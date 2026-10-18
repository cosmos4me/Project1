"""Calibrator: D v4 active calibration of one fixed camera on an xArm7.

Inputs
  K       3x3 camera intrinsics in pixels (undistorted images)
  T_init  4x4 initial camera <- robot-base transform in metres (X_cam = T @ X_base), e.g. from a pose estimator
  q       joint angles of each view: the 7 arm joints in radians (gripper joints are not used)
  mask    H x W robot mask of each view: non-zero on the silhouette of link_base and link1..link7 (not the gripper)
  robot   robot model files (default: the bundled xArm7 model, dv4calib.xarm7())

Per view: nl_ls fit of the camera pose from T_init with all views (dv4calib.fit), Laplace posterior (dv4calib.posterior),
and the D v4 decision of the next joint configuration (dv4calib.decide; joint 7 = 0, gripper unchanged).
The GPU is cuda:0 of the process; select it with CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import config
from . import robot as _robot

_DECIDER = None


def _decider():
    """the D v4 decider of this process (planners, distance fields and feasible set are built once)."""
    global _DECIDER
    if _DECIDER is None:
        from .feasibility import ensure_fields
        ensure_fields()
        from .decide import DecisionV4
        _DECIDER = DecisionV4("cuda:0")
    return _DECIDER


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def _check_K(K):
    K = np.asarray(K, np.float64)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)) or not np.allclose(K[2], (0, 0, 1)) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("K must be a finite 3x3 intrinsic matrix with last row (0, 0, 1) and positive focal lengths")
    return K


def _check_T(T):
    T = np.asarray(T, np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)) or not np.allclose(T[3], (0, 0, 0, 1)):
        raise ValueError("T_init must be a finite 4x4 homogeneous transform")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or np.linalg.det(R) < 0:
        raise ValueError("T_init[:3, :3] must be a rotation matrix")
    if np.linalg.norm(T[:3, 3]) > 10.0:
        raise ValueError("T_init translation is longer than 10 m: metres are expected")
    if T[2, 3] <= 0:
        raise ValueError("the robot base is behind the camera in T_init: T_init must map base coordinates to camera "
                         "coordinates (camera <- base); invert it if it is base <- camera")
    return T


def _check_q(q, limits=None):
    q = np.asarray(q, np.float64).reshape(-1)
    if len(q) < 7 or not np.all(np.isfinite(q[:7])):
        raise ValueError("q must hold at least the 7 arm joint angles (rad)")
    if limits is not None and (np.any(q[:7] < limits[0] - 0.02) or np.any(q[:7] > limits[1] + 0.02)):
        raise ValueError(f"joint angles {np.round(q[:7], 3).tolist()} are outside the joint limits of the robot model "
                         "(radians are expected)")
    return q[:7].copy()


def _check_mask(m, hw=None):
    m = np.asarray(m)
    if m.ndim == 3:
        if m.shape[-1] in (1, 3, 4):
            m = m[..., 0]
        elif m.shape[0] in (1, 3, 4):
            m = m[0]
    if m.ndim != 2:
        raise ValueError("mask must be an H x W array")
    if hw is not None and m.shape != tuple(hw):
        raise ValueError(f"mask size {m.shape} differs from the first view {tuple(hw)}")
    m = (m >= 0.5) if np.issubdtype(m.dtype, np.floating) else (m > 0)     # probabilities: threshold 0.5
    if not m.any():
        raise ValueError("mask is empty")
    return m


def _mask_sha(m):
    import hashlib
    return hashlib.sha256(np.packbits(np.asarray(m, bool)).tobytes() + str(np.shape(m)).encode()).hexdigest()


def pose_std(x_hat, Sigma, h=1e-6):
    """posterior standard deviation of the camera position in the robot-base frame (mm, x y z) and of the camera
    orientation (deg, rotation vector of R(x) R(x_hat)^T), by linearising the pose map at x_hat (float64)."""
    import torch
    from .se3 import pose64

    def feat(x):
        T = pose64(torch.as_tensor(np.asarray(x, np.float64))).numpy()
        return T

    R0 = feat(x_hat)[:3, :3]
    Jc, Jr = np.zeros((3, 6)), np.zeros((3, 6))
    for k in range(6):
        out = []
        for sg in (1.0, -1.0):
            x = np.asarray(x_hat, np.float64).copy(); x[k] += sg * h
            T = feat(x)
            c = -T[:3, :3].T @ T[:3, 3]
            dR = T[:3, :3] @ R0.T
            out.append((c, 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])))
        Jc[:, k] = (out[0][0] - out[1][0]) / (2 * h)
        Jr[:, k] = (out[0][1] - out[1][1]) / (2 * h)
    S = 0.5 * (np.asarray(Sigma) + np.asarray(Sigma).T)
    sc = np.sqrt(np.clip(np.diag(Jc @ S @ Jc.T), 0, None)) * 1000.0
    sr = np.degrees(np.sqrt(np.clip(np.diag(Jr @ S @ Jr.T), 0, None)))
    return sc, sr


class Calibrator:
    def __init__(self, K, T_init, q1, mask1, robot=None, workdir=None, seed: int = 0):
        import torch
        model = _robot.use(robot) if robot is not None else _robot.active()
        self.limits = model.arm_joint_limits()
        torch.zeros(1, device="cuda:0")
        from .geometry import ContourGeometry
        from .se3 import dof_of
        self.K = _check_K(K)
        m1 = _check_mask(mask1)
        self.image_hw = m1.shape
        self.geo = ContourGeometry(self.K, height=self.image_hw[0], width=self.image_hw[1], scale=1.0)
        self.T_init = _check_T(T_init)
        self.d0 = dof_of(self.T_init)
        self.views = [(_check_q(q1, self.limits), m1)]
        self.workdir = None if workdir is None else config.user_path(workdir)
        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.x_hat = None; self.v1 = None; self.post = None; self.warm = None; self._last = None
        self._fit_k = 0; self._post_k = 0
        self.history = []
        self.decisions = []

    # ------------------------------------------------------------------ steps
    def add_view(self, q, mask):
        """adds the observation of a new view (joint angles actually reached, robot mask).  The end points of the decision
        for this view become the warm start of the next decision, as in the sequential method."""
        k = len(self.views)
        self.views.append((_check_q(q, self.limits), _check_mask(mask, self.image_hw)))
        last = self._last if self._last is not None and self._last[0] == k else None
        self.warm = None if last is None else [(n, np.asarray(v, np.float64)) for n, v in last[2].log["warm_out"]]

    def fit(self):
        """nl_ls fit of the camera pose from T_init using all views; returns x_hat (6-d)."""
        from .fit import nl_ls, resid
        from .se3 import pose_of
        t0 = time.time()
        d_nl, d_ls, _ = nl_ls(self.geo, self.views, self.d0)
        self.x_hat = np.asarray(d_ls, np.float64)
        self._fit_k = len(self.views)
        quality = []
        T = pose_of(self.x_hat)
        for q, m in self.views:                       # fit quality per view: overlap and mean boundary offset (px)
            Cn, t, xor = resid(self.geo, q, T, m, self.x_hat)
            union = int((Cn.m | m).sum())
            quality.append(dict(iou=float((Cn.m & m).sum() / max(union, 1)),
                                mean_abs_offset_px=float(np.average(np.abs(t), weights=Cn.ds)) if Cn.ok else None))
        self.history.append(dict(views=self._fit_k, x_hat=self.x_hat.tolist(), T_cam_from_base=self.estimate().tolist(),
                                 fit_quality=quality, t_fit=time.time() - t0))
        self.save()
        return self.x_hat

    def posterior(self):
        """Laplace posterior of the current views (dict handed to the decision)."""
        from . import posterior as POST
        k = len(self.views)
        if self._fit_k != k:
            self.fit()
        if k == 1:
            P = POST.view1(self.geo, self.views[0][0], self.views[0][1], self.x_hat)
            self.v1 = P
            samples = POST.draws(self.x_hat, P["Sigma"], [POST.SEED0, self.seed], POST.M_DRAWS)
            I, Sigma = P["I"], P["Sigma"]
        else:
            self._ensure_v1()
            Pk = POST.k_views(self.geo, self.views, self.x_hat, self.v1["b_link"], self.v1["b_blk"])
            samples = POST.draws(self.x_hat, Pk["Sigma"], [POST.SEED0, self.seed, k], POST.M_DRAWS)
            I, Sigma = Pk["I"], Pk["Sigma"]
        self.post = dict(K=self.K, q1=self.views[-1][0], Rls_x_hat=self.x_hat, Rls_I_acq=I, Rls_Sigma=Sigma,
                         Rls_b_link=self.v1["b_link"], Rls_b_blk=np.float64(self.v1["b_blk"]), Rls_samples=samples,
                         image_hw=tuple(self.image_hw))
        self._post_k = k
        sc, sr = pose_std(self.x_hat, Sigma)
        self.history[-1].update(position_std_mm=sc.tolist(), rotation_std_deg=sr.tolist())
        self.save()
        return dict(I=I, Sigma=Sigma, samples=samples)

    def _ensure_v1(self):
        """view-1 noise scales b_link, b_blk, from the fit of view 1 alone (as in the sequential method); computed here
        when several views were added before the first posterior."""
        if self.v1 is None:
            from . import posterior as POST
            from .fit import nl_ls
            x1 = np.asarray(nl_ls(self.geo, self.views[:1], self.d0)[1], np.float64)
            self.v1 = POST.view1(self.geo, self.views[0][0], self.views[0][1], x1)
        return self.v1

    def next_view(self):
        """D v4 decision for the next view; returns (q7, decision).  It is made once per number of views: asking again
        returns the same pose (also a pose that no path check passed, flagged decision.log['unchecked'])."""
        k = len(self.views)
        if self._last is not None and self._last[0] == k:
            return self._last[1], self._last[2]
        if self._post_k != k:
            self.posterior()
        t0 = time.perf_counter()
        d = _decider().decide(self.post, k, self.warm)
        t = time.perf_counter() - t0
        q7 = np.r_[np.asarray(d.q, np.float64)[:6], 0.0]
        rec = dict(after_views=k, q=q7, R=d.R_exact, source=d.source, t=t, views_q=[v[0] for v in self.views],
                   views_mask_sha256=[_mask_sha(v[1]) for v in self.views], log=d.log)
        self.decisions.append(rec)
        if self.workdir is not None:
            (self.workdir / f"decision_k{k}.json").write_text(json.dumps(_jsonable(rec)))
        self.history[-1].update(decision=dict(q=q7.tolist(), R=d.R_exact, source=d.source, fallback=d.log["fallback"],
                                              unchecked=d.log["unchecked"], t=t))
        self._last = (k, q7, d)
        self.save()
        return q7, d

    def estimate(self):
        """current estimate as a 4x4 camera <- base transform."""
        from .se3 import pose_of
        if self.x_hat is None:
            raise RuntimeError("no estimate yet (call fit())")
        return np.asarray(pose_of(self.x_hat), np.float64)

    def save(self):
        if self.workdir is not None:
            (self.workdir / "history.json").write_text(json.dumps(_jsonable(self.history)))

    # ------------------------------------------------------------------ loop
    def run(self, capture, n_views=5, log=None, allow_unchecked=False):
        """full loop: capture(q7) moves the robot to q7 and returns (q_reached, mask) or just the mask.
        Returns the final 4x4 camera <- base estimate after n_views views (the final posterior is also computed, for the
        reported uncertainty).  If no candidate passed the released path check, the loop stops with an error unless
        allow_unchecked=True; calling run() again continues with the same pose."""
        if self._fit_k != len(self.views):
            self.fit()
        while len(self.views) < n_views:
            q7, d = self.next_view()
            if d.log["unchecked"] and not allow_unchecked:
                raise RuntimeError(f"view {len(self.views) + 1}: no candidate passed the released path check (best pose "
                                   f"{np.round(q7, 4).tolist()}). Check the motion yourself and continue with "
                                   "run(..., allow_unchecked=True) if it is safe.")
            if log:
                log(f"view {len(self.views) + 1}: q {np.round(q7, 4).tolist()} R {d.R_exact:.5f} ({d.source})")
            obs = capture(q7)
            q_reached, mask = obs if isinstance(obs, tuple) else (q7, obs)
            self.add_view(q_reached, mask)
            self.fit()
        if self._post_k != len(self.views):
            self.posterior()
        return self.estimate()
