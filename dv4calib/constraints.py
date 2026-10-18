"""Constraint rows of TR-SQP (g <= 0 feasible): reach sphere, goal and path clearances, validity of R~; plus the
link metric of the trust region and the reach restoration (second-order correction)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


EPS_REACH = 1e-4          # m, a-priori numerical margin (>= 10x the TorchXArmFK-SAPIEN FK difference 7.7e-6 m)
WS_RES = 1e-3             # m, MPLib point-cloud resolution (released default)


@dataclass
class Support:
    tau_max: float
    L_min: float
    D_ref: float
    r_res: float
    info: dict


def link_metric(robot, q, device="cpu"):
    """G = sum_l J_l^T J_l + 1e-3 tr(G)/n I over the metric frames (DESIGN §5.2, dref8)."""
    qt = torch.as_tensor(np.asarray(q, np.float64), device=device)
    f = lambda x: robot.frame_positions(robot.fk(robot.q_full(x))).reshape(-1)
    J = torch.autograd.functional.jacobian(f, qt).cpu().numpy()
    G = J.T @ J
    return G + 1e-3 * np.trace(G) / len(G) * np.eye(len(G))


def restore_reach(robot, q, device="cpu", target_eps=EPS_REACH, iters=3):
    """Gauss-Newton restoration onto {|p_l - c| <= r_max - eps} for the violated frames (min-norm step)."""
    q = np.asarray(q, np.float64).copy()
    c = torch.as_tensor(robot.dist_center, device=device)
    for _ in range(iters):
        qt = torch.as_tensor(q, device=device)
        f = lambda x: torch.linalg.norm(torch.stack([robot.fk(robot.q_full(x))[n][:3, 3] for n in robot.dist_frames]) - c, dim=1) - (robot.dist_max - target_eps)
        g = f(qt).detach().cpu().numpy()
        viol = g > 0
        if not viol.any():
            break
        J = torch.autograd.functional.jacobian(f, qt).cpu().numpy()[viol]
        q = q - J.T @ np.linalg.solve(J @ J.T + 1e-12 * np.eye(len(J)), g[viol] * 1.5)
        q = np.clip(q, robot.lower, robot.upper)
    return q


class ConstraintsV2:
    def __init__(self, robot_c, clear, support, q_anchor, q_now, F_start=None):
        self.robot, self.clear, self.sup = robot_c, clear, support
        self.q_anchor = np.asarray(q_anchor, np.float64)[:6]
        self.q_now = np.asarray(q_now, np.float64)[:6]
        self.center = torch.as_tensor(robot_c.dist_center)
        with torch.no_grad():
            Tl0 = robot_c.fk(robot_c.q_full(torch.as_tensor(self.q_anchor)))
            self.P0 = robot_c.frame_positions(Tl0).clone()
        self.tau_cap, self.L_floor = support.tau_max, support.L_min
        if F_start is not None:
            self.tau_cap = max(support.tau_max, float(F_start["tau_c"]))
            self.L_floor = min(support.L_min, float(F_start["L_in"]))
        self.lo = np.maximum(robot_c.lower, self.q_now - support.D_ref)
        self.hi = np.minimum(robot_c.upper, self.q_now + support.D_ref)
        self.n_geo = 0
        self.t_clear = 0.0

    # --------------------------------------------------- C2 + M2 (CPU torch, autograd)
    def _cm(self, q):
        rob = self.robot
        qt = q if torch.is_tensor(q) else torch.as_tensor(np.asarray(q, np.float64))
        Tl = rob.fk(rob.q_full(qt))
        P = torch.stack([Tl[n][:3, 3] for n in rob.dist_frames])
        g2 = torch.linalg.norm(P - self.center, dim=1) - (rob.dist_max - EPS_REACH)
        gm2 = (torch.linalg.norm(rob.frame_positions(Tl) - self.P0, dim=1).max() - self.sup.r_res).reshape(1)
        return torch.cat([g2, gm2])

    def _cm_names(self):
        return [f"C2:{n}" for n in self.robot.dist_frames] + ["M2"]

    # --------------------------------------------------- interface
    def geo_val(self, q):
        import time
        with torch.no_grad():
            gcm = self._cm(q).numpy()
        t0 = time.perf_counter()
        gc, names, info = self.clear.rows(np.asarray(q, np.float64)[:6], self.q_anchor)
        self.t_clear += time.perf_counter() - t0; self.n_geo += 1
        self._last = (np.asarray(q, np.float64).copy(), info)
        return np.r_[gcm, gc], self._cm_names() + names

    def geo_jac(self, q):
        import time
        q = np.asarray(q, np.float64)[:6]
        qt = torch.as_tensor(q)
        gcm = self._cm(qt).detach().numpy()
        Jcm = torch.autograd.functional.jacobian(lambda x: self._cm(x), qt).numpy()
        t0 = time.perf_counter()
        gc, names, info = self.clear.rows(q, self.q_anchor)
        Jc = self.clear.rows_jac(q, self.q_anchor, info)
        self.t_clear += time.perf_counter() - t0; self.n_geo += 1
        return np.r_[gcm, gc], np.r_[Jcm, Jc], self._cm_names() + names

    def validity(self, F):
        return torch.stack([F["tau_c"] - self.tau_cap, self.L_floor - F["L_in"]]), ["V", "U"]

    def box_ok(self, q):
        return bool(np.all(q >= self.lo - 1e-12) and np.all(q <= self.hi + 1e-12))
