"""Exact risk R(q): the expected posterior error after a view at q, from the rasterized half-resolution contour of the
moving links (nvdiffrast), averaged over the estimate and the posterior samples (scores8.RScoreG of the experiments)."""
from __future__ import annotations

import math

import numpy as np
import torch

from . import robot as _robot
from .geometry import TorchXArmFK, link_meshes
from .se3 import SCALE, pose_of

LINKS8 = ["link_base"] + [f"link{i}" for i in range(1, 8)]


SOLVER_LINKS = [f"link{i}" for i in range(1, 8)]                    # v4 solver / mask contract (no base)


class LinearBayesScore:

    def __init__(self, fk, K: np.ndarray, height: int = 720, width: int = 1280, scale: float = 0.5,
                 links=SOLVER_LINKS, normal_sigma_px: float = 1.5):
        import nvdiffrast.torch as dr
        import torch
        self.torch, self.dr, self.fk, self.links = torch, dr, fk, list(links)
        self.dev = torch.device("cuda:0")
        self.ctx = dr.RasterizeCudaContext()
        self.scale = scale
        self.H, self.W = int(round(height * scale)), int(round(width * scale))
        Ks = np.asarray(K, np.float64).copy(); Ks[:2] *= scale
        self.Ks = torch.as_tensor(Ks, device=self.dev)
        meshes = link_meshes()
        verts, faces, off = [], [], 0
        for name in self.links:
            m = meshes[name]
            v = torch.as_tensor(np.asarray(m.vertices, np.float64), device=self.dev)
            verts.append(torch.cat([v, torch.ones_like(v[:, :1])], 1))
            faces.append(torch.as_tensor(np.asarray(m.faces, np.int32), device=self.dev) + off)
            off += len(m.vertices)
        self.verts, self.faces = verts, torch.cat(faces).contiguous()
        near, far = 0.001, 10.0
        fu, fv, cu, cv = Ks[0, 0], Ks[1, 1], Ks[0, 2], Ks[1, 2]
        proj = torch.tensor(((2 * fu / self.W, 0, -2 * cu / self.W + 1, 0), (0, 2 * fv / self.H, 2 * cv / self.H - 1, 0),
                             (0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)), (0, 0, -1, 0)),
                            dtype=torch.float64, device=self.dev)
        flip = torch.tensor(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)), dtype=torch.float64, device=self.dev)
        self.pv = proj @ flip
        r = max(1, int(math.ceil(3 * normal_sigma_px)))
        x = torch.arange(-r, r + 1, dtype=torch.float32, device=self.dev)
        g = torch.exp(-0.5 * (x / normal_sigma_px) ** 2); g = g / g.sum()
        self.gx, self.gy, self.r = g.view(1, 1, 1, -1), g.view(1, 1, -1, 1), r


class SandwichGeometry(LinearBayesScore):

    def __init__(self, fk, K: np.ndarray, height: int = 720, width: int = 1280, scale: float = 0.5,
                 pad_frac: float = 0.5, border_px: float = 4.0, links=None):
        super().__init__(fk, K, height, width, scale, **({} if links is None else {"links": links}))
        torch = self.torch
        self.full_h, self.full_w, self.border_px = height, width, border_px
        self.ph, self.pw = int(round(self.H * pad_frac)), int(round(self.W * pad_frac))
        self.He, self.We = self.H + 2 * self.ph, self.W + 2 * self.pw
        Ke = np.asarray(K, np.float64).copy(); Ke[:2] *= scale; Ke[0, 2] += self.pw; Ke[1, 2] += self.ph
        near, far = 0.001, 10.0
        fu, fv, cu, cv = Ke[0, 0], Ke[1, 1], Ke[0, 2], Ke[1, 2]
        proj = torch.tensor(((2 * fu / self.We, 0, -2 * cu / self.We + 1, 0), (0, 2 * fv / self.He, 2 * cv / self.He - 1, 0),
                             (0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)), (0, 0, -1, 0)),
                            dtype=torch.float64, device=self.dev)
        flip = torch.tensor(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)), dtype=torch.float64, device=self.dev)
        self.pv_e = proj @ flip

    def _world(self, q7):
        torch = self.torch
        Tl = self.fk(torch.as_tensor(np.asarray(q7, np.float64), device=self.dev))
        return torch.cat([v @ Tl[n].T for n, v in zip(self.links, self.verts)], 0)

    def contour_px_links(self, q7, T_cam):
        """contour_px plus the index (into self.links) of the link whose triangle covers each contour pixel."""
        torch, dr, F = self.torch, self.dr, self.torch.nn.functional
        if not hasattr(self, "_face_cum"):
            meshes = link_meshes()
            self._face_cum = np.cumsum([len(meshes[nm].faces) for nm in self.links])
        with torch.no_grad():
            world = self._world(q7)
            Tc = torch.as_tensor(np.asarray(T_cam, np.float64), device=self.dev)
            clip = ((world @ Tc.T) @ self.pv.T).float()[None].contiguous()
            rast, _ = dr.rasterize(self.ctx, clip, self.faces, resolution=(self.H, self.W))
            pts, _ = dr.interpolate(world[None, :, :3].float().contiguous(), rast, self.faces)
            rast, pts = torch.flip(rast, dims=[1]), torch.flip(pts, dims=[1])
            m = rast[0, ..., 3] > 0
            mi = m.float()[None, None]
            boundary = m & (F.max_pool2d(1 - mi, 3, 1, 1)[0, 0] > 0)
            sm = F.conv2d(F.pad(mi, (self.r, self.r, 0, 0), mode="replicate"), self.gx)
            sm = F.conv2d(F.pad(sm, (0, 0, self.r, self.r), mode="replicate"), self.gy)[0, 0]
            gy, gx = torch.gradient(sm)
            ys, xs = torch.nonzero(boundary, as_tuple=True)
            n = -torch.stack([gx[ys, xs], gy[ys, xs]], 1).double()
            n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)
            P = pts[0, ys, xs].double()
            tri = rast[0, ys, xs, 3].round().long().cpu().numpy() - 1
        link_idx = np.searchsorted(self._face_cum, tri, side="right")
        return P, n, m, ys, xs, link_idx


class RScoreG:

    def __init__(self, K, x_hat, I_acq, b_link, b_blk, samples, with_delta=False, scale=0.5, fk=None, h=1e-3,
                 min_px=10, geo=None):
        self.torch = torch
        self.pose_of = pose_of
        self.dev = torch.device("cuda:0")
        self.fk = fk if fk is not None else TorchXArmFK(_robot.active().urdf, self.dev)
        self.geo = geo if geo is not None else SandwichGeometry(self.fk, np.asarray(K, np.float64), scale=scale, links=LINKS8)
        self.s = self.geo.scale
        self.W = np.diag(SCALE ** 2)
        self.with_delta = bool(with_delta)
        self.na = 7 if self.with_delta else 6
        self.x_hat = np.asarray(x_hat, np.float64)
        self.I_acq = np.asarray(I_acq, np.float64)
        assert self.I_acq.shape == (self.na, self.na), self.I_acq.shape
        self.b_link = np.asarray(b_link, np.float64)
        self.b_blk = float(b_blk)
        self.xs = np.concatenate([self.x_hat[None], np.asarray(samples, np.float64).reshape(-1, 6)], 0)
        self.h = h
        self.min_px = min_px
        self._poses = [self._pose_pack(x) for x in self.xs]
        self.T0 = self._poses[0][0]
        self.prior_risk = self._risk(np.zeros((self.na, self.na)))

    def _pose_pack(self, x):
        torch = self.torch
        T = self.pose_of(x)
        pert = []
        for k in range(6):
            for sg in (1, -1):
                d = np.asarray(x, np.float64).copy(); d[k] += sg * self.h
                pert.append(torch.as_tensor(self.pose_of(d), device=self.dev))
        return T, pert

    def _jac(self, P, n, k):
        torch = self.torch
        Ph = torch.cat([P, torch.ones_like(P[:, :1])], 1)
        pert = self._poses[k][1]; cols = []
        for j in range(6):
            uv = []
            for T in (pert[2 * j], pert[2 * j + 1]):
                u = (Ph @ T.T)[:, :3] @ self.geo.Ks.T
                uv.append(u[:, :2] / u[:, 2:3])
            cols.append(((uv[0] - uv[1]) / (2 * self.h) * n).sum(1))
        return torch.stack(cols, 1) / self.s

    def info_q(self, q7, k):
        torch = self.torch
        T = self._poses[k][0]
        P, n, m, ys, xs, li = self.geo.contour_px_links(np.asarray(q7, np.float64), T)
        mv = (li >= 1) & ((n[:, 0].abs() + n[:, 1].abs()).cpu().numpy() >= 0.5)
        if int(mv.sum()) < self.min_px:
            return np.zeros((self.na, self.na)), int(mv.sum())
        mvt = torch.as_tensor(mv, device=P.device)
        P, n = P[mvt], n[mvt]
        J = self._jac(P, n, k)
        ds = 1.0 / ((n[:, 0].abs() + n[:, 1].abs()) * self.s)
        b = torch.as_tensor(self.b_link[li[mv]], device=P.device)
        w = ds / b ** 2
        A = torch.cat([J, torch.ones_like(J[:, :1])], 1) if self.with_delta else J
        return (A.T @ (w[:, None] * A)).cpu().numpy(), int(mv.sum())

    def _risk(self, Iq):
        Ipost = (self.I_acq + Iq) / self.b_blk
        S = np.linalg.inv(Ipost)[:6, :6]
        return float(np.trace(self.W @ S))

    def R(self, q7, poses=None):
        ks = range(len(self.xs)) if poses is None else poses
        terms, npx = [], []
        for k in ks:
            Iq, n = self.info_q(q7, k)
            terms.append(self._risk(Iq)); npx.append(n)
        terms = np.asarray(terms)
        return float(terms.mean()), float(terms[0]), terms, np.asarray(npx)


class ExactR:
    """R(q) for one posterior (experiments: diffsel_v2.api.ExactR with lscape_common.Scorer)."""

    def __init__(self, post):
        K = np.asarray(post["K"], np.float64)
        fk = TorchXArmFK(_robot.active().urdf, torch.device("cuda:0"))
        geo = SandwichGeometry(fk, K, *post["image_hw"], scale=0.5, links=LINKS8)
        self.rs = RScoreG(K, post["Rls_x_hat"], post["Rls_I_acq"], post["Rls_b_link"], float(post["Rls_b_blk"]),
                          post["Rls_samples"], with_delta=False, fk=fk, geo=geo)
        self.n = 0

    def __call__(self, q):
        self.n += 1
        q7 = np.r_[np.asarray(q, np.float64)[:6], 0.0]
        v, _, terms, px = self.rs.R(q7)
        return float(v)
