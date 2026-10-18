"""Robot geometry for the fit and the posterior: forward kinematics from the URDF, the visual link meshes, the mask
rasterizer (GPU) and the contour of the rendered silhouette with normals, 3D points, links and the boundary Jacobian."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy import ndimage

from . import robot as _robot
from .se3 import SCALE, dof_of, pose_of

LINKS8 = ["link_base"] + [f"link{i}" for i in range(1, 8)]          # the mask links
W_TABLE = np.diag(SCALE ** 2)
_MESHES = None


def link_meshes():
    """visual meshes of the robot links (EasyHeC RenderXarmApiHelper.get_meshes: same loader call, same link order) of the
    active robot model, from absolute paths."""
    global _MESHES
    if _MESHES is None:
        _MESHES = {k: trimesh.load(str(v), force="mesh") for k, v in _robot.active().link_meshes}
    return _MESHES


ARM = ["link_base"] + [f"link{i}" for i in range(1, 8)]


def _origin(xyz, rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                  [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                  [-sp, cp * sr, cp * cr]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = xyz
    return T


class TorchXArmFK:
    """link_base..link7 (+ fixed tool frames) poses in the base frame, differentiable in q (float64)."""

    def __init__(self, urdf: Path, device):
        import torch
        self.torch = torch
        root = ET.parse(str(urdf)).getroot()
        joints = {j.find("child").get("link"): j for j in root.findall("joint")}
        def origin(j):
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
            return _origin(xyz, rpy)
        self.origins = []
        for name in ARM[1:]:
            j = joints[name]
            assert j.get("type") == "revolute" and j.find("axis").get("xyz").split() == ["0", "0", "1"], name
            self.origins.append(torch.tensor(origin(j), dtype=torch.float64, device=device))
        # fixed frames after link7 used only by the max-distance surrogate (fingers at q=0)
        self.tool = {}
        for name, parent in (("link_eef", "link7"), ("xarm_gripper_base_link", "link_eef"),
                             ("link_tcp", "xarm_gripper_base_link"), ("left_finger", "xarm_gripper_base_link"),
                             ("right_finger", "xarm_gripper_base_link")):
            self.tool[name] = (parent, torch.tensor(origin(joints[name]), dtype=torch.float64, device=device))
        self.device = device

    def __call__(self, q7):
        torch = self.torch
        T = torch.eye(4, dtype=torch.float64, device=self.device)
        out = {"link_base": T}
        for i, name in enumerate(ARM[1:]):
            c, s = torch.cos(q7[i]), torch.sin(q7[i])
            z, o = torch.zeros_like(c), torch.ones_like(c)
            Rz = torch.stack([torch.stack([c, -s, z, z]), torch.stack([s, c, z, z]),
                              torch.stack([z, z, o, z]), torch.stack([z, z, z, o])])
            T = T @ self.origins[i] @ Rz
            out[name] = T
        for name, (parent, O) in self.tool.items():
            out[name] = out[parent] @ O
        return out


def gpu_raster(self, q7, T_cam, dev, chunk=40_000_000):
    Pw = self.world(q7)
    Pc = Pw @ T_cam[:3, :3].T + T_cam[:3, 3]
    z = Pc[:, 2]
    u = self.Ks[0, 0] * Pc[:, 0] / z + self.Ks[0, 1] * Pc[:, 1] / z + self.Ks[0, 2]
    v = self.Ks[1, 1] * Pc[:, 1] / z + self.Ks[1, 2]
    F = self.faces
    good = (z[F] > 1e-3).all(1)
    F = F[good]; fl = self.face_link[good]
    U, Vv, Z = u[F], v[F], z[F]
    H, W = self.H, self.W
    c0 = np.clip(np.ceil(U.min(1) - 0.5), 0, W).astype(np.int64)
    c1 = np.clip(np.floor(U.max(1) - 0.5), -1, W - 1).astype(np.int64)
    r0 = np.clip(np.ceil(Vv.min(1) - 0.5), 0, H).astype(np.int64)
    r1 = np.clip(np.floor(Vv.max(1) - 0.5), -1, H - 1).astype(np.int64)
    bw = np.maximum(c1 - c0 + 1, 0); bh = np.maximum(r1 - r0 + 1, 0)
    nb = bw * bh
    keep = nb > 0
    fidx = np.nonzero(keep)[0]
    nb_k = nb[keep]
    tot = int(nb_k.sum())

    t = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    fidx_t, nb_t = t(fidx), t(nb_k)
    U_t, V_t, Z_t = t(U), t(Vv), t(Z)
    bw_t, r0_t, c0_t = t(bw), t(r0), t(c0)
    starts_all = torch.cumsum(nb_t, 0) - nb_t
    best_invz = torch.full((H * W,), -float("inf"), dtype=torch.float64, device=dev)
    pieces = []                                  # (global index offset, f, pid, invz, b0, b1, b2) of inside pairs
    # chunk over faces so that the pixel-pair arrays stay bounded
    bounds = [0]
    cs = np.cumsum(nb_k)
    while bounds[-1] < len(fidx):
        lo = bounds[-1]
        base = cs[lo - 1] if lo > 0 else 0
        hi = int(np.searchsorted(cs, base + chunk, side="right"))
        bounds.append(max(hi, lo + 1))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        nbc = nb_t[lo:hi]
        n = int(nbc.sum())
        if n == 0:
            continue
        rep = torch.repeat_interleave(torch.arange(lo, hi, device=dev), nbc)
        g0 = int(starts_all[lo])
        gidx = torch.arange(g0, g0 + n, device=dev)                 # global pair index = position in the CPU arrays
        start = starts_all[rep]
        loc = gidx - start
        f = fidx_t[rep]
        dr_ = torch.div(loc, bw_t[f], rounding_mode="floor"); dc_ = loc - dr_ * bw_t[f]
        rr = r0_t[f] + dr_; cc = c0_t[f] + dc_
        px = cc.to(torch.float64) + 0.5; py = rr.to(torch.float64) + 0.5
        u0, u1, u2 = U_t[f, 0], U_t[f, 1], U_t[f, 2]
        v0, v1, v2 = V_t[f, 0], V_t[f, 1], V_t[f, 2]
        area = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)
        w0 = (u1 - px) * (v2 - py) - (u2 - px) * (v1 - py)
        w1 = (u2 - px) * (v0 - py) - (u0 - px) * (v2 - py)
        w2 = (u0 - px) * (v1 - py) - (u1 - px) * (v0 - py)
        sgn = torch.sign(area)
        inside = (torch.abs(area) > 1e-12) & (w0 * sgn >= 0) & (w1 * sgn >= 0) & (w2 * sgn >= 0)
        gi = gidx[inside]; f = f[inside]; rr = rr[inside]; cc = cc[inside]
        ar = area[inside]
        b0, b1, b2 = w0[inside] / ar, w1[inside] / ar, w2[inside] / ar
        invz = b0 / Z_t[f, 0] + b1 / Z_t[f, 1] + b2 / Z_t[f, 2]
        pid = rr * W + cc
        best_invz.scatter_reduce_(0, pid, invz, reduce="amax")
        pieces.append((gi, f, pid, invz, b0, b1, b2))
    # winner per pixel: max invz, ties -> smallest global index (= first in the stable lexsort)
    BIG = torch.iinfo(torch.int64).max
    best_gi = torch.full((H * W,), BIG, dtype=torch.int64, device=dev)
    for gi, f, pid, invz, *_ in pieces:
        win = invz == best_invz[pid]
        best_gi.scatter_reduce_(0, pid[win], gi[win], reduce="amin")
    sel_f, sel_pid, sel_b = [], [], []
    for gi, f, pid, invz, b0, b1, b2 in pieces:
        w = best_gi[pid] == gi
        sel_f.append(f[w]); sel_pid.append(pid[w]); sel_b.append(torch.stack([b0[w], b1[w], b2[w]], 1))
    fs = torch.cat(sel_f).cpu().numpy() if sel_f else np.zeros(0, np.int64)
    ps = torch.cat(sel_pid).cpu().numpy() if sel_pid else np.zeros(0, np.int64)
    bs = torch.cat(sel_b).cpu().numpy() if sel_b else np.zeros((0, 3))
    m = np.zeros(H * W, bool); m[ps] = True
    face_img = -np.ones(H * W, np.int64); face_img[ps] = fs
    pb0, pb1, pb2 = bs[:, 0] / Z[fs, 0], bs[:, 1] / Z[fs, 1], bs[:, 2] / Z[fs, 2]
    ssum = pb0 + pb1 + pb2
    Pw_f = Pw[F[fs]]
    pts = (Pw_f[:, 0] * (pb0 / ssum)[:, None] + Pw_f[:, 1] * (pb1 / ssum)[:, None] + Pw_f[:, 2] * (pb2 / ssum)[:, None])
    pts_img = np.zeros((H * W, 3)); pts_img[ps] = pts
    link_img = -np.ones(H * W, np.int64); link_img[ps] = fl[fs]
    return m.reshape(H, W), link_img.reshape(H, W), pts_img.reshape(H, W, 3), tot


class ContourGeometry:

    def __init__(self, K, height=720, width=1280, scale=0.5, links=LINKS8, normal_sigma_px=1.5):
        self.fk = TorchXArmFK(_robot.active().urdf, torch.device("cpu"))
        self.dev = torch.device("cuda:0")
        meshes = link_meshes()
        self.links = list(links)
        self.verts, faces, lid, off = [], [], [], 0
        for li, name in enumerate(self.links):
            m = meshes[name]
            v = np.asarray(m.vertices, np.float64)
            self.verts.append(np.concatenate([v, np.ones((len(v), 1))], 1))
            faces.append(np.asarray(m.faces, np.int64) + off)
            lid.append(np.full(len(m.faces), li))
            off += len(v)
        self.faces = np.concatenate(faces); self.face_link = np.concatenate(lid)
        self.scale = scale
        self.H, self.W = int(round(height * scale)), int(round(width * scale))
        Ks = np.asarray(K, np.float64).copy(); Ks[:2] *= scale
        self.Ks = Ks
        r = max(1, int(math.ceil(3 * normal_sigma_px)))
        x = np.arange(-r, r + 1, dtype=np.float64)
        g = np.exp(-0.5 * (x / normal_sigma_px) ** 2); self.g = g / g.sum()

    def raster(self, q7, T_cam):
        """(mask, link image, 3D points) of the mask links; GPU port of the experiments' CPU rasterizer (bit-identical)."""
        r = gpu_raster(self, q7, T_cam, self.dev, chunk=8_000_000)[:3]
        torch.cuda.empty_cache()
        return r

    def world(self, q7):
        Tl = self.fk(torch.as_tensor(np.asarray(q7, np.float64)))
        return np.concatenate([v @ Tl[n].numpy().T for n, v in zip(self.links, self.verts)], 0)[:, :3]

    def boundary_jacobian(self, P, n, dof0, h=1e-3):
        Ph = np.concatenate([P, np.ones((len(P), 1))], 1)
        cols = []
        for k in range(6):
            uv = []
            for s in (1, -1):
                d = np.asarray(dof0, np.float64).copy(); d[k] += s * h
                T = pose_of(d)
                pc = Ph @ T.T
                uu = pc[:, :3] @ self.Ks.T
                uv.append(uu[:, :2] / uu[:, 2:3])
            cols.append(((uv[0] - uv[1]) / (2 * h) * n).sum(1))
        return np.stack(cols, 1) / self.scale


def frame_mask(shape) -> np.ndarray:
    f = np.zeros(shape, bool)
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = True
    return f


def render_contours(R: np.ndarray) -> list:
    """ordered contours of R as (n, 2) int arrays of (row, col) boundary pixels of R, image-frame pixels removed."""
    import cv2
    H, W = R.shape
    cs, _ = cv2.findContours(np.ascontiguousarray(R, np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        l = c[:, 0, ::-1].astype(np.int64)
        l = l[(l[:, 0] > 0) & (l[:, 0] < H - 1) & (l[:, 1] > 0) & (l[:, 1] < W - 1)]
        if len(l) >= 3:
            out.append(l)
    return out


class FullContour:
    """full-res render contour (ordered loops, image frame removed) with normals, 3D points, links, J."""

    def __init__(self, geo: ContourGeometry, q, T, dof=None, sigma=3.0):
        m, link_img, pts = geo.raster(q, T)
        self.m = m
        self.loops = render_contours(m)
        self.ok = bool(self.loops)
        if not self.ok:
            return
        pix = np.concatenate(self.loops)
        self.pix = pix
        r = int(math.ceil(3 * sigma)); x = np.arange(-r, r + 1, dtype=np.float64)
        g = np.exp(-0.5 * (x / sigma) ** 2); g /= g.sum()
        sm = ndimage.correlate1d(m.astype(np.float64), g, axis=1, mode="nearest")
        sm = ndimage.correlate1d(sm, g, axis=0, mode="nearest")
        gy, gx = np.gradient(sm)
        n = -np.stack([gx[pix[:, 0], pix[:, 1]], gy[pix[:, 0], pix[:, 1]]], 1)
        self.n = n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-12, None)
        self.P = pts[pix[:, 0], pix[:, 1]]
        self.li = link_img[pix[:, 0], pix[:, 1]]
        self.ds = 1.0 / np.clip(np.abs(self.n).max(1), 0.5, None)       # arc length per 8-connected chain pixel
        self.J = geo.boundary_jacobian(self.P, self.n, dof_of(T) if dof is None else dof)
        self.lens = [len(l) for l in self.loops]
