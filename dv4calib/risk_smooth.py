"""Smooth risk R~(q) with exact reverse-mode gradients (the objective of TR-SQP): soft facing, soft coverage and soft
image border of the silhouette edges of the robot meshes, tau = 1 px; plus the differentiable arm model (forward
kinematics, reach frames, joint limits), the fixed camera and the camera belief (posterior) it uses."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from . import robot as _robot
from .geometry import TorchXArmFK, link_meshes
from .se3 import SCALE, pose64

URDF = _robot.active().urdf


LINKS8 = ["link_base"] + [f"link{i}" for i in range(1, 8)]
TOOL = ["link_eef", "xarm_gripper_base_link", "link_tcp", "left_finger", "right_finger"]
LINKS_ALL = LINKS8 + TOOL
W_XARM = np.diag(SCALE ** 2)
POST_KEYS = ("K", "q1", "Rls_x_hat", "Rls_I_acq", "Rls_Sigma", "Rls_b_link", "Rls_b_blk", "Rls_samples")
_MESH_CACHE = {}


@dataclass
class ArmModel:
    name: str
    n: int
    q_full: Callable                       # active q (tensor) -> full joint vector (tensor)
    lower: np.ndarray
    upper: np.ndarray
    fk: Callable                           # full q tensor -> {link: 4x4 base<-link} float64 autograd
    render_links: list
    meshes: dict                           # link -> PolyMesh
    parent: dict                           # link -> (parent link | None, joint name | None)  (kinematic tree)
    active_joints: set
    area_sign: float = 1.0                 # sigma of §3.3, fixed by the T0 unit test
    metric_frames: list = field(default_factory=list)   # frames of the link-displacement metric G / M2
    dist_frames: list | None = None        # C2 frames (xArm)
    dist_center: np.ndarray | None = None
    dist_max: float | None = None
    extra: dict = field(default_factory=dict)

    def frame_positions(self, Tl):
        return torch.stack([Tl[n][:3, 3] for n in self.metric_frames])


@dataclass
class CameraBelief:
    x_hat: np.ndarray
    draws: np.ndarray                      # (8,6) fixed-seed posterior draws
    I_acq: np.ndarray
    b_link: dict                           # link -> b
    b_blk: float
    W: np.ndarray
    Sigma: np.ndarray | None = None


class CameraSpec:
    """base class; subclasses implement T_cam / D / p_tilde for one mounting / parametrisation."""
    name: str
    K: np.ndarray
    W: int
    H: int
    s: float
    z_near: float = 1e-3
    mount: str | None = None

    def static_links(self, robot: ArmModel) -> set:
        """links with no active joint between them and the mounting link (§6.3)."""
        mount = self.mount
        out = set()
        for l in robot.render_links:
            if self._rigid(robot, l, mount):
                out.add(l)
        return out

    @staticmethod
    def _chain(robot, l):
        ch = [l]
        while robot.parent.get(ch[-1], (None, None))[0] is not None:
            ch.append(robot.parent[ch[-1]][0])
        return ch

    def _rigid(self, robot, l, mount):
        root = self._chain(robot, l)[-1]
        m = root if mount is None else mount
        cl, cm = self._chain(robot, l), self._chain(robot, m)
        common = next(a for a in cl if a in cm)
        joints = []
        for ch in (cl, cm):
            for a in ch:
                if a == common:
                    break
                joints.append(robot.parent[a][1])
        return not any(j in robot.active_joints for j in joints)


@dataclass
class PolyMesh:
    V: np.ndarray            # (nv,3) welded vertices, link frame, float64
    F: np.ndarray            # (nf,3) faces (welded ids), consistent orientation
    edges: np.ndarray        # (ne,2) vertex ids (v0 < v1)
    edge_tris: np.ndarray    # (ne,2) incident faces (-1 = boundary edge)
    edge_dir_sign: np.ndarray  # (ne,2) +1 if that face traverses v0->v1, -1 if v1->v0 (0 for missing face)
    facet_of_tri: np.ndarray  # (nf,) facet id (sliver absorption applied)
    facet_diam: np.ndarray    # (n_facet,) D_F, 3D max vertex distance [m]
    facet_centroid: np.ndarray  # (n_facet,3) area-weighted 3D centroid (link frame)
    edge_len3d: np.ndarray    # (ne,) L_e [m]
    internal: np.ndarray      # (ne,) both incident faces in the same facet
    boundary: np.ndarray      # (ne,) only one incident face
    stats: dict = field(default_factory=dict)


def _face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a2 = np.linalg.norm(n, axis=1)
    return n / np.maximum(a2, 1e-300)[:, None], 0.5 * a2


def build_mesh(V, F, weld=1e-6, theta_flat=1e-3, a_min_mm2=1e-3) -> PolyMesh:
    V = np.asarray(V, np.float64); F = np.asarray(F, np.int64)
    key = np.round(V / weld).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    wv = V[first]                                   # representative = first raw vertex of the group (deterministic)
    wf = inv[F]
    degen = (wf[:, 0] == wf[:, 1]) | (wf[:, 1] == wf[:, 2]) | (wf[:, 0] == wf[:, 2])
    wf = wf[~degen]
    nf = len(wf)
    he = np.concatenate([wf[:, [0, 1]], wf[:, [1, 2]], wf[:, [2, 0]]], 0)
    hf = np.tile(np.arange(nf), 3)
    und = np.sort(he, 1)
    uniq, einv, cnt = np.unique(und, axis=0, return_inverse=True, return_counts=True)
    einv = einv.reshape(-1)
    ne = len(uniq)
    order = np.argsort(einv, kind="stable")
    starts = np.r_[0, np.cumsum(cnt)[:-1]]
    edge_tris = -np.ones((ne, 2), np.int64)
    edge_sign = np.zeros((ne, 2), np.int64)
    nonman = cnt > 2
    for slot in range(2):
        has = cnt > slot
        h = order[starts[has] + slot]
        edge_tris[has, slot] = hf[h]
        edge_sign[has, slot] = np.where(he[h, 0] == uniq[has, 0], 1, -1)
    man = cnt == 2
    consistent = edge_sign[man, 0] == -edge_sign[man, 1]
    # ---- facets: region growing over manifold edges, dihedral < theta_flat and seed-normal angle < theta_flat
    nrm, area = _face_normals(wv, wf)
    sliver = area < a_min_mm2 * 1e-6                # mm^2 -> m^2
    cos_t = np.cos(theta_flat)
    adj_ptr_f = np.concatenate([edge_tris[man, 0], edge_tris[man, 1]])
    adj_nb_f = np.concatenate([edge_tris[man, 1], edge_tris[man, 0]])
    o = np.argsort(adj_ptr_f, kind="stable")
    adj_src, adj_dst = adj_ptr_f[o], adj_nb_f[o]
    ptr = np.searchsorted(adj_src, np.arange(nf + 1))
    facet = -np.ones(nf, np.int64)
    nfac = 0
    for f0 in range(nf):
        if facet[f0] >= 0 or sliver[f0]:
            continue
        facet[f0] = nfac
        stack = [f0]
        n0 = nrm[f0]
        while stack:
            f = stack.pop()
            for g in adj_dst[ptr[f]:ptr[f + 1]]:
                if facet[g] >= 0 or sliver[g]:
                    continue
                if nrm[f] @ nrm[g] >= cos_t and n0 @ nrm[g] >= cos_t:
                    facet[g] = nfac
                    stack.append(g)
        nfac += 1
    # ---- sliver absorption: facet across the longest edge, following sliver chains
    e_of_face = einv.reshape(3, nf).T                # face -> its 3 edge ids (he order: 01,12,20)
    L3 = np.linalg.norm(wv[uniq[:, 1]] - wv[uniq[:, 0]], axis=1)
    n_unabs = 0
    for f0 in np.flatnonzero(sliver):
        if facet[f0] >= 0:
            continue
        seen = {f0}
        f = f0
        target = -1
        while True:
            es = e_of_face[f]
            e = es[np.argmax(L3[es])]
            t = edge_tris[e]
            g = t[0] if t[0] != f else t[1]
            if g < 0 or g in seen:
                break
            if facet[g] >= 0:
                target = facet[g]
                break
            seen.add(g); f = g
        if target < 0:
            target = nfac; nfac += 1; n_unabs += 1
        for g in seen:
            if facet[g] < 0:
                facet[g] = target
    # ---- facet constants
    cen_num = np.zeros((nfac, 3)); cen_den = np.zeros(nfac)
    tri_c = wv[wf].mean(1)
    np.add.at(cen_num, facet, tri_c * np.maximum(area, 1e-300)[:, None])
    np.add.at(cen_den, facet, np.maximum(area, 1e-300))
    centroid = cen_num / cen_den[:, None]
    diam = np.zeros(nfac)
    fo = np.argsort(facet, kind="stable")
    fptr = np.searchsorted(facet[fo], np.arange(nfac + 1))
    for k in range(nfac):
        vs = np.unique(wf[fo[fptr[k]:fptr[k + 1]]].reshape(-1))
        P = wv[vs]
        if len(P) > 400:                              # exact max distance via convex-hull vertices would be cheaper;
            from scipy.spatial import ConvexHull      # fall back to hull for very large facets
            try:
                P = P[ConvexHull(P, qhull_options="QJ").vertices]
            except Exception:
                pass
        d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)).max() if len(P) > 1 else 0.0
        diam[k] = d
    boundary = edge_tris[:, 1] < 0
    internal = (~boundary) & (facet[np.maximum(edge_tris[:, 0], 0)] == facet[np.maximum(edge_tris[:, 1], 0)])
    stats = {"V_raw": int(len(V)), "V_welded": int(len(wv)), "F": int(nf), "F_degenerate_dropped": int(degen.sum()),
             "E": int(ne), "boundary_edges": int(boundary.sum()), "nonmanifold_edges": int(nonman.sum()),
             "orient_consistent_frac": float(consistent.mean()) if man.any() else 1.0,
             "n_facets": int(nfac), "n_sliver_faces": int(sliver.sum()), "n_sliver_unabsorbed_facets": int(n_unabs),
             "internal_edges": int(internal.sum()),
             "candidate_edges(non-internal)": int((~internal).sum())}
    return PolyMesh(V=wv, F=wf, edges=uniq, edge_tris=edge_tris, edge_dir_sign=edge_sign, facet_of_tri=facet,
                    facet_diam=diam, facet_centroid=centroid, edge_len3d=L3, internal=internal, boundary=boundary,
                    stats=stats)


def belief_of(post) -> CameraBelief:
    bl = {n: float(post["Rls_b_link"][i]) for i, n in enumerate(LINKS8)}
    return CameraBelief(x_hat=np.asarray(post["Rls_x_hat"], np.float64), draws=np.asarray(post["Rls_samples"], np.float64),
                        I_acq=np.asarray(post["Rls_I_acq"], np.float64), b_link=bl, b_blk=float(post["Rls_b_blk"]),
                        W=W_XARM, Sigma=np.asarray(post["Rls_Sigma"], np.float64))


class XArmFixedCamera(CameraSpec):
    def __init__(self, K, W=1280, H=720, s=0.5, name="fixed"):
        self.name, self.K, self.W, self.H, self.s = name, np.asarray(K, np.float64), W, H, s
        self.z_near = 1e-3
        self.mount = None
        self._Dcache = {}

    def T_cam(self, Tl, x):
        return pose64(x)

    def D(self, x):
        key = tuple(np.round(np.asarray(x, np.float64), 15))
        if key not in self._Dcache:
            J = torch.autograd.functional.jacobian(pose64, torch.as_tensor(np.asarray(x, np.float64)))
            self._Dcache[key] = J.permute(2, 0, 1).contiguous()          # (6,4,4) = d exp(x) / d x_j
        return self._Dcache[key]

    def p_tilde(self, Tl, pb):
        return torch.cat([pb, torch.ones_like(pb[:, :1])], 1)


def xarm_meshes():
    if "xarm" not in _MESH_CACHE:
        meshes = link_meshes()
        _MESH_CACHE["xarm"] = {n: build_mesh(np.asarray(meshes[n].vertices, np.float64), np.asarray(meshes[n].faces, np.int64))
                               for n in LINKS8}
    return _MESH_CACHE["xarm"]


def joint_limits():
    """PyMP joint-limit box of the released candidate bank (summary values, same as FrozenBankCandidateSource.low/high)."""
    from .feasibility import _Source                   # the PyMP joint limits of the released backend
    s = _Source()
    return np.asarray(s.low[:7], np.float64).copy(), np.asarray(s.high[:7], np.float64).copy()


def xarm_model(device="cpu", center=None) -> ArmModel:
    dev = torch.device(device)
    fk = TorchXArmFK(URDF, dev)
    low7, high7 = joint_limits()

    def q_full(q):
        return torch.cat([q.reshape(-1)[:6], torch.zeros(1, dtype=q.dtype, device=q.device)])

    parent = {"link_base": (None, None)}
    for i in range(1, 8):
        parent[f"link{i}"] = (LINKS8[i - 1], f"joint{i}")
    rm = ArmModel(name="xarm", n=6, q_full=q_full, lower=low7[:6].copy(), upper=high7[:6].copy(), fk=fk,
                    render_links=list(LINKS8), meshes=xarm_meshes(), parent=parent,
                    active_joints={f"joint{i}" for i in range(1, 7)}, area_sign=-1.0,
                    metric_frames=list(LINKS_ALL), dist_frames=[n for n in LINKS_ALL if n != "link_base"],
                    dist_center=None if center is None else np.asarray(center, np.float64), dist_max=0.5)
    rm.extra["low7"], rm.extra["high7"] = low7, high7
    return rm


def sstep(t):
    """quintic C2 smoothstep, 0 for t <= -1/2, 1 for t >= 1/2."""
    s = (t + 0.5).clamp(0.0, 1.0)
    return s * s * s * (s * (6.0 * s - 15.0) + 10.0)


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class RobotGeom:
    """all render links concatenated (global vertex / face / facet / edge ids), torch tensors on `device`."""

    def __init__(self, robot, device):
        self.device = device
        dt = torch.float64
        V, F, vl, fl, ftf, fdiam, fcen, flink = [], [], [], [], [], [], [], []
        ev, etri, esgn, efac, elink, elen, ebnd = [], [], [], [], [], [], []
        voff = foff = facoff = 0
        self.link_names = list(robot.render_links)
        self.link_vrange = []
        for li, name in enumerate(self.link_names):
            pm = robot.meshes[name]
            nv, nf, nfac = len(pm.V), len(pm.F), len(pm.facet_diam)
            V.append(pm.V); F.append(pm.F + voff)
            vl.append(np.full(nv, li)); fl.append(np.full(nf, li))
            ftf.append(pm.facet_of_tri + facoff); fdiam.append(pm.facet_diam); fcen.append(pm.facet_centroid)
            flink.append(np.full(nfac, li))
            keep = ~pm.internal
            E = pm.edges[keep]; T = pm.edge_tris[keep]; S = pm.edge_dir_sign[keep]
            fac = np.where(T >= 0, pm.facet_of_tri[np.maximum(T, 0)] + facoff, -1)
            ev.append(E + voff); etri.append(np.where(T >= 0, T + foff, -1)); esgn.append(S); efac.append(fac)
            elink.append(np.full(len(E), li)); elen.append(pm.edge_len3d[keep]); ebnd.append(pm.boundary[keep])
            self.link_vrange.append((voff, voff + nv))
            voff += nv; foff += nf; facoff += nfac
        cat = lambda xs, d=None: torch.as_tensor(np.concatenate(xs), device=device, dtype=d)
        self.V = cat(V, dt); self.F = cat(F, torch.long); self.v_link = cat(vl, torch.long); self.f_link = cat(fl, torch.long)
        self.facet_of_tri = cat(ftf, torch.long); self.facet_diam = cat(fdiam, dt); self.facet_cen = cat(fcen, dt)
        self.facet_link = cat(flink, torch.long)
        self.e_v = cat(ev, torch.long); self.e_tri = cat(etri, torch.long); self.e_sign = cat(esgn, dt)
        self.e_fac = cat(efac, torch.long); self.e_link = cat(elink, torch.long); self.e_len3d = cat(elen, dt)
        self.e_bnd = cat(ebnd, torch.bool)
        self.nV, self.nF, self.nFacet, self.nE = len(self.V), len(self.F), len(self.facet_diam), len(self.e_v)

    def base_points(self, Tl: dict):
        """(nV,3) vertices and (nFacet,3) facet centroids in the robot base frame (autograd through Tl)."""
        Pv, Pc = [], []
        for li, name in enumerate(self.link_names):
            T = Tl[name]
            a, b = self.link_vrange[li]
            Pv.append(self.V[a:b] @ T[:3, :3].T + T[:3, 3])
        sel = [self.facet_link == li for li in range(len(self.link_names))]
        cen = torch.empty_like(self.facet_cen)
        for li, name in enumerate(self.link_names):
            T = Tl[name]
            cen = torch.where(sel[li][:, None], self.facet_cen @ T[:3, :3].T + T[:3, 3], cen)
        return torch.cat(Pv, 0), cen


def _inside(px, py, ax, ay, bx, by, cx, cy):
    c0 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    c1 = (cx - bx) * (py - by) - (cy - by) * (px - bx)
    c2 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return (((c0 >= 0) & (c1 >= 0) & (c2 >= 0)) | ((c0 <= 0) & (c1 <= 0) & (c2 <= 0))) & (area.abs() > 1e-12)


def _segd2(px, py, ax, ay, bx, by):
    ux, uy = bx - ax, by - ay
    den = ux * ux + uy * uy
    t = ((px - ax) * ux + (py - ay) * uy) / den.clamp_min(1e-300)
    t = torch.where(den > 0, t.clamp(0.0, 1.0), torch.zeros_like(t))
    dx, dy = px - ax - t * ux, py - ay - t * uy
    return dx * dx + dy * dy


@torch.no_grad()
def cover_pairs(P, Tri, tri_ok, rad, cell, chunk=6_000_000):
    """exact superset search.  P (n,2) probes, Tri (m,3,2) projected triangles, tri_ok (m,) bool (z > 0, used).
    Returns inside (n,) bool (probe inside or on any used non-degenerate triangle, winding-independent) and the pairs
    (pi, ti) with probe not inside and dist(probe, triangle) < rad.  Triangles are registered in a uniform grid
    (cell) by their bbox dilated by rad; a triangle at distance < rad from a probe always shares the probe's cell.
    Two passes: inside test over all candidate pairs, distances only for the probes that are not inside."""
    dev = P.device
    n = len(P)
    inside = torch.zeros(n, dtype=torch.bool, device=dev)
    if n == 0:
        return inside, torch.zeros(0, dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.long, device=dev)
    pmin, pmax = P.min(0).values, P.max(0).values
    idx_t = torch.nonzero(tri_ok).squeeze(1)
    T = Tri[idx_t]
    lo = T.min(1).values - rad; hi = T.max(1).values + rad
    hit = (hi[:, 0] >= pmin[0]) & (lo[:, 0] <= pmax[0]) & (hi[:, 1] >= pmin[1]) & (lo[:, 1] <= pmax[1])
    idx_t, lo, hi = idx_t[hit], lo[hit], hi[hit]
    org = pmin - cell
    ncx = int(torch.floor((pmax[0] - org[0]) / cell).item()) + 2
    ncy = int(torch.floor((pmax[1] - org[1]) / cell).item()) + 2
    cl = torch.floor((lo - org) / cell).long(); ch = torch.floor((hi - org) / cell).long()
    cl[:, 0].clamp_(0, ncx - 1); ch[:, 0].clamp_(0, ncx - 1); cl[:, 1].clamp_(0, ncy - 1); ch[:, 1].clamp_(0, ncy - 1)
    bw = ch[:, 0] - cl[:, 0] + 1; bh = ch[:, 1] - cl[:, 1] + 1; nb = bw * bh
    rep = torch.repeat_interleave(torch.arange(len(idx_t), device=dev), nb)
    st = torch.repeat_interleave(torch.cumsum(nb, 0) - nb, nb)
    loc = torch.arange(len(rep), device=dev) - st
    dy = torch.div(loc, bw[rep], rounding_mode="floor"); dx = loc - dy * bw[rep]
    cid = (cl[rep, 1] + dy) * ncx + (cl[rep, 0] + dx)
    o = torch.argsort(cid, stable=True)
    cid_s, tri_s = cid[o], idx_t[rep[o]]
    ptr = torch.searchsorted(cid_s, torch.arange(ncx * ncy + 1, device=dev))
    pc = torch.floor((P - org) / cell).long()
    pcid = pc[:, 1].clamp(0, ncy - 1) * ncx + pc[:, 0].clamp(0, ncx - 1)
    cnt = ptr[pcid + 1] - ptr[pcid]
    TX, TY = Tri[..., 0], Tri[..., 1]
    PX, PY = P[:, 0], P[:, 1]

    def expand(sel):
        c = cnt[sel]
        csum = torch.cumsum(c, 0)
        start, m = 0, len(sel)
        while start < m:
            base = csum[start - 1].item() if start > 0 else 0
            stop = int(torch.searchsorted(csum, torch.tensor(base + chunk, device=dev, dtype=csum.dtype)).item()) + 1
            stop = min(max(stop, start + 1), m)
            cc = c[start:stop]
            pp = torch.repeat_interleave(sel[start:stop], cc)
            pst = torch.repeat_interleave(torch.cumsum(cc, 0) - cc, cc)
            k = torch.arange(len(pp), device=dev) - pst
            tt = tri_s[ptr[pcid[pp]] + k]
            yield pp, tt
            start = stop

    allp = torch.arange(n, device=dev)
    for pp, tt in expand(allp):
        ins = _inside(PX[pp], PY[pp], TX[tt, 0], TY[tt, 0], TX[tt, 1], TY[tt, 1], TX[tt, 2], TY[tt, 2])
        inside.index_fill_(0, pp[ins], True)
    out_p, out_t = [], []
    r2 = rad * rad
    for pp, tt in expand(torch.nonzero(~inside).squeeze(1)):
        px, py = PX[pp], PY[pp]
        ax, ay, bx, by, cx, cy = TX[tt, 0], TY[tt, 0], TX[tt, 1], TY[tt, 1], TX[tt, 2], TY[tt, 2]
        d2 = torch.minimum(torch.minimum(_segd2(px, py, ax, ay, bx, by), _segd2(px, py, bx, by, cx, cy)),
                           _segd2(px, py, cx, cy, ax, ay))
        keep = d2 < r2
        out_p.append(pp[keep]); out_t.append(tt[keep])
    if out_p:
        return inside, torch.cat(out_p), torch.cat(out_t)
    return inside, torch.zeros(0, dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.long, device=dev)


def seg_dist(X, A, B):
    AB = B - A
    den = (AB * AB).sum(-1)
    t = ((X - A) * AB).sum(-1) / den.clamp_min(1e-300)
    t = torch.where(den > 0, t.clamp(0.0, 1.0), torch.zeros_like(t))
    D = X - (A + t[..., None] * AB)
    return torch.sqrt((D * D).sum(-1).clamp_min(1e-300))


def seg_tri_dist(X, A, B, C):
    """distance from X to the triangle boundary (= distance to the triangle for X outside)."""
    return torch.minimum(torch.minimum(seg_dist(X, A, B), seg_dist(X, B, C)), seg_dist(X, C, A))


@dataclass
class Disc:
    K: dict                  # camera name -> (nE,) long samples per edge (fixed for the epoch)
    epoch: int = 0
    q_ref: np.ndarray | None = None


class SmoothRisk:

    def __init__(self, robot, cams, beliefs, tau_px=1.0, facing="F1", device="cpu", sigma=None):
        """robot: ArmModel; cams: list[CameraSpec]; beliefs: dict cam.name -> CameraBelief."""
        self.robot, self.cams, self.beliefs = robot, list(cams), beliefs
        self.dev = torch.device(device)
        self.geo = RobotGeom(robot, self.dev)
        self.tau = float(tau_px)                     # raster px
        self.facing = facing
        self.sigma = robot.area_sign if sigma is None else sigma
        self.stats = {}
        g = self.geo
        self.cam_data = {}
        for c in self.cams:
            b = beliefs[c.name]
            xs = np.concatenate([np.asarray(b.x_hat, np.float64)[None], np.asarray(b.draws, np.float64)], 0)
            Ds = [c.D(x).to(self.dev) for x in xs]
            static = c.static_links(robot)
            minfo = torch.tensor([0.0 if n in static else 1.0 for n in g.link_names], dtype=torch.float64, device=self.dev)
            bl = torch.as_tensor(np.asarray([b.b_link[n] for n in g.link_names], np.float64), device=self.dev)
            self.cam_data[c.name] = dict(xs=xs, D=Ds, minfo=minfo, b_link=bl,
                                         I_acq=torch.as_tensor(np.asarray(b.I_acq, np.float64), device=self.dev),
                                         b_blk=float(b.b_blk),
                                         W=torch.as_tensor(np.asarray(b.W, np.float64), device=self.dev),
                                         Kf=torch.as_tensor(np.asarray(c.K, np.float64), device=self.dev))
        self.n_eval = 0
        self.n_grad = 0
        self.debug = None
        self.no_edge_prune = False         # T1(b): certificate (i) off (omega = 0 edges sampled too)

    @torch.no_grad()
    def prepare_run(self, q_start, epoch=0) -> Disc:
        Tl = self.robot.fk(self._qt(q_start))
        Pb, _ = self.geo.base_points(Tl)
        K = {}
        for c in self.cams:
            cd = self.cam_data[c.name]
            Tc = c.T_cam(Tl, torch.as_tensor(cd["xs"][0], device=self.dev))
            Pc = Pb @ Tc[:3, :3].T + Tc[:3, 3]
            z = Pc[:, 2]
            ze = torch.minimum(z[self.geo.e_v[:, 0]], z[self.geo.e_v[:, 1]]).clamp_min(c.z_near)
            fr = c.s * float(c.K[0, 0])
            K[c.name] = torch.clamp(torch.ceil((fr * self.geo.e_len3d / ze) / (self.tau / 2.0)), min=1).long()
        return Disc(K=K, epoch=epoch, q_ref=np.asarray(q_start, np.float64).copy())

    def _qt(self, q):
        if isinstance(q, torch.Tensor):
            return self.robot.q_full(q.to(self.dev))
        return self.robot.q_full(torch.as_tensor(np.asarray(q, np.float64), device=self.dev))

    def _term(self, c, k, Tl, Pb, Cb, Ke, need_all=False, hard=False, timing=None):
        cd = self.cam_data[c.name]
        g, tau, dev = self.geo, self.tau, self.dev
        t0 = time.perf_counter()
        x = torch.as_tensor(cd["xs"][k], device=dev)
        Tc = c.T_cam(Tl, x)
        Pc = Pb @ Tc[:3, :3].T + Tc[:3, 3]
        Kf = cd["Kf"]
        fx, fy, sk, cx, cy = Kf[0, 0], Kf[1, 1], Kf[0, 1], Kf[0, 2], Kf[1, 2]
        z = Pc[:, 2]
        zs = z.clamp_min(c.z_near)
        uf = torch.stack([(fx * Pc[:, 0] + sk * Pc[:, 1]) / zs + cx, fy * Pc[:, 1] / zs + cy], 1)   # full res
        U = c.s * uf                                                                                # raster px
        # ---- facing
        Fa, Fb, Fc = g.F[:, 0], g.F[:, 1], g.F[:, 2]
        Atri = self.sigma * 0.5 * cross2(U[Fb] - U[Fa], U[Fc] - U[Fa])
        A_F = torch.zeros(g.nFacet, dtype=torch.float64, device=dev).index_add(0, g.facet_of_tri, Atri)
        if hard:
            S_F = (A_F > 0).double()
        else:
            ZF = (Cb @ Tc[:3, :3].T + Tc[:3, 3])[:, 2]
            fr = c.s * Kf[0, 0]
            if self.facing == "F1":
                tF = 2.0 * A_F * ZF / (fr * g.facet_diam)
            else:
                raise NotImplementedError("F2 facing is pre-registered but only built if T3 requires it")
            S_F = sstep(tF / tau)
        t1 = time.perf_counter()
        # ---- edge weights
        f1 = g.e_fac[:, 0]; f2 = g.e_fac[:, 1]
        S1 = S_F[f1]
        S2 = torch.where(g.e_bnd, torch.zeros_like(S1), S_F[f2.clamp_min(0)])
        om = torch.where(g.e_bnd, torch.ones_like(S1), (S1 - S2).abs())
        minfo = cd["minfo"][g.e_link]
        with torch.no_grad():
            ok_e = (z[g.e_v[:, 0]] > c.z_near) & (z[g.e_v[:, 1]] > c.z_near) & (minfo > 0)
            cand = (om > 0) & ok_e if not self.no_edge_prune else ok_e
            E = torch.nonzero(cand).squeeze(1)
        hi1 = g.e_bnd[E] | (S1[E] >= S2[E])
        sgn = torch.where(hi1, g.e_sign[E, 0], g.e_sign[E, 1])
        ia, ib = g.e_v[E, 0], g.e_v[E, 1]
        d01 = U[ib] - U[ia]
        L01 = torch.sqrt((d01 * d01).sum(1))
        perp = torch.stack([-d01[:, 1], d01[:, 0]], 1)
        nhat = perp / L01.clamp_min(1e-300)[:, None]
        n_out = (-self.sigma * sgn)[:, None] * nhat
        # samples
        if hard:
            with torch.no_grad():
                Kh = torch.clamp(torch.ceil(L01 / 0.05), min=1).long()
            Kk = Kh
        else:
            Kk = Ke[E]
        rep = torch.repeat_interleave(torch.arange(len(E), device=dev), Kk)
        mloc = torch.arange(len(rep), device=dev) - torch.repeat_interleave(torch.cumsum(Kk, 0) - Kk, Kk)
        tm = (mloc.double() + 0.5) / Kk[rep].double()
        um = U[ia][rep] * (1 - tm)[:, None] + U[ib][rep] * tm[:, None]
        probe = 1e-3 if hard else tau
        om_ = um + probe * n_out[rep]
        Wr, Hr = c.s * c.W, c.s * c.H
        if hard:
            inimg = lambda p: (p[:, 0] >= 0) & (p[:, 0] < Wr) & (p[:, 1] >= 0) & (p[:, 1] < Hr)
            beta = (inimg(um) & inimg(om_) & inimg(um - n_out[rep])).double()
        else:
            dx = torch.minimum(om_[:, 0], Wr - om_[:, 0]); dy = torch.minimum(om_[:, 1], Hr - om_[:, 1])
            beta = sstep(dx / tau) * sstep(dy / tau)
        t2 = time.perf_counter()
        # ---- coverage (no_grad index sets) + soft coverage with autograd
        with torch.no_grad():
            live = (beta > 0) if not need_all else torch.ones_like(beta, dtype=torch.bool)
            live_idx = torch.nonzero(live).squeeze(1)
            Tri = U[g.F]                                                   # (nF,3,2)
            tri_ok = (z[g.F] > c.z_near).all(1)
            inside, pi, ti = cover_pairs(om_[live_idx].detach(), Tri.detach(), tri_ok, (1e-3 if hard else tau),
                                         cell=(2.0 * tau if not hard else 2.0))
            keep_local = ~inside
            kept = live_idx[keep_local]
            remap = -torch.ones(len(live_idx), dtype=torch.long, device=dev)
            remap[keep_local] = torch.arange(int(keep_local.sum()), device=dev)
            pi2 = remap[pi]
        t3 = time.perf_counter()
        if hard:
            v = torch.ones(len(kept), dtype=torch.float64, device=dev)
        else:
            X = om_[kept][pi2]
            Tt = U[g.F[ti]]                                                   # with autograd (Tri above is no_grad)
            dist = seg_tri_dist(X, Tt[:, 0], Tt[:, 1], Tt[:, 2])
            # 1 - kappa = S(dist/tau - 1/2) exactly (symmetric smoothstep); evaluated directly to avoid 1 - (1 + ulp)
            fpair = sstep(dist / tau - 0.5)
            with torch.no_grad():
                zero_s = torch.zeros(len(kept), dtype=torch.bool, device=dev)
                zero_s.index_fill_(0, pi2[fpair <= 0], True)             # a factor exactly 0: v = 0, dv = 0
            logf = torch.log(torch.where(fpair > 0, fpair, torch.ones_like(fpair)))
            logv = torch.zeros(len(kept), dtype=torch.float64, device=dev).index_add(0, pi2, logf)
            v = torch.exp(logv) * (~zero_s).double()
        rk = rep[kept]
        Ek = E[rk]
        ds = (L01[rk] / c.s) / Kk[rk].double()                                 # full-res px
        bl = cd["b_link"][g.e_link[Ek]]
        wl = om[Ek] * v * ds * minfo[Ek] / bl ** 2                             # without beta (for L_all)
        w = wl * beta[kept]
        # ---- boundary Jacobian (closed form) at the perspective-correct sample
        t_k = tm[kept]
        za, zb = zs[ia][rk], zs[ib][rk]
        al = ((1 - t_k) / za) / ((1 - t_k) / za + t_k / zb)
        pc = al[:, None] * Pc[ia][rk] + (1 - al)[:, None] * Pc[ib][rk]
        pb = al[:, None] * Pb[ia][rk] + (1 - al)[:, None] * Pb[ib][rk]
        X_, Y_, Z_ = pc[:, 0], pc[:, 1], pc[:, 2]
        u_ = (fx * X_ + sk * Y_) / Z_ + cx; v_ = fy * Y_ / Z_ + cy
        ne = nhat[rk]
        a0 = ne[:, 0] * fx / Z_
        a1 = (ne[:, 0] * sk + ne[:, 1] * fy) / Z_
        a2 = -(ne[:, 0] * (u_ - cx) + ne[:, 1] * (v_ - cy)) / Z_
        am = torch.stack([a0, a1, a2], 1)
        pt = c.p_tilde(Tl, pb)                                                  # (n,4)
        Dp = torch.einsum("jab,nb->nja", cd["D"][k], pt)[:, :, :3]
        J = torch.einsum("na,nja->nj", am, Dp)
        H = J.T @ (w[:, None] * J)
        A = cd["b_blk"] * torch.linalg.inv(cd["I_acq"] + H)
        r = torch.trace(cd["W"] @ A)
        t4 = time.perf_counter()
        if timing is not None:
            for key, dtm in (("project+facing", t1 - t0), ("edges+samples", t2 - t1), ("coverage_nograd", t3 - t2),
                             ("soft_cov+J+H", t4 - t3)):
                timing[key] = timing.get(key, 0.0) + dtm
        if self.debug is not None:
            with torch.no_grad():
                pos = (w > 0) & (~g.e_bnd[Ek])
                gap = (S1[Ek] - S2[Ek]).abs()[pos]
                self.debug.setdefault("min_gap", []).append(float(gap.min().item()) if len(gap) else 1.0)
                self.debug.setdefault("n_pos", []).append(int(pos.sum().item()))
        feats = None
        if need_all:
            dsm = ds * om[Ek] * minfo[Ek]
            L_in = (dsm * v * beta[kept]).sum()
            L_all = (dsm * v).sum()
            with torch.no_grad():
                pos = w > 0
                spac = (L01[rk][pos] / Kk[rk][pos].double()).max().item() if bool(pos.any()) else 0.0
                partial = (w > 0) & ((om[Ek] * v * beta[kept]) < 1 - 1e-12)
            feats = dict(L_in=L_in, L_all=L_all, tau_c=1.0 - L_in / L_all.clamp_min(1e-300), n_samples=int(len(kept)),
                         n_cand_edges=int(len(E)), n_cand_samples=int(len(rep)),
                         band_mass=float((dsm * v * beta[kept])[partial].sum().item()),
                         max_spacing=spac, L_in_value=float(L_in.item()))
        return r, feats

    def __call__(self, q, disc: Disc, terms="all", feats=True, hard=False, timing=None):
        """R~ (scalar, autograd w.r.t. q if q requires grad) and features of the x^ term per camera."""
        Tl = self.robot.fk(self._qt(q))
        Pb, Cb = self.geo.base_points(Tl)
        tot = 0.0
        out = {}
        for c in self.cams:
            cd = self.cam_data[c.name]
            ks = range(len(cd["xs"])) if terms == "all" else list(terms)
            rs = []
            for k in ks:
                r, f = self._term(c, k, Tl, Pb, Cb, disc.K[c.name] if disc is not None else None,
                                  need_all=(feats and k == 0), hard=hard, timing=timing)
                rs.append(r)
                if f is not None:
                    out[c.name] = f
            Rc = torch.stack(rs).mean()
            out.setdefault("R_cam", {})[c.name] = Rc
            tot = tot + Rc
        self.n_eval += 1
        return tot, out

    @torch.no_grad()
    def value(self, q, disc, terms="all", feats=True, hard=False):
        R, F = self(q, disc, terms=terms, feats=feats, hard=hard)
        return float(R), F
