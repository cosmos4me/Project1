"""D v2 exact-geometry clearance constraints (GT-free, a-priori constants only).

Geometry = the geometry of the RELEASED checks:
  * environment / path (MPLib 0.2.1 path_is_feasible, use_convex=False): the URDF collision MESHES vs the released workspace point
    cloud (MPLib octree, 1 mm voxels).  FCL BVH collision is a surface test, so the faithful clearance is the unsigned distance of
    each workspace point to each link's mesh surface minus the voxel half-diagonal.
  * self collision (PyMP computeCollisions, use_convex=True): the convex hulls '<mesh>.convex.stl' of the SRDF-enabled link pairs.
Fields (cached, computed once from the meshes): per link an unsigned distance grid of the mesh surface (env) and a signed distance
grid of the convex hull (self; inside exact via the face planes, outside the distance to dense hull-surface samples), spacing H, on
the link bounding box padded by PAD.  Lookups: trilinear (torch grid_sample, float64), differentiable in q through a batched FK;
points outside a padded box get the value PAD (a valid lower bound, no gradient).
Conservative margins (a priori):  trilinear interpolation of a 1-Lipschitz field errs by <= (sqrt(3)/2) H;  dense-sample distance
over-estimates by <= EPS_S;  MPLib voxel half-diagonal VOX;  self queries use hull-surface samples of spacing S_SELF (covering
radius <= S_SELF).  Path: straight joint segment q_now -> q sampled so that no robot point moves more than DELTA between samples
(a-priori per-joint sweep radii from the kinematic chain and mesh extents); the path margin adds DELTA / 2.
Gradients: Danskin - the gradient of each min-row is the gradient of its arg-min term, recomputed with autograd."""
from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from . import config
from . import robot as _robot

URDF = _robot.active().urdf
SRDF = _robot.active().srdf
CACHE = config.fields_cache()
GEOM = ["link_base", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "xarm_gripper_base_link", "left_finger",
        "right_finger"]
ARM = ["link_base"] + [f"link{i}" for i in range(1, 8)]
H = 0.002                    # grid spacing (m)
PAD = 0.03                   # grid padding around each link (m); outside value = PAD (lower bound)
EPS_S = 0.0005               # bound of the dense-sample distance over-estimate (mesh sampling density, see build)
VOX = 0.5 * math.sqrt(3) * 1e-3   # MPLib point-cloud voxel (1 mm) half-diagonal
S_SELF = 0.003               # hull-surface sample spacing for self queries (m)
DELTA = 0.005                # max robot-point motion between path samples (m)
NUM = 2e-4                   # numerical slack (m)
INTERP = 0.5 * math.sqrt(3) * H    # trilinear error bound of a 1-Lipschitz field: sum_i w_i |x - c_i| <= sqrt(sum_axes u(1-u) H^2)
                                   # <= (sqrt(3)/2) H  (Jensen); corrected 2026-10-17 13:00 from the loose sqrt(3) H
M_ENV = INTERP + EPS_S + VOX + NUM                         # goal environment margin  (~3.3 mm)
M_SELF = INTERP + EPS_S + S_SELF + NUM                     # goal self margin        (~5.4 mm)
M_PENV = M_ENV + DELTA / 2                                   # path environment margin (~5.8 mm)
M_PSELF = M_SELF + DELTA / 2                                 # path self margin        (~7.9 mm)


def _origin(xyz, rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    T = np.eye(4); T[:3, :3] = Rz @ Ry @ Rx; T[:3, 3] = xyz
    return T


def link_meshes():
    import trimesh
    root = ET.parse(URDF).getroot()
    out = {}
    for l in root.findall("link"):
        name = l.get("name")
        if name not in GEOM:
            continue
        c = l.find("collision")
        o = c.find("origin")
        T = _origin([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()],
                    [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
        fn = URDF.parent / c.find("geometry")[0].get("filename")
        mesh = trimesh.load(str(fn), force="mesh"); mesh.apply_transform(T)
        hull = trimesh.load(str(fn) + ".convex.stl", force="mesh"); hull.apply_transform(T)
        out[name] = (mesh, hull)
    return out


def enabled_pairs():
    dis = {frozenset((d.get("link1"), d.get("link2"))) for d in ET.parse(SRDF).getroot().findall("disable_collisions")}
    return [(a, b) for i, a in enumerate(GEOM) for b in GEOM[i + 1:] if frozenset((a, b)) not in dis]


def build_fields(cache=CACHE, n_per_m2=6.0e6, seed=0):
    """one-time geometry preprocessing (no scene data): grids + hull samples -> npz."""
    import trimesh
    from scipy.spatial import cKDTree
    t0 = time.time()
    rng = np.random.default_rng(seed)
    arrs, meta = {}, {}
    for name, (mesh, hull) in link_meshes().items():
        lo = np.minimum(mesh.bounds[0], hull.bounds[0]) - PAD
        hi = np.maximum(mesh.bounds[1], hull.bounds[1]) + PAD
        shape = np.ceil((hi - lo) / H).astype(int) + 1
        hi = lo + (shape - 1) * H
        xs = [lo[k] + H * np.arange(shape[k]) for k in range(3)]
        G = np.stack(np.meshgrid(*xs, indexing="ij"), -1).reshape(-1, 3)
        # env: unsigned distance to the mesh surface (dense samples + vertices)
        n = int(max(2e5, min(3e6, mesh.area * n_per_m2)))
        smp, _ = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 31)))
        P = np.concatenate([smp, mesh.vertices])
        udf, _ = cKDTree(P).query(G, workers=8)
        # self: signed distance of the convex hull (inside exact: max face-plane distance; outside: dense hull samples)
        nrm = hull.face_normals; off = np.einsum("ij,ij->i", nrm, hull.triangles[:, 0])
        phi = np.full(len(G), -np.inf)
        for s in range(0, len(G), 200000):
            phi[s:s + 200000] = (G[s:s + 200000] @ nrm.T - off).max(1)
        nh = int(max(1e5, min(2e6, hull.area * n_per_m2)))
        hs, _ = trimesh.sample.sample_surface(hull, nh, seed=int(rng.integers(1 << 31)))
        dh, _ = cKDTree(np.concatenate([hs, hull.vertices])).query(G, workers=8)
        sdf = np.where(phi < 0, phi, dh)
        # hull-surface samples for self queries (spacing ~S_SELF) + vertices
        ns = int(max(200, hull.area / S_SELF ** 2))
        qs, _ = trimesh.sample.sample_surface_even(hull, ns, radius=0.5 * S_SELF, seed=int(rng.integers(1 << 31)))
        qs = np.concatenate([qs, hull.vertices])
        arrs[f"{name}__udf"] = udf.reshape(shape).astype(np.float32)
        arrs[f"{name}__sdf"] = sdf.reshape(shape).astype(np.float32)
        arrs[f"{name}__lo"] = lo; arrs[f"{name}__hi"] = hi
        arrs[f"{name}__selfpts"] = qs
        arrs[f"{name}__rho"] = np.array(float(np.linalg.norm(np.concatenate([mesh.vertices, hull.vertices]), axis=1).max()))
        meta[name] = dict(shape=shape.tolist(), n_env_samples=len(P), n_hull_samples=nh, n_self_pts=len(qs), area=float(mesh.area))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrs)
    return meta, time.time() - t0


class Clearance:
    def __init__(self, fk, env_points, device="cuda:0", cache=CACHE):
        self.dev = torch.device(device)
        self.fk = fk
        if not Path(cache).exists():
            build_fields(cache)
        z = np.load(cache)
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=self.dev)  # noqa: E731
        self.udf = {n: t(z[f"{n}__udf"])[None, None] for n in GEOM}
        self.sdf = {n: t(z[f"{n}__sdf"])[None, None] for n in GEOM}
        self.lo = {n: t(z[f"{n}__lo"]) for n in GEOM}
        self.hi = {n: t(z[f"{n}__hi"]) for n in GEOM}
        self.spts = {n: t(z[f"{n}__selfpts"]) for n in GEOM}
        self.rho = {n: float(z[f"{n}__rho"]) for n in GEOM}
        self.env = t(env_points)
        self.pairs = enabled_pairs()
        self.origins = [o.to(self.dev) for o in fk.origins]
        self.tool = {k: (p, O.to(self.dev)) for k, (p, O) in fk.tool.items()}
        self.radii = self._sweep_radii()

    # ---------------------------------------------------------------- kinematics
    def fk_batch(self, Q7):
        B = Q7.shape[0]
        T = torch.eye(4, dtype=torch.float64, device=self.dev).expand(B, 4, 4)
        out = {"link_base": T}
        for i, name in enumerate(ARM[1:]):
            c, s = torch.cos(Q7[:, i]), torch.sin(Q7[:, i])
            z, o = torch.zeros_like(c), torch.ones_like(c)
            Rz = torch.stack([torch.stack([c, -s, z, z], -1), torch.stack([s, c, z, z], -1),
                              torch.stack([z, z, o, z], -1), torch.stack([z, z, z, o], -1)], -2)
            T = T @ self.origins[i] @ Rz
            out[name] = T
        for name, (parent, O) in self.tool.items():
            out[name] = out[parent] @ O
        return out

    def _sweep_radii(self):
        """a-priori bound r_j: any geometry point distal to joint j is within r_j of joint j's axis origin (triangle inequality
        over the joint-origin translations of the chain + the link's maximal vertex norm)."""
        tr = [float(torch.linalg.norm(o[:3, 3])) for o in self.origins]          # joint1..joint7 origin translations
        tool_tr = {k: float(torch.linalg.norm(O[:3, 3])) for k, (p, O) in self.tool.items()}
        chain_after7 = max(self.rho["link7"], tool_tr["link_eef"] + tool_tr["xarm_gripper_base_link"] + max(
            self.rho["xarm_gripper_base_link"], tool_tr["left_finger"] + self.rho["left_finger"],
            tool_tr["right_finger"] + self.rho["right_finger"]))
        r = []
        for j in range(7):                       # joint j+1 rotates links j+1..7 (+ tools)
            best = 0.0
            for l in range(j + 1, 8):
                d = sum(tr[j + 1:l]) + (self.rho[f"link{l}"] if l < 7 else chain_after7)
                best = max(best, d)
            r.append(best)
        return np.asarray(r)

    def n_path(self, q_now, q):
        dq = np.abs(np.r_[np.asarray(q, np.float64)[:6], 0.0] - np.r_[np.asarray(q_now, np.float64)[:6], 0.0])
        return max(1, int(math.ceil(float(self.radii @ dq) / DELTA)))

    # ---------------------------------------------------------------- field lookups
    def _look(self, field, name, P):
        """P (..., 3) link-frame points -> trilinear values (...), outside the padded box -> PAD."""
        lo, hi = self.lo[name], self.hi[name]
        u = 2.0 * (P - lo) / (hi - lo) - 1.0
        inside = (u.abs() <= 1.0).all(-1)
        g = u[..., [2, 1, 0]].reshape(1, -1, 1, 1, 3)       # grid_sample order (W, H, D) = (z, y, x) of the (x, y, z) grid
        v = Fn.grid_sample(field[name], g, mode="bilinear", padding_mode="border", align_corners=True).reshape(P.shape[:-1])
        return torch.where(inside, v, torch.full_like(v, PAD))

    @staticmethod
    def _to_local(T, X):
        """T (B,4,4), X (N,3) world -> (B,N,3) in the frame of T."""
        R, t = T[:, :3, :3], T[:, :3, 3]
        return torch.einsum("bji,bnj->bni", R, X[None] - t[:, None])

    def _look_masked(self, field, name, P, mask):
        """lookups only where mask (B, N) is true; elsewhere PAD (those points are farther than rho + PAD from the link
        origin, hence farther than PAD from the geometry: PAD is a valid lower bound there)."""
        out = torch.full(P.shape[:-1], PAD, dtype=P.dtype, device=P.device)
        if mask.any():
            out[mask] = self._look(field, name, P[mask])
        return out

    def env_min(self, Tl):
        """per link: min over workspace points of the mesh UDF -> (B, L), argmin point index (B, L)."""
        vals, idx = [], []
        for n in GEOM:
            T = Tl[n]
            near = torch.cdist(T[:, :3, 3][:, None], self.env[None].expand(T.shape[0], -1, -1))[:, 0] < self.rho[n] + PAD
            v = self._look_masked(self.udf, n, self._to_local(T, self.env), near)
            m, i = v.min(1); vals.append(m); idx.append(i)
        return torch.stack(vals, 1), torch.stack(idx, 1)

    def self_min(self, Tl):
        """per enabled pair (both directions): min hull SDF -> (B, P), arg-min (direction, point index)."""
        vals, idx = [], []
        for a, b in self.pairs:
            best = None
            for src, dst, dflag in ((a, b, 0), (b, a, 1)):
                Ts, Td = Tl[src], Tl[dst]
                B = Ts.shape[0]
                close = torch.linalg.norm(Ts[:, :3, 3] - Td[:, :3, 3], dim=1) < self.rho[src] + self.rho[dst] + PAD
                if not close.any():
                    v = torch.full((B, len(self.spts[src])), PAD, dtype=torch.float64, device=self.dev)
                else:
                    Xw = torch.einsum("bij,nj->bni", Ts[:, :3, :3], self.spts[src]) + Ts[:, None, :3, 3]
                    Xl = torch.einsum("bji,bnj->bni", Td[:, :3, :3], Xw - Td[:, None, :3, 3])
                    near = (torch.linalg.norm(Xl, dim=-1) < self.rho[dst] + PAD) & close[:, None]
                    v = self._look_masked(self.sdf, dst, Xl, near)
                m, i = v.min(1)
                cand = (m, i, torch.full_like(i, dflag))
                if best is None:
                    best = cand
                else:
                    sel = cand[0] < best[0]
                    best = tuple(torch.where(sel, c, bb) for c, bb in zip(cand, best))
            vals.append(best[0]); idx.append(torch.stack([best[2], best[1]], -1))
        return torch.stack(vals, 1), torch.stack(idx, 1)

    # ---------------------------------------------------------------- constraint rows
    def _q7(self, q):
        q = torch.as_tensor(np.asarray(q, np.float64), device=self.dev) if not torch.is_tensor(q) else q
        return torch.cat([q.reshape(-1)[:6], torch.zeros(1, dtype=torch.float64, device=self.dev)])

    @torch.no_grad()
    def rows(self, q, q_now, chunk=256):
        """g rows (<= 0 feasible) + names + arg-min info.  Rows: CE:<link> goal environment, CS:<a>|<b> goal self,
        PE:<link> path environment (min over samples), PS path self (min over samples and pairs)."""
        q7, q07 = self._q7(q), self._q7(q_now)
        Tl = self.fk_batch(q7[None])
        e, ei = self.env_min(Tl); s, si = self.self_min(Tl)
        N = self.n_path(q_now, q)
        ts = torch.arange(1, N + 1, dtype=torch.float64, device=self.dev) / N
        pe_best, pe_arg = None, None
        ps_best, ps_arg = None, None
        for c in range(0, N, chunk):
            tt = ts[c:c + chunk]
            Q = q07[None] + tt[:, None] * (q7 - q07)[None]
            T2 = self.fk_batch(Q)
            pe, pei = self.env_min(T2)                    # (b, L)
            pmin, pk = pe.min(0)                          # per link over samples
            arg = torch.stack([tt[pk], pei[pk, torch.arange(len(GEOM), device=self.dev)].double()], -1)
            if pe_best is None:
                pe_best, pe_arg = pmin, arg
            else:
                sel = pmin < pe_best
                pe_best = torch.where(sel, pmin, pe_best); pe_arg = torch.where(sel[:, None], arg, pe_arg)
            sv, sidx = self.self_min(T2)                   # (b, P)
            flat = sv.reshape(-1); k = int(flat.argmin())
            bi, pi = divmod(k, sv.shape[1])
            if ps_best is None or flat[k] < ps_best:
                ps_best = flat[k]; ps_arg = (float(tt[bi]), pi, int(sidx[bi, pi, 0]), int(sidx[bi, pi, 1]))
        g = torch.cat([M_ENV - e[0], M_SELF - s[0], M_PENV - pe_best, (M_PSELF - ps_best).reshape(1)])
        names = ([f"CE:{n}" for n in GEOM] + [f"CS:{a}|{b}" for a, b in self.pairs] + [f"PE:{n}" for n in GEOM] + ["PS"])
        info = dict(ei=ei[0].cpu().numpy(), si=si[0].cpu().numpy(), pe=pe_arg.cpu().numpy(), ps=ps_arg, N=N,
                    min_env=float(e.min()), min_self=float(s.min()), min_penv=float(pe_best.min()), min_pself=float(ps_best))
        return g.cpu().numpy(), names, info



    def rows_jac(self, q, q_now, info):
        """Danskin Jacobian: the gradient of every min-row is minus the gradient of its arg-min term.  All arg-min terms are
        recomputed in ONE graph (one batched FK over the distinct path parameters t), then one backward pass per row."""
        x = torch.as_tensor(np.asarray(q, np.float64)[:6], device=self.dev).clone().requires_grad_(True)
        q07 = self._q7(q_now)
        q7 = torch.cat([x, torch.zeros(1, dtype=torch.float64, device=self.dev)])
        spec = []                                   # (kind, t, link | (a, b, dflag), point index)
        for li, n in enumerate(GEOM):
            spec.append(("e", 1.0, n, int(info["ei"][li])))
        for pi, (a, b) in enumerate(self.pairs):
            spec.append(("s", 1.0, (a, b, int(info["si"][pi, 0])), int(info["si"][pi, 1])))
        for li, n in enumerate(GEOM):
            spec.append(("e", float(info["pe"][li, 0]), n, int(info["pe"][li, 1])))
        t, pi, dflag, p = info["ps"]
        a, b = self.pairs[pi]
        spec.append(("s", float(t), (a, b, int(dflag)), int(p)))
        ts = sorted({sp[1] for sp in spec})
        tt = torch.as_tensor(ts, dtype=torch.float64, device=self.dev)
        Tl = self.fk_batch(q07[None] + tt[:, None] * (q7 - q07)[None])
        vals = []
        for kind, tv, what, pidx in spec:
            k = ts.index(tv)
            if kind == "e":
                v = self._look(self.udf, what, self._to_local(Tl[what][k:k + 1], self.env[pidx:pidx + 1]))[0, 0]
            else:
                a, b, d = what
                src, dst = (a, b) if d == 0 else (b, a)
                Ts, Td = Tl[src][k], Tl[dst][k]
                xw = Ts[:3, :3] @ self.spts[src][pidx] + Ts[:3, 3]
                v = self._look(self.sdf, dst, (Td[:3, :3].T @ (xw - Td[:3, 3]))[None])[0]
            vals.append(v)
        J = np.zeros((len(vals), 6))
        for r, v in enumerate(vals):
            if v.requires_grad:
                g, = torch.autograd.grad(v, x, retain_graph=True, allow_unused=True)
                if g is not None:
                    J[r] = -g.detach().cpu().numpy()
        return J
