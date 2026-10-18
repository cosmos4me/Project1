"""nl_ls: robust silhouette fit of the camera pose.

nl_fit   Gauss-Newton with a Huber loss (0.5 px) on the area-attributed boundary displacement of every contour pixel of
         the rendered silhouette, all views at once, from the initial pose.
nl_ls    then per (view, link) Laplace scales b = sqrt(m^2 + 0.25^2), m = ds-weighted mean |t| of the group, re-estimated
         in 3 outer rounds with Gauss-Newton-Huber steps on the reweighted objective."""
import math

import numpy as np

from .geometry import FullContour, frame_mask
from .se3 import pose_of


def cells(R: np.ndarray, loops: list):
    """Voronoi cells of the contour: for every pixel the index (into the concatenated loops) of its nearest contour
    pixel, or a value >= n for pixels nearest to the part of R that lies on the image frame (pseudo-boundary with no
    error: the silhouette continues outside the image).  Returns (cell index image, n)."""
    from scipy import ndimage
    R = np.asarray(R, bool)
    pts = np.concatenate(loops); fpts = np.argwhere(R & frame_mask(R.shape))
    idx = -np.ones(R.shape, np.int64)
    if len(fpts):
        idx[fpts[:, 0], fpts[:, 1]] = len(pts) + np.arange(len(fpts))
    idx[pts[:, 0], pts[:, 1]] = np.arange(len(pts))
    _, (iy, ix) = ndimage.distance_transform_edt(idx < 0, return_indices=True)
    return idx[iy, ix], len(pts)


def detached(mask: np.ndarray, R: np.ndarray):
    """residual components that touch no contour of R: additions away from R and holes inside R."""
    from scipy import ndimage
    st = np.ones((3, 3), bool)
    m, R = np.asarray(mask, bool), np.asarray(R, bool)
    out = []
    for comp, near in ((m & ~R, ndimage.binary_dilation(R, st)), (R & ~m, ndimage.binary_dilation(~R, st))):
        lab, n = ndimage.label(comp, st)
        touch = set(np.unique(lab[near & (lab > 0)]).tolist())
        keep = [k for k in range(1, n + 1) if k not in touch]
        out.append(np.isin(lab, keep) if keep else np.zeros(m.shape, bool))
    return out[0], out[1]


def attributed_displacement(mask: np.ndarray, R: np.ndarray, loops: list, keep_add, keep_del) -> list:
    """per contour pixel: (#additions - #deletions) among the attached, non-frame residual pixels of its cell."""
    m, R = np.asarray(mask, bool), np.asarray(R, bool)
    near, N = cells(R, loops)
    fr = frame_mask(R.shape)
    add = m & ~R & ~keep_add & ~fr & (near < N)                 # residual nearest to the frame part of R: not attributed
    dele = R & ~m & ~keep_del & ~fr & (near < N)
    d = (np.bincount(near[add], minlength=N)[:N] - np.bincount(near[dele], minlength=N)[:N]).astype(np.float64)
    return np.split(d, np.cumsum([len(l) for l in loops])[:-1])


def huber_solve(J, t, w, c=0.5, A0=None, b0=None, iters=60, x0=None):
    """argmin_x  sum_i w_i rho_c(J_i x - t_i) + 0.5 (x - b0)^T A0 (x - b0)   with rho_c = 2*Huber_c / (2c) scaled so
    rho(r) ~ |r| for |r| >> c and r^2/(2c) near 0.  IRLS."""
    n = J.shape[1]
    A0 = np.zeros((n, n)) if A0 is None else A0
    b0 = np.zeros(n) if b0 is None else b0
    if x0 is None:
        H = J.T @ (w[:, None] * J) / c + A0
        x = np.linalg.solve(H + 1e-12 * np.trace(H) / n * np.eye(n), J.T @ (w * t) / c + A0 @ b0)
    else:
        x = np.asarray(x0, np.float64).copy()
    for _ in range(iters):
        r = J @ x - t
        om = w / np.maximum(np.abs(r), c)                  # IRLS weight for rho(r) = r^2/(2c) | |r| - c/2
        H = J.T @ (om[:, None] * J) + A0
        xn = np.linalg.solve(H + 1e-12 * np.trace(H) / n * np.eye(n), J.T @ (om * t) + A0 @ b0)
        if np.max(np.abs(xn - x)) < 1e-10:
            x = xn
            break
        x = xn
    return x


def resid(geo, q, T, mask, dof):
    Cn = FullContour(geo, q, T, dof)
    ka, kd = detached(mask, Cn.m)
    d = np.concatenate(attributed_displacement(mask, Cn.m, Cn.loops, ka, kd))
    return Cn, d / Cn.ds, int((Cn.m != mask).sum())


def nl_fit(geo, views, dof0, iters=25, c=0.5, verbose=False):
    """views: list of (q, mask).  returns dof, history"""
    dof = np.asarray(dof0, np.float64).copy(); hist = []
    def evaluate(d):
        T = pose_of(d)
        Js, ts, ws, L = [], [], [], 0
        for q, m in views:
            Cn, t, xor = resid(geo, q, T, m, d)
            Js.append(Cn.J); ts.append(t); ws.append(Cn.ds); L += xor
        return np.concatenate(Js), np.concatenate(ts), np.concatenate(ws), L
    J, t, w, L = evaluate(dof)
    hist.append((dof.copy(), L))
    for it in range(iters):
        step = huber_solve(J, t, w, c=c)
        a = 1.0; acc = False
        for _ in range(6):
            dn = dof + a * step
            Jn, tn, wn, Ln = evaluate(dn)
            if Ln <= L:
                acc = True; break
            a *= 0.5
        if not acc:
            break
        dof, J, t, w, L = dn, Jn, tn, wn, Ln
        hist.append((dof.copy(), L))
        if verbose:
            print(it, L, a, np.round(step, 5), flush=True)
        if np.max(np.abs(a * step)) < 1e-6:
            break
    return dof, hist


Q0 = 0.25


def evaluate(geo, views, d):
    T = pose_of(d); out = []
    for v, (q, m) in enumerate(views):
        Cn, t, xor = resid(geo, q, T, m, d)
        out.append((Cn.J, t, Cn.ds, Cn.li, xor))
    return out


def group_scales(ev, nmin=20):
    b = {}
    for v, (J, t, ds, li, _) in enumerate(ev):
        pooled = float(np.average(np.abs(t), weights=ds))
        for l in np.unique(li):
            s = li == l
            m = float(np.average(np.abs(t[s]), weights=ds[s])) if s.sum() >= nmin else pooled
            b[(v, int(l))] = math.sqrt(m * m + Q0 * Q0)
    return b


def stack(ev, b):
    J = np.concatenate([e[0] for e in ev]); t = np.concatenate([e[1] for e in ev])
    w = np.concatenate([e[2] / np.array([b.get((v, int(l)), 1.0) for l in e[3]]) for v, e in enumerate(ev)])
    return J, t, w


def objective(ev, b):
    return float(sum((e[2] * np.abs(e[1]) / np.array([b.get((v, int(l)), 1.0) for l in e[3]])).sum() for v, e in enumerate(ev)))


def nl_ls(geo, views, dof0, iters=30, outer=3, inner=10, c=0.5, dof_init=None):
    """returns (dof_nl, dof_ls, info)"""
    d_nl = np.asarray(dof_init, np.float64) if dof_init is not None else nl_fit(geo, views, dof0, iters=iters, c=c)[0]
    dof = d_nl.copy(); hist = []
    for o in range(outer):
        ev = evaluate(geo, views, dof); b = group_scales(ev)
        L = objective(ev, b)
        for it in range(inner):
            J, t, w = stack(ev, b)
            step = huber_solve(J, t, w, c=c)
            a = 1.0; acc = False
            for _ in range(6):
                dn = dof + a * step; evn = evaluate(geo, views, dn); Ln = objective(evn, b)
                if Ln <= L:
                    acc = True; break
                a *= 0.5
            if not acc:
                break
            dof, ev, L = dn, evn, Ln
            if np.max(np.abs(a * step)) < 1e-6:
                break
        hist.append({"outer": o, "L": L, "b": {f"{k[0]}:{k[1]}": round(v, 3) for k, v in b.items()}})
    return d_nl, dof, {"hist": hist}
