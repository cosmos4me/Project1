"""Laplace posterior of the camera extrinsic x (6-d exponential coordinates of camera <- base) after the acquired views.

view 1  (same as the experiments' l2_at):  residuals t_i of the view-1 fit at x_hat; per-link Laplace scale
        b_l = sqrt(m_l^2 + Q0^2), m_l = ds-weighted mean |t_i| over the link's contour pixels (the view's pooled mean if
        the link has < NMIN pixels), Q0 = 0.25 px
        (mean quantisation error of a binary mask); I = sum_i J_i (ds_i / b_{l_i}^2) J_i^T; correlation factor
        b_blk = automatic circular-block-bootstrap block length of the residual sequence; Sigma = b_blk I^-1.
view k  (k >= 2): I_k = view-1 information (all 8 links) + views 2..k information (moving links 1-7 only, the
        links the risk counts; the base looks the same in every view) at the k-view x_hat with the view-1 b_l, b_blk.
draws:  M = 8 Laplace samples, seed [SEED0, key] for view 1 and [SEED0, key, k] for k views."""
from __future__ import annotations

import math

import numpy as np

from .fit import resid
from .geometry import FullContour
from .se3 import pose_of


def block_length_cb(x) -> float:
    """automatic circular-block-bootstrap block length (Politis & White 2004; Patton, Politis & White 2009)."""
    x = np.asarray(x, np.float64); n = len(x)
    e = x - x.mean()
    if n < 16 or float(e @ e) <= 0:
        return 1.0
    b_max = math.ceil(min(3 * math.sqrt(n), n / 3))
    kn = max(5, int(math.log10(n)))
    m_max = int(math.ceil(math.sqrt(n))) + kn
    acv = np.array([float(e[k:] @ e[:n - k]) / n for k in range(m_max + kn + 1)])
    rho = np.abs(acv / acv[0])
    cv = 2.0 * math.sqrt(math.log10(n) / n)
    m_hat = None
    for k in range(1, m_max + 1):                      # first run of kn insignificant autocorrelations
        if np.all(rho[k:k + kn] < cv):
            m_hat = k - 1
            break
    M = min(2 * max(m_hat, 1), m_max) if m_hat is not None else m_max
    lam = lambda t: 1.0 if t <= 0.5 else 2.0 * (1.0 - t)
    G = sum(2.0 * lam(k / M) * k * acv[k] for k in range(1, M + 1))
    g0 = acv[0] + sum(2.0 * lam(k / M) * acv[k] for k in range(1, M + 1))
    D = 4.0 / 3.0 * g0 ** 2
    if D <= 0:
        return float(b_max)
    b = (2.0 * G ** 2 / D) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(min(max(b, 1.0), b_max))


Q0 = 0.25
NMIN = 20
SEED0 = 20261015
M_DRAWS = 8


def draws(x_hat, Sigma, rng_key, n=M_DRAWS):
    rng = np.random.default_rng(rng_key)
    L = np.linalg.cholesky(0.5 * (np.asarray(Sigma) + np.asarray(Sigma).T))
    z = rng.standard_normal((n, 6))
    return np.asarray(x_hat, np.float64)[None] + z @ L.T


def view1(geo, q1, m1, x):
    """returns dict(I, Sigma, b_link (8,), b_blk, n_px)."""
    x = np.asarray(x, np.float64)
    Cn, r, _ = resid(geo, q1, pose_of(x), m1, x)
    J, ds, li = Cn.J, Cn.ds, Cn.li
    pooled = float(np.average(np.abs(r), weights=ds))
    bg = np.zeros(8)
    for l in range(8):
        s = li == l
        m = float(np.average(np.abs(r[s]), weights=ds[s])) if s.sum() >= NMIN else pooled
        bg[l] = math.sqrt(m * m + Q0 * Q0)
    b = bg[li.astype(int)]
    I = (J * (ds / b ** 2)[:, None]).T @ J
    b_blk = float(block_length_cb(r))
    return dict(I=I, Sigma=b_blk * np.linalg.inv(I), b_link=bg, b_blk=b_blk, n_px=int(len(r)))


def info_view(geo, q, x, b_link, lmin):
    x = np.asarray(x, np.float64)
    Cn = FullContour(geo, q, pose_of(x), x)
    if not Cn.ok:
        return np.zeros((6, 6)), 0
    li = Cn.li.astype(int); s = li >= lmin
    J, ds = Cn.J[s], Cn.ds[s]
    return (J * (ds / b_link[li[s]] ** 2)[:, None]).T @ J, int(s.sum())


def k_views(geo, views, x_hat, b_link, b_blk):
    I = np.zeros((6, 6)); npx = []
    for j, (q, _m) in enumerate(views):
        Ij, n = info_view(geo, q, x_hat, np.asarray(b_link, np.float64), 0 if j == 0 else 1)
        I += Ij; npx.append(n)
    return dict(I=I, Sigma=float(b_blk) * np.linalg.inv(I), n_px=npx)
