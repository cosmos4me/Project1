"""Capsule risk R_cap(q): every link as a capsule, the silhouette approximated by capsule side lines and end caps,
soft visibility (occlusion by the other capsules and the image border); used for the coarse search of D v4."""
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss

from . import robot as _robot
from .geometry import TorchXArmFK, link_meshes
from .se3 import SCALE, pose_of

LINKS8 = ["link_base"] + [f"link{i}" for i in range(1, 8)]
TAU = 2.0
W = np.diag(SCALE ** 2)
_GEO = None


def geometry():
    """the capsule geometry of the robot (built once per process)."""
    global _GEO
    if _GEO is None:
        _GEO = Geo()
    return _GEO


class Geo:
    def __init__(self):
        self.fkt = TorchXArmFK(_robot.active().urdf, torch.device("cpu"))
        self.O = [o.numpy() for o in self.fkt.origins]
        meshes = link_meshes()
        self.V = [np.asarray(meshes[n].vertices, np.float64) for n in LINKS8]
        # capsules (6.1)
        self.cap = []
        for v in self.V:
            c = v.mean(0)
            w, U = np.linalg.eigh(np.cov((v - c).T))
            e = U[:, -1]
            t = (v - c) @ e
            perp = (v - c) - t[:, None] * e[None]
            r = float(np.median(np.linalg.norm(perp, axis=1)))
            tmin, tmax = float(t.min()), float(t.max())
            if tmax - tmin < 2 * r:
                A = B = c + 0.5 * (tmin + tmax) * e
                r = max(r, 0.5 * (tmax - tmin)); sph = True
            else:
                A, B = c + (tmin + r) * e, c + (tmax - r) * e; sph = False
            self.cap.append(dict(A=A, B=B, e=e, r=r, sph=sph))
        xs, ws = leggauss(16)
        self.t16, self.w16 = 0.5 * (xs + 1), 0.5 * ws                  # [0, 1]
        xs, ws = leggauss(8)
        self.th8, self.w8 = 0.5 * np.pi * xs, 0.5 * np.pi * ws          # [-pi/2, pi/2]

    def fk(self, Q):
        """(B,7) -> (B,8,4,4) link_base..link7 in base frame (TorchXArmFK, batched)."""
        Q = np.atleast_2d(Q); B = len(Q)
        T = np.tile(np.eye(4), (B, 1, 1)); out = [T]
        for i in range(7):
            c, s = np.cos(Q[:, i]), np.sin(Q[:, i])
            Rz = np.zeros((B, 4, 4)); Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s; Rz[:, 1, 0] = s; Rz[:, 1, 1] = c
            Rz[:, 2, 2] = 1; Rz[:, 3, 3] = 1
            T = np.einsum("bij,jk,bkl->bil", T, self.O[i], Rz)
            out.append(T)
        return np.stack(out, 1)


def smooth(z):
    x = np.clip((z + 1.0) * 0.5, 0.0, 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


class Scene:

    def __init__(self, G: Geo, post: dict):
        self.G = G
        P = post
        H_img, W_img = post.get("image_hw", (720, 1280))
        self.WIMG, self.HIMG = float(W_img) - 1.0, float(H_img) - 1.0    # last pixel centre (1279, 719 at 1280 x 720)
        self.K = np.asarray(P["K"], np.float64)
        self.fx, self.fy, self.sk, self.cx, self.cy = self.K[0, 0], self.K[1, 1], self.K[0, 1], self.K[0, 2], self.K[1, 2]
        self.f = 0.5 * (self.fx + self.fy)
        self.xh = np.asarray(P["Rls_x_hat"], np.float64)
        self.T = pose_of(self.xh)
        self.Rc, self.tc = self.T[:3, :3], self.T[:3, 3]
        self.o_cam = -self.Rc.T @ self.tc
        self.b_link = np.asarray(P["Rls_b_link"], np.float64)
        self.b_blk = float(P["Rls_b_blk"])
        self.Sinv = np.asarray(P["Rls_I_acq"], np.float64) / self.b_blk
        self.Sigma = np.linalg.inv(self.Sinv)
        self.Sigma_file = np.asarray(P["Rls_Sigma"], np.float64)
        self.q1 = np.asarray(P["q1"], np.float64)
        self.Jxi = self.jac_xi()                     # 6x6, d xi / d x at x_hat (central difference, pose_of)

    def jac_xi(self, h=1e-3):
        T0inv = np.linalg.inv(self.T)
        J = np.zeros((6, 6))
        for k in range(6):
            dT = []
            for sg in (1, -1):
                d = self.xh.copy(); d[k] += sg * h
                dT.append(pose_of(d) @ T0inv)
            rho = (dT[0][:3, 3] - dT[1][:3, 3]) / (2 * h)
            A0 = 0.5 * (dT[0][:3, :3] - dT[0][:3, :3].T); A1 = 0.5 * (dT[1][:3, :3] - dT[1][:3, :3].T)
            S = (A0 - A1) / (2 * h)
            J[:3, k] = rho; J[3:, k] = (S[2, 1], S[0, 2], S[1, 0])
        return J

    def proj(self, p):
        Z = p[..., 2]
        u = (self.fx * p[..., 0] + self.sk * p[..., 1]) / Z + self.cx
        v = self.fy * p[..., 1] / Z + self.cy
        return u, v, Z

    def dproj(self, p):
        """du/dp (..,2,3)."""
        X, Y, Z = p[..., 0], p[..., 1], p[..., 2]
        D = np.zeros(p.shape[:-1] + (2, 3))
        D[..., 0, 0] = self.fx / Z; D[..., 0, 1] = self.sk / Z; D[..., 0, 2] = -(self.fx * X + self.sk * Y) / Z ** 2
        D[..., 1, 1] = self.fy / Z; D[..., 1, 2] = -self.fy * Y / Z ** 2
        return D

    def cam_links(self, Q):
        Tl = self.G.fk(Q)
        Tc = np.einsum("ij,bljk->blik", self.T, Tl)
        return Tl, Tc

    def surrogate(self, Q, chunk=64):
        G = self.G
        R_out, Lvis_out, iso_out = [], [], []
        Ws = np.sqrt(W)
        for s0 in range(0, len(Q), chunk):
            Tl, Tc = self.cam_links(Q[s0:s0 + chunk])
            Bn = len(Tc)
            A = np.stack([Tc[:, l, :3, :3] @ G.cap[l]["A"] + Tc[:, l, :3, 3] for l in range(8)], 1)   # (B,8,3)
            Bp = np.stack([Tc[:, l, :3, :3] @ G.cap[l]["B"] + Tc[:, l, :3, 3] for l in range(8)], 1)
            dax = np.stack([Tc[:, l, :3, :3] @ G.cap[l]["e"] for l in range(8)], 1)                     # unit axis
            rr = np.array([G.cap[l]["r"] for l in range(8)])
            uA, vA, ZA = self.proj(A); uB, vB, ZB = self.proj(Bp)
            a2 = np.stack([uA, vA], -1); b2 = np.stack([uB, vB], -1)
            # iso (phi8): projected capsule axes of link1-7
            ab = b2 - a2
            ell = np.linalg.norm(ab, axis=-1)
            tt = ab / np.maximum(ell, 1e-12)[..., None]
            nrm = np.stack([-tt[..., 1], tt[..., 0]], -1)
            good = (ZA > 0) & (ZB > 0)
            Nm = np.einsum("bl,bli,blj->bij", np.where(good, ell, 0)[:, 1:], nrm[:, 1:], nrm[:, 1:])
            ev = np.linalg.eigvalsh(Nm)
            iso_out.append(np.where(ev[:, 1] > 0, ev[:, 0] / np.maximum(ev[:, 1], 1e-12), 0.0))
            pts, nor, ds, lid = [], [], [], []
            for l in range(1, 8):
                r = rr[l]
                Al, Bl, d = A[:, l], Bp[:, l], dax[:, l]
                # side lines
                Pt = Al[:, None] + G.t16[None, :, None] * (Bl - Al)[:, None]               # (B,16,3)
                mvec = np.cross(d[:, None], Pt); mvec /= np.maximum(np.linalg.norm(mvec, axis=-1, keepdims=True), 1e-12)
                D = self.dproj(Pt)                                                          # (B,16,2,3)
                dpi = np.einsum("btij,bj->bti", D, Bl - Al)                                 # (B,16,2)
                L = np.linalg.norm(dpi, axis=-1)
                tg = dpi / np.maximum(L, 1e-12)[..., None]
                nperp = np.stack([-tg[..., 1], tg[..., 0]], -1)
                uP, vP, _ = self.proj(Pt)
                for sg in (1.0, -1.0):
                    S = Pt + sg * r * mvec
                    uS, vS, _ = self.proj(S)
                    sgn = np.sign(((uS - uP)[..., None] * nperp[..., :1] + (vS - vP)[..., None] * nperp[..., 1:]))
                    sgn = np.where(sgn == 0, 1.0, sgn)
                    pts.append(S); nor.append(nperp * sgn); ds.append(L * G.w16[None]); lid.append(np.full(16, l))
                # end hemispheres
                for E, dout in ((Al, -d), (Bl, d)):
                    En = np.linalg.norm(E, axis=-1)
                    eh = E / En[:, None]
                    e1 = dout - np.sum(dout * eh, -1, keepdims=True) * eh
                    e1 /= np.maximum(np.linalg.norm(e1, axis=-1, keepdims=True), 1e-12)
                    e2 = np.cross(eh, e1)
                    Qp = E[:, None] + r * (np.cos(G.th8)[None, :, None] * e1[:, None] + np.sin(G.th8)[None, :, None] * e2[:, None])
                    uQ, vQ, _ = self.proj(Qp); uE, vE, _ = self.proj(E)
                    nv = np.stack([uQ - uE[:, None], vQ - vE[:, None]], -1)
                    nv /= np.maximum(np.linalg.norm(nv, axis=-1, keepdims=True), 1e-12)
                    pts.append(Qp); nor.append(nv); ds.append((self.f * r / En)[:, None] * G.w8[None]); lid.append(np.full(8, l))
            pts = np.concatenate(pts, 1); nor = np.concatenate(nor, 1); ds = np.concatenate(ds, 1); lid = np.concatenate(lid)
            us, vs, Zs = self.proj(pts)                                                    # (B,336)
            x = np.stack([us, vs], -1)
            # visibility: union occlusion by other capsules (incl. base) + screen margin
            vis = np.ones_like(us)
            for lp in range(8):
                a = a2[:, lp][:, None]; b = b2[:, lp][:, None]
                ab2 = b - a
                den = np.sum(ab2 * ab2, -1)
                sstar = np.where(den > 1e-12, np.clip(np.sum((x - a) * ab2, -1) / np.maximum(den, 1e-12), 0, 1), 0.0)
                cp = a + sstar[..., None] * ab2
                dist = np.linalg.norm(x - cp, axis=-1)
                Zi = ZA[:, lp][:, None] + sstar * (ZB[:, lp] - ZA[:, lp])[:, None]
                rho = self.f * rr[lp] / np.where(Zi > 1e-6, Zi, 1e-6)
                cov = smooth((rho - dist) / TAU)
                cov = np.where((Zi > 1e-6), cov, 0.0)
                cov = np.where(lid[None] == lp, 0.0, cov)
                vis = vis * (1.0 - cov)
            mm = np.minimum(np.minimum(us, self.WIMG - us), np.minimum(vs, self.HIMG - vs))
            vis = vis * smooth(mm / TAU) * (Zs > 1e-6)
            # information
            Dp = self.dproj(pts)                                                           # (B,336,2,3)
            avec = np.einsum("bnij,bni->bnj", Dp, nor)                                      # (B,336,3)
            g = np.concatenate([avec, np.cross(pts, avec)], -1)                             # (B,336,6)
            Jx = g @ self.Jxi                                                               # J_x = Jxi^T g
            bl = self.b_link[lid]
            w = vis * ds / bl[None] ** 2
            H = np.einsum("bn,bni,bnj->bij", w, Jx, Jx) / self.b_blk
            Apost = self.Sinv[None] + H
            R_out.append(np.einsum("ij,bji->b", W, np.linalg.inv(Apost)))
            Lv = np.stack([(vis * ds)[:, lid == l].sum(1) for l in range(1, 8)], 1)
            Lvis_out.append(Lv)
        return np.concatenate(R_out), np.concatenate(Lvis_out), np.concatenate(iso_out)


def frames(G, Q):
    T = G.fk(Q)
    P = [T[:, l, :3, 3] for l in range(8)]
    Tm = {LINKS8[l]: T[:, l] for l in range(8)}
    for name, (parent, O) in G.fkt.tool.items():
        Tm[name] = Tm[parent] @ O.numpy()
        P.append(Tm[name][:, :3, 3])
    return np.stack(P, 1)
