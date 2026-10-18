"""DESIGN §5.2 feasible-iterate trust-region SLP/SQP on f = log R~_tau with exact (autograd) gradients.
Linear model + ellipsoidal trust region in the link-displacement metric G; linearised constraints by the weight-free
dual-norm inclusion rule (reach constraints C2 always included, parent (c)); box = joint limits and motion budget;
acceptance (Armijo 1e-4) and radius update use the same f.  Trial points that violate only C2 by a numerical amount get
one Gauss-Newton restoration onto the reach sphere (second-order correction, deviation logged)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import minimize

from .constraints import link_metric, restore_reach

TR = dict(D0=0.01, Dmin=0.001, eta1=0.25, eta2=0.75, armijo=1e-4, max_it=20, small=1e-4, n_small=3)


@dataclass
class Trace:
    q: np.ndarray
    f: float
    R: float
    iters: list = field(default_factory=list)
    accepted: list = field(default_factory=list)       # accepted iterates (q, f)
    n_f: int = 0
    n_grad: int = 0
    n_trials: int = 0
    epochs: int = 0
    epoch_jumps: list = field(default_factory=list)
    t: float = 0.0
    stop: str = ""


class TRSQP:
    def __init__(self, risk, robot, cons, device="cpu", consts=TR, soc=True, log=None):
        self.risk, self.robot, self.cons = risk, robot, cons
        self.dev, self.c, self.soc, self.log = device, dict(consts), soc, log

    def _eval(self, q, disc):
        """f, grad f, R, features, validity g and its gradient (one forward, three backward passes)."""
        qt = torch.as_tensor(np.asarray(q, np.float64), device=self.risk.dev).clone().requires_grad_(True)
        R, F = self.risk(qt, disc, feats=True)
        cam = self.risk.cams[0].name
        gv, _ = self.cons.validity(F[cam])
        gR, = torch.autograd.grad(R, qt, retain_graph=True)
        JV = torch.stack([torch.autograd.grad(gv[i], qt, retain_graph=(i == 0))[0] for i in range(len(gv))])
        R = float(R.item())
        feats = {k: (float(v.item()) if torch.is_tensor(v) else v) for k, v in F[cam].items()}
        return np.log(R), gR.detach().cpu().numpy() / R, R, feats, gv.detach().cpu().numpy(), JV.detach().cpu().numpy()

    def _qp(self, g, G, Delta, q, cons_g, cons_J, lo, hi, force):
        Ginv = np.linalg.inv(G)
        incl = []
        for m in range(len(cons_g)):
            dn = float(np.sqrt(max(cons_J[m] @ Ginv @ cons_J[m], 0.0)))
            incl.append(bool(force[m]) or (cons_g[m] + 2 * Delta * dn >= 0))
        incl = np.asarray(incl)
        A = cons_J[incl]; b = cons_g[incl]
        cs = [dict(type="ineq", fun=lambda d: Delta ** 2 - d @ G @ d, jac=lambda d: -2 * G @ d)]
        if len(A):
            cs.append(dict(type="ineq", fun=lambda d: -(b + A @ d), jac=lambda d: -A))
        bounds = list(zip(lo - q, hi - q))
        # start from a small feasible steepest-descent direction scaled into the region (SLSQP needs a non-stationary start)
        d0 = np.zeros_like(q)
        r = minimize(lambda d: g @ d, d0, jac=lambda d: g, method="SLSQP", bounds=bounds, constraints=cs,
                     options=dict(ftol=1e-12, maxiter=100))
        d = np.clip(r.x, lo - q, hi - q)
        return d, int(incl.sum()), bool(r.success)

    def run(self, q0, disc, q_now=None) -> Trace:
        t_start = time.perf_counter()
        c = self.c
        q = np.asarray(q0, np.float64).copy()
        f, g, R, F, gv, JV = self._eval(q, disc)
        tr = Trace(q=q.copy(), f=f, R=R)
        tr.n_f += 1; tr.n_grad += 1
        tr.accepted.append((q.copy(), f, R))
        Delta, Dmax = c["D0"], self.cons.sup.r_res
        n_small = 0
        lo, hi = self.cons.lo, self.cons.hi
        for it in range(c["max_it"]):
            G = link_metric(self.robot, q, self.dev)
            cg, cJ, names = self.cons.geo_jac(q)
            allg = np.r_[cg, gv]; allJ = np.r_[cJ, JV]
            nm = list(names) + ["V", "U"]
            force = np.array([n.startswith("C2") for n in nm])
            d, n_inc, qp_ok = self._qp(g, G, Delta, q, allg, allJ, lo, hi, force)   # C2 rows always included
            pred = float(-g @ d)
            rec = dict(it=it, f=f, R=R, Delta=Delta, n_included=n_inc, qp_ok=qp_ok, pred=pred,
                       stepG=float(np.sqrt(max(d @ G @ d, 0.0))))
            if pred <= 1e-12:
                rec["result"] = "stationary"; tr.iters.append(rec); tr.stop = "stationary"; break
            qt = q + d
            gt, _ = self.cons.geo_val(qt)
            viol = gt > 0
            soc_used = False
            if viol.any() and self.soc and all(names[i].startswith("C2") for i in np.flatnonzero(viol)):
                qt = restore_reach(self.robot, qt, self.dev)
                gt, _ = self.cons.geo_val(qt)
                viol = gt > 0
                soc_used = True
            tr.n_trials += 1
            if viol.any() or not self.cons.box_ok(qt):
                rec.update(result="reject_constraint", viol=[names[i] for i in np.flatnonzero(viol)][:5], soc=soc_used)
                Delta /= 2; tr.iters.append(rec)
                if Delta < c["Dmin"]:
                    tr.stop = "Delta<Dmin"; break
                continue
            f2, g2, R2, F2, gv2, JV2 = self._eval(qt, disc)
            tr.n_f += 1; tr.n_grad += 1
            if np.any(gv2 > 0):
                rec.update(result="reject_validity", gv=gv2.tolist(), soc=soc_used)
                Delta /= 2; tr.iters.append(rec)
                if Delta < c["Dmin"]:
                    tr.stop = "Delta<Dmin"; break
                continue
            ared = f - f2
            rho = ared / pred
            if ared >= c["armijo"] * pred:
                rec.update(result="accept", ared=ared, rho=rho, soc=soc_used, f_new=f2, R_new=R2)
                q, f, g, R, F, gv, JV = qt, f2, g2, R2, F2, gv2, JV2
                tr.accepted.append((q.copy(), f, R))
                n_small = n_small + 1 if ared < c["small"] else 0
                # re-discretisation check (spacing > 0.6 tau on weighted edges)
                if F.get("max_spacing", 0.0) > 0.6 * self.risk.tau:
                    disc = self.risk.prepare_run(q, epoch=disc.epoch + 1)
                    fo = f
                    f, g, R, F, gv, JV = self._eval(q, disc)
                    tr.n_f += 1; tr.n_grad += 1
                    tr.epochs += 1; tr.epoch_jumps.append(f - fo)
                    rec["new_epoch"] = disc.epoch
            else:
                rec.update(result="reject_armijo", ared=ared, rho=rho, soc=soc_used)
            if rho < c["eta1"]:
                Delta /= 2
            elif rho > c["eta2"] and rec["stepG"] >= 0.9 * Delta:
                Delta = min(2 * Delta, Dmax)
            tr.iters.append(rec)
            if self.log:
                self.log(f"it {it} {rec.get('result')} f {f:.5f} Delta {Delta:.4f} pred {pred:.2e} ared {rec.get('ared', float('nan')):.2e} inc {n_inc}")
            if Delta < c["Dmin"]:
                tr.stop = "Delta<Dmin"; break
            if n_small >= c["n_small"]:
                tr.stop = "small_steps"; break
        else:
            tr.stop = "max_it"
        tr.q, tr.f, tr.R = q, f, R
        tr.t = time.perf_counter() - t_start
        tr.disc = disc
        return tr
