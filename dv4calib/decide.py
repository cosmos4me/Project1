"""D v4 decision: the next joint configuration for one fixed camera on the xArm7.

  1 starts   view 2  : the N_FEAS pose-feasible configurations of a Sobol sequence (seed 0, drawn in batches until N_FEAS
                       pass the released cheap check and the pose clearances; scene independent, cached), ranked by the
                       capsule risk R_cap, N_START kept with the 0.25 m link-position diversity rule;
             view >= 3: the previous decision's end points (warm start) + q_now, pose-feasible ones ranked by R_cap with
                       the diversity rule; filled up to N_START from the feasible set in R_cap order.
  2 move     each start: SLSQP on R_cap (forward-difference gradient, reach constraint) and pull-back along start -> end to
             the last cheap_ok and pose-feasible point.
  3 order    end points in exact-R order; released cheap check + released path check from q_now (<= CERT_ATTEMPTS); the
             first N_ANCHOR certified end points are the anchors.
  4 refine   TR-SQP on log R~ (exact reverse-mode gradients; reach, clearance and the straight path anchor -> q inside)
             from each anchor.  Support constants: D_ref (motion budget) and r_res (trust cap) from the N_FEAS set;
             tau_max and L_min (R~ validity) from a fixed seeded subset of N_SUPPORT feasible configurations.
  5 select   refined end points + anchors in exact-R order, released final check (cheap + path <= 3); first pass.  Fallback:
             continue below the point where step 3 stopped; if nothing passes: best exact R, flagged unchecked.
Path checks go through _path(): a candidate within NEAR (max-abs 1e-4 rad) of q_now counts as reachable, and a
RuntimeError of the released planner counts as a failed attempt."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import qmc

from . import clearance as C
from . import risk_capsule as FT
from . import risk_smooth as RX
from .clearance import Clearance
from .constraints import ConstraintsV2, Support
from .feasibility import Feas, Released
from .geometry import TorchXArmFK
from .risk_exact import ExactR
from .risk_smooth import SmoothRisk
from .trsqp import TRSQP


N_FEAS = 320          # a priori (= the v3 feasible count of Sobol 4096; same set)
N_START = 8           # a priori (v3: 8 starts)
N_ANCHOR = 2          # a priori time budget (v3: 2)
DIV = 0.25            # m, diversity rule (v3)
N_SUPPORT = 64        # a priori: subset for the camera-dependent R~ validity constants
SEED_SUP = 20261018
BATCH = 512
NEAR = 1e-4           # rad, max-abs distance to q_now below which a candidate is the current pose (no plan needed)


CERT_ATTEMPTS = 3       # released path-check attempts for certifying a branch / restored point (as the final check)


@dataclass
class Decision:
    q: np.ndarray
    source: str
    R_exact: float
    R_tilde: float
    log: dict = field(default_factory=dict)


class DecisionV4:
    _FEAS = None

    def __init__(self, device="cuda:0"):
        self.dev = device
        torch.zeros(1, device=device)
        self.rel = Released()
        self.robot = RX.xarm_model(device, center=self.rel.center)
        self.robot_c = RX.xarm_model("cpu", center=self.rel.center)
        self.clear = Clearance(TorchXArmFK(RX.URDF, torch.device(device)), self.rel.workspace, device)
        FT.geometry()
        self.feas = Feas(self.rel.center)

    @torch.no_grad()
    def pose_ok(self, Q, chunk=256):
        """Q (n, 7): all clearance rows of Clearance.rows(q, q) <= 0, batched (env per link >= M_PENV, self per pair >= M_PSELF)."""
        Q = np.atleast_2d(np.asarray(Q, np.float64))
        out = np.zeros(len(Q), bool)
        for s in range(0, len(Q), chunk):
            Q7 = torch.as_tensor(np.c_[Q[s:s + chunk, :6], np.zeros(len(Q[s:s + chunk]))], device=self.clear.dev)
            Tl = self.clear.fk_batch(Q7)
            e = self.clear.env_min(Tl)[0].min(1).values
            sm = self.clear.self_min(Tl)[0].min(1).values
            out[s:s + chunk] = ((e >= C.M_PENV) & (sm >= C.M_PSELF)).cpu().numpy()
        return out

    def feasible_set(self):
        if DecisionV4._FEAS is None:
            t0 = time.perf_counter()
            eng = qmc.Sobol(d=6, scramble=True, seed=0)
            low, high = self.feas.low7, self.feas.high7
            keep, drawn = [], 0
            while len(keep) < N_FEAS:
                U = eng.random(BATCH); drawn += BATCH
                Q = np.zeros((BATCH, 7)); Q[:, :6] = low[:6] + U * (high[:6] - low[:6])
                idx = np.flatnonzero(self.feas.cheap(Q)["cheap_ok"])
                if len(idx):
                    idx = idx[self.pose_ok(Q[idx])]
                keep.extend(Q[idx])
            Qf = np.asarray(keep[:N_FEAS])
            LPf = np.stack([np.asarray(self.feas.cands.backend.link_positions(self.feas.pad(q, self.feas.nq)), np.float64) for q in Qf])
            with torch.no_grad():
                P = np.stack([self.robot_c.frame_positions(self.robot_c.fk(self.robot_c.q_full(torch.as_tensor(q[:6])))).numpy() for q in Qf])
            Dm = np.linalg.norm(P[:, None] - P[None], axis=-1).max(-1); np.fill_diagonal(Dm, np.inf)
            r_res = float(np.median(Dm.min(1)))
            sub = np.sort(np.random.default_rng(SEED_SUP).choice(len(Qf), N_SUPPORT, replace=False))
            DecisionV4._FEAS = dict(Q=Qf, LP=LPf, r_res=r_res, sub=sub, drawn=drawn, t=time.perf_counter() - t0)
        return DecisionV4._FEAS

    def support_v4(self, risk, q_now):
        F = self.feasible_set()
        taus, Ls = [], []
        for q in F["Q"][F["sub"]]:
            disc = risk.prepare_run(q[:6])
            _, Fe = risk.value(q[:6], disc, terms=[0], feats=True)
            f = Fe[risk.cams[0].name]
            taus.append(float(f["tau_c"])); Ls.append(float(f["L_in"]))
        D_ref = float(np.max(np.abs(F["Q"][:, :6] - np.asarray(q_now)[None, :6]).max(1)))
        return Support(tau_max=float(np.max(taus)), L_min=float(np.min(Ls)), D_ref=D_ref, r_res=F["r_res"],
                       info=dict(n_support=len(taus), source="v4 feasible set"))

    def _path(self, q_now, q):
        if np.abs(np.asarray(q, np.float64)[:6] - np.asarray(q_now, np.float64)[:6]).max() < NEAR:
            self._cnt["path_near"] += 1
            return True
        try:
            return self.rel.path_ok(q_now, q)
        except RuntimeError:
            self._cnt["path_exc"] += 1
            return False

    def _final_ok(self, q, q_now, attempts=3):
        info = self.rel.checks(q)
        if not info["cheap_ok"]:
            return False, 0, info
        for t in range(1, attempts + 1):
            if self._path(q_now, q):
                return True, t, info
        return False, attempts, info

    def _lp(self, q):
        return np.asarray(self.feas.cands.backend.link_positions(self.feas.pad(np.r_[np.asarray(q)[:6], 0.0], self.feas.nq)), np.float64)

    def _diverse(self, order, LP, n, taken=()):
        sel, ref = [], [np.asarray(l) for l in taken]
        for i in order:
            if all(np.linalg.norm(LP[i] - l, axis=-1).max() >= DIV for l in ref):
                sel.append(int(i)); ref.append(LP[i])
                if len(sel) == n:
                    break
        return sel

    def _local_rcap(self, sc, x0):
        """local optimiser on R_cap (SLSQP, forward-difference gradient, reach constraint) + pull-back."""
        G = FT.geometry()
        c, rmax = self.feas.center, self.feas.r_max
        low, high = self.feas.low7, self.feas.high7
        h = 1e-4
        cache = {}

        def fg(x):
            key = x.tobytes()
            if key not in cache:
                B = np.zeros((7, 7)); B[:, :6] = x[None]; B[1:, :6] += h * np.eye(6)
                v = sc.surrogate(B)[0]
                cache[key] = (v[0], (v[1:] - v[0]) / h)
            return cache[key]

        def cons(x):
            P = FT.frames(G, np.r_[x, 0][None])[0]
            return rmax - np.linalg.norm(P - c[None], axis=1)

        def cons_j(x):
            B = np.zeros((7, 7)); B[:, :6] = x[None]; B[1:, :6] += 1e-6 * np.eye(6)
            P = FT.frames(G, B)
            g = rmax - np.linalg.norm(P - c[None, None], axis=2)
            return ((g[1:] - g[0][None]) / 1e-6).T

        r = minimize(lambda x: fg(x)[0], x0, jac=lambda x: fg(x)[1], method="SLSQP",
                     bounds=list(zip(low[:6], high[:6])), constraints=[dict(type="ineq", fun=cons, jac=cons_j)],
                     options=dict(maxiter=100, ftol=1e-9))
        xs = np.clip(r.x, low[:6], high[:6])
        A_ = np.linspace(1.0, 0.0, 11)
        L = np.c_[x0[None] + A_[:, None] * (xs - x0)[None], np.zeros(len(A_))]
        ok = self.feas.cheap(L)["cheap_ok"] & self.pose_ok(L)
        a_ok = float(A_[int(np.argmax(ok))]) if ok.any() else 0.0
        return x0 + a_ok * (xs - x0), a_ok

    def decide(self, post_in, after_view, warm=None):
        """post_in: posterior dict (K, q1 = current joints, Rls_x_hat, Rls_I_acq, Rls_Sigma, Rls_b_link, Rls_b_blk,
        Rls_samples, image_hw); after_view = number of views so far; warm = previous decision log["warm_out"]."""
        T0 = time.perf_counter(); tm = {}
        self._cnt = dict(path_near=0, path_exc=0)
        copy = lambda: {**{k: np.array(post_in[k]) for k in RX.POST_KEYS}, "image_hw": tuple(post_in["image_hw"])}
        post = copy()
        q_now = np.asarray(post["q1"], np.float64)[:6]
        G = FT.geometry()
        sc = FT.Scene(G, copy())
        F = self.feasible_set()
        # ---- 1 starts
        t = time.perf_counter()
        Rf = sc.surrogate(F["Q"])[0]
        starts, src_names = [], []
        if after_view >= 2 and warm:
            W, uniq = [], set()
            for name, q in list(warm) + [("q_now", q_now)]:
                q = np.clip(np.asarray(q, np.float64)[:6], self.feas.low7[:6], self.feas.high7[:6])
                key = tuple(np.round(q, 6))
                if key in uniq:
                    continue
                uniq.add(key)
                q7 = np.r_[q, 0.0][None]
                if bool(self.feas.cheap(q7)["cheap_ok"][0] and self.pose_ok(q7)[0]):
                    W.append((name, q))
            if W:
                Rw = sc.surrogate(np.c_[np.asarray([q for _, q in W]), np.zeros(len(W))])[0]
                LPw = np.stack([self._lp(q) for _, q in W])
                for i in self._diverse(np.argsort(Rw, kind="stable"), LPw, N_START):
                    starts.append(W[i][1]); src_names.append("warm:" + W[i][0])
        if len(starts) < N_START:
            taken = [self._lp(q) for q in starts]
            for i in self._diverse(np.argsort(Rf, kind="stable"), F["LP"], N_START - len(starts), taken):
                starts.append(F["Q"][i, :6].copy()); src_names.append(f"feas{i}")
        tm["starts"] = time.perf_counter() - t
        # ---- 2 move (R_cap SLSQP + pull-back)
        t = time.perf_counter()
        ends = []
        for name, q0 in zip(src_names, starts):
            q, a = self._local_rcap(sc, np.asarray(q0, np.float64).copy())
            ends.append((name, q, a))
        tm["move"] = time.perf_counter() - t
        # ---- 3 order (exact R) + certification
        t = time.perf_counter()
        ex = ExactR(copy())
        uniq, E = set(), []
        for name, q, a in ends:
            key = tuple(np.round(q, 9))
            if key not in uniq:
                uniq.add(key); E.append((name, q, a, ex(q)))
        E.sort(key=lambda e: e[3])
        tm["exact_R"] = time.perf_counter() - t
        t = time.perf_counter()
        anchors, cert_log, n_path_cert, checked = [], [], 0, set()
        for j, (name, q, a, Rq) in enumerate(E):
            checked.add(j)                                  # examined by step 3 (anchor, cheap_fail or path_fail)
            chk = self.rel.checks(q)
            if not chk["cheap_ok"]:
                cert_log.append(dict(name=name, R=Rq, certified=False, why="cheap_fail")); continue
            cert, att = False, 0
            for att in range(1, CERT_ATTEMPTS + 1):
                n_path_cert += 1
                if self._path(q_now, q):
                    cert = True; break
            cert_log.append(dict(name=name, R=Rq, certified=bool(cert), why="path_ok" if cert else "path_fail", attempts=att))
            if cert:
                anchors.append((name, q, Rq))
                if len(anchors) == N_ANCHOR:
                    break
        tm["certify"] = time.perf_counter() - t
        # ---- 4 refine (TR-SQP)
        t = time.perf_counter()
        cam = RX.XArmFixedCamera(post["K"], int(post["image_hw"][1]), int(post["image_hw"][0]))
        risk = SmoothRisk(self.robot, [cam], {cam.name: RX.belief_of(post)}, device=self.dev)
        tm["risk"] = time.perf_counter() - t
        runs = []
        tm["support"] = 0.0
        if anchors:
            ts = time.perf_counter(); sup = self.support_v4(risk, q_now); tm["support"] = time.perf_counter() - ts
        t = time.perf_counter()
        if anchors:
            for name, qa, R0 in anchors:
                disc = risk.prepare_run(qa)
                Rt0, F0 = risk.value(qa, disc, feats=True)
                F0c = {k: (float(v) if torch.is_tensor(v) else v) for k, v in F0[cam.name].items()}
                cons = ConstraintsV2(self.robot_c, self.clear, sup, qa, q_now, F_start=F0c)
                g0, names = cons.geo_val(qa)
                tr = TRSQP(risk, self.robot_c, cons, device="cpu").run(qa, disc, q_now)
                runs.append(dict(anchor=name, q0=np.asarray(qa).tolist(), R0_exact=R0, Rt0=float(Rt0),
                                 start_viol=[names[i] for i in np.flatnonzero(g0 > 0)], q=np.asarray(tr.q).tolist(),
                                 Rt=float(tr.R), n_f=tr.n_f, n_trials=tr.n_trials, iters=len(tr.iters), stop=tr.stop, t=tr.t,
                                 t_clear=cons.t_clear, n_geo=cons.n_geo))
        tm["trsqp"] = time.perf_counter() - t
        # ---- 5 select
        t = time.perf_counter()
        fin, seen = [], set()
        for r in runs:
            q = np.asarray(r["q"]); k = tuple(np.round(q, 9))
            if k not in seen:
                seen.add(k); fin.append((f"refined:{r['anchor']}", q, ex(q)))
        for name, q, Rq in anchors:
            k = tuple(np.round(q, 9))
            if k not in seen:
                seen.add(k); fin.append((f"anchor:{name}", q, Rq))
        fin.sort(key=lambda e: e[2])
        pick, checks, n_path = None, [], 0
        for name, q, Rq in fin:
            ok, att, info = self._final_ok(q, q_now, attempts=3); n_path += att
            checks.append(dict(name=name, R=Rq, ok=bool(ok), attempts=int(att)))
            if ok:
                pick = (name, q, Rq); break
        if pick is None:                                   # fallback: continue down the step-3 order
            for j, (name, q, a, Rq) in enumerate(E):
                if j in checked:
                    continue
                ok, att, info = self._final_ok(q, q_now, attempts=3); n_path += att
                checks.append(dict(name="order:" + name, R=Rq, ok=bool(ok), attempts=int(att)))
                if ok:
                    pick = ("order:" + name, q, Rq); break
        unchecked = pick is None
        if unchecked:
            cand = fin + [("order:" + n, q, Rq) for n, q, a, Rq in E]
            pick = min(cand, key=lambda e: e[2])
            pick = (pick[0] + "(unchecked)", pick[1], pick[2])
        tm["select"] = time.perf_counter() - t
        warm_out = [(f"end:{n}", np.asarray(q).tolist()) for n, q, a, Rq in E] + \
                   [(f"refined:{r['anchor']}", r["q"]) for r in runs]
        log = dict(after_view=after_view, times=tm, t_total=time.perf_counter() - T0, starts=src_names,
                   pull=[float(a) for _, _, a in ends], order=[dict(name=n, R=Rq) for n, q, a, Rq in E], certify=cert_log,
                   n_path_certify=n_path_cert, runs=runs, final_checks=checks, n_path_final=n_path, source=pick[0],
                   fallback=bool(pick[0].startswith("order:") or unchecked), unchecked=bool(unchecked),
                   warm_out=warm_out, counts=dict(R_tilde_evals=risk.n_eval, R_exact_evals=ex.n, path_checks=n_path_cert + n_path,
                                                  path_near=self._cnt["path_near"], path_exc=self._cnt["path_exc"]))
        return Decision(q=np.asarray(pick[1]), source=pick[0], R_exact=float(pick[2]), R_tilde=float("nan"), log=log)
