"""Command line.

  python -m dv4calib next <folder>        fit the views in <folder>, then save the next joint configuration
  python -m dv4calib estimate <folder>    fit the views in <folder>, then save the camera pose and its uncertainty

<folder> uses the format of dv4calib.io. Results go to <folder>/dv4calib_out (or --out):
  estimate.txt (4x4 camera <- base), next_q.txt (7 joint angles), decision_k<k>.json, history.json, summary.json.
A capture loop alternates `next` with moving the robot and saving the new view as views/<k+1>/{q.txt, mask.png}.
Each `next` uses the previous decision (decision_k<k-1>.json) as warm start, as in the sequential method.
If no candidate passed the released path check, next_q_unchecked.txt is written instead of next_q.txt and the exit
status is 3 (use --allow-unchecked to accept it).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _calibrator(folder, out, seed):
    from .calibrator import Calibrator
    from .io import load_folder
    data = load_folder(folder)
    (q1, m1), rest = data["views"][0], data["views"][1:]
    cal = Calibrator(data["K"], data["T_init"], q1, m1, workdir=out, seed=seed)
    cal.posterior()                      # view-1 fit and noise scales (used by every later posterior)
    for q, m in rest:
        cal.add_view(q, m)
    if rest:
        cal.fit()
    return cal, data


def _summary(cal, data, extra=None):
    h = dict(cal.history[-1])
    s = dict(views=len(cal.views), view_dirs=[str(d) for d in data["view_dirs"]], T_cam_from_base=cal.estimate().tolist(),
             position_std_mm=h.get("position_std_mm"), rotation_std_deg=h.get("rotation_std_deg"),
             fit_quality=h.get("fit_quality"))
    if data["T_gt"] is not None:
        from .se3 import dof_of
        import numpy as np
        e = np.asarray(cal.x_hat) - dof_of(np.asarray(data["T_gt"], np.float64))
        s["error_vs_T_gt"] = dict(rot_deg=float(np.abs(e[3:]).max() * 180 / np.pi), trans_mm=float(np.linalg.norm(e[:3]) * 1000))
    s.update(extra or {})
    low = [i + 1 for i, f in enumerate(s["fit_quality"] or []) if f["iou"] < 0.8]
    if low:
        print(f"WARNING: the fitted silhouette overlaps the mask by less than 80% in views {low}; check T_init and the masks")
    return s


def cmd_estimate(a):
    import numpy as np
    out = Path(a.out) if a.out else Path(a.folder) / "dv4calib_out"
    cal, data = _calibrator(a.folder, out, a.seed)
    if len(cal.views) > 1:
        cal.posterior()
    np.savetxt(out / "estimate.txt", cal.estimate())
    s = _summary(cal, data)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: s[k] for k in ("views", "position_std_mm", "rotation_std_deg")}))
    print("T_cam_from_base =\n" + np.array2string(cal.estimate(), precision=6, suppress_small=True))
    return 0


def cmd_next(a):
    import numpy as np
    out = Path(a.out) if a.out else Path(a.folder) / "dv4calib_out"
    cal, data = _calibrator(a.folder, out, a.seed)
    k = len(cal.views)
    prev = out / f"decision_k{k - 1}.json"
    warm = False
    if k >= 2 and prev.is_file():
        from .calibrator import _mask_sha
        rec = json.loads(prev.read_text())
        same = (len(rec.get("views_q", [])) == k - 1
                and all(np.allclose(a, b, rtol=0, atol=1e-12) for a, b in zip(rec["views_q"], [v[0] for v in cal.views]))
                and rec.get("views_mask_sha256") == [_mask_sha(v[1]) for v in cal.views[:k - 1]])
        if same:
            cal.warm = [(n, np.asarray(q, np.float64)) for n, q in rec["log"]["warm_out"]]
            warm = True
        else:
            print(f"note: {prev.name} was computed for other views; no warm start")
    q7, d = cal.next_view()
    np.savetxt(out / "estimate.txt", cal.estimate())
    unchecked = bool(d.log["unchecked"])
    target = out / ("next_q_unchecked.txt" if unchecked and not a.allow_unchecked else "next_q.txt")
    for f in (out / "next_q.txt", out / "next_q_unchecked.txt"):
        f.unlink(missing_ok=True)
    np.savetxt(target, q7.reshape(1, -1))
    s = _summary(cal, data, dict(next_q=q7.tolist(), risk=d.R_exact, source=d.source, warm_start=warm,
                                 fallback=d.log["fallback"], unchecked=unchecked))
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print("next q (rad):", " ".join(f"{v:.6f}" for v in q7))
    print(f"predicted risk R = {d.R_exact:.5f} ({d.source}); saved to {target}")
    if unchecked and not a.allow_unchecked:
        print("WARNING: no candidate passed the released path check; check this motion yourself "
              "(rerun with --allow-unchecked to accept it)")
        return 3
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dv4calib", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default=None, help="GPU index (sets CUDA_VISIBLE_DEVICES before CUDA starts)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("next", "estimate"):
        p = sub.add_parser(name)
        p.add_argument("folder")
        p.add_argument("--gpu", dest="gpu_sub", default=None, help="GPU index (same as --gpu before the command)")
        p.add_argument("--out", default=None)
        p.add_argument("--seed", type=int, default=0, help="seed of the posterior samples")
        if name == "next":
            p.add_argument("--allow-unchecked", action="store_true", help="accept a pose that no path check passed")
    a = ap.parse_args(argv)
    gpu = a.gpu_sub if a.gpu_sub is not None else a.gpu
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return {"next": cmd_next, "estimate": cmd_estimate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
