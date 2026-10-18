"""Synthetic end-to-end example: masks rendered with the package's own rasterizer at a known camera pose.

A camera (1280 x 720, f = 900 px) looks at the bundled xArm7 from about 1.3 m. The initial pose is the true pose moved by
1.2 deg and 12 mm. View 1 is the zero configuration. Every further view is chosen by D v4 and its mask is rendered at the
true pose. Because the fit renders with the same rasterizer that made the masks, the errors show that the loop works,
not the accuracy on real or simulated camera images.

usage: python examples/synthetic.py [--out dv4calib_synthetic] [--views 5]
The views are also written as a data folder (<out>/data) for `python -m dv4calib next|estimate`."""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """4x4 camera <- base transform of an OpenCV camera (x right, y down, z forward) at `eye` looking at `target`."""
    eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = -R @ eye
    return T


def perturb(T, rot_deg, trans_mm, axis=(0.3, -0.5, 0.8), direction=(0.6, 0.2, -0.77)):
    a = np.asarray(axis, np.float64); a /= np.linalg.norm(a)
    th = np.radians(rot_deg)
    A = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    dR = np.eye(3) + np.sin(th) * A + (1 - np.cos(th)) * A @ A
    d = np.asarray(direction, np.float64); d /= np.linalg.norm(d)
    out = np.eye(4); out[:3, :3] = dR @ T[:3, :3]; out[:3, 3] = T[:3, 3] + 1e-3 * trans_mm * d
    return out


def errors(x_hat, T_gt):
    """EasyHeC's error measure on the se(3) logarithms of the estimate and the ground truth."""
    from dv4calib.se3 import dof_of
    e = np.asarray(x_hat, np.float64) - dof_of(np.asarray(T_gt, np.float64))
    rot, trans = float(np.abs(e[3:]).max() * 180 / np.pi), float(np.linalg.norm(e[:3]) * 1000)
    return dict(rot_deg=rot, trans_mm=trans, sqrt_c2=float(np.hypot(rot / 0.1, trans / 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dv4calib_synthetic")
    ap.add_argument("--views", type=int, default=5)
    a = ap.parse_args()
    from dv4calib import Calibrator, io, robot
    robot.use(robot.xarm7())
    from dv4calib.geometry import ContourGeometry
    out = Path(a.out); data = out / "data"; data.mkdir(parents=True, exist_ok=True)
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    T_gt = look_at(eye=(1.05, 0.65, 0.55), target=(0.0, 0.0, 0.32))
    T_init = perturb(T_gt, 1.2, 12.0)
    geo = ContourGeometry(K, height=720, width=1280, scale=1.0)

    def render(q7):
        return np.asarray(geo.raster(np.asarray(q7, np.float64)[:7], T_gt)[0], bool)

    np.savetxt(data / "K.txt", K); np.savetxt(data / "T_init.txt", T_init); np.savetxt(data / "T_gt.txt", T_gt)
    q1 = np.zeros(7)
    m1 = render(q1)
    io.write_view(data, 1, q1, m1)
    cal = Calibrator(K, T_init, q1, m1, workdir=out / "log")

    def capture(q7):
        m = render(q7)
        io.write_view(data, len(cal.views) + 1, q7, m)
        return m

    t0 = time.time()
    cal.run(capture, n_views=a.views, log=print)
    e0 = errors(cal.d0, T_gt)
    print(f"initial pose: rot {e0['rot_deg']:.3f} deg, trans {e0['trans_mm']:.2f} mm")
    rows = []
    for h in cal.history:
        r = dict(views=h["views"], **errors(h["x_hat"], T_gt)); rows.append(r)
        print(f"after {r['views']} views: rot {r['rot_deg']:.4f} deg, trans {r['trans_mm']:.3f} mm, sqrt_c2 {r['sqrt_c2']:.4f}")
    print("posterior std after the last view: position (mm)", np.round(cal.history[-1]["position_std_mm"], 4),
          "rotation (deg)", np.round(cal.history[-1]["rotation_std_deg"], 5))
    (out / "result.json").write_text(json.dumps(dict(initial=e0, per_view=rows, seconds=time.time() - t0,
                                                     T_estimate=cal.estimate().tolist(), T_gt=T_gt.tolist()), indent=1))


if __name__ == "__main__":
    main()
