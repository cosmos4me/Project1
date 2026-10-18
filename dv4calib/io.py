"""Input folder format of the command-line interface.

<folder>/
  K.txt              3x3 camera intrinsics (pixels)
  T_init.txt         4x4 initial camera <- robot-base transform (metres)    [also read: initialization_T_camera_from_base.txt]
  T_gt.txt           optional 4x4 ground truth, used only for error reports [also read: ground_truth_T_camera_from_base.txt]
  views/01/q.txt     joint angles of view 1: the 7 arm joints in radians    [also read: qpos.npy]
  views/01/mask.png  robot mask of view 1: non-zero on link_base and link1..link7 (not the gripper)
  views/02/...       further views, in capture order
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

T_INIT_NAMES = ("T_init.txt", "initialization_T_camera_from_base.txt")
T_GT_NAMES = ("T_gt.txt", "ground_truth_T_camera_from_base.txt")


def read_mask(path) -> np.ndarray:
    """boolean robot mask; same rule as the experiments: first channel > 0."""
    import imageio.v2 as imageio
    m = imageio.imread(path)
    return (m[..., 0] if m.ndim == 3 else m) > 0


def write_mask(path, mask) -> None:
    import imageio.v2 as imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, (np.asarray(mask, bool) * 255).astype(np.uint8))


def read_q(view_dir) -> np.ndarray:
    d = Path(view_dir)
    if (d / "q.txt").is_file():
        return np.loadtxt(d / "q.txt", dtype=np.float64).reshape(-1)
    if (d / "qpos.npy").is_file():
        return np.load(d / "qpos.npy").astype(np.float64).reshape(-1)
    raise FileNotFoundError(f"{d}: no q.txt or qpos.npy")


def _first(folder: Path, names):
    for n in names:
        if (folder / n).is_file():
            return folder / n
    return None


def view_dirs(folder) -> list:
    """view folders in capture order: numeric names by value (2 before 10), other names after them."""
    root = Path(folder) / "views"
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found")
    dirs = [d for d in root.iterdir() if d.is_dir()]
    bad = sorted(d.name for d in dirs if not (d / "mask.png").is_file())
    if bad:
        raise FileNotFoundError(f"{root}: no mask.png in {bad}")
    return sorted(dirs, key=lambda d: (0, int(d.name), "") if d.name.isdigit() else (1, 0, d.name))


def load_folder(folder) -> dict:
    """dict(K, T_init, T_gt (or None), views = [(q, mask), ...] in capture order, view_dirs)."""
    folder = Path(folder)
    if not (folder / "K.txt").is_file():
        raise FileNotFoundError(f"{folder}/K.txt not found")
    t_init = _first(folder, T_INIT_NAMES)
    if t_init is None:
        raise FileNotFoundError(f"{folder}: none of {T_INIT_NAMES} found")
    t_gt = _first(folder, T_GT_NAMES)
    dirs = view_dirs(folder)
    if not dirs:
        raise FileNotFoundError(f"{folder}/views has no view with mask.png")
    return dict(K=np.loadtxt(folder / "K.txt"), T_init=np.loadtxt(t_init),
                T_gt=None if t_gt is None else np.loadtxt(t_gt),
                views=[(read_q(d), read_mask(d / "mask.png")) for d in dirs], view_dirs=dirs)


def write_view(folder, index: int, q, mask) -> Path:
    """writes views/<index:02d>/{q.txt, mask.png} (e.g. from a capture loop)."""
    d = Path(folder) / "views" / f"{index:02d}"
    d.mkdir(parents=True, exist_ok=True)
    np.savetxt(d / "q.txt", np.asarray(q, np.float64).reshape(1, -1))
    write_mask(d / "mask.png", mask)
    return d
