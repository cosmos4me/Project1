"""dv4calib: active camera calibration from robot masks with a differentiable A-optimal risk (D v4).

    from dv4calib import Calibrator
    cal = Calibrator(K, T_init, q1, mask1)          # bundled xArm7 model; robot=dv4calib.xarm7(root) for other files
    T = cal.run(capture, n_views=5)                 # capture(q7) -> mask (or (q_reached, mask))

The computation modules are imported when the first Calibrator is created, after the robot model is set."""
from __future__ import annotations

__version__ = "0.4.0"
__all__ = ["Calibrator", "RobotModel", "xarm7", "load_folder"]


def __getattr__(name):
    if name == "Calibrator":
        from .calibrator import Calibrator
        return Calibrator
    if name in ("RobotModel", "xarm7"):
        from . import robot
        return getattr(robot, name)
    if name == "load_folder":
        from .io import load_folder
        return load_folder
    raise AttributeError(name)
