"""Robot model input: the URDF (kinematics, collision geometry), the SRDF (self-collision pairs) and the visual mesh of
every link (used to render masks and contours).

dv4calib implements the method for the xArm7 with gripper (EasyHeC's `xarm7_with_gripper_reduced_dof` model).
Several parts are specific to this robot: the mask link set, the capsule surrogate, the reach rule and the planner
set-up.
- `xarm7()` returns the bundled model (dv4calib/assets/xarm7).
- `xarm7(root)` returns the same model read from another directory with the same layout, for example the `assets/`
  folder of an EasyHeC checkout.

The vendored modules read the model when they are imported, so a process uses one robot model.
"""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets" / "xarm7"
URDF_NAME = "xarm7_with_gripper_reduced_dof.urdf"
SRDF_NAME = "xarm7_with_gripper_reduced_dof.srdf"

# EasyHeC easyhec/structures/xarm_mapping.py: link -> visual mesh (relative to the URDF directory), same order
XARM7_LINK_MESHES = (
    ("link_base", "xarm_description/meshes/xarm7/visual/link_base.STL"),
    ("link1", "xarm_description/meshes/xarm7/visual/link1.STL"),
    ("link2", "xarm_description/meshes/xarm7/visual/link2.STL"),
    ("link3", "xarm_description/meshes/xarm7/visual/link3.STL"),
    ("link4", "xarm_description/meshes/xarm7/visual/link4.STL"),
    ("link5", "xarm_description/meshes/xarm7/visual/link5.STL"),
    ("link6", "xarm_description/meshes/xarm7/visual/link6.STL"),
    ("link7", "xarm_description/meshes/xarm7/visual/link7.STL"),
    ("xarm_gripper_base_link", "xarm_gripper/meshes/base_link.STL"),
    ("left_outer_knuckle", "xarm_gripper/meshes/left_outer_knuckle.STL"),
    ("left_finger", "xarm_gripper/meshes/left_finger.STL"),
    ("left_inner_knuckle", "xarm_gripper/meshes/left_inner_knuckle.STL"),
    ("right_outer_knuckle", "xarm_gripper/meshes/right_outer_knuckle.STL"),
    ("right_finger", "xarm_gripper/meshes/right_finger.STL"),
    ("right_inner_knuckle", "xarm_gripper/meshes/right_inner_knuckle.STL"),
)
# The input masks are silhouettes of these links (the gripper is not part of the mask), as in the experiments.
MASK_LINKS = ("link_base",) + tuple(f"link{i}" for i in range(1, 8))
REQUIRED_LINKS = MASK_LINKS + ("link_eef", "xarm_gripper_base_link", "link_tcp", "left_finger", "right_finger")


@dataclass(frozen=True)
class RobotModel:
    urdf: Path
    srdf: Path
    link_meshes: tuple                     # ((link name, absolute mesh path), ...)
    name: str = "xarm7"

    @property
    def asset_dir(self) -> Path:
        return self.urdf.parent

    def referenced_files(self) -> list:
        """mesh files referenced by the URDF (loaded by SAPIEN, MPLib and PyMP) and their convex hulls
        (<mesh>.convex.stl, read by the distance-field build and by SAPIEN), absolute paths."""
        refs = {(self.asset_dir / f).resolve() for f in re.findall(r'filename="([^"]+)"', self.urdf.read_text())}
        hulls = {f.with_name(f.name + ".convex.stl") for f in refs}
        return sorted(refs | {h for h in hulls if h.is_file()} | self.collision_hulls())

    def collision_hulls(self) -> set:
        """<collision mesh>.convex.stl of every collision mesh in the URDF (required by the distance-field build)."""
        root = ET.parse(self.urdf).getroot()
        out = set()
        for c in root.iter("collision"):
            for g in c.iter("mesh"):
                f = (self.asset_dir / g.get("filename")).resolve()
                out.add(f.with_name(f.name + ".convex.stl"))
        return out

    def files(self) -> list:
        return sorted({self.urdf, self.srdf, *[Path(p) for _, p in self.link_meshes], *self.referenced_files()})

    @property
    def digest(self) -> str:
        """sha256 over the content of every model file (keys the distance-field cache)."""
        h = hashlib.sha256()
        for f in self.files():
            h.update(str(f.relative_to(self.asset_dir) if f.is_relative_to(self.asset_dir) else f.name).encode())
            h.update(hashlib.sha256(f.read_bytes()).digest())
        return h.hexdigest()

    def arm_joint_limits(self):
        """(lower, upper) of joint1..joint7 from the URDF (rad)."""
        root = ET.parse(self.urdf).getroot()
        lim = {j.get("name"): j.find("limit") for j in root.findall("joint")}
        lo = np.array([float(lim[f"joint{i}"].get("lower")) for i in range(1, 8)])
        hi = np.array([float(lim[f"joint{i}"].get("upper")) for i in range(1, 8)])
        return lo, hi

    def absolute(self) -> "RobotModel":
        r = lambda p: Path(p).expanduser().resolve()
        return RobotModel(urdf=r(self.urdf), srdf=r(self.srdf), link_meshes=tuple((n, r(p)) for n, p in self.link_meshes),
                          name=self.name)

    def validate(self) -> "RobotModel":
        missing = [str(f) for f in self.files() if not f.is_file()]
        if missing:
            raise FileNotFoundError(f"robot model files not found: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        links = set(re.findall(r'<link\s+name="([^"]+)"', self.urdf.read_text()))
        absent = [n for n in REQUIRED_LINKS if n not in links]
        if absent:
            raise ValueError(f"{self.urdf.name}: links {absent} not found; dv4calib supports the xArm7 with gripper "
                             f"({URDF_NAME}) only")
        if [n for n, _ in self.link_meshes][:len(MASK_LINKS)] != list(MASK_LINKS):
            raise ValueError("link_meshes must start with the mask links " + ", ".join(MASK_LINKS))
        return self


def xarm7(root=None) -> RobotModel:
    """the xArm7 model from `root` (a directory with the layout of dv4calib/assets/xarm7 or EasyHeC's assets/)."""
    root = ASSETS if root is None else Path(root).expanduser().resolve()
    return RobotModel(urdf=root / URDF_NAME, srdf=root / SRDF_NAME,
                      link_meshes=tuple((n, root / p) for n, p in XARM7_LINK_MESHES)).validate()


_ACTIVE: RobotModel | None = None


def use(model: RobotModel) -> RobotModel:
    """set the robot model of this process (before the first calibration); the same model may be set again."""
    global _ACTIVE
    model = model.absolute()
    if _ACTIVE is None:
        _ACTIVE = model.validate()
    elif model != _ACTIVE and model.digest != _ACTIVE.digest:
        raise RuntimeError("a different robot model is already in use in this process (the vendored modules read the "
                           "model at import); start a new process to use another model")
    return _ACTIVE


def active() -> RobotModel:
    return _ACTIVE if _ACTIVE is not None else use(xarm7())
