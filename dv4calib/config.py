"""Fixed constants of the released xArm7 set-up, the cache directory, and the files of the active robot model.

DV4_CACHE  cache directory (default ~/.cache/dv4calib) for the distance fields of the clearance constraints (about 60 MB,
           built once in about 8 min, one file per robot model)."""
from __future__ import annotations

import os
from pathlib import Path

USER_CWD = Path.cwd()                  # working directory when dv4calib was imported


def user_path(p) -> Path:
    """a path given by the user; a relative path is taken relative to the working directory at import time."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else USER_CWD / p


def _robot():
    from .robot import active
    return active()


def asset_dir() -> Path:
    return _robot().asset_dir


def urdf() -> Path:
    return _robot().urdf


def srdf() -> Path:
    return _robot().srdf


def cache_dir() -> Path:
    p = user_path(os.environ.get("DV4_CACHE", Path.home() / ".cache" / "dv4calib"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def fields_cache() -> Path:
    return cache_dir() / f"fields_v2_{_robot().digest[:16]}.npz"


# Released EasyHeC exploration region: every link frame within R_MAX of REACH_CENTER.  REACH_CENTER is the released rule
# "midpoint of the y-extremal link positions of the first 20000 rows of the frozen candidate bank, x = y = 0"; its value
# is stored here so that the bank is not needed.
REACH_CENTER = (0.0, 0.0, 0.31899435818195343)
R_MAX = 0.5
MOVE_GROUP = "link_eef"            # MPLib move group of the released planner
PLANNER_NQ = 9                     # 7 arm joints + 2 gripper joints in the PyMP / MPLib model
PLANNING_TIME = 0.1                # s, released MPLib RRTConnect budget (unseeded)
WORKSPACE_SEED = 20230817          # released workspace point cloud (5000 points), see feasibility.workspace_points()
WORKSPACE_SHA256 = "9ee4502298fd6dc7a99313852ba197eea0b52ee73cdabb5d021aa71aebbb9fcb"   # of that cloud in the experiments
