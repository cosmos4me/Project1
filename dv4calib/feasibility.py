"""Released xArm feasibility checks without the frozen candidate bank.

Same checks as faithful_xarm.frozen_candidate_source.ReleasedPyMPFeasibilityBackend (copied below, unchanged except that
the bank-specific bookkeeping is removed):
  cheap_ok = PyMP joint limits + PyMP self-collision + every SAPIEN link position within R_MAX of REACH_CENTER
  path_ok  = MPLib RRTConnect plan from the current configuration (planning time 0.1 s, unseeded) with the released
             workspace point cloud.
The classes Feas and Released keep the attribute names that the vendored code expects
(cands.backend, cands.planner_nq, cands.low, cands.high, cands.workspace_points, cands.max_distance_m,
cands._ensure_max_distance_center(), pad, nq, low7, high7, center, r_max, cheap(), checks(), path_ok(), final_ok())."""
from __future__ import annotations

import hashlib
import os
import threading
import time
import warnings
from types import SimpleNamespace

import mplib
import numpy as np
import sapien.core as sapien

from . import config as paths

_WORKSPACE_RNG_LOCK = threading.Lock()


# ---- EasyHeC (MIT, Copyright (c) 2023 SU Lab, UCSD): easyhec/structures/sapien_kin.py and
# easyhec/modeling/models/rb_solve/collision_checker.py (the methods dv4calib uses, unchanged)
class SAPIENKinematicsModelStandalone:

    def __init__(self, urdf_path):
        self.engine = sapien.Engine()

        self.scene = self.engine.create_scene()

        loader = self.scene.create_urdf_loader()
        # loader.scale = 10
        builder = loader.load_file_as_articulation_builder(urdf_path)

        self.robot: sapien.Articulation = builder.build(fix_root_link=True)

        self.robot.set_pose(sapien.Pose())

        self.robot.set_qpos(np.zeros(self.robot.dof))

        self.scene.step()

        self.model: sapien.PinocchioModel = self.robot.create_pinocchio_model()

    def compute_forward_kinematics(self, qpos, link_index):
        assert len(qpos) == self.robot.dof, f"qpos {len(qpos)} != {self.robot.dof}"
        self.model.compute_forward_kinematics(np.array(qpos))

        return self.model.get_link_pose(link_index)


class CollisionChecker:

    def __init__(self, cfg):

        urdf_path = cfg.urdf_path
        srdf_path = cfg.srdf_path
        move_group = cfg.move_group
        self.engine = sapien.Engine()

        self.scene = self.engine.create_scene()

        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot: sapien.Articulation = loader.load(urdf_path)
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        self.planner = mplib.Planner(
            urdf=urdf_path,
            srdf=srdf_path,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=move_group,
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7))

    def add_point_cloud(self, point_cloud):
        """
        :param point_cloud: (N, 3) numpy array in arm base frame
        """
        # import trimesh
        # box = trimesh.creation.box([0.1, 0.4, 0.2])
        # points, _ = trimesh.sample.sample_surface(box, 1000)
        # points += [0.55, 0, 0.1]
        self.planner.update_point_cloud(point_cloud)
        return


def _released_workspace_boundary_numeric() -> np.ndarray:
    """Exact numeric body of the released get_workspace_boundary (without Vis3D)."""
    import trimesh

    xmin, ymin, zmin, xmax, ymax, zmax = -0.2, -0.5, -0.5, 1.0, 0.5, 1.0
    box = trimesh.primitives.Box(extents=[xmax - xmin, ymax - ymin, zmax - zmin])
    box.apply_translation([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])
    points_workspace = box.sample(20_000)
    points_plane = np.zeros((20_000, 3))
    points_plane[:, 0] = np.random.uniform(-0.2, 0.15, size=20_000)
    points_plane[:, 1] = np.random.uniform(-0.4, 0.4, size=20_000)
    points_plane[:, 2] = 0.0
    points_plane = points_plane[np.linalg.norm(points_plane, axis=1) > 0.1]
    points = np.concatenate((points_workspace, points_plane), axis=0)
    indices = np.random.choice(len(points), 5_000, replace=False)
    return points[indices]


def workspace_points(seed: int = paths.WORKSPACE_SEED) -> np.ndarray:
    """The released 5000-point workspace cloud, replayed under a fixed and restored NumPy RNG."""
    with _WORKSPACE_RNG_LOCK:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            pts = np.asarray(_released_workspace_boundary_numeric(), dtype=np.float64)
        finally:
            np.random.set_state(state)
    pts = np.ascontiguousarray(pts)
    if seed == paths.WORKSPACE_SEED and hashlib.sha256(pts.tobytes()).hexdigest() != paths.WORKSPACE_SHA256:
        warnings.warn("the regenerated workspace cloud differs from the one of the experiments (another trimesh or NumPy "
                      "version?); feasibility checks and decisions can differ from the reported runs")
    return pts


def ensure_fields(log=print):
    """builds the distance-field cache of the active robot model once (file lock, atomic rename) and returns its path."""
    import fcntl
    final = paths.fields_cache()
    if final.is_file():
        return final
    with open(final.with_suffix(".lock"), "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        if not final.is_file():
            from . import clearance
            tmp = final.with_name(f"{final.stem}.tmp{os.getpid()}.npz")
            log(f"dv4calib: building the distance fields once (about 8 min) -> {final}")
            try:
                clearance.build_fields(cache=tmp)
                os.replace(tmp, final)
            finally:
                tmp.unlink(missing_ok=True)
    return final


def pad_qpos(qpos, planner_nq: int = paths.PLANNER_NQ) -> np.ndarray:
    value = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError("qpos must be finite shape (7,)")
    return np.concatenate((value, np.zeros(planner_nq - 7, dtype=np.float64)))


class ReleasedPyMPFeasibilityBackend:
    """Released PyMP self-collision, SAPIEN FK and MPLib path adapter (copied from faithful_xarm.frozen_candidate_source)."""

    def __init__(self, urdf, srdf, move_group, workspace, *, timestep=0.01, planning_time=paths.PLANNING_TIME):
        from pymp import Planner

        self.urdf, self.srdf, self.move_group = str(urdf), str(srdf), move_group
        self.timestep, self.planning_time = float(timestep), float(planning_time)
        self.workspace_points = np.ascontiguousarray(workspace, dtype=np.float64)
        self._self_planner = Planner(self.urdf, user_joint_names=None, ee_link_name=move_group, srdf=self.srdf)
        self._path_checker = CollisionChecker(SimpleNamespace(urdf_path=self.urdf, srdf_path=self.srdf, move_group=move_group))
        self._path_checker.add_point_cloud(self.workspace_points)
        if hasattr(self._path_checker.planner.planning_world, "set_use_point_cloud"):
            raise ImportError("dv4calib needs the MPLib 0.2 planning API (mplib==0.2.1)")
        self._path_api = "mplib_plan_qpos_point_cloud_always_active"
        self._kinematics = SAPIENKinematicsModelStandalone(self.urdf)
        self._link_count = len(self._kinematics.robot.get_links())

    def joint_limits(self):
        return (np.asarray(self._self_planner.robot.joint_limits[0], dtype=np.float64),
                np.asarray(self._self_planner.robot.joint_limits[1], dtype=np.float64))

    def in_self_collision(self, qpos) -> bool:
        return bool(self._self_planner.robot.computeCollisions(np.asarray(qpos).tolist()))

    def link_positions(self, qpos) -> np.ndarray:
        return np.asarray([self._kinematics.compute_forward_kinematics(qpos, i).p for i in range(self._link_count)], dtype=np.float64)

    def path_is_feasible(self, current_qpos, target_qpos, workspace=None):
        result = self._path_checker.planner.plan_qpos([np.asarray(target_qpos, dtype=np.float64)],
                                                      np.asarray(current_qpos, dtype=np.float64),
                                                      time_step=self.timestep, planning_time=self.planning_time)
        code = 0 if (isinstance(result, dict) and result.get("status") == "Success") else -1
        return code == 0, {"code": int(code), "path_api": self._path_api}


class _Source:
    """Bank-free stand-in for FrozenBankCandidateSource (only the attributes the decision code uses)."""
    _BACKEND = None

    def __init__(self):
        # working directory while the planners load the model: the directory above the asset directory, as in the
        # experiments (EasyHeC root above assets/); the loaders resolve mesh paths from the URDF directory
        cwd = os.getcwd(); os.chdir(paths.asset_dir().parent)
        try:
            if _Source._BACKEND is None:
                _Source._BACKEND = ReleasedPyMPFeasibilityBackend(paths.urdf(), paths.srdf(), paths.MOVE_GROUP, workspace_points())
        finally:
            os.chdir(cwd)
        self.backend = _Source._BACKEND
        self.urdf, self.srdf = paths.urdf(), paths.srdf()
        self.planner_nq = paths.PLANNER_NQ
        self.low, self.high = self.backend.joint_limits()
        assert len(self.low) == self.planner_nq, "PyMP model size differs from PLANNER_NQ"
        self.workspace_points = self.backend.workspace_points
        self.max_distance_m = paths.R_MAX

    def _ensure_max_distance_center(self):
        return np.asarray(paths.REACH_CENTER, np.float64)

    def _slice(self, *a, **k):
        raise NotImplementedError("dv4calib has no frozen candidate bank (D v4 does not use it)")


class Feas:
    """lscape_common.Feas interface: released cheap checks (vectorised) + released path check."""

    def __init__(self, center=None):
        self.cands = _Source()
        self.pad = pad_qpos
        self.nq = self.cands.planner_nq
        self.low7, self.high7 = np.asarray(self.cands.low[:7]), np.asarray(self.cands.high[:7])
        self.center = np.asarray(self.cands._ensure_max_distance_center() if center is None else center, np.float64)
        self.r_max = float(self.cands.max_distance_m)

    def cheap(self, Q):
        Q = np.atleast_2d(Q); n = len(Q)
        joint_ok = np.all((Q >= self.low7) & (Q <= self.high7), axis=1)
        self_coll = np.zeros(n, bool); dmax = np.full(n, np.nan)
        for i, q in enumerate(Q):
            qp = self.pad(np.asarray(q, np.float64), self.nq)
            self_coll[i] = bool(self.cands.backend.in_self_collision(qp))
            L = np.asarray(self.cands.backend.link_positions(qp), np.float64)
            dmax[i] = float(np.linalg.norm(L - self.center[None], axis=1).max())
        dist_ok = dmax <= self.r_max
        return {"joint_ok": joint_ok, "self_coll": self_coll, "dmax": dmax, "dist_ok": dist_ok, "cheap_ok": joint_ok & ~self_coll & dist_ok}

    def path(self, q1, q):
        t0 = time.time()
        ok, audit = self.cands.backend.path_is_feasible(self.pad(np.asarray(q1, np.float64), self.nq),
                                                        self.pad(np.asarray(q, np.float64), self.nq), self.cands.workspace_points)
        return bool(ok), str(audit.get("code")), time.time() - t0


class Released:
    """diffsel_v2.released.Released interface (cheap check, path check, final check)."""

    def __init__(self):
        self.cands = _Source()
        self.pad = pad_qpos
        self.nq = self.cands.planner_nq
        self.low7, self.high7 = np.asarray(self.cands.low[:7], np.float64), np.asarray(self.cands.high[:7], np.float64)
        self.center = np.asarray(self.cands._ensure_max_distance_center(), np.float64)
        self.r_max = float(self.cands.max_distance_m)
        self.workspace = np.asarray(self.cands.workspace_points, np.float64)
        self.n_path, self.t_path, self.n_cheap = 0, 0.0, 0

    @staticmethod
    def q7(q):
        q = np.asarray(q, np.float64).reshape(-1)
        return np.r_[q[:6], 0.0] if len(q) == 6 else q

    def link_positions(self, q):
        return np.asarray(self.cands.backend.link_positions(self.pad(self.q7(q), self.nq)), np.float64)

    def dmax(self, q):
        return float(np.linalg.norm(self.link_positions(q) - self.center[None], axis=1).max())

    def checks(self, q):
        q7 = self.q7(q); self.n_cheap += 1
        joint = bool(np.all((q7 >= self.low7) & (q7 <= self.high7)))
        selfc = bool(self.cands.backend.in_self_collision(self.pad(q7, self.nq)))
        dm = self.dmax(q7)
        return dict(joint_ok=joint, self_coll=selfc, dmax=dm, dist_ok=dm <= self.r_max, cheap_ok=joint and (not selfc) and dm <= self.r_max)

    def cheap_ok(self, q):
        return self.checks(q)["cheap_ok"]

    def path_ok(self, q_now, q):
        t0 = time.perf_counter()
        ok, _ = self.cands.backend.path_is_feasible(self.pad(self.q7(q_now), self.nq), self.pad(self.q7(q), self.nq), self.cands.workspace_points)
        self.n_path += 1; self.t_path += time.perf_counter() - t0
        return bool(ok)

    def final_ok(self, q, q_now, attempts=3):
        info = self.checks(q)
        if not info["cheap_ok"]:
            return False, 0, info
        for t in range(1, attempts + 1):
            if self.path_ok(q_now, q):
                return True, t, info
        return False, attempts, info

    def env_collision_goal(self, q):
        pl = self.cands.backend._path_checker.planner
        return bool(len(pl.check_for_env_collision(self.pad(self.q7(q), self.nq))) > 0)
