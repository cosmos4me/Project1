"""Template of a capture loop on a real xArm7 with a fixed camera. Fill in the three functions.

Safety: dv4calib checks the paths to its poses with the released EasyHeC planner. That planner sees a synthetic workspace:
the bounding box of the robot region and the floor plane near the base, not your cell. Plan and check every motion with
your own collision model before you execute it.
"""
import numpy as np

from dv4calib import Calibrator


def move_robot(q7):
    """move the arm to the 7 joint angles q7 (rad) on a collision-free path; the gripper stays as it is."""
    raise NotImplementedError


def read_joints():
    """the 7 arm joint angles the robot reached (rad)."""
    raise NotImplementedError


def segment_robot():
    """H x W boolean mask of link_base and link1..link7 in the current (undistorted) camera image, without the gripper."""
    raise NotImplementedError


def main():
    K = np.loadtxt("K.txt")                 # 3x3 intrinsics (pixels)
    T_init = np.loadtxt("T_init.txt")       # 4x4 initial camera <- base transform (metres)
    cal = Calibrator(K, T_init, read_joints(), segment_robot(), workdir="runs/cam0")

    def capture(q7):
        move_robot(q7)
        return read_joints(), segment_robot()

    T = cal.run(capture, n_views=5, log=print)
    np.savetxt("T_cam_from_base.txt", T)
    print(T)
    print("posterior std: position (mm)", cal.history[-1]["position_std_mm"], "rotation (deg)", cal.history[-1]["rotation_std_deg"])


if __name__ == "__main__":
    main()
