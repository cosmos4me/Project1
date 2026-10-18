# Notice

This repository includes code and model files from the projects below. Their license texts are in `third_party_licenses/`.

| files | origin | license |
|---|---|---|
| `dv4calib/se3.py`: `hat` | PyTorch3D 0.7.9 (`pytorch3d/transforms/so3.py`) | BSD License, Copyright (c) Meta Platforms, Inc. and affiliates (`third_party_licenses/pytorch3d-BSD.txt`) |
| `dv4calib/se3.py`: `_so3_exp_map`, `se3_exp_map`, `_se3_V_matrix`, `_get_se3_V_input` | PyTorch3D code, as copied by EasyHeC (`easyhec/utils/pytorch3d_se3.py`) | BSD License, Copyright (c) Meta Platforms, Inc. and affiliates (`third_party_licenses/pytorch3d-BSD.txt`) |
| `dv4calib/se3.py`: `to_array`, `se3_log_map`; `dv4calib/feasibility.py`: `SAPIENKinematicsModelStandalone`, `CollisionChecker`, the released workspace boundary and planner adapter; `dv4calib/geometry.py`: `link_meshes` (link-to-mesh mapping) | EasyHeC, https://github.com/ootts/EasyHeC, commit bc9771c | MIT License, Copyright (c) 2023 SU Lab, UCSD (`third_party_licenses/EasyHeC-MIT.txt`) |
| `dv4calib/assets/xarm7/` (URDF, SRDF and meshes, copied from EasyHeC's `assets/`) | UFACTORY xArm ROS packages `xarm_description` and `xarm_gripper` (https://github.com/xArm-Developer/xarm_ros) | BSD 3-Clause License, Copyright (c) 2018, UFACTORY Inc. (`third_party_licenses/xarm_ros-BSD-3-Clause.txt`) |

Dependencies installed separately keep their own licenses. nvdiffrast, which computes the exact risk, is distributed
under the Nvidia Source Code License (1-Way Commercial): others may use it for research or evaluation purposes only.

The license of the remaining code is to be chosen by the authors.
