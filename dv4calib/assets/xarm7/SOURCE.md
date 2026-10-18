# xArm7 robot model (bundled)

Copied unchanged from EasyHeC (https://github.com/ootts/EasyHeC, commit bc9771cbc1137e450cd0f33d5bac0fcaf0f1015b,
directory `assets/`). Only the files that dv4calib reads are included:
- the URDF and SRDF;
- the meshes the URDF references, which SAPIEN, MPLib and PyMP load, and their convex hulls
  (`*.convex.stl`, tracked in EasyHeC), which the distance-field build reads;
- the visual STL meshes of the links, which are used to render masks and contours.

Licenses:
- EasyHeC is released under the MIT License, Copyright (c) 2023 SU Lab, UCSD (see `NOTICE.md` at the repository root).
- `xarm_description/` and `xarm_gripper/` are UFACTORY's xArm ROS description packages (https://github.com/xArm-Developer/xarm_ros):
  BSD 3-Clause License, Copyright (c) 2018, UFACTORY Inc.; full text in `third_party_licenses/xarm_ros-BSD-3-Clause.txt`
  at the repository root (their `package.xml` files are included).

| file | sha256 |
|---|---|
| `xarm7_with_gripper_reduced_dof.srdf` | `ee4952c440820bffbcb7ec8d3ca862e9f850cc7b74526cc30b2b68d48fec8887` |
| `xarm7_with_gripper_reduced_dof.urdf` | `5b90c89fddc9e00bd20f215494b57b3fec92f7bb71b023f693f335ba00021d43` |
| `xarm_description/meshes/collision/end_tool.STL` | `e06112781cdd615b7df536569f50addb445a6f4b4acced6d53b0742018b6c768` |
| `xarm_description/meshes/collision/end_tool.STL.convex.stl` | `d4fca06433ddf85f97b4ba95bd8bb1e89e41b1e0b6afc0157526e0fbc53bb8df` |
| `xarm_description/meshes/xarm7/visual/link1.STL` | `34b541122df84d2ef5fcb91b715eb19659dc15ad8d44a191dde481f780265636` |
| `xarm_description/meshes/xarm7/visual/link1_textured.glb` | `fc699576efc528fa469ce6136f72939f6b9c99615e1fca705da88416f9bfda0b` |
| `xarm_description/meshes/xarm7/visual/link1_textured.glb.convex.stl` | `df8a78e619be4d2feec876e35a731bdaf64278a74d6bb07ddebeae33a9cd718c` |
| `xarm_description/meshes/xarm7/visual/link2.STL` | `61e641cd47c169ecef779683332e00e4914db729bf02dfb61bfbe69351827455` |
| `xarm_description/meshes/xarm7/visual/link2_textured.glb` | `83c89dbf7f1e264a6e47a00f757a30666aac405e4b4b478d84efc9dc73563cdb` |
| `xarm_description/meshes/xarm7/visual/link2_textured.glb.convex.stl` | `e1bddd284da06450ed855ed49984236bc34d1dda54cde304c3835f1ae26f2246` |
| `xarm_description/meshes/xarm7/visual/link3.STL` | `9e2798e7946dd70046c95455d5ba96392d0b54a6069caba91dc4ca66e1379b42` |
| `xarm_description/meshes/xarm7/visual/link3_textured.glb` | `ae0ce44cd2531b6b5e41f760ec681474b0f437b586038497c0cd5e3ae710047a` |
| `xarm_description/meshes/xarm7/visual/link3_textured.glb.convex.stl` | `c497498c63d0b7da509e563953111700945632db564fba787186e91a4c8e4103` |
| `xarm_description/meshes/xarm7/visual/link4.STL` | `c757fee95f873191a0633c355c07a360032960771cabbd7593a6cdb0f1ffb089` |
| `xarm_description/meshes/xarm7/visual/link4_textured.glb` | `94af657f4434478d1f0db4bc0994a4f42c3659f15eb452a9f6306bf63161669d` |
| `xarm_description/meshes/xarm7/visual/link4_textured.glb.convex.stl` | `90e402d84d26ceeefae586086b222c24c5e2915f852f044425bbcca96a9151d6` |
| `xarm_description/meshes/xarm7/visual/link5.STL` | `715ad5787c5dab57589937fd47289882707b5e1eb997e340d567785b02f4ec90` |
| `xarm_description/meshes/xarm7/visual/link5_textured.glb` | `48630b4b7d1162d8a2f47995d05a09263296aa2ae50970ac893c5f95e8838661` |
| `xarm_description/meshes/xarm7/visual/link5_textured.glb.convex.stl` | `493e82fc1d5b1388b9669bc730bb1243c86320a79f45001282072403d3b679d7` |
| `xarm_description/meshes/xarm7/visual/link6.STL` | `85b320aa420497827223d16d492bba8de091173374e361396fc7a5dad7bdb0cb` |
| `xarm_description/meshes/xarm7/visual/link6_textured.glb` | `ed865e4c91983dd194d2a9e2de08d5eba2a3ffe2a35cf70901f7609fa4aafd28` |
| `xarm_description/meshes/xarm7/visual/link6_textured.glb.convex.stl` | `56bceedfd354943eb79bc7fb621ec31dda1865f66817b0cfec7b48a3f25a6025` |
| `xarm_description/meshes/xarm7/visual/link7.STL` | `97115d848fbf802cb770cd9be639ae2af993103b9d9bbb0c50c943c738a36f18` |
| `xarm_description/meshes/xarm7/visual/link7_silver.glb` | `484e3ee74e345f1e6ef24c37e684fd1a4328a56af558ee057f539c177f60c1cb` |
| `xarm_description/meshes/xarm7/visual/link7_silver.glb.convex.stl` | `cd8e1f0a980ad196bcb528cfc69e9a766f7a8e99e85e93a5ec8bc7f9b5c79c4d` |
| `xarm_description/meshes/xarm7/visual/link_base.STL` | `cf52275a913a0977595f5d12373f06583513e26fe18b8cf2827c74da94a4f0be` |
| `xarm_description/meshes/xarm7/visual/link_base_textured.glb` | `bdccb694dd55ef68c8b120d14f464a56c911ff4a55397b43bc19313b6184356b` |
| `xarm_description/meshes/xarm7/visual/link_base_textured.glb.convex.stl` | `53e365cffdebb7a3c8b9010bb9a4175136fdde8ec1d0a46cc4d57ec69da0e36e` |
| `xarm_description/package.xml` | `a5053deaaa6bc5386b5945bcbadd06eb1a648cfd83ad8851cc0e5c1149905939` |
| `xarm_gripper/meshes/base_link.STL` | `cdaa4cff22f7c9cff05c6a8ed32f94fd2b11a69d37dd97a159ccee2d8dd32f13` |
| `xarm_gripper/meshes/base_link_collision.STL` | `48a86ad100dc8fd508d942fced3a5ee61f755ab1c32ddf4050c1443cbb87dcea` |
| `xarm_gripper/meshes/base_link_collision.STL.convex.stl` | `3ca5f03107c6f335210710d945f13370aa9bf3755da9e5ec6c3005bda4b6d8df` |
| `xarm_gripper/meshes/left_finger.STL` | `a44756eb72f9c214cb37e61dc209cd7073fdff3e4271a7423476ef6fd090d2d4` |
| `xarm_gripper/meshes/left_finger_black.glb` | `afcd80affab381fe18904b5af6c436ef789f8765c753e2a1023c64aacd198e55` |
| `xarm_gripper/meshes/left_finger_black.glb.convex.stl` | `d1489062af6ffbdc247b244f55df0397961c772e22ce66a398986e10c8c97dcf` |
| `xarm_gripper/meshes/left_inner_knuckle.STL` | `e8e48692ad26837bb3d6a97582c89784d09948fc09bfe4e5a59017859ff04dac` |
| `xarm_gripper/meshes/left_outer_knuckle.STL` | `501665812b08d67e764390db781e839adc6896a9540301d60adf606f57648921` |
| `xarm_gripper/meshes/right_finger.STL` | `c5dee87c7f37baf554b8456ebfe0b3e8ed0b22b8938bd1add6505c2ad6d32c7d` |
| `xarm_gripper/meshes/right_finger_black.glb` | `73b40635529c81c7a89484aa88ae21c3b19d7408a15f5622360c05b6a9fb9514` |
| `xarm_gripper/meshes/right_finger_black.glb.convex.stl` | `4ed44c27dda19969713b611e370d0276e79afd33dd6a03f23dc9aebb7fbdbd42` |
| `xarm_gripper/meshes/right_inner_knuckle.STL` | `b41dd2c2c550281bf78d7cc6fa117b14786700e5c453560a0cb5fd6dfa0ffb3e` |
| `xarm_gripper/meshes/right_outer_knuckle.STL` | `75ca1107d0a42a0f03802a9a49cab48419b31851ee8935f8f1ca06be1c1c91e8` |
| `xarm_gripper/package.xml` | `99a35db070df1d61485ce644b5b738d23ff226fe3347a512bee117c61b6bab98` |
