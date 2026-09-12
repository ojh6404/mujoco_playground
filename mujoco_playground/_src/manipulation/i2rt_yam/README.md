# i2rt YAM

Pick-cube and stacking tasks for the [i2rt YAM](https://i2rt.com/products/yam-manipulator)
(Yet Another Manipulator), the arm of the [ABC](https://github.com/amazon-far/abc)
project. They are the reBot Arm B601-DM tasks of `../rebot_b601_dm` run with the
YAM model, so that the two arms can be compared on the same tasks.

| Env | Task | Observation |
| --- | --- | --- |
| `YamPickCube` | Grasp a 4×4×6 cm box and bring it to a random target position. | State |
| `YamPickCubeOrientation` | Same, with a random target orientation (up to 45°). | State |
| `YamPickCubeCartesian` | Lift the box to a fixed height with Cartesian (y, z, gripper) actions. | State, or pixels with `vision=True` |
| `YamStackCube` | Pick up a blue 4 cm cube and place it on top of a red one, then let go. | State |
| `YamStackCubeCartesian` | Same, with Cartesian (y, z, gripper) actions and both cubes on the gripper's plane. | State, or pixels with `vision=True` |

```sh
train-jax-ppo --env_name YamPickCube --impl warp
train-jax-ppo --env_name YamPickCubeCartesian --impl warp --vision
```

The task classes subclass the reBot ones (`base.YamMixin` swaps the joint
names and the assets, `SCENE_XML` the scene), and `default_config()` of every
task returns the reBot config: the two arms have the same joint layout and
link lengths, so the box spawn, target and gripper ranges carry over. Scenes,
sensors, keyframe names and geom names (`hand_box`, `left_finger_pad`,
`right_finger_pad`, site `gripper`) match the reBot scenes.

## Model

`xmls/yam.xml` is the [mujoco_menagerie `i2rt_yam`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/i2rt_yam)
model (derived from i2rt's `yam.urdf`; the meshes are read from the menagerie
clone) with the changes ABC made to it and a few for MJX:

- ABC (`assets/put_bottles/assets/i2rt_yam/yam.xml`) widens the finger travel
  from 37.5 mm to 47.5 mm per finger (96 mm between the pads) and gives the
  finger pads a friction of 4.0. Both are kept.
- Only the gripper collides: a box fitted to the wrist link meshes
  (`hand_box`) and, per finger, the two pad boxes of the original model: the
  vertical 12 × 80 mm strip at the fingertip (`left_finger_pad`,
  `right_finger_pad`, used by the contact sensors) and the inclined strip
  above it (`*_finger_pad_inner`). The original collision capsules stay as
  non-colliding geoms so that the finger bodies keep their mass; the 0.6 mm
  fingertip spheres of the ABC model are dropped. Arm links do not collide,
  as for the reBot.
- The `gripper` site of the tasks sits on the fingertip strips, 44 mm off the
  wrist axis (the YAM fingers are offset from the wrist) and 17 mm behind the
  fingertips, with the reBot site convention (x toward the fingertips, y along
  the finger travel). The `tcp_site` of the i2rt model is kept.
- The base mesh reaches 10 mm below the base frame, so the arm body is mounted
  10 mm above the floor.
- Joint ranges, effort limits, armature, friction loss and the position
  actuator gains are the menagerie values: kp 40 / kv 2.5 on the DM-J4340
  joints 1–3, kp 10 / kv 1 on the DM-J4310 joints 5–6, kp 20 / kv 0.5 on
  joint 4, and kp 100 / kv 10 without a force limit on the gripper. All links
  are gravity compensated (`gravcomp="1"`) as in the menagerie model.

The keyframes are the reBot ones solved for the YAM: `home` and `low_home`
have the gripper pointing down at (0.28, 0, 0.16) and (0.28, 0, 0.13) with the
fingers 80 mm apart, joints (0, 1.7736, 1.6395, −1.4367, 0, 0) and
(0, 1.7682, 1.5139, −1.3165, 0, 0). `picked` was produced by closing the
gripper on the cube in simulation at the `low_home` pose.

## YAM vs reBot B601-DM

Both arms are yaw–pitch–pitch–pitch–roll–roll chains with Damiao motors, the
same 264 mm upper arm and nearly the same forearm; the YAM's joints 2–4 turn
the opposite way (positive folds the arm forward). Model values as loaded by
MJX:

| | YAM (`i2rt_yam`, ABC) | reBot B601-DM (`rebot_b601_dm`) |
| --- | --- | --- |
| Motors | DM-J4340 (joints 1–3), DM-J4310 (4–6) | DM-J4340P (joints 1–3), DM-J4310 (4–6, gripper) |
| Effort limits | 28 Nm / 10 Nm | 27 Nm / 7 Nm |
| Joint ranges | j1 −150…175°, j2 0…210°, j3 0…210°, j4 ±90°, j5 ±90°, j6 ±120° | j1 ±160°, j2 −180…0°, j3 −180…0°, j4 −107…90°, j5 ±90°, j6 ±180° |
| Base → joint 1 | 73 mm (with the 10 mm mount) | 85 mm |
| Upper arm (joint 2 → 3) | 264 mm | 264 mm |
| Forearm (joint 3 → 4) | 245 mm, 60 mm lateral offset | 243 mm, 54 mm lateral offset |
| Joint 6 → fingertips | 147 mm | 155 mm |
| Mass | 3.98 kg: base 0.22, links 0.12 / 1.24 / 0.85 / 0.46 / 0.35 / 0.37, fingers 2 × 0.18 | 4.99 kg: base 0.84, links 0.16 / 1.33 / 0.84 / 0.52 / 0.38 / 0.37, gripper 0.50 + 2 × 0.03 |
| Position-servo gains kp / kv | 40/2.5, 40/2.5, 40/2.5, 20/0.5, 10/1, 10/1 (menagerie) | 140/7, 140/7, 110/5.5, 55/2.5, 38/1.8, 28/1.4 (DM MuJoCo node) |
| Joint armature, damping | 0.032 (4340) / 0.0018 (4310), friction loss 0.1 | 0.01, damping 0.8 |
| Gripper | 47.5 mm per finger (96 mm), kp 100 N/m, no force limit (1 N per cm of error), fingertip strips 12 × 80 mm + inclined strips, friction 4.0, grasp centre 44 mm off the wrist axis | 50 mm per finger (100 mm), kp 1800 N/m, ±8 N, pads 49 × 27 mm, friction 1.5, on the wrist axis |
| Pointing-down reach at y = 0 (max x at 3 / 10 / 15 / 18 cm height) | 0.46 / 0.42 / 0.37 / 0.32 m | 0.51 / 0.47 / 0.43 / 0.38 m |
| Highest pointing-down pose at x = 0.28 m | 0.19 m | 0.21 m |

The YAM reaches 5 cm less because its grasp centre sits 44 mm behind the
wrist; joint 4 (±90° vs −107…90°) limits the pointing-down height of both arms
in the same way. The simulated YAM is a much softer position servo (3.5×
lower kp) with a slow, weak gripper: with kp 100 N/m the fingers take about
0.3 s to close and hold a 4 cm cube with about 1.5 N per pad, against 8 N for
the reBot.
