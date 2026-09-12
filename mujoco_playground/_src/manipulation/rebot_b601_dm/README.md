# reBot Arm B601-DM

Pick-cube tasks for the reBot Arm B601-DM (Damiao motor edition), mirroring the
`PandaPickCube` family.

| Env | Task | Observation |
| --- | --- | --- |
| `RebotDmPickCube` | Grasp a 4×4×6 cm box and bring it to a random target position. | State |
| `RebotDmPickCubeOrientation` | Same, with a random target orientation (up to 45°). | State |
| `RebotDmPickCubeCartesian` | Lift the box to a fixed height with Cartesian (y, z, gripper) actions. | State, or pixels with `vision=True` |
| `RebotDmStackCube` | Pick up a blue 4 cm cube and place it on top of a red one, then let go. | State |
| `RebotDmStackCubeCartesian` | Same, with Cartesian (y, z, gripper) actions and both cubes on the gripper's plane. | State, or pixels with `vision=True` |
| `RebotDmPickCubeReal` | Lift a 5 cm cube in the calibrated real setup of rebot_serl, with (x, y, z, yaw, gripper) actions at 10 Hz, the real controller gains and image augmentation. | State, or pixels from the calibrated left D435i with `vision=True` |

```sh
train-jax-ppo --env_name RebotDmPickCube --impl warp
train-jax-ppo --env_name RebotDmPickCubeCartesian --impl warp --vision
```

## Model

`xmls/rebot_b601_dm.xml` is written for MJX from `DM/urdf/ReBot_Arm_DM.urdf` in
[Rebot_Arm_description](https://github.com/Yang-Ci/Rebot_Arm_description):

- Kinematics, inertials, joint ranges and effort limits (27 Nm on joint1–3,
  7 Nm on joint4–6, 8 N gripper) come from the URDF. URDF `rpy` values are used
  verbatim through `eulerseq="XYZ"`.
- The DM firmware tracks position targets with its own PID (`POS_VEL` mode), so
  the arm is modeled as gravity-compensated (`gravcomp="1"`) position actuators.
  They use the PD gains of the DM MuJoCo physics-grasp node, clamped to the URDF
  effort limits.
- The gripper is a single position actuator on `finger_left`: 0 is closed and
  0.05 m is fully open (100 mm opening). `finger_right` follows through a joint
  equality, matching the URDF mimic.
- Only the gripper collides. There is one box for the gripper base and one pad
  box per finger, fitted to the front and mid segments in
  `DM/meshes/mujoco_collision`. Arm links are visual only.

Do not reuse these parameters for the B601-RS. Its joint axes and ranges,
gripper, meshes and motors (RobStride, MIT mode) are different.

### Parameter sources

| Parameter | Value | Status |
| --- | --- | --- |
| Joint origins, axes, ranges | DM URDF | Identical in Rebot_Arm_description, reBotArm_control_py, and the DM ROS 2 description. Forward kinematics matches the DM MJCF of the ROS 2 package. |
| Masses, COMs, inertia tensors | DM URDF (`fullinertia`) | Identical in all DM URDFs. No DM measurement to check them against. |
| Effort limits | 27 / 7 Nm, 8 N | URDF values. |
| Gripper travel | 0.05 m per finger | Matches Rebot_Arm_description and the DM ROS 2 description. The reBotArm_control_py URDF lists 0.0285 m (57 mm opening). |
| Arm PD gains | kp 140/140/110/55/38/28, kv 7/7/5.5/2.5/1.8/1.4 | From the DM MuJoCo node; not identified against the real `POS_VEL` loop. |
| Joint damping, armature | 0.8, 0.01 | Copied from the DM MJCF; no motor data behind them. |
| Joint friction, velocity limits | Not modeled | The SDK caps `POS_VEL` speed at 5 rad/s (joint1–3) and 3 rad/s (joint4–6). |

## Differences from the Panda tasks

`RebotDmPickCube` and `RebotDmPickCubeOrientation` follow `PandaPickCube` for
observations, actions (per-step deltas of the joint and gripper position
targets) and rewards. The workspace is smaller:

- The box spawns within ±7 cm (x) and ±12 cm (y) of (0.28, 0), instead of
  ±20 cm.
- Targets are 8–15 cm above the box's starting height, instead of 20–40 cm.
  joint4's range stops the gripper from pointing straight down above about
  0.18 m.

`RebotDmPickCubeCartesian` follows `PandaPickCubeCartesian`:

- `kinematics.py` provides the IK in pure JAX, so it works with both MJX
  backends.
- The gripper keeps pointing down and moves in the vertical plane x = 0.28 m,
  where the box spawns within ±5 cm in y.
- The target is fixed at 0.15 m height.
- Each control step moves the gripper by at most 5 mm, and the fingers move at
  0.08 m/s, the URDF and MoveIt limit.
- The front camera `front` looks at the spawn line from x = 0.85 m.

### Stacking

`RebotDmStackCube` spawns the two cubes on opposite sides of y = 0 (sides
chosen at random) within `spawn_x_range` × `spawn_y_range`. The target for the
blue cube is the spot on top of the red cube wherever the red cube currently
is. Dense rewards pull the gripper to the blue cube and the blue cube to the
target; `stacked` needs the cubes in contact within `stack_xy_tolerance` /
`stack_z_tolerance`, and `success` additionally needs both finger pads off the
blue cube. A `red_still` term discourages pushing the red cube around. Episodes
do not end on success.

`RebotDmStackCubeCartesian` is the vision variant: it uses the Cartesian
controller and front camera of `RebotDmPickCubeCartesian` (shared in
`cartesian.py`), spawns both cubes on the line x = 0.28 m with |y| in
`spawn_y_range`, and keeps the stacking reward terms. Like
`PandaPickCubeCartesian`, the per-step reward is the improvement of the dense
score over the episode (so hovering next to the cube earns nothing), episodes
end on success, and `guide_prob` of the episodes start from the `picked`
keyframe with the blue cube already grasped. With the plain per-step reward of
`RebotDmStackCube`, a vision policy only learned to park the gripper next to
the blue cube. `vision_mode` works the same way as for the pick task.

### Sim-to-real pick (`RebotDmPickCubeReal`)

`pick_real.py` and `xmls/mjx_real_cube_camera.xml` reproduce the calibrated
setup of `~/robot/rebot_ws/rebot_serl` (calibration of 2026-09-11):

- The robot base frame is the world frame and the table (the real2sim table
  box, 1.33 × 1.10 m, top 13.4 mm above the base) is white over a grey room
  floor. The cube is 5 cm.
- The camera is `cam_b`, the D435i on the robot's left (serial 021222071805),
  at its calibrated pose with the SDK intrinsics (vertical FOV 42.66°,
  principal point within 3 px of the centre). The policy sees 96 × 72 pixels
  (4:3 like the 640 × 480 colour stream), rendered at 192 × 144 and
  average-pooled, because the Warp renderer casts one ray per pixel and a
  direct low-resolution render is aliased in a way a resized camera image is
  not (a policy trained on direct renders picked the cube in only 3 of 10
  episodes when fed images from MuJoCo's own renderer instead). On the
  robot, resize the colour image to 96 × 72 and scale to [0, 1].
- The episode starts with the gripper pointing down 0.19 m above the base
  (TCP 0.20 m, about 19 cm above the table) at x = 0.30 m, joints
  (0, −1.7095, −1.6319, 1.4934, 0, 0), with the fingers fully open. This is
  the `READY_Q` pose of rebot_serl raised by 5 cm, close to the highest
  top-down pose joint4 allows; use these joint angles as the reset pose on
  the robot. `gripper_travel` sets the opening: 0.05 m per finger (100 mm, the
  URDF value) by default; rebot_serl measured 28.5 mm per finger between the
  motor hard stops, which leaves only 3.5 mm per side around the cube and
  makes the grasp a sub-pixel alignment problem at 96 × 72 (RGB reached 27%
  and depth 44% success at 20M steps with that opening). Set it to the real
  opening before training for the robot.
- Actions, applied at 10 Hz: translation of the grasp-site target by up to
  1 cm per step along x, y, z, yaw about the vertical by up to 0.1 rad per
  step (the compliance clips of rebot_serl), and open/close (a < 0 closes).
  The target is clipped to `tip_x_range` × `tip_y_range` × `tip_z_range`
  and `yaw_range`; the arm joints follow through the same IK as the other
  Cartesian tasks. On the robot, apply the same increments to the TCP target
  of `RobotServer.set_target_pose` (the grasp site is 10 mm above the pinch
  point TCP along the approach axis, which does not matter for increments).
- Low-level control follows `config/controller.yaml` of rebot_serl
  (`actuation`): the arm joints use the MIT-mode PD gains of its compliance
  profile (kp 60/60/60/10/10/8, kd 4/4/4/1.2/1/0.8), the PD torque is bounded
  at kp × the reference lead it allows (9 Nm on joints 1–3, 3 Nm on 4–6;
  gravity is compensated separately, as on the robot), and the joint targets
  pass through the 50 ms first-order filter of its 500 Hz loop. The gripper
  speed and travel are the real ones; its force is bounded at 20 N rather
  than the 92 N stall force of the real motor, which would sink the pads
  into the soft-contact cube. With these gains the gripper lags a 1 cm step
  by 7–10 mm for a few steps, which the policy has to account for.
- The cube spawns at x 0.24–0.38 m, y ±0.12 m with a yaw of ±45°; the task
  is to lift it to 0.15 m (`target_height`, success within 5 cm in height).
- Every episode draws a cube colour (yellow, light green, purple or red with
  ±0.08 jitter), a table shade (0.85–1.0), a room-floor shade, a camera pose
  offset (±1 cm, ≤2°), a brightness factor (0.7–1.3), a gain per colour
  channel (±0.1), a blend with a 3 × 3 box blur and a pixel-noise level (up
  to 0.03). The colours and the camera pose are applied through per-world
  model fields at render time, so they work under the cached auto-reset of
  the training wrapper.
- `learning/rebot_real_policy.py` runs a trained checkpoint on camera images
  without a simulator and returns the target increments and the gripper
  command for `RobotServer`.

```sh
train-jax-ppo --env_name RebotDmPickCubeReal --impl warp --vision --use_tb --eval_video_envs 2 --render_camera cam_b
```

### RGB, depth and RGB-D

Pixel observations need `vision=True` and `impl="warp"`, which renders with the
MuJoCo Warp batch renderer. `vision_mode` selects the channels of the single
`pixels/view_0` observation:

| `vision_mode` | Observation | Channels |
| --- | --- | --- |
| `rgb` (default) | RGB | 3 |
| `depth` | depth | 1 |
| `rgbd` | RGB and depth stacked | 4 |

Depth is the distance from the camera divided by `depth_scale` (1.5 m by
default), clipped to [0, 1]. All modes use the same CNN policy, so they can be
compared directly:

```sh
for mode in rgb depth rgbd; do
  train-jax-ppo --env_name RebotDmPickCubeCartesian --vision --impl warp \
    --use_tb --eval_video_envs 2 --suffix $mode \
    --playground_config_overrides="{\"vision_mode\": \"$mode\"}"
done
tensorboard --logdir logs
```

## Results

Brax PPO with the `PandaPickCube` hyperparameters (32.8M steps, 2048 envs,
`impl="warp"`, about 90 s on an RTX 4090), evaluated on 1024 deterministic
episodes:

| Env | Success (box within 2 cm of target) | Median final distance | Median rotation error |
| --- | --- | --- | --- |
| `RebotDmPickCube` | 100.0% | 0.1 cm | 0.3° |
| `RebotDmPickCubeOrientation` | 99.9% | 0.2 cm | 1.4° |

`RebotDmStackCube` with its own hyperparameters (65.5M steps, 2048 envs, about
6 min): 99.6% of 256 deterministic episodes end with the blue cube stacked and
released, all 256 stack at some point, and the red cube moves 0.04 cm (median).

Vision PPO on `RebotDmPickCubeCartesian` (10M steps, 1024 envs, 64×64 pixels,
one front camera, same CNN for all modes), evaluated on 128 episodes at every
2.5M steps:

| `vision_mode` | Success at 2.5M / 5M / 7.5M / 10M steps | Final eval reward | Final avg. episode length | Train time (RTX 4090) |
| --- | --- | --- | --- | --- |
| `rgb` | 100% / 100% / 100% / 100% | 9.91 | 31 steps | 14.8 min |
| `depth` | 99.2% / 100% / 100% / 100% | 9.88 | 32 steps | 15.8 min |
| `rgbd` | 100% / 99.2% / 100% / 100% | 9.89 | 31 steps | 17.4 min (GPU shared with other jobs) |

All three modes solve this task; depth alone learns slightly slower (average
episode length 42 steps at 2.5M vs 31 for RGB) and catches up by 7.5M steps.

Vision PPO on `RebotDmStackCubeCartesian` (20M steps, 1024 envs, eval every 2M
steps on 128 episodes, episodes end on success):

| `vision_mode` | Success at 2M / 4M / 10M / 20M steps | Final avg. episode length |
| --- | --- | --- |
| `rgb` | 90% / 95% / 97% / 98% | 56 steps |
| `depth` | 6% / 27% / 53% / 53% | 170 steps |
| `rgbd` | 51% / 90% / 97% / 98% | 52 steps |

Stacking separates the modes: with RGB the red and blue cubes are trivially
told apart, RGB-D matches it after a slower start, and depth alone plateaus at
about half the episodes (the two 4 cm cubes are nearly indistinguishable in a
64×64 depth image).

## Assets

Meshes are not stored in this repository. On first load, `Rebot_Arm_description`
is cloned into `mujoco_playground/external_deps/` at the commit pinned in
`base.py` (`DESCRIPTION_COMMIT_SHA`), the same way mujoco_menagerie is. To use an
existing checkout instead, set `REBOT_ARM_DESCRIPTION_PATH`.

`Rebot_Arm_description` has no license file. The reBot Arm MuJoCo project it
comes from is licensed under CERN-OHL-W v2, so check the upstream terms before
redistributing the meshes.
