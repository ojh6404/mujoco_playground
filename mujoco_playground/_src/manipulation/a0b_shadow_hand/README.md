# A0B arm + Shadow Hand (a0bsrh)

Tasks for a Shadow Dexterous Hand mounted on the JSK A0B arm: in-hand cube
reorientation (a port of the LEAP hand task `LeapCubeReorient`) and, with the
arm moving, the pick-and-reorient and regrasping tasks of
[SAPG](https://sapg-rl.github.io/) (AllegroKuka), with one or two arms.

| Env | Task | Actions | Observation |
| --- | --- | --- | --- |
| `A0bShadowCubeReorient` | Reorient a 6 cm cube resting in the palm to a goal orientation; a new goal is drawn after each success. The arm holds a fixed palm-up pose. | 20 (hand) | `state` (noisy hand joints, tracking error, cube pose error histories, last action) and `privileged_state` for the critic |
| `A0bShadowCubePickReorient` | Lift the cube off a table and hold it at a goal pose (four cube corners as keypoints); a new goal after each success. | 26 (arm + hand) | `state` (full state as in SAPG) and `privileged_state` |
| `A0bShadowCubeRegrasp` | Same, but the goal is a position (one keypoint at the cube centre). | 26 | same |
| `A0bShadowTwoArmsCubeReorient` | Two arms facing each other across the table; the cube and the goals are placed near a random arm, so goals on the other side need a handover. | 52 | same, for both arms |
| `A0bShadowTwoArmsCubeRegrasp` | Two arms, goal position only. | 52 | same |

```sh
train-jax-ppo --env_name A0bShadowCubeReorient --impl warp --domain_randomization
train-jax-ppo --env_name A0bShadowTwoArmsCubeReorient --impl warp --use_tb --eval_video_envs 2 --render_camera side
```

## Model

`xmls/a0bsrh_mjx.xml` is generated from `xml/robot.xml`, `shared.xml` and
`shared_asset.xml` of the `a0bsrh_model` ROS package (JSK GitLab, the hand
part comes from the OpenAI Gym Shadow Hand assets). Changes for MJX:

- Mesh assets are referenced by file name and loaded from
  `$A0BSRH_MODEL_PATH/xml/assets` (default `~/robot/shadowhand_ws/a0bsrh_model`).
  The package is not public, so nothing is cloned automatically.
- The arm and forearm collision meshes are disabled (`contype/conaffinity 0`);
  only the hand collides.
- The 0.5 mm contact margins are set to 0: the MJX JAX backend rejects margins
  on mesh contacts (the fingertips are convex meshes).
- Fingertip sites (`ff_tip`, `mf_tip`, `rf_tip`, `lf_tip`, `th_tip`) and a
  `grasp_site` 4.5 cm above the palm are added for the sensors.
- Joints, fixed tendons (the coupled distal finger joints), contact pairs and
  position actuators are the original ones: 6 arm + 2 wrist + 18 finger
  actuators, 30 joints.

The scene `xmls/scene_mjx_cube.xml` reuses the LEAP dex-cube mesh and textures
scaled to 6 cm. Its `home` keyframe holds the arm palm-up (`a0b_joint5` = π,
the other arm joints at 0, palm 1.37 m above the floor) with the cube settled
on the palm.

## Task

The arm actuators keep their keyframe targets; the policy outputs deltas for
the 20 hand actuators (`action_size` = 20). Rewards, observation noise, goal
updates, perturbations and domain randomization follow `LeapCubeReorient`;
the episode ends when the cube drops more than 10 cm below its resting height.
`reset_noise` controls the hand-pose and cube-position randomization at reset,
which is smaller than for the LEAP hand because the Shadow palm is flatter.
By default the cube starts face down with a random yaw
(`reset_noise.cube_yaw_only`); with a uniformly random start orientation
about 8% of the cubes roll off the flat palm before the policy acts, and
goals are uniformly random orientations either way.

## Pick-and-reorient and regrasping (SAPG tasks)

`pick_reorient.py` follows `allegro_kuka_two_arms.py` of SAPG. The scenes
attach `a0bsrh_table_mjx.xml` (the robot with the forearm colliding with the
table) once (`scene_mjx_pick_reorient.xml`, prefix `arm0/`, table in front of
the arm) or twice (`scene_mjx_two_arms.xml`, prefixes `arm0/` and `arm1/`,
arms 2 m apart facing each other across a 1.2 m table). The hand geoms of the
two arms get different collision bits in `_configure_mj_model` so the hands
collide with each other but not with themselves.

- **Home pose.** The arm starts palm down with the grasp site 0.72 m in front
  of its base, 0.2 m above the table (`ARM_HOME_QPOS`, an IK solution; this
  arm keeps the palm down at that reach only by folding in the horizontal
  plane). The cube spawns 0.62–0.82 m in front of a random arm, ±0.15 m
  sideways, with a uniformly random orientation, and drops onto the table.
  Every episode draws a new robot pose, cube pose and goal, also under the
  training wrapper's cached reset.
- **Actions.** Per arm, 6 arm and 20 hand actuator target deltas, clipped to
  `arm_action_scale` (0.1 rad) and `hand_action_scale` (0.5 rad) per step and
  to the actuator ranges. SAPG uses absolute targets; deltas suit the A0B's
  stiff position actuators better.
- **Goals.** A pose (reorientation) or position (regrasping) drawn 0.55–0.85 m
  in front of a random arm, ±0.3 m sideways, 0.1–0.38 m above the table.
  Keypoints are the cube corners scaled by `keypoint_scale` (1.5), or the
  centre for regrasping; a goal is reached when the furthest keypoint is
  within `success_tolerance` × `keypoint_scale` for `success_steps` steps.
- **Rewards** (SAPG scales, so `reward_scaling` is 0.01 in the PPO config):
  fingertips getting closer to the cube than ever in the episode (×50; with
  one arm only until the cube is lifted), cube height until it counts as
  lifted (×20), a one-off lift bonus (300), the furthest keypoint getting
  closer to the goal than ever (×200, once lifted), a bonus per step near the
  goal (1000), and small action penalties. Unlike SAPG, dropping a lifted
  cube off the table costs the lift bonus again (−300): without that penalty
  the policy learned to lift the cube and throw it off the table, ending the
  episode to collect the next lift bonus. The penalty is not charged before
  the cube was lifted, because charging it then stopped the policy from
  touching the cube at all.
- **Curriculum.** As in SAPG the tolerance starts at 7.5 cm and is multiplied
  by 0.9, down to 1 cm, but per environment: whenever an episode ends with at
  least three goals reached. The tolerance is kept in `info`, which the
  training wrapper preserves across episodes. Evaluation episodes start at
  the initial tolerance; `success_strict` reports successes at the target
  tolerance.
- **Termination.** The cube falling off the table, `max_consecutive_successes`
  goals, or 600 steps.

Differences from SAPG: one 6 cm cube instead of randomized cuboids, no random
forces on the object, and no domain randomization.

### Results

Brax PPO with the config in `manipulation_params.py` (8192 envs,
`impl="warp"`, evaluated on 128 episodes of 600 steps at the initial 7.5 cm
tolerance, i.e. 11 cm for the furthest keypoint), on an RTX 4090:

| Env | Steps | Goals reached per episode | Episodes ending with the cube off the table | Time |
| --- | --- | --- | --- | --- |
| `A0bShadowCubePickReorient` | 258M (the run died in a sporadic Warp out-of-memory during an evaluation; the last checkpoint is the final policy) | 3.4 | 35% | 4.3 min per 21.5M steps |

Learning curve of the single-arm task: lifting is learned at about 40M steps,
the first goals are reached at 100M (1.7 per episode), then 2.9 at 170M and
3.4 at 258M. Two failure modes shaped the reward: without a fall penalty the
policy lifted the cube and threw it off the table to restart the episode and
collect the lift bonus again; with a fall penalty charged also before the
first lift, it stopped touching the cube.
