# A0B arm + Shadow Hand (a0bsrh)

In-hand cube reorientation with a Shadow Dexterous Hand mounted on the JSK A0B
arm, a port of the LEAP hand task `LeapCubeReorient`.

| Env | Task | Actions | Observation |
| --- | --- | --- | --- |
| `A0bShadowCubeReorient` | Reorient a 6 cm cube resting in the palm to a goal orientation; a new goal is drawn after each success. The arm holds a fixed palm-up pose. | 20 (hand) | `state` (noisy hand joints, tracking error, cube pose error histories, last action) and `privileged_state` for the critic |

```sh
train-jax-ppo --env_name A0bShadowCubeReorient --impl warp --domain_randomization
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

