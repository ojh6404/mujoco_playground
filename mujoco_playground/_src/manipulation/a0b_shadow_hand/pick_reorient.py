# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pick up a cube from a table and hold it at a goal, with one or two A0B arms.

A port of the AllegroKuka tasks of SAPG (https://sapg-rl.github.io/) to the
A0B arm with a Shadow Hand. The policy drives the arm and the hand, lifts the
cube off the table and brings it to a goal: a pose, matched through four cube
corners (reorientation), or a position, matched through the cube centre
(regrasping). A new goal is drawn after each success. With two arms facing
each other across the table, the cube and the goals are placed near a random
arm, so goals on the other side need a handover.
"""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.a0b_shadow_hand import a0bsrh_constants as consts
from mujoco_playground._src.manipulation.a0b_shadow_hand import base as a0bsrh_base
from mujoco_playground._src.manipulation.leap_hand import base as leap_hand_base

NU_ARM = len(consts.ARM_ACTUATOR_NAMES)
NU_ARM_HAND = NU_ARM + consts.NU_HAND
NQ_ARM_HAND = len(consts.ARM_JOINT_NAMES) + consts.NQ_HAND
NUM_FINGERTIPS = len(consts.FINGERTIP_NAMES)

# Keypoints relative to the cube centre, in units of half the cube size.
_KEYPOINT_OFFSETS = {
    "reorient": [[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]],
    "regrasp": [[0, 0, 0]],
}


def default_config(
    num_arms: int = 1, task: str = "reorient"
) -> config_dict.ConfigDict:
  """Returns the config of the pick tasks.

  Args:
    num_arms: 1 (scene_mjx_pick_reorient.xml) or 2 (scene_mjx_two_arms.xml).
    task: "reorient" (goal pose, four corner keypoints) or "regrasp" (goal
      position, one keypoint at the cube centre).
  """
  if task not in _KEYPOINT_OFFSETS:
    raise ValueError(f"Unknown task {task!r}.")
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.01,
      episode_length=600,
      action_repeat=1,
      num_arms=num_arms,
      task=task,
      # Change of the position targets per step (rad) for an action of 1.
      arm_action_scale=0.1,
      hand_action_scale=0.5,
      # Uniform noise (rad) around the home pose at the start of an episode.
      reset_noise=config_dict.create(arm_pose=0.05, hand_pose=0.03),
      # Cube spawn box in the frame of one arm base (x forward, m); `height`
      # is the height of the cube centre above the table top.
      cube_spawn=config_dict.create(
          x=[0.62, 0.82], y=[-0.15, 0.15], height=[0.06, 0.08]
      ),
      # Goal volume in the frame of one arm base; `z` is above the table top.
      target_volume=config_dict.create(
          x=[0.55, 0.85], y=[-0.3, 0.3], z=[0.1, 0.38]
      ),
      # The keypoints sit on the cube corners scaled by this factor (SAPG).
      keypoint_scale=1.5,
      # A goal is reached when the furthest keypoint is within
      # success_tolerance * keypoint_scale of its goal position for
      # success_steps consecutive steps.
      success_tolerance=0.075,
      success_steps=1,
      # Tolerance curriculum after SAPG: whenever an episode ends with at
      # least min_successes goals, that environment's tolerance is multiplied
      # by `increment`, down to target_tolerance. The curriculum state lives
      # in `info`, which the training wrapper keeps across episodes.
      curriculum=config_dict.create(
          enable=True,
          target_tolerance=0.01,
          increment=0.9,
          min_successes=3,
      ),
      # The episode ends after this many goals.
      max_consecutive_successes=50,
      # Height (m) above its resting height at which the cube counts as lifted.
      lift_threshold=0.15,
      reward_config=config_dict.create(
          # Fingertips getting closer to the cube than ever in the episode.
          fingertip_delta=50.0,
          # Cube height above the table, until it counts as lifted.
          lifting=20.0,
          # Once, when the cube first counts as lifted.
          lift_bonus=300.0,
          # Furthest keypoint getting closer to the goal than ever for this
          # goal, once the cube is lifted.
          keypoint=200.0,
          # Per step near the goal, spread over success_steps.
          reach_goal_bonus=1000.0,
          arm_action_penalty=0.003,
          hand_action_penalty=0.0003,
          # Once when a lifted cube falls off the table, which ends the
          # episode. Cancels the lift bonus: otherwise lifting the cube and
          # throwing it off the table to start a new episode pays better than
          # holding it at the goal (seen at 40M steps with no penalty). It is
          # not charged before the cube was lifted, or the policy stops
          # touching the cube at all (seen at 64M steps).
          fall_penalty=-300.0,
      ),
      obs_clip=10.0,
      impl="warp",
      # Contact buffers for the training batch (8192 envs with one arm, 4096
      # with two). The Warp convex narrowphase allocates scratch arrays
      # proportional to naccdmax (= naconmax) on every step, several GB for
      # the two-arm scene at 64 * 8192, and training runs then died with
      # sporadic Warp out-of-memory errors.
      naconmax=40 * 8192 if num_arms == 1 else 64 * 4096,
      njmax=300 if num_arms == 1 else 500,
  )


class CubePickReorient(a0bsrh_base.A0bShadowHandEnv):
  """Lift a cube off a table and hold it at a goal pose or position."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    overrides = config_overrides or {}
    self._num_arms = int(overrides.get("num_arms", config.num_arms))
    if self._num_arms not in (1, 2):
      raise ValueError("num_arms must be 1 or 2.")
    xml = consts.PICK_XML if self._num_arms == 1 else consts.TWO_ARMS_XML
    super().__init__(
        xml_path=xml.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _configure_mj_model(self, mj_model: mujoco.MjModel) -> None:
    """Makes the two hands collide with each other, but not with themselves.

    The hand collision geoms come with contype 1 and conaffinity 0 (their
    self-collisions are explicit contact pairs). Arm 0 keeps contype 1 and
    gets conaffinity 4; arm 1 gets contype 4 and conaffinity 1. Geoms of the
    same hand then never match, geoms of different hands do, and both hands
    still collide with the table (conaffinity 3), the cube and the floor
    (contype and conaffinity 1).
    """
    if self._num_arms != 2:
      return
    for gid in range(mj_model.ngeom):
      name = mj_model.geom(gid).name
      if mj_model.geom_contype[gid] != 1 or mj_model.geom_conaffinity[gid]:
        continue  # Not a hand collision geom.
      if name.startswith("arm0/"):
        mj_model.geom_conaffinity[gid] = 4
      elif name.startswith("arm1/"):
        mj_model.geom_contype[gid] = 4
        mj_model.geom_conaffinity[gid] = 1

  def _post_init(self) -> None:
    m = self._mj_model
    num_arms = self._num_arms

    table = m.geom("table")
    self._table_top = float(table.pos[2] + table.size[2])
    self._cube_half_size = float(m.geom("cube").size[0])
    self._cube_rest_height = self._table_top + self._cube_half_size
    self._cube_body_id = m.body("cube").id
    cube_jid = m.joint("cube_freejoint").id
    self._cube_qadr = int(m.jnt_qposadr[cube_jid])
    self._goal_mocap_id = int(m.body_mocapid[m.body("goal").id])

    # Fixed frames of the arm bases.
    self._base_pos = np.stack(
        [m.body(f"arm{i}").pos for i in range(num_arms)]
    ).astype(np.float32)
    mats = []
    for i in range(num_arms):
      mat = np.zeros(9)
      mujoco.mju_quat2Mat(mat, m.body(f"arm{i}").quat)
      mats.append(mat.reshape(3, 3))
    self._base_mat = np.stack(mats).astype(np.float32)

    # Joints, actuators and sites, per arm.
    joint_names = consts.ARM_JOINT_NAMES + consts.HAND_JOINT_NAMES
    act_names = consts.ARM_ACTUATOR_NAMES + consts.HAND_ACTUATOR_NAMES
    self._qids = np.stack([
        mjx_env.get_qpos_ids(m, [f"arm{i}/{n}" for n in joint_names])
        for i in range(num_arms)
    ])
    self._dqids = np.stack([
        mjx_env.get_qvel_ids(m, [f"arm{i}/{n}" for n in joint_names])
        for i in range(num_arms)
    ])
    jids = np.array([
        [m.joint(f"arm{i}/{n}").id for n in joint_names]
        for i in range(num_arms)
    ])
    self._jnt_lower = jp.array(m.jnt_range[jids, 0])
    self._jnt_upper = jp.array(m.jnt_range[jids, 1])
    self._act_ids = np.array([
        [m.actuator(f"arm{i}/{n}").id for n in act_names]
        for i in range(num_arms)
    ])
    self._act_qids = np.stack([
        mjx_env.get_qpos_ids(m, [f"arm{i}/{n}" for n in act_names])
        for i in range(num_arms)
    ])
    self._ctrl_lower = jp.array(m.actuator_ctrlrange[self._act_ids, 0])
    self._ctrl_upper = jp.array(m.actuator_ctrlrange[self._act_ids, 1])
    self._palm_sites = np.array(
        [m.site(f"arm{i}/grasp_site").id for i in range(num_arms)]
    )
    self._tip_sites = np.array([
        [m.site(f"arm{i}/{n}").id for n in consts.FINGERTIP_NAMES]
        for i in range(num_arms)
    ])

    self._home_q = jp.tile(
        jp.array(consts.ARM_HOME_QPOS + consts.HAND_HOME_QPOS), (num_arms, 1)
    )
    self._init_q = jp.array(m.qpos0)
    self._action_scale = jp.concatenate([
        jp.full((NU_ARM,), self._config.arm_action_scale),
        jp.full((consts.NU_HAND,), self._config.hand_action_scale),
    ])

    self._keypoint_offsets = (
        jp.array(_KEYPOINT_OFFSETS[self._config.task], dtype=float)
        * self._cube_half_size
        * self._config.keypoint_scale
    )
    self._num_keypoints = self._keypoint_offsets.shape[0]

  # Sampling.

  def _local_to_world(self, arm: jax.Array, local: jax.Array) -> jax.Array:
    """Maps a point in the frame of arm `arm` (z ignored) to the world."""
    base_pos = jp.array(self._base_pos)[arm]
    base_mat = jp.array(self._base_mat)[arm]
    return base_pos + base_mat @ local

  def _sample_goal(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
    rng_arm, rng_pos, rng_quat = jax.random.split(rng, 3)
    arm = jax.random.randint(rng_arm, (), 0, self._num_arms)
    vol = self._config.target_volume
    lo = jp.array([vol.x[0], vol.y[0], vol.z[0]])
    hi = jp.array([vol.x[1], vol.y[1], vol.z[1]])
    local = jax.random.uniform(rng_pos, (3,), minval=lo, maxval=hi)
    pos = self._local_to_world(arm, local.at[2].set(0.0))
    pos = pos.at[2].set(self._table_top + local[2])
    if self._config.task == "reorient":
      quat = leap_hand_base.uniform_quat(rng_quat)
    else:
      quat = jp.array([1.0, 0.0, 0.0, 0.0])
    return pos, quat

  def _sample_episode(self, rng: jax.Array) -> Dict[str, jax.Array]:
    """Samples the robot pose, cube pose, controls and goal of an episode."""
    rng_arm, rng_hand, rng_side, rng_cube, rng_quat, rng_goal = (
        jax.random.split(rng, 6)
    )
    noise = jp.concatenate(
        [
            self._config.reset_noise.arm_pose
            * jax.random.uniform(
                rng_arm, (self._num_arms, NU_ARM), minval=-1, maxval=1
            ),
            self._config.reset_noise.hand_pose
            * jax.random.uniform(
                rng_hand, (self._num_arms, consts.NQ_HAND), minval=-1, maxval=1
            ),
        ],
        axis=-1,
    )
    q_robot = jp.clip(self._home_q + noise, self._jnt_lower, self._jnt_upper)
    qpos = self._init_q.at[self._qids.ravel()].set(q_robot.ravel())

    arm = jax.random.randint(rng_side, (), 0, self._num_arms)
    spawn = self._config.cube_spawn
    lo = jp.array([spawn.x[0], spawn.y[0], spawn.height[0]])
    hi = jp.array([spawn.x[1], spawn.y[1], spawn.height[1]])
    local = jax.random.uniform(rng_cube, (3,), minval=lo, maxval=hi)
    cube_pos = self._local_to_world(arm, local.at[2].set(0.0))
    cube_pos = cube_pos.at[2].set(self._table_top + local[2])
    cube_quat = leap_hand_base.uniform_quat(rng_quat)
    qpos = qpos.at[self._cube_qadr : self._cube_qadr + 7].set(
        jp.concatenate([cube_pos, cube_quat])
    )

    ctrl = jp.zeros(self._mjx_model.nu)
    ctrl = ctrl.at[self._act_ids.ravel()].set(qpos[self._act_qids.ravel()])
    goal_pos, goal_quat = self._sample_goal(rng_goal)
    mocap_pos = jp.zeros((self._mjx_model.nmocap, 3))
    mocap_quat = jp.tile(
        jp.array([1.0, 0.0, 0.0, 0.0]), (self._mjx_model.nmocap, 1)
    )
    return {
        "qpos": qpos,
        "qvel": jp.zeros(self._mjx_model.nv),
        "ctrl": ctrl,
        "mocap_pos": mocap_pos.at[self._goal_mocap_id].set(goal_pos),
        "mocap_quat": mocap_quat.at[self._goal_mocap_id].set(goal_quat),
    }

  # Environment API.

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng_episode = jax.random.split(rng)
    sample = self._sample_episode(rng_episode)
    data = mjx_env.make_data(
        self._mj_model,
        qpos=sample["qpos"],
        qvel=sample["qvel"],
        ctrl=sample["ctrl"],
        mocap_pos=sample["mocap_pos"],
        mocap_quat=sample["mocap_quat"],
        impl=self._mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self._mjx_model, data)

    geom = self._geometry(data)
    info = {
        "rng": rng,
        "_steps": jp.array(0, dtype=int),
        "motor_targets": sample["ctrl"][self._act_ids.ravel()],
        "last_act": jp.zeros(self.action_size),
        "closest_fingertip_dist": geom["fingertip_dist"],
        "closest_keypoint_max_dist": geom["keypoint_max_dist"],
        "lifted": jp.array(False),
        "near_goal_steps": jp.array(0, dtype=int),
        "successes": jp.array(0, dtype=int),
        "success_tolerance": jp.array(
            self._config.success_tolerance, dtype=float
        ),
    }
    metrics = {
        f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.keys()
    }
    metrics.update({
        "success": jp.zeros(()),
        "success_strict": jp.zeros(()),
        "successes": jp.zeros(()),
        "success_tolerance": info["success_tolerance"],
        "lifted": jp.zeros(()),
        "keypoint_max_dist": geom["keypoint_max_dist"],
        "fell": jp.zeros(()),
    })
    obs = self._get_obs(data, info, geom)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    info = state.info
    newly_reset = info["_steps"] == 0

    # The autoreset wrapper restores the first state of the environment;
    # draw a fresh robot pose, cube pose and goal for every episode instead.
    info["rng"], rng_episode = jax.random.split(info["rng"])
    sample = self._sample_episode(rng_episode)
    data = state.data.replace(**{
        k: jp.where(newly_reset, v, getattr(state.data, k))
        for k, v in sample.items()
    })
    motor_targets = jp.where(
        newly_reset,
        sample["ctrl"][self._act_ids.ravel()],
        info["motor_targets"],
    )
    lifted_prev = jp.where(newly_reset, False, info["lifted"])
    successes = jp.where(newly_reset, 0, info["successes"])
    near_goal_steps = jp.where(newly_reset, 0, info["near_goal_steps"])

    # Position targets move by at most the action scale per step.
    action = jp.clip(action, -1.0, 1.0)
    act = action.reshape(self._num_arms, NU_ARM_HAND)
    targets = motor_targets.reshape(self._num_arms, NU_ARM_HAND)
    targets = jp.clip(
        targets + act * self._action_scale, self._ctrl_lower, self._ctrl_upper
    )
    motor_targets = targets.ravel()
    ctrl = data.ctrl.at[self._act_ids.ravel()].set(motor_targets)
    data = mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps)

    geom = self._geometry(data)
    closest_ft = jp.where(
        newly_reset, geom["fingertip_dist"], info["closest_fingertip_dist"]
    )
    closest_kp = jp.where(
        newly_reset,
        geom["keypoint_max_dist"],
        info["closest_keypoint_max_dist"],
    )
    cfg = self._config.reward_config

    # Fingertips approaching the cube. With one arm this only matters until
    # the cube is lifted; with two arms the other hand should stay close.
    ft_delta = jp.sum(jp.clip(closest_ft - geom["fingertip_dist"], 0.0, 10.0))
    closest_ft = jp.minimum(closest_ft, geom["fingertip_dist"])

    z_lift = geom["cube_pos"][2] - self._cube_rest_height
    lifted = (z_lift > self._config.lift_threshold) | lifted_prev
    lifting = jp.clip(z_lift, 0.0, 0.5) * (~lifted)
    lift_bonus = (lifted & ~lifted_prev).astype(float)
    if self._num_arms == 1:
      ft_delta = ft_delta * (~lifted)

    kp_delta = jp.clip(closest_kp - geom["keypoint_max_dist"], 0.0, 100.0)
    kp_delta = kp_delta * lifted
    closest_kp = jp.minimum(closest_kp, geom["keypoint_max_dist"])

    tolerance = info["success_tolerance"] * self._config.keypoint_scale
    near_goal = geom["keypoint_max_dist"] <= tolerance
    near_goal_steps = jp.where(near_goal, near_goal_steps + 1, 0)
    success = near_goal_steps >= self._config.success_steps
    reach_bonus = near_goal.astype(float) / self._config.success_steps

    arm_penalty = jp.sum(jp.square(act[:, :NU_ARM]))
    hand_penalty = jp.sum(jp.square(act[:, NU_ARM:]))

    fell = geom["cube_pos"][2] < self._table_top - 0.05
    rewards = {
        "fingertip_delta": ft_delta,
        "lifting": lifting,
        "lift_bonus": lift_bonus,
        "keypoint": kp_delta,
        "reach_goal_bonus": reach_bonus,
        "arm_action_penalty": -arm_penalty,
        "hand_action_penalty": -hand_penalty,
        "fall_penalty": (fell & lifted).astype(float),
    }
    reward = sum(v * cfg[k] for k, v in rewards.items())
    reward = jp.where(jp.isnan(reward), 0.0, reward)

    # A new goal after each success.
    successes = successes + success.astype(int)
    info["rng"], rng_goal = jax.random.split(info["rng"])
    new_goal_pos, new_goal_quat = self._sample_goal(rng_goal)
    goal_pos = jp.where(success, new_goal_pos, geom["goal_pos"])
    goal_quat = jp.where(success, new_goal_quat, geom["goal_quat"])
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._goal_mocap_id].set(goal_pos),
        mocap_quat=data.mocap_quat.at[self._goal_mocap_id].set(goal_quat),
    )
    geom = self._geometry(data)  # Keypoint errors to the current goal.
    closest_kp = jp.where(success, geom["keypoint_max_dist"], closest_kp)
    near_goal_steps = jp.where(success, 0, near_goal_steps)

    nans = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = fell | nans | (successes >= self._config.max_consecutive_successes)

    # Tolerance curriculum, evaluated when the episode ends.
    curriculum = self._config.curriculum
    episode_end = done | (
        info["_steps"] + self._config.action_repeat
        >= self._config.episode_length
    )
    tighten = (
        curriculum.enable
        & episode_end
        & (successes >= curriculum.min_successes)
    )
    success_tolerance = jp.where(
        tighten,
        jp.maximum(
            info["success_tolerance"] * curriculum.increment,
            curriculum.target_tolerance,
        ),
        info["success_tolerance"],
    )
    strict_tolerance = curriculum.target_tolerance * self._config.keypoint_scale

    info.update({
        "motor_targets": motor_targets,
        "last_act": action,
        "closest_fingertip_dist": closest_ft,
        "closest_keypoint_max_dist": closest_kp,
        "lifted": lifted,
        "near_goal_steps": near_goal_steps,
        "successes": successes,
        "success_tolerance": success_tolerance,
    })
    info["_steps"] = info["_steps"] + self._config.action_repeat
    info["_steps"] = jp.where(
        done | (info["_steps"] >= self._config.episode_length),
        0,
        info["_steps"],
    )

    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v * cfg[k]
    state.metrics.update({
        "success": success.astype(float),
        "success_strict": (
            geom["keypoint_max_dist"] <= strict_tolerance
        ).astype(float),
        "successes": successes.astype(float),
        "success_tolerance": success_tolerance,
        "lifted": lifted.astype(float),
        "keypoint_max_dist": geom["keypoint_max_dist"],
        "fell": fell.astype(float),
    })
    obs = self._get_obs(data, info, geom)
    return state.replace(  # pyrefly: ignore[missing-attribute]
        data=data, obs=obs, reward=reward, done=done.astype(float), info=info
    )

  # Helpers.

  def _geometry(self, data: mjx.Data) -> Dict[str, jax.Array]:
    """Cube, goal, keypoint and hand geometry of a state."""
    cube_pos = data.xpos[self._cube_body_id]
    cube_quat = data.xquat[self._cube_body_id]
    goal_pos = data.mocap_pos[self._goal_mocap_id]
    goal_quat = data.mocap_quat[self._goal_mocap_id]
    rotate = jax.vmap(math.rotate, in_axes=(0, None))
    obj_kp = cube_pos + rotate(self._keypoint_offsets, cube_quat)
    goal_kp = goal_pos + rotate(self._keypoint_offsets, goal_quat)
    keypoint_dist = jp.linalg.norm(obj_kp - goal_kp, axis=-1)
    tips = data.site_xpos[self._tip_sites]  # (num_arms, 5, 3)
    palm_pos = data.site_xpos[self._palm_sites]  # (num_arms, 3)
    return {
        "cube_pos": cube_pos,
        "cube_quat": cube_quat,
        "goal_pos": goal_pos,
        "goal_quat": goal_quat,
        "obj_keypoints": obj_kp,
        "goal_keypoints": goal_kp,
        "keypoint_max_dist": jp.max(keypoint_dist),
        "fingertips": tips,
        "fingertip_dist": jp.linalg.norm(tips - cube_pos, axis=-1),
        "palm_pos": palm_pos,
        "palm_mat": data.site_xmat[self._palm_sites].reshape(self._num_arms, 9),
    }

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], geom: Dict[str, jax.Array]
  ) -> mjx_env.Observation:
    q = data.qpos[self._qids]
    q_unscaled = (
        2.0 * (q - self._jnt_lower) / (self._jnt_upper - self._jnt_lower) - 1.0
    )
    dq = data.qvel[self._dqids]
    palm_pos = geom["palm_pos"]
    palm_vel = jp.stack([
        jp.concatenate([
            mjx_env.get_sensor_data(self.mj_model, data, f"arm{i}_palm_linvel"),
            mjx_env.get_sensor_data(self.mj_model, data, f"arm{i}_palm_angvel"),
        ])
        for i in range(self._num_arms)
    ])
    tips_rel_palm = geom["fingertips"] - palm_pos[:, None, :]
    kp_rel_palm = geom["obj_keypoints"][None, :, :] - palm_pos[:, None, :]
    per_arm = jp.concatenate(
        [
            q_unscaled,  # 30
            dq,  # 30
            palm_pos,  # 3
            geom["palm_mat"][:, 3:],  # 6, two rows of the rotation matrix
            palm_vel,  # 6
            tips_rel_palm.reshape(self._num_arms, -1),  # 15
            kp_rel_palm.reshape(self._num_arms, -1),  # 3 * num_keypoints
        ],
        axis=-1,
    ).ravel()
    steps = info["_steps"].astype(float)
    shared = jp.concatenate([
        geom["cube_quat"],  # 4
        mjx_env.get_sensor_data(self.mj_model, data, "cube_linvel"),  # 3
        mjx_env.get_sensor_data(self.mj_model, data, "cube_angvel"),  # 3
        (geom["obj_keypoints"] - geom["goal_keypoints"]).ravel(),
        jp.array([
            info["closest_keypoint_max_dist"],
            info["lifted"].astype(float),
            jp.log(steps / 10.0 + 1.0),
            jp.log(info["successes"].astype(float) + 1.0),
        ]),
        info["last_act"],
    ])
    state = jp.clip(
        jp.concatenate([per_arm, shared]),
        -self._config.obs_clip,
        self._config.obs_clip,
    )
    privileged_state = jp.concatenate([
        state,
        geom["cube_pos"],
        geom["goal_pos"],
        geom["goal_quat"],
        info["closest_fingertip_dist"].ravel(),
    ])
    return {"state": state, "privileged_state": privileged_state}

  @property
  def action_size(self) -> int:
    return self._num_arms * NU_ARM_HAND
