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

"""In-hand cube reorientation with the Shadow Hand on the A0B arm.

A port of the LEAP hand `CubeReorient` task. The arm holds a fixed palm-up
pose; the policy drives the 20 hand actuators.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.manipulation.a0b_shadow_hand import a0bsrh_constants as consts
from mujoco_playground._src.manipulation.a0b_shadow_hand import base as a0bsrh_base
from mujoco_playground._src.manipulation.leap_hand import base as leap_hand_base


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.01,
      action_scale=0.5,
      action_repeat=1,
      ema_alpha=1.0,
      episode_length=1000,
      success_threshold=0.1,
      history_len=1,
      # Reset randomization: hand joint angles (rad), cube position (m) and
      # whether the cube starts face down with a random yaw (True) or in a
      # uniformly random orientation (False, as for the LEAP hand).
      reset_noise=config_dict.create(
          hand_pose=0.03, cube_pos=0.005, cube_yaw_only=True
      ),
      # The cube spawns this far above its resting height so a randomly rotated
      # cube does not start inside the palm.
      cube_spawn_height=0.01,
      obs_noise=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
              cube_pos=0.02,
              cube_ori=0.1,
          ),
          random_ori_injection_prob=0.0,
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              orientation=5.0,
              position=0.5,
              termination=-100.0,
              hand_pose=-0.5,
              action_rate=-0.001,
              joint_vel=0.0,
              energy=-1e-3,
          ),
          success_reward=100.0,
      ),
      pert_config=config_dict.create(
          enable=False,
          linear_velocity_pert=[0.0, 3.0],
          angular_velocity_pert=[0.0, 0.5],
          pert_duration_steps=[1, 100],
          pert_wait_steps=[60, 150],
      ),
      impl="warp",
      naconmax=30 * 8192,
      njmax=200,
  )


class CubeReorient(a0bsrh_base.A0bShadowHandEnv):
  """Reorient a cube in the Shadow Hand to match a goal orientation."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.CUBE_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    home_key = self._mj_model.keyframe("home")
    self._init_q = jp.array(home_key.qpos, dtype=float)
    self._init_ctrl = jp.array(home_key.ctrl, dtype=float)
    self._init_mpos = jp.array(home_key.mpos, dtype=float)
    self._init_mquat = jp.array(home_key.mquat, dtype=float)

    self._hand_act_ids = np.array(
        [self._mj_model.actuator(n).id for n in consts.HAND_ACTUATOR_NAMES]
    )
    self._lowers = self._mj_model.actuator_ctrlrange[self._hand_act_ids, 0]
    self._uppers = self._mj_model.actuator_ctrlrange[self._hand_act_ids, 1]
    self._hand_qids = mjx_env.get_qpos_ids(
        self.mj_model, consts.HAND_JOINT_NAMES
    )
    self._hand_dqids = mjx_env.get_qvel_ids(
        self.mj_model, consts.HAND_JOINT_NAMES
    )
    # Joints driven by the hand actuators (the coupled distal joints are not).
    self._hand_act_qids = mjx_env.get_qpos_ids(
        self.mj_model, consts.HAND_ACTUATOR_NAMES
    )
    self._hand_act_dqids = mjx_env.get_qvel_ids(
        self.mj_model, consts.HAND_ACTUATOR_NAMES
    )
    self._hand_joint_range = jp.array(
        self._mj_model.jnt_range[
            [self._mj_model.joint(n).id for n in consts.HAND_JOINT_NAMES]
        ]
    )
    self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._cube_geom_id = self._mj_model.geom("cube").id
    self._cube_body_id = self._mj_model.body("cube").id
    self._cube_mass = self._mj_model.body_subtreemass[self._cube_body_id]
    self._default_pose = self._init_q[self._hand_qids]
    self._cube_start_pos = self._init_q[self._cube_qids[:3]]
    # The arm does not move, so the cube has fallen once it is this far below
    # its resting height.
    self._fall_height = float(self._cube_start_pos[2]) - 0.1

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Randomize the goal orientation.
    rng, goal_rng = jax.random.split(rng)
    goal_quat = leap_hand_base.uniform_quat(goal_rng)

    # Randomize the hand pose.
    rng, pos_rng = jax.random.split(rng)
    q_hand = jp.clip(
        self._default_pose
        + self._config.reset_noise.hand_pose
        * jax.random.normal(pos_rng, (consts.NQ_HAND,)),
        self._hand_joint_range[:, 0],
        self._hand_joint_range[:, 1],
    )

    # Randomize the cube pose.
    rng, p_rng, quat_rng = jax.random.split(rng, 3)
    start_pos = (
        self._cube_start_pos
        + jp.array([0.0, 0.0, self._config.cube_spawn_height])
        + jax.random.uniform(
            p_rng,
            (3,),
            minval=-self._config.reset_noise.cube_pos,
            maxval=self._config.reset_noise.cube_pos,
        )
    )
    if self._config.reset_noise.cube_yaw_only:
      yaw = jax.random.uniform(quat_rng, minval=-jp.pi, maxval=jp.pi)
      start_quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
    else:
      start_quat = leap_hand_base.uniform_quat(quat_rng)
    q_cube = jp.array([*start_pos, *start_quat])

    qpos = self._init_q.at[self._hand_qids].set(q_hand)
    qpos = qpos.at[self._cube_qids].set(q_cube)
    ctrl = self._init_ctrl.at[self._hand_act_ids].set(qpos[self._hand_act_qids])
    data = mjx_env.make_data(
        self._mj_model,
        qpos=qpos,
        ctrl=ctrl,
        qvel=jp.zeros(self._mjx_model.nv),
        mocap_pos=self._init_mpos,
        mocap_quat=goal_quat,
        impl=self._mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )

    rng, pert1, pert2, pert3 = jax.random.split(rng, 4)
    pert_wait_steps = jax.random.randint(
        pert1,
        (1,),
        minval=self._config.pert_config.pert_wait_steps[0],
        maxval=self._config.pert_config.pert_wait_steps[1],
    )
    pert_duration_steps = jax.random.randint(
        pert2,
        (1,),
        minval=self._config.pert_config.pert_duration_steps[0],
        maxval=self._config.pert_config.pert_duration_steps[1],
    )
    pert_lin = jax.random.uniform(
        pert3,
        minval=self._config.pert_config.linear_velocity_pert[0],
        maxval=self._config.pert_config.linear_velocity_pert[1],
    )
    pert_ang = jax.random.uniform(
        pert3,
        minval=self._config.pert_config.angular_velocity_pert[0],
        maxval=self._config.pert_config.angular_velocity_pert[1],
    )
    pert_velocity = jp.array([pert_lin] * 3 + [pert_ang] * 3)

    info = {
        "rng": rng,
        "step": 0,
        "steps_since_last_success": 0,
        "success_count": 0,
        "last_act": jp.zeros(consts.NU_HAND),
        "last_last_act": jp.zeros(consts.NU_HAND),
        "motor_targets": ctrl[self._hand_act_ids],
        "qpos_error_history": jp.zeros(
            self._config.history_len * consts.NU_HAND
        ),
        "cube_pos_error_history": jp.zeros(self._config.history_len * 3),
        "cube_ori_error_history": jp.zeros(self._config.history_len * 6),
        "goal_quat_dquat": jp.zeros(3),
        "pert_wait_steps": pert_wait_steps,
        "pert_duration_steps": pert_duration_steps,
        "pert_vel": pert_velocity,
        "pert_dir": jp.zeros(6, dtype=float),
        "last_pert_step": jp.array([-jp.inf], dtype=float),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["reward/success"] = jp.zeros((), dtype=float)
    metrics["steps_since_last_success"] = 0
    metrics["success_count"] = 0

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state, state.info["rng"])

    delta = action * self._config.action_scale
    motor_targets = state.info["motor_targets"] + delta
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    motor_targets = (
        self._config.ema_alpha * motor_targets
        + (1 - self._config.ema_alpha) * state.info["motor_targets"]
    )
    # The arm actuators keep their home targets.
    ctrl = self._init_ctrl.at[self._hand_act_ids].set(motor_targets)

    data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
    state.info["motor_targets"] = motor_targets

    ori_error = self._cube_orientation_error(data)
    success = ori_error < self._config.success_threshold
    state.info["steps_since_last_success"] = jp.where(
        success, 0, state.info["steps_since_last_success"] + 1
    )
    state.info["success_count"] = jp.where(
        success, state.info["success_count"] + 1, state.info["success_count"]
    )
    state.metrics["steps_since_last_success"] = state.info[
        "steps_since_last_success"
    ]
    state.metrics["success_count"] = state.info["success_count"]

    done = self._get_termination(data, state.info)
    obs = self._get_obs(data, state.info)

    rewards = self._get_reward(data, action, state.info, state.metrics, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt  # pylint: disable=redefined-outer-name

    # Move the goal on success, so one episode covers several goals.
    state.info["rng"], goal_rng = jax.random.split(state.info["rng"])
    state.info["goal_quat_dquat"] = jp.where(
        success,
        3 + jax.random.uniform(goal_rng, (3,), minval=-2, maxval=2),
        state.info["goal_quat_dquat"] * 0.8,
    )
    goal_quat = math.quat_integrate(
        state.data.mocap_quat[0],
        state.info["goal_quat_dquat"],
        2 * jp.array(self.dt),
    )
    data = data.replace(mocap_quat=jp.array([goal_quat]))
    state.metrics["reward/success"] = success.astype(float)
    reward += success * self._config.reward_config.success_reward

    state.info["step"] += 1
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    return state.replace(  # pyrefly: ignore[missing-attribute]
        data=data, obs=obs, reward=reward, done=done
    )

  def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    fall_termination = self.get_cube_position(data)[2] < self._fall_height
    nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
    return fall_termination | nans

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> mjx_env.Observation:
    joint_angles = data.qpos[self._hand_qids]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.obs_noise.level
        * self._config.obs_noise.scales.joint_pos
    )

    # Tracking error of the actuated joints.
    noisy_act_angles = (
        data.qpos[self._hand_act_qids]
        + (2 * jax.random.uniform(noise_rng, shape=(consts.NU_HAND,)) - 1)
        * self._config.obs_noise.level
        * self._config.obs_noise.scales.joint_pos
    )
    qpos_error_history = (
        jp.roll(info["qpos_error_history"], consts.NU_HAND)
        .at[: consts.NU_HAND]
        .set(noisy_act_angles - info["motor_targets"])
    )
    info["qpos_error_history"] = qpos_error_history

    def _get_cube_pose(data: mjx.Data) -> jax.Array:
      """Returns (potentially) noisy cube pose (xyz,wxyz)."""
      cube_pos = self.get_cube_position(data)
      cube_quat = self.get_cube_orientation(data)
      info["rng"], pos_rng, ori_rng = jax.random.split(info["rng"], 3)
      noisy_cube_quat = math.normalize(
          cube_quat
          + jax.random.normal(ori_rng, shape=(4,))
          * self._config.obs_noise.level
          * self._config.obs_noise.scales.cube_ori
      )
      noisy_cube_pos = (
          cube_pos
          + (2 * jax.random.uniform(pos_rng, shape=cube_pos.shape) - 1)
          * self._config.obs_noise.level
          * self._config.obs_noise.scales.cube_pos
      )
      return jp.concatenate([noisy_cube_pos, noisy_cube_quat])

    noisy_pose = _get_cube_pose(data)
    info["rng"], key1, key2, key3 = jax.random.split(info["rng"], 4)
    rand_quat = leap_hand_base.uniform_quat(key1)
    rand_pos = jax.random.uniform(key2, (3,), minval=-0.5, maxval=0.5)
    rand_pose = jp.concatenate([rand_pos, rand_quat])
    m = self._config.obs_noise.level * jax.random.bernoulli(
        key3, self._config.obs_noise.random_ori_injection_prob
    )
    noisy_pose = noisy_pose * (1 - m) + rand_pose * m

    palm_pos = self.get_palm_position(data)
    cube_pos_error = palm_pos - noisy_pose[:3]
    cube_pos_error_history = (
        jp.roll(info["cube_pos_error_history"], 3).at[:3].set(cube_pos_error)
    )
    info["cube_pos_error_history"] = cube_pos_error_history

    goal_quat = self.get_cube_goal_orientation(data)
    quat_diff = math.quat_mul(noisy_pose[3:], math.quat_inv(goal_quat))
    xmat_diff = math.quat_to_mat(quat_diff).ravel()[3:]
    cube_ori_error_history = (
        jp.roll(info["cube_ori_error_history"], 6).at[:6].set(xmat_diff)
    )
    info["cube_ori_error_history"] = cube_ori_error_history

    cube_pos_error_uncorrupted = palm_pos - self.get_cube_position(data)
    cube_quat_uncorrupted = self.get_cube_orientation(data)
    quat_diff_uncorrupted = math.quat_mul(
        cube_quat_uncorrupted, math.quat_inv(goal_quat)
    )
    xmat_diff_uncorrupted = math.quat_to_mat(quat_diff_uncorrupted).ravel()[3:]

    state = jp.concatenate([
        noisy_joint_angles,  # 24
        qpos_error_history,  # 20 * history_len
        cube_pos_error_history,  # 3 * history_len
        cube_ori_error_history,  # 6 * history_len
        info["last_act"],  # 20
    ])

    privileged_state = jp.concatenate([
        state,
        data.qpos[self._hand_qids],
        data.qvel[self._hand_dqids],
        self.get_fingertip_positions(data),
        cube_pos_error_uncorrupted,
        xmat_diff_uncorrupted,
        self.get_cube_linvel(data),
        self.get_cube_angvel(data),
        info["pert_dir"],
        data.xfrc_applied[self._cube_body_id],
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    del done, metrics  # Unused.

    cube_pos = self.get_cube_position(data)
    palm_pos = self.get_palm_position(data)
    cube_pose_mse = jp.linalg.norm(palm_pos - cube_pos)
    cube_pos_reward = reward.tolerance(
        cube_pose_mse, (0, 0.02), margin=0.05, sigmoid="linear"
    )

    terminated = self._get_termination(data, info)

    hand_pose_reward = jp.sum(
        jp.square(data.qpos[self._hand_qids] - self._default_pose)
    )

    return {
        "orientation": self._reward_cube_orientation(data),
        "position": cube_pos_reward,
        "termination": terminated,
        "hand_pose": hand_pose_reward,
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "joint_vel": self._cost_joint_vel(data),
        "energy": self._cost_energy(
            data.qvel[self._hand_act_dqids],
            data.actuator_force[self._hand_act_ids],
        ),
    }

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cube_orientation_error(self, data: mjx.Data):
    cube_ori = self.get_cube_orientation(data)
    cube_goal_ori = self.get_cube_goal_orientation(data)
    quat_diff = math.quat_mul(cube_ori, math.quat_inv(cube_goal_ori))
    quat_diff = math.normalize(quat_diff)
    return 2.0 * jp.asin(jp.clip(math.norm(quat_diff[1:]), max=1.0))

  def _reward_cube_orientation(self, data: mjx.Data) -> jax.Array:
    ori_error = self._cube_orientation_error(data)
    return reward.tolerance(ori_error, (0, 0.2), margin=jp.pi, sigmoid="linear")

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    c1 = jp.sum(jp.square(act - last_act))
    c2 = jp.sum(jp.square(act - 2 * last_act + last_last_act))
    return c1 + c2

  def _cost_joint_vel(self, data: mjx.Data) -> jax.Array:
    max_velocity = 5.0
    vel_tolerance = 1.0
    hand_qvel = data.qvel[self._hand_dqids]
    return jp.sum((hand_qvel / (max_velocity - vel_tolerance)) ** 2)

  def _maybe_apply_perturbation(
      self, state: mjx_env.State, rng: jax.Array
  ) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      directory = jax.random.normal(rng, (6,))
      return directory / jp.linalg.norm(directory)

    def get_xfrc(
        state: mjx_env.State, pert_dir: jax.Array, i: jax.Array
    ) -> jax.Array:
      u_t = 0.5 * jp.sin(jp.pi * i / state.info["pert_duration_steps"])
      force = (
          u_t
          * self._cube_mass
          * state.info["pert_vel"]
          / (state.info["pert_duration_steps"] * self.dt)
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._cube_body_id].set(force * pert_dir)
      return xfrc_applied

    step, last_pert_step = state.info["step"], state.info["last_pert_step"]
    start_pert = jp.mod(step, state.info["pert_wait_steps"]) == 0
    start_pert &= step != 0  # No perturbation at the beginning of the episode.
    last_pert_step = jp.where(start_pert, step, last_pert_step)
    duration = jp.clip(step - last_pert_step, 0, 100_000)
    in_pert_interval = duration < state.info["pert_duration_steps"]

    pert_dir = jp.where(start_pert, gen_dir(rng), state.info["pert_dir"])
    xfrc = get_xfrc(state, pert_dir, duration) * in_pert_interval

    state.info["pert_dir"] = pert_dir
    state.info["last_pert_step"] = last_pert_step
    data = state.data.replace(xfrc_applied=xfrc)
    return state.replace(data=data)  # pyrefly: ignore[missing-attribute]

  @property
  def action_size(self) -> int:
    return consts.NU_HAND


def domain_randomize(model: mjx.Model, rng: jax.Array):
  mj_model = CubeReorient().mj_model
  cube_body_id = mj_model.body("cube").id
  hand_qids = mjx_env.get_qpos_ids(mj_model, consts.HAND_JOINT_NAMES)
  hand_act_ids = np.array(
      [mj_model.actuator(n).id for n in consts.HAND_ACTUATOR_NAMES]
  )
  hand_body_ids = np.array([
      b
      for b in range(mj_model.nbody)
      if mj_model.body(b).name.startswith("robot0:")
      and mj_model.body(b).name not in ("robot0:body",)
      and not mj_model.body(b).name.startswith("robot0:arm_link")
  ])
  fingertip_geoms = [
      "robot0:C_thdistal",
      "robot0:C_ffdistal",
      "robot0:C_mfdistal",
      "robot0:C_rfdistal",
      "robot0:C_lfdistal",
  ]
  fingertip_geom_ids = [mj_model.geom(g).id for g in fingertip_geoms]
  nq_hand = len(hand_qids)

  @jax.vmap
  def rand(rng):
    rng, key = jax.random.split(rng)
    fingertip_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=1.0)
    geom_friction = model.geom_friction.at[fingertip_geom_ids, 0].set(
        fingertip_friction
    )

    rng, key1, key2 = jax.random.split(rng, 3)
    dmass = jax.random.uniform(key1, minval=0.8, maxval=1.2)
    body_inertia = model.body_inertia.at[cube_body_id].set(
        model.body_inertia[cube_body_id] * dmass
    )
    dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
    body_ipos = model.body_ipos.at[cube_body_id].set(
        model.body_ipos[cube_body_id] + dpos
    )

    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0.at[hand_qids].set(
        model.qpos0[hand_qids]
        + jax.random.uniform(key, shape=(nq_hand,), minval=-0.05, maxval=0.05)
    )

    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
        key, shape=(nq_hand,), minval=0.5, maxval=2.0
    )
    dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

    rng, key = jax.random.split(rng)
    armature = model.dof_armature[hand_qids] * jax.random.uniform(
        key, shape=(nq_hand,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[hand_qids].set(armature)

    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[hand_body_ids].set(
        model.body_mass[hand_body_ids] * dmass
    )

    rng, key = jax.random.split(rng)
    kp = model.actuator_gainprm[hand_act_ids, 0] * jax.random.uniform(
        key, (len(hand_act_ids),), minval=0.8, maxval=1.2
    )
    actuator_gainprm = model.actuator_gainprm.at[hand_act_ids, 0].set(kp)
    actuator_biasprm = model.actuator_biasprm.at[hand_act_ids, 1].set(-kp)

    rng, key = jax.random.split(rng)
    kd = model.dof_damping[hand_qids] * jax.random.uniform(
        key, (nq_hand,), minval=0.8, maxval=1.2
    )
    dof_damping = model.dof_damping.at[hand_qids].set(kd)

    return (
        geom_friction,
        body_mass,
        body_inertia,
        body_ipos,
        qpos0,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        actuator_gainprm,
        actuator_biasprm,
    )

  (
      geom_friction,
      body_mass,
      body_inertia,
      body_ipos,
      qpos0,
      dof_frictionloss,
      dof_armature,
      dof_damping,
      actuator_gainprm,
      actuator_biasprm,
  ) = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_mass": 0,
      "body_inertia": 0,
      "body_ipos": 0,
      "qpos0": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "dof_damping": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
  })

  model = model.tree_replace({  # pyrefly: ignore[bad-assignment]
      "geom_friction": geom_friction,
      "body_mass": body_mass,
      "body_inertia": body_inertia,
      "body_ipos": body_ipos,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "dof_damping": dof_damping,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
  })

  return model, in_axes
