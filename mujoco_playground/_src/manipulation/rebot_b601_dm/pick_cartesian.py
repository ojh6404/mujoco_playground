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

"""Pick up a cube to a fixed height with the reBot B601-DM from pixels.

The reBot counterpart of PandaPickCubeCartesian: a Cartesian controller moves
the gripper in the vertical plane through the cube's spawn line.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.rebot_b601_dm import base
from mujoco_playground._src.manipulation.rebot_b601_dm import cartesian
from mujoco_playground._src.manipulation.rebot_b601_dm import pick


def default_config():
  config = config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.005,
      episode_length=200,
      action_repeat=1,
      reward_config=config_dict.create(
          reward_scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=4.0,
              # Box goes to the target mocap.
              box_target=8.0,
              # Do not collide the gripper with the floor.
              no_floor_collision=0.25,
              # Do not collide cube with gripper
              no_box_collision=0.05,
              # Destabilizes training in cartesian action space.
              robot_target_qpos=0.0,
          ),
          action_rate=-0.0005,
          no_soln_reward=-0.01,
          lifted_reward=0.5,
          success_reward=2.0,
      ),
      **cartesian.cartesian_config_fields(),
      box_init_range=0.05,
      # Height of the fixed box target.
      target_height=0.15,
      success_threshold=0.05,
      action_history_length=1,
      impl="warp",
      naconmax=24 * 2048,
      naccdmax=24 * 2048,
      njmax=128,
  )
  return config


class RebotDmPickCubeCartesian(
    cartesian.CartesianVisionMixin, pick.RebotDmPickCube
):
  """Lift a cube to a fixed height with Cartesian (y, z, gripper) actions.

  The gripper keeps pointing down and moves in the vertical plane through the
  cube's spawn line. Pixel observations need `vision=True` and `impl="warp"`.
  """

  SCENE_XML = base.XML_PATH / "mjx_single_cube_camera.xml"

  def __init__(  # pylint: disable=non-parent-init-called,super-init-not-called
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    base.RebotDmBase.__init__(self, self.SCENE_XML, config, config_overrides)
    self._sample_orientation = False
    self._post_init(obj_name="box", keyframe="low_home")

    # Contact sensor IDs.
    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_box"]
    ]
    self._box_hand_found_sensor = self._mj_model.sensor("box_hand_found").id

    self._guide_q = self._mj_model.keyframe("picked").qpos
    self._guide_ctrl = self._mj_model.keyframe("picked").ctrl
    self._init_cartesian_vision()

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Resets the environment to an initial state."""
    x_plane = self._start_tip_pos[0]

    # intialize box position
    rng, rng_box = jax.random.split(rng)
    r_range = self._config.box_init_range
    box_pos = jp.array([
        x_plane,
        jax.random.uniform(rng_box, (), minval=-r_range, maxval=r_range),
        self._init_obj_pos[2],
    ])

    # Fixed target position to simplify pixels-only training.
    target_pos = jp.array([x_plane, 0.0, self._config.target_height])

    # initialize pipeline state
    init_q = (
        jp.array(self._init_q)
        .at[self._obj_qposadr : self._obj_qposadr + 3]
        .set(box_pos)
    )
    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax,
        njmax=self._config.njmax,
    )

    target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    data = data.replace(
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat)
    )
    if not self._vision:
      # mocap target should not appear in the pixels observation.
      data = data.replace(
          mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos)
      )

    # initialize env state and info
    metrics = {
        "out_of_bounds": jp.array(0.0),
        **{
            f"reward/{k}": 0.0
            for k in self._config.reward_config.reward_scales.keys()
        },
        "reward/success": jp.array(0.0),
        "reward/lifted": jp.array(0.0),
    }

    info = {
        "rng": rng,
        "target_pos": target_pos,
        "reached_box": jp.array(0.0, dtype=float),
        "prev_reward": jp.array(0.0, dtype=float),
        "current_pos": self._start_tip_pos,
        "newly_reset": jp.array(False, dtype=bool),
        "prev_action": jp.zeros(3),
        "_steps": jp.array(0, dtype=int),
        "action_history": jp.zeros((
            self._config.action_history_length,
        )),  # Gripper only
    }

    reward, done = jp.zeros(2)

    obs = self._get_obs(data, info)
    obs = jp.concat([obs, jp.zeros(1), jp.zeros(3)], axis=0)
    if self._vision:
      rng_brightness, rng = jax.random.split(rng)
      info.update({"brightness": self._sample_brightness(rng_brightness)})
      data = mjx.forward(self._mjx_model, data)

    state = mjx_env.State(data, obs, reward, done, metrics, info)
    if self._vision and not self._defer_rendering:
      state = self.render_state(state)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Runs one timestep of the environment's dynamics."""
    action_history = (
        jp.roll(state.info["action_history"], 1).at[0].set(action[2])
    )
    state.info["action_history"] = action_history
    # Add action delay
    state.info["rng"], key = jax.random.split(state.info["rng"])
    action_idx = jax.random.randint(
        key, (), minval=0, maxval=self._config.action_history_length
    )
    action = action.at[2].set(state.info["action_history"][action_idx])

    state.info["newly_reset"] = state.info["_steps"] == 0

    newly_reset = state.info["newly_reset"]
    state.info["prev_reward"] = jp.where(
        newly_reset, 0.0, state.info["prev_reward"]
    )
    state.info["current_pos"] = jp.where(
        newly_reset, self._start_tip_pos, state.info["current_pos"]
    )
    state.info["reached_box"] = jp.where(
        newly_reset, 0.0, state.info["reached_box"]
    )
    state.info["prev_action"] = jp.where(
        newly_reset, jp.zeros(3), state.info["prev_action"]
    )

    # Ocassionally aid exploration.
    state.info["rng"], key_swap = jax.random.split(state.info["rng"])
    to_sample = newly_reset * jax.random.bernoulli(key_swap, 0.05)
    swapped_data = state.data.replace(
        qpos=self._guide_q, ctrl=self._guide_ctrl
    )  # help hit the terminal sparse reward.
    data = jax.tree_util.tree_map_with_path(
        lambda path, x, y: ((1 - to_sample) * x + to_sample * y).astype(x.dtype)
        if len(path) == 1
        else x,
        state.data,
        swapped_data,
    )

    # Cartesian control
    ctrl, new_tip_position, no_soln = self._move_tip(
        state.info["current_pos"], data.ctrl, action
    )
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)
    state.info.update({"current_pos": new_tip_position})

    # Simulator step
    data = mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps)

    # Dense rewards
    raw_rewards = self._get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.reward_scales[k]
        for k, v in raw_rewards.items()
    }

    # Penalize collision with box.
    hand_box = (
        data.sensordata[self._mj_model.sensor_adr[self._box_hand_found_sensor]]
        > 0
    )
    raw_rewards["no_box_collision"] = jp.where(hand_box, 0.0, 1.0)

    total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

    if not self._vision:
      # Vision policy cannot access the required state-based observations.
      da = jp.linalg.norm(action - state.info["prev_action"])
      state.info["prev_action"] = action
      total_reward += self._config.reward_config.action_rate * da
      total_reward += no_soln * self._config.reward_config.no_soln_reward

    # Sparse rewards
    box_pos = data.xpos[self._obj_body]
    lifted = (box_pos[2] > 0.05) * self._config.reward_config.lifted_reward
    total_reward += lifted
    success = self._get_success(data, state.info)
    total_reward += success * self._config.reward_config.success_reward

    # Reward progress
    reward = jp.maximum(
        total_reward - state.info["prev_reward"], jp.zeros_like(total_reward)
    )
    state.info["prev_reward"] = jp.maximum(
        total_reward, state.info["prev_reward"]
    )
    reward = jp.where(newly_reset, 0.0, reward)  # Prevent first-step artifact
    # A NaN physics state ends the episode below; keep its reward finite so
    # evaluation averages stay meaningful.
    reward = jp.where(jp.isnan(reward), 0.0, reward)

    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    state.metrics.update(out_of_bounds=out_of_bounds.astype(float))
    state.metrics.update({f"reward/{k}": v for k, v in raw_rewards.items()})
    state.metrics.update({
        "reward/lifted": lifted.astype(float),
        "reward/success": success.astype(float),
    })

    done = (
        out_of_bounds
        | jp.isnan(data.qpos).any()
        | jp.isnan(data.qvel).any()
        | success
    )

    # Ensure exact sync between newly_reset and the autoresetwrapper.
    state.info["_steps"] += self._config.action_repeat
    state.info["_steps"] = jp.where(
        done | (state.info["_steps"] >= self._config.episode_length),
        0,
        state.info["_steps"],
    )

    obs = self._get_obs(data, state.info)
    obs = jp.concat([obs, no_soln.reshape(1), action], axis=0)
    state = state.replace(  # pyrefly: ignore[missing-attribute]
        data=data,
        obs=obs,
        reward=reward,
        done=done.astype(float),
        info=state.info,
    )
    if self._vision and not self._defer_rendering:
      state = self.render_state(state)
    return state

  def _get_success(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    box_pos = data.xpos[self._obj_body]
    target_pos = info["target_pos"]
    if self._vision:
      # Randomized camera positions cannot see location along y line.
      box_pos, target_pos = box_pos[2], target_pos[2]
    return jp.linalg.norm(box_pos - target_pos) < self._config.success_threshold
