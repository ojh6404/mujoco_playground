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

"""Stack a blue cube on a red cube with the reBot B601-DM from pixels.

Both cubes spawn on the line x = 0.28 m in front of the robot, so the
Cartesian (y, z, gripper) controller of RebotDmPickCubeCartesian can reach
them. Like PandaPickCubeCartesian, the reward is the improvement of a dense
score over the episode, episodes end on success, and a few episodes start
with the blue cube already grasped to help exploration.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.rebot_b601_dm import base
from mujoco_playground._src.manipulation.rebot_b601_dm import cartesian
from mujoco_playground._src.manipulation.rebot_b601_dm import stack


def default_config() -> config_dict.ConfigDict:
  """Returns the default config for cube stacking with Cartesian control."""
  fields = cartesian.cartesian_config_fields()
  fields["tip_z_range"] = [0.02, 0.22]
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.005,
      episode_length=300,
      action_repeat=1,
      cube_half_size=0.02,
      # Each cube spawns on the line through the gripper with |y| in this range,
      # the two cubes on opposite sides of y = 0.
      spawn_y_range=[0.03, 0.12],
      stack_xy_tolerance=0.015,
      stack_z_tolerance=0.01,
      reward_config=config_dict.create(
          scales=config_dict.create(
              gripper_blue=4.0,
              blue_target=8.0,
              stacked=4.0,
              success=4.0,
              red_still=1.0,
              no_floor_collision=0.25,
              # Destabilizes training in cartesian action space.
              robot_target_qpos=0.0,
          ),
          # Blue cube lifted off the table.
          lifted_reward=0.5,
      ),
      # Fraction of episodes that start from the 'picked' keyframe.
      guide_prob=0.05,
      **fields,
      impl="warp",
      naconmax=24 * 2048,
      naccdmax=24 * 2048,
      njmax=128,
  )


class RebotDmStackCubeCartesian(
    cartesian.CartesianVisionMixin, stack.RebotDmStackCube
):
  """Stack the blue cube on the red one with (y, z, gripper) actions.

  Success (blue cube resting on the red one, fingers off it) ends the episode.
  Pixel observations need `vision=True` and `impl="warp"`.
  """

  SCENE_XML = base.XML_PATH / "mjx_stack_cubes_camera.xml"

  def __init__(  # pylint: disable=non-parent-init-called,super-init-not-called
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    base.RebotDmBase.__init__(self, self.SCENE_XML, config, config_overrides)
    self._post_init_stack(keyframe="low_home")
    self._init_cartesian_vision()
    self._guide_q = self._mj_model.keyframe("picked").qpos
    self._guide_ctrl = self._mj_model.keyframe("picked").ctrl
    self._guide_red_pos = jp.array(
        self._guide_q[self._red_qposadr : self._red_qposadr + 3]
    )

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng_y, rng_side = jax.random.split(rng, 3)

    # Spawn the cubes on the gripper's plane, on opposite sides of y = 0.
    half = self._config.cube_half_size
    x_plane = self._start_tip_pos[0]
    y_lo, y_hi = self._config.spawn_y_range
    ys = jax.random.uniform(rng_y, (2,), minval=y_lo, maxval=y_hi)
    side = jp.where(jax.random.bernoulli(rng_side), 1.0, -1.0)
    red_pos = jp.array([x_plane, -side * ys[0], half])
    blue_pos = jp.array([x_plane, side * ys[1], half])

    init_q = jp.array(self._init_q)
    init_q = init_q.at[self._red_qposadr : self._red_qposadr + 3].set(red_pos)
    init_q = init_q.at[self._obj_qposadr : self._obj_qposadr + 3].set(blue_pos)
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
    if not self._vision:
      # The marker only helps state-based viewing; it is hidden from pixels.
      target_pos = red_pos + jp.array([0.0, 0.0, 2 * half])
      data = data.replace(
          mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos)
      )

    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
        "lifted": jp.array(0.0, dtype=float),
    }
    info = {
        "rng": rng,
        "red_init_pos": red_pos,
        "reached_blue": jp.array(0.0, dtype=float),
        "prev_reward": jp.array(0.0, dtype=float),
        "current_pos": self._start_tip_pos,
        "newly_reset": jp.array(False, dtype=bool),
        "_steps": jp.array(0, dtype=int),
    }
    obs = jp.concat([self._get_obs(data, info), jp.zeros(1), jp.zeros(3)])
    if self._vision:
      rng_brightness, rng = jax.random.split(rng)
      info.update({"brightness": self._sample_brightness(rng_brightness)})
      data = mjx.forward(self._mjx_model, data)

    reward, done = jp.zeros(2)
    state = mjx_env.State(data, obs, reward, done, metrics, info)
    if self._vision and not self._defer_rendering:
      state = self.render_state(state)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state.info["newly_reset"] = state.info["_steps"] == 0
    newly_reset = state.info["newly_reset"]
    state.info["prev_reward"] = jp.where(
        newly_reset, 0.0, state.info["prev_reward"]
    )
    state.info["current_pos"] = jp.where(
        newly_reset, self._start_tip_pos, state.info["current_pos"]
    )
    state.info["reached_blue"] = jp.where(
        newly_reset, 0.0, state.info["reached_blue"]
    )

    # Occasionally start with the blue cube already in the gripper.
    state.info["rng"], key_swap = jax.random.split(state.info["rng"])
    to_sample = newly_reset * jax.random.bernoulli(
        key_swap, self._config.guide_prob
    )
    swapped_data = state.data.replace(qpos=self._guide_q, ctrl=self._guide_ctrl)
    data = jax.tree_util.tree_map_with_path(
        lambda path, x, y: ((1 - to_sample) * x + to_sample * y).astype(x.dtype)
        if len(path) == 1
        else x,
        state.data,
        swapped_data,
    )
    state.info["red_init_pos"] = jp.where(
        to_sample, self._guide_red_pos, state.info["red_init_pos"]
    )

    # Cartesian control
    ctrl, new_tip_position, no_soln = self._move_tip(
        state.info["current_pos"], data.ctrl, action
    )
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)
    state.info.update({"current_pos": new_tip_position})

    data = mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps)

    raw_rewards = self._get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    blue_pos = data.xpos[self._obj_body]
    lifted = (blue_pos[2] > 0.05) * self._config.reward_config.lifted_reward
    total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4) + lifted

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

    cube_pos = data.xpos[jp.array([self._obj_body, self._red_body])]
    out_of_bounds = jp.any(jp.abs(cube_pos) > 1.0)
    out_of_bounds |= jp.any(cube_pos[:, 2] < 0.0)
    success = raw_rewards["success"] > 0.5
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

    state.metrics.update(
        **raw_rewards,
        out_of_bounds=out_of_bounds.astype(float),
        lifted=(lifted > 0).astype(float),
    )
    obs = jp.concat(
        [self._get_obs(data, state.info), no_soln.reshape(1), action]
    )
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
