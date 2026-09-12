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

"""Stack a blue cube on a red cube with the reBot Arm B601-DM."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.rebot_b601_dm import base
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member


def default_config() -> config_dict.ConfigDict:
  """Returns the default config for the reBot B601-DM cube stacking task."""
  config = config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=250,
      action_repeat=1,
      action_scale=0.04,
      cube_half_size=0.02,
      # Each cube spawns with x in spawn_x_range and |y| in spawn_y_range, the
      # two cubes on opposite sides of y = 0.
      spawn_x_range=[0.22, 0.34],
      spawn_y_range=[0.03, 0.12],
      # The blue cube counts as stacked when it touches the red cube and is
      # within these distances of the spot on top of it.
      stack_xy_tolerance=0.015,
      stack_z_tolerance=0.01,
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Gripper goes to the blue cube.
              gripper_blue=4.0,
              # Blue cube goes to the spot on top of the red cube.
              blue_target=8.0,
              # Blue cube rests on the red cube.
              stacked=4.0,
              # Blue cube rests on the red cube and the fingers let go of it.
              success=4.0,
              # Red cube stays where it spawned.
              red_still=1.0,
              # Do not collide the gripper with the floor.
              no_floor_collision=0.25,
              # Arm stays close to the home pose.
              robot_target_qpos=0.3,
          )
      ),
      impl="warp",
      naconmax=24 * 2048,
      naccdmax=24 * 2048,
      njmax=128,
  )
  return config


class RebotDmStackCube(base.RebotDmBase):
  """Pick up the blue cube and place it on top of the red cube."""

  SCENE_XML = base.XML_PATH / "mjx_stack_cubes.xml"

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(self.SCENE_XML, config, config_overrides)
    self._post_init_stack(keyframe="home")

  def _post_init_stack(self, keyframe: str) -> None:
    """Looks up the cube bodies and the contact sensors."""
    self._post_init(obj_name="blue_cube", keyframe=keyframe)
    self._red_body = self._mj_model.body("red_cube").id
    self._red_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.body("red_cube").jntadr[0]
    ]
    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_box"]
    ]
    self._blue_red_found_sensor = self._mj_model.sensor("blue_red_found").id
    self._finger_blue_found_sensor = [
        self._mj_model.sensor(f"{geom}_blue_found").id
        for geom in ["left_finger_pad", "right_finger_pad"]
    ]

  def _sensor_found(self, data: mjx.Data, sensor_id: int) -> jax.Array:
    return data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0

  def reset(self, rng: jax.Array) -> State:
    rng, rng_x, rng_y, rng_side = jax.random.split(rng, 4)

    # Spawn the cubes on opposite sides of y = 0, sides chosen at random.
    half = self._config.cube_half_size
    x_lo, x_hi = self._config.spawn_x_range
    y_lo, y_hi = self._config.spawn_y_range
    xs = jax.random.uniform(rng_x, (2,), minval=x_lo, maxval=x_hi)
    ys = jax.random.uniform(rng_y, (2,), minval=y_lo, maxval=y_hi)
    side = jp.where(jax.random.bernoulli(rng_side), 1.0, -1.0)
    red_pos = jp.array([xs[0], -side * ys[0], half])
    blue_pos = jp.array([xs[1], side * ys[1], half])

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

    # The mocap body only marks where the blue cube should end up.
    target_pos = red_pos + jp.array([0.0, 0.0, 2 * half])
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos)
    )

    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }
    info = {"rng": rng, "red_init_pos": red_pos, "reached_blue": 0.0}
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return State(data, obs, reward, done, metrics, info)

  def step(self, state: State, action: jax.Array) -> State:
    delta = action * self._action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    raw_rewards = self._get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

    cube_pos = data.xpos[jp.array([self._obj_body, self._red_body])]
    out_of_bounds = jp.any(jp.abs(cube_pos) > 1.0)
    out_of_bounds |= jp.any(cube_pos[:, 2] < 0.0)
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)

    state.metrics.update(
        **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
    )

    obs = self._get_obs(data, state.info)
    return State(data, obs, reward, done, state.metrics, state.info)

  def _target_pos(self, data: mjx.Data) -> jax.Array:
    """Center of the blue cube once it sits on top of the red cube."""
    half = self._config.cube_half_size
    return data.xpos[self._red_body] + jp.array([0.0, 0.0, 2 * half])

  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    blue_pos = data.xpos[self._obj_body]
    red_pos = data.xpos[self._red_body]
    target_pos = self._target_pos(data)
    gripper_pos = data.site_xpos[self._gripper_site]

    gripper_blue_dist = jp.linalg.norm(blue_pos - gripper_pos)
    gripper_blue = 1 - jp.tanh(5 * gripper_blue_dist)
    blue_target = 1 - jp.tanh(5 * jp.linalg.norm(blue_pos - target_pos))
    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )
    red_still = 1 - jp.tanh(10 * jp.linalg.norm(red_pos - info["red_init_pos"]))

    xy_err = jp.linalg.norm((blue_pos - red_pos)[:2])
    z_err = jp.abs(blue_pos[2] - target_pos[2])
    stacked = (
        (xy_err < self._config.stack_xy_tolerance)
        & (z_err < self._config.stack_z_tolerance)
        & self._sensor_found(data, self._blue_red_found_sensor)
    )
    fingers_touch_blue = jp.any(
        jp.array([
            self._sensor_found(data, sensor_id)
            for sensor_id in self._finger_blue_found_sensor
        ])
    )
    success = stacked & ~fingers_touch_blue

    floor_collision = jp.any(
        jp.array([
            self._sensor_found(data, sensor_id)
            for sensor_id in self._floor_hand_found_sensor
        ])
    )
    no_floor_collision = 1 - floor_collision.astype(float)

    info["reached_blue"] = 1.0 * jp.maximum(
        info["reached_blue"], gripper_blue_dist < 0.012
    )

    return {
        "gripper_blue": gripper_blue,
        "blue_target": blue_target * info["reached_blue"],
        "stacked": stacked.astype(float),
        "success": success.astype(float),
        "red_still": red_still,
        "no_floor_collision": no_floor_collision,
        "robot_target_qpos": robot_target_qpos,
    }

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    blue_pos = data.xpos[self._obj_body]
    red_pos = data.xpos[self._red_body]
    return jp.concatenate([
        data.qpos,
        data.qvel,
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        data.xmat[self._red_body].ravel()[3:],
        blue_pos - gripper_pos,
        red_pos - gripper_pos,
        self._target_pos(data) - blue_pos,
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])
