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

"""Cartesian (y, z, gripper) control and pixel observations for reBot tasks."""

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.rebot_b601_dm import base
from mujoco_playground._src.manipulation.rebot_b601_dm import kinematics

VISION_MODE_CHANNELS = {"rgb": 3, "depth": 1, "rgbd": 4}


def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      nworld=1024,
      cam_res=(64, 64),
      use_textures=False,
      use_shadows=False,
      enabled_geom_groups=[0, 1, 2],
      cam_active=None,  # Use all cameras.
  )


def cartesian_config_fields() -> dict:
  """Config entries shared by the Cartesian-control tasks."""
  return dict(
      # Size of cartesian increment.
      action_scale=0.005,
      # Finger speed in m/s, the URDF and MoveIt limit of the DM gripper.
      gripper_speed=0.08,
      # Reachable range of the gripper site along y and z.
      tip_y_range=[-0.15, 0.15],
      tip_z_range=[0.025, 0.22],
      vision=False,
      vision_config=default_vision_config(),
      # Channels of the 'pixels/view_0' observation: 'rgb', 'depth' or 'rgbd'.
      vision_mode="rgb",
      # Distance from the camera in meters that maps to 1 in depth images.
      depth_scale=1.5,
      obs_noise=config_dict.create(brightness=[1.0, 1.0]),
  )


def adjust_brightness(img, scale):
  """Adjusts the brightness of an image by scaling the pixel values."""
  return jp.clip(img * scale, 0, 1)


class CartesianVisionMixin:
  """Moves the gripper in the y-z plane with IK and renders pixel observations.

  The gripper keeps the orientation it has in the keyframe (pointing down).
  Subclasses call `_init_cartesian_vision()` after `_post_init` and set
  `info["current_pos"]` / `info["brightness"]` in `reset`.
  """

  def _init_cartesian_vision(self) -> None:
    self._vision = self._config.vision
    self._defer_rendering = False
    self._kinematics = kinematics.SiteKinematics(
        self._mj_model, self.ARM_JOINTS, "gripper"
    )
    self._start_tip_pos, self._start_tip_rot = self._kinematics.forward(
        jp.array(self._init_ctrl[:6])
    )
    if self._vision:
      mode = self._config.vision_mode
      if mode not in VISION_MODE_CHANNELS:
        raise ValueError(
            f"vision_mode must be one of {list(VISION_MODE_CHANNELS)}, got"
            f" '{mode}'."
        )
      self._render_rgb = mode != "depth"
      self._render_depth = mode != "rgb"
      self._rc = mjx.create_render_context(
          mjm=self._mj_model,
          render_rgb=(self._render_rgb,),
          render_depth=(self._render_depth,),
          **self._config.vision_config.to_dict(),
      )
      self._rc_pytree = self._rc.pytree()

  @property
  def observation_size(self) -> mjx_env.ObservationSize:
    if self._vision:
      width, height = self._config.vision_config.cam_res
      channels = VISION_MODE_CHANNELS[self._config.vision_mode]
      return {"pixels/view_0": (height, width, channels)}
    return super().observation_size

  @property
  def action_size(self) -> int:
    return 3

  def defer_rendering(self) -> None:
    self._defer_rendering = True

  def render_state(self, state: mjx_env.State) -> mjx_env.State:
    data = mjx.refit_bvh(self._mjx_model, state.data, self._rc_pytree)
    rgb_data, depth_data, data = mjx.render(
        self._mjx_model, data, self._rc_pytree
    )
    channels = []
    if self._render_rgb:
      rgb = mjx.get_rgb(self._rc_pytree, 0, rgb_data)
      brightness = state.info["brightness"][..., None, None, :]
      channels.append(adjust_brightness(rgb, brightness))
    if self._render_depth:
      # Camera distance / depth_scale, clipped to [0, 1].
      channels.append(
          mjx.get_depth(
              self._rc_pytree, 0, depth_data, self._config.depth_scale
          )
      )
    obs = {"pixels/view_0": jp.concatenate(channels, axis=-1)}
    return state.replace(
        data=data, obs=obs
    )  # pyrefly: ignore[missing-attribute]

  def _sample_brightness(self, rng: jax.Array) -> jax.Array:
    return jax.random.uniform(
        rng,
        (1,),
        minval=self._config.obs_noise.brightness[0],
        maxval=self._config.obs_noise.brightness[1],
    )

  def _move_tip(
      self,
      current_tip_pos: jax.Array,
      current_ctrl: jax.Array,
      action: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Calculate new joint targets from a (y, z, gripper) increment.

    Returns:
      The new ctrl vector, the new gripper site target and whether the IK
      failed (in which case the arm targets are left unchanged).
    """
    y_lo, y_hi = self._config.tip_y_range
    z_lo, z_hi = self._config.tip_z_range
    new_tip_pos = current_tip_pos + jp.array([0.0, action[0], action[1]]) * (
        self._config.action_scale
    )
    new_tip_pos = new_tip_pos.at[1].set(jp.clip(new_tip_pos[1], y_lo, y_hi))
    new_tip_pos = new_tip_pos.at[2].set(jp.clip(new_tip_pos[2], z_lo, z_hi))

    # The gripper keeps its starting (pointing down) orientation.
    arm_q, err = self._kinematics.inverse(
        current_ctrl[:6], new_tip_pos, self._start_tip_rot, iterations=2
    )
    no_soln = (err > 2e-3) | jp.any(jp.isnan(arm_q))
    arm_q = jp.where(no_soln, current_ctrl[:6], arm_q)
    new_tip_pos = jp.where(no_soln, current_tip_pos, new_tip_pos)

    # Discrete gripper action where a < 0 := closed.
    finger_step = self._config.gripper_speed * self.dt
    finger = current_ctrl[6] + jp.where(
        action[2] < 0, -finger_step, finger_step
    )
    new_ctrl = current_ctrl.at[:6].set(arm_q).at[6].set(finger)
    return new_ctrl, new_tip_pos, no_soln
