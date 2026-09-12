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

"""Base class for the A0B arm with a Shadow Hand."""

import os
from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.a0b_shadow_hand import a0bsrh_constants as consts

# Meshes come from the a0bsrh_model package (JSK GitLab, not public), which
# must be checked out locally.
MODEL_PATH = epath.Path(
    os.environ.get(
        "A0BSRH_MODEL_PATH",
        os.path.expanduser("~/robot/shadowhand_ws/a0bsrh_model"),
    )
)


def get_assets() -> Dict[str, bytes]:
  if not (MODEL_PATH / "xml" / "assets").exists():
    raise FileNotFoundError(
        f"a0bsrh_model not found at {MODEL_PATH}. Clone"
        " https://gitlab.jsk.imi.i.u-tokyo.ac.jp/wu/a0bsrh_model and point"
        " A0BSRH_MODEL_PATH at it."
    )
  assets = {}
  mjx_env.update_assets(assets, MODEL_PATH / "xml" / "assets" / "srh")
  mjx_env.update_assets(assets, MODEL_PATH / "xml" / "assets" / "a0b")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  # The reorientation cube and its textures are shared with the LEAP hand.
  mjx_env.update_assets(assets, consts.LEAP_XML_PATH / "meshes")
  mjx_env.update_assets(
      assets, consts.LEAP_XML_PATH / "reorientation_cube_textures"
  )
  return assets


class A0bShadowHandEnv(mjx_env.MjxEnv):
  """Base class for A0B + Shadow Hand environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)
    self._model_assets = get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mj_model.opt.ccd_iterations = 10

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._configure_mj_model(self._mj_model)
    self._mjx_model = mjx_env.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

  def _configure_mj_model(self, mj_model: mujoco.MjModel) -> None:
    """Hook for subclasses to patch the MjModel before it is put on MJX."""
    del mj_model  # Unused.

  # Sensor readings.

  def get_palm_position(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "palm_position")

  def get_cube_position(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_position")

  def get_cube_orientation(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_orientation")

  def get_cube_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_linvel")

  def get_cube_angvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_angvel")

  def get_cube_goal_orientation(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_goal_orientation")

  def get_fingertip_positions(self, data: mjx.Data) -> jax.Array:
    """Get fingertip positions relative to the grasp site."""
    return jp.concatenate([
        mjx_env.get_sensor_data(self.mj_model, data, f"{name}_position")
        for name in consts.FINGERTIP_NAMES
    ])

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
