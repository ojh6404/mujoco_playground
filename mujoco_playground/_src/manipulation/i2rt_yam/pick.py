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

"""Bring a box to a target and orientation with the i2rt YAM."""

from typing import Any, Dict, Optional, Union

from ml_collections import config_dict

from mujoco_playground._src.manipulation.i2rt_yam import base
from mujoco_playground._src.manipulation.rebot_b601_dm import pick as rebot_pick


def default_config() -> config_dict.ConfigDict:
  """Returns the default config for the YAM pick-cube tasks.

  The task is RebotDmPickCube's: the YAM and the reBot B601-DM have the same
  link lengths, so the box spawn and target ranges are shared.
  """
  return rebot_pick.default_config()


class YamPickCube(base.YamMixin, rebot_pick.RebotDmPickCube):
  """Bring a box to a target."""

  SCENE_XML = base.XML_PATH / "mjx_single_cube.xml"

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      sample_orientation: bool = False,
  ):
    super().__init__(config, config_overrides, sample_orientation)


class YamPickCubeOrientation(YamPickCube):
  """Bring a box to a target and orientation."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides, sample_orientation=True)
