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

"""Stack a blue cube on a red cube with the i2rt YAM."""

from typing import Any, Dict, Optional, Union

from ml_collections import config_dict

from mujoco_playground._src.manipulation.i2rt_yam import base
from mujoco_playground._src.manipulation.rebot_b601_dm import stack as rebot_stack


def default_config() -> config_dict.ConfigDict:
  """Returns the default config, RebotDmStackCube's.

  The Warp contact and constraint buffers are larger: the four pad strips of
  the YAM gripper make about twice the contacts of the reBot pads.
  """
  config = rebot_stack.default_config()
  config.naconmax = 64 * 2048
  config.naccdmax = 64 * 2048
  config.njmax = 512
  return config


class YamStackCube(base.YamMixin, rebot_stack.RebotDmStackCube):
  """Pick up the blue cube and place it on top of the red cube."""

  SCENE_XML = base.XML_PATH / "mjx_stack_cubes.xml"

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
