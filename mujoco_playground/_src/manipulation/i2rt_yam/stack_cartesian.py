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

"""Stack a blue cube on a red cube with the i2rt YAM from pixels."""

from typing import Any, Dict, Optional, Union

from ml_collections import config_dict

from mujoco_playground._src.manipulation.i2rt_yam import base
from mujoco_playground._src.manipulation.rebot_b601_dm import stack_cartesian as rebot_stack_cartesian


def default_config() -> config_dict.ConfigDict:
  """Returns the default config, RebotDmStackCubeCartesian's."""
  return rebot_stack_cartesian.default_config()


class YamStackCubeCartesian(
    base.YamMixin, rebot_stack_cartesian.RebotDmStackCubeCartesian
):
  """Stack the blue cube on the red one with (y, z, gripper) actions.

  Success (blue cube resting on the red one, fingers off it) ends the episode.
  Pixel observations need `vision=True` and `impl="warp"`.
  """

  SCENE_XML = base.XML_PATH / "mjx_stack_cubes_camera.xml"

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
