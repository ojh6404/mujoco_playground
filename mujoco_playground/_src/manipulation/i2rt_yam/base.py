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

"""i2rt YAM (Yet Another Manipulator) assets and joint names.

The YAM tasks reuse the reBot B601-DM task classes: the two arms have the same
joint layout, link lengths and motors, so only the robot model, the keyframes
and the joint names differ.
"""

from typing import Dict

from mujoco_playground._src import mjx_env

XML_PATH = mjx_env.ROOT_PATH / "manipulation" / "i2rt_yam" / "xmls"
# Meshes come from the mujoco_menagerie i2rt_yam model.
MENAGERIE_YAM_PATH = mjx_env.MENAGERIE_PATH / "i2rt_yam"

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["left_finger", "right_finger"]


def get_assets() -> Dict[str, bytes]:
  mjx_env.ensure_menagerie_exists()
  assets = {}
  mjx_env.update_assets(assets, XML_PATH, "*.xml")
  mjx_env.update_assets(assets, MENAGERIE_YAM_PATH / "assets")
  return assets


class YamMixin:
  """Swaps the reBot Arm B601-DM of a reBot task class for the YAM.

  Put it first in the bases: it overrides the joint names and the assets of
  `rebot_b601_dm.base.RebotDmBase`; the task class sets `SCENE_XML`.
  """

  ARM_JOINTS = ARM_JOINTS
  FINGER_JOINTS = FINGER_JOINTS

  def _get_assets(self) -> Dict[str, bytes]:
    return get_assets()
