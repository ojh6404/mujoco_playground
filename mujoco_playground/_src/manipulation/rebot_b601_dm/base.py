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

"""reBot Arm B601-DM (Damiao motors) base class."""

import os
import subprocess
from typing import Any, Dict, Optional, Union

from etils import epath
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

XML_PATH = mjx_env.ROOT_PATH / "manipulation" / "rebot_b601_dm" / "xmls"

# Meshes come from the DM variant of the reBot Arm B601 description package.
DESCRIPTION_URL = "https://github.com/Yang-Ci/Rebot_Arm_description.git"
DESCRIPTION_COMMIT_SHA = "3774165c935f6f03665a0a5f14f17477638bd7dd"
DESCRIPTION_PATH = epath.Path(
    os.environ.get(
        "REBOT_ARM_DESCRIPTION_PATH",
        mjx_env.EXTERNAL_DEPS_PATH / "Rebot_Arm_description",
    )
)

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["finger_left", "finger_right"]


def ensure_description_exists() -> None:
  """Clones Rebot_Arm_description at the pinned commit if it is missing."""
  if DESCRIPTION_PATH.exists():
    return
  print("Rebot_Arm_description not found. Downloading...")
  DESCRIPTION_PATH.parent.mkdir(exist_ok=True, parents=True)
  subprocess.run(
      ["git", "clone", "--quiet", DESCRIPTION_URL, str(DESCRIPTION_PATH)],
      check=True,
  )
  subprocess.run(
      [
          "git",
          "-C",
          str(DESCRIPTION_PATH),
          "checkout",
          "--quiet",
          DESCRIPTION_COMMIT_SHA,
      ],
      check=True,
  )


def get_assets() -> Dict[str, bytes]:
  ensure_description_exists()
  assets = {}
  mjx_env.update_assets(assets, XML_PATH, "*.xml")
  meshes = DESCRIPTION_PATH / "DM" / "meshes"
  mjx_env.update_assets(assets, meshes / "visual")
  mjx_env.update_assets(assets, meshes / "shared")
  return assets


class RebotDmBase(mjx_env.MjxEnv):
  """Base environment for the reBot Arm B601-DM.

  Ports of the same tasks to other arms (see `i2rt_yam`) override the joint
  names, `_get_assets` and the task classes' `SCENE_XML`; the scene must keep
  the geom, site and keyframe names of the reBot scenes.
  """

  ARM_JOINTS = ARM_JOINTS
  FINGER_JOINTS = FINGER_JOINTS

  def __init__(
      self,
      xml_path: epath.Path,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)

    self._xml_path = xml_path.as_posix()
    xml = xml_path.read_text()
    self._model_assets = self._get_assets()
    mj_model = mujoco.MjModel.from_xml_string(xml, assets=self._model_assets)
    mj_model.opt.timestep = self.sim_dt

    self._mj_model = mj_model
    self._mjx_model = mjx_env.put_model(mj_model, impl=self._config.impl)
    self._action_scale = self._config.action_scale

  def _get_assets(self) -> Dict[str, bytes]:
    return get_assets()

  def _post_init(self, obj_name: str, keyframe: str):
    all_joints = self.ARM_JOINTS + self.FINGER_JOINTS
    self._robot_arm_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in self.ARM_JOINTS
    ])
    self._robot_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in all_joints
    ])
    self._gripper_site = self._mj_model.site("gripper").id
    self._left_finger_geom = self._mj_model.geom("left_finger_pad").id
    self._right_finger_geom = self._mj_model.geom("right_finger_pad").id
    self._hand_geom = self._mj_model.geom("hand_box").id
    self._obj_body = self._mj_model.body(obj_name).id
    self._obj_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.body(obj_name).jntadr[0]
    ]
    self._mocap_target = self._mj_model.body("mocap_target").mocapid
    self._floor_geom = self._mj_model.geom("floor").id
    self._init_q = self._mj_model.keyframe(keyframe).qpos
    self._init_obj_pos = jp.array(
        self._init_q[self._obj_qposadr : self._obj_qposadr + 3],
        dtype=jp.float32,
    )
    self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl
    self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
