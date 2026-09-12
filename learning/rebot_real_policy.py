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

"""Runs a trained RebotDmPickCubeReal vision policy on camera images.

For the real reBot with rebot_serl: feed the colour and/or depth image of the
left D435i (cam_b) at 10 Hz and apply the returned increments to the TCP
target of `RobotServer.set_target_pose` (base frame, top-down orientation)
and to the gripper. No simulator or GPU renderer is needed at inference time.

Example:
  policy = RebotRealPolicy("logs/rebot_real/<run>/checkpoints")
  cmd = policy.command(bgr_image[..., ::-1])   # RGB uint8, any resolution
  cmd = policy.command(depth=depth_m)          # depth policy: metres, float
  cmd = policy.command(rgb, depth=depth_m)     # RGB-D policy
  # cmd.delta_xyz (m), cmd.delta_yaw (rad), cmd.gripper (0 closed .. 1 open)
"""

import dataclasses
import functools
import json
import os
from typing import Any, Dict, Optional

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from etils import epath
import jax
import jax.numpy as jp
import numpy as np
from PIL import Image

from mujoco_playground.config import manipulation_params

ENV_NAME = "RebotDmPickCubeReal"
ACTION_SIZE = 5


@dataclasses.dataclass
class Command:
  """One control step for the robot, from a policy action in [-1, 1]."""

  action: np.ndarray  # Raw policy output (dx, dy, dz, dyaw, gripper).
  delta_xyz: np.ndarray  # Target translation increment (m, base frame).
  delta_yaw: float  # Target rotation increment about the vertical (rad).
  gripper: float  # Opening target in [0, 1] (0 closed, 1 open), as
  # RobotServer.set_gripper takes it; the env moves the fingers toward it at
  # their speed limit, which the robot does on its own.
  close: bool  # Opening target below half.


class RebotRealPolicy:
  """Loads a brax PPO vision checkpoint of RebotDmPickCubeReal."""

  def __init__(
      self,
      checkpoint_path: str,
      env_config: Optional[Dict[str, Any]] = None,
      seed: int = 0,
  ):
    # Either the `checkpoints` directory of a run (latest step is used) or
    # one of its numbered step directories.
    path = epath.Path(checkpoint_path).resolve()
    steps = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
    if steps:
      path = max(steps, key=lambda p: int(p.name))
    self.checkpoint_step_path = path
    if env_config is None:
      config_file = path.parent / "config.json"
      with open(config_file, "r", encoding="utf-8") as fp:
        env_config = json.load(fp)
    self.env_config = env_config
    vision_cfg = env_config["vision_config"]
    width, height = vision_cfg["cam_res"]
    # The env renders at cam_res and average-pools by `supersample`.
    ss = int(env_config.get("supersample", 1))
    width, height = width // ss, height // ss
    self.vision_mode = env_config["vision_mode"]
    channels = {"rgb": 3, "depth": 1, "rgbd": 4}[self.vision_mode]
    self.width, self.height = int(width), int(height)
    self.depth_scale = float(env_config["depth_scale"])
    self.action_scale = float(env_config["action_scale"])
    self.yaw_scale = float(env_config["yaw_scale"])

    rl_config = manipulation_params.brax_vision_ppo_config(ENV_NAME)
    network_factory = functools.partial(
        ppo_networks_vision.make_ppo_networks_vision,
        **rl_config.network_factory,
    )
    networks = network_factory(
        observation_size={"pixels/view_0": (self.height, self.width, channels)},
        action_size=ACTION_SIZE,
    )
    make_policy = ppo_networks.make_inference_fn(networks)
    params = ppo_checkpoint.load(path.as_posix())
    self._policy = jax.jit(make_policy(params, deterministic=True))
    self._rng = jax.random.PRNGKey(seed)

  def preprocess(
      self,
      image: Optional[np.ndarray] = None,
      depth: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    """Builds the policy input from an RGB image and/or a depth image.

    Args:
      image: RGB, uint8 or float in [0, 1], any resolution.
      depth: distance from the camera in metres, any resolution. Invalid
        pixels (0 or NaN) are treated as far away.
    """
    channels = []
    if self.vision_mode in ("rgb", "rgbd"):
      if image is None:
        raise ValueError(f"A {self.vision_mode} policy needs an RGB image.")
      if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
      resized = Image.fromarray(image).resize(
          (self.width, self.height), Image.BILINEAR
      )
      channels.append(np.asarray(resized, dtype=np.float32) / 255.0)
    if self.vision_mode in ("depth", "rgbd"):
      if depth is None:
        raise ValueError(f"A {self.vision_mode} policy needs a depth image.")
      depth = np.asarray(depth, dtype=np.float32)
      depth = np.where(np.isfinite(depth) & (depth > 0), depth, np.inf)
      depth = np.clip(depth / self.depth_scale, 0.0, 1.0)
      resized = Image.fromarray(depth, mode="F").resize(
          (self.width, self.height), Image.BILINEAR
      )
      channels.append(np.asarray(resized, dtype=np.float32)[..., None])
    return np.concatenate(channels, axis=-1)

  def act(
      self,
      image: Optional[np.ndarray] = None,
      depth: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    """Returns the policy action in [-1, 1]."""
    obs = {"pixels/view_0": jp.asarray(self.preprocess(image, depth))[None]}
    self._rng, key = jax.random.split(self._rng)
    action, _ = self._policy(obs, key)
    return np.clip(np.asarray(action[0]), -1.0, 1.0)

  def command(
      self,
      image: Optional[np.ndarray] = None,
      depth: Optional[np.ndarray] = None,
  ) -> Command:
    action = self.act(image, depth)
    gripper = float(0.5 * (1.0 + action[4]))
    return Command(
        action=action,
        delta_xyz=action[:3] * self.action_scale,
        delta_yaw=float(action[3] * self.yaw_scale),
        gripper=gripper,
        close=gripper < 0.5,
    )


if __name__ == "__main__":
  import sys  # pylint: disable=g-import-not-at-top

  ckpt, img_path = sys.argv[1], sys.argv[2]
  os.environ.setdefault("JAX_PLATFORMS", "cpu")
  pol = RebotRealPolicy(ckpt)
  print(pol.command(np.asarray(Image.open(img_path).convert("RGB"))))
