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

"""Pick up a 5 cm cube with the reBot B601-DM in its calibrated real setup.

Made for sim-to-real with rebot_serl: the scene, the camera (the D435i on the
robot's left) and the reset pose follow its calibration, the arm follows the
gains and target filter of its controller, control runs at 10 Hz with 1 cm
Cartesian steps, 0.1 rad yaw steps (its compliance clips) and a continuous
gripper opening, and the cube colour, table shade, camera pose and image
appearance are randomized every episode.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.rebot_b601_dm import base
from mujoco_playground._src.manipulation.rebot_b601_dm import cartesian
from mujoco_playground._src.manipulation.rebot_b601_dm import pick_cartesian

# Cube colours to train with (RGB): yellow, light green, purple, red.
CUBE_COLORS = [
    [1.0, 0.85, 0.0],
    [0.55, 0.85, 0.25],
    [0.55, 0.3, 0.8],
    [0.9, 0.1, 0.1],
]


def default_config() -> config_dict.ConfigDict:
  config = pick_cartesian.default_config()
  del config["box_init_range"]
  # Both finger pads touching the cube: rewards the closing sequence, which
  # takes five steps from fully open, before any lift.
  config.reward_config.grasped_reward = 1.0
  # Fraction of episodes that start from the `picked` keyframe (cube held
  # 3 cm above the table), so lifting is explored from a grasped state.
  config.guide_prob = 0.1
  config.ctrl_dt = 0.1  # The rebot_serl env rate.
  config.episode_length = 100
  # Per step: 1 cm (0.1 m/s) and 0.1 rad, the compliance clips of rebot_serl.
  config.action_scale = 0.01
  config.yaw_scale = 0.1
  # Finger speed (m/s): motor 12 rad/s through the 205.6 rad/m rack.
  config.gripper_speed = 0.05
  # Travel per finger (m). The URDF gives 0.05 (100 mm opening) and the
  # gripper is said to open that far; rebot_serl measured 28.5 mm per finger
  # (57 mm) between its motor hard stops, which leaves 3.5 mm per side around
  # the 5 cm cube. Set this to the opening of the real gripper.
  config.gripper_travel = 0.05
  config.tip_x_range = [0.20, 0.42]
  config.tip_y_range = [-0.18, 0.18]
  config.tip_z_range = [0.035, 0.20]
  config.yaw_range = [-1.57, 1.57]
  # Height of the grasp site at which the cube counts as picked.
  config.target_height = 0.15
  config.box_spawn = config_dict.create(
      x=[0.24, 0.38], y=[-0.12, 0.12], yaw=[-0.785, 0.785]
  )
  config.randomization = config_dict.create(
      cube_colors=CUBE_COLORS,
      color_jitter=0.08,
      floor_gray=[0.85, 1.0],
      ground_gray=[0.2, 0.6],
      # Camera pose noise, to cover calibration error: m and degrees.
      cam_pos_noise=0.01,
      cam_rot_noise=2.0,
  )
  config.obs_noise.brightness = [0.7, 1.3]
  # Low-level control after rebot_serl (config/controller.yaml): the MIT-mode
  # PD gains of its "compliance" profile run in the motor firmware (Nm/rad,
  # Nm s/rad), the reference may lead the joint by at most max_joint_lead
  # (bounds the PD torque at kp * lead; gravity is compensated separately),
  # and the 500 Hz loop filters the target with a first-order lag. The real
  # gripper motor is stall-limited at 0.45 Nm (92.5 N on the rack); with the
  # soft contacts of the 50 g cube that force sinks the pads 1 cm into it, so
  # the sim gripper is bounded at 20 N (the old 8 N of the URDF holds the
  # cube too). With enable=False the playground gains of the DM MuJoCo node
  # are kept.
  config.actuation = config_dict.create(
      enable=True,
      joint_stiffness=[60.0, 60.0, 60.0, 10.0, 10.0, 8.0],
      joint_damping=[4.0, 4.0, 4.0, 1.2, 1.0, 0.8],
      max_joint_lead=[0.15, 0.15, 0.15, 0.30, 0.30, 0.40],
      target_filter_tau=0.05,
      gripper_force=20.0,
  )
  # Rendered at twice the policy resolution and average-pooled, like a camera
  # image resized to the 96 x 72 (4:3, as the D435i colour stream) input; the
  # Warp renderer casts one ray per pixel, so a direct 96 x 72 render is
  # aliased in a way a resized camera image is not.
  config.vision_config.cam_res = (192, 144)
  config.supersample = 2
  # Per-episode image augmentation on top of the brightness factor: a gain per
  # colour channel (white balance), a blend with a 3 x 3 box blur, and
  # per-step Gaussian pixel noise (standard deviation drawn per episode).
  config.image_noise = config_dict.create(
      channel_gain=0.1, blur=[0.0, 1.0], noise_std=[0.0, 0.03]
  )
  return config


class RebotDmPickCubeReal(pick_cartesian.RebotDmPickCubeCartesian):
  """Lift a 5 cm cube with (x, y, z, yaw, gripper) actions and a real camera."""

  def __init__(  # pylint: disable=non-parent-init-called,super-init-not-called
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    base.RebotDmBase.__init__(
        self,
        base.XML_PATH / "mjx_real_cube_camera.xml",
        config,
        config_overrides,
    )
    self._set_gripper_travel(self._config.gripper_travel)
    if self._config.actuation.enable:
      self._set_actuation()
    self._mjx_model = mjx_env.put_model(self._mj_model, impl=self._config.impl)
    self._sample_orientation = False
    self._post_init(obj_name="box", keyframe="ready")
    # Start with the fingers fully open, whatever the travel.
    m = self._mj_model
    travel = self._config.gripper_travel
    init_q = np.array(self._init_q)
    init_q[m.jnt_qposadr[m.joint("finger_left").id]] = travel
    init_q[m.jnt_qposadr[m.joint("finger_right").id]] = -travel
    self._init_q = init_q
    init_ctrl = np.array(self._init_ctrl)
    init_ctrl[m.actuator("gripper").id] = travel
    self._init_ctrl = init_ctrl

    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_box"]
    ]
    self._box_hand_found_sensor = self._mj_model.sensor("box_hand_found").id
    self._pad_box_found_sensors = [
        self._mj_model.sensor(f"{side}_finger_pad_box_found").id
        for side in ["left", "right"]
    ]
    self._guide_q = self._mj_model.keyframe("picked").qpos
    self._guide_ctrl = self._mj_model.keyframe("picked").ctrl
    self._init_cartesian_vision()

    self._box_geom = self._mj_model.geom("box").id
    self._ground_geom = self._mj_model.geom("ground").id
    self._cam_id = self._mj_model.camera("cam_b").id
    self._cube_colors = jp.array(
        self._config.randomization.cube_colors, dtype=float
    )

  # Proprioception given to the vision policy next to the pixels: the
  # integrated target of the grasp site relative to its start (3), the yaw
  # target (1), the finger opening as a fraction of the travel (1) and the
  # last action (5); all of it is known on the robot too. The critic also
  # sees the cube position relative to the grasp site and its height.
  PROPRIO_SIZE = 10
  PRIVILEGED_SIZE = 14

  @property
  def observation_size(self) -> mjx_env.ObservationSize:
    if self._vision:
      width, height = self._config.vision_config.cam_res
      ss = self._config.supersample
      channels = cartesian.VISION_MODE_CHANNELS[self._config.vision_mode]
      return {
          "pixels/view_0": (height // ss, width // ss, channels),
          "state": (self.PROPRIO_SIZE,),
          "privileged_state": (self.PRIVILEGED_SIZE,),
      }
    return super().observation_size

  def _proprio(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    opening = data.ctrl[..., 6] / self._config.gripper_travel
    return jp.concatenate(
        [
            info["current_pos"] - self._start_tip_pos,
            info["current_yaw"][..., None],
            opening[..., None],
            info["last_action"],
        ],
        axis=-1,
    )

  @staticmethod
  def _pool(img: jax.Array, ss: int) -> jax.Array:
    """Average-pools the last two spatial axes by `ss`."""
    if ss == 1:
      return img
    *batch, h, w, c = img.shape
    img = img.reshape(*batch, h // ss, ss, w // ss, ss, c)
    return img.mean(axis=(-4, -2))

  @staticmethod
  def _box_blur(img: jax.Array) -> jax.Array:
    """3 x 3 box blur with edge padding over the last two spatial axes."""
    padded = jp.pad(
        img, [(0, 0)] * (img.ndim - 3) + [(1, 1), (1, 1), (0, 0)], mode="edge"
    )
    h, w = img.shape[-3], img.shape[-2]
    total = 0.0
    for dy in range(3):
      for dx in range(3):
        total = total + padded[..., dy : dy + h, dx : dx + w, :]
    return total / 9.0

  def _set_gripper_travel(self, travel: float) -> None:
    """Limits the fingers to the travel of the real gripper."""
    m = self._mj_model
    m.jnt_range[m.joint("finger_left").id] = [0.0, travel]
    m.jnt_range[m.joint("finger_right").id] = [-travel, 0.0]
    m.actuator_ctrlrange[m.actuator("gripper").id] = [0.0, travel]

  def _set_actuation(self) -> None:
    """Sets the PD gains and torque bounds of the real controller."""
    m = self._mj_model
    cfg = self._config.actuation
    for i, name in enumerate(base.ARM_JOINTS):
      aid = m.actuator(name).id
      kp, kd = cfg.joint_stiffness[i], cfg.joint_damping[i]
      m.actuator_gainprm[aid, 0] = kp
      m.actuator_biasprm[aid, 1] = -kp
      m.actuator_biasprm[aid, 2] = -kd
      limit = kp * cfg.max_joint_lead[i]
      m.actuator_forcerange[aid] = [-limit, limit]
    gid = m.actuator("gripper").id
    m.actuator_forcerange[gid] = [-cfg.gripper_force, cfg.gripper_force]

  def _physics_step(
      self, data: mjx.Data, ctrl: jax.Array, filtered: jax.Array
  ) -> tuple[mjx.Data, jax.Array]:
    """Steps the physics; the arm targets go through the target filter."""
    tau = self._config.actuation.target_filter_tau
    if not self._config.actuation.enable or tau <= 0.0:
      return (
          mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps),
          ctrl[:6],
      )
    alpha = 1.0 - jp.exp(-self.sim_dt / tau)

    def substep(carry, _):
      data, filtered = carry
      filtered = filtered + alpha * (ctrl[:6] - filtered)
      data = data.replace(ctrl=ctrl.at[:6].set(filtered))
      return (mjx.step(self._mjx_model, data), filtered), None

    (data, filtered), _ = jax.lax.scan(
        substep, (data, filtered), (), self.n_substeps
    )
    return data, filtered

  # Sampling.

  def _sample_visuals(self, rng: jax.Array) -> Dict[str, jax.Array]:
    """Cube and table colours, camera pose and brightness of an episode."""
    cfg = self._config.randomization
    (
        k_color,
        k_jitter,
        k_floor,
        k_ground,
        k_pos,
        k_axis,
        k_angle,
        k_bright,
        k_gain,
        k_blur,
        k_noise,
    ) = jax.random.split(rng, 11)
    idx = jax.random.randint(k_color, (), 0, self._cube_colors.shape[0])
    rgb = self._cube_colors[idx] + jax.random.uniform(
        k_jitter, (3,), minval=-cfg.color_jitter, maxval=cfg.color_jitter
    )
    gray = jax.random.uniform(
        k_floor, (), minval=cfg.floor_gray[0], maxval=cfg.floor_gray[1]
    )
    ground = jax.random.uniform(
        k_ground, (), minval=cfg.ground_gray[0], maxval=cfg.ground_gray[1]
    )
    m = self._mjx_model
    cam_pos = m.cam_pos[self._cam_id] + jax.random.uniform(
        k_pos, (3,), minval=-cfg.cam_pos_noise, maxval=cfg.cam_pos_noise
    )
    axis = jax.random.normal(k_axis, (3,))
    axis = axis / jp.linalg.norm(axis)
    angle = jax.random.uniform(
        k_angle, (), minval=0.0, maxval=jp.deg2rad(cfg.cam_rot_noise)
    )
    cam_quat = math.quat_mul(
        math.axis_angle_to_quat(axis, angle), m.cam_quat[self._cam_id]
    )
    noise = self._config.image_noise
    return {
        "cube_rgba": jp.concatenate([jp.clip(rgb, 0.0, 1.0), jp.ones(1)]),
        "floor_rgba": jp.array([gray, gray, gray, 1.0]),
        "ground_rgba": jp.array([ground, ground, ground, 1.0]),
        "cam_pos": cam_pos,
        "cam_quat": cam_quat,
        "brightness": self._sample_brightness(k_bright),
        "channel_gain": 1.0 + jax.random.uniform(
            k_gain, (3,), minval=-noise.channel_gain, maxval=noise.channel_gain
        ),
        "blur": jax.random.uniform(
            k_blur, (), minval=noise.blur[0], maxval=noise.blur[1]
        ),
        "noise_std": jax.random.uniform(
            k_noise, (), minval=noise.noise_std[0], maxval=noise.noise_std[1]
        ),
    }

  def _sample_box(self, rng: jax.Array) -> Dict[str, jax.Array]:
    """Cube pose and lift target of an episode."""
    k_xy, k_yaw = jax.random.split(rng)
    spawn = self._config.box_spawn
    xy = jax.random.uniform(
        k_xy,
        (2,),
        minval=jp.array([spawn.x[0], spawn.y[0]]),
        maxval=jp.array([spawn.x[1], spawn.y[1]]),
    )
    yaw = jax.random.uniform(
        k_yaw, (), minval=spawn.yaw[0], maxval=spawn.yaw[1]
    )
    pos = jp.array([xy[0], xy[1], self._init_obj_pos[2]])
    quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
    target_pos = jp.array([xy[0], xy[1], self._config.target_height])
    return {"pos": pos, "quat": quat, "target_pos": target_pos}

  @staticmethod
  def _yaw_rot(yaw: jax.Array) -> jax.Array:
    c, s = jp.cos(yaw), jp.sin(yaw)
    return jp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

  # Environment API.

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, k_box, k_vis = jax.random.split(rng, 3)
    box = self._sample_box(k_box)
    adr = self._obj_qposadr
    init_q = jp.array(self._init_q)
    init_q = init_q.at[adr : adr + 3].set(box["pos"])
    init_q = init_q.at[adr + 3 : adr + 7].set(box["quat"])
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
      # The mocap target must not appear in pixels.
      data = data.replace(
          mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(
              box["target_pos"]
          )
      )

    metrics = {
        "out_of_bounds": jp.array(0.0),
        **{
            f"reward/{k}": 0.0
            for k in self._config.reward_config.reward_scales.keys()
        },
        "reward/success": jp.array(0.0),
        "reward/grasped": jp.array(0.0),
        "reward/lifted": jp.array(0.0),
        "no_soln": jp.array(0.0),
    }
    info = {
        "rng": rng,
        "target_pos": box["target_pos"],
        "reached_box": jp.array(0.0, dtype=float),
        "prev_reward": jp.array(0.0, dtype=float),
        "current_pos": self._start_tip_pos,
        "current_yaw": jp.array(0.0, dtype=float),
        "newly_reset": jp.array(False, dtype=bool),
        "prev_action": jp.zeros(self.action_size),
        "last_action": jp.zeros(self.action_size),
        "_steps": jp.array(0, dtype=int),
        "action_history": jp.zeros((self._config.action_history_length,)),
        "filtered_ctrl": jp.array(self._init_ctrl[:6], dtype=float),
        **self._sample_visuals(k_vis),
    }

    reward, done = jp.zeros(2)
    obs = self._get_obs(data, info)
    obs = jp.concat([obs, jp.zeros(1), jp.zeros(self.action_size)], axis=0)
    if self._vision:
      data = mjx.forward(self._mjx_model, data)
    state = mjx_env.State(data, obs, reward, done, metrics, info)
    if self._vision and not self._defer_rendering:
      state = self.render_state(state)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    info = state.info
    gripper_idx = self.action_size - 1
    action_history = (
        jp.roll(info["action_history"], 1).at[0].set(action[gripper_idx])
    )
    info["action_history"] = action_history
    info["rng"], key = jax.random.split(info["rng"])
    action_idx = jax.random.randint(
        key, (), minval=0, maxval=self._config.action_history_length
    )
    action = action.at[gripper_idx].set(action_history[action_idx])

    newly_reset = info["_steps"] == 0
    info["newly_reset"] = newly_reset
    info["prev_reward"] = jp.where(newly_reset, 0.0, info["prev_reward"])
    info["current_pos"] = jp.where(
        newly_reset, self._start_tip_pos, info["current_pos"]
    )
    info["current_yaw"] = jp.where(newly_reset, 0.0, info["current_yaw"])
    info["reached_box"] = jp.where(newly_reset, 0.0, info["reached_box"])
    info["prev_action"] = jp.where(
        newly_reset, jp.zeros(self.action_size), info["prev_action"]
    )
    info["last_action"] = jp.where(
        newly_reset, jp.zeros(self.action_size), info["last_action"]
    )

    # The training wrapper restores the first state of the environment at
    # every reset; draw a new cube pose and new visuals for each episode.
    info["rng"], k_box, k_vis = jax.random.split(info["rng"], 3)
    box = self._sample_box(k_box)
    adr = self._obj_qposadr
    qpos = state.data.qpos
    qpos = qpos.at[adr : adr + 3].set(
        jp.where(newly_reset, box["pos"], qpos[adr : adr + 3])
    )
    qpos = qpos.at[adr + 3 : adr + 7].set(
        jp.where(newly_reset, box["quat"], qpos[adr + 3 : adr + 7])
    )
    data = state.data.replace(qpos=qpos)
    info["target_pos"] = jp.where(
        newly_reset, box["target_pos"], info["target_pos"]
    )
    visuals = self._sample_visuals(k_vis)
    for k, v in visuals.items():
      info[k] = jp.where(newly_reset, v, info[k])

    # Occasionally start with the cube already grasped.
    info["rng"], key_swap = jax.random.split(info["rng"])
    to_sample = newly_reset * jax.random.bernoulli(
        key_swap, self._config.guide_prob
    )
    swapped_data = data.replace(qpos=self._guide_q, ctrl=self._guide_ctrl)
    data = jax.tree_util.tree_map_with_path(
        lambda path, x, y: ((1 - to_sample) * x + to_sample * y).astype(x.dtype)
        if len(path) == 1
        else x,
        data,
        swapped_data,
    )

    ctrl, new_tip_pos, new_yaw, no_soln = self._move_tip_yaw(
        info["current_pos"], info["current_yaw"], data.ctrl, action
    )
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)
    info["current_pos"] = new_tip_pos
    info["current_yaw"] = new_yaw

    filtered = jp.where(newly_reset, data.ctrl[:6], info["filtered_ctrl"])
    data, info["filtered_ctrl"] = self._physics_step(data, ctrl, filtered)

    raw_rewards = self._get_reward(data, info)
    rewards = {
        k: v * self._config.reward_config.reward_scales[k]
        for k, v in raw_rewards.items()
    }
    hand_box = (
        data.sensordata[self._mj_model.sensor_adr[self._box_hand_found_sensor]]
        > 0
    )
    raw_rewards["no_box_collision"] = jp.where(hand_box, 0.0, 1.0)
    total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

    if not self._vision:
      da = jp.linalg.norm(action - info["prev_action"])
      info["prev_action"] = action
      total_reward += self._config.reward_config.action_rate * da
      total_reward += no_soln * self._config.reward_config.no_soln_reward

    grasped = jp.all(
        jp.array([
            data.sensordata[self._mj_model.sensor_adr[sid]] > 0
            for sid in self._pad_box_found_sensors
        ])
    )
    total_reward += grasped * self._config.reward_config.grasped_reward

    box_pos = data.xpos[self._obj_body]
    lifted = (
        box_pos[2] > self._init_obj_pos[2] + 0.02
    ) * self._config.reward_config.lifted_reward
    total_reward += lifted
    success = self._get_success(data, info)
    total_reward += success * self._config.reward_config.success_reward

    reward = jp.maximum(total_reward - info["prev_reward"], 0.0)
    info["prev_reward"] = jp.maximum(total_reward, info["prev_reward"])
    reward = jp.where(newly_reset, 0.0, reward)
    reward = jp.where(jp.isnan(reward), 0.0, reward)

    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    state.metrics.update(out_of_bounds=out_of_bounds.astype(float))
    state.metrics.update({f"reward/{k}": v for k, v in raw_rewards.items()})
    state.metrics.update({
        "reward/grasped": grasped.astype(float),
        "reward/lifted": lifted.astype(float),
        "reward/success": success.astype(float),
        "no_soln": no_soln.astype(float),
    })

    done = (
        out_of_bounds
        | jp.isnan(data.qpos).any()
        | jp.isnan(data.qvel).any()
        | success
    )
    info["_steps"] += self._config.action_repeat
    info["_steps"] = jp.where(
        done | (info["_steps"] >= self._config.episode_length),
        0,
        info["_steps"],
    )

    info["last_action"] = action
    obs = self._get_obs(data, info)
    obs = jp.concat([obs, no_soln.reshape(1), action], axis=0)
    state = state.replace(  # pyrefly: ignore[missing-attribute]
        data=data,
        obs=obs,
        reward=reward,
        done=done.astype(float),
        info=info,
    )
    if self._vision and not self._defer_rendering:
      state = self.render_state(state)
    return state

  def _move_tip_yaw(
      self,
      current_tip_pos: jax.Array,
      current_yaw: jax.Array,
      current_ctrl: jax.Array,
      action: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Joint targets for an (x, y, z, yaw, gripper) increment."""
    lo = jp.array([
        self._config.tip_x_range[0],
        self._config.tip_y_range[0],
        self._config.tip_z_range[0],
    ])
    hi = jp.array([
        self._config.tip_x_range[1],
        self._config.tip_y_range[1],
        self._config.tip_z_range[1],
    ])
    new_tip_pos = jp.clip(
        current_tip_pos + action[:3] * self._config.action_scale, lo, hi
    )
    new_yaw = jp.clip(
        current_yaw + action[3] * self._config.yaw_scale,
        self._config.yaw_range[0],
        self._config.yaw_range[1],
    )
    # The gripper points down; the yaw turns it about the vertical axis.
    target_rot = self._yaw_rot(new_yaw) @ self._start_tip_rot
    arm_q, err = self._kinematics.inverse(
        current_ctrl[:6], new_tip_pos, target_rot, iterations=4
    )
    no_soln = (err > 2e-3) | jp.any(jp.isnan(arm_q))
    arm_q = jp.where(no_soln, current_ctrl[:6], arm_q)
    new_tip_pos = jp.where(no_soln, current_tip_pos, new_tip_pos)
    new_yaw = jp.where(no_soln, current_yaw, new_yaw)

    # Continuous opening target: -1 closed, +1 fully open, reached at the
    # finger speed. (With a discrete open/close threshold at 0 the trained
    # policy dithered around it and dropped grasped cubes.)
    finger_step = self._config.gripper_speed * self.dt
    opening = 0.5 * (1.0 + action[4]) * self._config.gripper_travel
    finger = jp.clip(
        opening, current_ctrl[6] - finger_step, current_ctrl[6] + finger_step
    )
    new_ctrl = current_ctrl.at[:6].set(arm_q).at[6].set(finger)
    return new_ctrl, new_tip_pos, new_yaw, no_soln

  def render_state(self, state: mjx_env.State) -> mjx_env.State:
    """Renders with the episode's colours and camera pose."""
    info = state.info
    batch = info["cube_rgba"].shape[:-1]
    m = self._mjx_model
    geom_rgba = jp.broadcast_to(m.geom_rgba, batch + m.geom_rgba.shape)
    geom_rgba = geom_rgba.at[..., self._box_geom, :].set(info["cube_rgba"])
    geom_rgba = geom_rgba.at[..., self._floor_geom, :].set(info["floor_rgba"])
    geom_rgba = geom_rgba.at[..., self._ground_geom, :].set(info["ground_rgba"])
    cam_pos = jp.broadcast_to(m.cam_pos, batch + m.cam_pos.shape)
    cam_pos = cam_pos.at[..., self._cam_id, :].set(info["cam_pos"])
    cam_quat = jp.broadcast_to(m.cam_quat, batch + m.cam_quat.shape)
    cam_quat = cam_quat.at[..., self._cam_id, :].set(info["cam_quat"])
    model = m.tree_replace({
        "geom_rgba": geom_rgba,
        "cam_pos": cam_pos,
        "cam_quat": cam_quat,
    })
    data = mjx.refit_bvh(model, state.data, self._rc_pytree)
    rgb_data, depth_data, data = mjx.render(model, data, self._rc_pytree)
    ss = self._config.supersample
    channels = []
    if self._render_rgb:
      rgb = self._pool(mjx.get_rgb(self._rc_pytree, 0, rgb_data), ss)
      gain = (
          info["brightness"][..., None, None, :]
          * info["channel_gain"][..., None, None, :]
      )
      rgb = jp.clip(rgb * gain, 0.0, 1.0)
      blur = info["blur"][..., None, None, None]
      rgb = (1.0 - blur) * rgb + blur * self._box_blur(rgb)
      channels.append(rgb)
    if self._render_depth:
      depth = mjx.get_depth(
          self._rc_pytree, 0, depth_data, self._config.depth_scale
      )
      channels.append(self._pool(depth, ss))
    pixels = jp.concatenate(channels, axis=-1)
    # Per-step pixel noise; keys per world when the state is batched.
    rng = info["rng"]
    fold = lambda k: jax.random.fold_in(k, 7)
    key = jax.vmap(fold)(rng) if rng.ndim == 2 else fold(rng)
    if rng.ndim == 2:
      noise = jax.vmap(lambda k: jax.random.normal(k, pixels.shape[1:]))(key)
    else:
      noise = jax.random.normal(key, pixels.shape)
    pixels = jp.clip(
        pixels + info["noise_std"][..., None, None, None] * noise, 0.0, 1.0
    )
    proprio = self._proprio(data, info)
    cube_rel = (
        data.xpos[..., self._obj_body, :]
        - data.site_xpos[..., self._gripper_site, :]
    )
    privileged = jp.concatenate(
        [proprio, cube_rel, data.xpos[..., self._obj_body, 2:3]], axis=-1
    )
    obs = {
        "pixels/view_0": pixels,
        "state": proprio,
        "privileged_state": privileged,
    }
    return state.replace(  # pyrefly: ignore[missing-attribute]
        data=data, obs=obs
    )

  @property
  def action_size(self) -> int:
    return 5
