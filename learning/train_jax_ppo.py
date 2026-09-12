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

"""Train a PPO agent using JAX on the specified environment."""

import os

# These must be set before importing mujoco and jax: mujoco picks its GL
# backend when it is imported and XLA reads XLA_FLAGS when it starts.
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# pylint: disable=wrong-import-position
# ruff: noqa: E402
import datetime
import functools
import json
import time
from types import SimpleNamespace
from typing import Optional
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import numpy as np

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

try:
  import tensorboardX
except ImportError:
  tensorboardX = None

try:
  import wandb
except ImportError:
  wandb = None


if not hasattr(jax, "device_put_replicated"):
  # brax <= 0.14.2 still calls jax.device_put_replicated, which jax >= 0.11
  # removed. Replicate the pytree onto every device along a leading axis, as
  # the old function did.
  def _device_put_replicated(tree, devices):
    mesh = jax.sharding.Mesh(np.array(devices), ("devices",))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("devices")
    )
    return jax.tree.map(
        lambda x: jax.device_put(jp.stack([x] * len(devices)), sharding), tree
    )

  jax.device_put_replicated = _device_put_replicated

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string(
    "playground_config_overrides",
    None,
    "Overrides for the playground env config.",
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)
_EVAL_VIDEO_ENVS = flags.DEFINE_integer(
    "eval_video_envs",
    0,
    "Number of environments to roll out and render with the current policy at"
    " every evaluation. Videos go to <logdir>/eval_videos; with --use_tb, frame"
    " strips are also logged to TensorBoard.",
)
_RENDER_CAMERA = flags.DEFINE_string(
    "render_camera",
    None,
    "Name of the MJCF camera used for rollout videos (default: free camera).",
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.3, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)
_WARP_KERNEL_CACHE_DIR = flags.DEFINE_string(
    "warp_kernel_cache_dir",
    None,
    "Directory for caching compiled Warp kernels.",
)
_LOGDIR = flags.DEFINE_string("logdir", None, "Directory for logging.")


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in mujoco_playground.manipulation._envs:
    if _VISION.value:
      return manipulation_params.brax_vision_ppo_config(env_name, _IMPL.value)
    return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.locomotion._envs:
    return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.dm_control_suite._envs:
    if _VISION.value:
      return dm_control_suite_params.brax_vision_ppo_config(
          env_name, _IMPL.value
      )
    return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def scale_contact_buffers(
    overrides: dict, env_name: str, ref_num_envs: int, num_envs: int
) -> dict:
  """Sizes the Warp contact buffers of a config for `num_envs` worlds.

  `naconmax` and `naccdmax` count contacts across all worlds, and the tuned
  env configs size them for the training batch (`ref_num_envs`). Copying them
  to an evaluation or rendering env with a handful of worlds wastes GPU memory
  and can run the training out of it.
  """
  env_cfg = registry.get_default_config(env_name)
  overrides = dict(overrides)
  for key in ("naconmax", "naccdmax"):
    if key not in env_cfg:
      continue
    total = int(overrides.get(key, env_cfg[key]))
    per_env = max(-(-total // max(ref_num_envs, 1)), 1)
    # The buffer is shared by all worlds, so a small batch gets headroom for
    # worlds with many contacts.
    overrides[key] = min(total, max(4 * per_env * num_envs, 2048))
  return overrides


def rscope_fn(full_states, obs, rew, done):
  """
  All arrays are of shape (unroll_length, rscope_envs, ...)
  full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
  obs: nd.array or dict obs based on env configuration
  rew: nd.array rewards
  done: nd.array done flags
  """
  # Calculate cumulative rewards per episode, stopping at first done flag
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "Collected rscope rollouts with reward"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )


class RolloutRenderer:
  """Rolls out a policy in a few environments and renders the episodes.

  Environments are wrapped like in training (episode bookkeeping, autoreset,
  deferred vision rendering). Each rendered frame shows an external camera and,
  for vision policies, the pixels the policy receives.
  """

  _RENDER_EVERY = 2
  _STRIP_FRAMES = 8

  def __init__(
      self,
      env_name: str,
      env_cfg_overrides: dict,
      ppo_params: config_dict.ConfigDict,
      num_envs: int,
      vision: bool,
      seed: int,
      camera: Optional[str] = None,
  ):
    self._camera = camera
    overrides = scale_contact_buffers(
        env_cfg_overrides, env_name, ppo_params.num_envs, num_envs
    )
    if vision:
      overrides["vision_config.nworld"] = num_envs
    self._env = registry.load(
        env_name,
        config=registry.get_default_config(env_name),
        config_overrides=overrides,
    )
    self._wrapped = wrapper.wrap_for_brax_training(
        self._env,
        episode_length=ppo_params.episode_length,
        action_repeat=ppo_params.get("action_repeat", 1),
    )
    self._num_envs = num_envs
    self._vision = vision
    self._episode_length = ppo_params.episode_length
    self._seed = seed
    self._make_policy = None
    self._reset = jax.jit(self._wrapped.reset)
    self._rollout = None
    self._scene_option = mujoco.MjvOption()
    self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  @property
  def fps(self) -> float:
    return 1.0 / self._env.dt / self._RENDER_EVERY

  def _do_rollout(self, state, rng, params):
    policy = self._make_policy(params, deterministic=True)

    def step(carry, _):
      state, rng = carry
      rng, act_key = jax.random.split(rng)
      act_keys = jax.random.split(act_key, self._num_envs)
      act = jax.vmap(policy)(state.obs, act_keys)[0]
      state = self._wrapped.step(state, act)
      out = {
          "qpos": state.data.qpos,
          "qvel": state.data.qvel,
          "mocap_pos": state.data.mocap_pos,
          "mocap_quat": state.data.mocap_quat,
          "xfrc_applied": state.data.xfrc_applied,
          "reward": state.reward,
          "done": state.done,
      }
      if self._vision:
        out["pixels"] = {
            k: v for k, v in state.obs.items() if k.startswith("pixels/")
        }
      return (state, rng), out

    _, traj = jax.lax.scan(
        step, (state, rng), None, length=self._episode_length
    )
    # (time, num_envs, ...) -> (num_envs, time, ...).
    return jax.tree.map(lambda x: jp.moveaxis(x, 0, 1), traj)

  def _policy_view(self, pixels: dict, env_idx: int, t: int, height: int):
    """Tiles the policy's pixel observations for one frame as uint8."""
    tiles = []
    for key in sorted(pixels):
      img = np.asarray(pixels[key][env_idx, t])
      # Split RGB-D into an RGB tile and a depth tile.
      splits = [img[..., :3], img[..., 3:]] if img.shape[-1] == 4 else [img]
      for split in splits:
        if split.shape[-1] == 1:
          split = np.repeat(split, 3, axis=-1)
        split = (np.clip(split, 0, 1) * 255).astype(np.uint8)
        tiles.append(media.resize_image(split, (height, height)))
    return np.concatenate(tiles, axis=1)

  def render(
      self,
      make_policy,
      params,
      out_dir: epath.Path,
      name: str,
      writer=None,
      step: int = 0,
      height: int = 240,
      width: int = 320,
  ):
    """Writes <out_dir>/<name><i>.mp4 per env and logs strips to TensorBoard."""
    if self._make_policy is not make_policy:
      self._make_policy = make_policy
      self._rollout = jax.jit(self._do_rollout)
    rng = jax.random.split(jax.random.PRNGKey(self._seed), self._num_envs)
    state = self._reset(rng)
    traj = self._rollout(state, jax.random.PRNGKey(self._seed + 1), params)
    traj = jax.tree.map(np.asarray, traj)

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ("qpos", "qvel", "mocap_pos", "mocap_quat", "xfrc_applied")
    steps = range(0, self._episode_length, self._RENDER_EVERY)
    for i in range(self._num_envs):
      states = [
          SimpleNamespace(
              data=SimpleNamespace(**{k: traj[k][i, t] for k in fields})
          )
          for t in steps
      ]
      frames = self._env.render(
          states,
          height=height,
          width=width,
          camera=self._camera,
          scene_option=self._scene_option,
      )
      if self._vision:
        frames = [
            np.concatenate(
                [f, self._policy_view(traj["pixels"], i, t, f.shape[0])],
                axis=1,
            )
            for f, t in zip(frames, steps)
        ]
      path = out_dir / f"{name}{i}.mp4"
      media.write_video(path, frames, fps=self.fps)
      print(f"Rollout video saved as '{path}'.")
      if writer is not None:
        idx = np.linspace(0, len(frames) - 1, self._STRIP_FRAMES).astype(int)
        strip = np.concatenate([frames[j] for j in idx], axis=1)
        writer.add_image(f"eval/rollout_env{i}", strip, step, dataformats="HWC")
    if writer is not None:
      writer.add_scalar(
          "eval/video_episode_reward", traj["reward"].sum(axis=1).mean(), step
      )
      writer.flush()


def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv

  if _WARP_KERNEL_CACHE_DIR.value is not None:
    import warp as wp  # pylint: disable=g-import-not-at-top

    wp.config.kernel_cache_dir = _WARP_KERNEL_CACHE_DIR.value

  # Load environment configuration
  env_cfg = registry.get_default_config(_ENV_NAME.value)

  ppo_params = get_rl_config(_ENV_NAME.value)

  if _NUM_TIMESTEPS.present:
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
    ppo_params.num_timesteps = 0
  if _NUM_EVALS.present:
    ppo_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    ppo_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    ppo_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    ppo_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present:
    ppo_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    ppo_params.learning_rate = _LEARNING_RATE.value
  if _ENTROPY_COST.present:
    ppo_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present:
    ppo_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    ppo_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present:
    ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present:
    ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value

  env_cfg_overrides = {"impl": _IMPL.value}
  if _VISION.value:
    env_cfg_overrides["vision"] = True
    env_cfg_overrides["vision_config.nworld"] = ppo_params.num_envs
  if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
    env_cfg_overrides.update(json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value))

  env = registry.load(
      _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
  )
  if _RUN_EVALS.present:
    ppo_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present:
    ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present:
    ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

  print(f"Environment Config:\n{env_cfg}")
  if env_cfg_overrides:
    print(f"Environment Config Overrides:\n{env_cfg_overrides}\n")
  print(f"PPO Training Parameters:\n{ppo_params}")

  # Generate unique experiment name
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Set up logging directory
  logdir = epath.Path(_LOGDIR.value or "logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Initialize Weights & Biases if required
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    if wandb is None:
      raise ImportError(
          "wandb is required for --use_wandb. Install via: pip install wandb"
      )
    wandb.init(project="mjxrl", name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})

  # Initialize TensorBoard if required
  writer = None
  if _USE_TB.value and not _PLAY_ONLY.value and tensorboardX is not None:
    writer = tensorboardX.SummaryWriter(logdir)

  # Handle checkpoint loading
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
      latest_ckpts.sort(key=lambda x: int(x.name))
      latest_ckpt = latest_ckpts[-1]
      restore_checkpoint_path = latest_ckpt
      print(f"Restoring from: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  network_fn = (
      ppo_networks_vision.make_ppo_networks_vision
      if _VISION.value
      else ppo_networks.make_ppo_networks
  )
  if hasattr(ppo_params, "network_factory"):
    network_factory = functools.partial(
        network_fn, **ppo_params.network_factory
    )
  else:
    network_factory = network_fn

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  num_eval_envs = ppo_params.get("num_eval_envs", 128)

  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]

  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      restore_checkpoint_path=restore_checkpoint_path,
      save_checkpoint_path=ckpt_path,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
      vision=_VISION.value,
  )

  times = [time.monotonic()]

  # Progress function for logging
  def progress(num_steps, metrics):
    times.append(time.monotonic())

    # Log to Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics, step=num_steps)

    # Log to TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value and writer is not None:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    if _RUN_EVALS.value:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
    if _LOG_TRAINING_METRICS.value:
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )

  eval_env_overrides = scale_contact_buffers(
      env_cfg_overrides, _ENV_NAME.value, ppo_params.num_envs, num_eval_envs
  )
  if _VISION.value:
    eval_env_overrides["vision_config.nworld"] = num_eval_envs
  eval_env = registry.load(
      _ENV_NAME.value,
      config=registry.get_default_config(_ENV_NAME.value),
      config_overrides=eval_env_overrides,
  )

  eval_renderer = None
  if _EVAL_VIDEO_ENVS.value > 0:
    eval_renderer = RolloutRenderer(
        _ENV_NAME.value,
        env_cfg_overrides,
        ppo_params,
        _EVAL_VIDEO_ENVS.value,
        _VISION.value,
        _SEED.value,
        camera=_RENDER_CAMERA.value,
    )

  rscope_handle = None
  if _RSCOPE_ENVS.value:
    # Interactive visualisation of policy checkpoints
    from rscope import brax as rscope_utils

    if not _VISION.value:
      rscope_env = registry.load(
          _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
      )
      rscope_env = wrapper.wrap_for_brax_training(
          rscope_env,
          episode_length=ppo_params.episode_length,
          action_repeat=ppo_params.action_repeat,
          randomization_fn=training_params.get("randomization_fn"),
      )
    else:
      rscope_env = env

    rscope_handle = rscope_utils.BraxRolloutSaver(
        rscope_env,
        ppo_params,
        _VISION.value,
        _RSCOPE_ENVS.value,
        _DETERMINISTIC_RSCOPE.value,
        jax.random.PRNGKey(_SEED.value),
        rscope_fn,
    )

  def policy_params_fn(current_step, make_policy, params):
    if rscope_handle is not None:
      rscope_handle.set_make_policy(make_policy)
      # rscope_handle.dump_rollout(params) # Disabled to prevent rendering slice crash
    if eval_renderer is not None:
      eval_renderer.render(
          make_policy,
          params,
          logdir / "eval_videos",
          f"step_{current_step:012d}_env",
          writer=writer,
          step=current_step,
      )

  # Train or load the model
  make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
      environment=env,
      progress_fn=progress,
      policy_params_fn=policy_params_fn,
      eval_env=eval_env,
  )

  print("Done training.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

  print("Starting inference...")

  # Run evaluation rollouts matching how training handles batched environments.
  if eval_renderer is not None and _EVAL_VIDEO_ENVS.value == _NUM_VIDEOS.value:
    final_renderer = eval_renderer
  else:
    final_renderer = RolloutRenderer(
        _ENV_NAME.value,
        env_cfg_overrides,
        ppo_params,
        _NUM_VIDEOS.value,
        _VISION.value,
        _SEED.value,
        camera=_RENDER_CAMERA.value,
    )
  print(f"FPS for rendering: {final_renderer.fps}")
  final_renderer.render(
      make_inference_fn, params, logdir, "rollout", height=480, width=640
  )


def run():
  """Entry point for uv/pip script."""
  app.run(main)


if __name__ == "__main__":
  run()
