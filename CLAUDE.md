# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MuJoCo Playground: GPU-accelerated RL environments (DM Control Suite, locomotion, manipulation, vision) built on MuJoCo MJX, supporting both the JAX (`impl="jax"`) and MuJoCo Warp (`impl="warp"`) backends. The PyPI package is named `playground`; the import name is `mujoco_playground`. Python >= 3.11 (CI tests 3.11 and 3.12).

## Commands

Setup from source (GPU), per README:
```sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple
uv --no-config sync --all-extras
```
CI (CPU only) instead runs `uv pip install --system -e ".[test]"`.

Tests (`*_test.py` under `mujoco_playground/_src/`):
```sh
# Clone menagerie once before running in parallel, as CI does, so xdist workers don't each try to clone it.
python -c "from mujoco_playground._src import mjx_env; mjx_env.ensure_menagerie_exists()"
pytest -n auto mujoco_playground/_src/                                    # full suite (CI)
pytest mujoco_playground/_src/locomotion/locomotion_test.py -k Go1Getup   # single env
pytest mujoco_playground/_src/wrapper_test.py                             # single file
```

Lint / format (pre-commit runs pyink, isort, and a copyright-header check):
```sh
pre-commit run --all-files
pylint . --rcfile=pylintrc   # not in pre-commit; run manually
pytype .
```

Training:
```sh
train-jax-ppo --env_name CartpoleBalance --impl warp                  # brax PPO: learning/train_jax_ppo.py
train-jax-ppo --env_name PandaPickCubeCartesian --impl warp --vision  # pixel observations
train-rsl-ppo --env_name LeapCubeReorient --impl warp                 # RSL-RL: learning/train_rsl_rl.py
```
Other useful flags: `--domain_randomization`, `--playground_config_overrides='{"dotted.key": value}'` (JSON), `--num_timesteps`, `--play_only`, `--use_tb` (TensorBoard scalars), `--eval_video_envs N` (render N rollouts at every evaluation into `eval_videos/` and, with `--use_tb`, frame strips into TensorBoard). Output goes to `logs/<env>-<timestamp>/` (checkpoints, `config.json`, rollout mp4s). A CLI flag overrides the tuned RL config only when it is passed explicitly.

`train_jax_ppo.py` sets `MUJOCO_GL`, `XLA_FLAGS` and `XLA_PYTHON_CLIENT_PREALLOCATE` before its other imports on purpose: mujoco >= 3.13 picks the GL backend at import time, so setting them later breaks the final rendering.

On Ampere and newer GPUs, `export JAX_DEFAULT_MATMUL_PRECISION=highest` avoids TF32-induced training instability and irreproducibility (README FAQ).

## Code style

- Google style: 2-space indentation, 80 columns (pyink with `pyink-indentation = 2` and majority quotes). isort uses `force_single_line`, so one import per line.
- Every Python file carries the `Copyright <year> Google LLC` Apache header, and pre-commit enforces it.
- The repo is synced from Google-internal code via Copybara. That is why `internal_paths` fallbacks and `# go/keep-sorted start/end` markers appear; keep those blocks sorted.

## Architecture

### Suites and registry
`mujoco_playground/_src/` holds three suites: `dm_control_suite`, `locomotion`, and `manipulation`. Each suite's `__init__.py` owns its registry dicts:
- `_envs`: env name → class, often a `functools.partial(Cls, task=...)` for variants such as flat or rough terrain
- `_cfgs`: env name → `default_config()` function
- `_randomizer` (locomotion and manipulation only): env name → `domain_randomize`

Each suite computes `ALL_ENVS` lazily through a module `__getattr__`, so `register_environment()` additions show up. `registry.ALL_ENVS` is instead a snapshot taken at import. `_src/registry.py` only dispatches `load`, `get_default_config`, and `get_domain_randomizer` to the owning suite.

To add an env: implement the class and `default_config()`, register both in the suite's `_envs` and `_cfgs` (plus `_randomizer` if needed), and add tuned hyperparameters in `mujoco_playground/config/<suite>_params.py`. The suite tests are parameterized over `ALL_ENVS`: each one does a jitted `reset` and `step` with `impl="jax"`, checks the obs shape against `observation_size`, and checks for NaNs. A registered env is therefore smoke-tested automatically.

### Environment contract (`_src/mjx_env.py`)
- `MjxEnv` subclasses implement `reset(rng) -> State`, `step(state, action) -> State`, and the properties `xml_path`, `action_size`, `mj_model`, and `mjx_model`. Callers jit and vmap `reset`/`step`.
- `State` is a flax `struct.dataclass` of `(data, obs, reward, done, metrics, info)`. `obs` is either an array or a dict. Dict keys such as `state` and `privileged_state` feed asymmetric actor-critic through `policy_obs_key` / `value_obs_key` in the RL config's `network_factory`. Vision obs use the key `pixels/view_0`.
- The config is an `ml_collections.ConfigDict` that `MjxEnv.__init__` **locks**. `config_overrides` is a flat dict with dotted keys, such as `{"impl": "jax", "vision_config.nworld": 1024}`, and can only change keys that already exist.
- A typical constructor:
  1. Builds an assets dict with `mjx_env.update_assets(...)` from the env's local `xmls/` plus `MENAGERIE_PATH / <robot>`.
  2. Loads the model with `MjModel.from_xml_string(...)`.
  3. Patches `mj_model` (timestep = `sim_dt`, PD gains, and so on).
  4. Calls `mjx_env.put_model(mj_model, impl=config.impl)`.

  `reset` builds data with `mjx_env.make_data(..., impl=..., naconmax=..., njmax=...)`. `step` calls `mjx_env.step(model, data, action, self.n_substeps)`, where `n_substeps = ctrl_dt / sim_dt`.
- Robot families are split into:
  - `<robot>/base.py`: model loading and sensor accessors
  - `<robot>_constants.py`: paths and geom/sensor names
  - task files such as `joystick.py` and `getup.py`
  - `randomize.py`

  Contacts are read from MuJoCo contact sensors (`*_found`) via `mjx_env.get_sensor_data`.

### Backends (`impl`)
Env configs carry `impl`, which has defaulted to `"warp"` since 0.2.0, plus the Warp buffer sizes `naconmax`, `naccdmax`, and `njmax`. The suite tests override `impl="jax"`. Both training scripts define an `--impl` flag that defaults to `jax` and overrides the env config. `put_model` deliberately masks Warp's solver and linesearch iteration-overflow warnings.

### Menagerie assets
Locomotion and manipulation XMLs include robot models from `mujoco_menagerie`. It is git-cloned on the first `locomotion.load` or `manipulation.load` into `mujoco_playground/external_deps/mujoco_menagerie`, pinned to `MENAGERIE_COMMIT_SHA` in `mjx_env.py`. The check only tests whether the directory exists, so bumping the SHA does not update an existing clone.

The reBot B601-DM envs (`manipulation/rebot_b601_dm/`) clone their meshes the same way from `Rebot_Arm_description` into `external_deps/`, pinned in `base.py`; `REBOT_ARM_DESCRIPTION_PATH` points at an existing checkout instead.

The i2rt YAM envs (`manipulation/i2rt_yam/`) subclass the reBot task classes (`base.YamMixin` swaps joint names and assets, `SCENE_XML` the scene) and read their meshes from the menagerie `i2rt_yam` model; scenes and keyframes mirror the reBot ones.

The A0B + Shadow Hand env (`manipulation/a0b_shadow_hand/`) reads meshes from a local checkout of the private `a0bsrh_model` package (`A0BSRH_MODEL_PATH`, default `~/robot/shadowhand_ws/a0bsrh_model`); its MJCF is a generated, MJX-adapted copy of that package's XML (see its README).

### MuJoCo Warp import
mujoco-mjx >= 3.13 vendors MuJoCo Warp as `mujoco.mjx.third_party.mujoco_warp` and no separate `mujoco_warp` package is installed. Use `mjx_env.import_mujoco_warp()` rather than `import mujoco_warp`; it tries the top-level package first and falls back to the vendored copy.

### Wrappers (`_src/wrapper.py`)
`wrap_for_brax_training` stacks these wrappers: `VmapWrapper` (or `BraxDomainRandomizationVmapWrapper`) → brax `EpisodeWrapper` → `BraxAutoResetWrapper` → optional `DeferredVisionWrapper`.
- **Auto-reset:** by default it restores a cached first state, resetting only `data` and `obs` and keeping `info`. With `full_reset=True` it calls `env.reset` instead. That path preserves `info["AutoResetWrapper_preserve_info"]`, which serves as a curriculum hook, and `info["AutoResetWrapper_done_count"]` counts completed episodes.
- **Domain randomizers:** they have the signature `domain_randomize(model, rng) -> (batched_model, in_axes)`. The DR wrapper vmaps by temporarily swapping `env.unwrapped._mjx_model`. Envs must therefore store their model as `self._mjx_model` and read it through that attribute (or `self.mjx_model`) inside `reset` and `step`.

### Vision
Vision envs set `vision=True`; currently these are the Cartpole family and `PandaPickCubeCartesian`. They create a Warp batch-render context with `mjx.create_render_context(mjm=..., **vision_config)`, where `vision_config.nworld` is the batch size. The train script sets `nworld` to `num_envs` for training and to `num_eval_envs` for eval. For envs that implement `defer_rendering()` and `render_state(state)` (see `pick_cartesian.py`), `DeferredVisionWrapper` renders after autoreset, so the cached reset state never holds pixels.

### Other directories
- `learning/`: the training entry points (the `train-jax-ppo` and `train-rsl-ppo` console scripts) and Colab notebooks; included in the wheel.
- `mujoco_playground/config/`: tuned RL hyperparameters (`brax_ppo_config`, `brax_vision_ppo_config`, `rsl_rl_config`), chosen per env name by if/elif chains that override a shared default.
- `mujoco_playground/experimental/`: notebooks, sim2sim ONNX playback scripts, and Madrona benchmarks; excluded from pylint and from the wheel.
- `mujoco_playground.wrapper_torch`: `__init__.py` imports it from an internal `mujoco_playground.rss2025` path that is not in this repo, so here it resolves to `None`. The real module is `_src/wrapper_torch.py`, but `learning/train_rsl_rl.py` imports it through the top-level name.
