# RL100 + QAM Integration Log

## Goal

This integration keeps the original RL100 pipeline intact while adding a second offline-RL backend based on QAM and flow matching.

The resulting framework now supports:

- Original RL100 path:
  - diffusion policy
  - IQL critics
  - PPO offline/online optimization
  - transition model / AM-Q / consistency model
- New QAM path:
  - flow-matching policy on top of the same UNet backbone
  - QAM critic ensemble with pessimistic target backup
  - offline QAM actor optimization with adjoint matching
  - RGB-oriented DP-style visual encoder option

## Core Changes

### 1. Policy generation path is now configurable

File:

- `diffusion_policy_3d/policy/dp3.py`

Added `policy.generative_mode`:

- `diffusion`
  - original behavior
  - original diffusion BC loss
  - original DDIM sampling path
- `flow_matching`
  - flow-matching BC loss: interpolate between Gaussian noise and target trajectory
  - ODE-style rollout for inference
  - QAM SDE rollout helper for offline RL actor training

This means IL and IL-retrain automatically become flow-matching IL whenever `generative_mode=flow_matching`.

### 2. Offline RL backend is now configurable

File:

- `diffusion_policy_3d/rl100_trainer.py`

Added `training.offline_policy_backend`:

- `ppo`
  - original RL100 behavior
  - transition model + IQL + PPO + optional online RL
- `qam`
  - QAM critic training
  - QAM actor update with:
    - flow BC anchor
    - adjoint matching loss
  - keeps the existing RL100 outer loop:
    - train
    - rollout
    - merge
    - IL-retrain

Important behavior:

- `ppo` requires `policy.generative_mode=diffusion`
- `qam` requires `policy.generative_mode=flow_matching`
- `qam` currently supports only the offline stage
- online RL stays on the original RL100/PPO path only

### 3. Added QAM critic ensemble

File:

- `diffusion_policy_3d/model/rl/qam_critics.py`

Implemented:

- ensemble Q critics
- EMA target ensemble
- pessimistic target:
  - `Q_mean - rho * Q_std`

This follows the QAM paper / repo direction instead of reusing IQL.

### 4. Added DP-style RGB encoder option

Files:

- `diffusion_policy_3d/model/vision/dp_image_encoder.py`
- `diffusion_policy_3d/model/vision/multimodal_obs_encoder.py`

The old RGB path used only a shallow `SimpleRGBEncoder`.

Now `MultiModalObsEncoder` supports:

- `rgb_encoder_type: simple`
  - old lightweight conv encoder
- `rgb_encoder_type: dp_resnet`
  - new ResNet-style encoder inspired by Diffusion Policy’s image stack
  - configurable crop
  - configurable stage widths
  - GroupNorm path to better match DP-style visual encoders

This is dependency-free inside the current repo:

- no new `torchvision`
- no new `robomimic`

so it can be used directly in the existing workspace.

## Config Changes

Main config:

- `diffusion_policy_3d/config/rl100.yaml`

New knobs:

- `policy.generative_mode`
- `policy.flow_sampling_steps`
- `policy.rgb_encoder_type`
- `policy.rgb_encoder_cfg.*`
- `training.offline_policy_backend`
- `qam.*`

Added preset:

- `diffusion_policy_3d/config/rl100_qam_rgb.yaml`

This preset switches to:

- `task=realrobot_dualcam13dof`
- `offline_policy_backend=qam`
- `generative_mode=flow_matching`
- `rgb_encoder_type=dp_resnet`
- `run_online_rl=false`

## Recommended Usage

### Original RL100 path

```bash
python train_rl100.py --config-name rl100
```

### QAM + RGB real-robot-oriented path

```bash
python train_rl100.py --config-name rl100_qam_rgb
```

### Manual override example

```bash
python train_rl100.py \
  task=realrobot_dualcam13dof \
  training.offline_policy_backend=qam \
  policy.generative_mode=flow_matching \
  policy.rgb_encoder_type=dp_resnet \
  training.run_online_rl=false
```

## Offline Loop Behavior Under QAM

For each offline iteration:

1. train QAM critics
2. optimize flow-matching policy with QAM adjoint matching
3. rollout new data
4. merge new data into replay
5. re-run IL on the merged dataset using flow-matching BC

This preserves the RL100 high-level structure while swapping the policy-learning backend.

## Current Boundaries

The following parts are intentionally left unchanged or limited:

- online RL is still RL100/PPO-only
- consistency model / CM runtime path is still tied to the PPO/diffusion branch
- AM-Q / transition-model path remains part of the original RL100 backend
- QAM backend currently assumes:
  - `obs_as_global_cond=true`
  - film-style conditioning

These are deliberate guardrails to keep the original RL100 functionality intact while adding the new QAM path in parallel.

## Verification Performed

Static verification completed:

- Python syntax check via `python -m py_compile` on all modified files

Runtime import smoke-check status in the available conda environment:

- the new files themselves were syntactically loadable
- full project import is still blocked by existing environment dependency gaps:
  - `termcolor`
  - `einops`
  - `wandb`

Those missing packages are pre-existing runtime requirements of this project and are not introduced by this patch.

## Real-Robot UR5e Adapter

- Added `diffusion_policy_3d/env/real_robot/ur5e_inspire_dualcam_env.py`
  - Adapts the original `/home/yrz/dex-data-collection` UR5e + Inspire hand + dual-camera stack to RL100 `reset()/step()` execution.
  - Keeps the existing `RealRobotRunner` contract unchanged.
  - Supports configurable camera routing so `rgbm` and `right_cam_img` can be mapped from `orbbec_femto_bolt` / `l515`.
  - Reuses the original replay idea of interpolating high-level 20 Hz actions into 100 Hz low-level hardware commands.
- Added `rl100_qam_pick_place_ur5e.yaml`
  - Enables the real-hardware env through Hydra without changing the pure-offline config.

### Important Assumptions

- The dataset action/state layout is treated as:
  - first 6 dims: UR5e joint positions, normalized from `[-2π, 2π]` to `[-1, 1]`
  - last 6 dims: Inspire hand control values, normalized from `[0, 2000]` to `[-1, 1]`
- The current real-hardware preset assumes the hand values stored in zarr were linearly normalized from `[0, 2000]` to `[-1, 1]` via:
  - `normalized = raw / 1000 - 1`
  - `raw = normalized * 1000 + 1000`
- `rgbm[..., 3]` is currently synthesized online as a configurable constant mask (`zeros` by default), because the teleop collection stack only exposes raw RGB cameras at runtime. If an online segmentation pipeline is later available, this channel should be replaced with the real mask.

## Raw dex-data-collection Processing Path

- Added `scripts/convert_dex_data_collection_pkl_to_zarr.py`
  - Converts the original `dex-data-collection` per-episode `.pkl` outputs directly to RL100/QAM zarr.
  - Uses a streaming two-pass implementation:
    - first pass scans episode lengths / shapes
    - second pass preprocesses each episode and writes directly to zarr
  - Writes training-friendly time-only chunks from the start, so no extra rechunk step is required.
- Added `diffusion_policy_3d/config/task/realrobot_pick_place_dexcollector.yaml`
  - New task config aligned to the processed output of the raw collector data.
  - Exposes the important deployment knobs directly in config / env vars:
    - zarr path
    - UR5e IP
    - Inspire hand serial port
    - head / wrist camera routing
    - head mask mode
    - hand encoding
- Added `diffusion_policy_3d/config/rl100_qam_pick_place_dexcollector.yaml`
  - End-to-end training preset for:
    - IL on processed collector data
    - offline RL with QAM
    - real-robot rollout using the same UR5e/Inspire dual-camera env

### Processing Decisions Kept On Purpose

- Keep hand normalization by default:
  - raw collector hand values are `0~2000`
  - processed zarr stores them as `[-1, 1]`
  - the real-robot env decodes them back before execution
- Keep a constant 4th `rgbm` channel by default:
  - raw collector data only has RGB cameras
  - using a constant mask keeps the offline zarr shape aligned with the online env shape
  - this is intentional for train / rollout consistency, not for segmentation quality
- Use `action[t] = state[t+1]` by default:
  - this matches the observed behavior of the already-processed dataset that was inspected earlier

## Processed-Zarr Pilot Path

Because the currently available dataset is the already-processed zarr rather than the raw collector output, the integration now also includes a "pilot" path that does not require:

- new teleop collection
- raw `.pkl` data
- a live rollout env during offline RL

Added:

- `scripts/preprocess_real_robot_zarr.py`
  - takes the existing processed zarr
  - resizes `rgbm` / `right_cam_img`
  - rewrites time-only chunks
  - optionally writes sparse terminal `reward/done`
- `diffusion_policy_3d/config/task/realrobot_pick_place_processed.yaml`
  - training-only task config for the processed zarr
  - uses the real-robot dataset format but sets `env_runner: null`
  - reads the replay directly from zarr instead of copying it fully into RAM
- `diffusion_policy_3d/config/rl100_qam_pick_place_processed.yaml`
  - QAM + flow-matching preset for the fixed processed dataset
  - disables offline data collection explicitly

### Interface Changes For Fixed-Dataset Offline RL

Added `training.offline_collect_new_data` to `rl100.yaml`.

Behavior:

- `true`
  - original RL100 loop behavior
  - offline iteration includes rollout / merge / optional IL retrain
- `false`
  - offline iteration keeps critic training + actor optimization
  - skips rollout / merge / collection-triggered IL retrain

This keeps the original framework intact while allowing a first end-to-end run on the already-processed zarr alone.

## Offline RL Real-Robot Alignment Update

The current IL checkpoint was trained with:

- `use_mask=false`
- `use_right_cam_img=true`
- 84x84 image inputs
- flow-matching policy with QAM-compatible config

To make the subsequent offline RL rollout match that checkpoint and the mentor-processed zarr format, the real-robot path was updated:

- `UR5eInspireDualCamEnv` now supports normalized joint actions/states.
  - `arm_action_mode=joint`
  - `arm_state_mode=joint`
  - `arm_encoding=normalized_minus1_1`
  - `arm_norm_scale=2π`
  - policy action first 6 dims are decoded from `[-1, 1]` to raw UR5e joint radians before `servoJ`
  - observed UR5e joints are encoded back to `[-1, 1]` before being written as `right_state`
- `RealRobotRunner` now stores rollout episodes using zarr storage keys, not just active policy obs keys.
  - policy may consume `head_rgb = rgbm[..., :3]`
  - replay still writes `rgbm`
  - this prevents mixed `head_rgb/rgbm` keys after merge
- `RealRobotDataset.merge_episodes()` now appends rollout data by storage key and preserves existing zarr arrays.
  - existing `rgbm/right_cam_img/right_state/action` arrays are extended together
  - `reward/done` are appended for RL training
  - if only `head_rgb` is available, a zero mask channel can be synthesized to recover `rgbm`
- Manual sparse reward prompts now accept:
  - `c` / Enter: continue
  - `1` or `s`: success, terminate, terminal reward = 1
  - `0` or `f`: failure, terminate, terminal reward = 0
- Added `rl100_qam_pick_place_offline_ur5e.yaml`.
  - Starts from an IL checkpoint
  - Keeps QAM + flow matching
  - Enables real-robot rollout / merge / IL retrain
  - Defaults to the current processed-zarr obs layout: `use_mask=false`, `use_right_cam_img=true`
