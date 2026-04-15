# RL-100 on 3D-Diffusion-Policy

本仓库是在 [3D Diffusion Policy (DP3)](https://3d-diffusion-policy.github.io) 基础上实现并整理的 **RL-100** 版本，包含：

- DP3 行为克隆训练与评测
- RL-100 的 `IL -> Offline RL -> Online RL` 三阶段训练
- MetaWorld / Adroit / DexArt 演示数据采集脚本
- DDIM 主策略与 Consistency Model 的评测入口

论文：

- DP3: <https://arxiv.org/abs/2403.03954>
- RL-100: <https://arxiv.org/abs/2510.14830>

<div align="center">
  <img src="rl100.png" alt="RL-100 framework" width="100%">
</div>

## 仓库结构

核心代码位于 [3D-Diffusion-Policy](/home/yrz/RL-100/3D-Diffusion-Policy)：

- [train_rl100.py](/home/yrz/RL-100/3D-Diffusion-Policy/train_rl100.py)：RL-100 训练入口
- [eval_rl100.py](/home/yrz/RL-100/3D-Diffusion-Policy/eval_rl100.py)：RL-100 单 checkpoint 评测入口
- [rl100.yaml](/home/yrz/RL-100/3D-Diffusion-Policy/diffusion_policy_3d/config/rl100.yaml)：RL-100 主配置
- [config/task](/home/yrz/RL-100/3D-Diffusion-Policy/diffusion_policy_3d/config/task)：各任务配置
- [scripts](/home/yrz/RL-100/scripts)：数据采集、DP3 训练与评测脚本

## 环境安装

环境配置 **直接沿用 DP3**，这里没有额外改动。

- 安装说明见 [INSTALL.md](/home/yrz/RL-100/INSTALL.md)
- 常见报错见 [ERROR_CATCH.md](/home/yrz/RL-100/ERROR_CATCH.md)

如果你已经能正常跑 DP3，就可以直接跑 RL-100。

## 数据采集

所有演示数据默认写入 [3D-Diffusion-Policy/data](/home/yrz/RL-100/3D-Diffusion-Policy/data)。

### MetaWorld

脚本： [gen_demonstration_metaworld.sh](/home/yrz/RL-100/scripts/gen_demonstration_metaworld.sh)

```bash
bash scripts/gen_demonstration_metaworld.sh dial-turn
bash scripts/gen_demonstration_metaworld.sh basketball sparse
bash scripts/gen_demonstration_metaworld.sh push dense
```

说明：

- 第一个参数是 MetaWorld 任务名
- 第二个参数是奖励类型，默认 `sparse`
- 当前脚本固定采集 `100` 个 episode

### Adroit

脚本： [gen_demonstration_adroit.sh](/home/yrz/RL-100/scripts/gen_demonstration_adroit.sh)

```bash
bash scripts/gen_demonstration_adroit.sh door
bash scripts/gen_demonstration_adroit.sh hammer
bash scripts/gen_demonstration_adroit.sh pen
```

说明：

- 当前脚本固定采集 `10` 个 episode
- 依赖 `third_party/VRL3/ckpts/` 下的 expert checkpoint

### DexArt

脚本： [gen_demonstration_dexart.sh](/home/yrz/RL-100/scripts/gen_demonstration_dexart.sh)

```bash
bash scripts/gen_demonstration_dexart.sh laptop
bash scripts/gen_demonstration_dexart.sh faucet
bash scripts/gen_demonstration_dexart.sh bucket
bash scripts/gen_demonstration_dexart.sh toilet
```

说明：

- 当前脚本固定采集 `100` 个 episode
- 依赖 `third_party/dexart-release/assets/rl_checkpoints/`

### 真机 HDF5 转 zarr

如果你的真机演示数据是：

- `data/action`: `(K, 13)`
- `data/right_state`: `(K, 13)`
- `data/rgbm`: `(K, H, W, 4)`
- `data/right_cam_img`: `(K, H, W, 3)`
- `meta/episode_ends`: `(J,)`

可以直接转换成 RL-100 训练用 zarr：

```bash
python scripts/convert_real_robot_hdf5_to_zarr.py \
  --input /path/to/real_robot_demo.hdf5 \
  --output 3D-Diffusion-Policy/data/realrobot_dualcam13dof.zarr \
  --overwrite
```

默认会把 `rgbm` 和 `right_cam_img` resize 到 `84x84`，并把 RGB 归一化到 `[0, 1]`。  
如果你想保留原分辨率，可以把 `--resize-height` 和 `--resize-width` 设成 `0`。

### dex-data-collection 原生 `.pkl` 转 zarr

如果你的原始数据直接来自 `/home/yrz/dex-data-collection` 的采集输出，也就是每个 episode 一个 `.pkl`，包含：

- `episode_ur5e_pos_eef` 或 `episode_ur5e_pos_j`
- `episode_inspire_hand_pos`
- `episode_l515_color`
- `episode_orbbec_femto_bolt_color`

可以直接转换成 RL100/QAM 可训练的 zarr：

```bash
python scripts/convert_dex_data_collection_pkl_to_zarr.py \
  --input-dir /path/to/raw_dexcollector_pkls \
  --output /path/to/realrobot_pick_place_dexcollector.zarr \
  --arm-source eef \
  --action-source next_state \
  --head-camera-key episode_orbbec_femto_bolt_color \
  --wrist-camera-key episode_l515_color \
  --hand-encoding normalized_minus1_1 \
  --mask-mode zeros \
  --resize-height 224 \
  --resize-width 224 \
  --overwrite
```

这条脚本默认做了几件事：

- `right_state[t] = [ur5e_state(t), inspire_hand_state(t)]`
- `action[t] = right_state[t+1]`
  - 最后一帧会重复最后一个状态
- 头相机写成 `rgbm`
  - 前 `3` 通道是归一化后的图像
  - 第 `4` 通道是常量 mask
- 腕部相机写成 `right_cam_img`
- hand 的后 `6` 维默认从原始 `0~2000` 映射到 `[-1, 1]`
- zarr 直接按“只沿时间维 chunk”写出，不需要再单独 rechunk

### 只有现成 processed zarr 时

如果你现在手里只有师兄已经处理好的 zarr，例如：

- `data/action`
- `data/right_state`
- `data/rgbm`
- `data/right_cam_img`
- `meta/episode_ends`

可以先再做一层“训练友好版”预处理：

```bash
python scripts/preprocess_real_robot_zarr.py \
  --input /path/to/dex-data.zarr \
  --output 3D-Diffusion-Policy/data/dex-data-train84.zarr \
  --resize-height 84 \
  --resize-width 84 \
  --process-batch-size 32 \
  --overwrite
```

如果你希望预处理阶段就把 sparse terminal `reward/done` 也写进去，可以加：

```bash
python scripts/preprocess_real_robot_zarr.py \
  --input /path/to/dex-data.zarr \
  --output 3D-Diffusion-Policy/data/dex-data-train84.zarr \
  --resize-height 84 \
  --resize-width 84 \
  --write-terminal-labels \
  --overwrite
```

推荐先缩到 `84x84` 再跑，原因很直接：

- 当前 `RealRobotDataset` 默认历史实现会把 replay 复制到内存
- 这份数据有 `79799` 帧、双相机
- 即便只是 RGB `uint8`，`224x224` 也会非常大
- 现在仓库已经给 `RealRobotDataset` 加了 `zarr` 直读后端，但第一版建议仍然用 `84x84`，更稳

对应的固定数据集训练 preset：

- [realrobot_pick_place_processed.yaml](/home/yrz/RL-100/3D-Diffusion-Policy/diffusion_policy_3d/config/task/realrobot_pick_place_processed.yaml)
- [rl100_qam_pick_place_processed.yaml](/home/yrz/RL-100/3D-Diffusion-Policy/diffusion_policy_3d/config/rl100_qam_pick_place_processed.yaml)

只跑 IL：

```bash
cd 3D-Diffusion-Policy

python train_rl100.py \
  --config-name rl100_qam_pick_place_processed \
  training.num_offline_iterations=0 \
  training.run_online_rl=false
```

跑一版固定数据集的 QAM offline RL：

```bash
cd 3D-Diffusion-Policy

python train_rl100.py --config-name rl100_qam_pick_place_processed
```

默认数据路径已经写在 [realrobot_pick_place_processed.yaml](/home/yrz/RL-100/3D-Diffusion-Policy/diffusion_policy_3d/config/task/realrobot_pick_place_processed.yaml)：

```yaml
dataset_zarr_path: data/dex-data-train84.zarr
use_mask: false
use_right_cam_img: true
```

如果你想临时换数据，不需要改文件，也可以命令行覆盖：

```bash
python train_rl100.py \
  --config-name rl100_qam_pick_place_processed \
  task.dataset_zarr_path=/path/to/other.zarr
```

观测输入用两个接口切换：

```bash
python train_rl100.py \
  --config-name rl100_qam_pick_place_processed \
  task.use_mask=false \
  task.use_right_cam_img=false
```

这两个开关的含义：

- `use_mask=false, use_right_cam_img=false`：`right_state + rgbm[..., :3]`，默认推荐第一版使用
- `use_mask=true, use_right_cam_img=false`：`right_state + rgbm`，使用第三视角 RGB+mask
- `use_mask=false, use_right_cam_img=true`：`right_state + rgbm[..., :3] + right_cam_img`
- `use_mask=true, use_right_cam_img=true`：`right_state + rgbm + right_cam_img`，恢复之前最重的双相机+mask输入

`use_mask=false` 不要求 zarr 里真的有 `data/head_rgb`。当前 dataset 会从现有 `data/rgbm` 自动取前三通道生成。

如果确实需要手动指定 obs key，也保留了高级覆盖：

```bash
python train_rl100.py \
  --config-name rl100_qam_pick_place_processed \
  task.obs_mode=head_rgb_right_cam
```

这个 preset 的特点是：

- `training.offline_collect_new_data=false`
- `collection_episodes=0`
- `task.env_runner=null`
- offline RL 只在固定 zarr 上训练 critics + QAM actor
- 不做 rollout
- 不做 merge
- 不做 collection 后的 IL retrain

## DP3 基线训练与评测

如果你只想跑原始 DP3 行为克隆流程，可以继续用原脚本。

### 训练

脚本： [train_policy.sh](/home/yrz/RL-100/scripts/train_policy.sh)

```bash
bash scripts/train_policy.sh dp3 metaworld_dial-turn exp1 0 0
bash scripts/train_policy.sh dp3 adroit_hammer exp1 0 0
bash scripts/train_policy.sh simple_dp3 dexart_laptop exp1 0 0
```

参数顺序：

1. 算法名：`dp3` 或 `simple_dp3`
2. 任务名：例如 `metaworld_dial-turn`
3. 附加字符串：用于组成实验名
4. 随机种子
5. GPU id

### 评测

脚本： [eval_policy.sh](/home/yrz/RL-100/scripts/eval_policy.sh)

```bash
bash scripts/eval_policy.sh dp3 metaworld_dial-turn exp1 0 0
```

## RL-100 训练

RL-100 不走 shell 脚本，直接用 Hydra 入口。

先进入项目目录：

```bash
cd 3D-Diffusion-Policy
```

### 基本训练

```bash
python train_rl100.py task=metaworld_dial-turn
```

### 真机任务配置

仓库新增了一个真机模板任务：

- [realrobot_dualcam13dof.yaml](/home/yrz/RL-100/3D-Diffusion-Policy/diffusion_policy_3d/config/task/realrobot_dualcam13dof.yaml)

它默认使用：

- `right_state` 作为低维 proprio 输入
- `rgbm` 作为头相机 `4` 通道输入
- `right_cam_img` 作为腕部相机 `3` 通道输入
- `action` 维度 `13`
- `task.execution.enable_eval=false`，训练过程中不做真机 eval
- `task.execution.enable_amq=false`，offline RL 跳过 AM-Q / OPE gate
- `task.execution.enable_cm_policy=false`，禁用 consistency-model runtime policy
- `task.execution.stop_env_on_keyboard_interrupt=true`，`Ctrl+C` 时对 env 做 best-effort stop

如果只想先验证离线 IL，可直接关掉 rollout 阶段：

```bash
cd 3D-Diffusion-Policy

python train_rl100.py \
  task=realrobot_dualcam13dof \
  training.num_offline_iterations=0 \
  training.run_online_rl=false
```

如果要跑真机 offline-collection / online RL，需要在配置里给 `task.env_runner.env` 提供真实机器人环境：

```bash
python train_rl100.py \
  task=realrobot_dualcam13dof \
  task.env_runner.env._target_=your_robot_pkg.envs.YourRealRobotEnv
```

如果你之后想显式打开真机评测或 `cm`，可以在命令行覆盖：

```bash
python train_rl100.py \
  task=realrobot_dualcam13dof \
  task.execution.enable_eval=true \
  task.execution.enable_cm_policy=true
```

这个 env 需要至少提供：

- `reset() -> obs_dict` 或 `(obs_dict, info)`
- `step(action) -> obs_dict`
- `step(action) -> (obs_dict, info)`
- `step(action) -> (obs_dict, reward, done, info)`
- `step(action) -> (obs_dict, reward, terminated, truncated, info)`

其中 `obs_dict` 的 key 需要和 task 配置一致，例如：

- `right_state`
- `rgbm`
- `right_cam_img`

### dex-data-collection 真机训练预设

如果你后面直接把项目拷到连着 UR5e 的电脑上，推荐用这套配置：

- [realrobot_pick_place_dexcollector.yaml](/home/yrz/RL-100/3D-Diffusion-Policy/diffusion_policy_3d/config/task/realrobot_pick_place_dexcollector.yaml)
- [rl100_qam_pick_place_dexcollector.yaml](/home/yrz/RL-100/3D-Diffusion-Policy/diffusion_policy_3d/config/rl100_qam_pick_place_dexcollector.yaml)
- [rl100_qam_pick_place_offline_ur5e.yaml](/home/yrz/RL-100/3D-Diffusion-Policy/diffusion_policy_3d/config/rl100_qam_pick_place_offline_ur5e.yaml)

这套配置默认假定：

- 第一版 offline RL 可以直接从已经处理好的 `data/dex-data-train84.zarr` 启动
- `action/right_state` 都是 `12` 维
- 前 `6` 维是 UR5e joint，按 `[-2π, 2π] -> [-1, 1]` 归一化
- 后 `6` 维是 Inspire hand，按 `[0, 2000] -> [-1, 1]` 归一化
- rollout 真机执行也用 `UR5e + Inspire + Orbbec + L515`
- 默认观测输入和 processed 版本保持一致：
  - `task.use_mask=false`
  - `task.use_right_cam_img=true`
  - 也就是 `right_state + rgbm[..., :3] + right_cam_img`

真机执行时 env 会把 policy 输出的前 6 维 normalized joint 反解成 UR5e 弧度 joint，再通过 RTDE 下发；采集到的新 rollout 仍按 `rgbm/right_cam_img/right_state/action/reward/done` 写回 zarr，和师兄处理后的数据格式对齐。

如果要在 dexcollector 真机路径里打开 mask 或腕部相机，也用同样两个接口：

```bash
python train_rl100.py \
  --config-name rl100_qam_pick_place_dexcollector \
  task.use_mask=true \
  task.use_right_cam_img=true
```

注意：当前 raw `.pkl` 转换脚本默认只能写常量 mask。除非你同时接入师兄的 `segmask` 生成结果，否则不建议打开 `task.use_mask=true`。

关键路径都暴露成了环境变量，迁到机械臂电脑后直接改：

```bash
export REALROBOT_ZARR_PATH=/path/to/realrobot_pick_place_dexcollector.zarr
export UR5E_IP=192.168.1.109
export INSPIRE_HAND_PORT=/dev/ttyUSB0
export REALROBOT_HEAD_CAMERA=orbbec_femto_bolt
export REALROBOT_WRIST_CAMERA=l515
export REALROBOT_HEAD_MASK_MODE=zeros
export REALROBOT_ARM_ACTION_MODE=joint
export REALROBOT_ARM_STATE_MODE=joint
export REALROBOT_ARM_ENCODING=normalized_minus1_1
export REALROBOT_HAND_ENCODING=normalized_minus1_1
```

只跑 IL：

```bash
cd 3D-Diffusion-Policy

python train_rl100.py \
  --config-name rl100_qam_pick_place_dexcollector \
  training.num_offline_iterations=0 \
  training.run_online_rl=false
```

跑 IL + offline RL + 真机 rollout：

```bash
cd 3D-Diffusion-Policy

python train_rl100.py \
  --config-name rl100_qam_pick_place_offline_ur5e \
  training.resume_path=/path/to/your_il.ckpt \
  task.dataset_zarr_path=data/dex-data-train84.zarr \
  training.num_offline_iterations=1 \
  training.collection_episodes=1
```

这个 preset 默认比较保守：

- `num_offline_iterations=1`
- `collection_episodes=1`
- `eval_episodes=1`
- `runtime.collection_policy=flow`
- `training.resume_load_rl_state=false`，即从 IL policy 开始，critics/QAM optimizer 重新初始化

避免第一次上真机就沿用仿真里的大规模采样设置。

### 常见覆盖写法

```bash
python train_rl100.py \
  task=metaworld_dial-turn \
  training.seed=0 \
  training.device=cuda:0 \
  logging.use_wandb=true \
  task.env_runner.eval_episodes=100
```

### 指定从某个 checkpoint 恢复

```bash
python train_rl100.py \
  task=metaworld_dial-turn \
  training.resume=true \
  training.resume_path=/path/to/checkpoints/after_il.ckpt
```

### 训练阶段

`train_rl100.py` 默认执行以下流程：

1. `IL`：先用 demonstration 训练 DP3/RL100 policy
2. `Offline RL`：训练 transition model、IQL critics、offline PPO，并做 OPE gate
3. `Data Collection + IL Retrain`：收集新轨迹并并回数据集，再做 IL retrain
4. `Online RL`：对 fresh rollout 做 on-policy PPO + GAE
5. `Final Eval`：按配置评测 `ddim` 和/或 `cm`

如果你把：

- `training.offline_collect_new_data=false`
- `training.collection_episodes=0`

那第 `2` 阶段会变成“固定数据集 offline RL”：

- 仍然训练 critics
- 仍然训练 PPO 或 QAM actor
- 但跳过 rollout / merge / IL retrain

相关主配置见 [rl100.yaml](/home/yrz/RL-100/3D-Diffusion-Policy/diffusion_policy_3d/config/rl100.yaml)。

## RL-100 评测

脚本入口： [eval_rl100.py](/home/yrz/RL-100/3D-Diffusion-Policy/eval_rl100.py)

### 评测主模型 DDIM

```bash
python eval_rl100.py \
  task=metaworld_dial-turn \
  checkpoint_path=/path/to/checkpoints/final.ckpt \
  runtime.eval_policy_mode=ddim \
  runtime.eval_use_ema=false \
  task.env_runner.eval_episodes=100
```

### 评测 EMA-DDIM

```bash
python eval_rl100.py \
  task=metaworld_dial-turn \
  checkpoint_path=/path/to/checkpoints/final.ckpt \
  runtime.eval_policy_mode=ddim \
  runtime.eval_use_ema=true \
  task.env_runner.eval_episodes=100
```

### 评测 Consistency Model

```bash
python eval_rl100.py \
  task=metaworld_dial-turn \
  checkpoint_path=/path/to/checkpoints/final.ckpt \
  runtime.eval_policy_mode=cm \
  runtime.eval_use_ema=true \
  task.env_runner.eval_episodes=100
```

## 输出内容

训练输出目录由 Hydra 管理，默认在：

```bash
3D-Diffusion-Policy/outputs/rl100_<task>_seed<seed>/<date>_<time>/
```

其中通常包含：

- `checkpoints/after_il.ckpt`
- `checkpoints/offline_iter_<N>.ckpt`
- `checkpoints/online_iter_<N>.ckpt`
- `checkpoints/final.ckpt`
- `plots/` 下的各类 loss / success 曲线

## 重要配置项

常用项基本都在 [rl100.yaml](/home/yrz/RL-100/3D-Diffusion-Policy/diffusion_policy_3d/config/rl100.yaml)：

- `training.num_offline_iterations`
- `training.offline_collect_new_data`
- `training.critic_epochs`
- `training.ppo_epochs`
- `training.ppo_inner_steps`
- `training.collection_episodes`
- `training.online_rl_iterations`
- `training.online_collection_episodes`
- `training.rl_policy_lr`
- `runtime.collection_policy`
- `runtime.collection_use_ema`
- `runtime.il_retrain_success_only`
- `runtime.final_eval_policies`
- `runtime.final_eval_use_ema`
- `task.env_runner.eval_episodes`

任务数据路径、观测维度、评测 episode 数在各自 task yaml 里定义，例如：

- [metaworld_dial-turn.yaml](/home/yrz/RL-100/3D-Diffusion-Policy/diffusion_policy_3d/config/task/metaworld_dial-turn.yaml)

## 注意事项

### 1. `eval_episodes` 以 task yaml 为准

最终训练评测和 `eval_rl100.py` 都直接读取 `task.env_runner.eval_episodes`。  
如果要改评测轮数，改 task yaml 或在命令行覆盖：

```bash
python train_rl100.py task=metaworld_dial-turn task.env_runner.eval_episodes=100
```

### 2. `il_retrain_success_only` 只影响 IL retrain，不影响 RL 本身

当前逻辑是：

- offline RL / online RL 都使用完整采样轨迹，包括失败轨迹
- `success-only` 只作用于后续 `IL retrain` 的数据筛选

这和 RL 训练、IL 重训的语义已经拆开了。

### 3. `prediction_type` 必须是 `epsilon`

RL-100 的 PPO ratio 计算依赖 `epsilon` 参数化。当前配置已固定为：

```yaml
policy:
  noise_scheduler:
    prediction_type: epsilon
```

不要改成 `sample`。

### 4. `sigma_max` 的值

根据 RL-100 论文 `2510.14830 v4` 的消融结论，stochastic DDIM 的标准差上界需要按控制模式区分：

- `sigma_max = 0.8`
  - Adroit
  - Mujoco locomotion
  - 真机单步控制任务
- `sigma_max = 0.1`
  - MetaWorld
  - 真机 chunk-action 控制任务

仓库当前默认：

```yaml
policy:
  sigma_max: 0.1
```

这适合 MetaWorld / chunk-action。  
如果你做真机单步控制，应该按论文建议改到 `0.8`。

### 5. `reward_type=dense` 只能用于真的有 dense reward 标签的数据

当前 MetaWorld 演示脚本默认是：

```bash
bash scripts/gen_demonstration_metaworld.sh <task> sparse
```

如果你要跑 `dense`，需要保证：

- 采集脚本真的生成了 dense reward
- 配置里的 `critics.reward_type` 与数据一致

不要拿 sparse 数据去伪装 dense reward。

### 5.1 真机 sparse reward / episode 结束建议

RL-100 论文里的真机 rollout 本身就是“人工给 sparse success signal”；仓库里的真机 runner 也按这个思路实现了。默认真机模板配置是：

- `task.env_runner.reward_mode=terminal_sparse_manual`
- `task.env_runner.episode_end_mode=env_or_manual_or_max_steps`

也就是：

- 每个 episode 的中间步 reward 默认全是 `0`
- episode 结束时，如果人工标记成功，则终止步 reward=`1`
- 如果人工标记失败，则终止步 reward=`0`
- episode 可以由 env 自己结束、达到 `max_steps` 结束，或者每个 action chunk 后人工决定 `continue/success/failure`
- 当前终端输入兼容 `c` 继续、`1`/`s` 成功结束、`0`/`f` 失败结束

这更接近主流真机 RL 在“没有可靠 success classifier 时”的做法。  
如果你之后接入了自动 success classifier / 自动 reset 逻辑，可以切到：

- `task.env_runner.reward_mode=terminal_sparse_env_success`
- `task.env_runner.episode_end_mode=env_or_max_steps`

### 6. `use_recon_vib=true` 时要从头训

如果 checkpoint 不是用 Recon/VIB 训练出来的，不要直接在中途打开：

```yaml
policy:
  use_recon_vib: true
```

这会把随机初始化的 decoder/VIB 分支引进来，破坏已有 policy。  
如果要用，应该从 demonstration 开始重新训练。

### 7. EMA 与主模型不是一回事

- 训练时真正反向更新的是主模型 `policy`
- `ema_policy` 是主模型参数的指数滑动平均
- 最终评测是否用 EMA，取决于：
  - `runtime.final_eval_use_ema`
  - `runtime.eval_use_ema`

### 8. WandB 不是必须的

如果不想用 WandB，直接在配置里关掉：

```bash
python train_rl100.py task=metaworld_dial-turn logging.use_wandb=false
```

## 参考文档

- RL-100 代码说明： [RL100_README.md](/home/yrz/RL-100/3D-Diffusion-Policy/RL100_README.md)
- DP3 安装说明： [INSTALL.md](/home/yrz/RL-100/INSTALL.md)
- 安装踩坑记录： [ERROR_CATCH.md](/home/yrz/RL-100/ERROR_CATCH.md)

## Citation

如果这个仓库对你有帮助，可以引用：

```bibtex
@inproceedings{Ze2024DP3,
  title={3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations},
  author={Yanjie Ze and Gu Zhang and Kangning Zhang and Chenyuan Hu and Muhan Wang and Huazhe Xu},
  booktitle={Proceedings of Robotics: Science and Systems (RSS)},
  year={2024}
}

@article{lei2025rl100,
  title={RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning},
  author={Lei, Kun and Li, Huanyu and Yu, Dongjie and Wei, Zhenyu and Guo, Lingxiao and Jiang, Zhennan and Wang, Ziyu and Liang, Shiyu and Xu, Huazhe},
  journal={arXiv preprint arXiv:2510.14830},
  year={2025}
}
```
