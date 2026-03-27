Job start at 2026-03-25 17:23:40
Job run at:
   Static hostname: localhost.localdomain
Transient hostname: r8a30-a06
         Icon name: computer-server
           Chassis: server
        Machine ID: df5d662d31bf4cc3ab86556a4c775478
           Boot ID: 873af846a6f24b3da0e9ef5d75044627
  Operating System: Rocky Linux 8.7 (Green Obsidian)
       CPE OS Name: cpe:/o:rocky:rocky:8:GA
            Kernel: Linux 4.18.0-425.10.1.el8_7.x86_64
      Architecture: x86-64
Filesystem                                        Size  Used Avail Use% Mounted on
/dev/mapper/rl-root                               120G   29G   92G  24% /
/dev/sda2                                         2.0G  305M  1.7G  15% /boot
/dev/mapper/rl-local                              512G  172G  341G  34% /local
/dev/mapper/rl-var                                256G   23G  234G   9% /var
/dev/sda1                                         599M  5.8M  594M   1% /boot/efi
ssd.nas00.future.cn:/rocky8_home                   16G   15G  2.0G  88% /home
ssd.nas00.future.cn:/rocky8_workspace             400G     0  400G   0% /workspace
ssd.nas00.future.cn:/rocky8_tools                 5.0T   99G  5.0T   2% /tools
ssd.nas00.future.cn:/centos7_home                  16G  4.2G   12G  26% /centos7/home
ssd.nas00.future.cn:/centos7_workspace            400G     0  400G   0% /centos7/workspace
ssd.nas00.future.cn:/centos7_tools                5.0T  235G  4.8T   5% /centos7/tools
ssd.nas00.future.cn:/eda-tools                    8.0T  6.4T  1.7T  79% /centos7/eda-tools
hdd.nas00.future.cn:/share_personal               500G     0  500G   0% /share/personal
zone05.nas01.future.cn:/NAS_HPC_collab_codemodel   40T   37T  3.4T  92% /share/collab/codemodel
ext-zone00.nas02.future.cn:/nfs_global            408T  393T   16T  97% /nfs_global
ssd.nas00.future.cn:/common_datasets               75T   64T   12T  85% /datasets
192.168.12.10@o2ib:192.168.12.11@o2ib:/lustre     1.9P   12T  1.7P   1% /lustre
beegfs_nodev                                       70T   15T   56T  21% /fast
Currently Loaded Modulefiles: 1) cluster-tools/v1.0 3) cuda-cudnn/12.1-8.9.3 5) git/2.31.1 2) cmake/3.21.7 4) gcc/9.3.0 6) slurm-tools/v1.0
/tools/cluster-software/gcc/gcc-9.3.0/bin/gcc
/home/S/yangrongzheng/miniconda3/bin/python
/home/S/yangrongzheng/miniconda3/bin/python3
############### /home : /home/S/yangrongzheng
Disk quotas for user yangrongzheng (uid 6215): 
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
          /home  14339M  16384M  20480M            160k       0       0        

############### /workspace
Disk quotas for user yangrongzheng (uid 6215): 
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
     /workspace      0K    400G    500G               1       0       0        

############### /nfs_global
Disk quotas for user yangrongzheng (uid 6215): 
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
    /nfs_global    492G   5120G   7168G            354k   5000k  10000k        

############### /lustre
Disk quotas for usr yangrongzheng (uid 6215):
     Filesystem    used   quota   limit   grace   files   quota   limit   grace
        /lustre      0k      8T     10T       -       0  3000000 36000000       -
uid 6215 is using default block quota setting
uid 6215 is using default file quota setting
name, driver_version, power.limit [W]
NVIDIA A30, 570.124.06, 165.00 W
NVIDIA A30, 570.124.06, 165.00 W
NVIDIA A30, 570.124.06, 165.00 W
NVIDIA A30, 570.124.06, 165.00 W
NVIDIA A30, 570.124.06, 165.00 W
NVIDIA A30, 570.124.06, 165.00 W
NVIDIA A30, 570.124.06, 165.00 W
NVIDIA A30, 570.124.06, 165.00 W
Using GPU(s) 0,1,2,3,4,5,6,7
This job is assigned the following resources by SLURM:
CPU_IDs=0-63 GRES=gpu:8(IDX:0-7)
Main program continues to run. Monitoring information will be exported after three hours.
Have already added /tools/cluster-modulefiles into $MODULEPATH
no change     /home/S/yangrongzheng/miniconda3/condabin/conda
no change     /home/S/yangrongzheng/miniconda3/bin/conda
no change     /home/S/yangrongzheng/miniconda3/bin/conda-env
no change     /home/S/yangrongzheng/miniconda3/bin/activate
no change     /home/S/yangrongzheng/miniconda3/bin/deactivate
no change     /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
no change     /home/S/yangrongzheng/miniconda3/etc/fish/conf.d/conda.fish
no change     /home/S/yangrongzheng/miniconda3/shell/condabin/Conda.psm1
no change     /home/S/yangrongzheng/miniconda3/shell/condabin/conda-hook.ps1
no change     /home/S/yangrongzheng/miniconda3/lib/python3.13/site-packages/xontrib/conda.xsh
no change     /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.csh
no change     /home/S/yangrongzheng/.bashrc
No action taken.

================================================================================
                              RL-100 TRAINING
================================================================================

Configuration:
task:
  name: dial-turn
  task_name: ${.name}
  shape_meta:
    obs:
      point_cloud:
        shape:
        - 512
        - 3
        type: point_cloud
      agent_pos:
        shape:
        - 9
        type: low_dim
    action:
      shape:
      - 4
  env_runner:
    _target_: diffusion_policy_3d.env_runner.metaworld_runner.MetaworldRunner
    eval_episodes: 100
    n_obs_steps: ${n_obs_steps}
    n_action_steps: ${n_action_steps}
    fps: 10
    n_envs: null
    n_train: null
    n_test: null
    task_name: ${task_name}
    device: ${training.device}
    use_point_crop: ${policy.use_point_crop}
    num_points: 512
  dataset:
    _target_: diffusion_policy_3d.dataset.metaworld_dataset.MetaworldDataset
    zarr_path: data/metaworld_dial-turn_expert.zarr
    horizon: ${horizon}
    pad_before: ${eval:'${n_obs_steps}-1'}
    pad_after: ${eval:'${n_action_steps}-1'}
    seed: 42
    val_ratio: 0.02
    max_train_episodes: 90
    n_action_steps: ${n_action_steps}
    gamma: 0.99
name: train_rl100
task_name: ${task.name}
shape_meta: ${task.shape_meta}
exp_name: debug
horizon: 16
n_obs_steps: 2
n_action_steps: 8
obs_as_global_cond: true
policy:
  _target_: diffusion_policy_3d.policy.rl100_policy.RL100Policy
  shape_meta: ${shape_meta}
  horizon: ${horizon}
  n_obs_steps: ${n_obs_steps}
  n_action_steps: ${n_action_steps}
  num_inference_steps: 10
  use_point_crop: true
  encoder_output_dim: 64
  diffusion_step_embed_dim: 128
  down_dims:
  - 512
  - 1024
  - 2048
  kernel_size: 5
  n_groups: 8
  condition_type: film
  use_pc_color: false
  pointnet_type: pointnet
  obs_as_global_cond: ${obs_as_global_cond}
  noise_scheduler:
    _target_: diffusers.schedulers.scheduling_ddim.DDIMScheduler
    num_train_timesteps: 100
    beta_schedule: squaredcos_cap_v2
    clip_sample: true
    prediction_type: epsilon
  pointcloud_encoder_cfg:
    in_channels: 3
    out_channels: 64
    use_layernorm: true
    final_norm: layernorm
    normal_channel: false
  use_recon_vib: false
  beta_recon: 0.0001
  beta_kl: 0.0001
  ppo_clip_eps: 0.2
  sigma_min: 0.01
  sigma_max: 0.1
  use_variance_clip: true
critics:
  hidden_dims:
  - 256
  - 256
  - 256
  gamma: 0.9227
  tau: 0.7
  target_update_tau: 0.005
  reward_scale: 1.0
  reward_type: sparse
optimizer:
  policy:
    _target_: torch.optim.AdamW
    lr: 0.0001
    weight_decay: 1.0e-06
    betas:
    - 0.9
    - 0.999
  v_network:
    _target_: torch.optim.AdamW
    lr: 0.0003
    weight_decay: 1.0e-06
  q_network:
    _target_: torch.optim.AdamW
    lr: 0.0003
    weight_decay: 1.0e-06
  consistency:
    _target_: torch.optim.AdamW
    lr: 0.0001
    weight_decay: 1.0e-06
ema:
  _target_: diffusion_policy_3d.model.diffusion.ema_model.EMAModel
  power: 0.75
  max_value: 0.9999
  min_value: 0.0
dataloader:
  batch_size: 256
  num_workers: 8
  shuffle: true
  pin_memory: true
  drop_last: false
  persistent_workers: true
training:
  device: cuda:0
  seed: 42
  use_ema: true
  il_epochs: 1000
  il_retrain_epochs: 100
  retrain_il_after_collection: true
  num_offline_iterations: 10
  critic_epochs: 20
  ppo_epochs: 3
  ppo_inner_steps: 1
  collection_episodes: 20
  cd_every: 5
  lambda_cd: 0
  rl_policy_lr: 1.0e-05
  run_online_rl: true
  online_rl_iterations: 10
  online_collection_episodes: 20
  lambda_v: 0.5
  gae_lambda: 0.95
  gradient_accumulate_every: 1
  max_grad_norm: 0.5
  log_every: 10
  eval_every: 100
  checkpoint_every: 200
  resume: true
  resume_path: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/after_il.ckpt
ope:
  num_batches: 100
  rollout_horizon: 3
  delta_coef: 0.01
  delta_abs_min: 0.0
runtime:
  collection_policy: ddim
  collection_use_ema: false
  il_retrain_success_only: true
  merge_success_only: true
  final_eval_policies:
  - ddim
  - cm
  final_eval_use_ema: true
  eval_policy_mode: ddim
  eval_use_ema: true
checkpoint:
  save_ckpt: true
  save_last_ckpt: true
  save_last_snapshot: false
  topk:
    monitor_key: test_mean_score
    mode: max
    k: 1
logging:
  use_wandb: false
  project: rl100-dp3
  group: ${task.name}
  name: rl100_${task.name}_seed${training.seed}
  mode: online


[Setup] Random seed: 42
[Setup] Output directory: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy

[Setup] Loading dataset...
Replay Buffer: state, shape (20000, 9), dtype float32, range -0.16~0.88
Replay Buffer: action, shape (20000, 4), dtype float32, range -1.51~1.69
Replay Buffer: point_cloud, shape (20000, 1024, 6), dtype float32, range -0.95~255.00
Replay Buffer: reward, shape (20000,), dtype float32, range 0.00~1.00
Replay Buffer: done, shape (20000,), dtype float32, range 0.00~1.00
--------------------------
[Setup] Dataset loaded: 17370 samples across 100 episodes
[Setup] Dataset has reward/done labels: True
[Setup] Dataset point_cloud points: 1024
[Setup] Initializing environment runner...
[Setup] env_runner.num_points=512 differs from dataset=1024. Override to dataset value to avoid merge shape mismatch.
[MetaWorldEnv] use_point_crop: True
[Setup] Environment runner initialized

[Setup] Initializing RL100Trainer...
[RL100Trainer] Initializing RL100Policy...
[DP3Encoder] point cloud shape: [512, 3]
[DP3Encoder] state shape: [9]
[DP3Encoder] imagination point shape: None
[DP3Encoder] use_recon_vib: False
[DP3Encoder] beta_recon: 0.0001, beta_kl: 0.0001
[PointNetEncoderXYZ] use_layernorm: True
[PointNetEncoderXYZ] use_final_norm: layernorm
[PointNetEncoderXYZ] use_vib: False
[DP3Encoder] output dim: 128
[DiffusionUnetHybridPointcloudPolicy] use_pc_color: False
[DiffusionUnetHybridPointcloudPolicy] pointnet_type: pointnet
[2026-03-25 17:23:58,015][diffusion_policy_3d.model.diffusion.conditional_unet1d][INFO] - number of parameters: 2.550744e+08
----------------------------------
Class name: RL100Policy
  Number of parameters: 255.1383M
   _dummy_variable: 0.0000M (0.00%)
   obs_encoder: 0.0639M (0.03%)
   model: 255.0744M (99.97%)
   mask_generator: 0.0000M (0.00%)
----------------------------------
[RL100Trainer] Initializing IQL Critics...
[RL100Trainer] Initializing Consistency Model...
[2026-03-25 17:23:59,794][diffusion_policy_3d.model.diffusion.conditional_unet1d][INFO] - number of parameters: 2.550744e+08
[RL100Trainer] Initializing Transition Model T_θ(s'|s,a)...
[Setup] RL100Trainer initialized

[Setup] Resuming from checkpoint: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/after_il.ckpt
[TransitionModel] Checkpoint loaded.
[Checkpoint] policy_optimizer not restored (ValueError: loaded state dict contains a parameter group that doesn't match the size of optimizer's group)
[Checkpoint] Loaded from /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/after_il.ckpt
[Setup] IL phase will be skipped — starting directly from offline RL.

[Training] Starting RL-100 pipeline...

================================================================================
                    RL-100 TRAINING PIPELINE
================================================================================


[RL100Trainer] Skipping IL phase — loaded from checkpoint.
[RL100Trainer] Normalizer synced from dataset. Resuming offline RL from iteration 0.
[RL100Trainer] Dataset already contains reward/done labels; keep existing rewards.

================================================================================
               OFFLINE RL ITERATION 1/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 0)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 17370 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=5.39324 | val=0.00488 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-31.21736 | val=0.00051 | no-improve=0/5
[TransitionModel] Epoch   40 | train=-37.27446 | val=0.00039 | no-improve=0/5
[TransitionModel] Epoch   60 | train=-41.41919 | val=0.00033 | no-improve=0/5
[TransitionModel] Epoch   80 | train=-45.35632 | val=0.00029 | no-improve=0/5
[TransitionModel] Epoch  100 | train=-49.11932 | val=0.00021 | no-improve=0/5
[TransitionModel] Epoch  120 | train=-52.74034 | val=0.00017 | no-improve=0/5
[TransitionModel] Epoch  140 | train=-56.12397 | val=0.00015 | no-improve=0/5
[TransitionModel] Epoch  160 | train=-59.82198 | val=0.00013 | no-improve=0/5
[TransitionModel] Epoch  180 | train=-63.17687 | val=0.00013 | no-improve=0/5
[TransitionModel] Training complete. Elites=[2, 0, 6, 3, 4], val_loss=0.00012
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_00.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_00.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 0)
[IQL] Epoch 0/20, V Loss: 0.0314, Q Loss: 0.0613
[IQL] Epoch 1/20, V Loss: 0.0001, Q Loss: 0.0084
[IQL] Epoch 2/20, V Loss: 0.0001, Q Loss: 0.0081
[IQL] Epoch 3/20, V Loss: 0.0001, Q Loss: 0.0079
[IQL] Epoch 4/20, V Loss: 0.0001, Q Loss: 0.0078
[IQL] Epoch 5/20, V Loss: 0.0001, Q Loss: 0.0077
[IQL] Epoch 6/20, V Loss: 0.0001, Q Loss: 0.0078
[IQL] Epoch 7/20, V Loss: 0.0002, Q Loss: 0.0077
[IQL] Epoch 8/20, V Loss: 0.0001, Q Loss: 0.0075
[IQL] Epoch 9/20, V Loss: 0.0001, Q Loss: 0.0075
[IQL] Epoch 10/20, V Loss: 0.0002, Q Loss: 0.0075
[IQL] Epoch 11/20, V Loss: 0.0001, Q Loss: 0.0073
[IQL] Epoch 12/20, V Loss: 0.0002, Q Loss: 0.0074
[IQL] Epoch 13/20, V Loss: 0.0003, Q Loss: 0.0075
[IQL] Epoch 14/20, V Loss: 0.0002, Q Loss: 0.0072
[IQL] Epoch 15/20, V Loss: 0.0002, Q Loss: 0.0071
[IQL] Epoch 16/20, V Loss: 0.0002, Q Loss: 0.0072
[IQL] Epoch 17/20, V Loss: 0.0002, Q Loss: 0.0073
[IQL] Epoch 18/20, V Loss: 0.0002, Q Loss: 0.0070
[IQL] Epoch 19/20, V Loss: 0.0004, Q Loss: 0.0071
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_00.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_00.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 0)
[OPE] Behavior policy value J_old = 0.2632
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 1.00e-05
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 68 mini-batches, 17370 samples, raw advantage mean=-0.0067, std=0.0078
[Offline RL] Epoch 0/3, PPO Loss: -0.0284, PostKL: 5.247e-02, PostClipFrac: 0.298530, PostMeanRatio: 0.995786, PostRatioDev: 1.942e-01, GradNorm: 17.5871, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 1/3, PPO Loss: -0.0078, PostKL: 5.223e-02, PostClipFrac: 0.269868, PostMeanRatio: 0.997292, PostRatioDev: 1.832e-01, GradNorm: 17.6725, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 2/3, PPO Loss: 0.0051, PostKL: 5.577e-02, PostClipFrac: 0.276185, PostMeanRatio: 0.995021, PostRatioDev: 1.851e-01, GradNorm: 16.1219, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_00.png
[OPE] Policy REJECTED: J_new=0.2631 ≤ J_old=0.2632 + δ=0.0026. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 0)
[Collect] 20 episodes, success=0.700, env_return=1067.19, rl_reward=0.70, steps=4000
[Data Collection] Success Rate: 0.700, EnvReturn: 1067.19, RLReward: 0.70, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 20 episodes remain in RL replay; IL retrain keeps 14/20 successful episodes (drops 6 failures).
[Dataset] Merged 20 episodes (4000 steps) → total 24000 steps, 120 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0503, Val Loss: 0.0124
[IL] Epoch 1/100, Loss: 0.0141, Val Loss: 0.0106
[IL] Epoch 2/100, Loss: 0.0118, Val Loss: 0.0095
[IL] Epoch 3/100, Loss: 0.0107, Val Loss: 0.0092
[IL] Epoch 4/100, Loss: 0.0098, Val Loss: 0.0079
[IL] Epoch 5/100, Loss: 0.0091, Val Loss: 0.0092
[IL] Epoch 6/100, Loss: 0.0094, Val Loss: 0.0091
[IL] Epoch 7/100, Loss: 0.0087, Val Loss: 0.0103
[IL] Epoch 8/100, Loss: 0.0087, Val Loss: 0.0109
[IL] Epoch 9/100, Loss: 0.0081, Val Loss: 0.0096
[IL] Epoch 10/100, Loss: 0.0085, Val Loss: 0.0069
[IL] Epoch 11/100, Loss: 0.0075, Val Loss: 0.0065
[IL] Epoch 12/100, Loss: 0.0069, Val Loss: 0.0072
[IL] Epoch 13/100, Loss: 0.0070, Val Loss: 0.0059
[IL] Epoch 14/100, Loss: 0.0067, Val Loss: 0.0069
[IL] Epoch 15/100, Loss: 0.0064, Val Loss: 0.0089
[IL] Epoch 16/100, Loss: 0.0078, Val Loss: 0.0073
[IL] Epoch 17/100, Loss: 0.0067, Val Loss: 0.0071
[IL] Epoch 18/100, Loss: 0.0068, Val Loss: 0.0094
[IL] Epoch 19/100, Loss: 0.0069, Val Loss: 0.0043
[IL] Epoch 20/100, Loss: 0.0065, Val Loss: 0.0049
[IL] Epoch 21/100, Loss: 0.0061, Val Loss: 0.0056
[IL] Epoch 22/100, Loss: 0.0062, Val Loss: 0.0071
[IL] Epoch 23/100, Loss: 0.0069, Val Loss: 0.0094
[IL] Epoch 24/100, Loss: 0.0068, Val Loss: 0.0075
[IL] Epoch 25/100, Loss: 0.0063, Val Loss: 0.0102
[IL] Epoch 26/100, Loss: 0.0061, Val Loss: 0.0076
[IL] Epoch 27/100, Loss: 0.0069, Val Loss: 0.0074
[IL] Epoch 28/100, Loss: 0.0061, Val Loss: 0.0084
[IL] Epoch 29/100, Loss: 0.0063, Val Loss: 0.0087
[IL] Epoch 30/100, Loss: 0.0061, Val Loss: 0.0077
[IL] Epoch 31/100, Loss: 0.0062, Val Loss: 0.0073
[IL] Epoch 32/100, Loss: 0.0060, Val Loss: 0.0084
[IL] Epoch 33/100, Loss: 0.0065, Val Loss: 0.0078
[IL] Epoch 34/100, Loss: 0.0054, Val Loss: 0.0085
[IL] Epoch 35/100, Loss: 0.0058, Val Loss: 0.0090
[IL] Epoch 36/100, Loss: 0.0057, Val Loss: 0.0083
[IL] Epoch 37/100, Loss: 0.0054, Val Loss: 0.0062
[IL] Epoch 38/100, Loss: 0.0052, Val Loss: 0.0079
[IL] Epoch 39/100, Loss: 0.0055, Val Loss: 0.0075
[IL] Epoch 40/100, Loss: 0.0060, Val Loss: 0.0083
[IL] Epoch 41/100, Loss: 0.0062, Val Loss: 0.0057
[IL] Epoch 42/100, Loss: 0.0059, Val Loss: 0.0073
[IL] Epoch 43/100, Loss: 0.0061, Val Loss: 0.0083
[IL] Epoch 44/100, Loss: 0.0062, Val Loss: 0.0075
[IL] Epoch 45/100, Loss: 0.0055, Val Loss: 0.0082
[IL] Epoch 46/100, Loss: 0.0051, Val Loss: 0.0073
[IL] Epoch 47/100, Loss: 0.0055, Val Loss: 0.0075
[IL] Epoch 48/100, Loss: 0.0058, Val Loss: 0.0067
[IL] Epoch 49/100, Loss: 0.0058, Val Loss: 0.0051
[IL] Epoch 50/100, Loss: 0.0056, Val Loss: 0.0080
[IL] Epoch 51/100, Loss: 0.0052, Val Loss: 0.0067
[IL] Epoch 52/100, Loss: 0.0056, Val Loss: 0.0086
[IL] Epoch 53/100, Loss: 0.0057, Val Loss: 0.0090
[IL] Epoch 54/100, Loss: 0.0052, Val Loss: 0.0082
[IL] Epoch 55/100, Loss: 0.0058, Val Loss: 0.0085
[IL] Epoch 56/100, Loss: 0.0051, Val Loss: 0.0071
[IL] Epoch 57/100, Loss: 0.0051, Val Loss: 0.0087
[IL] Epoch 58/100, Loss: 0.0053, Val Loss: 0.0070
[IL] Epoch 59/100, Loss: 0.0048, Val Loss: 0.0101
[IL] Epoch 60/100, Loss: 0.0053, Val Loss: 0.0077
[IL] Epoch 61/100, Loss: 0.0050, Val Loss: 0.0077
[IL] Epoch 62/100, Loss: 0.0048, Val Loss: 0.0091
[IL] Epoch 63/100, Loss: 0.0051, Val Loss: 0.0078
[IL] Epoch 64/100, Loss: 0.0052, Val Loss: 0.0068
[IL] Epoch 65/100, Loss: 0.0051, Val Loss: 0.0073
[IL] Epoch 66/100, Loss: 0.0056, Val Loss: 0.0080
[IL] Epoch 67/100, Loss: 0.0056, Val Loss: 0.0064
[IL] Epoch 68/100, Loss: 0.0046, Val Loss: 0.0100
[IL] Epoch 69/100, Loss: 0.0058, Val Loss: 0.0065
[IL] Epoch 70/100, Loss: 0.0058, Val Loss: 0.0060
[IL] Epoch 71/100, Loss: 0.0050, Val Loss: 0.0083
[IL] Epoch 72/100, Loss: 0.0050, Val Loss: 0.0087
[IL] Epoch 73/100, Loss: 0.0048, Val Loss: 0.0081
[IL] Epoch 74/100, Loss: 0.0053, Val Loss: 0.0077
[IL] Epoch 75/100, Loss: 0.0045, Val Loss: 0.0070
[IL] Epoch 76/100, Loss: 0.0047, Val Loss: 0.0084
[IL] Epoch 77/100, Loss: 0.0047, Val Loss: 0.0063
[IL] Epoch 78/100, Loss: 0.0044, Val Loss: 0.0077
[IL] Epoch 79/100, Loss: 0.0046, Val Loss: 0.0077
[IL] Epoch 80/100, Loss: 0.0045, Val Loss: 0.0059
[IL] Epoch 81/100, Loss: 0.0045, Val Loss: 0.0069
[IL] Epoch 82/100, Loss: 0.0049, Val Loss: 0.0061
[IL] Epoch 83/100, Loss: 0.0049, Val Loss: 0.0071
[IL] Epoch 84/100, Loss: 0.0050, Val Loss: 0.0139
[IL] Epoch 85/100, Loss: 0.0048, Val Loss: 0.0071
[IL] Epoch 86/100, Loss: 0.0050, Val Loss: 0.0058
[IL] Epoch 87/100, Loss: 0.0049, Val Loss: 0.0072
[IL] Epoch 88/100, Loss: 0.0049, Val Loss: 0.0082
[IL] Epoch 89/100, Loss: 0.0043, Val Loss: 0.0089
[IL] Epoch 90/100, Loss: 0.0053, Val Loss: 0.0075
[IL] Epoch 91/100, Loss: 0.0052, Val Loss: 0.0088
[IL] Epoch 92/100, Loss: 0.0044, Val Loss: 0.0069
[IL] Epoch 93/100, Loss: 0.0047, Val Loss: 0.0063
[IL] Epoch 94/100, Loss: 0.0051, Val Loss: 0.0069
[IL] Epoch 95/100, Loss: 0.0049, Val Loss: 0.0065
[IL] Epoch 96/100, Loss: 0.0051, Val Loss: 0.0112
[IL] Epoch 97/100, Loss: 0.0048, Val Loss: 0.0084
[IL] Epoch 98/100, Loss: 0.0046, Val Loss: 0.0123
[IL] Epoch 99/100, Loss: 0.0044, Val Loss: 0.0066
test_mean_score: 0.45
[IL] Eval - Success Rate: 0.450
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_00.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_0.ckpt

================================================================================
               OFFLINE RL ITERATION 2/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 1)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 21230 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-36.06926 | val=0.00050 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-68.30794 | val=0.00024 | no-improve=0/5
[TransitionModel] Epoch   40 | train=-73.39562 | val=0.00023 | no-improve=0/5
[TransitionModel] Epoch   60 | train=-77.50230 | val=0.00022 | no-improve=4/5
[TransitionModel] Epoch   61 | train=-77.90658 | val=0.00024 | no-improve=5/5
[TransitionModel] Training complete. Elites=[1, 3, 6, 5, 0], val_loss=0.00021
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_01.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_01.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 1)
[IQL] Epoch 0/20, V Loss: 0.0001, Q Loss: 0.0067
[IQL] Epoch 1/20, V Loss: 0.0002, Q Loss: 0.0066
[IQL] Epoch 2/20, V Loss: 0.0002, Q Loss: 0.0068
[IQL] Epoch 3/20, V Loss: 0.0003, Q Loss: 0.0067
[IQL] Epoch 4/20, V Loss: 0.0002, Q Loss: 0.0066
[IQL] Epoch 5/20, V Loss: 0.0002, Q Loss: 0.0065
[IQL] Epoch 6/20, V Loss: 0.0002, Q Loss: 0.0065
[IQL] Epoch 7/20, V Loss: 0.0002, Q Loss: 0.0064
[IQL] Epoch 8/20, V Loss: 0.0003, Q Loss: 0.0064
[IQL] Epoch 9/20, V Loss: 0.0001, Q Loss: 0.0063
[IQL] Epoch 10/20, V Loss: 0.0003, Q Loss: 0.0064
[IQL] Epoch 11/20, V Loss: 0.0003, Q Loss: 0.0064
[IQL] Epoch 12/20, V Loss: 0.0002, Q Loss: 0.0063
[IQL] Epoch 13/20, V Loss: 0.0002, Q Loss: 0.0063
[IQL] Epoch 14/20, V Loss: 0.0003, Q Loss: 0.0063
[IQL] Epoch 15/20, V Loss: 0.0002, Q Loss: 0.0062
[IQL] Epoch 16/20, V Loss: 0.0002, Q Loss: 0.0062
[IQL] Epoch 17/20, V Loss: 0.0003, Q Loss: 0.0061
[IQL] Epoch 18/20, V Loss: 0.0001, Q Loss: 0.0060
[IQL] Epoch 19/20, V Loss: 0.0002, Q Loss: 0.0060
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_01.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_01.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 1)
[OPE] Behavior policy value J_old = 0.3943
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 1.00e-05
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 83 mini-batches, 21230 samples, raw advantage mean=0.0056, std=0.0124
[Offline RL] Epoch 0/3, PPO Loss: -0.0398, PostKL: 5.775e-02, PostClipFrac: 0.298988, PostMeanRatio: 0.996363, PostRatioDev: 1.968e-01, GradNorm: 27.5530, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 1/3, PPO Loss: -0.0108, PostKL: 8.086e-02, PostClipFrac: 0.287835, PostMeanRatio: 1.007448, PostRatioDev: 2.057e-01, GradNorm: 21.4160, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 2/3, PPO Loss: 0.0063, PostKL: 7.729e-02, PostClipFrac: 0.290873, PostMeanRatio: 1.005747, PostRatioDev: 2.046e-01, GradNorm: 15.3145, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_01.png
[OPE] Policy REJECTED: J_new=0.3943 ≤ J_old=0.3943 + δ=0.0039. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 1)
[Collect] 20 episodes, success=0.800, env_return=1049.10, rl_reward=0.80, steps=4000
[Data Collection] Success Rate: 0.800, EnvReturn: 1049.10, RLReward: 0.80, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 20 episodes remain in RL replay; IL retrain keeps 16/20 successful episodes (drops 4 failures).
[Dataset] Merged 20 episodes (4000 steps) → total 28000 steps, 140 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0083, Val Loss: 0.0073
[IL] Epoch 1/100, Loss: 0.0068, Val Loss: 0.0095
[IL] Epoch 2/100, Loss: 0.0063, Val Loss: 0.0069
[IL] Epoch 3/100, Loss: 0.0070, Val Loss: 0.0075
[IL] Epoch 4/100, Loss: 0.0063, Val Loss: 0.0072
[IL] Epoch 5/100, Loss: 0.0063, Val Loss: 0.0067
[IL] Epoch 6/100, Loss: 0.0063, Val Loss: 0.0076
[IL] Epoch 7/100, Loss: 0.0061, Val Loss: 0.0064
[IL] Epoch 8/100, Loss: 0.0065, Val Loss: 0.0078
[IL] Epoch 9/100, Loss: 0.0059, Val Loss: 0.0052
[IL] Epoch 10/100, Loss: 0.0066, Val Loss: 0.0096
[IL] Epoch 11/100, Loss: 0.0061, Val Loss: 0.0072
[IL] Epoch 12/100, Loss: 0.0057, Val Loss: 0.0083
[IL] Epoch 13/100, Loss: 0.0055, Val Loss: 0.0102
[IL] Epoch 14/100, Loss: 0.0053, Val Loss: 0.0099
[IL] Epoch 15/100, Loss: 0.0060, Val Loss: 0.0082
[IL] Epoch 16/100, Loss: 0.0059, Val Loss: 0.0073
[IL] Epoch 17/100, Loss: 0.0055, Val Loss: 0.0114
[IL] Epoch 18/100, Loss: 0.0060, Val Loss: 0.0074
[IL] Epoch 19/100, Loss: 0.0057, Val Loss: 0.0082
[IL] Epoch 20/100, Loss: 0.0057, Val Loss: 0.0068
[IL] Epoch 21/100, Loss: 0.0058, Val Loss: 0.0090
[IL] Epoch 22/100, Loss: 0.0056, Val Loss: 0.0096
[IL] Epoch 23/100, Loss: 0.0056, Val Loss: 0.0087
[IL] Epoch 24/100, Loss: 0.0057, Val Loss: 0.0091
[IL] Epoch 25/100, Loss: 0.0054, Val Loss: 0.0096
[IL] Epoch 26/100, Loss: 0.0049, Val Loss: 0.0082
[IL] Epoch 27/100, Loss: 0.0054, Val Loss: 0.0069
[IL] Epoch 28/100, Loss: 0.0057, Val Loss: 0.0097
[IL] Epoch 29/100, Loss: 0.0058, Val Loss: 0.0087
[IL] Epoch 30/100, Loss: 0.0056, Val Loss: 0.0074
[IL] Epoch 31/100, Loss: 0.0053, Val Loss: 0.0076
[IL] Epoch 32/100, Loss: 0.0056, Val Loss: 0.0062
[IL] Epoch 33/100, Loss: 0.0052, Val Loss: 0.0061
[IL] Epoch 34/100, Loss: 0.0051, Val Loss: 0.0050
[IL] Epoch 35/100, Loss: 0.0054, Val Loss: 0.0084
[IL] Epoch 36/100, Loss: 0.0052, Val Loss: 0.0074
[IL] Epoch 37/100, Loss: 0.0058, Val Loss: 0.0077
[IL] Epoch 38/100, Loss: 0.0055, Val Loss: 0.0074
[IL] Epoch 39/100, Loss: 0.0055, Val Loss: 0.0058
[IL] Epoch 40/100, Loss: 0.0056, Val Loss: 0.0102
[IL] Epoch 41/100, Loss: 0.0051, Val Loss: 0.0082
[IL] Epoch 42/100, Loss: 0.0056, Val Loss: 0.0098
[IL] Epoch 43/100, Loss: 0.0053, Val Loss: 0.0082
[IL] Epoch 44/100, Loss: 0.0047, Val Loss: 0.0091
[IL] Epoch 45/100, Loss: 0.0048, Val Loss: 0.0083
[IL] Epoch 46/100, Loss: 0.0047, Val Loss: 0.0072
[IL] Epoch 47/100, Loss: 0.0051, Val Loss: 0.0068
[IL] Epoch 48/100, Loss: 0.0050, Val Loss: 0.0088
[IL] Epoch 49/100, Loss: 0.0051, Val Loss: 0.0081
[IL] Epoch 50/100, Loss: 0.0051, Val Loss: 0.0075
[IL] Epoch 51/100, Loss: 0.0049, Val Loss: 0.0053
[IL] Epoch 52/100, Loss: 0.0048, Val Loss: 0.0079
[IL] Epoch 53/100, Loss: 0.0053, Val Loss: 0.0074
[IL] Epoch 54/100, Loss: 0.0053, Val Loss: 0.0126
[IL] Epoch 55/100, Loss: 0.0054, Val Loss: 0.0069
[IL] Epoch 56/100, Loss: 0.0052, Val Loss: 0.0079
[IL] Epoch 57/100, Loss: 0.0050, Val Loss: 0.0085
[IL] Epoch 58/100, Loss: 0.0046, Val Loss: 0.0079
[IL] Epoch 59/100, Loss: 0.0052, Val Loss: 0.0082
[IL] Epoch 60/100, Loss: 0.0047, Val Loss: 0.0097
[IL] Epoch 61/100, Loss: 0.0048, Val Loss: 0.0071
[IL] Epoch 62/100, Loss: 0.0044, Val Loss: 0.0098
[IL] Epoch 63/100, Loss: 0.0051, Val Loss: 0.0075
[IL] Epoch 64/100, Loss: 0.0054, Val Loss: 0.0067
[IL] Epoch 65/100, Loss: 0.0048, Val Loss: 0.0093
[IL] Epoch 66/100, Loss: 0.0049, Val Loss: 0.0075
[IL] Epoch 67/100, Loss: 0.0045, Val Loss: 0.0063
[IL] Epoch 68/100, Loss: 0.0047, Val Loss: 0.0068
[IL] Epoch 69/100, Loss: 0.0050, Val Loss: 0.0097
[IL] Epoch 70/100, Loss: 0.0050, Val Loss: 0.0088
[IL] Epoch 71/100, Loss: 0.0051, Val Loss: 0.0066
[IL] Epoch 72/100, Loss: 0.0046, Val Loss: 0.0089
[IL] Epoch 73/100, Loss: 0.0050, Val Loss: 0.0093
[IL] Epoch 74/100, Loss: 0.0054, Val Loss: 0.0095
[IL] Epoch 75/100, Loss: 0.0044, Val Loss: 0.0091
[IL] Epoch 76/100, Loss: 0.0046, Val Loss: 0.0097
[IL] Epoch 77/100, Loss: 0.0044, Val Loss: 0.0059
[IL] Epoch 78/100, Loss: 0.0046, Val Loss: 0.0111
[IL] Epoch 79/100, Loss: 0.0049, Val Loss: 0.0108
[IL] Epoch 80/100, Loss: 0.0044, Val Loss: 0.0098
[IL] Epoch 81/100, Loss: 0.0052, Val Loss: 0.0060
[IL] Epoch 82/100, Loss: 0.0044, Val Loss: 0.0065
[IL] Epoch 83/100, Loss: 0.0045, Val Loss: 0.0089
[IL] Epoch 84/100, Loss: 0.0042, Val Loss: 0.0067
[IL] Epoch 85/100, Loss: 0.0046, Val Loss: 0.0096
[IL] Epoch 86/100, Loss: 0.0045, Val Loss: 0.0115
[IL] Epoch 87/100, Loss: 0.0052, Val Loss: 0.0078
[IL] Epoch 88/100, Loss: 0.0048, Val Loss: 0.0092
[IL] Epoch 89/100, Loss: 0.0044, Val Loss: 0.0052
[IL] Epoch 90/100, Loss: 0.0049, Val Loss: 0.0059
[IL] Epoch 91/100, Loss: 0.0045, Val Loss: 0.0110
[IL] Epoch 92/100, Loss: 0.0052, Val Loss: 0.0102
[IL] Epoch 93/100, Loss: 0.0045, Val Loss: 0.0092
[IL] Epoch 94/100, Loss: 0.0048, Val Loss: 0.0095
[IL] Epoch 95/100, Loss: 0.0049, Val Loss: 0.0096
[IL] Epoch 96/100, Loss: 0.0045, Val Loss: 0.0082
[IL] Epoch 97/100, Loss: 0.0047, Val Loss: 0.0071
[IL] Epoch 98/100, Loss: 0.0045, Val Loss: 0.0076
[IL] Epoch 99/100, Loss: 0.0039, Val Loss: 0.0077
test_mean_score: 0.65
[IL] Eval - Success Rate: 0.650
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_01.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_1.ckpt

================================================================================
               OFFLINE RL ITERATION 3/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 2)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 25090 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-69.48351 | val=0.00045 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-81.72370 | val=0.00039 | no-improve=3/5
[TransitionModel] Epoch   27 | train=-83.84827 | val=0.00037 | no-improve=5/5
[TransitionModel] Training complete. Elites=[6, 3, 4, 1, 2], val_loss=0.00034
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_02.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_02.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 2)
[IQL] Epoch 0/20, V Loss: 0.0002, Q Loss: 0.0059
[IQL] Epoch 1/20, V Loss: 0.0003, Q Loss: 0.0060
[IQL] Epoch 2/20, V Loss: 0.0002, Q Loss: 0.0059
[IQL] Epoch 3/20, V Loss: 0.0003, Q Loss: 0.0058
[IQL] Epoch 4/20, V Loss: 0.0003, Q Loss: 0.0058
[IQL] Epoch 5/20, V Loss: 0.0002, Q Loss: 0.0058
[IQL] Epoch 6/20, V Loss: 0.0003, Q Loss: 0.0057
[IQL] Epoch 7/20, V Loss: 0.0002, Q Loss: 0.0058
[IQL] Epoch 8/20, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 9/20, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 10/20, V Loss: 0.0002, Q Loss: 0.0057
[IQL] Epoch 11/20, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 12/20, V Loss: 0.0001, Q Loss: 0.0055
[IQL] Epoch 13/20, V Loss: 0.0003, Q Loss: 0.0056
[IQL] Epoch 14/20, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 15/20, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 16/20, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 17/20, V Loss: 0.0003, Q Loss: 0.0056
[IQL] Epoch 18/20, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 19/20, V Loss: 0.0002, Q Loss: 0.0055
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_02.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_02.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 2)
[OPE] Behavior policy value J_old = 0.3821
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 1.00e-05
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 99 mini-batches, 25090 samples, raw advantage mean=0.0044, std=0.0094
[Offline RL] Epoch 0/3, PPO Loss: -0.0356, PostKL: 6.157e-02, PostClipFrac: 0.294854, PostMeanRatio: 0.992339, PostRatioDev: 1.935e-01, GradNorm: 19.8459, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 1/3, PPO Loss: -0.0123, PostKL: 1.021e-01, PostClipFrac: 0.290382, PostMeanRatio: 1.012604, PostRatioDev: 2.180e-01, GradNorm: 17.1577, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 2/3, PPO Loss: 0.0048, PostKL: 9.285e-02, PostClipFrac: 0.283946, PostMeanRatio: 1.001146, PostRatioDev: 2.045e-01, GradNorm: 15.5801, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_02.png
[OPE] Policy REJECTED: J_new=0.3822 ≤ J_old=0.3821 + δ=0.0038. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 2)
[Collect] 20 episodes, success=1.000, env_return=1145.66, rl_reward=1.00, steps=4000
[Data Collection] Success Rate: 1.000, EnvReturn: 1145.66, RLReward: 1.00, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 20 episodes remain in RL replay; IL retrain keeps 20/20 successful episodes (drops 0 failures).
[Dataset] Merged 20 episodes (4000 steps) → total 32000 steps, 160 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0068, Val Loss: 0.0078
[IL] Epoch 1/100, Loss: 0.0058, Val Loss: 0.0084
[IL] Epoch 2/100, Loss: 0.0063, Val Loss: 0.0075
[IL] Epoch 3/100, Loss: 0.0061, Val Loss: 0.0073
[IL] Epoch 4/100, Loss: 0.0058, Val Loss: 0.0094
[IL] Epoch 5/100, Loss: 0.0058, Val Loss: 0.0088
[IL] Epoch 6/100, Loss: 0.0057, Val Loss: 0.0083
[IL] Epoch 7/100, Loss: 0.0057, Val Loss: 0.0086
[IL] Epoch 8/100, Loss: 0.0060, Val Loss: 0.0101
[IL] Epoch 9/100, Loss: 0.0059, Val Loss: 0.0069
[IL] Epoch 10/100, Loss: 0.0059, Val Loss: 0.0111
[IL] Epoch 11/100, Loss: 0.0055, Val Loss: 0.0065
[IL] Epoch 12/100, Loss: 0.0053, Val Loss: 0.0082
[IL] Epoch 13/100, Loss: 0.0054, Val Loss: 0.0072
[IL] Epoch 14/100, Loss: 0.0053, Val Loss: 0.0080
[IL] Epoch 15/100, Loss: 0.0053, Val Loss: 0.0079
[IL] Epoch 16/100, Loss: 0.0055, Val Loss: 0.0092
[IL] Epoch 17/100, Loss: 0.0050, Val Loss: 0.0104
[IL] Epoch 18/100, Loss: 0.0051, Val Loss: 0.0086
[IL] Epoch 19/100, Loss: 0.0052, Val Loss: 0.0132
[IL] Epoch 20/100, Loss: 0.0051, Val Loss: 0.0081
[IL] Epoch 21/100, Loss: 0.0050, Val Loss: 0.0113
[IL] Epoch 22/100, Loss: 0.0054, Val Loss: 0.0100
[IL] Epoch 23/100, Loss: 0.0050, Val Loss: 0.0115
[IL] Epoch 24/100, Loss: 0.0051, Val Loss: 0.0128
[IL] Epoch 25/100, Loss: 0.0052, Val Loss: 0.0094
[IL] Epoch 26/100, Loss: 0.0051, Val Loss: 0.0070
[IL] Epoch 27/100, Loss: 0.0053, Val Loss: 0.0101
[IL] Epoch 28/100, Loss: 0.0048, Val Loss: 0.0050
[IL] Epoch 29/100, Loss: 0.0047, Val Loss: 0.0083
[IL] Epoch 30/100, Loss: 0.0051, Val Loss: 0.0066
[IL] Epoch 31/100, Loss: 0.0053, Val Loss: 0.0083
[IL] Epoch 32/100, Loss: 0.0051, Val Loss: 0.0088
[IL] Epoch 33/100, Loss: 0.0048, Val Loss: 0.0076
[IL] Epoch 34/100, Loss: 0.0052, Val Loss: 0.0099
[IL] Epoch 35/100, Loss: 0.0048, Val Loss: 0.0079
[IL] Epoch 36/100, Loss: 0.0051, Val Loss: 0.0067
[IL] Epoch 37/100, Loss: 0.0051, Val Loss: 0.0072
[IL] Epoch 38/100, Loss: 0.0050, Val Loss: 0.0106
[IL] Epoch 39/100, Loss: 0.0049, Val Loss: 0.0085
[IL] Epoch 40/100, Loss: 0.0049, Val Loss: 0.0065
[IL] Epoch 41/100, Loss: 0.0051, Val Loss: 0.0079
[IL] Epoch 42/100, Loss: 0.0048, Val Loss: 0.0082
[IL] Epoch 43/100, Loss: 0.0051, Val Loss: 0.0079
[IL] Epoch 44/100, Loss: 0.0050, Val Loss: 0.0054
[IL] Epoch 45/100, Loss: 0.0047, Val Loss: 0.0105
[IL] Epoch 46/100, Loss: 0.0047, Val Loss: 0.0073
[IL] Epoch 47/100, Loss: 0.0047, Val Loss: 0.0094
[IL] Epoch 48/100, Loss: 0.0045, Val Loss: 0.0065
[IL] Epoch 49/100, Loss: 0.0047, Val Loss: 0.0108
[IL] Epoch 50/100, Loss: 0.0046, Val Loss: 0.0107
[IL] Epoch 51/100, Loss: 0.0042, Val Loss: 0.0110
[IL] Epoch 52/100, Loss: 0.0048, Val Loss: 0.0116
[IL] Epoch 53/100, Loss: 0.0050, Val Loss: 0.0091
[IL] Epoch 54/100, Loss: 0.0046, Val Loss: 0.0080
[IL] Epoch 55/100, Loss: 0.0046, Val Loss: 0.0062
[IL] Epoch 56/100, Loss: 0.0044, Val Loss: 0.0133
[IL] Epoch 57/100, Loss: 0.0044, Val Loss: 0.0088
[IL] Epoch 58/100, Loss: 0.0044, Val Loss: 0.0072
[IL] Epoch 59/100, Loss: 0.0047, Val Loss: 0.0072
[IL] Epoch 60/100, Loss: 0.0047, Val Loss: 0.0077
[IL] Epoch 61/100, Loss: 0.0054, Val Loss: 0.0087
[IL] Epoch 62/100, Loss: 0.0046, Val Loss: 0.0113
[IL] Epoch 63/100, Loss: 0.0043, Val Loss: 0.0092
[IL] Epoch 64/100, Loss: 0.0045, Val Loss: 0.0101
[IL] Epoch 65/100, Loss: 0.0044, Val Loss: 0.0082
[IL] Epoch 66/100, Loss: 0.0043, Val Loss: 0.0083
[IL] Epoch 67/100, Loss: 0.0047, Val Loss: 0.0117
[IL] Epoch 68/100, Loss: 0.0043, Val Loss: 0.0082
[IL] Epoch 69/100, Loss: 0.0040, Val Loss: 0.0073
[IL] Epoch 70/100, Loss: 0.0044, Val Loss: 0.0087
[IL] Epoch 71/100, Loss: 0.0043, Val Loss: 0.0137
[IL] Epoch 72/100, Loss: 0.0043, Val Loss: 0.0090
[IL] Epoch 73/100, Loss: 0.0046, Val Loss: 0.0140
[IL] Epoch 74/100, Loss: 0.0048, Val Loss: 0.0068
[IL] Epoch 75/100, Loss: 0.0044, Val Loss: 0.0075
[IL] Epoch 76/100, Loss: 0.0045, Val Loss: 0.0109
[IL] Epoch 77/100, Loss: 0.0044, Val Loss: 0.0101
[IL] Epoch 78/100, Loss: 0.0048, Val Loss: 0.0100
[IL] Epoch 79/100, Loss: 0.0045, Val Loss: 0.0062
[IL] Epoch 80/100, Loss: 0.0045, Val Loss: 0.0084
[IL] Epoch 81/100, Loss: 0.0044, Val Loss: 0.0059
[IL] Epoch 82/100, Loss: 0.0044, Val Loss: 0.0108
[IL] Epoch 83/100, Loss: 0.0046, Val Loss: 0.0070
[IL] Epoch 84/100, Loss: 0.0042, Val Loss: 0.0072
[IL] Epoch 85/100, Loss: 0.0038, Val Loss: 0.0079
[IL] Epoch 86/100, Loss: 0.0042, Val Loss: 0.0065
[IL] Epoch 87/100, Loss: 0.0039, Val Loss: 0.0091
[IL] Epoch 88/100, Loss: 0.0039, Val Loss: 0.0080
[IL] Epoch 89/100, Loss: 0.0041, Val Loss: 0.0063
[IL] Epoch 90/100, Loss: 0.0041, Val Loss: 0.0065
[IL] Epoch 91/100, Loss: 0.0044, Val Loss: 0.0048
[IL] Epoch 92/100, Loss: 0.0045, Val Loss: 0.0076
[IL] Epoch 93/100, Loss: 0.0042, Val Loss: 0.0090
[IL] Epoch 94/100, Loss: 0.0047, Val Loss: 0.0093
[IL] Epoch 95/100, Loss: 0.0042, Val Loss: 0.0065
[IL] Epoch 96/100, Loss: 0.0042, Val Loss: 0.0094
[IL] Epoch 97/100, Loss: 0.0039, Val Loss: 0.0071
[IL] Epoch 98/100, Loss: 0.0040, Val Loss: 0.0082
[IL] Epoch 99/100, Loss: 0.0040, Val Loss: 0.0110
test_mean_score: 0.71
[IL] Eval - Success Rate: 0.710
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_02.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_2.ckpt

================================================================================
               OFFLINE RL ITERATION 4/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 3)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 28950 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-78.95151 | val=0.00032 | no-improve=0/5
[TransitionModel] Epoch   11 | train=-85.99391 | val=0.00033 | no-improve=5/5
[TransitionModel] Training complete. Elites=[2, 1, 3, 5, 4], val_loss=0.00027
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_03.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_03.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 3)
[IQL] Epoch 0/20, V Loss: 0.0002, Q Loss: 0.0057
[IQL] Epoch 1/20, V Loss: 0.0001, Q Loss: 0.0056
[IQL] Epoch 2/20, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 3/20, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 4/20, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 5/20, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 6/20, V Loss: 0.0003, Q Loss: 0.0056
[IQL] Epoch 7/20, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 8/20, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 9/20, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 10/20, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 11/20, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 12/20, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 13/20, V Loss: 0.0002, Q Loss: 0.0058
[IQL] Epoch 14/20, V Loss: 0.0003, Q Loss: 0.0055
[IQL] Epoch 15/20, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 16/20, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 17/20, V Loss: 0.0002, Q Loss: 0.0057
[IQL] Epoch 18/20, V Loss: 0.0003, Q Loss: 0.0061
[IQL] Epoch 19/20, V Loss: 0.0005, Q Loss: 0.0057
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_03.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_03.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 3)
[OPE] Behavior policy value J_old = 0.4184
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 1.00e-05
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 114 mini-batches, 28950 samples, raw advantage mean=-0.0030, std=0.0153
[Offline RL] Epoch 0/3, PPO Loss: -0.0275, PostKL: 6.638e-02, PostClipFrac: 0.314794, PostMeanRatio: 0.996136, PostRatioDev: 2.060e-01, GradNorm: 18.1557, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 1/3, PPO Loss: -0.0055, PostKL: 9.536e-02, PostClipFrac: 0.301024, PostMeanRatio: 1.018323, PostRatioDev: 2.206e-01, GradNorm: 14.7459, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 2/3, PPO Loss: 0.0087, PostKL: 8.281e-02, PostClipFrac: 0.279486, PostMeanRatio: 1.014482, PostRatioDev: 2.017e-01, GradNorm: 14.3654, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_03.png
[OPE] Policy REJECTED: J_new=0.4185 ≤ J_old=0.4184 + δ=0.0042. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 3)
[Collect] 20 episodes, success=0.800, env_return=1107.54, rl_reward=0.80, steps=4000
[Data Collection] Success Rate: 0.800, EnvReturn: 1107.54, RLReward: 0.80, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 20 episodes remain in RL replay; IL retrain keeps 16/20 successful episodes (drops 4 failures).
[Dataset] Merged 20 episodes (4000 steps) → total 36000 steps, 180 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0113, Val Loss: 0.0098
[IL] Epoch 1/100, Loss: 0.0062, Val Loss: 0.0068
[IL] Epoch 2/100, Loss: 0.0056, Val Loss: 0.0096
[IL] Epoch 3/100, Loss: 0.0053, Val Loss: 0.0108
[IL] Epoch 4/100, Loss: 0.0056, Val Loss: 0.0064
[IL] Epoch 5/100, Loss: 0.0055, Val Loss: 0.0073
[IL] Epoch 6/100, Loss: 0.0054, Val Loss: 0.0112
[IL] Epoch 7/100, Loss: 0.0049, Val Loss: 0.0090
[IL] Epoch 8/100, Loss: 0.0052, Val Loss: 0.0071
[IL] Epoch 9/100, Loss: 0.0053, Val Loss: 0.0085
[IL] Epoch 10/100, Loss: 0.0048, Val Loss: 0.0070
[IL] Epoch 11/100, Loss: 0.0055, Val Loss: 0.0112
[IL] Epoch 12/100, Loss: 0.0052, Val Loss: 0.0127
[IL] Epoch 13/100, Loss: 0.0049, Val Loss: 0.0070
[IL] Epoch 14/100, Loss: 0.0050, Val Loss: 0.0087
[IL] Epoch 15/100, Loss: 0.0054, Val Loss: 0.0111
[IL] Epoch 16/100, Loss: 0.0051, Val Loss: 0.0088
[IL] Epoch 17/100, Loss: 0.0050, Val Loss: 0.0136
Extracting GPU stats logs using atop has been completed on r8a30-a06.
Logs are being saved to: /nfs_global/S/yangrongzheng/atop-740901-r8a30-a06-gpustat.log
[IL] Epoch 18/100, Loss: 0.0050, Val Loss: 0.0106
[IL] Epoch 19/100, Loss: 0.0053, Val Loss: 0.0067
[IL] Epoch 20/100, Loss: 0.0046, Val Loss: 0.0092
[IL] Epoch 21/100, Loss: 0.0048, Val Loss: 0.0080
[IL] Epoch 22/100, Loss: 0.0051, Val Loss: 0.0113
[IL] Epoch 23/100, Loss: 0.0053, Val Loss: 0.0095
[IL] Epoch 24/100, Loss: 0.0048, Val Loss: 0.0082
[IL] Epoch 25/100, Loss: 0.0048, Val Loss: 0.0057
[IL] Epoch 26/100, Loss: 0.0048, Val Loss: 0.0087
[IL] Epoch 27/100, Loss: 0.0051, Val Loss: 0.0082
[IL] Epoch 28/100, Loss: 0.0045, Val Loss: 0.0074
[IL] Epoch 29/100, Loss: 0.0047, Val Loss: 0.0079
[IL] Epoch 30/100, Loss: 0.0048, Val Loss: 0.0066
[IL] Epoch 31/100, Loss: 0.0047, Val Loss: 0.0081
[IL] Epoch 32/100, Loss: 0.0050, Val Loss: 0.0076
[IL] Epoch 33/100, Loss: 0.0047, Val Loss: 0.0092
[IL] Epoch 34/100, Loss: 0.0046, Val Loss: 0.0067
[IL] Epoch 35/100, Loss: 0.0049, Val Loss: 0.0104
[IL] Epoch 36/100, Loss: 0.0046, Val Loss: 0.0070
[IL] Epoch 37/100, Loss: 0.0045, Val Loss: 0.0074
[IL] Epoch 38/100, Loss: 0.0043, Val Loss: 0.0080
[IL] Epoch 39/100, Loss: 0.0042, Val Loss: 0.0073
[IL] Epoch 40/100, Loss: 0.0043, Val Loss: 0.0083
[IL] Epoch 41/100, Loss: 0.0048, Val Loss: 0.0090
[IL] Epoch 42/100, Loss: 0.0051, Val Loss: 0.0096
[IL] Epoch 43/100, Loss: 0.0047, Val Loss: 0.0058
[IL] Epoch 44/100, Loss: 0.0048, Val Loss: 0.0117
[IL] Epoch 45/100, Loss: 0.0046, Val Loss: 0.0078
[IL] Epoch 46/100, Loss: 0.0042, Val Loss: 0.0076
[IL] Epoch 47/100, Loss: 0.0052, Val Loss: 0.0066
[IL] Epoch 48/100, Loss: 0.0045, Val Loss: 0.0128
[IL] Epoch 49/100, Loss: 0.0045, Val Loss: 0.0075
[IL] Epoch 50/100, Loss: 0.0047, Val Loss: 0.0080
[IL] Epoch 51/100, Loss: 0.0041, Val Loss: 0.0068
[IL] Epoch 52/100, Loss: 0.0042, Val Loss: 0.0077
[IL] Epoch 53/100, Loss: 0.0042, Val Loss: 0.0090
[IL] Epoch 54/100, Loss: 0.0042, Val Loss: 0.0093
[IL] Epoch 55/100, Loss: 0.0040, Val Loss: 0.0089
[IL] Epoch 56/100, Loss: 0.0042, Val Loss: 0.0091
[IL] Epoch 57/100, Loss: 0.0047, Val Loss: 0.0109
[IL] Epoch 58/100, Loss: 0.0043, Val Loss: 0.0081
[IL] Epoch 59/100, Loss: 0.0045, Val Loss: 0.0069
[IL] Epoch 60/100, Loss: 0.0042, Val Loss: 0.0113
[IL] Epoch 61/100, Loss: 0.0046, Val Loss: 0.0082
[IL] Epoch 62/100, Loss: 0.0043, Val Loss: 0.0096
[IL] Epoch 63/100, Loss: 0.0040, Val Loss: 0.0093
[IL] Epoch 64/100, Loss: 0.0041, Val Loss: 0.0089
[IL] Epoch 65/100, Loss: 0.0044, Val Loss: 0.0071
[IL] Epoch 66/100, Loss: 0.0041, Val Loss: 0.0083
[IL] Epoch 67/100, Loss: 0.0040, Val Loss: 0.0112
[IL] Epoch 68/100, Loss: 0.0042, Val Loss: 0.0087
[IL] Epoch 69/100, Loss: 0.0046, Val Loss: 0.0106
[IL] Epoch 70/100, Loss: 0.0042, Val Loss: 0.0078
[IL] Epoch 71/100, Loss: 0.0048, Val Loss: 0.0061
[IL] Epoch 72/100, Loss: 0.0045, Val Loss: 0.0095
[IL] Epoch 73/100, Loss: 0.0046, Val Loss: 0.0093
[IL] Epoch 74/100, Loss: 0.0041, Val Loss: 0.0071
[IL] Epoch 75/100, Loss: 0.0040, Val Loss: 0.0119
[IL] Epoch 76/100, Loss: 0.0044, Val Loss: 0.0076
[IL] Epoch 77/100, Loss: 0.0039, Val Loss: 0.0091
[IL] Epoch 78/100, Loss: 0.0038, Val Loss: 0.0121
[IL] Epoch 79/100, Loss: 0.0039, Val Loss: 0.0097
[IL] Epoch 80/100, Loss: 0.0040, Val Loss: 0.0117
[IL] Epoch 81/100, Loss: 0.0041, Val Loss: 0.0083
[IL] Epoch 82/100, Loss: 0.0044, Val Loss: 0.0109
[IL] Epoch 83/100, Loss: 0.0042, Val Loss: 0.0107
[IL] Epoch 84/100, Loss: 0.0041, Val Loss: 0.0096
[IL] Epoch 85/100, Loss: 0.0038, Val Loss: 0.0076
[IL] Epoch 86/100, Loss: 0.0042, Val Loss: 0.0187
[IL] Epoch 87/100, Loss: 0.0041, Val Loss: 0.0094
[IL] Epoch 88/100, Loss: 0.0045, Val Loss: 0.0097
[IL] Epoch 89/100, Loss: 0.0046, Val Loss: 0.0100
[IL] Epoch 90/100, Loss: 0.0040, Val Loss: 0.0075
[IL] Epoch 91/100, Loss: 0.0042, Val Loss: 0.0063
[IL] Epoch 92/100, Loss: 0.0038, Val Loss: 0.0124
[IL] Epoch 93/100, Loss: 0.0040, Val Loss: 0.0089
[IL] Epoch 94/100, Loss: 0.0039, Val Loss: 0.0132
[IL] Epoch 95/100, Loss: 0.0037, Val Loss: 0.0075
[IL] Epoch 96/100, Loss: 0.0043, Val Loss: 0.0125
[IL] Epoch 97/100, Loss: 0.0045, Val Loss: 0.0075
[IL] Epoch 98/100, Loss: 0.0038, Val Loss: 0.0127
[IL] Epoch 99/100, Loss: 0.0037, Val Loss: 0.0094
test_mean_score: 0.82
[IL] Eval - Success Rate: 0.820
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_03.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_3.ckpt

================================================================================
               OFFLINE RL ITERATION 5/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 4)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 32810 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-81.90551 | val=0.00040 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-92.38055 | val=0.00031 | no-improve=0/5
[TransitionModel] Epoch   29 | train=-95.64887 | val=0.00030 | no-improve=5/5
[TransitionModel] Training complete. Elites=[2, 5, 3, 4, 0], val_loss=0.00027
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_04.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_04.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 4)
[IQL] Epoch 0/20, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 1/20, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 2/20, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 3/20, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 4/20, V Loss: 0.0001, Q Loss: 0.0052
[IQL] Epoch 5/20, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 6/20, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 7/20, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 8/20, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 9/20, V Loss: 0.0001, Q Loss: 0.0053
[IQL] Epoch 10/20, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 11/20, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 12/20, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 13/20, V Loss: 0.0001, Q Loss: 0.0052
[IQL] Epoch 14/20, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 15/20, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 16/20, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 17/20, V Loss: 0.0001, Q Loss: 0.0054
[IQL] Epoch 18/20, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 19/20, V Loss: 0.0002, Q Loss: 0.0053
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_04.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_04.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 4)
[OPE] Behavior policy value J_old = 0.4639
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 1.00e-05
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 129 mini-batches, 32810 samples, raw advantage mean=0.0058, std=0.0145
[Offline RL] Epoch 0/3, PPO Loss: -0.0260, PostKL: 6.610e-02, PostClipFrac: 0.298769, PostMeanRatio: 0.997555, PostRatioDev: 1.987e-01, GradNorm: 16.1472, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 1/3, PPO Loss: -0.0076, PostKL: 1.093e-01, PostClipFrac: 0.296912, PostMeanRatio: 1.027857, PostRatioDev: 2.286e-01, GradNorm: 14.0937, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 2/3, PPO Loss: 0.0045, PostKL: 1.553e-01, PostClipFrac: 0.287328, PostMeanRatio: 1.077080, PostRatioDev: 2.701e-01, GradNorm: 13.1569, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_04.png
[OPE] Policy REJECTED: J_new=0.4639 ≤ J_old=0.4639 + δ=0.0046. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 4)
[Collect] 20 episodes, success=0.600, env_return=980.61, rl_reward=0.60, steps=4000
[Data Collection] Success Rate: 0.600, EnvReturn: 980.61, RLReward: 0.60, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 20 episodes remain in RL replay; IL retrain keeps 12/20 successful episodes (drops 8 failures).
[Dataset] Merged 20 episodes (4000 steps) → total 40000 steps, 200 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0049, Val Loss: 0.0078
[IL] Epoch 1/100, Loss: 0.0046, Val Loss: 0.0066
[IL] Epoch 2/100, Loss: 0.0046, Val Loss: 0.0081
[IL] Epoch 3/100, Loss: 0.0047, Val Loss: 0.0077
[IL] Epoch 4/100, Loss: 0.0045, Val Loss: 0.0076
[IL] Epoch 5/100, Loss: 0.0043, Val Loss: 0.0093
[IL] Epoch 6/100, Loss: 0.0042, Val Loss: 0.0093
[IL] Epoch 7/100, Loss: 0.0044, Val Loss: 0.0063
[IL] Epoch 8/100, Loss: 0.0044, Val Loss: 0.0093
[IL] Epoch 9/100, Loss: 0.0041, Val Loss: 0.0125
[IL] Epoch 10/100, Loss: 0.0041, Val Loss: 0.0072
[IL] Epoch 11/100, Loss: 0.0044, Val Loss: 0.0083
[IL] Epoch 12/100, Loss: 0.0044, Val Loss: 0.0072
[IL] Epoch 13/100, Loss: 0.0041, Val Loss: 0.0071
[IL] Epoch 14/100, Loss: 0.0042, Val Loss: 0.0103
[IL] Epoch 15/100, Loss: 0.0040, Val Loss: 0.0086
[IL] Epoch 16/100, Loss: 0.0042, Val Loss: 0.0079
[IL] Epoch 17/100, Loss: 0.0043, Val Loss: 0.0085
[IL] Epoch 18/100, Loss: 0.0041, Val Loss: 0.0084
[IL] Epoch 19/100, Loss: 0.0041, Val Loss: 0.0085
[IL] Epoch 20/100, Loss: 0.0040, Val Loss: 0.0063
[IL] Epoch 21/100, Loss: 0.0042, Val Loss: 0.0093
[IL] Epoch 22/100, Loss: 0.0039, Val Loss: 0.0086
[IL] Epoch 23/100, Loss: 0.0042, Val Loss: 0.0088
[IL] Epoch 24/100, Loss: 0.0042, Val Loss: 0.0079
[IL] Epoch 25/100, Loss: 0.0044, Val Loss: 0.0107
[IL] Epoch 26/100, Loss: 0.0041, Val Loss: 0.0086
[IL] Epoch 27/100, Loss: 0.0039, Val Loss: 0.0084
[IL] Epoch 28/100, Loss: 0.0041, Val Loss: 0.0078
[IL] Epoch 29/100, Loss: 0.0040, Val Loss: 0.0063
[IL] Epoch 30/100, Loss: 0.0039, Val Loss: 0.0073
[IL] Epoch 31/100, Loss: 0.0040, Val Loss: 0.0081
[IL] Epoch 32/100, Loss: 0.0038, Val Loss: 0.0123
[IL] Epoch 33/100, Loss: 0.0039, Val Loss: 0.0096
[IL] Epoch 34/100, Loss: 0.0042, Val Loss: 0.0108
[IL] Epoch 35/100, Loss: 0.0040, Val Loss: 0.0102
[IL] Epoch 36/100, Loss: 0.0041, Val Loss: 0.0099
[IL] Epoch 37/100, Loss: 0.0040, Val Loss: 0.0104
[IL] Epoch 38/100, Loss: 0.0039, Val Loss: 0.0098
[IL] Epoch 39/100, Loss: 0.0040, Val Loss: 0.0070
[IL] Epoch 40/100, Loss: 0.0042, Val Loss: 0.0115
[IL] Epoch 41/100, Loss: 0.0043, Val Loss: 0.0096
[IL] Epoch 42/100, Loss: 0.0039, Val Loss: 0.0099
[IL] Epoch 43/100, Loss: 0.0036, Val Loss: 0.0116
[IL] Epoch 44/100, Loss: 0.0039, Val Loss: 0.0069
[IL] Epoch 45/100, Loss: 0.0039, Val Loss: 0.0087
[IL] Epoch 46/100, Loss: 0.0036, Val Loss: 0.0095
[IL] Epoch 47/100, Loss: 0.0039, Val Loss: 0.0092
[IL] Epoch 48/100, Loss: 0.0035, Val Loss: 0.0091
[IL] Epoch 49/100, Loss: 0.0035, Val Loss: 0.0128
[IL] Epoch 50/100, Loss: 0.0040, Val Loss: 0.0121
[IL] Epoch 51/100, Loss: 0.0038, Val Loss: 0.0097
[IL] Epoch 52/100, Loss: 0.0036, Val Loss: 0.0096
[IL] Epoch 53/100, Loss: 0.0038, Val Loss: 0.0118
[IL] Epoch 54/100, Loss: 0.0036, Val Loss: 0.0122
[IL] Epoch 55/100, Loss: 0.0038, Val Loss: 0.0080
[IL] Epoch 56/100, Loss: 0.0037, Val Loss: 0.0057
[IL] Epoch 57/100, Loss: 0.0037, Val Loss: 0.0069
[IL] Epoch 58/100, Loss: 0.0043, Val Loss: 0.0089
[IL] Epoch 59/100, Loss: 0.0040, Val Loss: 0.0094
[IL] Epoch 60/100, Loss: 0.0039, Val Loss: 0.0090
[IL] Epoch 61/100, Loss: 0.0040, Val Loss: 0.0114
[IL] Epoch 62/100, Loss: 0.0038, Val Loss: 0.0060
[IL] Epoch 63/100, Loss: 0.0035, Val Loss: 0.0061
[IL] Epoch 64/100, Loss: 0.0036, Val Loss: 0.0083
[IL] Epoch 65/100, Loss: 0.0037, Val Loss: 0.0068
[IL] Epoch 66/100, Loss: 0.0037, Val Loss: 0.0073
[IL] Epoch 67/100, Loss: 0.0035, Val Loss: 0.0120
[IL] Epoch 68/100, Loss: 0.0037, Val Loss: 0.0073
[IL] Epoch 69/100, Loss: 0.0034, Val Loss: 0.0103
[IL] Epoch 70/100, Loss: 0.0040, Val Loss: 0.0101
[IL] Epoch 71/100, Loss: 0.0038, Val Loss: 0.0091
[IL] Epoch 72/100, Loss: 0.0036, Val Loss: 0.0086
[IL] Epoch 73/100, Loss: 0.0037, Val Loss: 0.0094
[IL] Epoch 74/100, Loss: 0.0037, Val Loss: 0.0077
[IL] Epoch 75/100, Loss: 0.0039, Val Loss: 0.0083
[IL] Epoch 76/100, Loss: 0.0033, Val Loss: 0.0085
[IL] Epoch 77/100, Loss: 0.0034, Val Loss: 0.0096
[IL] Epoch 78/100, Loss: 0.0038, Val Loss: 0.0068
[IL] Epoch 79/100, Loss: 0.0037, Val Loss: 0.0131
[IL] Epoch 80/100, Loss: 0.0036, Val Loss: 0.0149
[IL] Epoch 81/100, Loss: 0.0040, Val Loss: 0.0066
[IL] Epoch 82/100, Loss: 0.0037, Val Loss: 0.0067
[IL] Epoch 83/100, Loss: 0.0037, Val Loss: 0.0106
[IL] Epoch 84/100, Loss: 0.0035, Val Loss: 0.0114
[IL] Epoch 85/100, Loss: 0.0034, Val Loss: 0.0148
[IL] Epoch 86/100, Loss: 0.0036, Val Loss: 0.0086
[IL] Epoch 87/100, Loss: 0.0037, Val Loss: 0.0092
[IL] Epoch 88/100, Loss: 0.0040, Val Loss: 0.0105
[IL] Epoch 89/100, Loss: 0.0035, Val Loss: 0.0073
[IL] Epoch 90/100, Loss: 0.0034, Val Loss: 0.0077
[IL] Epoch 91/100, Loss: 0.0036, Val Loss: 0.0140
[IL] Epoch 92/100, Loss: 0.0038, Val Loss: 0.0070
[IL] Epoch 93/100, Loss: 0.0038, Val Loss: 0.0098
[IL] Epoch 94/100, Loss: 0.0036, Val Loss: 0.0108
[IL] Epoch 95/100, Loss: 0.0036, Val Loss: 0.0124
[IL] Epoch 96/100, Loss: 0.0039, Val Loss: 0.0090
[IL] Epoch 97/100, Loss: 0.0039, Val Loss: 0.0096
[IL] Epoch 98/100, Loss: 0.0035, Val Loss: 0.0121
[IL] Epoch 99/100, Loss: 0.0031, Val Loss: 0.0113
test_mean_score: 0.76
[IL] Eval - Success Rate: 0.760
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_04.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_4.ckpt

================================================================================
               OFFLINE RL ITERATION 6/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 5)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 36670 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-90.51939 | val=0.00055 | no-improve=0/5
[TransitionModel] Epoch   15 | train=-100.80250 | val=0.00049 | no-improve=5/5
[TransitionModel] Training complete. Elites=[3, 1, 2, 6, 0], val_loss=0.00044
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_05.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_05.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 5)
[IQL] Epoch 0/20, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 1/20, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 2/20, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 3/20, V Loss: 0.0001, Q Loss: 0.0049
[IQL] Epoch 4/20, V Loss: 0.0001, Q Loss: 0.0051
[IQL] Epoch 5/20, V Loss: 0.0001, Q Loss: 0.0050
[IQL] Epoch 6/20, V Loss: 0.0001, Q Loss: 0.0050
[IQL] Epoch 7/20, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 8/20, V Loss: 0.0001, Q Loss: 0.0050
[IQL] Epoch 9/20, V Loss: 0.0001, Q Loss: 0.0050
[IQL] Epoch 10/20, V Loss: 0.0001, Q Loss: 0.0050
[IQL] Epoch 11/20, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 12/20, V Loss: 0.0001, Q Loss: 0.0052
[IQL] Epoch 13/20, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 14/20, V Loss: 0.0001, Q Loss: 0.0050
[IQL] Epoch 15/20, V Loss: 0.0001, Q Loss: 0.0051
[IQL] Epoch 16/20, V Loss: 0.0001, Q Loss: 0.0050
[IQL] Epoch 17/20, V Loss: 0.0001, Q Loss: 0.0050
[IQL] Epoch 18/20, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 19/20, V Loss: 0.0002, Q Loss: 0.0049
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_05.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_05.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 5)
[OPE] Behavior policy value J_old = 0.4281
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 1.00e-05
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 144 mini-batches, 36670 samples, raw advantage mean=0.0052, std=0.0196
[Offline RL] Epoch 0/3, PPO Loss: -0.0276, PostKL: 5.306e-02, PostClipFrac: 0.266893, PostMeanRatio: 0.995936, PostRatioDev: 1.798e-01, GradNorm: 17.4797, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 1/3, PPO Loss: -0.0081, PostKL: 7.712e-02, PostClipFrac: 0.265932, PostMeanRatio: 1.008784, PostRatioDev: 1.916e-01, GradNorm: 15.3196, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 2/3, PPO Loss: 0.0075, PostKL: 8.319e-02, PostClipFrac: 0.254165, PostMeanRatio: 1.019640, PostRatioDev: 1.909e-01, GradNorm: 13.8551, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_05.png
[OPE] Policy REJECTED: J_new=0.4281 ≤ J_old=0.4281 + δ=0.0043. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 5)
[Collect] 20 episodes, success=0.650, env_return=1204.28, rl_reward=0.65, steps=4000
[Data Collection] Success Rate: 0.650, EnvReturn: 1204.28, RLReward: 0.65, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 20 episodes remain in RL replay; IL retrain keeps 13/20 successful episodes (drops 7 failures).
[Dataset] Merged 20 episodes (4000 steps) → total 44000 steps, 220 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0046, Val Loss: 0.0116
[IL] Epoch 1/100, Loss: 0.0041, Val Loss: 0.0087
[IL] Epoch 2/100, Loss: 0.0044, Val Loss: 0.0059
[IL] Epoch 3/100, Loss: 0.0043, Val Loss: 0.0083
[IL] Epoch 4/100, Loss: 0.0042, Val Loss: 0.0088
[IL] Epoch 5/100, Loss: 0.0042, Val Loss: 0.0073
[IL] Epoch 6/100, Loss: 0.0040, Val Loss: 0.0096
[IL] Epoch 7/100, Loss: 0.0040, Val Loss: 0.0091
[IL] Epoch 8/100, Loss: 0.0038, Val Loss: 0.0097
[IL] Epoch 9/100, Loss: 0.0038, Val Loss: 0.0082
[IL] Epoch 10/100, Loss: 0.0037, Val Loss: 0.0121
[IL] Epoch 11/100, Loss: 0.0035, Val Loss: 0.0101
[IL] Epoch 12/100, Loss: 0.0039, Val Loss: 0.0088
[IL] Epoch 13/100, Loss: 0.0038, Val Loss: 0.0076
[IL] Epoch 14/100, Loss: 0.0038, Val Loss: 0.0111
[IL] Epoch 15/100, Loss: 0.0040, Val Loss: 0.0060
[IL] Epoch 16/100, Loss: 0.0035, Val Loss: 0.0096
[IL] Epoch 17/100, Loss: 0.0037, Val Loss: 0.0076
[IL] Epoch 18/100, Loss: 0.0033, Val Loss: 0.0144
[IL] Epoch 19/100, Loss: 0.0037, Val Loss: 0.0113
[IL] Epoch 20/100, Loss: 0.0036, Val Loss: 0.0118
[IL] Epoch 21/100, Loss: 0.0037, Val Loss: 0.0092
[IL] Epoch 22/100, Loss: 0.0038, Val Loss: 0.0081
[IL] Epoch 23/100, Loss: 0.0037, Val Loss: 0.0085
[IL] Epoch 24/100, Loss: 0.0033, Val Loss: 0.0120
[IL] Epoch 25/100, Loss: 0.0039, Val Loss: 0.0090
[IL] Epoch 26/100, Loss: 0.0035, Val Loss: 0.0126
[IL] Epoch 27/100, Loss: 0.0038, Val Loss: 0.0085
[IL] Epoch 28/100, Loss: 0.0038, Val Loss: 0.0065
[IL] Epoch 29/100, Loss: 0.0037, Val Loss: 0.0106
[IL] Epoch 30/100, Loss: 0.0036, Val Loss: 0.0080
[IL] Epoch 31/100, Loss: 0.0035, Val Loss: 0.0078
[IL] Epoch 32/100, Loss: 0.0038, Val Loss: 0.0095
[IL] Epoch 33/100, Loss: 0.0036, Val Loss: 0.0094
[IL] Epoch 34/100, Loss: 0.0038, Val Loss: 0.0072
[IL] Epoch 35/100, Loss: 0.0035, Val Loss: 0.0083
[IL] Epoch 36/100, Loss: 0.0039, Val Loss: 0.0061
[IL] Epoch 37/100, Loss: 0.0036, Val Loss: 0.0070
[IL] Epoch 38/100, Loss: 0.0035, Val Loss: 0.0072
[IL] Epoch 39/100, Loss: 0.0038, Val Loss: 0.0096
[IL] Epoch 40/100, Loss: 0.0034, Val Loss: 0.0075
[IL] Epoch 41/100, Loss: 0.0033, Val Loss: 0.0083
[IL] Epoch 42/100, Loss: 0.0034, Val Loss: 0.0109
[IL] Epoch 43/100, Loss: 0.0034, Val Loss: 0.0135
[IL] Epoch 44/100, Loss: 0.0035, Val Loss: 0.0080
[IL] Epoch 45/100, Loss: 0.0034, Val Loss: 0.0068
[IL] Epoch 46/100, Loss: 0.0037, Val Loss: 0.0095
[IL] Epoch 47/100, Loss: 0.0038, Val Loss: 0.0108
[IL] Epoch 48/100, Loss: 0.0037, Val Loss: 0.0091
[IL] Epoch 49/100, Loss: 0.0035, Val Loss: 0.0144
[IL] Epoch 50/100, Loss: 0.0050, Val Loss: 0.0134
[IL] Epoch 51/100, Loss: 0.0036, Val Loss: 0.0106
[IL] Epoch 52/100, Loss: 0.0033, Val Loss: 0.0090
[IL] Epoch 53/100, Loss: 0.0036, Val Loss: 0.0107
[IL] Epoch 54/100, Loss: 0.0032, Val Loss: 0.0081
[IL] Epoch 55/100, Loss: 0.0040, Val Loss: 0.0111
[IL] Epoch 56/100, Loss: 0.0040, Val Loss: 0.0094
[IL] Epoch 57/100, Loss: 0.0032, Val Loss: 0.0101
[IL] Epoch 58/100, Loss: 0.0032, Val Loss: 0.0125
[IL] Epoch 59/100, Loss: 0.0029, Val Loss: 0.0097
[IL] Epoch 60/100, Loss: 0.0036, Val Loss: 0.0085
[IL] Epoch 61/100, Loss: 0.0037, Val Loss: 0.0114
[IL] Epoch 62/100, Loss: 0.0035, Val Loss: 0.0110
[IL] Epoch 63/100, Loss: 0.0040, Val Loss: 0.0084
[IL] Epoch 64/100, Loss: 0.0032, Val Loss: 0.0119
[IL] Epoch 65/100, Loss: 0.0036, Val Loss: 0.0081
[IL] Epoch 66/100, Loss: 0.0033, Val Loss: 0.0077
[IL] Epoch 67/100, Loss: 0.0035, Val Loss: 0.0091
[IL] Epoch 68/100, Loss: 0.0033, Val Loss: 0.0127
[IL] Epoch 69/100, Loss: 0.0032, Val Loss: 0.0157
[IL] Epoch 70/100, Loss: 0.0034, Val Loss: 0.0068
[IL] Epoch 71/100, Loss: 0.0036, Val Loss: 0.0131
[IL] Epoch 72/100, Loss: 0.0037, Val Loss: 0.0083
[IL] Epoch 73/100, Loss: 0.0036, Val Loss: 0.0123
[IL] Epoch 74/100, Loss: 0.0035, Val Loss: 0.0072
[IL] Epoch 75/100, Loss: 0.0032, Val Loss: 0.0103
[IL] Epoch 76/100, Loss: 0.0036, Val Loss: 0.0094
[IL] Epoch 77/100, Loss: 0.0033, Val Loss: 0.0086
[IL] Epoch 78/100, Loss: 0.0035, Val Loss: 0.0100
[IL] Epoch 79/100, Loss: 0.0032, Val Loss: 0.0106
[IL] Epoch 80/100, Loss: 0.0030, Val Loss: 0.0075
[IL] Epoch 81/100, Loss: 0.0033, Val Loss: 0.0071
[IL] Epoch 82/100, Loss: 0.0036, Val Loss: 0.0099
[IL] Epoch 83/100, Loss: 0.0037, Val Loss: 0.0080
[IL] Epoch 84/100, Loss: 0.0032, Val Loss: 0.0078
[IL] Epoch 85/100, Loss: 0.0031, Val Loss: 0.0085
[IL] Epoch 86/100, Loss: 0.0032, Val Loss: 0.0091
[IL] Epoch 87/100, Loss: 0.0031, Val Loss: 0.0098
[IL] Epoch 88/100, Loss: 0.0034, Val Loss: 0.0070
[IL] Epoch 89/100, Loss: 0.0035, Val Loss: 0.0078
[IL] Epoch 90/100, Loss: 0.0034, Val Loss: 0.0137
[IL] Epoch 91/100, Loss: 0.0033, Val Loss: 0.0105
[IL] Epoch 92/100, Loss: 0.0034, Val Loss: 0.0091
[IL] Epoch 93/100, Loss: 0.0037, Val Loss: 0.0095
[IL] Epoch 94/100, Loss: 0.0033, Val Loss: 0.0078
[IL] Epoch 95/100, Loss: 0.0034, Val Loss: 0.0117
[IL] Epoch 96/100, Loss: 0.0032, Val Loss: 0.0095
[IL] Epoch 97/100, Loss: 0.0032, Val Loss: 0.0143
[IL] Epoch 98/100, Loss: 0.0033, Val Loss: 0.0082
[IL] Epoch 99/100, Loss: 0.0033, Val Loss: 0.0114
test_mean_score: 0.85
[IL] Eval - Success Rate: 0.850
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_05.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_5.ckpt

================================================================================
               OFFLINE RL ITERATION 7/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 6)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 40530 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-97.64641 | val=0.00034 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-108.76204 | val=0.00031 | no-improve=1/5
[TransitionModel] Epoch   40 | train=-117.34272 | val=0.00028 | no-improve=0/5
[TransitionModel] Epoch   45 | train=-119.49972 | val=0.00029 | no-improve=5/5
[TransitionModel] Training complete. Elites=[1, 3, 6, 4, 2], val_loss=0.00027
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_06.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_06.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 6)
[IQL] Epoch 0/20, V Loss: 0.0001, Q Loss: 0.0049
[IQL] Epoch 1/20, V Loss: 0.0001, Q Loss: 0.0049
[IQL] Epoch 2/20, V Loss: 0.0001, Q Loss: 0.0049
[IQL] Epoch 3/20, V Loss: 0.0001, Q Loss: 0.0049
[IQL] Epoch 4/20, V Loss: 0.0002, Q Loss: 0.0049
[IQL] Epoch 5/20, V Loss: 0.0002, Q Loss: 0.0049
[IQL] Epoch 6/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 7/20, V Loss: 0.0002, Q Loss: 0.0049
[IQL] Epoch 8/20, V Loss: 0.0001, Q Loss: 0.0050
[IQL] Epoch 9/20, V Loss: 0.0002, Q Loss: 0.0049
[IQL] Epoch 10/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 11/20, V Loss: 0.0002, Q Loss: 0.0049
[IQL] Epoch 12/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 13/20, V Loss: 0.0001, Q Loss: 0.0049
[IQL] Epoch 14/20, V Loss: 0.0002, Q Loss: 0.0048
[IQL] Epoch 15/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 16/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 17/20, V Loss: 0.0002, Q Loss: 0.0049
[IQL] Epoch 18/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 19/20, V Loss: 0.0001, Q Loss: 0.0049
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_06.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_06.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 6)
[OPE] Behavior policy value J_old = 0.4178
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 1.00e-05
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 159 mini-batches, 40530 samples, raw advantage mean=-0.0117, std=0.0116
[Offline RL] Epoch 0/3, PPO Loss: -0.0217, PostKL: 6.745e-02, PostClipFrac: 0.275203, PostMeanRatio: 0.996094, PostRatioDev: 1.833e-01, GradNorm: 16.1773, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 1/3, PPO Loss: -0.0045, PostKL: 9.758e-02, PostClipFrac: 0.270956, PostMeanRatio: 1.008288, PostRatioDev: 1.953e-01, GradNorm: 13.2812, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 2/3, PPO Loss: 0.0067, PostKL: 9.417e-02, PostClipFrac: 0.266379, PostMeanRatio: 1.016911, PostRatioDev: 1.999e-01, GradNorm: 12.2965, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_06.png
[OPE] Policy REJECTED: J_new=0.4178 ≤ J_old=0.4178 + δ=0.0042. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 6)
[Collect] 20 episodes, success=0.750, env_return=984.91, rl_reward=0.75, steps=4000
[Data Collection] Success Rate: 0.750, EnvReturn: 984.91, RLReward: 0.75, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 20 episodes remain in RL replay; IL retrain keeps 15/20 successful episodes (drops 5 failures).
[Dataset] Merged 20 episodes (4000 steps) → total 48000 steps, 240 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0048, Val Loss: 0.0070
[IL] Epoch 1/100, Loss: 0.0038, Val Loss: 0.0092
[IL] Epoch 2/100, Loss: 0.0041, Val Loss: 0.0088
[IL] Epoch 3/100, Loss: 0.0039, Val Loss: 0.0072
[IL] Epoch 4/100, Loss: 0.0038, Val Loss: 0.0081
[IL] Epoch 5/100, Loss: 0.0039, Val Loss: 0.0101
[IL] Epoch 6/100, Loss: 0.0039, Val Loss: 0.0085
[IL] Epoch 7/100, Loss: 0.0038, Val Loss: 0.0108
[IL] Epoch 8/100, Loss: 0.0036, Val Loss: 0.0083
[IL] Epoch 9/100, Loss: 0.0039, Val Loss: 0.0069
[IL] Epoch 10/100, Loss: 0.0036, Val Loss: 0.0078
[IL] Epoch 11/100, Loss: 0.0039, Val Loss: 0.0076
[IL] Epoch 12/100, Loss: 0.0039, Val Loss: 0.0064
[IL] Epoch 13/100, Loss: 0.0038, Val Loss: 0.0096
[IL] Epoch 14/100, Loss: 0.0036, Val Loss: 0.0129
[IL] Epoch 15/100, Loss: 0.0037, Val Loss: 0.0093
[IL] Epoch 16/100, Loss: 0.0036, Val Loss: 0.0081
[IL] Epoch 17/100, Loss: 0.0036, Val Loss: 0.0092
[IL] Epoch 18/100, Loss: 0.0033, Val Loss: 0.0066
[IL] Epoch 19/100, Loss: 0.0035, Val Loss: 0.0076
[IL] Epoch 20/100, Loss: 0.0034, Val Loss: 0.0099
[IL] Epoch 21/100, Loss: 0.0035, Val Loss: 0.0080
[IL] Epoch 22/100, Loss: 0.0036, Val Loss: 0.0074
[IL] Epoch 23/100, Loss: 0.0033, Val Loss: 0.0136
[IL] Epoch 24/100, Loss: 0.0037, Val Loss: 0.0088
[IL] Epoch 25/100, Loss: 0.0035, Val Loss: 0.0112
[IL] Epoch 26/100, Loss: 0.0033, Val Loss: 0.0066
[IL] Epoch 27/100, Loss: 0.0034, Val Loss: 0.0080
[IL] Epoch 28/100, Loss: 0.0033, Val Loss: 0.0057
[IL] Epoch 29/100, Loss: 0.0034, Val Loss: 0.0082
[IL] Epoch 30/100, Loss: 0.0036, Val Loss: 0.0093
[IL] Epoch 31/100, Loss: 0.0037, Val Loss: 0.0072
[IL] Epoch 32/100, Loss: 0.0037, Val Loss: 0.0094
[IL] Epoch 33/100, Loss: 0.0036, Val Loss: 0.0078
[IL] Epoch 34/100, Loss: 0.0034, Val Loss: 0.0086
[IL] Epoch 35/100, Loss: 0.0031, Val Loss: 0.0089
[IL] Epoch 36/100, Loss: 0.0030, Val Loss: 0.0130
[IL] Epoch 37/100, Loss: 0.0036, Val Loss: 0.0114
[IL] Epoch 38/100, Loss: 0.0034, Val Loss: 0.0088
[IL] Epoch 39/100, Loss: 0.0036, Val Loss: 0.0085
[IL] Epoch 40/100, Loss: 0.0035, Val Loss: 0.0108
[IL] Epoch 41/100, Loss: 0.0036, Val Loss: 0.0073
[IL] Epoch 42/100, Loss: 0.0034, Val Loss: 0.0091
[IL] Epoch 43/100, Loss: 0.0036, Val Loss: 0.0076
[IL] Epoch 44/100, Loss: 0.0035, Val Loss: 0.0123
[IL] Epoch 45/100, Loss: 0.0033, Val Loss: 0.0091
[IL] Epoch 46/100, Loss: 0.0033, Val Loss: 0.0113
[IL] Epoch 47/100, Loss: 0.0031, Val Loss: 0.0122
[IL] Epoch 48/100, Loss: 0.0034, Val Loss: 0.0083
[IL] Epoch 49/100, Loss: 0.0034, Val Loss: 0.0097
[IL] Epoch 50/100, Loss: 0.0034, Val Loss: 0.0085
[IL] Epoch 51/100, Loss: 0.0034, Val Loss: 0.0118
[IL] Epoch 52/100, Loss: 0.0032, Val Loss: 0.0089
[IL] Epoch 53/100, Loss: 0.0032, Val Loss: 0.0123
[IL] Epoch 54/100, Loss: 0.0030, Val Loss: 0.0109
[IL] Epoch 55/100, Loss: 0.0032, Val Loss: 0.0106
[IL] Epoch 56/100, Loss: 0.0033, Val Loss: 0.0051
[IL] Epoch 57/100, Loss: 0.0031, Val Loss: 0.0114
[IL] Epoch 58/100, Loss: 0.0033, Val Loss: 0.0076
[IL] Epoch 59/100, Loss: 0.0033, Val Loss: 0.0095
[IL] Epoch 60/100, Loss: 0.0031, Val Loss: 0.0113
[IL] Epoch 61/100, Loss: 0.0030, Val Loss: 0.0056
[IL] Epoch 62/100, Loss: 0.0032, Val Loss: 0.0121
[IL] Epoch 63/100, Loss: 0.0034, Val Loss: 0.0093
[IL] Epoch 64/100, Loss: 0.0032, Val Loss: 0.0099
[IL] Epoch 65/100, Loss: 0.0033, Val Loss: 0.0103
[IL] Epoch 66/100, Loss: 0.0032, Val Loss: 0.0094
[IL] Epoch 67/100, Loss: 0.0032, Val Loss: 0.0084
[IL] Epoch 68/100, Loss: 0.0032, Val Loss: 0.0064
[IL] Epoch 69/100, Loss: 0.0037, Val Loss: 0.0110
[IL] Epoch 70/100, Loss: 0.0030, Val Loss: 0.0068
[IL] Epoch 71/100, Loss: 0.0032, Val Loss: 0.0102
[IL] Epoch 72/100, Loss: 0.0033, Val Loss: 0.0077
[IL] Epoch 73/100, Loss: 0.0032, Val Loss: 0.0065
[IL] Epoch 74/100, Loss: 0.0035, Val Loss: 0.0101
[IL] Epoch 75/100, Loss: 0.0035, Val Loss: 0.0079
[IL] Epoch 76/100, Loss: 0.0029, Val Loss: 0.0093
[IL] Epoch 77/100, Loss: 0.0034, Val Loss: 0.0078
[IL] Epoch 78/100, Loss: 0.0034, Val Loss: 0.0087
[IL] Epoch 79/100, Loss: 0.0033, Val Loss: 0.0094
[IL] Epoch 80/100, Loss: 0.0030, Val Loss: 0.0089
[IL] Epoch 81/100, Loss: 0.0031, Val Loss: 0.0089
[IL] Epoch 82/100, Loss: 0.0030, Val Loss: 0.0082
[IL] Epoch 83/100, Loss: 0.0030, Val Loss: 0.0074
[IL] Epoch 84/100, Loss: 0.0030, Val Loss: 0.0099
[IL] Epoch 85/100, Loss: 0.0032, Val Loss: 0.0078
[IL] Epoch 86/100, Loss: 0.0030, Val Loss: 0.0049
[IL] Epoch 87/100, Loss: 0.0034, Val Loss: 0.0065
[IL] Epoch 88/100, Loss: 0.0030, Val Loss: 0.0059
[IL] Epoch 89/100, Loss: 0.0031, Val Loss: 0.0134
[IL] Epoch 90/100, Loss: 0.0033, Val Loss: 0.0115
[IL] Epoch 91/100, Loss: 0.0032, Val Loss: 0.0077
[IL] Epoch 92/100, Loss: 0.0030, Val Loss: 0.0093
[IL] Epoch 93/100, Loss: 0.0031, Val Loss: 0.0091
[IL] Epoch 94/100, Loss: 0.0031, Val Loss: 0.0168
[IL] Epoch 95/100, Loss: 0.0029, Val Loss: 0.0116
[IL] Epoch 96/100, Loss: 0.0029, Val Loss: 0.0096
[IL] Epoch 97/100, Loss: 0.0031, Val Loss: 0.0089
[IL] Epoch 98/100, Loss: 0.0034, Val Loss: 0.0100
[IL] Epoch 99/100, Loss: 0.0035, Val Loss: 0.0076
test_mean_score: 0.84
[IL] Eval - Success Rate: 0.840
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_06.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_6.ckpt

================================================================================
               OFFLINE RL ITERATION 8/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 7)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 44390 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-114.75710 | val=0.00037 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-128.36311 | val=0.00036 | no-improve=5/5
[TransitionModel] Training complete. Elites=[4, 0, 5, 3, 1], val_loss=0.00032
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_07.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_07.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 7)
[IQL] Epoch 0/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 1/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 2/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 3/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 4/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 5/20, V Loss: 0.0002, Q Loss: 0.0048
[IQL] Epoch 6/20, V Loss: 0.0001, Q Loss: 0.0049
[IQL] Epoch 7/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 8/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 9/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 10/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 11/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 12/20, V Loss: 0.0001, Q Loss: 0.0047
[IQL] Epoch 13/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 14/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 15/20, V Loss: 0.0002, Q Loss: 0.0048
[IQL] Epoch 16/20, V Loss: 0.0002, Q Loss: 0.0048
[IQL] Epoch 17/20, V Loss: 0.0001, Q Loss: 0.0047
[IQL] Epoch 18/20, V Loss: 0.0002, Q Loss: 0.0048
[IQL] Epoch 19/20, V Loss: 0.0001, Q Loss: 0.0049
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_07.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_07.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 7)
[OPE] Behavior policy value J_old = 0.4156
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 1.00e-05
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 174 mini-batches, 44390 samples, raw advantage mean=-0.0140, std=0.0104
[Offline RL] Epoch 0/3, PPO Loss: -0.0214, PostKL: 8.236e-02, PostClipFrac: 0.237084, PostMeanRatio: 1.037753, PostRatioDev: 2.029e-01, GradNorm: 14.9234, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 1/3, PPO Loss: -0.0032, PostKL: 1.495e+00, PostClipFrac: 0.241285, PostMeanRatio: 2.434896, PostRatioDev: 1.606e+00, GradNorm: 13.4826, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 2/3, PPO Loss: 0.0093, PostKL: 4.075e-01, PostClipFrac: 0.242386, PostMeanRatio: 1.349116, PostRatioDev: 5.197e-01, GradNorm: 12.7026, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_07.png
[OPE] Policy REJECTED: J_new=0.4157 ≤ J_old=0.4156 + δ=0.0042. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 7)
[Collect] 20 episodes, success=0.850, env_return=1013.80, rl_reward=0.85, steps=4000
[Data Collection] Success Rate: 0.850, EnvReturn: 1013.80, RLReward: 0.85, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 20 episodes remain in RL replay; IL retrain keeps 17/20 successful episodes (drops 3 failures).
[Dataset] Merged 20 episodes (4000 steps) → total 52000 steps, 260 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0041, Val Loss: 0.0117
[IL] Epoch 1/100, Loss: 0.0040, Val Loss: 0.0058
[IL] Epoch 2/100, Loss: 0.0037, Val Loss: 0.0081
[IL] Epoch 3/100, Loss: 0.0040, Val Loss: 0.0097
[IL] Epoch 4/100, Loss: 0.0038, Val Loss: 0.0111
[IL] Epoch 5/100, Loss: 0.0035, Val Loss: 0.0093
[IL] Epoch 6/100, Loss: 0.0033, Val Loss: 0.0101
[IL] Epoch 7/100, Loss: 0.0037, Val Loss: 0.0084
[IL] Epoch 8/100, Loss: 0.0036, Val Loss: 0.0076
[IL] Epoch 9/100, Loss: 0.0036, Val Loss: 0.0109
[IL] Epoch 10/100, Loss: 0.0035, Val Loss: 0.0110
[IL] Epoch 11/100, Loss: 0.0036, Val Loss: 0.0082
[IL] Epoch 12/100, Loss: 0.0033, Val Loss: 0.0092
[IL] Epoch 13/100, Loss: 0.0033, Val Loss: 0.0090
[IL] Epoch 14/100, Loss: 0.0036, Val Loss: 0.0122
[IL] Epoch 15/100, Loss: 0.0039, Val Loss: 0.0121
[IL] Epoch 16/100, Loss: 0.0036, Val Loss: 0.0085
[IL] Epoch 17/100, Loss: 0.0035, Val Loss: 0.0132
[IL] Epoch 18/100, Loss: 0.0032, Val Loss: 0.0061
[IL] Epoch 19/100, Loss: 0.0031, Val Loss: 0.0073
[IL] Epoch 20/100, Loss: 0.0032, Val Loss: 0.0098
[IL] Epoch 21/100, Loss: 0.0031, Val Loss: 0.0107
[IL] Epoch 22/100, Loss: 0.0032, Val Loss: 0.0084
[IL] Epoch 23/100, Loss: 0.0035, Val Loss: 0.0077
[IL] Epoch 24/100, Loss: 0.0036, Val Loss: 0.0064
[IL] Epoch 25/100, Loss: 0.0033, Val Loss: 0.0079
[IL] Epoch 26/100, Loss: 0.0033, Val Loss: 0.0057
[IL] Epoch 27/100, Loss: 0.0034, Val Loss: 0.0082
[IL] Epoch 28/100, Loss: 0.0039, Val Loss: 0.0067
[IL] Epoch 29/100, Loss: 0.0032, Val Loss: 0.0081
[IL] Epoch 30/100, Loss: 0.0032, Val Loss: 0.0069
[IL] Epoch 31/100, Loss: 0.0034, Val Loss: 0.0079
[IL] Epoch 32/100, Loss: 0.0033, Val Loss: 0.0094
[IL] Epoch 33/100, Loss: 0.0030, Val Loss: 0.0086
[IL] Epoch 34/100, Loss: 0.0030, Val Loss: 0.0104
[IL] Epoch 35/100, Loss: 0.0031, Val Loss: 0.0073
[IL] Epoch 36/100, Loss: 0.0032, Val Loss: 0.0075
[IL] Epoch 37/100, Loss: 0.0031, Val Loss: 0.0115
[IL] Epoch 38/100, Loss: 0.0032, Val Loss: 0.0068
[IL] Epoch 39/100, Loss: 0.0033, Val Loss: 0.0104
[IL] Epoch 40/100, Loss: 0.0030, Val Loss: 0.0130
[IL] Epoch 41/100, Loss: 0.0030, Val Loss: 0.0099
[IL] Epoch 42/100, Loss: 0.0033, Val Loss: 0.0103
[IL] Epoch 43/100, Loss: 0.0033, Val Loss: 0.0099
[IL] Epoch 44/100, Loss: 0.0035, Val Loss: 0.0097
[IL] Epoch 45/100, Loss: 0.0029, Val Loss: 0.0080
[IL] Epoch 46/100, Loss: 0.0031, Val Loss: 0.0078
[IL] Epoch 47/100, Loss: 0.0031, Val Loss: 0.0092
[IL] Epoch 48/100, Loss: 0.0032, Val Loss: 0.0070
[IL] Epoch 49/100, Loss: 0.0031, Val Loss: 0.0084
[IL] Epoch 50/100, Loss: 0.0030, Val Loss: 0.0149
[IL] Epoch 51/100, Loss: 0.0032, Val Loss: 0.0069
[IL] Epoch 52/100, Loss: 0.0030, Val Loss: 0.0092
[IL] Epoch 53/100, Loss: 0.0034, Val Loss: 0.0100
[IL] Epoch 54/100, Loss: 0.0035, Val Loss: 0.0111
[IL] Epoch 55/100, Loss: 0.0030, Val Loss: 0.0055
[IL] Epoch 56/100, Loss: 0.0032, Val Loss: 0.0067
[IL] Epoch 57/100, Loss: 0.0028, Val Loss: 0.0096
[IL] Epoch 58/100, Loss: 0.0029, Val Loss: 0.0079
[IL] Epoch 59/100, Loss: 0.0034, Val Loss: 0.0106
[IL] Epoch 60/100, Loss: 0.0032, Val Loss: 0.0107
[IL] Epoch 61/100, Loss: 0.0031, Val Loss: 0.0078
[IL] Epoch 62/100, Loss: 0.0030, Val Loss: 0.0081
[IL] Epoch 63/100, Loss: 0.0029, Val Loss: 0.0103
[IL] Epoch 64/100, Loss: 0.0031, Val Loss: 0.0076
[IL] Epoch 65/100, Loss: 0.0030, Val Loss: 0.0131
[IL] Epoch 66/100, Loss: 0.0030, Val Loss: 0.0106
[IL] Epoch 67/100, Loss: 0.0029, Val Loss: 0.0097
[IL] Epoch 68/100, Loss: 0.0032, Val Loss: 0.0088
[IL] Epoch 69/100, Loss: 0.0028, Val Loss: 0.0092
[IL] Epoch 70/100, Loss: 0.0034, Val Loss: 0.0082
[IL] Epoch 71/100, Loss: 0.0027, Val Loss: 0.0094
[IL] Epoch 72/100, Loss: 0.0028, Val Loss: 0.0106
[IL] Epoch 73/100, Loss: 0.0030, Val Loss: 0.0103
[IL] Epoch 74/100, Loss: 0.0028, Val Loss: 0.0114
[IL] Epoch 75/100, Loss: 0.0030, Val Loss: 0.0091
[IL] Epoch 76/100, Loss: 0.0032, Val Loss: 0.0089
[IL] Epoch 77/100, Loss: 0.0031, Val Loss: 0.0080
[IL] Epoch 78/100, Loss: 0.0031, Val Loss: 0.0093
[IL] Epoch 79/100, Loss: 0.0030, Val Loss: 0.0083
[IL] Epoch 80/100, Loss: 0.0028, Val Loss: 0.0096
[IL] Epoch 81/100, Loss: 0.0034, Val Loss: 0.0148
[IL] Epoch 82/100, Loss: 0.0032, Val Loss: 0.0108
[IL] Epoch 83/100, Loss: 0.0030, Val Loss: 0.0077
[IL] Epoch 84/100, Loss: 0.0030, Val Loss: 0.0079
[IL] Epoch 85/100, Loss: 0.0028, Val Loss: 0.0110
[IL] Epoch 86/100, Loss: 0.0028, Val Loss: 0.0129
[IL] Epoch 87/100, Loss: 0.0028, Val Loss: 0.0110
[IL] Epoch 88/100, Loss: 0.0028, Val Loss: 0.0081
[IL] Epoch 89/100, Loss: 0.0026, Val Loss: 0.0142
[IL] Epoch 90/100, Loss: 0.0028, Val Loss: 0.0098
[IL] Epoch 91/100, Loss: 0.0029, Val Loss: 0.0101
[IL] Epoch 92/100, Loss: 0.0031, Val Loss: 0.0132
[IL] Epoch 93/100, Loss: 0.0029, Val Loss: 0.0075
[IL] Epoch 94/100, Loss: 0.0032, Val Loss: 0.0109
[IL] Epoch 95/100, Loss: 0.0030, Val Loss: 0.0080
[IL] Epoch 96/100, Loss: 0.0031, Val Loss: 0.0089
[IL] Epoch 97/100, Loss: 0.0032, Val Loss: 0.0099
[IL] Epoch 98/100, Loss: 0.0029, Val Loss: 0.0053
[IL] Epoch 99/100, Loss: 0.0029, Val Loss: 0.0129
test_mean_score: 0.83
[IL] Eval - Success Rate: 0.830
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_07.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_7.ckpt

================================================================================
               OFFLINE RL ITERATION 9/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 8)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 48250 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-126.38094 | val=0.00040 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-138.15325 | val=0.00038 | no-improve=1/5
[TransitionModel] Epoch   31 | train=-143.61961 | val=0.00040 | no-improve=5/5
[TransitionModel] Training complete. Elites=[2, 1, 6, 3, 5], val_loss=0.00034
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_08.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_08.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 8)
[IQL] Epoch 0/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 1/20, V Loss: 0.0002, Q Loss: 0.0048
[IQL] Epoch 2/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 3/20, V Loss: 0.0001, Q Loss: 0.0047
[IQL] Epoch 4/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 5/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 6/20, V Loss: 0.0001, Q Loss: 0.0047
[IQL] Epoch 7/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 8/20, V Loss: 0.0001, Q Loss: 0.0047
[IQL] Epoch 9/20, V Loss: 0.0001, Q Loss: 0.0047
[IQL] Epoch 10/20, V Loss: 0.0001, Q Loss: 0.0047
[IQL] Epoch 11/20, V Loss: 0.0001, Q Loss: 0.0047
[IQL] Epoch 12/20, V Loss: 0.0002, Q Loss: 0.0047
[IQL] Epoch 13/20, V Loss: 0.0002, Q Loss: 0.0048
[IQL] Epoch 14/20, V Loss: 0.0001, Q Loss: 0.0047
[IQL] Epoch 15/20, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 16/20, V Loss: 0.0002, Q Loss: 0.0047
[IQL] Epoch 17/20, V Loss: 0.0001, Q Loss: 0.0047
[IQL] Epoch 18/20, V Loss: 0.0002, Q Loss: 0.0047
[IQL] Epoch 19/20, V Loss: 0.0002, Q Loss: 0.0047
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_08.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_08.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 8)
[OPE] Behavior policy value J_old = 0.4319
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 1.00e-05
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 189 mini-batches, 48250 samples, raw advantage mean=0.0005, std=0.0161
[Offline RL] Epoch 0/3, PPO Loss: -0.0213, PostKL: 4.781e-02, PostClipFrac: 0.261167, PostMeanRatio: 0.997215, PostRatioDev: 1.773e-01, GradNorm: 14.7275, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 1/3, PPO Loss: -0.0060, PostKL: 1.718e-01, PostClipFrac: 0.262598, PostMeanRatio: 1.104879, PostRatioDev: 2.877e-01, GradNorm: 13.9995, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 2/3, PPO Loss: 0.0069, PostKL: 2.017e-01, PostClipFrac: 0.258207, PostMeanRatio: 1.136867, PostRatioDev: 3.130e-01, GradNorm: 11.5377, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_08.png
[OPE] Policy REJECTED: J_new=0.4319 ≤ J_old=0.4319 + δ=0.0043. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 8)
[Collect] 20 episodes, success=0.700, env_return=1044.70, rl_reward=0.70, steps=4000
[Data Collection] Success Rate: 0.700, EnvReturn: 1044.70, RLReward: 0.70, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 20 episodes remain in RL replay; IL retrain keeps 14/20 successful episodes (drops 6 failures).
[Dataset] Merged 20 episodes (4000 steps) → total 56000 steps, 280 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0041, Val Loss: 0.0045
[IL] Epoch 1/100, Loss: 0.0037, Val Loss: 0.0092
[IL] Epoch 2/100, Loss: 0.0038, Val Loss: 0.0089
[IL] Epoch 3/100, Loss: 0.0036, Val Loss: 0.0081
[IL] Epoch 4/100, Loss: 0.0033, Val Loss: 0.0070
[IL] Epoch 5/100, Loss: 0.0032, Val Loss: 0.0103
[IL] Epoch 6/100, Loss: 0.0036, Val Loss: 0.0083
[IL] Epoch 7/100, Loss: 0.0034, Val Loss: 0.0113
[IL] Epoch 8/100, Loss: 0.0032, Val Loss: 0.0087
[IL] Epoch 9/100, Loss: 0.0031, Val Loss: 0.0103
[IL] Epoch 10/100, Loss: 0.0030, Val Loss: 0.0096
[IL] Epoch 11/100, Loss: 0.0035, Val Loss: 0.0130
[IL] Epoch 12/100, Loss: 0.0034, Val Loss: 0.0099
[IL] Epoch 13/100, Loss: 0.0032, Val Loss: 0.0118
[IL] Epoch 14/100, Loss: 0.0043, Val Loss: 0.0104
[IL] Epoch 15/100, Loss: 0.0034, Val Loss: 0.0094
[IL] Epoch 16/100, Loss: 0.0030, Val Loss: 0.0117
[IL] Epoch 17/100, Loss: 0.0031, Val Loss: 0.0105
[IL] Epoch 18/100, Loss: 0.0038, Val Loss: 0.0090
[IL] Epoch 19/100, Loss: 0.0030, Val Loss: 0.0085
[IL] Epoch 20/100, Loss: 0.0030, Val Loss: 0.0129
[IL] Epoch 21/100, Loss: 0.0030, Val Loss: 0.0084
[IL] Epoch 22/100, Loss: 0.0030, Val Loss: 0.0115
[IL] Epoch 23/100, Loss: 0.0031, Val Loss: 0.0068
[IL] Epoch 24/100, Loss: 0.0029, Val Loss: 0.0100
[IL] Epoch 25/100, Loss: 0.0031, Val Loss: 0.0097
[IL] Epoch 26/100, Loss: 0.0029, Val Loss: 0.0069
[IL] Epoch 27/100, Loss: 0.0032, Val Loss: 0.0101
[IL] Epoch 28/100, Loss: 0.0032, Val Loss: 0.0094
[IL] Epoch 29/100, Loss: 0.0030, Val Loss: 0.0058
[IL] Epoch 30/100, Loss: 0.0030, Val Loss: 0.0092
[IL] Epoch 31/100, Loss: 0.0031, Val Loss: 0.0110
[IL] Epoch 32/100, Loss: 0.0030, Val Loss: 0.0094
[IL] Epoch 33/100, Loss: 0.0032, Val Loss: 0.0086
[IL] Epoch 34/100, Loss: 0.0029, Val Loss: 0.0093
[IL] Epoch 35/100, Loss: 0.0028, Val Loss: 0.0109
[IL] Epoch 36/100, Loss: 0.0029, Val Loss: 0.0102
[IL] Epoch 37/100, Loss: 0.0032, Val Loss: 0.0068
[IL] Epoch 38/100, Loss: 0.0034, Val Loss: 0.0086
[IL] Epoch 39/100, Loss: 0.0028, Val Loss: 0.0112
[IL] Epoch 40/100, Loss: 0.0034, Val Loss: 0.0137
[IL] Epoch 41/100, Loss: 0.0029, Val Loss: 0.0114
[IL] Epoch 42/100, Loss: 0.0030, Val Loss: 0.0107
[IL] Epoch 43/100, Loss: 0.0038, Val Loss: 0.0095
[IL] Epoch 44/100, Loss: 0.0031, Val Loss: 0.0117
[IL] Epoch 45/100, Loss: 0.0029, Val Loss: 0.0084
[IL] Epoch 46/100, Loss: 0.0030, Val Loss: 0.0114
[IL] Epoch 47/100, Loss: 0.0031, Val Loss: 0.0132
[IL] Epoch 48/100, Loss: 0.0029, Val Loss: 0.0082
[IL] Epoch 49/100, Loss: 0.0028, Val Loss: 0.0124
[IL] Epoch 50/100, Loss: 0.0030, Val Loss: 0.0061
[IL] Epoch 51/100, Loss: 0.0029, Val Loss: 0.0172
[IL] Epoch 52/100, Loss: 0.0028, Val Loss: 0.0127
[IL] Epoch 53/100, Loss: 0.0029, Val Loss: 0.0132
[IL] Epoch 54/100, Loss: 0.0027, Val Loss: 0.0127
[IL] Epoch 55/100, Loss: 0.0032, Val Loss: 0.0094
[IL] Epoch 56/100, Loss: 0.0029, Val Loss: 0.0080
[IL] Epoch 57/100, Loss: 0.0028, Val Loss: 0.0105
[IL] Epoch 58/100, Loss: 0.0030, Val Loss: 0.0085
[IL] Epoch 59/100, Loss: 0.0029, Val Loss: 0.0125
[IL] Epoch 60/100, Loss: 0.0029, Val Loss: 0.0114
[IL] Epoch 61/100, Loss: 0.0030, Val Loss: 0.0117
[IL] Epoch 62/100, Loss: 0.0027, Val Loss: 0.0127
[IL] Epoch 63/100, Loss: 0.0029, Val Loss: 0.0125
[IL] Epoch 64/100, Loss: 0.0026, Val Loss: 0.0103
[IL] Epoch 65/100, Loss: 0.0029, Val Loss: 0.0099
[IL] Epoch 66/100, Loss: 0.0029, Val Loss: 0.0109
[IL] Epoch 67/100, Loss: 0.0030, Val Loss: 0.0143
[IL] Epoch 68/100, Loss: 0.0027, Val Loss: 0.0085
[IL] Epoch 69/100, Loss: 0.0028, Val Loss: 0.0083
[IL] Epoch 70/100, Loss: 0.0029, Val Loss: 0.0235
[IL] Epoch 71/100, Loss: 0.0029, Val Loss: 0.0101
[IL] Epoch 72/100, Loss: 0.0030, Val Loss: 0.0113
[IL] Epoch 73/100, Loss: 0.0031, Val Loss: 0.0136
[IL] Epoch 74/100, Loss: 0.0030, Val Loss: 0.0099
[IL] Epoch 75/100, Loss: 0.0028, Val Loss: 0.0073
[IL] Epoch 76/100, Loss: 0.0029, Val Loss: 0.0081
[IL] Epoch 77/100, Loss: 0.0027, Val Loss: 0.0142
[IL] Epoch 78/100, Loss: 0.0026, Val Loss: 0.0146
[IL] Epoch 79/100, Loss: 0.0030, Val Loss: 0.0131
[IL] Epoch 80/100, Loss: 0.0032, Val Loss: 0.0143
[IL] Epoch 81/100, Loss: 0.0028, Val Loss: 0.0088
[IL] Epoch 82/100, Loss: 0.0030, Val Loss: 0.0122
[IL] Epoch 83/100, Loss: 0.0031, Val Loss: 0.0060
[IL] Epoch 84/100, Loss: 0.0027, Val Loss: 0.0121
[IL] Epoch 85/100, Loss: 0.0029, Val Loss: 0.0102
[IL] Epoch 86/100, Loss: 0.0027, Val Loss: 0.0108
[IL] Epoch 87/100, Loss: 0.0029, Val Loss: 0.0075
[IL] Epoch 88/100, Loss: 0.0031, Val Loss: 0.0091
[IL] Epoch 89/100, Loss: 0.0029, Val Loss: 0.0154
[IL] Epoch 90/100, Loss: 0.0030, Val Loss: 0.0099
[IL] Epoch 91/100, Loss: 0.0027, Val Loss: 0.0084
[IL] Epoch 92/100, Loss: 0.0027, Val Loss: 0.0109
[IL] Epoch 93/100, Loss: 0.0028, Val Loss: 0.0099
[IL] Epoch 94/100, Loss: 0.0027, Val Loss: 0.0114
[IL] Epoch 95/100, Loss: 0.0027, Val Loss: 0.0132
[IL] Epoch 96/100, Loss: 0.0027, Val Loss: 0.0126
[IL] Epoch 97/100, Loss: 0.0028, Val Loss: 0.0152
[IL] Epoch 98/100, Loss: 0.0032, Val Loss: 0.0108
[IL] Epoch 99/100, Loss: 0.0030, Val Loss: 0.0095
test_mean_score: 0.82
[IL] Eval - Success Rate: 0.820
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_08.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_8.ckpt

================================================================================
               OFFLINE RL ITERATION 10/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 9)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 52110 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-139.31363 | val=0.00032 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-153.79916 | val=0.00026 | no-improve=0/5
[TransitionModel] Epoch   27 | train=-157.52415 | val=0.00026 | no-improve=5/5
[TransitionModel] Training complete. Elites=[2, 0, 6, 4, 3], val_loss=0.00025
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_09.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_09.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 9)
[IQL] Epoch 0/20, V Loss: 0.0002, Q Loss: 0.0047
[IQL] Epoch 1/20, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Epoch 2/20, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 3/20, V Loss: 0.0002, Q Loss: 0.0047
[IQL] Epoch 4/20, V Loss: 0.0002, Q Loss: 0.0047
[IQL] Epoch 5/20, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 6/20, V Loss: 0.0001, Q Loss: 0.0047
[IQL] Epoch 7/20, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 8/20, V Loss: 0.0001, Q Loss: 0.0046
[IQL] Epoch 9/20, V Loss: 0.0002, Q Loss: 0.0047
[IQL] Epoch 10/20, V Loss: 0.0002, Q Loss: 0.0047
[IQL] Epoch 11/20, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 12/20, V Loss: 0.0001, Q Loss: 0.0046
[IQL] Epoch 13/20, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 14/20, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 15/20, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 16/20, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 17/20, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 18/20, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 19/20, V Loss: 0.0002, Q Loss: 0.0046
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_09.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_09.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 9)
[OPE] Behavior policy value J_old = 0.4181
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 1.00e-05
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 204 mini-batches, 52110 samples, raw advantage mean=-0.0002, std=0.0231
[Offline RL] Epoch 0/3, PPO Loss: -0.0238, PostKL: 5.795e-02, PostClipFrac: 0.278312, PostMeanRatio: 0.997406, PostRatioDev: 1.867e-01, GradNorm: 17.0130, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 1/3, PPO Loss: -0.0058, PostKL: 8.566e-02, PostClipFrac: 0.268277, PostMeanRatio: 1.016866, PostRatioDev: 2.002e-01, GradNorm: 13.0787, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Epoch 2/3, PPO Loss: 0.0071, PostKL: 9.973e-02, PostClipFrac: 0.254785, PostMeanRatio: 1.039301, PostRatioDev: 2.114e-01, GradNorm: 11.8013, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_09.png
[OPE] Policy REJECTED: J_new=0.4183 ≤ J_old=0.4181 + δ=0.0042. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.900, env_return=1238.16, rl_reward=0.90, steps=4000
[Data Collection] Success Rate: 0.900, EnvReturn: 1238.16, RLReward: 0.90, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 20 episodes remain in RL replay; IL retrain keeps 18/20 successful episodes (drops 2 failures).
[Dataset] Merged 20 episodes (4000 steps) → total 60000 steps, 300 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0038, Val Loss: 0.0111
[IL] Epoch 1/100, Loss: 0.0033, Val Loss: 0.0090
[IL] Epoch 2/100, Loss: 0.0035, Val Loss: 0.0096
[IL] Epoch 3/100, Loss: 0.0034, Val Loss: 0.0089
[IL] Epoch 4/100, Loss: 0.0031, Val Loss: 0.0062
[IL] Epoch 5/100, Loss: 0.0030, Val Loss: 0.0095
[IL] Epoch 6/100, Loss: 0.0030, Val Loss: 0.0072
[IL] Epoch 7/100, Loss: 0.0031, Val Loss: 0.0091
[IL] Epoch 8/100, Loss: 0.0031, Val Loss: 0.0117
[IL] Epoch 9/100, Loss: 0.0031, Val Loss: 0.0113
[IL] Epoch 10/100, Loss: 0.0031, Val Loss: 0.0114
[IL] Epoch 11/100, Loss: 0.0031, Val Loss: 0.0093
[IL] Epoch 12/100, Loss: 0.0033, Val Loss: 0.0099
[IL] Epoch 13/100, Loss: 0.0029, Val Loss: 0.0110
[IL] Epoch 14/100, Loss: 0.0030, Val Loss: 0.0095
[IL] Epoch 15/100, Loss: 0.0030, Val Loss: 0.0101
[IL] Epoch 16/100, Loss: 0.0030, Val Loss: 0.0048
[IL] Epoch 17/100, Loss: 0.0028, Val Loss: 0.0069
[IL] Epoch 18/100, Loss: 0.0029, Val Loss: 0.0082
[IL] Epoch 19/100, Loss: 0.0029, Val Loss: 0.0103
[IL] Epoch 20/100, Loss: 0.0032, Val Loss: 0.0095
[IL] Epoch 21/100, Loss: 0.0033, Val Loss: 0.0120
[IL] Epoch 22/100, Loss: 0.0031, Val Loss: 0.0083
[IL] Epoch 23/100, Loss: 0.0030, Val Loss: 0.0087
[IL] Epoch 24/100, Loss: 0.0030, Val Loss: 0.0076
[IL] Epoch 25/100, Loss: 0.0028, Val Loss: 0.0073
[IL] Epoch 26/100, Loss: 0.0027, Val Loss: 0.0082
[IL] Epoch 27/100, Loss: 0.0026, Val Loss: 0.0092
[IL] Epoch 28/100, Loss: 0.0028, Val Loss: 0.0082
[IL] Epoch 29/100, Loss: 0.0031, Val Loss: 0.0096
[IL] Epoch 30/100, Loss: 0.0029, Val Loss: 0.0125
[IL] Epoch 31/100, Loss: 0.0032, Val Loss: 0.0104
[IL] Epoch 32/100, Loss: 0.0028, Val Loss: 0.0088
[IL] Epoch 33/100, Loss: 0.0028, Val Loss: 0.0081
[IL] Epoch 34/100, Loss: 0.0030, Val Loss: 0.0118
[IL] Epoch 35/100, Loss: 0.0029, Val Loss: 0.0122
[IL] Epoch 36/100, Loss: 0.0030, Val Loss: 0.0096
[IL] Epoch 37/100, Loss: 0.0028, Val Loss: 0.0086
[IL] Epoch 38/100, Loss: 0.0026, Val Loss: 0.0134
[IL] Epoch 39/100, Loss: 0.0028, Val Loss: 0.0113
[IL] Epoch 40/100, Loss: 0.0032, Val Loss: 0.0107
[IL] Epoch 41/100, Loss: 0.0029, Val Loss: 0.0096
[IL] Epoch 42/100, Loss: 0.0026, Val Loss: 0.0121
[IL] Epoch 43/100, Loss: 0.0026, Val Loss: 0.0111
[IL] Epoch 44/100, Loss: 0.0028, Val Loss: 0.0110
[IL] Epoch 45/100, Loss: 0.0028, Val Loss: 0.0132
[IL] Epoch 46/100, Loss: 0.0029, Val Loss: 0.0095
[IL] Epoch 47/100, Loss: 0.0038, Val Loss: 0.0073
[IL] Epoch 48/100, Loss: 0.0030, Val Loss: 0.0146
[IL] Epoch 49/100, Loss: 0.0029, Val Loss: 0.0092
[IL] Epoch 50/100, Loss: 0.0029, Val Loss: 0.0085
[IL] Epoch 51/100, Loss: 0.0028, Val Loss: 0.0090
[IL] Epoch 52/100, Loss: 0.0029, Val Loss: 0.0073
[IL] Epoch 53/100, Loss: 0.0028, Val Loss: 0.0076
[IL] Epoch 54/100, Loss: 0.0030, Val Loss: 0.0106
[IL] Epoch 55/100, Loss: 0.0025, Val Loss: 0.0076
[IL] Epoch 56/100, Loss: 0.0030, Val Loss: 0.0097
[IL] Epoch 57/100, Loss: 0.0033, Val Loss: 0.0098
[IL] Epoch 58/100, Loss: 0.0026, Val Loss: 0.0101
[IL] Epoch 59/100, Loss: 0.0025, Val Loss: 0.0108
[IL] Epoch 60/100, Loss: 0.0028, Val Loss: 0.0074
[IL] Epoch 61/100, Loss: 0.0028, Val Loss: 0.0111
[IL] Epoch 62/100, Loss: 0.0026, Val Loss: 0.0096
[IL] Epoch 63/100, Loss: 0.0027, Val Loss: 0.0155
[IL] Epoch 64/100, Loss: 0.0028, Val Loss: 0.0107
[IL] Epoch 65/100, Loss: 0.0028, Val Loss: 0.0125
[IL] Epoch 66/100, Loss: 0.0030, Val Loss: 0.0115
[IL] Epoch 67/100, Loss: 0.0027, Val Loss: 0.0113
[IL] Epoch 68/100, Loss: 0.0024, Val Loss: 0.0097
[IL] Epoch 69/100, Loss: 0.0026, Val Loss: 0.0090
[IL] Epoch 70/100, Loss: 0.0026, Val Loss: 0.0119
[IL] Epoch 71/100, Loss: 0.0026, Val Loss: 0.0112
[IL] Epoch 72/100, Loss: 0.0027, Val Loss: 0.0080
[IL] Epoch 73/100, Loss: 0.0026, Val Loss: 0.0098
[IL] Epoch 74/100, Loss: 0.0024, Val Loss: 0.0086
[IL] Epoch 75/100, Loss: 0.0025, Val Loss: 0.0116
[IL] Epoch 76/100, Loss: 0.0028, Val Loss: 0.0079
[IL] Epoch 77/100, Loss: 0.0028, Val Loss: 0.0108
[IL] Epoch 78/100, Loss: 0.0027, Val Loss: 0.0156
[IL] Epoch 79/100, Loss: 0.0028, Val Loss: 0.0097
[IL] Epoch 80/100, Loss: 0.0026, Val Loss: 0.0113
[IL] Epoch 81/100, Loss: 0.0030, Val Loss: 0.0099
[IL] Epoch 82/100, Loss: 0.0028, Val Loss: 0.0088
[IL] Epoch 83/100, Loss: 0.0028, Val Loss: 0.0067
[IL] Epoch 84/100, Loss: 0.0027, Val Loss: 0.0120
[IL] Epoch 85/100, Loss: 0.0025, Val Loss: 0.0077
[IL] Epoch 86/100, Loss: 0.0026, Val Loss: 0.0088
[IL] Epoch 87/100, Loss: 0.0026, Val Loss: 0.0098
[IL] Epoch 88/100, Loss: 0.0026, Val Loss: 0.0071
[IL] Epoch 89/100, Loss: 0.0027, Val Loss: 0.0097
[IL] Epoch 90/100, Loss: 0.0029, Val Loss: 0.0102
[IL] Epoch 91/100, Loss: 0.0032, Val Loss: 0.0079
[IL] Epoch 92/100, Loss: 0.0026, Val Loss: 0.0075
[IL] Epoch 93/100, Loss: 0.0027, Val Loss: 0.0104
[IL] Epoch 94/100, Loss: 0.0028, Val Loss: 0.0154
[IL] Epoch 95/100, Loss: 0.0028, Val Loss: 0.0089
[IL] Epoch 96/100, Loss: 0.0029, Val Loss: 0.0093
[IL] Epoch 97/100, Loss: 0.0025, Val Loss: 0.0094
[IL] Epoch 98/100, Loss: 0.0025, Val Loss: 0.0080
[IL] Epoch 99/100, Loss: 0.0024, Val Loss: 0.0115
test_mean_score: 0.75
[IL] Eval - Success Rate: 0.750
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_09.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_9.ckpt

================================================================================
                    PHASE 3: ONLINE RL FINE-TUNING
================================================================================

[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 1.00e-05

[Online RL] Iteration 1/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.950, env_return=1062.16, rl_reward=0.95, steps=4000
[Data Collection] Success Rate: 0.950, EnvReturn: 1062.16, RLReward: 0.95, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_collection_success_rate.png
[Dataset] online collection: all 20 episodes remain in RL replay; IL retrain keeps 19/20 successful episodes (drops 1 failures).
[Online RL] Epoch 1/3, PPO Loss: -0.0072, PostKL: 4.461e-02, PostClipFrac: 0.198938, PostMeanRatio: 0.995572, PostRatioDev: 1.449e-01, GradNorm: 20.7342, Reg Loss: 0.0000, CD Loss: 0.3409
[Online RL] Epoch 2/3, PPO Loss: 0.0280, PostKL: 1.207e-01, PostClipFrac: 0.285238, PostMeanRatio: 1.004484, PostRatioDev: 2.102e-01, GradNorm: 19.7415, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/3, PPO Loss: 0.0460, PostKL: 1.494e-01, PostClipFrac: 0.312372, PostMeanRatio: 1.007336, PostRatioDev: 2.366e-01, GradNorm: 15.3715, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_v_loss_iter_00.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_loss_iter_00.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_kl_iter_00.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_clipfrac_iter_00.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_gradnorm_iter_00.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_cd_loss_iter_00.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_0.ckpt

[Online RL] Iteration 2/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.800, env_return=1085.81, rl_reward=0.80, steps=4000
[Data Collection] Success Rate: 0.800, EnvReturn: 1085.81, RLReward: 0.80, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_collection_success_rate.png
[Dataset] online collection: all 20 episodes remain in RL replay; IL retrain keeps 16/20 successful episodes (drops 4 failures).
[Online RL] Epoch 1/3, PPO Loss: 0.0015, PostKL: 9.602e-03, PostClipFrac: 0.070426, PostMeanRatio: 0.997733, PostRatioDev: 6.558e-02, GradNorm: 13.9762, Reg Loss: 0.0000, CD Loss: 0.3340
[Online RL] Epoch 2/3, PPO Loss: 0.0222, PostKL: 4.720e-02, PostClipFrac: 0.198781, PostMeanRatio: 1.000635, PostRatioDev: 1.401e-01, GradNorm: 11.0975, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/3, PPO Loss: 0.0382, PostKL: 7.766e-02, PostClipFrac: 0.257304, PostMeanRatio: 1.010251, PostRatioDev: 1.794e-01, GradNorm: 11.0762, Reg Loss: 0.0000, CD Loss: 0.3298
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_v_loss_iter_01.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_loss_iter_01.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_kl_iter_01.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_clipfrac_iter_01.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_gradnorm_iter_01.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_cd_loss_iter_01.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_1.ckpt

[Online RL] Iteration 3/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.950, env_return=1183.40, rl_reward=0.95, steps=4000
[Data Collection] Success Rate: 0.950, EnvReturn: 1183.40, RLReward: 0.95, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_collection_success_rate.png
[Dataset] online collection: all 20 episodes remain in RL replay; IL retrain keeps 19/20 successful episodes (drops 1 failures).
[Online RL] Epoch 1/3, PPO Loss: -0.0006, PostKL: 5.309e-03, PostClipFrac: 0.029368, PostMeanRatio: 1.002040, PostRatioDev: 4.413e-02, GradNorm: 15.7164, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 2/3, PPO Loss: 0.0180, PostKL: 3.645e-02, PostClipFrac: 0.129194, PostMeanRatio: 1.019167, PostRatioDev: 1.184e-01, GradNorm: 13.1024, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/3, PPO Loss: 0.0327, PostKL: 1.708e-01, PostClipFrac: 0.206914, PostMeanRatio: 1.135030, PostRatioDev: 2.794e-01, GradNorm: 11.3758, Reg Loss: 0.0000, CD Loss: 0.3321
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_v_loss_iter_02.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_loss_iter_02.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_kl_iter_02.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_clipfrac_iter_02.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_gradnorm_iter_02.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_cd_loss_iter_02.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_2.ckpt

[Online RL] Iteration 4/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.750, env_return=1137.37, rl_reward=0.75, steps=4000
[Data Collection] Success Rate: 0.750, EnvReturn: 1137.37, RLReward: 0.75, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_collection_success_rate.png
[Dataset] online collection: all 20 episodes remain in RL replay; IL retrain keeps 15/20 successful episodes (drops 5 failures).
[Online RL] Epoch 1/3, PPO Loss: 0.0003, PostKL: 4.476e-03, PostClipFrac: 0.031620, PostMeanRatio: 1.000277, PostRatioDev: 4.322e-02, GradNorm: 17.7674, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 2/3, PPO Loss: 0.0125, PostKL: 3.342e-02, PostClipFrac: 0.144752, PostMeanRatio: 1.008495, PostRatioDev: 1.137e-01, GradNorm: 14.5147, Reg Loss: 0.0000, CD Loss: 0.3397
[Online RL] Epoch 3/3, PPO Loss: 0.0263, PostKL: 4.670e-02, PostClipFrac: 0.199009, PostMeanRatio: 1.012300, PostRatioDev: 1.400e-01, GradNorm: 9.4037, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_v_loss_iter_03.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_loss_iter_03.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_kl_iter_03.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_clipfrac_iter_03.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_gradnorm_iter_03.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_cd_loss_iter_03.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_3.ckpt

[Online RL] Iteration 5/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.900, env_return=1129.04, rl_reward=0.90, steps=4000
[Data Collection] Success Rate: 0.900, EnvReturn: 1129.04, RLReward: 0.90, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_collection_success_rate.png
[Dataset] online collection: all 20 episodes remain in RL replay; IL retrain keeps 18/20 successful episodes (drops 2 failures).
[Online RL] Epoch 1/3, PPO Loss: 0.0013, PostKL: 4.418e-03, PostClipFrac: 0.024117, PostMeanRatio: 0.998931, PostRatioDev: 3.875e-02, GradNorm: 16.5048, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 2/3, PPO Loss: 0.0175, PostKL: 2.839e-02, PostClipFrac: 0.115017, PostMeanRatio: 1.006241, PostRatioDev: 9.832e-02, GradNorm: 12.5167, Reg Loss: 0.0000, CD Loss: 0.3380
[Online RL] Epoch 3/3, PPO Loss: 0.0337, PostKL: 5.789e-02, PostClipFrac: 0.185262, PostMeanRatio: 1.016828, PostRatioDev: 1.492e-01, GradNorm: 13.4394, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_v_loss_iter_04.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_loss_iter_04.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_kl_iter_04.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_clipfrac_iter_04.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_gradnorm_iter_04.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_cd_loss_iter_04.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_4.ckpt

[Online RL] Iteration 6/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.750, env_return=1075.74, rl_reward=0.75, steps=4000
[Data Collection] Success Rate: 0.750, EnvReturn: 1075.74, RLReward: 0.75, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_collection_success_rate.png
[Dataset] online collection: all 20 episodes remain in RL replay; IL retrain keeps 15/20 successful episodes (drops 5 failures).
[Online RL] Epoch 1/3, PPO Loss: -0.0004, PostKL: 3.538e-03, PostClipFrac: 0.024932, PostMeanRatio: 0.998135, PostRatioDev: 3.807e-02, GradNorm: 15.0061, Reg Loss: 0.0000, CD Loss: 0.3254
[Online RL] Epoch 2/3, PPO Loss: 0.0195, PostKL: 2.166e-02, PostClipFrac: 0.124534, PostMeanRatio: 0.999982, PostRatioDev: 9.761e-02, GradNorm: 10.6553, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/3, PPO Loss: 0.0342, PostKL: 4.644e-02, PostClipFrac: 0.205363, PostMeanRatio: 1.007339, PostRatioDev: 1.491e-01, GradNorm: 10.8639, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_v_loss_iter_05.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_loss_iter_05.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_kl_iter_05.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_clipfrac_iter_05.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_gradnorm_iter_05.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_cd_loss_iter_05.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_5.ckpt

[Online RL] Iteration 7/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.950, env_return=1084.18, rl_reward=0.95, steps=4000
[Data Collection] Success Rate: 0.950, EnvReturn: 1084.18, RLReward: 0.95, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_collection_success_rate.png
[Dataset] online collection: all 20 episodes remain in RL replay; IL retrain keeps 19/20 successful episodes (drops 1 failures).
[Online RL] Epoch 1/3, PPO Loss: -0.0011, PostKL: 4.283e-03, PostClipFrac: 0.031481, PostMeanRatio: 1.001827, PostRatioDev: 4.894e-02, GradNorm: 18.3318, Reg Loss: 0.0000, CD Loss: 0.3261
[Online RL] Epoch 2/3, PPO Loss: 0.0161, PostKL: 1.970e-02, PostClipFrac: 0.136224, PostMeanRatio: 1.006311, PostRatioDev: 1.065e-01, GradNorm: 11.8597, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/3, PPO Loss: 0.0286, PostKL: 4.077e-02, PostClipFrac: 0.209415, PostMeanRatio: 1.012371, PostRatioDev: 1.519e-01, GradNorm: 10.6521, Reg Loss: 0.0000, CD Loss: 0.3378
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_v_loss_iter_06.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_loss_iter_06.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_kl_iter_06.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_clipfrac_iter_06.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_gradnorm_iter_06.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_cd_loss_iter_06.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_6.ckpt

[Online RL] Iteration 8/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.750, env_return=1124.26, rl_reward=0.75, steps=4000
[Data Collection] Success Rate: 0.750, EnvReturn: 1124.26, RLReward: 0.75, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_collection_success_rate.png
[Dataset] online collection: all 20 episodes remain in RL replay; IL retrain keeps 15/20 successful episodes (drops 5 failures).
[Online RL] Epoch 1/3, PPO Loss: 0.0010, PostKL: 3.682e-03, PostClipFrac: 0.034299, PostMeanRatio: 1.000044, PostRatioDev: 4.410e-02, GradNorm: 16.4316, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 2/3, PPO Loss: 0.0151, PostKL: 2.139e-02, PostClipFrac: 0.131027, PostMeanRatio: 1.004782, PostRatioDev: 1.063e-01, GradNorm: 13.8643, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/3, PPO Loss: 0.0286, PostKL: 4.406e-02, PostClipFrac: 0.195487, PostMeanRatio: 1.013818, PostRatioDev: 1.499e-01, GradNorm: 11.4867, Reg Loss: 0.0000, CD Loss: 0.3376
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_v_loss_iter_07.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_loss_iter_07.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_kl_iter_07.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_clipfrac_iter_07.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_gradnorm_iter_07.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_cd_loss_iter_07.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_7.ckpt

[Online RL] Iteration 9/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.900, env_return=1151.23, rl_reward=0.90, steps=4000
[Data Collection] Success Rate: 0.900, EnvReturn: 1151.23, RLReward: 0.90, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_collection_success_rate.png
[Dataset] online collection: all 20 episodes remain in RL replay; IL retrain keeps 18/20 successful episodes (drops 2 failures).
[Online RL] Epoch 1/3, PPO Loss: 0.0017, PostKL: 2.583e-03, PostClipFrac: 0.025504, PostMeanRatio: 0.997845, PostRatioDev: 3.860e-02, GradNorm: 16.5628, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 2/3, PPO Loss: 0.0160, PostKL: 1.483e-02, PostClipFrac: 0.124989, PostMeanRatio: 0.996575, PostRatioDev: 9.440e-02, GradNorm: 11.3704, Reg Loss: 0.0000, CD Loss: 0.3312
[Online RL] Epoch 3/3, PPO Loss: 0.0270, PostKL: 3.156e-02, PostClipFrac: 0.188791, PostMeanRatio: 1.000904, PostRatioDev: 1.367e-01, GradNorm: 13.2614, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_v_loss_iter_08.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_loss_iter_08.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_kl_iter_08.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_clipfrac_iter_08.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_gradnorm_iter_08.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_cd_loss_iter_08.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_8.ckpt

[Online RL] Iteration 10/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.750, env_return=975.11, rl_reward=0.75, steps=4000
[Data Collection] Success Rate: 0.750, EnvReturn: 975.11, RLReward: 0.75, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_collection_success_rate.png
[Dataset] online collection: all 20 episodes remain in RL replay; IL retrain keeps 15/20 successful episodes (drops 5 failures).
[Online RL] Epoch 1/3, PPO Loss: -0.0010, PostKL: 5.204e-03, PostClipFrac: 0.024736, PostMeanRatio: 0.999142, PostRatioDev: 3.832e-02, GradNorm: 19.2709, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 2/3, PPO Loss: 0.0153, PostKL: 2.577e-02, PostClipFrac: 0.108080, PostMeanRatio: 1.001921, PostRatioDev: 8.857e-02, GradNorm: 14.1572, Reg Loss: 0.0000, CD Loss: 0.3334
[Online RL] Epoch 3/3, PPO Loss: 0.0321, PostKL: 5.521e-02, PostClipFrac: 0.172423, PostMeanRatio: 1.016988, PostRatioDev: 1.383e-01, GradNorm: 12.9106, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_v_loss_iter_09.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_loss_iter_09.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_kl_iter_09.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_clipfrac_iter_09.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_ppo_gradnorm_iter_09.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/online_cd_loss_iter_09.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_9.ckpt

================================================================================
                         TRAINING COMPLETE!
================================================================================

[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/final.ckpt

[Evaluation] Running final evaluation...
test_mean_score: 0.83
test_mean_score: 0.01

================================================================================
                         FINAL RESULTS
================================================================================
[ddim]
mean_traj_rewards: 12285.9861
mean_success_rates: 0.8300
test_mean_score: 0.8300
SR_test_L3: 0.8400
SR_test_L5: 0.8340
[cm]
mean_traj_rewards: 2152.7849
mean_success_rates: 0.0100
test_mean_score: 0.0100
SR_test_L3: 0.8400
SR_test_L5: 0.8340

[Training] Complete! Checkpoints saved to:
  /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints
Found 8 GPUs for rendering. Using device 0.
Extracting GPU stats logs using atop has been completed on r8a30-a06.
Logs are being saved to: /nfs_global/S/yangrongzheng/atop-740901-r8a30-a06-gpustat.log
Job end at 2026-03-26 05:40:49
