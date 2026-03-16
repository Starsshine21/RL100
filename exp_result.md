Job start at 2026-03-13 15:23:11
Job run at:
   Static hostname: localhost.localdomain
Transient hostname: r8a100-a02.ib.future.cn
         Icon name: computer-server
           Chassis: server
        Machine ID: d6529da74c4847f5ae1fed83744eba13
           Boot ID: 5633f0e2abd44074870b9ecbbc204d9c
  Operating System: Rocky Linux 8.7 (Green Obsidian)
       CPE OS Name: cpe:/o:rocky:rocky:8:GA
            Kernel: Linux 4.18.0-425.19.2.el8_7.x86_64
      Architecture: x86-64
Filesystem                                        Size  Used Avail Use% Mounted on
/dev/mapper/rl-root                               120G   26G   95G  22% /
/dev/sdb1                                         1.1T   11G  1.1T   1% /tmp
/dev/sdb2                                         4.2T  177G  4.1T   5% /local
/dev/mapper/rl-var                                768G   28G  741G   4% /var
/dev/sda2                                         2.0G  304M  1.7G  15% /boot
/dev/sda1                                         599M  5.8M  594M   1% /boot/efi
ssd.nas00.future.cn:/rocky8_home                   16G   15G  1.9G  89% /home
ssd.nas00.future.cn:/rocky8_workspace             400G     0  400G   0% /workspace
ssd.nas00.future.cn:/rocky8_tools                 5.0T   99G  5.0T   2% /tools
ssd.nas00.future.cn:/centos7_home                  16G  4.2G   12G  26% /centos7/home
ssd.nas00.future.cn:/centos7_workspace            400G     0  400G   0% /centos7/workspace
ssd.nas00.future.cn:/centos7_tools                5.0T  235G  4.8T   5% /centos7/tools
ssd.nas00.future.cn:/eda-tools                    8.0T  6.3T  1.8T  79% /centos7/eda-tools
hdd.nas00.future.cn:/share_personal               500G     0  500G   0% /share/personal
zone05.nas01.future.cn:/NAS_HPC_collab_codemodel   40T   37T  3.7T  91% /share/collab/codemodel
ext-zone00.nas02.future.cn:/nfs_global            406T  397T  9.4T  98% /nfs_global
ssd.nas00.future.cn:/common_datasets               75T   64T   12T  85% /datasets
192.168.12.10@o2ib:192.168.12.11@o2ib:/lustre     1.9P   12T  1.8P   1% /lustre
beegfs_nodev                                       70T   15T   56T  21% /fast
Currently Loaded Modulefiles: 1) cluster-tools/v1.0 3) cuda-cudnn/12.1-8.9.3 5) git/2.31.1 2) cmake/3.21.7 4) gcc/9.3.0 6) slurm-tools/v1.0
/tools/cluster-software/gcc/gcc-9.3.0/bin/gcc
/home/S/yangrongzheng/miniconda3/bin/python
/home/S/yangrongzheng/miniconda3/bin/python3
############### /home : /home/S/yangrongzheng
Disk quotas for user yangrongzheng (uid 6215): 
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
          /home  14453M  16384M  20480M            168k       0       0        

############### /workspace
Disk quotas for user yangrongzheng (uid 6215): 
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
     /workspace      0K    400G    500G               1       0       0        

############### /nfs_global
Disk quotas for user yangrongzheng (uid 6215): 
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
    /nfs_global    262G   5120G   7168G            350k   5000k  10000k        

############### /lustre
Disk quotas for usr yangrongzheng (uid 6215):
     Filesystem    used   quota   limit   grace   files   quota   limit   grace
        /lustre      0k      8T     10T       -       0  3000000 36000000       -
uid 6215 is using default block quota setting
uid 6215 is using default file quota setting
name, driver_version, power.limit [W]
NVIDIA A100-PCIE-40GB, 570.124.06, 225.00 W
NVIDIA A100-PCIE-40GB, 570.124.06, 225.00 W
NVIDIA A100-PCIE-40GB, 570.124.06, 225.00 W
NVIDIA A100-PCIE-40GB, 570.124.06, 225.00 W
NVIDIA A100-PCIE-40GB, 570.124.06, 225.00 W
NVIDIA A100-PCIE-40GB, 570.124.06, 225.00 W
NVIDIA A100-PCIE-40GB, 570.124.06, 225.00 W
NVIDIA A100-PCIE-40GB, 570.124.06, 225.00 W
Using GPU(s) 0,1,2,3,4,5,6,7
This job is assigned the following resources by SLURM:
CPU_IDs=0-111 GRES=gpu:8(IDX:0-7)
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
    eval_episodes: 20
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
  num_offline_iterations: 5
  critic_epochs: 40
  ppo_epochs: 20
  ppo_inner_steps: 4
  collection_episodes: 20
  cd_every: 5
  lambda_cd: 1.0
  rl_policy_lr: 1.0e-05
  run_online_rl: true
  online_rl_iterations: 10
  online_collection_episodes: 20
  lambda_v: 0.5
  gae_lambda: 0.95
  gradient_accumulate_every: 1
  max_grad_norm: 1.0
  log_every: 10
  eval_every: 100
  checkpoint_every: 200
  resume: true
  resume_path: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/after_il.ckpt
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
[Setup] Dataset loaded: 17370 episodes
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
[2026-03-13 15:23:24,407][diffusion_policy_3d.model.diffusion.conditional_unet1d][INFO] - number of parameters: 2.550744e+08
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
[2026-03-13 15:23:26,064][diffusion_policy_3d.model.diffusion.conditional_unet1d][INFO] - number of parameters: 2.550744e+08
[RL100Trainer] Initializing Transition Model T_θ(s'|s,a)...
[Setup] RL100Trainer initialized

[Setup] Resuming from checkpoint: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/after_il.ckpt
[Checkpoint] critics: skipped 2 incompatible key(s): ['q_network.q1.network.0.weight', 'q_network.q2.network.0.weight']
[Checkpoint] critics: 2 missing key(s): ['q_network.q1.network.0.weight', 'q_network.q2.network.0.weight']
[Checkpoint] transition_model full restore failed (RuntimeError: Error(s) in loading state_dict for EnsembleDynamicsModel:
	size mismatch for backbones.0.weight: copying a param with shape torch.Size([7, 260, 200]) from checkpoint, the shape in current model is torch.Size([7, 288, 200]).
	size mismatch for backbones.0.saved_weight: copying a param with shape torch.Size([7, 260, 200]) from checkpoint, the shape in current model is torch.Size([7, 288, 200]).); trying partial model restore.
[Checkpoint] transition_model.model: skipped 2 incompatible key(s): ['backbones.0.weight', 'backbones.0.saved_weight']
[Checkpoint] transition_model.model: 2 missing key(s): ['backbones.0.weight', 'backbones.0.saved_weight']
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
               OFFLINE RL ITERATION 1/5
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 0)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 17370 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=5.39305 | val=0.00466 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-31.27108 | val=0.00050 | no-improve=0/5
[TransitionModel] Epoch   40 | train=-37.27283 | val=0.00038 | no-improve=0/5
[TransitionModel] Epoch   60 | train=-41.47412 | val=0.00032 | no-improve=0/5
[TransitionModel] Epoch   80 | train=-45.42663 | val=0.00027 | no-improve=0/5
[TransitionModel] Epoch  100 | train=-49.15089 | val=0.00021 | no-improve=0/5
[TransitionModel] Epoch  120 | train=-52.82454 | val=0.00017 | no-improve=0/5
[TransitionModel] Epoch  140 | train=-56.30543 | val=0.00015 | no-improve=0/5
[TransitionModel] Epoch  160 | train=-59.88554 | val=0.00014 | no-improve=3/5
[TransitionModel] Epoch  180 | train=-63.24838 | val=0.00013 | no-improve=1/5
[TransitionModel] Training complete. Elites=[5, 1, 3, 4, 0], val_loss=0.00012

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 0)
[IQL] Epoch 0/40, V Loss: 0.0089, Q Loss: 0.0591
[IQL] Epoch 1/40, V Loss: 0.0002, Q Loss: 0.0089
[IQL] Epoch 2/40, V Loss: 0.0001, Q Loss: 0.0088
[IQL] Epoch 3/40, V Loss: 0.0001, Q Loss: 0.0087
[IQL] Epoch 4/40, V Loss: 0.0002, Q Loss: 0.0085
[IQL] Epoch 5/40, V Loss: 0.0002, Q Loss: 0.0084
[IQL] Epoch 6/40, V Loss: 0.0002, Q Loss: 0.0085
[IQL] Epoch 7/40, V Loss: 0.0002, Q Loss: 0.0084
[IQL] Epoch 8/40, V Loss: 0.0001, Q Loss: 0.0082
[IQL] Epoch 9/40, V Loss: 0.0002, Q Loss: 0.0082
[IQL] Epoch 10/40, V Loss: 0.0003, Q Loss: 0.0081
[IQL] Epoch 11/40, V Loss: 0.0001, Q Loss: 0.0080
[IQL] Epoch 12/40, V Loss: 0.0002, Q Loss: 0.0080
[IQL] Epoch 13/40, V Loss: 0.0003, Q Loss: 0.0081
[IQL] Epoch 14/40, V Loss: 0.0002, Q Loss: 0.0078
[IQL] Epoch 15/40, V Loss: 0.0002, Q Loss: 0.0077
[IQL] Epoch 16/40, V Loss: 0.0003, Q Loss: 0.0078
[IQL] Epoch 17/40, V Loss: 0.0002, Q Loss: 0.0078
[IQL] Epoch 18/40, V Loss: 0.0002, Q Loss: 0.0075
[IQL] Epoch 19/40, V Loss: 0.0004, Q Loss: 0.0077
[IQL] Epoch 20/40, V Loss: 0.0003, Q Loss: 0.0076
[IQL] Epoch 21/40, V Loss: 0.0002, Q Loss: 0.0074
[IQL] Epoch 22/40, V Loss: 0.0004, Q Loss: 0.0076
[IQL] Epoch 23/40, V Loss: 0.0003, Q Loss: 0.0073
[IQL] Epoch 24/40, V Loss: 0.0002, Q Loss: 0.0071
[IQL] Epoch 25/40, V Loss: 0.0001, Q Loss: 0.0071
[IQL] Epoch 26/40, V Loss: 0.0002, Q Loss: 0.0070
[IQL] Epoch 27/40, V Loss: 0.0002, Q Loss: 0.0071
[IQL] Epoch 28/40, V Loss: 0.0002, Q Loss: 0.0070
[IQL] Epoch 29/40, V Loss: 0.0002, Q Loss: 0.0069
[IQL] Epoch 30/40, V Loss: 0.0001, Q Loss: 0.0067
[IQL] Epoch 31/40, V Loss: 0.0002, Q Loss: 0.0068
[IQL] Epoch 32/40, V Loss: 0.0002, Q Loss: 0.0068
[IQL] Epoch 33/40, V Loss: 0.0002, Q Loss: 0.0067
[IQL] Epoch 34/40, V Loss: 0.0002, Q Loss: 0.0067
[IQL] Epoch 35/40, V Loss: 0.0002, Q Loss: 0.0066
[IQL] Epoch 36/40, V Loss: 0.0002, Q Loss: 0.0065
[IQL] Epoch 37/40, V Loss: 0.0002, Q Loss: 0.0065
[IQL] Epoch 38/40, V Loss: 0.0002, Q Loss: 0.0065
[IQL] Epoch 39/40, V Loss: 0.0002, Q Loss: 0.0064
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_00.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_00.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 0)
[OPE] Behavior policy value J_old = 0.4146
[RL PPO] Reducing policy LR: 1.00e-04 → 1.00e-05
[Offline RL] Epoch 0/20, PPO Loss: -0.0421, Reg Loss: 0.0000, CD Loss: 0.3154
[Offline RL] Epoch 1/20, PPO Loss: -0.0477, Reg Loss: 0.0000, CD Loss: 0.0738
[Offline RL] Epoch 2/20, PPO Loss: -0.0468, Reg Loss: 0.0000, CD Loss: 0.0593
[Offline RL] Epoch 3/20, PPO Loss: -0.0469, Reg Loss: 0.0000, CD Loss: 0.0517
[Offline RL] Epoch 4/20, PPO Loss: -0.0460, Reg Loss: 0.0000, CD Loss: 0.0437
[Offline RL] Epoch 5/20, PPO Loss: -0.0474, Reg Loss: 0.0000, CD Loss: 0.0329
[Offline RL] Epoch 6/20, PPO Loss: -0.0474, Reg Loss: 0.0000, CD Loss: 0.0256
[Offline RL] Epoch 7/20, PPO Loss: -0.0449, Reg Loss: 0.0000, CD Loss: 0.0224
[Offline RL] Epoch 8/20, PPO Loss: -0.0456, Reg Loss: 0.0000, CD Loss: 0.0188
[Offline RL] Epoch 9/20, PPO Loss: -0.0445, Reg Loss: 0.0000, CD Loss: 0.0169
[Offline RL] Epoch 10/20, PPO Loss: -0.0451, Reg Loss: 0.0000, CD Loss: 0.0158
[Offline RL] Epoch 11/20, PPO Loss: -0.0456, Reg Loss: 0.0000, CD Loss: 0.0146
[Offline RL] Epoch 12/20, PPO Loss: -0.0469, Reg Loss: 0.0000, CD Loss: 0.0150
[Offline RL] Epoch 13/20, PPO Loss: -0.0472, Reg Loss: 0.0000, CD Loss: 0.0138
[Offline RL] Epoch 14/20, PPO Loss: -0.0477, Reg Loss: 0.0000, CD Loss: 0.0130
[Offline RL] Epoch 15/20, PPO Loss: -0.0479, Reg Loss: 0.0000, CD Loss: 0.0125
[Offline RL] Epoch 16/20, PPO Loss: -0.0479, Reg Loss: 0.0000, CD Loss: 0.0120
[Offline RL] Epoch 17/20, PPO Loss: -0.0473, Reg Loss: 0.0000, CD Loss: 0.0124
[Offline RL] Epoch 18/20, PPO Loss: -0.0480, Reg Loss: 0.0000, CD Loss: 0.0119
[Offline RL] Epoch 19/20, PPO Loss: -0.0477, Reg Loss: 0.0000, CD Loss: 0.0111
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_00.png
[OPE] Policy REJECTED: J_new=0.4264 ≤ J_old=0.4146 + δ=0.0207. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 0)
[Collect] 20 episodes, success=0.600, env_return=1022.39, rl_reward=0.60, steps=4000
[Data Collection] Success Rate: 0.600, EnvReturn: 1022.39, RLReward: 0.60, Episodes: 20, Steps: 4000
[Dataset] Merged 20 episodes (4000 steps) → total 24000 steps, 120 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.0447, Val Loss: 0.0094
[IL] Epoch 1/100, Loss: 0.0157, Val Loss: 0.0076
[IL] Epoch 2/100, Loss: 0.0139, Val Loss: 0.0078
[IL] Epoch 3/100, Loss: 0.0128, Val Loss: 0.0063
[IL] Epoch 4/100, Loss: 0.0122, Val Loss: 0.0052
[IL] Epoch 5/100, Loss: 0.0115, Val Loss: 0.0074
[IL] Epoch 6/100, Loss: 0.0107, Val Loss: 0.0072
[IL] Epoch 7/100, Loss: 0.0111, Val Loss: 0.0075
[IL] Epoch 8/100, Loss: 0.0109, Val Loss: 0.0060
[IL] Epoch 9/100, Loss: 0.0110, Val Loss: 0.0059
[IL] Epoch 10/100, Loss: 0.0096, Val Loss: 0.0069
[IL] Epoch 11/100, Loss: 0.0104, Val Loss: 0.0058
[IL] Epoch 12/100, Loss: 0.0094, Val Loss: 0.0065
[IL] Epoch 13/100, Loss: 0.0096, Val Loss: 0.0082
[IL] Epoch 14/100, Loss: 0.0092, Val Loss: 0.0054
[IL] Epoch 15/100, Loss: 0.0093, Val Loss: 0.0061
[IL] Epoch 16/100, Loss: 0.0096, Val Loss: 0.0068
[IL] Epoch 17/100, Loss: 0.0098, Val Loss: 0.0059
[IL] Epoch 18/100, Loss: 0.0090, Val Loss: 0.0068
[IL] Epoch 19/100, Loss: 0.0083, Val Loss: 0.0059
[IL] Epoch 20/100, Loss: 0.0084, Val Loss: 0.0079
[IL] Epoch 21/100, Loss: 0.0085, Val Loss: 0.0061
[IL] Epoch 22/100, Loss: 0.0086, Val Loss: 0.0110
[IL] Epoch 23/100, Loss: 0.0084, Val Loss: 0.0067
[IL] Epoch 24/100, Loss: 0.0083, Val Loss: 0.0054
[IL] Epoch 25/100, Loss: 0.0088, Val Loss: 0.0054
[IL] Epoch 26/100, Loss: 0.0086, Val Loss: 0.0078
[IL] Epoch 27/100, Loss: 0.0082, Val Loss: 0.0049
[IL] Epoch 28/100, Loss: 0.0077, Val Loss: 0.0064
[IL] Epoch 29/100, Loss: 0.0088, Val Loss: 0.0093
[IL] Epoch 30/100, Loss: 0.0082, Val Loss: 0.0099
[IL] Epoch 31/100, Loss: 0.0082, Val Loss: 0.0050
[IL] Epoch 32/100, Loss: 0.0078, Val Loss: 0.0072
[IL] Epoch 33/100, Loss: 0.0078, Val Loss: 0.0055
[IL] Epoch 34/100, Loss: 0.0081, Val Loss: 0.0065
[IL] Epoch 35/100, Loss: 0.0080, Val Loss: 0.0066
[IL] Epoch 36/100, Loss: 0.0078, Val Loss: 0.0079
[IL] Epoch 37/100, Loss: 0.0078, Val Loss: 0.0063
[IL] Epoch 38/100, Loss: 0.0076, Val Loss: 0.0083
[IL] Epoch 39/100, Loss: 0.0082, Val Loss: 0.0047
[IL] Epoch 40/100, Loss: 0.0079, Val Loss: 0.0058
[IL] Epoch 41/100, Loss: 0.0074, Val Loss: 0.0065
[IL] Epoch 42/100, Loss: 0.0076, Val Loss: 0.0088
[IL] Epoch 43/100, Loss: 0.0071, Val Loss: 0.0084
[IL] Epoch 44/100, Loss: 0.0075, Val Loss: 0.0075
[IL] Epoch 45/100, Loss: 0.0076, Val Loss: 0.0079
[IL] Epoch 46/100, Loss: 0.0069, Val Loss: 0.0051
[IL] Epoch 47/100, Loss: 0.0079, Val Loss: 0.0069
[IL] Epoch 48/100, Loss: 0.0079, Val Loss: 0.0085
[IL] Epoch 49/100, Loss: 0.0071, Val Loss: 0.0069
[IL] Epoch 50/100, Loss: 0.0070, Val Loss: 0.0099
[IL] Epoch 51/100, Loss: 0.0073, Val Loss: 0.0075
[IL] Epoch 52/100, Loss: 0.0073, Val Loss: 0.0064
[IL] Epoch 53/100, Loss: 0.0073, Val Loss: 0.0085
[IL] Epoch 54/100, Loss: 0.0074, Val Loss: 0.0061
[IL] Epoch 55/100, Loss: 0.0071, Val Loss: 0.0048
[IL] Epoch 56/100, Loss: 0.0066, Val Loss: 0.0075
[IL] Epoch 57/100, Loss: 0.0069, Val Loss: 0.0053
[IL] Epoch 58/100, Loss: 0.0069, Val Loss: 0.0059
[IL] Epoch 59/100, Loss: 0.0068, Val Loss: 0.0058
[IL] Epoch 60/100, Loss: 0.0071, Val Loss: 0.0052
[IL] Epoch 61/100, Loss: 0.0063, Val Loss: 0.0064
[IL] Epoch 62/100, Loss: 0.0067, Val Loss: 0.0085
[IL] Epoch 63/100, Loss: 0.0066, Val Loss: 0.0059
[IL] Epoch 64/100, Loss: 0.0066, Val Loss: 0.0076
[IL] Epoch 65/100, Loss: 0.0073, Val Loss: 0.0064
[IL] Epoch 66/100, Loss: 0.0068, Val Loss: 0.0104
[IL] Epoch 67/100, Loss: 0.0065, Val Loss: 0.0057
[IL] Epoch 68/100, Loss: 0.0071, Val Loss: 0.0071
[IL] Epoch 69/100, Loss: 0.0066, Val Loss: 0.0065
[IL] Epoch 70/100, Loss: 0.0066, Val Loss: 0.0065
[IL] Epoch 71/100, Loss: 0.0068, Val Loss: 0.0083
[IL] Epoch 72/100, Loss: 0.0063, Val Loss: 0.0061
[IL] Epoch 73/100, Loss: 0.0065, Val Loss: 0.0081
[IL] Epoch 74/100, Loss: 0.0063, Val Loss: 0.0060
[IL] Epoch 75/100, Loss: 0.0066, Val Loss: 0.0058
[IL] Epoch 76/100, Loss: 0.0061, Val Loss: 0.0058
[IL] Epoch 77/100, Loss: 0.0071, Val Loss: 0.0085
[IL] Epoch 78/100, Loss: 0.0065, Val Loss: 0.0077
[IL] Epoch 79/100, Loss: 0.0066, Val Loss: 0.0069
[IL] Epoch 80/100, Loss: 0.0060, Val Loss: 0.0070
[IL] Epoch 81/100, Loss: 0.0061, Val Loss: 0.0074
[IL] Epoch 82/100, Loss: 0.0063, Val Loss: 0.0073
[IL] Epoch 83/100, Loss: 0.0061, Val Loss: 0.0059
[IL] Epoch 84/100, Loss: 0.0063, Val Loss: 0.0112
[IL] Epoch 85/100, Loss: 0.0068, Val Loss: 0.0082
[IL] Epoch 86/100, Loss: 0.0061, Val Loss: 0.0086
[IL] Epoch 87/100, Loss: 0.0065, Val Loss: 0.0075
[IL] Epoch 88/100, Loss: 0.0061, Val Loss: 0.0076
[IL] Epoch 89/100, Loss: 0.0067, Val Loss: 0.0061
[IL] Epoch 90/100, Loss: 0.0060, Val Loss: 0.0060
[IL] Epoch 91/100, Loss: 0.0064, Val Loss: 0.0064
[IL] Epoch 92/100, Loss: 0.0059, Val Loss: 0.0061
[IL] Epoch 93/100, Loss: 0.0060, Val Loss: 0.0085
[IL] Epoch 94/100, Loss: 0.0059, Val Loss: 0.0111
[IL] Epoch 95/100, Loss: 0.0062, Val Loss: 0.0059
[IL] Epoch 96/100, Loss: 0.0064, Val Loss: 0.0083
[IL] Epoch 97/100, Loss: 0.0058, Val Loss: 0.0107
[IL] Epoch 98/100, Loss: 0.0058, Val Loss: 0.0093
[IL] Epoch 99/100, Loss: 0.0058, Val Loss: 0.0067
test_mean_score: 0.7
[IL] Eval - Success Rate: 0.700
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_00.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_0.ckpt

================================================================================
               OFFLINE RL ITERATION 2/5
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 1)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 21230 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-37.23734 | val=0.00055 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-68.10624 | val=0.00029 | no-improve=0/5
[TransitionModel] Epoch   40 | train=-72.59191 | val=0.00027 | no-improve=4/5
[TransitionModel] Epoch   41 | train=-73.07665 | val=0.00028 | no-improve=5/5
[TransitionModel] Training complete. Elites=[4, 5, 0, 3, 1], val_loss=0.00025

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 1)
[IQL] Epoch 0/40, V Loss: 0.0003, Q Loss: 0.0062
[IQL] Epoch 1/40, V Loss: 0.0002, Q Loss: 0.0061
[IQL] Epoch 2/40, V Loss: 0.0001, Q Loss: 0.0060
[IQL] Epoch 3/40, V Loss: 0.0002, Q Loss: 0.0060
[IQL] Epoch 4/40, V Loss: 0.0002, Q Loss: 0.0061
[IQL] Epoch 5/40, V Loss: 0.0003, Q Loss: 0.0061
[IQL] Epoch 6/40, V Loss: 0.0001, Q Loss: 0.0059
[IQL] Epoch 7/40, V Loss: 0.0002, Q Loss: 0.0059
[IQL] Epoch 8/40, V Loss: 0.0002, Q Loss: 0.0059
[IQL] Epoch 9/40, V Loss: 0.0003, Q Loss: 0.0061
[IQL] Epoch 10/40, V Loss: 0.0003, Q Loss: 0.0059
[IQL] Epoch 11/40, V Loss: 0.0003, Q Loss: 0.0058
[IQL] Epoch 12/40, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 13/40, V Loss: 0.0002, Q Loss: 0.0058
[IQL] Epoch 14/40, V Loss: 0.0002, Q Loss: 0.0057
[IQL] Epoch 15/40, V Loss: 0.0003, Q Loss: 0.0056
[IQL] Epoch 16/40, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 17/40, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 18/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 19/40, V Loss: 0.0003, Q Loss: 0.0056
[IQL] Epoch 20/40, V Loss: 0.0003, Q Loss: 0.0055
[IQL] Epoch 21/40, V Loss: 0.0003, Q Loss: 0.0055
[IQL] Epoch 22/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 23/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 24/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 25/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 26/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 27/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 28/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 29/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 30/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 31/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 32/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 33/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 34/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 35/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 36/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 37/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 38/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 39/40, V Loss: 0.0003, Q Loss: 0.0052
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_01.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_01.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 1)
[OPE] Behavior policy value J_old = 0.6264
[RL PPO] Reducing policy LR: 1.00e-04 → 1.00e-05
[Offline RL] Epoch 0/20, PPO Loss: -0.0440, Reg Loss: 0.0000, CD Loss: 0.2783
[Offline RL] Epoch 1/20, PPO Loss: -0.0451, Reg Loss: 0.0000, CD Loss: 0.0666
[Offline RL] Epoch 2/20, PPO Loss: -0.0454, Reg Loss: 0.0000, CD Loss: 0.0554
[Offline RL] Epoch 3/20, PPO Loss: -0.0448, Reg Loss: 0.0000, CD Loss: 0.0447
[Offline RL] Epoch 4/20, PPO Loss: -0.0450, Reg Loss: 0.0000, CD Loss: 0.0367
[Offline RL] Epoch 5/20, PPO Loss: -0.0450, Reg Loss: 0.0000, CD Loss: 0.0297
[Offline RL] Epoch 6/20, PPO Loss: -0.0470, Reg Loss: 0.0000, CD Loss: 0.0243
[Offline RL] Epoch 7/20, PPO Loss: -0.0473, Reg Loss: 0.0000, CD Loss: 0.0233
[Offline RL] Epoch 8/20, PPO Loss: -0.0468, Reg Loss: 0.0000, CD Loss: 0.0206
[Offline RL] Epoch 9/20, PPO Loss: -0.0466, Reg Loss: 0.0000, CD Loss: 0.0201
[Offline RL] Epoch 10/20, PPO Loss: -0.0454, Reg Loss: 0.0000, CD Loss: 0.0195
[Offline RL] Epoch 11/20, PPO Loss: -0.0462, Reg Loss: 0.0000, CD Loss: 0.0183
[Offline RL] Epoch 12/20, PPO Loss: -0.0461, Reg Loss: 0.0000, CD Loss: 0.0182
[Offline RL] Epoch 13/20, PPO Loss: -0.0457, Reg Loss: 0.0000, CD Loss: 0.0182
[Offline RL] Epoch 14/20, PPO Loss: -0.0466, Reg Loss: 0.0000, CD Loss: 0.0173
[Offline RL] Epoch 15/20, PPO Loss: -0.0467, Reg Loss: 0.0000, CD Loss: 0.0176
[Offline RL] Epoch 16/20, PPO Loss: -0.0475, Reg Loss: 0.0000, CD Loss: 0.0170
[Offline RL] Epoch 17/20, PPO Loss: -0.0482, Reg Loss: 0.0000, CD Loss: 0.0185
[Offline RL] Epoch 18/20, PPO Loss: -0.0474, Reg Loss: 0.0000, CD Loss: 0.0187
[Offline RL] Epoch 19/20, PPO Loss: -0.0474, Reg Loss: 0.0000, CD Loss: 0.0180
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_01.png
[OPE] Policy REJECTED: J_new=0.6335 ≤ J_old=0.6264 + δ=0.0313. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 1)
[Collect] 20 episodes, success=0.900, env_return=1257.14, rl_reward=0.90, steps=4000
[Data Collection] Success Rate: 0.900, EnvReturn: 1257.14, RLReward: 0.90, Episodes: 20, Steps: 4000
[Dataset] Merged 20 episodes (4000 steps) → total 28000 steps, 140 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.0095, Val Loss: 0.0104
[IL] Epoch 1/100, Loss: 0.0083, Val Loss: 0.0076
[IL] Epoch 2/100, Loss: 0.0086, Val Loss: 0.0071
[IL] Epoch 3/100, Loss: 0.0076, Val Loss: 0.0080
[IL] Epoch 4/100, Loss: 0.0081, Val Loss: 0.0089
[IL] Epoch 5/100, Loss: 0.0078, Val Loss: 0.0076
[IL] Epoch 6/100, Loss: 0.0082, Val Loss: 0.0068
[IL] Epoch 7/100, Loss: 0.0080, Val Loss: 0.0063
[IL] Epoch 8/100, Loss: 0.0077, Val Loss: 0.0116
[IL] Epoch 9/100, Loss: 0.0075, Val Loss: 0.0109
[IL] Epoch 10/100, Loss: 0.0081, Val Loss: 0.0082
[IL] Epoch 11/100, Loss: 0.0073, Val Loss: 0.0093
[IL] Epoch 12/100, Loss: 0.0073, Val Loss: 0.0084
[IL] Epoch 13/100, Loss: 0.0071, Val Loss: 0.0066
[IL] Epoch 14/100, Loss: 0.0074, Val Loss: 0.0081
[IL] Epoch 15/100, Loss: 0.0074, Val Loss: 0.0077
[IL] Epoch 16/100, Loss: 0.0071, Val Loss: 0.0114
[IL] Epoch 17/100, Loss: 0.0072, Val Loss: 0.0072
[IL] Epoch 18/100, Loss: 0.0067, Val Loss: 0.0090
[IL] Epoch 19/100, Loss: 0.0082, Val Loss: 0.0070
[IL] Epoch 20/100, Loss: 0.0072, Val Loss: 0.0101
[IL] Epoch 21/100, Loss: 0.0077, Val Loss: 0.0078
[IL] Epoch 22/100, Loss: 0.0070, Val Loss: 0.0069
[IL] Epoch 23/100, Loss: 0.0077, Val Loss: 0.0106
[IL] Epoch 24/100, Loss: 0.0102, Val Loss: 0.0090
[IL] Epoch 25/100, Loss: 0.0073, Val Loss: 0.0097
[IL] Epoch 26/100, Loss: 0.0077, Val Loss: 0.0085
[IL] Epoch 27/100, Loss: 0.0069, Val Loss: 0.0075
[IL] Epoch 28/100, Loss: 0.0068, Val Loss: 0.0081
[IL] Epoch 29/100, Loss: 0.0069, Val Loss: 0.0062
[IL] Epoch 30/100, Loss: 0.0068, Val Loss: 0.0084
[IL] Epoch 31/100, Loss: 0.0065, Val Loss: 0.0083
[IL] Epoch 32/100, Loss: 0.0071, Val Loss: 0.0121
[IL] Epoch 33/100, Loss: 0.0079, Val Loss: 0.0072
[IL] Epoch 34/100, Loss: 0.0072, Val Loss: 0.0140
[IL] Epoch 35/100, Loss: 0.0115, Val Loss: 0.0070
[IL] Epoch 36/100, Loss: 0.0071, Val Loss: 0.0096
[IL] Epoch 37/100, Loss: 0.0074, Val Loss: 0.0279
[IL] Epoch 38/100, Loss: 0.0124, Val Loss: 0.0111
[IL] Epoch 39/100, Loss: 0.0092, Val Loss: 0.0105
[IL] Epoch 40/100, Loss: 0.0077, Val Loss: 0.0076
[IL] Epoch 41/100, Loss: 0.0070, Val Loss: 0.0086
[IL] Epoch 42/100, Loss: 0.0080, Val Loss: 0.0108
[IL] Epoch 43/100, Loss: 0.0089, Val Loss: 0.0079
[IL] Epoch 44/100, Loss: 0.0069, Val Loss: 0.0093
[IL] Epoch 45/100, Loss: 0.0070, Val Loss: 0.0067
[IL] Epoch 46/100, Loss: 0.0077, Val Loss: 0.0193
[IL] Epoch 47/100, Loss: 0.0110, Val Loss: 0.0089
[IL] Epoch 48/100, Loss: 0.0070, Val Loss: 0.0082
[IL] Epoch 49/100, Loss: 0.0067, Val Loss: 0.0089
[IL] Epoch 50/100, Loss: 0.0073, Val Loss: 0.0123
[IL] Epoch 51/100, Loss: 0.0062, Val Loss: 0.0082
[IL] Epoch 52/100, Loss: 0.0069, Val Loss: 0.0408
[IL] Epoch 53/100, Loss: 0.0173, Val Loss: 0.0086
[IL] Epoch 54/100, Loss: 0.0080, Val Loss: 0.0092
[IL] Epoch 55/100, Loss: 0.0075, Val Loss: 0.0324
[IL] Epoch 56/100, Loss: 0.0210, Val Loss: 0.0107
[IL] Epoch 57/100, Loss: 0.0114, Val Loss: 0.0092
[IL] Epoch 58/100, Loss: 0.0102, Val Loss: 0.0136
[IL] Epoch 59/100, Loss: 0.0096, Val Loss: 0.0100
[IL] Epoch 60/100, Loss: 0.0078, Val Loss: 0.0082
[IL] Epoch 61/100, Loss: 0.0077, Val Loss: 0.0107
[IL] Epoch 62/100, Loss: 0.0139, Val Loss: 0.0292
[IL] Epoch 63/100, Loss: 0.0153, Val Loss: 0.0137
[IL] Epoch 64/100, Loss: 0.0115, Val Loss: 0.0083
[IL] Epoch 65/100, Loss: 0.0079, Val Loss: 0.0077
[IL] Epoch 66/100, Loss: 0.0069, Val Loss: 0.0080
[IL] Epoch 67/100, Loss: 0.0072, Val Loss: 0.0156
[IL] Epoch 68/100, Loss: 0.0094, Val Loss: 0.0171
[IL] Epoch 69/100, Loss: 0.0100, Val Loss: 0.0089
[IL] Epoch 70/100, Loss: 0.0071, Val Loss: 0.0088
[IL] Epoch 71/100, Loss: 0.0065, Val Loss: 0.0068
[IL] Epoch 72/100, Loss: 0.0066, Val Loss: 0.0102
[IL] Epoch 73/100, Loss: 0.0064, Val Loss: 0.0072
[IL] Epoch 74/100, Loss: 0.0061, Val Loss: 0.0088
[IL] Epoch 75/100, Loss: 0.0058, Val Loss: 0.0103
[IL] Epoch 76/100, Loss: 0.0065, Val Loss: 0.0099
[IL] Epoch 77/100, Loss: 0.0062, Val Loss: 0.0095
[IL] Epoch 78/100, Loss: 0.0066, Val Loss: 0.0094
[IL] Epoch 79/100, Loss: 0.0064, Val Loss: 0.0078
[IL] Epoch 80/100, Loss: 0.0063, Val Loss: 0.0064
[IL] Epoch 81/100, Loss: 0.0059, Val Loss: 0.0095
[IL] Epoch 82/100, Loss: 0.0068, Val Loss: 0.0068
[IL] Epoch 83/100, Loss: 0.0067, Val Loss: 0.0085
[IL] Epoch 84/100, Loss: 0.0066, Val Loss: 0.0077
[IL] Epoch 85/100, Loss: 0.0061, Val Loss: 0.0104
[IL] Epoch 86/100, Loss: 0.0056, Val Loss: 0.0055
[IL] Epoch 87/100, Loss: 0.0059, Val Loss: 0.0065
[IL] Epoch 88/100, Loss: 0.0061, Val Loss: 0.0085
[IL] Epoch 89/100, Loss: 0.0060, Val Loss: 0.0097
[IL] Epoch 90/100, Loss: 0.0069, Val Loss: 0.0075
[IL] Epoch 91/100, Loss: 0.0061, Val Loss: 0.0102
[IL] Epoch 92/100, Loss: 0.0067, Val Loss: 0.0105
[IL] Epoch 93/100, Loss: 0.0064, Val Loss: 0.0075
[IL] Epoch 94/100, Loss: 0.0069, Val Loss: 0.0087
[IL] Epoch 95/100, Loss: 0.0061, Val Loss: 0.0087
[IL] Epoch 96/100, Loss: 0.0063, Val Loss: 0.0133
[IL] Epoch 97/100, Loss: 0.0091, Val Loss: 0.0105
[IL] Epoch 98/100, Loss: 0.0065, Val Loss: 0.0104
[IL] Epoch 99/100, Loss: 0.0066, Val Loss: 0.0084
test_mean_score: 0.65
[IL] Eval - Success Rate: 0.650
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_01.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_1.ckpt

================================================================================
               OFFLINE RL ITERATION 3/5
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 2)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 25090 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-64.17257 | val=0.00047 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-76.85288 | val=0.00032 | no-improve=1/5
[TransitionModel] Epoch   40 | train=-82.38041 | val=0.00029 | no-improve=0/5
[TransitionModel] Epoch   60 | train=-87.68386 | val=0.00027 | no-improve=0/5
[TransitionModel] Epoch   80 | train=-92.89704 | val=0.00026 | no-improve=0/5
[TransitionModel] Epoch   87 | train=-94.85630 | val=0.00027 | no-improve=5/5
[TransitionModel] Training complete. Elites=[5, 3, 4, 1, 0], val_loss=0.00025

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 2)
[IQL] Epoch 0/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 1/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 2/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 3/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 4/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 5/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 6/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 7/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 8/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 9/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 10/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 11/40, V Loss: 0.0001, Q Loss: 0.0052
[IQL] Epoch 12/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 13/40, V Loss: 0.0001, Q Loss: 0.0052
[IQL] Epoch 14/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 15/40, V Loss: 0.0003, Q Loss: 0.0051
[IQL] Epoch 16/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 17/40, V Loss: 0.0003, Q Loss: 0.0050
[IQL] Epoch 18/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 19/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 20/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 21/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 22/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 23/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 24/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 25/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 26/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 27/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 28/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 29/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 30/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 31/40, V Loss: 0.0001, Q Loss: 0.0051
[IQL] Epoch 32/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 33/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 34/40, V Loss: 0.0004, Q Loss: 0.0056
[IQL] Epoch 35/40, V Loss: 0.0001, Q Loss: 0.0052
[IQL] Epoch 36/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 37/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 38/40, V Loss: 0.0001, Q Loss: 0.0051
[IQL] Epoch 39/40, V Loss: 0.0001, Q Loss: 0.0050
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_02.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_02.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 2)
[OPE] Behavior policy value J_old = 0.6067
[RL PPO] Reducing policy LR: 1.00e-04 → 1.00e-05
[Offline RL] Epoch 0/20, PPO Loss: -0.0429, Reg Loss: 0.0000, CD Loss: 0.2462
[Offline RL] Epoch 1/20, PPO Loss: -0.0451, Reg Loss: 0.0000, CD Loss: 0.0619
[Offline RL] Epoch 2/20, PPO Loss: -0.0444, Reg Loss: 0.0000, CD Loss: 0.0456
[Offline RL] Epoch 3/20, PPO Loss: -0.0429, Reg Loss: 0.0000, CD Loss: 0.0367
[Offline RL] Epoch 4/20, PPO Loss: -0.0428, Reg Loss: 0.0000, CD Loss: 0.0303
[Offline RL] Epoch 5/20, PPO Loss: -0.0414, Reg Loss: 0.0000, CD Loss: 0.0246
[Offline RL] Epoch 6/20, PPO Loss: -0.0414, Reg Loss: 0.0000, CD Loss: 0.0223
[Offline RL] Epoch 7/20, PPO Loss: -0.0418, Reg Loss: 0.0000, CD Loss: 0.0200
[Offline RL] Epoch 8/20, PPO Loss: -0.0417, Reg Loss: 0.0000, CD Loss: 0.0173
[Offline RL] Epoch 9/20, PPO Loss: -0.0424, Reg Loss: 0.0000, CD Loss: 0.0186
[Offline RL] Epoch 10/20, PPO Loss: -0.0402, Reg Loss: 0.0000, CD Loss: 0.0160
[Offline RL] Epoch 11/20, PPO Loss: -0.0397, Reg Loss: 0.0000, CD Loss: 0.0150
[Offline RL] Epoch 12/20, PPO Loss: -0.0414, Reg Loss: 0.0000, CD Loss: 0.0146
[Offline RL] Epoch 13/20, PPO Loss: -0.0424, Reg Loss: 0.0000, CD Loss: 0.0144
[Offline RL] Epoch 14/20, PPO Loss: -0.0422, Reg Loss: 0.0000, CD Loss: 0.0163
[Offline RL] Epoch 15/20, PPO Loss: -0.0423, Reg Loss: 0.0000, CD Loss: 0.0153
[Offline RL] Epoch 16/20, PPO Loss: -0.0415, Reg Loss: 0.0000, CD Loss: 0.0142
[Offline RL] Epoch 17/20, PPO Loss: -0.0416, Reg Loss: 0.0000, CD Loss: 0.0155
[Offline RL] Epoch 18/20, PPO Loss: -0.0421, Reg Loss: 0.0000, CD Loss: 0.0146
[Offline RL] Epoch 19/20, PPO Loss: -0.0429, Reg Loss: 0.0000, CD Loss: 0.0172
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_02.png
[OPE] Policy REJECTED: J_new=0.6182 ≤ J_old=0.6067 + δ=0.0303. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 2)
[Collect] 20 episodes, success=0.500, env_return=995.99, rl_reward=0.50, steps=4000
[Data Collection] Success Rate: 0.500, EnvReturn: 995.99, RLReward: 0.50, Episodes: 20, Steps: 4000
[Dataset] Merged 20 episodes (4000 steps) → total 32000 steps, 160 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.0286, Val Loss: 0.0092
[IL] Epoch 1/100, Loss: 0.0143, Val Loss: 0.0077
[IL] Epoch 2/100, Loss: 0.0131, Val Loss: 0.0110
[IL] Epoch 3/100, Loss: 0.0122, Val Loss: 0.0124
[IL] Epoch 4/100, Loss: 0.0123, Val Loss: 0.0095
[IL] Epoch 5/100, Loss: 0.0114, Val Loss: 0.0073
[IL] Epoch 6/100, Loss: 0.0111, Val Loss: 0.0082
[IL] Epoch 7/100, Loss: 0.0108, Val Loss: 0.0071
[IL] Epoch 8/100, Loss: 0.0106, Val Loss: 0.0100
[IL] Epoch 9/100, Loss: 0.0107, Val Loss: 0.0084
[IL] Epoch 10/100, Loss: 0.0101, Val Loss: 0.0060
[IL] Epoch 11/100, Loss: 0.0096, Val Loss: 0.0073
[IL] Epoch 12/100, Loss: 0.0100, Val Loss: 0.0069
[IL] Epoch 13/100, Loss: 0.0096, Val Loss: 0.0055
[IL] Epoch 14/100, Loss: 0.0096, Val Loss: 0.0069
[IL] Epoch 15/100, Loss: 0.0092, Val Loss: 0.0077
[IL] Epoch 16/100, Loss: 0.0094, Val Loss: 0.0072
[IL] Epoch 17/100, Loss: 0.0092, Val Loss: 0.0079
[IL] Epoch 18/100, Loss: 0.0094, Val Loss: 0.0090
[IL] Epoch 19/100, Loss: 0.0092, Val Loss: 0.0071
[IL] Epoch 20/100, Loss: 0.0096, Val Loss: 0.0061
[IL] Epoch 21/100, Loss: 0.0089, Val Loss: 0.0113
[IL] Epoch 22/100, Loss: 0.0090, Val Loss: 0.0081
[IL] Epoch 23/100, Loss: 0.0088, Val Loss: 0.0065
[IL] Epoch 24/100, Loss: 0.0093, Val Loss: 0.0105
[IL] Epoch 25/100, Loss: 0.0097, Val Loss: 0.0078
[IL] Epoch 26/100, Loss: 0.0092, Val Loss: 0.0075
[IL] Epoch 27/100, Loss: 0.0088, Val Loss: 0.0072
[IL] Epoch 28/100, Loss: 0.0088, Val Loss: 0.0095
[IL] Epoch 29/100, Loss: 0.0088, Val Loss: 0.0104
[IL] Epoch 30/100, Loss: 0.0101, Val Loss: 0.0087
[IL] Epoch 31/100, Loss: 0.0082, Val Loss: 0.0076
[IL] Epoch 32/100, Loss: 0.0088, Val Loss: 0.0087
[IL] Epoch 33/100, Loss: 0.0083, Val Loss: 0.0077
[IL] Epoch 34/100, Loss: 0.0079, Val Loss: 0.0079
[IL] Epoch 35/100, Loss: 0.0086, Val Loss: 0.0069
[IL] Epoch 36/100, Loss: 0.0084, Val Loss: 0.0060
[IL] Epoch 37/100, Loss: 0.0082, Val Loss: 0.0080
[IL] Epoch 38/100, Loss: 0.0083, Val Loss: 0.0084
[IL] Epoch 39/100, Loss: 0.0082, Val Loss: 0.0066
[IL] Epoch 40/100, Loss: 0.0081, Val Loss: 0.0089
[IL] Epoch 41/100, Loss: 0.0080, Val Loss: 0.0061
[IL] Epoch 42/100, Loss: 0.0081, Val Loss: 0.0085
[IL] Epoch 43/100, Loss: 0.0079, Val Loss: 0.0093
[IL] Epoch 44/100, Loss: 0.0083, Val Loss: 0.0095
[IL] Epoch 45/100, Loss: 0.0082, Val Loss: 0.0093
[IL] Epoch 46/100, Loss: 0.0075, Val Loss: 0.0071
[IL] Epoch 47/100, Loss: 0.0082, Val Loss: 0.0130
[IL] Epoch 48/100, Loss: 0.0100, Val Loss: 0.0070
[IL] Epoch 49/100, Loss: 0.0083, Val Loss: 0.0098
[IL] Epoch 50/100, Loss: 0.0076, Val Loss: 0.0076
[IL] Epoch 51/100, Loss: 0.0075, Val Loss: 0.0080
[IL] Epoch 52/100, Loss: 0.0082, Val Loss: 0.0101
[IL] Epoch 53/100, Loss: 0.0076, Val Loss: 0.0113
[IL] Epoch 54/100, Loss: 0.0081, Val Loss: 0.0091
[IL] Epoch 55/100, Loss: 0.0074, Val Loss: 0.0086
[IL] Epoch 56/100, Loss: 0.0079, Val Loss: 0.0073
[IL] Epoch 57/100, Loss: 0.0081, Val Loss: 0.0115
[IL] Epoch 58/100, Loss: 0.0078, Val Loss: 0.0128
[IL] Epoch 59/100, Loss: 0.0070, Val Loss: 0.0081
[IL] Epoch 60/100, Loss: 0.0075, Val Loss: 0.0108
[IL] Epoch 61/100, Loss: 0.0078, Val Loss: 0.0073
[IL] Epoch 62/100, Loss: 0.0078, Val Loss: 0.0092
[IL] Epoch 63/100, Loss: 0.0079, Val Loss: 0.0085
[IL] Epoch 64/100, Loss: 0.0071, Val Loss: 0.0104
[IL] Epoch 65/100, Loss: 0.0070, Val Loss: 0.0097
[IL] Epoch 66/100, Loss: 0.0075, Val Loss: 0.0080
[IL] Epoch 67/100, Loss: 0.0072, Val Loss: 0.0118
[IL] Epoch 68/100, Loss: 0.0073, Val Loss: 0.0093
[IL] Epoch 69/100, Loss: 0.0074, Val Loss: 0.0069
[IL] Epoch 70/100, Loss: 0.0075, Val Loss: 0.0111
[IL] Epoch 71/100, Loss: 0.0073, Val Loss: 0.0088
[IL] Epoch 72/100, Loss: 0.0076, Val Loss: 0.0100
[IL] Epoch 73/100, Loss: 0.0074, Val Loss: 0.0068
[IL] Epoch 74/100, Loss: 0.0072, Val Loss: 0.0083
[IL] Epoch 75/100, Loss: 0.0071, Val Loss: 0.0118
[IL] Epoch 76/100, Loss: 0.0071, Val Loss: 0.0079
[IL] Epoch 77/100, Loss: 0.0073, Val Loss: 0.0124
[IL] Epoch 78/100, Loss: 0.0084, Val Loss: 0.0091
[IL] Epoch 79/100, Loss: 0.0074, Val Loss: 0.0097
[IL] Epoch 80/100, Loss: 0.0073, Val Loss: 0.0063
[IL] Epoch 81/100, Loss: 0.0069, Val Loss: 0.0084
[IL] Epoch 82/100, Loss: 0.0071, Val Loss: 0.0114
[IL] Epoch 83/100, Loss: 0.0070, Val Loss: 0.0115
[IL] Epoch 84/100, Loss: 0.0073, Val Loss: 0.0112
[IL] Epoch 85/100, Loss: 0.0071, Val Loss: 0.0108
[IL] Epoch 86/100, Loss: 0.0068, Val Loss: 0.0109
[IL] Epoch 87/100, Loss: 0.0074, Val Loss: 0.0104
[IL] Epoch 88/100, Loss: 0.0069, Val Loss: 0.0096
[IL] Epoch 89/100, Loss: 0.0070, Val Loss: 0.0089
[IL] Epoch 90/100, Loss: 0.0067, Val Loss: 0.0080
[IL] Epoch 91/100, Loss: 0.0067, Val Loss: 0.0074
[IL] Epoch 92/100, Loss: 0.0066, Val Loss: 0.0083
[IL] Epoch 93/100, Loss: 0.0071, Val Loss: 0.0124
[IL] Epoch 94/100, Loss: 0.0070, Val Loss: 0.0096
[IL] Epoch 95/100, Loss: 0.0076, Val Loss: 0.0104
[IL] Epoch 96/100, Loss: 0.0071, Val Loss: 0.0056
[IL] Epoch 97/100, Loss: 0.0068, Val Loss: 0.0087
[IL] Epoch 98/100, Loss: 0.0066, Val Loss: 0.0115
[IL] Epoch 99/100, Loss: 0.0068, Val Loss: 0.0103
test_mean_score: 0.5
[IL] Eval - Success Rate: 0.500
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_02.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_2.ckpt

================================================================================
               OFFLINE RL ITERATION 4/5
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 3)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 28950 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-78.15644 | val=0.00055 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-99.24724 | val=0.00039 | no-improve=0/5
[TransitionModel] Epoch   31 | train=-102.76687 | val=0.00036 | no-improve=5/5
[TransitionModel] Training complete. Elites=[2, 1, 3, 5, 6], val_loss=0.00034

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 3)
[IQL] Epoch 0/40, V Loss: 0.0001, Q Loss: 0.0051
[IQL] Epoch 1/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 2/40, V Loss: 0.0001, Q Loss: 0.0053
[IQL] Epoch 3/40, V Loss: 0.0002, Q Loss: 0.0058
[IQL] Epoch 4/40, V Loss: 0.0001, Q Loss: 0.0053
[IQL] Epoch 5/40, V Loss: 0.0001, Q Loss: 0.0051
[IQL] Epoch 6/40, V Loss: 0.0001, Q Loss: 0.0051
[IQL] Epoch 7/40, V Loss: 0.0001, Q Loss: 0.0050
[IQL] Epoch 8/40, V Loss: 0.0001, Q Loss: 0.0051
[IQL] Epoch 9/40, V Loss: 0.0001, Q Loss: 0.0051
[IQL] Epoch 10/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 11/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 12/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 13/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 14/40, V Loss: 0.0001, Q Loss: 0.0050
[IQL] Epoch 15/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 16/40, V Loss: 0.0001, Q Loss: 0.0050
[IQL] Epoch 17/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 18/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 19/40, V Loss: 0.0001, Q Loss: 0.0050
[IQL] Epoch 20/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 21/40, V Loss: 0.0001, Q Loss: 0.0051
[IQL] Epoch 22/40, V Loss: 0.0001, Q Loss: 0.0050
[IQL] Epoch 23/40, V Loss: 0.0001, Q Loss: 0.0051
[IQL] Epoch 24/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 25/40, V Loss: 0.0001, Q Loss: 0.0051
[IQL] Epoch 26/40, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 27/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 28/40, V Loss: 0.0001, Q Loss: 0.0051
