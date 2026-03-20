Job start at 2026-03-20 10:33:50
Job run at:
   Static hostname: localhost.localdomain
Transient hostname: r8l40-a06.ib.future.cn
         Icon name: computer-server
           Chassis: server
        Machine ID: 0015f69f25f84ea8a6acc9f89250fcb2
           Boot ID: a8482d2006984ff68b034529138a7b89
  Operating System: Rocky Linux 8.7 (Green Obsidian)
       CPE OS Name: cpe:/o:rocky:rocky:8:GA
            Kernel: Linux 4.18.0-425.10.1.el8_7.x86_64
      Architecture: x86-64
Filesystem                                        Size  Used Avail Use% Mounted on
/dev/mapper/rl-root                               376G   24G  352G   7% /
/dev/nvme4n1p1                                    3.5T   48G  3.5T   2% /tmp
/dev/nvme1n1p1                                    3.5T   25G  3.5T   1% /local
/dev/mapper/rl-var                                512G   15G  498G   3% /var
/dev/nvme0n1p2                                    2.0G  366M  1.7G  18% /boot
/dev/nvme0n1p1                                    599M  5.8M  594M   1% /boot/efi
/dev/nvme3n1p1                                    3.5T   39G  3.5T   2% /local/nfscache
ssd.nas00.future.cn:/rocky8_home                   16G   15G  1.9G  89% /home
ssd.nas00.future.cn:/rocky8_workspace             400G     0  400G   0% /workspace
ssd.nas00.future.cn:/rocky8_tools                 5.0T   99G  5.0T   2% /tools
ssd.nas00.future.cn:/centos7_home                  16G  4.2G   12G  26% /centos7/home
ssd.nas00.future.cn:/centos7_workspace            400G     0  400G   0% /centos7/workspace
ssd.nas00.future.cn:/centos7_tools                5.0T  235G  4.8T   5% /centos7/tools
ssd.nas00.future.cn:/eda-tools                    8.0T  6.3T  1.8T  79% /centos7/eda-tools
hdd.nas00.future.cn:/share_personal               500G     0  500G   0% /share/personal
zone05.nas01.future.cn:/NAS_HPC_collab_codemodel   40T   37T  3.7T  91% /share/collab/codemodel
ext-zone00.nas02.future.cn:/nfs_global            407T  397T   11T  98% /nfs_global
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
          /home  14454M  16384M  20480M            168k       0       0        

############### /workspace
Disk quotas for user yangrongzheng (uid 6215): 
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
     /workspace      0K    400G    500G               1       0       0        

############### /nfs_global
Disk quotas for user yangrongzheng (uid 6215): 
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
    /nfs_global    363G   5120G   7168G            350k   5000k  10000k        

############### /lustre
Disk quotas for usr yangrongzheng (uid 6215):
     Filesystem    used   quota   limit   grace   files   quota   limit   grace
        /lustre      0k      8T     10T       -       0  3000000 36000000       -
uid 6215 is using default block quota setting
uid 6215 is using default file quota setting
name, driver_version, power.limit [W]
NVIDIA L40, 570.124.06, 275.00 W
NVIDIA L40, 570.124.06, 275.00 W
NVIDIA L40, 570.124.06, 275.00 W
NVIDIA L40, 570.124.06, 275.00 W
NVIDIA L40, 570.124.06, 275.00 W
NVIDIA L40, 570.124.06, 275.00 W
NVIDIA L40, 570.124.06, 275.00 W
NVIDIA L40, 570.124.06, 275.00 W
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
  ppo_epochs: 10
  ppo_inner_steps: 1
  collection_episodes: 20
  cd_every: 5
  lambda_cd: 0
  rl_policy_lr: 2.0e-05
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
runtime:
  collection_policy: ddim
  collection_use_ema: false
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
[2026-03-20 10:34:18,295][diffusion_policy_3d.model.diffusion.conditional_unet1d][INFO] - number of parameters: 2.550744e+08
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
[2026-03-20 10:34:19,899][diffusion_policy_3d.model.diffusion.conditional_unet1d][INFO] - number of parameters: 2.550744e+08
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
               OFFLINE RL ITERATION 1/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 0)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 17370 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=5.41231 | val=0.00466 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-31.27199 | val=0.00050 | no-improve=0/5
[TransitionModel] Epoch   40 | train=-37.28982 | val=0.00038 | no-improve=0/5
[TransitionModel] Epoch   60 | train=-41.48960 | val=0.00032 | no-improve=0/5
[TransitionModel] Epoch   80 | train=-45.43257 | val=0.00027 | no-improve=0/5
[TransitionModel] Epoch  100 | train=-49.11585 | val=0.00021 | no-improve=0/5
[TransitionModel] Epoch  120 | train=-52.82123 | val=0.00017 | no-improve=1/5
[TransitionModel] Epoch  140 | train=-56.22911 | val=0.00014 | no-improve=0/5
[TransitionModel] Epoch  160 | train=-59.92276 | val=0.00014 | no-improve=0/5
[TransitionModel] Epoch  169 | train=-61.39241 | val=0.00013 | no-improve=5/5
[TransitionModel] Training complete. Elites=[5, 1, 3, 4, 0], val_loss=0.00013
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_00.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_00.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 0)
[IQL] Epoch 0/20, V Loss: 0.0089, Q Loss: 0.0591
[IQL] Epoch 1/20, V Loss: 0.0002, Q Loss: 0.0089
[IQL] Epoch 2/20, V Loss: 0.0001, Q Loss: 0.0088
[IQL] Epoch 3/20, V Loss: 0.0001, Q Loss: 0.0087
[IQL] Epoch 4/20, V Loss: 0.0002, Q Loss: 0.0085
[IQL] Epoch 5/20, V Loss: 0.0002, Q Loss: 0.0084
[IQL] Epoch 6/20, V Loss: 0.0002, Q Loss: 0.0085
[IQL] Epoch 7/20, V Loss: 0.0002, Q Loss: 0.0084
[IQL] Epoch 8/20, V Loss: 0.0001, Q Loss: 0.0082
[IQL] Epoch 9/20, V Loss: 0.0002, Q Loss: 0.0082
[IQL] Epoch 10/20, V Loss: 0.0003, Q Loss: 0.0081
[IQL] Epoch 11/20, V Loss: 0.0001, Q Loss: 0.0080
[IQL] Epoch 12/20, V Loss: 0.0002, Q Loss: 0.0080
[IQL] Epoch 13/20, V Loss: 0.0003, Q Loss: 0.0081
[IQL] Epoch 14/20, V Loss: 0.0002, Q Loss: 0.0078
[IQL] Epoch 15/20, V Loss: 0.0002, Q Loss: 0.0077
[IQL] Epoch 16/20, V Loss: 0.0003, Q Loss: 0.0078
[IQL] Epoch 17/20, V Loss: 0.0002, Q Loss: 0.0078
[IQL] Epoch 18/20, V Loss: 0.0002, Q Loss: 0.0075
[IQL] Epoch 19/20, V Loss: 0.0004, Q Loss: 0.0077
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_00.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_00.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 0)
[OPE] Behavior policy value J_old = 0.2797
[RL PPO] Reducing policy LR: 1.00e-04 → 2.00e-05
[Offline RL] Epoch 0/10, PPO Loss: 0.0000, PostKL: 1.370e+00, PostClipFrac: 0.769804, PostMeanRatio: 1.401446, PostRatioDev: 1.097e+00, GradNorm: 15.0788, Reg Loss: 0.0000, CD Loss: 0.3337
[Offline RL] Epoch 1/10, PPO Loss: 0.0000, PostKL: 1.134e+01, PostClipFrac: 0.827837, PostMeanRatio: 0.966676, PostRatioDev: 8.908e-01, GradNorm: 18.5137, Reg Loss: 0.0000, CD Loss: 0.3279
[Offline RL] Epoch 2/10, PPO Loss: -0.0000, PostKL: 2.722e+01, PostClipFrac: 0.946800, PostMeanRatio: 0.720077, PostRatioDev: 1.224e+00, GradNorm: 21.5167, Reg Loss: 0.0000, CD Loss: 0.3552
[Offline RL] Epoch 3/10, PPO Loss: -0.0000, PostKL: 2.017e+01, PostClipFrac: 0.967809, PostMeanRatio: 0.571874, PostRatioDev: 1.207e+00, GradNorm: 20.7299, Reg Loss: 0.0000, CD Loss: 0.5105
[Offline RL] Epoch 4/10, PPO Loss: 0.0000, PostKL: 1.427e+02, PostClipFrac: 0.950291, PostMeanRatio: 1.152372, PostRatioDev: 1.643e+00, GradNorm: 24.6605, Reg Loss: 0.0000, CD Loss: 0.5310
[Offline RL] Epoch 5/10, PPO Loss: 0.0000, PostKL: 9.807e+01, PostClipFrac: 0.931035, PostMeanRatio: 0.657878, PostRatioDev: 1.128e+00, GradNorm: 34.1631, Reg Loss: 0.0000, CD Loss: 0.5531
[Offline RL] Epoch 6/10, PPO Loss: -0.0000, PostKL: 5.218e+00, PostClipFrac: 0.852992, PostMeanRatio: 0.967401, PostRatioDev: 9.500e-01, GradNorm: 22.1384, Reg Loss: 0.0000, CD Loss: 0.6408
[Offline RL] Epoch 7/10, PPO Loss: 0.0000, PostKL: 8.212e+00, PostClipFrac: 0.777409, PostMeanRatio: 0.953627, PostRatioDev: 7.051e-01, GradNorm: 22.2594, Reg Loss: 0.0000, CD Loss: 0.5598
[Offline RL] Epoch 8/10, PPO Loss: -0.0000, PostKL: 8.193e+00, PostClipFrac: 0.758056, PostMeanRatio: 0.931886, PostRatioDev: 6.571e-01, GradNorm: 23.9665, Reg Loss: 0.0000, CD Loss: 0.5138
[Offline RL] Epoch 9/10, PPO Loss: -0.0000, PostKL: 5.280e+00, PostClipFrac: 0.713136, PostMeanRatio: 0.942454, PostRatioDev: 5.598e-01, GradNorm: 21.3501, Reg Loss: 0.0000, CD Loss: 0.4922
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_00.png
[OPE] Policy REJECTED: J_new=0.2437 ≤ J_old=0.2797 + δ=0.0140. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 0)
[Collect] 20 episodes, success=0.700, env_return=1173.32, rl_reward=0.70, steps=4000
[Data Collection] Success Rate: 0.700, EnvReturn: 1173.32, RLReward: 0.70, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: keeping 14/20 successful episodes (dropped 6 failures) before merge.
[Dataset] Merged 14 episodes (2800 steps) → total 22800 steps, 114 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.0187, Val Loss: 0.0101
[IL] Epoch 1/100, Loss: 0.0114, Val Loss: 0.0070
[IL] Epoch 2/100, Loss: 0.0108, Val Loss: 0.0102
[IL] Epoch 3/100, Loss: 0.0100, Val Loss: 0.0064
[IL] Epoch 4/100, Loss: 0.0099, Val Loss: 0.0074
[IL] Epoch 5/100, Loss: 0.0086, Val Loss: 0.0068
[IL] Epoch 6/100, Loss: 0.0088, Val Loss: 0.0097
[IL] Epoch 7/100, Loss: 0.0086, Val Loss: 0.0071
[IL] Epoch 8/100, Loss: 0.0086, Val Loss: 0.0062
[IL] Epoch 9/100, Loss: 0.0081, Val Loss: 0.0105
[IL] Epoch 10/100, Loss: 0.0081, Val Loss: 0.0063
[IL] Epoch 11/100, Loss: 0.0077, Val Loss: 0.0090
[IL] Epoch 12/100, Loss: 0.0085, Val Loss: 0.0070
[IL] Epoch 13/100, Loss: 0.0073, Val Loss: 0.0059
[IL] Epoch 14/100, Loss: 0.0079, Val Loss: 0.0068
[IL] Epoch 15/100, Loss: 0.0075, Val Loss: 0.0077
[IL] Epoch 16/100, Loss: 0.0074, Val Loss: 0.0061
[IL] Epoch 17/100, Loss: 0.0070, Val Loss: 0.0057
[IL] Epoch 18/100, Loss: 0.0068, Val Loss: 0.0059
[IL] Epoch 19/100, Loss: 0.0077, Val Loss: 0.0080
[IL] Epoch 20/100, Loss: 0.0070, Val Loss: 0.0072
[IL] Epoch 21/100, Loss: 0.0070, Val Loss: 0.0063
[IL] Epoch 22/100, Loss: 0.0073, Val Loss: 0.0083
[IL] Epoch 23/100, Loss: 0.0066, Val Loss: 0.0072
[IL] Epoch 24/100, Loss: 0.0067, Val Loss: 0.0069
[IL] Epoch 25/100, Loss: 0.0064, Val Loss: 0.0046
[IL] Epoch 26/100, Loss: 0.0068, Val Loss: 0.0063
[IL] Epoch 27/100, Loss: 0.0070, Val Loss: 0.0050
[IL] Epoch 28/100, Loss: 0.0064, Val Loss: 0.0069
[IL] Epoch 29/100, Loss: 0.0069, Val Loss: 0.0069
[IL] Epoch 30/100, Loss: 0.0070, Val Loss: 0.0075
[IL] Epoch 31/100, Loss: 0.0064, Val Loss: 0.0061
[IL] Epoch 32/100, Loss: 0.0058, Val Loss: 0.0068
[IL] Epoch 33/100, Loss: 0.0066, Val Loss: 0.0116
[IL] Epoch 34/100, Loss: 0.0063, Val Loss: 0.0080
[IL] Epoch 35/100, Loss: 0.0062, Val Loss: 0.0089
[IL] Epoch 36/100, Loss: 0.0063, Val Loss: 0.0061
[IL] Epoch 37/100, Loss: 0.0062, Val Loss: 0.0106
[IL] Epoch 38/100, Loss: 0.0065, Val Loss: 0.0081
[IL] Epoch 39/100, Loss: 0.0061, Val Loss: 0.0083
[IL] Epoch 40/100, Loss: 0.0056, Val Loss: 0.0067
[IL] Epoch 41/100, Loss: 0.0058, Val Loss: 0.0079
[IL] Epoch 42/100, Loss: 0.0063, Val Loss: 0.0074
[IL] Epoch 43/100, Loss: 0.0066, Val Loss: 0.0063
[IL] Epoch 44/100, Loss: 0.0064, Val Loss: 0.0072
[IL] Epoch 45/100, Loss: 0.0065, Val Loss: 0.0099
[IL] Epoch 46/100, Loss: 0.0060, Val Loss: 0.0066
[IL] Epoch 47/100, Loss: 0.0060, Val Loss: 0.0051
[IL] Epoch 48/100, Loss: 0.0060, Val Loss: 0.0083
[IL] Epoch 49/100, Loss: 0.0065, Val Loss: 0.0093
[IL] Epoch 50/100, Loss: 0.0065, Val Loss: 0.0088
[IL] Epoch 51/100, Loss: 0.0061, Val Loss: 0.0085
[IL] Epoch 52/100, Loss: 0.0057, Val Loss: 0.0062
[IL] Epoch 53/100, Loss: 0.0057, Val Loss: 0.0075
[IL] Epoch 54/100, Loss: 0.0061, Val Loss: 0.0062
[IL] Epoch 55/100, Loss: 0.0063, Val Loss: 0.0076
[IL] Epoch 56/100, Loss: 0.0057, Val Loss: 0.0080
[IL] Epoch 57/100, Loss: 0.0056, Val Loss: 0.0075
[IL] Epoch 58/100, Loss: 0.0057, Val Loss: 0.0070
[IL] Epoch 59/100, Loss: 0.0059, Val Loss: 0.0056
[IL] Epoch 60/100, Loss: 0.0055, Val Loss: 0.0074
[IL] Epoch 61/100, Loss: 0.0053, Val Loss: 0.0112
[IL] Epoch 62/100, Loss: 0.0057, Val Loss: 0.0073
[IL] Epoch 63/100, Loss: 0.0059, Val Loss: 0.0086
[IL] Epoch 64/100, Loss: 0.0060, Val Loss: 0.0064
[IL] Epoch 65/100, Loss: 0.0059, Val Loss: 0.0075
[IL] Epoch 66/100, Loss: 0.0057, Val Loss: 0.0055
[IL] Epoch 67/100, Loss: 0.0055, Val Loss: 0.0084
[IL] Epoch 68/100, Loss: 0.0053, Val Loss: 0.0072
[IL] Epoch 69/100, Loss: 0.0058, Val Loss: 0.0110
[IL] Epoch 70/100, Loss: 0.0055, Val Loss: 0.0078
[IL] Epoch 71/100, Loss: 0.0053, Val Loss: 0.0089
[IL] Epoch 72/100, Loss: 0.0052, Val Loss: 0.0054
[IL] Epoch 73/100, Loss: 0.0049, Val Loss: 0.0076
[IL] Epoch 74/100, Loss: 0.0055, Val Loss: 0.0093
[IL] Epoch 75/100, Loss: 0.0057, Val Loss: 0.0073
[IL] Epoch 76/100, Loss: 0.0059, Val Loss: 0.0076
[IL] Epoch 77/100, Loss: 0.0053, Val Loss: 0.0073
[IL] Epoch 78/100, Loss: 0.0058, Val Loss: 0.0060
[IL] Epoch 79/100, Loss: 0.0053, Val Loss: 0.0063
[IL] Epoch 80/100, Loss: 0.0052, Val Loss: 0.0078
[IL] Epoch 81/100, Loss: 0.0053, Val Loss: 0.0075
[IL] Epoch 82/100, Loss: 0.0051, Val Loss: 0.0057
[IL] Epoch 83/100, Loss: 0.0063, Val Loss: 0.0075
[IL] Epoch 84/100, Loss: 0.0050, Val Loss: 0.0087
[IL] Epoch 85/100, Loss: 0.0057, Val Loss: 0.0087
[IL] Epoch 86/100, Loss: 0.0052, Val Loss: 0.0119
[IL] Epoch 87/100, Loss: 0.0052, Val Loss: 0.0065
[IL] Epoch 88/100, Loss: 0.0050, Val Loss: 0.0052
[IL] Epoch 89/100, Loss: 0.0049, Val Loss: 0.0086
[IL] Epoch 90/100, Loss: 0.0050, Val Loss: 0.0065
[IL] Epoch 91/100, Loss: 0.0054, Val Loss: 0.0088
[IL] Epoch 92/100, Loss: 0.0053, Val Loss: 0.0055
[IL] Epoch 93/100, Loss: 0.0057, Val Loss: 0.0047
[IL] Epoch 94/100, Loss: 0.0056, Val Loss: 0.0074
[IL] Epoch 95/100, Loss: 0.0051, Val Loss: 0.0070
[IL] Epoch 96/100, Loss: 0.0049, Val Loss: 0.0074
[IL] Epoch 97/100, Loss: 0.0054, Val Loss: 0.0072
[IL] Epoch 98/100, Loss: 0.0052, Val Loss: 0.0080
[IL] Epoch 99/100, Loss: 0.0052, Val Loss: 0.0070
test_mean_score: 0.66
[IL] Eval - Success Rate: 0.660
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_00.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_0.ckpt

================================================================================
               OFFLINE RL ITERATION 2/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 1)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 20072 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-48.29878 | val=0.00037 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-63.25528 | val=0.00022 | no-improve=1/5
[TransitionModel] Epoch   40 | train=-67.96560 | val=0.00021 | no-improve=4/5
[TransitionModel] Epoch   60 | train=-72.33469 | val=0.00020 | no-improve=1/5
[TransitionModel] Epoch   80 | train=-76.47630 | val=0.00019 | no-improve=1/5
[TransitionModel] Epoch  100 | train=-80.43251 | val=0.00019 | no-improve=3/5
[TransitionModel] Epoch  110 | train=-82.55774 | val=0.00019 | no-improve=5/5
[TransitionModel] Training complete. Elites=[5, 6, 0, 1, 2], val_loss=0.00018
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_01.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_01.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 1)
[IQL] Epoch 0/20, V Loss: 0.0002, Q Loss: 0.0074
[IQL] Epoch 1/20, V Loss: 0.0001, Q Loss: 0.0073
[IQL] Epoch 2/20, V Loss: 0.0003, Q Loss: 0.0073
[IQL] Epoch 3/20, V Loss: 0.0001, Q Loss: 0.0072
[IQL] Epoch 4/20, V Loss: 0.0002, Q Loss: 0.0072
[IQL] Epoch 5/20, V Loss: 0.0002, Q Loss: 0.0072
[IQL] Epoch 6/20, V Loss: 0.0002, Q Loss: 0.0070
[IQL] Epoch 7/20, V Loss: 0.0003, Q Loss: 0.0072
[IQL] Epoch 8/20, V Loss: 0.0003, Q Loss: 0.0070
[IQL] Epoch 9/20, V Loss: 0.0001, Q Loss: 0.0068
[IQL] Epoch 10/20, V Loss: 0.0002, Q Loss: 0.0067
[IQL] Epoch 11/20, V Loss: 0.0003, Q Loss: 0.0069
[IQL] Epoch 12/20, V Loss: 0.0002, Q Loss: 0.0068
[IQL] Epoch 13/20, V Loss: 0.0001, Q Loss: 0.0065
[IQL] Epoch 14/20, V Loss: 0.0002, Q Loss: 0.0065
[IQL] Epoch 15/20, V Loss: 0.0002, Q Loss: 0.0065
[IQL] Epoch 16/20, V Loss: 0.0002, Q Loss: 0.0065
[IQL] Epoch 17/20, V Loss: 0.0003, Q Loss: 0.0065
[IQL] Epoch 18/20, V Loss: 0.0003, Q Loss: 0.0065
[IQL] Epoch 19/20, V Loss: 0.0002, Q Loss: 0.0063
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_01.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_01.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 1)
[OPE] Behavior policy value J_old = 0.4647
[RL PPO] Reducing policy LR: 1.00e-04 → 2.00e-05
[Offline RL] Epoch 0/10, PPO Loss: 0.0000, PostKL: 1.313e+00, PostClipFrac: 0.764587, PostMeanRatio: 1.124077, PostRatioDev: 8.022e-01, GradNorm: 15.2205, Reg Loss: 0.0000, CD Loss: 0.3565
[Offline RL] Epoch 1/10, PPO Loss: 0.0000, PostKL: 7.540e-01, PostClipFrac: 0.753314, PostMeanRatio: 0.998267, PostRatioDev: 6.416e-01, GradNorm: 15.1119, Reg Loss: 0.0000, CD Loss: 0.3371
[Offline RL] Epoch 2/10, PPO Loss: -0.0000, PostKL: 2.496e+00, PostClipFrac: 0.857355, PostMeanRatio: 0.902203, PostRatioDev: 8.943e-01, GradNorm: 16.3158, Reg Loss: 0.0000, CD Loss: 0.3710
[Offline RL] Epoch 3/10, PPO Loss: -0.0000, PostKL: 2.894e+00, PostClipFrac: 0.871692, PostMeanRatio: 0.891121, PostRatioDev: 9.360e-01, GradNorm: 16.1403, Reg Loss: 0.0000, CD Loss: 0.4599
[Offline RL] Epoch 4/10, PPO Loss: 0.0000, PostKL: 3.965e+00, PostClipFrac: 0.898505, PostMeanRatio: 0.871292, PostRatioDev: 1.008e+00, GradNorm: 18.5210, Reg Loss: 0.0000, CD Loss: 0.6329
[Offline RL] Epoch 5/10, PPO Loss: 0.0000, PostKL: 1.689e+00, PostClipFrac: 0.837712, PostMeanRatio: 0.991705, PostRatioDev: 8.679e-01, GradNorm: 16.5546, Reg Loss: 0.0000, CD Loss: 0.4766
[Offline RL] Epoch 6/10, PPO Loss: 0.0000, PostKL: 1.905e+00, PostClipFrac: 0.773948, PostMeanRatio: 0.984428, PostRatioDev: 7.144e-01, GradNorm: 17.9597, Reg Loss: 0.0000, CD Loss: 0.5153
[Offline RL] Epoch 7/10, PPO Loss: 0.0000, PostKL: 2.766e+00, PostClipFrac: 0.806461, PostMeanRatio: 0.925313, PostRatioDev: 7.454e-01, GradNorm: 19.2236, Reg Loss: 0.0000, CD Loss: 0.4997
[Offline RL] Epoch 8/10, PPO Loss: -0.0000, PostKL: 4.557e+00, PostClipFrac: 0.795062, PostMeanRatio: 0.933181, PostRatioDev: 7.507e-01, GradNorm: 21.0244, Reg Loss: 0.0000, CD Loss: 0.5355
[Offline RL] Epoch 9/10, PPO Loss: 0.0000, PostKL: 2.026e+01, PostClipFrac: 0.813600, PostMeanRatio: 0.850236, PostRatioDev: 7.932e-01, GradNorm: 23.9521, Reg Loss: 0.0000, CD Loss: 0.7566
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_01.png
[OPE] Policy REJECTED: J_new=0.4682 ≤ J_old=0.4647 + δ=0.0232. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 1)
[Collect] 20 episodes, success=0.650, env_return=919.96, rl_reward=0.65, steps=4000
[Data Collection] Success Rate: 0.650, EnvReturn: 919.96, RLReward: 0.65, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: keeping 13/20 successful episodes (dropped 7 failures) before merge.
[Dataset] Merged 13 episodes (2600 steps) → total 25400 steps, 127 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.0156, Val Loss: 0.0113
[IL] Epoch 1/100, Loss: 0.0092, Val Loss: 0.0088
[IL] Epoch 2/100, Loss: 0.0088, Val Loss: 0.0078
[IL] Epoch 3/100, Loss: 0.0087, Val Loss: 0.0066
[IL] Epoch 4/100, Loss: 0.0083, Val Loss: 0.0086
[IL] Epoch 5/100, Loss: 0.0080, Val Loss: 0.0070
[IL] Epoch 6/100, Loss: 0.0078, Val Loss: 0.0078
[IL] Epoch 7/100, Loss: 0.0072, Val Loss: 0.0069
[IL] Epoch 8/100, Loss: 0.0071, Val Loss: 0.0071
[IL] Epoch 9/100, Loss: 0.0071, Val Loss: 0.0066
[IL] Epoch 10/100, Loss: 0.0072, Val Loss: 0.0075
[IL] Epoch 11/100, Loss: 0.0071, Val Loss: 0.0050
[IL] Epoch 12/100, Loss: 0.0075, Val Loss: 0.0076
[IL] Epoch 13/100, Loss: 0.0071, Val Loss: 0.0080
[IL] Epoch 14/100, Loss: 0.0069, Val Loss: 0.0078
[IL] Epoch 15/100, Loss: 0.0070, Val Loss: 0.0070
[IL] Epoch 16/100, Loss: 0.0062, Val Loss: 0.0080
[IL] Epoch 17/100, Loss: 0.0065, Val Loss: 0.0066
[IL] Epoch 18/100, Loss: 0.0069, Val Loss: 0.0086
[IL] Epoch 19/100, Loss: 0.0068, Val Loss: 0.0077
[IL] Epoch 20/100, Loss: 0.0070, Val Loss: 0.0089
[IL] Epoch 21/100, Loss: 0.0064, Val Loss: 0.0074
[IL] Epoch 22/100, Loss: 0.0068, Val Loss: 0.0061
[IL] Epoch 23/100, Loss: 0.0066, Val Loss: 0.0080
[IL] Epoch 24/100, Loss: 0.0064, Val Loss: 0.0048
[IL] Epoch 25/100, Loss: 0.0062, Val Loss: 0.0085
[IL] Epoch 26/100, Loss: 0.0065, Val Loss: 0.0055
[IL] Epoch 27/100, Loss: 0.0063, Val Loss: 0.0057
[IL] Epoch 28/100, Loss: 0.0061, Val Loss: 0.0082
[IL] Epoch 29/100, Loss: 0.0067, Val Loss: 0.0053
[IL] Epoch 30/100, Loss: 0.0064, Val Loss: 0.0064
[IL] Epoch 31/100, Loss: 0.0063, Val Loss: 0.0100
[IL] Epoch 32/100, Loss: 0.0062, Val Loss: 0.0073
[IL] Epoch 33/100, Loss: 0.0061, Val Loss: 0.0074
[IL] Epoch 34/100, Loss: 0.0065, Val Loss: 0.0091
[IL] Epoch 35/100, Loss: 0.0063, Val Loss: 0.0070
[IL] Epoch 36/100, Loss: 0.0058, Val Loss: 0.0067
[IL] Epoch 37/100, Loss: 0.0062, Val Loss: 0.0103
[IL] Epoch 38/100, Loss: 0.0065, Val Loss: 0.0073
[IL] Epoch 39/100, Loss: 0.0064, Val Loss: 0.0077
[IL] Epoch 40/100, Loss: 0.0065, Val Loss: 0.0084
[IL] Epoch 41/100, Loss: 0.0061, Val Loss: 0.0079
[IL] Epoch 42/100, Loss: 0.0060, Val Loss: 0.0084
[IL] Epoch 43/100, Loss: 0.0063, Val Loss: 0.0116
[IL] Epoch 44/100, Loss: 0.0069, Val Loss: 0.0074
[IL] Epoch 45/100, Loss: 0.0062, Val Loss: 0.0084
[IL] Epoch 46/100, Loss: 0.0059, Val Loss: 0.0104
[IL] Epoch 47/100, Loss: 0.0059, Val Loss: 0.0081
[IL] Epoch 48/100, Loss: 0.0057, Val Loss: 0.0094
[IL] Epoch 49/100, Loss: 0.0058, Val Loss: 0.0075
[IL] Epoch 50/100, Loss: 0.0061, Val Loss: 0.0078
[IL] Epoch 51/100, Loss: 0.0067, Val Loss: 0.0091
[IL] Epoch 52/100, Loss: 0.0060, Val Loss: 0.0061
[IL] Epoch 53/100, Loss: 0.0056, Val Loss: 0.0072
[IL] Epoch 54/100, Loss: 0.0056, Val Loss: 0.0072
[IL] Epoch 55/100, Loss: 0.0060, Val Loss: 0.0084
[IL] Epoch 56/100, Loss: 0.0055, Val Loss: 0.0102
[IL] Epoch 57/100, Loss: 0.0055, Val Loss: 0.0095
[IL] Epoch 58/100, Loss: 0.0059, Val Loss: 0.0106
[IL] Epoch 59/100, Loss: 0.0054, Val Loss: 0.0083
[IL] Epoch 60/100, Loss: 0.0053, Val Loss: 0.0068
[IL] Epoch 61/100, Loss: 0.0056, Val Loss: 0.0056
[IL] Epoch 62/100, Loss: 0.0059, Val Loss: 0.0067
[IL] Epoch 63/100, Loss: 0.0063, Val Loss: 0.0078
[IL] Epoch 64/100, Loss: 0.0058, Val Loss: 0.0088
[IL] Epoch 65/100, Loss: 0.0055, Val Loss: 0.0091
[IL] Epoch 66/100, Loss: 0.0055, Val Loss: 0.0064
[IL] Epoch 67/100, Loss: 0.0056, Val Loss: 0.0058
[IL] Epoch 68/100, Loss: 0.0063, Val Loss: 0.0063
[IL] Epoch 69/100, Loss: 0.0054, Val Loss: 0.0046
[IL] Epoch 70/100, Loss: 0.0053, Val Loss: 0.0059
[IL] Epoch 71/100, Loss: 0.0057, Val Loss: 0.0079
[IL] Epoch 72/100, Loss: 0.0054, Val Loss: 0.0081
[IL] Epoch 73/100, Loss: 0.0057, Val Loss: 0.0081
[IL] Epoch 74/100, Loss: 0.0056, Val Loss: 0.0063
[IL] Epoch 75/100, Loss: 0.0052, Val Loss: 0.0108
[IL] Epoch 76/100, Loss: 0.0052, Val Loss: 0.0064
[IL] Epoch 77/100, Loss: 0.0052, Val Loss: 0.0057
[IL] Epoch 78/100, Loss: 0.0055, Val Loss: 0.0076
[IL] Epoch 79/100, Loss: 0.0057, Val Loss: 0.0065
[IL] Epoch 80/100, Loss: 0.0048, Val Loss: 0.0068
[IL] Epoch 81/100, Loss: 0.0059, Val Loss: 0.0070
[IL] Epoch 82/100, Loss: 0.0054, Val Loss: 0.0089
[IL] Epoch 83/100, Loss: 0.0055, Val Loss: 0.0094
[IL] Epoch 84/100, Loss: 0.0052, Val Loss: 0.0097
[IL] Epoch 85/100, Loss: 0.0055, Val Loss: 0.0104
[IL] Epoch 86/100, Loss: 0.0055, Val Loss: 0.0083
[IL] Epoch 87/100, Loss: 0.0052, Val Loss: 0.0108
[IL] Epoch 88/100, Loss: 0.0049, Val Loss: 0.0136
[IL] Epoch 89/100, Loss: 0.0063, Val Loss: 0.0088
[IL] Epoch 90/100, Loss: 0.0064, Val Loss: 0.0077
[IL] Epoch 91/100, Loss: 0.0052, Val Loss: 0.0090
[IL] Epoch 92/100, Loss: 0.0059, Val Loss: 0.0085
[IL] Epoch 93/100, Loss: 0.0056, Val Loss: 0.0063
[IL] Epoch 94/100, Loss: 0.0050, Val Loss: 0.0069
[IL] Epoch 95/100, Loss: 0.0047, Val Loss: 0.0102
[IL] Epoch 96/100, Loss: 0.0050, Val Loss: 0.0085
[IL] Epoch 97/100, Loss: 0.0048, Val Loss: 0.0083
[IL] Epoch 98/100, Loss: 0.0049, Val Loss: 0.0057
[IL] Epoch 99/100, Loss: 0.0050, Val Loss: 0.0075
test_mean_score: 0.73
[IL] Eval - Success Rate: 0.730
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_01.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_1.ckpt

================================================================================
               OFFLINE RL ITERATION 3/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 2)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 22581 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-64.96760 | val=0.00033 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-85.16837 | val=0.00021 | no-improve=1/5
[TransitionModel] Epoch   40 | train=-89.94386 | val=0.00020 | no-improve=1/5
[TransitionModel] Epoch   44 | train=-90.82616 | val=0.00020 | no-improve=5/5
[TransitionModel] Training complete. Elites=[6, 0, 3, 1, 2], val_loss=0.00019
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_02.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_02.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 2)
[IQL] Epoch 0/20, V Loss: 0.0003, Q Loss: 0.0066
[IQL] Epoch 1/20, V Loss: 0.0003, Q Loss: 0.0064
[IQL] Epoch 2/20, V Loss: 0.0002, Q Loss: 0.0063
[IQL] Epoch 3/20, V Loss: 0.0003, Q Loss: 0.0062
[IQL] Epoch 4/20, V Loss: 0.0002, Q Loss: 0.0063
[IQL] Epoch 5/20, V Loss: 0.0002, Q Loss: 0.0061
[IQL] Epoch 6/20, V Loss: 0.0003, Q Loss: 0.0062
[IQL] Epoch 7/20, V Loss: 0.0002, Q Loss: 0.0061
[IQL] Epoch 8/20, V Loss: 0.0004, Q Loss: 0.0061
[IQL] Epoch 9/20, V Loss: 0.0002, Q Loss: 0.0059
[IQL] Epoch 10/20, V Loss: 0.0003, Q Loss: 0.0059
[IQL] Epoch 11/20, V Loss: 0.0002, Q Loss: 0.0059
[IQL] Epoch 12/20, V Loss: 0.0003, Q Loss: 0.0058
[IQL] Epoch 13/20, V Loss: 0.0003, Q Loss: 0.0058
[IQL] Epoch 14/20, V Loss: 0.0002, Q Loss: 0.0060
[IQL] Epoch 15/20, V Loss: 0.0003, Q Loss: 0.0060
[IQL] Epoch 16/20, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 17/20, V Loss: 0.0002, Q Loss: 0.0060
[IQL] Epoch 18/20, V Loss: 0.0002, Q Loss: 0.0058
[IQL] Epoch 19/20, V Loss: 0.0002, Q Loss: 0.0057
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_02.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_02.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 2)
[OPE] Behavior policy value J_old = 0.5350
[RL PPO] Reducing policy LR: 1.00e-04 → 2.00e-05
[Offline RL] Epoch 0/10, PPO Loss: -0.0000, PostKL: 1.276e+03, PostClipFrac: 0.783362, PostMeanRatio: 1272.886949, PostRatioDev: 1.273e+03, GradNorm: 17.8321, Reg Loss: 0.0000, CD Loss: 0.3289
[Offline RL] Epoch 1/10, PPO Loss: 0.0000, PostKL: 2.311e+01, PostClipFrac: 0.864727, PostMeanRatio: 0.807003, PostRatioDev: 9.764e-01, GradNorm: 20.9924, Reg Loss: 0.0000, CD Loss: 0.3656
[Offline RL] Epoch 2/10, PPO Loss: 0.0000, PostKL: 2.293e+00, PostClipFrac: 0.725868, PostMeanRatio: 0.944579, PostRatioDev: 6.872e-01, GradNorm: 16.0783, Reg Loss: 0.0000, CD Loss: 0.4335
[Offline RL] Epoch 3/10, PPO Loss: -0.0000, PostKL: 9.935e-01, PostClipFrac: 0.617469, PostMeanRatio: 0.979597, PostRatioDev: 5.092e-01, GradNorm: 15.5112, Reg Loss: 0.0000, CD Loss: 0.5344
[Offline RL] Epoch 4/10, PPO Loss: 0.0000, PostKL: 1.151e+00, PostClipFrac: 0.668452, PostMeanRatio: 0.994035, PostRatioDev: 5.556e-01, GradNorm: 16.4278, Reg Loss: 0.0000, CD Loss: 0.5191
[Offline RL] Epoch 5/10, PPO Loss: 0.0000, PostKL: 1.965e+00, PostClipFrac: 0.626129, PostMeanRatio: 0.981078, PostRatioDev: 5.191e-01, GradNorm: 16.8914, Reg Loss: 0.0000, CD Loss: 0.5856
[Offline RL] Epoch 6/10, PPO Loss: -0.0000, PostKL: 3.732e-01, PostClipFrac: 0.363477, PostMeanRatio: 0.995007, PostRatioDev: 2.405e-01, GradNorm: 15.1905, Reg Loss: 0.0000, CD Loss: 0.8062
[Offline RL] Epoch 7/10, PPO Loss: -0.0000, PostKL: 1.362e-01, PostClipFrac: 0.310495, PostMeanRatio: 1.001088, PostRatioDev: 2.024e-01, GradNorm: 14.5875, Reg Loss: 0.0000, CD Loss: 0.8057
[Offline RL] Epoch 8/10, PPO Loss: -0.0000, PostKL: 1.147e-01, PostClipFrac: 0.376795, PostMeanRatio: 0.999353, PostRatioDev: 2.392e-01, GradNorm: 13.9705, Reg Loss: 0.0000, CD Loss: 0.8674
[Offline RL] Epoch 9/10, PPO Loss: -0.0000, PostKL: 8.503e-01, PostClipFrac: 0.565540, PostMeanRatio: 0.996877, PostRatioDev: 4.132e-01, GradNorm: 14.9702, Reg Loss: 0.0000, CD Loss: 0.7546
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_02.png
[OPE] Policy ACCEPTED: J_new=1.0742 > J_old=0.5350 + δ=0.0267

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 2)
[Collect] 20 episodes, success=0.000, env_return=103.86, rl_reward=0.00, steps=4000
[Data Collection] Success Rate: 0.000, EnvReturn: 103.86, RLReward: 0.00, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: keeping 0/20 successful episodes (dropped 20 failures) before merge.

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.3660, Val Loss: 0.0392
[IL] Epoch 1/100, Loss: 0.0309, Val Loss: 0.0174
[IL] Epoch 2/100, Loss: 0.0187, Val Loss: 0.0154
[IL] Epoch 3/100, Loss: 0.0134, Val Loss: 0.0131
[IL] Epoch 4/100, Loss: 0.0104, Val Loss: 0.0108
[IL] Epoch 5/100, Loss: 0.0089, Val Loss: 0.0077
[IL] Epoch 6/100, Loss: 0.0078, Val Loss: 0.0072
[IL] Epoch 7/100, Loss: 0.0070, Val Loss: 0.0086
[IL] Epoch 8/100, Loss: 0.0063, Val Loss: 0.0059
[IL] Epoch 9/100, Loss: 0.0061, Val Loss: 0.0058
[IL] Epoch 10/100, Loss: 0.0054, Val Loss: 0.0063
[IL] Epoch 11/100, Loss: 0.0051, Val Loss: 0.0065
[IL] Epoch 12/100, Loss: 0.0050, Val Loss: 0.0078
[IL] Epoch 13/100, Loss: 0.0049, Val Loss: 0.0085
[IL] Epoch 14/100, Loss: 0.0044, Val Loss: 0.0070
[IL] Epoch 15/100, Loss: 0.0046, Val Loss: 0.0079
[IL] Epoch 16/100, Loss: 0.0043, Val Loss: 0.0106
[IL] Epoch 17/100, Loss: 0.0041, Val Loss: 0.0072
[IL] Epoch 18/100, Loss: 0.0042, Val Loss: 0.0078
[IL] Epoch 19/100, Loss: 0.0041, Val Loss: 0.0060
[IL] Epoch 20/100, Loss: 0.0040, Val Loss: 0.0066
[IL] Epoch 21/100, Loss: 0.0036, Val Loss: 0.0080
[IL] Epoch 22/100, Loss: 0.0039, Val Loss: 0.0048
[IL] Epoch 23/100, Loss: 0.0037, Val Loss: 0.0078
[IL] Epoch 24/100, Loss: 0.0035, Val Loss: 0.0070
[IL] Epoch 25/100, Loss: 0.0037, Val Loss: 0.0080
[IL] Epoch 26/100, Loss: 0.0037, Val Loss: 0.0079
[IL] Epoch 27/100, Loss: 0.0036, Val Loss: 0.0089
[IL] Epoch 28/100, Loss: 0.0033, Val Loss: 0.0074
[IL] Epoch 29/100, Loss: 0.0034, Val Loss: 0.0063
[IL] Epoch 30/100, Loss: 0.0033, Val Loss: 0.0067
[IL] Epoch 31/100, Loss: 0.0034, Val Loss: 0.0088
[IL] Epoch 32/100, Loss: 0.0034, Val Loss: 0.0077
[IL] Epoch 33/100, Loss: 0.0032, Val Loss: 0.0099
[IL] Epoch 34/100, Loss: 0.0031, Val Loss: 0.0090
[IL] Epoch 35/100, Loss: 0.0031, Val Loss: 0.0081
[IL] Epoch 36/100, Loss: 0.0033, Val Loss: 0.0053
[IL] Epoch 37/100, Loss: 0.0032, Val Loss: 0.0095
[IL] Epoch 38/100, Loss: 0.0032, Val Loss: 0.0073
[IL] Epoch 39/100, Loss: 0.0031, Val Loss: 0.0110
[IL] Epoch 40/100, Loss: 0.0030, Val Loss: 0.0111
[IL] Epoch 41/100, Loss: 0.0030, Val Loss: 0.0091
[IL] Epoch 42/100, Loss: 0.0033, Val Loss: 0.0078
[IL] Epoch 43/100, Loss: 0.0030, Val Loss: 0.0087
[IL] Epoch 44/100, Loss: 0.0029, Val Loss: 0.0069
[IL] Epoch 45/100, Loss: 0.0029, Val Loss: 0.0113
[IL] Epoch 46/100, Loss: 0.0031, Val Loss: 0.0078
[IL] Epoch 47/100, Loss: 0.0030, Val Loss: 0.0077
[IL] Epoch 48/100, Loss: 0.0029, Val Loss: 0.0063
[IL] Epoch 49/100, Loss: 0.0029, Val Loss: 0.0068
[IL] Epoch 50/100, Loss: 0.0029, Val Loss: 0.0071
[IL] Epoch 51/100, Loss: 0.0030, Val Loss: 0.0060
[IL] Epoch 52/100, Loss: 0.0029, Val Loss: 0.0120
[IL] Epoch 53/100, Loss: 0.0028, Val Loss: 0.0156
[IL] Epoch 54/100, Loss: 0.0028, Val Loss: 0.0130
[IL] Epoch 55/100, Loss: 0.0029, Val Loss: 0.0074
[IL] Epoch 56/100, Loss: 0.0029, Val Loss: 0.0130
[IL] Epoch 57/100, Loss: 0.0028, Val Loss: 0.0083
[IL] Epoch 58/100, Loss: 0.0028, Val Loss: 0.0084
[IL] Epoch 59/100, Loss: 0.0028, Val Loss: 0.0093
[IL] Epoch 60/100, Loss: 0.0029, Val Loss: 0.0101
[IL] Epoch 61/100, Loss: 0.0027, Val Loss: 0.0129
[IL] Epoch 62/100, Loss: 0.0027, Val Loss: 0.0086
[IL] Epoch 63/100, Loss: 0.0028, Val Loss: 0.0141
[IL] Epoch 64/100, Loss: 0.0025, Val Loss: 0.0106
[IL] Epoch 65/100, Loss: 0.0026, Val Loss: 0.0095
[IL] Epoch 66/100, Loss: 0.0027, Val Loss: 0.0130
[IL] Epoch 67/100, Loss: 0.0027, Val Loss: 0.0074
[IL] Epoch 68/100, Loss: 0.0027, Val Loss: 0.0119
[IL] Epoch 69/100, Loss: 0.0027, Val Loss: 0.0083
[IL] Epoch 70/100, Loss: 0.0027, Val Loss: 0.0077
[IL] Epoch 71/100, Loss: 0.0026, Val Loss: 0.0145
[IL] Epoch 72/100, Loss: 0.0025, Val Loss: 0.0099
[IL] Epoch 73/100, Loss: 0.0026, Val Loss: 0.0067
[IL] Epoch 74/100, Loss: 0.0027, Val Loss: 0.0109
[IL] Epoch 75/100, Loss: 0.0026, Val Loss: 0.0079
[IL] Epoch 76/100, Loss: 0.0026, Val Loss: 0.0085
[IL] Epoch 77/100, Loss: 0.0026, Val Loss: 0.0110
[IL] Epoch 78/100, Loss: 0.0027, Val Loss: 0.0088
[IL] Epoch 79/100, Loss: 0.0027, Val Loss: 0.0107
[IL] Epoch 80/100, Loss: 0.0024, Val Loss: 0.0092
[IL] Epoch 81/100, Loss: 0.0028, Val Loss: 0.0073
[IL] Epoch 82/100, Loss: 0.0024, Val Loss: 0.0133
[IL] Epoch 83/100, Loss: 0.0026, Val Loss: 0.0084
[IL] Epoch 84/100, Loss: 0.0023, Val Loss: 0.0118
[IL] Epoch 85/100, Loss: 0.0025, Val Loss: 0.0106
[IL] Epoch 86/100, Loss: 0.0026, Val Loss: 0.0089
[IL] Epoch 87/100, Loss: 0.0024, Val Loss: 0.0072
[IL] Epoch 88/100, Loss: 0.0024, Val Loss: 0.0111
[IL] Epoch 89/100, Loss: 0.0024, Val Loss: 0.0120
[IL] Epoch 90/100, Loss: 0.0025, Val Loss: 0.0129
[IL] Epoch 91/100, Loss: 0.0024, Val Loss: 0.0162
[IL] Epoch 92/100, Loss: 0.0025, Val Loss: 0.0104
[IL] Epoch 93/100, Loss: 0.0024, Val Loss: 0.0103
[IL] Epoch 94/100, Loss: 0.0025, Val Loss: 0.0113
[IL] Epoch 95/100, Loss: 0.0026, Val Loss: 0.0105
[IL] Epoch 96/100, Loss: 0.0025, Val Loss: 0.0091
[IL] Epoch 97/100, Loss: 0.0024, Val Loss: 0.0098
[IL] Epoch 98/100, Loss: 0.0025, Val Loss: 0.0079
[IL] Epoch 99/100, Loss: 0.0023, Val Loss: 0.0105
test_mean_score: 0.74
[IL] Eval - Success Rate: 0.740
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_02.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_2.ckpt

================================================================================
               OFFLINE RL ITERATION 4/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 3)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 22581 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-88.34069 | val=0.00025 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-95.42639 | val=0.00022 | no-improve=5/5
[TransitionModel] Training complete. Elites=[4, 2, 3, 1, 6], val_loss=0.00018
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_03.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_03.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 3)
[IQL] Epoch 0/20, V Loss: 0.0003, Q Loss: 0.0058
[IQL] Epoch 1/20, V Loss: 0.0002, Q Loss: 0.0057
[IQL] Epoch 2/20, V Loss: 0.0002, Q Loss: 0.0059
[IQL] Epoch 3/20, V Loss: 0.0002, Q Loss: 0.0059
[IQL] Epoch 4/20, V Loss: 0.0002, Q Loss: 0.0058
[IQL] Epoch 5/20, V Loss: 0.0002, Q Loss: 0.0058
[IQL] Epoch 6/20, V Loss: 0.0002, Q Loss: 0.0062
[IQL] Epoch 7/20, V Loss: 0.0002, Q Loss: 0.0059
[IQL] Epoch 8/20, V Loss: 0.0002, Q Loss: 0.0057
[IQL] Epoch 9/20, V Loss: 0.0002, Q Loss: 0.0059
[IQL] Epoch 10/20, V Loss: 0.0002, Q Loss: 0.0058
[IQL] Epoch 11/20, V Loss: 0.0002, Q Loss: 0.0059
[IQL] Epoch 12/20, V Loss: 0.0003, Q Loss: 0.0059
[IQL] Epoch 13/20, V Loss: 0.0002, Q Loss: 0.0057
[IQL] Epoch 14/20, V Loss: 0.0003, Q Loss: 0.0055
[IQL] Epoch 15/20, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 16/20, V Loss: 0.0003, Q Loss: 0.0061
[IQL] Epoch 17/20, V Loss: 0.0003, Q Loss: 0.0057
[IQL] Epoch 18/20, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 19/20, V Loss: 0.0002, Q Loss: 0.0054
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_03.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_03.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 3)
[OPE] Behavior policy value J_old = 0.6256
[Offline RL] Epoch 0/10, PPO Loss: -0.0000, PostKL: 1.375e+01, PostClipFrac: 0.860205, PostMeanRatio: 4.919765, PostRatioDev: 4.948e+00, GradNorm: 17.0579, Reg Loss: 0.0000, CD Loss: 0.3566
[Offline RL] Epoch 1/10, PPO Loss: 0.0000, PostKL: 8.699e-01, PostClipFrac: 0.703502, PostMeanRatio: 0.978684, PostRatioDev: 5.700e-01, GradNorm: 14.0019, Reg Loss: 0.0000, CD Loss: 0.4475
[Offline RL] Epoch 2/10, PPO Loss: 0.0000, PostKL: 3.203e+00, PostClipFrac: 0.767352, PostMeanRatio: 0.919437, PostRatioDev: 7.645e-01, GradNorm: 15.7766, Reg Loss: 0.0000, CD Loss: 0.3739
[Offline RL] Epoch 3/10, PPO Loss: 0.0000, PostKL: 8.583e-01, PostClipFrac: 0.587700, PostMeanRatio: 0.979514, PostRatioDev: 4.674e-01, GradNorm: 15.2131, Reg Loss: 0.0000, CD Loss: 0.3824
[Offline RL] Epoch 4/10, PPO Loss: -0.0000, PostKL: 7.151e-01, PostClipFrac: 0.460806, PostMeanRatio: 0.978736, PostRatioDev: 3.393e-01, GradNorm: 14.2064, Reg Loss: 0.0000, CD Loss: 0.5146
[Offline RL] Epoch 5/10, PPO Loss: 0.0000, PostKL: 4.130e-01, PostClipFrac: 0.363766, PostMeanRatio: 0.993484, PostRatioDev: 2.632e-01, GradNorm: 14.5892, Reg Loss: 0.0000, CD Loss: 0.5956
[Offline RL] Epoch 6/10, PPO Loss: 0.0000, PostKL: 4.693e-01, PostClipFrac: 0.325814, PostMeanRatio: 0.990627, PostRatioDev: 2.377e-01, GradNorm: 14.6390, Reg Loss: 0.0000, CD Loss: 0.6098
[Offline RL] Epoch 7/10, PPO Loss: 0.0000, PostKL: 2.600e-01, PostClipFrac: 0.243736, PostMeanRatio: 0.994021, PostRatioDev: 1.748e-01, GradNorm: 14.4120, Reg Loss: 0.0000, CD Loss: 0.5715
[Offline RL] Epoch 8/10, PPO Loss: -0.0000, PostKL: 3.066e+00, PostClipFrac: 0.384166, PostMeanRatio: 1.000677, PostRatioDev: 3.570e-01, GradNorm: 18.5905, Reg Loss: 0.0000, CD Loss: 0.5639
[Offline RL] Epoch 9/10, PPO Loss: 0.0000, PostKL: 4.052e+00, PostClipFrac: 0.471744, PostMeanRatio: 0.933615, PostRatioDev: 4.118e-01, GradNorm: 20.2740, Reg Loss: 0.0000, CD Loss: 0.5568
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_03.png
[OPE] Policy ACCEPTED: J_new=0.7502 > J_old=0.6256 + δ=0.0313

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 3)
[Collect] 20 episodes, success=0.000, env_return=90.82, rl_reward=0.00, steps=4000
[Data Collection] Success Rate: 0.000, EnvReturn: 90.82, RLReward: 0.00, Episodes: 20, Steps: 4000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: keeping 0/20 successful episodes (dropped 20 failures) before merge.

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.2325, Val Loss: 0.0205
[IL] Epoch 1/100, Loss: 0.0146, Val Loss: 0.0110
[IL] Epoch 2/100, Loss: 0.0093, Val Loss: 0.0100
[IL] Epoch 3/100, Loss: 0.0071, Val Loss: 0.0076
[IL] Epoch 4/100, Loss: 0.0062, Val Loss: 0.0073
[IL] Epoch 5/100, Loss: 0.0053, Val Loss: 0.0081
[IL] Epoch 6/100, Loss: 0.0047, Val Loss: 0.0065
[IL] Epoch 7/100, Loss: 0.0046, Val Loss: 0.0066
[IL] Epoch 8/100, Loss: 0.0042, Val Loss: 0.0077
[IL] Epoch 9/100, Loss: 0.0039, Val Loss: 0.0074
[IL] Epoch 10/100, Loss: 0.0036, Val Loss: 0.0091
[IL] Epoch 11/100, Loss: 0.0038, Val Loss: 0.0057
[IL] Epoch 12/100, Loss: 0.0034, Val Loss: 0.0097
[IL] Epoch 13/100, Loss: 0.0033, Val Loss: 0.0089
[IL] Epoch 14/100, Loss: 0.0032, Val Loss: 0.0061
[IL] Epoch 15/100, Loss: 0.0031, Val Loss: 0.0081
[IL] Epoch 16/100, Loss: 0.0031, Val Loss: 0.0067
[IL] Epoch 17/100, Loss: 0.0028, Val Loss: 0.0056
[IL] Epoch 18/100, Loss: 0.0032, Val Loss: 0.0074
[IL] Epoch 19/100, Loss: 0.0030, Val Loss: 0.0074
[IL] Epoch 20/100, Loss: 0.0029, Val Loss: 0.0073
[IL] Epoch 21/100, Loss: 0.0027, Val Loss: 0.0101
[IL] Epoch 22/100, Loss: 0.0026, Val Loss: 0.0073
[IL] Epoch 23/100, Loss: 0.0027, Val Loss: 0.0065
[IL] Epoch 24/100, Loss: 0.0026, Val Loss: 0.0098
[IL] Epoch 25/100, Loss: 0.0026, Val Loss: 0.0091
[IL] Epoch 26/100, Loss: 0.0027, Val Loss: 0.0105
[IL] Epoch 27/100, Loss: 0.0026, Val Loss: 0.0127
[IL] Epoch 28/100, Loss: 0.0025, Val Loss: 0.0100
[IL] Epoch 29/100, Loss: 0.0024, Val Loss: 0.0122
[IL] Epoch 30/100, Loss: 0.0026, Val Loss: 0.0077
[IL] Epoch 31/100, Loss: 0.0025, Val Loss: 0.0100
[IL] Epoch 32/100, Loss: 0.0026, Val Loss: 0.0112
[IL] Epoch 33/100, Loss: 0.0025, Val Loss: 0.0116
[IL] Epoch 34/100, Loss: 0.0026, Val Loss: 0.0108
[IL] Epoch 35/100, Loss: 0.0023, Val Loss: 0.0074
[IL] Epoch 36/100, Loss: 0.0025, Val Loss: 0.0111
[IL] Epoch 37/100, Loss: 0.0023, Val Loss: 0.0074
[IL] Epoch 38/100, Loss: 0.0024, Val Loss: 0.0130
[IL] Epoch 39/100, Loss: 0.0023, Val Loss: 0.0100
[IL] Epoch 40/100, Loss: 0.0024, Val Loss: 0.0089
[IL] Epoch 41/100, Loss: 0.0022, Val Loss: 0.0135
[IL] Epoch 42/100, Loss: 0.0023, Val Loss: 0.0141
[IL] Epoch 43/100, Loss: 0.0021, Val Loss: 0.0102
[IL] Epoch 44/100, Loss: 0.0023, Val Loss: 0.0098
[IL] Epoch 45/100, Loss: 0.0022, Val Loss: 0.0086
[IL] Epoch 46/100, Loss: 0.0023, Val Loss: 0.0106
[IL] Epoch 47/100, Loss: 0.0024, Val Loss: 0.0099
[IL] Epoch 48/100, Loss: 0.0024, Val Loss: 0.0094
[IL] Epoch 49/100, Loss: 0.0024, Val Loss: 0.0093
[IL] Epoch 50/100, Loss: 0.0023, Val Loss: 0.0092
[IL] Epoch 51/100, Loss: 0.0024, Val Loss: 0.0126
[IL] Epoch 52/100, Loss: 0.0023, Val Loss: 0.0131
[IL] Epoch 53/100, Loss: 0.0023, Val Loss: 0.0055
[IL] Epoch 54/100, Loss: 0.0023, Val Loss: 0.0126
[IL] Epoch 55/100, Loss: 0.0021, Val Loss: 0.0058
[IL] Epoch 56/100, Loss: 0.0022, Val Loss: 0.0060
[IL] Epoch 57/100, Loss: 0.0021, Val Loss: 0.0103
[IL] Epoch 58/100, Loss: 0.0024, Val Loss: 0.0113
[IL] Epoch 59/100, Loss: 0.0022, Val Loss: 0.0083
[IL] Epoch 60/100, Loss: 0.0020, Val Loss: 0.0089
[IL] Epoch 61/100, Loss: 0.0021, Val Loss: 0.0084
[IL] Epoch 62/100, Loss: 0.0023, Val Loss: 0.0108
[IL] Epoch 63/100, Loss: 0.0022, Val Loss: 0.0119
[IL] Epoch 64/100, Loss: 0.0021, Val Loss: 0.0149
[IL] Epoch 65/100, Loss: 0.0021, Val Loss: 0.0128
[IL] Epoch 66/100, Loss: 0.0022, Val Loss: 0.0137
[IL] Epoch 67/100, Loss: 0.0021, Val Loss: 0.0083
[IL] Epoch 68/100, Loss: 0.0020, Val Loss: 0.0103
[IL] Epoch 69/100, Loss: 0.0022, Val Loss: 0.0163
[IL] Epoch 70/100, Loss: 0.0021, Val Loss: 0.0126
[IL] Epoch 71/100, Loss: 0.0022, Val Loss: 0.0091
[IL] Epoch 72/100, Loss: 0.0025, Val Loss: 0.0105
[IL] Epoch 73/100, Loss: 0.0022, Val Loss: 0.0098
[IL] Epoch 74/100, Loss: 0.0022, Val Loss: 0.0108
[IL] Epoch 75/100, Loss: 0.0023, Val Loss: 0.0160
[IL] Epoch 76/100, Loss: 0.0021, Val Loss: 0.0176
[IL] Epoch 77/100, Loss: 0.0022, Val Loss: 0.0092
[IL] Epoch 78/100, Loss: 0.0021, Val Loss: 0.0153
[IL] Epoch 79/100, Loss: 0.0021, Val Loss: 0.0085
[IL] Epoch 80/100, Loss: 0.0023, Val Loss: 0.0107
[IL] Epoch 81/100, Loss: 0.0022, Val Loss: 0.0080
[IL] Epoch 82/100, Loss: 0.0021, Val Loss: 0.0159
[IL] Epoch 83/100, Loss: 0.0021, Val Loss: 0.0137
[IL] Epoch 84/100, Loss: 0.0024, Val Loss: 0.0120
[IL] Epoch 85/100, Loss: 0.0022, Val Loss: 0.0117
[IL] Epoch 86/100, Loss: 0.0020, Val Loss: 0.0096
[IL] Epoch 87/100, Loss: 0.0020, Val Loss: 0.0125
[IL] Epoch 88/100, Loss: 0.0021, Val Loss: 0.0121
[IL] Epoch 89/100, Loss: 0.0021, Val Loss: 0.0155
[IL] Epoch 90/100, Loss: 0.0022, Val Loss: 0.0112
[IL] Epoch 91/100, Loss: 0.0026, Val Loss: 0.0088
[IL] Epoch 92/100, Loss: 0.0022, Val Loss: 0.0104
[IL] Epoch 93/100, Loss: 0.0019, Val Loss: 0.0108
[IL] Epoch 94/100, Loss: 0.0019, Val Loss: 0.0076
[IL] Epoch 95/100, Loss: 0.0020, Val Loss: 0.0136
[IL] Epoch 96/100, Loss: 0.0020, Val Loss: 0.0107
[IL] Epoch 97/100, Loss: 0.0022, Val Loss: 0.0077
[IL] Epoch 98/100, Loss: 0.0020, Val Loss: 0.0084
[IL] Epoch 99/100, Loss: 0.0022, Val Loss: 0.0072
