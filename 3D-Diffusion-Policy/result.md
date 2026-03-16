Job start at 2026-03-14 09:49:54
Job run at:
   Static hostname: localhost.localdomain
Transient hostname: r8l40-a02
         Icon name: computer-server
           Chassis: server
        Machine ID: d7f5671651c94cdf81ff3115fae1ffa3
           Boot ID: 8d5c4fd19b9a4ce69f18a77f036b455a
  Operating System: Rocky Linux 8.7 (Green Obsidian)
       CPE OS Name: cpe:/o:rocky:rocky:8:GA
            Kernel: Linux 4.18.0-425.10.1.el8_7.x86_64
      Architecture: x86-64
Filesystem                                        Size  Used Avail Use% Mounted on
/dev/mapper/rl-root                               376G   24G  352G   7% /
/dev/nvme1n1p1                                    3.5T   26G  3.5T   1% /tmp
/dev/nvme3n1p1                                    3.5T   25G  3.5T   1% /local
/dev/mapper/rl-var                                512G   15G  498G   3% /var
/dev/nvme4n1p1                                    3.5T   39G  3.5T   2% /local/nfscache
/dev/nvme0n1p2                                    2.0G  366M  1.7G  18% /boot
/dev/nvme0n1p1                                    599M  5.8M  594M   1% /boot/efi
ssd.nas00.future.cn:/rocky8_home                   16G   15G  1.9G  89% /home
ssd.nas00.future.cn:/rocky8_workspace             400G     0  400G   0% /workspace
ssd.nas00.future.cn:/rocky8_tools                 5.0T   99G  5.0T   2% /tools
ssd.nas00.future.cn:/centos7_home                  16G  4.2G   12G  26% /centos7/home
ssd.nas00.future.cn:/centos7_workspace            400G     0  400G   0% /centos7/workspace
ssd.nas00.future.cn:/centos7_tools                5.0T  235G  4.8T   5% /centos7/tools
ssd.nas00.future.cn:/eda-tools                    8.0T  6.3T  1.8T  79% /centos7/eda-tools
hdd.nas00.future.cn:/share_personal               500G     0  500G   0% /share/personal
zone05.nas01.future.cn:/NAS_HPC_collab_codemodel   40T   37T  3.7T  91% /share/collab/codemodel
ext-zone00.nas02.future.cn:/nfs_global            407T  394T   14T  97% /nfs_global
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
    /nfs_global    256G   5120G   7168G            350k   5000k  10000k        

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
  num_offline_iterations: 10
  critic_epochs: 40
  ppo_epochs: 30
  ppo_inner_steps: 1
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
[2026-03-14 09:50:09,395][diffusion_policy_3d.model.diffusion.conditional_unet1d][INFO] - number of parameters: 2.550744e+08
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
[2026-03-14 09:50:10,995][diffusion_policy_3d.model.diffusion.conditional_unet1d][INFO] - number of parameters: 2.550744e+08
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
[OPE] Behavior policy value J_old = 0.4134
[RL PPO] Reducing policy LR: 1.00e-04 → 1.00e-05
[Offline RL] Epoch 0/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.3124
[Offline RL] Epoch 1/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0765
[Offline RL] Epoch 2/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0656
[Offline RL] Epoch 3/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0675
[Offline RL] Epoch 4/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0590
[Offline RL] Epoch 5/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0531
[Offline RL] Epoch 6/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0540
[Offline RL] Epoch 7/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0533
[Offline RL] Epoch 8/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0687
[Offline RL] Epoch 9/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0625
[Offline RL] Epoch 10/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0583
[Offline RL] Epoch 11/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0544
[Offline RL] Epoch 12/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0554
[Offline RL] Epoch 13/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0858
[Offline RL] Epoch 14/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0919
[Offline RL] Epoch 15/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0658
[Offline RL] Epoch 16/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0994
[Offline RL] Epoch 17/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1385
[Offline RL] Epoch 18/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1406
[Offline RL] Epoch 19/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1254
[Offline RL] Epoch 20/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1256
[Offline RL] Epoch 21/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1255
[Offline RL] Epoch 22/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1330
[Offline RL] Epoch 23/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1064
[Offline RL] Epoch 24/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0786
[Offline RL] Epoch 25/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0935
[Offline RL] Epoch 26/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1076
[Offline RL] Epoch 27/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0984
[Offline RL] Epoch 28/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1019
[Offline RL] Epoch 29/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1265
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_00.png
[OPE] Policy ACCEPTED: J_new=0.4418 > J_old=0.4134 + δ=0.0207

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 0)
[Collect] 20 episodes, success=0.000, env_return=76.96, rl_reward=0.00, steps=4000
[Data Collection] Success Rate: 0.000, EnvReturn: 76.96, RLReward: 0.00, Episodes: 20, Steps: 4000
[Dataset] offline collection: keeping 0/20 successful episodes (dropped 20 failures) before merge.

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.2056, Val Loss: 0.0389
[IL] Epoch 1/100, Loss: 0.0207, Val Loss: 0.0180
[IL] Epoch 2/100, Loss: 0.0084, Val Loss: 0.0094
[IL] Epoch 3/100, Loss: 0.0052, Val Loss: 0.0092
[IL] Epoch 4/100, Loss: 0.0040, Val Loss: 0.0084
[IL] Epoch 5/100, Loss: 0.0033, Val Loss: 0.0076
[IL] Epoch 6/100, Loss: 0.0030, Val Loss: 0.0098
[IL] Epoch 7/100, Loss: 0.0027, Val Loss: 0.0060
[IL] Epoch 8/100, Loss: 0.0025, Val Loss: 0.0071
[IL] Epoch 9/100, Loss: 0.0023, Val Loss: 0.0058
[IL] Epoch 10/100, Loss: 0.0022, Val Loss: 0.0080
[IL] Epoch 11/100, Loss: 0.0021, Val Loss: 0.0067
[IL] Epoch 12/100, Loss: 0.0020, Val Loss: 0.0049
[IL] Epoch 13/100, Loss: 0.0020, Val Loss: 0.0065
[IL] Epoch 14/100, Loss: 0.0018, Val Loss: 0.0075
[IL] Epoch 15/100, Loss: 0.0019, Val Loss: 0.0054
[IL] Epoch 16/100, Loss: 0.0017, Val Loss: 0.0047
[IL] Epoch 17/100, Loss: 0.0017, Val Loss: 0.0069
[IL] Epoch 18/100, Loss: 0.0017, Val Loss: 0.0064
[IL] Epoch 19/100, Loss: 0.0015, Val Loss: 0.0078
[IL] Epoch 20/100, Loss: 0.0015, Val Loss: 0.0061
[IL] Epoch 21/100, Loss: 0.0016, Val Loss: 0.0078
[IL] Epoch 22/100, Loss: 0.0015, Val Loss: 0.0064
[IL] Epoch 23/100, Loss: 0.0015, Val Loss: 0.0082
[IL] Epoch 24/100, Loss: 0.0014, Val Loss: 0.0053
[IL] Epoch 25/100, Loss: 0.0014, Val Loss: 0.0068
[IL] Epoch 26/100, Loss: 0.0014, Val Loss: 0.0062
[IL] Epoch 27/100, Loss: 0.0014, Val Loss: 0.0059
[IL] Epoch 28/100, Loss: 0.0014, Val Loss: 0.0087
[IL] Epoch 29/100, Loss: 0.0014, Val Loss: 0.0070
[IL] Epoch 30/100, Loss: 0.0014, Val Loss: 0.0067
[IL] Epoch 31/100, Loss: 0.0013, Val Loss: 0.0078
[IL] Epoch 32/100, Loss: 0.0013, Val Loss: 0.0072
[IL] Epoch 33/100, Loss: 0.0012, Val Loss: 0.0055
[IL] Epoch 34/100, Loss: 0.0013, Val Loss: 0.0076
[IL] Epoch 35/100, Loss: 0.0012, Val Loss: 0.0077
[IL] Epoch 36/100, Loss: 0.0013, Val Loss: 0.0083
[IL] Epoch 37/100, Loss: 0.0013, Val Loss: 0.0060
[IL] Epoch 38/100, Loss: 0.0013, Val Loss: 0.0061
[IL] Epoch 39/100, Loss: 0.0011, Val Loss: 0.0069
[IL] Epoch 40/100, Loss: 0.0011, Val Loss: 0.0069
[IL] Epoch 41/100, Loss: 0.0011, Val Loss: 0.0076
[IL] Epoch 42/100, Loss: 0.0012, Val Loss: 0.0068
[IL] Epoch 43/100, Loss: 0.0011, Val Loss: 0.0069
[IL] Epoch 44/100, Loss: 0.0011, Val Loss: 0.0064
[IL] Epoch 45/100, Loss: 0.0011, Val Loss: 0.0110
[IL] Epoch 46/100, Loss: 0.0011, Val Loss: 0.0057
[IL] Epoch 47/100, Loss: 0.0012, Val Loss: 0.0059
[IL] Epoch 48/100, Loss: 0.0011, Val Loss: 0.0049
[IL] Epoch 49/100, Loss: 0.0012, Val Loss: 0.0073
[IL] Epoch 50/100, Loss: 0.0011, Val Loss: 0.0064
[IL] Epoch 51/100, Loss: 0.0011, Val Loss: 0.0107
[IL] Epoch 52/100, Loss: 0.0011, Val Loss: 0.0081
[IL] Epoch 53/100, Loss: 0.0011, Val Loss: 0.0095
[IL] Epoch 54/100, Loss: 0.0010, Val Loss: 0.0090
[IL] Epoch 55/100, Loss: 0.0010, Val Loss: 0.0066
[IL] Epoch 56/100, Loss: 0.0011, Val Loss: 0.0051
[IL] Epoch 57/100, Loss: 0.0011, Val Loss: 0.0098
[IL] Epoch 58/100, Loss: 0.0011, Val Loss: 0.0090
[IL] Epoch 59/100, Loss: 0.0011, Val Loss: 0.0121
[IL] Epoch 60/100, Loss: 0.0010, Val Loss: 0.0073
[IL] Epoch 61/100, Loss: 0.0011, Val Loss: 0.0075
[IL] Epoch 62/100, Loss: 0.0011, Val Loss: 0.0069
[IL] Epoch 63/100, Loss: 0.0009, Val Loss: 0.0093
[IL] Epoch 64/100, Loss: 0.0010, Val Loss: 0.0118
[IL] Epoch 65/100, Loss: 0.0010, Val Loss: 0.0088
[IL] Epoch 66/100, Loss: 0.0010, Val Loss: 0.0070
[IL] Epoch 67/100, Loss: 0.0009, Val Loss: 0.0055
[IL] Epoch 68/100, Loss: 0.0010, Val Loss: 0.0093
[IL] Epoch 69/100, Loss: 0.0010, Val Loss: 0.0066
[IL] Epoch 70/100, Loss: 0.0009, Val Loss: 0.0112
[IL] Epoch 71/100, Loss: 0.0009, Val Loss: 0.0090
[IL] Epoch 72/100, Loss: 0.0009, Val Loss: 0.0113
[IL] Epoch 73/100, Loss: 0.0010, Val Loss: 0.0076
[IL] Epoch 74/100, Loss: 0.0010, Val Loss: 0.0080
[IL] Epoch 75/100, Loss: 0.0009, Val Loss: 0.0099
[IL] Epoch 76/100, Loss: 0.0010, Val Loss: 0.0110
[IL] Epoch 77/100, Loss: 0.0010, Val Loss: 0.0097
[IL] Epoch 78/100, Loss: 0.0009, Val Loss: 0.0106
[IL] Epoch 79/100, Loss: 0.0009, Val Loss: 0.0072
[IL] Epoch 80/100, Loss: 0.0009, Val Loss: 0.0117
[IL] Epoch 81/100, Loss: 0.0009, Val Loss: 0.0074
[IL] Epoch 82/100, Loss: 0.0009, Val Loss: 0.0074
[IL] Epoch 83/100, Loss: 0.0009, Val Loss: 0.0086
[IL] Epoch 84/100, Loss: 0.0009, Val Loss: 0.0103
[IL] Epoch 85/100, Loss: 0.0008, Val Loss: 0.0070
[IL] Epoch 86/100, Loss: 0.0009, Val Loss: 0.0062
[IL] Epoch 87/100, Loss: 0.0010, Val Loss: 0.0084
[IL] Epoch 88/100, Loss: 0.0008, Val Loss: 0.0095
[IL] Epoch 89/100, Loss: 0.0009, Val Loss: 0.0088
[IL] Epoch 90/100, Loss: 0.0009, Val Loss: 0.0069
[IL] Epoch 91/100, Loss: 0.0009, Val Loss: 0.0110
[IL] Epoch 92/100, Loss: 0.0009, Val Loss: 0.0099
[IL] Epoch 93/100, Loss: 0.0010, Val Loss: 0.0107
[IL] Epoch 94/100, Loss: 0.0009, Val Loss: 0.0090
[IL] Epoch 95/100, Loss: 0.0009, Val Loss: 0.0068
[IL] Epoch 96/100, Loss: 0.0009, Val Loss: 0.0085
[IL] Epoch 97/100, Loss: 0.0009, Val Loss: 0.0065
[IL] Epoch 98/100, Loss: 0.0009, Val Loss: 0.0085
[IL] Epoch 99/100, Loss: 0.0008, Val Loss: 0.0112
test_mean_score: 0.75
[IL] Eval - Success Rate: 0.750
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_00.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_0.ckpt

================================================================================
               OFFLINE RL ITERATION 2/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 1)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 17370 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-60.64431 | val=0.00013 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-64.80546 | val=0.00012 | no-improve=2/5
[TransitionModel] Epoch   29 | train=-66.57573 | val=0.00012 | no-improve=5/5
[TransitionModel] Training complete. Elites=[1, 3, 5, 6, 4], val_loss=0.00012

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 1)
[IQL] Epoch 0/40, V Loss: 0.0003, Q Loss: 0.0066
[IQL] Epoch 1/40, V Loss: 0.0002, Q Loss: 0.0064
[IQL] Epoch 2/40, V Loss: 0.0002, Q Loss: 0.0065
[IQL] Epoch 3/40, V Loss: 0.0002, Q Loss: 0.0063
[IQL] Epoch 4/40, V Loss: 0.0003, Q Loss: 0.0066
[IQL] Epoch 5/40, V Loss: 0.0002, Q Loss: 0.0062
[IQL] Epoch 6/40, V Loss: 0.0003, Q Loss: 0.0063
[IQL] Epoch 7/40, V Loss: 0.0002, Q Loss: 0.0062
[IQL] Epoch 8/40, V Loss: 0.0002, Q Loss: 0.0062
[IQL] Epoch 9/40, V Loss: 0.0002, Q Loss: 0.0061
[IQL] Epoch 10/40, V Loss: 0.0003, Q Loss: 0.0063
[IQL] Epoch 11/40, V Loss: 0.0003, Q Loss: 0.0061
[IQL] Epoch 12/40, V Loss: 0.0003, Q Loss: 0.0062
[IQL] Epoch 13/40, V Loss: 0.0002, Q Loss: 0.0060
[IQL] Epoch 14/40, V Loss: 0.0003, Q Loss: 0.0060
[IQL] Epoch 15/40, V Loss: 0.0002, Q Loss: 0.0059
[IQL] Epoch 16/40, V Loss: 0.0003, Q Loss: 0.0059
[IQL] Epoch 17/40, V Loss: 0.0003, Q Loss: 0.0059
[IQL] Epoch 18/40, V Loss: 0.0002, Q Loss: 0.0059
[IQL] Epoch 19/40, V Loss: 0.0003, Q Loss: 0.0058
[IQL] Epoch 20/40, V Loss: 0.0002, Q Loss: 0.0059
[IQL] Epoch 21/40, V Loss: 0.0003, Q Loss: 0.0058
[IQL] Epoch 22/40, V Loss: 0.0003, Q Loss: 0.0058
[IQL] Epoch 23/40, V Loss: 0.0003, Q Loss: 0.0058
[IQL] Epoch 24/40, V Loss: 0.0003, Q Loss: 0.0059
[IQL] Epoch 25/40, V Loss: 0.0002, Q Loss: 0.0058
[IQL] Epoch 26/40, V Loss: 0.0003, Q Loss: 0.0056
[IQL] Epoch 27/40, V Loss: 0.0003, Q Loss: 0.0056
[IQL] Epoch 28/40, V Loss: 0.0002, Q Loss: 0.0057
[IQL] Epoch 29/40, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 30/40, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 31/40, V Loss: 0.0004, Q Loss: 0.0057
[IQL] Epoch 32/40, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 33/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 34/40, V Loss: 0.0003, Q Loss: 0.0056
[IQL] Epoch 35/40, V Loss: 0.0004, Q Loss: 0.0057
[IQL] Epoch 36/40, V Loss: 0.0003, Q Loss: 0.0055
[IQL] Epoch 37/40, V Loss: 0.0003, Q Loss: 0.0055
[IQL] Epoch 38/40, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 39/40, V Loss: 0.0003, Q Loss: 0.0053
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_01.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_01.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 1)
[OPE] Behavior policy value J_old = 0.6424
[Offline RL] Epoch 0/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0630
[Offline RL] Epoch 1/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0258
[Offline RL] Epoch 2/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0205
[Offline RL] Epoch 3/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0195
[Offline RL] Epoch 4/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0189
[Offline RL] Epoch 5/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0168
[Offline RL] Epoch 6/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0183
[Offline RL] Epoch 7/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0333
[Offline RL] Epoch 8/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0515
[Offline RL] Epoch 9/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0491
[Offline RL] Epoch 10/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0479
[Offline RL] Epoch 11/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0467
[Offline RL] Epoch 12/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0345
[Offline RL] Epoch 13/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0870
[Offline RL] Epoch 14/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0798
[Offline RL] Epoch 15/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0653
[Offline RL] Epoch 16/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0467
[Offline RL] Epoch 17/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0649
[Offline RL] Epoch 18/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0678
[Offline RL] Epoch 19/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0613
[Offline RL] Epoch 20/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0715
[Offline RL] Epoch 21/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1021
[Offline RL] Epoch 22/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0803
[Offline RL] Epoch 23/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0590
[Offline RL] Epoch 24/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0470
[Offline RL] Epoch 25/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0504
[Offline RL] Epoch 26/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0825
[Offline RL] Epoch 27/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0732
[Offline RL] Epoch 28/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0446
[Offline RL] Epoch 29/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0686
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_01.png
[OPE] Policy ACCEPTED: J_new=0.7341 > J_old=0.6424 + δ=0.0321

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 1)
[Collect] 20 episodes, success=0.000, env_return=54.51, rl_reward=0.00, steps=4000
[Data Collection] Success Rate: 0.000, EnvReturn: 54.51, RLReward: 0.00, Episodes: 20, Steps: 4000
[Dataset] offline collection: keeping 0/20 successful episodes (dropped 20 failures) before merge.

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.2809, Val Loss: 0.0582
[IL] Epoch 1/100, Loss: 0.0354, Val Loss: 0.0257
[IL] Epoch 2/100, Loss: 0.0170, Val Loss: 0.0174
[IL] Epoch 3/100, Loss: 0.0108, Val Loss: 0.0138
[IL] Epoch 4/100, Loss: 0.0078, Val Loss: 0.0128
[IL] Epoch 5/100, Loss: 0.0061, Val Loss: 0.0101
[IL] Epoch 6/100, Loss: 0.0050, Val Loss: 0.0094
[IL] Epoch 7/100, Loss: 0.0043, Val Loss: 0.0095
[IL] Epoch 8/100, Loss: 0.0038, Val Loss: 0.0098
[IL] Epoch 9/100, Loss: 0.0031, Val Loss: 0.0085
[IL] Epoch 10/100, Loss: 0.0030, Val Loss: 0.0072
[IL] Epoch 11/100, Loss: 0.0026, Val Loss: 0.0081
[IL] Epoch 12/100, Loss: 0.0027, Val Loss: 0.0072
[IL] Epoch 13/100, Loss: 0.0023, Val Loss: 0.0085
[IL] Epoch 14/100, Loss: 0.0022, Val Loss: 0.0050
[IL] Epoch 15/100, Loss: 0.0023, Val Loss: 0.0086
[IL] Epoch 16/100, Loss: 0.0020, Val Loss: 0.0087
[IL] Epoch 17/100, Loss: 0.0019, Val Loss: 0.0069
[IL] Epoch 18/100, Loss: 0.0018, Val Loss: 0.0075
[IL] Epoch 19/100, Loss: 0.0017, Val Loss: 0.0082
[IL] Epoch 20/100, Loss: 0.0017, Val Loss: 0.0069
[IL] Epoch 21/100, Loss: 0.0016, Val Loss: 0.0081
[IL] Epoch 22/100, Loss: 0.0016, Val Loss: 0.0071
[IL] Epoch 23/100, Loss: 0.0016, Val Loss: 0.0085
[IL] Epoch 24/100, Loss: 0.0015, Val Loss: 0.0108
[IL] Epoch 25/100, Loss: 0.0014, Val Loss: 0.0071
[IL] Epoch 26/100, Loss: 0.0014, Val Loss: 0.0070
[IL] Epoch 27/100, Loss: 0.0013, Val Loss: 0.0065
[IL] Epoch 28/100, Loss: 0.0013, Val Loss: 0.0099
[IL] Epoch 29/100, Loss: 0.0013, Val Loss: 0.0071
[IL] Epoch 30/100, Loss: 0.0012, Val Loss: 0.0078
[IL] Epoch 31/100, Loss: 0.0012, Val Loss: 0.0086
[IL] Epoch 32/100, Loss: 0.0012, Val Loss: 0.0085
[IL] Epoch 33/100, Loss: 0.0012, Val Loss: 0.0080
[IL] Epoch 34/100, Loss: 0.0011, Val Loss: 0.0123
[IL] Epoch 35/100, Loss: 0.0013, Val Loss: 0.0081
[IL] Epoch 36/100, Loss: 0.0012, Val Loss: 0.0067
[IL] Epoch 37/100, Loss: 0.0011, Val Loss: 0.0086
[IL] Epoch 38/100, Loss: 0.0011, Val Loss: 0.0075
[IL] Epoch 39/100, Loss: 0.0012, Val Loss: 0.0063
[IL] Epoch 40/100, Loss: 0.0010, Val Loss: 0.0075
[IL] Epoch 41/100, Loss: 0.0011, Val Loss: 0.0101
[IL] Epoch 42/100, Loss: 0.0011, Val Loss: 0.0090
[IL] Epoch 43/100, Loss: 0.0011, Val Loss: 0.0083
[IL] Epoch 44/100, Loss: 0.0010, Val Loss: 0.0057
[IL] Epoch 45/100, Loss: 0.0010, Val Loss: 0.0062
[IL] Epoch 46/100, Loss: 0.0010, Val Loss: 0.0065
[IL] Epoch 47/100, Loss: 0.0010, Val Loss: 0.0117
[IL] Epoch 48/100, Loss: 0.0010, Val Loss: 0.0060
[IL] Epoch 49/100, Loss: 0.0009, Val Loss: 0.0062
[IL] Epoch 50/100, Loss: 0.0010, Val Loss: 0.0109
[IL] Epoch 51/100, Loss: 0.0010, Val Loss: 0.0094
[IL] Epoch 52/100, Loss: 0.0010, Val Loss: 0.0090
[IL] Epoch 53/100, Loss: 0.0009, Val Loss: 0.0084
[IL] Epoch 54/100, Loss: 0.0009, Val Loss: 0.0111
[IL] Epoch 55/100, Loss: 0.0008, Val Loss: 0.0083
[IL] Epoch 56/100, Loss: 0.0009, Val Loss: 0.0102
[IL] Epoch 57/100, Loss: 0.0009, Val Loss: 0.0098
[IL] Epoch 58/100, Loss: 0.0009, Val Loss: 0.0058
[IL] Epoch 59/100, Loss: 0.0009, Val Loss: 0.0109
[IL] Epoch 60/100, Loss: 0.0009, Val Loss: 0.0065
[IL] Epoch 61/100, Loss: 0.0009, Val Loss: 0.0077
[IL] Epoch 62/100, Loss: 0.0010, Val Loss: 0.0069
[IL] Epoch 63/100, Loss: 0.0009, Val Loss: 0.0076
[IL] Epoch 64/100, Loss: 0.0008, Val Loss: 0.0080
[IL] Epoch 65/100, Loss: 0.0010, Val Loss: 0.0069
[IL] Epoch 66/100, Loss: 0.0008, Val Loss: 0.0058
[IL] Epoch 67/100, Loss: 0.0008, Val Loss: 0.0102
[IL] Epoch 68/100, Loss: 0.0009, Val Loss: 0.0101
[IL] Epoch 69/100, Loss: 0.0008, Val Loss: 0.0098
[IL] Epoch 70/100, Loss: 0.0009, Val Loss: 0.0137
[IL] Epoch 71/100, Loss: 0.0009, Val Loss: 0.0058
[IL] Epoch 72/100, Loss: 0.0008, Val Loss: 0.0095
[IL] Epoch 73/100, Loss: 0.0009, Val Loss: 0.0087
[IL] Epoch 74/100, Loss: 0.0008, Val Loss: 0.0120
[IL] Epoch 75/100, Loss: 0.0008, Val Loss: 0.0082
[IL] Epoch 76/100, Loss: 0.0009, Val Loss: 0.0112
[IL] Epoch 77/100, Loss: 0.0008, Val Loss: 0.0086
[IL] Epoch 78/100, Loss: 0.0008, Val Loss: 0.0108
[IL] Epoch 79/100, Loss: 0.0009, Val Loss: 0.0088
[IL] Epoch 80/100, Loss: 0.0008, Val Loss: 0.0078
[IL] Epoch 81/100, Loss: 0.0009, Val Loss: 0.0087
[IL] Epoch 82/100, Loss: 0.0009, Val Loss: 0.0088
[IL] Epoch 83/100, Loss: 0.0008, Val Loss: 0.0103
[IL] Epoch 84/100, Loss: 0.0008, Val Loss: 0.0103
[IL] Epoch 85/100, Loss: 0.0009, Val Loss: 0.0072
[IL] Epoch 86/100, Loss: 0.0008, Val Loss: 0.0095
[IL] Epoch 87/100, Loss: 0.0008, Val Loss: 0.0090
[IL] Epoch 88/100, Loss: 0.0008, Val Loss: 0.0103
[IL] Epoch 89/100, Loss: 0.0008, Val Loss: 0.0089
[IL] Epoch 90/100, Loss: 0.0008, Val Loss: 0.0088
[IL] Epoch 91/100, Loss: 0.0007, Val Loss: 0.0107
[IL] Epoch 92/100, Loss: 0.0008, Val Loss: 0.0137
[IL] Epoch 93/100, Loss: 0.0008, Val Loss: 0.0088
[IL] Epoch 94/100, Loss: 0.0008, Val Loss: 0.0097
[IL] Epoch 95/100, Loss: 0.0008, Val Loss: 0.0117
[IL] Epoch 96/100, Loss: 0.0009, Val Loss: 0.0087
[IL] Epoch 97/100, Loss: 0.0008, Val Loss: 0.0101
[IL] Epoch 98/100, Loss: 0.0008, Val Loss: 0.0084
[IL] Epoch 99/100, Loss: 0.0007, Val Loss: 0.0097
test_mean_score: 0.6
[IL] Eval - Success Rate: 0.600
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_01.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_1.ckpt

================================================================================
               OFFLINE RL ITERATION 3/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 2)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 17370 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-66.16608 | val=0.00014 | no-improve=0/5
[TransitionModel] Epoch   14 | train=-69.30994 | val=0.00014 | no-improve=5/5
[TransitionModel] Training complete. Elites=[3, 4, 6, 0, 2], val_loss=0.00014

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 2)
[IQL] Epoch 0/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 1/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 2/40, V Loss: 0.0003, Q Loss: 0.0055
[IQL] Epoch 3/40, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 4/40, V Loss: 0.0003, Q Loss: 0.0056
[IQL] Epoch 5/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 6/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 7/40, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 8/40, V Loss: 0.0004, Q Loss: 0.0055
[IQL] Epoch 9/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 10/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 11/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 12/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 13/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 14/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 15/40, V Loss: 0.0003, Q Loss: 0.0053
[IQL] Epoch 16/40, V Loss: 0.0003, Q Loss: 0.0051
[IQL] Epoch 17/40, V Loss: 0.0003, Q Loss: 0.0053
[IQL] Epoch 18/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 19/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 20/40, V Loss: 0.0003, Q Loss: 0.0053
[IQL] Epoch 21/40, V Loss: 0.0004, Q Loss: 0.0054
[IQL] Epoch 22/40, V Loss: 0.0003, Q Loss: 0.0053
[IQL] Epoch 23/40, V Loss: 0.0003, Q Loss: 0.0055
[IQL] Epoch 24/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 25/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 26/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 27/40, V Loss: 0.0003, Q Loss: 0.0051
[IQL] Epoch 28/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 29/40, V Loss: 0.0003, Q Loss: 0.0053
[IQL] Epoch 30/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 31/40, V Loss: 0.0003, Q Loss: 0.0051
[IQL] Epoch 32/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 33/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 34/40, V Loss: 0.0003, Q Loss: 0.0050
[IQL] Epoch 35/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 36/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 37/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 38/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 39/40, V Loss: 0.0003, Q Loss: 0.0052
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_02.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_02.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 2)
[OPE] Behavior policy value J_old = 0.6657
[Offline RL] Epoch 0/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0706
[Offline RL] Epoch 1/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0265
[Offline RL] Epoch 2/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0166
[Offline RL] Epoch 3/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0141
[Offline RL] Epoch 4/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0116
[Offline RL] Epoch 5/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0110
[Offline RL] Epoch 6/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0118
[Offline RL] Epoch 7/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0113
[Offline RL] Epoch 8/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0110
[Offline RL] Epoch 9/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0111
[Offline RL] Epoch 10/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0119
[Offline RL] Epoch 11/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0160
[Offline RL] Epoch 12/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0179
[Offline RL] Epoch 13/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0132
[Offline RL] Epoch 14/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0113
[Offline RL] Epoch 15/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0108
[Offline RL] Epoch 16/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0109
[Offline RL] Epoch 17/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0114
[Offline RL] Epoch 18/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0130
[Offline RL] Epoch 19/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0129
[Offline RL] Epoch 20/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0124
[Offline RL] Epoch 21/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0122
[Offline RL] Epoch 22/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0124
[Offline RL] Epoch 23/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0168
[Offline RL] Epoch 24/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0161
[Offline RL] Epoch 25/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0136
[Offline RL] Epoch 26/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0130
[Offline RL] Epoch 27/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0113
[Offline RL] Epoch 28/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0107
[Offline RL] Epoch 29/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0090
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_02.png
[OPE] Policy REJECTED: J_new=0.6695 ≤ J_old=0.6657 + δ=0.0333. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 2)
[Collect] 20 episodes, success=0.900, env_return=1133.25, rl_reward=0.90, steps=4000
[Data Collection] Success Rate: 0.900, EnvReturn: 1133.25, RLReward: 0.90, Episodes: 20, Steps: 4000
[Dataset] offline collection: keeping 18/20 successful episodes (dropped 2 failures) before merge.
[Dataset] Merged 18 episodes (3600 steps) → total 23600 steps, 118 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.0230, Val Loss: 0.0079
[IL] Epoch 1/100, Loss: 0.0137, Val Loss: 0.0090
[IL] Epoch 2/100, Loss: 0.0144, Val Loss: 0.0054
[IL] Epoch 3/100, Loss: 0.0115, Val Loss: 0.0085
[IL] Epoch 4/100, Loss: 0.0109, Val Loss: 0.0061
[IL] Epoch 5/100, Loss: 0.0115, Val Loss: 0.0063
[IL] Epoch 6/100, Loss: 0.0104, Val Loss: 0.0049
[IL] Epoch 7/100, Loss: 0.0104, Val Loss: 0.0090
[IL] Epoch 8/100, Loss: 0.0097, Val Loss: 0.0071
[IL] Epoch 9/100, Loss: 0.0103, Val Loss: 0.0068
[IL] Epoch 10/100, Loss: 0.0098, Val Loss: 0.0057
[IL] Epoch 11/100, Loss: 0.0095, Val Loss: 0.0055
[IL] Epoch 12/100, Loss: 0.0093, Val Loss: 0.0081
[IL] Epoch 13/100, Loss: 0.0095, Val Loss: 0.0051
[IL] Epoch 14/100, Loss: 0.0094, Val Loss: 0.0057
[IL] Epoch 15/100, Loss: 0.0090, Val Loss: 0.0055
[IL] Epoch 16/100, Loss: 0.0097, Val Loss: 0.0056
[IL] Epoch 17/100, Loss: 0.0092, Val Loss: 0.0096
[IL] Epoch 18/100, Loss: 0.0093, Val Loss: 0.0088
[IL] Epoch 19/100, Loss: 0.0087, Val Loss: 0.0056
[IL] Epoch 20/100, Loss: 0.0087, Val Loss: 0.0062
[IL] Epoch 21/100, Loss: 0.0086, Val Loss: 0.0046
[IL] Epoch 22/100, Loss: 0.0086, Val Loss: 0.0079
[IL] Epoch 23/100, Loss: 0.0085, Val Loss: 0.0059
[IL] Epoch 24/100, Loss: 0.0079, Val Loss: 0.0051
[IL] Epoch 25/100, Loss: 0.0078, Val Loss: 0.0068
[IL] Epoch 26/100, Loss: 0.0078, Val Loss: 0.0083
[IL] Epoch 27/100, Loss: 0.0075, Val Loss: 0.0062
[IL] Epoch 28/100, Loss: 0.0075, Val Loss: 0.0050
[IL] Epoch 29/100, Loss: 0.0078, Val Loss: 0.0056
[IL] Epoch 30/100, Loss: 0.0075, Val Loss: 0.0058
[IL] Epoch 31/100, Loss: 0.0078, Val Loss: 0.0047
[IL] Epoch 32/100, Loss: 0.0077, Val Loss: 0.0067
[IL] Epoch 33/100, Loss: 0.0072, Val Loss: 0.0051
[IL] Epoch 34/100, Loss: 0.0073, Val Loss: 0.0073
[IL] Epoch 35/100, Loss: 0.0071, Val Loss: 0.0060
[IL] Epoch 36/100, Loss: 0.0074, Val Loss: 0.0039
[IL] Epoch 37/100, Loss: 0.0074, Val Loss: 0.0051
[IL] Epoch 38/100, Loss: 0.0068, Val Loss: 0.0066
[IL] Epoch 39/100, Loss: 0.0071, Val Loss: 0.0073
[IL] Epoch 40/100, Loss: 0.0074, Val Loss: 0.0056
[IL] Epoch 41/100, Loss: 0.0071, Val Loss: 0.0056
[IL] Epoch 42/100, Loss: 0.0071, Val Loss: 0.0052
[IL] Epoch 43/100, Loss: 0.0067, Val Loss: 0.0077
[IL] Epoch 44/100, Loss: 0.0070, Val Loss: 0.0050
[IL] Epoch 45/100, Loss: 0.0068, Val Loss: 0.0059
[IL] Epoch 46/100, Loss: 0.0071, Val Loss: 0.0067
[IL] Epoch 47/100, Loss: 0.0070, Val Loss: 0.0055
[IL] Epoch 48/100, Loss: 0.0067, Val Loss: 0.0065
[IL] Epoch 49/100, Loss: 0.0062, Val Loss: 0.0052
[IL] Epoch 50/100, Loss: 0.0067, Val Loss: 0.0062
[IL] Epoch 51/100, Loss: 0.0066, Val Loss: 0.0069
[IL] Epoch 52/100, Loss: 0.0065, Val Loss: 0.0055
[IL] Epoch 53/100, Loss: 0.0063, Val Loss: 0.0070
[IL] Epoch 54/100, Loss: 0.0066, Val Loss: 0.0058
[IL] Epoch 55/100, Loss: 0.0064, Val Loss: 0.0061
[IL] Epoch 56/100, Loss: 0.0064, Val Loss: 0.0078
[IL] Epoch 57/100, Loss: 0.0063, Val Loss: 0.0067
[IL] Epoch 58/100, Loss: 0.0065, Val Loss: 0.0059
[IL] Epoch 59/100, Loss: 0.0065, Val Loss: 0.0072
[IL] Epoch 60/100, Loss: 0.0065, Val Loss: 0.0052
[IL] Epoch 61/100, Loss: 0.0065, Val Loss: 0.0045
[IL] Epoch 62/100, Loss: 0.0066, Val Loss: 0.0081
[IL] Epoch 63/100, Loss: 0.0064, Val Loss: 0.0071
[IL] Epoch 64/100, Loss: 0.0058, Val Loss: 0.0075
[IL] Epoch 65/100, Loss: 0.0062, Val Loss: 0.0059
[IL] Epoch 66/100, Loss: 0.0061, Val Loss: 0.0055
[IL] Epoch 67/100, Loss: 0.0061, Val Loss: 0.0059
[IL] Epoch 68/100, Loss: 0.0059, Val Loss: 0.0072
[IL] Epoch 69/100, Loss: 0.0058, Val Loss: 0.0093
[IL] Epoch 70/100, Loss: 0.0062, Val Loss: 0.0089
[IL] Epoch 71/100, Loss: 0.0060, Val Loss: 0.0095
[IL] Epoch 72/100, Loss: 0.0061, Val Loss: 0.0085
[IL] Epoch 73/100, Loss: 0.0061, Val Loss: 0.0071
[IL] Epoch 74/100, Loss: 0.0061, Val Loss: 0.0100
[IL] Epoch 75/100, Loss: 0.0058, Val Loss: 0.0061
[IL] Epoch 76/100, Loss: 0.0060, Val Loss: 0.0086
[IL] Epoch 77/100, Loss: 0.0060, Val Loss: 0.0071
[IL] Epoch 78/100, Loss: 0.0056, Val Loss: 0.0075
[IL] Epoch 79/100, Loss: 0.0058, Val Loss: 0.0056
[IL] Epoch 80/100, Loss: 0.0061, Val Loss: 0.0063
[IL] Epoch 81/100, Loss: 0.0059, Val Loss: 0.0062
[IL] Epoch 82/100, Loss: 0.0056, Val Loss: 0.0066
[IL] Epoch 83/100, Loss: 0.0060, Val Loss: 0.0087
[IL] Epoch 84/100, Loss: 0.0061, Val Loss: 0.0048
[IL] Epoch 85/100, Loss: 0.0059, Val Loss: 0.0084
[IL] Epoch 86/100, Loss: 0.0054, Val Loss: 0.0086
[IL] Epoch 87/100, Loss: 0.0062, Val Loss: 0.0106
[IL] Epoch 88/100, Loss: 0.0059, Val Loss: 0.0062
[IL] Epoch 89/100, Loss: 0.0060, Val Loss: 0.0068
[IL] Epoch 90/100, Loss: 0.0058, Val Loss: 0.0066
[IL] Epoch 91/100, Loss: 0.0058, Val Loss: 0.0071
[IL] Epoch 92/100, Loss: 0.0056, Val Loss: 0.0089
[IL] Epoch 93/100, Loss: 0.0057, Val Loss: 0.0101
[IL] Epoch 94/100, Loss: 0.0058, Val Loss: 0.0080
[IL] Epoch 95/100, Loss: 0.0054, Val Loss: 0.0087
[IL] Epoch 96/100, Loss: 0.0055, Val Loss: 0.0079
[IL] Epoch 97/100, Loss: 0.0055, Val Loss: 0.0088
[IL] Epoch 98/100, Loss: 0.0058, Val Loss: 0.0090
[IL] Epoch 99/100, Loss: 0.0052, Val Loss: 0.0091
test_mean_score: 0.75
[IL] Eval - Success Rate: 0.750
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_02.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_2.ckpt

================================================================================
               OFFLINE RL ITERATION 4/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 3)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 20844 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-51.96623 | val=0.00034 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-71.84540 | val=0.00019 | no-improve=0/5
[TransitionModel] Epoch   35 | train=-75.30700 | val=0.00018 | no-improve=5/5
[TransitionModel] Training complete. Elites=[3, 4, 1, 2, 0], val_loss=0.00017

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 3)
[IQL] Epoch 0/40, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 1/40, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 2/40, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 3/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 4/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 5/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 6/40, V Loss: 0.0003, Q Loss: 0.0055
[IQL] Epoch 7/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 8/40, V Loss: 0.0001, Q Loss: 0.0054
[IQL] Epoch 9/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 10/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 11/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 12/40, V Loss: 0.0003, Q Loss: 0.0055
[IQL] Epoch 13/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 14/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 15/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 16/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 17/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 18/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 19/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 20/40, V Loss: 0.0002, Q Loss: 0.0056
[IQL] Epoch 21/40, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 22/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 23/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 24/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 25/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 26/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 27/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 28/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 29/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 30/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 31/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 32/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 33/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 34/40, V Loss: 0.0003, Q Loss: 0.0053
[IQL] Epoch 35/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 36/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 37/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 38/40, V Loss: 0.0002, Q Loss: 0.0055
[IQL] Epoch 39/40, V Loss: 0.0002, Q Loss: 0.0052
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_03.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_03.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 3)
[OPE] Behavior policy value J_old = 0.5739
[Offline RL] Epoch 0/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0800
[Offline RL] Epoch 1/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1023
[Offline RL] Epoch 2/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1279
[Offline RL] Epoch 3/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1213
[Offline RL] Epoch 4/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1650
[Offline RL] Epoch 5/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1949
[Offline RL] Epoch 6/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1894
[Offline RL] Epoch 7/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2529
[Offline RL] Epoch 8/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2596
[Offline RL] Epoch 9/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2914
[Offline RL] Epoch 10/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2921
[Offline RL] Epoch 11/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2522
[Offline RL] Epoch 12/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2649
[Offline RL] Epoch 13/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2257
[Offline RL] Epoch 14/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1987
[Offline RL] Epoch 15/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1804
Extracting GPU stats logs using atop has been completed on r8l40-a02.
Logs are being saved to: /nfs_global/S/yangrongzheng/atop-737411-r8l40-a02-gpustat.log
[Offline RL] Epoch 16/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1840
[Offline RL] Epoch 17/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1996
[Offline RL] Epoch 18/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1803
[Offline RL] Epoch 19/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1991
[Offline RL] Epoch 20/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1919
[Offline RL] Epoch 21/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2428
[Offline RL] Epoch 22/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2781
[Offline RL] Epoch 23/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2949
[Offline RL] Epoch 24/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2039
[Offline RL] Epoch 25/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1630
[Offline RL] Epoch 26/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1417
[Offline RL] Epoch 27/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1326
[Offline RL] Epoch 28/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1158
[Offline RL] Epoch 29/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1036
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_03.png
[OPE] Policy ACCEPTED: J_new=0.8487 > J_old=0.5739 + δ=0.0287

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 3)
[Collect] 20 episodes, success=0.000, env_return=45.79, rl_reward=0.00, steps=4000
[Data Collection] Success Rate: 0.000, EnvReturn: 45.79, RLReward: 0.00, Episodes: 20, Steps: 4000
[Dataset] offline collection: keeping 0/20 successful episodes (dropped 20 failures) before merge.

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.7194, Val Loss: 0.0544
[IL] Epoch 1/100, Loss: 0.0367, Val Loss: 0.0261
[IL] Epoch 2/100, Loss: 0.0220, Val Loss: 0.0211
[IL] Epoch 3/100, Loss: 0.0185, Val Loss: 0.0158
[IL] Epoch 4/100, Loss: 0.0153, Val Loss: 0.0151
[IL] Epoch 5/100, Loss: 0.0145, Val Loss: 0.0127
[IL] Epoch 6/100, Loss: 0.0132, Val Loss: 0.0117
[IL] Epoch 7/100, Loss: 0.0124, Val Loss: 0.0104
[IL] Epoch 8/100, Loss: 0.0122, Val Loss: 0.0133
[IL] Epoch 9/100, Loss: 0.0111, Val Loss: 0.0105
[IL] Epoch 10/100, Loss: 0.0108, Val Loss: 0.0101
[IL] Epoch 11/100, Loss: 0.0100, Val Loss: 0.0095
[IL] Epoch 12/100, Loss: 0.0101, Val Loss: 0.0088
[IL] Epoch 13/100, Loss: 0.0099, Val Loss: 0.0098
[IL] Epoch 14/100, Loss: 0.0096, Val Loss: 0.0085
[IL] Epoch 15/100, Loss: 0.0095, Val Loss: 0.0076
[IL] Epoch 16/100, Loss: 0.0092, Val Loss: 0.0098
[IL] Epoch 17/100, Loss: 0.0089, Val Loss: 0.0083
[IL] Epoch 18/100, Loss: 0.0085, Val Loss: 0.0105
[IL] Epoch 19/100, Loss: 0.0081, Val Loss: 0.0066
[IL] Epoch 20/100, Loss: 0.0083, Val Loss: 0.0078
[IL] Epoch 21/100, Loss: 0.0083, Val Loss: 0.0080
[IL] Epoch 22/100, Loss: 0.0074, Val Loss: 0.0084
[IL] Epoch 23/100, Loss: 0.0073, Val Loss: 0.0076
[IL] Epoch 24/100, Loss: 0.0076, Val Loss: 0.0075
[IL] Epoch 25/100, Loss: 0.0071, Val Loss: 0.0072
[IL] Epoch 26/100, Loss: 0.0065, Val Loss: 0.0072
[IL] Epoch 27/100, Loss: 0.0067, Val Loss: 0.0072
[IL] Epoch 28/100, Loss: 0.0064, Val Loss: 0.0066
[IL] Epoch 29/100, Loss: 0.0066, Val Loss: 0.0060
[IL] Epoch 30/100, Loss: 0.0064, Val Loss: 0.0070
[IL] Epoch 31/100, Loss: 0.0066, Val Loss: 0.0074
[IL] Epoch 32/100, Loss: 0.0063, Val Loss: 0.0055
[IL] Epoch 33/100, Loss: 0.0063, Val Loss: 0.0055
[IL] Epoch 34/100, Loss: 0.0061, Val Loss: 0.0081
[IL] Epoch 35/100, Loss: 0.0060, Val Loss: 0.0067
[IL] Epoch 36/100, Loss: 0.0060, Val Loss: 0.0056
[IL] Epoch 37/100, Loss: 0.0062, Val Loss: 0.0094
[IL] Epoch 38/100, Loss: 0.0054, Val Loss: 0.0051
[IL] Epoch 39/100, Loss: 0.0061, Val Loss: 0.0090
[IL] Epoch 40/100, Loss: 0.0058, Val Loss: 0.0072
[IL] Epoch 41/100, Loss: 0.0055, Val Loss: 0.0072
[IL] Epoch 42/100, Loss: 0.0060, Val Loss: 0.0084
[IL] Epoch 43/100, Loss: 0.0054, Val Loss: 0.0058
[IL] Epoch 44/100, Loss: 0.0055, Val Loss: 0.0071
[IL] Epoch 45/100, Loss: 0.0061, Val Loss: 0.0066
[IL] Epoch 46/100, Loss: 0.0060, Val Loss: 0.0072
[IL] Epoch 47/100, Loss: 0.0053, Val Loss: 0.0070
[IL] Epoch 48/100, Loss: 0.0056, Val Loss: 0.0060
[IL] Epoch 49/100, Loss: 0.0060, Val Loss: 0.0077
[IL] Epoch 50/100, Loss: 0.0053, Val Loss: 0.0057
[IL] Epoch 51/100, Loss: 0.0054, Val Loss: 0.0053
[IL] Epoch 52/100, Loss: 0.0057, Val Loss: 0.0079
[IL] Epoch 53/100, Loss: 0.0057, Val Loss: 0.0069
[IL] Epoch 54/100, Loss: 0.0052, Val Loss: 0.0084
[IL] Epoch 55/100, Loss: 0.0054, Val Loss: 0.0057
[IL] Epoch 56/100, Loss: 0.0053, Val Loss: 0.0068
[IL] Epoch 57/100, Loss: 0.0051, Val Loss: 0.0079
[IL] Epoch 58/100, Loss: 0.0051, Val Loss: 0.0052
[IL] Epoch 59/100, Loss: 0.0053, Val Loss: 0.0045
[IL] Epoch 60/100, Loss: 0.0058, Val Loss: 0.0069
[IL] Epoch 61/100, Loss: 0.0054, Val Loss: 0.0081
[IL] Epoch 62/100, Loss: 0.0055, Val Loss: 0.0071
[IL] Epoch 63/100, Loss: 0.0051, Val Loss: 0.0103
[IL] Epoch 64/100, Loss: 0.0054, Val Loss: 0.0078
[IL] Epoch 65/100, Loss: 0.0056, Val Loss: 0.0070
[IL] Epoch 66/100, Loss: 0.0051, Val Loss: 0.0075
[IL] Epoch 67/100, Loss: 0.0050, Val Loss: 0.0103
[IL] Epoch 68/100, Loss: 0.0053, Val Loss: 0.0085
[IL] Epoch 69/100, Loss: 0.0053, Val Loss: 0.0068
[IL] Epoch 70/100, Loss: 0.0055, Val Loss: 0.0068
[IL] Epoch 71/100, Loss: 0.0050, Val Loss: 0.0084
[IL] Epoch 72/100, Loss: 0.0051, Val Loss: 0.0084
[IL] Epoch 73/100, Loss: 0.0054, Val Loss: 0.0080
[IL] Epoch 74/100, Loss: 0.0053, Val Loss: 0.0097
[IL] Epoch 75/100, Loss: 0.0050, Val Loss: 0.0074
[IL] Epoch 76/100, Loss: 0.0054, Val Loss: 0.0090
[IL] Epoch 77/100, Loss: 0.0057, Val Loss: 0.0049
[IL] Epoch 78/100, Loss: 0.0053, Val Loss: 0.0056
[IL] Epoch 79/100, Loss: 0.0051, Val Loss: 0.0103
[IL] Epoch 80/100, Loss: 0.0051, Val Loss: 0.0087
[IL] Epoch 81/100, Loss: 0.0056, Val Loss: 0.0083
[IL] Epoch 82/100, Loss: 0.0051, Val Loss: 0.0059
[IL] Epoch 83/100, Loss: 0.0051, Val Loss: 0.0058
[IL] Epoch 84/100, Loss: 0.0050, Val Loss: 0.0080
[IL] Epoch 85/100, Loss: 0.0051, Val Loss: 0.0077
[IL] Epoch 86/100, Loss: 0.0052, Val Loss: 0.0093
[IL] Epoch 87/100, Loss: 0.0052, Val Loss: 0.0057
[IL] Epoch 88/100, Loss: 0.0048, Val Loss: 0.0061
[IL] Epoch 89/100, Loss: 0.0049, Val Loss: 0.0101
[IL] Epoch 90/100, Loss: 0.0047, Val Loss: 0.0085
[IL] Epoch 91/100, Loss: 0.0052, Val Loss: 0.0080
[IL] Epoch 92/100, Loss: 0.0051, Val Loss: 0.0103
[IL] Epoch 93/100, Loss: 0.0050, Val Loss: 0.0061
[IL] Epoch 94/100, Loss: 0.0051, Val Loss: 0.0063
[IL] Epoch 95/100, Loss: 0.0050, Val Loss: 0.0070
[IL] Epoch 96/100, Loss: 0.0050, Val Loss: 0.0083
[IL] Epoch 97/100, Loss: 0.0047, Val Loss: 0.0063
[IL] Epoch 98/100, Loss: 0.0048, Val Loss: 0.0086
[IL] Epoch 99/100, Loss: 0.0048, Val Loss: 0.0058
test_mean_score: 0.55
[IL] Eval - Success Rate: 0.550
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_03.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_3.ckpt

================================================================================
               OFFLINE RL ITERATION 5/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 4)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 20844 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-74.46673 | val=0.00017 | no-improve=0/5
[TransitionModel] Epoch   11 | train=-77.69132 | val=0.00018 | no-improve=5/5
[TransitionModel] Training complete. Elites=[3, 1, 4, 2, 0], val_loss=0.00016

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 4)
[IQL] Epoch 0/40, V Loss: 0.0003, Q Loss: 0.0055
[IQL] Epoch 1/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 2/40, V Loss: 0.0002, Q Loss: 0.0053
[IQL] Epoch 3/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 4/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 5/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 6/40, V Loss: 0.0003, Q Loss: 0.0053
[IQL] Epoch 7/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 8/40, V Loss: 0.0003, Q Loss: 0.0053
[IQL] Epoch 9/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 10/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 11/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 12/40, V Loss: 0.0004, Q Loss: 0.0056
[IQL] Epoch 13/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 14/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 15/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 16/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 17/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 18/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 19/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 20/40, V Loss: 0.0002, Q Loss: 0.0054
[IQL] Epoch 21/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 22/40, V Loss: 0.0003, Q Loss: 0.0054
[IQL] Epoch 23/40, V Loss: 0.0003, Q Loss: 0.0051
[IQL] Epoch 24/40, V Loss: 0.0003, Q Loss: 0.0056
[IQL] Epoch 25/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 26/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 27/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 28/40, V Loss: 0.0003, Q Loss: 0.0051
[IQL] Epoch 29/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 30/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 31/40, V Loss: 0.0003, Q Loss: 0.0053
[IQL] Epoch 32/40, V Loss: 0.0003, Q Loss: 0.0050
[IQL] Epoch 33/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 34/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 35/40, V Loss: 0.0002, Q Loss: 0.0049
[IQL] Epoch 36/40, V Loss: 0.0003, Q Loss: 0.0050
[IQL] Epoch 37/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 38/40, V Loss: 0.0002, Q Loss: 0.0051
[IQL] Epoch 39/40, V Loss: 0.0002, Q Loss: 0.0051
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_04.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_04.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 4)
[OPE] Behavior policy value J_old = 0.6803
[Offline RL] Epoch 0/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0891
[Offline RL] Epoch 1/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0384
[Offline RL] Epoch 2/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0291
[Offline RL] Epoch 3/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0240
[Offline RL] Epoch 4/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0281
[Offline RL] Epoch 5/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0256
[Offline RL] Epoch 6/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0451
[Offline RL] Epoch 7/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0834
[Offline RL] Epoch 8/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1619
[Offline RL] Epoch 9/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1750
[Offline RL] Epoch 10/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1675
[Offline RL] Epoch 11/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0976
[Offline RL] Epoch 12/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0747
[Offline RL] Epoch 13/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0586
[Offline RL] Epoch 14/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0486
[Offline RL] Epoch 15/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0491
[Offline RL] Epoch 16/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0387
[Offline RL] Epoch 17/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0394
[Offline RL] Epoch 18/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0494
[Offline RL] Epoch 19/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0738
[Offline RL] Epoch 20/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0710
[Offline RL] Epoch 21/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0381
[Offline RL] Epoch 22/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0303
[Offline RL] Epoch 23/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0389
[Offline RL] Epoch 24/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0388
[Offline RL] Epoch 25/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0764
[Offline RL] Epoch 26/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1359
[Offline RL] Epoch 27/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1281
[Offline RL] Epoch 28/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1418
[Offline RL] Epoch 29/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1457
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_04.png
[OPE] Policy ACCEPTED: J_new=0.7618 > J_old=0.6803 + δ=0.0340

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 4)
[Collect] 20 episodes, success=0.000, env_return=57.69, rl_reward=0.00, steps=4000
[Data Collection] Success Rate: 0.000, EnvReturn: 57.69, RLReward: 0.00, Episodes: 20, Steps: 4000
[Dataset] offline collection: keeping 0/20 successful episodes (dropped 20 failures) before merge.

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.1357, Val Loss: 0.0391
[IL] Epoch 1/100, Loss: 0.0241, Val Loss: 0.0167
[IL] Epoch 2/100, Loss: 0.0133, Val Loss: 0.0112
[IL] Epoch 3/100, Loss: 0.0104, Val Loss: 0.0111
[IL] Epoch 4/100, Loss: 0.0088, Val Loss: 0.0108
[IL] Epoch 5/100, Loss: 0.0084, Val Loss: 0.0093
[IL] Epoch 6/100, Loss: 0.0076, Val Loss: 0.0080
[IL] Epoch 7/100, Loss: 0.0074, Val Loss: 0.0070
[IL] Epoch 8/100, Loss: 0.0070, Val Loss: 0.0061
[IL] Epoch 9/100, Loss: 0.0065, Val Loss: 0.0065
[IL] Epoch 10/100, Loss: 0.0062, Val Loss: 0.0070
[IL] Epoch 11/100, Loss: 0.0065, Val Loss: 0.0083
[IL] Epoch 12/100, Loss: 0.0064, Val Loss: 0.0042
[IL] Epoch 13/100, Loss: 0.0061, Val Loss: 0.0060
[IL] Epoch 14/100, Loss: 0.0057, Val Loss: 0.0076
[IL] Epoch 15/100, Loss: 0.0058, Val Loss: 0.0081
[IL] Epoch 16/100, Loss: 0.0060, Val Loss: 0.0073
[IL] Epoch 17/100, Loss: 0.0056, Val Loss: 0.0063
[IL] Epoch 18/100, Loss: 0.0053, Val Loss: 0.0059
[IL] Epoch 19/100, Loss: 0.0055, Val Loss: 0.0087
[IL] Epoch 20/100, Loss: 0.0057, Val Loss: 0.0134
[IL] Epoch 21/100, Loss: 0.0052, Val Loss: 0.0083
[IL] Epoch 22/100, Loss: 0.0052, Val Loss: 0.0073
[IL] Epoch 23/100, Loss: 0.0053, Val Loss: 0.0062
[IL] Epoch 24/100, Loss: 0.0053, Val Loss: 0.0068
[IL] Epoch 25/100, Loss: 0.0052, Val Loss: 0.0071
[IL] Epoch 26/100, Loss: 0.0050, Val Loss: 0.0084
[IL] Epoch 27/100, Loss: 0.0051, Val Loss: 0.0062
[IL] Epoch 28/100, Loss: 0.0050, Val Loss: 0.0077
[IL] Epoch 29/100, Loss: 0.0051, Val Loss: 0.0070
[IL] Epoch 30/100, Loss: 0.0056, Val Loss: 0.0070
[IL] Epoch 31/100, Loss: 0.0055, Val Loss: 0.0081
[IL] Epoch 32/100, Loss: 0.0053, Val Loss: 0.0089
[IL] Epoch 33/100, Loss: 0.0053, Val Loss: 0.0086
[IL] Epoch 34/100, Loss: 0.0049, Val Loss: 0.0073
[IL] Epoch 35/100, Loss: 0.0051, Val Loss: 0.0072
[IL] Epoch 36/100, Loss: 0.0051, Val Loss: 0.0093
[IL] Epoch 37/100, Loss: 0.0051, Val Loss: 0.0083
[IL] Epoch 38/100, Loss: 0.0049, Val Loss: 0.0067
[IL] Epoch 39/100, Loss: 0.0049, Val Loss: 0.0063
[IL] Epoch 40/100, Loss: 0.0048, Val Loss: 0.0086
[IL] Epoch 41/100, Loss: 0.0051, Val Loss: 0.0072
[IL] Epoch 42/100, Loss: 0.0049, Val Loss: 0.0076
[IL] Epoch 43/100, Loss: 0.0050, Val Loss: 0.0079
[IL] Epoch 44/100, Loss: 0.0046, Val Loss: 0.0080
[IL] Epoch 45/100, Loss: 0.0053, Val Loss: 0.0101
[IL] Epoch 46/100, Loss: 0.0047, Val Loss: 0.0102
[IL] Epoch 47/100, Loss: 0.0051, Val Loss: 0.0090
[IL] Epoch 48/100, Loss: 0.0048, Val Loss: 0.0071
[IL] Epoch 49/100, Loss: 0.0049, Val Loss: 0.0067
[IL] Epoch 50/100, Loss: 0.0049, Val Loss: 0.0090
[IL] Epoch 51/100, Loss: 0.0048, Val Loss: 0.0069
[IL] Epoch 52/100, Loss: 0.0050, Val Loss: 0.0111
[IL] Epoch 53/100, Loss: 0.0046, Val Loss: 0.0086
[IL] Epoch 54/100, Loss: 0.0047, Val Loss: 0.0094
[IL] Epoch 55/100, Loss: 0.0048, Val Loss: 0.0091
[IL] Epoch 56/100, Loss: 0.0049, Val Loss: 0.0097
[IL] Epoch 57/100, Loss: 0.0048, Val Loss: 0.0076
[IL] Epoch 58/100, Loss: 0.0045, Val Loss: 0.0081
[IL] Epoch 59/100, Loss: 0.0044, Val Loss: 0.0079
[IL] Epoch 60/100, Loss: 0.0047, Val Loss: 0.0106
[IL] Epoch 61/100, Loss: 0.0047, Val Loss: 0.0075
[IL] Epoch 62/100, Loss: 0.0046, Val Loss: 0.0060
[IL] Epoch 63/100, Loss: 0.0048, Val Loss: 0.0066
[IL] Epoch 64/100, Loss: 0.0049, Val Loss: 0.0073
[IL] Epoch 65/100, Loss: 0.0047, Val Loss: 0.0085
[IL] Epoch 66/100, Loss: 0.0049, Val Loss: 0.0080
[IL] Epoch 67/100, Loss: 0.0045, Val Loss: 0.0091
[IL] Epoch 68/100, Loss: 0.0048, Val Loss: 0.0090
[IL] Epoch 69/100, Loss: 0.0045, Val Loss: 0.0097
[IL] Epoch 70/100, Loss: 0.0045, Val Loss: 0.0078
[IL] Epoch 71/100, Loss: 0.0048, Val Loss: 0.0066
[IL] Epoch 72/100, Loss: 0.0046, Val Loss: 0.0088
[IL] Epoch 73/100, Loss: 0.0046, Val Loss: 0.0120
[IL] Epoch 74/100, Loss: 0.0046, Val Loss: 0.0070
[IL] Epoch 75/100, Loss: 0.0044, Val Loss: 0.0083
[IL] Epoch 76/100, Loss: 0.0043, Val Loss: 0.0089
[IL] Epoch 77/100, Loss: 0.0045, Val Loss: 0.0095
[IL] Epoch 78/100, Loss: 0.0047, Val Loss: 0.0078
[IL] Epoch 79/100, Loss: 0.0046, Val Loss: 0.0083
[IL] Epoch 80/100, Loss: 0.0045, Val Loss: 0.0101
[IL] Epoch 81/100, Loss: 0.0047, Val Loss: 0.0119
[IL] Epoch 82/100, Loss: 0.0048, Val Loss: 0.0094
[IL] Epoch 83/100, Loss: 0.0043, Val Loss: 0.0080
[IL] Epoch 84/100, Loss: 0.0048, Val Loss: 0.0071
[IL] Epoch 85/100, Loss: 0.0043, Val Loss: 0.0082
[IL] Epoch 86/100, Loss: 0.0045, Val Loss: 0.0082
[IL] Epoch 87/100, Loss: 0.0046, Val Loss: 0.0086
[IL] Epoch 88/100, Loss: 0.0046, Val Loss: 0.0084
[IL] Epoch 89/100, Loss: 0.0042, Val Loss: 0.0095
[IL] Epoch 90/100, Loss: 0.0045, Val Loss: 0.0100
[IL] Epoch 91/100, Loss: 0.0044, Val Loss: 0.0091
[IL] Epoch 92/100, Loss: 0.0044, Val Loss: 0.0117
[IL] Epoch 93/100, Loss: 0.0045, Val Loss: 0.0110
[IL] Epoch 94/100, Loss: 0.0044, Val Loss: 0.0092
[IL] Epoch 95/100, Loss: 0.0045, Val Loss: 0.0076
[IL] Epoch 96/100, Loss: 0.0045, Val Loss: 0.0084
[IL] Epoch 97/100, Loss: 0.0043, Val Loss: 0.0076
[IL] Epoch 98/100, Loss: 0.0043, Val Loss: 0.0091
[IL] Epoch 99/100, Loss: 0.0043, Val Loss: 0.0092
test_mean_score: 0.65
[IL] Eval - Success Rate: 0.650
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_04.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_4.ckpt

================================================================================
               OFFLINE RL ITERATION 6/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 5)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 20844 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-77.17187 | val=0.00015 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-82.14141 | val=0.00015 | no-improve=2/5
[TransitionModel] Epoch   28 | train=-83.80465 | val=0.00014 | no-improve=5/5
[TransitionModel] Training complete. Elites=[2, 6, 3, 1, 4], val_loss=0.00013

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 5)
[IQL] Epoch 0/40, V Loss: 0.0003, Q Loss: 0.0053
[IQL] Epoch 1/40, V Loss: 0.0003, Q Loss: 0.0052
[IQL] Epoch 2/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 3/40, V Loss: 0.0003, Q Loss: 0.0051
[IQL] Epoch 4/40, V Loss: 0.0003, Q Loss: 0.0050
[IQL] Epoch 5/40, V Loss: 0.0003, Q Loss: 0.0049
[IQL] Epoch 6/40, V Loss: 0.0003, Q Loss: 0.0049
[IQL] Epoch 7/40, V Loss: 0.0003, Q Loss: 0.0050
[IQL] Epoch 8/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 9/40, V Loss: 0.0003, Q Loss: 0.0049
[IQL] Epoch 10/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 11/40, V Loss: 0.0003, Q Loss: 0.0050
[IQL] Epoch 12/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 13/40, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 14/40, V Loss: 0.0003, Q Loss: 0.0050
[IQL] Epoch 15/40, V Loss: 0.0002, Q Loss: 0.0047
[IQL] Epoch 16/40, V Loss: 0.0003, Q Loss: 0.0050
[IQL] Epoch 17/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 18/40, V Loss: 0.0003, Q Loss: 0.0050
[IQL] Epoch 19/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 20/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 21/40, V Loss: 0.0002, Q Loss: 0.0048
[IQL] Epoch 22/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 23/40, V Loss: 0.0003, Q Loss: 0.0049
[IQL] Epoch 24/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 25/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 26/40, V Loss: 0.0002, Q Loss: 0.0049
[IQL] Epoch 27/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 28/40, V Loss: 0.0002, Q Loss: 0.0048
[IQL] Epoch 29/40, V Loss: 0.0003, Q Loss: 0.0051
[IQL] Epoch 30/40, V Loss: 0.0004, Q Loss: 0.0049
[IQL] Epoch 31/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 32/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 33/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 34/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 35/40, V Loss: 0.0003, Q Loss: 0.0049
[IQL] Epoch 36/40, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 37/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 38/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 39/40, V Loss: 0.0003, Q Loss: 0.0050
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_05.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_05.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 5)
[OPE] Behavior policy value J_old = 0.6867
[Offline RL] Epoch 0/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0507
[Offline RL] Epoch 1/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0299
[Offline RL] Epoch 2/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0372
[Offline RL] Epoch 3/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0902
[Offline RL] Epoch 4/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0964
[Offline RL] Epoch 5/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0972
[Offline RL] Epoch 6/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0940
[Offline RL] Epoch 7/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0857
[Offline RL] Epoch 8/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0624
[Offline RL] Epoch 9/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0689
[Offline RL] Epoch 10/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0674
[Offline RL] Epoch 11/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1042
[Offline RL] Epoch 12/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1346
[Offline RL] Epoch 13/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1336
[Offline RL] Epoch 14/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1470
[Offline RL] Epoch 15/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1721
[Offline RL] Epoch 16/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2359
[Offline RL] Epoch 17/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2082
[Offline RL] Epoch 18/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1848
[Offline RL] Epoch 19/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0884
[Offline RL] Epoch 20/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0899
[Offline RL] Epoch 21/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1253
[Offline RL] Epoch 22/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1787
[Offline RL] Epoch 23/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2961
[Offline RL] Epoch 24/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.3378
[Offline RL] Epoch 25/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.4009
[Offline RL] Epoch 26/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.3955
[Offline RL] Epoch 27/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.3572
[Offline RL] Epoch 28/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.3411
[Offline RL] Epoch 29/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2881
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_05.png
[OPE] Policy REJECTED: J_new=0.7056 ≤ J_old=0.6867 + δ=0.0343. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 5)
[Collect] 20 episodes, success=0.700, env_return=1062.97, rl_reward=0.70, steps=4000
[Data Collection] Success Rate: 0.700, EnvReturn: 1062.97, RLReward: 0.70, Episodes: 20, Steps: 4000
[Dataset] offline collection: keeping 14/20 successful episodes (dropped 6 failures) before merge.
[Dataset] Merged 14 episodes (2800 steps) → total 26400 steps, 132 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.0065, Val Loss: 0.0084
[IL] Epoch 1/100, Loss: 0.0060, Val Loss: 0.0062
[IL] Epoch 2/100, Loss: 0.0056, Val Loss: 0.0072
[IL] Epoch 3/100, Loss: 0.0057, Val Loss: 0.0102
[IL] Epoch 4/100, Loss: 0.0057, Val Loss: 0.0093
[IL] Epoch 5/100, Loss: 0.0055, Val Loss: 0.0100
[IL] Epoch 6/100, Loss: 0.0057, Val Loss: 0.0062
[IL] Epoch 7/100, Loss: 0.0058, Val Loss: 0.0059
[IL] Epoch 8/100, Loss: 0.0057, Val Loss: 0.0084
[IL] Epoch 9/100, Loss: 0.0056, Val Loss: 0.0070
[IL] Epoch 10/100, Loss: 0.0058, Val Loss: 0.0079
[IL] Epoch 11/100, Loss: 0.0056, Val Loss: 0.0116
[IL] Epoch 12/100, Loss: 0.0060, Val Loss: 0.0077
[IL] Epoch 13/100, Loss: 0.0054, Val Loss: 0.0068
[IL] Epoch 14/100, Loss: 0.0055, Val Loss: 0.0097
[IL] Epoch 15/100, Loss: 0.0051, Val Loss: 0.0103
[IL] Epoch 16/100, Loss: 0.0055, Val Loss: 0.0111
[IL] Epoch 17/100, Loss: 0.0056, Val Loss: 0.0067
[IL] Epoch 18/100, Loss: 0.0056, Val Loss: 0.0100
[IL] Epoch 19/100, Loss: 0.0050, Val Loss: 0.0065
[IL] Epoch 20/100, Loss: 0.0058, Val Loss: 0.0092
[IL] Epoch 21/100, Loss: 0.0053, Val Loss: 0.0118
[IL] Epoch 22/100, Loss: 0.0051, Val Loss: 0.0076
[IL] Epoch 23/100, Loss: 0.0050, Val Loss: 0.0063
[IL] Epoch 24/100, Loss: 0.0050, Val Loss: 0.0071
[IL] Epoch 25/100, Loss: 0.0053, Val Loss: 0.0063
[IL] Epoch 26/100, Loss: 0.0055, Val Loss: 0.0057
[IL] Epoch 27/100, Loss: 0.0053, Val Loss: 0.0084
[IL] Epoch 28/100, Loss: 0.0054, Val Loss: 0.0103
[IL] Epoch 29/100, Loss: 0.0049, Val Loss: 0.0075
[IL] Epoch 30/100, Loss: 0.0050, Val Loss: 0.0077
[IL] Epoch 31/100, Loss: 0.0049, Val Loss: 0.0056
[IL] Epoch 32/100, Loss: 0.0049, Val Loss: 0.0099
[IL] Epoch 33/100, Loss: 0.0050, Val Loss: 0.0093
[IL] Epoch 34/100, Loss: 0.0051, Val Loss: 0.0084
[IL] Epoch 35/100, Loss: 0.0050, Val Loss: 0.0095
[IL] Epoch 36/100, Loss: 0.0049, Val Loss: 0.0074
[IL] Epoch 37/100, Loss: 0.0054, Val Loss: 0.0089
[IL] Epoch 38/100, Loss: 0.0049, Val Loss: 0.0098
[IL] Epoch 39/100, Loss: 0.0049, Val Loss: 0.0085
[IL] Epoch 40/100, Loss: 0.0049, Val Loss: 0.0101
[IL] Epoch 41/100, Loss: 0.0049, Val Loss: 0.0086
[IL] Epoch 42/100, Loss: 0.0054, Val Loss: 0.0121
[IL] Epoch 43/100, Loss: 0.0047, Val Loss: 0.0074
[IL] Epoch 44/100, Loss: 0.0048, Val Loss: 0.0113
[IL] Epoch 45/100, Loss: 0.0051, Val Loss: 0.0090
[IL] Epoch 46/100, Loss: 0.0049, Val Loss: 0.0097
[IL] Epoch 47/100, Loss: 0.0047, Val Loss: 0.0096
[IL] Epoch 48/100, Loss: 0.0049, Val Loss: 0.0091
[IL] Epoch 49/100, Loss: 0.0049, Val Loss: 0.0071
[IL] Epoch 50/100, Loss: 0.0048, Val Loss: 0.0067
[IL] Epoch 51/100, Loss: 0.0047, Val Loss: 0.0058
[IL] Epoch 52/100, Loss: 0.0048, Val Loss: 0.0089
[IL] Epoch 53/100, Loss: 0.0048, Val Loss: 0.0087
[IL] Epoch 54/100, Loss: 0.0050, Val Loss: 0.0056
[IL] Epoch 55/100, Loss: 0.0049, Val Loss: 0.0098
[IL] Epoch 56/100, Loss: 0.0046, Val Loss: 0.0120
[IL] Epoch 57/100, Loss: 0.0047, Val Loss: 0.0127
[IL] Epoch 58/100, Loss: 0.0044, Val Loss: 0.0076
[IL] Epoch 59/100, Loss: 0.0050, Val Loss: 0.0088
[IL] Epoch 60/100, Loss: 0.0048, Val Loss: 0.0110
[IL] Epoch 61/100, Loss: 0.0049, Val Loss: 0.0081
[IL] Epoch 62/100, Loss: 0.0048, Val Loss: 0.0097
[IL] Epoch 63/100, Loss: 0.0048, Val Loss: 0.0111
[IL] Epoch 64/100, Loss: 0.0046, Val Loss: 0.0097
[IL] Epoch 65/100, Loss: 0.0046, Val Loss: 0.0087
[IL] Epoch 66/100, Loss: 0.0046, Val Loss: 0.0077
[IL] Epoch 67/100, Loss: 0.0049, Val Loss: 0.0071
[IL] Epoch 68/100, Loss: 0.0045, Val Loss: 0.0099
[IL] Epoch 69/100, Loss: 0.0047, Val Loss: 0.0095
[IL] Epoch 70/100, Loss: 0.0051, Val Loss: 0.0099
[IL] Epoch 71/100, Loss: 0.0046, Val Loss: 0.0099
[IL] Epoch 72/100, Loss: 0.0048, Val Loss: 0.0061
[IL] Epoch 73/100, Loss: 0.0048, Val Loss: 0.0091
[IL] Epoch 74/100, Loss: 0.0046, Val Loss: 0.0081
[IL] Epoch 75/100, Loss: 0.0048, Val Loss: 0.0070
[IL] Epoch 76/100, Loss: 0.0046, Val Loss: 0.0104
[IL] Epoch 77/100, Loss: 0.0045, Val Loss: 0.0098
[IL] Epoch 78/100, Loss: 0.0045, Val Loss: 0.0100
[IL] Epoch 79/100, Loss: 0.0044, Val Loss: 0.0069
[IL] Epoch 80/100, Loss: 0.0047, Val Loss: 0.0114
[IL] Epoch 81/100, Loss: 0.0045, Val Loss: 0.0102
[IL] Epoch 82/100, Loss: 0.0044, Val Loss: 0.0096
[IL] Epoch 83/100, Loss: 0.0046, Val Loss: 0.0139
[IL] Epoch 84/100, Loss: 0.0046, Val Loss: 0.0101
[IL] Epoch 85/100, Loss: 0.0044, Val Loss: 0.0094
[IL] Epoch 86/100, Loss: 0.0045, Val Loss: 0.0090
[IL] Epoch 87/100, Loss: 0.0043, Val Loss: 0.0124
[IL] Epoch 88/100, Loss: 0.0044, Val Loss: 0.0132
[IL] Epoch 89/100, Loss: 0.0042, Val Loss: 0.0155
[IL] Epoch 90/100, Loss: 0.0045, Val Loss: 0.0095
[IL] Epoch 91/100, Loss: 0.0044, Val Loss: 0.0115
[IL] Epoch 92/100, Loss: 0.0046, Val Loss: 0.0111
[IL] Epoch 93/100, Loss: 0.0042, Val Loss: 0.0084
[IL] Epoch 94/100, Loss: 0.0044, Val Loss: 0.0083
[IL] Epoch 95/100, Loss: 0.0042, Val Loss: 0.0117
[IL] Epoch 96/100, Loss: 0.0043, Val Loss: 0.0130
[IL] Epoch 97/100, Loss: 0.0045, Val Loss: 0.0108
[IL] Epoch 98/100, Loss: 0.0043, Val Loss: 0.0074
[IL] Epoch 99/100, Loss: 0.0043, Val Loss: 0.0092
test_mean_score: 0.55
[IL] Eval - Success Rate: 0.550
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_05.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_5.ckpt

================================================================================
               OFFLINE RL ITERATION 7/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 6)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 23546 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-79.72528 | val=0.00021 | no-improve=0/5
[TransitionModel] Epoch   16 | train=-86.66191 | val=0.00018 | no-improve=5/5
[TransitionModel] Training complete. Elites=[0, 4, 5, 3, 6], val_loss=0.00016

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 6)
[IQL] Epoch 0/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 1/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 2/40, V Loss: 0.0003, Q Loss: 0.0050
[IQL] Epoch 3/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 4/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 5/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 6/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 7/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 8/40, V Loss: 0.0003, Q Loss: 0.0049
[IQL] Epoch 9/40, V Loss: 0.0002, Q Loss: 0.0048
[IQL] Epoch 10/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 11/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 12/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 13/40, V Loss: 0.0003, Q Loss: 0.0049
[IQL] Epoch 14/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 15/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 16/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 17/40, V Loss: 0.0002, Q Loss: 0.0048
[IQL] Epoch 18/40, V Loss: 0.0003, Q Loss: 0.0049
[IQL] Epoch 19/40, V Loss: 0.0002, Q Loss: 0.0048
[IQL] Epoch 20/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 21/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 22/40, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 23/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 24/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 25/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 26/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 27/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 28/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 29/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 30/40, V Loss: 0.0004, Q Loss: 0.0043
[IQL] Epoch 31/40, V Loss: 0.0004, Q Loss: 0.0046
[IQL] Epoch 32/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 33/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 34/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 35/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 36/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 37/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 38/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 39/40, V Loss: 0.0003, Q Loss: 0.0047
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_06.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_06.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 6)
[OPE] Behavior policy value J_old = 0.6888
[Offline RL] Epoch 0/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0387
[Offline RL] Epoch 1/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0250
[Offline RL] Epoch 2/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0231
[Offline RL] Epoch 3/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0508
[Offline RL] Epoch 4/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0662
[Offline RL] Epoch 5/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0987
[Offline RL] Epoch 6/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1537
[Offline RL] Epoch 7/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1791
[Offline RL] Epoch 8/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1934
[Offline RL] Epoch 9/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1909
[Offline RL] Epoch 10/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1539
[Offline RL] Epoch 11/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2218
[Offline RL] Epoch 12/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2694
[Offline RL] Epoch 13/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1830
[Offline RL] Epoch 14/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1662
[Offline RL] Epoch 15/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2116
[Offline RL] Epoch 16/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2615
[Offline RL] Epoch 17/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2676
[Offline RL] Epoch 18/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2326
[Offline RL] Epoch 19/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.3292
[Offline RL] Epoch 20/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2998
[Offline RL] Epoch 21/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2932
[Offline RL] Epoch 22/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2968
[Offline RL] Epoch 23/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2130
[Offline RL] Epoch 24/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1565
[Offline RL] Epoch 25/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1193
[Offline RL] Epoch 26/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1555
[Offline RL] Epoch 27/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1861
[Offline RL] Epoch 28/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1257
[Offline RL] Epoch 29/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1485
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_06.png
[OPE] Policy REJECTED: J_new=0.5701 ≤ J_old=0.6888 + δ=0.0344. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 6)
[Collect] 20 episodes, success=0.750, env_return=1107.65, rl_reward=0.75, steps=4000
[Data Collection] Success Rate: 0.750, EnvReturn: 1107.65, RLReward: 0.75, Episodes: 20, Steps: 4000
[Dataset] offline collection: keeping 15/20 successful episodes (dropped 5 failures) before merge.
[Dataset] Merged 15 episodes (3000 steps) → total 29400 steps, 147 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.0065, Val Loss: 0.0094
[IL] Epoch 1/100, Loss: 0.0058, Val Loss: 0.0067
[IL] Epoch 2/100, Loss: 0.0055, Val Loss: 0.0063
[IL] Epoch 3/100, Loss: 0.0057, Val Loss: 0.0093
[IL] Epoch 4/100, Loss: 0.0055, Val Loss: 0.0103
[IL] Epoch 5/100, Loss: 0.0056, Val Loss: 0.0097
[IL] Epoch 6/100, Loss: 0.0056, Val Loss: 0.0068
[IL] Epoch 7/100, Loss: 0.0055, Val Loss: 0.0075
[IL] Epoch 8/100, Loss: 0.0053, Val Loss: 0.0092
[IL] Epoch 9/100, Loss: 0.0052, Val Loss: 0.0100
[IL] Epoch 10/100, Loss: 0.0053, Val Loss: 0.0077
[IL] Epoch 11/100, Loss: 0.0053, Val Loss: 0.0094
[IL] Epoch 12/100, Loss: 0.0049, Val Loss: 0.0076
[IL] Epoch 13/100, Loss: 0.0053, Val Loss: 0.0100
[IL] Epoch 14/100, Loss: 0.0054, Val Loss: 0.0105
[IL] Epoch 15/100, Loss: 0.0056, Val Loss: 0.0071
[IL] Epoch 16/100, Loss: 0.0049, Val Loss: 0.0094
[IL] Epoch 17/100, Loss: 0.0049, Val Loss: 0.0073
[IL] Epoch 18/100, Loss: 0.0050, Val Loss: 0.0090
[IL] Epoch 19/100, Loss: 0.0049, Val Loss: 0.0088
[IL] Epoch 20/100, Loss: 0.0051, Val Loss: 0.0078
[IL] Epoch 21/100, Loss: 0.0050, Val Loss: 0.0087
[IL] Epoch 22/100, Loss: 0.0048, Val Loss: 0.0094
[IL] Epoch 23/100, Loss: 0.0048, Val Loss: 0.0082
[IL] Epoch 24/100, Loss: 0.0049, Val Loss: 0.0155
[IL] Epoch 25/100, Loss: 0.0050, Val Loss: 0.0138
[IL] Epoch 26/100, Loss: 0.0050, Val Loss: 0.0089
[IL] Epoch 27/100, Loss: 0.0048, Val Loss: 0.0092
[IL] Epoch 28/100, Loss: 0.0048, Val Loss: 0.0091
[IL] Epoch 29/100, Loss: 0.0052, Val Loss: 0.0071
[IL] Epoch 30/100, Loss: 0.0050, Val Loss: 0.0078
[IL] Epoch 31/100, Loss: 0.0050, Val Loss: 0.0089
[IL] Epoch 32/100, Loss: 0.0049, Val Loss: 0.0096
[IL] Epoch 33/100, Loss: 0.0049, Val Loss: 0.0100
[IL] Epoch 34/100, Loss: 0.0046, Val Loss: 0.0091
[IL] Epoch 35/100, Loss: 0.0049, Val Loss: 0.0131
[IL] Epoch 36/100, Loss: 0.0051, Val Loss: 0.0132
[IL] Epoch 37/100, Loss: 0.0049, Val Loss: 0.0105
[IL] Epoch 38/100, Loss: 0.0044, Val Loss: 0.0088
[IL] Epoch 39/100, Loss: 0.0047, Val Loss: 0.0087
[IL] Epoch 40/100, Loss: 0.0046, Val Loss: 0.0121
[IL] Epoch 41/100, Loss: 0.0050, Val Loss: 0.0082
[IL] Epoch 42/100, Loss: 0.0047, Val Loss: 0.0098
[IL] Epoch 43/100, Loss: 0.0049, Val Loss: 0.0059
[IL] Epoch 44/100, Loss: 0.0044, Val Loss: 0.0165
[IL] Epoch 45/100, Loss: 0.0047, Val Loss: 0.0066
[IL] Epoch 46/100, Loss: 0.0047, Val Loss: 0.0081
[IL] Epoch 47/100, Loss: 0.0045, Val Loss: 0.0110
[IL] Epoch 48/100, Loss: 0.0045, Val Loss: 0.0126
[IL] Epoch 49/100, Loss: 0.0044, Val Loss: 0.0105
[IL] Epoch 50/100, Loss: 0.0045, Val Loss: 0.0119
[IL] Epoch 51/100, Loss: 0.0046, Val Loss: 0.0084
[IL] Epoch 52/100, Loss: 0.0047, Val Loss: 0.0059
[IL] Epoch 53/100, Loss: 0.0048, Val Loss: 0.0091
[IL] Epoch 54/100, Loss: 0.0046, Val Loss: 0.0082
[IL] Epoch 55/100, Loss: 0.0043, Val Loss: 0.0122
[IL] Epoch 56/100, Loss: 0.0045, Val Loss: 0.0090
[IL] Epoch 57/100, Loss: 0.0048, Val Loss: 0.0117
[IL] Epoch 58/100, Loss: 0.0045, Val Loss: 0.0104
[IL] Epoch 59/100, Loss: 0.0047, Val Loss: 0.0074
[IL] Epoch 60/100, Loss: 0.0045, Val Loss: 0.0098
[IL] Epoch 61/100, Loss: 0.0044, Val Loss: 0.0097
[IL] Epoch 62/100, Loss: 0.0045, Val Loss: 0.0057
[IL] Epoch 63/100, Loss: 0.0046, Val Loss: 0.0086
[IL] Epoch 64/100, Loss: 0.0045, Val Loss: 0.0104
[IL] Epoch 65/100, Loss: 0.0042, Val Loss: 0.0111
[IL] Epoch 66/100, Loss: 0.0043, Val Loss: 0.0100
[IL] Epoch 67/100, Loss: 0.0045, Val Loss: 0.0098
[IL] Epoch 68/100, Loss: 0.0043, Val Loss: 0.0123
[IL] Epoch 69/100, Loss: 0.0045, Val Loss: 0.0107
[IL] Epoch 70/100, Loss: 0.0042, Val Loss: 0.0083
[IL] Epoch 71/100, Loss: 0.0047, Val Loss: 0.0155
[IL] Epoch 72/100, Loss: 0.0044, Val Loss: 0.0071
[IL] Epoch 73/100, Loss: 0.0040, Val Loss: 0.0098
[IL] Epoch 74/100, Loss: 0.0046, Val Loss: 0.0104
[IL] Epoch 75/100, Loss: 0.0043, Val Loss: 0.0080
[IL] Epoch 76/100, Loss: 0.0044, Val Loss: 0.0088
[IL] Epoch 77/100, Loss: 0.0042, Val Loss: 0.0113
[IL] Epoch 78/100, Loss: 0.0041, Val Loss: 0.0057
[IL] Epoch 79/100, Loss: 0.0043, Val Loss: 0.0115
[IL] Epoch 80/100, Loss: 0.0044, Val Loss: 0.0113
[IL] Epoch 81/100, Loss: 0.0046, Val Loss: 0.0098
[IL] Epoch 82/100, Loss: 0.0045, Val Loss: 0.0075
[IL] Epoch 83/100, Loss: 0.0043, Val Loss: 0.0088
[IL] Epoch 84/100, Loss: 0.0042, Val Loss: 0.0069
[IL] Epoch 85/100, Loss: 0.0045, Val Loss: 0.0113
[IL] Epoch 86/100, Loss: 0.0043, Val Loss: 0.0110
[IL] Epoch 87/100, Loss: 0.0044, Val Loss: 0.0108
[IL] Epoch 88/100, Loss: 0.0043, Val Loss: 0.0107
[IL] Epoch 89/100, Loss: 0.0042, Val Loss: 0.0150
[IL] Epoch 90/100, Loss: 0.0043, Val Loss: 0.0120
[IL] Epoch 91/100, Loss: 0.0041, Val Loss: 0.0116
[IL] Epoch 92/100, Loss: 0.0042, Val Loss: 0.0151
[IL] Epoch 93/100, Loss: 0.0045, Val Loss: 0.0144
[IL] Epoch 94/100, Loss: 0.0041, Val Loss: 0.0109
[IL] Epoch 95/100, Loss: 0.0043, Val Loss: 0.0107
[IL] Epoch 96/100, Loss: 0.0041, Val Loss: 0.0116
[IL] Epoch 97/100, Loss: 0.0041, Val Loss: 0.0061
[IL] Epoch 98/100, Loss: 0.0045, Val Loss: 0.0120
[IL] Epoch 99/100, Loss: 0.0040, Val Loss: 0.0082
test_mean_score: 0.7
[IL] Eval - Success Rate: 0.700
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_06.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_6.ckpt

================================================================================
               OFFLINE RL ITERATION 8/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 7)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 26441 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-85.06786 | val=0.00021 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-91.86670 | val=0.00018 | no-improve=3/5
[TransitionModel] Epoch   22 | train=-92.47798 | val=0.00018 | no-improve=5/5
[TransitionModel] Training complete. Elites=[1, 2, 6, 5, 0], val_loss=0.00017

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 7)
[IQL] Epoch 0/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 1/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 2/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 3/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 4/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 5/40, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 6/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 7/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 8/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 9/40, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 10/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 11/40, V Loss: 0.0003, Q Loss: 0.0049
[IQL] Epoch 12/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 13/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 14/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 15/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 16/40, V Loss: 0.0002, Q Loss: 0.0052
[IQL] Epoch 17/40, V Loss: 0.0002, Q Loss: 0.0047
[IQL] Epoch 18/40, V Loss: 0.0002, Q Loss: 0.0045
[IQL] Epoch 19/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 20/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 21/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 22/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 23/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 24/40, V Loss: 0.0003, Q Loss: 0.0049
[IQL] Epoch 25/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 26/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 27/40, V Loss: 0.0003, Q Loss: 0.0048
[IQL] Epoch 28/40, V Loss: 0.0003, Q Loss: 0.0047
[IQL] Epoch 29/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 30/40, V Loss: 0.0002, Q Loss: 0.0044
[IQL] Epoch 31/40, V Loss: 0.0002, Q Loss: 0.0047
[IQL] Epoch 32/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 33/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 34/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 35/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 36/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 37/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 38/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 39/40, V Loss: 0.0004, Q Loss: 0.0048
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_07.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_07.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 7)
[OPE] Behavior policy value J_old = 0.6898
[Offline RL] Epoch 0/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0399
[Offline RL] Epoch 1/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0337
[Offline RL] Epoch 2/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0544
[Offline RL] Epoch 3/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0395
[Offline RL] Epoch 4/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0274
[Offline RL] Epoch 5/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0370
[Offline RL] Epoch 6/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1236
[Offline RL] Epoch 7/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0818
[Offline RL] Epoch 8/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0569
[Offline RL] Epoch 9/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0800
[Offline RL] Epoch 10/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1131
[Offline RL] Epoch 11/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1275
[Offline RL] Epoch 12/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1157
[Offline RL] Epoch 13/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1288
[Offline RL] Epoch 14/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1154
[Offline RL] Epoch 15/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1365
[Offline RL] Epoch 16/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1943
[Offline RL] Epoch 17/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1734
[Offline RL] Epoch 18/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2360
[Offline RL] Epoch 19/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2676
[Offline RL] Epoch 20/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2357
[Offline RL] Epoch 21/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1571
[Offline RL] Epoch 22/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1553
[Offline RL] Epoch 23/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1740
[Offline RL] Epoch 24/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1800
[Offline RL] Epoch 25/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1632
[Offline RL] Epoch 26/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1380
[Offline RL] Epoch 27/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1331
[Offline RL] Epoch 28/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1418
[Offline RL] Epoch 29/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1636
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_07.png
[OPE] Policy ACCEPTED: J_new=0.8349 > J_old=0.6898 + δ=0.0345

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 7)
[Collect] 20 episodes, success=0.000, env_return=80.46, rl_reward=0.00, steps=4000
[Data Collection] Success Rate: 0.000, EnvReturn: 80.46, RLReward: 0.00, Episodes: 20, Steps: 4000
[Dataset] offline collection: keeping 0/20 successful episodes (dropped 20 failures) before merge.

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.4782, Val Loss: 0.0453
[IL] Epoch 1/100, Loss: 0.0369, Val Loss: 0.0205
[IL] Epoch 2/100, Loss: 0.0243, Val Loss: 0.0140
[IL] Epoch 3/100, Loss: 0.0204, Val Loss: 0.0159
[IL] Epoch 4/100, Loss: 0.0178, Val Loss: 0.0145
[IL] Epoch 5/100, Loss: 0.0162, Val Loss: 0.0145
[IL] Epoch 6/100, Loss: 0.0149, Val Loss: 0.0127
[IL] Epoch 7/100, Loss: 0.0140, Val Loss: 0.0119
[IL] Epoch 8/100, Loss: 0.0134, Val Loss: 0.0079
[IL] Epoch 9/100, Loss: 0.0125, Val Loss: 0.0101
[IL] Epoch 10/100, Loss: 0.0126, Val Loss: 0.0096
[IL] Epoch 11/100, Loss: 0.0117, Val Loss: 0.0124
[IL] Epoch 12/100, Loss: 0.0111, Val Loss: 0.0073
[IL] Epoch 13/100, Loss: 0.0110, Val Loss: 0.0127
[IL] Epoch 14/100, Loss: 0.0109, Val Loss: 0.0078
[IL] Epoch 15/100, Loss: 0.0106, Val Loss: 0.0088
[IL] Epoch 16/100, Loss: 0.0100, Val Loss: 0.0104
[IL] Epoch 17/100, Loss: 0.0099, Val Loss: 0.0102
[IL] Epoch 18/100, Loss: 0.0100, Val Loss: 0.0107
[IL] Epoch 19/100, Loss: 0.0089, Val Loss: 0.0114
[IL] Epoch 20/100, Loss: 0.0089, Val Loss: 0.0105
[IL] Epoch 21/100, Loss: 0.0091, Val Loss: 0.0100
[IL] Epoch 22/100, Loss: 0.0084, Val Loss: 0.0096
[IL] Epoch 23/100, Loss: 0.0083, Val Loss: 0.0079
[IL] Epoch 24/100, Loss: 0.0080, Val Loss: 0.0072
[IL] Epoch 25/100, Loss: 0.0080, Val Loss: 0.0104
[IL] Epoch 26/100, Loss: 0.0079, Val Loss: 0.0067
[IL] Epoch 27/100, Loss: 0.0072, Val Loss: 0.0106
[IL] Epoch 28/100, Loss: 0.0073, Val Loss: 0.0083
[IL] Epoch 29/100, Loss: 0.0073, Val Loss: 0.0102
[IL] Epoch 30/100, Loss: 0.0068, Val Loss: 0.0073
[IL] Epoch 31/100, Loss: 0.0069, Val Loss: 0.0081
[IL] Epoch 32/100, Loss: 0.0069, Val Loss: 0.0095
[IL] Epoch 33/100, Loss: 0.0064, Val Loss: 0.0097
[IL] Epoch 34/100, Loss: 0.0063, Val Loss: 0.0087
[IL] Epoch 35/100, Loss: 0.0061, Val Loss: 0.0101
[IL] Epoch 36/100, Loss: 0.0061, Val Loss: 0.0095
[IL] Epoch 37/100, Loss: 0.0060, Val Loss: 0.0078
[IL] Epoch 38/100, Loss: 0.0060, Val Loss: 0.0104
[IL] Epoch 39/100, Loss: 0.0060, Val Loss: 0.0097
[IL] Epoch 40/100, Loss: 0.0057, Val Loss: 0.0115
[IL] Epoch 41/100, Loss: 0.0056, Val Loss: 0.0073
[IL] Epoch 42/100, Loss: 0.0053, Val Loss: 0.0067
[IL] Epoch 43/100, Loss: 0.0054, Val Loss: 0.0110
[IL] Epoch 44/100, Loss: 0.0054, Val Loss: 0.0121
[IL] Epoch 45/100, Loss: 0.0054, Val Loss: 0.0106
[IL] Epoch 46/100, Loss: 0.0051, Val Loss: 0.0109
[IL] Epoch 47/100, Loss: 0.0052, Val Loss: 0.0078
[IL] Epoch 48/100, Loss: 0.0051, Val Loss: 0.0091
[IL] Epoch 49/100, Loss: 0.0049, Val Loss: 0.0063
[IL] Epoch 50/100, Loss: 0.0050, Val Loss: 0.0113
[IL] Epoch 51/100, Loss: 0.0051, Val Loss: 0.0093
[IL] Epoch 52/100, Loss: 0.0047, Val Loss: 0.0073
[IL] Epoch 53/100, Loss: 0.0047, Val Loss: 0.0112
[IL] Epoch 54/100, Loss: 0.0046, Val Loss: 0.0099
[IL] Epoch 55/100, Loss: 0.0049, Val Loss: 0.0115
[IL] Epoch 56/100, Loss: 0.0048, Val Loss: 0.0148
[IL] Epoch 57/100, Loss: 0.0046, Val Loss: 0.0072
[IL] Epoch 58/100, Loss: 0.0047, Val Loss: 0.0153
[IL] Epoch 59/100, Loss: 0.0043, Val Loss: 0.0080
[IL] Epoch 60/100, Loss: 0.0048, Val Loss: 0.0148
[IL] Epoch 61/100, Loss: 0.0046, Val Loss: 0.0124
[IL] Epoch 62/100, Loss: 0.0045, Val Loss: 0.0100
[IL] Epoch 63/100, Loss: 0.0044, Val Loss: 0.0074
[IL] Epoch 64/100, Loss: 0.0046, Val Loss: 0.0128
[IL] Epoch 65/100, Loss: 0.0046, Val Loss: 0.0081
[IL] Epoch 66/100, Loss: 0.0044, Val Loss: 0.0088
[IL] Epoch 67/100, Loss: 0.0044, Val Loss: 0.0156
[IL] Epoch 68/100, Loss: 0.0044, Val Loss: 0.0095
[IL] Epoch 69/100, Loss: 0.0046, Val Loss: 0.0087
[IL] Epoch 70/100, Loss: 0.0045, Val Loss: 0.0095
[IL] Epoch 71/100, Loss: 0.0043, Val Loss: 0.0077
[IL] Epoch 72/100, Loss: 0.0041, Val Loss: 0.0094
[IL] Epoch 73/100, Loss: 0.0045, Val Loss: 0.0133
[IL] Epoch 74/100, Loss: 0.0045, Val Loss: 0.0088
[IL] Epoch 75/100, Loss: 0.0043, Val Loss: 0.0144
[IL] Epoch 76/100, Loss: 0.0040, Val Loss: 0.0079
[IL] Epoch 77/100, Loss: 0.0039, Val Loss: 0.0111
[IL] Epoch 78/100, Loss: 0.0041, Val Loss: 0.0066
[IL] Epoch 79/100, Loss: 0.0041, Val Loss: 0.0103
[IL] Epoch 80/100, Loss: 0.0041, Val Loss: 0.0079
[IL] Epoch 81/100, Loss: 0.0039, Val Loss: 0.0097
[IL] Epoch 82/100, Loss: 0.0042, Val Loss: 0.0128
[IL] Epoch 83/100, Loss: 0.0041, Val Loss: 0.0080
[IL] Epoch 84/100, Loss: 0.0039, Val Loss: 0.0151
[IL] Epoch 85/100, Loss: 0.0043, Val Loss: 0.0106
[IL] Epoch 86/100, Loss: 0.0039, Val Loss: 0.0117
[IL] Epoch 87/100, Loss: 0.0042, Val Loss: 0.0142
[IL] Epoch 88/100, Loss: 0.0041, Val Loss: 0.0098
[IL] Epoch 89/100, Loss: 0.0042, Val Loss: 0.0101
[IL] Epoch 90/100, Loss: 0.0042, Val Loss: 0.0075
[IL] Epoch 91/100, Loss: 0.0039, Val Loss: 0.0105
[IL] Epoch 92/100, Loss: 0.0041, Val Loss: 0.0082
[IL] Epoch 93/100, Loss: 0.0040, Val Loss: 0.0135
[IL] Epoch 94/100, Loss: 0.0042, Val Loss: 0.0097
[IL] Epoch 95/100, Loss: 0.0040, Val Loss: 0.0059
[IL] Epoch 96/100, Loss: 0.0039, Val Loss: 0.0117
[IL] Epoch 97/100, Loss: 0.0036, Val Loss: 0.0098
[IL] Epoch 98/100, Loss: 0.0037, Val Loss: 0.0105
[IL] Epoch 99/100, Loss: 0.0043, Val Loss: 0.0098
test_mean_score: 0.5
[IL] Eval - Success Rate: 0.500
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_07.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_7.ckpt

================================================================================
               OFFLINE RL ITERATION 9/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 8)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 26441 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-91.52436 | val=0.00019 | no-improve=0/5
[TransitionModel] Epoch   18 | train=-97.34731 | val=0.00020 | no-improve=5/5
[TransitionModel] Training complete. Elites=[1, 2, 4, 5, 6], val_loss=0.00017

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 8)
[IQL] Epoch 0/40, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 1/40, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Epoch 2/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 3/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 4/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 5/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 6/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 7/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 8/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 9/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 10/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 11/40, V Loss: 0.0002, Q Loss: 0.0050
[IQL] Epoch 12/40, V Loss: 0.0003, Q Loss: 0.0051
[IQL] Epoch 13/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 14/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 15/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 16/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 17/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 18/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 19/40, V Loss: 0.0003, Q Loss: 0.0043
[IQL] Epoch 20/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 21/40, V Loss: 0.0003, Q Loss: 0.0043
[IQL] Epoch 22/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 23/40, V Loss: 0.0004, Q Loss: 0.0045
[IQL] Epoch 24/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 25/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 26/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 27/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 28/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 29/40, V Loss: 0.0004, Q Loss: 0.0046
[IQL] Epoch 30/40, V Loss: 0.0003, Q Loss: 0.0042
[IQL] Epoch 31/40, V Loss: 0.0003, Q Loss: 0.0043
[IQL] Epoch 32/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 33/40, V Loss: 0.0003, Q Loss: 0.0042
[IQL] Epoch 34/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 35/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 36/40, V Loss: 0.0003, Q Loss: 0.0042
[IQL] Epoch 37/40, V Loss: 0.0003, Q Loss: 0.0043
[IQL] Epoch 38/40, V Loss: 0.0003, Q Loss: 0.0041
[IQL] Epoch 39/40, V Loss: 0.0004, Q Loss: 0.0042
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_08.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_08.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 8)
[OPE] Behavior policy value J_old = 0.6417
[Offline RL] Epoch 0/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0642
[Offline RL] Epoch 1/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0409
[Offline RL] Epoch 2/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0632
[Offline RL] Epoch 3/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0707
[Offline RL] Epoch 4/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1386
[Offline RL] Epoch 5/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1315
[Offline RL] Epoch 6/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1065
[Offline RL] Epoch 7/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1241
[Offline RL] Epoch 8/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1747
[Offline RL] Epoch 9/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1934
[Offline RL] Epoch 10/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1387
[Offline RL] Epoch 11/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2409
[Offline RL] Epoch 12/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2608
[Offline RL] Epoch 13/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2721
[Offline RL] Epoch 14/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2550
[Offline RL] Epoch 15/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2659
[Offline RL] Epoch 16/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.3280
[Offline RL] Epoch 17/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.3492
[Offline RL] Epoch 18/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.4059
[Offline RL] Epoch 19/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.3931
[Offline RL] Epoch 20/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2839
[Offline RL] Epoch 21/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2398
[Offline RL] Epoch 22/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2129
[Offline RL] Epoch 23/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2197
[Offline RL] Epoch 24/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1255
[Offline RL] Epoch 25/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1497
[Offline RL] Epoch 26/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.1993
[Offline RL] Epoch 27/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.2067
[Offline RL] Epoch 28/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.1782
[Offline RL] Epoch 29/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.2299
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_08.png
[OPE] Policy REJECTED: J_new=0.6465 ≤ J_old=0.6417 + δ=0.0321. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 8)
[Collect] 20 episodes, success=0.700, env_return=1130.64, rl_reward=0.70, steps=4000
[Data Collection] Success Rate: 0.700, EnvReturn: 1130.64, RLReward: 0.70, Episodes: 20, Steps: 4000
[Dataset] offline collection: keeping 14/20 successful episodes (dropped 6 failures) before merge.
[Dataset] Merged 14 episodes (2800 steps) → total 32200 steps, 161 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.0058, Val Loss: 0.0064
[IL] Epoch 1/100, Loss: 0.0054, Val Loss: 0.0105
[IL] Epoch 2/100, Loss: 0.0052, Val Loss: 0.0083
[IL] Epoch 3/100, Loss: 0.0052, Val Loss: 0.0072
[IL] Epoch 4/100, Loss: 0.0050, Val Loss: 0.0101
[IL] Epoch 5/100, Loss: 0.0052, Val Loss: 0.0085
[IL] Epoch 6/100, Loss: 0.0050, Val Loss: 0.0068
[IL] Epoch 7/100, Loss: 0.0050, Val Loss: 0.0078
[IL] Epoch 8/100, Loss: 0.0049, Val Loss: 0.0101
[IL] Epoch 9/100, Loss: 0.0048, Val Loss: 0.0069
[IL] Epoch 10/100, Loss: 0.0046, Val Loss: 0.0086
[IL] Epoch 11/100, Loss: 0.0048, Val Loss: 0.0077
[IL] Epoch 12/100, Loss: 0.0048, Val Loss: 0.0078
[IL] Epoch 13/100, Loss: 0.0046, Val Loss: 0.0069
[IL] Epoch 14/100, Loss: 0.0046, Val Loss: 0.0091
[IL] Epoch 15/100, Loss: 0.0045, Val Loss: 0.0109
[IL] Epoch 16/100, Loss: 0.0049, Val Loss: 0.0090
[IL] Epoch 17/100, Loss: 0.0047, Val Loss: 0.0083
[IL] Epoch 18/100, Loss: 0.0045, Val Loss: 0.0081
[IL] Epoch 19/100, Loss: 0.0046, Val Loss: 0.0069
[IL] Epoch 20/100, Loss: 0.0045, Val Loss: 0.0098
[IL] Epoch 21/100, Loss: 0.0045, Val Loss: 0.0105
[IL] Epoch 22/100, Loss: 0.0044, Val Loss: 0.0077
[IL] Epoch 23/100, Loss: 0.0045, Val Loss: 0.0084
[IL] Epoch 24/100, Loss: 0.0044, Val Loss: 0.0090
[IL] Epoch 25/100, Loss: 0.0043, Val Loss: 0.0091
[IL] Epoch 26/100, Loss: 0.0045, Val Loss: 0.0122
[IL] Epoch 27/100, Loss: 0.0045, Val Loss: 0.0063
[IL] Epoch 28/100, Loss: 0.0046, Val Loss: 0.0082
[IL] Epoch 29/100, Loss: 0.0047, Val Loss: 0.0068
[IL] Epoch 30/100, Loss: 0.0046, Val Loss: 0.0074
[IL] Epoch 31/100, Loss: 0.0043, Val Loss: 0.0087
[IL] Epoch 32/100, Loss: 0.0045, Val Loss: 0.0077
[IL] Epoch 33/100, Loss: 0.0043, Val Loss: 0.0077
[IL] Epoch 34/100, Loss: 0.0045, Val Loss: 0.0097
[IL] Epoch 35/100, Loss: 0.0042, Val Loss: 0.0088
[IL] Epoch 36/100, Loss: 0.0045, Val Loss: 0.0068
[IL] Epoch 37/100, Loss: 0.0044, Val Loss: 0.0105
[IL] Epoch 38/100, Loss: 0.0042, Val Loss: 0.0107
[IL] Epoch 39/100, Loss: 0.0044, Val Loss: 0.0058
[IL] Epoch 40/100, Loss: 0.0045, Val Loss: 0.0085
[IL] Epoch 41/100, Loss: 0.0041, Val Loss: 0.0077
[IL] Epoch 42/100, Loss: 0.0045, Val Loss: 0.0070
[IL] Epoch 43/100, Loss: 0.0042, Val Loss: 0.0093
[IL] Epoch 44/100, Loss: 0.0041, Val Loss: 0.0145
[IL] Epoch 45/100, Loss: 0.0042, Val Loss: 0.0112
[IL] Epoch 46/100, Loss: 0.0043, Val Loss: 0.0109
[IL] Epoch 47/100, Loss: 0.0040, Val Loss: 0.0068
[IL] Epoch 48/100, Loss: 0.0043, Val Loss: 0.0093
[IL] Epoch 49/100, Loss: 0.0043, Val Loss: 0.0125
[IL] Epoch 50/100, Loss: 0.0044, Val Loss: 0.0083
[IL] Epoch 51/100, Loss: 0.0041, Val Loss: 0.0073
[IL] Epoch 52/100, Loss: 0.0042, Val Loss: 0.0091
[IL] Epoch 53/100, Loss: 0.0042, Val Loss: 0.0088
[IL] Epoch 54/100, Loss: 0.0041, Val Loss: 0.0075
[IL] Epoch 55/100, Loss: 0.0040, Val Loss: 0.0071
[IL] Epoch 56/100, Loss: 0.0041, Val Loss: 0.0141
[IL] Epoch 57/100, Loss: 0.0043, Val Loss: 0.0109
[IL] Epoch 58/100, Loss: 0.0042, Val Loss: 0.0109
[IL] Epoch 59/100, Loss: 0.0039, Val Loss: 0.0081
[IL] Epoch 60/100, Loss: 0.0043, Val Loss: 0.0103
[IL] Epoch 61/100, Loss: 0.0041, Val Loss: 0.0101
[IL] Epoch 62/100, Loss: 0.0040, Val Loss: 0.0087
[IL] Epoch 63/100, Loss: 0.0047, Val Loss: 0.0133
[IL] Epoch 64/100, Loss: 0.0041, Val Loss: 0.0091
[IL] Epoch 65/100, Loss: 0.0043, Val Loss: 0.0098
[IL] Epoch 66/100, Loss: 0.0039, Val Loss: 0.0083
[IL] Epoch 67/100, Loss: 0.0041, Val Loss: 0.0135
[IL] Epoch 68/100, Loss: 0.0040, Val Loss: 0.0126
[IL] Epoch 69/100, Loss: 0.0038, Val Loss: 0.0092
[IL] Epoch 70/100, Loss: 0.0040, Val Loss: 0.0112
[IL] Epoch 71/100, Loss: 0.0041, Val Loss: 0.0103
[IL] Epoch 72/100, Loss: 0.0040, Val Loss: 0.0100
[IL] Epoch 73/100, Loss: 0.0042, Val Loss: 0.0134
[IL] Epoch 74/100, Loss: 0.0040, Val Loss: 0.0077
[IL] Epoch 75/100, Loss: 0.0041, Val Loss: 0.0106
[IL] Epoch 76/100, Loss: 0.0040, Val Loss: 0.0116
[IL] Epoch 77/100, Loss: 0.0040, Val Loss: 0.0118
[IL] Epoch 78/100, Loss: 0.0039, Val Loss: 0.0104
[IL] Epoch 79/100, Loss: 0.0042, Val Loss: 0.0104
[IL] Epoch 80/100, Loss: 0.0038, Val Loss: 0.0126
[IL] Epoch 81/100, Loss: 0.0039, Val Loss: 0.0077
[IL] Epoch 82/100, Loss: 0.0038, Val Loss: 0.0085
[IL] Epoch 83/100, Loss: 0.0039, Val Loss: 0.0085
[IL] Epoch 84/100, Loss: 0.0040, Val Loss: 0.0088
[IL] Epoch 85/100, Loss: 0.0039, Val Loss: 0.0133
[IL] Epoch 86/100, Loss: 0.0038, Val Loss: 0.0137
[IL] Epoch 87/100, Loss: 0.0040, Val Loss: 0.0068
[IL] Epoch 88/100, Loss: 0.0039, Val Loss: 0.0094
[IL] Epoch 89/100, Loss: 0.0041, Val Loss: 0.0109
[IL] Epoch 90/100, Loss: 0.0041, Val Loss: 0.0155
[IL] Epoch 91/100, Loss: 0.0040, Val Loss: 0.0142
[IL] Epoch 92/100, Loss: 0.0037, Val Loss: 0.0095
[IL] Epoch 93/100, Loss: 0.0037, Val Loss: 0.0075
[IL] Epoch 94/100, Loss: 0.0038, Val Loss: 0.0106
[IL] Epoch 95/100, Loss: 0.0037, Val Loss: 0.0098
[IL] Epoch 96/100, Loss: 0.0038, Val Loss: 0.0064
[IL] Epoch 97/100, Loss: 0.0039, Val Loss: 0.0127
[IL] Epoch 98/100, Loss: 0.0038, Val Loss: 0.0111
[IL] Epoch 99/100, Loss: 0.0038, Val Loss: 0.0122
test_mean_score: 0.75
[IL] Eval - Success Rate: 0.750
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_08.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_8.ckpt

================================================================================
               OFFLINE RL ITERATION 10/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 9)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 29143 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-95.39620 | val=0.00023 | no-improve=0/5
[TransitionModel] Epoch   16 | train=-101.61488 | val=0.00022 | no-improve=5/5
[TransitionModel] Training complete. Elites=[2, 1, 6, 0, 5], val_loss=0.00020

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 9)
[IQL] Epoch 0/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 1/40, V Loss: 0.0003, Q Loss: 0.0043
[IQL] Epoch 2/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 3/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 4/40, V Loss: 0.0004, Q Loss: 0.0045
[IQL] Epoch 5/40, V Loss: 0.0003, Q Loss: 0.0042
[IQL] Epoch 6/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 7/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 8/40, V Loss: 0.0003, Q Loss: 0.0043
[IQL] Epoch 9/40, V Loss: 0.0003, Q Loss: 0.0042
[IQL] Epoch 10/40, V Loss: 0.0003, Q Loss: 0.0045
[IQL] Epoch 11/40, V Loss: 0.0003, Q Loss: 0.0043
[IQL] Epoch 12/40, V Loss: 0.0004, Q Loss: 0.0041
[IQL] Epoch 13/40, V Loss: 0.0004, Q Loss: 0.0042
[IQL] Epoch 14/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 15/40, V Loss: 0.0003, Q Loss: 0.0042
[IQL] Epoch 16/40, V Loss: 0.0003, Q Loss: 0.0043
[IQL] Epoch 17/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 18/40, V Loss: 0.0003, Q Loss: 0.0042
[IQL] Epoch 19/40, V Loss: 0.0003, Q Loss: 0.0042
[IQL] Epoch 20/40, V Loss: 0.0003, Q Loss: 0.0042
[IQL] Epoch 21/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 22/40, V Loss: 0.0003, Q Loss: 0.0041
[IQL] Epoch 23/40, V Loss: 0.0003, Q Loss: 0.0040
[IQL] Epoch 24/40, V Loss: 0.0003, Q Loss: 0.0043
[IQL] Epoch 25/40, V Loss: 0.0003, Q Loss: 0.0043
[IQL] Epoch 26/40, V Loss: 0.0003, Q Loss: 0.0042
[IQL] Epoch 27/40, V Loss: 0.0003, Q Loss: 0.0042
[IQL] Epoch 28/40, V Loss: 0.0003, Q Loss: 0.0041
[IQL] Epoch 29/40, V Loss: 0.0003, Q Loss: 0.0044
[IQL] Epoch 30/40, V Loss: 0.0003, Q Loss: 0.0041
[IQL] Epoch 31/40, V Loss: 0.0003, Q Loss: 0.0041
[IQL] Epoch 32/40, V Loss: 0.0003, Q Loss: 0.0041
[IQL] Epoch 33/40, V Loss: 0.0004, Q Loss: 0.0040
[IQL] Epoch 34/40, V Loss: 0.0003, Q Loss: 0.0040
[IQL] Epoch 35/40, V Loss: 0.0003, Q Loss: 0.0046
[IQL] Epoch 36/40, V Loss: 0.0002, Q Loss: 0.0042
[IQL] Epoch 37/40, V Loss: 0.0003, Q Loss: 0.0041
[IQL] Epoch 38/40, V Loss: 0.0003, Q Loss: 0.0040
[IQL] Epoch 39/40, V Loss: 0.0003, Q Loss: 0.0039
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_09.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_09.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 9)
[OPE] Behavior policy value J_old = 0.6885
[Offline RL] Epoch 0/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0620
[Offline RL] Epoch 1/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0206
[Offline RL] Epoch 2/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0215
[Offline RL] Epoch 3/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0245
[Offline RL] Epoch 4/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0319
[Offline RL] Epoch 5/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0242
[Offline RL] Epoch 6/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0249
[Offline RL] Epoch 7/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0206
[Offline RL] Epoch 8/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0168
[Offline RL] Epoch 9/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0193
[Offline RL] Epoch 10/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0442
[Offline RL] Epoch 11/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0331
[Offline RL] Epoch 12/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0201
[Offline RL] Epoch 13/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0173
[Offline RL] Epoch 14/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0168
[Offline RL] Epoch 15/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0188
[Offline RL] Epoch 16/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0270
[Offline RL] Epoch 17/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0444
[Offline RL] Epoch 18/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0444
[Offline RL] Epoch 19/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0370
[Offline RL] Epoch 20/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0369
[Offline RL] Epoch 21/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0354
[Offline RL] Epoch 22/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0402
[Offline RL] Epoch 23/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0408
[Offline RL] Epoch 24/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0521
[Offline RL] Epoch 25/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0569
[Offline RL] Epoch 26/30, PPO Loss: 0.0000, Reg Loss: 0.0000, CD Loss: 0.0804
[Offline RL] Epoch 27/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0742
[Offline RL] Epoch 28/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0663
[Offline RL] Epoch 29/30, PPO Loss: -0.0000, Reg Loss: 0.0000, CD Loss: 0.0766
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_09.png
[OPE] Policy REJECTED: J_new=0.6370 ≤ J_old=0.6885 + δ=0.0344. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.700, env_return=1020.76, rl_reward=0.70, steps=4000
[Data Collection] Success Rate: 0.700, EnvReturn: 1020.76, RLReward: 0.70, Episodes: 20, Steps: 4000
[Dataset] offline collection: keeping 14/20 successful episodes (dropped 6 failures) before merge.
[Dataset] Merged 14 episodes (2800 steps) → total 35000 steps, 175 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[IL] Epoch 0/100, Loss: 0.0056, Val Loss: 0.0051
[IL] Epoch 1/100, Loss: 0.0049, Val Loss: 0.0110
[IL] Epoch 2/100, Loss: 0.0048, Val Loss: 0.0080
[IL] Epoch 3/100, Loss: 0.0046, Val Loss: 0.0087
[IL] Epoch 4/100, Loss: 0.0048, Val Loss: 0.0080
[IL] Epoch 5/100, Loss: 0.0046, Val Loss: 0.0096
[IL] Epoch 6/100, Loss: 0.0045, Val Loss: 0.0107
[IL] Epoch 7/100, Loss: 0.0043, Val Loss: 0.0052
[IL] Epoch 8/100, Loss: 0.0045, Val Loss: 0.0071
[IL] Epoch 9/100, Loss: 0.0047, Val Loss: 0.0110
[IL] Epoch 10/100, Loss: 0.0045, Val Loss: 0.0098
[IL] Epoch 11/100, Loss: 0.0045, Val Loss: 0.0142
[IL] Epoch 12/100, Loss: 0.0047, Val Loss: 0.0092
[IL] Epoch 13/100, Loss: 0.0042, Val Loss: 0.0080
[IL] Epoch 14/100, Loss: 0.0044, Val Loss: 0.0106
[IL] Epoch 15/100, Loss: 0.0042, Val Loss: 0.0096
[IL] Epoch 16/100, Loss: 0.0041, Val Loss: 0.0114
[IL] Epoch 17/100, Loss: 0.0041, Val Loss: 0.0118
[IL] Epoch 18/100, Loss: 0.0040, Val Loss: 0.0120
[IL] Epoch 19/100, Loss: 0.0039, Val Loss: 0.0151
[IL] Epoch 20/100, Loss: 0.0043, Val Loss: 0.0067
[IL] Epoch 21/100, Loss: 0.0041, Val Loss: 0.0102
[IL] Epoch 22/100, Loss: 0.0041, Val Loss: 0.0089
[IL] Epoch 23/100, Loss: 0.0038, Val Loss: 0.0080
[IL] Epoch 24/100, Loss: 0.0042, Val Loss: 0.0075
[IL] Epoch 25/100, Loss: 0.0040, Val Loss: 0.0101
[IL] Epoch 26/100, Loss: 0.0039, Val Loss: 0.0089
[IL] Epoch 27/100, Loss: 0.0040, Val Loss: 0.0109
[IL] Epoch 28/100, Loss: 0.0041, Val Loss: 0.0108
[IL] Epoch 29/100, Loss: 0.0039, Val Loss: 0.0147
[IL] Epoch 30/100, Loss: 0.0041, Val Loss: 0.0133
[IL] Epoch 31/100, Loss: 0.0039, Val Loss: 0.0073
[IL] Epoch 32/100, Loss: 0.0039, Val Loss: 0.0120
[IL] Epoch 33/100, Loss: 0.0039, Val Loss: 0.0075
[IL] Epoch 34/100, Loss: 0.0038, Val Loss: 0.0081
[IL] Epoch 35/100, Loss: 0.0039, Val Loss: 0.0120
[IL] Epoch 36/100, Loss: 0.0037, Val Loss: 0.0107
[IL] Epoch 37/100, Loss: 0.0040, Val Loss: 0.0101
[IL] Epoch 38/100, Loss: 0.0039, Val Loss: 0.0085
[IL] Epoch 39/100, Loss: 0.0040, Val Loss: 0.0089
[IL] Epoch 40/100, Loss: 0.0040, Val Loss: 0.0105
[IL] Epoch 41/100, Loss: 0.0037, Val Loss: 0.0104
[IL] Epoch 42/100, Loss: 0.0040, Val Loss: 0.0110
[IL] Epoch 43/100, Loss: 0.0039, Val Loss: 0.0128
[IL] Epoch 44/100, Loss: 0.0040, Val Loss: 0.0095
[IL] Epoch 45/100, Loss: 0.0040, Val Loss: 0.0147
[IL] Epoch 46/100, Loss: 0.0039, Val Loss: 0.0094
[IL] Epoch 47/100, Loss: 0.0037, Val Loss: 0.0116
[IL] Epoch 48/100, Loss: 0.0040, Val Loss: 0.0113
[IL] Epoch 49/100, Loss: 0.0038, Val Loss: 0.0121
[IL] Epoch 50/100, Loss: 0.0037, Val Loss: 0.0127
[IL] Epoch 51/100, Loss: 0.0040, Val Loss: 0.0086
[IL] Epoch 52/100, Loss: 0.0036, Val Loss: 0.0108
[IL] Epoch 53/100, Loss: 0.0036, Val Loss: 0.0107
[IL] Epoch 54/100, Loss: 0.0036, Val Loss: 0.0087
[IL] Epoch 55/100, Loss: 0.0036, Val Loss: 0.0114
[IL] Epoch 56/100, Loss: 0.0039, Val Loss: 0.0096
[IL] Epoch 57/100, Loss: 0.0037, Val Loss: 0.0159
[IL] Epoch 58/100, Loss: 0.0037, Val Loss: 0.0155
[IL] Epoch 59/100, Loss: 0.0038, Val Loss: 0.0091
[IL] Epoch 60/100, Loss: 0.0037, Val Loss: 0.0095
[IL] Epoch 61/100, Loss: 0.0040, Val Loss: 0.0100
[IL] Epoch 62/100, Loss: 0.0039, Val Loss: 0.0092
[IL] Epoch 63/100, Loss: 0.0037, Val Loss: 0.0083
[IL] Epoch 64/100, Loss: 0.0035, Val Loss: 0.0129
[IL] Epoch 65/100, Loss: 0.0037, Val Loss: 0.0102
[IL] Epoch 66/100, Loss: 0.0038, Val Loss: 0.0108
[IL] Epoch 67/100, Loss: 0.0037, Val Loss: 0.0084
[IL] Epoch 68/100, Loss: 0.0036, Val Loss: 0.0089
[IL] Epoch 69/100, Loss: 0.0035, Val Loss: 0.0091
[IL] Epoch 70/100, Loss: 0.0036, Val Loss: 0.0105
[IL] Epoch 71/100, Loss: 0.0037, Val Loss: 0.0149
[IL] Epoch 72/100, Loss: 0.0037, Val Loss: 0.0146
[IL] Epoch 73/100, Loss: 0.0038, Val Loss: 0.0166
[IL] Epoch 74/100, Loss: 0.0037, Val Loss: 0.0108
[IL] Epoch 75/100, Loss: 0.0033, Val Loss: 0.0127
[IL] Epoch 76/100, Loss: 0.0037, Val Loss: 0.0128
[IL] Epoch 77/100, Loss: 0.0035, Val Loss: 0.0093
[IL] Epoch 78/100, Loss: 0.0036, Val Loss: 0.0146
[IL] Epoch 79/100, Loss: 0.0036, Val Loss: 0.0093
[IL] Epoch 80/100, Loss: 0.0038, Val Loss: 0.0135
[IL] Epoch 81/100, Loss: 0.0037, Val Loss: 0.0087
[IL] Epoch 82/100, Loss: 0.0037, Val Loss: 0.0109
[IL] Epoch 83/100, Loss: 0.0037, Val Loss: 0.0125
[IL] Epoch 84/100, Loss: 0.0034, Val Loss: 0.0168
[IL] Epoch 85/100, Loss: 0.0037, Val Loss: 0.0056
[IL] Epoch 86/100, Loss: 0.0037, Val Loss: 0.0085
[IL] Epoch 87/100, Loss: 0.0037, Val Loss: 0.0119
[IL] Epoch 88/100, Loss: 0.0037, Val Loss: 0.0113
[IL] Epoch 89/100, Loss: 0.0034, Val Loss: 0.0079
[IL] Epoch 90/100, Loss: 0.0035, Val Loss: 0.0125
[IL] Epoch 91/100, Loss: 0.0039, Val Loss: 0.0081
[IL] Epoch 92/100, Loss: 0.0035, Val Loss: 0.0135
[IL] Epoch 93/100, Loss: 0.0035, Val Loss: 0.0107
[IL] Epoch 94/100, Loss: 0.0035, Val Loss: 0.0113
[IL] Epoch 95/100, Loss: 0.0037, Val Loss: 0.0097
[IL] Epoch 96/100, Loss: 0.0037, Val Loss: 0.0104
[IL] Epoch 97/100, Loss: 0.0035, Val Loss: 0.0154
[IL] Epoch 98/100, Loss: 0.0037, Val Loss: 0.0110
[IL] Epoch 99/100, Loss: 0.0038, Val Loss: 0.0067
test_mean_score: 0.8
[IL] Eval - Success Rate: 0.800
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_09.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_9.ckpt

================================================================================
                    PHASE 3: ONLINE RL FINE-TUNING
================================================================================


[Online RL] Iteration 1/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.750, env_return=1133.42, rl_reward=0.75, steps=4000
[Data Collection] Success Rate: 0.750, EnvReturn: 1133.42, RLReward: 0.75, Episodes: 20, Steps: 4000
[Dataset] online collection: keeping 15/20 successful episodes (dropped 5 failures) before merge.
[Online RL] Epoch 1/20, PPO Loss: 0.0198, Reg Loss: 0.0000, CD Loss: 0.2292
[Online RL] Epoch 2/20, PPO Loss: -0.0186, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/20, PPO Loss: -0.0109, Reg Loss: 0.0000, CD Loss: 0.1609
[Online RL] Epoch 4/20, PPO Loss: -0.0174, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 5/20, PPO Loss: -0.0243, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 6/20, PPO Loss: -0.0825, Reg Loss: 0.0000, CD Loss: 0.1074
[Online RL] Epoch 7/20, PPO Loss: -0.0539, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 8/20, PPO Loss: -0.0693, Reg Loss: 0.0000, CD Loss: 0.0720
[Online RL] Epoch 9/20, PPO Loss: -0.0608, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 10/20, PPO Loss: -0.0933, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 11/20, PPO Loss: -0.0986, Reg Loss: 0.0000, CD Loss: 0.0669
[Online RL] Epoch 12/20, PPO Loss: -0.1291, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 13/20, PPO Loss: -0.0973, Reg Loss: 0.0000, CD Loss: 0.0714
[Online RL] Epoch 14/20, PPO Loss: -0.0999, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 15/20, PPO Loss: -0.1244, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 16/20, PPO Loss: -0.1322, Reg Loss: 0.0000, CD Loss: 0.0748
[Online RL] Epoch 17/20, PPO Loss: -0.1105, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 18/20, PPO Loss: -0.0963, Reg Loss: 0.0000, CD Loss: 0.0650
[Online RL] Epoch 19/20, PPO Loss: -0.1191, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 20/20, PPO Loss: -0.0941, Reg Loss: 0.0000, CD Loss: 0.0000
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_0.ckpt

[Online RL] Iteration 2/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.650, env_return=1120.59, rl_reward=0.65, steps=4000
[Data Collection] Success Rate: 0.650, EnvReturn: 1120.59, RLReward: 0.65, Episodes: 20, Steps: 4000
[Dataset] online collection: keeping 13/20 successful episodes (dropped 7 failures) before merge.
[Online RL] Epoch 1/20, PPO Loss: 0.0080, Reg Loss: 0.0000, CD Loss: 0.0565
[Online RL] Epoch 2/20, PPO Loss: 0.0535, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/20, PPO Loss: -0.0975, Reg Loss: 0.0000, CD Loss: 0.0398
[Online RL] Epoch 4/20, PPO Loss: -0.0781, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 5/20, PPO Loss: -0.0256, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 6/20, PPO Loss: -0.1175, Reg Loss: 0.0000, CD Loss: 0.0424
[Online RL] Epoch 7/20, PPO Loss: -0.1018, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 8/20, PPO Loss: -0.1742, Reg Loss: 0.0000, CD Loss: 0.0278
[Online RL] Epoch 9/20, PPO Loss: -0.1046, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 10/20, PPO Loss: -0.1707, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 11/20, PPO Loss: -0.1380, Reg Loss: 0.0000, CD Loss: 0.0338
[Online RL] Epoch 12/20, PPO Loss: -0.1841, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 13/20, PPO Loss: -0.0922, Reg Loss: 0.0000, CD Loss: 0.0342
[Online RL] Epoch 14/20, PPO Loss: -0.1320, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 15/20, PPO Loss: -0.1023, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 16/20, PPO Loss: -0.1266, Reg Loss: 0.0000, CD Loss: 0.0382
[Online RL] Epoch 17/20, PPO Loss: -0.2079, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 18/20, PPO Loss: -0.2061, Reg Loss: 0.0000, CD Loss: 0.0377
[Online RL] Epoch 19/20, PPO Loss: -0.0992, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 20/20, PPO Loss: -0.1329, Reg Loss: 0.0000, CD Loss: 0.0000
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_1.ckpt

[Online RL] Iteration 3/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.750, env_return=1034.69, rl_reward=0.75, steps=4000
[Data Collection] Success Rate: 0.750, EnvReturn: 1034.69, RLReward: 0.75, Episodes: 20, Steps: 4000
[Dataset] online collection: keeping 15/20 successful episodes (dropped 5 failures) before merge.
[Online RL] Epoch 1/20, PPO Loss: 0.0071, Reg Loss: 0.0000, CD Loss: 0.0335
[Online RL] Epoch 2/20, PPO Loss: 0.0158, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/20, PPO Loss: -0.0113, Reg Loss: 0.0000, CD Loss: 0.0294
[Online RL] Epoch 4/20, PPO Loss: -0.0573, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 5/20, PPO Loss: -0.0685, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 6/20, PPO Loss: -0.0477, Reg Loss: 0.0000, CD Loss: 0.0228
[Online RL] Epoch 7/20, PPO Loss: -0.1070, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 8/20, PPO Loss: -0.0978, Reg Loss: 0.0000, CD Loss: 0.0217
[Online RL] Epoch 9/20, PPO Loss: -0.1306, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 10/20, PPO Loss: -0.1046, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 11/20, PPO Loss: -0.1406, Reg Loss: 0.0000, CD Loss: 0.0205
[Online RL] Epoch 12/20, PPO Loss: -0.1347, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 13/20, PPO Loss: -0.0895, Reg Loss: 0.0000, CD Loss: 0.0229
[Online RL] Epoch 14/20, PPO Loss: -0.0949, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 15/20, PPO Loss: -0.1131, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 16/20, PPO Loss: -0.1082, Reg Loss: 0.0000, CD Loss: 0.0220
[Online RL] Epoch 17/20, PPO Loss: -0.1194, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 18/20, PPO Loss: -0.0994, Reg Loss: 0.0000, CD Loss: 0.0217
[Online RL] Epoch 19/20, PPO Loss: -0.0715, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 20/20, PPO Loss: -0.0995, Reg Loss: 0.0000, CD Loss: 0.0000
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_2.ckpt

[Online RL] Iteration 4/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.650, env_return=1122.46, rl_reward=0.65, steps=4000
[Data Collection] Success Rate: 0.650, EnvReturn: 1122.46, RLReward: 0.65, Episodes: 20, Steps: 4000
[Dataset] online collection: keeping 13/20 successful episodes (dropped 7 failures) before merge.
[Online RL] Epoch 1/20, PPO Loss: -0.0263, Reg Loss: 0.0000, CD Loss: 0.0244
[Online RL] Epoch 2/20, PPO Loss: 0.0111, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/20, PPO Loss: -0.1520, Reg Loss: 0.0000, CD Loss: 0.0256
[Online RL] Epoch 4/20, PPO Loss: -0.0308, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 5/20, PPO Loss: -0.0615, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 6/20, PPO Loss: -0.1000, Reg Loss: 0.0000, CD Loss: 0.0199
[Online RL] Epoch 7/20, PPO Loss: -0.0713, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 8/20, PPO Loss: -0.0823, Reg Loss: 0.0000, CD Loss: 0.0190
[Online RL] Epoch 9/20, PPO Loss: -0.1414, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 10/20, PPO Loss: -0.0754, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 11/20, PPO Loss: -0.0928, Reg Loss: 0.0000, CD Loss: 0.0199
[Online RL] Epoch 12/20, PPO Loss: -0.1257, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 13/20, PPO Loss: -0.0234, Reg Loss: 0.0000, CD Loss: 0.0192
[Online RL] Epoch 14/20, PPO Loss: -0.0987, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 15/20, PPO Loss: -0.0996, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 16/20, PPO Loss: -0.1152, Reg Loss: 0.0000, CD Loss: 0.0203
[Online RL] Epoch 17/20, PPO Loss: -0.0759, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 18/20, PPO Loss: -0.0984, Reg Loss: 0.0000, CD Loss: 0.0192
[Online RL] Epoch 19/20, PPO Loss: -0.1940, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 20/20, PPO Loss: -0.1463, Reg Loss: 0.0000, CD Loss: 0.0000
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_3.ckpt

[Online RL] Iteration 5/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.700, env_return=1023.02, rl_reward=0.70, steps=4000
[Data Collection] Success Rate: 0.700, EnvReturn: 1023.02, RLReward: 0.70, Episodes: 20, Steps: 4000
[Dataset] online collection: keeping 14/20 successful episodes (dropped 6 failures) before merge.
[Online RL] Epoch 1/20, PPO Loss: 0.0159, Reg Loss: 0.0000, CD Loss: 0.0180
[Online RL] Epoch 2/20, PPO Loss: -0.0665, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/20, PPO Loss: -0.0556, Reg Loss: 0.0000, CD Loss: 0.0158
[Online RL] Epoch 4/20, PPO Loss: -0.0476, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 5/20, PPO Loss: -0.0675, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 6/20, PPO Loss: -0.0750, Reg Loss: 0.0000, CD Loss: 0.0164
[Online RL] Epoch 7/20, PPO Loss: -0.1173, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 8/20, PPO Loss: -0.1216, Reg Loss: 0.0000, CD Loss: 0.0127
[Online RL] Epoch 9/20, PPO Loss: -0.0998, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 10/20, PPO Loss: -0.0601, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 11/20, PPO Loss: -0.0979, Reg Loss: 0.0000, CD Loss: 0.0158
[Online RL] Epoch 12/20, PPO Loss: -0.1388, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 13/20, PPO Loss: -0.1073, Reg Loss: 0.0000, CD Loss: 0.0168
[Online RL] Epoch 14/20, PPO Loss: -0.1510, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 15/20, PPO Loss: -0.1455, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 16/20, PPO Loss: -0.1281, Reg Loss: 0.0000, CD Loss: 0.0177
[Online RL] Epoch 17/20, PPO Loss: -0.1942, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 18/20, PPO Loss: -0.1133, Reg Loss: 0.0000, CD Loss: 0.0144
[Online RL] Epoch 19/20, PPO Loss: -0.0585, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 20/20, PPO Loss: -0.1188, Reg Loss: 0.0000, CD Loss: 0.0000
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_4.ckpt

[Online RL] Iteration 6/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.800, env_return=998.68, rl_reward=0.80, steps=4000
[Data Collection] Success Rate: 0.800, EnvReturn: 998.68, RLReward: 0.80, Episodes: 20, Steps: 4000
[Dataset] online collection: keeping 16/20 successful episodes (dropped 4 failures) before merge.
[Online RL] Epoch 1/20, PPO Loss: -0.0029, Reg Loss: 0.0000, CD Loss: 0.0161
[Online RL] Epoch 2/20, PPO Loss: 0.0175, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/20, PPO Loss: -0.0382, Reg Loss: 0.0000, CD Loss: 0.0160
[Online RL] Epoch 4/20, PPO Loss: -0.0814, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 5/20, PPO Loss: -0.0900, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 6/20, PPO Loss: -0.1084, Reg Loss: 0.0000, CD Loss: 0.0149
[Online RL] Epoch 7/20, PPO Loss: -0.1009, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 8/20, PPO Loss: -0.1240, Reg Loss: 0.0000, CD Loss: 0.0150
[Online RL] Epoch 9/20, PPO Loss: -0.1231, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 10/20, PPO Loss: -0.0997, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 11/20, PPO Loss: -0.1218, Reg Loss: 0.0000, CD Loss: 0.0136
[Online RL] Epoch 12/20, PPO Loss: -0.1260, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 13/20, PPO Loss: -0.1151, Reg Loss: 0.0000, CD Loss: 0.0138
[Online RL] Epoch 14/20, PPO Loss: -0.1077, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 15/20, PPO Loss: -0.1507, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 16/20, PPO Loss: -0.1124, Reg Loss: 0.0000, CD Loss: 0.0137
[Online RL] Epoch 17/20, PPO Loss: -0.1207, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 18/20, PPO Loss: -0.1217, Reg Loss: 0.0000, CD Loss: 0.0143
[Online RL] Epoch 19/20, PPO Loss: -0.1127, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 20/20, PPO Loss: -0.1024, Reg Loss: 0.0000, CD Loss: 0.0000
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_5.ckpt

[Online RL] Iteration 7/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.750, env_return=951.26, rl_reward=0.75, steps=4000
[Data Collection] Success Rate: 0.750, EnvReturn: 951.26, RLReward: 0.75, Episodes: 20, Steps: 4000
[Dataset] online collection: keeping 15/20 successful episodes (dropped 5 failures) before merge.
[Online RL] Epoch 1/20, PPO Loss: -0.0006, Reg Loss: 0.0000, CD Loss: 0.0141
[Online RL] Epoch 2/20, PPO Loss: -0.0023, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/20, PPO Loss: -0.0450, Reg Loss: 0.0000, CD Loss: 0.0139
[Online RL] Epoch 4/20, PPO Loss: -0.0272, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 5/20, PPO Loss: -0.0819, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 6/20, PPO Loss: -0.1022, Reg Loss: 0.0000, CD Loss: 0.0136
[Online RL] Epoch 7/20, PPO Loss: -0.1066, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 8/20, PPO Loss: -0.1481, Reg Loss: 0.0000, CD Loss: 0.0119
[Online RL] Epoch 9/20, PPO Loss: -0.0967, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 10/20, PPO Loss: -0.1517, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 11/20, PPO Loss: -0.1240, Reg Loss: 0.0000, CD Loss: 0.0119
[Online RL] Epoch 12/20, PPO Loss: -0.1065, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 13/20, PPO Loss: -0.0895, Reg Loss: 0.0000, CD Loss: 0.0129
[Online RL] Epoch 14/20, PPO Loss: -0.0897, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 15/20, PPO Loss: -0.0982, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 16/20, PPO Loss: -0.1437, Reg Loss: 0.0000, CD Loss: 0.0121
[Online RL] Epoch 17/20, PPO Loss: -0.1355, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 18/20, PPO Loss: -0.1201, Reg Loss: 0.0000, CD Loss: 0.0126
[Online RL] Epoch 19/20, PPO Loss: -0.1436, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 20/20, PPO Loss: -0.1315, Reg Loss: 0.0000, CD Loss: 0.0000
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_6.ckpt

[Online RL] Iteration 8/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.550, env_return=954.69, rl_reward=0.55, steps=4000
[Data Collection] Success Rate: 0.550, EnvReturn: 954.69, RLReward: 0.55, Episodes: 20, Steps: 4000
[Dataset] online collection: keeping 11/20 successful episodes (dropped 9 failures) before merge.
[Online RL] Epoch 1/20, PPO Loss: 0.0903, Reg Loss: 0.0000, CD Loss: 0.0125
[Online RL] Epoch 2/20, PPO Loss: 0.1681, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/20, PPO Loss: -0.0123, Reg Loss: 0.0000, CD Loss: 0.0150
[Online RL] Epoch 4/20, PPO Loss: -0.0201, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 5/20, PPO Loss: 0.0337, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 6/20, PPO Loss: -0.1172, Reg Loss: 0.0000, CD Loss: 0.0117
[Online RL] Epoch 7/20, PPO Loss: -0.1191, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 8/20, PPO Loss: -0.1181, Reg Loss: 0.0000, CD Loss: 0.0102
[Online RL] Epoch 9/20, PPO Loss: -0.1264, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 10/20, PPO Loss: -0.0894, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 11/20, PPO Loss: -0.2350, Reg Loss: 0.0000, CD Loss: 0.0117
[Online RL] Epoch 12/20, PPO Loss: -0.0739, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 13/20, PPO Loss: -0.0791, Reg Loss: 0.0000, CD Loss: 0.0148
[Online RL] Epoch 14/20, PPO Loss: -0.0700, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 15/20, PPO Loss: 0.0125, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 16/20, PPO Loss: -0.1351, Reg Loss: 0.0000, CD Loss: 0.0118
[Online RL] Epoch 17/20, PPO Loss: -0.0536, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 18/20, PPO Loss: -0.0465, Reg Loss: 0.0000, CD Loss: 0.0157
[Online RL] Epoch 19/20, PPO Loss: -0.1119, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 20/20, PPO Loss: -0.1176, Reg Loss: 0.0000, CD Loss: 0.0000
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_7.ckpt

[Online RL] Iteration 9/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.800, env_return=1012.60, rl_reward=0.80, steps=4000
[Data Collection] Success Rate: 0.800, EnvReturn: 1012.60, RLReward: 0.80, Episodes: 20, Steps: 4000
[Dataset] online collection: keeping 16/20 successful episodes (dropped 4 failures) before merge.
[Online RL] Epoch 1/20, PPO Loss: 0.0161, Reg Loss: 0.0000, CD Loss: 0.0137
[Online RL] Epoch 2/20, PPO Loss: 0.0024, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/20, PPO Loss: -0.0323, Reg Loss: 0.0000, CD Loss: 0.0129
[Online RL] Epoch 4/20, PPO Loss: -0.0500, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 5/20, PPO Loss: -0.0814, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 6/20, PPO Loss: -0.1277, Reg Loss: 0.0000, CD Loss: 0.0135
[Online RL] Epoch 7/20, PPO Loss: -0.0876, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 8/20, PPO Loss: -0.1172, Reg Loss: 0.0000, CD Loss: 0.0122
[Online RL] Epoch 9/20, PPO Loss: -0.1206, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 10/20, PPO Loss: -0.1058, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 11/20, PPO Loss: -0.1090, Reg Loss: 0.0000, CD Loss: 0.0133
[Online RL] Epoch 12/20, PPO Loss: -0.1337, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 13/20, PPO Loss: -0.1444, Reg Loss: 0.0000, CD Loss: 0.0143
[Online RL] Epoch 14/20, PPO Loss: -0.1222, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 15/20, PPO Loss: -0.1258, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 16/20, PPO Loss: -0.1070, Reg Loss: 0.0000, CD Loss: 0.0138
[Online RL] Epoch 17/20, PPO Loss: -0.1085, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 18/20, PPO Loss: -0.1089, Reg Loss: 0.0000, CD Loss: 0.0114
[Online RL] Epoch 19/20, PPO Loss: -0.1210, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 20/20, PPO Loss: -0.0995, Reg Loss: 0.0000, CD Loss: 0.0000
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_8.ckpt

[Online RL] Iteration 10/10

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 9)
[Collect] 20 episodes, success=0.700, env_return=1002.01, rl_reward=0.70, steps=4000
[Data Collection] Success Rate: 0.700, EnvReturn: 1002.01, RLReward: 0.70, Episodes: 20, Steps: 4000
[Dataset] online collection: keeping 14/20 successful episodes (dropped 6 failures) before merge.
[Online RL] Epoch 1/20, PPO Loss: 0.0232, Reg Loss: 0.0000, CD Loss: 0.0139
[Online RL] Epoch 2/20, PPO Loss: -0.0251, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 3/20, PPO Loss: -0.0447, Reg Loss: 0.0000, CD Loss: 0.0132
[Online RL] Epoch 4/20, PPO Loss: -0.0655, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 5/20, PPO Loss: -0.0046, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 6/20, PPO Loss: -0.1039, Reg Loss: 0.0000, CD Loss: 0.0132
[Online RL] Epoch 7/20, PPO Loss: -0.0601, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 8/20, PPO Loss: -0.1060, Reg Loss: 0.0000, CD Loss: 0.0137
[Online RL] Epoch 9/20, PPO Loss: -0.1062, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 10/20, PPO Loss: -0.0912, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 11/20, PPO Loss: -0.1464, Reg Loss: 0.0000, CD Loss: 0.0149
[Online RL] Epoch 12/20, PPO Loss: -0.1514, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 13/20, PPO Loss: -0.1266, Reg Loss: 0.0000, CD Loss: 0.0134
[Online RL] Epoch 14/20, PPO Loss: -0.0945, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 15/20, PPO Loss: -0.0818, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 16/20, PPO Loss: -0.1126, Reg Loss: 0.0000, CD Loss: 0.0115
[Online RL] Epoch 17/20, PPO Loss: -0.1428, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 18/20, PPO Loss: -0.1395, Reg Loss: 0.0000, CD Loss: 0.0111
[Online RL] Epoch 19/20, PPO Loss: -0.1074, Reg Loss: 0.0000, CD Loss: 0.0000
[Online RL] Epoch 20/20, PPO Loss: -0.1239, Reg Loss: 0.0000, CD Loss: 0.0000
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/online_iter_9.ckpt

================================================================================
                         TRAINING COMPLETE!
================================================================================

[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/final.ckpt

[Evaluation] Running final evaluation...
test_mean_score: 0.65
test_mean_score: 0.75

================================================================================
                         FINAL RESULTS
================================================================================
[ddim]
mean_traj_rewards: 11455.1179
mean_success_rates: 0.6500
test_mean_score: 0.6500
SR_test_L3: 0.7667
SR_test_L5: 0.7500
[cm]
mean_traj_rewards: 11729.9273
mean_success_rates: 0.7500
test_mean_score: 0.7500
SR_test_L3: 0.7667
SR_test_L5: 0.7600

[Training] Complete! Checkpoints saved to:
  /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints
Found 8 GPUs for rendering. Using device 0.
Extracting GPU stats logs using atop has been completed on r8l40-a02.
Logs are being saved to: /nfs_global/S/yangrongzheng/atop-737411-r8l40-a02-gpustat.log
Job end at 2026-03-14 20:56:18
