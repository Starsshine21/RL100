Job start at 2026-04-02 11:01:03
Job run at:
   Static hostname: localhost.localdomain
Transient hostname: r8l40s-a05
         Icon name: computer-server
           Chassis: server
        Machine ID: 3ca2905c1b7740bb8ba855b252a5c4a9
           Boot ID: 11804f13fb1a497c9344d1b66b9abab7
  Operating System: Rocky Linux 8.7 (Green Obsidian)
       CPE OS Name: cpe:/o:rocky:rocky:8:GA
            Kernel: Linux 4.18.0-425.10.1.el8_7.x86_64
      Architecture: x86-64
Filesystem                                        Size  Used Avail Use% Mounted on
/dev/mapper/rl-root                               376G   30G  347G   8% /
/dev/nvme1n1p1                                    1.8T   13G  1.8T   1% /tmp
/dev/nvme2n1p1                                    3.5T   25G  3.5T   1% /local
/dev/mapper/rl-var                                512G   20G  493G   4% /var
/dev/nvme0n1p2                                    2.0G  367M  1.7G  18% /boot
/dev/nvme1n1p2                                    1.8T   14G  1.8T   1% /local/nfscache
/dev/nvme0n1p1                                    599M  5.8M  594M   1% /boot/efi
ssd.nas00.future.cn:/rocky8_home                   16G   15G  2.0G  88% /home
ssd.nas00.future.cn:/rocky8_workspace             400G     0  400G   0% /workspace
ssd.nas00.future.cn:/rocky8_tools                 5.0T   99G  5.0T   2% /tools
ssd.nas00.future.cn:/centos7_home                  16G  4.2G   12G  26% /centos7/home
ssd.nas00.future.cn:/centos7_workspace            400G     0  400G   0% /centos7/workspace
ssd.nas00.future.cn:/centos7_tools                5.0T  235G  4.8T   5% /centos7/tools
ssd.nas00.future.cn:/eda-tools                    8.0T  6.4T  1.7T  79% /centos7/eda-tools
hdd.nas00.future.cn:/share_personal               500G     0  500G   0% /share/personal
zone05.nas01.future.cn:/NAS_HPC_collab_codemodel   40T   37T  3.3T  92% /share/collab/codemodel
ext-zone00.nas02.future.cn:/nfs_global            425T  400T   26T  95% /nfs_global
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
          /home  14340M  16384M  20480M            160k       0       0        

############### /workspace
Disk quotas for user yangrongzheng (uid 6215): 
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
     /workspace      0K    400G    500G               1       0       0        

############### /nfs_global
Disk quotas for user yangrongzheng (uid 6215): 
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
    /nfs_global    479G   5120G   7168G            352k   5000k  10000k        

############### /lustre
Disk quotas for usr yangrongzheng (uid 6215):
     Filesystem    used   quota   limit   grace   files   quota   limit   grace
        /lustre      0k      8T     10T       -       0  3000000 36000000       -
uid 6215 is using default block quota setting
uid 6215 is using default file quota setting
name, driver_version, power.limit [W]
NVIDIA L40S, 570.124.06, 325.00 W
NVIDIA L40S, 570.124.06, 325.00 W
NVIDIA L40S, 570.124.06, 325.00 W
NVIDIA L40S, 570.124.06, 325.00 W
NVIDIA L40S, 570.124.06, 325.00 W
NVIDIA L40S, 570.124.06, 325.00 W
NVIDIA L40S, 570.124.06, 325.00 W
NVIDIA L40S, 570.124.06, 325.00 W
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
  name: peg-insert-side
  task_name: ${.name}
  point_cloud:
    num_points: 512
    channels: 3
    use_pc_color: false
    use_point_crop: true
    sampling_method: fps
  demonstration:
    num_episodes: 10
  shape_meta:
    obs:
      point_cloud:
        shape:
        - ${task.point_cloud.num_points}
        - ${task.point_cloud.channels}
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
    use_point_crop: ${task.point_cloud.use_point_crop}
    use_pc_color: ${task.point_cloud.use_pc_color}
    point_sampling_method: ${task.point_cloud.sampling_method}
    num_points: ${task.point_cloud.num_points}
  dataset:
    _target_: diffusion_policy_3d.dataset.metaworld_dataset.MetaworldDataset
    zarr_path: data/metaworld_peg-insert-side_expert.zarr
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
  transition_max_epochs: 50
  transition_patience: 5
  critic_epochs: 20
  offline_critic_steps: 1000
  ppo_epochs: 3
  offline_ppo_steps: 1000
  offline_ppo_max_passes: 8
  ppo_inner_steps: 1
  collection_episodes: 50
  cd_every: 5
  lambda_cd: 0
  rl_policy_lr: 4.0e-06
  ppo_target_kl: 0.5
  ppo_target_clip_frac: 0.6
  ppo_early_stop_min_steps: 10
  run_online_rl: true
  online_rl_iterations: 10
  online_collection_episodes: 50
  online_value_steps: 200
  online_ppo_steps: 100
  lambda_v: 0.5
  gae_lambda: 0.95
  gradient_accumulate_every: 1
  max_grad_norm: 0.5
  log_every: 10
  eval_every: 100
  checkpoint_every: 200
  resume: true
  resume_load_rl_state: false
  resume_path: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/after_il.ckpt
ope:
  num_batches: 100
  rollout_horizon: 5
  delta_coef: 0.0
  delta_abs_min: 0.0
runtime:
  collection_policy: ddim
  collection_use_ema: false
  il_retrain_success_only: true
  restore_best_ddim_before_final_eval: true
  final_eval_policies:
  - ddim
  - cm
  final_eval_use_ema: false
  eval_policy_mode: ddim
  eval_use_ema: false
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
Replay Buffer: state, shape (2000, 9), dtype float32, range -0.23~0.76
Replay Buffer: action, shape (2000, 4), dtype float32, range -11.52~6.85
Replay Buffer: point_cloud, shape (2000, 512, 3), dtype float32, range -1.10~0.63
Replay Buffer: reward, shape (2000,), dtype float32, range 0.00~1.00
Replay Buffer: done, shape (2000,), dtype float32, range 0.00~1.00
--------------------------
[Setup] Dataset loaded: 1737 samples across 10 episodes
[Setup] Dataset has reward/done labels: True
[Setup] Dataset point_cloud shape: (512, 3)
[Setup] Initializing environment runner...
obj_low: (0.0, 0.5, 0.02), obj_high: (0.2, 0.7, 0.02)
goal_low: (-0.35, 0.4, -0.001), goal_high: (-0.25, 0.7, 0.001)
[MetaWorldEnv] use_point_crop: True
[MetaWorldEnv] use_pc_color: False
[MetaWorldEnv] point_sampling_method: fps
[MetaWorldEnv] num_points: 512
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
[2026-04-02 11:01:31,037][diffusion_policy_3d.model.diffusion.conditional_unet1d][INFO] - number of parameters: 2.550744e+08
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
[2026-04-02 11:01:32,920][diffusion_policy_3d.model.diffusion.conditional_unet1d][INFO] - number of parameters: 2.550744e+08
[RL100Trainer] Initializing Transition Model T_θ(s'|s,a)...
[Setup] RL100Trainer initialized

[Setup] Resuming from checkpoint: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/after_il.ckpt
[Checkpoint] Loaded policy/EMA only; RL heads and optimizers keep fresh initialization.
[Setup] IL phase will be skipped — starting offline RL from restored IL policy with fresh RL heads.

[Training] Starting RL-100 pipeline...

================================================================================
                    RL-100 TRAINING PIPELINE
================================================================================


[RL100Trainer] Skipping IL phase — loaded from checkpoint.
[RL100Trainer] Normalizer synced from dataset. Resuming offline RL from iteration 0.
[RL100Trainer] Evaluating resumed DDIM policy to establish best-checkpoint baseline.
test_mean_score: 0.6
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/best_ddim.ckpt
[BestDDIM] Recorded initial best DDIM checkpoint: 0.6000 at resume_loaded_policy.
[RL100Trainer] Dataset already contains reward/done labels; keep existing rewards.

================================================================================
               OFFLINE RL ITERATION 1/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 0)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 1737 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=23.73646 | val=0.00666 | no-improve=0/5
[TransitionModel] Epoch    5 | train=-0.08544 | val=0.00825 | no-improve=5/5
[TransitionModel] Training complete. Elites=[4, 0, 6, 5, 2], val_loss=0.00666
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_00.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_00.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 0)
[IQL] Step 7/1000, V Loss: 0.5649, Q Loss: 0.3743
[IQL] Step 14/1000, V Loss: 0.0174, Q Loss: 0.0530
[IQL] Step 21/1000, V Loss: 0.0019, Q Loss: 0.0287
[IQL] Step 28/1000, V Loss: 0.0025, Q Loss: 0.0151
[IQL] Step 35/1000, V Loss: 0.0043, Q Loss: 0.0088
[IQL] Step 42/1000, V Loss: 0.0030, Q Loss: 0.0062
[IQL] Step 49/1000, V Loss: 0.0019, Q Loss: 0.0047
[IQL] Step 56/1000, V Loss: 0.0016, Q Loss: 0.0039
[IQL] Step 63/1000, V Loss: 0.0008, Q Loss: 0.0034
[IQL] Step 70/1000, V Loss: 0.0003, Q Loss: 0.0031
[IQL] Step 77/1000, V Loss: 0.0003, Q Loss: 0.0029
[IQL] Step 84/1000, V Loss: 0.0004, Q Loss: 0.0029
[IQL] Step 91/1000, V Loss: 0.0002, Q Loss: 0.0026
[IQL] Step 98/1000, V Loss: 0.0002, Q Loss: 0.0026
[IQL] Step 105/1000, V Loss: 0.0003, Q Loss: 0.0025
[IQL] Step 112/1000, V Loss: 0.0002, Q Loss: 0.0024
[IQL] Step 119/1000, V Loss: 0.0002, Q Loss: 0.0024
[IQL] Step 126/1000, V Loss: 0.0002, Q Loss: 0.0024
[IQL] Step 133/1000, V Loss: 0.0002, Q Loss: 0.0025
[IQL] Step 140/1000, V Loss: 0.0002, Q Loss: 0.0024
[IQL] Step 147/1000, V Loss: 0.0002, Q Loss: 0.0023
[IQL] Step 154/1000, V Loss: 0.0002, Q Loss: 0.0024
[IQL] Step 161/1000, V Loss: 0.0002, Q Loss: 0.0022
[IQL] Step 168/1000, V Loss: 0.0002, Q Loss: 0.0023
[IQL] Step 175/1000, V Loss: 0.0002, Q Loss: 0.0023
[IQL] Step 182/1000, V Loss: 0.0002, Q Loss: 0.0023
[IQL] Step 189/1000, V Loss: 0.0002, Q Loss: 0.0023
[IQL] Step 196/1000, V Loss: 0.0002, Q Loss: 0.0023
[IQL] Step 203/1000, V Loss: 0.0002, Q Loss: 0.0022
[IQL] Step 210/1000, V Loss: 0.0002, Q Loss: 0.0023
[IQL] Step 217/1000, V Loss: 0.0002, Q Loss: 0.0022
[IQL] Step 224/1000, V Loss: 0.0001, Q Loss: 0.0021
[IQL] Step 231/1000, V Loss: 0.0002, Q Loss: 0.0022
[IQL] Step 238/1000, V Loss: 0.0001, Q Loss: 0.0022
[IQL] Step 245/1000, V Loss: 0.0002, Q Loss: 0.0022
[IQL] Step 252/1000, V Loss: 0.0002, Q Loss: 0.0022
[IQL] Step 259/1000, V Loss: 0.0001, Q Loss: 0.0021
[IQL] Step 266/1000, V Loss: 0.0001, Q Loss: 0.0022
[IQL] Step 273/1000, V Loss: 0.0001, Q Loss: 0.0022
[IQL] Step 280/1000, V Loss: 0.0001, Q Loss: 0.0023
[IQL] Step 287/1000, V Loss: 0.0001, Q Loss: 0.0021
[IQL] Step 294/1000, V Loss: 0.0001, Q Loss: 0.0023
[IQL] Step 301/1000, V Loss: 0.0001, Q Loss: 0.0022
[IQL] Step 308/1000, V Loss: 0.0001, Q Loss: 0.0023
[IQL] Step 315/1000, V Loss: 0.0002, Q Loss: 0.0023
[IQL] Step 322/1000, V Loss: 0.0002, Q Loss: 0.0022
[IQL] Step 329/1000, V Loss: 0.0002, Q Loss: 0.0023
[IQL] Step 336/1000, V Loss: 0.0001, Q Loss: 0.0021
[IQL] Step 343/1000, V Loss: 0.0001, Q Loss: 0.0022
[IQL] Step 350/1000, V Loss: 0.0001, Q Loss: 0.0021
[IQL] Step 357/1000, V Loss: 0.0002, Q Loss: 0.0022
[IQL] Step 364/1000, V Loss: 0.0002, Q Loss: 0.0023
[IQL] Step 371/1000, V Loss: 0.0001, Q Loss: 0.0021
[IQL] Step 378/1000, V Loss: 0.0001, Q Loss: 0.0023
[IQL] Step 385/1000, V Loss: 0.0001, Q Loss: 0.0024
[IQL] Step 392/1000, V Loss: 0.0001, Q Loss: 0.0025
[IQL] Step 399/1000, V Loss: 0.0002, Q Loss: 0.0023
[IQL] Step 406/1000, V Loss: 0.0001, Q Loss: 0.0023
[IQL] Step 413/1000, V Loss: 0.0001, Q Loss: 0.0022
[IQL] Step 420/1000, V Loss: 0.0001, Q Loss: 0.0023
[IQL] Step 427/1000, V Loss: 0.0001, Q Loss: 0.0022
[IQL] Step 434/1000, V Loss: 0.0001, Q Loss: 0.0023
[IQL] Step 441/1000, V Loss: 0.0001, Q Loss: 0.0023
[IQL] Step 448/1000, V Loss: 0.0001, Q Loss: 0.0022
[IQL] Step 455/1000, V Loss: 0.0001, Q Loss: 0.0022
[IQL] Step 462/1000, V Loss: 0.0001, Q Loss: 0.0022
[IQL] Step 469/1000, V Loss: 0.0001, Q Loss: 0.0022
[IQL] Step 476/1000, V Loss: 0.0001, Q Loss: 0.0022
[IQL] Step 483/1000, V Loss: 0.0001, Q Loss: 0.0024
[IQL] Step 490/1000, V Loss: 0.0002, Q Loss: 0.0024
[IQL] Step 497/1000, V Loss: 0.0001, Q Loss: 0.0025
[IQL] Step 504/1000, V Loss: 0.0001, Q Loss: 0.0024
[IQL] Step 511/1000, V Loss: 0.0001, Q Loss: 0.0022
[IQL] Step 518/1000, V Loss: 0.0001, Q Loss: 0.0023
[IQL] Step 525/1000, V Loss: 0.0001, Q Loss: 0.0024
[IQL] Step 532/1000, V Loss: 0.0001, Q Loss: 0.0023
[IQL] Step 539/1000, V Loss: 0.0002, Q Loss: 0.0024
[IQL] Step 546/1000, V Loss: 0.0002, Q Loss: 0.0023
[IQL] Step 553/1000, V Loss: 0.0001, Q Loss: 0.0024
[IQL] Step 560/1000, V Loss: 0.0001, Q Loss: 0.0026
[IQL] Step 567/1000, V Loss: 0.0001, Q Loss: 0.0025
[IQL] Step 574/1000, V Loss: 0.0002, Q Loss: 0.0023
[IQL] Step 581/1000, V Loss: 0.0002, Q Loss: 0.0024
[IQL] Step 588/1000, V Loss: 0.0001, Q Loss: 0.0023
[IQL] Step 595/1000, V Loss: 0.0001, Q Loss: 0.0024
[IQL] Step 602/1000, V Loss: 0.0001, Q Loss: 0.0023
[IQL] Step 609/1000, V Loss: 0.0001, Q Loss: 0.0024
[IQL] Step 616/1000, V Loss: 0.0001, Q Loss: 0.0024
[IQL] Step 623/1000, V Loss: 0.0002, Q Loss: 0.0025
[IQL] Step 630/1000, V Loss: 0.0002, Q Loss: 0.0025
[IQL] Step 637/1000, V Loss: 0.0001, Q Loss: 0.0025
[IQL] Step 644/1000, V Loss: 0.0001, Q Loss: 0.0024
[IQL] Step 651/1000, V Loss: 0.0001, Q Loss: 0.0025
[IQL] Step 658/1000, V Loss: 0.0001, Q Loss: 0.0025
[IQL] Step 665/1000, V Loss: 0.0001, Q Loss: 0.0025
[IQL] Step 672/1000, V Loss: 0.0001, Q Loss: 0.0025
[IQL] Step 679/1000, V Loss: 0.0001, Q Loss: 0.0027
[IQL] Step 686/1000, V Loss: 0.0002, Q Loss: 0.0028
[IQL] Step 693/1000, V Loss: 0.0001, Q Loss: 0.0027
[IQL] Step 700/1000, V Loss: 0.0001, Q Loss: 0.0026
[IQL] Step 707/1000, V Loss: 0.0001, Q Loss: 0.0025
[IQL] Step 714/1000, V Loss: 0.0001, Q Loss: 0.0029
[IQL] Step 721/1000, V Loss: 0.0001, Q Loss: 0.0026
[IQL] Step 728/1000, V Loss: 0.0002, Q Loss: 0.0027
[IQL] Step 735/1000, V Loss: 0.0003, Q Loss: 0.0027
[IQL] Step 742/1000, V Loss: 0.0003, Q Loss: 0.0028
[IQL] Step 749/1000, V Loss: 0.0002, Q Loss: 0.0029
[IQL] Step 756/1000, V Loss: 0.0002, Q Loss: 0.0029
[IQL] Step 763/1000, V Loss: 0.0001, Q Loss: 0.0026
[IQL] Step 770/1000, V Loss: 0.0001, Q Loss: 0.0026
[IQL] Step 777/1000, V Loss: 0.0001, Q Loss: 0.0026
[IQL] Step 784/1000, V Loss: 0.0001, Q Loss: 0.0026
[IQL] Step 791/1000, V Loss: 0.0001, Q Loss: 0.0027
[IQL] Step 798/1000, V Loss: 0.0002, Q Loss: 0.0027
[IQL] Step 805/1000, V Loss: 0.0002, Q Loss: 0.0027
[IQL] Step 812/1000, V Loss: 0.0001, Q Loss: 0.0026
[IQL] Step 819/1000, V Loss: 0.0001, Q Loss: 0.0028
[IQL] Step 826/1000, V Loss: 0.0001, Q Loss: 0.0027
[IQL] Step 833/1000, V Loss: 0.0001, Q Loss: 0.0028
[IQL] Step 840/1000, V Loss: 0.0002, Q Loss: 0.0030
[IQL] Step 847/1000, V Loss: 0.0001, Q Loss: 0.0030
[IQL] Step 854/1000, V Loss: 0.0002, Q Loss: 0.0028
[IQL] Step 861/1000, V Loss: 0.0002, Q Loss: 0.0033
[IQL] Step 868/1000, V Loss: 0.0002, Q Loss: 0.0031
[IQL] Step 875/1000, V Loss: 0.0003, Q Loss: 0.0032
[IQL] Step 882/1000, V Loss: 0.0003, Q Loss: 0.0032
[IQL] Step 889/1000, V Loss: 0.0002, Q Loss: 0.0030
[IQL] Step 896/1000, V Loss: 0.0001, Q Loss: 0.0028
[IQL] Step 903/1000, V Loss: 0.0002, Q Loss: 0.0028
[IQL] Step 910/1000, V Loss: 0.0002, Q Loss: 0.0029
[IQL] Step 917/1000, V Loss: 0.0001, Q Loss: 0.0030
[IQL] Step 924/1000, V Loss: 0.0002, Q Loss: 0.0032
[IQL] Step 931/1000, V Loss: 0.0004, Q Loss: 0.0030
[IQL] Step 938/1000, V Loss: 0.0002, Q Loss: 0.0029
[IQL] Step 945/1000, V Loss: 0.0001, Q Loss: 0.0028
[IQL] Step 952/1000, V Loss: 0.0001, Q Loss: 0.0028
[IQL] Step 959/1000, V Loss: 0.0001, Q Loss: 0.0029
[IQL] Step 966/1000, V Loss: 0.0001, Q Loss: 0.0029
[IQL] Step 973/1000, V Loss: 0.0001, Q Loss: 0.0029
[IQL] Step 980/1000, V Loss: 0.0001, Q Loss: 0.0030
[IQL] Step 987/1000, V Loss: 0.0001, Q Loss: 0.0029
[IQL] Step 994/1000, V Loss: 0.0002, Q Loss: 0.0028
[IQL] Step 1000/1000, V Loss: 0.0002, Q Loss: 0.0033
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_00.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_00.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 0)
[OPE] AM-Q eval uses episode-start states: 9 unique episodes, 100 batch(es) x 256.
[OPE] Behavior policy value J_old = 0.6379
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 4.00e-06
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 7 mini-batches, 1737 samples, raw advantage mean=-0.0206, std=0.0142
[Offline RL] Cap PPO steps: requested 1000, using 56 (8 passes over 7 fixed PPO batches).
[Offline RL] Step 7/56, PPO Loss: -0.0394, PostKL: 6.692e-02, PostClipFrac: 0.356355, PostMeanRatio: 0.991875, PostRatioDev: 2.254e-01, GradNorm: 27.3847, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 14/56, PPO Loss: -0.0036, PostKL: 6.298e-02, PostClipFrac: 0.317171, PostMeanRatio: 0.989268, PostRatioDev: 2.058e-01, GradNorm: 21.2163, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 21/56, PPO Loss: 0.0151, PostKL: 5.800e-02, PostClipFrac: 0.310660, PostMeanRatio: 0.983265, PostRatioDev: 1.963e-01, GradNorm: 21.3648, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 28/56, PPO Loss: 0.0345, PostKL: 5.569e-02, PostClipFrac: 0.288306, PostMeanRatio: 0.984308, PostRatioDev: 1.861e-01, GradNorm: 16.6363, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 35/56, PPO Loss: 0.0455, PostKL: 6.004e-02, PostClipFrac: 0.306245, PostMeanRatio: 0.984295, PostRatioDev: 1.934e-01, GradNorm: 17.3905, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 42/56, PPO Loss: 0.0555, PostKL: 6.143e-02, PostClipFrac: 0.303371, PostMeanRatio: 0.985124, PostRatioDev: 1.926e-01, GradNorm: 15.7735, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 49/56, PPO Loss: 0.0646, PostKL: 6.156e-02, PostClipFrac: 0.313095, PostMeanRatio: 0.985666, PostRatioDev: 1.963e-01, GradNorm: 15.2138, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 56/56, PPO Loss: 0.0695, PostKL: 6.453e-02, PostClipFrac: 0.336426, PostMeanRatio: 0.988455, PostRatioDev: 2.058e-01, GradNorm: 18.9971, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_00.png
[OPE] Policy ACCEPTED: J_new=0.6412 > J_old=0.6379 + δ=0.0000

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 0)
[Collect] 50 episodes, success=0.780, env_return=845.54, rl_reward=0.78, steps=10000
[Data Collection] Success Rate: 0.780, EnvReturn: 845.54, RLReward: 0.78, Episodes: 50, Steps: 10000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 50 episodes remain in RL replay; IL retrain keeps 39/50 successful episodes (drops 11 failures).
[Dataset] Merged 50 episodes (10000 steps) → total 12000 steps, 60 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0415, Val Loss: 0.0496
[IL] Epoch 1/100, Loss: 0.0226, Val Loss: 0.0534
[IL] Epoch 2/100, Loss: 0.0211, Val Loss: 0.0371
[IL] Epoch 3/100, Loss: 0.0205, Val Loss: 0.0422
[IL] Epoch 4/100, Loss: 0.0194, Val Loss: 0.0531
[IL] Epoch 5/100, Loss: 0.0188, Val Loss: 0.0529
[IL] Epoch 6/100, Loss: 0.0197, Val Loss: 0.0572
[IL] Epoch 7/100, Loss: 0.0186, Val Loss: 0.0710
[IL] Epoch 8/100, Loss: 0.0189, Val Loss: 0.0617
[IL] Epoch 9/100, Loss: 0.0191, Val Loss: 0.0566
[IL] Epoch 10/100, Loss: 0.0178, Val Loss: 0.0685
[IL] Epoch 11/100, Loss: 0.0181, Val Loss: 0.0522
[IL] Epoch 12/100, Loss: 0.0189, Val Loss: 0.0517
[IL] Epoch 13/100, Loss: 0.0186, Val Loss: 0.0515
[IL] Epoch 14/100, Loss: 0.0181, Val Loss: 0.0590
[IL] Epoch 15/100, Loss: 0.0171, Val Loss: 0.0821
[IL] Epoch 16/100, Loss: 0.0176, Val Loss: 0.0517
[IL] Epoch 17/100, Loss: 0.0173, Val Loss: 0.0509
[IL] Epoch 18/100, Loss: 0.0173, Val Loss: 0.0469
[IL] Epoch 19/100, Loss: 0.0168, Val Loss: 0.0533
[IL] Epoch 20/100, Loss: 0.0168, Val Loss: 0.0662
[IL] Epoch 21/100, Loss: 0.0163, Val Loss: 0.0575
[IL] Epoch 22/100, Loss: 0.0168, Val Loss: 0.0598
[IL] Epoch 23/100, Loss: 0.0168, Val Loss: 0.0565
[IL] Epoch 24/100, Loss: 0.0166, Val Loss: 0.0526
[IL] Epoch 25/100, Loss: 0.0161, Val Loss: 0.0563
[IL] Epoch 26/100, Loss: 0.0168, Val Loss: 0.0500
[IL] Epoch 27/100, Loss: 0.0162, Val Loss: 0.0631
[IL] Epoch 28/100, Loss: 0.0166, Val Loss: 0.0589
[IL] Epoch 29/100, Loss: 0.0185, Val Loss: 0.0498
[IL] Epoch 30/100, Loss: 0.0163, Val Loss: 0.0472
[IL] Epoch 31/100, Loss: 0.0168, Val Loss: 0.0699
[IL] Epoch 32/100, Loss: 0.0167, Val Loss: 0.0644
[IL] Epoch 33/100, Loss: 0.0154, Val Loss: 0.0536
[IL] Epoch 34/100, Loss: 0.0155, Val Loss: 0.0600
[IL] Epoch 35/100, Loss: 0.0159, Val Loss: 0.0647
[IL] Epoch 36/100, Loss: 0.0158, Val Loss: 0.0625
[IL] Epoch 37/100, Loss: 0.0166, Val Loss: 0.0421
[IL] Epoch 38/100, Loss: 0.0147, Val Loss: 0.0421
[IL] Epoch 39/100, Loss: 0.0155, Val Loss: 0.0656
[IL] Epoch 40/100, Loss: 0.0147, Val Loss: 0.0661
[IL] Epoch 41/100, Loss: 0.0156, Val Loss: 0.0642
[IL] Epoch 42/100, Loss: 0.0156, Val Loss: 0.0685
[IL] Epoch 43/100, Loss: 0.0156, Val Loss: 0.0540
[IL] Epoch 44/100, Loss: 0.0149, Val Loss: 0.0598
[IL] Epoch 45/100, Loss: 0.0156, Val Loss: 0.0660
[IL] Epoch 46/100, Loss: 0.0154, Val Loss: 0.0600
[IL] Epoch 47/100, Loss: 0.0152, Val Loss: 0.0541
[IL] Epoch 48/100, Loss: 0.0152, Val Loss: 0.0744
[IL] Epoch 49/100, Loss: 0.0144, Val Loss: 0.0523
[IL] Epoch 50/100, Loss: 0.0148, Val Loss: 0.0695
[IL] Epoch 51/100, Loss: 0.0149, Val Loss: 0.0580
[IL] Epoch 52/100, Loss: 0.0151, Val Loss: 0.0615
[IL] Epoch 53/100, Loss: 0.0158, Val Loss: 0.0486
[IL] Epoch 54/100, Loss: 0.0159, Val Loss: 0.0602
[IL] Epoch 55/100, Loss: 0.0148, Val Loss: 0.0623
[IL] Epoch 56/100, Loss: 0.0158, Val Loss: 0.0624
[IL] Epoch 57/100, Loss: 0.0149, Val Loss: 0.0619
[IL] Epoch 58/100, Loss: 0.0144, Val Loss: 0.0796
[IL] Epoch 59/100, Loss: 0.0150, Val Loss: 0.0727
[IL] Epoch 60/100, Loss: 0.0145, Val Loss: 0.0659
[IL] Epoch 61/100, Loss: 0.0157, Val Loss: 0.0659
[IL] Epoch 62/100, Loss: 0.0140, Val Loss: 0.0553
[IL] Epoch 63/100, Loss: 0.0144, Val Loss: 0.0577
[IL] Epoch 64/100, Loss: 0.0139, Val Loss: 0.0736
[IL] Epoch 65/100, Loss: 0.0138, Val Loss: 0.0737
[IL] Epoch 66/100, Loss: 0.0151, Val Loss: 0.0541
[IL] Epoch 67/100, Loss: 0.0144, Val Loss: 0.0598
[IL] Epoch 68/100, Loss: 0.0143, Val Loss: 0.0501
[IL] Epoch 69/100, Loss: 0.0137, Val Loss: 0.0717
[IL] Epoch 70/100, Loss: 0.0135, Val Loss: 0.0653
[IL] Epoch 71/100, Loss: 0.0143, Val Loss: 0.0721
[IL] Epoch 72/100, Loss: 0.0133, Val Loss: 0.0465
[IL] Epoch 73/100, Loss: 0.0137, Val Loss: 0.0555
[IL] Epoch 74/100, Loss: 0.0137, Val Loss: 0.0582
[IL] Epoch 75/100, Loss: 0.0134, Val Loss: 0.0660
[IL] Epoch 76/100, Loss: 0.0148, Val Loss: 0.0759
[IL] Epoch 77/100, Loss: 0.0132, Val Loss: 0.0778
[IL] Epoch 78/100, Loss: 0.0131, Val Loss: 0.0601
[IL] Epoch 79/100, Loss: 0.0129, Val Loss: 0.0656
[IL] Epoch 80/100, Loss: 0.0142, Val Loss: 0.0587
[IL] Epoch 81/100, Loss: 0.0141, Val Loss: 0.0591
[IL] Epoch 82/100, Loss: 0.0140, Val Loss: 0.0638
[IL] Epoch 83/100, Loss: 0.0133, Val Loss: 0.0703
[IL] Epoch 84/100, Loss: 0.0128, Val Loss: 0.0550
[IL] Epoch 85/100, Loss: 0.0130, Val Loss: 0.0857
[IL] Epoch 86/100, Loss: 0.0131, Val Loss: 0.0631
[IL] Epoch 87/100, Loss: 0.0132, Val Loss: 0.0810
[IL] Epoch 88/100, Loss: 0.0138, Val Loss: 0.0588
[IL] Epoch 89/100, Loss: 0.0135, Val Loss: 0.0626
[IL] Epoch 90/100, Loss: 0.0125, Val Loss: 0.0769
[IL] Epoch 91/100, Loss: 0.0126, Val Loss: 0.0719
[IL] Epoch 92/100, Loss: 0.0129, Val Loss: 0.0666
[IL] Epoch 93/100, Loss: 0.0126, Val Loss: 0.0658
[IL] Epoch 94/100, Loss: 0.0135, Val Loss: 0.0651
[IL] Epoch 95/100, Loss: 0.0128, Val Loss: 0.0993
[IL] Epoch 96/100, Loss: 0.0147, Val Loss: 0.0716
[IL] Epoch 97/100, Loss: 0.0122, Val Loss: 0.0686
[IL] Epoch 98/100, Loss: 0.0140, Val Loss: 0.0755
[IL] Epoch 99/100, Loss: 0.0128, Val Loss: 0.0746
test_mean_score: 0.73
[IL] Eval - Success Rate: 0.730
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_00.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/best_ddim.ckpt
[BestDDIM] Updated best DDIM checkpoint from 0.6000 to 0.7300 at offline_iter_0.
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_0.ckpt

================================================================================
               OFFLINE RL ITERATION 2/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 1)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 11387 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=21.51633 | val=0.02684 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-22.84668 | val=0.00263 | no-improve=0/5
[TransitionModel] Epoch   40 | train=-27.95476 | val=0.00217 | no-improve=0/5
[TransitionModel] Training complete. Elites=[2, 6, 4, 0, 5], val_loss=0.00203
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_01.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_01.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 1)
[IQL] Step 45/1000, V Loss: 0.0002, Q Loss: 0.0032
[IQL] Step 90/1000, V Loss: 0.0002, Q Loss: 0.0031
[IQL] Step 135/1000, V Loss: 0.0002, Q Loss: 0.0032
[IQL] Step 180/1000, V Loss: 0.0002, Q Loss: 0.0031
[IQL] Step 225/1000, V Loss: 0.0001, Q Loss: 0.0031
[IQL] Step 270/1000, V Loss: 0.0002, Q Loss: 0.0032
[IQL] Step 315/1000, V Loss: 0.0002, Q Loss: 0.0032
[IQL] Step 360/1000, V Loss: 0.0001, Q Loss: 0.0031
[IQL] Step 405/1000, V Loss: 0.0002, Q Loss: 0.0033
[IQL] Step 450/1000, V Loss: 0.0002, Q Loss: 0.0032
[IQL] Step 495/1000, V Loss: 0.0002, Q Loss: 0.0032
[IQL] Step 540/1000, V Loss: 0.0003, Q Loss: 0.0033
[IQL] Step 585/1000, V Loss: 0.0002, Q Loss: 0.0033
[IQL] Step 630/1000, V Loss: 0.0003, Q Loss: 0.0036
[IQL] Step 675/1000, V Loss: 0.0002, Q Loss: 0.0034
[IQL] Step 720/1000, V Loss: 0.0001, Q Loss: 0.0033
[IQL] Step 765/1000, V Loss: 0.0001, Q Loss: 0.0034
[IQL] Step 810/1000, V Loss: 0.0002, Q Loss: 0.0035
[IQL] Step 855/1000, V Loss: 0.0002, Q Loss: 0.0035
[IQL] Step 900/1000, V Loss: 0.0005, Q Loss: 0.0039
[IQL] Step 945/1000, V Loss: 0.0004, Q Loss: 0.0038
[IQL] Step 990/1000, V Loss: 0.0002, Q Loss: 0.0035
[IQL] Step 1000/1000, V Loss: 0.0007, Q Loss: 0.0051
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_01.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_01.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 1)
[OPE] AM-Q eval uses episode-start states: 59 unique episodes, 100 batch(es) x 256.
[OPE] Behavior policy value J_old = 0.9431
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 4.00e-06
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 45 mini-batches, 11387 samples, raw advantage mean=0.0276, std=0.0071
[Offline RL] Cap PPO steps: requested 1000, using 360 (8 passes over 45 fixed PPO batches).
[Offline RL] Step 45/360, PPO Loss: -0.0199, PostKL: 3.101e-02, PostClipFrac: 0.224462, PostMeanRatio: 0.996974, PostRatioDev: 1.526e-01, GradNorm: 14.9390, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 90/360, PPO Loss: -0.0014, PostKL: 3.405e-02, PostClipFrac: 0.217697, PostMeanRatio: 0.998582, PostRatioDev: 1.515e-01, GradNorm: 13.6157, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 135/360, PPO Loss: 0.0097, PostKL: 3.676e-02, PostClipFrac: 0.220234, PostMeanRatio: 1.000047, PostRatioDev: 1.534e-01, GradNorm: 13.2884, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 180/360, PPO Loss: 0.0195, PostKL: 3.425e-02, PostClipFrac: 0.214734, PostMeanRatio: 0.999235, PostRatioDev: 1.488e-01, GradNorm: 12.8166, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 225/360, PPO Loss: 0.0250, PostKL: 3.622e-02, PostClipFrac: 0.222052, PostMeanRatio: 1.002247, PostRatioDev: 1.546e-01, GradNorm: 13.2183, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 270/360, PPO Loss: 0.0301, PostKL: 3.772e-02, PostClipFrac: 0.233101, PostMeanRatio: 1.002453, PostRatioDev: 1.590e-01, GradNorm: 12.8823, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 315/360, PPO Loss: 0.0355, PostKL: 3.955e-02, PostClipFrac: 0.246268, PostMeanRatio: 1.005090, PostRatioDev: 1.653e-01, GradNorm: 12.7617, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 360/360, PPO Loss: 0.0393, PostKL: 4.000e-02, PostClipFrac: 0.246380, PostMeanRatio: 1.006359, PostRatioDev: 1.656e-01, GradNorm: 12.1648, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_01.png
[OPE] Policy ACCEPTED: J_new=0.9431 > J_old=0.9431 + δ=0.0000

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 1)
[Collect] 50 episodes, success=0.760, env_return=852.07, rl_reward=0.76, steps=10000
[Data Collection] Success Rate: 0.760, EnvReturn: 852.07, RLReward: 0.76, Episodes: 50, Steps: 10000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 50 episodes remain in RL replay; IL retrain keeps 38/50 successful episodes (drops 12 failures).
[Dataset] Merged 50 episodes (10000 steps) → total 22000 steps, 110 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0200, Val Loss: 0.0674
[IL] Epoch 1/100, Loss: 0.0157, Val Loss: 0.0634
[IL] Epoch 2/100, Loss: 0.0149, Val Loss: 0.0583
[IL] Epoch 3/100, Loss: 0.0150, Val Loss: 0.0708
[IL] Epoch 4/100, Loss: 0.0145, Val Loss: 0.0728
[IL] Epoch 5/100, Loss: 0.0143, Val Loss: 0.0570
[IL] Epoch 6/100, Loss: 0.0144, Val Loss: 0.0703
[IL] Epoch 7/100, Loss: 0.0147, Val Loss: 0.0585
[IL] Epoch 8/100, Loss: 0.0148, Val Loss: 0.0594
[IL] Epoch 9/100, Loss: 0.0143, Val Loss: 0.0626
[IL] Epoch 10/100, Loss: 0.0145, Val Loss: 0.0743
[IL] Epoch 11/100, Loss: 0.0149, Val Loss: 0.0826
[IL] Epoch 12/100, Loss: 0.0147, Val Loss: 0.0895
[IL] Epoch 13/100, Loss: 0.0138, Val Loss: 0.0626
[IL] Epoch 14/100, Loss: 0.0147, Val Loss: 0.0699
[IL] Epoch 15/100, Loss: 0.0141, Val Loss: 0.0690
[IL] Epoch 16/100, Loss: 0.0145, Val Loss: 0.0587
[IL] Epoch 17/100, Loss: 0.0137, Val Loss: 0.0583
[IL] Epoch 18/100, Loss: 0.0138, Val Loss: 0.0764
[IL] Epoch 19/100, Loss: 0.0139, Val Loss: 0.0720
[IL] Epoch 20/100, Loss: 0.0134, Val Loss: 0.0711
[IL] Epoch 21/100, Loss: 0.0138, Val Loss: 0.0696
[IL] Epoch 22/100, Loss: 0.0138, Val Loss: 0.0609
[IL] Epoch 23/100, Loss: 0.0137, Val Loss: 0.0680
[IL] Epoch 24/100, Loss: 0.0139, Val Loss: 0.0693
[IL] Epoch 25/100, Loss: 0.0136, Val Loss: 0.0680
[IL] Epoch 26/100, Loss: 0.0140, Val Loss: 0.0824
[IL] Epoch 27/100, Loss: 0.0136, Val Loss: 0.0612
[IL] Epoch 28/100, Loss: 0.0129, Val Loss: 0.0732
[IL] Epoch 29/100, Loss: 0.0133, Val Loss: 0.0550
[IL] Epoch 30/100, Loss: 0.0135, Val Loss: 0.0746
[IL] Epoch 31/100, Loss: 0.0142, Val Loss: 0.0777
[IL] Epoch 32/100, Loss: 0.0135, Val Loss: 0.0891
[IL] Epoch 33/100, Loss: 0.0133, Val Loss: 0.0769
[IL] Epoch 34/100, Loss: 0.0135, Val Loss: 0.0639
[IL] Epoch 35/100, Loss: 0.0129, Val Loss: 0.0810
[IL] Epoch 36/100, Loss: 0.0130, Val Loss: 0.0737
[IL] Epoch 37/100, Loss: 0.0134, Val Loss: 0.0694
[IL] Epoch 38/100, Loss: 0.0133, Val Loss: 0.0780
[IL] Epoch 39/100, Loss: 0.0130, Val Loss: 0.0574
[IL] Epoch 40/100, Loss: 0.0128, Val Loss: 0.0855
[IL] Epoch 41/100, Loss: 0.0128, Val Loss: 0.0887
[IL] Epoch 42/100, Loss: 0.0132, Val Loss: 0.0787
[IL] Epoch 43/100, Loss: 0.0131, Val Loss: 0.0781
[IL] Epoch 44/100, Loss: 0.0129, Val Loss: 0.0696
[IL] Epoch 45/100, Loss: 0.0129, Val Loss: 0.0671
[IL] Epoch 46/100, Loss: 0.0132, Val Loss: 0.0920
[IL] Epoch 47/100, Loss: 0.0130, Val Loss: 0.0603
[IL] Epoch 48/100, Loss: 0.0127, Val Loss: 0.0736
[IL] Epoch 49/100, Loss: 0.0129, Val Loss: 0.0989
[IL] Epoch 50/100, Loss: 0.0126, Val Loss: 0.0744
[IL] Epoch 51/100, Loss: 0.0125, Val Loss: 0.0994
[IL] Epoch 52/100, Loss: 0.0124, Val Loss: 0.0647
[IL] Epoch 53/100, Loss: 0.0126, Val Loss: 0.0835
[IL] Epoch 54/100, Loss: 0.0127, Val Loss: 0.0833
[IL] Epoch 55/100, Loss: 0.0129, Val Loss: 0.0942
[IL] Epoch 56/100, Loss: 0.0126, Val Loss: 0.0866
[IL] Epoch 57/100, Loss: 0.0128, Val Loss: 0.0626
[IL] Epoch 58/100, Loss: 0.0124, Val Loss: 0.0798
[IL] Epoch 59/100, Loss: 0.0118, Val Loss: 0.0785
[IL] Epoch 60/100, Loss: 0.0123, Val Loss: 0.0737
[IL] Epoch 61/100, Loss: 0.0122, Val Loss: 0.0817
[IL] Epoch 62/100, Loss: 0.0122, Val Loss: 0.0820
[IL] Epoch 63/100, Loss: 0.0123, Val Loss: 0.0888
[IL] Epoch 64/100, Loss: 0.0124, Val Loss: 0.0920
[IL] Epoch 65/100, Loss: 0.0125, Val Loss: 0.0780
[IL] Epoch 66/100, Loss: 0.0120, Val Loss: 0.0792
[IL] Epoch 67/100, Loss: 0.0119, Val Loss: 0.1106
[IL] Epoch 68/100, Loss: 0.0116, Val Loss: 0.1066
[IL] Epoch 69/100, Loss: 0.0118, Val Loss: 0.0825
[IL] Epoch 70/100, Loss: 0.0116, Val Loss: 0.0966
[IL] Epoch 71/100, Loss: 0.0118, Val Loss: 0.0915
[IL] Epoch 72/100, Loss: 0.0118, Val Loss: 0.0872
[IL] Epoch 73/100, Loss: 0.0117, Val Loss: 0.0742
[IL] Epoch 74/100, Loss: 0.0118, Val Loss: 0.0885
[IL] Epoch 75/100, Loss: 0.0119, Val Loss: 0.0716
[IL] Epoch 76/100, Loss: 0.0123, Val Loss: 0.0842
[IL] Epoch 77/100, Loss: 0.0120, Val Loss: 0.0596
[IL] Epoch 78/100, Loss: 0.0120, Val Loss: 0.0801
[IL] Epoch 79/100, Loss: 0.0120, Val Loss: 0.0895
[IL] Epoch 80/100, Loss: 0.0123, Val Loss: 0.0747
[IL] Epoch 81/100, Loss: 0.0114, Val Loss: 0.0868
[IL] Epoch 82/100, Loss: 0.0118, Val Loss: 0.0888
[IL] Epoch 83/100, Loss: 0.0111, Val Loss: 0.0993
[IL] Epoch 84/100, Loss: 0.0119, Val Loss: 0.0803
[IL] Epoch 85/100, Loss: 0.0116, Val Loss: 0.0850
[IL] Epoch 86/100, Loss: 0.0118, Val Loss: 0.0773
[IL] Epoch 87/100, Loss: 0.0115, Val Loss: 0.1052
[IL] Epoch 88/100, Loss: 0.0111, Val Loss: 0.0863
[IL] Epoch 89/100, Loss: 0.0111, Val Loss: 0.0919
[IL] Epoch 90/100, Loss: 0.0115, Val Loss: 0.0777
[IL] Epoch 91/100, Loss: 0.0111, Val Loss: 0.0773
[IL] Epoch 92/100, Loss: 0.0105, Val Loss: 0.0906
[IL] Epoch 93/100, Loss: 0.0112, Val Loss: 0.0790
[IL] Epoch 94/100, Loss: 0.0108, Val Loss: 0.0994
[IL] Epoch 95/100, Loss: 0.0116, Val Loss: 0.0844
[IL] Epoch 96/100, Loss: 0.0109, Val Loss: 0.0851
[IL] Epoch 97/100, Loss: 0.0110, Val Loss: 0.0768
[IL] Epoch 98/100, Loss: 0.0104, Val Loss: 0.1036
[IL] Epoch 99/100, Loss: 0.0111, Val Loss: 0.1067
test_mean_score: 0.8
[IL] Eval - Success Rate: 0.800
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_01.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/best_ddim.ckpt
[BestDDIM] Updated best DDIM checkpoint from 0.7300 to 0.8000 at offline_iter_1.
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_1.ckpt

================================================================================
               OFFLINE RL ITERATION 3/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 2)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 21037 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-27.86191 | val=0.00232 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-35.41321 | val=0.00186 | no-improve=0/5
[TransitionModel] Epoch   40 | train=-40.68650 | val=0.00160 | no-improve=0/5
[TransitionModel] Training complete. Elites=[1, 2, 3, 0, 4], val_loss=0.00152
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_02.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_02.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 2)
[IQL] Step 83/1000, V Loss: 0.0002, Q Loss: 0.0036
[IQL] Step 166/1000, V Loss: 0.0001, Q Loss: 0.0037
[IQL] Step 249/1000, V Loss: 0.0001, Q Loss: 0.0036
[IQL] Step 332/1000, V Loss: 0.0002, Q Loss: 0.0038
[IQL] Step 415/1000, V Loss: 0.0001, Q Loss: 0.0037
[IQL] Step 498/1000, V Loss: 0.0001, Q Loss: 0.0037
[IQL] Step 581/1000, V Loss: 0.0002, Q Loss: 0.0038
[IQL] Step 664/1000, V Loss: 0.0002, Q Loss: 0.0038
[IQL] Step 747/1000, V Loss: 0.0002, Q Loss: 0.0039
[IQL] Step 830/1000, V Loss: 0.0001, Q Loss: 0.0039
[IQL] Step 913/1000, V Loss: 0.0001, Q Loss: 0.0039
[IQL] Step 996/1000, V Loss: 0.0002, Q Loss: 0.0040
[IQL] Step 1000/1000, V Loss: 0.0000, Q Loss: 0.0036
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_02.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_02.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 2)
[OPE] AM-Q eval uses episode-start states: 109 unique episodes, 100 batch(es) x 256.
[OPE] Behavior policy value J_old = 0.7531
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 4.00e-06
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 83 mini-batches, 21037 samples, raw advantage mean=-0.0208, std=0.0058
[Offline RL] Cap PPO steps: requested 1000, using 664 (8 passes over 83 fixed PPO batches).
[Offline RL] Step 83/664, PPO Loss: -0.0172, PostKL: 3.714e-02, PostClipFrac: 0.201364, PostMeanRatio: 0.997293, PostRatioDev: 1.427e-01, GradNorm: 17.3481, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 166/664, PPO Loss: -0.0003, PostKL: 4.980e-02, PostClipFrac: 0.196821, PostMeanRatio: 0.992922, PostRatioDev: 1.407e-01, GradNorm: 14.7061, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 249/664, PPO Loss: 0.0092, PostKL: 4.611e-02, PostClipFrac: 0.198545, PostMeanRatio: 0.990914, PostRatioDev: 1.405e-01, GradNorm: 13.2300, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 332/664, PPO Loss: 0.0156, PostKL: 4.637e-02, PostClipFrac: 0.201943, PostMeanRatio: 0.991062, PostRatioDev: 1.429e-01, GradNorm: 13.5201, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 415/664, PPO Loss: 0.0223, PostKL: 4.604e-02, PostClipFrac: 0.198700, PostMeanRatio: 0.989921, PostRatioDev: 1.413e-01, GradNorm: 13.4378, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 498/664, PPO Loss: 0.0264, PostKL: 4.679e-02, PostClipFrac: 0.202885, PostMeanRatio: 0.990368, PostRatioDev: 1.435e-01, GradNorm: 12.4704, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 581/664, PPO Loss: 0.0302, PostKL: 4.798e-02, PostClipFrac: 0.214510, PostMeanRatio: 0.990575, PostRatioDev: 1.491e-01, GradNorm: 12.1129, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 664/664, PPO Loss: 0.0334, PostKL: 5.207e-02, PostClipFrac: 0.225748, PostMeanRatio: 0.990637, PostRatioDev: 1.543e-01, GradNorm: 12.5252, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_02.png
[OPE] Policy ACCEPTED: J_new=0.7531 > J_old=0.7531 + δ=0.0000

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 2)
[Collect] 50 episodes, success=0.700, env_return=847.88, rl_reward=0.70, steps=10000
[Data Collection] Success Rate: 0.700, EnvReturn: 847.88, RLReward: 0.70, Episodes: 50, Steps: 10000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 50 episodes remain in RL replay; IL retrain keeps 35/50 successful episodes (drops 15 failures).
[Dataset] Merged 50 episodes (10000 steps) → total 32000 steps, 160 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0140, Val Loss: 0.0749
[IL] Epoch 1/100, Loss: 0.0121, Val Loss: 0.0716
[IL] Epoch 2/100, Loss: 0.0128, Val Loss: 0.0814
[IL] Epoch 3/100, Loss: 0.0121, Val Loss: 0.0865
[IL] Epoch 4/100, Loss: 0.0123, Val Loss: 0.0705
[IL] Epoch 5/100, Loss: 0.0116, Val Loss: 0.0863
[IL] Epoch 6/100, Loss: 0.0122, Val Loss: 0.0582
[IL] Epoch 7/100, Loss: 0.0120, Val Loss: 0.0733
[IL] Epoch 8/100, Loss: 0.0119, Val Loss: 0.0844
[IL] Epoch 9/100, Loss: 0.0120, Val Loss: 0.1063
[IL] Epoch 10/100, Loss: 0.0119, Val Loss: 0.1015
[IL] Epoch 11/100, Loss: 0.0120, Val Loss: 0.0674
[IL] Epoch 12/100, Loss: 0.0117, Val Loss: 0.0839
[IL] Epoch 13/100, Loss: 0.0120, Val Loss: 0.0655
[IL] Epoch 14/100, Loss: 0.0117, Val Loss: 0.0723
[IL] Epoch 15/100, Loss: 0.0119, Val Loss: 0.0966
[IL] Epoch 16/100, Loss: 0.0118, Val Loss: 0.1172
[IL] Epoch 17/100, Loss: 0.0111, Val Loss: 0.1112
[IL] Epoch 18/100, Loss: 0.0111, Val Loss: 0.0860
[IL] Epoch 19/100, Loss: 0.0111, Val Loss: 0.1027
[IL] Epoch 20/100, Loss: 0.0112, Val Loss: 0.0775
[IL] Epoch 21/100, Loss: 0.0111, Val Loss: 0.0927
[IL] Epoch 22/100, Loss: 0.0111, Val Loss: 0.0876
[IL] Epoch 23/100, Loss: 0.0112, Val Loss: 0.0776
[IL] Epoch 24/100, Loss: 0.0109, Val Loss: 0.0760
[IL] Epoch 25/100, Loss: 0.0110, Val Loss: 0.0893
[IL] Epoch 26/100, Loss: 0.0118, Val Loss: 0.0843
[IL] Epoch 27/100, Loss: 0.0115, Val Loss: 0.0816
[IL] Epoch 28/100, Loss: 0.0108, Val Loss: 0.0772
[IL] Epoch 29/100, Loss: 0.0110, Val Loss: 0.1130
[IL] Epoch 30/100, Loss: 0.0110, Val Loss: 0.0909
[IL] Epoch 31/100, Loss: 0.0106, Val Loss: 0.0799
[IL] Epoch 32/100, Loss: 0.0108, Val Loss: 0.0794
[IL] Epoch 33/100, Loss: 0.0119, Val Loss: 0.0882
[IL] Epoch 34/100, Loss: 0.0111, Val Loss: 0.1034
[IL] Epoch 35/100, Loss: 0.0105, Val Loss: 0.0759
[IL] Epoch 36/100, Loss: 0.0111, Val Loss: 0.0886
[IL] Epoch 37/100, Loss: 0.0106, Val Loss: 0.1044
[IL] Epoch 38/100, Loss: 0.0114, Val Loss: 0.1128
[IL] Epoch 39/100, Loss: 0.0112, Val Loss: 0.0913
[IL] Epoch 40/100, Loss: 0.0105, Val Loss: 0.0859
[IL] Epoch 41/100, Loss: 0.0103, Val Loss: 0.1033
[IL] Epoch 42/100, Loss: 0.0105, Val Loss: 0.0851
[IL] Epoch 43/100, Loss: 0.0103, Val Loss: 0.0977
[IL] Epoch 44/100, Loss: 0.0105, Val Loss: 0.0875
[IL] Epoch 45/100, Loss: 0.0108, Val Loss: 0.0883
[IL] Epoch 46/100, Loss: 0.0105, Val Loss: 0.1147
[IL] Epoch 47/100, Loss: 0.0101, Val Loss: 0.1000
[IL] Epoch 48/100, Loss: 0.0111, Val Loss: 0.1065
[IL] Epoch 49/100, Loss: 0.0106, Val Loss: 0.1033
[IL] Epoch 50/100, Loss: 0.0105, Val Loss: 0.0931
[IL] Epoch 51/100, Loss: 0.0108, Val Loss: 0.0890
[IL] Epoch 52/100, Loss: 0.0100, Val Loss: 0.1216
[IL] Epoch 53/100, Loss: 0.0110, Val Loss: 0.0806
[IL] Epoch 54/100, Loss: 0.0104, Val Loss: 0.1085
[IL] Epoch 55/100, Loss: 0.0098, Val Loss: 0.1201
[IL] Epoch 56/100, Loss: 0.0100, Val Loss: 0.0933
[IL] Epoch 57/100, Loss: 0.0103, Val Loss: 0.1067
[IL] Epoch 58/100, Loss: 0.0104, Val Loss: 0.0969
[IL] Epoch 59/100, Loss: 0.0108, Val Loss: 0.0771
[IL] Epoch 60/100, Loss: 0.0101, Val Loss: 0.1029
[IL] Epoch 61/100, Loss: 0.0108, Val Loss: 0.0962
[IL] Epoch 62/100, Loss: 0.0100, Val Loss: 0.0798
[IL] Epoch 63/100, Loss: 0.0097, Val Loss: 0.1032
[IL] Epoch 64/100, Loss: 0.0102, Val Loss: 0.1016
[IL] Epoch 65/100, Loss: 0.0104, Val Loss: 0.0988
[IL] Epoch 66/100, Loss: 0.0100, Val Loss: 0.1069
[IL] Epoch 67/100, Loss: 0.0108, Val Loss: 0.1063
[IL] Epoch 68/100, Loss: 0.0103, Val Loss: 0.1168
[IL] Epoch 69/100, Loss: 0.0094, Val Loss: 0.0941
[IL] Epoch 70/100, Loss: 0.0093, Val Loss: 0.0939
[IL] Epoch 71/100, Loss: 0.0100, Val Loss: 0.1239
[IL] Epoch 72/100, Loss: 0.0101, Val Loss: 0.1833
[IL] Epoch 73/100, Loss: 0.0094, Val Loss: 0.0838
[IL] Epoch 74/100, Loss: 0.0097, Val Loss: 0.1284
[IL] Epoch 75/100, Loss: 0.0094, Val Loss: 0.1188
[IL] Epoch 76/100, Loss: 0.0096, Val Loss: 0.1211
[IL] Epoch 77/100, Loss: 0.0098, Val Loss: 0.1370
[IL] Epoch 78/100, Loss: 0.0097, Val Loss: 0.1102
[IL] Epoch 79/100, Loss: 0.0097, Val Loss: 0.1046
[IL] Epoch 80/100, Loss: 0.0098, Val Loss: 0.1426
[IL] Epoch 81/100, Loss: 0.0094, Val Loss: 0.1321
[IL] Epoch 82/100, Loss: 0.0094, Val Loss: 0.1044
[IL] Epoch 83/100, Loss: 0.0095, Val Loss: 0.1235
[IL] Epoch 84/100, Loss: 0.0096, Val Loss: 0.1497
[IL] Epoch 85/100, Loss: 0.0092, Val Loss: 0.0953
[IL] Epoch 86/100, Loss: 0.0095, Val Loss: 0.1008
[IL] Epoch 87/100, Loss: 0.0095, Val Loss: 0.1091
[IL] Epoch 88/100, Loss: 0.0096, Val Loss: 0.1089
[IL] Epoch 89/100, Loss: 0.0088, Val Loss: 0.1147
[IL] Epoch 90/100, Loss: 0.0096, Val Loss: 0.0992
[IL] Epoch 91/100, Loss: 0.0091, Val Loss: 0.1192
[IL] Epoch 92/100, Loss: 0.0092, Val Loss: 0.1184
[IL] Epoch 93/100, Loss: 0.0095, Val Loss: 0.1097
[IL] Epoch 94/100, Loss: 0.0095, Val Loss: 0.1045
[IL] Epoch 95/100, Loss: 0.0093, Val Loss: 0.1164
[IL] Epoch 96/100, Loss: 0.0093, Val Loss: 0.1212
[IL] Epoch 97/100, Loss: 0.0088, Val Loss: 0.1105
[IL] Epoch 98/100, Loss: 0.0092, Val Loss: 0.1180
[IL] Epoch 99/100, Loss: 0.0088, Val Loss: 0.1080
test_mean_score: 0.87
[IL] Eval - Success Rate: 0.870
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_02.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/best_ddim.ckpt
[BestDDIM] Updated best DDIM checkpoint from 0.8000 to 0.8700 at offline_iter_2.
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_2.ckpt

================================================================================
               OFFLINE RL ITERATION 4/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 3)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 30687 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-41.20499 | val=0.00193 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-48.89771 | val=0.00161 | no-improve=0/5
[TransitionModel] Epoch   40 | train=-55.50157 | val=0.00149 | no-improve=1/5
[TransitionModel] Training complete. Elites=[1, 3, 0, 4, 2], val_loss=0.00146
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_03.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_03.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 3)
[IQL] Step 120/1000, V Loss: 0.0002, Q Loss: 0.0040
[IQL] Step 240/1000, V Loss: 0.0001, Q Loss: 0.0040
[IQL] Step 360/1000, V Loss: 0.0001, Q Loss: 0.0041
[IQL] Step 480/1000, V Loss: 0.0001, Q Loss: 0.0041
[IQL] Step 600/1000, V Loss: 0.0001, Q Loss: 0.0041
[IQL] Step 720/1000, V Loss: 0.0001, Q Loss: 0.0042
[IQL] Step 840/1000, V Loss: 0.0001, Q Loss: 0.0043
[IQL] Step 960/1000, V Loss: 0.0001, Q Loss: 0.0043
[IQL] Step 1000/1000, V Loss: 0.0001, Q Loss: 0.0045
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_03.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_03.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 3)
[OPE] AM-Q eval uses episode-start states: 159 unique episodes, 100 batch(es) x 256.
[OPE] Behavior policy value J_old = 0.6081
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 4.00e-06
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 120 mini-batches, 30687 samples, raw advantage mean=-0.0169, std=0.0041
[Offline RL] Cap PPO steps: requested 1000, using 960 (8 passes over 120 fixed PPO batches).
[Offline RL] Step 120/960, PPO Loss: -0.0113, PostKL: 1.762e-02, PostClipFrac: 0.152651, PostMeanRatio: 0.998735, PostRatioDev: 1.148e-01, GradNorm: 12.8221, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 240/960, PPO Loss: 0.0025, PostKL: 2.218e-02, PostClipFrac: 0.151469, PostMeanRatio: 0.998692, PostRatioDev: 1.163e-01, GradNorm: 12.9340, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 360/960, PPO Loss: 0.0095, PostKL: 2.337e-02, PostClipFrac: 0.160553, PostMeanRatio: 0.998957, PostRatioDev: 1.212e-01, GradNorm: 12.0969, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 480/960, PPO Loss: 0.0159, PostKL: 2.423e-02, PostClipFrac: 0.162460, PostMeanRatio: 1.000292, PostRatioDev: 1.231e-01, GradNorm: 11.8975, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 600/960, PPO Loss: 0.0203, PostKL: 2.660e-02, PostClipFrac: 0.173879, PostMeanRatio: 1.001425, PostRatioDev: 1.302e-01, GradNorm: 12.5905, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 720/960, PPO Loss: 0.0245, PostKL: 2.791e-02, PostClipFrac: 0.179870, PostMeanRatio: 1.002174, PostRatioDev: 1.334e-01, GradNorm: 11.5968, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 840/960, PPO Loss: 0.0277, PostKL: 3.062e-02, PostClipFrac: 0.189433, PostMeanRatio: 1.003477, PostRatioDev: 1.393e-01, GradNorm: 11.7380, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 960/960, PPO Loss: 0.0312, PostKL: 3.178e-02, PostClipFrac: 0.194868, PostMeanRatio: 1.004807, PostRatioDev: 1.423e-01, GradNorm: 11.4060, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_03.png
[OPE] Policy ACCEPTED: J_new=0.6082 > J_old=0.6081 + δ=0.0000

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 3)
[Collect] 50 episodes, success=0.840, env_return=938.87, rl_reward=0.84, steps=10000
[Data Collection] Success Rate: 0.840, EnvReturn: 938.87, RLReward: 0.84, Episodes: 50, Steps: 10000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 50 episodes remain in RL replay; IL retrain keeps 42/50 successful episodes (drops 8 failures).
[Dataset] Merged 50 episodes (10000 steps) → total 42000 steps, 210 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0110, Val Loss: 0.0911
[IL] Epoch 1/100, Loss: 0.0107, Val Loss: 0.0973
[IL] Epoch 2/100, Loss: 0.0108, Val Loss: 0.0720
[IL] Epoch 3/100, Loss: 0.0103, Val Loss: 0.0625
[IL] Epoch 4/100, Loss: 0.0102, Val Loss: 0.0972
[IL] Epoch 5/100, Loss: 0.0100, Val Loss: 0.0928
[IL] Epoch 6/100, Loss: 0.0098, Val Loss: 0.1002
[IL] Epoch 7/100, Loss: 0.0101, Val Loss: 0.0936
[IL] Epoch 8/100, Loss: 0.0102, Val Loss: 0.0951
[IL] Epoch 9/100, Loss: 0.0100, Val Loss: 0.1111
[IL] Epoch 10/100, Loss: 0.0095, Val Loss: 0.0847
[IL] Epoch 11/100, Loss: 0.0097, Val Loss: 0.0955
[IL] Epoch 12/100, Loss: 0.0096, Val Loss: 0.1114
[IL] Epoch 13/100, Loss: 0.0100, Val Loss: 0.0941
[IL] Epoch 14/100, Loss: 0.0097, Val Loss: 0.0750
[IL] Epoch 15/100, Loss: 0.0097, Val Loss: 0.0945
[IL] Epoch 16/100, Loss: 0.0097, Val Loss: 0.1189
[IL] Epoch 17/100, Loss: 0.0099, Val Loss: 0.0928
[IL] Epoch 18/100, Loss: 0.0092, Val Loss: 0.1145
[IL] Epoch 19/100, Loss: 0.0094, Val Loss: 0.1095
[IL] Epoch 20/100, Loss: 0.0096, Val Loss: 0.1112
[IL] Epoch 21/100, Loss: 0.0096, Val Loss: 0.0795
[IL] Epoch 22/100, Loss: 0.0098, Val Loss: 0.0952
[IL] Epoch 23/100, Loss: 0.0093, Val Loss: 0.0847
[IL] Epoch 24/100, Loss: 0.0094, Val Loss: 0.0916
[IL] Epoch 25/100, Loss: 0.0093, Val Loss: 0.0976
[IL] Epoch 26/100, Loss: 0.0094, Val Loss: 0.1131
[IL] Epoch 27/100, Loss: 0.0090, Val Loss: 0.1195
[IL] Epoch 28/100, Loss: 0.0089, Val Loss: 0.0943
[IL] Epoch 29/100, Loss: 0.0094, Val Loss: 0.1025
[IL] Epoch 30/100, Loss: 0.0090, Val Loss: 0.0975
[IL] Epoch 31/100, Loss: 0.0093, Val Loss: 0.1259
[IL] Epoch 32/100, Loss: 0.0092, Val Loss: 0.1228
[IL] Epoch 33/100, Loss: 0.0091, Val Loss: 0.0982
[IL] Epoch 34/100, Loss: 0.0088, Val Loss: 0.1059
[IL] Epoch 35/100, Loss: 0.0096, Val Loss: 0.1134
[IL] Epoch 36/100, Loss: 0.0093, Val Loss: 0.1135
[IL] Epoch 37/100, Loss: 0.0092, Val Loss: 0.1151
[IL] Epoch 38/100, Loss: 0.0092, Val Loss: 0.0884
[IL] Epoch 39/100, Loss: 0.0088, Val Loss: 0.0976
[IL] Epoch 40/100, Loss: 0.0089, Val Loss: 0.1316
[IL] Epoch 41/100, Loss: 0.0091, Val Loss: 0.1280
[IL] Epoch 42/100, Loss: 0.0090, Val Loss: 0.1185
[IL] Epoch 43/100, Loss: 0.0087, Val Loss: 0.1232
[IL] Epoch 44/100, Loss: 0.0086, Val Loss: 0.1149
[IL] Epoch 45/100, Loss: 0.0089, Val Loss: 0.0993
[IL] Epoch 46/100, Loss: 0.0085, Val Loss: 0.1084
[IL] Epoch 47/100, Loss: 0.0088, Val Loss: 0.1302
[IL] Epoch 48/100, Loss: 0.0086, Val Loss: 0.1231
[IL] Epoch 49/100, Loss: 0.0084, Val Loss: 0.1089
[IL] Epoch 50/100, Loss: 0.0090, Val Loss: 0.1227
[IL] Epoch 51/100, Loss: 0.0087, Val Loss: 0.1237
[IL] Epoch 52/100, Loss: 0.0086, Val Loss: 0.1204
[IL] Epoch 53/100, Loss: 0.0089, Val Loss: 0.1259
[IL] Epoch 54/100, Loss: 0.0087, Val Loss: 0.1015
[IL] Epoch 55/100, Loss: 0.0084, Val Loss: 0.1213
[IL] Epoch 56/100, Loss: 0.0087, Val Loss: 0.1321
[IL] Epoch 57/100, Loss: 0.0085, Val Loss: 0.1072
[IL] Epoch 58/100, Loss: 0.0084, Val Loss: 0.1558
[IL] Epoch 59/100, Loss: 0.0085, Val Loss: 0.1033
[IL] Epoch 60/100, Loss: 0.0087, Val Loss: 0.1254
[IL] Epoch 61/100, Loss: 0.0085, Val Loss: 0.1236
[IL] Epoch 62/100, Loss: 0.0087, Val Loss: 0.1111
[IL] Epoch 63/100, Loss: 0.0084, Val Loss: 0.1333
[IL] Epoch 64/100, Loss: 0.0081, Val Loss: 0.1179
[IL] Epoch 65/100, Loss: 0.0084, Val Loss: 0.1151
[IL] Epoch 66/100, Loss: 0.0080, Val Loss: 0.1467
[IL] Epoch 67/100, Loss: 0.0086, Val Loss: 0.1394
[IL] Epoch 68/100, Loss: 0.0083, Val Loss: 0.1360
[IL] Epoch 69/100, Loss: 0.0083, Val Loss: 0.1092
[IL] Epoch 70/100, Loss: 0.0083, Val Loss: 0.1276
[IL] Epoch 71/100, Loss: 0.0085, Val Loss: 0.1418
[IL] Epoch 72/100, Loss: 0.0083, Val Loss: 0.1277
[IL] Epoch 73/100, Loss: 0.0083, Val Loss: 0.1112
[IL] Epoch 74/100, Loss: 0.0081, Val Loss: 0.1277
[IL] Epoch 75/100, Loss: 0.0080, Val Loss: 0.0965
[IL] Epoch 76/100, Loss: 0.0081, Val Loss: 0.1571
[IL] Epoch 77/100, Loss: 0.0078, Val Loss: 0.1473
[IL] Epoch 78/100, Loss: 0.0081, Val Loss: 0.1147
[IL] Epoch 79/100, Loss: 0.0079, Val Loss: 0.1447
[IL] Epoch 80/100, Loss: 0.0081, Val Loss: 0.1368
[IL] Epoch 81/100, Loss: 0.0080, Val Loss: 0.1432
[IL] Epoch 82/100, Loss: 0.0082, Val Loss: 0.1571
[IL] Epoch 83/100, Loss: 0.0080, Val Loss: 0.1333
[IL] Epoch 84/100, Loss: 0.0080, Val Loss: 0.1187
[IL] Epoch 85/100, Loss: 0.0080, Val Loss: 0.1419
[IL] Epoch 86/100, Loss: 0.0079, Val Loss: 0.1179
[IL] Epoch 87/100, Loss: 0.0079, Val Loss: 0.1156
[IL] Epoch 88/100, Loss: 0.0081, Val Loss: 0.1359
[IL] Epoch 89/100, Loss: 0.0078, Val Loss: 0.0921
[IL] Epoch 90/100, Loss: 0.0077, Val Loss: 0.1172
[IL] Epoch 91/100, Loss: 0.0081, Val Loss: 0.0944
[IL] Epoch 92/100, Loss: 0.0080, Val Loss: 0.1270
[IL] Epoch 93/100, Loss: 0.0079, Val Loss: 0.1560
[IL] Epoch 94/100, Loss: 0.0076, Val Loss: 0.1577
[IL] Epoch 95/100, Loss: 0.0077, Val Loss: 0.0840
[IL] Epoch 96/100, Loss: 0.0080, Val Loss: 0.1178
[IL] Epoch 97/100, Loss: 0.0078, Val Loss: 0.1454
[IL] Epoch 98/100, Loss: 0.0076, Val Loss: 0.1282
[IL] Epoch 99/100, Loss: 0.0077, Val Loss: 0.1295
test_mean_score: 0.84
[IL] Eval - Success Rate: 0.840
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/il_loss_retrain_iter_03.png
[Checkpoint] Saved to /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints/offline_iter_3.ckpt

================================================================================
               OFFLINE RL ITERATION 5/10
================================================================================


[RL100Trainer] Line 6 — Training Transition Model T_θ (Iteration 4)

[TransitionModel] Encoding dataset for transition model training...
[TransitionModel] Dataset: 40337 samples, input_dim=288, target_dim=257
[TransitionModel] Epoch    0 | train=-56.03929 | val=0.00165 | no-improve=0/5
[TransitionModel] Epoch   20 | train=-66.16281 | val=0.00152 | no-improve=0/5
[TransitionModel] Epoch   40 | train=-74.54797 | val=0.00145 | no-improve=0/5
[TransitionModel] Training complete. Elites=[3, 5, 4, 2, 1], val_loss=0.00143
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_train_loss_iter_04.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/transition_val_loss_iter_04.png

[RL100Trainer] Phase 2a: Training IQL Critics (Iteration 4)
[IQL] Step 158/1000, V Loss: 0.0001, Q Loss: 0.0045
[IQL] Step 316/1000, V Loss: 0.0002, Q Loss: 0.0046
[IQL] Step 474/1000, V Loss: 0.0001, Q Loss: 0.0046
[IQL] Step 632/1000, V Loss: 0.0001, Q Loss: 0.0046
[IQL] Step 790/1000, V Loss: 0.0001, Q Loss: 0.0047
[IQL] Step 948/1000, V Loss: 0.0001, Q Loss: 0.0048
[IQL] Step 1000/1000, V Loss: 0.0001, Q Loss: 0.0059
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_q_loss_iter_04.png
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/iql_v_loss_iter_04.png

[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration 4)
[OPE] AM-Q eval uses episode-start states: 209 unique episodes, 100 batch(es) x 256.
[OPE] Behavior policy value J_old = 0.4986
[RL PPO] Reset policy optimizer state and set LR: 1.00e-04 → 4.00e-06
[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...
[Offline RL] Fixed PPO buffer ready: 158 mini-batches, 40337 samples, raw advantage mean=-0.0048, std=0.0059
[Offline RL] Step 158/1000, PPO Loss: -0.0086, PostKL: 1.265e-02, PostClipFrac: 0.115814, PostMeanRatio: 0.999610, PostRatioDev: 9.545e-02, GradNorm: 12.9940, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 316/1000, PPO Loss: 0.0024, PostKL: 1.939e-02, PostClipFrac: 0.123860, PostMeanRatio: 1.003049, PostRatioDev: 1.034e-01, GradNorm: 12.2110, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 474/1000, PPO Loss: 0.0090, PostKL: 2.143e-02, PostClipFrac: 0.129779, PostMeanRatio: 1.005311, PostRatioDev: 1.080e-01, GradNorm: 12.2516, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 632/1000, PPO Loss: 0.0137, PostKL: 2.301e-02, PostClipFrac: 0.139790, PostMeanRatio: 1.008499, PostRatioDev: 1.145e-01, GradNorm: 12.5375, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 790/1000, PPO Loss: 0.0182, PostKL: 2.574e-02, PostClipFrac: 0.145804, PostMeanRatio: 1.011547, PostRatioDev: 1.196e-01, GradNorm: 12.4184, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 948/1000, PPO Loss: 0.0211, PostKL: 2.602e-02, PostClipFrac: 0.152883, PostMeanRatio: 1.013096, PostRatioDev: 1.235e-01, GradNorm: 12.2577, Reg Loss: 0.0000, CD Loss: 0.0000
[Offline RL] Step 1000/1000, PPO Loss: 0.0222, PostKL: 2.831e-02, PostClipFrac: 0.161049, PostMeanRatio: 1.016330, PostRatioDev: 1.295e-01, GradNorm: 11.8956, Reg Loss: 0.0000, CD Loss: 0.0000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/ppo_loss_iter_04.png
[OPE] Policy REJECTED: J_new=0.4986 ≤ J_old=0.4986 + δ=0.0000. Rolling back to behavior policy.

[RL100Trainer] Phase 2c: Collecting New Data (Iteration 4)
[Collect] 50 episodes, success=0.680, env_return=857.92, rl_reward=0.68, steps=10000
[Data Collection] Success Rate: 0.680, EnvReturn: 857.92, RLReward: 0.68, Episodes: 50, Steps: 10000
[Plot] Saved: /nfs_global/S/yangrongzheng/RL100/3D-Diffusion-Policy/3D-Diffusion-Policy/plots/offline_collection_success_rate.png
[Dataset] offline collection: all 50 episodes remain in RL replay; IL retrain keeps 34/50 successful episodes (drops 16 failures).
[Dataset] Merged 50 episodes (10000 steps) → total 52000 steps, 260 episodes

[RL100Trainer] Retraining IL on merged dataset...

============================================================
[RL100Trainer] Phase 1: Imitation Learning
============================================================

[BC] Reset policy optimizer state and set LR: 1.00e-04
[IL] Epoch 0/100, Loss: 0.0087, Val Loss: 0.0802
[IL] Epoch 1/100, Loss: 0.0081, Val Loss: 0.0971
[IL] Epoch 2/100, Loss: 0.0082, Val Loss: 0.1184
[IL] Epoch 3/100, Loss: 0.0083, Val Loss: 0.1127
[IL] Epoch 4/100, Loss: 0.0079, Val Loss: 0.1208
[IL] Epoch 5/100, Loss: 0.0079, Val Loss: 0.1310
[IL] Epoch 6/100, Loss: 0.0078, Val Loss: 0.1271
[IL] Epoch 7/100, Loss: 0.0082, Val Loss: 0.1165
[IL] Epoch 8/100, Loss: 0.0078, Val Loss: 0.1483
[IL] Epoch 9/100, Loss: 0.0080, Val Loss: 0.1307
[IL] Epoch 10/100, Loss: 0.0079, Val Loss: 0.1134
[IL] Epoch 11/100, Loss: 0.0079, Val Loss: 0.1110
[IL] Epoch 12/100, Loss: 0.0080, Val Loss: 0.1006
[IL] Epoch 13/100, Loss: 0.0077, Val Loss: 0.1163
[IL] Epoch 14/100, Loss: 0.0079, Val Loss: 0.1155
[IL] Epoch 15/100, Loss: 0.0076, Val Loss: 0.1212
[IL] Epoch 16/100, Loss: 0.0077, Val Loss: 0.1241
[IL] Epoch 17/100, Loss: 0.0078, Val Loss: 0.1046
[IL] Epoch 18/100, Loss: 0.0079, Val Loss: 0.1180
[IL] Epoch 19/100, Loss: 0.0077, Val Loss: 0.1600
[IL] Epoch 20/100, Loss: 0.0078, Val Loss: 0.1456
[IL] Epoch 21/100, Loss: 0.0077, Val Loss: 0.1286
[IL] Epoch 22/100, Loss: 0.0076, Val Loss: 0.1173
[IL] Epoch 23/100, Loss: 0.0075, Val Loss: 0.1223
[IL] Epoch 24/100, Loss: 0.0074, Val Loss: 0.1179
[IL] Epoch 25/100, Loss: 0.0074, Val Loss: 0.1427
[IL] Epoch 26/100, Loss: 0.0078, Val Loss: 0.1660
[IL] Epoch 27/100, Loss: 0.0073, Val Loss: 0.1243
[IL] Epoch 28/100, Loss: 0.0074, Val Loss: 0.1154
[IL] Epoch 29/100, Loss: 0.0078, Val Loss: 0.1184
[IL] Epoch 30/100, Loss: 0.0073, Val Loss: 0.1373
[IL] Epoch 31/100, Loss: 0.0074, Val Loss: 0.1448
[IL] Epoch 32/100, Loss: 0.0077, Val Loss: 0.1567
[IL] Epoch 33/100, Loss: 0.0072, Val Loss: 0.1270
[IL] Epoch 34/100, Loss: 0.0075, Val Loss: 0.1309
[IL] Epoch 35/100, Loss: 0.0075, Val Loss: 0.1103
[IL] Epoch 36/100, Loss: 0.0074, Val Loss: 0.1174
[IL] Epoch 37/100, Loss: 0.0072, Val Loss: 0.1274
[IL] Epoch 38/100, Loss: 0.0074, Val Loss: 0.1360
[IL] Epoch 39/100, Loss: 0.0074, Val Loss: 0.1575
[IL] Epoch 40/100, Loss: 0.0073, Val Loss: 0.1411
[IL] Epoch 41/100, Loss: 0.0074, Val Loss: 0.1221
[IL] Epoch 42/100, Loss: 0.0072, Val Loss: 0.1241
[IL] Epoch 43/100, Loss: 0.0073, Val Loss: 0.1396
[IL] Epoch 44/100, Loss: 0.0073, Val Loss: 0.1407
[IL] Epoch 45/100, Loss: 0.0070, Val Loss: 0.1457
[IL] Epoch 46/100, Loss: 0.0071, Val Loss: 0.1352
[IL] Epoch 47/100, Loss: 0.0073, Val Loss: 0.1285
[IL] Epoch 48/100, Loss: 0.0073, Val Loss: 0.1602
[IL] Epoch 49/100, Loss: 0.0074, Val Loss: 0.0850
[IL] Epoch 50/100, Loss: 0.0070, Val Loss: 0.1094
[IL] Epoch 51/100, Loss: 0.0069, Val Loss: 0.1333
[IL] Epoch 52/100, Loss: 0.0068, Val Loss: 0.1230
[IL] Epoch 53/100, Loss: 0.0070, Val Loss: 0.1286
[IL] Epoch 54/100, Loss: 0.0071, Val Loss: 0.1578
[IL] Epoch 55/100, Loss: 0.0070, Val Loss: 0.1345
[IL] Epoch 56/100, Loss: 0.0071, Val Loss: 0.1651
[IL] Epoch 57/100, Loss: 0.0071, Val Loss: 0.1279
[IL] Epoch 58/100, Loss: 0.0071, Val Loss: 0.1265
[IL] Epoch 59/100, Loss: 0.0071, Val Loss: 0.1338
[IL] Epoch 60/100, Loss: 0.0067, Val Loss: 0.1381
[IL] Epoch 61/100, Loss: 0.0066, Val Loss: 0.1713
[IL] Epoch 62/100, Loss: 0.0070, Val Loss: 0.1255
[IL] Epoch 63/100, Loss: 0.0069, Val Loss: 0.1590
[IL] Epoch 64/100, Loss: 0.0071, Val Loss: 0.1298
[IL] Epoch 65/100, Loss: 0.0071, Val Loss: 0.1404
[IL] Epoch 66/100, Loss: 0.0069, Val Loss: 0.1328
[IL] Epoch 67/100, Loss: 0.0066, Val Loss: 0.1555
[IL] Epoch 68/100, Loss: 0.0067, Val Loss: 0.1539
[IL] Epoch 69/100, Loss: 0.0069, Val Loss: 0.1036
[IL] Epoch 70/100, Loss: 0.0066, Val Loss: 0.1191
[IL] Epoch 71/100, Loss: 0.0068, Val Loss: 0.1509
[IL] Epoch 72/100, Loss: 0.0069, Val Loss: 0.1270
[IL] Epoch 73/100, Loss: 0.0069, Val Loss: 0.1297
[IL] Epoch 74/100, Loss: 0.0071, Val Loss: 0.1153
[IL] Epoch 75/100, Loss: 0.0067, Val Loss: 0.0981
[IL] Epoch 76/100, Loss: 0.0068, Val Loss: 0.1426
[IL] Epoch 77/100, Loss: 0.0065, Val Loss: 0.1271
[IL] Epoch 78/100, Loss: 0.0066, Val Loss: 0.1441
[IL] Epoch 79/100, Loss: 0.0067, Val Loss: 0.1379
[IL] Epoch 80/100, Loss: 0.0067, Val Loss: 0.1427
[IL] Epoch 81/100, Loss: 0.0066, Val Loss: 0.1289
[IL] Epoch 82/100, Loss: 0.0068, Val Loss: 0.1718
[IL] Epoch 83/100, Loss: 0.0066, Val Loss: 0.1702
[IL] Epoch 84/100, Loss: 0.0067, Val Loss: 0.1469
[IL] Epoch 85/100, Loss: 0.0066, Val Loss: 0.0931
[IL] Epoch 86/100, Loss: 0.0065, Val Loss: 0.1361
[IL] Epoch 87/100, Loss: 0.0067, Val Loss: 0.1209
[IL] Epoch 88/100, Loss: 0.0065, Val Loss: 0.1557
[IL] Epoch 89/100, Loss: 0.0065, Val Loss: 0.1380
[IL] Epoch 90/100, Loss: 0.0065, Val Loss: 0.1430
[IL] Epoch 91/100, Loss: 0.0063, Val Loss: 0.1090
[IL] Epoch 92/100, Loss: 0.0064, Val Loss: 0.1575
[IL] Epoch 93/100, Loss: 0.0066, Val Loss: 0.1807
[IL] Epoch 94/100, Loss: 0.0066, Val Loss: 0.1752
[IL] Epoch 95/100, Loss: 0.0063, Val Loss: 0.1312
[IL] Epoch 96/100, Loss: 0.0065, Val Loss: 0.1846
[IL] Epoch 97/100, Loss: 0.0062, Val Loss: 0.1521
[IL] Epoch 98/100, Loss: 0.0065, Val Loss: 0.1716
[IL] Epoch 99/100, Loss: 0.0064, Val Loss: 0.1546
Extracting GPU stats logs using atop has been completed on r8l40s-a05.
Logs are being saved to: /nfs_global/S/yangrongzheng/atop-743903-r8l40s-a05-gpustat.log
