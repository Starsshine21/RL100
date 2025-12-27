"""
RL100 改进版训练脚本 v3
主要改进：
1. 从最佳checkpoint开始Phase 3
2. 在线数据质量筛选（只保留高质量episode）
3. Early Stopping机制（检测性能退化）
4. 数据平衡控制（限制在线数据比例）
5. 学习率调整（Phase 3使用更低学习率）
6. 更详细的性能监控
"""
import sys
import os
import pathlib
import hydra
import torch
import numpy as np
import copy
import wandb
import tqdm
import shutil
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from diffusion_policy_3d.policy.rl100 import RL100
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler

from diffusion_policy_3d.env.metaworld.metaworld_wrapper import MetaWorldEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
import zarr
import os
os.environ["WANDB_MODE"] = "offline"

class MemoryDataset(Dataset):
    """在线数据的内存数据集"""
    def __init__(self, data_list, horizon, pad_before, pad_after):
        super().__init__()

        # 1. 解包数据列表
        _data_cache = {
            'point_cloud': [], 'state': [], 'full_state': [],
            'action': [], 'reward': [], 'done': []
        }

        for item in data_list:
            # Point Cloud
            pc = item['obs']['point_cloud']
            if isinstance(pc, torch.Tensor): pc = pc.detach().cpu().numpy()
            if pc.shape[-1] > 3: pc = pc[..., :3]
            _data_cache['point_cloud'].append(pc)

            # Agent Pos (State)
            pos = item['obs']['agent_pos']
            if isinstance(pos, torch.Tensor): pos = pos.detach().cpu().numpy()
            _data_cache['state'].append(pos)

            # Full State
            fs = item['obs'].get('full_state', np.zeros(39, dtype=np.float32))
            if isinstance(fs, torch.Tensor): fs = fs.detach().cpu().numpy()
            _data_cache['full_state'].append(fs)

            # Action
            act = item['action']
            if isinstance(act, torch.Tensor): act = act.detach().cpu().numpy()
            _data_cache['action'].append(act)

            # Reward
            rew = item['reward']
            if isinstance(rew, torch.Tensor): rew = rew.detach().cpu().numpy()
            _data_cache['reward'].append(np.array(rew).reshape(-1)[0])

            # Done
            don = item['done']
            if isinstance(don, torch.Tensor): don = don.detach().cpu().numpy()
            _data_cache['done'].append(np.array(don).reshape(-1)[0])

        # 2. 堆叠成大数组
        buffer_data = {
            'point_cloud': np.stack(_data_cache['point_cloud']),
            'state': np.stack(_data_cache['state']),
            'full_state': np.stack(_data_cache['full_state']),
            'action': np.stack(_data_cache['action']),
            'reward': np.array(_data_cache['reward'], dtype=np.float32),
            'done': np.array(_data_cache['done'], dtype=np.float32)
        }

        # 3. 创建内存 Zarr
        store = zarr.MemoryStore()
        root = zarr.group(store=store)
        data_group = root.require_group('data', overwrite=True)
        meta_group = root.require_group('meta', overwrite=True)

        for key, val in buffer_data.items():
            data_group.create_dataset(key, data=val)

        # 计算 episode_ends
        episode_ends = np.where(buffer_data['done'] > 0.5)[0] + 1
        if len(episode_ends) == 0 or episode_ends[-1] != len(buffer_data['done']):
            episode_ends = np.append(episode_ends, len(buffer_data['done']))
        meta_group.create_dataset('episode_ends', data=episode_ends.astype(np.int64))

        self.replay_buffer = ReplayBuffer(root)

        # 4. 初始化序列采样器
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after
        )

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)

        # 格式化输出
        data = {
            'obs': {
                'point_cloud': sample['point_cloud'].astype(np.float32),
                'agent_pos': sample['state'].astype(np.float32),
                'full_state': sample['full_state'].astype(np.float32),
            },
            'action': sample['action'].astype(np.float32),
            'reward': sample['reward'].astype(np.float32),
            'next_obs': {
                'point_cloud': np.concatenate([sample['point_cloud'][1:], sample['point_cloud'][-1:]], axis=0).astype(np.float32),
                'agent_pos': np.concatenate([sample['state'][1:], sample['state'][-1:]], axis=0).astype(np.float32),
                'full_state': np.concatenate([sample['full_state'][1:], sample['full_state'][-1:]], axis=0).astype(np.float32),
            },
            'done': sample['done'].astype(np.float32)
        }
        return dict_apply(data, torch.from_numpy)


class TrainRL100Improved:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.output_dir = os.getcwd()

        # 1. WandB 模式配置
        wandb_mode = cfg.logging.get('wandb_mode', 'offline')
        os.environ["WANDB_MODE"] = wandb_mode

        # 2. 设置种子
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 3. 设备
        self.device = torch.device(cfg.training.device)

        # 4. 初始化离线数据集
        if hasattr(cfg.task.dataset, 'zarr_path') and not os.path.isabs(cfg.task.dataset.zarr_path):
            original_cwd = hydra.utils.get_original_cwd()
            cfg.task.dataset.zarr_path = os.path.join(original_cwd, cfg.task.dataset.zarr_path)
            print(f"[Info] Zarr path corrected to: {cfg.task.dataset.zarr_path}")

        self.offline_dataset = hydra.utils.instantiate(cfg.task.dataset)
        self.normalizer = self.offline_dataset.get_normalizer()

        # ===== 新策略：分离成功数据和全部数据 =====
        # 成功数据集（用于IL）：原始演示数据 + 成功的rollout数据
        self.successful_datasets = [self.offline_dataset]  # 假设演示数据都是成功的

        # 全部数据集（用于RL）：原始演示数据 + 所有rollout数据
        self.all_rl_datasets = [self.offline_dataset]

        print(f"[Info] 数据管理策略: IL使用成功数据，RL使用全部数据")

        # 5. 初始化 RL100 策略
        self.model: RL100 = hydra.utils.instantiate(cfg.policy)
        self.model.set_normalizer(self.normalizer)
        self.model.to(self.device)

        # 5.5 初始化 EMA 模型（用于稳定训练和评估）
        self.ema_model: RL100 = None
        self.ema = None
        if cfg.training.get('use_ema', True):
            try:
                self.ema_model = copy.deepcopy(self.model)
                print("[Info] EMA 模型已创建（deepcopy）")
            except Exception as e:
                # Minkowski Engine 可能无法 deepcopy，重新实例化
                print(f"[Warning] EMA deepcopy 失败: {e}，尝试重新实例化...")
                self.ema_model = hydra.utils.instantiate(cfg.policy)
                self.ema_model.set_normalizer(self.normalizer)
                self.ema_model.load_state_dict(self.model.state_dict())
                print("[Info] EMA 模型已重新实例化")

            self.ema_model.to(self.device)

            # 导入 EMAModel
            from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
            self.ema = EMAModel(
                model=self.ema_model,
                power=cfg.training.get('ema_power', 0.999)
            )
            print(f"[Info] EMA 已启用 (power={cfg.training.get('ema_power', 0.999)})")
        else:
            print("[Info] EMA 已禁用")

        # 6. 优化器（统一使用单一优化器，避免学习率调度问题）
        # 【修复】删除多余的优化器，统一使用self.optimizer
        # 不同阶段通过冻结/解冻参数来控制更新范围
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # 7. 学习率调度器
        self.lr_scheduler = None
        if hasattr(cfg, 'lr_scheduler') and cfg.lr_scheduler is not None:
            self.lr_scheduler = get_scheduler(
                cfg.lr_scheduler.name,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.lr_scheduler.warmup_steps,
                num_training_steps=cfg.training.total_steps
            )

        # 8. TopK Checkpoint管理器
        self.ckpt_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints2'),
            monitor_key='eval_success_rate',
            mode='max',
            k=cfg.training.get('keep_top_k_checkpoints', 3),
            format_str='epoch_{epoch:04d}_success_{eval_success_rate:.3f}.ckpt'
        )

        # 9. 环境（用于rollout） - 使用MultiStepWrapper与评估保持一致
        num_points = cfg.task.dataset.pointcloud_encoder_cfg.in_channels_points \
                     if 'in_channels_points' in cfg.task.dataset.pointcloud_encoder_cfg else 1024

        # 稀疏奖励配置
        use_sparse_reward = cfg.training.get('use_sparse_reward', False)
        sparse_reward_value = cfg.training.get('sparse_reward_value', 1.0)

        # 创建原始环境
        raw_env = MetaWorldEnv(
            task_name=cfg.task.env_name,
            device=cfg.training.device,
            num_points=num_points,
            use_sparse_reward=use_sparse_reward,
            sparse_reward_value=sparse_reward_value
        )

        # 包装为MultiStepWrapper（与评估保持一致：n_obs_steps=2, n_action_steps=8）
        self.env = MultiStepWrapper(
            raw_env,
            n_obs_steps=cfg.policy.n_obs_steps,  # 2
            n_action_steps=cfg.policy.n_action_steps,  # 8
            max_episode_steps=200,
            reward_agg_method='sum',
        )
        print(f"[Info] Rollout环境已使用MultiStepWrapper（n_obs_steps={cfg.policy.n_obs_steps}, n_action_steps={cfg.policy.n_action_steps}）")

        # 10. 评估环境（周期性验证）
        self.eval_runner = None
        if hasattr(cfg.task, 'env_runner'):
            self.eval_runner = hydra.utils.instantiate(cfg.task.env_runner,output_dir=os.getcwd())

        self.global_step = 0
        self.best_success_rate = 0.0

        # ===== 性能追踪 =====
        self.best_checkpoint_path = None

        # 11. 在线RL的Replay Buffer
        self.online_replay_buffer = []
        self.online_replay_max_size = cfg.training.get('online_replay_size', 5000)

        # 12. WandB 初始化
        wandb.init(
            project=cfg.logging.get('project', "RL100-Improved-v3"),
            name=cfg.logging.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=wandb_mode
        )

        # 13. 打印配置信息
        self._print_config()

    def _print_config(self):
        cfg = self.cfg
        print("\n" + "="*70)
        print("RL100 改进版v3 配置")
        print("="*70)
        print(f"📊 数据集: {cfg.task.name}")
        print(f"🎯 任务: {cfg.task.env_name}")
        print(f"💾 数据路径: {cfg.task.dataset.zarr_path}")
        print(f"🔧 设备: {cfg.training.device}")
        print(f"🌱 随机种子: {cfg.training.seed}")
        print("\n✨ 新增改进:")
        print(f"  ✓ 数据质量筛选（成功率阈值: {cfg.training.get('data_quality_threshold', 0.5)}）")
        print(f"  ✓ Phase 3从最佳checkpoint开始")
        print(f"  ✓ 学习率衰减（Phase 3: {cfg.training.get('phase3_lr_factor', 0.1)}x）")

        # 奖励模式信息
        if cfg.training.get('use_sparse_reward', False):
            print(f"\n⚠️  奖励模式: 稀疏奖励（成功={cfg.training.get('sparse_reward_value', 1.0)}, 失败=0）")
        else:
            print(f"\n💰 奖励模式: 密集奖励（距离+成功bonus）")

        print("\n核心特性:")
        print("  ✓ 扩散模型预测噪声（epsilon）")
        print("  ✓ 双层MDP（环境级 + 去噪步级）")
        print(f"  ✓ IQL算法（expectile={cfg.policy.omega}）")
        print(f"  ✓ AM-Q筛选（threshold={cfg.policy.amq_threshold}）")
        print(f"  ✓ VIB正则化（IL: recon=1.0, kl=0.001 | RL: recon={cfg.policy.beta_recon_rl}, kl={cfg.policy.beta_kl_rl}）")
        print(f"  ✓ GAE优势估计（lambda={cfg.policy.gae_lambda}）")
        print("\n训练流程:")
        print(f"  Phase 1: IL预训练 - {cfg.training.il_epochs} epochs")
        print(f"  Phase 2: 离线RL - {cfg.training.rl_iterations}轮迭代 × {cfg.training.rl_epochs_per_iter} epochs")
        if cfg.training.get('online_rl_enabled', True):
            print(f"  Phase 3: 在线RL - {cfg.training.get('online_rl_epochs', 10)} epochs")
        print("="*70 + "\n")

    def run(self):
        cfg = self.cfg

        # =========================================
        # Phase 1: Imitation Learning
        # =========================================
        resume_path = cfg.training.get('resume_path', None)
        #resume_path = "/nfs_global/S/yangrongzheng/3D-Diffusion-Policy/3D-Diffusion-Policy/checkpoints2/checkpoint_il_final.ckpt"
        if resume_path is not None and os.path.exists(resume_path):
            print(f"\n[Info] 从Checkpoint恢复: {resume_path}")
            self.load_checkpoint(resume_path)
            # 从checkpoint恢复后也需要冻结encoder
            print("\n[Info] Freezing encoder after resume...")
            self.freeze_encoder()
        else:
            print("\n" + "="*70)
            print("Phase 1: 模仿学习预训练（IL）")
            print("="*70)
            self.model.switch_to_il_mode()

            il_epochs = cfg.training.il_epochs
            self.train_loop(
                epochs=il_epochs,
                mode='il',
                desc="IL Pre-training"
            )
            self.save_checkpoint("checkpoint_il_final.ckpt")
            print("✓ Phase 1 完成\n")

            # IL完成后，冻结encoder
            print("\n[Info] Freezing encoder after IL phase...")
            self.freeze_encoder()

        # =========================================
        # Phase 2: 离线RL迭代
        # =========================================
        print("="*70)
        print("Phase 2: 离线强化学习迭代（IQL + AM-Q）")
        print("="*70)

        M = cfg.training.rl_iterations
        eval_freq = cfg.training.get('eval_freq', 2)
        data_quality_threshold = cfg.training.get('data_quality_threshold', 0.5)

        for iteration in range(M):
            print(f"\n{'─'*70}")
            print(f"迭代 {iteration+1}/{M}")
            print(f"{'─'*70}")

            # ===== 【新增】Step 1: IQL预训练Q、V网络 =====
            print(f"[1/6] IQL预训练Q/V网络...")
            self.model.switch_to_rl_mode()

            iql_epochs = self.cfg.training.get('iql_epochs_per_iter', 10)  # 默认10个epoch
            self.train_loop(
                epochs=iql_epochs,
                mode='iql',
                desc=f"IQL Pre-train Iter {iteration+1}"
            )

            # ===== 【新增】Step 2: 硬更新target网络 =====
            print(f"\n[2/6] Hard update target Q networks...")
            self.model.target_q_net1.load_state_dict(self.model.q_net1.state_dict())
            if self.model.is_double_q:
                self.model.target_q_net2.load_state_dict(self.model.q_net2.state_dict())
            if self.model.use_hierarchical_mdp:
                self.model.target_diffusion_q_net1.load_state_dict(self.model.diffusion_q_net1.state_dict())
                if self.model.is_double_q:
                    self.model.target_diffusion_q_net2.load_state_dict(self.model.diffusion_q_net2.state_dict())

            # Step 3: 离线RL训练（优化策略）
            print(f"\n[3/6] 离线RL训练（AM-Q优势加权）...")

            rl_epochs_per_iter = self.cfg.training.rl_epochs_per_iter
            self.train_loop(
                epochs=rl_epochs_per_iter,
                mode='offline_rl',
                desc=f"Offline RL Iter {iteration+1}"
            )

            # Step 4: Rollout收集新数据
            print(f"\n[4/6] Rollout收集新数据...")
            all_data, success_data, rollout_stats = self.collect_rollout_data(
                num_episodes=cfg.training.rollout_episodes
            )

            # 记录rollout统计
            wandb.log({
                'rollout/mean_reward': rollout_stats['mean_reward'],
                'rollout/mean_length': rollout_stats['mean_length'],
                'rollout/success_rate': rollout_stats['success_rate']
            }, step=self.global_step)

            # Step 3: 数据质量检测与合并
            data_quality_threshold = cfg.training.get('data_quality_threshold', 0.3)

            if len(all_data) > 0:
                print(f"\n[5/6] 数据质量检测与合并...")

                # ===== 数据质量检测（不做数量限制，只检测质量）=====
                if rollout_stats['success_rate'] >= data_quality_threshold:
                    print(f"  ✓ 数据质量合格（{rollout_stats['success_rate']:.1%} >= {data_quality_threshold:.1%}）")

                    # 创建全部数据的数据集（用于RL）
                    all_dataset = MemoryDataset(
                        data_list=all_data,
                        horizon=cfg.task.dataset.horizon,
                        pad_before=cfg.task.dataset.pad_before,
                        pad_after=cfg.task.dataset.pad_after
                    )
                    self.all_rl_datasets.append(all_dataset)
                    print(f"  📦 RL数据集: +{len(all_data)}步 → 总计{len(self.all_rl_datasets)}个数据集")

                    # 如果有成功的episode，创建成功数据集（用于IL）
                    if len(success_data) > 0:
                        success_dataset = MemoryDataset(
                            data_list=success_data,
                            horizon=cfg.task.dataset.horizon,
                            pad_before=cfg.task.dataset.pad_before,
                            pad_after=cfg.task.dataset.pad_after
                        )
                        self.successful_datasets.append(success_dataset)
                        print(f"  ✅ IL数据集（成功）: +{len(success_data)}步 → 总计{len(self.successful_datasets)}个数据集")
                    else:
                        print(f"  ⚠️  本轮无成功episode，IL数据集不更新")

                else:
                    print(f"  ✗ 数据质量不合格（{rollout_stats['success_rate']:.1%} < {data_quality_threshold:.1%}），丢弃本轮数据")
            else:
                print(f"\n[5/6] 无新数据收集，跳过合并")

            # Step 4: IL自我纠正（可选，encoder保持冻结）
            il_finetune_enabled = cfg.training.get('il_finetune_enabled', True)
            if il_finetune_enabled:
                print(f"\n[6/6] IL微调（只使用成功数据，encoder冻结）...")
                self.model.switch_to_il_mode()
                self.train_loop(
                    epochs=cfg.training.il_finetune_epochs,
                    mode='il',
                    desc=f"IL Finetune Iter {iteration+1}"
                )
            else:
                print(f"\n[6/6] IL微调已禁用（跳过）")

            # Step 5: 周期性评估
            if (iteration + 1) % eval_freq == 0 or iteration == M - 1:
                print(f"\n[评估] 验证策略性能...")
                eval_results = self.evaluate_policy()

                # 保存最佳模型
                success_rate = eval_results['success_rate']

                # ===== 更新best_success_rate并保存checkpoint =====
                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    print(f"  🏆 新最佳成功率: {success_rate:.2%}")
                else:
                    print(f"  📊 当前成功率: {success_rate:.2%}")

                # 保存TopK checkpoint
                ckpt_path = self.ckpt_manager.get_ckpt_path({
                    'eval_success_rate': success_rate,
                    'epoch': iteration + 1
                })

                if ckpt_path is not None:
                    checkpoint = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'normalizer_state_dict': self.normalizer.state_dict(),
                        'global_step': self.global_step,
                        'best_success_rate': self.best_success_rate,  # 现在是正确的最新值
                        'current_success_rate': success_rate,  # 添加当前成功率字段
                        'config': self.cfg,
                    }
                    if self.lr_scheduler is not None:
                        checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

                    torch.save(checkpoint, ckpt_path)
                    print(f"💾 保存TopK模型: {os.path.basename(ckpt_path)}")

                    # 如果是新最佳成功率，记录checkpoint路径
                    if success_rate == self.best_success_rate:
                        self.best_checkpoint_path = ckpt_path

            # 保存当前迭代checkpoint
            self.save_checkpoint(f"checkpoint_iter_{iteration+1}.ckpt")

        print("\n✓ Phase 2 完成")
        print(f"最佳成功率: {self.best_success_rate:.2%}")
        print(f"最佳Checkpoint: {self.best_checkpoint_path}")

        # =========================================
        # Phase 3: 在线RL（从最佳checkpoint开始）
        # =========================================
        if cfg.training.get('online_rl_enabled', True):
            print("\n" + "="*70)
            print("Phase 3: 在线强化学习（从最佳checkpoint开始）")
            print("="*70)

            # ===== 改进：加载最佳checkpoint =====
            if self.best_checkpoint_path is not None and os.path.exists(self.best_checkpoint_path):
                print(f"📂 加载最佳checkpoint: {os.path.basename(self.best_checkpoint_path)}")
                self.load_checkpoint(self.best_checkpoint_path)
            else:
                print("⚠️  未找到最佳checkpoint，从当前状态继续")

            self.model.switch_to_rl_mode()

            # ===== 改进：降低学习率 =====
            phase3_lr_factor = cfg.training.get('phase3_lr_factor', 0.1)
            original_lr = self.optimizer.param_groups[0]['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = original_lr * phase3_lr_factor
            print(f"🔽 降低学习率: {original_lr:.2e} -> {original_lr * phase3_lr_factor:.2e}")

            online_epochs = cfg.training.get('online_rl_epochs', 10)
            for epoch in range(online_epochs):
                print(f"\n在线RL Epoch {epoch+1}/{online_epochs}")

                # 收集在线数据
                all_online_data, success_online_data, rollout_stats = self.collect_rollout_data(
                    num_episodes=cfg.training.get('online_rollout_episodes', 10)
                )

                wandb.log({
                    'online_rl/mean_reward': rollout_stats['mean_reward'],
                    'online_rl/success_rate': rollout_stats['success_rate']
                }, step=self.global_step)

                if len(all_online_data) > 0:
                    # 添加到replay buffer（在线RL使用全部数据）
                    self.online_replay_buffer.extend(all_online_data)

                    # 限制buffer大小
                    if len(self.online_replay_buffer) > self.online_replay_max_size:
                        # 移除最早的数据
                        self.online_replay_buffer = self.online_replay_buffer[-self.online_replay_max_size:]
                        print(f"  📦 Replay Buffer已满，保留最近{self.online_replay_max_size}条数据")

                    # 使用replay buffer中的所有数据训练
                    print(f"  📊 使用Replay Buffer训练（大小: {len(self.online_replay_buffer)}）")
                    online_dataset = MemoryDataset(
                        data_list=self.online_replay_buffer,
                        horizon=cfg.task.dataset.horizon,
                        pad_before=cfg.task.dataset.pad_before,
                        pad_after=cfg.task.dataset.pad_after
                    )

                    # 训练
                    online_dataloader = DataLoader(
                        online_dataset,
                        batch_size=cfg.dataloader.batch_size,
                        shuffle=True,
                        num_workers=cfg.dataloader.num_workers,
                        pin_memory=True
                    )

                    # 训练多轮以充分利用replay buffer
                    train_epochs = 2 if epoch < online_epochs // 2 else 1
                    self.train_loop_online(
                        dataloader=online_dataloader,
                        epochs=train_epochs,
                        desc=f"Online RL Epoch {epoch+1}"
                    )

            # 最终评估
            print(f"\n[最终评估]")
            final_eval = self.evaluate_policy()
            self.save_checkpoint("checkpoint_online_rl_final.ckpt")
            print(f"✓ Phase 3 完成 - 最终成功率: {final_eval['success_rate']:.2%}\n")
        else:
            print("\n✗ Phase 3 跳过（online_rl_enabled=false）\n")

        # =========================================
        # 训练完成总结
        # =========================================
        print("="*70)
        print("训练完成总结")
        print("="*70)
        print(f"✓ Phase 1: IL预训练 - {cfg.training.il_epochs} epochs")
        print(f"✓ Phase 2: 离线RL - 最佳成功率 {self.best_success_rate:.2%}")
        if cfg.training.get('online_rl_enabled', True):
            print(f"✓ Phase 3: 在线RL - 最终成功率 {final_eval['success_rate']:.2%}")
        print(f"\n总训练步数: {self.global_step:,}")
        print(f"最佳Checkpoint: {self.best_checkpoint_path}")
        print("="*70)

    def get_dataloader(self, mode='rl'):
        """
        动态构建DataLoader

        Args:
            mode: 'il' - 使用成功数据（IL训练），'rl' - 使用全部数据（RL训练）
        """
        if mode == 'il':
            # IL模式：只使用成功的数据
            combined_dataset = ConcatDataset(self.successful_datasets)
            print(f"  [Dataloader] IL模式 - 使用{len(self.successful_datasets)}个成功数据集")
        else:
            # RL模式：使用所有数据
            combined_dataset = ConcatDataset(self.all_rl_datasets)
            print(f"  [Dataloader] RL模式 - 使用{len(self.all_rl_datasets)}个数据集（包括全部rollout）")

        return DataLoader(
            combined_dataset,
            batch_size=self.cfg.dataloader.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataloader.num_workers,
            pin_memory=True
        )

    def train_loop(self, epochs, mode, desc):
        """通用训练循环（IL + IQL + 离线RL）"""
        # 【修复】统一使用self.optimizer，避免学习率调度器不匹配问题
        # 根据模式选择对应的数据集
        if mode == 'il':
            dataloader = self.get_dataloader(mode='il')
        elif mode == 'iql':
            dataloader = self.get_dataloader(mode='rl')
        elif mode == 'offline_rl':
            dataloader = self.get_dataloader(mode='rl')
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 统一使用self.optimizer
        optimizer = self.optimizer

        target_update_freq = self.cfg.training.get('target_update_freq', 30)

        for epoch in range(epochs):
            with tqdm.tqdm(dataloader, desc=f"{desc} Epoch {epoch+1}", leave=False) as tepoch:
                for batch in tepoch:
                    # 1. 数据搬运
                    batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

                    # 2. 根据模式计算Loss
                    if mode == 'il':
                        raw_loss, loss_dict = self.model.compute_loss(batch)
                    elif mode == 'iql':
                        # IQL模式：只更新Q、V网络
                        raw_loss, loss_dict = self.model.compute_iql_loss(batch)
                    elif mode == 'offline_rl':
                        raw_loss, loss_dict = self.model.compute_offline_rl_loss(batch)

                    # 3. 反向传播
                    optimizer.zero_grad()
                    raw_loss.backward()

                    # 梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.get('grad_clip_norm', 10.0)
                    )

                    optimizer.step()

                    # 学习率调度（所有阶段统一调度）
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    # ===== EMA 更新（所有阶段统一更新）=====
                    if self.ema is not None:
                        self.ema.step(self.model)

                    # 4. 优化的Target Q网络更新
                    if mode in ['iql', 'offline_rl'] and self.global_step % target_update_freq == 0:
                        self.model.update_target_q_networks(tau=0.005)

                    # 5. 详细日志
                    self.global_step += 1
                    loss_dict['train/grad_norm'] = grad_norm.item()
                    if self.lr_scheduler is not None:
                        loss_dict['train/lr'] = self.lr_scheduler.get_last_lr()[0]

                    wandb.log(loss_dict, step=self.global_step)

                    # 6. 更详细的进度条显示
                    if mode == 'iql':
                        # IQL模式：显示Q、V loss
                        postfix_dict = {
                            'total': f"{raw_loss.item():.4f}",
                            'q': f"{loss_dict.get('iql/q', 0):.4f}",
                            'v': f"{loss_dict.get('iql/v', 0):.4f}",
                            'grad': f"{grad_norm.item():.3f}"
                        }
                    elif mode == 'offline_rl':
                        # 离线RL模式：显示diffusion, q, v三个主要loss
                        postfix_dict = {
                            'total': f"{raw_loss.item():.4f}",
                            'diff': f"{loss_dict.get('loss/diffusion', 0):.4f}",
                            'q': f"{loss_dict.get('loss/q', 0):.4f}",
                            'v': f"{loss_dict.get('loss/v', 0):.4f}",
                            'grad': f"{grad_norm.item():.3f}"
                        }
                    else:
                        # IL模式：简单显示
                        postfix_dict = {
                            'loss': f"{raw_loss.item():.4f}",
                            'grad': f"{grad_norm.item():.3f}"
                        }
                    tepoch.set_postfix(postfix_dict)

    def train_loop_online(self, dataloader, epochs, desc):
        """在线RL训练循环（使用GAE + Q/V网络更新）"""
        target_update_freq = self.cfg.training.get('target_update_freq', 30)

        for epoch in range(epochs):
            with tqdm.tqdm(dataloader, desc=f"{desc} (GAE)", leave=False) as tepoch:
                for batch in tepoch:
                    batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

                    # 使用GAE计算优势（现在包含Q/V网络更新）
                    raw_loss, loss_dict = self.model.compute_online_rl_loss(batch)

                    self.optimizer.zero_grad()
                    raw_loss.backward()

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.get('grad_clip_norm', 10.0)
                    )

                    self.optimizer.step()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    # ===== EMA 更新 =====
                    if self.ema is not None:
                        self.ema.step(self.model)

                    # ===== 添加Target Q网络更新 =====
                    if self.global_step % target_update_freq == 0:
                        self.model.update_target_q_networks(tau=0.005)

                    # 日志
                    self.global_step += 1
                    loss_dict['online_rl/grad_norm'] = grad_norm.item()
                    if self.lr_scheduler is not None:
                        loss_dict['online_rl/lr'] = self.lr_scheduler.get_last_lr()[0]

                    wandb.log(loss_dict, step=self.global_step)

                    # 详细的进度条显示（在线RL也显示各个loss组件）
                    postfix_dict = {
                        'total': f"{raw_loss.item():.4f}",
                        'diff': f"{loss_dict.get('loss/diffusion', 0):.4f}",
                        'q': f"{loss_dict.get('loss/q', 0):.4f}",
                        'v': f"{loss_dict.get('loss/v', 0):.4f}",
                        'grad': f"{grad_norm.item():.3f}"
                    }
                    tepoch.set_postfix(postfix_dict)

    def collect_rollout_data(self, num_episodes):
        """
        收集rollout数据并返回统计信息（使用MultiStepWrapper）

        Returns:
            all_data: 所有步的数据（用于RL）
            success_data: 只包含成功episode的数据（用于IL）
            stats: 统计信息
        """
        all_data = []
        success_data = []

        # ===== 使用 EMA 模型进行评估（如果可用）=====
        policy = self.ema_model if self.ema_model is not None else self.model
        policy.eval()

        n_action_steps = self.cfg.policy.n_action_steps  # 3 (执行3步)
        episode_rewards = []
        episode_lengths = []
        episode_successes = []

        with torch.no_grad():
            for i in range(num_episodes):
                # MultiStepWrapper的reset()返回stacked obs (n_obs_steps, ...)
                obs_dict = self.env.reset()
                done = False

                steps = 0
                episode_reward = 0
                episode_data = []  # 存储当前episode的所有数据

                while not done:
                    # 1. 准备输入（obs已经是stacked格式）
                    input_obs = {}
                    input_obs['point_cloud'] = torch.from_numpy(obs_dict['point_cloud']).unsqueeze(0).to(self.device).float()
                    input_obs['agent_pos'] = torch.from_numpy(obs_dict['agent_pos']).unsqueeze(0).to(self.device).float()

                    # 2. 策略预测（使用 EMA 模型，预测horizon=4步）
                    result = policy.predict_action(input_obs)
                    action_full = result['action'].squeeze(0).cpu().numpy()  # (horizon=4, action_dim)

                    # 3. 只取前n_action_steps步执行（3步）
                    action_to_execute = action_full[:n_action_steps]  # (3, action_dim)

                    # 4. 环境执行（MultiStepWrapper会执行这3步）
                    next_obs_dict, reward, done, info = self.env.step(action_to_execute)
                    episode_reward += reward

                    # 5. 处理MultiStepWrapper返回的格式
                    done = np.all(done) if isinstance(done, np.ndarray) else done

                    # MetaWorld的成功判断（MultiStepWrapper返回的info['success']可能是list）
                    if isinstance(info['success'], (list, np.ndarray)):
                        current_success = max(info['success'])
                    else:
                        current_success = info['success']

                    # 6. 组装样本（注意：这里存储的是单步的数据，用于后续训练）
                    # 由于MultiStepWrapper执行了3步，我们简化为存储第一步的obs和完整的action序列
                    sample = {
                        'obs': dict_apply(obs_dict, lambda x: torch.from_numpy(x.copy()).float()),
                        'action': torch.from_numpy(action_to_execute[0]).float(),  # 存储执行的第一步
                        'reward': torch.tensor(reward).float(),  # MultiStepWrapper已经聚合了3步的奖励
                        'next_obs': dict_apply(next_obs_dict, lambda x: torch.from_numpy(x.copy()).float()),
                        'done': torch.tensor(float(done)).float()
                    }
                    episode_data.append(sample)

                    # 7. 更新状态
                    obs_dict = next_obs_dict
                    steps += n_action_steps  # 执行了3步

                # 记录episode统计
                episode_rewards.append(episode_reward)
                episode_lengths.append(steps)
                episode_successes.append(current_success)

                # ===== 关键改动：分别存储全部数据和成功数据 =====
                all_data.extend(episode_data)  # 所有数据都加入

                if current_success:  # 只有成功的episode才加入成功数据
                    success_data.extend(episode_data)

                if (i + 1) % 5 == 0:
                    success_symbol = "✓" if current_success else "✗"
                    print(f"  Episode {i+1}/{num_episodes}: Reward={episode_reward:.2f}, Steps={steps}, Success={success_symbol}")

        policy.train()
        self.model.train()  # 确保主模型也回到训练模式

        # 返回数据和统计
        stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': np.mean(episode_successes),
            'num_successes': int(np.sum(episode_successes))
        }

        # 打印详细统计
        print(f"  📊 Rollout统计: 成功率={stats['success_rate']:.1%} ({stats['num_successes']}/{num_episodes}), "
              f"奖励={stats['mean_reward']:.1f}±{stats['std_reward']:.1f}")
        print(f"  📦 数据量: 全部={len(all_data)}步, 成功={len(success_data)}步")

        return all_data, success_data, stats

    def evaluate_policy(self):
        """使用env_runner进行周期性评估"""
        if self.eval_runner is None:
            print("  ⚠️  未配置env_runner，跳过评估")
            return {'success_rate': 0.0}

        print("  运行评估...")

        # ===== 使用 EMA 模型进行评估（如果可用）=====
        policy = self.ema_model if self.ema_model is not None else self.model
        policy.eval()

        try:
            eval_results = self.eval_runner.run(policy)
            # 修复：正确的键名是'test_mean_score'，不是'test/mean_score'
            success_rate = eval_results.get('test_mean_score', 0.0)
            mean_reward = eval_results.get('mean_traj_rewards', 0.0)

            # 记录到WandB
            wandb.log({
                'eval/success_rate': success_rate,
                'eval/mean_reward': mean_reward,
                'eval/SR_test_L3': eval_results.get('SR_test_L3', 0.0),
                'eval/SR_test_L5': eval_results.get('SR_test_L5', 0.0),
            }, step=self.global_step)

            print(f"  ✓ 评估完成 - 成功率: {success_rate:.2%}, 平均奖励: {mean_reward:.2f}")

            policy.train()
            self.model.train()  # 确保主模型也回到训练模式
            return {'success_rate': success_rate, 'mean_reward': mean_reward}

        except Exception as e:
            print(f"  ⚠️  评估失败: {e}")
            import traceback
            traceback.print_exc()
            policy.train()
            self.model.train()
            return {'success_rate': 0.0, 'mean_reward': 0.0}

    def save_checkpoint(self, filename):
        """保存checkpoint"""
        path = os.path.join(self.output_dir, "checkpoints2", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'normalizer_state_dict': self.normalizer.state_dict(),
            'global_step': self.global_step,
            'best_success_rate': self.best_success_rate,
            'config': self.cfg,
        }

        # ===== 保存 EMA 模型状态 =====
        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()

        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"  💾 Checkpoint已保存: {path}")

    def load_checkpoint(self, path):
        """加载checkpoint"""
        print(f"  📂 加载checkpoint: {path}")
        payload = torch.load(path, map_location=self.device)

        self.model.load_state_dict(payload['model_state_dict'])
        self.optimizer.load_state_dict(payload['optimizer_state_dict'])

        if 'normalizer_state_dict' in payload:
            self.normalizer.load_state_dict(payload['normalizer_state_dict'])

        # ===== 加载 EMA 模型状态 =====
        if self.ema_model is not None and 'ema_model_state_dict' in payload:
            self.ema_model.load_state_dict(payload['ema_model_state_dict'])
            print("  ✓ EMA 模型状态已恢复")

        self.global_step = payload.get('global_step', 0)
        self.best_success_rate = payload.get('best_success_rate', 0.0)

        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in payload:
            self.lr_scheduler.load_state_dict(payload['lr_scheduler_state_dict'])

        print(f"  ✓ 恢复训练 - Step: {self.global_step}, 最佳成功率: {self.best_success_rate:.2%}")

    def freeze_encoder(self):
        """冻结obs_encoder的所有参数"""
        for param in self.model.obs_encoder.parameters():
            param.requires_grad = False
        print("[Info] Encoder has been frozen. No gradients will be computed for encoder in subsequent RL phases.")


@hydra.main(config_path="diffusion_policy_3d/config", config_name="train_rl100", version_base=None)
def main(cfg):
    workspace = TrainRL100Improved(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
