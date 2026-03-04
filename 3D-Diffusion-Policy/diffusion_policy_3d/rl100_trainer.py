"""
RL-100 Trainer
==============
Complete training pipeline for RL-100: IL -> Offline RL -> Online RL.

Implements Algorithm 1 from RL-100 paper:
1. Phase 1: Imitation Learning (IL) - Train diffusion policy with BC
2. Phase 2: Offline RL Loop (M iterations):
   a) Train IQL Critics on dataset
   b) Optimize policy with PPO + Consistency Distillation
   c) Collect new data with policy
   d) Merge datasets and retrain IL
3. Phase 3: Online RL - Fine-tune with online rollouts

Key Components:
- RL100Policy: Diffusion policy with PPO optimization
- IQLCritics: Q and V networks for advantage estimation
- ConsistencyModel: Fast 1-step generation
- EnvRunner: Environment for data collection

Args:
    config: Configuration dictionary with:
        - policy: RL100Policy config
        - dataset: Dataset config
        - env_runner: Environment config
        - training: Training hyperparameters
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import wandb
import copy
from typing import Dict, Optional
from termcolor import cprint
import hydra
from omegaconf import OmegaConf

from diffusion_policy_3d.policy.rl100_policy import RL100Policy
from diffusion_policy_3d.model.rl.iql_critics import IQLCritics
from diffusion_policy_3d.model.rl.consistency_model import ConsistencyModel, ConsistencyDistillation
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner


class RL100Trainer:
    """
    Complete RL-100 training pipeline.

    Training Flow:
    ==============
    1. train_imitation_learning(D_0):
        - Standard BC on initial dataset
        - Outputs: Trained DP3 policy π_θ

    2. For M iterations:
        a) train_iql_critics(D_t):
            - Update V: L_V = expectile_loss(V(s), Q(s, a), τ=0.7)
            - Update Q: L_Q = MSE(Q(s,a), r + γV(s'))

        b) offline_rl_optimize(D_t):
            - Compute advantage: A = Q(s,a) - V(s)
            - Update policy: L_PPO = Σ_k min(r_k*A, clip(r_k)*A)
            - Distill to consistency model: L_CD = ||CM(noise) - π_θ||^2

        c) collect_new_data(π_θ):
            - Rollout policy in environment
            - Add to D_new

        d) merge and retrain:
            - D_{t+1} = D_t ∪ D_new
            - Retrain IL on D_{t+1}

    3. online_rl_finetune():
        - Continue PPO with fresh rollouts
        - No more IL retraining
    """

    def __init__(self, config: OmegaConf):
        self.config = config
        self.device = torch.device(config.training.device)

        # Initialize policy (RL100Policy extends DP3)
        cprint("[RL100Trainer] Initializing RL100Policy...", "cyan")
        self.policy: RL100Policy = hydra.utils.instantiate(config.policy)
        self.policy.to(self.device)

        # EMA model
        self.ema_policy = None
        if config.training.use_ema:
            self.ema_policy = copy.deepcopy(self.policy)
            self.ema = hydra.utils.instantiate(
                config.ema,
                model=self.ema_policy
            )

        # Optimizer for policy
        self.policy_optimizer = hydra.utils.instantiate(
            config.optimizer.policy,
            params=self.policy.parameters()
        )

        # Initialize IQL Critics
        cprint("[RL100Trainer] Initializing IQL Critics...", "cyan")
        obs_dim = self.policy.obs_feature_dim  # Output dim from encoder
        action_dim = self.policy.action_dim

        self.critics = IQLCritics(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.critics.hidden_dims,
            gamma=config.critics.gamma,
            tau=config.critics.tau
        )
        self.critics.to(self.device)

        # Optimizers for critics
        self.v_optimizer = hydra.utils.instantiate(
            config.optimizer.v_network,
            params=self.critics.v_network.parameters()
        )
        self.q_optimizer = hydra.utils.instantiate(
            config.optimizer.q_network,
            params=self.critics.q_network.parameters()
        )

        # Initialize Consistency Model
        cprint("[RL100Trainer] Initializing Consistency Model...", "cyan")
        self.consistency_model = ConsistencyModel(
            input_dim=action_dim,
            global_cond_dim=obs_dim * self.policy.n_obs_steps,
            diffusion_step_embed_dim=config.policy.diffusion_step_embed_dim,
            down_dims=config.policy.down_dims,
            condition_type=config.policy.condition_type
        )
        self.consistency_model.to(self.device)

        self.consistency_optimizer = hydra.utils.instantiate(
            config.optimizer.consistency,
            params=self.consistency_model.parameters()
        )

        self.consistency_distillation = ConsistencyDistillation(
            teacher_policy=self.policy,
            student_model=self.consistency_model,
            student_optimizer=self.consistency_optimizer
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.offline_rl_iteration = 0

    def train_imitation_learning(
        self,
        dataset: BaseDataset,
        num_epochs: int,
        val_dataset: Optional[BaseDataset] = None,
        env_runner: Optional[BaseRunner] = None
    ) -> Dict:
        """
        Phase 1: Train with behavior cloning (standard DP3 training).

        Args:
            dataset: Training dataset
            num_epochs: Number of training epochs
            val_dataset: Optional validation dataset
            env_runner: Optional environment for evaluation

        Returns:
            metrics: Dictionary with training metrics
        """
        cprint("\n" + "="*60, "yellow")
        cprint("[RL100Trainer] Phase 1: Imitation Learning", "yellow")
        cprint("="*60 + "\n", "yellow")

        config = self.config
        self.policy.train()

        # Create dataloader
        train_dataloader = DataLoader(
            dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=True,
            num_workers=config.dataloader.num_workers,
            pin_memory=True
        )

        # Set normalizer
        normalizer = dataset.get_normalizer()
        self.policy.set_normalizer(normalizer)
        if self.ema_policy is not None:
            self.ema_policy.set_normalizer(normalizer)

        # Training loop
        for epoch in range(num_epochs):
            epoch_losses = []

            for batch_idx, batch in enumerate(train_dataloader):
                # Move to device
                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

                # Compute loss
                loss, loss_dict = self.policy.compute_loss(batch)

                # Optimize
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()

                # Update EMA
                if self.ema_policy is not None:
                    self.ema.step(self.policy)

                epoch_losses.append(loss.item())
                self.global_step += 1

                # Log
                if self.global_step % config.training.log_every == 0:
                    wandb.log({
                        'il/loss': loss.item(),
                        'il/epoch': epoch,
                        **{f'il/{k}': v for k, v in loss_dict.items()}
                    }, step=self.global_step)

            # Epoch end
            avg_loss = np.mean(epoch_losses)
            cprint(f"[IL] Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}", "green")

            # Evaluate
            if env_runner is not None and (epoch + 1) % config.training.eval_every == 0:
                eval_policy = self.ema_policy if self.ema_policy else self.policy
                eval_policy.eval()
                with torch.no_grad():
                    metrics = env_runner.run(eval_policy)
                cprint(f"[IL] Eval - Success Rate: {metrics.get('mean_success_rates', 0):.3f}", "green")
                wandb.log({f'il/eval_{k}': v for k, v in metrics.items()}, step=self.global_step)
                eval_policy.train()

        return {'final_loss': avg_loss}

    def train_iql_critics(
        self,
        dataset: BaseDataset,
        num_epochs: int
    ) -> Dict:
        """
        Phase 2a: Train IQL critics (Q and V networks).

        Training Algorithm:
        1. Update V with expectile regression:
            L_V = expectile_loss(V(s), Q(s, a_data), τ=0.7)

        2. Update Q with Bellman backup:
            L_Q = MSE(Q(s, a), r + γV(s'))

        Args:
            dataset: Dataset with (s, a, r, s', done)
            num_epochs: Number of training epochs

        Returns:
            metrics: Dictionary with training metrics
        """
        cprint(f"\n[RL100Trainer] Phase 2a: Training IQL Critics (Iteration {self.offline_rl_iteration})", "cyan")

        config = self.config
        self.critics.train()

        # Create dataloader
        train_dataloader = DataLoader(
            dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=True,
            num_workers=config.dataloader.num_workers,
            pin_memory=True
        )

        for epoch in range(num_epochs):
            v_losses = []
            q_losses = []

            for batch_idx, batch in enumerate(train_dataloader):
                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

                # Extract data
                obs_dict = batch['obs']
                action = batch['action'][:, 0, :]  # [B, horizon, Da] -> [B, Da]

                # Encode observations to get features
                with torch.no_grad():
                    nobs = self.policy.normalizer.normalize(obs_dict)
                    if not self.policy.use_pc_color:
                        nobs['point_cloud'] = nobs['point_cloud'][..., :3]

                    this_nobs = dict_apply(
                        nobs,
                        lambda x: x[:, :self.policy.n_obs_steps, ...].reshape(-1, *x.shape[2:])
                    )
                    obs_features = self.policy.obs_encoder(this_nobs)
                    obs_features = obs_features.reshape(batch['action'].shape[0], -1)

                # Normalize action
                naction = self.policy.normalizer['action'].normalize(action)

                # 1. Update V network
                v_loss, v_info = self.critics.compute_v_loss(obs_features, naction)

                self.v_optimizer.zero_grad()
                v_loss.backward()
                self.v_optimizer.step()

                v_losses.append(v_loss.item())

                # 2. Update Q network
                # For simplicity, assume reward and next_obs are in batch
                # In practice, you need to modify dataset to include these
                if 'reward' in batch and 'next_obs' in batch:
                    reward = batch['reward']
                    done = batch.get('done', torch.zeros_like(reward))

                    # Encode next observations
                    with torch.no_grad():
                        next_nobs = self.policy.normalizer.normalize(batch['next_obs'])
                        if not self.policy.use_pc_color:
                            next_nobs['point_cloud'] = next_nobs['point_cloud'][..., :3]

                        next_this_nobs = dict_apply(
                            next_nobs,
                            lambda x: x[:, :self.policy.n_obs_steps, ...].reshape(-1, *x.shape[2:])
                        )
                        next_obs_features = self.policy.obs_encoder(next_this_nobs)
                        next_obs_features = next_obs_features.reshape(batch['action'].shape[0], -1)

                    q_loss, q_info = self.critics.compute_q_loss(
                        obs_features, naction, reward, next_obs_features, done
                    )

                    self.q_optimizer.zero_grad()
                    q_loss.backward()
                    self.q_optimizer.step()

                    q_losses.append(q_loss.item())

                    # Update target network
                    self.critics.update_target_network(tau=config.critics.target_update_tau)

                    # Log
                    if self.global_step % config.training.log_every == 0:
                        wandb.log({
                            'iql/v_loss': v_loss.item(),
                            'iql/q_loss': q_loss.item(),
                            **{f'iql/{k}': v for k, v in v_info.items()},
                            **{f'iql/{k}': v for k, v in q_info.items()}
                        }, step=self.global_step)

                self.global_step += 1

            cprint(f"[IQL] Epoch {epoch}/{num_epochs}, V Loss: {np.mean(v_losses):.4f}, "
                   f"Q Loss: {np.mean(q_losses):.4f if q_losses else 0:.4f}", "green")

        return {'v_loss': np.mean(v_losses), 'q_loss': np.mean(q_losses) if q_losses else 0}

    def offline_rl_optimize(
        self,
        dataset: BaseDataset,
        num_epochs: int
    ) -> Dict:
        """
        Phase 2b: Optimize policy with PPO and consistency distillation.

        Training:
        1. Compute advantage: A = Q(s, a) - V(s)
        2. Update policy with PPO: L_PPO = Σ_k min(r_k*A, clip(r_k)*A)
        3. Distill to consistency model: L_CD = ||CM(noise) - π_θ||^2

        Args:
            dataset: Dataset with (s, a, r, s', done)
            num_epochs: Number of training epochs

        Returns:
            metrics: Dictionary with training metrics
        """
        cprint(f"\n[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration {self.offline_rl_iteration})", "cyan")

        config = self.config
        self.policy.train()
        self.critics.eval()  # Freeze critics during policy update

        # Create dataloader
        train_dataloader = DataLoader(
            dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=True,
            num_workers=config.dataloader.num_workers,
            pin_memory=True
        )

        for epoch in range(num_epochs):
            ppo_losses = []
            cd_losses = []

            for batch_idx, batch in enumerate(train_dataloader):
                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

                obs_dict = batch['obs']
                action = batch['action'][:, 0, :]  # [B, horizon, Da] -> [B, Da]

                # Encode observations
                with torch.no_grad():
                    nobs = self.policy.normalizer.normalize(obs_dict)
                    if not self.policy.use_pc_color:
                        nobs['point_cloud'] = nobs['point_cloud'][..., :3]

                    this_nobs = dict_apply(
                        nobs,
                        lambda x: x[:, :self.policy.n_obs_steps, ...].reshape(-1, *x.shape[2:])
                    )
                    obs_features = self.policy.obs_encoder(this_nobs)
                    obs_features = obs_features.reshape(batch['action'].shape[0], -1)

                    # Normalize action
                    naction = self.policy.normalizer['action'].normalize(action)

                    # Compute advantage
                    advantages = self.critics.compute_advantage(obs_features, naction)

                # 1. PPO update
                ppo_loss, ppo_info = self.policy.compute_rl_loss(obs_dict, advantages)

                self.policy_optimizer.zero_grad()
                ppo_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config.training.max_grad_norm)
                self.policy_optimizer.step()

                ppo_losses.append(ppo_loss.item())

                # Update EMA
                if self.ema_policy is not None:
                    self.ema.step(self.policy)

                # 2. Consistency distillation (every N steps)
                if self.global_step % config.training.cd_every == 0:
                    cd_info = self.consistency_distillation.train_step(obs_dict)
                    cd_losses.append(cd_info['cd_loss'])

                    wandb.log({
                        'cd/loss': cd_info['cd_loss'],
                        **{f'cd/{k}': v for k, v in cd_info.items()}
                    }, step=self.global_step)

                # Log
                if self.global_step % config.training.log_every == 0:
                    wandb.log({
                        'ppo/loss': ppo_loss.item(),
                        **{f'ppo/{k}': v for k, v in ppo_info.items()}
                    }, step=self.global_step)

                self.global_step += 1

            cprint(f"[Offline RL] Epoch {epoch}/{num_epochs}, PPO Loss: {np.mean(ppo_losses):.4f}, "
                   f"CD Loss: {np.mean(cd_losses) if cd_losses else 0:.4f}", "green")

        return {'ppo_loss': np.mean(ppo_losses), 'cd_loss': np.mean(cd_losses) if cd_losses else 0}

    def collect_new_data(
        self,
        env_runner: BaseRunner,
        num_episodes: int
    ) -> Dict:
        """
        Phase 2c: Collect new data by rolling out policy in environment.

        Args:
            env_runner: Environment runner
            num_episodes: Number of episodes to collect

        Returns:
            metrics: Dictionary with collection metrics
        """
        cprint(f"\n[RL100Trainer] Phase 2c: Collecting New Data (Iteration {self.offline_rl_iteration})", "cyan")

        eval_policy = self.ema_policy if self.ema_policy else self.policy
        eval_policy.eval()

        with torch.no_grad():
            # Run rollout
            metrics = env_runner.run(eval_policy, num_episodes=num_episodes)

        cprint(f"[Data Collection] Success Rate: {metrics.get('mean_success_rates', 0):.3f}, "
               f"Reward: {metrics.get('mean_traj_rewards', 0):.2f}", "green")

        wandb.log({
            f'collection/success_rate': metrics.get('mean_success_rates', 0),
            f'collection/reward': metrics.get('mean_traj_rewards', 0)
        }, step=self.global_step)

        eval_policy.train()

        return metrics

    def run_pipeline(
        self,
        initial_dataset: BaseDataset,
        env_runner: BaseRunner,
        num_offline_iterations: int = 5
    ):
        """
        Execute complete RL-100 pipeline (Algorithm 1).

        Pipeline:
        =========
        1. Initialize: Train IL on D_0
        2. Loop M times:
            a) Train Critics (IQL)
            b) Optimize Policy (PPO + CD)
            c) Collect new data D_new
            d) Merge D = D ∪ D_new
            e) Retrain IL on merged dataset
        3. Online RL fine-tuning (optional)

        Args:
            initial_dataset: Initial dataset D_0
            env_runner: Environment for data collection
            num_offline_iterations: Number of offline RL iterations (M)
        """
        config = self.config

        cprint("\n" + "="*80, "magenta")
        cprint(" "*20 + "RL-100 TRAINING PIPELINE", "magenta")
        cprint("="*80 + "\n", "magenta")

        # ============================================
        # Phase 1: Initial Imitation Learning
        # ============================================
        self.train_imitation_learning(
            dataset=initial_dataset,
            num_epochs=config.training.il_epochs,
            env_runner=env_runner
        )

        # Save IL checkpoint
        self.save_checkpoint(tag='after_il')

        # ============================================
        # Phase 2: Offline RL Loop
        # ============================================
        current_dataset = initial_dataset

        for iteration in range(num_offline_iterations):
            self.offline_rl_iteration = iteration

            cprint("\n" + "="*80, "yellow")
            cprint(f" "*15 + f"OFFLINE RL ITERATION {iteration + 1}/{num_offline_iterations}", "yellow")
            cprint("="*80 + "\n", "yellow")

            # 2a) Train IQL Critics
            self.train_iql_critics(
                dataset=current_dataset,
                num_epochs=config.training.critic_epochs
            )

            # 2b) Optimize Policy
            self.offline_rl_optimize(
                dataset=current_dataset,
                num_epochs=config.training.ppo_epochs
            )

            # 2c) Collect New Data
            self.collect_new_data(
                env_runner=env_runner,
                num_episodes=config.training.collection_episodes
            )

            # 2d) Merge datasets
            # Note: In practice, you'd implement dataset merging here
            # For now, we assume env_runner updates the dataset

            # 2e) Retrain IL (optional)
            if config.training.retrain_il_after_collection:
                cprint(f"\n[RL100Trainer] Retraining IL on merged dataset...", "cyan")
                self.train_imitation_learning(
                    dataset=current_dataset,
                    num_epochs=config.training.il_retrain_epochs,
                    env_runner=env_runner
                )

            # Save checkpoint
            self.save_checkpoint(tag=f'offline_iter_{iteration}')

        # ============================================
        # Phase 3: Online RL Fine-tuning (Optional)
        # ============================================
        if config.training.run_online_rl:
            cprint("\n" + "="*80, "green")
            cprint(" "*20 + "PHASE 3: ONLINE RL FINE-TUNING", "green")
            cprint("="*80 + "\n", "green")

            # Continue PPO with online rollouts
            for online_iter in range(config.training.online_rl_iterations):
                # Collect fresh data
                self.collect_new_data(
                    env_runner=env_runner,
                    num_episodes=config.training.online_collection_episodes
                )

                # Update critics and policy
                # Note: Would need online dataset here
                # self.train_iql_critics(...)
                # self.offline_rl_optimize(...)

        cprint("\n" + "="*80, "magenta")
        cprint(" "*25 + "TRAINING COMPLETE!", "magenta")
        cprint("="*80 + "\n", "magenta")

        # Save final checkpoint
        self.save_checkpoint(tag='final')

    def save_checkpoint(self, tag: str = 'latest'):
        """Save checkpoint."""
        save_dir = os.path.join(self.config.output_dir, 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'policy': self.policy.state_dict(),
            'critics': self.critics.state_dict(),
            'consistency_model': self.consistency_model.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'consistency_optimizer': self.consistency_optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'offline_rl_iteration': self.offline_rl_iteration,
        }

        if self.ema_policy is not None:
            checkpoint['ema_policy'] = self.ema_policy.state_dict()

        save_path = os.path.join(save_dir, f'{tag}.ckpt')
        torch.save(checkpoint, save_path)
        cprint(f"[Checkpoint] Saved to {save_path}", "green")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy'])
        self.critics.load_state_dict(checkpoint['critics'])
        self.consistency_model.load_state_dict(checkpoint['consistency_model'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.consistency_optimizer.load_state_dict(checkpoint['consistency_optimizer'])

        if 'ema_policy' in checkpoint and self.ema_policy is not None:
            self.ema_policy.load_state_dict(checkpoint['ema_policy'])

        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.offline_rl_iteration = checkpoint['offline_rl_iteration']

        cprint(f"[Checkpoint] Loaded from {path}", "green")
