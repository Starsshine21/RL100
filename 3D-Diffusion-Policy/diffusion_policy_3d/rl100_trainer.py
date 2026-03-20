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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import wandb
import copy
from typing import Dict, List, Optional, Tuple
from termcolor import cprint
import hydra
from omegaconf import OmegaConf
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from diffusion_policy_3d.policy.rl100_policy import RL100Policy
from diffusion_policy_3d.policy.consistency_policy import ConsistencyPolicyWrapper
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.rl.iql_critics import IQLCritics
from diffusion_policy_3d.model.rl.consistency_model import ConsistencyModel, ConsistencyDistillation
from diffusion_policy_3d.model.rl.transition_model import TransitionModel
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

    def __init__(self, config: OmegaConf, output_dir: Optional[str] = None):
        self.config = config
        self._output_dir = output_dir
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
        obs_dim = self.policy.obs_feature_dim * self.policy.n_obs_steps  # Flattened obs feature dim across all obs steps
        critic_action_dim = self.policy.chunk_action_dim
        model_action_dim = self.policy.action_dim

        self.critics = IQLCritics(
            obs_dim=obs_dim,
            action_dim=critic_action_dim,
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
            input_dim=model_action_dim,
            global_cond_dim=(
                self.policy.obs_feature_dim
                if "cross_attention" in self.policy.condition_type
                else obs_dim
            ),
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

        # Initialize Transition Model (Algorithm 1, Line 6)
        cprint("[RL100Trainer] Initializing Transition Model T_θ(s'|s,a)...", "cyan")
        self.transition_model = TransitionModel(
            obs_feature_dim=obs_dim,    # same flattened dim used by critics
            action_dim=critic_action_dim,
            hidden_dims=(200, 200, 200, 200),
            num_ensemble=7,
            num_elites=5,
            device=str(self.device),
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.offline_rl_iteration = 0
        self.offline_collection_success_history = []
        self.online_collection_success_history = []

    def _reset_policy_optimizer(self, lr: Optional[float] = None) -> None:
        """
        Recreate the policy optimizer and discard any carried optimizer state.
        This is useful when switching from IL to PPO so Adam moments from BC do
        not cause an oversized first PPO step.
        """
        self.policy_optimizer = hydra.utils.instantiate(
            self.config.optimizer.policy,
            params=self.policy.parameters()
        )
        if lr is not None:
            for pg in self.policy_optimizer.param_groups:
                pg['lr'] = float(lr)

    def _prepare_policy_optimizer_for_bc(self) -> None:
        bc_lr = float(self.config.optimizer.policy.lr)
        self._reset_policy_optimizer(lr=bc_lr)
        cprint(f"[BC] Reset policy optimizer state and set LR: {bc_lr:.2e}", "cyan")

    def _prepare_policy_optimizer_for_rl(self) -> float:
        rl_lr = float(getattr(self.config.training, 'rl_policy_lr', 1e-5))
        previous_lr = float(self.policy_optimizer.param_groups[0]['lr'])
        self._reset_policy_optimizer(lr=rl_lr)
        cprint(
            f"[RL PPO] Reset policy optimizer state and set LR: {previous_lr:.2e} → {rl_lr:.2e}",
            "cyan"
        )
        return rl_lr

    def _encode_obs_representations(
        self,
        obs_dict: Dict[str, torch.Tensor],
        policy: Optional[RL100Policy] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self.policy if policy is None else policy
        nobs = policy.normalizer.normalize(obs_dict)
        if not policy.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :policy.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        obs_features = policy.obs_encoder(this_nobs)
        batch_size = next(iter(obs_dict.values())).shape[0]
        flat_obs_features = obs_features.reshape(batch_size, -1)
        if "cross_attention" in policy.condition_type:
            global_cond = obs_features.reshape(batch_size, policy.n_obs_steps, -1)
        else:
            global_cond = flat_obs_features
        return flat_obs_features, global_cond

    def _encode_obs_features(self, obs_dict: Dict[str, torch.Tensor], policy: Optional[RL100Policy] = None) -> torch.Tensor:
        flat_obs_features, _ = self._encode_obs_representations(obs_dict, policy=policy)
        return flat_obs_features

    def _should_apply_obs_regularization(self) -> bool:
        if not getattr(self.policy, 'use_recon_vib', False):
            return False
        return any(param.requires_grad for param in self.policy.obs_encoder.parameters())

    def _compute_obs_regularization(self, obs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self._should_apply_obs_regularization():
            zero = torch.zeros((), device=self.device)
            return zero, {
                'kl_loss': 0.0,
                'recon_loss': 0.0,
                'recon_loss_pc': 0.0,
                'recon_loss_state': 0.0,
                'total_reg_loss': 0.0,
            }
        return self.policy.compute_obs_regularization(obs_dict)

    def _extract_chunk_action(self, action_trajectory: torch.Tensor) -> torch.Tensor:
        return self.policy.extract_action_chunk(action_trajectory)

    def _normalize_chunk_action(self, action_trajectory: torch.Tensor) -> torch.Tensor:
        chunk_action = self._extract_chunk_action(action_trajectory)
        return self.policy.normalizer['action'].normalize(chunk_action)

    def _flatten_normalized_chunk_action(self, action_trajectory: torch.Tensor) -> torch.Tensor:
        normalized_chunk = self._normalize_chunk_action(action_trajectory)
        return normalized_chunk.reshape(normalized_chunk.shape[0], -1)

    def get_runtime_policy(self, mode: str = 'ddim', use_ema: bool = False) -> BasePolicy:
        teacher_policy = self.ema_policy if use_ema and self.ema_policy is not None else self.policy
        mode = str(mode).lower()
        if mode == 'cm':
            runtime_policy = ConsistencyPolicyWrapper(teacher_policy=teacher_policy, consistency_model=self.consistency_model)
            runtime_policy.to(self.device)
            runtime_policy.eval()
            return runtime_policy
        if mode != 'ddim':
            raise ValueError(f"Unsupported runtime policy mode: {mode}")
        teacher_policy.eval()
        return teacher_policy

    @property
    def output_dir(self) -> str:
        if self._output_dir is not None:
            return self._output_dir
        return os.getcwd()

    def _save_loss_curve_plot(
        self,
        loss_history: list,
        title: str,
        ylabel: str,
        filename: str,
        xlabel: str = "Epoch",
    ) -> Optional[str]:
        """
        Save loss curve with:
          - top panel: full range (non-uniform y-axis via symlog when needed)
          - bottom panel: zoomed view in [0, 1] to show 0.x fluctuations.
        """
        if not loss_history:
            return None
        if plt is None:
            cprint("[Plot] matplotlib is not available; skip loss curve plotting.", "yellow")
            return None

        y = np.asarray(loss_history, dtype=np.float64)
        x = np.arange(1, len(y) + 1, dtype=np.int32)
        if y.size == 0:
            return None

        plot_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, filename)

        fig, (ax_full, ax_zoom) = plt.subplots(
            2, 1, figsize=(9, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]}
        )

        # Full-range panel: non-uniform axis when values exceed 1.
        y_max = float(np.nanmax(y))
        y_min = float(np.nanmin(y))
        ax_full.plot(x, y, color="#1f77b4", linewidth=2.0)
        if y_max > 1.0:
            ax_full.set_yscale('symlog', linthresh=1.0, linscale=1.2, base=10)
        ax_full.set_ylabel(ylabel)
        ax_full.set_title(f"{title} (top: adaptive axis, bottom: 0-1 zoom)")
        ax_full.grid(True, alpha=0.3)
        ax_full.set_ylim(bottom=max(0.0, y_min * 0.95), top=max(1e-6, y_max * 1.05))

        # Zoom panel: emphasize [0, 1] region.
        zoom_mask = (y >= 0.0) & (y <= 1.0)
        if np.any(zoom_mask):
            y_zoom = y[zoom_mask]
            zmin = float(np.min(y_zoom))
            zmax = float(np.max(y_zoom))
            pad = max(0.01, (zmax - zmin) * 0.2)
            zlow = max(0.0, zmin - pad)
            zhigh = min(1.0, zmax + pad)
            if zhigh - zlow < 0.05:
                mid = 0.5 * (zhigh + zlow)
                zlow = max(0.0, mid - 0.025)
                zhigh = min(1.0, mid + 0.025)
        else:
            zlow, zhigh = 0.0, 1.0

        ax_zoom.plot(x, y, color="#d62728", linewidth=1.8)
        ax_zoom.set_ylim(zlow, zhigh)
        ax_zoom.set_ylabel(f"{ylabel} (0-1)")
        ax_zoom.set_xlabel(xlabel)
        ax_zoom.grid(True, alpha=0.35)

        fig.tight_layout()
        fig.savefig(save_path, dpi=160)
        plt.close(fig)

        cprint(f"[Plot] Saved: {save_path}", "cyan")
        return save_path

    def _save_iteration_metric_plot(
        self,
        metric_history: list,
        title: str,
        ylabel: str,
        filename: str,
    ) -> Optional[str]:
        return self._save_loss_curve_plot(
            loss_history=metric_history,
            title=title,
            ylabel=ylabel,
            filename=filename,
            xlabel="Iteration",
        )

    def train_imitation_learning(
        self,
        dataset: BaseDataset,
        num_epochs: int,
        val_dataset: Optional[BaseDataset] = None,
        env_runner: Optional[BaseRunner] = None,
        plot_tag: str = "il"
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

        if val_dataset is None and hasattr(dataset, 'get_validation_dataset'):
            try:
                candidate_val_dataset = dataset.get_validation_dataset()
                if len(candidate_val_dataset) > 0:
                    val_dataset = candidate_val_dataset
            except Exception:
                val_dataset = None

        # Create dataloader
        train_dataloader = DataLoader(
            dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=True,
            num_workers=config.dataloader.num_workers,
            pin_memory=True
        )
        val_dataloader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config.dataloader.batch_size,
                shuffle=False,
                num_workers=config.dataloader.num_workers,
                pin_memory=True
            )

        # Set normalizer
        normalizer = dataset.get_normalizer()
        self.policy.set_normalizer(normalizer)
        self.policy.to(self.device) 

        if self.ema_policy is not None:
            self.ema_policy.set_normalizer(normalizer)
            self.ema_policy.to(self.device)

        self._prepare_policy_optimizer_for_bc()

        # Training loop
        il_loss_per_epoch = []
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
                if self.global_step % config.training.log_every == 0 and config.logging.use_wandb:
                    wandb.log({
                        'il/loss': loss.item(),
                        'il/epoch': epoch,
                        **{f'il/{k}': v for k, v in loss_dict.items()}
                    }, step=self.global_step)

            # Epoch end
            avg_loss = np.mean(epoch_losses)
            il_loss_per_epoch.append(float(avg_loss))
            val_loss = None
            if val_dataloader is not None:
                self.policy.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_batch = dict_apply(val_batch, lambda x: x.to(self.device, non_blocking=True))
                        batch_val_loss, _ = self.policy.compute_loss(val_batch)
                        val_losses.append(batch_val_loss.item())
                self.policy.train()
                if val_losses:
                    val_loss = float(np.mean(val_losses))

            if val_loss is None:
                cprint(f"[IL] Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}", "green")
            else:
                cprint(f"[IL] Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}", "green")

            if val_loss is not None and config.logging.use_wandb:
                wandb.log({
                    'il/val_loss': val_loss,
                    'il/epoch': epoch,
                }, step=self.global_step)

            # Evaluate
            if env_runner is not None and (epoch + 1) % config.training.eval_every == 0:
                eval_policy = self.ema_policy if self.ema_policy else self.policy
                eval_policy.eval()
                with torch.no_grad():
                    metrics = env_runner.run(eval_policy)
                cprint(f"[IL] Eval - Success Rate: {metrics.get('mean_success_rates', 0):.3f}", "green")
                if config.logging.use_wandb:
                    wandb.log({f'il/eval_{k}': v for k, v in metrics.items()}, step=self.global_step)
                eval_policy.train()

        safe_tag = str(plot_tag).replace('/', '_').replace(' ', '_')
        il_plot_path = self._save_loss_curve_plot(
            loss_history=il_loss_per_epoch,
            title=f"IL Loss ({safe_tag})",
            ylabel="IL Loss",
            filename=f"il_loss_{safe_tag}.png"
        )
        if config.logging.use_wandb and il_plot_path is not None:
            wandb.log({f'il/loss_curve_{safe_tag}': wandb.Image(il_plot_path)}, step=self.global_step)

        return {'final_loss': avg_loss}

    def train_transition_model(
        self,
        dataset: BaseDataset,
        max_epochs: int = 200,
        max_epochs_since_update: int = 5,
    ) -> Dict:
        """
        Algorithm 1, Line 6: Train transition T_θm(s'|s, a).

        Freezes the policy encoder and trains the ensemble dynamics model to
        predict (Δobs_features, reward) from (obs_features, norm_action).

        Args:
            dataset                : offline dataset D_m
            max_epochs             : upper bound on training epochs
            max_epochs_since_update: early-stop patience

        Returns:
            metrics: dict with 'transition_val_loss'
        """
        cprint(f"\n[RL100Trainer] Line 6 — Training Transition Model T_θ "
               f"(Iteration {self.offline_rl_iteration})", "cyan")

        # Freeze encoder during transition model training
        self.policy.eval()
        transition_metrics = self.transition_model.train_on_dataset(
            policy=self.policy,
            dataset=dataset,
            batch_size=self.config.dataloader.batch_size,
            num_workers=self.config.dataloader.num_workers,
            max_epochs=max_epochs,
            max_epochs_since_update=max_epochs_since_update,
        )
        val_loss = float(transition_metrics['final_val_loss'])
        train_loss_history = transition_metrics.get('train_loss_history', [])
        val_loss_history = transition_metrics.get('val_loss_history', [])

        train_plot_path = self._save_loss_curve_plot(
            loss_history=train_loss_history,
            title=f"Transition Train Loss (iter {int(self.offline_rl_iteration)})",
            ylabel="Train Loss",
            filename=f"transition_train_loss_iter_{int(self.offline_rl_iteration):02d}.png"
        )
        val_plot_path = self._save_loss_curve_plot(
            loss_history=val_loss_history,
            title=f"Transition Val Loss (iter {int(self.offline_rl_iteration)})",
            ylabel="Val Loss",
            filename=f"transition_val_loss_iter_{int(self.offline_rl_iteration):02d}.png"
        )

        if self.config.logging.use_wandb and self.config.logging.use_wandb:
            log_dict = {
                'transition/val_loss': val_loss,
                'transition/iteration': self.offline_rl_iteration,
            }
            if train_plot_path is not None:
                log_dict['transition/train_loss_curve'] = wandb.Image(train_plot_path)
            if val_plot_path is not None:
                log_dict['transition/val_loss_curve'] = wandb.Image(val_plot_path)
            wandb.log(log_dict, step=self.global_step)

        return {
            'transition_val_loss': val_loss,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }

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

        # Freeze Diffusion Actor during critic-only training (tag2: "冻结 Diffusion Actor，只练 QV 网络")
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)

        self.critics.train()

        # Create dataloader
        train_dataloader = DataLoader(
            dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=True,
            num_workers=config.dataloader.num_workers,
            pin_memory=True
        )

        q_loss_per_epoch = []
        v_loss_per_epoch = []
        for epoch in range(num_epochs):
            v_losses = []
            q_losses = []

            for batch_idx, batch in enumerate(train_dataloader):
                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

                # Extract data
                obs_dict = batch['obs']

                # Encode observations to get features
                with torch.no_grad():
                    obs_features = self._encode_obs_features(obs_dict)

                # Normalize chunk action: the full executed action chunk is one MDP decision.
                naction = self._flatten_normalized_chunk_action(batch['action'])

                # 1. Update V network
                v_loss, v_info = self.critics.compute_v_loss(obs_features, naction)

                self.v_optimizer.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critics.v_network.parameters(), max_norm=10.0)
                self.v_optimizer.step()

                v_losses.append(v_loss.item())

                # 2. Update Q network using actual next_obs from dataset.
                # Encoding the real next observation is more reliable than the
                # transition model, which can fail to generalise when new collected
                # data has a different distribution from the expert demos.
                if 'reward' in batch and 'next_obs' in batch:
                    reward = batch['reward']
                    if reward.dim() == 1:
                        reward = reward.unsqueeze(-1)  # [B] -> [B, 1]
                    # Normalise chunk reward by max per-step reward so that Q* ~ O(10)
                    # rather than O(1000).  Metaworld shaped rewards go up to 10/step;
                    # chunk reward = Σ γ^j r_j ≤ 10 * Σ γ^j ≈ 77.  Without scaling,
                    # Q_pred starts near 0 while Q_target ≈ 56 → Q_loss ≈ 6000+ at epoch 0.
                    reward_scale = float(getattr(config.critics, 'reward_scale', 10.0))
                    reward = reward / reward_scale
                    done = batch.get('done', torch.zeros_like(reward))
                    if done.dim() == 1:
                        done = done.unsqueeze(-1)

                    B = obs_features.shape[0]
                    # Encode actual next obs (at t + n_action_steps) from dataset
                    with torch.no_grad():
                        next_obs_raw = batch['next_obs']  # nested dict, already on device
                        next_obs_features = self._encode_obs_features(next_obs_raw)

                    q_loss, q_info = self.critics.compute_q_loss(
                        obs_features, naction, reward, next_obs_features, done
                    )

                    self.q_optimizer.zero_grad()
                    q_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critics.q_network.parameters(), max_norm=10.0)
                    self.q_optimizer.step()

                    q_losses.append(q_loss.item())

                    # Update target network
                    self.critics.update_target_network(tau=config.critics.target_update_tau)

                    # Log
                    if self.global_step % config.training.log_every == 0 and config.logging.use_wandb:
                        wandb.log({
                            'iql/v_loss': v_loss.item(),
                            'iql/q_loss': q_loss.item(),
                            **{f'iql/{k}': v for k, v in v_info.items()},
                            **{f'iql/{k}': v for k, v in q_info.items()}
                        }, step=self.global_step)

                self.global_step += 1

            v_loss_avg = np.mean(v_losses) if v_losses else 0.0
            q_loss_avg = np.mean(q_losses) if q_losses else 0.0

            # 2. 使用格式化打印
            cprint(f"[IQL] Epoch {epoch}/{num_epochs}, "
                   f"V Loss: {float(v_loss_avg):.4f}, "
                   f"Q Loss: {float(q_loss_avg):.4f}", "green")
            v_loss_per_epoch.append(float(v_loss_avg))
            q_loss_per_epoch.append(float(q_loss_avg))

        q_loss_plot_path = self._save_loss_curve_plot(
            loss_history=q_loss_per_epoch,
            title=f"IQL Q Loss (iter {int(self.offline_rl_iteration)})",
            ylabel="Q Loss",
            filename=f"iql_q_loss_iter_{int(self.offline_rl_iteration):02d}.png"
        )
        v_loss_plot_path = self._save_loss_curve_plot(
            loss_history=v_loss_per_epoch,
            title=f"IQL V Loss (iter {int(self.offline_rl_iteration)})",
            ylabel="V Loss",
            filename=f"iql_v_loss_iter_{int(self.offline_rl_iteration):02d}.png"
        )
        if self.config.logging.use_wandb and q_loss_plot_path is not None:
            wandb.log({
                'iql/q_loss_curve': wandb.Image(q_loss_plot_path),
                'iql/iteration': int(self.offline_rl_iteration),
            }, step=self.global_step)
        if self.config.logging.use_wandb and v_loss_plot_path is not None:
            wandb.log({
                'iql/v_loss_curve': wandb.Image(v_loss_plot_path),
                'iql/iteration': int(self.offline_rl_iteration),
            }, step=self.global_step)

        # Unfreeze policy after critic training
        for param in self.policy.parameters():
            param.requires_grad_(True)
        self.policy.train()

        return {'v_loss': np.mean(v_losses), 'q_loss': np.mean(q_losses) if q_losses else 0}

    def _relabel_demo_rewards(self, dataset: BaseDataset) -> None:
        """
        Label expert demonstration episodes with rewards when zarr has none.

        Reads ``self.config.critics.reward_type``:
          - 'sparse': reward=1 at last step, 0 elsewhere
          - 'dense' : unsupported for unlabeled demos because shaped rewards
                      cannot be reconstructed after the fact
        """
        import numpy as np
        rb = dataset.replay_buffer

        if dataset.has_rl_data:
            cprint("[RL100Trainer] Dataset already contains reward/done labels; keep existing rewards.", "yellow")
            return  # Already has rewards, nothing to do

        reward_type = getattr(self.config.critics, 'reward_type', 'sparse')
        if reward_type != 'sparse':
            raise ValueError(
                "Cannot auto-relabel unlabeled demos with true dense rewards. "
                "Use sparse rewards or provide replay data that already contains reward/done labels."
            )

        n_episodes = rb.n_episodes
        episode_ends = rb.episode_ends[:]
        episode_starts = np.concatenate([[0], episode_ends[:-1]])

        reward_array = np.zeros(rb.n_steps, dtype=np.float32)
        done_array   = np.zeros(rb.n_steps, dtype=np.float32)

        for ep_idx in range(n_episodes):
            start = int(episode_starts[ep_idx])
            end   = int(episode_ends[ep_idx])
            if end > start:
                reward_array[end - 1] = 1.0
                done_array[end - 1] = 1.0

        rb.data.create_dataset('reward', data=reward_array, overwrite=True)
        rb.data.create_dataset('done',   data=done_array,   overwrite=True)

        dataset.has_rl_data = True

        n_total = rb.n_episodes
        existing_mask = np.asarray(
            getattr(dataset, 'train_mask', np.ones(n_total, dtype=bool)),
            dtype=bool,
        )
        episode_mask = existing_mask if len(existing_mask) == n_total else np.ones(n_total, dtype=bool)
        if hasattr(dataset, '_rebuild_sampler'):
            dataset._rebuild_sampler(episode_mask)

        cprint(f"[RL100Trainer] Relabeled {n_episodes} episodes "
               f"({rb.n_steps} steps) with reward_type={reward_type}.", "green")

    def _prepare_amq_eval_feature_batches(
        self,
        dataset: BaseDataset,
        num_batches: int,
        eval_seed: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Prepare a fixed set of initial states for AM-Q evaluation."""
        if "cross_attention" in getattr(self.policy, 'condition_type', ''):
            raise NotImplementedError(
                "AM-Q feature-space rollout currently supports film-style conditioning only."
            )

        from torch.utils.data import DataLoader as _DL

        ope_cfg = getattr(self.config, 'ope', None)
        shuffle_batches = bool(getattr(ope_cfg, 'shuffle_batches', True))
        data_generator = None
        if eval_seed is not None:
            data_generator = torch.Generator()
            data_generator.manual_seed(int(eval_seed))

        loader = _DL(
            dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=shuffle_batches,
            num_workers=0,
            pin_memory=True,
            generator=data_generator,
        )

        feature_batches = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= num_batches:
                    break
                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                obs_features = self._encode_obs_features(batch['obs'])
                feature_batches.append(obs_features.detach().cpu())

        return feature_batches

    def _sample_amq_initial_noise(
        self,
        batch_size: int,
        policy: BasePolicy,
        seed: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        noise_generator = torch.Generator()
        noise_generator.manual_seed(int(seed))
        return torch.randn(
            (batch_size, policy.horizon, policy.action_dim),
            generator=noise_generator,
            dtype=dtype,
        ).to(self.device)

    def _evaluate_policy_amq(
        self,
        dataset: BaseDataset,
        num_batches: int = 20,
        rollout_horizon: int = 5,
        policy: Optional[BasePolicy] = None,
        eval_features: Optional[List[torch.Tensor]] = None,
        eval_seed: Optional[int] = None,
    ) -> float:
        """
        Offline Policy Evaluation using AM-Q (paper Eq.20).

        Estimates policy value by H-step rollout through the transition model:
          Ĵ^{AM-Q}(π) = E_{(s,a)~(T̂,π)} [ Σ_{h=0}^{H-1} Q_ψ(s_h, a_h) ]

        Steps:
          1. Sample initial states from the dataset
          2. For each horizon step h = 0..H-1:
             a) Run the policy to get action a_h = π(s_h)
             b) Accumulate Q(s_h, a_h)
             c) Predict next state s_{h+1} via transition model T̂
          3. Return average cumulative Q

        Args:
            dataset: offline dataset for initial states
            num_batches: number of batches to average over
            rollout_horizon: H-step rollout depth through transition model

        Returns:
            Estimated policy value J_hat.
        """
        ope_cfg = getattr(self.config, 'ope', None)
        deterministic_transition = bool(getattr(ope_cfg, 'deterministic_transition', True))
        use_common_random_numbers = bool(getattr(ope_cfg, 'use_common_random_numbers', True))

        eval_policy = self.policy if policy is None else policy
        policy_was_training = bool(getattr(eval_policy, 'training', False))
        critics_was_training = bool(getattr(self.critics, 'training', False))
        eval_policy.eval()
        self.critics.eval()
        cumulative_q = []
        gamma_chunk = self.critics.gamma  # chunk-level discount

        if eval_features is None:
            eval_features = self._prepare_amq_eval_feature_batches(
                dataset=dataset,
                num_batches=num_batches,
                eval_seed=eval_seed,
            )

        with torch.no_grad():
            for batch_idx, init_features in enumerate(eval_features):
                obs_features = init_features.to(self.device, non_blocking=True)
                # H-step rollout
                batch_q_sum = torch.zeros(obs_features.shape[0], 1, device=self.device)
                cur_features = obs_features

                for h in range(rollout_horizon):
                    initial_noise = None
                    if use_common_random_numbers and eval_seed is not None:
                        noise_seed = int(eval_seed + batch_idx * 1009 + h * 104729)
                        initial_noise = self._sample_amq_initial_noise(
                            batch_size=cur_features.shape[0],
                            policy=eval_policy,
                            seed=noise_seed,
                            dtype=cur_features.dtype,
                        )

                    action_pred = eval_policy.predict_action_from_global_cond(
                        cur_features,
                        initial_noise=initial_noise,
                    )
                    naction = eval_policy.normalizer['action'].normalize(
                        action_pred['action']
                    ).reshape(cur_features.shape[0], -1)

                    # Accumulate discounted Q
                    q = self.critics.get_q_value(cur_features, naction)  # [B, 1]
                    batch_q_sum += (gamma_chunk ** h) * q

                    # Predict next features via transition model
                    if h < rollout_horizon - 1 and hasattr(self, 'transition_model'):
                        try:
                            next_features, _ = self.transition_model.predict_next_features(
                                cur_features,
                                naction,
                                deterministic=deterministic_transition,
                            )
                            cur_features = next_features
                        except Exception:
                            break  # transition model not trained yet

                cumulative_q.append(batch_q_sum.mean().item())

        if policy_was_training:
            eval_policy.train()
        if critics_was_training:
            self.critics.train()
        return float(np.mean(cumulative_q)) if cumulative_q else 0.0

    def offline_rl_optimize(
        self,
        dataset: BaseDataset,
        num_epochs: int
    ) -> Dict:
        """
        Phase 2b: Optimize policy with PPO and consistency distillation.

        Implements paper lines 7-8 of Algorithm 1 (OfflineRL):
          1. Save behavior-policy snapshot for OPE gate
          2. Freeze obs_encoder (paper: "keep ϕIL fixed")
          3. Normalize advantages (prevent gradient explosion)
          4. K-step PPO update + Consistency Distillation
          5. OPE gate: accept update only if AM-Q improves by ≥ δ (paper Eq.20)

        Args:
            dataset: Dataset with (s, a, r, s', done)
            num_epochs: Number of training epochs

        Returns:
            metrics: Dictionary with training metrics
        """
        cprint(f"\n[RL100Trainer] Phase 2b: Offline RL Optimization (Iteration {self.offline_rl_iteration})", "cyan")

        config = self.config
        ope_cfg = getattr(config, 'ope', None)
        ope_num_batches = int(getattr(ope_cfg, 'num_batches', 20))
        ope_rollout_horizon = int(getattr(ope_cfg, 'rollout_horizon', 5))
        ope_seed_base = int(getattr(ope_cfg, 'seed', int(getattr(config.training, 'seed', 0))))
        ope_seed = ope_seed_base + 10007 * int(self.offline_rl_iteration)

        # ── Step 0: Prepare a fixed AM-Q evaluation context for fair OPE gating ─
        ope_eval_features = self._prepare_amq_eval_feature_batches(
            dataset=dataset,
            num_batches=ope_num_batches,
            eval_seed=ope_seed,
        )
        j_old = self._evaluate_policy_amq(
            dataset=dataset,
            num_batches=ope_num_batches,
            rollout_horizon=ope_rollout_horizon,
            policy=self.policy,
            eval_features=ope_eval_features,
            eval_seed=ope_seed,
        )
        cprint(f"[OPE] Behavior policy value J_old = {j_old:.4f}", "cyan")

        # ── Step 1: Reset policy optimizer state for RL fine-tuning ──────────────
        # Reusing Adam moments from IL / checkpoint resume can make the first PPO
        # step disproportionately large even with a small RL learning rate.
        self._prepare_policy_optimizer_for_rl()

        # ── Step 1: Save snapshot for potential rollback ──────────────────────────
        policy_snapshot = copy.deepcopy(self.policy.state_dict())
        ema_snapshot = copy.deepcopy(self.ema_policy.state_dict()) if self.ema_policy is not None else None
        ema_helper_snapshot = None
        if self.ema_policy is not None and hasattr(self, 'ema'):
            ema_helper_snapshot = {
                'decay': self.ema.decay,
                'optimization_step': self.ema.optimization_step,
            }
        consistency_snapshot = copy.deepcopy(self.consistency_model.state_dict())
        policy_optimizer_snapshot = copy.deepcopy(self.policy_optimizer.state_dict())
        consistency_optimizer_snapshot = copy.deepcopy(self.consistency_optimizer.state_dict())

        # ── Step 2: Freeze encoder (paper: ϕIL is fixed during offline RL) ───────
        for param in self.policy.obs_encoder.parameters():
            param.requires_grad_(False)

        self.policy.train()
        self.critics.eval()

        # Create dataloader
        train_dataloader = DataLoader(
            dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=True,
            num_workers=config.dataloader.num_workers,
            pin_memory=True
        )

        # Number of inner PPO gradient steps per batch (allows ratio to deviate from 1)
        ppo_inner_steps = getattr(config.training, 'ppo_inner_steps', 4)
        # Paper Eq.22: L_total = L_RL + λ_CD · L_CD
        lambda_cd = getattr(config.training, 'lambda_cd', 1.0)

        ppo_loss_per_epoch = []
        for epoch in range(num_epochs):
            ppo_losses = []
            cd_losses = []
            reg_losses = []
            grad_norms = []
            approx_kls = []
            clip_fracs = []
            mean_ratios = []
            min_ratios = []
            max_ratios = []
            mean_abs_ratio_devs = []

            for batch_idx, batch in enumerate(train_dataloader):
                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

                obs_dict = batch['obs']

                # Encode observations (encoder is frozen — no_grad is redundant but explicit)
                with torch.no_grad():
                    obs_features, global_cond = self._encode_obs_representations(obs_dict)
                    naction = self._flatten_normalized_chunk_action(batch['action'])

                    # Compute raw advantage A = Q(s,a) - V(s)
                    advantages = self.critics.compute_advantage(obs_features, naction)

                # ── Step 3: Normalise advantages (prevent gradient explosion) ─────
                adv_mean = advantages.mean()
                adv_std = advantages.std() + 1e-8
                advantages = (advantages - adv_mean) / adv_std

                # ── Sample old trajectory ONCE per batch (fix old policy) ─────────
                trajectory_old, log_probs_old = self.policy.sample_for_ppo(global_cond=global_cond)
                run_cd_this_batch = (self.global_step % config.training.cd_every == 0)
                batch_ppo_losses = []
                batch_approx_kls = []
                batch_clip_fracs = []
                batch_mean_ratios = []
                batch_min_ratios = []
                batch_max_ratios = []
                batch_mean_abs_ratio_devs = []

                # ── Inner PPO loop: N gradient steps with SAME old trajectory ─────
                for ppo_inner in range(ppo_inner_steps):
                    ppo_loss, _ = self.policy.compute_ppo_loss(
                        obs_dict=None,
                        old_log_probs=log_probs_old,
                        advantages=advantages,
                        trajectory=trajectory_old,
                        global_cond=global_cond,
                    )

                    # Apply consistency distillation at most once per outer batch.
                    total_loss = ppo_loss
                    reg_loss, reg_info = self._compute_obs_regularization(obs_dict)
                    if reg_info['total_reg_loss'] > 0:
                        total_loss = total_loss + reg_loss
                        reg_losses.append(reg_info['total_reg_loss'])
                    run_cd_step = run_cd_this_batch and ppo_inner == 0
                    if run_cd_step:
                        cd_loss, cd_info = self.consistency_distillation.compute_distillation_loss(
                            obs_dict,
                            global_cond=global_cond,
                        )
                        total_loss = total_loss + lambda_cd * cd_loss
                        cd_losses.append(cd_info['cd_loss'])

                    self.policy_optimizer.zero_grad()
                    # Also zero CD optimizer if joint
                    self.consistency_optimizer.zero_grad()
                    total_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        config.training.max_grad_norm
                    )
                    grad_norms.append(float(grad_norm))
                    self.policy_optimizer.step()
                    # Update consistency model params too
                    if run_cd_step:
                        self.consistency_optimizer.step()

                    ppo_loss_display = float(-ppo_loss.item())
                    with torch.no_grad():
                        _, ppo_info_post = self.policy.compute_ppo_loss(
                            obs_dict=None,
                            old_log_probs=log_probs_old,
                            advantages=advantages,
                            trajectory=trajectory_old,
                            global_cond=global_cond,
                        )
                    batch_ppo_losses.append(ppo_loss_display)
                    batch_approx_kls.append(float(ppo_info_post.get('approx_kl', 0.0)))
                    batch_clip_fracs.append(float(ppo_info_post.get('clip_frac', 0.0)))
                    batch_mean_ratios.append(float(ppo_info_post.get('mean_ratio', 1.0)))
                    batch_min_ratios.append(float(ppo_info_post.get('min_ratio', 1.0)))
                    batch_max_ratios.append(float(ppo_info_post.get('max_ratio', 1.0)))
                    batch_mean_abs_ratio_devs.append(float(ppo_info_post.get('mean_abs_ratio_dev', 0.0)))

                ppo_loss_display = float(np.mean(batch_ppo_losses)) if batch_ppo_losses else 0.0
                ppo_losses.append(ppo_loss_display)
                approx_kls.extend(batch_approx_kls)
                clip_fracs.extend(batch_clip_fracs)
                mean_ratios.extend(batch_mean_ratios)
                min_ratios.extend(batch_min_ratios)
                max_ratios.extend(batch_max_ratios)
                mean_abs_ratio_devs.extend(batch_mean_abs_ratio_devs)

                # Update EMA
                if self.ema_policy is not None:
                    self.ema.step(self.policy)

                # Log
                if self.global_step % config.training.log_every == 0 and config.logging.use_wandb:
                    log_dict = {
                        'ppo/loss': ppo_loss_display,
                        'ppo/grad_norm': grad_norms[-1] if grad_norms else 0.0,
                        'ppo/approx_kl': float(np.mean(batch_approx_kls)) if batch_approx_kls else 0.0,
                        'ppo/clip_frac': float(np.mean(batch_clip_fracs)) if batch_clip_fracs else 0.0,
                        'ppo/mean_ratio': float(np.mean(batch_mean_ratios)) if batch_mean_ratios else 1.0,
                        'ppo/min_ratio': float(np.min(batch_min_ratios)) if batch_min_ratios else 1.0,
                        'ppo/max_ratio': float(np.max(batch_max_ratios)) if batch_max_ratios else 1.0,
                        'ppo/mean_abs_ratio_dev': float(np.mean(batch_mean_abs_ratio_devs)) if batch_mean_abs_ratio_devs else 0.0,
                    }
                    if reg_losses:
                        log_dict['ppo/obs_reg_loss'] = reg_losses[-1]
                    if cd_losses:
                        log_dict['cd/loss'] = cd_losses[-1]
                    wandb.log(log_dict, step=self.global_step)

                self.global_step += 1

            epoch_ppo_loss = float(np.mean(ppo_losses) if ppo_losses else 0.0)
            cprint(f"[Offline RL] Epoch {epoch}/{num_epochs}, PPO Loss: {epoch_ppo_loss:.4f}, "
                   f"PostKL: {np.mean(approx_kls) if approx_kls else 0.0:.3e}, "
                   f"PostClipFrac: {np.mean(clip_fracs) if clip_fracs else 0.0:.6f}, "
                   f"PostMeanRatio: {np.mean(mean_ratios) if mean_ratios else 1.0:.6f}, "
                   f"PostRatioDev: {np.mean(mean_abs_ratio_devs) if mean_abs_ratio_devs else 0.0:.3e}, "
                   f"GradNorm: {np.mean(grad_norms) if grad_norms else 0.0:.4f}, "
                   f"Reg Loss: {np.mean(reg_losses) if reg_losses else 0:.4f}, "
                   f"CD Loss: {np.mean(cd_losses) if cd_losses else 0:.4f}", "green")
            ppo_loss_per_epoch.append(epoch_ppo_loss)

        ppo_loss_plot_path = self._save_loss_curve_plot(
            loss_history=ppo_loss_per_epoch,
            title=f"PPO Loss (iter {int(self.offline_rl_iteration)})",
            ylabel="PPO Loss",
            filename=f"ppo_loss_iter_{int(self.offline_rl_iteration):02d}.png"
        )
        if config.logging.use_wandb and ppo_loss_plot_path is not None:
            wandb.log({
                'ppo/loss_curve': wandb.Image(ppo_loss_plot_path),
                'ppo/iteration': int(self.offline_rl_iteration),
            }, step=self.global_step)

        # ── Step 4: Unfreeze encoder ──────────────────────────────────────────────
        for param in self.policy.obs_encoder.parameters():
            param.requires_grad_(True)

        # ── Step 5: OPE Gate (paper Eq.20) ───────────────────────────────────────
        # Accept the PPO-updated policy only if AM-Q improves by δ = 0.05·|J_old|.
        # Otherwise roll back to the behavior-policy snapshot.
        j_new = self._evaluate_policy_amq(
            dataset=dataset,
            num_batches=ope_num_batches,
            rollout_horizon=ope_rollout_horizon,
            policy=self.policy,
            eval_features=ope_eval_features,
            eval_seed=ope_seed,
        )
        delta_coef = float(getattr(ope_cfg, 'delta_coef', 0.05))
        delta_abs_min = float(getattr(ope_cfg, 'delta_abs_min', 0.0))
        delta = max(delta_abs_min, delta_coef * abs(j_old)) if j_old != 0 else delta_abs_min
        if j_new - j_old >= delta:
            cprint(f"[OPE] Policy ACCEPTED: J_new={j_new:.4f} > J_old={j_old:.4f} + δ={delta:.4f}", "green")
        else:
            cprint(f"[OPE] Policy REJECTED: J_new={j_new:.4f} ≤ J_old={j_old:.4f} + δ={delta:.4f}. "
                   f"Rolling back to behavior policy.", "yellow")
            self.policy.load_state_dict(policy_snapshot)
            if self.ema_policy is not None and ema_snapshot is not None:
                self.ema_policy.load_state_dict(ema_snapshot)
            if ema_helper_snapshot is not None and hasattr(self, 'ema'):
                self.ema.decay = ema_helper_snapshot['decay']
                self.ema.optimization_step = ema_helper_snapshot['optimization_step']
            self.consistency_model.load_state_dict(consistency_snapshot)
            self.policy_optimizer.load_state_dict(policy_optimizer_snapshot)
            self.consistency_optimizer.load_state_dict(consistency_optimizer_snapshot)

        if config.logging.use_wandb:
            wandb.log({
                'ope/j_old': j_old,
                'ope/j_new': j_new,
                'ope/accepted': int(j_new - j_old >= delta),
            }, step=self.global_step)

        return {
            'ppo_loss': np.mean(ppo_losses),
            'approx_kl': np.mean(approx_kls) if approx_kls else 0,
            'clip_frac': np.mean(clip_fracs) if clip_fracs else 0,
            'mean_ratio': np.mean(mean_ratios) if mean_ratios else 1.0,
            'mean_abs_ratio_dev': np.mean(mean_abs_ratio_devs) if mean_abs_ratio_devs else 0,
            'reg_loss': np.mean(reg_losses) if reg_losses else 0,
            'cd_loss': np.mean(cd_losses) if cd_losses else 0
        }

    def collect_new_data(
        self,
        env_runner,
        num_episodes: int,
        policy_mode: Optional[str] = None,
        use_ema: Optional[bool] = None,
        collect_trajectory: bool = False,
    ) -> Tuple[Dict, list]:
        """
        Phase 2c: Collect new data by rolling out policy in environment.

        Uses run_and_collect() to capture both metrics and raw trajectory data
        for subsequent dataset merging.

        Returns:
            metrics  : success rate, reward, etc.
            episodes : list of episode dicts (numpy arrays) for merging
        """
        cprint(f"\n[RL100Trainer] Phase 2c: Collecting New Data (Iteration {self.offline_rl_iteration})", "cyan")

        collection_mode = policy_mode
        if collection_mode is None:
            collection_mode = getattr(self.config.runtime, 'collection_policy', 'ddim') if 'runtime' in self.config else 'ddim'
        collection_use_ema = False if use_ema is None else bool(use_ema)
        if use_ema is None and 'runtime' in self.config:
            collection_use_ema = bool(getattr(self.config.runtime, 'collection_use_ema', False))
        eval_policy = self.get_runtime_policy(mode=collection_mode, use_ema=collection_use_ema)
        eval_policy.eval()

        with torch.no_grad():
            reward_type = getattr(self.config.critics, 'reward_type', 'sparse')
            chunk_gamma = float(getattr(self.config.task.dataset, 'gamma', 0.99))
            metrics, episodes = env_runner.run_and_collect(
                eval_policy,
                num_episodes=num_episodes,
                reward_type=reward_type,
                gamma=chunk_gamma,
                collect_trajectory=collect_trajectory,
            )

        cprint(f"[Data Collection] Success Rate: {metrics.get('mean_success_rates', 0):.3f}, "
               f"EnvReturn: {metrics.get('mean_env_rewards', metrics.get('mean_traj_rewards', 0)):.2f}, "
               f"RLReward: {metrics.get('mean_rl_rewards', 0):.2f}, "
               f"Episodes: {len(episodes)}, Steps: {metrics.get('n_steps', 0)}", "green")

        if self.config.logging.use_wandb:
            wandb.log({
                'collection/success_rate': metrics.get('mean_success_rates', 0),
                'collection/reward':       metrics.get('mean_rl_rewards', metrics.get('mean_traj_rewards', 0)),
                'collection/env_return':   metrics.get('mean_env_rewards', metrics.get('mean_traj_rewards', 0)),
                'collection/rl_reward':    metrics.get('mean_rl_rewards', 0),
                'collection/n_steps':      metrics.get('n_steps', 0),
            }, step=self.global_step)

        eval_policy.train()
        return metrics, episodes

    def _build_il_episode_mask(
        self,
        episodes: List[dict],
        stage: str,
    ) -> np.ndarray:
        if not episodes:
            return np.zeros((0,), dtype=bool)

        merge_success_only = bool(getattr(self.config.runtime, 'merge_success_only', False))
        if not merge_success_only:
            return np.ones(len(episodes), dtype=bool)

        il_episode_mask = []
        for episode in episodes:
            if 'success' in episode:
                is_success = bool(episode['success'])
            else:
                reward = np.asarray(episode.get('reward', []), dtype=np.float32)
                is_success = bool(np.any(reward > 0))
            il_episode_mask.append(is_success)

        il_episode_mask = np.asarray(il_episode_mask, dtype=bool)
        dropped = len(episodes) - int(il_episode_mask.sum())
        cprint(
            f"[Dataset] {stage}: all {len(episodes)} episodes remain in RL replay; "
            f"IL retrain keeps {int(il_episode_mask.sum())}/{len(episodes)} successful episodes "
            f"(drops {dropped} failures).",
            "cyan",
        )
        return il_episode_mask

    def online_rl_optimize(
        self,
        online_episodes: List[dict],
        num_epochs: int,
        online_iteration: Optional[int] = None,
    ) -> Dict:
        """
        Phase 3: On-policy online RL with GAE advantages and PPO updates.

        Unlike offline RL, this stage uses fresh rollout trajectories and the
        stored behavior-policy denoising trajectories collected during rollout.
        """
        config = self.config
        lambda_v = float(getattr(config.training, 'lambda_v', 0.5))
        gae_lambda = float(getattr(config.training, 'gae_lambda', 0.95))
        gamma_chunk = self.critics.gamma
        lambda_cd = float(getattr(config.training, 'lambda_cd', 1.0))

        decision_episodes = []
        for ep in online_episodes:
            steps = [
                step for step in ep.get('decision_steps', [])
                if 'trajectory' in step and 'log_probs_old' in step
            ]
            if steps:
                decision_episodes.append(steps)

        if not decision_episodes:
            cprint("[Online RL] No decision-level PPO data available; skip online optimization.", "yellow")
            return {'v_loss': 0.0, 'ppo_loss': 0.0, 'n_decisions': 0}

        self.policy.eval()
        self.critics.eval()

        rollout_buffer = []
        with torch.no_grad():
            for ep_steps in decision_episodes:
                obs_dict = {
                    'point_cloud': torch.from_numpy(
                        np.stack([step['obs']['point_cloud'] for step in ep_steps], axis=0)
                    ).to(self.device),
                    'agent_pos': torch.from_numpy(
                        np.stack([step['obs']['agent_pos'] for step in ep_steps], axis=0)
                    ).to(self.device),
                }
                next_obs_dict = {
                    'point_cloud': torch.from_numpy(
                        np.stack([step['next_obs']['point_cloud'] for step in ep_steps], axis=0)
                    ).to(self.device),
                    'agent_pos': torch.from_numpy(
                        np.stack([step['next_obs']['agent_pos'] for step in ep_steps], axis=0)
                    ).to(self.device),
                }
                rewards = torch.as_tensor(
                    [step['reward'] for step in ep_steps], dtype=torch.float32, device=self.device
                )
                dones = torch.as_tensor(
                    [step['done'] for step in ep_steps], dtype=torch.float32, device=self.device
                )

                obs_features = self._encode_obs_features(obs_dict)
                next_obs_features = self._encode_obs_features(next_obs_dict)
                values = self.critics.get_value(obs_features).squeeze(-1)
                next_values = self.critics.get_value(next_obs_features).squeeze(-1)

                advantages = torch.zeros_like(rewards)
                last_gae = torch.zeros((), dtype=torch.float32, device=self.device)
                for t_step in reversed(range(len(ep_steps))):
                    delta = rewards[t_step] + gamma_chunk * (1 - dones[t_step]) * next_values[t_step] - values[t_step]
                    last_gae = delta + gamma_chunk * gae_lambda * (1 - dones[t_step]) * last_gae
                    advantages[t_step] = last_gae
                returns = advantages + values

                for idx, step in enumerate(ep_steps):
                    rollout_buffer.append({
                        'obs': step['obs'],
                        'trajectory': step['trajectory'],
                        'log_probs_old': step['log_probs_old'],
                        'obs_features': obs_features[idx].detach().cpu(),
                        'return': float(returns[idx].item()),
                        'advantage': float(advantages[idx].item()),
                    })

        all_advantages = torch.as_tensor(
            [item['advantage'] for item in rollout_buffer],
            dtype=torch.float32,
            device=self.device,
        )
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        for item, value in zip(rollout_buffer, all_advantages.detach().cpu().tolist()):
            item['advantage'] = float(value)

        # Value-function regression from Eq.21.
        self.critics.train()
        batch_size = min(config.dataloader.batch_size, len(rollout_buffer))
        critic_indices = np.arange(len(rollout_buffer))
        v_losses = []
        v_loss_per_epoch = []
        for critic_epoch in range(5):
            np.random.shuffle(critic_indices)
            epoch_v_losses = []
            for start in range(0, len(critic_indices), batch_size):
                idx = critic_indices[start:start + batch_size]
                obs_features = torch.stack([rollout_buffer[i]['obs_features'] for i in idx]).to(self.device)
                returns = torch.as_tensor(
                    [rollout_buffer[i]['return'] for i in idx],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(-1)
                v_pred = self.critics.get_value(obs_features)
                v_loss = lambda_v * F.mse_loss(v_pred, returns)

                self.v_optimizer.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critics.v_network.parameters(), config.training.max_grad_norm)
                self.v_optimizer.step()
                self.critics.update_target_network(tau=config.critics.target_update_tau)
                v_losses.append(v_loss.item())
                epoch_v_losses.append(v_loss.item())
            if epoch_v_losses:
                v_loss_per_epoch.append(float(np.mean(epoch_v_losses)))

        # On-policy PPO over the stored rollout buffer.
        self.policy.train()
        self.critics.eval()
        ppo_losses = []
        cd_losses = []
        reg_losses = []
        grad_norms = []
        approx_kls = []
        clip_fracs = []
        policy_indices = np.arange(len(rollout_buffer))
        for epoch in range(num_epochs):
            np.random.shuffle(policy_indices)
            epoch_ppo_losses = []
            epoch_cd_losses = []
            epoch_reg_losses = []
            epoch_grad_norms = []
            epoch_approx_kls = []
            epoch_clip_fracs = []
            epoch_mean_ratios = []
            epoch_min_ratios = []
            epoch_max_ratios = []
            epoch_mean_abs_ratio_devs = []
            for start in range(0, len(policy_indices), batch_size):
                idx = policy_indices[start:start + batch_size]
                obs_dict = {
                    'point_cloud': torch.from_numpy(
                        np.stack([rollout_buffer[i]['obs']['point_cloud'] for i in idx], axis=0)
                    ).to(self.device),
                    'agent_pos': torch.from_numpy(
                        np.stack([rollout_buffer[i]['obs']['agent_pos'] for i in idx], axis=0)
                    ).to(self.device),
                }
                trajectories = [
                    torch.from_numpy(
                        np.stack([rollout_buffer[i]['trajectory'][k] for i in idx], axis=0)
                    ).to(self.device)
                    for k in range(len(rollout_buffer[idx[0]]['trajectory']))
                ]
                old_log_probs = [
                    torch.from_numpy(
                        np.stack([rollout_buffer[i]['log_probs_old'][k] for i in idx], axis=0)
                    ).to(self.device).view(-1)
                    for k in range(len(rollout_buffer[idx[0]]['log_probs_old']))
                ]
                advantages = torch.as_tensor(
                    [rollout_buffer[i]['advantage'] for i in idx],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(-1)
                _, global_cond = self._encode_obs_representations(obs_dict)

                ppo_loss, _ = self.policy.compute_ppo_loss(
                    obs_dict=None,
                    old_log_probs=old_log_probs,
                    advantages=advantages,
                    trajectory=trajectories,
                    global_cond=global_cond,
                )
                ppo_loss_display = float(-ppo_loss.item())

                total_loss = ppo_loss
                reg_loss, reg_info = self._compute_obs_regularization(obs_dict)
                if reg_info['total_reg_loss'] > 0:
                    total_loss = total_loss + reg_loss
                    epoch_reg_losses.append(reg_info['total_reg_loss'])
                run_cd_step = (self.global_step % config.training.cd_every == 0)
                if run_cd_step:
                    cd_loss, cd_info = self.consistency_distillation.compute_distillation_loss(
                        obs_dict,
                        global_cond=global_cond,
                    )
                    total_loss = total_loss + lambda_cd * cd_loss
                    epoch_cd_losses.append(cd_info['cd_loss'])

                self.policy_optimizer.zero_grad()
                self.consistency_optimizer.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    config.training.max_grad_norm
                )
                epoch_grad_norms.append(float(grad_norm))
                self.policy_optimizer.step()
                if run_cd_step:
                    self.consistency_optimizer.step()

                if self.ema_policy is not None:
                    self.ema.step(self.policy)

                with torch.no_grad():
                    _, ppo_info_post = self.policy.compute_ppo_loss(
                        obs_dict=None,
                        old_log_probs=old_log_probs,
                        advantages=advantages,
                        trajectory=trajectories,
                        global_cond=global_cond,
                    )

                if config.logging.use_wandb and self.global_step % config.training.log_every == 0:
                    log_dict = {
                        'online/ppo_loss': ppo_loss_display,
                        'online/grad_norm': epoch_grad_norms[-1] if epoch_grad_norms else 0.0,
                        'online/approx_kl': float(ppo_info_post.get('approx_kl', 0.0)),
                        'online/clip_frac': float(ppo_info_post.get('clip_frac', 0.0)),
                        'online/mean_ratio': float(ppo_info_post.get('mean_ratio', 1.0)),
                        'online/min_ratio': float(ppo_info_post.get('min_ratio', 1.0)),
                        'online/max_ratio': float(ppo_info_post.get('max_ratio', 1.0)),
                        'online/mean_abs_ratio_dev': float(ppo_info_post.get('mean_abs_ratio_dev', 0.0)),
                    }
                    if epoch_reg_losses:
                        log_dict['online/obs_reg_loss'] = epoch_reg_losses[-1]
                    if epoch_cd_losses:
                        log_dict['online/cd_loss'] = epoch_cd_losses[-1]
                    wandb.log(log_dict, step=self.global_step)

                self.global_step += 1
                epoch_ppo_losses.append(ppo_loss_display)
                epoch_approx_kls.append(float(ppo_info_post.get('approx_kl', 0.0)))
                epoch_clip_fracs.append(float(ppo_info_post.get('clip_frac', 0.0)))
                epoch_mean_ratios.append(float(ppo_info_post.get('mean_ratio', 1.0)))
                epoch_min_ratios.append(float(ppo_info_post.get('min_ratio', 1.0)))
                epoch_max_ratios.append(float(ppo_info_post.get('max_ratio', 1.0)))
                epoch_mean_abs_ratio_devs.append(float(ppo_info_post.get('mean_abs_ratio_dev', 0.0)))

            if epoch_ppo_losses:
                ppo_losses.append(float(np.mean(epoch_ppo_losses)))
            if epoch_cd_losses:
                cd_losses.append(float(np.mean(epoch_cd_losses)))
            if epoch_reg_losses:
                reg_losses.append(float(np.mean(epoch_reg_losses)))
            if epoch_grad_norms:
                grad_norms.append(float(np.mean(epoch_grad_norms)))
            if epoch_approx_kls:
                approx_kls.append(float(np.mean(epoch_approx_kls)))
            if epoch_clip_fracs:
                clip_fracs.append(float(np.mean(epoch_clip_fracs)))
            cprint(
                f"[Online RL] Epoch {epoch + 1}/{num_epochs}, "
                f"PPO Loss: {np.mean(epoch_ppo_losses) if epoch_ppo_losses else 0.0:.4f}, "
                f"PostKL: {np.mean(epoch_approx_kls) if epoch_approx_kls else 0.0:.3e}, "
                f"PostClipFrac: {np.mean(epoch_clip_fracs) if epoch_clip_fracs else 0.0:.6f}, "
                f"PostMeanRatio: {np.mean(epoch_mean_ratios) if epoch_mean_ratios else 1.0:.6f}, "
                f"PostRatioDev: {np.mean(epoch_mean_abs_ratio_devs) if epoch_mean_abs_ratio_devs else 0.0:.3e}, "
                f"GradNorm: {np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0:.4f}, "
                f"Reg Loss: {np.mean(epoch_reg_losses) if epoch_reg_losses else 0.0:.4f}, "
                f"CD Loss: {np.mean(epoch_cd_losses) if epoch_cd_losses else 0.0:.4f}",
                "green",
            )

        iter_tag = int(online_iteration) if online_iteration is not None else 0
        online_v_plot_path = self._save_loss_curve_plot(
            loss_history=v_loss_per_epoch,
            title=f"Online V Loss (iter {iter_tag})",
            ylabel="V Loss",
            filename=f"online_v_loss_iter_{iter_tag:02d}.png"
        )
        online_ppo_plot_path = self._save_loss_curve_plot(
            loss_history=ppo_losses,
            title=f"Online PPO Loss (iter {iter_tag})",
            ylabel="PPO Loss",
            filename=f"online_ppo_loss_iter_{iter_tag:02d}.png"
        )
        online_kl_plot_path = self._save_loss_curve_plot(
            loss_history=approx_kls,
            title=f"Online PPO KL (iter {iter_tag})",
            ylabel="Approx KL",
            filename=f"online_ppo_kl_iter_{iter_tag:02d}.png"
        )
        online_clip_plot_path = self._save_loss_curve_plot(
            loss_history=clip_fracs,
            title=f"Online PPO ClipFrac (iter {iter_tag})",
            ylabel="ClipFrac",
            filename=f"online_ppo_clipfrac_iter_{iter_tag:02d}.png"
        )
        online_grad_plot_path = self._save_loss_curve_plot(
            loss_history=grad_norms,
            title=f"Online PPO GradNorm (iter {iter_tag})",
            ylabel="GradNorm",
            filename=f"online_ppo_gradnorm_iter_{iter_tag:02d}.png"
        )
        online_reg_plot_path = self._save_loss_curve_plot(
            loss_history=reg_losses,
            title=f"Online Reg Loss (iter {iter_tag})",
            ylabel="Reg Loss",
            filename=f"online_reg_loss_iter_{iter_tag:02d}.png"
        )
        online_cd_plot_path = self._save_loss_curve_plot(
            loss_history=cd_losses,
            title=f"Online CD Loss (iter {iter_tag})",
            ylabel="CD Loss",
            filename=f"online_cd_loss_iter_{iter_tag:02d}.png"
        )
        if config.logging.use_wandb:
            image_logs = {}
            if online_v_plot_path is not None:
                image_logs['online/v_loss_curve'] = wandb.Image(online_v_plot_path)
            if online_ppo_plot_path is not None:
                image_logs['online/ppo_loss_curve'] = wandb.Image(online_ppo_plot_path)
            if online_kl_plot_path is not None:
                image_logs['online/ppo_kl_curve'] = wandb.Image(online_kl_plot_path)
            if online_clip_plot_path is not None:
                image_logs['online/ppo_clipfrac_curve'] = wandb.Image(online_clip_plot_path)
            if online_grad_plot_path is not None:
                image_logs['online/ppo_gradnorm_curve'] = wandb.Image(online_grad_plot_path)
            if online_reg_plot_path is not None:
                image_logs['online/reg_loss_curve'] = wandb.Image(online_reg_plot_path)
            if online_cd_plot_path is not None:
                image_logs['online/cd_loss_curve'] = wandb.Image(online_cd_plot_path)
            if image_logs:
                image_logs['online/iteration'] = iter_tag
                wandb.log(image_logs, step=self.global_step)

        return {
            'v_loss': float(np.mean(v_losses)) if v_losses else 0.0,
            'ppo_loss': float(np.mean(ppo_losses)) if ppo_losses else 0.0,
            'grad_norm': float(np.mean(grad_norms)) if grad_norms else 0.0,
            'approx_kl': float(np.mean(approx_kls)) if approx_kls else 0.0,
            'clip_frac': float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            'reg_loss': float(np.mean(reg_losses)) if reg_losses else 0.0,
            'cd_loss': float(np.mean(cd_losses)) if cd_losses else 0.0,
            'n_decisions': len(rollout_buffer),
        }

    def run_pipeline(
        self,
        initial_dataset: BaseDataset,
        env_runner: BaseRunner,
        num_offline_iterations: int = 5,
        skip_il: bool = False,
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
            skip_il: If True, skip Phase 1 (IL) and resume RL from loaded checkpoint.
                     Normalizer will be synced from the dataset but no training is done.
        """
        config = self.config

        cprint("\n" + "="*80, "magenta")
        cprint(" "*20 + "RL-100 TRAINING PIPELINE", "magenta")
        cprint("="*80 + "\n", "magenta")

        # ============================================
        # Phase 1: Initial Imitation Learning
        # ============================================
        if skip_il:
            cprint("\n[RL100Trainer] Skipping IL phase — loaded from checkpoint.", "yellow")
            # Sync normalizer from dataset so downstream code (IQL, PPO) works correctly
            normalizer = initial_dataset.get_normalizer()
            self.policy.set_normalizer(normalizer)
            self.policy.to(self.device)
            if self.ema_policy is not None:
                self.ema_policy.set_normalizer(normalizer)
                self.ema_policy.to(self.device)
            cprint(f"[RL100Trainer] Normalizer synced from dataset. "
                   f"Resuming offline RL from iteration {self.offline_rl_iteration}.", "yellow")
        else:
            self.train_imitation_learning(
                dataset=initial_dataset,
                num_epochs=config.training.il_epochs,
                env_runner=env_runner,
                plot_tag='initial'
            )
            # Save IL checkpoint
            self.save_checkpoint(tag='after_il')

        # ============================================
        # Phase 2: Offline RL Loop
        # ============================================
        current_dataset = initial_dataset

        # Store original VIB betas from config for dynamic reduction/restoration.
        # Paper Eq.17: during RL fine-tuning reduce by 10×; during IL retraining restore.
        _vib_beta_recon_orig = float(config.policy.get('beta_recon', 1.0))
        _vib_beta_kl_orig    = float(config.policy.get('beta_kl', 0.001))

        def _apply_vib_betas(factor: float):
            """Set VIB betas = original * factor on policy (and ema_policy)."""
            if not getattr(self.policy, 'use_recon_vib', False):
                return
            self.policy.obs_encoder.beta_recon = _vib_beta_recon_orig * factor
            self.policy.obs_encoder.beta_kl    = _vib_beta_kl_orig    * factor
            if self.ema_policy is not None:
                self.ema_policy.obs_encoder.beta_recon = _vib_beta_recon_orig * factor
                self.ema_policy.obs_encoder.beta_kl    = _vib_beta_kl_orig    * factor
            cprint(f"[RL100] VIB betas set to factor={factor}: "
                   f"beta_recon={self.policy.obs_encoder.beta_recon:.6f}, "
                   f"beta_kl={self.policy.obs_encoder.beta_kl:.6f}", "yellow")

        # Ensure rewards exist in the dataset before training critics.
        # Expert demo zarrs typically have no reward/done labels;
        # relabel them with sparse success rewards so Q-training works.
        self._relabel_demo_rewards(current_dataset)

        # When resuming from a checkpoint, skip iterations already completed.
        # - after_il.ckpt has offline_rl_iteration=0 → start_iteration=0 (nothing done yet)
        # - offline_iter_N.ckpt has offline_rl_iteration=N → start_iteration=N+1
        start_iteration = self.offline_rl_iteration + 1 if skip_il and self.offline_rl_iteration > 0 else 0
        if skip_il and start_iteration > 0:
            cprint(f"[RL100Trainer] Skipping offline RL iterations 0~{start_iteration - 1} "
                   f"(already completed in checkpoint).", "yellow")

        for iteration in range(start_iteration, num_offline_iterations):
            self.offline_rl_iteration = iteration

            cprint("\n" + "="*80, "yellow")
            cprint(f" "*15 + f"OFFLINE RL ITERATION {iteration + 1}/{num_offline_iterations}", "yellow")
            cprint("="*80 + "\n", "yellow")

            # 2a) Train Transition Model  (Algorithm 1, Line 6)
            self.train_transition_model(
                dataset=current_dataset,
                max_epochs=200,
                max_epochs_since_update=5,
            )

            # 2b) Train IQL Critics  (Algorithm 1, Line 5)
            self.train_iql_critics(
                dataset=current_dataset,
                num_epochs=config.training.critic_epochs
            )

            # 2c) Optimize Policy  (Algorithm 1, Lines 7-8)
            # Paper Eq.17: reduce VIB betas by 10× during RL fine-tuning.
            _apply_vib_betas(0.1)
            self.offline_rl_optimize(
                dataset=current_dataset,
                num_epochs=config.training.ppo_epochs
            )

            # 2d) Collect New Data + Merge  (Algorithm 1, Lines 10-11)
            collection_metrics, new_episodes = self.collect_new_data(
                env_runner=env_runner,
                num_episodes=config.training.collection_episodes,
                use_ema=False,
                collect_trajectory=False,
            )
            self.offline_collection_success_history.append(
                float(collection_metrics.get('mean_success_rates', 0.0))
            )
            offline_collection_plot_path = self._save_iteration_metric_plot(
                metric_history=self.offline_collection_success_history,
                title="Offline Collection Success Rate",
                ylabel="Success Rate",
                filename="offline_collection_success_rate.png",
            )
            if config.logging.use_wandb and offline_collection_plot_path is not None:
                wandb.log({
                    'collection/offline_success_curve': wandb.Image(offline_collection_plot_path),
                }, step=self.global_step)
            il_episode_mask_new = self._build_il_episode_mask(new_episodes, stage='offline collection')

            # Algorithm 1 Line 11: D_{m+1} = D_m ∪ D_new
            if new_episodes:
                n_new = current_dataset.merge_episodes(
                    new_episodes,
                    il_episode_mask_new=il_episode_mask_new,
                )
                cprint(f"[Dataset] Merged {len(new_episodes)} episodes "
                       f"({n_new} steps) → total {current_dataset.replay_buffer.n_steps} steps, "
                       f"{current_dataset.replay_buffer.n_episodes} episodes", "cyan")

            # 2e) Retrain IL (optional) — Algorithm 1 Line 13
            if config.training.retrain_il_after_collection:
                # Paper Eq.17: restore VIB betas to original values for IL re-training.
                # The 10× reduction only applies to RL fine-tuning (OfflineRL step).
                _apply_vib_betas(1.0)
                cprint(f"\n[RL100Trainer] Retraining IL on merged dataset...", "cyan")
                il_dataset = current_dataset.get_il_training_dataset() \
                    if hasattr(current_dataset, 'get_il_training_dataset') else current_dataset
                self.train_imitation_learning(
                    dataset=il_dataset,
                    num_epochs=config.training.il_retrain_epochs,
                    env_runner=env_runner,
                    plot_tag=f"retrain_iter_{int(iteration):02d}"
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

            # Switch to a clean RL optimizer state before online fine-tuning.
            self._prepare_policy_optimizer_for_rl()

            for online_iter in range(config.training.online_rl_iterations):
                cprint(f"\n[Online RL] Iteration {online_iter + 1}/{config.training.online_rl_iterations}", "green")

                # 1. Collect fresh data
                collection_metrics, online_episodes = self.collect_new_data(
                    env_runner=env_runner,
                    num_episodes=config.training.online_collection_episodes,
                    policy_mode='ddim',
                    use_ema=False,
                    collect_trajectory=True,
                )
                self.online_collection_success_history.append(
                    float(collection_metrics.get('mean_success_rates', 0.0))
                )
                online_collection_plot_path = self._save_iteration_metric_plot(
                    metric_history=self.online_collection_success_history,
                    title="Online Collection Success Rate",
                    ylabel="Success Rate",
                    filename="online_collection_success_rate.png",
                )
                if config.logging.use_wandb and online_collection_plot_path is not None:
                    wandb.log({
                        'collection/online_success_curve': wandb.Image(online_collection_plot_path),
                    }, step=self.global_step)
                online_il_episode_mask = self._build_il_episode_mask(online_episodes, stage='online collection')

                if not online_episodes:
                    cprint("[Online RL] No episodes collected, skipping.", "yellow")
                    continue

                # Keep the raw online episodes for bookkeeping / future analysis,
                # but optimize the policy with the on-policy GAE objective.
                current_dataset.merge_episodes(
                    online_episodes,
                    il_episode_mask_new=online_il_episode_mask,
                )

                # 2. Online PPO + GAE optimization (paper Eq.21).
                _apply_vib_betas(0.1)
                self.online_rl_optimize(
                    online_episodes=online_episodes,
                    num_epochs=min(20, config.training.ppo_epochs),
                    online_iteration=online_iter,
                )
                _apply_vib_betas(1.0)

                self.save_checkpoint(tag=f'online_iter_{online_iter}')

        cprint("\n" + "="*80, "magenta")
        cprint(" "*25 + "TRAINING COMPLETE!", "magenta")
        cprint("="*80 + "\n", "magenta")

        # Save final checkpoint
        self.save_checkpoint(tag='final')

    def save_checkpoint(self, tag: str = 'latest'):
        """Save checkpoint."""
        save_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'policy': self.policy.state_dict(),
            'critics': self.critics.state_dict(),
            'consistency_model': self.consistency_model.state_dict(),
            'transition_model': self.transition_model.state_dict(),
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
            checkpoint['ema_helper'] = {
                'decay': self.ema.decay,
                'optimization_step': self.ema.optimization_step,
            }

        save_path = os.path.join(save_dir, f'{tag}.ckpt')
        torch.save(checkpoint, save_path)
        cprint(f"[Checkpoint] Saved to {save_path}", "green")

    def _load_matching_state_dict(self, module: nn.Module, state_dict: Dict[str, torch.Tensor], name: str):
        current_state = module.state_dict()
        filtered_state = {}
        skipped = []
        for key, value in state_dict.items():
            if key in current_state and current_state[key].shape == value.shape:
                filtered_state[key] = value
            else:
                skipped.append(key)

        missing, unexpected = module.load_state_dict(filtered_state, strict=False)
        if skipped:
            cprint(
                f"[Checkpoint] {name}: skipped {len(skipped)} incompatible key(s): "
                f"{skipped[:3]}{'...' if len(skipped) > 3 else ''}",
                "yellow",
            )
        if missing:
            cprint(
                f"[Checkpoint] {name}: {len(missing)} missing key(s): "
                f"{missing[:3]}{'...' if len(missing) > 3 else ''}",
                "yellow",
            )
        if unexpected:
            cprint(
                f"[Checkpoint] {name}: {len(unexpected)} unexpected key(s): "
                f"{unexpected[:3]}{'...' if len(unexpected) > 3 else ''}",
                "yellow",
            )

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Use strict=False to tolerate architecture mismatches (e.g., checkpoint saved
        # without VIB but current config has use_recon_vib=True). Missing keys (new VIB
        # layers) stay randomly initialized and will be trained from scratch; unexpected
        # keys (VIB layers in checkpoint but not in model) are silently ignored.
        missing, unexpected = self.policy.load_state_dict(checkpoint['policy'], strict=False)
        if missing:
            cprint(f"[Checkpoint] policy: {len(missing)} missing key(s) "
                   f"(new layers, will train from scratch): {missing[:3]}{'...' if len(missing)>3 else ''}", "yellow")
        if unexpected:
            cprint(f"[Checkpoint] policy: {len(unexpected)} unexpected key(s) "
                   f"(ignored): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}", "yellow")

        self._load_matching_state_dict(self.critics, checkpoint['critics'], 'critics')
        self._load_matching_state_dict(self.consistency_model, checkpoint['consistency_model'], 'consistency_model')
        if 'transition_model' in checkpoint:
            try:
                self.transition_model.load_state_dict(checkpoint['transition_model'])
            except Exception as e:
                cprint(f"[Checkpoint] transition_model full restore failed ({type(e).__name__}: {e}); trying partial model restore.", "yellow")
                if isinstance(checkpoint['transition_model'], dict) and 'model' in checkpoint['transition_model']:
                    self._load_matching_state_dict(
                        self.transition_model.model,
                        checkpoint['transition_model']['model'],
                        'transition_model.model'
                    )
        try:
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        except Exception as e:
            cprint(f"[Checkpoint] policy_optimizer not restored ({type(e).__name__}: {e})", "yellow")
        try:
            self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
        except Exception as e:
            cprint(f"[Checkpoint] v_optimizer not restored ({type(e).__name__}: {e})", "yellow")
        try:
            self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        except Exception as e:
            cprint(f"[Checkpoint] q_optimizer not restored ({type(e).__name__}: {e})", "yellow")
        try:
            self.consistency_optimizer.load_state_dict(checkpoint['consistency_optimizer'])
        except Exception as e:
            cprint(f"[Checkpoint] consistency_optimizer not restored ({type(e).__name__}: {e})", "yellow")

        if 'ema_policy' in checkpoint and self.ema_policy is not None:
            self.ema_policy.load_state_dict(checkpoint['ema_policy'], strict=False)
        if 'ema_helper' in checkpoint and self.ema_policy is not None and hasattr(self, 'ema'):
            self.ema.decay = checkpoint['ema_helper'].get('decay', self.ema.decay)
            self.ema.optimization_step = checkpoint['ema_helper'].get('optimization_step', self.ema.optimization_step)

        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.offline_rl_iteration = checkpoint['offline_rl_iteration']

        cprint(f"[Checkpoint] Loaded from {path}", "green")
