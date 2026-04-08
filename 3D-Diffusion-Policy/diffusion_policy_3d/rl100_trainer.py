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
        critics_cfg = config.critics
        default_hidden_dims = tuple(getattr(critics_cfg, 'hidden_dims', [256, 256, 256]))
        q_hidden_dims = tuple(getattr(critics_cfg, 'q_hidden_dims', default_hidden_dims))
        v_hidden_dims = tuple(getattr(critics_cfg, 'v_hidden_dims', default_hidden_dims))

        self.critics = IQLCritics(
            obs_dim=obs_dim,
            action_dim=critic_action_dim,
            q_hidden_dims=q_hidden_dims,
            v_hidden_dims=v_hidden_dims,
            gamma=critics_cfg.gamma,
            tau=critics_cfg.tau,
            use_layernorm=bool(getattr(critics_cfg, 'use_layernorm', True)),
            dropout=float(getattr(critics_cfg, 'dropout', 0.0)),
            activation=str(getattr(critics_cfg, 'activation', 'mish')),
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
            condition_type=config.policy.condition_type,
            max_timestep=int(config.policy.noise_scheduler.num_train_timesteps) - 1,
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
        transition_cfg = getattr(config, 'transition', None)
        transition_hidden_dims = tuple(getattr(transition_cfg, 'hidden_dims', (200, 200, 200, 200)))
        transition_weight_decays_cfg = getattr(
            transition_cfg,
            'weight_decays',
            (2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4),
        )
        transition_weight_decays = (
            None if transition_weight_decays_cfg is None
            else tuple(transition_weight_decays_cfg)
        )
        self.transition_model = TransitionModel(
            obs_feature_dim=obs_dim,    # same flattened dim used by critics
            action_dim=critic_action_dim,
            hidden_dims=transition_hidden_dims,
            num_ensemble=int(getattr(transition_cfg, 'num_ensemble', 7)),
            num_elites=int(getattr(transition_cfg, 'num_elites', 5)),
            lr=float(getattr(transition_cfg, 'lr', 1e-3)),
            weight_decays=transition_weight_decays,
            device=str(self.device),
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.offline_rl_iteration = 0
        self.offline_collection_success_history = []
        self.online_collection_success_history = []
        self.online_eval_score_history = []
        self.best_ddim_score = float('-inf')
        self.best_ddim_source_tag = None
        self.best_ddim_checkpoint_path = os.path.join(
            self.output_dir, 'checkpoints', 'best_ddim.ckpt')

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

    def _normalize_chunk_action(
        self,
        action_trajectory: torch.Tensor,
        executed_action_steps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        chunk_action = self._extract_chunk_action(action_trajectory)
        normalized_chunk = self.policy.normalizer['action'].normalize(chunk_action)
        return self.policy.mask_action_chunk(
            normalized_chunk,
            executed_action_steps=executed_action_steps,
        )

    def _flatten_normalized_chunk_action(
        self,
        action_trajectory: torch.Tensor,
        executed_action_steps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normalized_chunk = self._normalize_chunk_action(
            action_trajectory,
            executed_action_steps=executed_action_steps,
        )
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

    def _extract_eval_score(self, metrics: Optional[Dict]) -> Optional[float]:
        if not metrics:
            return None
        for key in ('test_mean_score', 'mean_success_rates'):
            value = metrics.get(key, None)
            if value is None:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(score):
                return score
        return None

    def _maybe_update_best_ddim_checkpoint(self, metrics: Optional[Dict], tag: str) -> bool:
        score = self._extract_eval_score(metrics)
        if score is None:
            return False
        if score <= self.best_ddim_score:
            return False

        prev_score = self.best_ddim_score
        self.best_ddim_score = score
        self.best_ddim_source_tag = str(tag)
        self.save_checkpoint(tag='best_ddim')
        self.best_ddim_checkpoint_path = os.path.join(
            self.output_dir, 'checkpoints', 'best_ddim.ckpt')

        if np.isfinite(prev_score):
            cprint(
                f"[BestDDIM] Updated best DDIM checkpoint from {prev_score:.4f} to {score:.4f} "
                f"at {self.best_ddim_source_tag}.",
                "green",
            )
        else:
            cprint(
                f"[BestDDIM] Recorded initial best DDIM checkpoint: {score:.4f} "
                f"at {self.best_ddim_source_tag}.",
                "green",
            )
        return True

    def _evaluate_current_ddim(
        self,
        env_runner: Optional[BaseRunner],
        use_ema: Optional[bool] = None,
    ) -> Optional[Dict]:
        if env_runner is None:
            return None
        eval_use_ema = (
            bool(getattr(self.config.runtime, 'final_eval_use_ema', False))
            if use_ema is None else bool(use_ema)
        )
        eval_policy = self.get_runtime_policy(mode='ddim', use_ema=eval_use_ema)
        eval_policy.eval()
        with torch.no_grad():
            metrics = env_runner.run(eval_policy)
        return metrics

    def _select_verified_final_ddim_checkpoint(
        self,
        env_runner: Optional[BaseRunner],
    ) -> None:
        """
        Verify that the checkpoint recorded as "best" still outperforms the
        current in-memory policy under the deterministic evaluation runner.

        This guards against stale bookkeeping or save/load inconsistencies:
        the final checkpoint should be selected by the score we can reproduce
        right now, not only by the score that was logged earlier.
        """
        if env_runner is None:
            return

        restore_best_ddim = bool(
            getattr(self.config.runtime, 'restore_best_ddim_before_final_eval', True)
        )
        if not restore_best_ddim:
            return
        if self.best_ddim_source_tag is None:
            return
        if not os.path.isfile(self.best_ddim_checkpoint_path):
            return

        last_checkpoint_path = os.path.join(self.output_dir, 'checkpoints', 'last.ckpt')

        self.save_checkpoint(tag='last')
        current_metrics = self._evaluate_current_ddim(env_runner)
        current_score = self._extract_eval_score(current_metrics)

        cprint(
            f"[BestDDIM] Verifying recorded best checkpoint from {self.best_ddim_source_tag} "
            f"(tracked score={self.best_ddim_score:.4f}).",
            "yellow",
        )
        self.load_checkpoint(self.best_ddim_checkpoint_path, load_rl_state=True)
        best_metrics = self._evaluate_current_ddim(env_runner)
        best_score = self._extract_eval_score(best_metrics)

        if current_score is not None:
            cprint(f"[BestDDIM] Current final-state DDIM score: {current_score:.4f}", "yellow")
        if best_score is not None:
            cprint(f"[BestDDIM] Restored best.ckpt DDIM score: {best_score:.4f}", "yellow")

        # Keep the restored checkpoint only when it actually reproduces a better
        # deterministic evaluation score than the current final state.
        if (
            current_score is not None
            and best_score is not None
            and best_score + 1e-8 < current_score
        ):
            cprint(
                f"[BestDDIM] Restored checkpoint underperformed the current final state "
                f"({best_score:.4f} < {current_score:.4f}); keeping last.ckpt instead.",
                "yellow",
            )
            self.load_checkpoint(last_checkpoint_path, load_rl_state=True)
            return

        if best_score is None and current_score is not None:
            cprint(
                "[BestDDIM] Restored checkpoint could not be re-scored; keeping last.ckpt instead.",
                "yellow",
            )
            self.load_checkpoint(last_checkpoint_path, load_rl_state=True)
            return

        cprint("[BestDDIM] Using restored best checkpoint for final save/eval.", "yellow")

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
        last_eval_metrics = None
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
                last_eval_metrics = metrics
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

        return {
            'final_loss': avg_loss,
            'eval_metrics': last_eval_metrics,
        }

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
            holdout_ratio=float(getattr(getattr(self.config, 'transition', None), 'holdout_ratio', 0.2)),
            logvar_loss_coef=float(getattr(getattr(self.config, 'transition', None), 'logvar_loss_coef', 0.01)),
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
        num_epochs: Optional[int] = None,
        num_steps: Optional[int] = None,
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
            num_epochs: Fallback number of training epochs
            num_steps: Fixed number of critic updates (preferred when set)

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

        def _critic_update(batch) -> Tuple[float, Optional[float]]:
            batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

            obs_dict = batch['obs']
            with torch.no_grad():
                obs_features = self._encode_obs_features(obs_dict)

            # The full executed action chunk is one MDP decision.
            naction = self._flatten_normalized_chunk_action(
                batch['action'],
                executed_action_steps=batch.get('executed_action_steps', None),
            )

            v_loss, v_info = self.critics.compute_v_loss(obs_features, naction)

            self.v_optimizer.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critics.v_network.parameters(), max_norm=10.0)
            self.v_optimizer.step()

            q_loss_item = None
            if 'reward' in batch and 'next_obs' in batch:
                reward = batch['reward']
                if reward.dim() == 1:
                    reward = reward.unsqueeze(-1)
                reward_scale = float(getattr(config.critics, 'reward_scale', 10.0))
                reward = reward / reward_scale
                done = batch.get('done', torch.zeros_like(reward))
                if done.dim() == 1:
                    done = done.unsqueeze(-1)

                with torch.no_grad():
                    next_obs_features = self._encode_obs_features(batch['next_obs'])

                q_loss, q_info = self.critics.compute_q_loss(
                    obs_features, naction, reward, next_obs_features, done
                )

                self.q_optimizer.zero_grad()
                q_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critics.q_network.parameters(), max_norm=10.0)
                self.q_optimizer.step()

                q_loss_item = float(q_loss.item())
                self.critics.update_target_network(tau=config.critics.target_update_tau)

                if self.global_step % config.training.log_every == 0 and config.logging.use_wandb:
                    wandb.log({
                        'iql/v_loss': float(v_loss.item()),
                        'iql/q_loss': q_loss_item,
                        **{f'iql/{k}': v for k, v in v_info.items()},
                        **{f'iql/{k}': v for k, v in q_info.items()}
                    }, step=self.global_step)

            self.global_step += 1
            return float(v_loss.item()), q_loss_item

        q_loss_per_epoch = []
        v_loss_per_epoch = []
        all_v_losses = []
        all_q_losses = []

        if num_steps is not None and int(num_steps) > 0:
            num_steps = int(num_steps)
            train_iterator = iter(train_dataloader)
            steps_per_bucket = max(1, len(train_dataloader))
            bucket_v_losses = []
            bucket_q_losses = []

            for step in range(num_steps):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_dataloader)
                    batch = next(train_iterator)

                v_loss_item, q_loss_item = _critic_update(batch)
                all_v_losses.append(v_loss_item)
                bucket_v_losses.append(v_loss_item)
                if q_loss_item is not None:
                    all_q_losses.append(q_loss_item)
                    bucket_q_losses.append(q_loss_item)

                should_flush = ((step + 1) % steps_per_bucket == 0) or (step + 1 == num_steps)
                if should_flush:
                    v_loss_avg = float(np.mean(bucket_v_losses)) if bucket_v_losses else 0.0
                    q_loss_avg = float(np.mean(bucket_q_losses)) if bucket_q_losses else 0.0
                    cprint(f"[IQL] Step {step + 1}/{num_steps}, "
                           f"V Loss: {v_loss_avg:.4f}, "
                           f"Q Loss: {q_loss_avg:.4f}", "green")
                    v_loss_per_epoch.append(v_loss_avg)
                    q_loss_per_epoch.append(q_loss_avg)
                    bucket_v_losses = []
                    bucket_q_losses = []
        else:
            num_epochs = int(num_epochs if num_epochs is not None else 0)
            for epoch in range(num_epochs):
                epoch_v_losses = []
                epoch_q_losses = []

                for batch in train_dataloader:
                    v_loss_item, q_loss_item = _critic_update(batch)
                    all_v_losses.append(v_loss_item)
                    epoch_v_losses.append(v_loss_item)
                    if q_loss_item is not None:
                        all_q_losses.append(q_loss_item)
                        epoch_q_losses.append(q_loss_item)

                v_loss_avg = float(np.mean(epoch_v_losses)) if epoch_v_losses else 0.0
                q_loss_avg = float(np.mean(epoch_q_losses)) if epoch_q_losses else 0.0
                cprint(f"[IQL] Epoch {epoch}/{num_epochs}, "
                       f"V Loss: {v_loss_avg:.4f}, "
                       f"Q Loss: {q_loss_avg:.4f}", "green")
                v_loss_per_epoch.append(v_loss_avg)
                q_loss_per_epoch.append(q_loss_avg)

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

        return {
            'v_loss': float(np.mean(all_v_losses)) if all_v_losses else 0.0,
            'q_loss': float(np.mean(all_q_losses)) if all_q_losses else 0.0,
        }

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
        """
        Prepare a fixed set of episode-start states for AM-Q evaluation.

        RL-100's AM-Q gate is intended to compare policies from the initial
        state distribution. Sampling arbitrary replay-buffer decision points
        dilutes early-horizon policy improvements, especially on sparse-reward
        tasks. When the dataset exposes episode boundaries, build the AM-Q
        batches from episode-start decision windows only; otherwise fall back to
        the generic replay-sample path.
        """
        if "cross_attention" in getattr(self.policy, 'condition_type', ''):
            raise NotImplementedError(
                "AM-Q feature-space rollout currently supports film-style conditioning only."
            )

        from torch.utils.data import DataLoader as _DL

        ope_cfg = getattr(self.config, 'ope', None)
        shuffle_batches = bool(getattr(ope_cfg, 'shuffle_batches', True))
        feature_batches = []
        batch_size = int(self.config.dataloader.batch_size)
        rng = np.random.default_rng(int(eval_seed)) if eval_seed is not None else np.random.default_rng()

        can_use_episode_starts = all(
            hasattr(dataset, attr)
            for attr in ('episode_starts', 'episode_ends', '_build_obs_window', 'replay_buffer')
        )

        if can_use_episode_starts:
            n_episodes = int(dataset.replay_buffer.n_episodes)
            episode_mask = np.asarray(
                getattr(dataset, 'train_mask', np.ones(n_episodes, dtype=bool)),
                dtype=bool,
            )
            if len(episode_mask) != n_episodes:
                episode_mask = np.ones(n_episodes, dtype=bool)
            candidate_episode_indices = np.flatnonzero(episode_mask)
            if candidate_episode_indices.size == 0:
                candidate_episode_indices = np.arange(n_episodes, dtype=np.int64)

            total_required = max(int(num_batches), 0) * batch_size
            if total_required > 0 and candidate_episode_indices.size > 0:
                episode_schedule = []
                while len(episode_schedule) < total_required:
                    episode_chunk = candidate_episode_indices.copy()
                    if shuffle_batches and episode_chunk.size > 1:
                        rng.shuffle(episode_chunk)
                    episode_schedule.extend(episode_chunk.tolist())
                episode_schedule = episode_schedule[:total_required]

                cprint(
                    f"[OPE] AM-Q eval uses episode-start states: "
                    f"{candidate_episode_indices.size} unique episodes, "
                    f"{num_batches} batch(es) x {batch_size}.",
                    "cyan",
                )

                with torch.no_grad():
                    for batch_idx in range(int(num_batches)):
                        start = batch_idx * batch_size
                        end = start + batch_size
                        batch_episode_indices = episode_schedule[start:end]
                        if not batch_episode_indices:
                            break

                        obs_point_cloud = []
                        obs_agent_pos = []
                        executed_action_steps = []

                        for episode_idx in batch_episode_indices:
                            episode_idx = int(episode_idx)
                            episode_start = int(dataset.episode_starts[episode_idx])
                            episode_end = int(dataset.episode_ends[episode_idx])

                            state_window, pc_window = dataset._build_obs_window(
                                decision_start_idx=episode_start,
                                episode_start=episode_start,
                                episode_end=episode_end,
                            )
                            obs_agent_pos.append(state_window.astype(np.float32, copy=False))
                            obs_point_cloud.append(pc_window.astype(np.float32, copy=False))

                            chunk_buffer_end = min(
                                episode_start + int(getattr(dataset, 'n_action_steps', self.policy.n_action_steps)),
                                episode_end,
                            )
                            executed_action_steps.append(
                                max(int(chunk_buffer_end - episode_start), 1)
                            )

                        obs_batch = {
                            'point_cloud': torch.from_numpy(
                                np.stack(obs_point_cloud, axis=0)
                            ).to(self.device, non_blocking=True),
                            'agent_pos': torch.from_numpy(
                                np.stack(obs_agent_pos, axis=0)
                            ).to(self.device, non_blocking=True),
                        }
                        obs_features = self._encode_obs_features(obs_batch)
                        feature_batches.append({
                            'obs_features': obs_features.detach().cpu(),
                            'executed_action_steps': torch.as_tensor(
                                executed_action_steps,
                                dtype=torch.long,
                            ),
                        })

                return feature_batches

        data_generator = None
        if eval_seed is not None:
            data_generator = torch.Generator()
            data_generator.manual_seed(int(eval_seed))

        loader = _DL(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_batches,
            num_workers=0,
            pin_memory=True,
            generator=data_generator,
        )

        cprint("[OPE] Dataset does not expose episode starts; falling back to replay-sampled AM-Q states.", "yellow")
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= num_batches:
                    break
                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                obs_features = self._encode_obs_features(batch['obs'])
                feature_batches.append({
                    'obs_features': obs_features.detach().cpu(),
                    'executed_action_steps': (
                        batch['executed_action_steps'].detach().cpu()
                        if 'executed_action_steps' in batch else None
                    ),
                })

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

        if eval_features is None:
            eval_features = self._prepare_amq_eval_feature_batches(
                dataset=dataset,
                num_batches=num_batches,
                eval_seed=eval_seed,
            )

        with torch.no_grad():
            for batch_idx, feature_batch in enumerate(eval_features):
                initial_executed_action_steps = None
                if isinstance(feature_batch, dict):
                    init_features = feature_batch['obs_features']
                    initial_executed_action_steps = feature_batch.get('executed_action_steps', None)
                else:
                    init_features = feature_batch
                obs_features = init_features.to(self.device, non_blocking=True)
                # H-step rollout
                batch_q_sum = torch.zeros(obs_features.shape[0], 1, device=self.device)
                cur_features = obs_features

                for h in range(rollout_horizon):
                    current_executed_action_steps = None
                    if h == 0 and initial_executed_action_steps is not None:
                        current_executed_action_steps = initial_executed_action_steps.to(
                            self.device, non_blocking=True)
                    noise_seed = None
                    if use_common_random_numbers and eval_seed is not None:
                        noise_seed = int(eval_seed + batch_idx * 1009 + h * 104729)

                    # RL-100 PPO optimizes the stochastic DDIM sub-policy
                    # induced by conditional_sample_with_trajectory(). Use the
                    # same policy semantics for AM-Q whenever available.
                    if hasattr(eval_policy, 'sample_for_ppo_from_global_cond') and hasattr(eval_policy, 'flatten_action_chunk'):
                        noise_generator = None
                        if noise_seed is not None:
                            noise_generator = torch.Generator()
                            noise_generator.manual_seed(int(noise_seed))
                        nsample, _, _ = eval_policy.sample_for_ppo_from_global_cond(
                            cur_features,
                            generator=noise_generator,
                            executed_action_steps=current_executed_action_steps,
                        )
                        naction = eval_policy.flatten_action_chunk(
                            nsample,
                            executed_action_steps=current_executed_action_steps,
                        )
                    else:
                        initial_noise = None
                        if noise_seed is not None:
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
                        )
                        if hasattr(eval_policy, 'mask_action_chunk'):
                            naction = eval_policy.mask_action_chunk(
                                naction,
                                executed_action_steps=current_executed_action_steps,
                            )
                        naction = naction.reshape(cur_features.shape[0], -1)

                    # Paper Eq.20 sums Q-values along the imagined rollout directly.
                    # Do not apply an extra outer discount here because Q already
                    # represents a discounted return estimate from (s_h, a_h).
                    q = self.critics.get_q_value(cur_features, naction)  # [B, 1]
                    batch_q_sum += q

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

    def _build_offline_ppo_buffer(
        self,
        dataset: BaseDataset,
        behavior_policy: RL100Policy,
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Build a fixed offline PPO buffer once per OfflineRL iteration.

        The dataset provides the state distribution, while the frozen
        behavior policy π_i provides the old denoising trajectory, old
        log-probabilities, and the action chunk used to compute IQL
        advantages. This mirrors standard PPO more closely: old data are
        frozen for the whole optimization window and reused across epochs.
        """
        config = self.config
        need_obs_cache = (
            float(getattr(config.training, 'lambda_cd', 1.0)) > 0.0
            or self._should_apply_obs_regularization()
        )

        cprint("[Offline RL] Building fixed PPO buffer from D_m under frozen behavior policy π_i...", "cyan")
        rollout_dataloader = DataLoader(
            dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=True,
            num_workers=config.dataloader.num_workers,
            pin_memory=True,
        )

        offline_buffer = []
        raw_advantages = []

        with torch.no_grad():
            for batch in rollout_dataloader:
                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                obs_dict = batch['obs']
                obs_features, global_cond = self._encode_obs_representations(obs_dict)
                executed_action_steps = batch.get('executed_action_steps', None)
                trajectory_old, log_probs_old = behavior_policy.sample_for_ppo(
                    global_cond=global_cond,
                    executed_action_steps=executed_action_steps,
                )
                behavior_chunk_action = behavior_policy.flatten_action_chunk(
                    trajectory_old[-1],
                    executed_action_steps=executed_action_steps,
                )
                advantages = self.critics.compute_advantage(
                    obs_features,
                    behavior_chunk_action,
                ).detach()

                buffer_item = {
                    'global_cond': global_cond.detach().cpu(),
                    'trajectory_old': [traj.detach().cpu() for traj in trajectory_old],
                    'log_probs_old': [log_prob.detach().cpu() for log_prob in log_probs_old],
                    'executed_action_steps': (
                        executed_action_steps.detach().cpu()
                        if executed_action_steps is not None else None
                    ),
                    'advantages_raw': advantages.detach().cpu(),
                }
                if need_obs_cache:
                    buffer_item['obs'] = dict_apply(obs_dict, lambda x: x.detach().cpu())

                offline_buffer.append(buffer_item)
                raw_advantages.append(buffer_item['advantages_raw'])

        if not offline_buffer:
            return [], {
                'num_batches': 0,
                'num_samples': 0,
                'adv_mean': 0.0,
                'adv_std': 1.0,
            }

        all_advantages = torch.cat(raw_advantages, dim=0)
        adv_mean_tensor = all_advantages.mean()
        adv_std_tensor = all_advantages.std().clamp_min(1e-8)

        for buffer_item in offline_buffer:
            buffer_item['advantages'] = (
                buffer_item.pop('advantages_raw') - adv_mean_tensor
            ) / adv_std_tensor

        total_samples = int(sum(item['advantages'].shape[0] for item in offline_buffer))
        adv_mean = float(adv_mean_tensor.item())
        adv_std = float(adv_std_tensor.item())
        cprint(
            f"[Offline RL] Fixed PPO buffer ready: {len(offline_buffer)} mini-batches, "
            f"{total_samples} samples, raw advantage mean={adv_mean:.4f}, std={adv_std:.4f}",
            "cyan",
        )
        return offline_buffer, {
            'num_batches': float(len(offline_buffer)),
            'num_samples': float(total_samples),
            'adv_mean': adv_mean,
            'adv_std': adv_std,
        }

    def offline_rl_optimize(
        self,
        dataset: BaseDataset,
        num_epochs: Optional[int] = None,
        num_steps: Optional[int] = None,
    ) -> Dict:
        """
        Phase 2b: Optimize policy with PPO and consistency distillation.

        Implements paper lines 7-8 of Algorithm 1 (OfflineRL):
          1. Save behavior-policy snapshot for OPE gate
          2. Freeze obs_encoder (paper: "keep ϕIL fixed")
          3. Build a fixed offline PPO buffer under π_i
          4. Normalize advantages (prevent gradient explosion)
          5. K-step PPO update + Consistency Distillation
          6. OPE gate: accept update only if AM-Q improves by ≥ δ (paper Eq.20)

        Args:
            dataset: Dataset with (s, a, r, s', done)
            num_epochs: Fallback number of training epochs
            num_steps: Fixed number of PPO updates (preferred when set)

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
        behavior_policy = copy.deepcopy(self.policy)
        behavior_policy.to(self.device)
        behavior_policy.eval()
        for param in behavior_policy.parameters():
            param.requires_grad_(False)

        # ── Step 2: Freeze encoder (paper: ϕIL is fixed during offline RL) ───────
        for param in self.policy.obs_encoder.parameters():
            param.requires_grad_(False)

        self.policy.train()
        self.critics.eval()
        offline_ppo_buffer, offline_buffer_info = self._build_offline_ppo_buffer(
            dataset=dataset,
            behavior_policy=behavior_policy,
        )
        if not offline_ppo_buffer:
            cprint("[Offline RL] Fixed PPO buffer is empty; skip policy optimization.", "yellow")
            for param in self.policy.obs_encoder.parameters():
                param.requires_grad_(True)
            return {
                'ppo_loss': 0.0,
                'approx_kl': 0.0,
                'clip_frac': 0.0,
                'mean_ratio': 1.0,
                'mean_abs_ratio_dev': 0.0,
                'reg_loss': 0.0,
                'cd_loss': 0.0,
            }
        if config.logging.use_wandb:
            wandb.log({
                'offline_buffer/num_batches': offline_buffer_info['num_batches'],
                'offline_buffer/num_samples': offline_buffer_info['num_samples'],
                'offline_buffer/adv_mean_raw': offline_buffer_info['adv_mean'],
                'offline_buffer/adv_std_raw': offline_buffer_info['adv_std'],
            }, step=self.global_step)

        # Number of inner PPO gradient steps per batch (allows ratio to deviate from 1)
        ppo_inner_steps = getattr(config.training, 'ppo_inner_steps', 4)
        # Paper Eq.22: L_total = L_RL + λ_CD · L_CD
        lambda_cd = getattr(config.training, 'lambda_cd', 1.0)
        offline_ppo_max_passes = int(getattr(config.training, 'offline_ppo_max_passes', 0))
        ppo_target_kl = getattr(config.training, 'ppo_target_kl', None)
        ppo_target_kl = float(ppo_target_kl) if ppo_target_kl is not None and float(ppo_target_kl) > 0 else None
        ppo_target_clip_frac = getattr(config.training, 'ppo_target_clip_frac', None)
        ppo_target_clip_frac = (
            float(ppo_target_clip_frac)
            if ppo_target_clip_frac is not None and float(ppo_target_clip_frac) > 0
            else None
        )
        ppo_early_stop_min_steps = int(getattr(config.training, 'ppo_early_stop_min_steps', 0))

        def _should_early_stop_ppo(metrics: Dict[str, float], completed_steps: int) -> bool:
            if completed_steps < ppo_early_stop_min_steps:
                return False
            if ppo_target_kl is not None and metrics['approx_kl'] >= ppo_target_kl:
                return True
            if ppo_target_clip_frac is not None and metrics['clip_frac'] >= ppo_target_clip_frac:
                return True
            return False

        def _format_early_stop_reason(metrics: Dict[str, float]) -> str:
            reasons = []
            if ppo_target_kl is not None and metrics['approx_kl'] >= ppo_target_kl:
                reasons.append(
                    f"approx_kl={metrics['approx_kl']:.3e} >= target_kl={ppo_target_kl:.3e}"
                )
            if ppo_target_clip_frac is not None and metrics['clip_frac'] >= ppo_target_clip_frac:
                reasons.append(
                    f"clip_frac={metrics['clip_frac']:.6f} >= target_clip_frac={ppo_target_clip_frac:.6f}"
                )
            return ", ".join(reasons) if reasons else "threshold reached"

        def _log_offline_step_summary(step_idx: int, total_steps: int, summary: Dict[str, float]) -> None:
            cprint(
                f"[Offline RL] Step {step_idx}/{total_steps}, PPO Loss: {summary['ppo_loss']:.4f}, "
                f"PostKL: {summary['approx_kl']:.3e}, "
                f"PostClipFrac: {summary['clip_frac']:.6f}, "
                f"PostMeanRatio: {summary['mean_ratio']:.6f}, "
                f"PostRatioDev: {summary['mean_abs_ratio_dev']:.3e}, "
                f"GradNorm: {summary['grad_norm']:.4f}, "
                f"Reg Loss: {summary['reg_loss']:.4f}, "
                f"CD Loss: {summary['cd_loss']:.4f}",
                "green",
            )

        def _offline_ppo_update(buffer_batch) -> Dict[str, float]:
            obs_dict = None
            if 'obs' in buffer_batch:
                obs_dict = dict_apply(
                    buffer_batch['obs'],
                    lambda x: x.to(self.device, non_blocking=True)
                )
            global_cond = buffer_batch['global_cond'].to(self.device, non_blocking=True)
            trajectory_old = [
                traj.to(self.device, non_blocking=True)
                for traj in buffer_batch['trajectory_old']
            ]
            log_probs_old = [
                log_prob.to(self.device, non_blocking=True)
                for log_prob in buffer_batch['log_probs_old']
            ]
            advantages = buffer_batch['advantages'].to(self.device, non_blocking=True)
            executed_action_steps = buffer_batch.get('executed_action_steps', None)
            if executed_action_steps is not None:
                executed_action_steps = executed_action_steps.to(self.device, non_blocking=True)
            run_cd_this_batch = (self.global_step % config.training.cd_every == 0)

            batch_ppo_losses = []
            batch_approx_kls = []
            batch_clip_fracs = []
            batch_mean_ratios = []
            batch_min_ratios = []
            batch_max_ratios = []
            batch_mean_abs_ratio_devs = []
            batch_reg_losses = []
            batch_cd_losses = []
            batch_grad_norms = []

            for ppo_inner in range(ppo_inner_steps):
                ppo_loss, _ = self.policy.compute_ppo_loss(
                    obs_dict=None,
                    old_log_probs=log_probs_old,
                    advantages=advantages,
                    trajectory=trajectory_old,
                    global_cond=global_cond,
                    executed_action_steps=executed_action_steps,
                )

                total_loss = ppo_loss
                if obs_dict is not None:
                    reg_loss, reg_info = self._compute_obs_regularization(obs_dict)
                    if reg_info['total_reg_loss'] > 0:
                        total_loss = total_loss + reg_loss
                        batch_reg_losses.append(float(reg_info['total_reg_loss']))
                run_cd_step = (
                    obs_dict is not None
                    and lambda_cd > 0.0
                    and run_cd_this_batch
                    and ppo_inner == 0
                )
                if run_cd_step:
                    cd_loss, cd_info = self.consistency_distillation.compute_distillation_loss(
                        obs_dict,
                        global_cond=global_cond,
                    )
                    total_loss = total_loss + lambda_cd * cd_loss
                    batch_cd_losses.append(float(cd_info['cd_loss']))

                self.policy_optimizer.zero_grad()
                self.consistency_optimizer.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    config.training.max_grad_norm
                )
                batch_grad_norms.append(float(grad_norm))
                self.policy_optimizer.step()
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
                        executed_action_steps=executed_action_steps,
                    )
                batch_ppo_losses.append(ppo_loss_display)
                batch_approx_kls.append(float(ppo_info_post.get('approx_kl', 0.0)))
                batch_clip_fracs.append(float(ppo_info_post.get('clip_frac', 0.0)))
                batch_mean_ratios.append(float(ppo_info_post.get('mean_ratio', 1.0)))
                batch_min_ratios.append(float(ppo_info_post.get('min_ratio', 1.0)))
                batch_max_ratios.append(float(ppo_info_post.get('max_ratio', 1.0)))
                batch_mean_abs_ratio_devs.append(float(ppo_info_post.get('mean_abs_ratio_dev', 0.0)))

            if self.ema_policy is not None:
                self.ema.step(self.policy)

            metrics = {
                'ppo_loss': float(np.mean(batch_ppo_losses)) if batch_ppo_losses else 0.0,
                'approx_kl': float(np.mean(batch_approx_kls)) if batch_approx_kls else 0.0,
                'clip_frac': float(np.mean(batch_clip_fracs)) if batch_clip_fracs else 0.0,
                'mean_ratio': float(np.mean(batch_mean_ratios)) if batch_mean_ratios else 1.0,
                'min_ratio': float(np.min(batch_min_ratios)) if batch_min_ratios else 1.0,
                'max_ratio': float(np.max(batch_max_ratios)) if batch_max_ratios else 1.0,
                'mean_abs_ratio_dev': float(np.mean(batch_mean_abs_ratio_devs)) if batch_mean_abs_ratio_devs else 0.0,
                'grad_norm': float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0,
                'reg_loss': float(np.mean(batch_reg_losses)) if batch_reg_losses else 0.0,
                'cd_loss': float(np.mean(batch_cd_losses)) if batch_cd_losses else 0.0,
            }
            if self.global_step % config.training.log_every == 0 and config.logging.use_wandb:
                log_dict = {
                    'ppo/loss': metrics['ppo_loss'],
                    'ppo/grad_norm': metrics['grad_norm'],
                    'ppo/approx_kl': metrics['approx_kl'],
                    'ppo/clip_frac': metrics['clip_frac'],
                    'ppo/mean_ratio': metrics['mean_ratio'],
                    'ppo/min_ratio': metrics['min_ratio'],
                    'ppo/max_ratio': metrics['max_ratio'],
                    'ppo/mean_abs_ratio_dev': metrics['mean_abs_ratio_dev'],
                }
                if metrics['reg_loss'] > 0:
                    log_dict['ppo/obs_reg_loss'] = metrics['reg_loss']
                if metrics['cd_loss'] > 0:
                    log_dict['cd/loss'] = metrics['cd_loss']
                wandb.log(log_dict, step=self.global_step)

            self.global_step += 1
            return metrics

        ppo_loss_per_epoch = []
        ppo_losses = []
        cd_losses = []
        reg_losses = []
        grad_norms = []
        approx_kls = []
        clip_fracs = []
        mean_ratios = []
        mean_abs_ratio_devs = []

        def _append_update_metrics(target: Dict[str, float]):
            ppo_losses.append(target['ppo_loss'])
            cd_losses.append(target['cd_loss'])
            reg_losses.append(target['reg_loss'])
            grad_norms.append(target['grad_norm'])
            approx_kls.append(target['approx_kl'])
            clip_fracs.append(target['clip_frac'])
            mean_ratios.append(target['mean_ratio'])
            mean_abs_ratio_devs.append(target['mean_abs_ratio_dev'])

        def _summarize_metrics(metric_buffer: List[Dict[str, float]]) -> Dict[str, float]:
            if not metric_buffer:
                return {
                    'ppo_loss': 0.0,
                    'approx_kl': 0.0,
                    'clip_frac': 0.0,
                    'mean_ratio': 1.0,
                    'mean_abs_ratio_dev': 0.0,
                    'grad_norm': 0.0,
                    'reg_loss': 0.0,
                    'cd_loss': 0.0,
                }
            return {
                'ppo_loss': float(np.mean([item['ppo_loss'] for item in metric_buffer])),
                'approx_kl': float(np.mean([item['approx_kl'] for item in metric_buffer])),
                'clip_frac': float(np.mean([item['clip_frac'] for item in metric_buffer])),
                'mean_ratio': float(np.mean([item['mean_ratio'] for item in metric_buffer])),
                'mean_abs_ratio_dev': float(np.mean([item['mean_abs_ratio_dev'] for item in metric_buffer])),
                'grad_norm': float(np.mean([item['grad_norm'] for item in metric_buffer])),
                'reg_loss': float(np.mean([item['reg_loss'] for item in metric_buffer])),
                'cd_loss': float(np.mean([item['cd_loss'] for item in metric_buffer])),
            }

        if num_steps is not None and int(num_steps) > 0:
            num_steps = int(num_steps)
            if offline_ppo_max_passes > 0:
                capped_num_steps = min(num_steps, offline_ppo_max_passes * max(1, len(offline_ppo_buffer)))
                if capped_num_steps < num_steps:
                    cprint(
                        f"[Offline RL] Cap PPO steps: requested {num_steps}, using {capped_num_steps} "
                        f"({offline_ppo_max_passes} passes over {len(offline_ppo_buffer)} fixed PPO batches).",
                        "yellow",
                    )
                num_steps = capped_num_steps
            steps_per_bucket = max(1, len(offline_ppo_buffer))
            bucket_metrics = []
            buffer_schedule = []
            while len(buffer_schedule) < num_steps:
                buffer_schedule.extend(np.random.permutation(len(offline_ppo_buffer)).tolist())
            buffer_schedule = buffer_schedule[:num_steps]

            # Traverse the fixed PPO buffer in shuffled passes instead of
            # sampling mini-batches with replacement. This keeps coverage
            # balanced and makes PPO/OPE behavior much less noisy.
            for step, buffer_idx in enumerate(buffer_schedule):
                update_metrics = _offline_ppo_update(offline_ppo_buffer[buffer_idx])
                _append_update_metrics(update_metrics)
                bucket_metrics.append(update_metrics)

                should_flush = ((step + 1) % steps_per_bucket == 0) or (step + 1 == num_steps)
                if should_flush:
                    summary = _summarize_metrics(bucket_metrics)
                    _log_offline_step_summary(step + 1, num_steps, summary)
                    ppo_loss_per_epoch.append(summary['ppo_loss'])
                    bucket_metrics = []
                if _should_early_stop_ppo(update_metrics, step + 1):
                    if bucket_metrics:
                        summary = _summarize_metrics(bucket_metrics)
                        _log_offline_step_summary(step + 1, num_steps, summary)
                        ppo_loss_per_epoch.append(summary['ppo_loss'])
                        bucket_metrics = []
                    cprint(
                        f"[Offline RL] Early stop at step {step + 1}/{num_steps}: "
                        f"{_format_early_stop_reason(update_metrics)}",
                        "yellow",
                    )
                    break
        else:
            num_epochs = int(num_epochs if num_epochs is not None else 0)
            for epoch in range(num_epochs):
                epoch_metrics = []
                buffer_order = np.random.permutation(len(offline_ppo_buffer))
                for buffer_idx in buffer_order:
                    update_metrics = _offline_ppo_update(offline_ppo_buffer[int(buffer_idx)])
                    _append_update_metrics(update_metrics)
                    epoch_metrics.append(update_metrics)

                summary = _summarize_metrics(epoch_metrics)
                cprint(
                    f"[Offline RL] Epoch {epoch}/{num_epochs}, PPO Loss: {summary['ppo_loss']:.4f}, "
                    f"PostKL: {summary['approx_kl']:.3e}, "
                    f"PostClipFrac: {summary['clip_frac']:.6f}, "
                    f"PostMeanRatio: {summary['mean_ratio']:.6f}, "
                    f"PostRatioDev: {summary['mean_abs_ratio_dev']:.3e}, "
                    f"GradNorm: {summary['grad_norm']:.4f}, "
                    f"Reg Loss: {summary['reg_loss']:.4f}, "
                    f"CD Loss: {summary['cd_loss']:.4f}",
                    "green",
                )
                ppo_loss_per_epoch.append(summary['ppo_loss'])

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

        il_success_only = getattr(self.config.runtime, 'il_retrain_success_only', None)
        if il_success_only is None:
            # Backward compatibility for older configs.
            il_success_only = getattr(self.config.runtime, 'merge_success_only', False)
        il_success_only = bool(il_success_only)

        if not il_success_only:
            cprint(
                f"[Dataset] {stage}: all {len(episodes)} episodes remain in RL replay; "
                f"IL retrain also uses all {len(episodes)} episodes.",
                "cyan",
            )
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
        num_epochs: Optional[int] = None,
        num_value_steps: Optional[int] = None,
        num_policy_steps: Optional[int] = None,
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
        ppo_target_kl = getattr(config.training, 'ppo_target_kl', None)
        ppo_target_kl = float(ppo_target_kl) if ppo_target_kl is not None and float(ppo_target_kl) > 0 else None
        ppo_target_clip_frac = getattr(config.training, 'ppo_target_clip_frac', None)
        ppo_target_clip_frac = (
            float(ppo_target_clip_frac)
            if ppo_target_clip_frac is not None and float(ppo_target_clip_frac) > 0
            else None
        )
        ppo_early_stop_min_steps = int(getattr(config.training, 'ppo_early_stop_min_steps', 0))

        def _should_early_stop_online(metrics: Dict[str, float], completed_steps: int) -> bool:
            if completed_steps < ppo_early_stop_min_steps:
                return False
            if ppo_target_kl is not None and metrics['approx_kl'] >= ppo_target_kl:
                return True
            if ppo_target_clip_frac is not None and metrics['clip_frac'] >= ppo_target_clip_frac:
                return True
            return False

        def _format_online_early_stop_reason(metrics: Dict[str, float]) -> str:
            reasons = []
            if ppo_target_kl is not None and metrics['approx_kl'] >= ppo_target_kl:
                reasons.append(
                    f"approx_kl={metrics['approx_kl']:.3e} >= target_kl={ppo_target_kl:.3e}"
                )
            if ppo_target_clip_frac is not None and metrics['clip_frac'] >= ppo_target_clip_frac:
                reasons.append(
                    f"clip_frac={metrics['clip_frac']:.6f} >= target_clip_frac={ppo_target_clip_frac:.6f}"
                )
            return ", ".join(reasons) if reasons else "threshold reached"

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
                        'executed_action_steps': int(step.get('executed_action_steps', self.policy.n_action_steps)),
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
        value_steps_per_bucket = max(1, (len(rollout_buffer) + batch_size - 1) // batch_size)

        def _sample_rollout_indices() -> np.ndarray:
            replace = len(rollout_buffer) < batch_size
            return np.random.choice(len(rollout_buffer), size=batch_size, replace=replace)

        def _online_value_update(idx: np.ndarray) -> float:
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
            return float(v_loss.item())

        if num_value_steps is not None and int(num_value_steps) > 0:
            num_value_steps = int(num_value_steps)
            bucket_v_losses = []
            for step in range(num_value_steps):
                v_loss_item = _online_value_update(_sample_rollout_indices())
                v_losses.append(v_loss_item)
                bucket_v_losses.append(v_loss_item)
                should_flush = ((step + 1) % value_steps_per_bucket == 0) or (step + 1 == num_value_steps)
                if should_flush:
                    bucket_mean = float(np.mean(bucket_v_losses)) if bucket_v_losses else 0.0
                    cprint(f"[Online RL] Value Step {step + 1}/{num_value_steps}, "
                           f"V Loss: {bucket_mean:.4f}", "green")
                    v_loss_per_epoch.append(bucket_mean)
                    bucket_v_losses = []
        else:
            critic_indices = np.arange(len(rollout_buffer))
            for critic_epoch in range(5):
                np.random.shuffle(critic_indices)
                epoch_v_losses = []
                for start in range(0, len(critic_indices), batch_size):
                    idx = critic_indices[start:start + batch_size]
                    v_loss_item = _online_value_update(idx)
                    v_losses.append(v_loss_item)
                    epoch_v_losses.append(v_loss_item)
                if epoch_v_losses:
                    epoch_mean = float(np.mean(epoch_v_losses))
                    cprint(f"[Online RL] Value Epoch {critic_epoch + 1}/5, "
                           f"V Loss: {epoch_mean:.4f}", "green")
                    v_loss_per_epoch.append(epoch_mean)

        # On-policy PPO over the stored rollout buffer.
        self.policy.train()
        self.critics.eval()
        ppo_losses = []
        cd_losses = []
        reg_losses = []
        grad_norms = []
        approx_kls = []
        clip_fracs = []
        policy_steps_per_bucket = max(1, (len(rollout_buffer) + batch_size - 1) // batch_size)

        def _online_ppo_update(idx: np.ndarray) -> Dict[str, float]:
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
                for k in range(len(rollout_buffer[int(idx[0])]['trajectory']))
            ]
            old_log_probs = [
                torch.from_numpy(
                    np.stack([rollout_buffer[i]['log_probs_old'][k] for i in idx], axis=0)
                ).to(self.device).view(-1)
                for k in range(len(rollout_buffer[int(idx[0])]['log_probs_old']))
            ]
            advantages = torch.as_tensor(
                [rollout_buffer[i]['advantage'] for i in idx],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(-1)
            executed_action_steps = torch.as_tensor(
                [rollout_buffer[i]['executed_action_steps'] for i in idx],
                dtype=torch.long,
                device=self.device,
            )
            _, global_cond = self._encode_obs_representations(obs_dict)

            ppo_loss, _ = self.policy.compute_ppo_loss(
                obs_dict=None,
                old_log_probs=old_log_probs,
                advantages=advantages,
                trajectory=trajectories,
                global_cond=global_cond,
                executed_action_steps=executed_action_steps,
            )
            ppo_loss_display = float(-ppo_loss.item())

            total_loss = ppo_loss
            reg_loss, reg_info = self._compute_obs_regularization(obs_dict)
            reg_loss_value = 0.0
            if reg_info['total_reg_loss'] > 0:
                total_loss = total_loss + reg_loss
                reg_loss_value = float(reg_info['total_reg_loss'])
            cd_loss_value = 0.0
            run_cd_step = (
                lambda_cd > 0.0
                and (self.global_step % config.training.cd_every == 0)
            )
            if run_cd_step:
                cd_loss, cd_info = self.consistency_distillation.compute_distillation_loss(
                    obs_dict,
                    global_cond=global_cond,
                )
                total_loss = total_loss + lambda_cd * cd_loss
                cd_loss_value = float(cd_info['cd_loss'])

            self.policy_optimizer.zero_grad()
            self.consistency_optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                config.training.max_grad_norm
            )
            grad_norm_value = float(grad_norm)
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
                    executed_action_steps=executed_action_steps,
                )

            metrics = {
                'ppo_loss': ppo_loss_display,
                'grad_norm': grad_norm_value,
                'approx_kl': float(ppo_info_post.get('approx_kl', 0.0)),
                'clip_frac': float(ppo_info_post.get('clip_frac', 0.0)),
                'mean_ratio': float(ppo_info_post.get('mean_ratio', 1.0)),
                'min_ratio': float(ppo_info_post.get('min_ratio', 1.0)),
                'max_ratio': float(ppo_info_post.get('max_ratio', 1.0)),
                'mean_abs_ratio_dev': float(ppo_info_post.get('mean_abs_ratio_dev', 0.0)),
                'reg_loss': reg_loss_value,
                'cd_loss': cd_loss_value,
            }
            if config.logging.use_wandb and self.global_step % config.training.log_every == 0:
                log_dict = {
                    'online/ppo_loss': metrics['ppo_loss'],
                    'online/grad_norm': metrics['grad_norm'],
                    'online/approx_kl': metrics['approx_kl'],
                    'online/clip_frac': metrics['clip_frac'],
                    'online/mean_ratio': metrics['mean_ratio'],
                    'online/min_ratio': metrics['min_ratio'],
                    'online/max_ratio': metrics['max_ratio'],
                    'online/mean_abs_ratio_dev': metrics['mean_abs_ratio_dev'],
                }
                if metrics['reg_loss'] > 0:
                    log_dict['online/obs_reg_loss'] = metrics['reg_loss']
                if metrics['cd_loss'] > 0:
                    log_dict['online/cd_loss'] = metrics['cd_loss']
                wandb.log(log_dict, step=self.global_step)

            self.global_step += 1
            return metrics

        def _summarize_online_metrics(metric_buffer: List[Dict[str, float]]) -> Dict[str, float]:
            if not metric_buffer:
                return {
                    'ppo_loss': 0.0,
                    'grad_norm': 0.0,
                    'approx_kl': 0.0,
                    'clip_frac': 0.0,
                    'mean_ratio': 1.0,
                    'mean_abs_ratio_dev': 0.0,
                    'reg_loss': 0.0,
                    'cd_loss': 0.0,
                }
            return {
                'ppo_loss': float(np.mean([item['ppo_loss'] for item in metric_buffer])),
                'grad_norm': float(np.mean([item['grad_norm'] for item in metric_buffer])),
                'approx_kl': float(np.mean([item['approx_kl'] for item in metric_buffer])),
                'clip_frac': float(np.mean([item['clip_frac'] for item in metric_buffer])),
                'mean_ratio': float(np.mean([item['mean_ratio'] for item in metric_buffer])),
                'mean_abs_ratio_dev': float(np.mean([item['mean_abs_ratio_dev'] for item in metric_buffer])),
                'reg_loss': float(np.mean([item['reg_loss'] for item in metric_buffer])),
                'cd_loss': float(np.mean([item['cd_loss'] for item in metric_buffer])),
            }

        def _log_online_step_summary(step_idx: int, total_steps: int, summary: Dict[str, float]) -> None:
            cprint(
                f"[Online RL] PPO Step {step_idx}/{total_steps}, "
                f"PPO Loss: {summary['ppo_loss']:.4f}, "
                f"PostKL: {summary['approx_kl']:.3e}, "
                f"PostClipFrac: {summary['clip_frac']:.6f}, "
                f"PostMeanRatio: {summary['mean_ratio']:.6f}, "
                f"PostRatioDev: {summary['mean_abs_ratio_dev']:.3e}, "
                f"GradNorm: {summary['grad_norm']:.4f}, "
                f"Reg Loss: {summary['reg_loss']:.4f}, "
                f"CD Loss: {summary['cd_loss']:.4f}",
                "green",
            )

        if num_policy_steps is not None and int(num_policy_steps) > 0:
            num_policy_steps = int(num_policy_steps)
            bucket_metrics = []
            for step in range(num_policy_steps):
                update_metrics = _online_ppo_update(_sample_rollout_indices())
                bucket_metrics.append(update_metrics)
                ppo_losses.append(update_metrics['ppo_loss'])
                cd_losses.append(update_metrics['cd_loss'])
                reg_losses.append(update_metrics['reg_loss'])
                grad_norms.append(update_metrics['grad_norm'])
                approx_kls.append(update_metrics['approx_kl'])
                clip_fracs.append(update_metrics['clip_frac'])

                should_flush = ((step + 1) % policy_steps_per_bucket == 0) or (step + 1 == num_policy_steps)
                if should_flush:
                    summary = _summarize_online_metrics(bucket_metrics)
                    _log_online_step_summary(step + 1, num_policy_steps, summary)
                    bucket_metrics = []
                if _should_early_stop_online(update_metrics, step + 1):
                    if bucket_metrics:
                        summary = _summarize_online_metrics(bucket_metrics)
                        _log_online_step_summary(step + 1, num_policy_steps, summary)
                        bucket_metrics = []
                    cprint(
                        f"[Online RL] Early stop at PPO step {step + 1}/{num_policy_steps}: "
                        f"{_format_online_early_stop_reason(update_metrics)}",
                        "yellow",
                    )
                    break
        else:
            num_epochs = int(num_epochs if num_epochs is not None else 0)
            policy_indices = np.arange(len(rollout_buffer))
            for epoch in range(num_epochs):
                np.random.shuffle(policy_indices)
                epoch_metrics = []
                for start in range(0, len(policy_indices), batch_size):
                    idx = policy_indices[start:start + batch_size]
                    update_metrics = _online_ppo_update(idx)
                    epoch_metrics.append(update_metrics)
                    ppo_losses.append(update_metrics['ppo_loss'])
                    cd_losses.append(update_metrics['cd_loss'])
                    reg_losses.append(update_metrics['reg_loss'])
                    grad_norms.append(update_metrics['grad_norm'])
                    approx_kls.append(update_metrics['approx_kl'])
                    clip_fracs.append(update_metrics['clip_frac'])

                summary = _summarize_online_metrics(epoch_metrics)
                cprint(
                    f"[Online RL] Epoch {epoch + 1}/{num_epochs}, "
                    f"PPO Loss: {summary['ppo_loss']:.4f}, "
                    f"PostKL: {summary['approx_kl']:.3e}, "
                    f"PostClipFrac: {summary['clip_frac']:.6f}, "
                    f"PostMeanRatio: {summary['mean_ratio']:.6f}, "
                    f"PostRatioDev: {summary['mean_abs_ratio_dev']:.3e}, "
                    f"GradNorm: {summary['grad_norm']:.4f}, "
                    f"Reg Loss: {summary['reg_loss']:.4f}, "
                    f"CD Loss: {summary['cd_loss']:.4f}",
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
            if env_runner is not None:
                cprint("[RL100Trainer] Evaluating resumed DDIM policy to establish best-checkpoint baseline.", "cyan")
                resumed_metrics = self._evaluate_current_ddim(env_runner)
                self._maybe_update_best_ddim_checkpoint(
                    resumed_metrics,
                    tag='resume_loaded_policy'
                )
        else:
            il_metrics = self.train_imitation_learning(
                dataset=initial_dataset,
                num_epochs=config.training.il_epochs,
                env_runner=env_runner,
                plot_tag='initial'
            )
            self._maybe_update_best_ddim_checkpoint(
                il_metrics.get('eval_metrics'),
                tag='after_il'
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
            transition_cfg = getattr(config, 'transition', None)
            self.train_transition_model(
                dataset=current_dataset,
                max_epochs=int(getattr(transition_cfg, 'max_epochs',
                                       getattr(config.training, 'transition_max_epochs', 200))),
                max_epochs_since_update=int(getattr(
                    transition_cfg,
                    'max_epochs_since_update',
                    getattr(config.training, 'transition_patience', 5))),
            )

            # 2b) Train IQL Critics  (Algorithm 1, Line 5)
            self.train_iql_critics(
                dataset=current_dataset,
                num_epochs=config.training.critic_epochs,
                num_steps=getattr(config.training, 'offline_critic_steps', None),
            )

            # 2c) Optimize Policy  (Algorithm 1, Lines 7-8)
            # Paper Eq.17: reduce VIB betas by 10× during RL fine-tuning.
            _apply_vib_betas(0.1)
            self.offline_rl_optimize(
                dataset=current_dataset,
                num_epochs=config.training.ppo_epochs,
                num_steps=getattr(config.training, 'offline_ppo_steps', None),
            )

            # 2d) Collect New Data + Merge  (Algorithm 1, Lines 10-11)
            collection_metrics, new_episodes = self.collect_new_data(
                env_runner=env_runner,
                num_episodes=config.training.collection_episodes,
                use_ema=False,
                collect_trajectory=True,
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
                il_metrics = self.train_imitation_learning(
                    dataset=il_dataset,
                    num_epochs=config.training.il_retrain_epochs,
                    env_runner=env_runner,
                    plot_tag=f"retrain_iter_{int(iteration):02d}"
                )
                self._maybe_update_best_ddim_checkpoint(
                    il_metrics.get('eval_metrics'),
                    tag=f'offline_iter_{iteration}'
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
                    num_value_steps=getattr(config.training, 'online_value_steps', None),
                    num_policy_steps=getattr(config.training, 'online_ppo_steps', None),
                    online_iteration=online_iter,
                )
                _apply_vib_betas(1.0)

                eval_metrics = self._evaluate_current_ddim(env_runner)
                eval_score = self._extract_eval_score(eval_metrics)
                if eval_score is not None:
                    self.online_eval_score_history.append(eval_score)
                    online_eval_plot_path = self._save_iteration_metric_plot(
                        metric_history=self.online_eval_score_history,
                        title="Online Eval DDIM Score",
                        ylabel="Success Rate",
                        filename="online_eval_ddim_score.png",
                    )
                    if config.logging.use_wandb:
                        log_dict = {'eval/online_ddim_score': eval_score}
                        if online_eval_plot_path is not None:
                            log_dict['eval/online_ddim_curve'] = wandb.Image(online_eval_plot_path)
                        wandb.log(log_dict, step=self.global_step)
                    self._maybe_update_best_ddim_checkpoint(
                        eval_metrics,
                        tag=f'online_iter_{online_iter}'
                    )

                self.save_checkpoint(tag=f'online_iter_{online_iter}')

        cprint("\n" + "="*80, "magenta")
        cprint(" "*25 + "TRAINING COMPLETE!", "magenta")
        cprint("="*80 + "\n", "magenta")

        self._select_verified_final_ddim_checkpoint(env_runner)

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

    def load_checkpoint(self, path: str, load_rl_state: bool = True):
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

        if 'ema_policy' in checkpoint and self.ema_policy is not None:
            self.ema_policy.load_state_dict(checkpoint['ema_policy'], strict=False)
        if 'ema_helper' in checkpoint and self.ema_policy is not None and hasattr(self, 'ema'):
            self.ema.decay = checkpoint['ema_helper'].get('decay', self.ema.decay)
            self.ema.optimization_step = checkpoint['ema_helper'].get('optimization_step', self.ema.optimization_step)

        if not load_rl_state:
            self.global_step = int(checkpoint.get('global_step', 0))
            self.epoch = 0
            self.offline_rl_iteration = 0
            cprint(
                "[Checkpoint] Loaded policy/EMA only; RL heads and optimizers keep fresh initialization.",
                "green",
            )
            return

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

        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.offline_rl_iteration = checkpoint['offline_rl_iteration']

        cprint(f"[Checkpoint] Loaded from {path}", "green")
