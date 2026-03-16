"""
RL-100 Policy Implementation
=============================
Extends DP3 with RL-100's K-step sub-MDP PPO loss for offline/online RL.

Key Innovation:
- Treats each denoising step as a sub-decision in a K-step MDP
- Computes PPO loss over ALL K denoising steps with shared advantage
- Enables policy gradient optimization for diffusion policies

Mathematical Formulation:
- Traditional: π_θ(a|s) - hard to compute for diffusion
- RL-100: π_θ(a|s) = ∏_{k=1}^{K} π_θ(a_{τ_{k-1}} | a_{τ_k}, s)
  where each step is a Gaussian: N(μ_θ(a_{τ_k}, s, k), σ_k^2)

PPO Loss:
  L_PPO = Σ_{k=1}^{K} min(r_k * A, clip(r_k, 1-ε, 1+ε) * A)
  where:
  - r_k = π_θ_new(a_{τ_{k-1}} | a_{τ_k}, s) / π_θ_old(a_{τ_{k-1}} | a_{τ_k}, s)
  - A = Q(s, a) - V(s) (shared advantage for all K steps)
  - ε = 0.2 (clip threshold)

References:
- RL-100 Paper: Algorithm 1
- DDIM: https://arxiv.org/abs/2010.02502
- PPO: https://arxiv.org/abs/1707.06347
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import copy
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.common.pytorch_util import dict_apply


class RL100Policy(DP3):
    """
    RL-100 Policy extending DP3 with PPO optimization over denoising steps.

    Inherits all DP3 functionality and adds:
    1. K-step trajectory tracking during denoising
    2. Gaussian log probability computation for each step
    3. PPO loss computation with shared advantage
    4. Variance clipping for stable exploration

    Args:
        (inherits all DP3 args)
        ppo_clip_eps: PPO clipping threshold (default: 0.2)
        sigma_min: Minimum std for exploration (default: 0.01)
        sigma_max: Maximum std for exploration (default: 0.8)
        use_variance_clip: Whether to clip predicted variance (default: True)
    """

    def __init__(
        self,
        # DP3 args
        shape_meta: dict,
        noise_scheduler,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = None,
        obs_as_global_cond: bool = True,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        condition_type: str = "film",
        use_down_condition: bool = True,
        use_mid_condition: bool = True,
        use_up_condition: bool = True,
        encoder_output_dim: int = 256,
        crop_shape=None,
        use_pc_color: bool = False,
        pointnet_type: str = "pointnet",
        pointcloud_encoder_cfg=None,
        use_recon_vib: bool = False,
        beta_recon: float = 1.0,
        beta_kl: float = 0.001,
        # RL-100 specific args
        ppo_clip_eps: float = 0.2,
        sigma_min: float = 0.01,
        sigma_max: float = 0.8,
        use_variance_clip: bool = True,
        **kwargs
    ):
        super().__init__(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_global_cond=obs_as_global_cond,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
            encoder_output_dim=encoder_output_dim,
            crop_shape=crop_shape,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_recon_vib=use_recon_vib,
            beta_recon=beta_recon,
            beta_kl=beta_kl,
            **kwargs
        )

        # RL-100 hyperparameters
        self.ppo_clip_eps = ppo_clip_eps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.use_variance_clip = use_variance_clip

    def encode_obs_global_cond(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode observations once into the conditioning tensor consumed by the
        diffusion UNet. This can be cached across PPO sub-steps.
        """
        batch_size = next(iter(obs_dict.values())).shape[0]
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        if "cross_attention" in self.condition_type:
            return nobs_features.reshape(batch_size, self.n_obs_steps, -1)
        return nobs_features.reshape(batch_size, -1)

    def compute_obs_regularization(self, obs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the Recon/VIB regularization term for RL fine-tuning.

        This mirrors the observation-encoder branch used in BC/IL loss, but
        returns only the weighted regularizer so it can be added to PPO.
        """
        device = next(self.model.parameters()).device
        zero = torch.zeros((), device=device)
        zero_info = {
            'kl_loss': 0.0,
            'recon_loss': 0.0,
            'recon_loss_pc': 0.0,
            'recon_loss_state': 0.0,
            'total_reg_loss': 0.0,
        }
        if not self.use_recon_vib:
            return zero, zero_info

        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        _, reg_loss_dict = self.obs_encoder(this_nobs, return_reg_loss=True)
        reg_loss = reg_loss_dict.get('total_reg_loss', zero)
        if isinstance(reg_loss, torch.Tensor) and reg_loss.dim() > 0:
            reg_loss = reg_loss.mean()

        info = {
            'kl_loss': float(reg_loss_dict.get('kl_loss', 0.0)),
            'recon_loss': float(reg_loss_dict.get('recon_loss', 0.0)),
            'recon_loss_pc': float(reg_loss_dict.get('recon_loss_pc', 0.0)),
            'recon_loss_state': float(reg_loss_dict.get('recon_loss_state', 0.0)),
            'total_reg_loss': float(reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss),
        }
        return reg_loss, info

    def get_variance_at_timestep(
        self,
        timestep: torch.Tensor,
        prev_timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute variance σ_k^2 for DDIM step t → t_prev.

        Uses the ACTUAL previous timestep from the scheduler's schedule,
        NOT t-1.  With num_train_timesteps=100 and num_inference_steps=10
        the schedule is [90,80,...,0], so for t=90 prev=80, not 89.

        Paper Eq.23: σ̃_k = clip(σ_k, σ_min, σ_max)

        Args:
            timestep: [B] current timestep in the DDIM schedule
            prev_timestep: [B] previous timestep in the DDIM schedule
                           (use -1 or 0 convention for the final step)

        Returns:
            variance: [B] variance values (σ²)
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(timestep.device)

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if prev_timestep.dim() == 0:
            prev_timestep = prev_timestep.unsqueeze(0)

        alpha_t = alphas_cumprod[timestep]

        # For the final step (prev_timestep <= 0), alpha_t_prev = 1.0
        alpha_t_prev = torch.where(
            prev_timestep >= 0,
            alphas_cumprod[prev_timestep.clamp(min=0)],
            torch.ones_like(alpha_t)
        )

        # DDIM variance: σ² = (1 - ᾱ_{t_prev}) / (1 - ᾱ_t) * (1 - ᾱ_t / ᾱ_{t_prev})
        variance = (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        final_step_mask = prev_timestep < 0

        # Paper Eq.23: clip variance to control exploration
        if self.use_variance_clip:
            clipped_variance = torch.clamp(variance, self.sigma_min ** 2, self.sigma_max ** 2)
            variance = torch.where(final_step_mask, torch.zeros_like(clipped_variance), clipped_variance)
        else:
            variance = torch.where(final_step_mask, torch.zeros_like(variance), variance)

        return variance

    def compute_gaussian_log_prob(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of x under Gaussian N(mean, variance).

        log π(x | mean, var) = -0.5 * [d*log(2π) + d*log(var) + ||x - mean||^2 / var]

        Args:
            x: [B, T, D] samples
            mean: [B, T, D] means
            variance: [B] or [B, 1, 1] variances

        Returns:
            log_prob: [B] log probabilities
        """
        # Keep variance as [B] for consistent broadcasting
        variance = variance.view(-1)  # [B]

        # Dimensionality
        d = x.shape[1] * x.shape[2]  # T * D

        # Compute log prob — all terms are [B]
        log_prob = -0.5 * (
            d * torch.log(2 * torch.pi * variance) +
            torch.sum((x - mean) ** 2, dim=[1, 2]) / variance
        )

        return log_prob

    def denoising_step_with_log_prob(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        global_cond: torch.Tensor,
        return_mean: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one DDIM denoising step t → t_prev and compute log probability.

        IMPORTANT: t_prev must be the ACTUAL previous timestep from the DDIM
        schedule, NOT t-1.  E.g. for schedule [90,80,...,0], when t=90,
        t_prev=80.

        Args:
            x_t: [B, T, D] noisy action at timestep t
            t: timestep (scalar or [B]) — current
            t_prev: timestep (scalar or [B]) — previous in DDIM schedule
            global_cond: [B, cond_dim] observation features
            return_mean: if True, also return predicted mean

        Returns:
            x_t_prev: [B, T, D] denoised action at timestep t_prev
            log_prob: [B] log probability of transition
            mean: [B, T, D] predicted mean (if return_mean=True)
        """
        # Predict noise
        noise_pred = self.model(
            sample=x_t,
            timestep=t,
            global_cond=global_cond
        )

        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(x_t.device)

        if isinstance(t, int):
            t_tensor = torch.tensor([t], device=x_t.device).expand(x_t.shape[0])
        else:
            t_tensor = t.to(x_t.device)
            if t_tensor.dim() == 0:
                t_tensor = t_tensor.expand(x_t.shape[0])
        if isinstance(t_prev, int):
            t_prev_tensor = torch.tensor([t_prev], device=x_t.device).expand(x_t.shape[0])
        else:
            t_prev_tensor = t_prev.to(x_t.device)
            if t_prev_tensor.dim() == 0:
                t_prev_tensor = t_prev_tensor.expand(x_t.shape[0])

        t_tensor = t_tensor.long()
        t_prev_tensor = t_prev_tensor.long()

        alpha_t = alphas_cumprod[t_tensor].view(-1, 1, 1)

        # Use ACTUAL previous timestep from schedule (not t-1)
        alpha_t_prev = torch.where(
            (t_prev_tensor >= 0).view(-1, 1, 1),
            alphas_cumprod[t_prev_tensor.clamp(min=0)].view(-1, 1, 1),
            torch.ones_like(alpha_t)
        )

        # Predicted x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        # Variance σ² (clipped per paper Eq.23)
        variance = self.get_variance_at_timestep(t_tensor, t_prev_tensor).to(x_t.device)

        # Mean: paper Eq.5a  μ = √ᾱ_{t_prev}·x̂₀ + √(1 - ᾱ_{t_prev} - σ²)·εθ
        noise_coeff = torch.sqrt(
            torch.clamp(1 - alpha_t_prev - variance.view(-1, 1, 1), min=0.0)
        )
        mean = torch.sqrt(alpha_t_prev) * pred_x0 + noise_coeff * noise_pred

        # Sample x_{t_prev} = μ + σ·ε  (no noise at final step)
        is_final_step = (t_prev_tensor < 0).view(-1, 1, 1)
        noise = torch.randn_like(x_t)
        x_t_prev = torch.where(
            is_final_step,
            mean,
            mean + torch.sqrt(variance).view(-1, 1, 1) * noise
        )

        # The final DDIM step is deterministic (σ² = 0), so it does not define a
        # valid Gaussian likelihood for PPO ratio computation.
        if torch.all(t_prev_tensor < 0):
            log_prob = torch.zeros(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
        else:
            log_prob = self.compute_gaussian_log_prob(x_t_prev, mean, variance)

        if return_mean:
            return x_t_prev, log_prob, mean
        else:
            return x_t_prev, log_prob

    def conditional_sample_with_trajectory(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        global_cond: torch.Tensor,
        return_trajectory: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        DDIM sampling that tracks full denoising trajectory.

        This is the core function for computing K-step log probabilities.

        Args:
            condition_data: [B, T, D] conditioning data
            condition_mask: [B, T, D] conditioning mask
            global_cond: [B, cond_dim] observation features
            return_trajectory: if True, return full trajectory and log probs

        Returns:
            final_action: [B, T, D] final denoised action
            trajectory: List of [B, T, D] actions at each step (length K+1)
            log_probs: List of [B] log probabilities at each step (length K)
        """
        scheduler = self.noise_scheduler

        # Initialize
        trajectory_list = []
        log_probs_list = []

        # Start from pure noise
        x_t = torch.randn_like(condition_data)
        trajectory_list.append(x_t.clone())

        # Set timesteps
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps  # e.g. [90, 80, 70, ..., 0]

        # Denoising loop
        for i, t in enumerate(timesteps):
            # Apply conditioning
            x_t[condition_mask] = condition_data[condition_mask]

            # Compute the ACTUAL previous timestep from the schedule
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
            else:
                # Final step: t_prev = -1 signals "no more noise"
                t_prev = torch.tensor(-1, device=x_t.device)

            # Denoising step with log prob
            x_t_prev, log_prob = self.denoising_step_with_log_prob(
                x_t=x_t,
                t=t,
                t_prev=t_prev,
                global_cond=global_cond,
                return_mean=False
            )

            # Store
            trajectory_list.append(x_t_prev.clone())
            log_probs_list.append(log_prob)

            # Update
            x_t = x_t_prev

        # Final conditioning
        x_t[condition_mask] = condition_data[condition_mask]

        if return_trajectory:
            return x_t, trajectory_list, log_probs_list
        else:
            return x_t

    def compute_ppo_loss(
        self,
        obs_dict: Optional[Dict[str, torch.Tensor]],
        old_log_probs: List[torch.Tensor],
        advantages: torch.Tensor,
        trajectory: List[torch.Tensor],
        global_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RL-100's K-step PPO loss.

        L_PPO = Σ_{k=1}^{K} E[ min(r_k * A, clip(r_k, 1-ε, 1+ε) * A) ]

        where r_k = π_θ_new(a_{k-1} | a_k, s) / π_θ_old(a_{k-1} | a_k, s)

        Args:
            obs_dict: Dictionary with observations
            old_log_probs: List of [B] old log probs for each of K steps
            advantages: [B, 1] advantage values A(s, a)
            trajectory: List of [B, T, D] actions at each timestep (length K+1)
            global_cond: Optional cached conditioning tensor

        Returns:
            loss: scalar PPO loss
            info: dict with logging info
        """
        device = next(self.model.parameters()).device
        batch_size = advantages.shape[0]
        if global_cond is None:
            if obs_dict is None:
                raise ValueError("compute_ppo_loss requires obs_dict when global_cond is not provided.")
            global_cond = self.encode_obs_global_cond(obs_dict)

        # Compute new log probs for each stochastic denoising step. The final
        # deterministic step is excluded from the PPO likelihood ratio.
        new_log_probs = []
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)
        timesteps = scheduler.timesteps  # e.g. [90, 80, 70, ..., 0]
        alphas_cumprod = scheduler.alphas_cumprod.to(device)

        for i, t in enumerate(timesteps):
            x_t = trajectory[i]
            x_t_prev = trajectory[i + 1]

            # Compute the ACTUAL previous timestep from the schedule
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
            else:
                t_prev = torch.tensor(-1, device=device)

            t_tensor = torch.tensor([t], device=device).expand(batch_size)
            t_prev_tensor = torch.tensor([t_prev], device=device).expand(batch_size)

            if torch.all(t_prev_tensor < 0):
                continue

            # Predict noise
            noise_pred = self.model(
                sample=x_t,
                timestep=t,
                global_cond=global_cond
            )

            alpha_t = alphas_cumprod[t_tensor].view(-1, 1, 1)
            alpha_t_prev = torch.where(
                (t_prev_tensor >= 0).view(-1, 1, 1),
                alphas_cumprod[t_prev_tensor.clamp(min=0)].view(-1, 1, 1),
                torch.ones_like(alpha_t)
            )

            pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # Variance σ² (clipped, paper Eq.23)
            variance = self.get_variance_at_timestep(t_tensor, t_prev_tensor)

            # Mean: paper Eq.5a  μ = √ᾱ_{t_prev}·x̂₀ + √(1 - ᾱ_{t_prev} - σ²)·εθ
            noise_coeff = torch.sqrt(
                torch.clamp(1 - alpha_t_prev - variance.view(-1, 1, 1), min=0.0)
            )
            mean = torch.sqrt(alpha_t_prev) * pred_x0 + noise_coeff * noise_pred

            # Log prob of actual transition
            log_prob = self.compute_gaussian_log_prob(x_t_prev, mean, variance)
            new_log_probs.append((i, log_prob))

        # Compute PPO loss for each step
        ppo_losses = []
        ratios = []

        for k, log_prob_new in new_log_probs:
            # Probability ratio
            ratio = torch.exp(log_prob_new - old_log_probs[k])
            ratios.append(ratio.mean().item())

            # Clipped ratio
            ratio_clipped = torch.clamp(
                ratio,
                1 - self.ppo_clip_eps,
                1 + self.ppo_clip_eps
            )

            # PPO objective (negative because we want to maximize)
            # Expand advantages to match batch size if needed
            adv = advantages.squeeze()
            surr1 = ratio * adv
            surr2 = ratio_clipped * adv
            ppo_loss_k = -torch.min(surr1, surr2).mean()

            ppo_losses.append(ppo_loss_k)

        # Average over K steps (not sum) — prevents gradient magnitude scaling with K
        total_ppo_loss = sum(ppo_losses) / len(ppo_losses)

        info = {
            'ppo_loss': total_ppo_loss.item(),
            'mean_ratio': sum(ratios) / len(ratios),
            'min_ratio': min(ratios),
            'max_ratio': max(ratios),
            'mean_advantage': advantages.mean().item(),
            'std_advantage': advantages.std().item(),
        }

        return total_ppo_loss, info

    def sample_for_ppo(
        self,
        obs_dict: Optional[Dict[str, torch.Tensor]] = None,
        global_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Sample denoising trajectory for PPO training.

        Call this ONCE before the inner PPO loop to get fixed old log probs.
        Then call compute_ppo_loss() multiple times with the same trajectory.

        Returns:
            trajectory: List of [B, T, D] tensors (K+1 entries)
            log_probs_old: List of [B] tensors (K entries)
        """
        device = next(self.model.parameters()).device
        if global_cond is None:
            if obs_dict is None:
                raise ValueError("sample_for_ppo requires obs_dict when global_cond is not provided.")
            global_cond = self.encode_obs_global_cond(obs_dict)
        batch_size = global_cond.shape[0]

        cond_data = torch.zeros(
            (batch_size, self.horizon, self.action_dim),
            device=device,
            dtype=global_cond.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        with torch.no_grad():
            _, trajectory, log_probs_old = self.conditional_sample_with_trajectory(
                condition_data=cond_data,
                condition_mask=cond_mask,
                global_cond=global_cond,
                return_trajectory=True
            )

        return trajectory, log_probs_old

    def sample_for_ppo_from_global_cond(
        self,
        global_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        device = global_cond.device
        batch_size = global_cond.shape[0]
        cond_data = torch.zeros((batch_size, self.horizon, self.action_dim), device=device, dtype=global_cond.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        with torch.no_grad():
            nsample, trajectory, log_probs_old = self.conditional_sample_with_trajectory(
                condition_data=cond_data,
                condition_mask=cond_mask,
                global_cond=global_cond,
                return_trajectory=True
            )

        return nsample, trajectory, log_probs_old

    def predict_action_with_trajectory(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        batch_size = next(iter(obs_dict.values())).shape[0]

        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        nsample, trajectory, log_probs_old = self.sample_for_ppo_from_global_cond(global_cond)
        naction_pred = nsample[..., :self.action_dim]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        action = self.extract_action_chunk(action_pred)

        return {
            'action': action,
            'action_pred': action_pred,
            'trajectory': trajectory,
            'log_probs_old': log_probs_old,
        }

    def compute_rl_loss(
        self,
        obs_dict: Dict[str, torch.Tensor],
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute RL loss by sampling trajectory and computing PPO loss.

        This is the main training function for offline/online RL.

        Args:
            obs_dict: Dictionary with observations
            advantages: [B, 1] advantage values from IQL critics

        Returns:
            loss: scalar RL loss
            info: dict with logging info
        """
        device = next(self.model.parameters()).device
        batch_size = advantages.shape[0]
        horizon = self.horizon
        action_dim = self.action_dim

        global_cond = self.encode_obs_global_cond(obs_dict)

        # Sample trajectory with old policy (for computing old log probs)
        with torch.no_grad():
            cond_data = torch.zeros((batch_size, horizon, action_dim), device=device)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            _, trajectory_old, log_probs_old = self.conditional_sample_with_trajectory(
                condition_data=cond_data,
                condition_mask=cond_mask,
                global_cond=global_cond,
                return_trajectory=True
            )

        # Compute PPO loss with new policy
        ppo_loss, ppo_info = self.compute_ppo_loss(
            obs_dict=None,
            old_log_probs=log_probs_old,
            advantages=advantages,
            trajectory=trajectory_old,
            global_cond=global_cond
        )

        return ppo_loss, ppo_info


if __name__ == "__main__":
    print("RL100Policy requires full DP3 environment to test.")
    print("Please use train_rl100.py for complete testing.")
