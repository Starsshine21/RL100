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
from typing import Dict, Tuple, List
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

    def get_variance_at_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Compute variance σ_k^2 for timestep k.

        Uses DDIM variance schedule with optional clipping.

        Args:
            timestep: [B] or scalar timestep

        Returns:
            variance: [B] variance values
        """
        # Get alpha values from scheduler
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(timestep.device)

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        alpha_t = alphas_cumprod[timestep]

        # For DDIM, variance at step t->t-1
        if timestep[0] > 0:
            alpha_t_prev = alphas_cumprod[timestep - 1]
        else:
            alpha_t_prev = torch.ones_like(alpha_t)

        # DDIM variance: σ_t^2 = (1 - α_{t-1}) / (1 - α_t) * (1 - α_t / α_{t-1})
        variance = (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)

        # Clip variance to avoid numerical issues and control exploration
        if self.use_variance_clip:
            variance = torch.clamp(variance, self.sigma_min ** 2, self.sigma_max ** 2)

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
        if variance.dim() == 1:
            variance = variance.view(-1, 1, 1)

        # Dimensionality
        d = x.shape[1] * x.shape[2]  # T * D

        # Compute log prob
        log_prob = -0.5 * (
            d * torch.log(2 * torch.pi * variance) +
            torch.sum((x - mean) ** 2, dim=[1, 2]) / variance.squeeze()
        )

        return log_prob

    def denoising_step_with_log_prob(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        global_cond: torch.Tensor,
        return_mean: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one denoising step and compute log probability.

        Args:
            x_t: [B, T, D] noisy action at timestep t
            t: timestep (scalar or [B])
            global_cond: [B, cond_dim] observation features
            return_mean: if True, also return predicted mean

        Returns:
            x_t_prev: [B, T, D] denoised action at timestep t-1
            log_prob: [B] log probability of transition
            mean: [B, T, D] predicted mean (if return_mean=True)
        """
        # Predict noise
        noise_pred = self.model(
            sample=x_t,
            timestep=t,
            global_cond=global_cond
        )

        # Compute mean using DDIM update rule
        # x_{t-1} = √(α_{t-1}) * (x_t - √(1-α_t) * ε_θ) / √(α_t) + √(1-α_{t-1}) * ε_θ

        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(x_t.device)

        if isinstance(t, int):
            t_tensor = torch.tensor([t], device=x_t.device).expand(x_t.shape[0])
        else:
            t_tensor = t

        alpha_t = alphas_cumprod[t_tensor].view(-1, 1, 1)

        if t_tensor[0] > 0:
            alpha_t_prev = alphas_cumprod[t_tensor - 1].view(-1, 1, 1)
        else:
            alpha_t_prev = torch.ones_like(alpha_t)

        # Predicted x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        # Mean of p(x_{t-1} | x_t)
        mean = (
            torch.sqrt(alpha_t_prev) * pred_x0 +
            torch.sqrt(1 - alpha_t_prev) * noise_pred
        )

        # Variance
        variance = self.get_variance_at_timestep(t_tensor)

        # Sample x_{t-1}
        if t_tensor[0] > 0:
            noise = torch.randn_like(x_t)
            x_t_prev = mean + torch.sqrt(variance).view(-1, 1, 1) * noise
        else:
            x_t_prev = mean

        # Compute log probability
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

        # Denoising loop
        for i, t in enumerate(scheduler.timesteps):
            # Apply conditioning
            x_t[condition_mask] = condition_data[condition_mask]

            # Denoising step with log prob
            x_t_prev, log_prob = self.denoising_step_with_log_prob(
                x_t=x_t,
                t=t,
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
        obs_dict: Dict[str, torch.Tensor],
        old_log_probs: List[torch.Tensor],
        advantages: torch.Tensor,
        trajectory: List[torch.Tensor]
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

        Returns:
            loss: scalar PPO loss
            info: dict with logging info
        """
        device = next(self.model.parameters()).device
        batch_size = advantages.shape[0]

        # Normalize observations
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        # Encode observations
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        # Compute new log probs for each denoising step
        new_log_probs = []
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)

        for i, t in enumerate(scheduler.timesteps):
            x_t = trajectory[i]
            x_t_prev = trajectory[i + 1]

            # Predict noise
            noise_pred = self.model(
                sample=x_t,
                timestep=t,
                global_cond=global_cond
            )

            # Compute mean
            alphas_cumprod = scheduler.alphas_cumprod.to(device)
            t_tensor = torch.tensor([t], device=device).expand(batch_size)

            alpha_t = alphas_cumprod[t_tensor].view(-1, 1, 1)
            if t > 0:
                alpha_t_prev = alphas_cumprod[t_tensor - 1].view(-1, 1, 1)
            else:
                alpha_t_prev = torch.ones_like(alpha_t)

            pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            mean = (
                torch.sqrt(alpha_t_prev) * pred_x0 +
                torch.sqrt(1 - alpha_t_prev) * noise_pred
            )

            # Variance
            variance = self.get_variance_at_timestep(t_tensor)

            # Log prob of actual transition
            log_prob = self.compute_gaussian_log_prob(x_t_prev, mean, variance)
            new_log_probs.append(log_prob)

        # Compute PPO loss for each step
        ppo_losses = []
        ratios = []

        for k in range(len(scheduler.timesteps)):
            # Probability ratio
            ratio = torch.exp(new_log_probs[k] - old_log_probs[k])
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

        # Sum over K steps
        total_ppo_loss = sum(ppo_losses)

        info = {
            'ppo_loss': total_ppo_loss.item(),
            'mean_ratio': sum(ratios) / len(ratios),
            'min_ratio': min(ratios),
            'max_ratio': max(ratios),
            'mean_advantage': advantages.mean().item(),
            'std_advantage': advantages.std().item(),
        }

        return total_ppo_loss, info

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

        # Normalize observations
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        # Encode observations
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

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
            obs_dict=obs_dict,
            old_log_probs=log_probs_old,
            advantages=advantages,
            trajectory=trajectory_old
        )

        return ppo_loss, ppo_info


if __name__ == "__main__":
    print("RL100Policy requires full DP3 environment to test.")
    print("Please use train_rl100.py for complete testing.")
