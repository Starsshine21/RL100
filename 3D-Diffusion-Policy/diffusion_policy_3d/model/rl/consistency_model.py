"""
Consistency Distillation for RL-100
====================================
Implements consistency distillation to learn a 1-step generation model
from the multi-step DDIM teacher policy.

Key Idea:
- Teacher: K-step DDIM denoising
- Student: 1-step direct prediction
- Loss: L2 distance between student output and teacher output

Benefits:
- 10x faster inference (1 step vs 10 steps)
- Maintains quality of K-step generation
- Can be fine-tuned online

References:
- Consistency Models: https://arxiv.org/abs/2303.01469
- RL-100: Applies CD to diffusion policies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D


class ConsistencyModel(nn.Module):
    """
    Consistency Model for fast 1-step generation.

    Architecture:
    - Uses same UNet architecture as teacher (DP3)
    - Predicts clean action directly from noisy action
    - Conditioned on observations like the teacher

    Training:
    - Distillation loss: L2(student(x_T), teacher_K_step(x_T))
    - Student learns to match teacher's K-step output in 1 step

    Args:
        input_dim: Action dimension
        global_cond_dim: Observation feature dimension
        model_config: Configuration dict for ConditionalUnet1D
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple = (512, 1024, 2048),
        kernel_size: int = 5,
        n_groups: int = 8,
        condition_type: str = 'film',
        use_down_condition: bool = True,
        use_mid_condition: bool = True,
        use_up_condition: bool = True,
    ):
        super().__init__()

        # Student network (same architecture as teacher)
        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim

    def forward(
        self,
        noisy_action: torch.Tensor,
        global_cond: torch.Tensor,
        timestep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict clean action from noisy action in 1 step.

        Args:
            noisy_action: [B, T, Da] noisy action trajectory
            global_cond: [B, Do*n_obs_steps] observation features
            timestep: [B] timestep (optional, for compatibility)

        Returns:
            clean_action: [B, T, Da] predicted clean action
        """
        # Use a dummy timestep for consistency model (always predicts from noise)
        if timestep is None:
            batch_size = noisy_action.shape[0]
            # Use maximum timestep to indicate "fully noisy"
            timestep = torch.ones(batch_size, device=noisy_action.device, dtype=torch.long) * 999

        # Predict clean action directly
        clean_action_pred = self.model(
            sample=noisy_action,
            timestep=timestep,
            global_cond=global_cond
        )

        return clean_action_pred

    def predict_action(
        self,
        batch_size: int,
        horizon: int,
        global_cond: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate action in 1 step (for fast inference).

        Args:
            batch_size: Batch size
            horizon: Action trajectory length
            global_cond: [B, Do*n_obs_steps] observation features
            device: Device to generate on

        Returns:
            action: [B, horizon, Da] generated action trajectory
        """
        # Sample initial noise
        noisy_action = torch.randn(
            batch_size, horizon, self.input_dim,
            device=device
        )

        # One-step prediction
        clean_action = self.forward(noisy_action, global_cond)

        return clean_action


class ConsistencyDistillation:
    """
    Consistency Distillation Trainer.

    Distills a K-step teacher (DP3) into a 1-step student (ConsistencyModel).

    Algorithm:
    1. Sample batch from dataset
    2. Encode observations with shared encoder
    3. Generate K-step teacher output (DDIM)
    4. Generate 1-step student output
    5. Minimize L2 distance: L_CD = ||student - teacher||^2

    Args:
        teacher_policy: DP3 policy (used as no-grad teacher during distillation)
        student_model: ConsistencyModel
        student_optimizer: Optimizer for student
    """

    def __init__(
        self,
        teacher_policy,  # DP3 policy
        student_model: ConsistencyModel,
        student_optimizer: torch.optim.Optimizer
    ):
        self.teacher_policy = teacher_policy
        self.student_model = student_model
        self.student_optimizer = student_optimizer

        # Do not freeze teacher params here.
        # The teacher is the same policy being optimized by IL/PPO in RL100Trainer.
        # Freezing here would disable training and break backward in IL stage.

    def compute_distillation_loss(
        self,
        obs_dict: Dict[str, torch.Tensor],
        global_cond: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Compute consistency distillation loss.

        Args:
            obs_dict: Dictionary with observations
                - 'point_cloud': [B, n_obs_steps, N, 3]
                - 'agent_pos': [B, n_obs_steps, state_dim]

        Returns:
            loss: scalar distillation loss
            info: dict with logging info
        """
        device = next(self.student_model.parameters()).device
        batch_size = obs_dict['point_cloud'].shape[0]
        horizon = self.teacher_policy.horizon
        action_dim = self.teacher_policy.action_dim

        # Use teacher in eval mode for stable targets, then restore mode.
        teacher_was_training = self.teacher_policy.training
        self.teacher_policy.eval()

        try:
            if global_cond is None:
                with torch.no_grad():
                    if hasattr(self.teacher_policy, 'encode_obs_global_cond'):
                        global_cond = self.teacher_policy.encode_obs_global_cond(obs_dict)
                    else:
                        # Normalize observations (using teacher's normalizer)
                        nobs = self.teacher_policy.normalizer.normalize(obs_dict)

                        # Remove color if needed
                        if not self.teacher_policy.use_pc_color:
                            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

                        this_nobs = {}
                        for key, value in nobs.items():
                            this_nobs[key] = value[:, :self.teacher_policy.n_obs_steps, ...].reshape(
                                -1, *value.shape[2:]
                            )

                        nobs_features = self.teacher_policy.obs_encoder(this_nobs)
                        if "cross_attention" in getattr(self.teacher_policy, 'condition_type', ''):
                            global_cond = nobs_features.reshape(batch_size, self.teacher_policy.n_obs_steps, -1)
                        else:
                            global_cond = nobs_features.reshape(batch_size, -1)
            else:
                # CD historically does not update the observation encoder, so
                # keep cached conditions detached when they are provided.
                global_cond = global_cond.detach()

            # Generate teacher output (K-step DDIM)
            with torch.no_grad():
                # Sample initial noise
                noisy_action = torch.randn(
                    batch_size, horizon, action_dim,
                    device=device
                )

                # Run teacher denoising (K steps)
                teacher_output = self.teacher_policy.conditional_sample(
                    condition_data=noisy_action,
                    condition_mask=torch.zeros_like(noisy_action, dtype=torch.bool),
                    global_cond=global_cond,
                    initial_noise=noisy_action,
                )

            # Generate student output (1 step)
            student_output = self.student_model(
                noisy_action=noisy_action,
                global_cond=global_cond
            )

            # Compute L2 loss
            cd_loss = F.mse_loss(student_output, teacher_output)
        finally:
            if teacher_was_training:
                self.teacher_policy.train()

        info = {
            'cd_loss': cd_loss.item(),
            'teacher_mean': teacher_output.mean().item(),
            'teacher_std': teacher_output.std().item(),
            'student_mean': student_output.mean().item(),
            'student_std': student_output.std().item(),
        }

        return cd_loss, info

    def train_step(
        self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> Dict:
        """
        Perform one distillation training step.

        Args:
            obs_dict: Dictionary with observations

        Returns:
            info: dict with logging info
        """
        self.student_model.train()

        # Compute loss
        loss, info = self.compute_distillation_loss(obs_dict)

        # Optimize
        self.student_optimizer.zero_grad()
        loss.backward()
        self.student_optimizer.step()

        return info

    def save_student(self, path: str):
        """Save student model."""
        torch.save({
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.student_optimizer.state_dict(),
        }, path)

    def load_student(self, path: str):
        """Load student model."""
        checkpoint = torch.load(path)
        self.student_model.load_state_dict(checkpoint['model_state_dict'])
        self.student_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    # Test Consistency Model
    print("Testing Consistency Model...")

    batch_size = 16
    horizon = 16
    action_dim = 4
    obs_feature_dim = 64
    n_obs_steps = 2

    # Create model
    consistency_model = ConsistencyModel(
        input_dim=action_dim,
        global_cond_dim=obs_feature_dim * n_obs_steps,
        diffusion_step_embed_dim=128,
        down_dims=(512, 1024, 2048),
        condition_type='film'
    )

    # Test forward pass
    noisy_action = torch.randn(batch_size, horizon, action_dim)
    global_cond = torch.randn(batch_size, obs_feature_dim * n_obs_steps)

    clean_action = consistency_model(noisy_action, global_cond)
    print(f"Input shape: {noisy_action.shape}")
    print(f"Output shape: {clean_action.shape}")
    print(f"Output mean: {clean_action.mean().item():.4f}")
    print(f"Output std: {clean_action.std().item():.4f}")

    # Test fast generation
    fast_action = consistency_model.predict_action(
        batch_size=batch_size,
        horizon=horizon,
        global_cond=global_cond,
        device=torch.device('cpu')
    )
    print(f"Fast generation output shape: {fast_action.shape}")

    print("Consistency Model test passed!")
