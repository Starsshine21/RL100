"""
IQL Critics Implementation for RL-100
=====================================
Implements Q and V networks with IQL (Implicit Q-Learning) algorithm.

Key Features:
- V network: Expectile regression (τ=0.7)
- Q network: Learns r + γV(s')
- Twin Q-networks for stability

References:
- IQL Paper: https://arxiv.org/abs/2110.06169
- RL-100 Paper: Extends IQL for diffusion policies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class MLP(nn.Module):
    """Multi-layer perceptron with LayerNorm and Mish activation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (256, 256, 256),
        output_dim: int = 1,
        use_layernorm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Mish())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class VNetwork(nn.Module):
    """
    Value function V(s) for IQL.

    Uses Expectile Regression with τ=0.7 to approximate max_a Q(s,a).
    This is more stable than explicit max operation.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: tuple = (256, 256, 256),
        use_layernorm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        self.network = MLP(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            use_layernorm=use_layernorm,
            dropout=dropout
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, obs_dim] observation features

        Returns:
            value: [B, 1] state value
        """
        return self.network(obs)


class QNetwork(nn.Module):
    """
    Q function Q(s,a) for IQL.

    Twin Q-networks to reduce overestimation bias.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple = (256, 256, 256),
        use_layernorm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        input_dim = obs_dim + action_dim

        # Twin Q networks
        self.q1 = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            use_layernorm=use_layernorm,
            dropout=dropout
        )

        self.q2 = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            use_layernorm=use_layernorm,
            dropout=dropout
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        return_both: bool = True
    ) -> torch.Tensor:
        """
        Args:
            obs: [B, obs_dim] observation features
            action: [B, action_dim] actions
            return_both: if True, return (q1, q2); else return min(q1, q2)

        Returns:
            q_values: [B, 1] or tuple of [B, 1]
        """
        x = torch.cat([obs, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)

        if return_both:
            return q1, q2
        else:
            return torch.min(q1, q2)


class IQLCritics(nn.Module):
    """
    Combined IQL Critics (Q and V networks) for RL-100.

    Training Algorithm:
    1. Update V with Expectile Regression:
       L_V = expectile_loss(V(s), Q(s, a_data), τ=0.7)

    2. Update Q with Bellman backup:
       L_Q = MSE(Q(s, a), r + γV(s'))

    Args:
        obs_dim: Observation feature dimension (e.g., 64 from PointNet encoder)
        action_dim: Action dimension (e.g., 4 for Metaworld push)
        hidden_dims: Hidden layer dimensions for MLP
        gamma: Discount factor (default: 0.99)
        tau: Expectile parameter for V update (default: 0.7)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple = (256, 256, 256),
        gamma: float = 0.99,
        tau: float = 0.7,
        use_layernorm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        self.gamma = gamma
        self.tau = tau
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # V network
        self.v_network = VNetwork(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims,
            use_layernorm=use_layernorm,
            dropout=dropout
        )

        # Twin Q networks
        self.q_network = QNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            use_layernorm=use_layernorm,
            dropout=dropout
        )

        # Target V network for stable Q learning
        self.v_network_target = VNetwork(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims,
            use_layernorm=use_layernorm,
            dropout=dropout
        )
        self.v_network_target.load_state_dict(self.v_network.state_dict())

        # Freeze target network
        for param in self.v_network_target.parameters():
            param.requires_grad = False

    def update_target_network(self, tau: float = 0.005):
        """Soft update of target V network."""
        for param, target_param in zip(
            self.v_network.parameters(),
            self.v_network_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    @staticmethod
    def expectile_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        expectile: float = 0.7
    ) -> torch.Tensor:
        """
        Asymmetric L2 loss for expectile regression.

        When expectile=0.7:
        - Overestimations (pred > target) get weight 0.7
        - Underestimations (pred < target) get weight 0.3

        This biases V towards upper quantile of Q(s, a_data).

        Args:
            pred: [B, 1] predicted values
            target: [B, 1] target values
            expectile: asymmetry parameter τ ∈ [0, 1]

        Returns:
            loss: scalar expectile loss
        """
        diff = target - pred
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        loss = weight * (diff ** 2)
        return loss.mean()

    def compute_v_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute V network loss with expectile regression.

        L_V = expectile_loss(V(s), Q(s, a_data), τ=0.7)

        Args:
            obs: [B, obs_dim] observation features
            action: [B, action_dim] actions from dataset

        Returns:
            loss: scalar V loss
            info: dict with logging info
        """
        with torch.no_grad():
            # Get Q target (use minimum of twin Q)
            q1, q2 = self.q_network(obs, action, return_both=True)
            q_target = torch.min(q1, q2)

        # Predict V
        v_pred = self.v_network(obs)

        # Expectile regression loss
        v_loss = self.expectile_loss(v_pred, q_target, self.tau)

        info = {
            'v_loss': v_loss.item(),
            'v_mean': v_pred.mean().item(),
            'q_target_mean': q_target.mean().item(),
        }

        return v_loss, info

    def compute_q_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute Q network loss with Bellman backup.

        L_Q = MSE(Q(s, a), r + γ(1-done)V_target(s'))

        Args:
            obs: [B, obs_dim] observation features
            action: [B, action_dim] actions
            reward: [B, 1] rewards
            next_obs: [B, obs_dim] next observation features
            done: [B, 1] done flags

        Returns:
            loss: scalar Q loss
            info: dict with logging info
        """
        # Compute target
        with torch.no_grad():
            next_v = self.v_network_target(next_obs)
            q_target = reward + self.gamma * (1 - done) * next_v

        # Predict Q values
        q1_pred, q2_pred = self.q_network(obs, action, return_both=True)

        # MSE loss for both Q networks
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        q_loss = q1_loss + q2_loss

        info = {
            'q_loss': q_loss.item(),
            'q1_mean': q1_pred.mean().item(),
            'q2_mean': q2_pred.mean().item(),
            'q_target_mean': q_target.mean().item(),
            'reward_mean': reward.mean().item(),
        }

        return q_loss, info

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get V(s) for advantage computation."""
        return self.v_network(obs)

    def get_q_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get min Q(s,a) for advantage computation."""
        return self.q_network(obs, action, return_both=False)

    def compute_advantage(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantage A(s,a) = Q(s,a) - V(s).

        This is used as the shared advantage for K-step PPO loss.

        Args:
            obs: [B, obs_dim] observation features
            action: [B, action_dim] final denoised actions

        Returns:
            advantage: [B, 1] advantage values
        """
        q_value = self.get_q_value(obs, action)
        v_value = self.get_value(obs)
        advantage = q_value - v_value
        return advantage


if __name__ == "__main__":
    # Test IQL Critics
    print("Testing IQL Critics...")

    batch_size = 256
    obs_dim = 64
    action_dim = 4

    critics = IQLCritics(
        obs_dim=obs_dim,
        action_dim=action_dim,
        gamma=0.99,
        tau=0.7
    )

    # Create dummy data
    obs = torch.randn(batch_size, obs_dim)
    action = torch.randn(batch_size, action_dim)
    reward = torch.randn(batch_size, 1)
    next_obs = torch.randn(batch_size, obs_dim)
    done = torch.zeros(batch_size, 1)

    # Test V loss
    v_loss, v_info = critics.compute_v_loss(obs, action)
    print(f"V Loss: {v_loss.item():.4f}")
    print(f"V Info: {v_info}")

    # Test Q loss
    q_loss, q_info = critics.compute_q_loss(obs, action, reward, next_obs, done)
    print(f"Q Loss: {q_loss.item():.4f}")
    print(f"Q Info: {q_info}")

    # Test advantage
    advantage = critics.compute_advantage(obs, action)
    print(f"Advantage shape: {advantage.shape}")
    print(f"Advantage mean: {advantage.mean().item():.4f}")

    print("IQL Critics test passed!")
