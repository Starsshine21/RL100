from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy_3d.model.rl.iql_critics import MLP


class QAMCritics(nn.Module):
    """
    QAM critic ensemble with pessimistic target backup.

    This follows QAM Eq.26: the target value uses the mean minus rho times
    the standard deviation across an ensemble of target Q-functions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_qs: int = 10,
        hidden_dims: tuple = (1024, 1024),
        gamma: float = 0.99,
        rho: float = 0.5,
        target_update_tau: float = 0.005,
        use_layernorm: bool = False,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_qs = int(num_qs)
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.target_update_tau = float(target_update_tau)

        input_dim = self.obs_dim + self.action_dim
        self.q_networks = nn.ModuleList([
            MLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
                use_layernorm=use_layernorm,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(self.num_qs)
        ])
        self.target_q_networks = nn.ModuleList([
            MLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
                use_layernorm=use_layernorm,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(self.num_qs)
        ])
        self.target_q_networks.load_state_dict(self.q_networks.state_dict())
        for param in self.target_q_networks.parameters():
            param.requires_grad_(False)

    def update_target_network(self, tau: float = None):
        tau = self.target_update_tau if tau is None else float(tau)
        for online_q, target_q in zip(self.q_networks, self.target_q_networks):
            for param, target_param in zip(online_q.parameters(), target_q.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        use_target: bool = False,
    ) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        q_modules = self.target_q_networks if use_target else self.q_networks
        qs = [q_network(x) for q_network in q_modules]
        return torch.stack(qs, dim=0)

    def get_q_statistics(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        use_target: bool = False,
    ) -> Dict[str, torch.Tensor]:
        qs = self.forward(obs, action, use_target=use_target).squeeze(-1)
        q_mean = qs.mean(dim=0, keepdim=False)
        q_std = qs.std(dim=0, unbiased=False, keepdim=False)
        q_min, _ = qs.min(dim=0)
        q_max, _ = qs.max(dim=0)
        return {
            "all_q": qs,
            "mean": q_mean.unsqueeze(-1),
            "std": q_std.unsqueeze(-1),
            "min": q_min.unsqueeze(-1),
            "max": q_max.unsqueeze(-1),
        }

    def get_mean_q(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        use_target: bool = False,
    ) -> torch.Tensor:
        return self.get_q_statistics(obs, action, use_target=use_target)["mean"]

    def get_pessimistic_q(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        use_target: bool = False,
        rho: float = None,
    ) -> torch.Tensor:
        rho = self.rho if rho is None else float(rho)
        stats = self.get_q_statistics(obs, action, use_target=use_target)
        return stats["mean"] - rho * stats["std"]

    def compute_q_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        next_action: torch.Tensor,
        rho: float = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        rho = self.rho if rho is None else float(rho)
        with torch.no_grad():
            next_stats = self.get_q_statistics(next_obs, next_action, use_target=True)
            target_q = reward + self.gamma * (1.0 - done) * (
                next_stats["mean"] - rho * next_stats["std"]
            )

        pred_q = self.forward(obs, action, use_target=False)
        loss = F.mse_loss(pred_q, target_q.unsqueeze(0).expand_as(pred_q))
        info = {
            "critic_loss": float(loss.item()),
            "q_mean": float(pred_q.mean().item()),
            "q_min": float(pred_q.min().item()),
            "q_max": float(pred_q.max().item()),
            "target_q_mean": float(target_q.mean().item()),
            "target_q_std": float(next_stats["std"].mean().item()),
        }
        return loss, info
