import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple


class BaseValueNet(nn.Module):
    """基础价值网络类，封装MLP骨干和LayerNorm，避免重复代码"""
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: Tuple[int, ...] = (256, 256), 
                 activation_fn: nn.Module = nn.ReLU):
        super().__init__()
        # 构建MLP骨干网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # 稳定训练，RL-100推荐
            layers.append(activation_fn())
            prev_dim = hidden_dim
        # 输出层（价值预测为标量）
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：输入特征 → 标量价值"""
        return self.mlp(x).squeeze(-1)  # 输出形状：(B,) 或 (B, T)


class ValueNet(BaseValueNet):
    """状态价值网络（V-net）：输入3D点云+机器人状态的联合特征，输出状态价值"""
    def __init__(self, 
                 obs_feat_dim: int,  # 状态特征维度（DP3Encoder输出的特征维度）
                 hidden_dims: Tuple[int, ...] = (256, 256), 
                 activation_fn: nn.Module = nn.ReLU):
        super().__init__(
            input_dim=obs_feat_dim,  # 输入=DP3Encoder输出的全局特征（如256+64=320维）
            hidden_dims=hidden_dims,
            activation_fn=activation_fn
        )


class QValueNet(BaseValueNet):
    """动作价值网络（Q-net）：输入“状态特征+动作”，输出动作价值"""
    def __init__(self, 
                 obs_feat_dim: int,  # 状态特征维度（同ValueNet的obs_feat_dim）
                 action_dim: int,    # 动作维度（单步动作维度，如UR5为2维，灵巧手为22维）
                 hidden_dims: Tuple[int, ...] = (256, 256), 
                 activation_fn: nn.Module = nn.ReLU,
                 is_action_chunk: bool = False,  # 是否支持动作块（action-chunk）
                 chunk_size: Optional[int] = None  # 动作块长度（如8/16，仅is_action_chunk=True时生效）
    ):
        # 计算输入维度：状态特征维度 + 动作维度（动作块需展平为1维）
        if is_action_chunk and chunk_size is not None:
            input_action_dim = action_dim * chunk_size  # 动作块展平（如22维×8步=176维）
        else:
            input_action_dim = action_dim  # 单步动作（如22维）
        
        super().__init__(
            input_dim=obs_feat_dim + input_action_dim,  # 输入=状态特征+动作（展平后）
            hidden_dims=hidden_dims,
            activation_fn=activation_fn
        )
        self.is_action_chunk = is_action_chunk
        self.chunk_size = chunk_size

    def forward(self, 
                obs_feat: torch.Tensor,  # 状态特征：(B, obs_feat_dim) 或 (B, T, obs_feat_dim)
                action: torch.Tensor     # 动作：单步=(B, action_dim)；动作块=(B, chunk_size, action_dim)
                ) -> torch.Tensor:
        # 若为动作块，先展平动作维度（如(B,8,22) → (B, 176)）
        if self.is_action_chunk and self.chunk_size is not None:
            # 处理时序维度（若有）：(B, T, chunk_size, action_dim) → (B, T, chunk_size×action_dim)
            if action.dim() == 4:
                action = action.flatten(start_dim=-2, end_dim=-1)
            # 处理单时序维度：(B, chunk_size, action_dim) → (B, chunk_size×action_dim)
            elif action.dim() == 3:
                action = action.flatten(start_dim=-2, end_dim=-1)
            else:
                raise ValueError(f"Action shape {action.shape} not supported for action-chunk")
        
        # 拼接“状态特征”和“动作”（确保两者维度匹配）
        # 情况1：无时序（B, dim）→ (B, obs_dim + action_dim)
        if obs_feat.dim() == 2 and action.dim() == 2:
            x = torch.cat([obs_feat, action], dim=-1)
        # 情况2：有时序（B, T, dim）→ (B, T, obs_dim + action_dim)
        elif obs_feat.dim() == 3 and action.dim() == 3:
            x = torch.cat([obs_feat, action], dim=-1)
        else:
            raise ValueError(
                f"Obs_feat dim {obs_feat.dim()} and action dim {action.dim()} mismatch. "
                "Support (2,2) or (3,3) dim pairs (batch, [time], dim)."
            )
        
        # 调用父类MLP输出动作价值
        return super().forward(x)