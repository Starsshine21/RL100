from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.common.value_net import ValueNet, QValueNet
from diffusion_policy_3d.common.pytorch_util import dict_apply

class DiffusionStepQNet(nn.Module):
    """
    为每个去噪步学习Q值的网络（双层MDP的下层）
    输入：(obs_feat, action, timestep) -> Q(s, a, t)
    """
    def __init__(self, obs_feat_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        # 时间步embedding
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Q网络：[obs_feat, action, timestep_embed] -> Q
        input_dim = obs_feat_dim + action_dim + 64
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, obs_feat: torch.Tensor, action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        obs_feat: (B, obs_dim) 或 (B*T, obs_dim)
        action: (B, action_dim) 或 (B*T, action_dim)
        timestep: (B,) 或 (B*T,) 去噪时间步 [0, 100]
        """
        # 归一化timestep到[0,1]
        t_normalized = timestep.float().unsqueeze(-1) / 100.0  # (B, 1)
        t_emb = self.timestep_embed(t_normalized)  # (B, 64)

        # 拼接特征
        x = torch.cat([obs_feat, action, t_emb], dim=-1)
        return self.mlp(x).squeeze(-1)


class RL100(DP3):
    """
    RL100改进版：
    1. 扩散模型预测噪声（epsilon）而不是动作
    2. 双层MDP：为每个去噪步分配奖励
    3. 离线RL使用IQL + AM-Q (阈值0.05)
    4. 支持在线RL（GAE）
    """
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon,
            n_action_steps,
            n_obs_steps,
            # === RL100 核心参数 ===
            gamma: float = 0.99,          # 折扣因子
            beta_recon_rl: float = 0.01,  # RL阶段 VIB 重建权重（缩小10倍）
            beta_kl_rl: float = 0.00001,  # RL阶段 VIB KL权重（缩小10倍）
            value_hidden_dims: tuple = (256, 256),
            # AM-Q 参数
            amq_threshold: float = 0.05,  # AM-Q优势阈值
            amq_weight: float = 1.0,      # AM-Q损失权重
            # IQL参数
            omega: float = 0.7,           # IQL Expectile 系数
            is_double_q: bool = True,     # 使用双 Q 网络
            # 双层MDP参数
            use_hierarchical_mdp: bool = True,  # 使用双层MDP
            diffusion_step_gamma: float = 0.99,  # 去噪步的折扣因子
            # GAE参数
            gae_lambda: float = 0.95,     # GAE lambda
            # 方差裁剪参数（用于稳定RL探索）
            use_variance_clipping: bool = True,  # 是否启用方差裁剪
            sigma_min: float = 0.01,        # 最小方差
            sigma_max: float = 0.8,        # 最大方差

            # DP3 原生参数
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            **kwargs):

        # 1. 初始化父类 DP3
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
            use_recon_vib=True,       # 强制开启 VIB
            beta_recon=beta_recon_rl, # 初始权重
            beta_kl=beta_kl_rl,       # 初始权重
            **kwargs
        )

        # 2. 记录 RL 参数
        self.gamma = gamma
        self.beta_recon_rl = beta_recon_rl
        self.beta_kl_rl = beta_kl_rl
        self.amq_threshold = amq_threshold
        self.amq_weight = amq_weight
        self.omega = omega
        self.is_double_q = is_double_q
        self.use_hierarchical_mdp = use_hierarchical_mdp
        self.diffusion_step_gamma = diffusion_step_gamma
        self.gae_lambda = gae_lambda

        # 方差裁剪参数
        self.use_variance_clipping = use_variance_clipping
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # 3. 初始化价值网络 (上层MDP - 环境级别)
        obs_feat_dim = self.obs_encoder.output_shape()
        action_dim = self.action_dim

        # V-Net（状态价值）
        self.v_net = ValueNet(
            obs_feat_dim=obs_feat_dim,
            hidden_dims=value_hidden_dims
        ).to(self.device)

        # Q-Net (Double Q)（环境级别Q值）
        self.q_net1 = QValueNet(
            obs_feat_dim=obs_feat_dim,
            action_dim=action_dim,
            hidden_dims=value_hidden_dims
        ).to(self.device)

        self.target_q_net1 = QValueNet(
            obs_feat_dim=obs_feat_dim,
            action_dim=action_dim,
            hidden_dims=value_hidden_dims
        ).to(self.device)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())

        if self.is_double_q:
            self.q_net2 = QValueNet(
                obs_feat_dim=obs_feat_dim,
                action_dim=action_dim,
                hidden_dims=value_hidden_dims
            ).to(self.device)
            self.target_q_net2 = QValueNet(
                obs_feat_dim=obs_feat_dim,
                action_dim=action_dim,
                hidden_dims=value_hidden_dims
            ).to(self.device)
            self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # 4. 初始化双层MDP的去噪步Q网络（下层MDP）
        if self.use_hierarchical_mdp:
            self.diffusion_q_net1 = DiffusionStepQNet(
                obs_feat_dim=obs_feat_dim,
                action_dim=action_dim,
                hidden_dims=value_hidden_dims
            ).to(self.device)

            self.target_diffusion_q_net1 = DiffusionStepQNet(
                obs_feat_dim=obs_feat_dim,
                action_dim=action_dim,
                hidden_dims=value_hidden_dims
            ).to(self.device)
            self.target_diffusion_q_net1.load_state_dict(self.diffusion_q_net1.state_dict())

            if self.is_double_q:
                self.diffusion_q_net2 = DiffusionStepQNet(
                    obs_feat_dim=obs_feat_dim,
                    action_dim=action_dim,
                    hidden_dims=value_hidden_dims
                ).to(self.device)

                self.target_diffusion_q_net2 = DiffusionStepQNet(
                    obs_feat_dim=obs_feat_dim,
                    action_dim=action_dim,
                    hidden_dims=value_hidden_dims
                ).to(self.device)
                self.target_diffusion_q_net2.load_state_dict(self.diffusion_q_net2.state_dict())

        # 5. 强制 Prediction Type 为 epsilon（预测噪声）
        if self.noise_scheduler.config.prediction_type != 'epsilon':
            cprint(f"[RL100Improved] Prediction type is '{self.noise_scheduler.config.prediction_type}', forcing to 'epsilon' for RL.", "cyan")
            self.noise_scheduler.config.prediction_type = 'epsilon'

    # ================= 工具函数 =================
    def _get_q_value(self, obs_feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """获取环境级别Q值（上层MDP）"""
        q1 = self.q_net1(obs_feat, action)
        if self.is_double_q:
            q2 = self.q_net2(obs_feat, action)
            return torch.min(q1, q2)
        return q1

    def _get_target_q_value(self, obs_feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """获取Target Q值（上层MDP）"""
        q1 = self.target_q_net1(obs_feat, action)
        if self.is_double_q:
            q2 = self.target_q_net2(obs_feat, action)
            return torch.min(q1, q2)
        return q1

    def _get_diffusion_q_value(self, obs_feat: torch.Tensor, action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """获取去噪步Q值（下层MDP）"""
        if not self.use_hierarchical_mdp:
            return torch.zeros(obs_feat.shape[0], device=obs_feat.device)

        q1 = self.diffusion_q_net1(obs_feat, action, timestep)
        if self.is_double_q:
            q2 = self.diffusion_q_net2(obs_feat, action, timestep)
            return torch.min(q1, q2)
        return q1

    def _get_target_diffusion_q_value(self, obs_feat: torch.Tensor, action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """获取Target去噪步Q值（下层MDP）"""
        if not self.use_hierarchical_mdp:
            return torch.zeros(obs_feat.shape[0], device=obs_feat.device)

        q1 = self.target_diffusion_q_net1(obs_feat, action, timestep)
        if self.is_double_q:
            q2 = self.target_diffusion_q_net2(obs_feat, action, timestep)
            return torch.min(q1, q2)
        return q1

    def update_target_q_networks(self, tau: float = 0.005):
        """软更新 Target 网络"""
        # 更新环境级别Q网络
        for param, target_param in zip(self.q_net1.parameters(), self.target_q_net1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if self.is_double_q:
            for param, target_param in zip(self.q_net2.parameters(), self.target_q_net2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # 更新去噪步Q网络
        if self.use_hierarchical_mdp:
            for param, target_param in zip(self.diffusion_q_net1.parameters(), self.target_diffusion_q_net1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if self.is_double_q:
                for param, target_param in zip(self.diffusion_q_net2.parameters(), self.target_diffusion_q_net2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def switch_to_il_mode(self):
        """切换到 IL 模式（高VIB权重）"""
        # self.obs_encoder.beta_recon = 0.1
        # self.obs_encoder.beta_kl = 0.0001
        self.obs_encoder.beta_recon = 0
        self.obs_encoder.beta_kl = 0
        cprint(f"[RL100Improved] Switched to IL mode (High VIB weights: recon={self.beta_recon_rl}, kl={self.beta_kl_rl})", "cyan")

    def switch_to_rl_mode(self):
        """切换到 RL 模式（低VIB权重，缩小10倍）"""
        self.obs_encoder.beta_recon = self.beta_recon_rl  # 0.1 -> 0.01 (缩小10倍)
        self.obs_encoder.beta_kl = self.beta_kl_rl        # 0.0001 -> 0.00001 (缩小10倍)
        cprint(f"[RL100Improved] Switched to RL mode (Low VIB weights: recon={self.beta_recon_rl}, kl={self.beta_kl_rl})", "cyan")

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        计算Generalized Advantage Estimation (GAE)
        rewards: (B, T)
        values: (B, T)
        dones: (B, T)
        """
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(rewards.shape[1])):
            if t == rewards.shape[1] - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[:, t]) * gae
            advantages[:, t] = gae

        return advantages

    # ================= IL 阶段的 Loss 计算（重写父类以支持 VIB） =================
    def compute_loss(self, batch):
        """
        重写 DP3 的 compute_loss 方法，添加 VIB 正则化损失支持
        用于 IL 阶段训练
        """
        # 1. 数据归一化
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # 2. 特征提取（关键修改：添加 return_reg_loss=True）
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))

            # ===== 关键修改：获取正则化损失 =====
            nobs_features, reg_loss_dict = self.obs_encoder(this_nobs, return_reg_loss=True)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features, reg_loss_dict = self.obs_encoder(this_nobs, return_reg_loss=True)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # 3. 生成 inpainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # 4. 采样噪声
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        # 5. 采样时间步
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()

        # 6. 加噪（前向扩散过程）
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # 7. 计算 loss mask
        loss_mask = ~condition_mask

        # 8. 应用 conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # 9. 预测噪声
        pred = self.model(sample=noisy_trajectory,
                        timestep=timesteps,
                        local_cond=local_cond,
                        global_cond=global_cond)

        # 10. 根据 prediction type 确定目标
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # 11. 计算扩散损失（BC Loss）
        diffusion_loss = F.mse_loss(pred, target, reduction='none')
        diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype)
        diffusion_loss = reduce(diffusion_loss, 'b ... -> b (...)', 'mean')
        diffusion_loss = diffusion_loss.mean()

        # 12. ===== 关键修改：添加 VIB 正则化损失 =====
        reg_loss = reg_loss_dict.get('total_reg_loss', 0.0)
        if isinstance(reg_loss, torch.Tensor) and reg_loss.dim() > 0:
            reg_loss = reg_loss.mean()

        # 13. 总损失 = 扩散损失 + 正则化损失
        total_loss = diffusion_loss + reg_loss

        # 14. 构建损失字典（添加详细的分解）
        loss_dict = {
            'train/bc_loss': diffusion_loss.item(),
            'train/reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            'train/total_loss': total_loss.item(),
        }

        # 添加 VIB 分解损失（如果存在）
        if 'kl_loss' in reg_loss_dict:
            loss_dict['train/kl_loss'] = reg_loss_dict['kl_loss'].item() if isinstance(reg_loss_dict['kl_loss'], torch.Tensor) else reg_loss_dict['kl_loss']
        if 'recon_loss' in reg_loss_dict:
            loss_dict['train/recon_loss'] = reg_loss_dict['recon_loss'].item() if isinstance(reg_loss_dict['recon_loss'], torch.Tensor) else reg_loss_dict['recon_loss']

        return total_loss, loss_dict

    # ================= 核心 Loss 计算 =================
    def _compute_weighted_diffusion_loss(self,
                                         nactions: torch.Tensor,
                                         obs_feat: torch.Tensor,
                                         advantage: torch.Tensor,
                                         nobs: Dict,
                                         use_amq: bool = False):
        """
        计算优势加权的扩散损失
        nactions: (B, T, A) 归一化的动作
        obs_feat: (B, T, D) 观测特征
        advantage: (B, T) 优势值
        use_amq: 是否使用AM-Q算法筛选样本
        """
        batch_size = nactions.shape[0]

        # 1. 准备条件
        global_cond = obs_feat[:, :self.n_obs_steps, :]  # (B, 2, D)

        if "cross_attention" not in self.condition_type and global_cond.dim() == 3:
            global_cond = global_cond.reshape(batch_size, -1)

        trajectory = nactions  # (B, T, ActDim)

        # 2. 生成噪声和时间步
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=trajectory.device
        ).long()

        # 3. 加噪
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # 4. 模型预测噪声
        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            local_cond=None,
            global_cond=global_cond
        )

        # 5. 目标是噪声（epsilon）
        target = noise

        # 6. 基础MSE Loss
        loss_unreduced = F.mse_loss(pred, target, reduction='none')  # (B, T, A)

        # 7. AM-Q算法：只使用advantage > threshold的样本
        if use_amq and self.amq_weight > 0:
            # 计算每个样本的advantage均值
            adv_mean = advantage.mean(dim=1)  # (B,)

            # 创建mask：只保留advantage > threshold的样本
            amq_mask = (adv_mean > self.amq_threshold).float()  # (B,)

            # 统计被选中的样本比例
            selected_ratio = amq_mask.mean().item()

            # 扩展mask维度以匹配loss
            amq_mask = amq_mask.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

            # 应用mask
            loss_unreduced = loss_unreduced * amq_mask

            # 为了避免除以0，至少保留一个样本
            if amq_mask.sum() == 0:
                amq_mask = torch.ones_like(amq_mask)
                selected_ratio = 1.0
        else:
            selected_ratio = 1.0

        # 8. 优势加权
        if advantage.dim() == 2:
            adv_expanded = advantage.unsqueeze(-1)  # (B, T, 1)
        else:
            adv_expanded = advantage.reshape(batch_size, -1, 1)

        # 使用exp(advantage)作为权重，并裁剪
        adv_weight = torch.exp(adv_expanded * 3.0)
        adv_weight = torch.clamp(adv_weight, max=100.0)

        # 9. 加权loss
        weighted_loss = loss_unreduced * adv_weight

        return weighted_loss.mean(), selected_ratio

    def _compute_hierarchical_diffusion_loss(self,
                                            nactions: torch.Tensor,
                                            obs_feat_flat: torch.Tensor,
                                            rewards_flat: torch.Tensor,
                                            nobs: Dict):
        """
        双层MDP的扩散损失：为每个去噪步分配奖励
        nactions: (B, T, A)
        obs_feat_flat: (B*T, D)
        rewards_flat: (B*T,)
        """
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # 1. 准备条件
        obs_feat_seq = obs_feat_flat.reshape(batch_size, horizon, -1)
        global_cond = obs_feat_seq[:, :self.n_obs_steps, :]  # (B, 2, D)

        if "cross_attention" not in self.condition_type and global_cond.dim() == 3:
            global_cond = global_cond.reshape(batch_size, -1)

        trajectory = nactions  # (B, T, A)

        # 2. 生成噪声和时间步
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        # 为每个样本随机采样一个去噪时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=trajectory.device
        ).long()

        # 3. 加噪
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # 4. 模型预测噪声
        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            local_cond=None,
            global_cond=global_cond
        )

        # 5. 计算去噪步的Q值（使用第一帧的obs和action）
        # 取第一帧的特征和动作
        first_obs_feat = obs_feat_flat[:batch_size]  # (B, D)
        first_action = nactions[:, 0, :]  # (B, A)

        # 计算当前时间步的Q值
        with torch.no_grad():
            # 计算下一个时间步的Q值（t-1）
            next_timesteps = torch.clamp(timesteps - 1, min=0)
            next_q = self._get_target_diffusion_q_value(first_obs_feat, first_action, next_timesteps)

            # 计算目标Q值：r + gamma * Q(s, a, t-1)
            # 这里的reward是环境奖励的平均值（近似）
            avg_reward = rewards_flat.reshape(batch_size, horizon).mean(dim=1)  # (B,)
            target_q = avg_reward + self.diffusion_step_gamma * next_q

        # 计算当前Q值
        current_q = self._get_diffusion_q_value(first_obs_feat, first_action, timesteps)

        # Q损失
        q_loss = F.mse_loss(current_q, target_q)

        # 6. 计算advantage：Q(s,a,t) - 平均Q值
        with torch.no_grad():
            advantage_diffusion = current_q - current_q.mean()

        # 7. 优势加权的扩散损失
        adv_weight = torch.exp(advantage_diffusion * 3.0).clamp(max=100.0)
        adv_weight = adv_weight.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

        # 目标是噪声
        target = noise
        loss_unreduced = F.mse_loss(pred, target, reduction='none')  # (B, T, A)

        # 加权
        weighted_loss = loss_unreduced * adv_weight
        diff_loss = weighted_loss.mean()

        return diff_loss, q_loss, advantage_diffusion.mean()

    def compute_iql_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算纯IQL损失（只更新Q、V网络，不更新Policy）
        用于预训练价值网络，避免初始化不好导致的性能下降
        """
        # 1. 数据准备与归一化
        nobs = self.normalizer.normalize(batch['obs'])
        nnext_obs = self.normalizer.normalize(batch['next_obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        rewards = batch['reward'].float()
        dones = batch['done'].float()

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
            nnext_obs['point_cloud'] = nnext_obs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # 2. 特征提取（不需要梯度，因为不更新encoder）
        nobs_flat = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        with torch.no_grad():
            obs_feat_flat = self.obs_encoder(nobs_flat, return_reg_loss=False)

        nnext_obs_flat = dict_apply(nnext_obs, lambda x: x.reshape(-1, *x.shape[2:]))
        with torch.no_grad():
            next_obs_feat_flat = self.obs_encoder(nnext_obs_flat, return_reg_loss=False)

        # Action展平
        action_flat = nactions.reshape(-1, nactions.shape[-1])
        reward_flat = rewards.reshape(-1)
        done_flat = dones.reshape(-1)

        # 3. 计算Q Loss
        current_q = self._get_q_value(obs_feat_flat, action_flat).squeeze(-1)

        with torch.no_grad():
            next_v = self.v_net(next_obs_feat_flat).squeeze(-1)
            target_q = reward_flat + self.gamma * next_v * (1 - done_flat)

        q_loss = F.mse_loss(current_q, target_q)

        # 4. 计算V Loss (IQL Expectile)
        with torch.no_grad():
            target_q_for_v = self._get_target_q_value(obs_feat_flat, action_flat).squeeze(-1)

        current_v = self.v_net(obs_feat_flat).squeeze(-1)
        adv_for_v = target_q_for_v - current_v

        v_loss_weights = torch.where(adv_for_v > 0, self.omega, 1 - self.omega)
        v_loss = (v_loss_weights * (adv_for_v ** 2)).mean()

        # 5. 总损失（只有Q和V，没有policy和reg）
        total_loss = q_loss + v_loss

        # 6. 计算advantage用于监控
        advantage = (target_q_for_v - current_v).detach()

        loss_dict = {
            'iql/total': total_loss.item(),
            'iql/q': q_loss.item(),
            'iql/v': v_loss.item(),
            'iql/advantage_mean': advantage.mean().item(),
            'iql/q_value_mean': current_q.mean().item(),
            'iql/v_value_mean': current_v.mean().item(),
        }

        return total_loss, loss_dict

    def compute_offline_rl_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算离线 RL 的总损失（改进版）
        支持：
        1. 预测噪声而不是动作
        2. 双层MDP
        3. AM-Q算法
        """
        # 1. 数据准备与归一化
        nobs = self.normalizer.normalize(batch['obs'])
        nnext_obs = self.normalizer.normalize(batch['next_obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        rewards = batch['reward'].float()
        dones = batch['done'].float()

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
            nnext_obs['point_cloud'] = nnext_obs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # 2. 特征提取
        nobs_flat = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_feat_flat, reg_loss_dict = self.obs_encoder(nobs_flat, return_reg_loss=True)

        nnext_obs_flat = dict_apply(nnext_obs, lambda x: x.reshape(-1, *x.shape[2:]))
        with torch.no_grad():
            next_obs_feat_flat = self.obs_encoder(nnext_obs_flat, return_reg_loss=False)

        # 3. 准备不同模块需要的维度
        obs_feat_seq = obs_feat_flat.reshape(batch_size, horizon, -1)

        # Action 展平
        action_flat = nactions.reshape(-1, nactions.shape[-1])
        reward_flat = rewards.reshape(-1)
        done_flat = dones.reshape(-1)

        # 4. 计算环境级别的 Q Loss（上层MDP）
        current_q = self._get_q_value(obs_feat_flat, action_flat).squeeze(-1)

        with torch.no_grad():
            next_v = self.v_net(next_obs_feat_flat).squeeze(-1)
            target_q = reward_flat + self.gamma * next_v * (1 - done_flat)

        q_loss = F.mse_loss(current_q, target_q)

        # 5. 计算 V Loss (IQL Expectile)
        with torch.no_grad():
            target_q_for_v = self._get_target_q_value(obs_feat_flat, action_flat).squeeze(-1)

        current_v = self.v_net(obs_feat_flat).squeeze(-1)
        adv_for_v = target_q_for_v - current_v

        v_loss_weights = torch.where(adv_for_v > 0, self.omega, 1 - self.omega)
        v_loss = (v_loss_weights * (adv_for_v ** 2)).mean()

        # 6. 计算 Advantage 用于 Diffusion
        advantage = (target_q_for_v - current_v).detach()
        advantage = advantage.reshape(batch_size, horizon)

        # 7. Diffusion Loss
        if self.use_hierarchical_mdp:
            # 使用双层MDP的扩散损失
            diff_loss, diffusion_q_loss, avg_diffusion_adv = self._compute_hierarchical_diffusion_loss(
                nactions=nactions,
                obs_feat_flat=obs_feat_flat,
                rewards_flat=reward_flat,
                nobs=nobs
            )
        else:
            # 使用传统的优势加权扩散损失
            diff_loss, selected_ratio = self._compute_weighted_diffusion_loss(
                nactions=nactions,
                obs_feat=obs_feat_seq,
                advantage=advantage,
                nobs=nobs,
                use_amq=(self.amq_weight > 0)
            )
            diffusion_q_loss = torch.tensor(0.0, device=self.device)
            avg_diffusion_adv = 0.0

        # 8. 正则化损失
        reg_loss = reg_loss_dict.get('total_reg_loss', 0.0)
        if isinstance(reg_loss, torch.Tensor) and reg_loss.dim() > 0:
            reg_loss = reg_loss.mean()

        # 9. 总损失
        total_loss = diff_loss + q_loss + v_loss + reg_loss
        if self.use_hierarchical_mdp:
            total_loss = total_loss + diffusion_q_loss

        loss_dict = {
            'loss/total': total_loss.item(),
            'loss/diffusion': diff_loss.item(),
            'loss/q': q_loss.item(),
            'loss/v': v_loss.item(),
            'loss/reg': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            'val/advantage_mean': advantage.mean().item(),
        }

        if self.use_hierarchical_mdp:
            loss_dict['loss/diffusion_q'] = diffusion_q_loss.item()
            loss_dict['val/diffusion_advantage'] = avg_diffusion_adv if isinstance(avg_diffusion_adv, float) else avg_diffusion_adv.item()

        if self.amq_weight > 0 and not self.use_hierarchical_mdp:
            loss_dict['val/amq_selected_ratio'] = selected_ratio

        return total_loss, loss_dict

    # ================= 方差裁剪采样（用于稳定RL探索）=================
    def conditional_sample(self,
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
            ):
        """
        重写父类的conditional_sample方法，添加方差裁剪支持

        方差裁剪用于在RL微调期间提供稳定探索：
        σk = clip(σk, σmin, σmax)
        """
        if not self.use_variance_clipping:
            # 如果未启用方差裁剪，直接调用父类方法
            return super().conditional_sample(
                condition_data, condition_mask,
                condition_data_pc, condition_mask_pc,
                local_cond, global_cond,
                generator,
                **kwargs
            )

        # ===== 启用方差裁剪的采样 =====
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # 设置时间步
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. 应用conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. 模型预测
            model_output = model(
                sample=trajectory,
                timestep=t,
                local_cond=local_cond,
                global_cond=global_cond
            )

            # 3. 计算前一个样本：x_t -> x_t-1（带方差裁剪）
            trajectory = self._step_with_variance_clipping(
                model_output=model_output,
                timestep=t,
                sample=trajectory,
                generator=generator
            )

        # 最终确保conditioning被强制执行
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def _step_with_variance_clipping(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None
    ) -> torch.Tensor:
        """
        执行带方差裁剪的DDPM采样步骤

        Args:
            model_output: 模型预测的噪声（epsilon）或样本
            timestep: 当前时间步
            sample: 当前样本 x_t
            generator: 随机数生成器

        Returns:
            prev_sample: 上一步样本 x_{t-1}
        """
        scheduler = self.noise_scheduler

        # 获取时间步索引
        t = timestep

        # 获取调度参数
        alpha_prod_t = scheduler.alphas_cumprod[t]
        # DDIM兼容性：使用torch.tensor(1.0)替代scheduler.one
        if t > 0:
            alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1]
        else:
            alpha_prod_t_prev = torch.tensor(1.0, device=alpha_prod_t.device, dtype=alpha_prod_t.dtype)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 根据prediction_type计算预测的原始样本
        if scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(f"Unsupported prediction_type: {scheduler.config.prediction_type}")

        # 裁剪预测样本
        if scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-1, 1)

        # 计算方差
        variance = self._get_variance(t, t - 1)

        # ===== 关键：应用方差裁剪 =====
        variance = torch.clamp(variance, min=self.sigma_min**2, max=self.sigma_max**2)
        std_dev_t = variance ** 0.5

        # 计算 x_{t-1} 的均值
        # 使用DDPM公式：μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t))
        alpha_t = scheduler.alphas[t]
        beta_t = scheduler.betas[t]

        if scheduler.config.prediction_type == "epsilon":
            pred_sample_direction = (beta_t / (beta_prod_t ** 0.5)) * model_output
            prev_sample = (sample - pred_sample_direction) / (alpha_t ** 0.5)
        else:
            # 对于其他prediction_type，使用标准公式
            prev_sample = (alpha_prod_t_prev ** 0.5) * pred_original_sample
            if t > 0:
                prev_sample = prev_sample + (beta_prod_t_prev ** 0.5) * model_output

        # 添加噪声（如果不是最后一步）
        if t > 0:
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype
            )
            prev_sample = prev_sample + std_dev_t * noise

        return prev_sample

    def _get_variance(self, timestep, prev_timestep):
        """
        计算DDPM/DDIM方差

        根据variance_type配置计算方差：
        - fixed_small: β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        - fixed_large: β_t
        - learned: 从模型学习
        - None (DDIM): 使用默认计算值
        """
        scheduler = self.noise_scheduler

        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        # DDIM兼容性：使用torch.tensor(1.0)替代scheduler.one
        if prev_timestep >= 0:
            alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep]
        else:
            alpha_prod_t_prev = torch.tensor(1.0, device=alpha_prod_t.device, dtype=alpha_prod_t.dtype)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        # 根据variance_type调整（DDIM调度器没有variance_type，使用getattr安全访问）
        variance_type = getattr(scheduler.config, 'variance_type', None)
        if variance_type == "fixed_small":
            variance = torch.clamp(variance, min=1e-20)
        elif variance_type == "fixed_large":
            variance = scheduler.betas[timestep]
        elif variance_type is None:
            # DDIM调度器没有variance_type，使用默认计算值并裁剪
            variance = torch.clamp(variance, min=1e-20)

        return variance

    def compute_online_rl_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算在线 RL 的损失（使用GAE + 价值网络更新）
        """
        # 1. 数据准备
        nobs = self.normalizer.normalize(batch['obs'])
        nnext_obs = self.normalizer.normalize(batch['next_obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        rewards = batch['reward'].float()
        dones = batch['done'].float()

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
            nnext_obs['point_cloud'] = nnext_obs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # 2. 特征提取
        nobs_flat = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_feat_flat, reg_loss_dict = self.obs_encoder(nobs_flat, return_reg_loss=True)

        nnext_obs_flat = dict_apply(nnext_obs, lambda x: x.reshape(-1, *x.shape[2:]))
        with torch.no_grad():
            next_obs_feat_flat = self.obs_encoder(nnext_obs_flat, return_reg_loss=False)

        obs_feat_seq = obs_feat_flat.reshape(batch_size, horizon, -1)
        action_flat = nactions.reshape(-1, nactions.shape[-1])
        reward_flat = rewards.reshape(-1)
        done_flat = dones.reshape(-1)

        # 3. ===== 添加Q/V网络更新 =====
        # Q Loss
        current_q = self._get_q_value(obs_feat_flat, action_flat).squeeze(-1)
        with torch.no_grad():
            next_v = self.v_net(next_obs_feat_flat).squeeze(-1)
            target_q = reward_flat + self.gamma * next_v * (1 - done_flat)
        q_loss = F.mse_loss(current_q, target_q)

        # V Loss (IQL Expectile)
        with torch.no_grad():
            target_q_for_v = self._get_target_q_value(obs_feat_flat, action_flat).squeeze(-1)
        current_v = self.v_net(obs_feat_flat).squeeze(-1)
        adv_for_v = target_q_for_v - current_v
        v_loss_weights = torch.where(adv_for_v > 0, self.omega, 1 - self.omega)
        v_loss = (v_loss_weights * (adv_for_v ** 2)).mean()

        # 4. 计算状态价值（用于GAE）
        values = current_v.reshape(batch_size, horizon)

        # 5. 计算GAE
        advantages = self.compute_gae(rewards, values, dones)

        # 6. Diffusion Loss（使用GAE计算的advantage）
        diff_loss, selected_ratio = self._compute_weighted_diffusion_loss(
            nactions=nactions,
            obs_feat=obs_feat_seq,
            advantage=advantages,
            nobs=nobs,
            use_amq=False  # 在线RL不使用AM-Q
        )

        # 7. 正则化损失
        reg_loss = reg_loss_dict.get('total_reg_loss', 0.0)
        if isinstance(reg_loss, torch.Tensor) and reg_loss.dim() > 0:
            reg_loss = reg_loss.mean()

        # 8. 总损失（包含Q/V网络）
        total_loss = diff_loss + q_loss + v_loss + reg_loss

        loss_dict = {
            'loss/total': total_loss.item(),
            'loss/diffusion': diff_loss.item(),
            'loss/q': q_loss.item(),
            'loss/v': v_loss.item(),
            'loss/reg': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            'val/advantage_mean': advantages.mean().item(),
            'val/value_mean': values.mean().item(),
        }

        return total_loss, loss_dict
