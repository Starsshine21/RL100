import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
# from pytorch3d.loss import chamfer_distance

def chamfer_distance(pc1, pc2):
    """
    手写版 Chamfer Distance，无需 pytorch3d，永不报错。
    """
    # pc1: (B, N, C), pc2: (B, M, C)
    dist = torch.cdist(pc1, pc2, p=2) # 计算欧氏距离矩阵
    dist_sq = dist ** 2 

    # 找最近邻
    min_dist_pc1, _ = torch.min(dist_sq, dim=2) # (B, N)
    min_dist_pc2, _ = torch.min(dist_sq, dim=1) # (B, M)
    
    # 求和平均
    loss = torch.mean(min_dist_pc1) + torch.mean(min_dist_pc2)
    
    # 返回 loss 和 None (None是为了模拟 pytorch3d 的第二个返回值，防止代码报错)
    return loss, None
# =======================================================

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud (带颜色，新增VIB+重建)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 use_vib: bool=True,  # 新增VIB开关
                 **kwargs
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels    



        
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        cprint("pointnet use_vib: {}".format(use_vib), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
        # ========== 新增VIB概率编码 ==========
        self.use_vib = use_vib
        if self.use_vib:
            self.final_projection_mu = nn.Linear(block_channel[-1], out_channels)
            self.final_projection_logvar = nn.Linear(block_channel[-1], out_channels)
        else:
            if final_norm == 'layernorm':
                self.final_projection = nn.Sequential(
                    nn.Linear(block_channel[-1], out_channels),
                    nn.LayerNorm(out_channels)
                )
            elif final_norm == 'none':
                self.final_projection = nn.Linear(block_channel[-1], out_channels)
            else:
                raise NotImplementedError(f"final_norm: {final_norm}")

        # ========== 新增重建解码器 ==========
        self.use_recon = use_vib
        if self.use_recon:
            self.recon_decoder = nn.Sequential(
                nn.Linear(out_channels, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 512 * (in_channels))  # 带颜色则重建6维（xyz+rgb）
            )

    def forward(self, x, return_recon=False):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]

        if self.use_vib:
            mu = self.final_projection_mu(x)
            logvar = self.final_projection_logvar(x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            feat = mu + eps * std
        else:
            feat = self.final_projection(x)
            mu, logvar = None, None

        recon_pc = None
        if self.use_recon and return_recon:
            recon_feat = mu if self.use_vib else feat
            recon_pc = self.recon_decoder(recon_feat)
            recon_pc = recon_pc.reshape(-1, 512, self.in_channels)  # 匹配输入维度

        if return_recon:
            return feat, mu, logvar, recon_pc
        else:
            return feat

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud (新增VIB概率编码)
    """
    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 use_vib: bool=True,  # 新增：是否启用VIB
                 **kwargs
                 ):
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_vib: {}".format(use_vib), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        # ========== 新增VIB：将原单输出改为mu和logvar ==========
        self.use_vib = use_vib
        if self.use_vib:
            # 输出均值和方差（维度均为out_channels）
            self.final_projection_mu = nn.Linear(block_channel[-1], out_channels)
            self.final_projection_logvar = nn.Linear(block_channel[-1], out_channels)
            # 最终特征维度不变（重参数化后为out_channels）
        else:
            # 保留原确定性输出逻辑
            if final_norm == 'layernorm':
                self.final_projection = nn.Sequential(
                    nn.Linear(block_channel[-1], out_channels),
                    nn.LayerNorm(out_channels)
                )
            elif final_norm == 'none':
                self.final_projection = nn.Linear(block_channel[-1], out_channels)
            else:
                raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection and not self.use_vib:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        # ========== 新增：点云重建解码器（RL-100表征正则化） ==========
        self.use_recon = use_vib  # 与VIB联动启用
        if self.use_recon:
            # 解码器：将隐特征（out_channels）重建为点云（默认512个点，3维坐标）
            self.recon_decoder = nn.Sequential(
                nn.Linear(out_channels, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 512 * 3)  # 输出512个点的x/y/z
            )
            cprint("[PointNetEncoderXYZ] enable point cloud reconstruction", "cyan")

    def forward(self, x, return_recon=False):
        """
        新增返回值：
        - feat: 重参数化后的隐特征（用于主模型）
        - mu/logvar: 隐分布的均值和方差（用于VIB损失）
        - recon_pc: 重建的点云（用于Recon损失，仅return_recon=True时返回）
        """
        # 1. 原MLP编码
        x = self.mlp(x)
        x = torch.max(x, 1)[0]  # 全局池化：B, 256
        
        # 2. VIB概率编码/原确定性编码
        if self.use_vib:
            mu = self.final_projection_mu(x)  # B, out_channels
            logvar = self.final_projection_logvar(x)  # B, out_channels
            # 重参数化采样（训练时带噪声，推理时用mu）
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            feat = mu + eps * std  # B, out_channels
        else:
            feat = self.final_projection(x)
            mu, logvar = None, None
        
        # 3. 点云重建（仅训练时启用）
        recon_pc = None
        if self.use_recon and return_recon:
            recon_feat = mu if self.use_vib else feat  # 用均值重建更稳定
            recon_pc = self.recon_decoder(recon_feat)  # B, 512*3
            recon_pc = recon_pc.reshape(-1, 512, 3)  # B, 512, 3

        # 兼容原返回逻辑：主模型只需要feat，mu/logvar和recon_pc用于计算正则化损失
        if return_recon:
            return feat, mu, logvar, recon_pc
        else:
            return feat


class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 # 新增RL-100正则化参数
                 use_recon_vib=True,
                 beta_recon=1.0,  # 重建损失权重
                 beta_kl=0.001    # KL损失权重
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        cprint(f"[DP3Encoder] use_recon_vib: {use_recon_vib}", "red")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        # ========== 传递VIB开关到子编码器 ==========
        pointcloud_encoder_cfg['use_vib'] = use_recon_vib
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        # 原state_mlp逻辑不变
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        # ========== 新增RL-100正则化参数 ==========
        self.use_recon_vib = use_recon_vib
        if self.use_recon_vib:
            # 输入是 state_feat 的维度 (output_dim)，输出回原始状态维度 (self.state_shape[0])
            # 结构很简单：特征 -> 原始数据
            self.state_recon_decoder = nn.Linear(output_dim, self.state_shape[0])
        else:
            self.state_recon_decoder = nn.Identity()
        self.beta_recon = beta_recon
        self.beta_kl = beta_kl
        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict, return_reg_loss=False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        新增return_reg_loss：True时返回特征+正则化损失（Recon+KL），False时仅返回特征（兼容DP3原逻辑）
        """
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)  # B, N_total, 3/6

        # ========== 调用子编码器，获取隐特征和正则化相关输出 ==========
        if self.use_recon_vib and return_reg_loss:
            pn_feat, mu, logvar, recon_pc = self.extractor(points, return_recon=True)
        else:
            pn_feat = self.extractor(points)
            mu = logvar = recon_pc = None
            
        # 原state_mlp编码
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)  # B, out_channel+64

        # ========== 计算RL-100的表征正则化损失 ==========
        reg_loss_dict = {}
        if self.use_recon_vib and return_reg_loss:
            # 1. KL损失（VIB） - 没问题
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            reg_loss_dict['kl_loss'] = self.beta_kl * kl_loss

            # 2. 重建损失（Chamfer距离）
            raw_pc = points  # B, N, C
            B, N, C = raw_pc.shape
            target_n = 512
            
            # 修复采样/补全逻辑
            if N > target_n:
                idx = torch.randperm(N)[:target_n].to(raw_pc.device)
                sampled_pc = raw_pc[:, idx, :]
                
            elif N < target_n:
                # 修复方案: 使用重复填充代替补零
                repeat_times = (target_n // N) + 1
                sampled_pc = raw_pc.repeat(1, repeat_times, 1)
                sampled_pc = sampled_pc[:, :target_n, :]
            else:
                sampled_pc = raw_pc
            pc_recon_loss, _ = chamfer_distance(recon_pc, sampled_pc)
            
            # 3. [新增] 状态重建损失 (MSE) - 对应公式15的 ||q^ - q||^2
            # 先把特征还原
            recon_state = self.state_recon_decoder(state_feat)
            # 计算 MSE (均方误差)
            state_recon_loss = F.mse_loss(recon_state, state)
            
            # 4. 总重建损失 = 点云重建 + 状态重建
            # 注意：beta_recon 权重同时作用于两者
            total_recon_loss = self.beta_recon * (pc_recon_loss + state_recon_loss)
            
            reg_loss_dict['recon_loss'] = total_recon_loss
            reg_loss_dict['total_reg_loss'] = reg_loss_dict['kl_loss'] + total_recon_loss

        if return_reg_loss:
            return final_feat, reg_loss_dict
        else:
            return final_feat

    def output_shape(self):
        return self.n_output_channels