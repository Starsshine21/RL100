from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


class OfflineRLDataset(BaseDataset):
    """
    适配DP3框架的离线RL数据集
    核心：继承BaseDataset，复用ReplayBuffer/SequenceSampler，补充RL所需的reward/next_obs/done
    支持MetaWorld（3D点云）/RealDex（6D点云）任务
    """
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_type='metaworld',  # 'metaworld'/'realdex'，适配不同点云维度
            reward_type='sparse', # 奖励计算方式：'distance'（距离）/'sparse'（稀疏）
            target_pos=None,        # 奖励计算的目标位置（如推块任务的目标）
            augment_pc=False,       # 是否启用点云增强
            # MetaWorld新增配置
            full_state_dim=20,      # MetaWorld的full_state维度
            pc_transform=None,      # 点云变换矩阵（和MetaWorldWrapper对齐）
            pc_scale=np.array([1,1,1]), # 点云缩放
            pc_offset=np.array([0,0,0]), # 点云偏移
            **kwargs
            ):
        super().__init__()
        # 1. 加载zarr数据（包含RL所需字段，若没有则后续实时计算）
        # 优先加载reward/next_obs/done，没有则只加载基础字段
        try:
            # 尝试加载所有可能存在的字段
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, 
                keys=['state', 'action', 'point_cloud', 'reward', 'done', 'full_state', 'next_state', 'next_point_cloud']
            )
            self.has_reward_done = True
        except:
            # 如果上面失败了（通常是因为缺 next_state），回退到加载基础字段
            # 【关键】这里必须包含 'full_state'、'reward'、'done'
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, 
                keys=['state', 'action', 'point_cloud', 'reward', 'done', 'full_state']
            )
            self.has_reward_done = True
        # 2. 划分训练/验证集（复用现有逻辑）
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # 3. 时序采样（复用SequenceSampler，对齐DP3的horizon）
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # 4. RL相关配置
        self.task_type = task_type
        self.reward_type = reward_type
        self.target_pos = target_pos if target_pos is not None else np.array([0.5, 0.5, 0.0])  # 默认目标位置
        self.augment_pc = augment_pc

        # 5. MetaWorld专属配置
        self.full_state_dim = full_state_dim
        self.pc_transform = pc_transform
        self.pc_scale = pc_scale
        self.pc_offset = pc_offset

    def get_validation_dataset(self):
        """返回验证集（复用现有逻辑）"""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """生成归一化器（逻辑简化版）"""
        
        # 1. 尝试获取 full_state，如果没有就用 state 代替（兼容性）
        if 'full_state' in self.replay_buffer:
            full_state_data = self.replay_buffer['full_state']
        else:
            full_state_data = self.replay_buffer['state']

        data = {
            'action': self.replay_buffer['action'],
            # 【核心修复】现在 state 就是 9维的 agent_pos，直接用，不要切片
            'agent_pos': self.replay_buffer['state'], 
            'point_cloud': self.replay_buffer['point_cloud'],
            'full_state': full_state_data
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    def get_all_actions(self) -> torch.Tensor:
        """实现BaseDataset接口，返回所有动作"""
        return torch.from_numpy(self.replay_buffer['action']).float()

    def _augment_point_cloud(self, pc: np.ndarray) -> np.ndarray:
        """点云增强：适配MetaWorld的X/Y轴旋转 + 小平移"""
        # 只增强前3维（XYZ），RGB维不变
        xyz = pc[..., :3]
        
        # 1. 应用MetaWorld的点云变换（可选）
        if self.pc_transform is not None:
            xyz = xyz @ self.pc_transform.T
        
        # 2. 随机旋转（X轴：-10°~10°，Y轴：-10°~10°，Z轴：-15°~15°）
        # X轴旋转
        angle_x = np.random.uniform(-np.pi/18, np.pi/18)
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), np.sin(angle_x)],
            [0, -np.sin(angle_x), np.cos(angle_x)]
        ], dtype=np.float32)
        # Y轴旋转
        angle_y = np.random.uniform(-np.pi/18, np.pi/18)
        rot_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ], dtype=np.float32)
        # Z轴旋转
        angle_z = np.random.uniform(-np.pi/12, np.pi/12)
        rot_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0.],
            [np.sin(angle_z), np.cos(angle_z), 0.],
            [0., 0., 1.]
        ], dtype=np.float32)
        # 组合旋转
        rot_mat = rot_z @ rot_y @ rot_x
        xyz = xyz @ rot_mat
        
        # 3. 随机平移（±0.01m，适配MetaWorld的尺度）
        trans = np.random.normal(0, 0.01, size=3)
        xyz = xyz + trans
        
        # 4. 应用缩放和偏移
        xyz = xyz * self.pc_scale + self.pc_offset
        
        # 拼接回原维度（3D/6D）
        if pc.shape[-1] == 6:
            pc = np.concatenate([xyz, pc[..., 3:]], axis=-1)
        else:
            pc = xyz
        return pc

    def _compute_reward(self, sample: Dict) -> np.ndarray:
        """实时计算奖励（适配MetaWorld不同任务）"""
        T = len(sample['state'])
        reward = np.zeros(T, dtype=np.float32)

        if self.reward_type == 'distance':
            # 距离奖励：块到目标的负距离（越近奖励越高）
            if self.task_type == 'metaworld':
                # MetaWorld不同任务的目标位置映射
                # sample['state']结构：agent_pos（9维） + 块位置（3维） + 目标位置（3维） + 其他（5维）
                block_pos = sample['state'][..., 9:12]  # 块位置（MetaWorld标准）
                target_pos = sample['state'][..., 12:15] # 目标位置（从state中读取）
                distance = np.linalg.norm(block_pos - target_pos, axis=-1)
                reward = -distance  # (T,)
            else:
                # RealDex：点云第0个点是物体位置
                block_pos = sample['point_cloud'][:, 0, :3]  # (T, 3)
                distance = np.linalg.norm(block_pos - self.target_pos, axis=-1)
                reward = -distance
        elif self.reward_type == 'sparse':
            # 稀疏奖励：到达目标（距离<0.05）奖励1，否则0
            if self.task_type == 'metaworld':
                block_pos = sample['state'][..., 9:12]
                target_pos = sample['state'][..., 12:15]
                distance = np.linalg.norm(block_pos - target_pos, axis=-1)
                if 'success' in sample:                                                                                           
                    reward = sample['success'].astype(np.float32)                                                                 
                else:   
                    reward = (distance < 0.05).astype(np.float32)
            else:
                block_pos = sample['point_cloud'][:, 0, :3]
                distance = np.linalg.norm(block_pos - self.target_pos, axis=-1)
                reward = (distance < 0.05).astype(np.float32)
        return reward

    def _get_next_obs(self, sample: Dict) -> Dict:
        """生成next_obs（时序后移一步）"""
        T = len(sample['state'])
        # next_state：state[1:] + 最后一帧重复（避免长度不一致）
        next_state = np.concatenate([sample['state'][1:], sample['state'][-1:]], axis=0)
        # next_point_cloud：同理
        next_point_cloud = np.concatenate([sample['point_cloud'][1:], sample['point_cloud'][-1:]], axis=0)
        return {
            'agent_pos': next_state[..., :self.full_state_dim-11].astype(np.float32),
            'point_cloud': next_point_cloud.astype(np.float32),
            'full_state': next_state[..., :self.full_state_dim].astype(np.float32)
        }

    def _sample_to_data(self, sample: Dict) -> Dict:
        # 1. 读取基础数据
        # 【关键修正】Zarr里的 'state' 是 Wrapper 生成的 9维 agent_pos
        agent_pos = sample['state'].astype(np.float32) 
        point_cloud = sample['point_cloud'][..., :3].astype(np.float32)
        
        # 2. 读取 Full State (39维)
        if 'full_state' in sample:
            full_state = sample['full_state'].astype(np.float32)
        else:
            # 兼容性处理：如果真没有，为了防止报错，只能造假数据或者报错
            # 但既然我们用了新采集脚本，这里一定会有
            full_state = np.zeros((len(agent_pos), 39), dtype=np.float32)

        # 3. 数据增强 (不变)
        if self.augment_pc and self.train_mask is not None:
            point_cloud = self._augment_point_cloud(point_cloud)

        # 4. 构建 Next Obs (Shift 操作)
        next_agent_pos = np.concatenate([agent_pos[1:], agent_pos[-1:]], axis=0)
        next_point_cloud = np.concatenate([point_cloud[1:], point_cloud[-1:]], axis=0)
        next_full_state = np.concatenate([full_state[1:], full_state[-1:]], axis=0)

        # 5. 读取/计算 Reward 和 Done
        if self.has_reward_done:
            reward = sample['reward'].astype(np.float32)
            done = sample['done'].astype(np.float32)
        else:
            # 如果需要实时计算，必须用 full_state (39维)
            # 临时把 full_state 塞进 sample 字典给 _compute_reward 用
            sample['state_for_reward'] = full_state 
            reward = self._compute_reward(sample) 
            done = np.zeros_like(reward)
            done[-1] = 1.0

        data = {
            'obs': {
                'point_cloud': point_cloud,  # (16, 1024, 3)
                'agent_pos': agent_pos,      # (16, 9)
                'full_state': full_state     # (16, 39)
            },
            'action': sample['action'].astype(np.float32),
            'reward': reward,
            'next_obs': {
                'point_cloud': next_point_cloud,
                'agent_pos': next_agent_pos,
                'full_state': next_full_state
            },
            'done': done
        }
        return data


    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """核心接口：返回RL所需的完整样本"""
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        # 转为torch张量（复用现有工具函数）
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data