from typing import Dict
import torch
import numpy as np
import copy
import zarr
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class MetaworldDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            ):
        super().__init__()

        # Check which keys are available (support both old and new zarr formats)
        _zarr_store = zarr.open(zarr_path, 'r')
        _available_keys = list(_zarr_store['data'].keys())
        self.has_rl_data = 'reward' in _available_keys and 'done' in _available_keys
        keys_to_load = ['state', 'action', 'point_cloud']
        if self.has_rl_data:
            keys_to_load.extend(['reward', 'done'])

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keys_to_load)
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

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

    def get_validation_dataset(self):
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
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32)
        point_cloud = sample['point_cloud'][:,].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud, 
                'agent_pos': agent_pos, 
            },
            'action': sample['action'].astype(np.float32)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        if self.has_rl_data:
            # Reward and done at the first timestep of the sampled window
            data['reward'] = np.array(sample['reward'][0], dtype=np.float32)
            data['done'] = np.array(sample['done'][0], dtype=np.float32)

            # next_obs: load the obs window starting 1 step ahead of the current window
            buffer_start_idx, _, _, _ = self.sampler.indices[idx]
            total_len = len(self.replay_buffer['state'])
            next_idx = min(buffer_start_idx + 1, total_len - 1)
            next_end_idx = min(next_idx + self.horizon, total_len)

            next_state = self.replay_buffer['state'][next_idx:next_end_idx].astype(np.float32)
            next_pc = self.replay_buffer['point_cloud'][next_idx:next_end_idx].astype(np.float32)

            # Pad if we're near the end of the buffer
            actual_len = len(next_state)
            if actual_len < self.horizon:
                pad_len = self.horizon - actual_len
                next_state = np.concatenate(
                    [next_state, np.tile(next_state[-1:], (pad_len,) + (1,) * (next_state.ndim - 1))], axis=0)
                next_pc = np.concatenate(
                    [next_pc, np.tile(next_pc[-1:], (pad_len,) + (1,) * (next_pc.ndim - 1))], axis=0)

            data['next_obs'] = {
                'agent_pos': next_state,
                'point_cloud': next_pc,
            }

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

