from typing import Dict
import torch
import numpy as np
import copy
import zarr
from termcolor import cprint
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
            n_action_steps=8,
            gamma=0.99,
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
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.val_mask = np.asarray(val_mask, dtype=bool)

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_action_steps = n_action_steps
        self.gamma = gamma
        self.action_chunk_start = pad_before
        self.action_chunk_end = self.action_chunk_start + n_action_steps
        self._rebuild_sampler(train_mask)
        self.il_train_mask = np.asarray(train_mask, dtype=bool).copy()
        self._refresh_episode_boundaries()
        self._shape_mismatch_warned = False

    def _refresh_episode_boundaries(self):
        self.episode_ends = self.replay_buffer.episode_ends[:]
        if len(self.episode_ends) == 0:
            self.episode_starts = np.zeros((0,), dtype=np.int64)
        else:
            self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]]).astype(np.int64, copy=False)

    def _rebuild_sampler(self, episode_mask: np.ndarray):
        episode_mask = np.asarray(episode_mask, dtype=bool)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=episode_mask,
        )
        self.train_mask = episode_mask

    @staticmethod
    def _align_point_cloud_shape(point_cloud: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Align point cloud array to replay buffer shape [T, N_target, C_target].

        - Channel mismatch: crop or zero-pad last dim.
        - Point-count mismatch: deterministic resample (linspace index) or repeat.
        """
        if point_cloud.ndim != 3:
            raise ValueError(f"point_cloud must be rank-3 [T, N, C], got shape={point_cloud.shape}")

        target_n, target_c = target_shape
        pc = point_cloud

        # Align channels first.
        cur_c = pc.shape[-1]
        if cur_c > target_c:
            pc = pc[..., :target_c]
        elif cur_c < target_c:
            pad_width = ((0, 0), (0, 0), (0, target_c - cur_c))
            pc = np.pad(pc, pad_width=pad_width, mode='constant')

        # Align number of points.
        cur_n = pc.shape[1]
        if cur_n > target_n:
            # Deterministic uniform index selection to keep behavior reproducible.
            indices = np.linspace(0, cur_n - 1, target_n, dtype=np.int64)
            pc = pc[:, indices, :]
        elif cur_n < target_n:
            repeat_factor = (target_n + cur_n - 1) // cur_n
            pc = np.tile(pc, (1, repeat_factor, 1))[:, :target_n, :]

        return pc.astype(np.float32, copy=False)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        episode_mask = np.asarray(getattr(self, 'val_mask', np.zeros(self.replay_buffer.n_episodes, dtype=bool)), dtype=bool)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=episode_mask
            )
        val_set.train_mask = episode_mask
        val_set.val_mask = episode_mask
        return val_set

    def get_il_training_dataset(self):
        il_set = copy.copy(self)
        episode_mask = np.asarray(
            getattr(self, 'il_train_mask', getattr(self, 'train_mask', np.ones(self.replay_buffer.n_episodes, dtype=bool))),
            dtype=bool,
        )
        il_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=episode_mask
        )
        il_set.train_mask = episode_mask
        il_set.il_train_mask = episode_mask
        il_set.val_mask = np.asarray(getattr(self, 'val_mask', np.zeros(self.replay_buffer.n_episodes, dtype=bool)), dtype=bool)
        return il_set

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

    def _build_obs_window(self, decision_start_idx: int, episode_start: int, episode_end: int):
        """
        Build a horizon-length observation sequence whose first n_obs_steps align
        with the decision state at ``decision_start_idx``.
        """
        desired_start = decision_start_idx - self.pad_before
        desired_end = desired_start + self.horizon

        buffer_start = max(desired_start, episode_start)
        buffer_end = min(desired_end, episode_end)
        sample_start_idx = buffer_start - desired_start
        sample_end_idx = sample_start_idx + (buffer_end - buffer_start)

        state_sample = self.replay_buffer['state'][buffer_start:buffer_end].astype(np.float32)
        pc_sample = self.replay_buffer['point_cloud'][buffer_start:buffer_end].astype(np.float32)

        next_state = np.zeros((self.horizon,) + state_sample.shape[1:], dtype=np.float32)
        next_pc = np.zeros((self.horizon,) + pc_sample.shape[1:], dtype=np.float32)

        if sample_start_idx > 0:
            next_state[:sample_start_idx] = state_sample[0]
            next_pc[:sample_start_idx] = pc_sample[0]
        if sample_end_idx < self.horizon:
            next_state[sample_end_idx:] = state_sample[-1]
            next_pc[sample_end_idx:] = pc_sample[-1]

        next_state[sample_start_idx:sample_end_idx] = state_sample
        next_pc[sample_start_idx:sample_end_idx] = pc_sample

        return next_state, next_pc
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        if self.has_rl_data:
            nc = self.n_action_steps
            chunk_start = self.action_chunk_start
            buffer_start_idx, _, sample_start_idx, _ = self.sampler.indices[idx]
            decision_start_idx = buffer_start_idx + (chunk_start - sample_start_idx)
            decision_start_idx = max(int(decision_start_idx), 0)

            episode_idx = int(np.searchsorted(self.episode_ends, decision_start_idx, side='right'))
            episode_idx = min(max(episode_idx, 0), len(self.episode_ends) - 1)
            episode_start = int(self.episode_starts[episode_idx])
            episode_end = int(self.episode_ends[episode_idx])
            decision_start_idx = min(max(decision_start_idx, episode_start), episode_end - 1)

            # Paper: R_chunk = Σ_{j=0}^{n_c-1} γ^j · R_{t+j}.
            # Compute it from the REAL replay-buffer interval rather than the padded
            # sampled sequence; otherwise the terminal reward is duplicated near the
            # episode tail.
            chunk_buffer_end = min(decision_start_idx + nc, episode_end)
            raw_rewards = self.replay_buffer['reward'][decision_start_idx:chunk_buffer_end].astype(np.float32)
            discount = np.array([self.gamma ** j for j in range(len(raw_rewards))], dtype=np.float32)
            chunk_reward = float(np.dot(discount, raw_rewards))
            data['reward'] = np.array(chunk_reward, dtype=np.float32)

            raw_dones = self.replay_buffer['done'][decision_start_idx:chunk_buffer_end].astype(np.float32)
            data['done'] = np.array(raw_dones.any() if len(raw_dones) > 0 else 0.0, dtype=np.float32)

            next_decision_start_idx = min(decision_start_idx + nc, episode_end - 1)
            next_state, next_pc = self._build_obs_window(
                decision_start_idx=next_decision_start_idx,
                episode_start=episode_start,
                episode_end=episode_end,
            )

            data['next_obs'] = {
                'agent_pos': next_state,
                'point_cloud': next_pc,
            }

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def merge_episodes(self, episodes: list, il_episode_mask_new=None) -> int:
        """
        Merge newly collected episodes into the replay buffer in-place.

        Each episode is a dict with numpy arrays of shape [T, ...]:
            state, action, point_cloud, reward, done

        The SequenceSampler is rebuilt after merging so that the new data
        is immediately available for training.

        Args:
            episodes: list of episode dicts from MetaworldRunner.run_and_collect()

        Returns:
            n_new_steps: total number of new timesteps added
        """
        if not episodes:
            return 0

        n_new_steps = 0
        n_old_episodes = int(self.replay_buffer.n_episodes)
        old_train_mask = np.asarray(getattr(self, 'train_mask', np.ones(n_old_episodes, dtype=bool)), dtype=bool)
        old_val_mask = np.asarray(getattr(self, 'val_mask', np.zeros(n_old_episodes, dtype=bool)), dtype=bool)
        old_il_train_mask = np.asarray(getattr(self, 'il_train_mask', old_train_mask), dtype=bool)
        target_state_shape = tuple(self.replay_buffer['state'].shape[1:])
        target_action_shape = tuple(self.replay_buffer['action'].shape[1:])
        target_pc_shape = tuple(self.replay_buffer['point_cloud'].shape[1:])

        for ep in episodes:
            state = np.asarray(ep['state'], dtype=np.float32)
            action = np.asarray(ep['action'], dtype=np.float32)
            point_cloud = np.asarray(ep['point_cloud'], dtype=np.float32)
            reward = np.asarray(
                ep.get('reward', np.zeros(len(state), dtype=np.float32)),
                dtype=np.float32
            ).reshape(-1)
            done = np.asarray(
                ep.get('done', np.zeros(len(state), dtype=np.float32)),
                dtype=np.float32
            ).reshape(-1)

            if state.shape[1:] != target_state_shape:
                raise ValueError(
                    f"State shape mismatch: episode {state.shape[1:]} vs replay_buffer {target_state_shape}"
                )
            if action.shape[1:] != target_action_shape:
                raise ValueError(
                    f"Action shape mismatch: episode {action.shape[1:]} vs replay_buffer {target_action_shape}"
                )
            if point_cloud.shape[1:] != target_pc_shape:
                original_shape = point_cloud.shape
                point_cloud = self._align_point_cloud_shape(point_cloud, target_pc_shape)
                if not self._shape_mismatch_warned:
                    cprint(
                        f"[MetaworldDataset] point_cloud shape mismatch detected. "
                        f"Auto-aligned from {original_shape[1:]} to {point_cloud.shape[1:]} "
                        f"to match replay buffer.",
                        "yellow"
                    )
                    self._shape_mismatch_warned = True

            T = len(state)
            if not (len(action) == len(point_cloud) == len(reward) == len(done) == T):
                raise ValueError(
                    "Episode length mismatch among fields: "
                    f"state={len(state)}, action={len(action)}, point_cloud={len(point_cloud)}, "
                    f"reward={len(reward)}, done={len(done)}"
                )

            # Ensure reward/done exist (add zeros if this is demo data without RL labels)
            ep_data = {
                'state': state,
                'action': action,
                'point_cloud': point_cloud,
                'reward': reward,
                'done': done,
            }

            self.replay_buffer.add_episode(ep_data)
            n_new_steps += len(ep['state'])

        # Flag that RL data is now present
        self.has_rl_data = True
        self._refresh_episode_boundaries()

        n_total = self.replay_buffer.n_episodes
        n_new_episodes = n_total - n_old_episodes
        if il_episode_mask_new is None:
            il_episode_mask_new = np.ones(n_new_episodes, dtype=bool)
        else:
            il_episode_mask_new = np.asarray(il_episode_mask_new, dtype=bool)
            if len(il_episode_mask_new) != n_new_episodes:
                raise ValueError(
                    f"il_episode_mask_new length mismatch: expected {n_new_episodes}, got {len(il_episode_mask_new)}"
                )
        if len(old_train_mask) == n_old_episodes:
            # Preserve the original train/val split and append newly collected
            # episodes to the training set.
            episode_mask = np.concatenate(
                [old_train_mask, np.ones(n_new_episodes, dtype=bool)],
                axis=0,
            )
        else:
            episode_mask = np.ones(n_total, dtype=bool)
        self._rebuild_sampler(episode_mask)
        if len(old_val_mask) == n_old_episodes:
            self.val_mask = np.concatenate(
                [old_val_mask, np.zeros(n_new_episodes, dtype=bool)],
                axis=0,
            )
        else:
            self.val_mask = np.zeros(n_total, dtype=bool)
        if len(old_il_train_mask) == n_old_episodes:
            self.il_train_mask = np.concatenate(
                [old_il_train_mask, il_episode_mask_new],
                axis=0,
            )
        else:
            self.il_train_mask = episode_mask.copy()

        return n_new_steps
