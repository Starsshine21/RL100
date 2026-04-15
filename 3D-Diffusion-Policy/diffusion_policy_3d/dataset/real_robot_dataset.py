from typing import Dict, List, Optional

import copy
import numpy as np
import torch
import zarr

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


def _is_low_dim_type(obs_type: str) -> bool:
    return str(obs_type).startswith("low_dim")


def _resolve_obs_keys_from_config(
    obs_mode,
    available_keys,
    use_mask=None,
    use_right_cam_img=None,
):
    available_keys = list(available_keys)
    if obs_mode is None and use_mask is None and use_right_cam_img is None:
        return available_keys
    if obs_mode is None or (isinstance(obs_mode, str) and obs_mode.strip().lower() in {"", "auto", "none", "null"}):
        use_mask = bool(use_mask)
        use_right_cam_img = bool(use_right_cam_img)
        requested = ["right_state", "rgbm" if use_mask else "head_rgb"]
        if use_right_cam_img:
            requested.append("right_cam_img")
    elif isinstance(obs_mode, str):
        mode = obs_mode.strip().lower()
        presets = {
            "all": available_keys,
            "full": ["right_state", "rgbm", "right_cam_img"],
            "rgbm": ["right_state", "rgbm"],
            "state_rgbm": ["right_state", "rgbm"],
            "head_rgb": ["right_state", "head_rgb"],
            "head_rgb_only": ["right_state", "head_rgb"],
            "state_head_rgb": ["right_state", "head_rgb"],
            "head_rgb_right_cam": ["right_state", "head_rgb", "right_cam_img"],
            "dual_rgb": ["right_state", "head_rgb", "right_cam_img"],
            "state": ["right_state"],
            "state_only": ["right_state"],
        }
        if mode in presets:
            requested = presets[mode]
        elif "," in mode:
            requested = [key.strip() for key in mode.split(",") if key.strip()]
        else:
            requested = [obs_mode]
    else:
        requested = [str(key) for key in list(obs_mode)]

    missing = [key for key in requested if key not in available_keys]
    if missing:
        raise KeyError(
            f"Requested obs keys {missing} are not present in shape_meta.obs. "
            f"Available keys: {available_keys}"
        )
    return requested


def _derived_obs_source_key(obs_key: str):
    if obs_key == "head_rgb":
        return "rgbm"
    return obs_key


def _derive_obs_value(obs_key: str, source_value: np.ndarray) -> np.ndarray:
    if obs_key == "head_rgb":
        return source_value[..., :3]
    return source_value


def _maybe_scale_image_for_dtype(value: np.ndarray, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer) and value.size > 0:
        if np.issubdtype(value.dtype, np.floating) and float(np.nanmax(value)) <= 1.5:
            value = value * 255.0
        info = np.iinfo(dtype)
        value = np.clip(np.rint(value), info.min, info.max)
    return value.astype(dtype, copy=False)


def _head_rgb_to_rgbm(value: np.ndarray, target_shape) -> np.ndarray:
    value = np.asarray(value)
    target_shape = tuple(target_shape)
    if value.shape[1:] == target_shape:
        return value
    if len(target_shape) != 3 or len(value.shape) != 4:
        return value
    if target_shape[-1] == 4 and value.shape[-1] == 3:
        mask = np.zeros(value.shape[:-1] + (1,), dtype=value.dtype)
        return np.concatenate([value, mask], axis=-1)
    if target_shape[0] == 4 and value.shape[1] == 3:
        mask = np.zeros((value.shape[0], 1) + value.shape[2:], dtype=value.dtype)
        return np.concatenate([value, mask], axis=1)
    return value


class RealRobotDataset(BaseDataset):
    def __init__(
        self,
        zarr_path,
        shape_meta,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        n_action_steps=8,
        n_obs_steps=None,
        gamma=0.99,
        action_key="action",
        reward_key="reward",
        done_key="done",
        replay_buffer_backend="memory",
        replay_buffer_mode="a",
        obs_mode=None,
        use_mask=None,
        use_right_cam_img=None,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        all_obs_meta = dict(shape_meta["obs"])
        self.all_obs_meta = all_obs_meta
        self.all_obs_storage_keys = list(dict.fromkeys(
            _derived_obs_source_key(key)
            for key in all_obs_meta.keys()
        ))
        self.obs_mode = None if obs_mode is None else str(obs_mode)
        self.use_mask = None if use_mask is None else bool(use_mask)
        self.use_right_cam_img = None if use_right_cam_img is None else bool(use_right_cam_img)
        self.obs_keys = _resolve_obs_keys_from_config(
            obs_mode=obs_mode,
            available_keys=all_obs_meta.keys(),
            use_mask=use_mask,
            use_right_cam_img=use_right_cam_img,
        )
        self.obs_meta = {
            key: all_obs_meta[key]
            for key in self.obs_keys
        }
        self.obs_keys = list(self.obs_meta.keys())
        self.obs_type_map = {
            key: str(meta.get("type", "low_dim")).lower()
            for key, meta in self.obs_meta.items()
        }
        self.obs_storage_key_map = {
            key: _derived_obs_source_key(key)
            for key in self.obs_keys
        }
        self.action_key = str(action_key)
        self.reward_key = str(reward_key)
        self.done_key = str(done_key)
        self.replay_buffer_backend = str(replay_buffer_backend).lower()
        self.replay_buffer_mode = str(replay_buffer_mode)
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.max_train_episodes = max_train_episodes
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = None if n_obs_steps is None else int(n_obs_steps)
        self.gamma = float(gamma)
        self.action_chunk_start = self.pad_before
        self.action_chunk_end = self.action_chunk_start + self.n_action_steps
        self.obs_window_length = (
            self.horizon if self.n_obs_steps is None
            else min(self.horizon, self.n_obs_steps)
        )

        zarr_store = zarr.open(zarr_path, "r")
        available_keys = list(zarr_store["data"].keys())
        obs_storage_keys = list(dict.fromkeys(self.obs_storage_key_map.values()))
        required_keys = [self.action_key] + obs_storage_keys
        missing_keys = [key for key in required_keys if key not in available_keys]
        if missing_keys:
            raise KeyError(
                f"Missing keys in {zarr_path}: {missing_keys}. Available keys: {available_keys}"
            )

        self.has_rl_data = self.reward_key in available_keys and self.done_key in available_keys
        keys_to_load = required_keys.copy()
        if self.has_rl_data:
            keys_to_load.extend([self.reward_key, self.done_key])
        self.sample_keys = list(keys_to_load)

        if self.replay_buffer_backend in {"memory", "copy", "numpy"}:
            self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys_to_load)
        elif self.replay_buffer_backend in {"zarr", "disk", "on_disk"}:
            self.replay_buffer = ReplayBuffer.create_from_path(
                zarr_path,
                mode=self.replay_buffer_mode,
            )
        else:
            raise ValueError(
                f"Unsupported replay_buffer_backend={replay_buffer_backend}. "
                "Expected 'memory' or 'zarr'."
            )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=self.seed,
        )
        self.val_mask = np.asarray(val_mask, dtype=bool)
        self.il_train_mask = np.asarray(train_mask, dtype=bool).copy()
        self._rebuild_sampler(train_mask)
        self._refresh_episode_boundaries()

    def _get_obs_value_from_sample(self, sample: Dict[str, np.ndarray], obs_key: str) -> np.ndarray:
        storage_key = self.obs_storage_key_map.get(obs_key, obs_key)
        if storage_key not in sample:
            raise KeyError(
                f"Sample is missing storage key '{storage_key}' for obs '{obs_key}'. "
                f"Available keys: {list(sample.keys())}"
            )
        return _derive_obs_value(obs_key, sample[storage_key])

    def _read_obs_slice(self, obs_key: str, start: int, end: int) -> np.ndarray:
        storage_key = self.obs_storage_key_map.get(obs_key, obs_key)
        value = self.replay_buffer[storage_key][start:end]
        return _derive_obs_value(obs_key, value)

    def _refresh_episode_boundaries(self):
        self.episode_ends = self.replay_buffer.episode_ends[:]
        if len(self.episode_ends) == 0:
            self.episode_starts = np.zeros((0,), dtype=np.int64)
        else:
            self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]]).astype(
                np.int64, copy=False
            )

    def _rebuild_sampler(self, episode_mask: np.ndarray):
        episode_mask = np.asarray(episode_mask, dtype=bool)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self.sample_keys,
            episode_mask=episode_mask,
        )
        self.train_mask = episode_mask

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        episode_mask = np.asarray(
            getattr(self, "val_mask", np.zeros(self.replay_buffer.n_episodes, dtype=bool)),
            dtype=bool,
        )
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self.sample_keys,
            episode_mask=episode_mask,
        )
        val_set.train_mask = episode_mask
        val_set.val_mask = episode_mask
        return val_set

    def get_il_training_dataset(self):
        il_set = copy.copy(self)
        episode_mask = np.asarray(
            getattr(
                self,
                "il_train_mask",
                getattr(self, "train_mask", np.ones(self.replay_buffer.n_episodes, dtype=bool)),
            ),
            dtype=bool,
        )
        il_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self.sample_keys,
            episode_mask=episode_mask,
        )
        il_set.train_mask = episode_mask
        il_set.il_train_mask = episode_mask
        il_set.val_mask = np.asarray(
            getattr(self, "val_mask", np.zeros(self.replay_buffer.n_episodes, dtype=bool)),
            dtype=bool,
        )
        return il_set

    def get_normalizer(self, mode="limits", **kwargs):
        fit_data = {
            "action": self.replay_buffer[self.action_key],
        }
        identity_keys = []
        for key in self.obs_keys:
            obs_type = self.obs_type_map[key]
            if obs_type == "rgb":
                identity_keys.append(key)
            else:
                fit_data[key] = self.replay_buffer[key]

        normalizer = LinearNormalizer()
        if fit_data:
            normalizer.fit(data=fit_data, last_n_dims=1, mode=mode, **kwargs)
        for key in identity_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs = {
            key: self._get_obs_value_from_sample(sample, key).astype(np.float32)
            for key in self.obs_keys
        }
        data = {
            "obs": {
                key: value[:self.obs_window_length].copy()
                for key, value in obs.items()
            },
            "action": sample[self.action_key].astype(np.float32),
        }
        return data

    def build_obs_window_dict(
        self,
        decision_start_idx: int,
        episode_start: int,
        episode_end: int,
    ) -> Dict[str, np.ndarray]:
        desired_start = decision_start_idx - self.pad_before
        desired_end = desired_start + self.obs_window_length

        buffer_start = max(desired_start, episode_start)
        buffer_end = min(desired_end, episode_end)
        sample_start_idx = buffer_start - desired_start
        sample_end_idx = sample_start_idx + (buffer_end - buffer_start)

        obs_window = {}
        for key in self.obs_keys:
            sample = self._read_obs_slice(key, buffer_start, buffer_end).astype(np.float32)
            padded = np.zeros((self.obs_window_length,) + sample.shape[1:], dtype=np.float32)
            if sample_start_idx > 0:
                padded[:sample_start_idx] = sample[0]
            if sample_end_idx < self.obs_window_length:
                padded[sample_end_idx:] = sample[-1]
            padded[sample_start_idx:sample_end_idx] = sample
            obs_window[key] = padded
        return obs_window

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        if self.has_rl_data:
            nc = self.n_action_steps
            chunk_start = self.action_chunk_start
            buffer_start_idx, _, sample_start_idx, _ = self.sampler.indices[idx]
            decision_start_idx = buffer_start_idx + (chunk_start - sample_start_idx)
            decision_start_idx = max(int(decision_start_idx), 0)

            episode_idx = int(np.searchsorted(self.episode_ends, decision_start_idx, side="right"))
            episode_idx = min(max(episode_idx, 0), len(self.episode_ends) - 1)
            episode_start = int(self.episode_starts[episode_idx])
            episode_end = int(self.episode_ends[episode_idx])
            decision_start_idx = min(max(decision_start_idx, episode_start), episode_end - 1)

            chunk_buffer_end = min(decision_start_idx + nc, episode_end)
            executed_action_steps = max(int(chunk_buffer_end - decision_start_idx), 1)
            raw_rewards = self.replay_buffer[self.reward_key][decision_start_idx:chunk_buffer_end].astype(
                np.float32
            )
            discount = np.array([self.gamma ** j for j in range(len(raw_rewards))], dtype=np.float32)
            data["reward"] = np.array(float(np.dot(discount, raw_rewards)), dtype=np.float32)
            data["executed_action_steps"] = np.array(executed_action_steps, dtype=np.int64)

            raw_dones = self.replay_buffer[self.done_key][decision_start_idx:chunk_buffer_end].astype(
                np.float32
            )
            data["done"] = np.array(
                raw_dones.any() if len(raw_dones) > 0 else 0.0,
                dtype=np.float32,
            )

            next_decision_start_idx = min(decision_start_idx + nc, episode_end - 1)
            data["next_obs"] = self.build_obs_window_dict(
                decision_start_idx=next_decision_start_idx,
                episode_start=episode_start,
                episode_end=episode_end,
            )

        return dict_apply(data, torch.from_numpy)

    def merge_episodes(self, episodes: List[dict], il_episode_mask_new=None) -> int:
        if not episodes:
            return 0

        n_new_steps = 0
        n_old_episodes = int(self.replay_buffer.n_episodes)
        old_train_mask = np.asarray(
            getattr(self, "train_mask", np.ones(n_old_episodes, dtype=bool)),
            dtype=bool,
        )
        old_val_mask = np.asarray(
            getattr(self, "val_mask", np.zeros(n_old_episodes, dtype=bool)),
            dtype=bool,
        )
        old_il_train_mask = np.asarray(
            getattr(self, "il_train_mask", old_train_mask),
            dtype=bool,
        )

        target_shapes = {
            self.action_key: tuple(self.replay_buffer[self.action_key].shape[1:])
        }
        target_dtypes = {
            self.action_key: self.replay_buffer[self.action_key].dtype
        }
        active_storage_keys = list(dict.fromkeys(self.obs_storage_key_map.values()))
        obs_storage_keys_to_write = [
            key for key in self.all_obs_storage_keys
            if key in self.replay_buffer or key in active_storage_keys
        ]
        for key in active_storage_keys:
            if key not in obs_storage_keys_to_write:
                obs_storage_keys_to_write.append(key)
        for key in obs_storage_keys_to_write:
            if key in self.replay_buffer:
                target_shapes[key] = tuple(self.replay_buffer[key].shape[1:])
                target_dtypes[key] = self.replay_buffer[key].dtype
            elif key in self.all_obs_meta:
                target_shapes[key] = tuple(self.all_obs_meta[key]["shape"])
                target_dtypes[key] = np.float32
            else:
                raise KeyError(f"No target shape found for storage key '{key}'.")

        for ep in episodes:
            ep_data = {}
            action_value = np.asarray(ep[self.action_key], dtype=np.float32)
            if action_value.shape[1:] != target_shapes[self.action_key]:
                raise ValueError(
                    f"Shape mismatch for key '{self.action_key}': "
                    f"{action_value.shape[1:]} vs {target_shapes[self.action_key]}"
                )
            ep_data[self.action_key] = action_value.astype(target_dtypes[self.action_key], copy=False)
            length = len(ep_data[self.action_key])

            for key in obs_storage_keys_to_write:
                found = False
                if key in ep:
                    value = np.asarray(ep[key])
                    found = True
                else:
                    value = None
                    for obs_key, storage_key in self.obs_storage_key_map.items():
                        if storage_key == key and obs_key in ep:
                            value = np.asarray(ep[obs_key])
                            if key == "rgbm" and obs_key == "head_rgb":
                                value = _head_rgb_to_rgbm(value, target_shapes[key])
                            found = True
                            break

                if not found:
                    if key in active_storage_keys:
                        raise KeyError(
                            f"Episode is missing required storage key '{key}'. "
                            f"Available keys: {list(ep.keys())}"
                        )
                    value = np.zeros((length,) + target_shapes[key], dtype=target_dtypes[key])

                if key == "rgbm":
                    value = _head_rgb_to_rgbm(value, target_shapes[key])
                if value.shape[1:] != target_shapes[key]:
                    raise ValueError(
                        f"Shape mismatch for key '{key}': {value.shape[1:]} vs {target_shapes[key]}"
                    )
                ep_data[key] = _maybe_scale_image_for_dtype(value, target_dtypes[key])

            reward = np.asarray(
                ep.get(self.reward_key, np.zeros(length, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            done = np.asarray(
                ep.get(self.done_key, np.zeros(length, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)

            if not all(len(ep_data[key]) == length for key in obs_storage_keys_to_write):
                raise ValueError("Episode length mismatch among observation keys.")
            if len(reward) != length or len(done) != length:
                raise ValueError("Episode length mismatch among action/reward/done.")

            ep_data[self.reward_key] = reward
            ep_data[self.done_key] = done
            self.replay_buffer.add_episode(ep_data)
            n_new_steps += length

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
