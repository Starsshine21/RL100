import collections
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import tqdm
from termcolor import cprint

from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.policy.base_policy import BasePolicy


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


def _derive_obs_from_raw(obs: Dict, key: str):
    if key in obs:
        return obs[key]
    if key == "head_rgb" and "rgbm" in obs:
        return np.asarray(obs["rgbm"])[..., :3]
    raise KeyError(f"Observation key '{key}' missing from env output. Available keys: {list(obs.keys())}")


def _storage_key_for_obs(obs_key: str) -> str:
    if obs_key == "head_rgb":
        return "rgbm"
    return obs_key


def _image_shape_is_chw(shape) -> bool:
    shape = tuple(shape)
    if len(shape) != 3:
        raise ValueError(f"RGB observation must be rank-3, got shape={shape}")
    c_first = shape[0] <= 4 and shape[-1] > 4
    c_last = shape[-1] <= 4 and shape[0] > 4
    if c_first:
        return True
    if c_last:
        return False
    return shape[0] <= 4


class RealRobotRunner(BaseRunner):
    def __init__(
        self,
        output_dir,
        env=None,
        shape_meta=None,
        eval_episodes=20,
        max_steps=200,
        n_obs_steps=2,
        n_action_steps=8,
        task_name=None,
        fps=10,
        tqdm_interval_sec=5.0,
        episode_end_mode="env_or_manual_or_max_steps",
        reward_mode="terminal_sparse_manual",
        success_reward=1.0,
        failure_reward=0.0,
        gamma=0.99,
        prompt_after_chunk=True,
        prompt_for_terminal_label=True,
        stop_on_keyboard_interrupt=True,
        obs_mode=None,
        use_mask=None,
        use_right_cam_img=None,
        seed=None,
    ):
        super().__init__(output_dir)
        self.env = env
        self.shape_meta = shape_meta
        all_obs_meta = dict(shape_meta["obs"]) if shape_meta is not None else {}
        self.all_obs_meta = all_obs_meta
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
        self.obs_type_map = {
            key: str(meta.get("type", "low_dim")).lower()
            for key, meta in self.obs_meta.items()
        }
        self.storage_obs_keys = list(dict.fromkeys(
            _storage_key_for_obs(key)
            for key in all_obs_meta.keys()
            if _storage_key_for_obs(key) in all_obs_meta
        ))
        self.task_name = task_name
        self.eval_episodes = int(eval_episodes)
        self.max_steps = int(max_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.fps = int(fps)
        self.tqdm_interval_sec = float(tqdm_interval_sec)
        self.episode_end_mode = str(episode_end_mode).lower()
        self.reward_mode = str(reward_mode).lower()
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.gamma = float(gamma)
        self.prompt_after_chunk = bool(prompt_after_chunk)
        self.prompt_for_terminal_label = bool(prompt_for_terminal_label)
        self.stop_on_keyboard_interrupt = bool(stop_on_keyboard_interrupt)
        self.seed = seed

        valid_episode_modes = {
            "env",
            "manual",
            "max_steps",
            "env_or_manual",
            "env_or_max_steps",
            "manual_or_max_steps",
            "env_or_manual_or_max_steps",
        }
        if self.episode_end_mode not in valid_episode_modes:
            raise ValueError(
                f"Unsupported episode_end_mode='{episode_end_mode}'. "
                f"Expected one of {sorted(valid_episode_modes)}"
            )

        valid_reward_modes = {
            "env",
            "dense_env",
            "terminal_sparse",
            "terminal_sparse_manual",
            "terminal_sparse_env_success",
        }
        if self.reward_mode not in valid_reward_modes:
            raise ValueError(
                f"Unsupported reward_mode='{reward_mode}'. Expected one of {sorted(valid_reward_modes)}"
            )

    def _require_env(self):
        if self.env is None:
            raise RuntimeError(
                "RealRobotRunner requires an instantiated real-robot env. "
                "Set task.env_runner.env._target_=... in the Hydra config."
            )
        return self.env

    def _uses_env_done(self) -> bool:
        return "env" in self.episode_end_mode

    def _uses_manual_done(self) -> bool:
        return "manual" in self.episode_end_mode

    def _uses_max_steps_done(self) -> bool:
        return "max_steps" in self.episode_end_mode

    def _extract_success(self, info: Dict) -> Optional[bool]:
        if not isinstance(info, dict) or "success" not in info:
            return None
        value = info["success"]
        try:
            if isinstance(value, (list, tuple, np.ndarray)):
                return bool(np.max(np.asarray(value)))
            return bool(value)
        except Exception:
            return None

    def _parse_reset_result(self, result):
        if isinstance(result, tuple):
            if len(result) >= 1:
                return result[0]
        return result

    def _parse_step_result(self, result) -> Tuple[Dict, float, bool, Dict]:
        if isinstance(result, tuple):
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                return obs, float(reward), bool(terminated or truncated), dict(info or {})
            if len(result) == 4:
                obs, reward, done, info = result
                return obs, float(reward), bool(done), dict(info or {})
            if len(result) == 2:
                obs, info = result
                return obs, 0.0, False, dict(info or {})
            if len(result) == 1:
                return result[0], 0.0, False, {}
        return result, 0.0, False, {}

    def _scale_rgb_like_array(self, array: np.ndarray, key: str) -> np.ndarray:
        spec_shape = tuple(self.obs_meta[key]["shape"])
        arr = array.astype(np.float32, copy=True)
        if _image_shape_is_chw(spec_shape):
            if arr.shape[0] >= 3 and np.max(arr[:3]) > 1.5:
                arr[:3] = arr[:3] / 255.0
            if arr.shape[0] > 3 and np.max(arr[3:]) > 1.5:
                arr[3:] = arr[3:] / 255.0
        else:
            if arr.shape[-1] >= 3 and np.max(arr[..., :3]) > 1.5:
                arr[..., :3] = arr[..., :3] / 255.0
            if arr.shape[-1] > 3 and np.max(arr[..., 3:]) > 1.5:
                arr[..., 3:] = arr[..., 3:] / 255.0
        return arr

    def _format_obs(self, obs: Dict) -> Dict[str, np.ndarray]:
        if not isinstance(obs, dict):
            raise TypeError(f"Expected env observation dict, got {type(obs)}")
        formatted = {}
        obs_keys = self.obs_keys if self.obs_keys else list(obs.keys())
        for key in obs_keys:
            value = np.asarray(_derive_obs_from_raw(obs, key))
            obs_type = self.obs_type_map.get(key, "low_dim")
            if obs_type == "rgb":
                formatted[key] = self._scale_rgb_like_array(value, key)
            elif obs_type == "point_cloud" or _is_low_dim_type(obs_type):
                formatted[key] = value.astype(np.float32, copy=False)
            else:
                formatted[key] = value.astype(np.float32, copy=False)
        return formatted

    def _extract_storage_obs(self, raw_obs: Dict) -> Dict[str, np.ndarray]:
        if not isinstance(raw_obs, dict):
            raise TypeError(f"Expected raw env observation dict, got {type(raw_obs)}")
        storage = {}
        for key in self.storage_obs_keys:
            if key not in raw_obs:
                continue
            storage[key] = np.asarray(raw_obs[key]).copy()
        return storage

    def _stack_obs_queue(self, obs_queue) -> Dict[str, np.ndarray]:
        return {
            key: np.stack([obs[key] for obs in obs_queue], axis=0).astype(np.float32)
            for key in obs_queue[0].keys()
        }

    def _obs_to_policy_input(self, stacked_obs: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(value).unsqueeze(0).to(device)
            for key, value in stacked_obs.items()
        }

    def _manual_chunk_decision(self) -> Optional[bool]:
        if not (self._uses_manual_done() and self.prompt_after_chunk):
            return None
        while True:
            choice = input(
                "[RealRobotRunner] Episode status after chunk? "
                "[c]ontinue / [1/s]uccess / [0/f]ailure: "
            ).strip().lower()
            if choice in ("", "c", "continue"):
                return None
            if choice in ("1", "s", "success", "y", "yes"):
                return True
            if choice in ("0", "f", "failure", "n", "no"):
                return False
            print("Please answer with c, 1/s, or 0/f.")

    def _manual_terminal_label(self) -> bool:
        while True:
            choice = input(
                "[RealRobotRunner] Mark terminal outcome? "
                "[1/s]uccess / [0/f]ailure: "
            ).strip().lower()
            if choice in ("1", "s", "success", "y", "yes"):
                return True
            if choice in ("0", "f", "failure", "n", "no"):
                return False
            print("Please answer with 1/s or 0/f.")

    def _resolve_terminal(
        self,
        env_done: bool,
        step_count: int,
        info: Dict,
        manual_chunk_label: Optional[bool],
    ) -> Tuple[bool, Optional[bool]]:
        success_label = manual_chunk_label
        done = False

        if manual_chunk_label is not None:
            done = True

        if not done and self._uses_env_done() and env_done:
            done = True

        if not done and step_count >= self.max_steps:
            done = True

        if done and success_label is None:
            info_success = self._extract_success(info)
            if self.reward_mode == "terminal_sparse_env_success":
                success_label = bool(info_success) if info_success is not None else False
            elif self.reward_mode in ("terminal_sparse_manual", "terminal_sparse"):
                if info_success is not None:
                    success_label = bool(info_success)
                elif self.prompt_for_terminal_label:
                    success_label = self._manual_terminal_label()
            elif info_success is not None:
                success_label = bool(info_success)

        return done, success_label

    def _best_effort_stop_env(self, env) -> bool:
        if env is None or not self.stop_on_keyboard_interrupt:
            return False

        stop_targets = [
            ("env", env),
            ("env.robot", getattr(env, "robot", None)),
            ("env.controller", getattr(env, "controller", None)),
        ]
        stop_methods = (
            "emergency_stop",
            "estop",
            "stop",
            "stop_robot",
            "halt",
            "pause",
        )

        for target_name, target in stop_targets:
            if target is None:
                continue
            for method_name in stop_methods:
                method = getattr(target, method_name, None)
                if not callable(method):
                    continue
                try:
                    method()
                    cprint(
                        f"[RealRobotRunner] KeyboardInterrupt -> called {target_name}.{method_name}()",
                        "yellow",
                    )
                    return True
                except Exception as exc:
                    cprint(
                        f"[RealRobotRunner] {target_name}.{method_name}() failed during stop: "
                        f"{type(exc).__name__}: {exc}",
                        "red",
                    )

        cprint(
            "[RealRobotRunner] KeyboardInterrupt received, but no supported stop hook was found on the env.",
            "red",
        )
        return False

    def _rollout_episode(
        self,
        env,
        policy: BasePolicy,
        collect_trajectory: bool,
        reward_mode: str,
        gamma: float,
    ):
        device = policy.device

        try:
            reset_result = env.reset()
            obs_raw = self._parse_reset_result(reset_result)
            obs = self._format_obs(obs_raw)
            policy.reset()

            obs_queue = collections.deque(maxlen=self.n_obs_steps)
            for _ in range(self.n_obs_steps):
                obs_queue.append({key: value.copy() for key, value in obs.items()})

            done = False
            step_count = 0
            traj_env_reward = 0.0
            is_success = False

            ep_obs_storage = {
                key: []
                for key in self._extract_storage_obs(obs_raw).keys()
            }
            ep_action, ep_reward, ep_done = [], [], []
            decision_steps = []

            while not done and step_count < self.max_steps:
                decision_obs = self._stack_obs_queue(obs_queue)
                obs_dict_input = self._obs_to_policy_input(decision_obs, device=device)

                with torch.no_grad():
                    if collect_trajectory and hasattr(policy, "predict_action_with_trajectory"):
                        action_dict = policy.predict_action_with_trajectory(obs_dict_input)
                    else:
                        action_dict = policy.predict_action(obs_dict_input)

                np_action = action_dict["action"].squeeze(0).detach().cpu().numpy()
                chunk_step_rewards = []
                chunk_done = False
                executed_action_steps = 0
                last_info = {}
                trajectory = None
                trajectory_tensors = None

                if "trajectory" in action_dict:
                    trajectory_tensors = [traj.detach().clone() for traj in action_dict["trajectory"]]
                    trajectory = np.stack(
                        [traj.squeeze(0).detach().cpu().numpy().astype(np.float32) for traj in action_dict["trajectory"]],
                        axis=0,
                    )

                for act in np_action:
                    if done or step_count >= self.max_steps:
                        break

                    current_obs = {key: value.copy() for key, value in obs.items()}
                    current_storage_obs = self._extract_storage_obs(obs_raw)
                    step_result = env.step(act)
                    next_obs_raw, reward_env, env_done, info = self._parse_step_result(step_result)
                    next_obs = self._format_obs(next_obs_raw)

                    traj_env_reward += float(reward_env)
                    last_info = info
                    info_success = self._extract_success(info)
                    if info_success is not None:
                        is_success = is_success or bool(info_success)

                    for key, value in current_storage_obs.items():
                        if key not in ep_obs_storage:
                            ep_obs_storage[key] = []
                        ep_obs_storage[key].append(np.asarray(value).copy())
                    ep_action.append(np.asarray(act, dtype=np.float32))
                    if reward_mode in ("env", "dense_env"):
                        step_reward = np.float32(reward_env)
                    else:
                        step_reward = np.float32(0.0)
                    ep_reward.append(step_reward)
                    chunk_step_rewards.append(step_reward)
                    ep_done.append(np.float32(env_done))

                    obs_queue.append({key: value.copy() for key, value in next_obs.items()})
                    obs = next_obs
                    obs_raw = next_obs_raw
                    executed_action_steps += 1
                    step_count += 1

                    if env_done and self._uses_env_done():
                        chunk_done = True
                        done = True
                        break
                    if step_count >= self.max_steps:
                        chunk_done = True
                        done = True
                        break

                if executed_action_steps == 0:
                    break

                manual_chunk_label = self._manual_chunk_decision() if not done else None
                chunk_done, success_label = self._resolve_terminal(
                    env_done=done,
                    step_count=step_count,
                    info=last_info,
                    manual_chunk_label=manual_chunk_label,
                )
                done = chunk_done
                if success_label is not None:
                    is_success = bool(success_label)

                if done:
                    ep_done[-1] = np.float32(1.0)
                    if reward_mode not in ("env", "dense_env"):
                        terminal_reward = self.success_reward if is_success else self.failure_reward
                        ep_reward[-1] = np.float32(terminal_reward)
                        chunk_step_rewards[-1] = np.float32(terminal_reward)

                next_obs_stack = self._stack_obs_queue(obs_queue)
                chunk_discount = np.array(
                    [gamma ** j for j in range(len(chunk_step_rewards))],
                    dtype=np.float32,
                )
                decision_reward = float(np.dot(chunk_discount, np.asarray(chunk_step_rewards, dtype=np.float32)))
                decision_step = {
                    "obs": decision_obs,
                    "action": np_action[:executed_action_steps].astype(np.float32),
                    "reward": np.float32(decision_reward),
                    "done": np.float32(done),
                    "executed_action_steps": np.int64(executed_action_steps),
                    "next_obs": next_obs_stack,
                }
                if trajectory is not None:
                    decision_step["trajectory"] = trajectory
                    log_probs_old = None
                    if trajectory_tensors is not None and hasattr(policy, "compute_trajectory_log_probs"):
                        with torch.no_grad():
                            log_prob_tensors = policy.compute_trajectory_log_probs(
                                obs_dict=obs_dict_input,
                                trajectory=trajectory_tensors,
                                executed_action_steps=torch.as_tensor(
                                    [executed_action_steps],
                                    device=device,
                                    dtype=torch.long,
                                ),
                            )
                        log_probs_old = np.stack(
                            [log_prob.squeeze(0).detach().cpu().numpy().astype(np.float32) for log_prob in log_prob_tensors],
                            axis=0,
                        )
                    elif "log_probs_old" in action_dict:
                        log_probs_old = np.stack(
                            [log_prob.squeeze(0).detach().cpu().numpy().astype(np.float32) for log_prob in action_dict["log_probs_old"]],
                            axis=0,
                        )
                    if log_probs_old is not None:
                        decision_step["log_probs_old"] = log_probs_old
                decision_steps.append(decision_step)
        except KeyboardInterrupt:
            self._best_effort_stop_env(env)
            raise

        episode = None
        if ep_action:
            episode = {
                key: np.stack(values, axis=0)
                for key, values in ep_obs_storage.items()
            }
            episode.update({
                "action": np.stack(ep_action, axis=0),
                "reward": np.asarray(ep_reward, dtype=np.float32),
                "done": np.asarray(ep_done, dtype=np.float32),
                "success": bool(is_success),
                "decision_steps": decision_steps,
            })

        traj_rl_reward = float(np.sum(ep_reward)) if ep_reward else 0.0
        return {
            "success": bool(is_success),
            "env_reward": float(traj_env_reward),
            "rl_reward": traj_rl_reward,
            "steps": len(ep_action),
            "episode": episode,
        }

    def run(self, policy: BasePolicy, save_video=False):
        del save_video
        env = self._require_env()
        all_success_rates = []
        all_env_rewards = []
        all_rl_rewards = []

        for _ in tqdm.tqdm(
            range(self.eval_episodes),
            desc=f"Eval on real robot {self.task_name}",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        ):
            rollout = self._rollout_episode(
                env=env,
                policy=policy,
                collect_trajectory=False,
                reward_mode=self.reward_mode,
                gamma=self.gamma,
            )
            all_success_rates.append(float(rollout["success"]))
            all_env_rewards.append(float(rollout["env_reward"]))
            all_rl_rewards.append(float(rollout["rl_reward"]))

        metrics = {
            "mean_traj_rewards": float(np.mean(all_env_rewards)) if all_env_rewards else 0.0,
            "mean_env_rewards": float(np.mean(all_env_rewards)) if all_env_rewards else 0.0,
            "mean_rl_rewards": float(np.mean(all_rl_rewards)) if all_rl_rewards else 0.0,
            "mean_success_rates": float(np.mean(all_success_rates)) if all_success_rates else 0.0,
            "test_mean_score": float(np.mean(all_success_rates)) if all_success_rates else 0.0,
        }
        cprint(
            f"[RealRobotRunner] eval success={metrics['mean_success_rates']:.3f}, "
            f"env_return={metrics['mean_env_rewards']:.3f}, rl_reward={metrics['mean_rl_rewards']:.3f}",
            "green",
        )
        return metrics

    def run_and_collect(
        self,
        policy: BasePolicy,
        num_episodes: int,
        reward_type: str = "sparse",
        gamma: float = 0.99,
        collect_trajectory: bool = False,
    ):
        env = self._require_env()
        if reward_type == "dense" and self.reward_mode not in ("env", "dense_env"):
            raise ValueError(
                "reward_type=dense requires env-based rewards. "
                "Set task.env_runner.reward_mode=env (or dense_env)."
            )

        effective_reward_mode = self.reward_mode
        if reward_type == "dense":
            effective_reward_mode = "dense_env"

        all_success_rates = []
        all_env_rewards = []
        all_rl_rewards = []
        collected_episodes = []

        for _ in tqdm.tqdm(
            range(num_episodes),
            desc=f"Collect on real robot {self.task_name}",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        ):
            rollout = self._rollout_episode(
                env=env,
                policy=policy,
                collect_trajectory=collect_trajectory,
                reward_mode=effective_reward_mode,
                gamma=gamma,
            )
            all_success_rates.append(float(rollout["success"]))
            all_env_rewards.append(float(rollout["env_reward"]))
            all_rl_rewards.append(float(rollout["rl_reward"]))
            if rollout["episode"] is not None:
                collected_episodes.append(rollout["episode"])

        metrics = {
            "mean_traj_rewards": float(np.mean(all_env_rewards)) if all_env_rewards else 0.0,
            "mean_env_rewards": float(np.mean(all_env_rewards)) if all_env_rewards else 0.0,
            "mean_rl_rewards": float(np.mean(all_rl_rewards)) if all_rl_rewards else 0.0,
            "mean_success_rates": float(np.mean(all_success_rates)) if all_success_rates else 0.0,
            "test_mean_score": float(np.mean(all_success_rates)) if all_success_rates else 0.0,
            "n_episodes": len(collected_episodes),
            "n_steps": sum(len(ep["action"]) for ep in collected_episodes),
        }
        cprint(
            f"[Collect] {len(collected_episodes)} real-robot episodes, "
            f"success={metrics['mean_success_rates']:.3f}, "
            f"env_return={metrics['mean_env_rewards']:.3f}, "
            f"rl_reward={metrics['mean_rl_rewards']:.3f}, "
            f"steps={metrics['n_steps']}",
            "cyan",
        )
        return metrics, collected_episodes
