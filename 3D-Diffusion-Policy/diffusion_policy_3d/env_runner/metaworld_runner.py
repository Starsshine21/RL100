import wandb
import numpy as np
import torch
import collections
import tqdm
from diffusion_policy_3d.env import MetaWorldEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

class MetaworldRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512
                 ):
        super().__init__(output_dir)
        self.task_name = task_name


        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy, save_video=False):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step(action)


                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        if save_video:
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()
        videos = None

        return log_data

    def run_and_collect(self, policy: BasePolicy, num_episodes: int,
                        reward_type: str = 'sparse',
                        gamma: float = 0.99):
        """
        Roll out policy and collect trajectory data for dataset merging.

        Args:
            reward_type: 'sparse' — reward=1 at last step if success, 0 elsewhere
                         'dense'  — use MetaWorld env shaped reward each step

        Returns:
            metrics  : same dict as run()
            episodes : list of per-episode dicts (numpy arrays)
        """
        device = policy.device
        # Unwrap MultiStepWrapper for single-step transitions; we will
        # execute the action chunk manually to keep replay data per-step.
        env = self.env.env

        all_env_rewards = []
        all_rl_rewards = []
        all_success_rates = []
        collected_episodes = []

        for _ in tqdm.tqdm(
            range(num_episodes),
            desc=f"Collect in {self.task_name}",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        ):
            obs = env.reset()
            policy.reset()

            done = False
            traj_env_reward = 0
            is_success = False

            ep_state, ep_action, ep_pc, ep_reward, ep_done = [], [], [], [], []
            decision_steps = []

            # Maintain a rolling window of observations for policy input.
            obs_queue = collections.deque(maxlen=self.n_obs_steps)
            init_obs = {
                'agent_pos': obs['agent_pos'].copy(),
                'point_cloud': obs['point_cloud'].copy(),
            }
            for _ in range(self.n_obs_steps):
                obs_queue.append(init_obs)

            step_count = 0
            while not done and step_count < self.max_steps:
                # Build stacked obs for the policy: [n_obs_steps, ...]
                pc_stack = np.stack([o['point_cloud'] for o in obs_queue], axis=0)
                state_stack = np.stack([o['agent_pos'] for o in obs_queue], axis=0)
                obs_dict_input = {
                    'point_cloud': torch.from_numpy(pc_stack).unsqueeze(0).to(device),
                    'agent_pos':   torch.from_numpy(state_stack).unsqueeze(0).to(device),
                }

                with torch.no_grad():
                    if hasattr(policy, 'predict_action_with_trajectory'):
                        action_dict = policy.predict_action_with_trajectory(obs_dict_input)
                    else:
                        action_dict = policy.predict_action(obs_dict_input)

                np_action = action_dict['action'].squeeze(0).detach().cpu().numpy()
                chunk_step_rewards = []
                chunk_done = False
                decision_obs = {
                    'point_cloud': pc_stack.astype(np.float32),
                    'agent_pos': state_stack.astype(np.float32),
                }
                trajectory = None
                log_probs_old = None
                if 'trajectory' in action_dict and 'log_probs_old' in action_dict:
                    trajectory = np.stack(
                        [traj.squeeze(0).detach().cpu().numpy().astype(np.float32)
                         for traj in action_dict['trajectory']],
                        axis=0,
                    )
                    log_probs_old = np.stack(
                        [log_prob.squeeze(0).detach().cpu().numpy().astype(np.float32)
                         for log_prob in action_dict['log_probs_old']],
                        axis=0,
                    )

                # Execute the chunk open-loop but record per-step transitions.
                for act in np_action:
                    if done or step_count >= self.max_steps:
                        break

                    cur_state = obs['agent_pos']
                    cur_pc = obs['point_cloud']

                    obs, reward, done, info = env.step(act)
                    reward = float(reward)
                    done = bool(np.all(done))
                    chunk_done = chunk_done or done

                    traj_env_reward += reward
                    if isinstance(info, dict) and 'success' in info:
                        try:
                            is_success = is_success or bool(max(info['success']))
                        except Exception:
                            is_success = is_success or bool(info['success'])

                    ep_state.append(cur_state.astype(np.float32))
                    ep_action.append(act.astype(np.float32))
                    ep_pc.append(cur_pc.astype(np.float32))
                    if reward_type == 'dense':
                        ep_reward.append(np.float32(reward))
                        chunk_step_rewards.append(np.float32(reward))
                    else:
                        ep_reward.append(np.float32(0.0))
                        chunk_step_rewards.append(np.float32(0.0))
                    ep_done.append(np.float32(done))

                    obs_queue.append({
                        'agent_pos': obs['agent_pos'].copy(),
                        'point_cloud': obs['point_cloud'].copy(),
                    })
                    step_count += 1
                    if step_count >= self.max_steps and not done:
                        done = True
                        chunk_done = True
                        ep_done[-1] = np.float32(1.0)

                if reward_type == 'sparse' and chunk_done and is_success and chunk_step_rewards:
                    chunk_step_rewards[-1] = np.float32(1.0)

                if np_action.shape[0] > 0:
                    next_pc_stack = np.stack([o['point_cloud'] for o in obs_queue], axis=0).astype(np.float32)
                    next_state_stack = np.stack([o['agent_pos'] for o in obs_queue], axis=0).astype(np.float32)
                    chunk_discount = np.array(
                        [gamma ** j for j in range(len(chunk_step_rewards))],
                        dtype=np.float32,
                    )
                    decision_reward = float(np.dot(chunk_discount, np.asarray(chunk_step_rewards, dtype=np.float32)))
                    decision_step = {
                        'obs': decision_obs,
                        'action': np_action.astype(np.float32),
                        'reward': np.float32(decision_reward),
                        'done': np.float32(chunk_done),
                        'next_obs': {
                            'point_cloud': next_pc_stack,
                            'agent_pos': next_state_stack,
                        },
                    }
                    if trajectory is not None and log_probs_old is not None:
                        decision_step['trajectory'] = trajectory
                        decision_step['log_probs_old'] = log_probs_old
                    decision_steps.append(decision_step)

            # Sparse: 1 at last step only if successful
            if reward_type == 'sparse' and is_success and len(ep_reward) > 0:
                ep_reward[-1] = np.float32(1.0)

            traj_rl_reward = float(np.sum(ep_reward)) if len(ep_reward) > 0 else 0.0

            all_success_rates.append(is_success)
            all_env_rewards.append(traj_env_reward)
            all_rl_rewards.append(traj_rl_reward)

            if len(ep_state) > 0:
                collected_episodes.append({
                    'state':       np.stack(ep_state,  axis=0),
                    'action':      np.stack(ep_action, axis=0),
                    'point_cloud': np.stack(ep_pc,     axis=0),
                    'reward':      np.array(ep_reward, dtype=np.float32),
                    'done':        np.array(ep_done,   dtype=np.float32),
                    'success':     bool(is_success),
                    'decision_steps': decision_steps,
                })

        metrics = {
            'mean_traj_rewards':  float(np.mean(all_env_rewards)),
            'mean_env_rewards':   float(np.mean(all_env_rewards)),
            'mean_rl_rewards':    float(np.mean(all_rl_rewards)),
            'mean_success_rates': float(np.mean(all_success_rates)),
            'test_mean_score':    float(np.mean(all_success_rates)),
            'n_episodes':         len(collected_episodes),
            'n_steps':            sum(len(e['state']) for e in collected_episodes),
        }

        cprint(f"[Collect] {len(collected_episodes)} episodes, "
               f"success={metrics['mean_success_rates']:.3f}, "
               f"env_return={metrics['mean_env_rewards']:.2f}, "
               f"rl_reward={metrics['mean_rl_rewards']:.2f}, "
               f"steps={metrics['n_steps']}", 'cyan')

        _ = env.reset()
        return metrics, collected_episodes
