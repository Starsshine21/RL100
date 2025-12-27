"""
RL100 评估脚本（修改版 - 使用MultiStepWrapper与原始DP3一致）
功能：加载checkpoint并评估策略性能
输出：成功率、平均奖励、episode长度等指标
"""
import sys
import os
import argparse
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_policy_3d.policy.rl100 import RL100
from diffusion_policy_3d.env.metaworld.metaworld_wrapper import MetaWorldEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.common.pytorch_util import dict_apply


class RL100Evaluator:
    """RL100策略评估器（使用MultiStepWrapper）"""

    def __init__(self, checkpoint_path: str, device: str = "cuda:0", render: bool = False, max_steps: int = 500):
        """
        Args:
            checkpoint_path: checkpoint文件路径
            device: 运行设备
            render: 是否渲染环境（可视化）
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.render = render

        print(f"\n{'='*60}")
        print(f"加载checkpoint: {checkpoint_path}")
        print(f"设备: {device}")
        print(f"{'='*60}\n")

        # 加载checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint['config']

        # 获取配置参数
        num_points = self.config.task.dataset.pointcloud_encoder_cfg.get('in_channels_points', 1024)
        self.n_obs_steps = self.config.policy.n_obs_steps
        self.horizon = self.config.policy.horizon  
        self.n_action_predict = self.horizon 
        self.n_action_execute = self.config.policy.n_action_steps      

        print(f"初始化环境（预测{self.n_action_predict}步，执行{self.n_action_execute}步）...")
        raw_env = MetaWorldEnv(
            task_name=self.config.task.env_name,
            device=str(self.device),
            num_points=num_points,
            max_episode_steps=max_steps,  # 传入max_steps参数
            #render=render
        )

        self.env = MultiStepWrapper(
            raw_env,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_execute,
            max_episode_steps=max_steps,
            reward_agg_method='sum',
        )
        print(f"  ✓ 环境已包装为MultiStepWrapper")
        print(f"  ✓ Action Chunking: 预测{self.n_action_predict}步，执行{self.n_action_execute}步")

        # 初始化策略
        print("初始化策略...")
        self.model = self._load_model()
        self.model.eval()

        print(f"✓ 模型加载成功")
        print(f"  任务: {self.config.task.env_name}")
        print(f"  训练步数: {self.checkpoint.get('global_step', 'unknown')}")
        print(f"  评估模式: Action Chunking（预测{self.n_action_predict}步，执行{self.n_action_execute}步）")
        print()

    def _load_model(self) -> RL100:
        """加载模型"""
        # 从config实例化模型
        import hydra
        model = hydra.utils.instantiate(self.config.policy)

        # 加载权重（优先加载EMA模型）
        # if 'ema_model_state_dict' in self.checkpoint:
        #     print("  ✓ 检测到EMA模型，加载EMA权重")
        #     model.load_state_dict(self.checkpoint['ema_model_state_dict'])
        # else:
        #     print("  ℹ 加载主模型权重")
        model.load_state_dict(self.checkpoint['model_state_dict'])

        model.to(self.device)

        # 设置normalizer
        if 'normalizer_state_dict' in self.checkpoint:
            from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(self.checkpoint['normalizer_state_dict'])
            model.set_normalizer(normalizer)
            print("  ✓ Normalizer已加载")

        return model

    def evaluate(self, num_episodes: int = 50, max_steps: int = None) -> dict:
        """
        评估策略（使用MultiStepWrapper）

        Args:
            num_episodes: 评估的episode数量
            max_steps: 每个episode的最大步数（None使用环境默认值）

        Returns:
            评估结果字典
        """
        if max_steps is None:
            max_steps = 500  # MetaWorld默认

        print(f"开始评估 ({num_episodes} episodes, 最大{max_steps}步)...\n")

        # 统计指标
        episode_rewards = []
        episode_lengths = []
        episode_successes = []

        with torch.no_grad():
            for episode_idx in tqdm(range(num_episodes), desc="评估进度"):
                # ===== MultiStepWrapper会自动管理观测历史 =====
                obs_dict = self.env.reset()  # 返回的obs已经是(n_obs_steps, ...)
                done = False

                episode_reward = 0
                episode_length = 0
                episode_success = False

                while not done:
                    # 准备输入（obs已经是stacked格式）
                    input_obs = {}
                    input_obs['point_cloud'] = torch.from_numpy(obs_dict['point_cloud']).unsqueeze(0).to(self.device).float()
                    input_obs['agent_pos'] = torch.from_numpy(obs_dict['agent_pos']).unsqueeze(0).to(self.device).float()

                    # 策略预测（返回horizon步的action序列，例如16步）
                    action_dict = self.model.predict_action(input_obs)
                    # action shape: (1, horizon, action_dim) 例如 (1, 16, 4)
                    action_full = action_dict['action'].squeeze(0).cpu().numpy()  # (horizon, action_dim)

                    # ===== 关键修改：只取前n_action_execute步执行 =====
                    action_to_execute = action_full[:self.n_action_execute]  # (8, action_dim)

                    # MultiStepWrapper会执行这8步
                    obs_dict, reward, done, info = self.env.step(action_to_execute)

                    episode_reward += reward
                    episode_length += self.n_action_execute  # 执行了8步

                    # 检查终止条件
                    done = np.all(done) if isinstance(done, np.ndarray) else done

                    # MetaWorld的成功判断（MultiStepWrapper返回的info是list）
                    if isinstance(info['success'], (list, np.ndarray)):
                        episode_success = episode_success or max(info['success'])
                    else:
                        episode_success = episode_success or info['success']

                # 记录统计
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_successes.append(1.0 if episode_success else 0.0)

        # 计算统计量
        results = {
            'success_rate': np.mean(episode_successes) * 100,
            'success_std': np.std(episode_successes) * 100,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'num_episodes': num_episodes,
            'checkpoint_path': self.checkpoint_path,
            'task_name': self.config.task.env_name,
            'global_step': self.checkpoint.get('global_step', 'unknown'),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_successes': episode_successes,
            'eval_method': f'Action Chunking: 预测{self.n_action_predict}步，执行{self.n_action_execute}步',
        }

        return results

    def print_results(self, results: dict):
        """打印评估结果"""
        print(f"\n{'='*60}")
        print("评估结果")
        print(f"{'='*60}")
        print(f"任务: {results['task_name']}")
        print(f"Checkpoint: {Path(results['checkpoint_path']).name}")
        print(f"训练步数: {results['global_step']}")
        print(f"评估Episodes: {results['num_episodes']}")
        print(f"评估方法: {results['eval_method']}")
        print(f"{'-'*60}")
        print(f"成功率: {results['success_rate']:.2f}% (±{results['success_std']:.2f}%)")
        print(f"平均奖励: {results['avg_reward']:.2f} (±{results['std_reward']:.2f})")
        print(f"平均长度: {results['avg_length']:.1f} (±{results['std_length']:.1f})")
        print(f"{'='*60}\n")

    def save_results(self, results: dict, output_path: str):
        """保存评估结果到JSON文件"""
        # 转换numpy类型为Python原生类型
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                results_serializable[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                results_serializable[key] = int(value)
            else:
                results_serializable[key] = value

        # 保存
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"✓ 结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RL100策略评估（MultiStepWrapper版本）")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint文件路径')
    parser.add_argument('--num_episodes', type=int, default=20,
                        help='评估的episode数量 (默认: 50)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='每个episode的最大步数 (默认: 使用环境默认值)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='运行设备 (默认: cuda:0)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果JSON文件路径 (默认: checkpoint同目录下)')
    parser.add_argument('--render', action='store_true',
                        help='是否渲染环境（可视化）')

    args = parser.parse_args()

    # 检查checkpoint是否存在
    if not os.path.exists(args.checkpoint):
        print(f"错误: checkpoint文件不存在: {args.checkpoint}")
        sys.exit(1)

    # 创建evaluator
    evaluator = RL100Evaluator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        # render=args.render
        max_steps=args.max_steps or 500
    )

    # 运行评估
    results = evaluator.evaluate(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps
    )

    # 打印结果
    evaluator.print_results(results)

    # 保存结果
    if args.output is None:
        # 默认保存到checkpoint同目录下
        checkpoint_dir = Path(args.checkpoint).parent
        checkpoint_name = Path(args.checkpoint).stem
        args.output = str(checkpoint_dir / f"{checkpoint_name}_eval_results.json")

    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
