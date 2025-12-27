"""
RL100 批量评估脚本
功能：对比不同训练阶段的checkpoint性能
输出：对比表格、图表、详细报告
"""
import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_rl100 import RL100Evaluator


class BatchEvaluator:
    """批量评估器"""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.results = []

    def evaluate_checkpoint(self, checkpoint_path: str, stage_name: str,
                            num_episodes: int = 50, max_steps: int = None) -> dict:
        """
        评估单个checkpoint

        Args:
            checkpoint_path: checkpoint路径
            stage_name: 阶段名称（用于显示）
            num_episodes: 评估episodes数
            max_steps: 最大步数

        Returns:
            评估结果
        """
        print(f"\n{'='*60}")
        print(f"评估阶段: {stage_name}")
        print(f"{'='*60}")

        evaluator = RL100Evaluator(
            checkpoint_path=checkpoint_path,
            device=self.device,
            render=False,
             max_steps=max_steps
        )

        results = evaluator.evaluate(num_episodes=num_episodes, max_steps=max_steps)
        results['stage_name'] = stage_name

        # 打印结果
        evaluator.print_results(results)

        self.results.append(results)
        return results

    def print_comparison_table(self):
        """打印对比表格"""
        if not self.results:
            print("没有评估结果")
            return

        print(f"\n{'='*80}")
        print("各阶段性能对比")
        print(f"{'='*80}\n")

        # 准备表格数据
        headers = ["阶段", "成功率 (%)", "平均奖励", "平均长度", "Episodes"]
        table_data = []

        for result in self.results:
            row = [
                result['stage_name'],
                f"{result['success_rate']:.2f} ± {result['success_std']:.2f}",
                f"{result['avg_reward']:.2f} ± {result['std_reward']:.2f}",
                f"{result['avg_length']:.1f} ± {result['std_length']:.1f}",
                result['num_episodes']
            ]
            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print()

        # 打印最佳阶段
        best_idx = np.argmax([r['success_rate'] for r in self.results])
        best_stage = self.results[best_idx]
        print(f"✓ 最佳阶段: {best_stage['stage_name']}")
        print(f"  成功率: {best_stage['success_rate']:.2f}%")
        print(f"  平均奖励: {best_stage['avg_reward']:.2f}")
        print()

    def plot_comparison(self, output_path: str = None):
        """绘制对比图表"""
        if not self.results:
            print("没有评估结果")
            return

        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            stage_names = [r['stage_name'] for r in self.results]
            success_rates = [r['success_rate'] for r in self.results]
            success_stds = [r['success_std'] for r in self.results]
            avg_rewards = [r['avg_reward'] for r in self.results]
            reward_stds = [r['std_reward'] for r in self.results]
            avg_lengths = [r['avg_length'] for r in self.results]
            length_stds = [r['std_length'] for r in self.results]

            x = np.arange(len(stage_names))
            width = 0.6

            # 图1: 成功率
            axes[0].bar(x, success_rates, width, yerr=success_stds,
                       capsize=5, color='#2ecc71', alpha=0.8)
            axes[0].set_xlabel('训练阶段', fontsize=12)
            axes[0].set_ylabel('成功率 (%)', fontsize=12)
            axes[0].set_title('各阶段成功率对比', fontsize=14, fontweight='bold')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(stage_names, rotation=15, ha='right')
            axes[0].grid(axis='y', alpha=0.3)
            axes[0].set_ylim(0, 105)

            # 在柱子上标注数值
            for i, v in enumerate(success_rates):
                axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

            # 图2: 平均奖励
            axes[1].bar(x, avg_rewards, width, yerr=reward_stds,
                       capsize=5, color='#3498db', alpha=0.8)
            axes[1].set_xlabel('训练阶段', fontsize=12)
            axes[1].set_ylabel('平均奖励', fontsize=12)
            axes[1].set_title('各阶段平均奖励对比', fontsize=14, fontweight='bold')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(stage_names, rotation=15, ha='right')
            axes[1].grid(axis='y', alpha=0.3)

            # 图3: Episode长度
            axes[2].bar(x, avg_lengths, width, yerr=length_stds,
                       capsize=5, color='#e74c3c', alpha=0.8)
            axes[2].set_xlabel('训练阶段', fontsize=12)
            axes[2].set_ylabel('平均步数', fontsize=12)
            axes[2].set_title('各阶段Episode长度对比', fontsize=14, fontweight='bold')
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(stage_names, rotation=15, ha='right')
            axes[2].grid(axis='y', alpha=0.3)

            plt.tight_layout()

            if output_path is None:
                output_path = 'evaluation_comparison.png'

            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ 对比图表已保存到: {output_path}")

        except ImportError:
            print("警告: 无法导入matplotlib，跳过绘图")
            print("可以通过 pip install matplotlib 安装")

    def save_results(self, output_path: str):
        """保存所有结果到JSON文件"""
        # 转换为可序列化格式
        results_serializable = []
        for result in self.results:
            result_dict = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    result_dict[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    result_dict[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    result_dict[key] = int(value)
                else:
                    result_dict[key] = value
            results_serializable.append(result_dict)

        # 保存
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"✓ 批量评估结果已保存到: {output_path}")


def auto_find_checkpoints(checkpoint_dir: str) -> Dict[str, str]:
    """
    自动查找checkpoint目录下的各阶段checkpoint

    Args:
        checkpoint_dir: checkpoint目录路径

    Returns:
        阶段名称到checkpoint路径的映射
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = {}

    # 定义checkpoint模式
    patterns = {
        'IL预训练': 'checkpoint_il_final.ckpt',
        'RL迭代0': 'checkpoint_iter_0.ckpt',
        'RL迭代1': 'checkpoint_iter_1.ckpt',
        'RL迭代2': 'checkpoint_iter_2.ckpt',
        'RL迭代3': 'checkpoint_iter_3.ckpt',
        '在线RL': 'checkpoint_online_rl_final.ckpt',
    }

    for stage_name, pattern in patterns.items():
        checkpoint_path = checkpoint_dir / pattern
        if checkpoint_path.exists():
            checkpoints[stage_name] = str(checkpoint_path)

    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="RL100批量评估")

    # 模式1: 自动查找checkpoint
    parser.add_argument('--checkpoint_dir', type=str,
                        help='checkpoint目录路径（自动查找各阶段checkpoint）')

    # 模式2: 手动指定checkpoint
    parser.add_argument('--checkpoints', nargs='+', type=str,
                        help='checkpoint文件路径列表')
    parser.add_argument('--stage_names', nargs='+', type=str,
                        help='对应的阶段名称列表')

    # 通用参数
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='每个checkpoint评估的episode数量 (默认: 50)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='每个episode的最大步数 (默认: 使用环境默认值)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='运行设备 (默认: cuda:0)')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='输出目录 (默认: ./eval_results)')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定要评估的checkpoints
    checkpoints_to_eval = {}

    if args.checkpoint_dir:
        # 模式1: 自动查找
        print(f"自动查找checkpoint目录: {args.checkpoint_dir}\n")
        checkpoints_to_eval = auto_find_checkpoints(args.checkpoint_dir)

        if not checkpoints_to_eval:
            print(f"错误: 在 {args.checkpoint_dir} 中未找到checkpoint文件")
            sys.exit(1)

        print(f"找到 {len(checkpoints_to_eval)} 个checkpoint:")
        for stage, path in checkpoints_to_eval.items():
            print(f"  - {stage}: {Path(path).name}")
        print()

    elif args.checkpoints:
        # 模式2: 手动指定
        if args.stage_names and len(args.stage_names) != len(args.checkpoints):
            print("错误: --stage_names 和 --checkpoints 的数量必须相同")
            sys.exit(1)

        stage_names = args.stage_names or [f"阶段{i+1}" for i in range(len(args.checkpoints))]

        for stage_name, checkpoint_path in zip(stage_names, args.checkpoints):
            if not os.path.exists(checkpoint_path):
                print(f"错误: checkpoint文件不存在: {checkpoint_path}")
                sys.exit(1)
            checkpoints_to_eval[stage_name] = checkpoint_path

    else:
        print("错误: 必须指定 --checkpoint_dir 或 --checkpoints")
        parser.print_help()
        sys.exit(1)

    # 创建批量评估器
    batch_evaluator = BatchEvaluator(device=args.device)

    # 依次评估每个checkpoint
    for stage_name, checkpoint_path in checkpoints_to_eval.items():
        batch_evaluator.evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            stage_name=stage_name,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps
        )

    # 打印对比表格
    batch_evaluator.print_comparison_table()

    # 绘制对比图表
    plot_path = output_dir / 'comparison_plot.png'
    batch_evaluator.plot_comparison(output_path=str(plot_path))

    # 保存结果
    results_path = output_dir / 'batch_evaluation_results.json'
    batch_evaluator.save_results(output_path=str(results_path))

    print(f"\n{'='*80}")
    print("批量评估完成！")
    print(f"结果已保存到: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
