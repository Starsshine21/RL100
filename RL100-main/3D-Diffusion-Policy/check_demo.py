"""
检查演示数据集的质量和成功率
"""
import sys
sys.path.insert(0, '/home/yrz/test/RL-100/3D-Diffusion-Policy')

import os
import zarr
import numpy as np
from pathlib import Path

def analyze_demonstration_data(zarr_path):
    """
    分析演示数据集的质量
    """
    print("="*80)
    print("演示数据集质量分析")
    print("="*80)
    print(f"数据路径: {zarr_path}")

    if not os.path.exists(zarr_path):
        print(f"\n❌ 错误: 数据路径不存在")
        print(f"   请检查路径是否正确: {zarr_path}")
        return

    # 打开zarr存储
    try:
        root = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"\n❌ 错误: 无法打开zarr文件")
        print(f"   错误信息: {e}")
        return

    # 获取数据和元数据
    try:
        data_group = root['data']
        meta_group = root['meta']
    except KeyError as e:
        print(f"\n❌ 错误: zarr文件结构不正确")
        print(f"   缺少组: {e}")
        print(f"   可用的键: {list(root.keys())}")
        return

    # 读取数据
    print(f"\n{'─'*80}")
    print("数据集基本信息")
    print(f"{'─'*80}")

    # 检查可用的数据字段
    available_keys = list(data_group.keys())
    print(f"可用字段: {', '.join(available_keys)}")

    # 读取基本数据
    try:
        actions = np.array(data_group['action'])
        rewards = np.array(data_group['reward'])
        dones = np.array(data_group['done'])
        episode_ends = np.array(meta_group['episode_ends'])

        print(f"\n总步数: {len(actions):,}")
        print(f"Episode数量: {len(episode_ends)}")

        # 其他可用字段
        if 'state' in data_group:
            states = np.array(data_group['state'])
            print(f"State维度: {states.shape}")

        if 'point_cloud' in data_group:
            pcs = np.array(data_group['point_cloud'])
            print(f"Point Cloud形状: {pcs.shape}")

    except KeyError as e:
        print(f"\n❌ 错误: 缺少必要的数据字段: {e}")
        return

    # 分析每个episode
    print(f"\n{'─'*80}")
    print("Episode分析")
    print(f"{'─'*80}")

    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episode_lengths = []
    episode_rewards = []
    episode_successes = []

    for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
        ep_length = end - start
        ep_reward = rewards[start:end].sum()
        ep_done = dones[start:end]

        # 检查是否成功（通常done=1表示episode结束）
        # MetaWorld中，成功通常通过最后几步的高reward判断
        # 或者检查done标志
        final_rewards = rewards[max(start, end-10):end]  # 最后10步
        is_success = ep_reward > 150  # 根据之前的分析，成功的episode reward > 150

        # 更精确的成功判断：最后几步reward是否持续高
        avg_final_reward = final_rewards.mean() if len(final_rewards) > 0 else 0
        is_success_refined = avg_final_reward > 0.5  # 最后几步平均reward > 0.5

        episode_lengths.append(ep_length)
        episode_rewards.append(ep_reward)
        episode_successes.append(is_success)

        # 打印前10个和后10个episode的详细信息
        if i < 10 or i >= len(episode_ends) - 10:
            success_mark = "✓" if is_success else "✗"
            print(f"Episode {i+1:3d}: 长度={ep_length:3d}, "
                  f"总reward={ep_reward:7.2f}, "
                  f"末段均值={avg_final_reward:5.2f}, "
                  f"成功={success_mark}")
        elif i == 10:
            print("  ...")

    # 统计信息
    print(f"\n{'='*80}")
    print("统计摘要")
    print(f"{'='*80}")

    episode_lengths = np.array(episode_lengths)
    episode_rewards = np.array(episode_rewards)
    episode_successes = np.array(episode_successes)

    success_count = episode_successes.sum()
    success_rate = success_count / len(episode_successes) * 100

    print(f"\nEpisode长度:")
    print(f"  平均: {episode_lengths.mean():.1f}")
    print(f"  最小: {episode_lengths.min()}")
    print(f"  最大: {episode_lengths.max()}")
    print(f"  标准差: {episode_lengths.std():.1f}")

    print(f"\nEpisode总Reward:")
    print(f"  平均: {episode_rewards.mean():.2f}")
    print(f"  最小: {episode_rewards.min():.2f}")
    print(f"  最大: {episode_rewards.max():.2f}")
    print(f"  标准差: {episode_rewards.std():.2f}")

    print(f"\n成功率:")
    print(f"  ✓ 成功: {success_count}/{len(episode_successes)} ({success_rate:.1f}%)")
    print(f"  ✗ 失败: {len(episode_successes) - success_count}/{len(episode_successes)} ({100-success_rate:.1f}%)")

    # Reward分布分析
    print(f"\n{'─'*80}")
    print("Reward分布")
    print(f"{'─'*80}")

    bins = [0, 100, 150, 180, 200, 220, 300]
    for i in range(len(bins)-1):
        count = ((episode_rewards >= bins[i]) & (episode_rewards < bins[i+1])).sum()
        pct = count / len(episode_rewards) * 100
        bar = "█" * int(pct / 2)
        print(f"  [{bins[i]:3d}, {bins[i+1]:3d}): {count:3d} ({pct:5.1f}%) {bar}")

    # 单步reward分析
    print(f"\n{'─'*80}")
    print("单步Reward分析")
    print(f"{'─'*80}")

    print(f"  平均单步reward: {rewards.mean():.4f}")
    print(f"  最小单步reward: {rewards.min():.4f}")
    print(f"  最大单步reward: {rewards.max():.4f}")
    print(f"  标准差: {rewards.std():.4f}")

    # 检查是否有异常值
    print(f"\n{'─'*80}")
    print("数据质量检查")
    print(f"{'─'*80}")

    # 检查NaN
    nan_count = np.isnan(rewards).sum()
    print(f"  NaN值: {nan_count} ({nan_count/len(rewards)*100:.2f}%)")

    # 检查无限值
    inf_count = np.isinf(rewards).sum()
    print(f"  Inf值: {inf_count} ({inf_count/len(rewards)*100:.2f}%)")

    # 检查done标志
    done_count = (dones > 0.5).sum()
    print(f"  Done标志: {done_count} 步")

    # 建议
    print(f"\n{'='*80}")
    print("数据集评估")
    print(f"{'='*80}")

    if success_rate > 90:
        print("  ✅ 优秀: 成功率超过90%，演示数据质量很高")
    elif success_rate > 70:
        print("  ✓ 良好: 成功率70-90%，演示数据质量不错")
    elif success_rate > 50:
        print("  ⚠️  中等: 成功率50-70%，建议检查失败的episode")
    else:
        print("  ❌ 较差: 成功率低于50%，建议重新收集演示数据")

    print(f"\n  推荐的数据质量阈值:")
    if success_rate > 80:
        print(f"    data_quality_threshold: 0.4  # 可以设置较高阈值")
    elif success_rate > 60:
        print(f"    data_quality_threshold: 0.3  # 当前设置合适")
    else:
        print(f"    data_quality_threshold: 0.2  # 建议降低阈值保留更多数据")

    print("="*80)

    return {
        'num_episodes': len(episode_ends),
        'total_steps': len(actions),
        'success_rate': success_rate,
        'avg_reward': episode_rewards.mean(),
        'avg_length': episode_lengths.mean()
    }


if __name__ == "__main__":
    # 数据路径（从配置文件中获取）
    data_paths = [
        # 主数据路径
        "/nfs_global/S/yangrongzheng/3D-Diffusion-Policy/3D-Diffusion-Policy/data/metaworld_wrapper_data",
        # 备选路径（如果你有其他位置）
        "./data/metaworld_wrapper_data",
        "../data/metaworld_wrapper_data",
    ]

    # 尝试找到存在的数据路径
    zarr_path = None
    for path in data_paths:
        if os.path.exists(path):
            zarr_path = path
            break

    if zarr_path is None:
        print("\n❌ 未找到演示数据")
        print("\n请指定正确的数据路径:")
        print("  python check_demo_data.py <path_to_zarr_data>")
        print("\n或者检查以下路径是否存在:")
        for path in data_paths:
            print(f"  - {path}")
        sys.exit(1)

    # 如果命令行指定了路径，使用命令行参数
    if len(sys.argv) > 1:
        zarr_path = sys.argv[1]

    # 分析数据
    stats = analyze_demonstration_data(zarr_path)

    if stats:
        print(f"\n✓ 分析完成")
        print(f"  Episodes: {stats['num_episodes']}")
        print(f"  总步数: {stats['total_steps']:,}")
        print(f"  成功率: {stats['success_rate']:.1f}%")
