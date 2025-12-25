"""
验证收集的演示数据是否完整、正确
"""
import os
import zarr
import numpy as np
from pathlib import Path

def verify_zarr_data(data_dir: str):
    """验证Zarr数据集的完整性和正确性"""

    print("="*70)
    print(f"验证数据集: {data_dir}")
    print("="*70)

    if not os.path.exists(data_dir):
        print(f"❌ 错误：数据目录不存在: {data_dir}")
        return False

    try:
        # 1. 加载Zarr数据
        root = zarr.open(data_dir, mode='r')
        data = root['data']
        meta = root['meta']

        # 2. 检查必需的键
        required_keys = ['state', 'full_state', 'action', 'point_cloud', 'reward', 'done']
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            print(f"❌ 缺少数据字段: {missing_keys}")
            return False

        print(f"✓ 所有必需字段存在: {required_keys}\n")

        # 3. 获取数据维度
        total_steps = len(data['done'])
        episode_ends = np.array(meta['episode_ends'][:])
        num_episodes = len(episode_ends)

        print(f"📊 数据集统计:")
        print(f"  总步数: {total_steps}")
        print(f"  Episode数: {num_episodes}")
        print(f"  平均Episode长度: {total_steps / num_episodes:.1f}\n")

        # 4. 检查每个字段的维度
        print(f"📐 数据维度检查:")
        expected_dims = {
            'state': (total_steps, 9),
            'full_state': (total_steps, 39),
            'action': (total_steps, 4),
            'point_cloud': (total_steps, 1024, 6),
            'reward': (total_steps,),
            'done': (total_steps,)
        }

        all_dims_correct = True
        for key, expected_shape in expected_dims.items():
            actual_shape = data[key].shape
            is_correct = actual_shape == expected_shape
            symbol = "✓" if is_correct else "❌"
            print(f"  {symbol} {key}: {actual_shape} (期望: {expected_shape})")
            if not is_correct:
                all_dims_correct = False

        print()

        # 5. 检查每个Episode的详细信息
        print(f"📋 Episode详细信息:")
        print(f"{'Episode':<10} {'起始索引':<12} {'结束索引':<12} {'步数':<8} {'总奖励':<12} {'成功':<6}")
        print("-"*70)

        episode_starts = [0] + list(episode_ends[:-1])
        episode_lengths = []
        episode_rewards = []
        episode_successes = []

        for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
            length = end - start
            episode_lengths.append(length)

            # 计算总奖励
            rewards = np.array(data['reward'][start:end])
            total_reward = np.sum(rewards)
            episode_rewards.append(total_reward)

            # 检查是否成功（最后一步的奖励通常表示成功）
            # MetaWorld通常成功时reward较高
            success = total_reward > 0  # 简单判断，可能需要根据任务调整
            episode_successes.append(success)

            success_symbol = "✓" if success else "✗"
            print(f"{i+1:<10} {start:<12} {end:<12} {length:<8} {total_reward:<12.2f} {success_symbol:<6}")

        print()

        # 6. 统计信息
        episode_lengths = np.array(episode_lengths)
        episode_rewards = np.array(episode_rewards)
        success_rate = np.mean(episode_successes) * 100

        print(f"📈 统计摘要:")
        print(f"  Episode长度: 最小={episode_lengths.min()}, "
              f"最大={episode_lengths.max()}, "
              f"平均={episode_lengths.mean():.1f}±{episode_lengths.std():.1f}")
        print(f"  总奖励: 最小={episode_rewards.min():.2f}, "
              f"最大={episode_rewards.max():.2f}, "
              f"平均={episode_rewards.mean():.2f}±{episode_rewards.std():.2f}")
        print(f"  成功率: {success_rate:.1f}% ({int(sum(episode_successes))}/{num_episodes})")
        print()

        # 7. 检查数据质量
        print(f"🔍 数据质量检查:")
        quality_ok = True

        # 检查NaN
        for key in required_keys:
            arr = np.array(data[key][:])
            has_nan = np.isnan(arr).any()
            symbol = "❌" if has_nan else "✓"
            print(f"  {symbol} {key}: {'发现NaN值' if has_nan else '无NaN值'}")
            if has_nan:
                quality_ok = False

        # 检查动作范围
        actions = np.array(data['action'][:])
        action_min, action_max = actions.min(), actions.max()
        action_in_range = (action_min >= -1.0) and (action_max <= 1.0)
        symbol = "✓" if action_in_range else "⚠️"
        print(f"  {symbol} action范围: [{action_min:.3f}, {action_max:.3f}] (期望: [-1.0, 1.0])")

        # 检查点云范围
        pc = np.array(data['point_cloud'][:])
        pc_min, pc_max = pc.min(), pc.max()
        print(f"  ℹ️  point_cloud范围: [{pc_min:.3f}, {pc_max:.3f}]")

        print()

        # 8. 检查episode_ends的一致性
        print(f"🔧 元数据一致性检查:")

        # 检查episode_ends是否递增
        is_monotonic = np.all(episode_ends[1:] > episode_ends[:-1])
        symbol = "✓" if is_monotonic else "❌"
        print(f"  {symbol} episode_ends递增: {is_monotonic}")

        # 检查最后一个episode_end是否等于总步数
        last_end_correct = episode_ends[-1] == total_steps
        symbol = "✓" if last_end_correct else "❌"
        print(f"  {symbol} 最后episode_end匹配总步数: {last_end_correct} ({episode_ends[-1]} == {total_steps})")

        # 检查done标志的位置
        done_indices = np.where(np.array(data['done'][:]) > 0.5)[0] + 1
        done_matches_ends = np.array_equal(done_indices, episode_ends)
        symbol = "✓" if done_matches_ends else "⚠️"
        print(f"  {symbol} done标志与episode_ends一致: {done_matches_ends}")
        if not done_matches_ends:
            print(f"      done位置: {done_indices}")
            print(f"      episode_ends: {episode_ends}")

        print()

        # 9. 最终判断
        print("="*70)
        if all_dims_correct and quality_ok and is_monotonic and last_end_correct:
            print("✅ 数据集验证通过！所有检查项均正常。")
            return True
        else:
            print("⚠️  数据集存在一些问题，请检查上述报告。")
            return False

    except Exception as e:
        print(f"❌ 验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="验证Zarr格式的演示数据")
    parser.add_argument('--data_dir', type=str, default='./data/dial_turn_demos',
                        help='数据目录路径')
    args = parser.parse_args()

    verify_zarr_data(args.data_dir)


if __name__ == "__main__":
    main()
