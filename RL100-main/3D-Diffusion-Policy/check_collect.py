"""
数据质量检查脚本
用于验证采集的专家演示数据是否符合训练要求

使用方法:
    python check_collected_data.py --data-path ./data/metaworld_wrapper_data/dial_turn_demos

检查内容:
    1. 数据维度是否正确
    2. 点云数量是否是512
    3. Episode质量统计
    4. 数据范围检查
    5. 异常值检测
    6. 数据分布可视化
"""

import os
import sys
import argparse
import zarr
import numpy as np
from termcolor import cprint
from collections import defaultdict

def print_section(title):
    """打印分节标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def check_data_structure(zarr_path):
    """检查数据结构是否完整"""
    print_section("1. 数据结构检查")

    if not os.path.exists(zarr_path):
        cprint(f"❌ 错误: 数据路径不存在: {zarr_path}", "red")
        return False

    try:
        root = zarr.open(zarr_path, mode='r')
    except Exception as e:
        cprint(f"❌ 错误: 无法打开Zarr数据: {e}", "red")
        return False

    # 检查必需的组和数据集
    required_groups = ['data', 'meta']
    required_datasets = {
        'data': ['state', 'full_state', 'action', 'point_cloud', 'reward', 'done'],
        'meta': ['episode_ends']
    }

    all_ok = True

    # 检查组
    for group_name in required_groups:
        if group_name in root:
            cprint(f"✅ 组 '{group_name}' 存在", "green")
        else:
            cprint(f"❌ 组 '{group_name}' 缺失", "red")
            all_ok = False
            continue

        # 检查该组下的数据集
        group = root[group_name]
        for dataset_name in required_datasets[group_name]:
            if dataset_name in group:
                shape = group[dataset_name].shape
                dtype = group[dataset_name].dtype
                cprint(f"  ✅ 数据集 '{dataset_name}': shape={shape}, dtype={dtype}", "cyan")
            else:
                cprint(f"  ❌ 数据集 '{dataset_name}' 缺失", "red")
                all_ok = False

    return all_ok, root

def check_dimensions(root):
    """检查数据维度是否正确"""
    print_section("2. 数据维度检查")

    data_group = root['data']

    # 期望的维度
    expected_dims = {
        'state': (None, 9),           # agent_pos: (N, 9)
        'full_state': (None, 39),     # MetaWorld完整状态: (N, 39)
        'point_cloud': (None, 512, None),  # 点云: (N, 512, 3或6)
        'action': (None, 4),          # MetaWorld动作: (N, 4)
        'reward': (None,),            # 奖励: (N,)
        'done': (None,)               # done标志: (N,)
    }

    all_ok = True
    total_steps = None

    for key, expected_shape in expected_dims.items():
        actual_shape = data_group[key].shape

        # 检查总步数一致性
        if total_steps is None:
            total_steps = actual_shape[0]
        elif actual_shape[0] != total_steps:
            cprint(f"❌ '{key}' 的步数 {actual_shape[0]} 与其他数据不一致 (期望 {total_steps})", "red")
            all_ok = False

        # 检查具体维度
        match = True
        for i, exp_dim in enumerate(expected_shape):
            if exp_dim is not None and i < len(actual_shape):
                if actual_shape[i] != exp_dim:
                    match = False
                    break

        if match:
            cprint(f"✅ '{key}': {actual_shape} - 维度正确", "green")
        else:
            cprint(f"❌ '{key}': {actual_shape} - 期望 {expected_shape}", "red")
            all_ok = False

            # 特殊检查：点云
            if key == 'point_cloud':
                if actual_shape[1] == 1024:
                    cprint(f"  ⚠️  警告: 点云数量是1024，应该是512（与DP3不一致）", "yellow")
                if len(actual_shape) == 3 and actual_shape[2] not in [3, 6]:
                    cprint(f"  ❌ 点云通道数 {actual_shape[2]} 异常（应为3或6）", "red")

    print(f"\n📊 总步数: {total_steps}")
    return all_ok, total_steps

def check_episodes(root, total_steps):
    """检查Episode质量"""
    print_section("3. Episode质量检查")

    episode_ends = root['meta']['episode_ends'][:]
    num_episodes = len(episode_ends)

    print(f"📦 总Episode数: {num_episodes}")

    if num_episodes == 0:
        cprint("❌ 错误: 没有Episode数据！", "red")
        return False

    # 计算每个episode的长度
    episode_lengths = []
    episode_starts = [0] + list(episode_ends[:-1])

    for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
        length = end - start
        episode_lengths.append(length)

    episode_lengths = np.array(episode_lengths)

    # 统计信息
    print(f"\nEpisode长度统计:")
    print(f"  平均长度: {np.mean(episode_lengths):.1f} 步")
    print(f"  最短长度: {np.min(episode_lengths)} 步")
    print(f"  最长长度: {np.max(episode_lengths)} 步")
    print(f"  标准差: {np.std(episode_lengths):.1f}")

    # 检查异常episode
    abnormal_episodes = []
    for i, length in enumerate(episode_lengths):
        if length < 10:
            abnormal_episodes.append((i, length, "太短"))
        elif length > 200:
            abnormal_episodes.append((i, length, "太长"))

    if abnormal_episodes:
        cprint(f"\n⚠️  发现 {len(abnormal_episodes)} 个异常Episode:", "yellow")
        for ep_idx, length, reason in abnormal_episodes[:5]:  # 只显示前5个
            print(f"  Episode {ep_idx}: {length}步 ({reason})")
        if len(abnormal_episodes) > 5:
            print(f"  ... 还有 {len(abnormal_episodes)-5} 个异常episode")
    else:
        cprint("✅ 所有Episode长度正常", "green")

    # 检查总步数一致性
    if episode_ends[-1] != total_steps:
        cprint(f"❌ 错误: episode_ends最后一个值 ({episode_ends[-1]}) 与总步数 ({total_steps}) 不一致", "red")
        return False
    else:
        cprint("✅ Episode索引一致性检查通过", "green")

    return True

def check_data_ranges(root):
    """检查数据范围是否合理"""
    print_section("4. 数据范围检查")

    data_group = root['data']

    checks = {
        'state': {
            'expected_range': (-2.0, 2.0),
            'check_nan': True,
            'check_inf': True
        },
        'full_state': {
            'expected_range': (-10.0, 10.0),
            'check_nan': True,
            'check_inf': True
        },
        'action': {
            'expected_range': (-1.0, 1.0),
            'check_nan': True,
            'check_inf': True
        },
        'point_cloud': {
            'expected_range': (-2.0, 2.0),
            'check_nan': True,
            'check_inf': True
        },
        'reward': {
            'expected_range': (-100.0, 100.0),
            'check_nan': True,
            'check_inf': True
        },
        'done': {
            'expected_range': (0.0, 1.0),
            'check_nan': False,
            'check_inf': False
        }
    }

    all_ok = True

    for key, config in checks.items():
        data = data_group[key][:]

        # 统计信息
        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        std_val = np.std(data)

        print(f"\n{key}:")
        print(f"  范围: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  均值: {mean_val:.4f}")
        print(f"  标准差: {std_val:.4f}")

        # 检查范围
        expected_min, expected_max = config['expected_range']
        if min_val < expected_min or max_val > expected_max:
            cprint(f"  ⚠️  警告: 数据超出期望范围 [{expected_min}, {expected_max}]", "yellow")
        else:
            cprint(f"  ✅ 范围正常", "green")

        # 检查NaN
        if config['check_nan']:
            nan_count = np.isnan(data).sum()
            if nan_count > 0:
                cprint(f"  ❌ 发现 {nan_count} 个NaN值！", "red")
                all_ok = False
            else:
                cprint(f"  ✅ 无NaN值", "green")

        # 检查Inf
        if config['check_inf']:
            inf_count = np.isinf(data).sum()
            if inf_count > 0:
                cprint(f"  ❌ 发现 {inf_count} 个Inf值！", "red")
                all_ok = False
            else:
                cprint(f"  ✅ 无Inf值", "green")

    return all_ok

def check_point_cloud_quality(root):
    """检查点云质量"""
    print_section("5. 点云质量检查")

    point_cloud = root['data']['point_cloud'][:]

    # 检查点云形状
    print(f"点云形状: {point_cloud.shape}")

    # 随机采样一些点云进行检查
    sample_indices = np.random.choice(len(point_cloud), min(100, len(point_cloud)), replace=False)

    # 检查每个点云的点数分布
    valid_point_counts = []
    for idx in sample_indices:
        pc = point_cloud[idx]
        # 检查有多少点不是全0
        valid_points = ~np.all(pc[:, :3] == 0, axis=1)
        valid_count = np.sum(valid_points)
        valid_point_counts.append(valid_count)

    valid_point_counts = np.array(valid_point_counts)

    print(f"\n有效点数统计（采样{len(sample_indices)}个点云）:")
    print(f"  平均: {np.mean(valid_point_counts):.1f} / {point_cloud.shape[1]}")
    print(f"  最少: {np.min(valid_point_counts)} / {point_cloud.shape[1]}")
    print(f"  最多: {np.max(valid_point_counts)} / {point_cloud.shape[1]}")

    if np.min(valid_point_counts) < point_cloud.shape[1] * 0.5:
        cprint(f"  ⚠️  警告: 某些点云的有效点数少于50%", "yellow")
    else:
        cprint(f"  ✅ 点云密度正常", "green")

    # 检查点云坐标范围
    pc_flat = point_cloud.reshape(-1, point_cloud.shape[-1])
    xyz = pc_flat[:, :3]

    print(f"\n点云XYZ坐标范围:")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        print(f"  {axis}: [{np.min(xyz[:, i]):.3f}, {np.max(xyz[:, i]):.3f}]")

    return True

def check_action_distribution(root):
    """检查动作分布"""
    print_section("6. 动作分布检查")

    actions = root['data']['action'][:]

    print(f"动作形状: {actions.shape}")
    print(f"\n各维度统计:")

    for i in range(actions.shape[1]):
        dim_data = actions[:, i]
        print(f"\n  维度 {i}:")
        print(f"    范围: [{np.min(dim_data):.4f}, {np.max(dim_data):.4f}]")
        print(f"    均值: {np.mean(dim_data):.4f}")
        print(f"    标准差: {np.std(dim_data):.4f}")

        # 检查是否所有动作都是0（可能表明没有实际动作）
        if np.all(np.abs(dim_data) < 1e-6):
            cprint(f"    ⚠️  警告: 该维度所有值接近0", "yellow")

    return True

def check_reward_success(root):
    """检查奖励和成功情况"""
    print_section("7. 奖励和成功情况检查")

    rewards = root['data']['reward'][:]
    dones = root['data']['done'][:]
    episode_ends = root['meta']['episode_ends'][:]

    # Episode级别的奖励统计
    episode_starts = [0] + list(episode_ends[:-1])
    episode_rewards = []

    for start, end in zip(episode_starts, episode_ends):
        ep_reward = np.sum(rewards[start:end])
        episode_rewards.append(ep_reward)

    episode_rewards = np.array(episode_rewards)

    print(f"Episode奖励统计:")
    print(f"  平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"  最低奖励: {np.min(episode_rewards):.2f}")
    print(f"  最高奖励: {np.max(episode_rewards):.2f}")
    print(f"  标准差: {np.std(episode_rewards):.2f}")

    # Done标志统计
    done_count = np.sum(dones > 0.5)
    print(f"\nDone标志统计:")
    print(f"  Done次数: {done_count}")
    print(f"  预期Done次数: {len(episode_ends)} (每个episode结束)")

    if done_count != len(episode_ends):
        cprint(f"  ⚠️  警告: Done次数与Episode数不匹配", "yellow")

    return True

def generate_summary(zarr_path):
    """生成数据摘要"""
    print_section("8. 数据摘要")

    root = zarr.open(zarr_path, mode='r')

    total_steps = root['data']['state'].shape[0]
    num_episodes = len(root['meta']['episode_ends'])

    point_cloud_shape = root['data']['point_cloud'].shape
    state_dim = root['data']['state'].shape[1]
    action_dim = root['data']['action'].shape[1]

    print(f"📊 数据集摘要:")
    print(f"  数据路径: {zarr_path}")
    print(f"  总步数: {total_steps:,}")
    print(f"  总Episode数: {num_episodes}")
    print(f"  平均Episode长度: {total_steps/num_episodes:.1f} 步")
    print(f"\n📐 数据维度:")
    print(f"  状态维度: {state_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  点云形状: {point_cloud_shape[1:]} (每步)")

    # 估算数据集大小
    total_size_mb = sum([
        root['data'][key].nbytes
        for key in root['data'].keys()
    ]) / (1024 ** 2)

    print(f"\n💾 数据集大小: {total_size_mb:.1f} MB")

    return True

def main():
    parser = argparse.ArgumentParser(description='检查采集的专家演示数据质量')
    parser.add_argument('--data-path', type=str,
                       default='./data/metaworld_wrapper_data/dial_turn_demos',
                       help='Zarr数据路径')
    args = parser.parse_args()

    print("\n" + "🔍 "*20)
    cprint("数据质量检查工具", "cyan", attrs=['bold'])
    print("🔍 "*20)

    # 1. 结构检查
    success, root = check_data_structure(args.data_path)
    if not success:
        cprint("\n❌ 数据结构检查失败！请检查数据采集是否正确。", "red")
        return

    # 2. 维度检查
    success, total_steps = check_dimensions(root)
    if not success:
        cprint("\n⚠️  数据维度存在问题！", "yellow")

    # 3. Episode检查
    check_episodes(root, total_steps)

    # 4. 数据范围检查
    check_data_ranges(root)

    # 5. 点云质量检查
    check_point_cloud_quality(root)

    # 6. 动作分布检查
    check_action_distribution(root)

    # 7. 奖励和成功检查
    check_reward_success(root)

    # 8. 生成摘要
    generate_summary(args.data_path)

    # 最终评估
    print_section("最终评估")

    # 检查关键指标
    point_cloud_shape = root['data']['point_cloud'].shape
    state_dim = root['data']['state'].shape[1]
    num_episodes = len(root['meta']['episode_ends'])

    issues = []

    # 检查点云数量
    if point_cloud_shape[1] != 512:
        issues.append(f"点云数量是{point_cloud_shape[1]}，应该是512")

    # 检查状态维度
    if state_dim != 9:
        issues.append(f"状态维度是{state_dim}，应该是9")

    # 检查Episode数量
    if num_episodes < 50:
        issues.append(f"Episode数量只有{num_episodes}，建议至少50个")

    if issues:
        cprint("\n⚠️  发现以下问题:", "yellow")
        for issue in issues:
            print(f"  - {issue}")
        cprint("\n建议: 检查数据采集脚本并重新采集", "yellow")
    else:
        cprint("\n🎉 数据质量检查通过！可以用于训练。", "green", attrs=['bold'])

    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
