import os
import numpy as np
import zarr
import torch
import shutil
import metaworld
from metaworld.policies import SawyerDialTurnV2Policy
# 导入你刚才保存的 Wrapper
from diffusion_policy_3d.env.metaworld.metaworld_wrapper import MetaWorldEnv

# ================= 配置区域 =================
TASK_NAME = 'dial-turn-v2-goal-observable'  # 任务名称
NUM_EPISODES = 10              # 采集多少条成功轨迹
MAX_STEPS = 200                # 每条轨迹最大步数
SAVE_DIR = './data/metaworld_wrapper_data/dial_turn_demos' # 保存路径
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_POINTS = 512               # 【修复】点云数量改为512（与DP3一致）

# 专家策略映射表 (根据任务名取策略)
POLICY_MAP = {
    'dial-turn-v2-goal-observable': SawyerDialTurnV2Policy,
    # 如果需要其他任务，在这里添加对应的 MetaWorld 策略
}
# ===========================================

def collect_data():
    # 1. 初始化环境 (使用 Wrapper!)
    print(f"正在初始化 MetaWorldEnv (Task: {TASK_NAME})...")
    # 注意：这里直接使用 Wrapper，保证采集的数据格式与训练/推理完全一致
    env = MetaWorldEnv(task_name=TASK_NAME, device=DEVICE, num_points=NUM_POINTS)

    # 2. 获取专家策略
    if TASK_NAME not in POLICY_MAP:
        raise ValueError(f"未找到任务 {TASK_NAME} 对应的专家策略，请在 POLICY_MAP 中添加。")
    policy = POLICY_MAP[TASK_NAME]()

    # 3. 准备 Zarr 存储结构
    if os.path.exists(SAVE_DIR):
        print(f"删除旧数据: {SAVE_DIR}")
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)

    # 内存暂存列表 (Buffer)
    buffer = {
        'state': [],        # 对应 Wrapper 的 agent_pos (9维)
        'full_state': [],   # 对应 Wrapper 的 full_state (原始39维，用于Debug或备用)
        'action': [],
        'point_cloud': [],
        'reward': [],
        'done': []
    }
    episode_ends = []
    total_steps = 0

    # 统计信息
    total_attempts = 0  # 总尝试次数
    successful_episodes = 0  # 成功的episode数

    # 4. 开始采集循环
    # 使用while循环，直到采集到足够的成功episode
    while successful_episodes < NUM_EPISODES:
        total_attempts += 1
        print(f"\n采集轨迹: 第{total_attempts}次尝试 (已成功: {successful_episodes}/{NUM_EPISODES})")

        # 重置环境，拿到第一帧观测
        # Wrapper 的 reset 返回的是一个字典
        obs_dict = env.reset()

        # Episode级别的临时buffer（用于质量检查后再决定是否保存）
        episode_buffer = {
            'state': [],
            'full_state': [],
            'action': [],
            'point_cloud': [],
            'reward': [],
            'done': []
        }
        ep_success_times = 0  # 记录episode中成功的次数
        ep_total_reward = 0   # 记录episode总奖励

        for step in range(MAX_STEPS):
            # === A. 获取动作 ===
            # 专家策略需要的是 MetaWorld 原始的 39维向量
            # Wrapper 在 obs_dict['full_state'] 里贴心地帮我们保留了这个原始数据
            raw_state = obs_dict['full_state']
            action = policy.get_action(raw_state)

            # 【修复】移除噪声：专家策略不需要添加噪声
            # 原代码: action = np.random.normal(action, 0.05)
            # 保持专家动作的原始质量，只做边界裁剪
            action = np.clip(action, -1.0, 1.0)

            # === B. 环境执行 ===
            # 这里的 next_obs_dict 里的 point_cloud 已经是处理好的了！
            next_obs_dict, reward, done, info = env.step(action)

            # === C. 存入 Episode Buffer（临时，用于质量检查）===
            # 存当前帧的数据到episode buffer
            episode_buffer['state'].append(obs_dict['agent_pos'])     # 9维
            episode_buffer['full_state'].append(obs_dict['full_state']) # 39维
            episode_buffer['point_cloud'].append(obs_dict['point_cloud'])
            episode_buffer['action'].append(action)
            episode_buffer['reward'].append(reward)
            episode_buffer['done'].append(float(done)) # 转为 float

            # 累计统计
            ep_total_reward += reward
            if info.get('success', False):
                ep_success_times += 1

            # 更新 Obs
            obs_dict = next_obs_dict

            # 如果成功了(info['success']) 或者 超时了
            # 注意：MetaWorld 的 done 信号有时候不准，通常以 success 或 max_steps 为准
            if done:
                break
            elif step == MAX_STEPS - 1:
                break

        # === D. 质量检查（关键修复）===
        # 遵循DP3的严格筛选标准：
        # 1. episode必须在最后成功 (info['success'] == True)
        # 2. 在episode过程中至少连续成功5次 (ep_success_times >= 5)
        ep_final_success = info.get('success', False)

        if ep_final_success and ep_success_times >= 5:
            # ✅ 质量合格，保存到全局buffer
            episode_steps = len(episode_buffer['state'])
            for key in buffer.keys():
                buffer[key].extend(episode_buffer[key])
            total_steps += episode_steps
            episode_ends.append(total_steps)
            successful_episodes += 1

            print(f"  ✅ 成功 - 奖励: {ep_total_reward:.2f}, 成功次数: {ep_success_times}, 步数: {episode_steps}")
        else:
            # ❌ 质量不合格，丢弃此episode
            reason = []
            if not ep_final_success:
                reason.append("最终未成功")
            if ep_success_times < 5:
                reason.append(f"成功次数不足({ep_success_times}<5)")
            print(f"  ❌ 失败 - {', '.join(reason)} - 奖励: {ep_total_reward:.2f}, 丢弃该轨迹")

    # 5. 写入 Zarr 文件
    print(f"正在写入硬盘: {SAVE_DIR} ...")
    root = zarr.open_group(SAVE_DIR, mode='w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    # 批量写入
    for key, val_list in buffer.items():
        # 转为 numpy 数组
        arr = np.array(val_list, dtype=np.float32)
        print(f"  -> 保存 {key}: 形状 {arr.shape}")
        data_group.create_dataset(key, data=arr, compressor=compressor)

    # 保存 meta 信息
    meta_group.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int64))

    print("\n" + "="*70)
    print("采集完成！")
    print("="*70)
    print(f"总尝试次数: {total_attempts}")
    print(f"成功Episode: {successful_episodes}/{total_attempts} ({successful_episodes/total_attempts*100:.1f}%)")
    print(f"总步数: {total_steps}")
    print(f"平均Episode长度: {total_steps/successful_episodes:.1f} 步")
    print(f"\n数据维度:")
    print(f"  State: {np.array(buffer['state']).shape} (期望是 [N, 9])")
    print(f"  Point Cloud: {np.array(buffer['point_cloud']).shape} (期望是 [N, 512, 3] 或 [N, 512, 6])")
    print(f"  Full State: {np.array(buffer['full_state']).shape} (期望是 [N, 39])")
    print(f"  Action: {np.array(buffer['action']).shape}")
    print("="*70)

if __name__ == "__main__":
    # 为了避免 MuJoCo 渲染在某些 headless 服务器上报错
    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'
    collect_data()
