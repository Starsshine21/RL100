import os
import numpy as np
import zarr
import torch
import shutil
import metaworld
from metaworld.policies import SawyerButtonPressV2Policy, SawyerHammerV2Policy
# 导入你刚才保存的 Wrapper
from diffusion_policy_3d.env.metaworld.metaworld_wrapper import MetaWorldEnv 

# ================= 配置区域 =================
TASK_NAME = 'button-press-v2-goal-observable'  # 任务名称
NUM_EPISODES = 100              # 采集多少条成功轨迹
MAX_STEPS = 200                # 每条轨迹最大步数
SAVE_DIR = './data/metaworld_wrapper_data' # 保存路径
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_POINTS = 512               # 【修复】点云数量改为512（与DP3一致）

# 专家策略映射表 (根据任务名取策略)
POLICY_MAP = {
    'button-press-v2-goal-observable': SawyerButtonPressV2Policy,
    'hammer-v2': SawyerHammerV2Policy,
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
        'state': [],
        'full_state': [],
        'action': [],
        'point_cloud': [],
        'reward': [],
        'done': []
    }
    episode_ends = []
    total_steps = 0

    # 统计信息
    total_attempts = 0
    successful_episodes = 0

    # 4. 开始采集循环
    while successful_episodes < NUM_EPISODES:
        total_attempts += 1
        print(f"\n采集轨迹: 第{total_attempts}次尝试 (已成功: {successful_episodes}/{NUM_EPISODES})")

        obs_dict = env.reset()

        episode_buffer = {
            'state': [],
            'full_state': [],
            'action': [],
            'point_cloud': [],
            'reward': [],
            'done': []
        }
        ep_success_times = 0
        ep_total_reward = 0

        for step in range(MAX_STEPS):
            raw_state = obs_dict['full_state']
            action = policy.get_action(raw_state)

            # 【修复】移除噪声
            action = np.clip(action, -1.0, 1.0)

            next_obs_dict, reward, done, info = env.step(action)

            episode_buffer['state'].append(obs_dict['agent_pos'])
            episode_buffer['full_state'].append(obs_dict['full_state'])
            episode_buffer['point_cloud'].append(obs_dict['point_cloud'])
            episode_buffer['action'].append(action)
            episode_buffer['reward'].append(reward)
            episode_buffer['done'].append(float(done))

            ep_total_reward += reward
            if info.get('success', False):
                ep_success_times += 1

            obs_dict = next_obs_dict

            if done or step == MAX_STEPS - 1:
                break

        # 【修复】质量检查
        ep_final_success = info.get('success', False)

        if ep_final_success and ep_success_times >= 5:
            episode_steps = len(episode_buffer['state'])
            for key in buffer.keys():
                buffer[key].extend(episode_buffer[key])
            total_steps += episode_steps
            episode_ends.append(total_steps)
            successful_episodes += 1
            print(f"  ✅ 成功 - 奖励: {ep_total_reward:.2f}, 成功次数: {ep_success_times}, 步数: {episode_steps}")
        else:
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
    print("="*70)

if __name__ == "__main__":
    # 为了避免 MuJoCo 渲染在某些 headless 服务器上报错
    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl' 
    collect_data()