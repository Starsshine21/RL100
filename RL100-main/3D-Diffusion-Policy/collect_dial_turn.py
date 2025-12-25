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
NUM_EPISODES = 100              # 采集多少条轨迹
MAX_STEPS = 200                # 每条轨迹最大步数
SAVE_DIR = './data/metaworld_wrapper_data/dial_turn_demos' # 保存路径
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
    env = MetaWorldEnv(task_name=TASK_NAME, device=DEVICE, num_points=1024)

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

    # 4. 开始采集循环
    for episode_idx in range(NUM_EPISODES):
        print(f"采集轨迹: {episode_idx+1}/{NUM_EPISODES}")

        # 重置环境，拿到第一帧观测
        # Wrapper 的 reset 返回的是一个字典
        obs_dict = env.reset()

        for step in range(MAX_STEPS):
            # === A. 获取动作 ===
            # 专家策略需要的是 MetaWorld 原始的 39维向量
            # Wrapper 在 obs_dict['full_state'] 里贴心地帮我们保留了这个原始数据
            raw_state = obs_dict['full_state']
            action = policy.get_action(raw_state)

            # 添加少量噪声，增加数据多样性 (对抗过拟合)
            action = np.random.normal(action, 0.05)
            action = np.clip(action, -1.0, 1.0)

            # === B. 环境执行 ===
            # 这里的 next_obs_dict 里的 point_cloud 已经是处理好的了！
            next_obs_dict, reward, done, info = env.step(action)

            # === C. 存入 Buffer ===
            # 存当前帧的数据
            buffer['state'].append(obs_dict['agent_pos'])     # 9维
            buffer['full_state'].append(obs_dict['full_state']) # 39维
            buffer['point_cloud'].append(obs_dict['point_cloud'])
            buffer['action'].append(action)
            buffer['reward'].append(reward)
            buffer['done'].append(float(done)) # 转为 float

            # 更新 Obs
            obs_dict = next_obs_dict
            total_steps += 1

            # 如果成功了(info['success']) 或者 超时了
            # 注意：MetaWorld 的 done 信号有时候不准，通常以 success 或 max_steps 为准
            # 这里为了简单，跑满 MAX_STEPS 或者遇到明确 done
            if done:
                print("成功")
                break
            elif step == MAX_STEPS - 1:
                print("超过步数限制")
                break

        # 记录每条轨迹的结束点
        episode_ends.append(total_steps)

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

    print("\n采集成功！")
    print(f"总步数: {total_steps}")
    print(f"State 维度: {np.array(buffer['state']).shape[1]} (期望是 9)")
    print(f"Point Cloud 维度: {np.array(buffer['point_cloud']).shape[1:]} (期望是 1024, 6)")

if __name__ == "__main__":
    # 为了避免 MuJoCo 渲染在某些 headless 服务器上报错
    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'
    collect_data()
