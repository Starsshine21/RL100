import os
import numpy as np
import zarr
import metaworld
from metaworld.policies import SawyerButtonPressV2Policy
import cv2
import shutil

# ================= 配置区域 =================
TASK_NAME = 'button-press-v2'  # 任务名称
NUM_EPISODES = 5               # 采集多少条轨迹（跑通为主，设小一点）
MAX_STEPS = 200                # 每条轨迹最大步数
SAVE_DIR = './data/metaworld2' # 保存路径
IMG_SIZE = 128                 # 摄像头分辨率
NUM_POINTS = 1024              # 每个点云采样的点数
# ===========================================

def get_camera_matrix(width, height, fov):
    """根据FOV计算相机的内参矩阵"""
    f = 0.5 * height / np.tan(fov * np.pi / 360)
    cx = width / 2
    cy = height / 2
    matrix = np.array([[f, 0, cx], 
                       [0, f, cy], 
                       [0, 0, 1]])
    return matrix

def depth_to_point_cloud(depth, rgb, camera_matrix):
    """将深度图转换为点云"""
    # 创建像素坐标网格
    rows, cols = depth.shape
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # 展平
    u = u.flatten()
    v = v.flatten()
    depth = depth.flatten()
    rgb = rgb.reshape(-1, 3)
    
    # 过滤掉无效深度 (MuJoCo背景通常深度很大或为0)
    valid = (depth > 0.1) & (depth < 2.0)
    u, v, depth, rgb = u[valid], v[valid], depth[valid], rgb[valid]
    
    # 反投影公式: Z=depth, X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy
    # 注意：MuJoCo的相机坐标系可能需要翻转，这里简化处理
    z = depth
    x = (u - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
    y = (v - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
    
    # 堆叠成 (N, 3)
    points = np.stack([x, y, z], axis=1)
    
    # 简单的坐标系转换（把相机坐标系转到类似于世界坐标系的方向）
    # 这里为了演示简单，做了一个近似的翻转
    points[:, 1] *= -1 
    points[:, 2] *= -1
    
    return points, rgb

def downsample_point_cloud(points, rgb, num_points):
    """随机采样点云到固定数量"""
    if len(points) == 0:
        return np.zeros((num_points, 3)), np.zeros((num_points, 3))
        
    if len(points) >= num_points:
        idxs = np.random.choice(len(points), num_points, replace=False)
    else:
        idxs = np.random.choice(len(points), num_points, replace=True)
    
    return points[idxs], rgb[idxs]

def collect_data():
    # 1. 初始化环境和专家策略
    print(f"正在初始化 MetaWorld 任务: {TASK_NAME} ...")
    ml1 = metaworld.ML1(TASK_NAME)
    env = ml1.train_classes[TASK_NAME]()
    task = ml1.train_tasks[0]
    env.set_task(task)
    
    # 获取专家策略 (MetaWorld 提供的硬编码策略)
    policy = SawyerButtonPressV2Policy()
    
    # 2. 准备 Zarr 存储结构
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR) # 清空旧数据
    os.makedirs(SAVE_DIR)
    
    root = zarr.open_group(SAVE_DIR, mode='w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')
    
    # 定义用来暂存所有轨迹的列表
    all_obs_pc = []
    all_obs_state = [] # agent_pos
    all_actions = []
    all_rewards = []
    all_dones = []
    episode_ends = [] # 记录每条轨迹结束的索引
    
    current_idx = 0
    
    # 3. 开始循环采集
    for episode_idx in range(NUM_EPISODES):
        obs = env.reset() # 注意：gym新版返回 (obs, info)，旧版只返回 obs
        if isinstance(obs, tuple): obs = obs[0]
        
        print(f"正在采集第 {episode_idx+1}/{NUM_EPISODES} 条轨迹...")
        
        for step in range(MAX_STEPS):
            # === A. 专家动作 ===
            action = policy.get_action(obs)
            # 添加一点随机噪声，让数据更多样化（这一步可选）
            action = np.random.normal(action, 0.05)
            
            # === B. 获取视觉数据 (RGB-D) ===
            # 使用 'corner' 相机，它是 MetaWorld 默认的侧上方视角
            # MuJoCo 渲染深度图
            rgb, depth = env.sim.render(width=IMG_SIZE, height=IMG_SIZE, camera_name='corner3', depth=True)
            # depth 是原本的距离，MuJoCo 返回的是归一化的，需要反转
            # 但 metaworld 的 env.sim.render depth=True 返回的是真实的 meter 距离 (如果配置对的话)
            # 这里做一个简单的翻转修正，因为 Mujoco 的 depth buffer 有时是倒的
            rgb = np.flipud(rgb) 
            depth = np.flipud(depth)
            
            # === C. 转换为点云 ===
            # 这里的 FOV 45 是 MetaWorld 默认的
            cam_matrix = get_camera_matrix(IMG_SIZE, IMG_SIZE, 45)
            pc_xyz, pc_rgb = depth_to_point_cloud(depth, rgb, cam_matrix)
            
            # 采样到 1024 个点
            pc_xyz, pc_rgb = downsample_point_cloud(pc_xyz, pc_rgb, NUM_POINTS)
            
            # === D. 获取本体状态 (Proprioception) ===
            # MetaWorld 的 obs 是 39维向量
            # 前4维是末端位置(3) + 夹爪(1)
            # 具体的索引取决于任务，但通常前几维是机器人状态
            agent_pos = obs[:4] 
            
            # === E. 环境执行一步 ===
            next_obs, reward, done, info = env.step(action)
            
            # === F. 存入暂存列表 ===
            all_obs_pc.append(pc_xyz)       # (1024, 3)
            all_obs_state.append(agent_pos) # (4,)
            all_actions.append(action)      # (4,)
            all_rewards.append(reward)
            all_dones.append(done)
            
            obs = next_obs
            current_idx += 1
            
            if done or step == MAX_STEPS - 1:
                break
        
        # 记录这条轨迹在哪里结束
        episode_ends.append(current_idx)
    
    # 4. 将所有数据转为 Numpy 数组并存入 Zarr
    print("正在保存数据到 Zarr...")
    
    # 压缩器
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    # 写入数据集
    # 注意：Zarr 通常存储形状为 (Total_Steps, Dim)
    data_group.create_dataset('point_cloud', data=np.array(all_obs_pc), compressor=compressor)
    data_group.create_dataset('state', data=np.array(all_obs_state), compressor=compressor)
    data_group.create_dataset('action', data=np.array(all_actions), compressor=compressor)
    data_group.create_dataset('reward', data=np.array(all_rewards), compressor=compressor)
    data_group.create_dataset('done', data=np.array(all_dones), compressor=compressor)
    
    meta_group.create_dataset('episode_ends', data=np.array(episode_ends), compressor=compressor)
    
    print(f"采集完成！数据保存在: {SAVE_DIR}")
    print(f"总步数: {current_idx}, 总轨迹数: {NUM_EPISODES}")

if __name__ == "__main__":
    collect_data()