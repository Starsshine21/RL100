import os
import argparse
import zarr
import numpy as np
import sys
from tqdm import tqdm
from termcolor import cprint

# 关闭gym警告
import gym
gym.logger.set_level(40)

# 导入你自己的环境封装
from diffusion_policy_3d.env.metaworld.metaworld_wrapper import MetaWorldEnv

# ========== 核心：添加MetaWorld policies目录到Python路径 ==========
# 替换为你本地的Metaworld policies目录路径！
POLICY_DIR = "/nfs_global/S/yangrongzheng/3D-Diffusion-Policy/third_party/Metaworld/metaworld/policies"
sys.path.append(POLICY_DIR)

# 直接导入pick-place-v2的策略（避免动态导入出错）
from sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy

def get_args():
    parser = argparse.ArgumentParser(description='采集MetaWorld pick-place-v2 专家演示数据')
    parser.add_argument('--num-episodes', type=int, default=100, help='采集的演示轮数')
    parser.add_argument('--save-dir', type=str, default='./data/metaworld', help='数据保存目录')
    parser.add_argument('--num-points', type=int, default=512, help='点云数量（和你的env一致）')
    parser.add_argument('--device', type=str, default='cuda:0', help='生成点云的设备')
    return parser.parse_args()

def main():
    args = get_args()
    task_name = "pick-place"  # 固定为pick-place-v2任务
    
    # 1. 初始化pick-place-v2专属专家策略
    cprint("加载 SawyerPickPlaceV2Policy 专家策略...", "blue")
    expert_policy = SawyerPickPlaceV2Policy()
    
    # 2. 初始化环境（复用你的MetaWorldEnv）
    env = MetaWorldEnv(
        task_name=task_name,
        device=args.device,
        num_points=args.num_points,
        use_point_crop=True
    )
    
    # 3. 创建zarr数据集（和RL100训练脚本对齐）
    save_path = os.path.join(args.save_dir, f"{task_name}_demos.zarr")
    root = zarr.open(save_path, mode='w')
    # 预分配数组（单轮最大步数200）
    total_steps = args.num_episodes * 200
    # 初始化数组（字段和你的env输出完全匹配）
    root.create_dataset('obs/agent_pos', shape=(total_steps, 9), dtype=np.float32)
    root.create_dataset('obs/point_cloud', shape=(total_steps, args.num_points, 3), dtype=np.float32)
    root.create_dataset('obs/full_state', shape=(total_steps, 20), dtype=np.float32)
    root.create_dataset('action', shape=(total_steps, 4), dtype=np.float32)
    root.create_dataset('reward', shape=(total_steps,), dtype=np.float32)
    root.create_dataset('done', shape=(total_steps,), dtype=np.bool_)
    root.create_dataset('episode_ends', shape=(args.num_episodes,), dtype=np.int64)
    
    # 4. 采集数据（核心逻辑：策略生成动作 + 环境生成视觉观测）
    step_idx = 0
    for episode in tqdm(range(args.num_episodes), desc=f"采集{task_name}演示数据"):
        # 重置环境
        obs_dict = env.reset()
        done = False
        episode_step = 0
        
        while not done and episode_step < 200:
            # a. 获取策略输入：full_state（和策略的obs解析逻辑对齐）
            raw_state = obs_dict['full_state']  # 策略需要的obs维度由环境保证
            
            # b. 调用策略生成动作（适配SawyerPickPlaceV2Policy的输出）
            action = expert_policy.get_action(raw_state)  # 返回4维动作数组
            
            # c. 环境步进（执行专家动作）
            next_obs_dict, reward, done, info = env.step(action)
            
            # d. 写入zarr数据集
            root['obs/agent_pos'][step_idx] = obs_dict['agent_pos']
            root['obs/point_cloud'][step_idx] = obs_dict['point_cloud'][..., :3]  # 仅保留XYZ
            root['obs/full_state'][step_idx] = raw_state
            root['action'][step_idx] = action
            root['reward'][step_idx] = reward
            root['done'][step_idx] = done
            
            # e. 更新索引和观测
            step_idx += 1
            episode_step += 1
            obs_dict = next_obs_dict
        
        # 记录本轮结束位置
        root['episode_ends'][episode] = step_idx
    
    # 5. 裁剪冗余数据（删除未使用的预留空间）
    for key in ['obs/agent_pos', 'obs/point_cloud', 'obs/full_state', 'action', 'reward', 'done']:
        root[key].resize(step_idx, axis=0)
    
    # 6. 输出采集结果
    cprint(f"\n✅ 数据采集完成！", "green")
    cprint(f"📁 保存路径：{save_path}", "blue")
    cprint(f"📊 总步数：{step_idx} | 采集轮数：{args.num_episodes}", "blue")
    cprint(f"📈 平均每轮步数：{step_idx/args.num_episodes:.1f}", "blue")

if __name__ == '__main__':
    main()