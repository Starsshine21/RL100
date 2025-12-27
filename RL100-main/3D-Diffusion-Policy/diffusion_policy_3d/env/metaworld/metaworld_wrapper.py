import torch
import gym
import numpy as np
import metaworld
from gym import spaces
from termcolor import cprint
from diffusion_policy_3d.gym_util.mujoco_point_cloud import PointCloudGenerator

# ================== 新增的 CPU 采样函数 ==================
# 放在类外面，避免类方法的缩进混淆
def cpu_point_cloud_sampling(point_cloud, num_points, method='fps'):
    """
    CPU 版本的点云采样 (修复版：兼容 XYZ 和 XYZRGB)
    """
    num_points = int(num_points)
    # 1. 补全点数
    if point_cloud.shape[0] < num_points:
        indices = np.random.choice(point_cloud.shape[0], num_points, replace=True)
        return point_cloud[indices]
    elif point_cloud.shape[0] == num_points:
        return point_cloud

    # 2. 随机采样
    if method == 'random':
        idxs = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        return point_cloud[idxs]

    # 3. 最远点采样 (FPS)
    elif method == 'fps':
        # 转为 Tensor
        points = torch.from_numpy(point_cloud).float()
        # [核心修复] 只提取前3维(XYZ)来计算几何距离，忽略RGB
        xyz = points[:, :3] 
        
        N = points.shape[0]
        centroids = torch.zeros(num_points, dtype=torch.long)
        distance = torch.ones(N) * 1e10
        farthest = torch.randint(0, N, (1,)).item()
        
        for i in range(num_points):
            centroids[i] = farthest
            # [核心修复] 只取 XYZ 计算距离
            centroid = xyz[farthest, :].view(1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.argmax(distance, -1).item()
            
        # 返回原始数据（包含RGB）
        return point_cloud[centroids.numpy()]
        
    else:
        raise NotImplementedError(f"Method {method} not implemented")

        
TASK_BOUDNS = {
    'default': [-0.5, -1.5, -0.795, 1, -0.4, 100],
}

class MetaWorldEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, device="cuda:0",
                 use_point_crop=True,
                 num_points=1024,
                 use_sparse_reward=False,  # 新增：是否使用稀疏奖励
                 sparse_reward_value=1.0,  # 新增：稀疏奖励的值
                 max_episode_steps=200,    # 新增：最大episode步数
                 ):
        super(MetaWorldEnv, self).__init__()

        # === 修复 1: Task Name 处理逻辑 ===
        task_name = task_name.replace('_', '-')
        if '-v2' not in task_name:
            task_name += '-v2'
        if 'goal-observable' not in task_name:
            task_name += '-goal-observable'
        
        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        self.env._freeze_rand_vec = False

        self.env.sim.model.cam_pos[2] = [0.6, 0.295, 0.8]
        self.env.sim.model.vis.map.znear = 0.1
        self.env.sim.model.vis.map.zfar = 1.5
        
        # === 修复 2: Device 处理逻辑 (兼容 CPU) ===
        if device == 'cpu':
            self.device_id = -1
        else:
            self.device_id = int(device.split(":")[-1])
        
        self.image_size = 128
        
        self.pc_generator = PointCloudGenerator(sim=self.env.sim, cam_names=['corner2'], img_size=self.image_size)
        self.use_point_crop = use_point_crop
        cprint("[MetaWorldEnv] use_point_crop: {}".format(self.use_point_crop), "cyan")
        self.num_points = num_points

        # === 新增：稀疏奖励配置 ===
        self.use_sparse_reward = use_sparse_reward
        self.sparse_reward_value = sparse_reward_value
        if self.use_sparse_reward:
            cprint(f"[MetaWorldEnv] 使用稀疏奖励模式: success={sparse_reward_value}, fail=0.0", "yellow")
        
        x_angle = 61.4
        y_angle = -7
        self.pc_transform = np.array([
            [1, 0, 0],
            [0, np.cos(np.deg2rad(x_angle)), np.sin(np.deg2rad(x_angle))],
            [0, -np.sin(np.deg2rad(x_angle)), np.cos(np.deg2rad(x_angle))]
        ]) @ np.array([
            [np.cos(np.deg2rad(y_angle)), 0, np.sin(np.deg2rad(y_angle))],
            [0, 1, 0],
            [-np.sin(np.deg2rad(y_angle)), 0, np.cos(np.deg2rad(y_angle))]
        ])
        
        self.pc_scale = np.array([1, 1, 1])
        self.pc_offset = np.array([0, 0, 0])
        if task_name in TASK_BOUDNS:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS[task_name]
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS['default']
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]

        # 使用传入的max_episode_steps参数，确保不为None
        if max_episode_steps is None:
            max_episode_steps = 200  # 默认值
            cprint(f"[MetaWorldEnv] max_episode_steps为None，使用默认值200", "yellow")
        self.episode_length = self._max_episode_steps = max_episode_steps
        self.action_space = self.env.action_space
        self.obs_sensor_dim = self.get_robot_state().shape[0]

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32
            ),
            'depth': spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
            'full_state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(39, ), # MetaWorld通常是39维
                dtype=np.float32
            ),
        })

    @staticmethod
    def get_pc_transform(task_name):
        x_angle = 61.4
        y_angle = -7
        pc_transform = np.array([
            [1, 0, 0],
            [0, np.cos(np.deg2rad(x_angle)), np.sin(np.deg2rad(x_angle))],
            [0, -np.sin(np.deg2rad(x_angle)), np.cos(np.deg2rad(x_angle))]
        ]) @ np.array([
            [np.cos(np.deg2rad(y_angle)), 0, np.sin(np.deg2rad(y_angle))],
            [0, 1, 0],
            [-np.sin(np.deg2rad(y_angle)), 0, np.cos(np.deg2rad(y_angle))]
        ])
        return pc_transform
        
    def get_robot_state(self):
        eef_pos = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env._get_site_pos('rightEndEffector'),
            self.env._get_site_pos('leftEndEffector')
        )
        return np.concatenate([eef_pos, finger_right, finger_left])

    def get_rgb(self):
        img = self.env.sim.render(width=self.image_size, height=self.image_size, camera_name="corner2", device_id=self.device_id)
        return img

    def render_high_res(self, resolution=1024):
        img = self.env.sim.render(width=resolution, height=resolution, camera_name="corner2", device_id=self.device_id)
        return img
    
    # === 修复 3: 使用 CPU 采样函数的 get_point_cloud ===
    def get_point_cloud(self, use_rgb=True):
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(device_id=self.device_id)
        
        if not use_rgb:
            point_cloud = point_cloud[..., :3]
        
        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        if self.pc_offset is not None:    
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset
        
        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]

        # 调用上面定义的 CPU 函数，不依赖 mjpc_wrapper
        point_cloud = cpu_point_cloud_sampling(point_cloud, self.num_points, 'fps')
        
        depth = depth[::-1]
        return point_cloud, depth
        
    def get_visual_obs(self):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
        }
        return obs_dict
            
    def step(self, action: np.array):
        raw_state, reward, done, env_info = self.env.step(action)
        self.cur_step += 1

        # === 稀疏奖励处理 ===
        if self.use_sparse_reward:
            # 只在成功时给予奖励，否则为0
            success = env_info.get('success', 0.0)
            if success > 0:
                reward = self.sparse_reward_value
            else:
                reward = 0.0

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_state,
        }

        done = done or self.cur_step >= self.episode_length
        return obs_dict, reward, done, env_info

    def reset(self):
        self.env.reset()
        self.env.reset_model()
        raw_obs = self.env.reset()
        self.cur_step = 0

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_obs,
        }
        return obs_dict

    def seed(self, seed=None):
        pass

    def set_seed(self, seed=None):
        pass

    def render(self, mode='rgb_array'):
        img = self.get_rgb()
        return img

    def close(self):
        pass