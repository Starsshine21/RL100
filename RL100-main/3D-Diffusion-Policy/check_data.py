import zarr
import numpy as np

# 你的数据路径
DATA_PATH = './data/metaworld_wrapper_data'

def inspect_zarr():
    print(f"正在读取数据: {DATA_PATH} ...\n")
    
    try:
        # 打开根目录，mode='r' 表示只读
        root = zarr.open(DATA_PATH, mode='r')
        
        # 1. 检查 data 组
        print("=== Data Group (主要数据) ===")
        data_group = root['data']
        
        # 遍历里面的每个数组
        for key in data_group.keys():
            arr = data_group[key]
            print(f"字段: {key:<15} | 形状: {str(arr.shape):<15} | 类型: {arr.dtype}")
            
            # 简单检查数值是否全为0（防止采集失败全是空数据）
            # 注意：reward 或 done 全为 0 是可能的，但 point_cloud 全为 0 肯定不对
            if arr.shape[0] > 0:
                sample_data = arr[0] # 取第一帧看看
                is_all_zero = np.all(sample_data == 0)
                if is_all_zero and key in ['point_cloud', 'action']:
                    print(f"  [警告] {key} 的第一帧全是 0，可能采集有问题！")
                else:
                    print(f"  [正常] 数据非空。")
        
        print("-" * 30)

        # 2. 检查 meta 组
        print("=== Meta Group (元数据) ===")
        meta_group = root['meta']
        episode_ends = meta_group['episode_ends'][:]
        print(f"episode_ends: {episode_ends}")
        print(f"一共采集了 {len(episode_ends)} 条轨迹")
        print(f"总步数 (Total Steps): {episode_ends[-1]}")
        
    except Exception as e:
        print(f"读取出错: {e}")
        print("可能路径不对，或者文件损坏。")

if __name__ == "__main__":
    inspect_zarr()