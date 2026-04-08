import zarr
import numpy as np
import matplotlib.pyplot as plt

path = "3D-Diffusion-Policy/data/metaworld_dial-turn_expert.zarr"

print("Loading dataset:", path)
z = zarr.open(path, 'r')

data = z['data']
meta = z['meta']

img = data['img'][:]
state = data['state'][:]
full_state = data['full_state'][:]
point_cloud = data['point_cloud'][:]
depth = data['depth'][:]
action = data['action'][:]
reward = data['reward'][:]
done = data['done'][:]

episode_ends = meta['episode_ends'][:]

print("\n================ DATA SHAPE ================")
print("img:", img.shape)
print("point_cloud:", point_cloud.shape)
print("depth:", depth.shape)
print("state:", state.shape)
print("full_state:", full_state.shape)
print("action:", action.shape)
print("reward:", reward.shape)
print("done:", done.shape)
print("episodes:", len(episode_ends))


print("\n================ VALUE RANGE ================")
print("img range:", img.min(), img.max())
print("point_cloud range:", point_cloud.min(), point_cloud.max())
print("depth range:", depth.min(), depth.max())
print("state range:", state.min(), state.max())
print("action range:", action.min(), action.max())
print("reward range:", reward.min(), reward.max())


print("\n================ NaN CHECK ================")

def check_nan(name, arr):
    if np.isnan(arr).any():
        print(f"{name}: contains NaN")
    else:
        print(f"{name}: OK")

check_nan("state", state)
check_nan("action", action)
check_nan("point_cloud", point_cloud)
check_nan("depth", depth)


print("\n================ REWARD CHECK ================")

print("reward unique values:", np.unique(reward))
print("total reward=1:", np.sum(reward == 1))
print("reward==done:", np.all(reward == done))


print("\n================ EPISODE CHECK ================")

start = 0
lengths = []
errors = 0

for i, end in enumerate(episode_ends):

    ep_reward = reward[start:end]
    ep_done = done[start:end]

    lengths.append(len(ep_reward))

    if ep_reward[-1] != 1:
        print(f"Episode {i}: last reward not 1")
        errors += 1

    if np.sum(ep_reward) != 1:
        print(f"Episode {i}: reward count != 1")
        errors += 1

    if ep_done[-1] != 1:
        print(f"Episode {i}: done last step not 1")
        errors += 1

    start = end


print("\n================ EPISODE STATS ================")

print("min length:", np.min(lengths))
print("max length:", np.max(lengths))
print("mean length:", np.mean(lengths))

print("episode errors:", errors)


print("\n================ VISUALIZE REWARD ================")

plt.figure()
plt.plot(reward[:1000])
plt.title("Reward timeline (first 1000 steps)")
plt.show()