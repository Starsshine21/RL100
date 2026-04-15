#!/usr/bin/env python3
import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from termcolor import cprint


def _resize_rgb_batch(images: np.ndarray, size):
    if size is None:
        return images.astype(np.float32)
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)


def _resize_mask_batch(masks: np.ndarray, size):
    if size is None:
        return masks.astype(np.float32)
    tensor = torch.from_numpy(masks).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=size, mode="nearest")
    tensor = (tensor > 0.5).float()
    return tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    if np.max(rgb) > 1.5:
        rgb = rgb / 255.0
    return rgb


def preprocess_rgbm(rgbm: np.ndarray, size):
    rgb = _normalize_rgb(rgbm[..., :3])
    mask = rgbm[..., 3:].astype(np.float32)
    if np.max(mask) > 1.5:
        mask = mask / 255.0
    rgb = _resize_rgb_batch(rgb, size=size)
    mask = _resize_mask_batch(mask, size=size)
    return np.concatenate([rgb, mask], axis=-1).astype(np.float32)


def preprocess_rgb(rgb: np.ndarray, size):
    rgb = _normalize_rgb(rgb)
    return _resize_rgb_batch(rgb, size=size).astype(np.float32)


def build_terminal_done(episode_ends: np.ndarray, total_steps: int) -> np.ndarray:
    done = np.zeros((total_steps,), dtype=np.float32)
    valid_ends = episode_ends[(episode_ends > 0) & (episode_ends <= total_steps)]
    done[valid_ends - 1] = 1.0
    return done


def main():
    parser = argparse.ArgumentParser(description="Convert real-robot HDF5 demos to RL-100 zarr format.")
    parser.add_argument("--input", required=True, help="Input HDF5 file path.")
    parser.add_argument("--output", required=True, help="Output zarr directory path.")
    parser.add_argument("--rgbm-key", default="data/rgbm", help="HDF5 dataset path for head RGB+mask.")
    parser.add_argument("--wrist-key", default="data/right_cam_img", help="HDF5 dataset path for wrist RGB.")
    parser.add_argument("--state-key", default="data/right_state", help="HDF5 dataset path for proprio state.")
    parser.add_argument("--action-key", default="data/action", help="HDF5 dataset path for action.")
    parser.add_argument("--episode-ends-key", default="meta/episode_ends", help="HDF5 dataset path for episode ends.")
    parser.add_argument("--resize-height", type=int, default=84, help="Resize image height. Set <=0 to keep raw height.")
    parser.add_argument("--resize-width", type=int, default=84, help="Resize image width. Set <=0 to keep raw width.")
    parser.add_argument("--write-done", action="store_true", help="Also write terminal done labels.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    args = parser.parse_args()

    output_path = os.path.expanduser(args.output)
    if os.path.exists(output_path):
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        import shutil
        shutil.rmtree(output_path)

    resize_size = None
    if args.resize_height > 0 and args.resize_width > 0:
        resize_size = (int(args.resize_height), int(args.resize_width))

    with h5py.File(os.path.expanduser(args.input), "r") as f:
        action = f[args.action_key][:]
        state = f[args.state_key][:]
        rgbm = f[args.rgbm_key][:]
        wrist_rgb = f[args.wrist_key][:]
        episode_ends = f[args.episode_ends_key][:]

    total_steps = int(action.shape[0])
    if state.shape[0] != total_steps or rgbm.shape[0] != total_steps or wrist_rgb.shape[0] != total_steps:
        raise ValueError(
            "Action/state/image lengths do not match: "
            f"action={action.shape[0]}, state={state.shape[0]}, rgbm={rgbm.shape[0]}, wrist={wrist_rgb.shape[0]}"
        )

    episode_ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
    if len(episode_ends) == 0 or int(episode_ends[-1]) != total_steps:
        raise ValueError(
            f"episode_ends must end at total_steps={total_steps}, got {episode_ends[-1] if len(episode_ends) else 'empty'}"
        )

    action = np.asarray(action, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    rgbm = preprocess_rgbm(np.asarray(rgbm), size=resize_size)
    wrist_rgb = preprocess_rgb(np.asarray(wrist_rgb), size=resize_size)

    zarr_root = zarr.group(output_path)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    zarr_data.create_dataset(
        "action",
        data=action,
        chunks=(min(256, total_steps), action.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "right_state",
        data=state,
        chunks=(min(256, total_steps), state.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "rgbm",
        data=rgbm,
        chunks=(min(64, total_steps), rgbm.shape[1], rgbm.shape[2], rgbm.shape[3]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "right_cam_img",
        data=wrist_rgb,
        chunks=(min(64, total_steps), wrist_rgb.shape[1], wrist_rgb.shape[2], wrist_rgb.shape[3]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    if args.write_done:
        done = build_terminal_done(episode_ends, total_steps=total_steps)
        zarr_data.create_dataset(
            "done",
            data=done,
            chunks=(min(256, total_steps),),
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends,
        chunks=(min(256, max(1, len(episode_ends))),),
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    cprint(f"Saved zarr dataset to {output_path}", "green")
    cprint(f"action: {action.shape} [{action.dtype}]", "green")
    cprint(f"right_state: {state.shape} [{state.dtype}]", "green")
    cprint(f"rgbm: {rgbm.shape} [{rgbm.dtype}] range=({rgbm.min():.4f}, {rgbm.max():.4f})", "green")
    cprint(
        f"right_cam_img: {wrist_rgb.shape} [{wrist_rgb.dtype}] range=({wrist_rgb.min():.4f}, {wrist_rgb.max():.4f})",
        "green",
    )
    cprint(f"episode_ends: {episode_ends.shape} [{episode_ends.dtype}]", "green")


if __name__ == "__main__":
    main()
