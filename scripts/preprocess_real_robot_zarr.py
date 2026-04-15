#!/usr/bin/env python3
import argparse
import os
import shutil
import time
from typing import Sequence

import numpy as np

try:
    from termcolor import cprint
except ImportError:
    def cprint(msg, *_args, **_kwargs):
        print(msg)


def infer_time_chunk(
    shape: Sequence[int],
    dtype,
    target_chunk_bytes: int,
    max_time_chunk: int,
) -> int:
    if len(shape) == 0:
        return 1
    frame_bytes = int(np.dtype(dtype).itemsize)
    for dim in shape[1:]:
        frame_bytes *= int(dim)
    if frame_bytes <= 0:
        return 1
    time_chunk = max(1, target_chunk_bytes // frame_bytes)
    time_chunk = min(int(shape[0]), int(max_time_chunk), int(time_chunk))
    return max(1, int(time_chunk))


def _cast_like_dtype(array: np.ndarray, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        array = np.clip(np.rint(array), info.min, info.max)
    return array.astype(dtype, copy=False)


def resize_rgb_batch(images: np.ndarray, size):
    images = np.asarray(images)
    if size is None:
        return images.copy()
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    array = tensor.permute(0, 2, 3, 1).cpu().numpy()
    return _cast_like_dtype(array, images.dtype)


def resize_mask_batch(masks: np.ndarray, size):
    masks = np.asarray(masks)
    if size is None:
        return masks.copy()
    import torch
    import torch.nn.functional as F

    mask_scale = 1.0 if float(np.max(masks)) <= 1.5 else 255.0
    tensor = torch.from_numpy(masks).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=size, mode="nearest")
    tensor = (tensor > (0.5 * mask_scale)).float() * mask_scale
    array = tensor.permute(0, 2, 3, 1).cpu().numpy()
    return _cast_like_dtype(array, masks.dtype)


def preprocess_rgbm_batch(rgbm: np.ndarray, size):
    rgbm = np.asarray(rgbm)
    if rgbm.shape[-1] != 4:
        raise ValueError(f"Expected rgbm with 4 channels, got shape={rgbm.shape}")
    rgb = resize_rgb_batch(rgbm[..., :3], size=size)
    mask = resize_mask_batch(rgbm[..., 3:], size=size)
    return np.concatenate([rgb, mask], axis=-1)


def preprocess_rgb_batch(rgb: np.ndarray, size):
    rgb = np.asarray(rgb)
    if rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape={rgb.shape}")
    return resize_rgb_batch(rgb, size=size)


def build_terminal_labels(episode_ends: np.ndarray, total_steps: int):
    reward = np.zeros((total_steps,), dtype=np.float32)
    done = np.zeros((total_steps,), dtype=np.float32)
    valid_ends = episode_ends[(episode_ends > 0) & (episode_ends <= total_steps)]
    reward[valid_ends - 1] = 1.0
    done[valid_ends - 1] = 1.0
    return reward, done


def main():
    parser = argparse.ArgumentParser(
        description="Resize / rechunk an existing real-robot zarr into a training-friendly RL100/QAM dataset."
    )
    parser.add_argument("--input", required=True, help="Input zarr path.")
    parser.add_argument("--output", required=True, help="Output zarr path.")
    parser.add_argument("--action-key", default="action", help="Key under data/ for actions.")
    parser.add_argument("--state-key", default="right_state", help="Key under data/ for proprio state.")
    parser.add_argument("--rgbm-key", default="rgbm", help="Key under data/ for head RGB+mask.")
    parser.add_argument("--wrist-key", default="right_cam_img", help="Key under data/ for wrist RGB.")
    parser.add_argument("--reward-key", default="reward", help="Key under data/ for reward, if present.")
    parser.add_argument("--done-key", default="done", help="Key under data/ for done, if present.")
    parser.add_argument("--episode-ends-key", default="episode_ends", help="Key under meta/ for episode ends.")
    parser.add_argument("--resize-height", type=int, default=84, help="Output image height. Set <=0 to keep raw height.")
    parser.add_argument("--resize-width", type=int, default=84, help="Output image width. Set <=0 to keep raw width.")
    parser.add_argument(
        "--process-batch-size",
        type=int,
        default=32,
        help="How many frames to preprocess per batch.",
    )
    parser.add_argument(
        "--target-chunk-mb",
        type=float,
        default=8.0,
        help="Target chunk size in MB. Only the time dimension is changed.",
    )
    parser.add_argument(
        "--max-time-chunk",
        type=int,
        default=512,
        help="Upper bound for the time chunk length.",
    )
    parser.add_argument(
        "--write-terminal-labels",
        action="store_true",
        help="Write sparse terminal reward/done labels when they do not already exist.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N preprocessing batches. Set <=0 to print only start/end messages.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output path.")
    args = parser.parse_args()

    import zarr

    input_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)
    process_batch_size = max(1, int(args.process_batch_size))
    progress_every = int(args.progress_every)
    target_chunk_bytes = int(max(args.target_chunk_mb, 1e-3) * (1024 ** 2))

    src_root = zarr.open(input_path, mode="r")
    if "data" not in src_root or "meta" not in src_root:
        raise ValueError(f"{input_path} is not a valid replay-buffer zarr: missing data/meta groups.")
    src_data = src_root["data"]
    src_meta = src_root["meta"]

    required_data_keys = [args.action_key, args.state_key, args.rgbm_key, args.wrist_key]
    missing_data = [key for key in required_data_keys if key not in src_data]
    if missing_data:
        raise KeyError(f"Missing required data keys: {missing_data}. Available keys: {list(src_data.keys())}")
    if args.episode_ends_key not in src_meta:
        raise KeyError(
            f"Missing meta/{args.episode_ends_key}. Available meta keys: {list(src_meta.keys())}"
        )

    action_src = src_data[args.action_key]
    state_src = src_data[args.state_key]
    rgbm_src = src_data[args.rgbm_key]
    wrist_src = src_data[args.wrist_key]
    episode_ends = np.asarray(src_meta[args.episode_ends_key][:], dtype=np.int64).reshape(-1)

    total_steps = int(action_src.shape[0])
    for key, arr in {
        args.state_key: state_src,
        args.rgbm_key: rgbm_src,
        args.wrist_key: wrist_src,
    }.items():
        if int(arr.shape[0]) != total_steps:
            raise ValueError(
                f"Length mismatch for data/{key}: expected {total_steps}, got {arr.shape[0]}"
            )
    if len(episode_ends) == 0 or int(episode_ends[-1]) != total_steps:
        raise ValueError(
            f"meta/{args.episode_ends_key} must end at total_steps={total_steps}, "
            f"got {episode_ends[-1] if len(episode_ends) else 'empty'}"
        )

    resize_size = None
    if args.resize_height > 0 and args.resize_width > 0:
        resize_size = (int(args.resize_height), int(args.resize_width))

    if resize_size is None:
        rgbm_height, rgbm_width = int(rgbm_src.shape[1]), int(rgbm_src.shape[2])
        wrist_height, wrist_width = int(wrist_src.shape[1]), int(wrist_src.shape[2])
    else:
        rgbm_height, rgbm_width = resize_size
        wrist_height, wrist_width = resize_size

    if os.path.exists(output_path):
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    dst_root = zarr.group(output_path)
    dst_data = dst_root.create_group("data")
    dst_meta = dst_root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    action_shape = tuple(int(x) for x in action_src.shape)
    state_shape = tuple(int(x) for x in state_src.shape)
    rgbm_shape = (total_steps, rgbm_height, rgbm_width, int(rgbm_src.shape[-1]))
    wrist_shape = (total_steps, wrist_height, wrist_width, int(wrist_src.shape[-1]))

    action_chunks = (
        infer_time_chunk(action_shape, np.float32, target_chunk_bytes, args.max_time_chunk),
        action_shape[1],
    )
    state_chunks = (
        infer_time_chunk(state_shape, np.float32, target_chunk_bytes, args.max_time_chunk),
        state_shape[1],
    )
    rgbm_chunks = (
        infer_time_chunk(rgbm_shape, rgbm_src.dtype, target_chunk_bytes, args.max_time_chunk),
        rgbm_shape[1],
        rgbm_shape[2],
        rgbm_shape[3],
    )
    wrist_chunks = (
        infer_time_chunk(wrist_shape, wrist_src.dtype, target_chunk_bytes, args.max_time_chunk),
        wrist_shape[1],
        wrist_shape[2],
        wrist_shape[3],
    )

    action_dst = dst_data.create_dataset(
        args.action_key,
        shape=action_shape,
        dtype=np.float32,
        chunks=action_chunks,
        compressor=compressor,
        overwrite=True,
    )
    state_dst = dst_data.create_dataset(
        args.state_key,
        shape=state_shape,
        dtype=np.float32,
        chunks=state_chunks,
        compressor=compressor,
        overwrite=True,
    )
    rgbm_dst = dst_data.create_dataset(
        args.rgbm_key,
        shape=rgbm_shape,
        dtype=rgbm_src.dtype,
        chunks=rgbm_chunks,
        compressor=compressor,
        overwrite=True,
    )
    wrist_dst = dst_data.create_dataset(
        args.wrist_key,
        shape=wrist_shape,
        dtype=wrist_src.dtype,
        chunks=wrist_chunks,
        compressor=compressor,
        overwrite=True,
    )

    copy_existing_terminal_labels = (
        args.reward_key in src_data and args.done_key in src_data
    )
    write_terminal_labels = bool(args.write_terminal_labels) and not copy_existing_terminal_labels
    reward_dst = None
    done_dst = None
    if copy_existing_terminal_labels or write_terminal_labels:
        label_chunks = (
            infer_time_chunk((total_steps,), np.float32, target_chunk_bytes, args.max_time_chunk),
        )
        reward_dst = dst_data.create_dataset(
            args.reward_key,
            shape=(total_steps,),
            dtype=np.float32,
            chunks=label_chunks,
            compressor=compressor,
            overwrite=True,
        )
        done_dst = dst_data.create_dataset(
            args.done_key,
            shape=(total_steps,),
            dtype=np.float32,
            chunks=label_chunks,
            compressor=compressor,
            overwrite=True,
        )

    dst_meta.create_dataset(
        args.episode_ends_key,
        data=episode_ends,
        shape=episode_ends.shape,
        dtype=np.int64,
        chunks=(min(max(1, len(episode_ends)), 256),),
        compressor=compressor,
        overwrite=True,
    )

    num_batches = (total_steps + process_batch_size - 1) // process_batch_size
    resize_text = (
        "keep_raw_resolution"
        if resize_size is None
        else f"{resize_size[0]}x{resize_size[1]}"
    )
    cprint(
        f"[preprocess] input={input_path} output={output_path}",
        "cyan",
    )
    cprint(
        f"[preprocess] total_steps={total_steps}, episodes={len(episode_ends)}, "
        f"batch_size={process_batch_size}, batches={num_batches}, resize={resize_text}",
        "cyan",
    )
    cprint(
        f"[preprocess] chunks: action={action_chunks}, right_state={state_chunks}, "
        f"rgbm={rgbm_chunks}, right_cam_img={wrist_chunks}",
        "cyan",
    )

    start_time = time.monotonic()
    for start in range(0, total_steps, process_batch_size):
        end = min(start + process_batch_size, total_steps)
        batch_idx = start // process_batch_size + 1
        sl = slice(start, end)
        action_dst[sl] = np.asarray(action_src[sl], dtype=np.float32)
        state_dst[sl] = np.asarray(state_src[sl], dtype=np.float32)
        rgbm_dst[sl] = preprocess_rgbm_batch(np.asarray(rgbm_src[sl]), size=resize_size)
        wrist_dst[sl] = preprocess_rgb_batch(np.asarray(wrist_src[sl]), size=resize_size)
        if copy_existing_terminal_labels:
            reward_dst[sl] = np.asarray(src_data[args.reward_key][sl], dtype=np.float32)
            done_dst[sl] = np.asarray(src_data[args.done_key][sl], dtype=np.float32)
        if (
            progress_every > 0
            and (batch_idx == 1 or batch_idx == num_batches or batch_idx % progress_every == 0)
        ):
            elapsed = time.monotonic() - start_time
            frames_per_sec = end / max(elapsed, 1e-6)
            remaining = total_steps - end
            eta_sec = remaining / max(frames_per_sec, 1e-6)
            percent = 100.0 * end / max(total_steps, 1)
            cprint(
                f"[preprocess] batch {batch_idx}/{num_batches} "
                f"frames {end}/{total_steps} ({percent:.1f}%) "
                f"elapsed={elapsed:.1f}s eta={eta_sec:.1f}s "
                f"speed={frames_per_sec:.1f} frames/s",
                "cyan",
            )

    if write_terminal_labels:
        reward, done = build_terminal_labels(episode_ends=episode_ends, total_steps=total_steps)
        reward_dst[:] = reward
        done_dst[:] = done

    elapsed = time.monotonic() - start_time
    cprint(f"Saved processed zarr to {output_path}", "green")
    cprint(f"total preprocessing time: {elapsed:.1f}s", "green")
    cprint(f"action: shape={action_shape}, chunks={action_chunks}, dtype=float32", "green")
    cprint(f"right_state: shape={state_shape}, chunks={state_chunks}, dtype=float32", "green")
    cprint(
        f"rgbm: shape={rgbm_shape}, chunks={rgbm_chunks}, dtype={rgbm_src.dtype}",
        "green",
    )
    cprint(
        f"right_cam_img: shape={wrist_shape}, chunks={wrist_chunks}, dtype={wrist_src.dtype}",
        "green",
    )
    if copy_existing_terminal_labels:
        cprint("reward/done: copied from input zarr", "green")
    elif write_terminal_labels:
        cprint("reward/done: synthesized as sparse terminal labels", "green")


if __name__ == "__main__":
    main()
