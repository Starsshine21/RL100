#!/usr/bin/env python3
import argparse
import os
import pickle
import re
import shutil
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
try:
    from termcolor import cprint
except ImportError:
    def cprint(msg, *_args, **_kwargs):
        print(msg)


def natural_key(text: str):
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", text)
    ]


def list_episode_files(input_dir: str, suffix: str = ".pkl") -> List[str]:
    files = []
    for name in os.listdir(input_dir):
        if name.endswith(suffix):
            files.append(os.path.join(input_dir, name))
    files.sort(key=lambda p: natural_key(os.path.basename(p)))
    return files


def infer_time_chunk(shape: Sequence[int], dtype, target_chunk_bytes: int, max_time_chunk: int) -> int:
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


def resize_rgb_batch(images: np.ndarray, size):
    if size is None:
        return images.astype(np.float32, copy=False)
    target_h, target_w = int(size[0]), int(size[1])
    resized = [
        cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        for frame in images
    ]
    return np.stack(resized, axis=0).astype(np.float32)


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32, copy=False)
    if np.max(rgb) > 1.5:
        rgb = rgb / 255.0
    return rgb.astype(np.float32, copy=False)


def preprocess_rgb(rgb: np.ndarray, size, convert_bgr_to_rgb: bool) -> np.ndarray:
    rgb = np.asarray(rgb)
    if convert_bgr_to_rgb:
        rgb = rgb[..., ::-1]
    rgb = normalize_rgb(rgb)
    return resize_rgb_batch(rgb, size=size).astype(np.float32)


def append_constant_mask(rgb: np.ndarray, mask_mode: str) -> np.ndarray:
    if mask_mode == "zeros":
        mask_value = 0.0
    elif mask_mode == "ones":
        mask_value = 1.0
    else:
        raise ValueError(f"Unsupported mask_mode={mask_mode}. Expected 'zeros' or 'ones'.")
    mask = np.full(rgb.shape[:-1] + (1,), fill_value=mask_value, dtype=np.float32)
    return np.concatenate([rgb, mask], axis=-1).astype(np.float32)


def encode_hand(hand_raw: np.ndarray, hand_encoding: str) -> np.ndarray:
    hand_raw = np.asarray(hand_raw, dtype=np.float32)
    if hand_encoding == "raw_0_2000":
        return hand_raw
    if hand_encoding == "normalized_minus1_1":
        return hand_raw / 1000.0 - 1.0
    raise ValueError(
        f"Unsupported hand_encoding={hand_encoding}. "
        "Expected 'normalized_minus1_1' or 'raw_0_2000'."
    )


def select_arm_sequence(episode: Dict, arm_source: str) -> np.ndarray:
    if arm_source == "eef":
        key = "episode_ur5e_pos_eef"
    elif arm_source == "joint":
        key = "episode_ur5e_pos_j"
    else:
        raise ValueError(f"Unsupported arm_source={arm_source}. Expected 'eef' or 'joint'.")
    if key not in episode:
        raise KeyError(f"Episode is missing key '{key}'. Available keys: {list(episode.keys())}")
    return np.asarray(episode[key], dtype=np.float32)


def build_state_sequence(episode: Dict, arm_source: str, hand_encoding: str) -> np.ndarray:
    arm = select_arm_sequence(episode, arm_source=arm_source)
    hand = np.asarray(episode["episode_inspire_hand_pos"], dtype=np.float32)
    hand = encode_hand(hand, hand_encoding=hand_encoding)
    if arm.shape[0] != hand.shape[0]:
        raise ValueError(f"Arm and hand lengths mismatch: arm={arm.shape[0]}, hand={hand.shape[0]}")
    return np.concatenate([arm, hand], axis=-1).astype(np.float32)


def build_action_sequence(state_seq: np.ndarray, action_source: str) -> np.ndarray:
    state_seq = np.asarray(state_seq, dtype=np.float32)
    if action_source in ("current_state", "state"):
        return state_seq.copy()
    if action_source == "next_state":
        action = np.empty_like(state_seq)
        if len(state_seq) == 0:
            return action
        action[:-1] = state_seq[1:]
        action[-1] = state_seq[-1]
        return action
    raise ValueError(
        f"Unsupported action_source={action_source}. "
        "Expected 'next_state' or 'current_state'."
    )


def build_terminal_done(episode_ends: np.ndarray, total_steps: int) -> np.ndarray:
    done = np.zeros((total_steps,), dtype=np.float32)
    valid_ends = episode_ends[(episode_ends > 0) & (episode_ends <= total_steps)]
    done[valid_ends - 1] = 1.0
    return done


def scan_episode_meta(
    episode_files: Sequence[str],
    arm_source: str,
    head_camera_key: str,
    wrist_camera_key: str,
) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    episode_lengths = []
    arm_shape = None
    head_shape = None
    wrist_shape = None

    for path in episode_files:
        with open(path, "rb") as f:
            episode = pickle.load(f)

        state_seq = build_state_sequence(episode, arm_source=arm_source, hand_encoding="raw_0_2000")
        head = np.asarray(episode[head_camera_key])
        wrist = np.asarray(episode[wrist_camera_key])

        length = int(state_seq.shape[0])
        if length <= 0:
            raise ValueError(f"Episode {path} is empty.")
        if head.shape[0] != length or wrist.shape[0] != length:
            raise ValueError(
                f"Image/state length mismatch in {path}: "
                f"state={length}, head={head.shape[0]}, wrist={wrist.shape[0]}"
            )

        episode_lengths.append(length)

        current_arm_shape = tuple(select_arm_sequence(episode, arm_source=arm_source).shape[1:])
        current_head_shape = tuple(head.shape[1:])
        current_wrist_shape = tuple(wrist.shape[1:])

        if arm_shape is None:
            arm_shape = current_arm_shape
            head_shape = current_head_shape
            wrist_shape = current_wrist_shape
        else:
            if current_arm_shape != arm_shape:
                raise ValueError(f"Inconsistent arm shape in {path}: {current_arm_shape} vs {arm_shape}")
            if current_head_shape != head_shape:
                raise ValueError(f"Inconsistent head image shape in {path}: {current_head_shape} vs {head_shape}")
            if current_wrist_shape != wrist_shape:
                raise ValueError(
                    f"Inconsistent wrist image shape in {path}: {current_wrist_shape} vs {wrist_shape}"
                )

    return episode_lengths, arm_shape, head_shape, wrist_shape


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw dex-data-collection .pkl episodes to RL100/QAM zarr format."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing dex-data-collection .pkl episodes.")
    parser.add_argument("--output", required=True, help="Output zarr directory path.")
    parser.add_argument(
        "--arm-source",
        choices=["eef", "joint"],
        default="eef",
        help="Which UR5e signal to use for state/action construction.",
    )
    parser.add_argument(
        "--action-source",
        choices=["next_state", "current_state"],
        default="next_state",
        help="How to derive training actions from the recorded trajectory.",
    )
    parser.add_argument(
        "--head-camera-key",
        default="episode_orbbec_femto_bolt_color",
        help="Episode key used as the third-view / head camera.",
    )
    parser.add_argument(
        "--wrist-camera-key",
        default="episode_l515_color",
        help="Episode key used as the wrist camera.",
    )
    parser.add_argument(
        "--hand-encoding",
        choices=["normalized_minus1_1", "raw_0_2000"],
        default="normalized_minus1_1",
        help="How to store the last 6 hand dimensions in zarr.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["zeros", "ones"],
        default="zeros",
        help="How to synthesize the 4th channel of rgbm from raw RGB collector data.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=224,
        help="Resize output image height. Set <=0 to keep raw height.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=224,
        help="Resize output image width. Set <=0 to keep raw width.",
    )
    parser.add_argument(
        "--convert-bgr-to-rgb",
        action="store_true",
        help="Convert collector BGR frames to RGB before writing zarr.",
    )
    parser.add_argument(
        "--target-chunk-mb",
        type=float,
        default=8.0,
        help="Target zarr chunk size in MB. Only the time dimension is chunked.",
    )
    parser.add_argument(
        "--max-time-chunk",
        type=int,
        default=512,
        help="Upper bound for the written time chunk size.",
    )
    parser.add_argument(
        "--write-done",
        action="store_true",
        help="Also write terminal done labels to data/done.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists.",
    )
    args = parser.parse_args()

    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "convert_dex_data_collection_pkl_to_zarr.py requires `zarr`. "
            "Install the RL100 training dependencies on the target machine first."
        ) from exc

    input_dir = os.path.expanduser(args.input_dir)
    output_path = os.path.expanduser(args.output)
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"{input_dir} is not a directory.")

    if os.path.exists(output_path):
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    resize_size = None
    if args.resize_height > 0 and args.resize_width > 0:
        resize_size = (int(args.resize_height), int(args.resize_width))

    episode_files = list_episode_files(input_dir)
    if not episode_files:
        raise FileNotFoundError(f"No .pkl episodes found under {input_dir}")

    cprint(f"Found {len(episode_files)} episode files.", "cyan")
    episode_lengths, arm_shape, head_shape_raw, wrist_shape_raw = scan_episode_meta(
        episode_files=episode_files,
        arm_source=args.arm_source,
        head_camera_key=args.head_camera_key,
        wrist_camera_key=args.wrist_camera_key,
    )

    total_steps = int(np.sum(episode_lengths))
    episode_ends = np.cumsum(np.asarray(episode_lengths, dtype=np.int64))
    state_dim = int(np.prod(arm_shape) + 6)

    if resize_size is None:
        head_h, head_w = int(head_shape_raw[0]), int(head_shape_raw[1])
        wrist_h, wrist_w = int(wrist_shape_raw[0]), int(wrist_shape_raw[1])
    else:
        head_h, head_w = resize_size
        wrist_h, wrist_w = resize_size

    target_chunk_bytes = int(max(args.target_chunk_mb, 1e-3) * (1024 ** 2))
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    zarr_root = zarr.group(output_path)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    action_shape = (total_steps, state_dim)
    rgbm_shape = (total_steps, head_h, head_w, 4)
    wrist_shape = (total_steps, wrist_h, wrist_w, 3)
    state_shape = (total_steps, state_dim)

    action_chunks = (
        infer_time_chunk(action_shape, np.float32, target_chunk_bytes, args.max_time_chunk),
        state_dim,
    )
    state_chunks = action_chunks
    rgbm_chunks = (
        infer_time_chunk(rgbm_shape, np.float32, target_chunk_bytes, args.max_time_chunk),
        head_h,
        head_w,
        4,
    )
    wrist_chunks = (
        infer_time_chunk(wrist_shape, np.float32, target_chunk_bytes, args.max_time_chunk),
        wrist_h,
        wrist_w,
        3,
    )

    action_arr = zarr_data.create_dataset(
        "action",
        shape=action_shape,
        dtype="float32",
        chunks=action_chunks,
        compressor=compressor,
        overwrite=True,
    )
    state_arr = zarr_data.create_dataset(
        "right_state",
        shape=state_shape,
        dtype="float32",
        chunks=state_chunks,
        compressor=compressor,
        overwrite=True,
    )
    rgbm_arr = zarr_data.create_dataset(
        "rgbm",
        shape=rgbm_shape,
        dtype="float32",
        chunks=rgbm_chunks,
        compressor=compressor,
        overwrite=True,
    )
    wrist_arr = zarr_data.create_dataset(
        "right_cam_img",
        shape=wrist_shape,
        dtype="float32",
        chunks=wrist_chunks,
        compressor=compressor,
        overwrite=True,
    )
    if args.write_done:
        done_arr = zarr_data.create_dataset(
            "done",
            shape=(total_steps,),
            dtype="float32",
            chunks=(infer_time_chunk((total_steps,), np.float32, target_chunk_bytes, args.max_time_chunk),),
            compressor=compressor,
            overwrite=True,
        )
    else:
        done_arr = None

    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends,
        dtype="int64",
        chunks=(min(max(1, len(episode_ends)), 256),),
        compressor=compressor,
        overwrite=True,
    )

    cursor = 0
    for ep_idx, path in enumerate(episode_files):
        with open(path, "rb") as f:
            episode = pickle.load(f)

        state_seq = build_state_sequence(
            episode=episode,
            arm_source=args.arm_source,
            hand_encoding=args.hand_encoding,
        )
        action_seq = build_action_sequence(state_seq, action_source=args.action_source)

        head_rgb = preprocess_rgb(
            np.asarray(episode[args.head_camera_key]),
            size=resize_size,
            convert_bgr_to_rgb=args.convert_bgr_to_rgb,
        )
        wrist_rgb = preprocess_rgb(
            np.asarray(episode[args.wrist_camera_key]),
            size=resize_size,
            convert_bgr_to_rgb=args.convert_bgr_to_rgb,
        )
        rgbm = append_constant_mask(head_rgb, mask_mode=args.mask_mode)

        length = int(state_seq.shape[0])
        sl = slice(cursor, cursor + length)
        action_arr[sl] = action_seq
        state_arr[sl] = state_seq
        rgbm_arr[sl] = rgbm
        wrist_arr[sl] = wrist_rgb

        if done_arr is not None:
            done = np.zeros((length,), dtype=np.float32)
            done[-1] = 1.0
            done_arr[sl] = done

        cursor += length
        cprint(
            f"[{ep_idx + 1}/{len(episode_files)}] {os.path.basename(path)} -> {length} steps",
            "cyan",
        )

    if cursor != total_steps:
        raise RuntimeError(f"Write length mismatch: cursor={cursor}, total_steps={total_steps}")

    cprint(f"Saved zarr dataset to {output_path}", "green")
    cprint(f"episodes: {len(episode_files)}", "green")
    cprint(f"total_steps: {total_steps}", "green")
    cprint(f"arm_source: {args.arm_source}", "green")
    cprint(f"action_source: {args.action_source}", "green")
    cprint(f"hand_encoding: {args.hand_encoding}", "green")
    cprint(
        f"rgbm: shape={rgbm_shape}, chunks={rgbm_chunks}, mask_mode={args.mask_mode}",
        "green",
    )
    cprint(
        f"right_cam_img: shape={wrist_shape}, chunks={wrist_chunks}",
        "green",
    )
    cprint(
        f"right_state/action: shape={state_shape}, chunks={state_chunks}",
        "green",
    )


if __name__ == "__main__":
    main()
