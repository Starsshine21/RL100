#!/usr/bin/env python3
import argparse
import math
import os
import shutil

import numpy as np
import zarr


def infer_time_chunk(shape, dtype, target_chunk_bytes: int, max_time_chunk: int) -> int:
    if len(shape) == 0:
        return 1
    frame_bytes = int(np.dtype(dtype).itemsize)
    for dim in shape[1:]:
        frame_bytes *= int(dim)
    if frame_bytes <= 0:
        return 1
    time_chunk = max(1, target_chunk_bytes // frame_bytes)
    time_chunk = min(int(shape[0]), int(max_time_chunk), int(time_chunk))
    return max(1, time_chunk)


def main():
    parser = argparse.ArgumentParser(
        description="Rechunk a zarr dataset only along the time/frame dimension."
    )
    parser.add_argument("--input", required=True, help="Input zarr path.")
    parser.add_argument("--output", required=True, help="Output zarr path.")
    parser.add_argument(
        "--target-chunk-mb",
        type=float,
        default=8.0,
        help="Target chunk size in MB. The script will only adjust the first dimension.",
    )
    parser.add_argument(
        "--max-time-chunk",
        type=int,
        default=2048,
        help="Upper bound for the rechunked frame dimension.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists.",
    )
    args = parser.parse_args()

    input_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)
    target_chunk_bytes = int(max(args.target_chunk_mb, 1e-3) * (1024 ** 2))

    src_root = zarr.open(input_path, mode="r")
    if "data" not in src_root or "meta" not in src_root:
        raise ValueError(f"{input_path} is not a valid replay-buffer zarr: missing data/meta groups.")

    if os.path.exists(output_path):
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    dst_root = zarr.group(output_path)
    dst_data = dst_root.create_group("data")
    dst_meta = dst_root.create_group("meta")

    print("== Copy meta ==")
    for key, src_arr in src_root["meta"].items():
        meta_value = src_arr[:]
        meta_chunks = src_arr.shape if len(src_arr.shape) > 0 else None
        dst_meta.create_dataset(
            key,
            data=meta_value,
            shape=src_arr.shape,
            dtype=src_arr.dtype,
            chunks=meta_chunks,
            compressor=src_arr.compressor,
            overwrite=True,
        )
        print(f"meta/{key}: shape={src_arr.shape}, dtype={src_arr.dtype}, chunks={meta_chunks}")

    print("\n== Rechunk data ==")
    for key, src_arr in src_root["data"].items():
        time_chunk = infer_time_chunk(
            shape=src_arr.shape,
            dtype=src_arr.dtype,
            target_chunk_bytes=target_chunk_bytes,
            max_time_chunk=args.max_time_chunk,
        )
        new_chunks = (time_chunk,) + tuple(src_arr.shape[1:])
        chunk_bytes = int(np.prod(new_chunks) * np.dtype(src_arr.dtype).itemsize)

        dst_arr = dst_data.create_dataset(
            key,
            shape=src_arr.shape,
            dtype=src_arr.dtype,
            chunks=new_chunks,
            compressor=src_arr.compressor,
            overwrite=True,
        )

        total_steps = int(src_arr.shape[0])
        for start in range(0, total_steps, time_chunk):
            end = min(start + time_chunk, total_steps)
            dst_arr[start:end] = src_arr[start:end]

        print(
            f"data/{key}: shape={src_arr.shape}, dtype={src_arr.dtype}, "
            f"old_chunks={src_arr.chunks}, new_chunks={new_chunks}, "
            f"chunk_mb={chunk_bytes / (1024 ** 2):.2f}"
        )

    print("\nDone.")
    print(f"input : {input_path}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
