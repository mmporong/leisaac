"""Resize Robomimic RGB observations while preserving all other HDF5 data."""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--cameras", nargs="+", default=["front", "wrist"])
    return parser.parse_args()


def copy_group(source: h5py.Group, target: h5py.Group, args: argparse.Namespace) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value

    for name, item in source.items():
        if isinstance(item, h5py.Group):
            copy_group(item, target.create_group(name), args)
            continue

        is_camera = item.parent.name.endswith("/obs") and name in args.cameras
        if not is_camera:
            source.copy(name, target)
            continue

        shape = (item.shape[0], args.height, args.width, item.shape[-1])
        resized = target.create_dataset(
            name,
            shape=shape,
            dtype=item.dtype,
            chunks=(1, args.height, args.width, item.shape[-1]),
            compression="gzip",
            compression_opts=4,
        )
        for index in range(item.shape[0]):
            resized[index] = cv2.resize(
                item[index], (args.width, args.height), interpolation=cv2.INTER_AREA
            )
        for key, value in item.attrs.items():
            resized.attrs[key] = value


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(input_path, "r") as source, h5py.File(output_path, "w") as target:
            copy_group(source, target, args)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise

    print(f"Resized {args.cameras} to {args.width}x{args.height}: {output_path}")


if __name__ == "__main__":
    main()
