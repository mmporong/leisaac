"""Convert a successful SO101 Isaac Lab HDF5 dataset to LeRobot v3."""

import argparse
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
CAMERA_KEYS = ("front", "wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/so101_mimic_act")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--task", default="Pick up the red cube and place it inside the blue box")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--max-action-step-norm", type=float, default=1.0)
    return parser.parse_args()


def sorted_demo_names(hdf: h5py.File) -> list[str]:
    return sorted(hdf["data"], key=lambda name: int(name.rsplit("_", 1)[1]))


def validate_demo(demo: h5py.Group, max_action_step_norm: float) -> int:
    if not bool(demo.attrs.get("success", False)):
        raise ValueError(f"dataset contains a non-successful episode: {demo.name}")

    frame_count = len(demo["actions"])
    expected_shapes = {
        "actions": (frame_count, 6),
        "obs/joint_pos": (frame_count, 6),
    }
    for key, shape in expected_shapes.items():
        if demo[key].shape != shape:
            raise ValueError(f"unexpected {key} shape in {demo.name}: {demo[key].shape}, expected {shape}")
        if not np.isfinite(demo[key][...]).all():
            raise ValueError(f"non-finite values in {demo.name}/{key}")

    for camera in CAMERA_KEYS:
        images = demo[f"obs/{camera}"]
        if images.shape != (frame_count, 84, 84, 3) or images.dtype != np.uint8:
            raise ValueError(
                f"unexpected {camera} images in {demo.name}: shape={images.shape}, dtype={images.dtype}"
            )
    action_step_norms = np.linalg.norm(np.diff(demo["actions"][...], axis=0), axis=1)
    if len(action_step_norms) and float(action_step_norms.max()) > max_action_step_norm:
        raise ValueError(
            f"action discontinuity in {demo.name}: max step norm={action_step_norms.max():.6f} "
            f"> limit={max_action_step_norm:.6f}"
        )
    return frame_count


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("max-episodes must be positive")
    if args.max_action_step_norm <= 0:
        raise ValueError("max-action-step-norm must be positive")
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")

    with h5py.File(input_path, "r") as hdf:
        demo_names = sorted_demo_names(hdf)
        if args.max_episodes is not None:
            demo_names = demo_names[: args.max_episodes]
        if not demo_names:
            raise ValueError("no demonstrations selected")

        frame_counts = [
            validate_demo(hdf[f"data/{name}"], args.max_action_step_norm) for name in demo_names
        ]
        features = {
            "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
            "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
            **{
                f"observation.images.{camera}": {
                    "dtype": "video",
                    "shape": (84, 84, 3),
                    "names": ["height", "width", "channel"],
                }
                for camera in CAMERA_KEYS
            },
        }
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            robot_type="so101_follower",
            features=features,
            root=output_root,
            use_videos=True,
            image_writer_threads=args.image_writer_threads,
            encoder_threads=2,
            batch_encoding_size=1,
        )

        for episode_index, (name, frame_count) in enumerate(zip(demo_names, frame_counts, strict=True)):
            demo = hdf[f"data/{name}"]
            for frame_index in range(frame_count):
                dataset.add_frame(
                    {
                        "action": demo["actions"][frame_index].astype(np.float32),
                        "observation.state": demo["obs/joint_pos"][frame_index].astype(np.float32),
                        "observation.images.front": demo["obs/front"][frame_index],
                        "observation.images.wrist": demo["obs/wrist"][frame_index],
                        "task": args.task,
                    }
                )
            dataset.save_episode()
            print(
                f"Saved episode {episode_index + 1}/{len(demo_names)}: {name} ({frame_count} frames)",
                flush=True,
            )

        dataset.finalize()

    print(f"Converted {len(demo_names)} episodes and {sum(frame_counts)} frames to {output_root}")


if __name__ == "__main__":
    main()
