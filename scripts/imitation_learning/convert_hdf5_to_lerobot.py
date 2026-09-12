"""Convert a successful SO101 Isaac Lab HDF5 dataset to LeRobot v3."""

import argparse
import hashlib
import json
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
DEFAULT_IMAGE_SIZE = 84
MAX_IMAGE_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/so101_mimic_act")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--task", default="Pick up the red cube and place it inside the blue box")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Skip this many numerically sorted episodes.",
    )
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--max-action-step-norm", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--action-source",
        choices=("stored", "next_observed"),
        default="stored",
        help="Use stored 6D actions or derive actions from the next observed joint position.",
    )
    return parser.parse_args()


def sorted_demo_names(hdf: h5py.File) -> list[str]:
    return sorted(hdf["data"], key=lambda name: int(name.rsplit("_", 1)[1]))


def validate_image_size(image_size: int) -> int:
    if not 1 <= image_size <= MAX_IMAGE_SIZE:
        raise ValueError(f"image-size must be between 1 and {MAX_IMAGE_SIZE}")
    return image_size


def get_demo_actions(demo: h5py.Group, action_source: str) -> np.ndarray:
    joint_pos = np.asarray(demo["obs/joint_pos"][...])
    frame_count = len(joint_pos)
    raw_actions = np.asarray(demo["actions"][...])
    if raw_actions.ndim != 2 or len(raw_actions) != frame_count:
        raise ValueError(f"actions length does not match observations or invalid rank in {demo.name}")
    if not np.isfinite(raw_actions).all():
        raise ValueError(f"non-finite raw actions in {demo.name}")
    if action_source == "stored":
        actions = raw_actions
    elif action_source == "next_observed":
        if frame_count == 0:
            actions = joint_pos.copy()
        else:
            actions = np.concatenate((joint_pos[1:], joint_pos[-1:]), axis=0)
    else:
        raise ValueError(f"unsupported action source: {action_source}")
    return actions


def build_features(image_size: int) -> dict:
    return {
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        **{
            f"observation.images.{camera}": {
                "dtype": "video",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            }
            for camera in CAMERA_KEYS
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_provenance(
    input_path: Path,
    input_sha256: str,
    action_source: str,
    image_size: int,
    demo_names: list[str],
    frame_counts: list[int],
) -> dict:
    return {
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "action_source": action_source,
        "image_size": image_size,
        "episode_names": demo_names,
        "frame_counts": frame_counts,
        "episodes": [
            {"name": name, "frame_count": frame_count}
            for name, frame_count in zip(demo_names, frame_counts, strict=True)
        ],
        "episode_count": len(demo_names),
        "frame_count": sum(frame_counts),
    }


def validate_demo(
    demo: h5py.Group,
    max_action_step_norm: float,
    image_size: int = DEFAULT_IMAGE_SIZE,
    action_source: str = "stored",
) -> int:
    if not np.isfinite(max_action_step_norm) or max_action_step_norm <= 0:
        raise ValueError("max-action-step-norm must be finite and positive")
    if not bool(demo.attrs.get("success", False)):
        raise ValueError(f"dataset contains a non-successful episode: {demo.name}")

    frame_count = len(demo["obs/joint_pos"])
    if frame_count < 2:
        raise ValueError(f"episode needs at least two frames: {demo.name}")
    actions = get_demo_actions(demo, action_source)
    expected_shapes = {
        "obs/joint_pos": (frame_count, 6),
    }
    for key, shape in expected_shapes.items():
        if demo[key].shape != shape:
            raise ValueError(
                f"unexpected {key} shape in {demo.name}: {demo[key].shape}, expected {shape}"
            )
        if not np.isfinite(demo[key][...]).all():
            raise ValueError(f"non-finite values in {demo.name}/{key}")
    if actions.shape != (frame_count, 6):
        raise ValueError(
            f"unexpected actions shape in {demo.name}: {actions.shape}, expected {(frame_count, 6)}"
        )
    if not np.isfinite(actions).all():
        raise ValueError(f"non-finite values in {demo.name}/actions")

    for camera in CAMERA_KEYS:
        images = demo[f"obs/{camera}"]
        if images.shape != (frame_count, image_size, image_size, 3) or images.dtype != np.uint8:
            raise ValueError(
                f"unexpected {camera} images in {demo.name}: shape={images.shape}, dtype={images.dtype}"
            )
    action_step_norms = np.linalg.norm(np.diff(actions, axis=0), axis=1)
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
    if args.start_episode < 0:
        raise ValueError("start-episode must be nonnegative")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("max-episodes must be positive")
    if not np.isfinite(args.max_action_step_norm) or args.max_action_step_norm <= 0:
        raise ValueError("max-action-step-norm must be finite and positive")
    validate_image_size(args.image_size)
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    input_sha256 = sha256_file(input_path)

    with h5py.File(input_path, "r") as hdf:
        demo_names = sorted_demo_names(hdf)[args.start_episode:]
        if args.max_episodes is not None:
            demo_names = demo_names[: args.max_episodes]
        if not demo_names:
            raise ValueError("no demonstrations selected")

        frame_counts = [
            validate_demo(
                hdf[f"data/{name}"],
                args.max_action_step_norm,
                args.image_size,
                args.action_source,
            )
            for name in demo_names
        ]
        features = build_features(args.image_size)
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

        for episode_index, (name, frame_count) in enumerate(
            zip(demo_names, frame_counts, strict=True)
        ):
            demo = hdf[f"data/{name}"]
            actions = get_demo_actions(demo, args.action_source)
            for frame_index in range(frame_count):
                dataset.add_frame(
                    {
                        "action": actions[frame_index].astype(np.float32),
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

    provenance = build_provenance(
        input_path,
        input_sha256,
        args.action_source,
        args.image_size,
        demo_names,
        frame_counts,
    )
    (output_root / "conversion_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Converted {len(demo_names)} episodes and {sum(frame_counts)} frames to {output_root}")


if __name__ == "__main__":
    main()
