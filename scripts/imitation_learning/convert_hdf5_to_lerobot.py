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
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--repo-id", default="local/so101_mimic_act")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--task", default="Pick up the red cube and place it inside the blue box")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--demo-names", nargs="+",
        help="Explicit demo names selected after a separate audit; cannot be combined with slicing.",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-output", type=Path)
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
        choices=("stored", "next_observed", "recorded_target"),
        default="stored",
        help=(
            "Use stored 6D actions, the next observed joint position, or the recorded "
            "actuator target aligned to the following transition."
        ),
    )
    parser.add_argument(
        "--joint-limits-file",
        type=Path,
        default=None,
        help="Evaluation JSON containing joint limits; required for recorded_target only.",
    )
    return parser.parse_args()


def sorted_demo_names(hdf: h5py.File) -> list[str]:
    return sorted(hdf["data"], key=lambda name: int(name.rsplit("_", 1)[1]))


def validate_image_size(image_size: int) -> int:
    if not 1 <= image_size <= MAX_IMAGE_SIZE:
        raise ValueError(f"image-size must be between 1 and {MAX_IMAGE_SIZE}")
    return image_size


def load_joint_limits(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    expected_names = [name.removesuffix(".pos") for name in JOINT_NAMES]
    if payload.get("joint_names") != expected_names:
        raise ValueError(
            f"joint_names in {resolved} must exactly match {expected_names}"
        )
    lower = np.asarray(payload.get("joint_lower_limits_rad"), dtype=np.float64)
    upper = np.asarray(payload.get("joint_upper_limits_rad"), dtype=np.float64)
    if lower.shape != (6,) or upper.shape != (6,):
        raise ValueError(f"joint limits in {resolved} must each contain exactly 6 values")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError(f"joint limits in {resolved} must be finite")
    if not np.all(lower < upper):
        raise ValueError(f"every lower joint limit must be below its upper limit in {resolved}")
    provenance = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "joint_names": expected_names,
        "joint_lower_limits_rad": lower.tolist(),
        "joint_upper_limits_rad": upper.tolist(),
    }
    return lower, upper, provenance


def get_demo_actions(
    demo: h5py.Group,
    action_source: str,
    joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
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
    elif action_source == "recorded_target":
        if joint_limits is None:
            raise ValueError("recorded_target requires validated joint limits")
        joint_targets = np.asarray(demo["obs/joint_pos_target"][...])
        if joint_targets.shape != (frame_count, 6):
            raise ValueError(
                f"unexpected obs/joint_pos_target shape in {demo.name}: "
                f"{joint_targets.shape}, expected {(frame_count, 6)}"
            )
        if not np.isfinite(joint_targets).all():
            raise ValueError(f"non-finite values in {demo.name}/obs/joint_pos_target")
        lower, upper = joint_limits
        actions = np.clip(joint_targets[1:], lower, upper)
    else:
        raise ValueError(f"unsupported action source: {action_source}")
    return actions


def recorded_target_clipping_stats(
    demo: h5py.Group,
    joint_limits: tuple[np.ndarray, np.ndarray],
) -> dict:
    targets = np.asarray(demo["obs/joint_pos_target"][1:], dtype=np.float64)
    lower, upper = joint_limits
    corrections = np.abs(targets - np.clip(targets, lower, upper))
    clipped = corrections > 0
    return {
        "clipped_frames": int(np.any(clipped, axis=1).sum()),
        "clipped_values": int(clipped.sum()),
        "max_abs_correction_rad": float(corrections.max(initial=0.0)),
        "by_joint": {
            name: {
                "clipped_values": int(clipped[:, index].sum()),
                "max_abs_correction_rad": float(corrections[:, index].max(initial=0.0)),
            }
            for index, name in enumerate(JOINT_NAMES)
        },
    }


def aggregate_clipping_stats(stats: list[dict]) -> dict:
    return {
        "clipped_frames": sum(item["clipped_frames"] for item in stats),
        "clipped_values": sum(item["clipped_values"] for item in stats),
        "max_abs_correction_rad": max(
            (item["max_abs_correction_rad"] for item in stats), default=0.0
        ),
        "by_joint": {
            name: {
                "clipped_values": sum(
                    item["by_joint"][name]["clipped_values"] for item in stats
                ),
                "max_abs_correction_rad": max(
                    (
                        item["by_joint"][name]["max_abs_correction_rad"]
                        for item in stats
                    ),
                    default=0.0,
                ),
            }
            for name in JOINT_NAMES
        },
    }


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
    joint_limits_provenance: dict | None = None,
    clipping_stats: list[dict] | None = None,
) -> dict:
    provenance = {
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
    if action_source == "recorded_target":
        if joint_limits_provenance is None:
            raise ValueError("recorded_target provenance requires joint limits")
        if clipping_stats is None or len(clipping_stats) != len(demo_names):
            raise ValueError("recorded_target provenance requires clipping stats per episode")
        for episode, stats in zip(provenance["episodes"], clipping_stats, strict=True):
            episode["clipping"] = stats
        provenance.update(
            {
                "joint_limits": joint_limits_provenance,
                "target_alignment": "action[t] = obs/joint_pos_target[t+1]",
                "last_source_frame_removed": True,
                "last_source_frame_removal_reason": (
                    "no recorded target exists for a transition after the final observation"
                ),
                "frame_alignment": "state and RGB use source slice [0:T-1], frames 0 through T-2",
                "action_clamping": {
                    "method": "elementwise clip to inclusive joint lower and upper limits",
                    "aggregate": aggregate_clipping_stats(clipping_stats),
                },
            }
        )
    return provenance


def validate_demo(
    demo: h5py.Group,
    max_action_step_norm: float,
    image_size: int = DEFAULT_IMAGE_SIZE,
    action_source: str = "stored",
    joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
) -> int:
    if not np.isfinite(max_action_step_norm) or max_action_step_norm <= 0:
        raise ValueError("max-action-step-norm must be finite and positive")
    if not bool(demo.attrs.get("success", False)):
        raise ValueError(f"dataset contains a non-successful episode: {demo.name}")

    frame_count = len(demo["obs/joint_pos"])
    if frame_count < 2:
        raise ValueError(f"episode needs at least two frames: {demo.name}")
    actions = get_demo_actions(demo, action_source, joint_limits)
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
    output_frame_count = frame_count - 1 if action_source == "recorded_target" else frame_count
    if actions.shape != (output_frame_count, 6):
        raise ValueError(
            f"unexpected actions shape in {demo.name}: {actions.shape}, "
            f"expected {(output_frame_count, 6)}"
        )
    if not np.isfinite(actions).all():
        raise ValueError(f"non-finite values in {demo.name}/actions")

    for camera in CAMERA_KEYS:
        images = demo[f"obs/{camera}"]
        if images.shape != (frame_count, image_size, image_size, 3) or images.dtype != np.uint8:
            raise ValueError(
                f"unexpected {camera} images in {demo.name}: shape={images.shape}, dtype={images.dtype}"
            )
    if action_source == "recorded_target":
        post_step_key = "states/articulation/robot/joint_position"
        if post_step_key not in demo:
            raise ValueError(f"missing required post-step joint positions in {demo.name}")
        post_step_joint_pos = np.asarray(
            demo[post_step_key][...]
        )
        if post_step_joint_pos.shape != (frame_count, 6):
            raise ValueError(
                f"unexpected post-step joint position shape in {demo.name}: "
                f"{post_step_joint_pos.shape}, expected {(frame_count, 6)}"
            )
        if not np.isfinite(post_step_joint_pos).all():
            raise ValueError(f"non-finite post-step joint positions in {demo.name}")
        if not np.allclose(
            post_step_joint_pos[:-1], demo["obs/joint_pos"][1:], rtol=0, atol=1e-6
        ):
            raise ValueError(f"pre/post-step recorder alignment does not match in {demo.name}")
    action_step_norms = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    if len(action_step_norms) and float(action_step_norms.max()) > max_action_step_norm:
        raise ValueError(
            f"action discontinuity in {demo.name}: max step norm={action_step_norms.max():.6f} "
            f"> limit={max_action_step_norm:.6f}"
        )
    return output_frame_count


def audit_demonstrations(
    hdf: h5py.File,
    demo_names: list[str],
    max_action_step_norm: float,
    image_size: int,
    action_source: str,
    joint_limits: tuple[np.ndarray, np.ndarray] | None,
) -> list[dict]:
    results = []
    for name in demo_names:
        try:
            frame_count = validate_demo(
                hdf[f"data/{name}"], max_action_step_norm, image_size,
                action_source, joint_limits,
            )
            results.append({"name": name, "accepted": True, "frame_count": frame_count})
        except Exception as error:
            results.append({"name": name, "accepted": False,
                            "error": f"{type(error).__name__}: {error}"})
    return results


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if args.output_root is None and not args.audit_only:
        raise ValueError("--output-root is required unless --audit-only is used")
    if args.audit_only and args.audit_output is None:
        raise ValueError("--audit-only requires --audit-output")
    if args.demo_names and (args.start_episode != 0 or args.max_episodes is not None):
        raise ValueError("--demo-names cannot be combined with --start-episode or --max-episodes")
    if args.demo_names and len(args.demo_names) != len(set(args.demo_names)):
        raise ValueError("--demo-names must not contain duplicates")
    output_root = args.output_root.expanduser().resolve() if args.output_root else None
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.start_episode < 0:
        raise ValueError("start-episode must be nonnegative")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("max-episodes must be positive")
    if not np.isfinite(args.max_action_step_norm) or args.max_action_step_norm <= 0:
        raise ValueError("max-action-step-norm must be finite and positive")
    validate_image_size(args.image_size)
    joint_limits = None
    joint_limits_provenance = None
    clipping_stats = None
    if args.action_source == "recorded_target":
        if args.joint_limits_file is None:
            raise ValueError(
                "--joint-limits-file is required for --action-source recorded_target"
            )
        lower, upper, joint_limits_provenance = load_joint_limits(args.joint_limits_file)
        joint_limits = (lower, upper)
    if output_root is not None and output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    audit_output = args.audit_output.expanduser().resolve() if args.audit_output else None
    if audit_output is not None and audit_output.exists():
        raise FileExistsError(f"audit output already exists: {audit_output}")
    input_sha256 = sha256_file(input_path)

    with h5py.File(input_path, "r") as hdf:
        all_demo_names = sorted_demo_names(hdf)
        if args.audit_only:
            audit = audit_demonstrations(
                hdf, all_demo_names, args.max_action_step_norm, args.image_size,
                args.action_source, joint_limits,
            )
            assert audit_output is not None
            audit_output.parent.mkdir(parents=True, exist_ok=True)
            audit_output.write_text(json.dumps({
                "schema_version": 1, "input_path": str(input_path),
                "input_sha256": input_sha256, "action_source": args.action_source,
                "max_action_step_norm": args.max_action_step_norm,
                "joint_limits": joint_limits_provenance,
                "episode_count": len(audit),
                "accepted_count": sum(item["accepted"] for item in audit),
                "rejected_count": sum(not item["accepted"] for item in audit),
                "episodes": audit,
            }, indent=2) + "\n", encoding="utf-8")
            print(f"Audited {len(audit)} episodes; no dataset was written", flush=True)
            return
        if args.demo_names:
            missing = sorted(set(args.demo_names) - set(all_demo_names))
            if missing:
                raise ValueError(f"selected demos do not exist: {missing}")
            demo_names = list(args.demo_names)
        else:
            demo_names = all_demo_names[args.start_episode:]
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
                joint_limits,
            )
            for name in demo_names
        ]
        if args.action_source == "recorded_target":
            assert joint_limits is not None
            clipping_stats = [
                recorded_target_clipping_stats(hdf[f"data/{name}"], joint_limits)
                for name in demo_names
            ]
        features = build_features(args.image_size)
        assert output_root is not None
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
            actions = get_demo_actions(demo, args.action_source, joint_limits)
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
        joint_limits_provenance,
        clipping_stats,
    )
    assert output_root is not None
    (output_root / "conversion_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Converted {len(demo_names)} episodes and {sum(frame_counts)} frames to {output_root}")


if __name__ == "__main__":
    main()
