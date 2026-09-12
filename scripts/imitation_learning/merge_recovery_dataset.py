"""Append successful raw recovery episodes to a next-state Robomimic dataset.

Recovery inputs are always treated as training data. Evaluation recovery
episodes must therefore be kept in separate files and must not be passed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import cv2
import h5py
import numpy as np


ACTION_REPRESENTATION = "next_observed_joint_position"
CAMERAS = ("front", "wrist")
LOW_DIM_OBSERVATIONS = ("joint_pos", "joint_vel")
IMAGE_SHAPE = (84, 84, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return (int(name.rsplit("_", 1)[1]), name)
    except (IndexError, ValueError):
        return (2**63 - 1, name)


def _decode_names(dataset: h5py.Dataset) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in dataset[...]
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory_fingerprint(joint_pos: np.ndarray, actions: np.ndarray) -> str:
    """Hash canonical action/state content, independent of source file layout."""
    digest = hashlib.sha256()
    for label, array in ((b"joint_pos", joint_pos), (b"actions", actions)):
        canonical = np.ascontiguousarray(array, dtype="<f4")
        digest.update(label)
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _next_state_actions(joint_pos: np.ndarray) -> np.ndarray:
    actions = np.asarray(joint_pos, dtype=np.float32).copy()
    if len(actions) > 1:
        actions[:-1] = actions[1:]
    return actions


def _previous_actions(actions: np.ndarray) -> np.ndarray:
    previous = np.empty_like(actions)
    previous[0] = 0.0
    if len(actions) > 1:
        previous[1:] = actions[:-1]
    return previous


def _validate_baseline(hdf: h5py.File) -> tuple[list[str], list[str], list[str]]:
    if hdf.attrs.get("action_representation") != ACTION_REPRESENTATION:
        raise ValueError(
            "baseline action_representation must be "
            f"{ACTION_REPRESENTATION!r}, got {hdf.attrs.get('action_representation')!r}"
        )
    for key in ("data", "mask/train", "mask/valid"):
        if key not in hdf:
            raise ValueError(f"baseline is missing {key}")

    demos = sorted(hdf["data"], key=_demo_sort_key)
    train = _decode_names(hdf["mask/train"])
    valid = _decode_names(hdf["mask/valid"])
    demo_set = set(demos)
    if len(set(train)) != len(train) or len(set(valid)) != len(valid) or set(train) & set(valid):
        raise ValueError("baseline masks contain duplicate or overlapping episode names")
    if not set(train + valid) <= demo_set:
        raise ValueError("baseline masks reference missing episodes")
    if not demos:
        raise ValueError("baseline contains no demonstrations")
    return demos, train, valid


def _load_recovery_demo(demo: h5py.Group) -> dict[str, np.ndarray]:
    required = [f"obs/{key}" for key in (*LOW_DIM_OBSERVATIONS, *CAMERAS)]
    missing = [key for key in required if key not in demo]
    if missing:
        raise ValueError(f"{demo.name} is missing required datasets: {missing}")

    joint_pos = demo["obs/joint_pos"][...]
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 6 or len(joint_pos) == 0:
        raise ValueError(f"{demo.name}/obs/joint_pos must have shape (T, 6), got {joint_pos.shape}")
    frame_count = len(joint_pos)
    joint_vel = demo["obs/joint_vel"][...]
    if joint_vel.shape != (frame_count, 6):
        raise ValueError(
            f"{demo.name}/obs/joint_vel must have shape {(frame_count, 6)}, got {joint_vel.shape}"
        )
    for key, array in (("joint_pos", joint_pos), ("joint_vel", joint_vel)):
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"{demo.name}/obs/{key} contains non-finite or non-numeric values")

    result = {
        "joint_pos": np.asarray(joint_pos, dtype=np.float32),
        "joint_vel": np.asarray(joint_vel, dtype=np.float32),
    }
    for camera in CAMERAS:
        images = demo[f"obs/{camera}"]
        if images.ndim != 4 or images.shape[0] != frame_count or images.shape[-1] != 3:
            raise ValueError(
                f"{demo.name}/obs/{camera} must have shape (T, H, W, 3), got {images.shape}"
            )
        if images.dtype != np.uint8:
            raise ValueError(f"{demo.name}/obs/{camera} must be uint8, got {images.dtype}")
    return result


def _copy_resized_images(
    source: h5py.Dataset, target_obs: h5py.Group, name: str, batch_size: int
) -> None:
    frame_count = source.shape[0]
    target = target_obs.create_dataset(
        name,
        shape=(frame_count, *IMAGE_SHAPE),
        dtype=np.uint8,
        chunks=(1, *IMAGE_SHAPE),
        compression="gzip",
        compression_opts=4,
    )
    for start in range(0, frame_count, batch_size):
        stop = min(start + batch_size, frame_count)
        frames = source[start:stop]
        if source.shape[1:] == IMAGE_SHAPE:
            target[start:stop] = frames
        else:
            target[start:stop] = np.stack(
                [cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) for frame in frames]
            )


def _write_recovery_demo(
    source: h5py.Group,
    target_data: h5py.Group,
    output_name: str,
    arrays: dict[str, np.ndarray],
    actions: np.ndarray,
    batch_size: int,
) -> None:
    target = target_data.create_group(output_name)
    target.attrs["num_samples"] = len(actions)
    target.attrs["success"] = True
    if "seed" in source.attrs:
        target.attrs["seed"] = source.attrs["seed"]
    obs = target.create_group("obs")
    target.create_dataset("actions", data=actions)
    obs.create_dataset("actions", data=_previous_actions(actions))
    for key in LOW_DIM_OBSERVATIONS:
        obs.create_dataset(key, data=arrays[key])
    for camera in CAMERAS:
        _copy_resized_images(source[f"obs/{camera}"], obs, camera, batch_size)


def merge_recovery_dataset(
    baseline: Path,
    recovery_paths: list[Path],
    output: Path,
    manifest: Path | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Create an atomic merged dataset and provenance manifest."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    baseline = baseline.expanduser().resolve()
    recovery_paths = [path.expanduser().resolve() for path in recovery_paths]
    output = output.expanduser().resolve()
    manifest = (manifest or output.with_suffix(".manifest.json")).expanduser().resolve()
    if output == manifest:
        raise ValueError("output and manifest paths must be different")
    if output.exists() or manifest.exists():
        existing = output if output.exists() else manifest
        raise FileExistsError(f"output already exists: {existing}")
    if output in [baseline, *recovery_paths] or manifest in [baseline, *recovery_paths]:
        raise ValueError("output and manifest must not replace an input file")
    if len(set(recovery_paths)) != len(recovery_paths):
        raise ValueError("the same recovery path was supplied more than once")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output_tmp: Path | None = None
    manifest_tmp: Path | None = None
    output_installed = False
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as tmp:
            output_tmp = Path(tmp.name)
        shutil.copyfile(baseline, output_tmp)

        source_records: list[dict[str, Any]] = []
        accepted: list[dict[str, Any]] = []
        with h5py.File(output_tmp, "r+") as merged:
            baseline_demos, baseline_train, baseline_valid = _validate_baseline(merged)
            seen: dict[str, str] = {}
            for name in baseline_demos:
                demo = merged[f"data/{name}"]
                joint_pos = demo["obs/joint_pos"][...]
                actions = demo["actions"][...]
                if joint_pos.shape != actions.shape or joint_pos.ndim != 2 or joint_pos.shape[1] != 6:
                    raise ValueError(f"baseline {demo.name} does not have matching (T, 6) state/actions")
                if not np.isfinite(joint_pos).all() or not np.isfinite(actions).all():
                    raise ValueError(f"baseline {demo.name} contains non-finite state/actions")
                seen[_trajectory_fingerprint(joint_pos, actions)] = name

            next_index = max(_demo_sort_key(name)[0] for name in baseline_demos) + 1
            for recovery_path in recovery_paths:
                record: dict[str, Any] = {
                    "path": str(recovery_path),
                    "sha256": _sha256_file(recovery_path),
                    "episodes": [],
                }
                with h5py.File(recovery_path, "r") as source_hdf:
                    if "data" not in source_hdf:
                        raise ValueError(f"recovery source is missing data group: {recovery_path}")
                    for source_name in sorted(source_hdf["data"], key=_demo_sort_key):
                        source_demo = source_hdf[f"data/{source_name}"]
                        seed = source_demo.attrs.get("seed")
                        episode: dict[str, Any] = {
                            "source_demo": source_name,
                            "seed": int(seed) if seed is not None else None,
                        }
                        success = source_demo.attrs.get("success", False)
                        if not isinstance(success, (bool, np.bool_)) or not bool(success):
                            episode.update(status="skipped", reason="success_attr_is_not_true")
                            record["episodes"].append(episode)
                            continue

                        arrays = _load_recovery_demo(source_demo)
                        actions = _next_state_actions(arrays["joint_pos"])
                        fingerprint = _trajectory_fingerprint(arrays["joint_pos"], actions)
                        episode["trajectory_sha256"] = fingerprint
                        if fingerprint in seen:
                            episode.update(
                                status="skipped",
                                reason="duplicate_action_state",
                                duplicate_of=seen[fingerprint],
                            )
                            record["episodes"].append(episode)
                            continue

                        output_name = f"demo_{next_index}"
                        next_index += 1
                        _write_recovery_demo(
                            source_demo, merged["data"], output_name, arrays, actions, batch_size
                        )
                        seen[fingerprint] = output_name
                        episode.update(status="accepted", output_demo=output_name, num_samples=len(actions))
                        record["episodes"].append(episode)
                        accepted.append(
                            {
                                "output_demo": output_name,
                                "source_path": str(recovery_path),
                                "source_sha256": record["sha256"],
                                **episode,
                            }
                        )
                source_records.append(record)

            if not accepted:
                raise ValueError("no valid, unique successful recovery episodes were found")

            new_train = baseline_train + [entry["output_demo"] for entry in accepted]
            train_dataset = merged["mask/train"]
            train_attrs = dict(train_dataset.attrs)
            del merged["mask/train"]
            string_dtype = h5py.string_dtype(encoding="utf-8")
            new_dataset = merged["mask"].create_dataset(
                "train", data=np.asarray(new_train, dtype=string_dtype)
            )
            for key, value in train_attrs.items():
                new_dataset.attrs[key] = value
            merged["data"].attrs["total"] = sum(
                int(demo.attrs.get("num_samples", len(demo["actions"])))
                for demo in merged["data"].values()
            )

        manifest_data: dict[str, Any] = {
            "schema_version": 1,
            "output": str(output),
            "action_representation": ACTION_REPRESENTATION,
            "recovery_policy": "successful unique episodes are appended to train only",
            "recovery_schema": {
                "actions": "obs/joint_pos[t+1], final frame repeated",
                "obs/actions": "previous 6D action, first frame zeros",
                "obs": ["joint_pos", "joint_vel", "front", "wrist", "actions"],
                "images": "uint8 RGB 84x84; cv2.INTER_AREA when resized",
                "omitted_raw_fields": "processed_actions, raw 8D actions, states, and unused observations",
            },
            "duplicate_detection": "sha256 of canonical float32 joint_pos and derived next-state actions",
            "baseline": {
                "path": str(baseline),
                "sha256": _sha256_file(baseline),
                "episodes": baseline_demos,
                "train": baseline_train,
                "valid": baseline_valid,
            },
            "recovery_sources": source_records,
            "accepted_recovery": accepted,
            "final_split": {
                "train": baseline_train + [entry["output_demo"] for entry in accepted],
                "valid": baseline_valid,
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=manifest.parent, prefix=f".{manifest.name}.", delete=False
        ) as tmp:
            json.dump(manifest_data, tmp, indent=2)
            tmp.write("\n")
            manifest_tmp = Path(tmp.name)
        os.replace(output_tmp, output)
        output_tmp = None
        output_installed = True
        os.replace(manifest_tmp, manifest)
        manifest_tmp = None
        return manifest_data
    except BaseException:
        if output_tmp is not None:
            output_tmp.unlink(missing_ok=True)
        if manifest_tmp is not None:
            manifest_tmp.unlink(missing_ok=True)
        if output_installed and not manifest.exists():
            output.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    manifest = args.manifest or args.output.with_suffix(".manifest.json")
    result = merge_recovery_dataset(
        args.baseline, args.recovery, args.output, manifest, args.batch_size
    )
    print(f"Merged {len(result['accepted_recovery'])} recovery episodes: {args.output}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
