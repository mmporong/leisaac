"""Run bounded image-resolution Mimic shards without an LLM monitoring loop.

Each successful shard is converted directly from observed next-joint positions.
Completed shards are reusable; incomplete outputs are never silently overwritten.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[2]


def shard_plan(total: int, chunk: int, seed_start: int) -> list[dict]:
    if total <= 0 or chunk <= 0 or seed_start < 0:
        raise ValueError("total and chunk must be positive; seed-start must be nonnegative")
    return [
        {"index": i, "seed": seed_start + i, "episodes": min(chunk, total - offset)}
        for i, offset in enumerate(range(0, total, chunk))
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_shard(path: Path, expected: int, image_size: int) -> int:
    with h5py.File(path, "r") as hdf:
        if len(hdf["data"]) != expected:
            raise ValueError(f"wrong number of episodes in {path}")
        total_frames = 0
        for demo in hdf["data"].values():
            if not bool(demo.attrs.get("success", False)):
                raise ValueError(f"non-successful demo in {path}: {demo.name}")
            frames = len(demo["actions"])
            if frames < 2 or demo["obs/joint_pos"].shape != (frames, 6):
                raise ValueError(f"invalid frame/joint shape in {path}: {demo.name}")
            raw_actions = demo["actions"][:]
            if raw_actions.ndim != 2 or not np.isfinite(raw_actions).all():
                raise ValueError(f"invalid raw actions in {path}: {demo.name}")
            if not np.isfinite(demo["obs/joint_pos"][:]).all():
                raise ValueError(f"invalid joint observations in {path}: {demo.name}")
            for camera in ("front", "wrist"):
                images = demo[f"obs/{camera}"]
                if images.shape != (frames, image_size, image_size, 3) or images.dtype.name != "uint8":
                    raise ValueError(f"wrong image contract in {path}: {camera}")
            total_frames += frames
    return total_frames


def validate_generation_manifest(path: Path, expected: dict) -> None:
    manifest = json.loads(path.read_text())
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"generation manifest does not match the planned shard: {path}")
    failures = manifest.get("failed_demos")
    attempts = manifest.get("attempts")
    successes = manifest.get("successful_demos")
    if not isinstance(failures, int) or failures < 0 or attempts != successes + failures:
        raise ValueError(f"inconsistent generation counts: {path}")
    if attempts > manifest["max_attempts"]:
        raise ValueError(f"generation exceeded planned attempt limit: {path}")


def check_free_space(path: Path, minimum_gib: float) -> None:
    if not math.isfinite(minimum_gib) or minimum_gib < 1:
        raise ValueError("minimum free space must be finite and at least 1 GiB")
    if shutil.disk_usage(path).free < minimum_gib * 2**30:
        raise RuntimeError(f"free space fell below {minimum_gib} GiB: {path}")


def validate_conversion(root: Path, raw: Path, episodes: int, frames: int, image_size: int) -> None:
    """Require the converter's final marker, not just partially written metadata."""
    provenance = json.loads((root / "conversion_provenance.json").read_text())
    expected = {
        "input_sha256": sha256(raw), "action_source": "next_observed",
        "image_size": image_size, "episode_count": episodes, "frame_count": frames,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError(f"conversion provenance mismatch: {root}")
    validate_dataset_info(root, episodes, frames, image_size)


def validate_dataset_info(root: Path, episodes: int, frames: int, image_size: int) -> None:
    info = json.loads((root / "meta/info.json").read_text())
    if info["total_episodes"] != episodes or info["total_frames"] != frames:
        raise ValueError(f"incomplete or mismatched dataset preserved: {root}")
    for camera in ("front", "wrist"):
        if info["features"][f"observation.images.{camera}"]["shape"] != [image_size, image_size, 3]:
            raise ValueError(f"dataset image shape mismatch: {root}: {camera}")


def run_child(command: list[str], log_path: Path) -> None:
    with log_path.open("x", encoding="utf-8") as stream:
        subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=6100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--max-attempts-per-success", type=int, default=10)
    args = parser.parse_args()
    if not 84 <= args.image_size <= 256 or not math.isfinite(args.min_free_gib) or args.min_free_gib < 1:
        raise ValueError("image-size must be 84..256 and min-free-gib finite and at least 1")
    if (args.render_width, args.render_height) != (640, 480):
        raise ValueError("render dimensions must be 640x480")
    if args.max_attempts_per_success < 1:
        raise ValueError("max-attempts-per-success must be positive")
    source = args.input.expanduser().resolve(strict=True)
    isaac_python = args.isaac_python.expanduser().resolve(strict=True)
    root = args.output_dir.expanduser().resolve()
    shards = shard_plan(args.total, args.chunk, args.seed_start)
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "source": str(source), "source_sha256": sha256(source), "shards": shards,
        "image_size": args.image_size, "render_width": args.render_width, "render_height": args.render_height,
        "min_free_gib": args.min_free_gib, "action_source": "next_observed",
        "successful_only": True, "max_attempts_per_success": args.max_attempts_per_success,
        "isaac_python": str(isaac_python), "lerobot_python": sys.executable,
        "code_sha256": {
            name: sha256(REPO / name) for name in (
                "scripts/mimic/generate_dataset.py", "scripts/mimic/runtime_options.py",
                "scripts/imitation_learning/convert_hdf5_to_lerobot.py",
                "scripts/imitation_learning/run_mimic_image_batch.py",
                "source/leisaac/leisaac/tasks/pick_cube_into_box/pick_cube_into_box_env_cfg.py",
                "source/leisaac/leisaac/tasks/pick_cube_into_box/pick_cube_into_box_mimic_env_cfg.py",
                "source/leisaac/leisaac/tasks/lift_cube/lift_cube_env_cfg.py",
            )
        },
    }
    with (root / "batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = root / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("existing batch plan or source/code fingerprint differs; use another output directory")
        if not plan_path.exists():
            atomic_json(plan_path, plan)
        progress_path = root / "progress.json"
        if progress_path.exists():
            previous = json.loads(progress_path.read_text())
            if previous.get("status") == "complete":
                validate_dataset_info(root / "lerobot_all", args.total, previous["total_frames"], args.image_size)
                print(json.dumps(previous), flush=True)
                return
        for directory in ("raw", "lerobot_shards", "logs"):
            (root / directory).mkdir(exist_ok=True)
        completed = []
        previous = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        status = {"status": "running", "target": args.total,
                  "completed_episodes": previous.get("completed_episodes", 0),
                  "completed_shards": previous.get("completed_shards", [])}

        def update(state: str, **extra) -> None:
            status.update(status=state, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
            atomic_json(root / "progress.json", status)

        try:
            update("running")
            for spec in shards:
                name = f"shard_{spec['index']:03d}"
                raw = root / "raw" / f"{name}.hdf5"
                manifest = raw.with_suffix(".generation.json")
                converted = root / "lerobot_shards" / name
                repo_id = f"local/so101_vision{args.image_size}_{name}"
                if (root / "STOP").exists():
                    update("paused", reason="STOP file found at shard boundary")
                    return
                check_free_space(root, args.min_free_gib)
                update("generating", current_shard=spec)
                if raw.exists():
                    if not manifest.is_file() or not json.loads(manifest.read_text()).get("completed"):
                        raise RuntimeError(f"partial generation preserved; inspect before resuming: {raw}")
                else:
                    run_child([
                        str(isaac_python), "-u", "scripts/mimic/generate_dataset.py",
                        "--task", "LeIsaac-SO101-PickCubeIntoBox-Mimic-v0", "--num_envs", "1",
                        "--input_file", str(source), "--output_file", str(raw),
                        "--generation_num_trials", str(spec["episodes"]), "--datagen-seed", str(spec["seed"]),
                        "--render-width", str(args.render_width), "--render-height", str(args.render_height),
                        "--observation-image-size", str(args.image_size), "--successful-only",
                        "--max-attempts", str(spec["episodes"] * args.max_attempts_per_success),
                        "--min-free-gib", str(args.min_free_gib),
                        "--progress-file", str(root / "generator_progress.json"),
                        "--headless", "--device", "cuda:0", "--enable_cameras",
                    ], root / "logs" / f"{name}_generate.log")
                validate_generation_manifest(manifest, {
                    "completed": True, "task": "LeIsaac-SO101-PickCubeIntoBox-Mimic-v0",
                    "input_file": str(source), "output_file": str(raw),
                    "generation_mode": "successes", "generation_num_trials": spec["episodes"],
                    "datagen_seed": spec["seed"], "num_envs": 1,
                    "successful_demos": spec["episodes"], "stop_reason": "target_successes_reached",
                    "render_dimensions": {camera: [args.render_width, args.render_height]
                                          for camera in ("front", "wrist")},
                    "observation_image_size": args.image_size, "successful_only": True,
                    "max_attempts": spec["episodes"] * args.max_attempts_per_success,
                    "min_free_gib": args.min_free_gib,
                })
                frame_count = validate_shard(raw, spec["episodes"], args.image_size)
                check_free_space(root, args.min_free_gib)
                update("converting", current_shard=spec)
                if not converted.exists():
                    run_child([
                        sys.executable, "-u", "scripts/imitation_learning/convert_hdf5_to_lerobot.py",
                        "--input", str(raw), "--output-root", str(converted), "--repo-id", repo_id,
                        "--image-size", str(args.image_size), "--action-source", "next_observed",
                    ], root / "logs" / f"{name}_convert.log")
                validate_conversion(converted, raw, spec["episodes"], frame_count, args.image_size)
                completed.append({**spec, "frames": frame_count, "root": str(converted), "repo_id": repo_id})
                update("shard_complete", completed_shards=completed,
                       completed_episodes=sum(item["episodes"] for item in completed))
                print(json.dumps({"completed": status["completed_episodes"], "target": args.total}), flush=True)

            aggregate = root / "lerobot_all"
            if (root / "STOP").exists():
                update("paused", reason="STOP file found before aggregation")
                return
            if aggregate.exists():
                raise FileExistsError(f"aggregate output already exists; inspect rather than overwrite: {aggregate}")
            check_free_space(root, args.min_free_gib)
            update("aggregating")
            from lerobot.datasets.aggregate import aggregate_datasets

            aggregate_datasets(
                repo_ids=[item["repo_id"] for item in completed],
                roots=[Path(item["root"]) for item in completed],
                aggr_repo_id=f"local/so101_mimic_vision{args.image_size}_{args.total}", aggr_root=aggregate,
            )
            total_frames = sum(item["frames"] for item in completed)
            validate_dataset_info(aggregate, args.total, total_frames, args.image_size)
            update("complete", dataset_root=str(aggregate), total_frames=total_frames)
        except BaseException as error:
            update("failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
