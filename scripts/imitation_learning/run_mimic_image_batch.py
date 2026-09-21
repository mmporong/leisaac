"""Run bounded image-resolution Mimic shards without an LLM monitoring loop.

Each successful shard is converted from recorded actuator targets by default.
Completed shards are reusable; incomplete outputs are never silently overwritten.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.imitation_learning.action_contract import (
    CONTRACT_FILENAME,
    atomic_json,
    build_aggregate_contract,
    sha256_file,
    validate_aggregate_contract,
)
from scripts.imitation_learning.convert_hdf5_to_lerobot import load_joint_limits


def shard_plan(total: int, chunk: int, seed_start: int) -> list[dict]:
    if total <= 0 or chunk <= 0 or seed_start < 0:
        raise ValueError("total and chunk must be positive; seed-start must be nonnegative")
    return [
        {"index": i, "seed": seed_start + i, "episodes": min(chunk, total - offset)}
        for i, offset in enumerate(range(0, total, chunk))
    ]


sha256 = sha256_file


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
    # Manifests written before the reset refresh option existed carry no key; they mean 0 (legacy frames).
    defaults = {"reset_render_frames": 0}
    if any(manifest.get(key, defaults.get(key)) != value for key, value in expected.items()):
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


def validate_conversion(
    root: Path,
    raw: Path,
    episodes: int,
    frames: int | None,
    image_size: int,
    action_source: str = "recorded_target",
    selected_demo_names: list[str] | None = None,
) -> dict:
    """Require the converter's final marker, not just partially written metadata."""
    provenance = json.loads((root / "conversion_provenance.json").read_text())
    expected = {
        "input_sha256": sha256(raw), "action_source": action_source,
        "image_size": image_size, "episode_count": episodes,
    }
    expected_frames = provenance.get("frame_count") if frames is None else (
        frames - episodes if action_source == "recorded_target" else frames
    )
    expected["frame_count"] = expected_frames
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError(f"conversion provenance mismatch: {root}")
    if selected_demo_names is not None and provenance.get("episode_names") != selected_demo_names:
        raise ValueError(f"conversion selected-demo order mismatch: {root}")
    validate_dataset_info(root, episodes, expected_frames, image_size)
    return provenance


def load_selection_manifest(
    path: Path, raw_dir: Path, action_source: str,
    max_action_step_norm: float, joint_limits: dict | None,
) -> list[dict]:
    if (type(max_action_step_norm) not in (int, float)
            or not math.isfinite(max_action_step_norm) or max_action_step_norm <= 0):
        raise ValueError("selection action-step threshold must be finite and positive")
    if action_source == "recorded_target" and joint_limits is None:
        raise ValueError("recorded_target selection requires joint limits")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("action_source") != action_source):
        raise ValueError("selection manifest schema or action source mismatch")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("selection manifest must contain a nonempty shards list")
    result = []
    selected_global = set()
    for index, item in enumerate(shards):
        filename = item.get("filename")
        names = item.get("selected_demo_names")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("selection shard filename must be a basename")
        if not isinstance(names, list) or not names or len(names) != len(set(names)):
            raise ValueError(f"selection for {filename} must be nonempty and unique")
        raw = (raw_dir / filename).resolve(strict=True)
        if item.get("raw_sha256") != sha256(raw):
            raise ValueError(f"selection raw hash mismatch: {raw}")
        audit_path = Path(item.get("audit_path", "")).expanduser().resolve(strict=True)
        if item.get("audit_sha256") != sha256(audit_path):
            raise ValueError(f"selection audit hash mismatch: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if type(audit.get("schema_version")) is not int or audit["schema_version"] != 1:
            raise ValueError("unsupported selection audit schema")
        threshold = audit.get("max_action_step_norm")
        if (type(threshold) not in (int, float) or not math.isfinite(threshold)
                or threshold <= 0 or threshold != max_action_step_norm):
            raise ValueError("selection audit action-step threshold differs from conversion")
        if audit.get("joint_limits") != joint_limits:
            raise ValueError("selection audit joint limits differ from conversion")
        if (audit.get("input_sha256") != item["raw_sha256"]
                or audit.get("action_source") != action_source):
            raise ValueError(f"audit does not match selection source: {filename}")
        episodes = audit.get("episodes", [])
        if (not isinstance(episodes, list) or not episodes
                or any(type(audit.get(key)) is not int
                       for key in ("episode_count", "accepted_count", "rejected_count"))
                or any(not isinstance(ep, dict) or not isinstance(ep.get("name"), str)
                       or type(ep.get("accepted")) is not bool for ep in episodes)
                or len({ep["name"] for ep in episodes}) != len(episodes)
                or audit.get("episode_count") != len(episodes)
                or audit.get("accepted_count") != sum(ep["accepted"] for ep in episodes)
                or audit.get("rejected_count") != sum(not ep["accepted"] for ep in episodes)):
            raise ValueError("selection audit episode counts or decisions are inconsistent")
        accepted = {episode["name"] for episode in episodes if episode["accepted"]}
        if not set(names) <= accepted:
            raise ValueError(f"selection contains unaudited or rejected demos: {filename}")
        keys = {(item["raw_sha256"], name) for name in names}
        if selected_global & keys:
            raise ValueError("selection contains duplicate raw/demo identities")
        selected_global.update(keys)
        result.append({"index": index, "filename": filename, "episodes": len(names),
                       "selected_demo_names": names, "audit_episode_count": audit["episode_count"]})
    return result


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
    parser.add_argument("--isaac-python", type=Path)
    parser.add_argument(
        "--raw-dir", type=Path,
        help="Reuse completed raw shards from this directory; missing shards fail instead of running simulation.",
    )
    parser.add_argument("--convert-only", action="store_true", help="Require existing raw shards and skip simulation.")
    parser.add_argument(
        "--action-source", choices=("recorded_target", "next_observed", "stored"),
        default="recorded_target",
    )
    parser.add_argument("--allow-legacy-action-source", action="store_true")
    parser.add_argument("--joint-limits-file", type=Path)
    parser.add_argument("--max-action-step-norm", type=float, default=1.0)
    parser.add_argument(
        "--selection-manifest", type=Path,
        help="Explicit audited demo selection for reusable raw shards; never auto-drops rejected demos.",
    )
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=6100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--max-attempts-per-success", type=int, default=10)
    parser.add_argument(
        "--reset-render-frames", type=int, default=0,
        help="Re-render after every reset so recorded frame 0 shows the reset scene; 0 keeps legacy stale frames.",
    )
    args = parser.parse_args()
    if args.reset_render_frames < 0:
        raise ValueError("reset-render-frames cannot be negative")
    if not 84 <= args.image_size <= 256 or not math.isfinite(args.min_free_gib) or args.min_free_gib < 1:
        raise ValueError("image-size must be 84..256 and min-free-gib finite and at least 1")
    if (args.render_width, args.render_height) != (640, 480):
        raise ValueError("render dimensions must be 640x480")
    if args.max_attempts_per_success < 1:
        raise ValueError("max-attempts-per-success must be positive")
    if not math.isfinite(args.max_action_step_norm) or args.max_action_step_norm <= 0:
        raise ValueError("max-action-step-norm must be finite and positive")
    source = args.input.expanduser().resolve(strict=True)
    if args.action_source != "recorded_target" and not args.allow_legacy_action_source:
        raise ValueError("legacy action sources require --allow-legacy-action-source")
    if args.action_source == "recorded_target" and args.joint_limits_file is None:
        raise ValueError("recorded_target requires --joint-limits-file")
    joint_limits_file = (
        args.joint_limits_file.expanduser().resolve(strict=True) if args.joint_limits_file else None
    )
    joint_limits = load_joint_limits(joint_limits_file)[2] if args.action_source == "recorded_target" else None
    if args.isaac_python is None and not (args.convert_only or args.raw_dir):
        raise ValueError("--isaac-python is required when generating raw shards")
    isaac_python = args.isaac_python.expanduser().resolve(strict=True) if args.isaac_python else None
    raw_dir = args.raw_dir.expanduser().resolve(strict=True) if args.raw_dir else None
    selection_manifest = (
        args.selection_manifest.expanduser().resolve(strict=True) if args.selection_manifest else None
    )
    if selection_manifest is not None and raw_dir is None:
        raise ValueError("--selection-manifest requires --raw-dir")
    root = args.output_dir.expanduser().resolve()
    shards = (
        load_selection_manifest(selection_manifest, raw_dir, args.action_source,
                                args.max_action_step_norm, joint_limits)
        if selection_manifest is not None else shard_plan(args.total, args.chunk, args.seed_start)
    )
    if sum(item["episodes"] for item in shards) != args.total:
        raise ValueError("selected demo count must equal --total")
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "source": str(source), "source_sha256": sha256(source), "shards": shards,
        "image_size": args.image_size, "render_width": args.render_width, "render_height": args.render_height,
        "min_free_gib": args.min_free_gib, "action_source": args.action_source,
        "max_action_step_norm": args.max_action_step_norm,
        "legacy_action_source_explicit": args.allow_legacy_action_source,
        "joint_limits_file": str(joint_limits_file) if joint_limits_file else None,
        "joint_limits_sha256": sha256(joint_limits_file) if joint_limits_file else None,
        "raw_dir": str(raw_dir) if raw_dir else None, "convert_only": args.convert_only,
        "selection_manifest": str(selection_manifest) if selection_manifest else None,
        "selection_manifest_sha256": sha256(selection_manifest) if selection_manifest else None,
        "successful_only": True, "max_attempts_per_success": args.max_attempts_per_success,
        "reset_render_frames": args.reset_render_frames,
        "isaac_python": str(isaac_python) if isaac_python else None, "lerobot_python": sys.executable,
        "code_sha256": {
            name: sha256(REPO / name) for name in (
                "scripts/mimic/generate_dataset.py", "scripts/mimic/runtime_options.py",
                "scripts/imitation_learning/convert_hdf5_to_lerobot.py",
                "scripts/imitation_learning/action_contract.py",
                "scripts/imitation_learning/run_mimic_image_batch.py",
                "source/leisaac/leisaac/tasks/pick_cube_into_box/pick_cube_into_box_env_cfg.py",
                "source/leisaac/leisaac/tasks/pick_cube_into_box/pick_cube_into_box_mimic_env_cfg.py",
                "source/leisaac/leisaac/tasks/pick_cube_into_box/mdp/terminations.py",
                "source/leisaac/leisaac/tasks/pick_cube_into_box/mdp/release_state.py",
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
                validate_aggregate_contract(root / "lerobot_all", allow_legacy=args.allow_legacy_action_source)
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
                raw = (raw_dir or (root / "raw")) / spec.get("filename", f"{name}.hdf5")
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
                elif args.convert_only or raw_dir is not None:
                    raise FileNotFoundError(f"required reusable raw shard is missing: {raw}")
                else:
                    assert isaac_python is not None
                    run_child([
                        str(isaac_python), "-u", "scripts/mimic/generate_dataset.py",
                        "--task", "LeIsaac-SO101-PickCubeIntoBox-Mimic-v0", "--num_envs", "1",
                        "--input_file", str(source), "--output_file", str(raw),
                        "--generation_num_trials", str(spec["episodes"]), "--datagen-seed", str(spec["seed"]),
                        "--render-width", str(args.render_width), "--render-height", str(args.render_height),
                        "--observation-image-size", str(args.image_size), "--successful-only",
                        "--max-attempts", str(spec["episodes"] * args.max_attempts_per_success),
                        "--min-free-gib", str(args.min_free_gib),
                        "--reset-render-frames", str(args.reset_render_frames),
                        "--progress-file", str(root / "generator_progress.json"),
                        "--headless", "--device", "cuda:0", "--enable_cameras",
                    ], root / "logs" / f"{name}_generate.log")
                if selection_manifest is None:
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
                        "reset_render_frames": args.reset_render_frames,
                    })
                    raw_episode_count = spec["episodes"]
                else:
                    raw_episode_count = spec["audit_episode_count"]
                raw_frame_count = validate_shard(raw, raw_episode_count, args.image_size)
                check_free_space(root, args.min_free_gib)
                update("converting", current_shard=spec)
                if not converted.exists():
                    convert_command = [
                        sys.executable, "-u", "scripts/imitation_learning/convert_hdf5_to_lerobot.py",
                        "--input", str(raw), "--output-root", str(converted), "--repo-id", repo_id,
                        "--image-size", str(args.image_size), "--action-source", args.action_source,
                        "--max-action-step-norm", str(args.max_action_step_norm),
                    ]
                    if joint_limits_file is not None:
                        convert_command.extend(["--joint-limits-file", str(joint_limits_file)])
                    if spec.get("selected_demo_names"):
                        convert_command.extend(["--demo-names", *spec["selected_demo_names"]])
                    run_child(convert_command, root / "logs" / f"{name}_convert.log")
                provenance = validate_conversion(
                    converted, raw, spec["episodes"],
                    None if spec.get("selected_demo_names") else raw_frame_count,
                    args.image_size, args.action_source, spec.get("selected_demo_names"),
                )
                completed.append({
                    **spec, "frames": provenance["frame_count"], "root": str(converted),
                    "repo_id": repo_id, "raw_path": str(raw),
                    "conversion_provenance_path": str(converted / "conversion_provenance.json"),
                })
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
            contract = build_aggregate_contract(
                aggregate, args.action_source, completed, args.total, total_frames
            )
            atomic_json(aggregate / CONTRACT_FILENAME, contract)
            validate_aggregate_contract(aggregate, allow_legacy=args.allow_legacy_action_source)
            update("complete", dataset_root=str(aggregate), total_frames=total_frames)
        except BaseException as error:
            update("failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
