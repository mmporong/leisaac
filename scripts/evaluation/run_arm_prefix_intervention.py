"""Audit and run finite teacher-arm prefixes; not autonomous policy evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image

from run_action_intervention import (
    HORIZON, REPO, SEEDS, audit_case, atomic_json, build_command, load_preflight,
    prepare_output_root, require_gpu_idle, sha256_file,
)

CASES = (
    ("teacher_all", "teacher_all", None),
    ("teacher_arm", "teacher_arm", None),
    ("policy", "policy", None),
    ("teacher_arm_until_240", "teacher_arm", 240),
    ("teacher_arm_until_300", "teacher_arm", 300),
    ("teacher_arm_until_420", "teacher_arm", 420),
)
IMAGE_START, IMAGE_END, IMAGE_STRIDE = 211, 451, 30
STAGE = "arm_prefix_intervention_diagnostic"


def build_case_command(args, source, cutoff, seed, output):
    command = build_command(args, source, seed, output)
    command.extend(("--dump-policy-images-every", str(IMAGE_STRIDE),
                    "--dump-policy-images-range", str(IMAGE_START), str(IMAGE_END)))
    if cutoff is not None:
        command.extend(("--diagnostic-teacher-arm-until-step", str(cutoff)))
    return command


def audit_images_and_video(output, audit):
    trace = json.loads((output / "trace_001.json").read_text())
    evaluation = json.loads((output / "evaluation.json").read_text())
    if (evaluation.get("dump_policy_images_every") != IMAGE_STRIDE
            or evaluation.get("dump_policy_images_range") != [IMAGE_START, IMAGE_END]):
        raise ValueError("input image dump configuration changed")
    manifest_path = output / "policy_images_001.json"
    images = json.loads(manifest_path.read_text())
    expected = list(range(IMAGE_START, min(IMAGE_END, audit["steps"]) + 1, IMAGE_STRIDE))
    if [row["step"] for row in images] != expected:
        raise ValueError("input image dump steps changed")
    for row in images:
        if (row["observation_index"] != row["step"] - 1
                or row["state_before"] != trace[row["step"] - 1]["state_before"]):
            raise ValueError("dumped state is not the pre-action trace state")
        for camera in ("front", "wrist"):
            record = row[camera]
            path = (output / record["path"]).resolve(strict=True)
            if not path.is_relative_to(output.resolve()) or sha256_file(path) != record["sha256"]:
                raise ValueError("image path or file hash changed")
            with Image.open(path) as image:
                pixels = np.asarray(image)
            if (pixels.shape != (224, 224, 3) or pixels.dtype != np.uint8
                    or record["shape"] != [224, 224, 3] or record["dtype"] != "uint8"
                    or hashlib.sha256(pixels.tobytes()).hexdigest() != record["pixel_sha256"]):
                raise ValueError("dumped RGB input bytes changed")
    video = cv2.VideoCapture(audit["video"])
    try:
        metadata = {"frame_count": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": float(video.get(cv2.CAP_PROP_FPS)),
                    "first_frame_decoded": bool(video.read()[0])}
    finally:
        video.release()
    if metadata != {"frame_count": audit["steps"], "width": 1280, "height": 480,
                    "fps": 60.0, "first_frame_decoded": True}:
        raise ValueError(f"video metadata contract changed: {metadata}")
    return {"input_dump_frames": len(images), "input_manifest_sha256": sha256_file(manifest_path),
            "video_metadata": metadata}


def audit_prefix_case(output, case, source, cutoff, seed, preflight):
    case_preflight = dict(preflight)
    if cutoff is not None:
        case_preflight["teacher_arm_until_step"] = cutoff
    audit = audit_case(output, source, seed, case_preflight)
    if case in ("teacher_all", "teacher_arm") and audit["clipped_steps"] != 0:
        raise ValueError("positive controls must have zero clipping")
    audit.update(case=case, **audit_images_and_video(output, audit))
    if cutoff is not None:
        trace = json.loads((output / "trace_001.json").read_text())
        if len(trace) <= cutoff:
            raise ValueError("prefix diagnostic must execute the policy handoff")
        before, after = trace[cutoff - 1], trace[cutoff]
        audit["handoff"] = {
            "first_policy_step": cutoff + 1,
            "requested_arm_jump_l2_rad": float(np.linalg.norm(
                np.asarray(after["requested_action"][:5]) - before["requested_action"][:5])),
            "policy_teacher_arm_error_l2_rad": float(np.linalg.norm(
                np.asarray(after["policy_action"][:5]) - after["teacher_action"][:5])),
            "state_before": after["state_before"], "cube_xyz_after": after["cube_xyz"],
        }
    return audit


def validate_saved_runs(root, manifest, args, preflight):
    expected = [(seed, *case) for seed in SEEDS for case in CASES]
    actual = [(run["audit"]["seed"], run["audit"]["case"]) for run in manifest["runs"]]
    if actual != [(row[0], row[1]) for row in expected[:len(actual)]]:
        raise ValueError("saved cases must be an exact preregistered prefix")
    for run, (seed, case, source, cutoff) in zip(manifest["runs"], expected):
        output = root / f"seed_{seed}" / case
        command = build_case_command(args, source, cutoff, seed, output)
        command_sha = hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
        log = root / "logs" / f"seed_{seed}_{case}.log"
        if (run["command"] != command or run["command_sha256"] != command_sha
                or run["log"] != str(log) or sha256_file(log) != run["log_sha256"]):
            raise ValueError("saved command or log changed")
        if audit_prefix_case(output, case, source, cutoff, seed, preflight) != run["audit"]:
            raise ValueError("saved audit changed")
        if case in ("teacher_all", "teacher_arm") and not run["audit"]["success"]:
            raise ValueError("positive control failed; followups are forbidden")


def main():
    user = os.environ.get("USER") or Path.home().name
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/arm_prefix_intervention_20260928")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Read-only preflight/prefix audit; no Isaac.")
    args = parser.parse_args()
    args.isaac_python = (Path("/data") / user / "conda-envs/leisaac/bin/python").resolve(strict=True)
    args.server_python = (Path.home() / "miniforge3/envs/lerobot/bin/python").resolve(strict=True)
    args.evaluator = REPO / "scripts/evaluation/lerobot_act_so101.py"
    args.checkpoint = REPO / "outputs/act_reset_fixed_76_20260921/model/checkpoints/010000/pretrained_model"
    args.raw_path = REPO / "outputs/mimic_vision224_500_20260913/raw/shard_009.hdf5"
    args.demo, args.gripper_effort_mode = "demo_9", "fixed"
    reference = REPO / "outputs/evaluation/control_diagnosis_20260921/shard009_recorded_target/evaluation.json"
    preflight = load_preflight(args.raw_path, args.demo, args.checkpoint, reference)
    args.gripper_effort_limit = preflight["effort"][0]
    preflight.update(gripper_effort_mode="fixed", implementation={
        label: {"path": str(path), "sha256": sha256_file(path)} for label, path in {
            "prefix_runner": Path(__file__).resolve(), "base_runner": REPO / "scripts/evaluation/run_action_intervention.py",
            "evaluator": args.evaluator, "helper": REPO / "scripts/evaluation/action_intervention.py",
            "image_dump": REPO / "scripts/evaluation/policy_image_dump.py",
            "server": REPO / "scripts/imitation_learning/serve_lerobot_act.py",
            "protocol": REPO / "docs/plans/remaining_causes_20260921/arm_prefix_intervention_20260928.md",
        }.items()})
    saved_preflight = {key: value for key, value in preflight.items() if key != "targets"}
    root = args.output_root.expanduser().resolve()
    if args.resume:
        if (root / "manifest.sha256").exists():
            raise FileExistsError("diagnostic already finalized")
        manifest = json.loads((root / "manifest.json").read_text())
        if (manifest.get("schema_version") != 1 or manifest.get("stage") != STAGE
                or manifest.get("preflight") != saved_preflight):
            raise ValueError("saved protocol, source, model or implementation changed")
        validate_saved_runs(root, manifest, args, preflight)
    else:
        if root.exists():
            raise FileExistsError("output root exists; audit it with --resume or use a new root")
        manifest = {"schema_version": 1, "stage": STAGE, "preflight": saved_preflight, "runs": []}
    if args.check_only:
        print(json.dumps({"preflight_pass": True, "completed_runs": len(manifest["runs"]),
                          "output_writes": False, "gpu_checked": False}))
        return
    if any(os.statvfs(path).f_bavail * os.statvfs(path).f_frsize < 1024**3 for path in (REPO, Path.home())):
        raise RuntimeError("less than 1 GiB free on an output/runtime filesystem")
    require_gpu_idle()
    if args.resume:
        index = len(list(root.glob("resume_snapshot_*.json"))) + 1
        with (root / f"resume_snapshot_{index:03d}.json").open("xb") as stream:
            stream.write((root / "manifest.json").read_bytes())
    else:
        root = prepare_output_root(root)
        atomic_json(root / "manifest.json", manifest)
    expected = [(seed, *case) for seed in SEEDS for case in CASES]
    for seed, case, source, cutoff in expected[len(manifest["runs"]):]:
        require_gpu_idle(wait_seconds=10)
        output = root / f"seed_{seed}" / case
        log = root / "logs" / f"seed_{seed}_{case}.log"
        if output.exists() or log.exists():
            raise FileExistsError("unmanifested partial case exists; preserve it and use a new root")
        command = build_case_command(args, source, cutoff, seed, output)
        print(f"START seed={seed} case={case}", flush=True)
        with log.open("x") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            raise RuntimeError(f"case failed ({result.returncode}); see {log}")
        audit = audit_prefix_case(output, case, source, cutoff, seed, preflight)
        manifest["runs"].append({"command": command,
            "command_sha256": hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest(),
            "log": str(log), "log_sha256": sha256_file(log), "audit": audit})
        atomic_json(root / "manifest.json", manifest)
        print(f"DONE seed={seed} case={case} outcome={audit['outcome']}", flush=True)
        if case in ("teacher_all", "teacher_arm") and not audit["success"]:
            raise RuntimeError("positive control failed; followups are forbidden")
    manifest["launch_comparison"] = {
        case: {"success_values": [run["audit"]["success"] for run in manifest["runs"] if run["audit"]["case"] == case],
               "outcomes": [run["audit"]["outcome"] for run in manifest["runs"] if run["audit"]["case"] == case]}
        for case, _, _ in CASES}
    manifest["raw_model_unchanged"] = True
    atomic_json(root / "manifest.json", manifest)
    with (root / "manifest.sha256").open("x") as stream:
        stream.write(f"{sha256_file(root / 'manifest.json')}  manifest.json\n")
    print(json.dumps({"output": str(root), "runs": len(manifest["runs"]),
                      "launch_comparison": manifest["launch_comparison"]}), flush=True)


if __name__ == "__main__":
    main()
