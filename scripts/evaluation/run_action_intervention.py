"""Run and audit the preregistered arm/gripper command intervention diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.imitation_learning.action_contract import atomic_json, sha256_file

SEEDS = (4101, 4102)
FOLLOWUPS = ("policy", "teacher_gripper", "teacher_arm")
EXPECTED_SCENE = "375aaad23fe61b89eb0eb3c79037f53acb55348823dd674385f9d1c0c66d2205"
HORIZON = 675
MODEL_SHA256 = "ebed7e5415bf1f5501e7b834bfcb5f97c84593335fda145822f3c71fe4aeebab"
TASK = "LeIsaac-SO101-PickCubeIntoBox-v0"


def finite_six(value, label):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite shape (6,)")
    return array


def parse_compute_processes(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def require_gpu_idle() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    active = parse_compute_processes(result.stdout)
    if active:
        raise RuntimeError(f"GPU compute process exists; refusing to stop it or start Isaac: {active}")


def prepare_output_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing overlapping output root: {path}")
    path.mkdir(parents=True)
    (path / "logs").mkdir()
    return path


def load_preflight(raw_path: Path, demo: str, model: Path, reference_path: Path) -> dict:
    raw_path, model, reference_path = (p.expanduser().resolve(strict=True)
                                       for p in (raw_path, model, reference_path))
    expected_model = (REPO / "outputs/act_reset_fixed_76_20260921/model/checkpoints/010000/pretrained_model").resolve()
    expected_raw = (REPO / "outputs/mimic_vision224_500_20260913/raw/shard_009.hdf5").resolve()
    if model != expected_model or raw_path != expected_raw or demo != "demo_9":
        raise ValueError("diagnostic model/raw/demo are fixed by the preregistration")
    model_hash = sha256_file(model / "model.safetensors")
    if model_hash != MODEL_SHA256:
        raise ValueError("fixed model sha256 changed")
    raw_hash = sha256_file(raw_path)
    with h5py.File(raw_path, "r") as file:
        group = file[f"data/{demo}"]
        target = group["obs/joint_pos_target"][:]
        if target.shape != (676, 6) or target.dtype != np.float32 or not np.isfinite(target).all():
            raise ValueError("raw source must be finite float32 (676,6)")
        if not bool(group.attrs.get("success", False)):
            raise ValueError("raw source is not marked successful")
    reference = json.loads(reference_path.read_text())
    result = reference["results"][0]
    if not (reference["mode"] == "recorded_target" and result["episode"] == demo
            and result["steps"] == HORIZON and result["source_frame_count"] == 676
            and result["success"] and result["final_success"] and result["raw_sha256"] == raw_hash
            and result["initial_physical_sha256"] == EXPECTED_SCENE
            and result["gripper_effort_limit_range"][0] == result["gripper_effort_limit_range"][1]):
        raise ValueError("existing recorded-target positive control contract changed")
    return {"raw_path": str(raw_path), "raw_sha256": raw_hash, "demo": demo,
            "model": str(model), "model_sha256": model_hash, "targets": target,
            "reference_path": str(reference_path), "reference_sha256": sha256_file(reference_path),
            "scene_sha256": EXPECTED_SCENE, "success_criteria": reference["success_criteria"],
            "control_dt_s": reference["control_dt_s"], "joint_names": reference["joint_names"],
            "lower": reference["joint_lower_limits_rad"], "upper": reference["joint_upper_limits_rad"],
            "effort": result["gripper_effort_limit_range"]}


def audit_case(case_dir: Path, condition: str, seed: int, preflight: dict) -> dict:
    evaluation_path, trace_path = case_dir / "evaluation.json", case_dir / "trace_001.json"
    evaluation, trace = json.loads(evaluation_path.read_text()), json.loads(trace_path.read_text())
    if len(evaluation.get("results", [])) != 1:
        raise ValueError("evaluation must contain exactly one rollout result")
    result = evaluation["results"][0]
    required = {
        "task": TASK, "checkpoint": preflight["model"], "num_rollouts": 1,
        "seed_start": seed, "horizon": HORIZON,
        "trace_steps": HORIZON, "n_action_steps": 30, "reset_render_frames": 4,
        "gripper_effort_mode": "task", "diagnostic_action_source": condition,
        "server_seed": 0, "server_device": "cpu", "render_width": 640,
        "render_height": 480, "policy_image_size": 224, "lift_threshold_m": 0.02,
        "autonomous_policy_evaluation": condition == "policy",
        "control_dt_s": preflight["control_dt_s"], "success_criteria": preflight["success_criteria"],
        "joint_names": preflight["joint_names"], "joint_lower_limits_rad": preflight["lower"],
        "joint_upper_limits_rad": preflight["upper"],
    }
    for key, expected in required.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"evaluation contract mismatch: {key}")
    source = evaluation.get("initial_state_source")
    if source != {"hdf5": preflight["raw_path"], "demo": preflight["demo"]}:
        raise ValueError("initial-state source mismatch")
    teacher_source = evaluation.get("teacher_command_source")
    if condition == "policy":
        if teacher_source is not None:
            raise ValueError("policy condition must not have a teacher source")
    else:
        expected_teacher = {
            "path": preflight["raw_path"], "demo": preflight["demo"],
            "sha256": preflight["raw_sha256"], "source_frames": 676,
            "command_alignment": "loop t -> raw joint_pos_target[t+1]",
            "horizon": HORIZON, "training_eligibility": "diagnostic_only",
        }
        if teacher_source != expected_teacher:
            raise ValueError("teacher command source mismatch")
    if result["initial_scene_state_sha256"] != preflight["scene_sha256"]:
        raise ValueError("initial physical scene mismatch")
    if result.get("seed") != seed:
        raise ValueError("rollout seed mismatch")
    if result["gripper_effort_limit_range"] != preflight["effort"]:
        raise ValueError("gripper effort contract mismatch")
    steps = result["steps"]
    if not 1 <= steps <= HORIZON or len(trace) != steps or [r.get("step") for r in trace] != list(range(1, steps + 1)):
        raise ValueError("trace must cover exactly the executed one-based prefix")
    first_success_step = result.get("first_success_step")
    if (result["success"] and first_success_step != steps) or (not result["success"] and first_success_step is not None):
        raise ValueError("first_success_step does not match early-stop result")
    lower, upper = np.asarray(preflight["lower"], np.float32), np.asarray(preflight["upper"], np.float32)
    targets = preflight["targets"]
    maxima = {"teacher_trace": 0.0, "teacher_preclip": 0.0,
              "policy_preclip": 0.0, "applied_clip": 0.0}
    min_requested_gripper = float("inf")
    teacher_indices = set(range(6)) if condition == "teacher_all" else ({5} if condition == "teacher_gripper" else (set(range(5)) if condition == "teacher_arm" else set()))
    for row in trace:
        step = row["step"]
        policy = finite_six(row.get("policy_action"), f"policy_action step {step}")
        requested = finite_six(row.get("requested_action"), f"requested_action step {step}")
        applied = finite_six(row.get("applied_action"), f"applied_action step {step}")
        teacher = np.asarray(targets[step], np.float32)
        recorded_teacher = row.get("teacher_action")
        if condition == "policy":
            if recorded_teacher is not None:
                raise ValueError("policy trace must not contain teacher_action")
        else:
            recorded_teacher = finite_six(recorded_teacher, f"teacher_action step {step}")
            maxima["teacher_trace"] = max(
                maxima["teacher_trace"], float(np.abs(recorded_teacher - teacher).max()))
        if teacher_indices:
            indices = sorted(teacher_indices)
            maxima["teacher_preclip"] = max(maxima["teacher_preclip"], float(np.abs(requested[indices] - teacher[indices]).max()))
        policy_indices = sorted(set(range(6)) - teacher_indices)
        if policy_indices:
            maxima["policy_preclip"] = max(maxima["policy_preclip"], float(np.abs(requested[policy_indices] - policy[policy_indices]).max()))
        maxima["applied_clip"] = max(maxima["applied_clip"], float(np.abs(applied - np.clip(requested, lower, upper)).max()))
        min_requested_gripper = min(min_requested_gripper, float(requested[5]))
    if any(value != 0 for value in maxima.values()):
        raise ValueError(f"action substitution/clip audit failed: {maxima}")
    if condition == "teacher_all" and (
        result["clipped_steps"] != 0 or any(result["max_clip_correction_rad"].values())
        or maxima["applied_clip"] != 0
    ):
        raise ValueError("teacher_all must have zero clipping")
    if sha256_file(Path(preflight["raw_path"])) != preflight["raw_sha256"] or sha256_file(Path(preflight["model"]) / "model.safetensors") != preflight["model_sha256"]:
        raise ValueError("raw or model changed during diagnostic")
    for label, record in preflight["implementation"].items():
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"implementation changed during diagnostic: {label}")
    videos = sorted(case_dir.glob("rollout_001_*.mp4"))
    if len(videos) != 1:
        raise ValueError("expected exactly one rollout video")
    return {"audit_pass": True, "condition": condition, "seed": seed, "steps": steps,
            "success": bool(result["success"]), "outcome": result["outcome"],
            "max_cube_lift_m": result["max_cube_lift_m"], "first_lift_step": result["first_lift_step"],
            "first_success_step": first_success_step,
            "min_requested_gripper_rad": min_requested_gripper,
            "clipped_steps": result["clipped_steps"],
            "max_clip_correction_rad": result["max_clip_correction_rad"],
            "initial_observation_sha256": result["initial_observation_sha256"],
            "first_policy_action": trace[0]["policy_action"], "max_errors": maxima,
            "evaluation": str(evaluation_path), "evaluation_sha256": sha256_file(evaluation_path),
            "trace": str(trace_path), "trace_sha256": sha256_file(trace_path),
            "video": str(videos[0]), "video_sha256": sha256_file(videos[0])}


def allowed_followups(teacher_audit: dict) -> tuple[str, ...]:
    return FOLLOWUPS if teacher_audit.get("audit_pass") is True and teacher_audit.get("success") is True else ()


def build_command(args, condition, seed, output):
    return [str(args.isaac_python), "-u", str(args.evaluator), "--checkpoint", str(args.checkpoint),
            "--initial-state-hdf5", str(args.raw_path), "--initial-state-demo", args.demo,
            "--num-rollouts", "1", "--seed", str(seed), "--horizon", str(HORIZON),
            "--n-action-steps", "30", "--render-width", "640", "--render-height", "480",
            "--policy-image-size", "224", "--reset-render-frames", "4", "--gripper-effort-mode", "task",
            "--lift-threshold-m", "0.02",
            "--trace-steps", str(HORIZON), "--diagnostic-action-source", condition,
            "--server-device", "cpu", "--server-seed", "0", "--server-python", str(args.server_python),
            "--video-count", "1", "--output-dir", str(output), "--headless", "--device", "cuda:0"]


def run_case(args, root, preflight, condition, seed):
    require_gpu_idle()
    output, log = root / f"seed_{seed}" / condition, root / "logs" / f"seed_{seed}_{condition}.log"
    command = build_command(args, condition, seed, output)
    command_sha256 = hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
    print(f"START seed={seed} condition={condition}", flush=True)
    with log.open("x") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError(f"case failed ({completed.returncode}); see {log}")
    audit = audit_case(output, condition, seed, preflight)
    print(f"DONE seed={seed} condition={condition} success={audit['success']}", flush=True)
    return {"command": command, "command_sha256": command_sha256,
            "log": str(log), "log_sha256": sha256_file(log), "audit": audit}


def summarize_launches(runs):
    summary = {}
    for condition in ("teacher_all", *FOLLOWUPS):
        rows = [run["audit"] for run in runs if run["audit"]["condition"] == condition]
        summary[condition] = {"executions": len(rows), "success_values": [r["success"] for r in rows],
                              "outcomes": [r["outcome"] for r in rows],
                              "lifted_values": [r["first_lift_step"] is not None for r in rows],
                              "launch_consistent": len(rows) == 2 and len({(r["success"], r["outcome"], r["first_lift_step"] is not None) for r in rows}) == 1,
                              "interpretation": "launch-sensitive diagnostic only; not an autonomy success rate"}
    return summary


def main():
    user = os.environ.get("USER") or Path.home().name
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/action_intervention_20260922")
    parser.add_argument("--checkpoint", type=Path, default=REPO / "outputs/act_reset_fixed_76_20260921/model/checkpoints/010000/pretrained_model")
    parser.add_argument("--raw-path", type=Path, default=REPO / "outputs/mimic_vision224_500_20260913/raw/shard_009.hdf5")
    parser.add_argument("--demo", default="demo_9")
    parser.add_argument("--reference", type=Path, default=REPO / "outputs/evaluation/control_diagnosis_20260921/shard009_recorded_target/evaluation.json")
    parser.add_argument("--evaluator", type=Path, default=REPO / "scripts/evaluation/lerobot_act_so101.py")
    parser.add_argument("--isaac-python", type=Path, default=Path("/data") / user / "conda-envs/leisaac/bin/python")
    parser.add_argument("--server-python", type=Path, default=Path.home() / "miniforge3/envs/lerobot/bin/python")
    args = parser.parse_args()
    args.evaluator, args.isaac_python, args.server_python = (
        path.expanduser().resolve(strict=True)
        for path in (args.evaluator, args.isaac_python, args.server_python)
    )
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    args.raw_path = args.raw_path.expanduser().resolve(strict=True)
    preflight = load_preflight(args.raw_path, args.demo, args.checkpoint, args.reference)
    implementation_paths = {
        "runner": Path(__file__).resolve(),
        "evaluator": args.evaluator,
        "intervention_helper": REPO / "scripts/evaluation/action_intervention.py",
    }
    preflight["implementation"] = {
        label: {"path": str(path), "sha256": sha256_file(path)}
        for label, path in implementation_paths.items()
    }
    require_gpu_idle()
    root = prepare_output_root(args.output_root)
    manifest = {"schema_version": 1, "stage": "action_intervention_diagnostic", "preflight": {k: v for k, v in preflight.items() if k != "targets"}, "runs": []}
    atomic_json(root / "manifest.json", manifest)
    for seed in SEEDS:
        teacher = run_case(args, root, preflight, "teacher_all", seed)
        manifest["runs"].append(teacher); atomic_json(root / "manifest.json", manifest)
        for condition in allowed_followups(teacher["audit"]):
            run = run_case(args, root, preflight, condition, seed)
            manifest["runs"].append(run); atomic_json(root / "manifest.json", manifest)
    manifest["launch_comparison"] = summarize_launches(manifest["runs"])
    manifest["raw_model_unchanged"] = True
    atomic_json(root / "manifest.json", manifest)
    manifest_hash = sha256_file(root / "manifest.json")
    (root / "manifest.sha256").write_text(f"{manifest_hash}  manifest.json\n", encoding="utf-8")
    print(json.dumps({"output": str(root), "runs": len(manifest["runs"]),
                      "manifest_sha256": manifest_hash,
                      "launch_comparison": manifest["launch_comparison"]}))


if __name__ == "__main__": main()
