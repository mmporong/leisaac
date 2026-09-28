"""Replay historical applied commands to test contact repeatability; no learning."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from run_action_intervention import (
    REPO, SEEDS, TASK, atomic_json, build_command, load_preflight, prepare_output_root,
    require_gpu_idle, sha256_file,
)
from run_arm_prefix_intervention import audit_images_and_video, IMAGE_START, IMAGE_END, IMAGE_STRIDE
from serve_recorded_actions import load_commands


SOURCES = {
    "old_success_4101": REPO / "outputs/action_intervention_fixed_effort_v2_20260927/seed_4101/teacher_arm",
    "old_success_4102": REPO / "outputs/action_intervention_fixed_effort_v2_20260927/seed_4102/teacher_arm",
    "new_failure_4101": REPO / "outputs/arm_prefix_intervention_20260928/seed_4101/teacher_arm",
}


def load_pinned_source(pin):
    parent_path = REPO / pin["parent_manifest"]
    if sha256_file(parent_path) != pin["parent_manifest_sha256"]:
        raise ValueError("preregistered parent manifest hash changed")
    parent = json.loads(parent_path.read_text())
    matches = [run["audit"] for run in parent["runs"]
               if run["audit"].get("condition") == pin["condition"] and run["audit"].get("seed") == pin["seed"]]
    if len(matches) != 1 or matches[0].get("audit_pass") is not True:
        raise ValueError("historical audited run identity changed")
    audit = matches[0]
    if audit["steps"] != pin["horizon"] or audit["success"] != pin["source_was_successful"]:
        raise ValueError("historical horizon/success changed")
    for key in ("trace", "evaluation"):
        path = (REPO / pin[key]).resolve(strict=True)
        if (audit[key] != str(path) or audit[key+"_sha256"] != pin[key+"_sha256"]
                or sha256_file(path) != pin[key+"_sha256"]):
            raise ValueError("historical trace/evaluation differs from the pinned parent audit")
    return {"kind": "historical_applied_action_replay", "model_loaded": False,
        "trace": str((REPO / pin["trace"]).resolve()), "trace_sha256": pin["trace_sha256"],
        "evaluation": str((REPO / pin["evaluation"]).resolve()), "evaluation_sha256": pin["evaluation_sha256"],
        "horizon": pin["horizon"], "source_was_successful": pin["source_was_successful"],
        "parent_manifest": str(parent_path), "parent_manifest_sha256": pin["parent_manifest_sha256"],
        "source_seed": pin["seed"], "source_condition": pin["condition"]}


def audit_replay(output, source, seed, config_path, preflight):
    evaluation_path, trace_path = output / "evaluation.json", output / "trace_001.json"
    evaluation = json.loads(evaluation_path.read_text())
    result = evaluation["results"][0]
    history = json.loads(Path(source["trace"]).read_text())
    trace = json.loads(trace_path.read_text())
    required = {"task": TASK, "checkpoint": str(config_path), "num_rollouts": 1, "seed_start": seed,
        "horizon": source["horizon"], "trace_steps": source["horizon"],
        "diagnostic_action_source": "teacher_arm", "diagnostic_teacher_arm_until_step": None,
        "autonomous_policy_evaluation": False, "gripper_effort_mode": "fixed",
        "gripper_effort_limit": preflight["effort"][0], "control_dt_s": preflight["control_dt_s"],
        "success_criteria": preflight["success_criteria"], "joint_names": preflight["joint_names"],
        "joint_lower_limits_rad": preflight["lower"], "joint_upper_limits_rad": preflight["upper"],
        "render_width": 640, "render_height": 480, "policy_image_size": 224,
        "reset_render_frames": 4, "server_device": "cpu", "server_seed": 0, "n_action_steps": 30,
        "initial_state_source": {"hdf5": preflight["raw_path"], "demo": preflight["demo"]}}
    if len(evaluation["results"]) != 1 or any(evaluation.get(k) != v for k, v in required.items()):
        raise ValueError("recorded replay evaluation contract changed")
    if (result["seed"] != seed or result["initial_scene_state_sha256"] != preflight["scene_sha256"]
            or result["gripper_effort_limit_range"] != preflight["effort"]
            or result["clipped_steps"] != 0 or len(trace) != result["steps"]
            or [r["step"] for r in trace] != list(range(1, result["steps"] + 1))
            or not 1 <= len(trace) <= source["horizon"]
            or (not result["success"] and len(trace) != source["horizon"])
            or result["first_success_step"] != (len(trace) if result["success"] else None)):
        raise ValueError("recorded replay scene, effort or complete prefix changed")
    for index, row in enumerate(trace):
        if row["gripper_effort_limit"] != preflight["effort"][0] or row["applied_action_source"] != "teacher_arm":
            raise ValueError("recorded replay effort/source changed")
        expected = np.asarray(history[index]["applied_action"], np.float32)
        for key in ("policy_action", "requested_action", "applied_action"):
            if not np.array_equal(np.asarray(row[key], np.float32), expected):
                raise ValueError("actual applied replay differs from the historical six-joint command")
    if (sha256_file(Path(source["parent_manifest"])) != source["parent_manifest_sha256"]
            or sha256_file(Path(source["trace"])) != source["trace_sha256"]
            or sha256_file(Path(source["evaluation"])) != source["evaluation_sha256"]
            or json.loads(config_path.read_text()) != source):
        raise ValueError("source trace/evaluation/manifest changed")
    if sha256_file(Path(preflight["raw_path"])) != preflight["raw_sha256"] or sha256_file(Path(preflight["model"]) / "model.safetensors") != preflight["model_sha256"]:
        raise ValueError("original raw/model changed")
    for record in preflight["implementation"].values():
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError("replay implementation changed")
    videos = list(output.glob("rollout_001_*.mp4"))
    if len(videos) != 1:
        raise ValueError("one complete replay video is required")
    audit = {"audit_pass": True, "kind": "historical_applied_action_replay", "model_loaded": False,
        "seed": seed, "steps": result["steps"], "success": result["success"], "outcome": result["outcome"],
        "max_cube_lift_m": result["max_cube_lift_m"], "first_lift_step": result["first_lift_step"],
        "first_success_step": result["first_success_step"], "clipped_steps": result["clipped_steps"],
        "source_was_successful": source["source_was_successful"],
        "initial_scene_state_sha256": result["initial_scene_state_sha256"],
        "effort": result["gripper_effort_limit_range"],
        "evaluation": str(evaluation_path), "evaluation_sha256": sha256_file(evaluation_path),
        "trace": str(trace_path), "trace_sha256": sha256_file(trace_path),
        "video": str(videos[0]), "video_sha256": sha256_file(videos[0])}
    audit.update(audit_images_and_video(output, audit))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/contact_replay_20260928")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    user = os.environ.get("USER") or Path.home().name
    raw = REPO / "outputs/mimic_vision224_500_20260913/raw/shard_009.hdf5"
    model = REPO / "outputs/act_reset_fixed_76_20260921/model/checkpoints/010000/pretrained_model"
    reference = REPO / "outputs/evaluation/control_diagnosis_20260921/shard009_recorded_target/evaluation.json"
    preflight = load_preflight(raw, "demo_9", model, reference)
    evaluator = REPO / "scripts/evaluation/lerobot_act_so101.py"
    server = REPO / "scripts/evaluation/serve_recorded_actions.py"
    pin_path = REPO / "docs/evidence/contact_replay_sources_20260928.json"
    pins = json.loads(pin_path.read_text())
    if pins.get("kind") != "audited_historical_source_pins" or set(pins["sources"]) != set(SOURCES):
        raise ValueError("unexpected preregistered source pins")
    sources = {}
    for name, path in SOURCES.items():
        source = load_pinned_source(pins["sources"][name])
        if source["trace"] != str(path / "trace_001.json") or source["evaluation"] != str(path / "evaluation.json"):
            raise ValueError("source path differs from the preregistration")
        evaluation = json.loads((path / "evaluation.json").read_text())
        result = evaluation["results"][0]
        trace = json.loads((path / "trace_001.json").read_text())
        commands = np.asarray([r["applied_action"] for r in trace], np.float32)
        if (result["steps"] != len(trace) or result["clipped_steps"] != 0
                or commands.shape != (len(trace), 6) or not np.isfinite(commands).all()
                or [r["step"] for r in trace] != list(range(1, len(trace)+1))
                or result["initial_scene_state_sha256"] != preflight["scene_sha256"]
                or evaluation["success_criteria"] != preflight["success_criteria"]
                or result["gripper_effort_limit_range"] != preflight["effort"]
                or not np.array_equal(commands[:, :5], preflight["targets"][1:len(trace)+1, :5])):
            raise ValueError("historical source does not match the replay physics/action contract")
        if result["success"] != source["source_was_successful"] or result["steps"] != source["horizon"]:
            raise ValueError("source result differs from the audited historical record")
        sources[name] = source
    implementation = {
        "runner": Path(__file__).resolve(), "server": server, "evaluator": evaluator,
        "helper": REPO / "scripts/evaluation/action_intervention.py",
        "base_runner": REPO / "scripts/evaluation/run_action_intervention.py",
        "prefix_runner": REPO / "scripts/evaluation/run_arm_prefix_intervention.py",
        "protocol_transport": REPO / "scripts/imitation_learning/serve_lerobot_act.py",
        "source_pins": pin_path,
        "image_dump": REPO / "scripts/evaluation/policy_image_dump.py",
        "protocol": REPO / "docs/plans/remaining_causes_20260921/contact_replay_20260928.md"}
    preflight["implementation"] = {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in implementation.items()}
    if args.check_only:
        print(json.dumps({"preflight_pass": True, "sources": {key: val["horizon"] for key,val in sources.items()},
                          "model_loaded": False, "output_writes": False}))
        return
    if any(os.statvfs(path).f_bavail * os.statvfs(path).f_frsize < 1024**3 for path in (REPO, Path.home())):
        raise RuntimeError("less than 1 GiB free on an output/runtime filesystem")
    require_gpu_idle()
    root = prepare_output_root(args.output_root)
    manifest = {"schema_version": 1, "stage": "contact_replay_diagnostic", "sources": sources,
                "preflight": {k:v for k,v in preflight.items() if k != "targets"}, "runs": []}
    atomic_json(root / "manifest.json", manifest)
    for name, source in sources.items():
        config = root / f"{name}_source.json"
        atomic_json(config, source)
        load_commands(config)
        for seed in SEEDS:
            require_gpu_idle(wait_seconds=10)
            output, log = root / name / f"seed_{seed}", root / "logs" / f"{name}_{seed}.log"
            cmd_args = argparse.Namespace(isaac_python=Path("/data") / user / "conda-envs/leisaac/bin/python",
                evaluator=evaluator, checkpoint=config, raw_path=raw, demo="demo_9",
                server_python=Path.home()/"miniforge3/envs/lerobot/bin/python", gripper_effort_mode="fixed",
                gripper_effort_limit=preflight["effort"][0])
            command = build_command(cmd_args, "teacher_arm", seed, output)
            command[command.index("--horizon")+1] = str(source["horizon"])
            command[command.index("--trace-steps")+1] = str(source["horizon"])
            command.extend(("--server-script", str(server), "--dump-policy-images-every", str(IMAGE_STRIDE),
                            "--dump-policy-images-range", str(IMAGE_START), str(IMAGE_END)))
            print(f"START source={name} seed={seed} recorded_only=true", flush=True)
            with log.open("x") as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
            if completed.returncode:
                raise RuntimeError(f"recorded replay failed ({completed.returncode}); see {log}")
            audit = audit_replay(output, source, seed, config, preflight)
            audit["source"] = name
            manifest["runs"].append({"command": command, "command_sha256": hashlib.sha256(
                json.dumps(command, separators=(",", ":")).encode()).hexdigest(),
                "source_manifest": str(config), "source_manifest_sha256": sha256_file(config),
                "log": str(log), "log_sha256": sha256_file(log), "audit": audit})
            atomic_json(root / "manifest.json", manifest)
            print(f"DONE source={name} seed={seed} outcome={audit['outcome']}", flush=True)
    manifest["raw_model_unchanged"] = True
    atomic_json(root / "manifest.json", manifest)
    with (root / "manifest.sha256").open("x") as stream:
        stream.write(f"{sha256_file(root / 'manifest.json')}  manifest.json\n")
    print(json.dumps({"runs":len(manifest["runs"]),"output":str(root)}), flush=True)


if __name__ == "__main__":
    main()
