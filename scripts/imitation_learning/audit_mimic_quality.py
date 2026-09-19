"""Screen saved Mimic trajectories without rewriting data or running physics.

Recorded-state stability is a screening signal, not a new-policy replay result.
The candidate selection must pass a separate replay check before training.
"""

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.imitation_learning.action_contract import atomic_json, sha256_file
from scripts.imitation_learning.convert_hdf5_to_lerobot import (
    audit_demonstrations, get_demo_actions, load_joint_limits, sorted_demo_names, validate_image_size,
)

RELEASE_PATH = REPO / "source/leisaac/leisaac/tasks/pick_cube_into_box/mdp/release_state.py"
SPEC = importlib.util.spec_from_file_location("quality_release_criteria", RELEASE_PATH)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def release_run_summary(candidate: np.ndarray, control_dt_s: float, hold_time_s: float) -> dict:
    """Post-step samples start at episode step one; only uninterrupted samples count."""
    if (candidate.ndim != 1 or candidate.dtype != np.bool_
            or not math.isfinite(control_dt_s) or control_dt_s <= 0
            or not math.isfinite(hold_time_s) or hold_time_s <= 0):
        raise ValueError("invalid release candidates or timing")
    required = math.ceil(hold_time_s / control_dt_s - 1e-9)
    count = longest = 0
    first = None
    for step, value in enumerate(candidate, start=1):
        count = count + 1 if value else 0
        longest = max(longest, count)
        if count >= required and first is None:
            first = step
    return {"required_steps": required, "longest_stable_steps": longest,
            "terminal_stable_steps": count, "first_success_step": first,
            "ever_stable": first is not None, "final_stable": count >= required}


def recorded_release(demo, criteria: dict, control_dt_s: float) -> dict:
    """Evaluate only T-1 post-step states corresponding to available target labels."""
    count = len(demo["obs/joint_pos"]) - 1
    if count < 1:
        raise ValueError("recorded release requires at least two source frames")
    def array(key, width):
        data = np.asarray(demo[key][...])
        if data.shape != (count + 1, width) or not np.isfinite(data).all():
            raise ValueError(f"invalid saved state: {key}")
        return torch.from_numpy(data[:count])
    cube = array("states/rigid_object/cube/root_pose", 7)
    box = array("states/rigid_object/box_target/root_pose", 7)
    velocity = array("states/rigid_object/cube/root_velocity", 6)
    joints = array("states/articulation/robot/joint_position", 6)
    candidate = release.stable_release_candidate(
        cube[:, :3] - box[:, :3], velocity[:, :3], velocity[:, 3:], joints[:, -1], criteria,
    )
    return release_run_summary(candidate.numpy(), control_dt_s, criteria["hold_time_s"])


def jump_diagnostics(demo, limits, threshold: float, joint_names: list[str]) -> dict:
    commands = get_demo_actions(demo, "recorded_target", limits)
    clipping = np.abs(np.asarray(demo["obs/joint_pos_target"][1:]) - commands)
    delta = np.diff(commands, axis=0)
    norms = np.linalg.norm(delta, axis=1)
    indices = np.flatnonzero(norms > threshold)
    observed_steps = np.diff(np.asarray(demo["obs/joint_pos"][:]), axis=0)
    raw_actions = np.asarray(demo["actions"][:])
    if raw_actions.shape != (len(commands) + 1, 8):
        raise ValueError("pose diagnostics require 8D Mimic EEF actions")
    pose_delta = np.diff(raw_actions[:-1, :3], axis=0)
    dominant = Counter(joint_names[int(np.abs(delta[index]).argmax())] for index in indices)
    events = []
    for index in indices:
        events.append({"command_index": int(index + 1), "source_target_index": int(index + 2),
                       "norm_rad": float(norms[index]), "delta_rad": delta[index].tolist(),
                       "observed_joint_step_norm_rad": float(np.linalg.norm(observed_steps[index + 1])),
                       "eef_target_translation_step_m": float(np.linalg.norm(pose_delta[index]))})
    return {"max_action_step_norm_rad": float(norms.max()) if len(norms) else 0.,
            "clipped_frames": int(np.any(clipping > 0, axis=1).sum()),
            "max_clip_correction_rad": dict(zip(joint_names, clipping.max(axis=0).tolist())),
            "jump_count": len(indices), "dominant_joint_counts": dict(dominant),
            "max_observed_joint_step_norm_rad": float(np.linalg.norm(observed_steps, axis=1).max())
            if len(observed_steps) else 0.,
            "events": events}


def run_audit(raw_dir: Path, limits_path: Path, reference_path: Path, output: Path,
              threshold: float = 1.0, image_size: int = 224,
              require_unclipped_targets: bool = False) -> dict:
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("action-step threshold must be finite and positive")
    validate_image_size(image_size)
    raw_dir, limits_path, reference_path = (p.expanduser().resolve(strict=True)
                                           for p in (raw_dir, limits_path, reference_path))
    reference = json.loads(reference_path.read_text())
    control_dt_s = reference.get("control_dt_s")
    if (type(control_dt_s) not in (int, float) or not math.isfinite(control_dt_s) or control_dt_s <= 0):
        raise ValueError("reference must specify a finite positive control_dt_s")
    criteria = release.release_criteria_metadata(reference.get("success_criteria"))
    if reference.get("success_criteria") != criteria:
        raise ValueError("reference success criteria must match the shared stable-release specification")
    lower, upper, limits_record = load_joint_limits(limits_path)
    shards = sorted(raw_dir.glob("shard_*.hdf5"))
    if not shards:
        raise ValueError("no raw shards found")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "audits").mkdir()
    plan = {"schema_version": 1, "raw_dir": str(raw_dir), "action_source": "recorded_target",
            "require_unclipped_targets": require_unclipped_targets,
            "max_action_step_norm": threshold, "image_size": image_size,
            "joint_limits": limits_record, "control_dt_s": control_dt_s,
            "reference_path": str(reference_path), "reference_sha256": sha256_file(reference_path),
            "success_criteria": criteria, "code_sha256": {
                str(p.relative_to(REPO)): sha256_file(p) for p in (
                    Path(__file__).resolve(), RELEASE_PATH,
                    REPO / "scripts/imitation_learning/convert_hdf5_to_lerobot.py")}}
    atomic_json(output / "plan.json", plan)
    all_episodes, selection, cohort = [], [], []
    for raw in shards:
        raw_hash = sha256_file(raw)
        with h5py.File(raw, "r") as hdf:
            names = sorted_demo_names(hdf)
            audits = audit_demonstrations(hdf, names, threshold, image_size,
                                           "recorded_target", (lower, upper))
            selected = []
            for result in audits:
                demo = hdf[f"data/{result['name']}"]
                details = {"shard": raw.name, "raw_sha256": raw_hash, **result}
                try:
                    details["recorded_release"] = recorded_release(demo, criteria, control_dt_s)
                    details["jumps"] = jump_diagnostics(demo, (lower, upper), threshold,
                                                        limits_record["joint_names"])
                except (KeyError, ValueError, TypeError) as error:
                    details["diagnostic_error"] = f"{type(error).__name__}: {error}"
                details["screen_pass"] = (result["accepted"] and "diagnostic_error" not in details
                                          and details["recorded_release"]["final_stable"]
                                          and (not require_unclipped_targets
                                               or details["jumps"]["clipped_frames"] == 0))
                if details["screen_pass"]:
                    selected.append(result["name"])
                all_episodes.append(details)
        audit_path = output / "audits" / f"{raw.stem}.json"
        atomic_json(audit_path, {"schema_version": 1, "input_path": str(raw), "input_sha256": raw_hash,
                                "action_source": "recorded_target", "max_action_step_norm": threshold,
                                "joint_limits": limits_record, "episode_count": len(audits),
                                "accepted_count": sum(ep["accepted"] for ep in audits),
                                "rejected_count": sum(not ep["accepted"] for ep in audits), "episodes": audits})
        if selected:
            record = {"filename": raw.name, "raw_path": str(raw), "raw_sha256": raw_hash,
                      "audit_path": str(audit_path), "audit_sha256": sha256_file(audit_path),
                      "selected_demo_names": selected}
            selection.append(record)
            # Predeclared deterministic middle candidate per raw shard, not selected by replay outcome.
            cohort.append(record | {"selected_demo_names": [selected[len(selected) // 2]]})
        atomic_json(output / "progress.json", {"status": "auditing", "last_shard": raw.name,
                                               "episodes_checked": len(all_episodes)})
        print(f"audited {raw.name}: {len(selected)}/{len(audits)} recorded-state candidates", flush=True)
    diagnostics = {"schema_version": 1, "episodes": all_episodes}
    atomic_json(output / "episode_diagnostics.json", diagnostics)
    for filename, records, purpose in (
        ("candidate_selection.json", selection, "recorded-state screen only; replay not yet verified"),
        ("replay_cohort.json", cohort, "one predeclared middle candidate per shard; no performance claim"),
    ):
        atomic_json(output / filename, {"schema_version": 1, "action_source": "recorded_target",
                                      "purpose": purpose, "shards": records})
    joint_counts = Counter()
    for ep in all_episodes:
        joint_counts.update(ep.get("jumps", {}).get("dominant_joint_counts", {}))
    summary = {"schema_version": 1, "status": "complete", "episode_count": len(all_episodes),
               "converter_pass": sum(ep["accepted"] for ep in all_episodes),
               "converter_rejected": sum(not ep["accepted"] for ep in all_episodes),
               "require_unclipped_targets": require_unclipped_targets,
               "screen_pass": sum(ep["screen_pass"] for ep in all_episodes),
               "recorded_final_stable": sum(ep.get("recorded_release", {}).get("final_stable", False)
                                             for ep in all_episodes),
               "diagnostic_errors": sum("diagnostic_error" in ep for ep in all_episodes),
               "jump_event_count": sum(ep.get("jumps", {}).get("jump_count", 0) for ep in all_episodes),
               "dominant_joint_counts": dict(joint_counts), "replay_cohort_size": len(cohort),
               "replay_executed": False, "training_started": False,
               "episode_diagnostics_sha256": sha256_file(output / "episode_diagnostics.json")}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--joint-limits-file", type=Path, required=True)
    parser.add_argument("--timing-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-action-step-norm", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--require-unclipped-targets", action="store_true",
                        help="Reject episodes requiring any change to the recorded targets at joint limits.")
    args = parser.parse_args()
    print(json.dumps(run_audit(args.raw_dir, args.joint_limits_file, args.timing_reference,
                              args.output_dir, args.max_action_step_norm, args.image_size,
                              args.require_unclipped_targets)), flush=True)


if __name__ == "__main__":
    main()
