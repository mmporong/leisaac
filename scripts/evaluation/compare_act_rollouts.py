"""Compare paired baseline and recovery ACT rollout evaluation reports."""

import argparse
import json
import math
import string
from pathlib import Path


CONDITION_KEYS = (
    "task",
    "horizon",
    "server_seed",
    "n_action_steps",
    "lift_threshold_m",
    "server_device",
    "render_width",
    "render_height",
    "policy_image_size",
)
OUTCOMES = ("success", "no_lift", "low_after_lift", "lifted_not_completed")
INITIAL_STATE_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require_finite_vector(value, size: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must be a list with shape ({size},)")
    vector = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"{label}[{index}] must be finite, got {item!r}")
        vector.append(float(item))
    return vector


def _require_sha256(value, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in string.hexdigits for character in value):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256 digest")
    return value


def _load_rollout(root: Path, seed: int, arm: str) -> dict:
    path = root / f"seed_{seed}" / "evaluation.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {arm} evaluation for seed {seed}: {path}")
    try:
        evaluation = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(evaluation, dict):
        raise ValueError(f"evaluation must be a JSON object: {path}")
    if type(evaluation.get("num_rollouts")) is not int or evaluation["num_rollouts"] != 1:
        raise ValueError(f"{path} must report exactly one rollout")
    results = evaluation.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"{path} must contain exactly one result")
    if evaluation.get("seed_start") != seed or results[0].get("seed") != seed:
        raise ValueError(
            f"seed mismatch in {path}: expected {seed}, "
            f"seed_start={evaluation.get('seed_start')!r}, result_seed={results[0].get('seed')!r}"
        )
    result = results[0]
    if type(result.get("success")) is not bool:
        raise ValueError(f"success must be bool in {path}")
    outcome = result.get("outcome")
    if outcome not in OUTCOMES:
        raise ValueError(f"invalid outcome in {path}: {outcome!r}")
    if result["success"] != (outcome == "success"):
        raise ValueError(f"success and outcome disagree in {path}")
    checkpoint = evaluation.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"checkpoint must be a non-empty string in {path}")
    hashes = result.get("initial_observation_sha256")
    if not isinstance(hashes, dict) or any(not isinstance(hashes.get(camera), str) for camera in ("front", "wrist")):
        raise ValueError(f"initial_observation_sha256 must contain front and wrist strings in {path}")
    conditions = {key: evaluation.get(key) for key in CONDITION_KEYS}
    if not isinstance(conditions["task"], str) or not conditions["task"]:
        raise ValueError(f"task must be a non-empty string in {path}")
    if type(conditions["horizon"]) is not int or conditions["horizon"] <= 0:
        raise ValueError(f"horizon must be a positive integer in {path}")
    if type(conditions["server_seed"]) is not int:
        raise ValueError(f"server_seed must be an integer in {path}")
    if conditions["n_action_steps"] is not None and (
        type(conditions["n_action_steps"]) is not int or conditions["n_action_steps"] <= 0
    ):
        raise ValueError(f"n_action_steps must be null or a positive integer in {path}")
    threshold = conditions["lift_threshold_m"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError(f"lift_threshold_m must be finite and positive in {path}")
    conditions["lift_threshold_m"] = float(threshold)
    if conditions["server_device"] not in ("cpu", "cuda"):
        raise ValueError(f"server_device must be cpu or cuda in {path}")
    for key in ("render_width", "render_height", "policy_image_size"):
        if type(conditions[key]) is not int or conditions[key] <= 0:
            raise ValueError(f"{key} must be a positive integer in {path}")
    return {
        "path": path,
        "checkpoint": checkpoint,
        "conditions": conditions,
        "success": result["success"],
        "outcome": outcome,
        "initial_cube_xyz": _require_finite_vector(
            result.get("initial_cube_xyz"), 3, f"{path}: initial_cube_xyz"
        ),
        "initial_joint_pos": _require_finite_vector(
            result.get("initial_joint_pos"), 6, f"{path}: initial_joint_pos"
        ),
        "initial_scene_state_sha256": _require_sha256(
            result.get("initial_scene_state_sha256"), f"{path}: initial_scene_state_sha256"
        ),
        "hashes": {camera: hashes[camera] for camera in ("front", "wrist")},
    }


def _assert_same(label: str, baseline, recovery, seed: int) -> None:
    if baseline != recovery:
        raise ValueError(
            f"comparison condition mismatch for seed {seed}, {label}: "
            f"baseline={baseline!r}, recovery={recovery!r}"
        )


def _assert_initial_state_equal(label: str, baseline: list[float], recovery: list[float], seed: int) -> None:
    for index, (baseline_value, recovery_value) in enumerate(zip(baseline, recovery)):
        difference = abs(baseline_value - recovery_value)
        if difference > INITIAL_STATE_TOLERANCE:
            raise ValueError(
                f"initial state mismatch for seed {seed}, {label}[{index}]: "
                f"baseline={baseline_value!r}, recovery={recovery_value!r}, "
                f"absolute_difference={difference!r}, tolerance={INITIAL_STATE_TOLERANCE!r}"
            )


def build_report(baseline_dir: Path, recovery_dir: Path, seeds: list[int]) -> dict:
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        duplicates = sorted({seed for seed in seeds if seeds.count(seed) > 1})
        raise ValueError(f"duplicate seeds are not allowed: {duplicates}")

    baseline_rollouts = [_load_rollout(baseline_dir, seed, "baseline") for seed in seeds]
    recovery_rollouts = [_load_rollout(recovery_dir, seed, "recovery") for seed in seeds]
    baseline_checkpoints = {rollout["checkpoint"] for rollout in baseline_rollouts}
    recovery_checkpoints = {rollout["checkpoint"] for rollout in recovery_rollouts}
    if len(baseline_checkpoints) != 1:
        raise ValueError(f"baseline arm uses multiple checkpoints: {sorted(baseline_checkpoints)}")
    if len(recovery_checkpoints) != 1:
        raise ValueError(f"recovery arm uses multiple checkpoints: {sorted(recovery_checkpoints)}")
    baseline_checkpoint = next(iter(baseline_checkpoints))
    recovery_checkpoint = next(iter(recovery_checkpoints))
    if baseline_checkpoint == recovery_checkpoint:
        raise ValueError("baseline and recovery checkpoints must differ")

    expected_conditions = baseline_rollouts[0]["conditions"]
    per_seed = []
    paired_counts = {key: 0 for key in ("both_success", "both_failure", "baseline_only", "recovery_only")}
    for seed, baseline, recovery in zip(seeds, baseline_rollouts, recovery_rollouts):
        for key in CONDITION_KEYS:
            _assert_same(key, expected_conditions[key], baseline["conditions"][key], seed)
            _assert_same(key, expected_conditions[key], recovery["conditions"][key], seed)
        if baseline["initial_scene_state_sha256"] != recovery["initial_scene_state_sha256"]:
            raise ValueError(
                f"initial scene state hash mismatch for seed {seed}: "
                f"baseline={baseline['initial_scene_state_sha256']!r}, "
                f"recovery={recovery['initial_scene_state_sha256']!r}"
            )
        _assert_initial_state_equal(
            "initial_cube_xyz", baseline["initial_cube_xyz"], recovery["initial_cube_xyz"], seed
        )
        _assert_initial_state_equal(
            "initial_joint_pos", baseline["initial_joint_pos"], recovery["initial_joint_pos"], seed
        )
        if baseline["success"] and recovery["success"]:
            paired_key = "both_success"
        elif not baseline["success"] and not recovery["success"]:
            paired_key = "both_failure"
        elif baseline["success"]:
            paired_key = "baseline_only"
        else:
            paired_key = "recovery_only"
        paired_counts[paired_key] += 1
        per_seed.append(
            {
                "seed": seed,
                "baseline": {"success": baseline["success"], "outcome": baseline["outcome"]},
                "recovery": {"success": recovery["success"], "outcome": recovery["outcome"]},
                "initial_state_equal": {
                    "initial_scene_state_sha256": True,
                    "initial_cube_xyz": True,
                    "initial_joint_pos": True,
                    "tolerance": INITIAL_STATE_TOLERANCE,
                },
                "initial_observation_sha256_equal": {
                    camera: baseline["hashes"][camera] == recovery["hashes"][camera]
                    for camera in ("front", "wrist")
                },
            }
        )

    def arm_summary(rollouts: list[dict], directory: Path, checkpoint: str) -> dict:
        successes = sum(rollout["success"] for rollout in rollouts)
        return {
            "directory": str(directory.resolve()),
            "checkpoint": checkpoint,
            "successes": successes,
            "success_rate": successes / len(rollouts),
            "outcome_counts": {
                outcome: sum(rollout["outcome"] == outcome for rollout in rollouts)
                for outcome in OUTCOMES
            },
        }

    baseline_summary = arm_summary(baseline_rollouts, baseline_dir, baseline_checkpoint)
    recovery_summary = arm_summary(recovery_rollouts, recovery_dir, recovery_checkpoint)
    return {
        "comparison": "paired exploratory comparison from one training seed",
        "interpretation": (
            f"This report compares {len(seeds)} paired rollout seeds from a single training seed; "
            "it does not establish statistical superiority."
        ),
        "seeds": seeds,
        "num_pairs": len(seeds),
        "conditions": expected_conditions,
        "baseline": baseline_summary,
        "recovery": recovery_summary,
        "delta_percentage_points": (
            recovery_summary["success_rate"] - baseline_summary["success_rate"]
        )
        * 100.0,
        "paired": paired_counts,
        "per_seed": per_seed,
    }


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    report = build_report(
        args.baseline_dir.expanduser().resolve(),
        args.recovery_dir.expanduser().resolve(),
        args.seeds,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
