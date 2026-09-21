"""Compare a policy rollout trace with the successful raw demo that shares its initial physics state.

Alignment (see replay_joint_contract.aligned_commands): the evaluator trace entry with
``step == s`` (1-based) holds the command applied during transition s-1 -> s, which
corresponds to the demo's ``obs/joint_pos_target[s]``; its ``state_before`` corresponds to
``obs/joint_pos[s-1]`` and ``state_after`` to ``obs/joint_pos[s]``.

CPU only; reads the trace JSON and the raw HDF5, writes one JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def first_index(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, default=None, help="evaluation.json next to the trace")
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--demo", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 10, 30, 60, 120, 240])
    parser.add_argument("--gripper-closed-rad", type=float, default=0.5)
    parser.add_argument("--lift-threshold-m", type=float, default=0.02)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    if not trace:
        raise ValueError("empty trace")
    steps = np.array([entry["step"] for entry in trace])
    if not np.array_equal(steps, np.arange(1, len(trace) + 1)):
        raise ValueError("trace steps must be 1..N without gaps")
    requested = np.array([entry["requested_action"] for entry in trace], dtype=np.float64)
    applied = np.array([entry["applied_action"] for entry in trace], dtype=np.float64)
    state_before = np.array([entry["state_before"] for entry in trace], dtype=np.float64)
    state_after = np.array([entry["state_after"] for entry in trace], dtype=np.float64)
    cube = np.array([entry["cube_xyz"] for entry in trace], dtype=np.float64)

    with h5py.File(args.raw_path, "r") as dataset:
        demo = dataset[f"data/{args.demo}"]
        target = demo["obs/joint_pos_target"][:].astype(np.float64)
        joint_pos = demo["obs/joint_pos"][:].astype(np.float64)
        demo_cube = demo["states/rigid_object/cube/root_pose"][:, :3].astype(np.float64)
        initial_cube = demo["initial_state/rigid_object/cube/root_pose"][0, :3].astype(np.float64)

    n = min(len(trace), len(target) - 1)
    command_error = requested[:n] - target[1 : n + 1]
    state_error = state_after[:n] - joint_pos[1 : n + 1]

    def per_joint(values: np.ndarray) -> dict:
        return dict(zip(JOINTS, np.round(values, 6).tolist()))

    windows = {}
    for w in args.windows:
        if w > n:
            continue
        windows[str(w)] = {
            "command_rmse_rad": per_joint(np.sqrt(np.mean(command_error[:w] ** 2, axis=0))),
            "command_rmse_all_rad": float(np.sqrt(np.mean(command_error[:w] ** 2))),
            "state_rmse_rad": per_joint(np.sqrt(np.mean(state_error[:w] ** 2, axis=0))),
            "state_rmse_all_rad": float(np.sqrt(np.mean(state_error[:w] ** 2))),
            "policy_command_at_step": per_joint(requested[w - 1]),
            "demo_target_at_step": per_joint(target[w]),
            "policy_state_after_step": per_joint(state_after[w - 1]),
            "demo_state_after_step": per_joint(joint_pos[w]),
        }

    demo_lift = demo_cube[:, 2] - initial_cube[2]
    policy_lift = cube[:, 2] - cube[0, 2]
    policy_cube_xy_drift = np.linalg.norm(cube[:, :2] - cube[0, :2], axis=1)
    demo_cube_xy_drift = np.linalg.norm(demo_cube[:, :2] - initial_cube[:2], axis=1)

    report = {
        "trace": str(args.trace.resolve()),
        "raw_path": str(args.raw_path.resolve()),
        "demo": args.demo,
        "compared_steps": int(n),
        "trace_steps": int(len(trace)),
        "demo_frames": int(len(target)),
        "initial_state_match": {
            "policy_state_before_step1": per_joint(state_before[0]),
            "demo_joint_pos_frame0": per_joint(joint_pos[0]),
            "policy_initial_cube_xyz": np.round(cube[0], 6).tolist(),
            "demo_initial_cube_xyz": np.round(initial_cube, 6).tolist(),
            "max_joint_abs_diff_rad": float(np.abs(state_before[0] - joint_pos[0]).max()),
        },
        "windows": windows,
        "gripper": {
            "demo_first_target_below_threshold_step": first_index(target[:, 5] < args.gripper_closed_rad),
            "policy_first_command_below_threshold_step": (
                None if first_index(requested[:, 5] < args.gripper_closed_rad) is None
                else first_index(requested[:, 5] < args.gripper_closed_rad) + 1
            ),
            "policy_min_gripper_command_rad": float(requested[:, 5].min()),
            "policy_min_gripper_command_step": int(requested[:, 5].argmin()) + 1,
            "demo_min_gripper_target_rad": float(target[:, 5].min()),
        },
        "cube": {
            "demo_first_lift_step": first_index(demo_lift > args.lift_threshold_m),
            "demo_max_lift_m": float(demo_lift.max()),
            "policy_max_lift_in_trace_m": float(policy_lift.max()),
            "policy_max_xy_drift_in_trace_m": float(policy_cube_xy_drift.max()),
            "policy_first_xy_drift_over_5mm_step": (
                None if first_index(policy_cube_xy_drift > 0.005) is None
                else first_index(policy_cube_xy_drift > 0.005) + 1
            ),
            "demo_first_xy_drift_over_5mm_frame": first_index(demo_cube_xy_drift > 0.005),
        },
        "clipping": {
            "steps_with_clipped_command": int(np.any(np.abs(requested[:n] - applied[:n]) > 1e-9, axis=1).sum()),
        },
    }
    if args.evaluation is not None and args.evaluation.exists():
        evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
        report["evaluation_summary"] = {
            "n_action_steps": evaluation.get("n_action_steps"),
            "reset_render_frames": evaluation.get("reset_render_frames"),
            "initial_state_source": evaluation.get("initial_state_source"),
            "results": [
                {k: r.get(k) for k in ("seed", "success", "steps", "outcome", "max_cube_lift_m",
                                        "clipped_steps", "initial_scene_state_sha256", "first_action")}
                for r in evaluation.get("results", [])
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "windows"}, indent=2))
    for w, entry in windows.items():
        print(f"window {w}: command_rmse_all={entry['command_rmse_all_rad']:.4f} state_rmse_all={entry['state_rmse_all_rad']:.4f}")


if __name__ == "__main__":
    main()
