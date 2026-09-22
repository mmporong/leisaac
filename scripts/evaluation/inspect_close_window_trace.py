"""Read-only CPU prerequisite check for plan 1.1.7; no fabricated image inputs."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from probe_gripper_close_offline import band_for, first_close, trace_index


def inspect(trace, states, c, start=240, end=340):
    records = []
    for entry in trace:
        step = entry["step"]
        if not start <= step <= end:
            continue
        s = trace_index(step)
        difference = np.abs(np.asarray(entry["state_before"]) - states[s])
        records.append({"trace_step": step, "s": s, "band": band_for(s, c),
                        "per_joint_abs_difference_rad": difference.tolist(),
                        "max_abs_difference_rad": float(difference.max()),
                        "retained": bool(np.all(difference <= .2))})
    if [r["trace_step"] for r in records] != list(range(start, end + 1)):
        raise ValueError("trace does not cover every preregistered step")
    close = [r for r in records if r["band"] == "close"]
    kept = sum(r["retained"] for r in close)
    return {"window": [start, end], "raw_close_index": c, "steps": len(records),
            "close_steps": len(close), "retained_close_steps": kept,
            "excluded_close_steps": len(close) - kept,
            "minimum_retained_close_steps": 15, "state_distance_gate_passed": kept >= 15,
            "close_max_joint_distance_quantiles_rad": np.quantile(
                [r["max_abs_difference_rad"] for r in close], [0, .25, .5, .75, 1]).tolist(),
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--demo", default="demo_9")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with h5py.File(args.raw_path, "r") as shard:
        group = shard["data"][args.demo]
        states = group["obs/joint_pos"][:]
        c = first_close(group["obs/joint_pos_target"][:])
    report = inspect(json.loads(args.trace.read_text()), states, c)
    report.update({"trace": str(args.trace), "raw_path": str(args.raw_path), "demo": args.demo,
                   "probe_status": "unresolved_missing_policy_images",
                   "blocker": "Existing rollout saved initial PNGs only, not policy input PNGs at steps 240..340. Stage 1.2 requires Isaac and remains deferred.",
                   "interpretation": "State-distance prerequisite only; no image-conditioned ACT inference or closed-loop hit measured.",
                   "counterfactual_note": "Most per-frame queries would be counterfactual because rollout n_action_steps=30.",
                   "available_png_files": sorted(str(p) for p in args.trace.parent.rglob("*.png"))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
