"""Validate full preregistered CPU probe coverage and collect compact evidence.

Reads all eight immutable run reports, rechecks checkpoint/split contracts, and
independently checks three state-copy rows per run against raw HDF5 targets.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from probe_gripper_close_offline import CHUNK, FORMULAS, sha256, summarize, validate_inputs, verdict

REPO = Path(__file__).resolve().parents[2]
EXPECTED_MODELS = {
    "fixedmodel": ("act_reset_fixed_76_20260921", "act_reset_fixed_76_20260921_split",
                   "ebed7e5415bf1f5501e7b834bfcb5f97c84593335fda145822f3c71fe4aeebab"),
    "oldmodel": ("act_resume_10000_20260921", "act_pilot_76_20260921_split",
                 "a82ab0e5ddc2c7d11e2a6e16955885dae6cd562b2958542f90931fda79330f51"),
}


def check_identity(report, model):
    model_dir, split_dir, model_hash = EXPECTED_MODELS[model]
    expected_checkpoint = REPO / "outputs" / model_dir / "model/checkpoints/010000/pretrained_model"
    expected_split = REPO / "outputs" / split_dir
    if (Path(report["checkpoint"]).resolve() != expected_checkpoint.resolve()
            or Path(report["split_root"]).resolve() != expected_split.resolve()
            or report["model_sha256"] != model_hash):
        raise ValueError("report does not belong to preregistered checkpoint/split pair")
    return expected_checkpoint, expected_split


def check_coverage(report, contract, part):
    rows = report["rows"]
    actual = {(r["episode"], r["s"]) for r in rows}
    entries = contract["splits"][part]["episodes"]
    expected = ({(i, s) for i, e in enumerate(entries) for s in range(e["frame_count"] - CHUNK + 1)}
                if part == "valid" else {(20, s) for s in range(238, 339)})
    if actual != expected or len(actual) != len(rows):
        raise ValueError("incomplete/duplicate preregistered observation coverage")
    if part == "valid":
        counts = {band: sum(r["band"] == band and r["predictions"]["act"]["target_closed"] for r in rows)
                  for band in ("close", "post_close", "pre_close", "baseline")}
        if counts != {"close": 450, "post_close": 465, "pre_close": 0, "baseline": 3549}:
            raise ValueError(f"plan 0.5 band count mismatch: {counts}")
        if sum(e["frame_count"] for e in entries) != 7998 or len(report["excluded"]) != 435:
            raise ValueError("plan 0.5 total/tail count mismatch")
    return len(expected)


def independent_null_checks(report, contract, part):
    entries = contract["splits"][part]["episodes"]
    lower = np.asarray(contract["joint_limits"]["joint_lower_limits_rad"])
    upper = np.asarray(contract["joint_limits"]["joint_upper_limits_rad"])
    close_rows = [r for r in report["rows"] if r["band"] == "close"]
    results = []
    for index in np.linspace(0, len(close_rows) - 1, 3, dtype=int):
        row = close_rows[index]
        entry = entries[row["episode"]]
        s = row["s"]
        with h5py.File(entry["raw_path"], "r") as shard:
            group = shard["data"][entry["raw_demo"]]
            state = group["obs/joint_pos"][s].astype(np.float64)
            target = np.clip(group["obs/joint_pos_target"][s+1:s+31], lower, upper)
        independently_calculated = float(np.mean((state[5] - target[:, 5]) ** 2))
        delta = abs(independently_calculated - row["predictions"]["state_copy"]["gripper_mse"])
        if delta > 1e-6:
            raise ValueError("state-copy raw recomputation failed")
        results.append({"episode": row["episode"], "s": s, "state_copy_gripper_mse_delta": delta})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/grasp_signal_probe_20260921"))
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/grasp_close_signal_20260921.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    torch.set_num_threads(2)
    evidence = {"date": "2026-09-22", "stage": "1.1 CPU teacher forcing; 1.1.7 state-distance prerequisite only",
                "formulas": FORMULAS, "runs": {}, "input_validation": {}}
    for model in ("fixedmodel", "oldmodel"):
        for part in ("valid", "train20"):
            for path in ("lerobot", "raw"):
                key = f"{part}_{model}_{path}"
                file = args.root / f"{key}.json"
                report = json.loads(file.read_text())
                checkpoint, split = check_identity(report, model)
                contract = json.loads((split / "action_contract.json").read_text())
                split_part = "train" if part == "train20" else "valid"
                if report["part"] != split_part or report["input_path"] != path or report["device"] != "cpu":
                    raise ValueError("report identity mismatch")
                coverage = check_coverage(report, contract, split_part)
                if report["processor_overrides"] != {"device_processor": {"device": "cpu"}} or report["formulas"] != FORMULAS:
                    raise ValueError("processor device or preregistered formulas differ")
                names = ("act", "mean", "always_closed", "state_copy")
                recomputed = {name: summarize(report["rows"], name) for name in names}
                if report["metrics"] != recomputed or report["verdict"] != verdict(recomputed):
                    raise ValueError("reported metrics/verdict do not match row-level recomputation")
                for family, family_report in report["families"].items():
                    subset = [row for row in report["rows"] if row["family"] == family]
                    count = len({row["episode"] for row in subset})
                    family_metrics = {name: summarize(subset, name) for name in names}
                    family_verdict = "insufficient_family_episodes" if split_part == "valid" and count <= 2 else verdict(family_metrics)
                    if family_report != {"episodes": count, "metrics": family_metrics, "verdict": family_verdict}:
                        raise ValueError("family metrics/verdict do not match row-level recomputation")
                if not report["raw_stat_unchanged"] or not report["determinism"]["bit_equal"] or report["determinism"]["repeated_frames"] != 20:
                    raise ValueError("immutability/determinism verification failed")
                if model not in evidence["input_validation"]:
                    evidence["input_validation"][model] = validate_inputs(checkpoint, split)
                if report["model_sha256"] != evidence["input_validation"][model]["model_sha256"]:
                    raise ValueError("checkpoint content changed since inference")
                for name in ("split_provenance", "action_contract"):
                    if sha256(split / f"{name}.json") != evidence["input_validation"][model][f"{name}_sha256"]:
                        raise ValueError("split metadata changed during aggregation")
                    if "input_validation" in report and report["input_validation"][f"{name}_sha256"] != evidence["input_validation"][model][f"{name}_sha256"]:
                        raise ValueError("split metadata changed since inference")
                for raw_path, stat in report["raw_stat_before"].items():
                    current = Path(raw_path).stat()
                    if stat != [current.st_size, current.st_mtime_ns]:
                        raise ValueError(f"raw changed: {raw_path}")
                evidence["runs"][key] = {"artifact": str(file), "sha256": sha256(file),
                    "coverage_verified_frames": coverage,
                    "selection": {"part": split_part, "stride": 1, "max_samples": None,
                                  "episode": 20 if part == "train20" else None,
                                  "frame_range": [238, 338] if part == "train20" else None},
                    "metrics": report["metrics"], "verdict": report["verdict"],
                    "families": {f: {"episodes": v["episodes"], "verdict": v["verdict"], "act": v["metrics"]["act"]}
                                 for f, v in report["families"].items()},
                    "pixel_equivalence": report["pixel_equivalence"], "determinism": report["determinism"],
                    "independent_raw_null_checks": independent_null_checks(report, contract, split_part)}
    closed = json.loads((args.root / "closed_loop_prerequisites.json").read_text())
    from inspect_close_window_trace import inspect
    expected_trace = REPO / "outputs/initial_frame_diag_20260921/rollout_demo9_fixedmodel_n30/trace_001.json"
    expected_raw = REPO / "outputs/mimic_vision224_500_20260913/raw/shard_009.hdf5"
    if (Path(closed["trace"]).resolve() != expected_trace or Path(closed["raw_path"]).resolve() != expected_raw
            or closed["demo"] != "demo_9" or closed["probe_status"] != "unresolved_missing_policy_images"):
        raise ValueError("closed-loop prerequisite report identity mismatch")
    with h5py.File(expected_raw, "r") as shard:
        recalculated = inspect(json.loads(expected_trace.read_text()), shard["data/demo_9/obs/joint_pos"][:], 278)
    for field, value in recalculated.items():
        if closed[field] != value:
            raise ValueError(f"closed-loop prerequisite recomputation mismatch: {field}")
    evidence["closed_loop"] = {k: v for k, v in closed.items() if k != "records"}
    evidence["closed_loop"]["artifact"] = str(args.root / "closed_loop_prerequisites.json")
    evidence["closed_loop"]["artifact_sha256"] = sha256(args.root / "closed_loop_prerequisites.json")
    evidence["closed_loop"]["trace_sha256"] = sha256(expected_trace)
    evidence["limitations"] = [
        "No Isaac, training, physical robot, raw-data mutation, or checkpoint mutation in this CPU pass.",
        "Missing per-step policy PNGs prevent stage 1.1.7 inference and the two-axis causal verdict.",
        "Pixel equivalence is measured against the preregistered 5/255 bound; failed paths remain separate.",
        "Early lerobot reports predate provenance hardening; this aggregation verifies their exact frame coverage and checkpoint/split contracts without rewriting them.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: {"verdict": v["verdict"], "close_hit": v["metrics"]["act"]["close"]["chunk_close_hit"]}
                      for k, v in evidence["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
