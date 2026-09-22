"""CPU-only, teacher-forced ACT chunk diagnostics; never trains or starts Isaac.

Indices follow plan_r5: raw close index c, LeRobot observation index s,
target chunk raw_target[s+1:s+31]. All prediction metrics use radians.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

CHUNK = 30
BANDS = ("pre_close", "close", "post_close", "baseline")
FORMULAS = {
    "c": "first index of raw_target[1:,5] < 0.5, plus 1; c==1 excluded as degenerate",
    "target": "raw_target[s+1:s+31]",
    "valid": "s+30 <= len(raw_target)-1",
    "pre_close": "c-60 <= s <= c-31",
    "close": "c-30 <= s <= c-1",
    "post_close": "c <= s <= c+30",
    "baseline": "all remaining valid s",
    "clipped_target_share": "fraction of future target frames (including repeated chunk exposures) with any raw joint outside contract limits",
}


def first_close(target):
    indices = np.flatnonzero(np.asarray(target)[1:, 5] < 0.5)
    return int(indices[0] + 1) if len(indices) else None


def band_for(s, c):
    if c - 60 <= s <= c - 31:
        return "pre_close"
    if c - 30 <= s <= c - 1:
        return "close"
    if c <= s <= c + 30:
        return "post_close"
    return "baseline"


def trace_index(step):
    if step < 1:
        raise ValueError("trace step is one-based")
    return step - 1


def closed_loop_status(retained_close, hit):
    if retained_close < 15 or hit is None:
        return "unresolved"
    return "high" if hit >= 0.8 else "low" if hit <= 0.4 else "partial"


def predict_chunk(model, preprocessor, postprocessor, sample):
    import torch
    observation = {k: v.unsqueeze(0).cpu() for k, v in sample.items() if k.startswith("observation.")}
    model.reset()
    with torch.inference_mode():
        prediction = postprocessor(model.predict_action_chunk(preprocessor(observation)))
    result = prediction.squeeze(0).detach().cpu().numpy()
    if result.shape != (CHUNK, 6) or not np.isfinite(result).all():
        raise ValueError(f"expected finite (30,6) rad chunk, got {result.shape}")
    return result


def metric_row(prediction, target):
    pred = np.flatnonzero(prediction[:, 5] < 0.5)
    truth = np.flatnonzero(target[:, 5] < 0.5)
    error = np.asarray(prediction, dtype=np.float64) - target
    return {
        "prediction_closed": bool(len(pred)), "target_closed": bool(len(truth)),
        "close_index_error": int(pred[0] - truth[0]) if len(pred) and len(truth) else None,
        "gripper_mse": float(np.mean(error[:, 5] ** 2)),
        "arm_mse": float(np.mean(error[:, :5] ** 2)),
    }


def summarize(rows, name):
    bands = {}
    for band in BANDS:
        group = [r for r in rows if r["band"] == band]
        metrics = [r["predictions"][name] for r in group]
        positives = [r for r in metrics if r["target_closed"]]
        errors = [r["close_index_error"] for r in positives if r["close_index_error"] is not None]
        bands[band] = {
            "frames": len(group), "target_positive_frames": len(positives),
            "chunk_close_hit": sum(r["prediction_closed"] for r in positives) / len(positives) if positives else None,
            "prediction_closed_fraction": sum(r["prediction_closed"] for r in metrics) / len(metrics) if metrics else None,
            "close_index_error_denominator": len(positives),
            "close_index_error_median": float(np.median(errors)) if errors else None,
            "prediction_exists_fraction": len(errors) / len(positives) if positives else None,
            "no_prediction": len(positives) - len(errors),
            "gripper_rmse": float(np.sqrt(np.mean([r["gripper_mse"] for r in metrics]))) if metrics else None,
            "arm_rmse": float(np.sqrt(np.mean([r["arm_mse"] for r in metrics]))) if metrics else None,
            "clipped_target_share": float(np.mean([r["clipped_target_share"] for r in group])) if group else None,
        }
    bands["pre_close_false_alarm"] = bands["pre_close"]["prediction_closed_fraction"]
    return bands


def passes_thresholds(summary):
    close = summary["close"]
    return bool(close["chunk_close_hit"] is not None and close["chunk_close_hit"] >= .8
                and close["close_index_error_median"] is not None
                and abs(close["close_index_error_median"]) <= 5
                and close["prediction_exists_fraction"] >= .8
                and summary["pre_close_false_alarm"] is not None
                and summary["pre_close_false_alarm"] <= .2)


def verdict(summaries):
    hit = summaries["act"]["close"]["chunk_close_hit"]
    if hit is None:
        return "unresolved"
    if hit < .2:
        return "not_learned"
    if passes_thresholds(summaries["act"]) and not any(passes_thresholds(summaries[n]) for n in ("mean", "always_closed", "state_copy")):
        return "teacher_forcing_learned"
    return "partial_or_unresolved"


def null_chunks(state, action_mean):
    # Only gripper differs between nulls; arm channels are current-state copy,
    # reported for completeness, never used by the gripper decision gate.
    base = np.broadcast_to(state, (CHUNK, 6)).copy()
    mean = base.copy()
    mean[:, 5] = action_mean[5]
    closed = base.copy()
    closed[:, 5] = 0
    return {"mean": mean, "always_closed": closed, "state_copy": base}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(checkpoint, split_root):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.imitation_learning.run_act_vision_experiment import validate_split, verify_checkpoint
    validate_split(split_root.resolve())
    result = verify_checkpoint(checkpoint.resolve(), split_root.resolve() / "train", 10000)
    result["split_provenance_sha256"] = sha256(split_root / "split_provenance.json")
    result["action_contract_sha256"] = sha256(split_root / "action_contract.json")
    result["processor_sha256"] = {p.name: sha256(p) for p in sorted(checkpoint.glob("policy_*")) if p.is_file()}
    return result


def registered_selection(part, stride, episode, frame_range, max_samples):
    if stride != 1 or max_samples is not None:
        return False
    return (part == "valid" and episode is None and frame_range is None) or (
        part == "train" and episode == 20 and frame_range == [238, 338])


def aligned_state(state, teacher_state):
    state, teacher_state = (np.asarray(value, dtype=np.float64) for value in (state, teacher_state))
    if any(value.shape != (6,) or not np.isfinite(value).all() for value in (state, teacher_state)):
        raise ValueError("state comparison requires two finite six-joint vectors")
    difference = np.abs(state - teacher_state)
    return bool(np.all(difference <= .2)), float(difference.max())


def run_closed_loop(args, model, pre, post, contract, input_validation):
    """Paired, counterfactual chunk queries on verified recorded rollout inputs."""
    import torch
    from compare_close_window_inputs import analyze, load_json, load_trace, validate_manifest

    if (args.part != "train" or args.episode != 20 or args.input_path != "raw"
            or args.stride != 1 or args.max_samples is not None or args.frame_range is not None):
        raise ValueError("closed-loop mode requires train episode 20, raw inputs, stride 1, no subsampling")
    entry = contract["splits"]["train"]["episodes"][20]
    if entry["raw_demo"] != "demo_9" or entry["aggregate_episode_index"] != 24:
        raise ValueError("closed-loop diagnostic must use registered train episode 20 / aggregate 24 / demo_9")
    raw_path = Path(entry["raw_path"])
    comparison = analyze(args.closed_loop_dir, args.reference_dir, raw_path, entry["raw_demo"])
    if not comparison["strict_reproduction_gate"]["pass"]:
        raise ValueError("strict scene/config/420-step replay gate failed; closed-loop inference is blocked")
    manifest_path = args.closed_loop_dir / "policy_images_001.json"
    inputs = validate_manifest(args.closed_loop_dir, load_json(manifest_path),
                               load_trace(args.closed_loop_dir / "trace_001.json"))
    if sha256(manifest_path) != comparison["inputs"]["manifest_sha256"]:
        raise ValueError("manifest changed between validation and inference snapshot")
    evaluation = load_json(args.closed_loop_dir / "evaluation.json")
    if Path(evaluation["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("dump and probe checkpoints differ")
    with h5py.File(raw_path, "r") as shard:
        group = shard["data"][entry["raw_demo"]]
        targets, states = group["obs/joint_pos_target"][:], group["obs/joint_pos"][:]
        images = {camera: group[f"obs/{camera}"][:] for camera in ("front", "wrist")}
    lower = np.asarray(contract["joint_limits"]["joint_lower_limits_rad"])
    upper = np.asarray(contract["joint_limits"]["joint_upper_limits_rad"])
    mean = np.asarray(json.loads((args.split_root / "train/meta/stats.json").read_text())["action"]["mean"]).reshape(6)
    c = first_close(targets)
    model_hash = sha256(args.checkpoint / "model.safetensors")
    raw_hash = sha256(raw_path)
    rows, teacher_rows, excluded, deterministic = [], [], [], []
    for step, observation in inputs.items():
        s = trace_index(step)
        retained, distance = aligned_state(observation["state_before"], states[s])
        if not retained:
            excluded.append({"step": step, "s": s, "band": band_for(s, c), "max_state_distance_rad": distance})
            continue
        if s + CHUNK >= len(targets):
            raise ValueError("closed-loop window lacks full target chunk")
        pair = []
        for kind in ("closed_loop", "paired_teacher"):
            state = observation["state_before"] if kind == "closed_loop" else states[s]
            sample = {"observation.state": torch.from_numpy(np.asarray(state, dtype=np.float32))}
            for camera in ("front", "wrist"):
                pixels = observation[camera]["pixels"] if kind == "closed_loop" else images[camera][s]
                sample[f"observation.images.{camera}"] = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255.
            prediction = predict_chunk(model, pre, post, sample)
            if len(deterministic) < 20:
                deterministic.append(bool(np.array_equal(prediction, predict_chunk(model, pre, post, sample))))
                if not deterministic[-1]:
                    raise ValueError("CPU prediction not bit deterministic")
            raw_target = targets[s + 1:s + CHUNK + 1]
            target = np.clip(raw_target, lower, upper)
            pair.append({"step": step, "s": s, "c": c, "band": band_for(s, c),
                         "actual_chunk_query_boundary": s % 30 == 0,
                         "max_state_distance_rad": distance,
                         "clipped_target_share": float(np.any(raw_target != target, axis=1).mean()),
                         "predictions": {name: metric_row(chunk, target) for name, chunk in
                                         {"act": prediction, **null_chunks(state, mean)}.items()}})
        rows.append(pair[0])
        teacher_rows.append(pair[1])
    names = ("act", "mean", "always_closed", "state_copy")
    summaries = {name: summarize(rows, name) for name in names}
    teacher = {name: summarize(teacher_rows, name) for name in names}
    close = summaries["act"]["close"]
    status = closed_loop_status(close["frames"], close["chunk_close_hit"])
    if sha256(raw_path) != raw_hash or sha256(args.checkpoint / "model.safetensors") != model_hash:
        raise ValueError("raw data or model changed during diagnostic")
    after_comparison = analyze(args.closed_loop_dir, args.reference_dir, raw_path, entry["raw_demo"])
    if after_comparison != comparison:
        raise ValueError("rollout/reference inputs changed during diagnostic")
    report = {"mode": "paired_closed_loop_counterfactual", "checkpoint": str(args.checkpoint.resolve()),
              "model_sha256": model_hash, "input_validation": input_validation,
              "device": "cpu", "processor_overrides": {"device_processor": {"device": "cpu"}},
              "input_comparison": comparison, "formulas": FORMULAS,
              "samples": len(rows), "excluded": excluded, "metrics": summaries,
              "paired_teacher_metrics": teacher, "paired_teacher_verdict": verdict(teacher),
              "closed_loop_status": status,
              "scope_limit": "One scene, step-aligned close window; most queries are counterfactual, not grasp trials",
              "determinism": {"repeated_queries": len(deterministic), "bit_equal": all(deterministic)},
              "raw_sha256": raw_hash, "inputs_unchanged": True,
              "unchanged_scope": ["raw shard", "model weights", "manifest", "candidate/reference evaluation and trace",
                                  "202 PNG file and pixel hashes"],
              "rows": rows, "paired_teacher_rows": teacher_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({"samples": len(rows), "retained_close": close["frames"],
                      "chunk_close_hit": close["chunk_close_hit"], "closed_loop_status": status,
                      "paired_teacher_verdict": report["paired_teacher_verdict"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--part", choices=("train", "valid"), required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--force-processor-device", choices=("cpu",), default="cpu")
    parser.add_argument("--input-path", choices=("lerobot", "raw"), required=True)
    parser.add_argument("--null-models", nargs="+", default=["mean", "always_closed", "state_copy"])
    parser.add_argument("--episode", type=int)
    parser.add_argument("--frame-range", type=int, nargs=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--closed-loop-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path,
                        default=Path("outputs/initial_frame_diag_20260921/rollout_demo9_fixedmodel_n30"))
    parser.add_argument("--source-hdf5", type=Path, default=Path("datasets/pick_cube_into_box_annotated_wrist_10_20260911.hdf5"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.stride < 1 or args.threads < 1 or set(args.null_models) != {"mean", "always_closed", "state_copy"}:
        raise ValueError("positive stride/threads and all three preregistered nulls required")

    import torch
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy, ACTConfig
    torch.set_num_threads(args.threads)
    torch.manual_seed(43)
    torch.use_deterministic_algorithms(True)
    input_validation = validate_inputs(args.checkpoint, args.split_root)
    config = ACTConfig.from_pretrained(args.checkpoint)
    config.device = "cpu"
    model = ACTPolicy.from_pretrained(args.checkpoint, config=config, local_files_only=True).cpu().eval()
    if config.chunk_size != CHUNK:
        raise ValueError("plan requires checkpoint chunk_size=30")
    override = {"device_processor": {"device": "cpu"}}
    pre, post = make_pre_post_processors(config, pretrained_path=str(args.checkpoint),
                                        preprocessor_overrides=override, postprocessor_overrides=override)
    provenance = json.loads((args.split_root / "split_provenance.json").read_text())
    contract = json.loads((args.split_root / "action_contract.json").read_text())
    if args.closed_loop_dir is not None:
        run_closed_loop(args, model, pre, post, contract, input_validation)
        return
    entries = contract["splits"][args.part]["episodes"]
    dataset = LeRobotDataset(provenance["splits"][args.part]["repo_id"], root=args.split_root / args.part)
    lower = np.asarray(contract["joint_limits"]["joint_lower_limits_rad"])
    upper = np.asarray(contract["joint_limits"]["joint_upper_limits_rad"])
    mean = np.asarray(json.loads((args.split_root / "train/meta/stats.json").read_text())["action"]["mean"]).reshape(6)
    families = {}
    with h5py.File(args.source_hdf5, "r") as source:
        for name, demo in source["data"].items():
            c = first_close(demo["obs/joint_pos_target"][:])
            generated_c = c if c == 1 else c + 6
            if generated_c in families:
                raise ValueError("ambiguous source close index")
            families[generated_c] = "src:" + name
    raw_paths = sorted({e["raw_path"] for e in entries})
    before = {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in raw_paths}
    model_hash = sha256(args.checkpoint / "model.safetensors")
    rows, excluded, deterministic = [], [], []
    offset = 0
    pixel_max = {"front": 0., "wrist": 0.}
    pixel_fail = {"front": 0, "wrist": 0}
    for episode, entry in enumerate(entries):
        length = entry["frame_count"]
        if args.episode is not None and episode != args.episode:
            offset += length
            continue
        with h5py.File(entry["raw_path"], "r") as shard:
            group = shard["data"][entry["raw_demo"]]
            targets = group["obs/joint_pos_target"][:]
            states = group["obs/joint_pos"][:]
            images = {camera: group[f"obs/{camera}"][:] for camera in ("front", "wrist")}
        if length != len(targets) - 1:
            raise ValueError("contract raw length mismatch")
        c = first_close(targets)
        if c is None or c == 1:
            excluded.append({"episode": episode, "reason": "degenerate_c1" if c == 1 else "no_close", "frames": length})
            offset += length
            continue
        family = families[c]
        for s in range(0, length, args.stride):
            if args.frame_range and not args.frame_range[0] <= s <= args.frame_range[1]:
                continue
            if s + CHUNK > len(targets) - 1:
                excluded.append({"episode": episode, "s": s, "reason": "tail_truncated"})
                continue
            sample = dataset[offset + s]
            if int(sample["episode_index"]) != episode or int(sample["frame_index"]) != s:
                raise ValueError("LeRobot index alignment mismatch")
            if not np.allclose(sample["observation.state"].numpy(), states[s], atol=1e-6, rtol=0):
                raise ValueError("raw/LeRobot state mismatch")
            raw_sample = {"observation.state": torch.from_numpy(states[s].astype(np.float32))}
            for camera in images:
                key = f"observation.images.{camera}"
                raw_sample[key] = torch.from_numpy(images[camera][s].copy()).permute(2, 0, 1).float() / 255.
                difference = float((raw_sample[key] - sample[key]).abs().max())
                pixel_max[camera] = max(pixel_max[camera], difference)
                pixel_fail[camera] += difference > 5 / 255 + 1e-7
            chosen = sample if args.input_path == "lerobot" else raw_sample
            prediction = predict_chunk(model, pre, post, chosen)
            if len(deterministic) < 20:
                repeat = predict_chunk(model, pre, post, chosen)
                deterministic.append(bool(np.array_equal(prediction, repeat)))
                if not deterministic[-1]:
                    raise ValueError("CPU prediction not bit deterministic")
            raw_target = targets[s + 1:s + 31]
            target = np.clip(raw_target, lower, upper)
            if not np.allclose(sample["action"].numpy(), target[0], atol=1e-6, rtol=0):
                raise ValueError("recorded_target t+1 contract mismatch")
            predictions = {"act": prediction, **null_chunks(states[s], mean)}
            rows.append({"episode": episode, "aggregate_episode_index": entry["aggregate_episode_index"],
                         "raw_shard": Path(entry["raw_path"]).name, "raw_demo": entry["raw_demo"],
                         "family": family, "s": s, "c": c, "band": band_for(s, c),
                         "clipped_target_share": float(np.any(raw_target != target, axis=1).mean()),
                         "predictions": {n: metric_row(p, target) for n, p in predictions.items()}})
            if len(rows) % 250 == 0:
                print(f"{args.output.name}: {len(rows)} frames; episode={episode} s={s}", flush=True)
            if args.max_samples and len(rows) >= args.max_samples:
                break
        offset += length
        if args.max_samples and len(rows) >= args.max_samples:
            break
    names = ("act", "mean", "always_closed", "state_copy")
    summaries = {name: summarize(rows, name) for name in names}
    registered = registered_selection(args.part, args.stride, args.episode, args.frame_range, args.max_samples)
    per_family = {}
    for family in sorted(set(families.values())):
        subset = [r for r in rows if r["family"] == family]
        count = len({r["episode"] for r in subset})
        metrics = {name: summarize(subset, name) for name in names}
        per_family[family] = {"episodes": count, "metrics": metrics,
                              "verdict": ("diagnostic_unregistered_selection" if not registered else
                                          "insufficient_family_episodes" if args.part == "valid" and count <= 2 else verdict(metrics))}
    after = {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in raw_paths}
    if before != after or model_hash != sha256(args.checkpoint / "model.safetensors"):
        raise ValueError("input changed during analysis")
    report = {"checkpoint": str(args.checkpoint), "model_sha256": model_hash,
              "split_root": str(args.split_root), "part": args.part, "input_path": args.input_path,
              "selection": {"stride": args.stride, "episode": args.episode, "frame_range": args.frame_range,
                            "max_samples": args.max_samples, "preregistered": registered},
              "input_validation": input_validation,
              "device": "cpu", "processor_overrides": override, "formulas": FORMULAS,
              "samples": len(rows), "excluded": excluded, "metrics": summaries,
              "verdict": verdict(summaries) if registered else "diagnostic_unregistered_selection", "families": per_family,
              "determinism": {"repeated_frames": len(deterministic), "bit_equal": all(deterministic)},
              "pixel_equivalence": {"threshold": 5 / 255, "max_abs_difference": pixel_max,
                                    "exceeding_frames": pixel_fail, "checked_frames": len(rows),
                                    "passed": not any(pixel_fail.values())},
              "target_metrics": "converter-clipped future labels; clipping itself measured from unmodified raw targets",
              "null_arm_channels": "current state copy; not a decision metric",
              "raw_stat_before": before, "raw_stat_unchanged": before == after, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("samples", "verdict", "determinism", "pixel_equivalence")}), flush=True)


if __name__ == "__main__":
    main()
