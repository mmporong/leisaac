"""Validate a close-window policy-image dump and compare it with the raw demonstration.

The strict reproduction gate compares the new 420-step rollout trace with the
pre-existing demo-9 rollout. Image and state comparisons use step indices only:
raw/LeRobot frame ``s = completed_step - 1``.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.imitation_learning.action_contract import sha256_file


DEFAULT_REFERENCE = REPO / "outputs/initial_frame_diag_20260921/rollout_demo9_fixedmodel_n30"
DEFAULT_RAW = REPO / "outputs/mimic_vision224_500_20260913/raw/shard_009.hdf5"
EXPECTED_SCENE_SHA256 = "375aaad23fe61b89eb0eb3c79037f53acb55348823dd674385f9d1c0c66d2205"
EXPECTED_TRACE_STEPS = 420
EXPECTED_DUMP_STEPS = list(range(240, 341))
TRACE_FIELDS = ("state_before", "requested_action", "applied_action")
CONFIG_FIELDS = (
    "task",
    "checkpoint",
    "num_rollouts",
    "seed_start",
    "horizon",
    "server_seed",
    "server_device",
    "n_action_steps",
    "reset_render_frames",
    "initial_state_source",
    "trace_steps",
    "gripper_effort_mode",
    "control_dt_s",
    "render_width",
    "render_height",
    "policy_image_size",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def completed_step_to_raw_frame(step: int) -> int:
    if type(step) is not int or step < 1:
        raise ValueError("completed step must be a positive integer")
    return step - 1


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_trace(path: Path) -> list[dict]:
    trace = load_json(path)
    if not isinstance(trace, list):
        raise ValueError(f"trace must be a list: {path}")
    expected_steps = list(range(1, len(trace) + 1))
    actual_steps = [row.get("step") for row in trace]
    if actual_steps != expected_steps:
        raise ValueError(f"trace step sequence must be 1..{len(trace)}: {path}")
    for row in trace:
        for field in TRACE_FIELDS:
            values = np.asarray(row.get(field), dtype=np.float64)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError(f"invalid trace {field} at step {row['step']}: {path}")
    return trace


def compare_trace(reference: list[dict], candidate: list[dict]) -> dict:
    count_match = len(reference) == len(candidate) == EXPECTED_TRACE_STEPS
    step_sequence_match = (
        [row["step"] for row in reference] == [row["step"] for row in candidate]
        == list(range(1, EXPECTED_TRACE_STEPS + 1))
    )
    fields = {}
    for field in TRACE_FIELDS:
        reference_values = np.asarray([row[field] for row in reference], dtype=np.float64)
        candidate_values = np.asarray([row[field] for row in candidate], dtype=np.float64)
        if reference_values.shape != candidate_values.shape:
            fields[field] = {
                "bit_identical": False,
                "max_abs_difference": None,
                "mismatched_scalars": None,
                "shape_match": False,
            }
            continue
        delta = np.abs(reference_values - candidate_values)
        fields[field] = {
            "bit_identical": bool(np.array_equal(reference_values, candidate_values)),
            "max_abs_difference": float(delta.max()) if delta.size else 0.0,
            "mismatched_scalars": int(np.count_nonzero(reference_values != candidate_values)),
            "shape_match": True,
        }
    bit_identical = count_match and step_sequence_match and all(
        record["bit_identical"] for record in fields.values()
    )
    return {
        "expected_steps": EXPECTED_TRACE_STEPS,
        "reference_steps": len(reference),
        "candidate_steps": len(candidate),
        "count_match": count_match,
        "step_sequence_match": step_sequence_match,
        "fields": fields,
        "bit_identical": bit_identical,
    }


def compare_required_config(reference: dict, candidate: dict) -> dict[str, dict]:
    comparison = {}
    for field in CONFIG_FIELDS:
        reference_present = field in reference
        candidate_present = field in candidate
        comparison[field] = {
            "reference": reference.get(field),
            "candidate": candidate.get(field),
            "reference_present": reference_present,
            "candidate_present": candidate_present,
            "equal": (
                reference_present
                and candidate_present
                and reference[field] == candidate[field]
            ),
        }
    return comparison


def write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def safe_manifest_path(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("manifest image path must be a non-empty string")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"manifest image path must be relative: {relative_path}")
    resolved_root = root.resolve(strict=True)
    resolved = (resolved_root / candidate).resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"manifest image path escapes rollout directory: {relative_path}")
    return resolved


def load_rgb_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError(f"policy image must be an RGB PNG: {path}")
        pixels = np.asarray(image)
    if pixels.shape != (224, 224, 3) or pixels.dtype != np.uint8:
        raise ValueError(f"policy image must be uint8 (224,224,3): {path}, got {pixels.shape}/{pixels.dtype}")
    return np.ascontiguousarray(pixels)


def validate_manifest(rollout_dir: Path, manifest: list[dict], trace: list[dict]) -> dict[int, dict]:
    if not isinstance(manifest, list):
        raise ValueError("policy image manifest must be a top-level list")
    steps = [row.get("step") for row in manifest]
    if steps != EXPECTED_DUMP_STEPS:
        raise ValueError("policy image manifest must contain ordered steps 240..340 exactly once")
    trace_by_step = {row["step"]: row for row in trace}
    validated = {}
    for row in manifest:
        step = row["step"]
        if row.get("observation_index") != completed_step_to_raw_frame(step):
            raise ValueError(f"manifest observation_index must equal step-1 at step {step}")
        state = np.asarray(row.get("state_before"), dtype=np.float64)
        trace_state = np.asarray(trace_by_step[step]["state_before"], dtype=np.float64)
        if state.shape != (6,) or not np.isfinite(state).all() or not np.array_equal(state, trace_state):
            raise ValueError(f"manifest state_before does not exactly match trace at step {step}")
        cameras = {}
        for camera in ("front", "wrist"):
            record = row.get(camera)
            if not isinstance(record, dict):
                raise ValueError(f"manifest {camera} record missing at step {step}")
            expected_name = f"policy_images/rollout_001/step_{step:06d}_{camera}.png"
            if record.get("path") != expected_name:
                raise ValueError(
                    f"manifest {camera} path at step {step} must be {expected_name}"
                )
            path = safe_manifest_path(rollout_dir, record["path"])
            file_sha256 = sha256_file(path)
            if record.get("sha256") != file_sha256:
                raise ValueError(f"PNG file sha256 mismatch at step {step} camera {camera}")
            pixels = load_rgb_png(path)
            pixel_sha256 = sha256_bytes(pixels.tobytes(order="C"))
            if record.get("pixel_sha256") != pixel_sha256:
                raise ValueError(f"RGB pixel sha256 mismatch at step {step} camera {camera}")
            shape = record.get("shape")
            if shape != [224, 224, 3]:
                raise ValueError(f"manifest shape mismatch at step {step} camera {camera}")
            if record.get("dtype") != "uint8":
                raise ValueError(f"manifest dtype mismatch at step {step} camera {camera}")
            cameras[camera] = {
                "path": str(path),
                "relative_path": record["path"],
                "file_sha256": file_sha256,
                "pixel_sha256": pixel_sha256,
                "pixels": pixels,
            }
        validated[step] = {"state_before": state, **cameras}
    return validated


def first_close_index(targets: np.ndarray) -> int:
    targets = np.asarray(targets)
    indices = np.flatnonzero(targets[1:, 5] < 0.5)
    if not len(indices):
        raise ValueError("raw demo has no close command after target[0]")
    return int(indices[0] + 1)


def image_difference(candidate: np.ndarray, raw: np.ndarray) -> dict:
    if raw.shape != (224, 224, 3) or raw.dtype != np.uint8:
        raise ValueError(f"raw image must be uint8 (224,224,3), got {raw.shape}/{raw.dtype}")
    difference = np.abs(candidate.astype(np.int16) - raw.astype(np.int16))
    return {
        "mae_uint8": float(difference.mean()),
        "mae_normalized": float(difference.mean() / 255.0),
        "max_abs_uint8": int(difference.max()),
        "pixel_values_equal_share": float(np.mean(difference == 0)),
    }


def scene_sha(evaluation: dict) -> str:
    results = evaluation.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("evaluation must contain exactly one rollout result")
    value = results[0].get("initial_scene_state_sha256")
    if not isinstance(value, str):
        raise ValueError("evaluation is missing initial_scene_state_sha256")
    return value


def analyze(
    rollout_dir: Path,
    reference_dir: Path,
    raw_path: Path,
    demo_name: str,
) -> dict:
    rollout_dir = rollout_dir.expanduser().resolve(strict=True)
    reference_dir = reference_dir.expanduser().resolve(strict=True)
    raw_path = raw_path.expanduser().resolve(strict=True)
    candidate_evaluation_path = rollout_dir / "evaluation.json"
    reference_evaluation_path = reference_dir / "evaluation.json"
    candidate_trace_path = rollout_dir / "trace_001.json"
    reference_trace_path = reference_dir / "trace_001.json"
    manifest_path = rollout_dir / "policy_images_001.json"
    for path in (
        candidate_evaluation_path,
        reference_evaluation_path,
        candidate_trace_path,
        reference_trace_path,
        manifest_path,
    ):
        path.resolve(strict=True)

    candidate_evaluation = load_json(candidate_evaluation_path)
    reference_evaluation = load_json(reference_evaluation_path)
    candidate_trace = load_trace(candidate_trace_path)
    reference_trace = load_trace(reference_trace_path)
    trace_comparison = compare_trace(reference_trace, candidate_trace)
    config_comparison = compare_required_config(reference_evaluation, candidate_evaluation)
    reference_scene_sha = scene_sha(reference_evaluation)
    candidate_scene_sha = scene_sha(candidate_evaluation)
    scene_comparison = {
        "expected_sha256": EXPECTED_SCENE_SHA256,
        "reference_sha256": reference_scene_sha,
        "candidate_sha256": candidate_scene_sha,
        "reference_matches_expected": reference_scene_sha == EXPECTED_SCENE_SHA256,
        "candidate_matches_expected": candidate_scene_sha == EXPECTED_SCENE_SHA256,
        "candidate_matches_reference": candidate_scene_sha == reference_scene_sha,
    }
    strict_reproduction_pass = (
        trace_comparison["bit_identical"]
        and all(record["equal"] for record in config_comparison.values())
        and all(scene_comparison[key] for key in (
            "reference_matches_expected", "candidate_matches_expected", "candidate_matches_reference"
        ))
    )

    manifest = load_json(manifest_path)
    validated_images = validate_manifest(rollout_dir, manifest, candidate_trace)
    rows = []
    with h5py.File(raw_path, "r") as raw_hdf:
        if f"data/{demo_name}" not in raw_hdf:
            raise ValueError(f"raw demo not found: {demo_name}")
        demo = raw_hdf[f"data/{demo_name}"]
        joint_pos = np.asarray(demo["obs/joint_pos"])
        targets = np.asarray(demo["obs/joint_pos_target"])
        raw_images = {camera: demo[f"obs/{camera}"] for camera in ("front", "wrist")}
        close_index = first_close_index(targets)
        if close_index != 278:
            raise ValueError(f"expected raw close index 278, got {close_index}")
        for step in EXPECTED_DUMP_STEPS:
            raw_frame = completed_step_to_raw_frame(step)
            if raw_frame >= len(joint_pos):
                raise ValueError(f"raw frame {raw_frame} is outside demo length {len(joint_pos)}")
            state_difference = np.abs(validated_images[step]["state_before"] - joint_pos[raw_frame])
            within_state_threshold = bool(np.all(state_difference <= 0.2))
            row = {
                "step": step,
                "raw_frame_s": raw_frame,
                "state_max_abs_rad": float(state_difference.max()),
                "state_abs_by_joint_rad": state_difference.tolist(),
                "within_state_threshold": within_state_threshold,
                "in_close_band": close_index - 30 <= raw_frame <= close_index - 1,
                "images": {},
            }
            for camera in ("front", "wrist"):
                row["images"][camera] = image_difference(
                    validated_images[step][camera]["pixels"], np.asarray(raw_images[camera][raw_frame])
                )
            rows.append(row)

    close_rows = [row for row in rows if row["in_close_band"]]
    retained_close = [row for row in close_rows if row["within_state_threshold"]]
    state_maxima = np.asarray([row["state_max_abs_rad"] for row in rows])
    image_summary = {}
    for camera in ("front", "wrist"):
        maes = np.asarray([row["images"][camera]["mae_uint8"] for row in rows])
        image_summary[camera] = {
            "steps": len(rows),
            "mean_mae_uint8": float(maes.mean()),
            "median_mae_uint8": float(np.median(maes)),
            "max_mae_uint8": float(maes.max()),
            "mean_mae_normalized": float(maes.mean() / 255.0),
        }
    return {
        "schema_version": 1,
        "date": str(date.today()),
        "scope": "CPU validation of a policy-image dump and step-index raw close-window alignment",
        "inputs": {
            "rollout_dir": str(rollout_dir),
            "reference_dir": str(reference_dir),
            "raw_path": str(raw_path),
            "raw_sha256": sha256_file(raw_path),
            "demo": demo_name,
            "candidate_evaluation_sha256": sha256_file(candidate_evaluation_path),
            "reference_evaluation_sha256": sha256_file(reference_evaluation_path),
            "candidate_trace_sha256": sha256_file(candidate_trace_path),
            "reference_trace_sha256": sha256_file(reference_trace_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "strict_reproduction_gate": {
            "pass": strict_reproduction_pass,
            "scene": scene_comparison,
            "config": config_comparison,
            "trace": trace_comparison,
        },
        "manifest_validation": {
            "format": "top-level list",
            "steps": len(validated_images),
            "first_step": min(validated_images),
            "last_step": max(validated_images),
            "state_before_exact_trace_match": True,
            "observation_index_equals_step_minus_1": True,
            "png_files": 2 * len(validated_images),
            "all_file_and_pixel_sha256_match": True,
            "image_contract": "RGB uint8 [224,224,3]",
        },
        "alignment": {
            "formula": "raw/LeRobot frame s = trace completed step - 1",
            "raw_close_index_c": close_index,
            "close_band_s": [close_index - 30, close_index - 1],
            "close_band_completed_steps": [close_index - 29, close_index],
            "state_exclusion_threshold_rad_per_joint": 0.2,
            "dump_steps": [EXPECTED_DUMP_STEPS[0], EXPECTED_DUMP_STEPS[-1]],
            "counterfactual_note": (
                "n_action_steps=30 means most per-frame dump observations were not policy query boundaries"
            ),
        },
        "state_distance": {
            "steps": len(rows),
            "within_threshold": sum(row["within_state_threshold"] for row in rows),
            "excluded": sum(not row["within_state_threshold"] for row in rows),
            "max_abs_rad": float(state_maxima.max()),
            "median_step_max_abs_rad": float(np.median(state_maxima)),
            "p95_step_max_abs_rad": float(np.quantile(state_maxima, 0.95)),
        },
        "close_band_retention": {
            "total": len(close_rows),
            "retained": len(retained_close),
            "excluded": len(close_rows) - len(retained_close),
            "minimum_required": 15,
            "status": "ready_for_closed_loop_probe" if len(retained_close) >= 15 else "unresolved",
            "retained_steps": [row["step"] for row in retained_close],
            "excluded_steps": [row["step"] for row in close_rows if not row["within_state_threshold"]],
        },
        "image_difference_from_raw": image_summary,
        "per_step": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--demo", default="demo_9")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {output}")
    report = analyze(args.rollout_dir, args.reference_dir, args.raw_path, args.demo)
    write_json_exclusive(output, report)
    print(json.dumps({
        "strict_reproduction_pass": report["strict_reproduction_gate"]["pass"],
        "close_band_retained": report["close_band_retention"]["retained"],
        "close_band_status": report["close_band_retention"]["status"],
        "output": str(output),
    }, ensure_ascii=False))
    if not report["strict_reproduction_gate"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
