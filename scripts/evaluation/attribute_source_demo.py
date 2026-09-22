"""Attribute all saved Mimic episodes to leader demos and decompose audit gates.

This is an offline, CPU-only audit. It reads the original HDF5 files and existing
quality diagnostics, and writes new JSON artifacts without modifying either input.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.imitation_learning.action_contract import atomic_json, sha256_file


DEFAULT_SOURCE = REPO / "datasets/pick_cube_into_box_annotated_wrist_10_20260911.hdf5"
DEFAULT_RAW = REPO / "outputs/mimic_vision224_500_20260913/raw"
DEFAULT_DIAGNOSTICS = (
    REPO / "outputs/evaluation/mimic_quality_verified_20260919/episode_diagnostics.json"
)
DEFAULT_CONTRACT = (
    REPO / "outputs/mimic_target_unclipped_76_20260919/lerobot_all/action_contract.json"
)
DEFAULT_OUTPUT = REPO / "outputs/source_attribution_20260921"
DEFAULT_EVIDENCE = REPO / "docs/evidence/source_gate_decomposition_20260921.json"
DEFAULT_FIXED_RAW = REPO / "outputs/mimic_reset_fixed_76_20260921/raw"
TOLERANCES = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)
JOINT_EPSILON = 1e-12


def numeric_demo_key(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def first_close_index(targets: np.ndarray, threshold: float = 0.5) -> int:
    """Return the raw target index of the first close command, ignoring target[0]."""
    targets = np.asarray(targets)
    if targets.ndim != 2 or targets.shape[1] != 6 or len(targets) < 2:
        raise ValueError("joint targets must have shape (T, 6) with T >= 2")
    indices = np.flatnonzero(targets[1:, 5] < threshold)
    if not len(indices):
        raise ValueError("episode has no close command after stale target[0]")
    return int(indices[0] + 1)


def signal_margins(signatures: dict[int, int]) -> dict:
    """Report exact-match attribution margins to the nearest competing source."""
    if len(set(signatures.values())) != len(signatures):
        raise ValueError("attribution signal is not injective")
    rows = []
    for source, value in sorted(signatures.items()):
        competing = [(abs(value - other_value), other) for other, other_value in signatures.items()
                     if other != source]
        separation, nearest = min(competing)
        rows.append({"source": f"src:demo_{source}", "signature": value,
                     "nearest_competing_source": f"src:demo_{nearest}",
                     "nearest_competing_distance": separation,
                     "exact_match_error": 0, "margin": separation})
    return {"definition": "second_best_absolute_error - best_absolute_error",
            "minimum_margin": min(row["margin"] for row in rows), "by_source": rows}


def require_exact_episode_set(computed: set[tuple[str, str]], contracted: set[tuple[str, str]]) -> None:
    missing = sorted(contracted - computed)
    unexpected = sorted(computed - contracted)
    if missing or unexpected:
        raise AssertionError(
            f"strict candidate set differs from contract: missing={missing}, unexpected={unexpected}"
        )


def decision_gate(signal_reports: dict[str, dict], clipping_pools: list[dict]) -> dict:
    margin_ok = min(report["minimum_margin"] for report in signal_reports.values()) > 0
    close_ok = all(pool["close_clipped_frame_share"] <= 0.02 for pool in clipping_pools)
    return {
        "minimum_attribution_margin_sufficient": margin_ok,
        "all_close_clipping_shares_lte_0_02": close_ok,
        "proceed_to_2_2_when_other_prerequisites_pass": margin_ok and close_ok,
    }


def stat_snapshot(paths: Iterable[Path]) -> list[dict]:
    rows = []
    for path in sorted((path.resolve(strict=True) for path in paths), key=str):
        stat = path.stat()
        rows.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return rows


def quat_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError("quaternion must have shape (4,)")
    return quaternion * np.array([1.0, -1.0, -1.0, -1.0])


def quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    return np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])


def quat_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector_quaternion = np.concatenate(([0.0], np.asarray(vector, dtype=np.float64)))
    return quat_multiply_wxyz(
        quat_multiply_wxyz(quaternion, vector_quaternion), quat_conjugate_wxyz(quaternion)
    )[1:]


def compose_pose_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compose T_world_left and T_left_right poses encoded as xyz+wxyz."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != (7,) or right.shape != (7,):
        raise ValueError("poses must have shape (7,)")
    position = left[:3] + quat_rotate_wxyz(left[3:], right[:3])
    quaternion = quat_multiply_wxyz(left[3:], right[3:])
    quaternion /= np.linalg.norm(quaternion)
    return np.concatenate((position, quaternion))


def invert_pose_wxyz(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError("pose must have shape (7,)")
    inverse_quaternion = quat_conjugate_wxyz(pose[3:])
    inverse_quaternion /= np.linalg.norm(inverse_quaternion)
    return np.concatenate((quat_rotate_wxyz(inverse_quaternion, -pose[:3]), inverse_quaternion))


def object_frame_eef_pose(
    world_robot_pose: np.ndarray, robot_eef_pose: np.ndarray, world_object_pose: np.ndarray
) -> np.ndarray:
    """Compute T_object_eef = inv(T_world_object) * T_world_robot * T_robot_eef."""
    world_eef = compose_pose_wxyz(world_robot_pose, robot_eef_pose)
    return compose_pose_wxyz(invert_pose_wxyz(world_object_pose), world_eef)


def load_source_signatures(source_path: Path) -> dict[int, dict]:
    signatures = {}
    with h5py.File(source_path, "r") as source:
        for name in sorted(source["data"], key=numeric_demo_key):
            index = numeric_demo_key(name)
            targets = np.asarray(source[f"data/{name}/obs/joint_pos_target"])
            signatures[index] = {
                "source": f"src:demo_{index}",
                "raw_length": len(targets),
                "close_index": first_close_index(targets),
                "expected_generated_length": len(targets) + 16,
                "expected_generated_close_index": (
                    1 if index == 6 else first_close_index(targets) + 6
                ),
                "close_offset": 0 if index == 6 else 6,
                "degenerate": index == 6,
            }
    if set(signatures) != set(range(10)):
        raise ValueError(f"expected source demos 0..9, got {sorted(signatures)}")
    return signatures


def clipping_correction(targets: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    actions = np.asarray(targets, dtype=np.float64)[1:]
    return np.maximum(np.maximum(lower - actions, actions - upper), 0.0)


def diagnostics_index(path: Path) -> tuple[dict[tuple[str, str], dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("diagnostics must contain an episodes list")
    indexed = {(episode["shard"], episode["name"]): episode for episode in episodes}
    if len(indexed) != len(episodes):
        raise ValueError("duplicate shard/name in diagnostics")
    return indexed, payload


def pool_summary(rows: Iterable[dict], tolerance: float) -> dict:
    selected = [row for row in rows if row["quality_pass"] and row["max_clip_correction_rad"] <= tolerance]
    return {
        "tolerance_rad": tolerance,
        "episodes": len(selected),
        "raw_frames": sum(row["raw_frames"] for row in selected),
        "lerobot_frames": sum(row["lerobot_frames"] for row in selected),
        "source_families": len({row["source"] for row in selected}),
        "sources": sorted({row["source"] for row in selected}, key=lambda x: int(x.rsplit("_", 1)[1])),
    }


def clipping_pool_summary(name: str, rows: list[dict]) -> dict:
    clipped_frames = sum(row["clipped_lerobot_frames"] for row in rows)
    total_frames = sum(row["lerobot_frames"] for row in rows)
    close_clipped = sum(row["close_clipped_frames"] for row in rows)
    close_total = 30 * len(rows)
    return {
        "pool": name,
        "episodes": len(rows),
        "clipped_episodes": sum(row["clipped_lerobot_frames"] > 0 for row in rows),
        "clipped_episode_share": sum(row["clipped_lerobot_frames"] > 0 for row in rows) / len(rows),
        "clipped_frames": clipped_frames,
        "lerobot_frames": total_frames,
        "clipped_frame_share": clipped_frames / total_frames,
        "close_clipped_frames": close_clipped,
        "close_frames": close_total,
        "close_clipped_frame_share": close_clipped / close_total,
    }


def analyze(source_path: Path, raw_dir: Path, diagnostics_path: Path, contract_path: Path) -> tuple[dict, list[dict]]:
    for path in (source_path, raw_dir, diagnostics_path, contract_path):
        path.resolve(strict=True)
    signatures = load_source_signatures(source_path)
    by_length = {record["expected_generated_length"]: source for source, record in signatures.items()}
    by_close = {record["expected_generated_close_index"]: source for source, record in signatures.items()}
    if len(by_length) != 10 or len(by_close) != 10:
        raise ValueError("source attribution signatures must be injective")

    diagnostics, diagnostics_payload = diagnostics_index(diagnostics_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    limits = contract["joint_limits"]
    lower = np.asarray(limits["joint_lower_limits_rad"], dtype=np.float64)
    upper = np.asarray(limits["joint_upper_limits_rad"], dtype=np.float64)
    joint_names = limits["joint_names"]
    if lower.shape != (6,) or upper.shape != (6,) or len(joint_names) != 6:
        raise ValueError("contract must contain six joint limits")

    rows = []
    seen = set()
    raw_shards = sorted(raw_dir.glob("shard_*.hdf5"))
    if len(raw_shards) != 20:
        raise ValueError(f"expected 20 raw shards, got {len(raw_shards)}")
    raw_shard_integrity = []
    raw_shard_hashes = {}
    for shard_path in raw_shards:
        actual_sha256 = sha256_file(shard_path)
        diagnostic_hashes = {
            episode.get("raw_sha256")
            for (shard_name, _), episode in diagnostics.items()
            if shard_name == shard_path.name
        }
        if diagnostic_hashes != {actual_sha256}:
            raise AssertionError(
                f"diagnostics raw_sha256 mismatch for {shard_path.name}: "
                f"recorded={sorted(str(value) for value in diagnostic_hashes)}, actual={actual_sha256}"
            )
        stat = shard_path.stat()
        raw_shard_hashes[shard_path.name] = actual_sha256
        raw_shard_integrity.append({
            "filename": shard_path.name,
            "sha256": actual_sha256,
            "diagnostic_episode_records": sum(
                shard_name == shard_path.name for shard_name, _ in diagnostics
            ),
            "diagnostics_sha256_match": True,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    for shard_path in raw_shards:
        with h5py.File(shard_path, "r") as shard:
            for demo_name in sorted(shard["data"], key=numeric_demo_key):
                key = (shard_path.name, demo_name)
                if key not in diagnostics:
                    raise ValueError(f"missing diagnostic record for {key}")
                diagnostic = diagnostics[key]
                demo = shard[f"data/{demo_name}"]
                targets = np.asarray(demo["obs/joint_pos_target"])
                raw_length = len(targets)
                close_index = first_close_index(targets)
                length_source = by_length.get(raw_length)
                close_source = by_close.get(close_index)
                if length_source is None or close_source is None or length_source != close_source:
                    raise ValueError(
                        f"attribution disagreement for {key}: length={raw_length}->{length_source}, "
                        f"close={close_index}->{close_source}"
                    )
                source = length_source
                correction = clipping_correction(targets, lower, upper)
                clipped = correction > JOINT_EPSILON
                max_by_joint = correction.max(axis=0)
                max_correction = float(max_by_joint.max())
                dominant_joint = joint_names[int(max_by_joint.argmax())] if max_correction > 0 else None
                close_start = close_index - 29
                close_stop = close_index + 1
                if close_index == 1 and source == 6:
                    close_clipped = None
                elif close_start < 1 or close_stop > raw_length:
                    raise ValueError(f"invalid close window for {key}: c={close_index}")
                else:
                    close_clipped = int(
                        np.any(clipped[close_start - 1: close_stop - 1], axis=1).sum()
                    )
                jump_pass = diagnostic.get("jumps", {}).get("jump_count", 1) == 0
                release_pass = diagnostic.get("recorded_release", {}).get("final_stable", False) is True
                quality_pass = jump_pass and release_pass
                accepted = diagnostic.get("accepted") is True
                if accepted != jump_pass:
                    raise ValueError(f"accepted/jump mismatch for {key}")
                recorded_frames = diagnostic.get("frame_count")
                if recorded_frames is not None and recorded_frames != raw_length - 1:
                    raise ValueError(f"frame_count mismatch for {key}")
                row = {
                    "shard": shard_path.name,
                    "demo": demo_name,
                    "source": f"src:demo_{source}",
                    "raw_frames": raw_length,
                    "lerobot_frames": raw_length - 1,
                    "diagnostic_frame_count": recorded_frames,
                    "diagnostic_frame_count_missing": recorded_frames is None,
                    "diagnostic_accepted": accepted,
                    "close_index": close_index,
                    "length_signal_match": True,
                    "close_signal_match": True,
                    "jump_pass": jump_pass,
                    "release_pass": release_pass,
                    "quality_pass": quality_pass,
                    "unclipped": max_correction <= JOINT_EPSILON,
                    "strict_final_pass": quality_pass and max_correction <= JOINT_EPSILON,
                    "max_clip_correction_rad": max_correction,
                    "max_clip_correction_by_joint_rad": dict(zip(joint_names, max_by_joint.tolist())),
                    "dominant_clip_joint": dominant_joint,
                    "clipped_lerobot_frames": int(np.any(clipped, axis=1).sum()),
                    "close_clipped_frames": close_clipped,
                }
                rows.append(row)
                seen.add(key)
    if len(rows) != 500 or seen != set(diagnostics):
        raise ValueError(f"expected identical 500-episode inputs, raw={len(rows)} diagnostics={len(diagnostics)}")

    by_source_rows = defaultdict(list)
    for row in rows:
        by_source_rows[row["source"]].append(row)
    source_gate_rows = []
    for source, family_rows in sorted(by_source_rows.items(), key=lambda item: numeric_demo_key(item[0])):
        clipped_quality = [row for row in family_rows if row["quality_pass"] and not row["unclipped"]]
        counts = {
            "jump_pass": sum(row["jump_pass"] for row in family_rows),
            "release_pass": sum(row["release_pass"] for row in family_rows),
            "quality_pass": sum(row["quality_pass"] for row in family_rows),
            "unclipped_all_generated": sum(row["unclipped"] for row in family_rows),
            "strict_final_pass": sum(row["strict_final_pass"] for row in family_rows),
        }
        source_gate_rows.append({
            "source": source,
            "generated": len(family_rows),
            **counts,
            "survival_rates_from_generated": {
                field: count / len(family_rows) for field, count in counts.items()
            },
            "median_raw_frames": int(np.median([row["raw_frames"] for row in family_rows])),
            "quality_clipped_median_max_correction_rad": (
                float(np.median([row["max_clip_correction_rad"] for row in clipped_quality]))
                if clipped_quality else None
            ),
        })

    totals = {field: sum(row[field] for row in source_gate_rows) for field in (
        "generated", "jump_pass", "release_pass", "quality_pass",
        "unclipped_all_generated", "strict_final_pass",
    )}
    expected = {"generated": 500, "jump_pass": 403, "quality_pass": 294, "strict_final_pass": 76}
    for field, value in expected.items():
        if totals[field] != value:
            raise AssertionError(f"{field}: expected {value}, got {totals[field]}")

    computed_strict_set = {
        (row["shard"], row["demo"]) for row in rows if row["strict_final_pass"]
    }
    contract_rows = contract.get("episodes")
    if not isinstance(contract_rows, list) or len(contract_rows) != 76:
        raise AssertionError("action contract must contain exactly 76 episodes")
    contracted_strict_set = {
        (Path(row["raw_path"]).name, row["raw_demo"]) for row in contract_rows
    }
    if len(contracted_strict_set) != len(contract_rows):
        raise AssertionError("action contract contains duplicate raw episode identities")
    require_exact_episode_set(computed_strict_set, contracted_strict_set)
    row_index = {(row["shard"], row["demo"]): row for row in rows}
    for contract_row in contract_rows:
        identity = (Path(contract_row["raw_path"]).name, contract_row["raw_demo"])
        computed_row = row_index[identity]
        if contract_row["frame_count"] != computed_row["lerobot_frames"]:
            raise AssertionError(f"contract frame_count mismatch for {identity}")
        if contract_row["raw_sha256"] != raw_shard_hashes[identity[0]]:
            raise AssertionError(f"contract raw_sha256 mismatch for {identity}")

    all_clipped = [row for row in rows if not row["unclipped"]]
    quality_clipped = [row for row in rows if row["quality_pass"] and not row["unclipped"]]
    dominant_counts = Counter(row["dominant_clip_joint"] for row in all_clipped)
    involved_all = {joint: sum(row["max_clip_correction_by_joint_rad"][joint] > JOINT_EPSILON
                               for row in all_clipped) for joint in joint_names}
    involved_quality = {joint: sum(row["max_clip_correction_by_joint_rad"][joint] > JOINT_EPSILON
                                   for row in quality_clipped) for joint in joint_names}
    if len(all_clipped) != 366 or dominant_counts != Counter({"wrist_flex": 315, "elbow_flex": 51}):
        raise AssertionError(f"unexpected all-clipped decomposition: {len(all_clipped)}, {dominant_counts}")
    if involved_all["wrist_flex"] != 366 or involved_all["elbow_flex"] != 115 \
            or involved_all["shoulder_lift"] != 16:
        raise AssertionError(f"unexpected all-clipped joint involvement: {involved_all}")
    if len(quality_clipped) != 218 or involved_quality["wrist_flex"] != 218 \
            or involved_quality["elbow_flex"] != 16:
        raise AssertionError(f"unexpected quality-clipped joint involvement: {involved_quality}")

    tolerance_pools = [pool_summary(rows, tolerance) for tolerance in TOLERANCES]
    expected_pools = {
        0.0: (76, 34_601, 34_525, 6), 0.02: (105, 52_058, 51_953, 7),
        0.05: (188, 119_868, 119_680, 9), 0.10: (208, 130_327, 130_119, 9),
        0.20: (229, 142_101, 141_872, 9), 0.40: (285, 175_215, 174_930, 9),
    }
    for pool in tolerance_pools:
        actual = (pool["episodes"], pool["raw_frames"], pool["lerobot_frames"], pool["source_families"])
        if actual != expected_pools[pool["tolerance_rad"]]:
            raise AssertionError(f"unexpected tolerance pool {pool['tolerance_rad']}: {actual}")

    pool_005 = [row for row in rows if row["quality_pass"] and row["max_clip_correction_rad"] <= 0.05]
    family_005 = []
    for source, family_rows in sorted(
        ((source, [row for row in pool_005 if row["source"] == source])
         for source in {row["source"] for row in pool_005}),
        key=lambda item: numeric_demo_key(item[0]),
    ):
        lengths = sorted({row["lerobot_frames"] for row in family_rows})
        if len(lengths) != 1:
            raise AssertionError(f"0.05 pool family length is not constant: {source} -> {lengths}")
        family_005.append({"source": source, "episodes": len(family_rows),
                           "lerobot_frames": sum(row["lerobot_frames"] for row in family_rows),
                           "unique_episode_lengths": lengths,
                           "lerobot_frames_per_episode": lengths[0]})

    current_sources = {"src:demo_1", "src:demo_4", "src:demo_5", "src:demo_7", "src:demo_8", "src:demo_9"}
    current_rows = [row for row in pool_005 if row["source"] in current_sources]
    added_rows = [row for row in pool_005 if row["source"] not in current_sources]
    clipping_pools = [
        clipping_pool_summary("current_6_families", current_rows),
        clipping_pool_summary("added_3_families", added_rows),
        clipping_pool_summary("all_9_families", pool_005),
    ]
    missing = [row for row in rows if row["diagnostic_frame_count_missing"]]
    if len(missing) != 97 or any(row["jump_pass"] or row["quality_pass"] for row in missing):
        raise AssertionError("the 97 missing frame_count records must all be rejected outside candidate pools")
    degenerate = by_source_rows["src:demo_6"]
    if len(degenerate) != 37 or {row["close_index"] for row in degenerate} != {1} \
            or any(row["release_pass"] for row in degenerate):
        raise AssertionError("src:demo_6 degenerate exception changed")

    length_signatures = {source: record["expected_generated_length"] for source, record in signatures.items()}
    close_signatures = {source: record["expected_generated_close_index"] for source, record in signatures.items()}
    signal_reports = {
        "generated_raw_length": signal_margins(length_signatures),
        "first_close_raw_index": signal_margins(close_signatures),
    }
    summary = {
        "schema_version": 1,
        "date": str(date.today()),
        "scope": "CPU-only attribution and gate decomposition of all 500 existing raw Mimic episodes",
        "inputs": {
            "source_hdf5": str(source_path.resolve()),
            "source_sha256": sha256_file(source_path),
            "raw_dir": str(raw_dir.resolve()),
            "raw_shards": len(raw_shards),
            "raw_shard_integrity": raw_shard_integrity,
            "diagnostics": str(diagnostics_path.resolve()),
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "diagnostics_schema_version": diagnostics_payload.get("schema_version"),
            "contract": str(contract_path.resolve()),
            "contract_sha256": sha256_file(contract_path),
        },
        "attribution": {
            "episodes": len(rows),
            "source_families": len(by_source_rows),
            "length_close_agreement": len(rows),
            "source_signatures": [signatures[index] for index in sorted(signatures)],
            "signals": signal_reports,
            "object_frame_eef_check": {
                "role": "implementation check, not an independent attribution signal",
                "formula": "inv(T_world_object) * T_world_robot * T_robot_eef",
                "quaternion_inverse": "normalized conjugate (wxyz)",
            },
        },
        "gate_decomposition": {"by_source": source_gate_rows, "totals": totals,
                               "asserted_totals": expected,
                               "strict_candidate_contract": {
                                   "computed_episodes": len(computed_strict_set),
                                   "contract_episodes": len(contracted_strict_set),
                                   "identity": "(raw basename, demo name)",
                                   "exact_set_equality": True,
                                   "frame_count_and_raw_sha256_match": True,
                               },
                               "additional_clipping_exclusions_after_quality": len(quality_clipped),
                               "quality_clipped_exclusions_in_added_3_families": sum(
                                   row["source"] in {"src:demo_0", "src:demo_2", "src:demo_3"}
                                   for row in quality_clipped
                               )},
        "clipping_joint_counts": {
            "all_clipped_episodes": len(all_clipped),
            "definition_1_episode_dominant_joint_partition": dict(sorted(dominant_counts.items())),
            "definition_2_any_joint_involvement_all_clipped": involved_all,
            "quality_pass_clipped_episodes": len(quality_clipped),
            "definition_3_any_joint_involvement_quality_clipped": involved_quality,
            "gate_narrative_count": len(quality_clipped),
        },
        "tolerance_pools": tolerance_pools,
        "tolerance_0_05_by_source": family_005,
        "raw_target_clipping_by_0_05_pool": clipping_pools,
        "missing_frame_count": {
            "records": len(missing),
            "all_accepted_false": all(not row["diagnostic_accepted"] for row in missing),
            "all_quality_pass_false": all(not row["quality_pass"] for row in missing),
            "handling": "raw length recovered from HDF5 for attribution; records excluded from candidate pools",
        },
        "degenerate_source": {
            "source": "src:demo_6", "generated_episodes": len(degenerate),
            "source_close_index": signatures[6]["close_index"],
            "generated_close_indices": sorted({row["close_index"] for row in degenerate}),
            "expected_plus_6_close_index_observed_count": sum(row["close_index"] == 7 for row in rows),
            "close_offset": 0, "release_pass": sum(row["release_pass"] for row in degenerate),
            "quality_pass": sum(row["quality_pass"] for row in degenerate),
            "classification": "degenerate; excluded from close-signal training decisions",
        },
        "decision_gate": decision_gate(signal_reports, clipping_pools),
    }
    return summary, rows


def run(
    source_path: Path,
    raw_dir: Path,
    diagnostics_path: Path,
    contract_path: Path,
    output_dir: Path,
    evidence_path: Path | None = None,
    fixed_raw_dir: Path = DEFAULT_FIXED_RAW,
) -> dict:
    preservation_files = [
        *sorted(raw_dir.expanduser().resolve(strict=True).glob("*")),
        *sorted(fixed_raw_dir.expanduser().resolve(strict=True).glob("*")),
    ]
    preservation_files = [path for path in preservation_files if path.is_file()]
    before = stat_snapshot(preservation_files)
    summary, rows = analyze(source_path, raw_dir, diagnostics_path, contract_path)
    after = stat_snapshot(preservation_files)
    if before != after:
        raise AssertionError("raw input size or mtime changed during the CPU audit")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(output_dir / "episodes.json", {"schema_version": 1, "episodes": rows})
    preservation = {
        "schema_version": 1,
        "original_raw_dir": str(raw_dir.expanduser().resolve()),
        "fixed_76_raw_dir": str(fixed_raw_dir.expanduser().resolve()),
        "before": before,
        "after": after,
        "size_and_mtime_unchanged": True,
    }
    atomic_json(output_dir / "input_preservation.json", preservation)
    summary["artifacts"] = {
        "episodes": str(output_dir / "episodes.json"),
        "episodes_sha256": sha256_file(output_dir / "episodes.json"),
        "input_preservation": str(output_dir / "input_preservation.json"),
        "input_preservation_sha256": sha256_file(output_dir / "input_preservation.json"),
        "summary": str(output_dir / "summary.json"),
    }
    atomic_json(output_dir / "summary.json", summary)
    if evidence_path is not None:
        evidence_path = evidence_path.expanduser().resolve()
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(evidence_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-hdf5", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--fixed-raw-dir", type=Path, default=DEFAULT_FIXED_RAW)
    args = parser.parse_args()
    summary = run(args.source_hdf5, args.raw_dir, args.diagnostics, args.contract,
                  args.output_dir, args.evidence, args.fixed_raw_dir)
    print(json.dumps({
        "status": "complete",
        "episodes": summary["attribution"]["episodes"],
        "gate_totals": summary["gate_decomposition"]["totals"],
        "close_clipping": {
            row["pool"]: row["close_clipped_frame_share"]
            for row in summary["raw_target_clipping_by_0_05_pool"]
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
