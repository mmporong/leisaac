"""Create and validate the SO101 action-label contract across IL pipeline stages."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
CONTRACT_FILENAME = "action_contract.json"
RECORDED_TARGET_ALIGNMENT = "action[t] = obs/joint_pos_target[t+1]"
LEGACY_ALIGNMENTS = {
    "next_observed": "action[t] = obs/joint_pos[min(t+1, T-1)]",
    "stored": "action[t] = raw actions[t]",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def alignment_for(action_source: str) -> str:
    if action_source == "recorded_target":
        return RECORDED_TARGET_ALIGNMENT
    try:
        return LEGACY_ALIGNMENTS[action_source]
    except KeyError as error:
        raise ValueError(f"unsupported action source: {action_source}") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_limits_record(limits: dict[str, Any]) -> None:
    limit_path = Path(limits.get("path", ""))
    _require(limit_path.is_file(), "joint-limits source file is missing")
    _require(limits.get("sha256") == sha256_file(limit_path), "joint-limits file hash changed")
    payload = json.loads(limit_path.read_text(encoding="utf-8"))
    names = limits.get("joint_names")
    lower = limits.get("joint_lower_limits_rad")
    upper = limits.get("joint_upper_limits_rad")
    _require(isinstance(names, list) and len(names) == 6, "invalid joint names")
    _require(isinstance(lower, list) and len(lower) == 6, "invalid lower joint limits")
    _require(isinstance(upper, list) and len(upper) == 6, "invalid upper joint limits")
    _require(all(isinstance(value, (int, float)) and math.isfinite(value) for value in lower + upper),
             "joint limits must be finite numbers")
    _require(all(low < high for low, high in zip(lower, upper, strict=True)),
             "every lower joint limit must be below its upper limit")
    for key in ("joint_names", "joint_lower_limits_rad", "joint_upper_limits_rad"):
        _require(payload.get(key) == limits.get(key), f"embedded joint limits differ from live file: {key}")


def validate_conversion_provenance(
    provenance_path: Path,
    raw_path: Path,
    action_source: str,
) -> dict[str, Any]:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    _require(provenance.get("action_source") == action_source, "conversion action source mismatch")
    _require(provenance.get("input_sha256") == sha256_file(raw_path), "raw file hash changed")
    episodes = provenance.get("episodes")
    _require(isinstance(episodes, list) and episodes, "conversion provenance has no episodes")
    _require(provenance.get("episode_count") == len(episodes), "conversion episode count mismatch")
    frame_count = sum(int(episode["frame_count"]) for episode in episodes)
    _require(provenance.get("frame_count") == frame_count, "conversion frame count mismatch")
    if action_source == "recorded_target":
        _require("target_alignment" in provenance, "recorded_target conversion has no target alignment")
    _require(provenance.get("target_alignment", alignment_for(action_source)) == alignment_for(action_source),
             "conversion action alignment mismatch")
    if action_source == "recorded_target":
        limits = provenance.get("joint_limits")
        _require(isinstance(limits, dict), "recorded_target conversion has no joint limits")
        _validate_limits_record(limits)
    return provenance


def build_aggregate_contract(
    aggregate_root: Path,
    action_source: str,
    shards: list[dict[str, Any]],
    aggregate_episode_count: int,
    aggregate_frame_count: int,
) -> dict[str, Any]:
    sources = []
    episode_map = []
    next_episode = 0
    canonical_limits = None
    for shard in shards:
        raw_path = Path(shard["raw_path"]).resolve(strict=True)
        provenance_path = Path(shard["conversion_provenance_path"]).resolve(strict=True)
        provenance = validate_conversion_provenance(provenance_path, raw_path, action_source)
        limits = provenance.get("joint_limits")
        if canonical_limits is None:
            canonical_limits = limits
        elif limits != canonical_limits:
            raise ValueError("joint-limit provenance differs between converted shards")
        source_record = {
            "raw_path": str(raw_path),
            "raw_sha256": provenance["input_sha256"],
            "conversion_provenance_path": str(provenance_path),
            "conversion_provenance_sha256": sha256_file(provenance_path),
            "episode_count": provenance["episode_count"],
            "frame_count": provenance["frame_count"],
        }
        sources.append(source_record)
        for episode in provenance["episodes"]:
            episode_map.append({
                "aggregate_episode_index": next_episode,
                "raw_path": str(raw_path),
                "raw_sha256": provenance["input_sha256"],
                "raw_demo": episode["name"],
                "frame_count": int(episode["frame_count"]),
            })
            next_episode += 1
    _require(next_episode == aggregate_episode_count, "aggregate episode count does not match conversions")
    _require(sum(item["frame_count"] for item in episode_map) == aggregate_frame_count,
             "aggregate frame count does not match conversions")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "stage": "aggregate",
        "dataset_root": str(aggregate_root.resolve()),
        "action_source": action_source,
        "alignment": alignment_for(action_source),
        "legacy_action_source": action_source != "recorded_target",
        "joint_limits": canonical_limits,
        "episode_count": aggregate_episode_count,
        "frame_count": aggregate_frame_count,
        "sources": sources,
        "episodes": episode_map,
    }
    if action_source == "recorded_target" and canonical_limits is None:
        raise ValueError("recorded_target aggregate contract requires joint limits")
    return contract


def validate_aggregate_contract(root: Path, *, allow_legacy: bool = False) -> dict[str, Any]:
    contract_path = root / CONTRACT_FILENAME
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(contract.get("schema_version") == SCHEMA_VERSION, "unsupported action contract schema")
    _require(contract.get("stage") == "aggregate", "expected aggregate action contract")
    action_source = contract.get("action_source")
    _require(action_source == "recorded_target" or allow_legacy, "legacy action source requires explicit opt-in")
    _require(contract.get("alignment") == alignment_for(action_source), "action alignment mismatch")
    _require(contract.get("legacy_action_source") is (action_source != "recorded_target"),
             "aggregate legacy action-source flag mismatch")
    _require(Path(contract["dataset_root"]).resolve() == root.resolve(), "action contract dataset root mismatch")
    episodes = contract.get("episodes", [])
    _require(len(episodes) == contract.get("episode_count"), "action contract episode count mismatch")
    _require(sum(int(item["frame_count"]) for item in episodes) == contract.get("frame_count"),
             "action contract frame count mismatch")
    _require([item["aggregate_episode_index"] for item in episodes] == list(range(len(episodes))),
             "aggregate episode mapping is not contiguous")
    sources = contract.get("sources", [])
    _require(isinstance(sources, list) and sources, "action contract has no raw sources")
    if action_source == "recorded_target":
        limits = contract.get("joint_limits")
        _require(isinstance(limits, dict), "recorded_target contract has no joint limits")
        _validate_limits_record(limits)
    rebuilt_episodes = []
    raw_demo_ids = set()
    for source in sources:
        raw_path = Path(source["raw_path"])
        provenance_path = Path(source["conversion_provenance_path"])
        _require(raw_path.is_file(), f"raw source is missing: {raw_path}")
        _require(sha256_file(raw_path) == source["raw_sha256"], f"raw source changed: {raw_path}")
        _require(provenance_path.is_file(), f"conversion provenance is missing: {provenance_path}")
        _require(sha256_file(provenance_path) == source["conversion_provenance_sha256"],
                 f"conversion provenance changed: {provenance_path}")
        provenance = validate_conversion_provenance(provenance_path, raw_path, action_source)
        _require(provenance.get("joint_limits") == contract.get("joint_limits"),
                 "source joint limits differ from aggregate action contract")
        _require(source.get("episode_count") == provenance["episode_count"],
                 "source episode count differs from conversion provenance")
        _require(source.get("frame_count") == provenance["frame_count"],
                 "source frame count differs from conversion provenance")
        for episode in provenance["episodes"]:
            identity = (provenance["input_sha256"], episode["name"])
            _require(identity not in raw_demo_ids, "duplicate raw-sha/demo identity in action contract")
            raw_demo_ids.add(identity)
            rebuilt_episodes.append({
                "aggregate_episode_index": len(rebuilt_episodes),
                "raw_path": str(raw_path.resolve()),
                "raw_sha256": provenance["input_sha256"],
                "raw_demo": episode["name"],
                "frame_count": int(episode["frame_count"]),
            })
    _require(rebuilt_episodes == episodes, "aggregate episode lineage differs from conversion provenance")
    return contract


def build_split_contract(
    source_root: Path,
    source_contract: dict[str, Any],
    split_indices: dict[str, list[int]],
) -> dict[str, Any]:
    episode_lookup = {item["aggregate_episode_index"]: item for item in source_contract["episodes"]}
    split_records = {}
    for name, indices in split_indices.items():
        mapped = [episode_lookup[index] for index in sorted(indices)]
        split_records[name] = {
            "source_episode_indices": sorted(indices),
            "episode_count": len(mapped),
            "frame_count": sum(item["frame_count"] for item in mapped),
            "episodes": mapped,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "split",
        "source_root": str(source_root.resolve()),
        "source_contract_sha256": sha256_file(source_root / CONTRACT_FILENAME),
        "action_source": source_contract["action_source"],
        "alignment": source_contract["alignment"],
        "legacy_action_source": source_contract["legacy_action_source"],
        "joint_limits": source_contract["joint_limits"],
        "splits": split_records,
    }


def validate_split_contract(root: Path, *, allow_legacy: bool = False) -> dict[str, Any]:
    contract = json.loads((root / CONTRACT_FILENAME).read_text(encoding="utf-8"))
    _require(contract.get("schema_version") == SCHEMA_VERSION, "unsupported action contract schema")
    _require(contract.get("stage") == "split", "expected split action contract")
    action_source = contract.get("action_source")
    _require(action_source == "recorded_target" or allow_legacy, "legacy action source requires explicit opt-in")
    _require(contract.get("alignment") == alignment_for(action_source), "split action alignment mismatch")
    source_root = Path(contract["source_root"])
    _require(sha256_file(source_root / CONTRACT_FILENAME) == contract["source_contract_sha256"],
             "source action contract changed after splitting")
    source = validate_aggregate_contract(source_root, allow_legacy=allow_legacy)
    for key in ("action_source", "alignment", "legacy_action_source", "joint_limits"):
        _require(source[key] == contract.get(key), f"split {key} differs from source action contract")
    all_indices = []
    for name in ("train", "valid"):
        part = contract.get("splits", {}).get(name, {})
        indices = part.get("source_episode_indices", [])
        _require(indices and len(indices) == len(set(indices)), f"{name} source episode IDs are invalid")
        _require(part.get("episode_count") == len(indices), f"{name} episode count mismatch")
        _require(part.get("frame_count") == sum(item["frame_count"] for item in part.get("episodes", [])),
                 f"{name} frame count mismatch")
        expected_episodes = [source["episodes"][index] for index in sorted(indices)]
        _require(part.get("episodes") == expected_episodes,
                 f"{name} episode lineage differs from source action contract")
        all_indices.extend(indices)
    _require(len(all_indices) == len(set(all_indices)), "train/validation action-contract overlap")
    _require(sorted(all_indices) == list(range(source["episode_count"])),
             "split action contract does not cover every source episode")
    return contract
