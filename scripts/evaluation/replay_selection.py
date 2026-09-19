"""CPU-only source selection and validation for joint-contract replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


DEFAULT_EPISODES = (0, 1, 2)
_DEMO_NAME = re.compile(r"demo_[0-9]+")
_MANIFEST_REQUIRED_KEYS = {"schema_version", "action_source", "shards"}
_MANIFEST_ALLOWED_KEYS = _MANIFEST_REQUIRED_KEYS | {"purpose"}
_SHARD_REQUIRED_KEYS = {"filename", "raw_path", "raw_sha256", "selected_demo_names"}
_SHARD_ALLOWED_KEYS = _SHARD_REQUIRED_KEYS | {"audit_path", "audit_sha256"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} does not resolve to an existing file: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def _validate_episode_indices(episodes: list[int] | None) -> list[int]:
    selected = list(DEFAULT_EPISODES if episodes is None else episodes)
    if (not selected or any(type(index) is not int or index < 0 for index in selected)
            or len(selected) != len(set(selected))):
        raise ValueError("episode indices must be nonnegative and unique")
    return selected


def load_replay_selection(
    dataset: Path | None,
    episodes: list[int] | None,
    selection_manifest: Path | None,
) -> tuple[list[dict], dict]:
    """Return validated replay shards and provenance metadata.

    ``episodes=None`` intentionally means the legacy dataset default.  It also
    lets callers distinguish an explicit ``--episodes`` from that default when
    a selection manifest is used.
    """
    if (dataset is None) == (selection_manifest is None):
        raise ValueError("exactly one of dataset and selection manifest is required")

    if dataset is not None:
        raw = _resolved_file(dataset, "dataset")
        names = [f"demo_{index}" for index in _validate_episode_indices(episodes)]
        raw_sha256 = sha256_file(raw)
        return ([{
            "filename": raw.name,
            "raw_path": raw,
            "raw_sha256": raw_sha256,
            "selected_demo_names": names,
            "manifest_shard_index": None,
        }], {
            "dataset": str(raw),
            "selection_manifest": None,
            "selection_manifest_sha256": None,
            "selection_action_source": None,
        })

    if episodes is not None:
        raise ValueError("--episodes cannot be combined with --selection-manifest")
    manifest_path = _resolved_file(selection_manifest, "selection manifest")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid selection manifest JSON: {manifest_path}") from exc
    if (not isinstance(manifest, dict)
            or not _MANIFEST_REQUIRED_KEYS <= set(manifest) <= _MANIFEST_ALLOWED_KEYS):
        raise ValueError("selection manifest has missing or unknown fields")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported selection manifest schema")
    if manifest["action_source"] != "recorded_target":
        raise ValueError("selection manifest action_source must be recorded_target")
    if "purpose" in manifest and (not isinstance(manifest["purpose"], str) or not manifest["purpose"]):
        raise ValueError("selection manifest purpose must be a nonempty string")
    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise ValueError("selection manifest shards must be a nonempty list")

    plan = []
    selected_identities: set[tuple[str, str]] = set()
    for index, shard in enumerate(shards):
        if (not isinstance(shard, dict)
                or not _SHARD_REQUIRED_KEYS <= set(shard) <= _SHARD_ALLOWED_KEYS):
            raise ValueError(f"selection shard {index} has missing or unknown fields")
        filename = shard["filename"]
        raw_path_value = shard["raw_path"]
        raw_sha256 = shard["raw_sha256"]
        names = shard["selected_demo_names"]
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError(f"selection shard {index} filename must be a basename")
        if not isinstance(raw_path_value, str) or not Path(raw_path_value).is_absolute():
            raise ValueError(f"selection shard {index} raw_path must be absolute")
        raw = _resolved_file(Path(raw_path_value), f"selection shard {index} raw_path")
        if filename != raw.name:
            raise ValueError(f"selection shard {index} filename does not match raw_path")
        if (not isinstance(raw_sha256, str) or len(raw_sha256) != 64
                or any(character not in "0123456789abcdef" for character in raw_sha256)):
            raise ValueError(f"selection shard {index} raw_sha256 must be lowercase SHA-256")
        if sha256_file(raw) != raw_sha256:
            raise ValueError(f"selection raw hash mismatch: {raw}")
        has_audit_path = "audit_path" in shard
        has_audit_sha256 = "audit_sha256" in shard
        if has_audit_path != has_audit_sha256:
            raise ValueError(f"selection shard {index} audit_path and audit_sha256 must appear together")
        if has_audit_path:
            audit_path_value = shard["audit_path"]
            audit_sha256 = shard["audit_sha256"]
            if not isinstance(audit_path_value, str) or not Path(audit_path_value).is_absolute():
                raise ValueError(f"selection shard {index} audit_path must be absolute")
            audit_path = _resolved_file(Path(audit_path_value), f"selection shard {index} audit_path")
            if (not isinstance(audit_sha256, str) or len(audit_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in audit_sha256)
                    or sha256_file(audit_path) != audit_sha256):
                raise ValueError(f"selection audit hash mismatch: {audit_path}")
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or _DEMO_NAME.fullmatch(name) is None for name in names)
                or len(names) != len(set(names))):
            raise ValueError(f"selection for {filename} must contain unique demo_N names")
        identities = {(raw_sha256, name) for name in names}
        if selected_identities & identities:
            raise ValueError("selection contains duplicate raw/demo identities")
        selected_identities.update(identities)
        plan.append({
            "filename": filename,
            "raw_path": raw,
            "raw_sha256": raw_sha256,
            "selected_demo_names": list(names),
            "manifest_shard_index": index,
        })

    return plan, {
        "dataset": None,
        "selection_manifest": str(manifest_path),
        "selection_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selection_action_source": manifest["action_source"],
    }
