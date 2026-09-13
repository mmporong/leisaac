"""Create a deterministic 400/100 train/valid split of the 500-episode Mimic ACT dataset."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.dataset_tools import _load_episode_with_stats, split_dataset
from lerobot.datasets.io_utils import load_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_DATA_PATH

EXPECTED_EPISODES = 500
FRAME_STATS_FEATURES = ("action", "observation.state")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_free_space(path: Path, minimum_gib: float = 8.0) -> None:
    """Apply the same free-space guard used by the Mimic image batch runner."""
    if not math.isfinite(minimum_gib) or minimum_gib < 1:
        raise ValueError("minimum free space must be finite and at least 1 GiB")
    if shutil.disk_usage(path).free < minimum_gib * 2**30:
        raise RuntimeError(f"free space fell below {minimum_gib} GiB: {path}")


def build_split_indices(
    total_episodes: int,
    seed: int = 43,
    shard_size: int = 25,
    valid_per_shard: int = 5,
) -> dict[str, list[int]]:
    """Select validation episodes independently within each contiguous source shard."""
    if total_episodes <= 0:
        raise ValueError("total episodes must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if shard_size <= 1 or total_episodes % shard_size:
        raise ValueError("shard-size must divide the total episode count and be greater than 1")
    if not 0 < valid_per_shard < shard_size:
        raise ValueError("valid-per-shard must be between 1 and shard-size - 1")

    rng = np.random.default_rng(seed)
    valid: list[int] = []
    for start in range(0, total_episodes, shard_size):
        shard = np.arange(start, start + shard_size)
        valid.extend(int(index) for index in rng.choice(shard, size=valid_per_shard, replace=False))
    valid_set = set(valid)
    return {
        "train": [index for index in range(total_episodes) if index not in valid_set],
        "valid": sorted(valid_set),
    }


def _episode_stats(dataset: LeRobotDataset, episode_indices: list[int]) -> dict:
    stats = []
    for episode_index in episode_indices:
        row = _load_episode_with_stats(dataset, episode_index)
        episode = {}
        for key, value in row.items():
            if not key.startswith("stats/"):
                continue
            parts = key.removeprefix("stats/").rsplit("/", 1)
            if len(parts) != 2:
                raise ValueError(f"invalid episode stats key: {key}")
            feature, statistic = parts
            value = np.asarray(value)
            feature_info = dataset.meta.features.get(feature)
            if feature_info and feature_info["dtype"] in ("image", "video") and statistic != "count":
                if value.dtype == object:
                    flat_values = []
                    for item in value:
                        while isinstance(item, np.ndarray):
                            item = item.flatten()[0]
                        flat_values.append(item)
                    value = np.asarray(flat_values, dtype=np.float64)
                if value.ndim == 1:
                    value = value.reshape(-1, 1, 1)
            episode.setdefault(feature, {})[statistic] = value
        stats.append(episode)
    if not stats:
        raise ValueError("cannot aggregate stats for an empty episode selection")
    return aggregate_stats(stats)


def _assert_stats_close(actual: dict, expected: dict, label: str) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{label} feature keys differ: {set(actual) ^ set(expected)}")
    for feature in expected:
        if set(actual[feature]) != set(expected[feature]):
            raise ValueError(f"{label} statistic keys differ for {feature}")
        for statistic, expected_value in expected[feature].items():
            actual_value = np.asarray(actual[feature][statistic])
            expected_value = np.asarray(expected_value)
            if actual_value.shape != expected_value.shape or not np.allclose(
                actual_value, expected_value, rtol=1e-6, atol=1e-8, equal_nan=False
            ):
                raise ValueError(f"{label} mismatch: {feature}/{statistic}")


def _frame_stats(dataset: LeRobotDataset, episode_indices: list[int]) -> dict[str, dict[str, np.ndarray]]:
    selected = set(episode_indices)
    files: dict[Path, set[int]] = {}
    for episode_index in episode_indices:
        episode = dataset.meta.episodes[episode_index]
        path = dataset.root / DEFAULT_DATA_PATH.format(
            chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
        )
        files.setdefault(path, set()).add(episode_index)

    batches = {feature: [] for feature in FRAME_STATS_FEATURES if feature in dataset.meta.features}
    for path, file_episodes in files.items():
        frame = pd.read_parquet(path, columns=["episode_index", *batches])
        frame = frame[frame["episode_index"].isin(file_episodes & selected)]
        for feature in batches:
            batches[feature].append(np.stack(frame[feature].map(np.asarray).to_list()).astype(np.float64))

    result = {}
    for feature, feature_batches in batches.items():
        values = np.concatenate(feature_batches, axis=0)
        result[feature] = {
            "count": np.asarray([len(values)]),
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
        }
    return result


def _validate_frame_stats(actual: dict, expected: dict, label: str) -> list[str]:
    for feature, feature_stats in expected.items():
        if feature not in actual:
            raise ValueError(f"{label} is missing frame stats for {feature}")
        for statistic, expected_value in feature_stats.items():
            actual_value = np.asarray(actual[feature][statistic])
            if actual_value.shape != expected_value.shape or not np.allclose(
                actual_value, expected_value, rtol=1e-5, atol=1e-7, equal_nan=False
            ):
                raise ValueError(f"{label} independent frame-stat mismatch: {feature}/{statistic}")
    return sorted(expected)


def validate_split(
    source: LeRobotDataset,
    split: LeRobotDataset,
    original_indices: list[int],
    split_root: Path,
) -> dict:
    expected_mapping = {old: new for new, old in enumerate(sorted(original_indices))}
    expected_frames = sum(int(source.meta.episodes[index]["length"]) for index in original_indices)
    if split.meta.total_episodes != len(original_indices) or split.meta.total_frames != expected_frames:
        raise ValueError(f"split episode/frame count mismatch: {split_root}")
    if [episode["episode_index"] for episode in split.meta.episodes] != list(range(len(original_indices))):
        raise ValueError(f"split episode indices are not contiguous: {split_root}")
    expected_lengths = [int(source.meta.episodes[index]["length"]) for index in sorted(original_indices)]
    if [int(episode["length"]) for episode in split.meta.episodes] != expected_lengths:
        raise ValueError(f"split episode lengths do not follow the source mapping: {split_root}")

    actual_stats = load_stats(split_root)
    if actual_stats is None:
        raise FileNotFoundError(f"split stats are missing: {split_root / 'meta/stats.json'}")
    expected_stats = _episode_stats(source, original_indices)
    _assert_stats_close(actual_stats, expected_stats, str(split_root))
    frame_features = _validate_frame_stats(actual_stats, _frame_stats(source, original_indices), str(split_root))
    return {
        "root": str(split_root),
        "repo_id": split.repo_id,
        "original_episode_indices": sorted(original_indices),
        "original_to_new_episode_index": {str(old): new for old, new in expected_mapping.items()},
        "episode_count": len(original_indices),
        "frame_count": expected_frames,
        "stats_sha256": sha256(split_root / "meta/stats.json"),
        "stats_validation": {
            "episode_stats_aggregate": "passed",
            "independent_full_frame_mean_std": "passed",
            "independent_features": frame_features,
        },
    }


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        candidate = candidate.parent
    return candidate


def run_split(
    dataset_root: Path,
    repo_id: str,
    output_dir: Path,
    seed: int,
    shard_size: int,
    valid_per_shard: int,
) -> dict:
    source_root = dataset_root.expanduser().resolve(strict=True)
    output_root = output_dir.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output directory already exists; refusing to overwrite: {output_root}")
    for required in (source_root / "meta/info.json", source_root / "meta/stats.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    check_free_space(_nearest_existing_parent(output_root.parent), 8.0)

    source = LeRobotDataset(repo_id=repo_id, root=source_root)
    if source.meta.total_episodes != EXPECTED_EPISODES:
        raise ValueError(
            f"source must contain exactly {EXPECTED_EPISODES} episodes, got {source.meta.total_episodes}"
        )
    indices = build_split_indices(source.meta.total_episodes, seed, shard_size, valid_per_shard)
    if set(indices["train"]) & set(indices["valid"]) or sorted(indices["train"] + indices["valid"]) != list(
        range(EXPECTED_EPISODES)
    ):
        raise AssertionError("internal split partition invariant failed")

    output_root.mkdir(parents=True, exist_ok=False)
    created = split_dataset(source, indices, output_dir=output_root)
    split_records = {
        name: validate_split(source, created[name], selected, output_root / name)
        for name, selected in indices.items()
    }
    source_expected_stats = _episode_stats(source, list(range(EXPECTED_EPISODES)))
    source_actual_stats = load_stats(source_root)
    if source_actual_stats is None:
        raise FileNotFoundError(source_root / "meta/stats.json")
    _assert_stats_close(source_actual_stats, source_expected_stats, "source")

    marker = {
        "schema_version": 1,
        "success": True,
        "source": str(source_root),
        "source_repo_id": repo_id,
        "source_info_sha256": sha256(source_root / "meta/info.json"),
        "source_stats_sha256": sha256(source_root / "meta/stats.json"),
        "seed": seed,
        "shard_size": shard_size,
        "valid_per_shard": valid_per_shard,
        "source_stats_validation": {"episode_stats_aggregate": "passed"},
        "splits": split_records,
    }
    atomic_json(output_root / "split_provenance.json", marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--valid-per-shard", type=int, default=5)
    args = parser.parse_args()
    if not args.repo_id.strip():
        raise ValueError("repo-id must not be empty")
    marker = run_split(
        args.dataset_root, args.repo_id, args.output_dir, args.seed, args.shard_size, args.valid_per_shard
    )
    print(json.dumps(marker, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
