"""Merge converted LeRobot datasets and write the aggregate action contract.

Each input is a convert_hdf5_to_lerobot.py output root. Its raw HDF5 path comes from
conversion_provenance.json and must still match the recorded SHA256. The merged root can
then be split with prepare_mimic_act_split.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.imitation_learning.action_contract import (  # noqa: E402
    CONTRACT_FILENAME,
    atomic_json,
    build_aggregate_contract,
    validate_aggregate_contract,
)


def converted_records(roots: list[Path]) -> list[dict]:
    """Read each converted root and require matching counts and identical features."""
    if not roots:
        raise ValueError("at least one converted dataset is required")
    records = []
    for root in roots:
        root = root.expanduser().resolve(strict=True)
        provenance_path = root / "conversion_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        counts = (provenance["episode_count"], provenance["frame_count"])
        if (info["total_episodes"], info["total_frames"]) != counts:
            raise ValueError(f"converted dataset counts differ from its provenance: {root}")
        records.append({
            "root": root,
            "raw_path": provenance["input_path"],
            "conversion_provenance_path": str(provenance_path),
            "episodes": counts[0],
            "frames": counts[1],
            "features": info["features"],
        })
    for record in records[1:]:
        if record["features"] != records[0]["features"]:
            raise ValueError(f"feature layout differs from {records[0]['root']}: {record['root']}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--action-source", choices=("recorded_target", "next_observed", "stored"),
                        default="recorded_target")
    parser.add_argument("--allow-legacy-action-source", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists; refusing to overwrite: {output_root}")
    records = converted_records(args.inputs)

    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(
        repo_ids=[f"local/{record['root'].name}" for record in records],
        roots=[record["root"] for record in records],
        aggr_repo_id=args.repo_id,
        aggr_root=output_root,
    )
    episodes = sum(record["episodes"] for record in records)
    frames = sum(record["frames"] for record in records)
    info = json.loads((output_root / "meta/info.json").read_text(encoding="utf-8"))
    if (info["total_episodes"], info["total_frames"]) != (episodes, frames):
        raise ValueError(f"aggregate counts {info['total_episodes']}/{info['total_frames']} != {episodes}/{frames}")
    contract = build_aggregate_contract(output_root, args.action_source, records, episodes, frames)
    atomic_json(output_root / CONTRACT_FILENAME, contract)
    validate_aggregate_contract(output_root, allow_legacy=args.allow_legacy_action_source)
    print(json.dumps({"output_root": str(output_root), "episodes": episodes, "frames": frames}), flush=True)


if __name__ == "__main__":
    main()
