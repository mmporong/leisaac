"""Inspect camera/joint-state synchronization in raw MimicGen HDF5 demonstrations.

The script is read-only. It checks, per demonstration:

* the first recorded joint state and the first two recorded joint targets,
* whether the first three camera frames repeat (frame 0 == 1, frame 1 == 2),
* the duplicate pattern over the whole episode, split by even/odd frame index,
  which is the fingerprint of a camera whose ``update_period`` is twice the
  physics step (30 Hz camera vs 60 Hz recording),
* how close frame 0 is to the last recorded frame of the previous demo in the
  same shard (candidate source of the stale reset image), and to frame 2.

Usage::

    python scripts/evaluation/inspect_initial_frames.py \
        --contract outputs/mimic_target_unclipped_76_20260919/lerobot_all/action_contract.json \
        --output docs/evidence/initial_frame_sync_20260921.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


CAMERAS = ("front", "wrist")


def frame_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def duplicate_pattern(images: np.ndarray) -> dict:
    """Count exact duplicates between consecutive frames, split by parity of the first index."""
    equal = np.array(
        [bool(np.array_equal(images[k], images[k + 1])) for k in range(len(images) - 1)],
        dtype=bool,
    )
    even = equal[0::2]
    odd = equal[1::2]
    return {
        "consecutive_pairs": int(len(equal)),
        "duplicate_pairs": int(equal.sum()),
        "even_start_pairs": int(len(even)),
        "even_start_duplicates": int(even.sum()),
        "odd_start_pairs": int(len(odd)),
        "odd_start_duplicates": int(odd.sum()),
        "first_non_duplicate_even_pair": (
            int(np.flatnonzero(~even)[0] * 2) if (~even).any() else None
        ),
    }


def inspect_demo(shard: h5py.File, demo: str, previous_demo: str | None) -> dict:
    group = shard["data"][demo]
    joint_pos = group["obs/joint_pos"][:]
    target = group["obs/joint_pos_target"][:]
    record = {
        "demo": demo,
        "frames": int(joint_pos.shape[0]),
        "first_joint_pos": joint_pos[0].tolist(),
        "first_joint_pos_all_zero": bool(np.all(joint_pos[0] == 0)),
        "joint_pos_target_0": target[0].tolist(),
        "joint_pos_target_1": target[1].tolist(),
        "cameras": {},
    }
    for camera in CAMERAS:
        images = group[f"obs/{camera}"][:]
        entry = {
            "frame0_eq_frame1": bool(np.array_equal(images[0], images[1])),
            "frame1_eq_frame2": bool(np.array_equal(images[1], images[2])),
            "mae_frame0_frame2": frame_mae(images[0], images[2]),
            "frame0_sha256": hashlib.sha256(images[0].tobytes()).hexdigest(),
            "duplicates": duplicate_pattern(images),
        }
        if previous_demo is not None and previous_demo in shard["data"]:
            previous_last = shard["data"][previous_demo][f"obs/{camera}"][-1]
            entry["mae_frame0_vs_previous_demo_last_frame"] = frame_mae(images[0], previous_last)
            entry["previous_demo"] = previous_demo
        record["cameras"][camera] = entry
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True, help="LeRobot action_contract.json with raw_path/raw_demo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Inspect only the first N contract episodes")
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    episodes = contract["episodes"][: args.limit] if args.limit else contract["episodes"]

    records = []
    open_shards: dict[str, h5py.File] = {}
    try:
        for episode in episodes:
            raw_path = episode["raw_path"]
            if raw_path not in open_shards:
                open_shards[raw_path] = h5py.File(raw_path, "r")
            shard = open_shards[raw_path]
            demo = episode["raw_demo"]
            index = int(demo.split("_")[1])
            previous_demo = f"demo_{index - 1}" if index > 0 else None
            record = inspect_demo(shard, demo, previous_demo)
            record["aggregate_episode_index"] = episode["aggregate_episode_index"]
            record["raw_path"] = raw_path
            records.append(record)
            print(
                f"{Path(raw_path).name}/{demo}: frames={record['frames']} "
                + " ".join(
                    f"{cam}[0==1:{int(c['frame0_eq_frame1'])} 1==2:{int(c['frame1_eq_frame2'])} "
                    f"even_dup={c['duplicates']['even_start_duplicates']}/{c['duplicates']['even_start_pairs']} "
                    f"odd_dup={c['duplicates']['odd_start_duplicates']}/{c['duplicates']['odd_start_pairs']}"
                    + (
                        f" prev_mae={c['mae_frame0_vs_previous_demo_last_frame']:.3f}"
                        if "mae_frame0_vs_previous_demo_last_frame" in c
                        else ""
                    )
                    + "]"
                    for cam, c in record["cameras"].items()
                ),
                flush=True,
            )
    finally:
        for shard in open_shards.values():
            shard.close()

    def summarize(camera: str) -> dict:
        cams = [r["cameras"][camera] for r in records]
        prev = [c["mae_frame0_vs_previous_demo_last_frame"] for c in cams if "mae_frame0_vs_previous_demo_last_frame" in c]
        return {
            "frame0_eq_frame1": sum(c["frame0_eq_frame1"] for c in cams),
            "frame1_eq_frame2": sum(c["frame1_eq_frame2"] for c in cams),
            "even_start_pairs": sum(c["duplicates"]["even_start_pairs"] for c in cams),
            "even_start_duplicates": sum(c["duplicates"]["even_start_duplicates"] for c in cams),
            "odd_start_pairs": sum(c["duplicates"]["odd_start_pairs"] for c in cams),
            "odd_start_duplicates": sum(c["duplicates"]["odd_start_duplicates"] for c in cams),
            "demos_with_all_even_pairs_duplicated": sum(
                c["duplicates"]["even_start_duplicates"] == c["duplicates"]["even_start_pairs"] for c in cams
            ),
            "mae_frame0_vs_previous_demo_last_frame": {
                "count": len(prev),
                "min": min(prev) if prev else None,
                "median": float(np.median(prev)) if prev else None,
                "max": max(prev) if prev else None,
                "below_0_5": sum(v < 0.5 for v in prev),
            },
            "mae_frame0_frame2": {
                "min": min(c["mae_frame0_frame2"] for c in cams),
                "median": float(np.median([c["mae_frame0_frame2"] for c in cams])),
                "max": max(c["mae_frame0_frame2"] for c in cams),
            },
        }

    summary = {
        "contract": str(args.contract.resolve()),
        "demo_count": len(records),
        "first_joint_pos_all_zero": sum(r["first_joint_pos_all_zero"] for r in records),
        "cameras": {camera: summarize(camera) for camera in CAMERAS},
        "interpretation_note": (
            "even_start_duplicates == even_start_pairs for every demo means each recorded image is "
            "repeated for frames (2k, 2k+1): the camera refreshes every second 60 Hz step. "
            "A small mae_frame0_vs_previous_demo_last_frame means frame 0 is the previous attempt's "
            "final rendered scene; larger values are expected when a discarded failed attempt sat "
            "between the two exported demos."
        ),
        "demos": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "demos"}, indent=2))


if __name__ == "__main__":
    main()
