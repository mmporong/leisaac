"""Create deterministic train/validation masks for the SO101 MimicGen dataset."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--validation-count", type=int, default=10)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def select_spatial_validation(positions: np.ndarray, count: int) -> list[int]:
    """Select workspace-spanning validation points with farthest-point sampling."""
    if not 0 < count < len(positions):
        raise ValueError("validation-count must be between 1 and the number of demos minus 1")

    center = positions.mean(axis=0)
    selected = [int(np.argmax(np.linalg.norm(positions - center, axis=1)))]
    while len(selected) < count:
        distances = np.linalg.norm(positions[:, None, :] - positions[selected][None, :, :], axis=2)
        min_distances = distances.min(axis=1)
        min_distances[selected] = -1.0
        selected.append(int(np.argmax(min_distances)))
    return selected


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    manifest = args.manifest or dataset.with_suffix(".split.json")

    with h5py.File(dataset, "r+") as hdf:
        demos = sorted(hdf["data"], key=lambda name: int(name.rsplit("_", 1)[1]))
        positions = np.asarray(
            [hdf[f"data/{name}/initial_state/rigid_object/cube/root_pose"][0, :2] for name in demos]
        )
        validation_indices = select_spatial_validation(positions, args.validation_count)
        validation_names = [demos[index] for index in validation_indices]
        validation_set = set(validation_names)
        train_names = [name for name in demos if name not in validation_set]

        mask = hdf.require_group("mask")
        for key in ("train", "valid"):
            if key in mask:
                del mask[key]
        string_dtype = h5py.string_dtype(encoding="utf-8")
        mask.create_dataset("train", data=np.asarray(train_names, dtype=string_dtype))
        mask.create_dataset("valid", data=np.asarray(validation_names, dtype=string_dtype))

        manifest_data = {
            "dataset": str(dataset),
            "method": "farthest_point_sampling_on_initial_cube_xy",
            "train": train_names,
            "validation": validation_names,
            "initial_cube_xy": {
                name: [float(value) for value in positions[index]] for index, name in enumerate(demos)
            },
            "limitations": (
                "MimicGen source-subtask lineage is not present in the generated HDF5. "
                "The masks are episode-disjoint and spatially distributed, but source-lineage independence "
                "must be assessed with separate unseen-seed simulator rollouts."
            ),
        }

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"Prepared {len(train_names)} train and {len(validation_names)} validation demonstrations.")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
