"""Convert SO101 demonstrations to a selected 6D joint action representation."""

import argparse
import shutil
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--representation",
        choices=("joint_delta", "joint_target", "joint_next_state"),
        default="joint_delta",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(input_path, output_path)
        with h5py.File(output_path, "r+") as hdf:
            for demo in hdf["data"].values():
                actions = demo["actions"]
                if "joint_position_targets" in demo:
                    raise ValueError(f"already converted: {demo.name}")
                targets = actions[()]
                current = demo["obs/joint_pos"][()]
                if targets.ndim != 2 or targets.shape[1] != 6:
                    raise ValueError(f"expected 6D joint targets in {demo.name}, got {targets.shape}")
                if current.shape != targets.shape:
                    raise ValueError(
                        f"joint position shape {current.shape} does not match actions {targets.shape} "
                        f"in {demo.name}"
                    )
                if len(targets) < 2:
                    raise ValueError(f"episode is too short to repair its first action: {demo.name}")
                demo.create_dataset("joint_position_targets", data=targets)
                if args.representation == "joint_delta":
                    converted = targets - current
                    converted[0] = targets[1] - current[0]
                    hdf.attrs["action_representation"] = "joint_position_delta"
                    hdf.attrs["action_reference"] = "obs/joint_pos"
                elif args.representation == "joint_target":
                    converted = targets.copy()
                    converted[0] = targets[1]
                    hdf.attrs["action_representation"] = "joint_position_target"
                else:
                    converted = current.copy()
                    converted[:-1] = current[1:]
                    hdf.attrs["action_representation"] = "next_observed_joint_position"
                    hdf.attrs["action_reference"] = "obs/joint_pos[t+1], final frame repeated"
                actions[...] = converted
                previous_actions = converted.copy()
                previous_actions[0] = 0.0
                previous_actions[1:] = converted[:-1]
                del demo["obs/actions"]
                demo["obs"].create_dataset("actions", data=previous_actions)
            first_action_repairs = {
                "joint_delta": "next_joint_target_minus_current_joint_position",
                "joint_target": "next_joint_target",
                "joint_next_state": "not_required_next_observed_state",
            }
            hdf.attrs["first_action_repair"] = first_action_repairs[args.representation]
            hdf.attrs["observation_actions"] = "previous_applied_6d_action"
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise

    print(f"Prepared {args.representation} actions: {output_path}")


if __name__ == "__main__":
    main()
