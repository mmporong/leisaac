"""Measure ACT one-step predictions on stored LeRobot demonstrations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def predict(
    model: ACTPolicy,
    preprocessor,
    postprocessor,
    sample: dict,
    device: torch.device,
) -> np.ndarray:
    observation = {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if key.startswith("observation.")
    }
    model.reset()
    with torch.inference_mode():
        action = postprocessor(model.select_action(preprocessor(observation)))
    return action.squeeze(0).detach().cpu().numpy()


def error_summary(errors: np.ndarray) -> dict:
    return {
        "samples": len(errors),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "per_joint_rmse": np.sqrt(np.mean(np.square(errors), axis=0)).tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("max-samples must be positive")

    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    device = torch.device(args.device)

    dataset = LeRobotDataset(args.repo_id, root=dataset_root, episodes=args.episodes)
    model = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
    device_override = {"device": str(device)}
    preprocessor, postprocessor = make_pre_post_processors(
        model.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": device_override},
        postprocessor_overrides={"device_processor": device_override},
    )

    episode_indices = np.asarray(dataset.hf_dataset["episode_index"])
    start_indices = np.flatnonzero(
        np.concatenate(([True], episode_indices[1:] != episode_indices[:-1]))
    ).tolist()
    sampled_indices = np.linspace(
        0,
        len(dataset) - 1,
        num=min(args.max_samples, len(dataset)),
        dtype=int,
    ).tolist()

    first_errors = []
    for index in start_indices:
        sample = dataset[index]
        first_errors.append(predict(model, preprocessor, postprocessor, sample, device) - sample["action"].numpy())

    sampled_errors = []
    for index in sampled_indices:
        sample = dataset[index]
        sampled_errors.append(
            predict(model, preprocessor, postprocessor, sample, device) - sample["action"].numpy()
        )

    report = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "episodes": dataset.num_episodes,
        "frames": dataset.num_frames,
        "first_frames": error_summary(np.asarray(first_errors)),
        "uniform_frames": error_summary(np.asarray(sampled_errors)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
