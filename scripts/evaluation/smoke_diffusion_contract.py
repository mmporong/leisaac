"""CPU-only Diffusion Policy wiring smoke on one registered fixed76 window."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from lerobot.configs import FeatureType, NormalizationMode
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features

REPO = Path(__file__).resolve().parents[2]
SPLIT = REPO / "outputs/act_reset_fixed_76_20260921_split"
OUTPUT = REPO / "outputs/diffusion_contract_smoke_20260922.json"
EPISODE, SAMPLE = 20, 248


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(root: Path) -> dict:
    return {str(path.relative_to(root)): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in sorted(root.rglob("*")) if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    output = parser.parse_args().output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output}")
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.manual_seed(20260922); np.random.seed(20260922)
    root, contract_path = SPLIT / "train", SPLIT / "action_contract.json"
    before = snapshot(root); contract = json.loads(contract_path.read_text())
    entry = contract["splits"]["train"]["episodes"][EPISODE]
    raw_path = Path(entry["raw_path"]); raw_before = sha256(raw_path)
    if raw_before != entry["raw_sha256"] or contract["alignment"] != "action[t] = obs/joint_pos_target[t+1]":
        raise ValueError("registered raw/action contract changed")

    repo_id = "local/so101_mimic_vision224_76_train"
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    features = dataset_to_policy_features(meta.features)
    output_features = {key: value for key, value in features.items() if value.type is FeatureType.ACTION}
    input_features = {key: value for key, value in features.items() if key not in output_features}
    cfg = DiffusionConfig(
        input_features=input_features, output_features=output_features, device="cpu", push_to_hub=False,
        n_obs_steps=1, horizon=32, n_action_steps=30, drop_n_last_frames=2,
        down_dims=(64, 128, 256), pretrained_backbone_weights=None,
        use_separate_rgb_encoder_per_camera=False, num_inference_steps=2,
    )
    deltas = {key: [index / meta.fps for index in cfg.observation_delta_indices]
              for key in ["observation.state", *cfg.image_features]}
    deltas["action"] = [index / meta.fps for index in cfg.action_delta_indices]
    dataset = LeRobotDataset(repo_id, root=root, episodes=[EPISODE], delta_timestamps=deltas)
    sample = dataset[SAMPLE]
    if sample["action"].shape != (32, 6) or bool(sample["action_is_pad"].any()):
        raise ValueError("expected an unpadded 32x6 action window")
    with h5py.File(raw_path, "r") as file:
        raw = file[f"data/{entry['raw_demo']}/obs/joint_pos_target"][SAMPLE + 1:SAMPLE + 33]
    limits = contract["joint_limits"]
    clipped = np.clip(raw, limits["joint_lower_limits_rad"], limits["joint_upper_limits_rad"]).astype(np.float32)
    action = sample["action"].numpy()
    if not np.array_equal(action, clipped) or not np.allclose(action, clipped, atol=1e-6, rtol=0):
        raise ValueError("LeRobot action window differs from clipped raw target[t+1:t+33]")

    batch = {key: (value.unsqueeze(0) if torch.is_tensor(value) else [value]) for key, value in sample.items()}
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)
    processed = preprocessor(batch)
    roundtrip = postprocessor(processed["action"])
    roundtrip_error = float((roundtrip - batch["action"]).abs().max())
    if cfg.normalization_mapping[FeatureType.ACTION] is not NormalizationMode.MIN_MAX or roundtrip_error > 1e-6:
        raise ValueError("MIN_MAX action normalization roundtrip failed")
    policy = DiffusionPolicy(cfg); policy.train()
    loss, _ = policy(processed)
    if not torch.isfinite(loss):
        raise ValueError("non-finite training loss")
    loss.backward()
    gradient_tensors = sum(parameter.grad is not None for parameter in policy.parameters())
    if not gradient_tensors or not all(torch.isfinite(parameter.grad).all() for parameter in policy.parameters()
                                      if parameter.grad is not None):
        raise ValueError("missing or non-finite gradients")
    policy.eval(); policy.reset()
    with torch.inference_mode():
        prediction = policy.predict_action_chunk({key: processed[key] for key in cfg.input_features})
        prediction = postprocessor(prediction)
    if prediction.shape != (1, 30, 6) or not torch.isfinite(prediction).all():
        raise ValueError("invalid postprocessed prediction")
    after, raw_after = snapshot(root), sha256(raw_path)
    if before != after or raw_before != raw_after:
        raise RuntimeError("source dataset or raw HDF5 changed during read-only smoke")

    report = {
        "schema_version": 1, "success": True, "device": "cpu", "torch_threads": 2,
        "source": {"root": str(root), "repo_id": repo_id, "split": "train", "episode": EPISODE,
                   "sample": SAMPLE, "global_frame": int(sample["index"]), "raw_path": str(raw_path),
                   "raw_demo": entry["raw_demo"], "raw_sha256": raw_before,
                   "contract": str(contract_path), "contract_sha256": sha256(contract_path)},
        "sample": {"state_shape": list(sample["observation.state"].shape),
                   "front_shape": list(sample["observation.images.front"].shape),
                   "wrist_shape": list(sample["observation.images.wrist"].shape),
                   "action_shape": list(sample["action"].shape), "action_padding_any": False,
                   "raw_clipped_exact": True, "raw_clipped_max_abs_error": float(np.abs(action - clipped).max())},
        "config": {"n_obs_steps": 1, "horizon": 32, "n_action_steps": 30, "drop_n_last_frames": 2,
                   "down_dims": [64, 128, 256], "num_inference_steps": 2,
                   "pretrained_backbone_weights": None, "shared_rgb_encoder": True,
                   "feature_keys": sorted([*input_features, *output_features])},
        "model_smoke": {"loss": float(loss.detach()), "loss_finite": True,
                        "backward_gradient_tensors": gradient_tensors, "gradients_finite": True, "optimizer_step": False,
                        "action_normalization": "MIN_MAX", "roundtrip_max_abs_error": roundtrip_error,
                        "prediction_shape": list(prediction.shape), "prediction_finite": True},
        "source_unchanged": True, "artifacts_written": [str(output)],
        "limitations": ["random-weight wiring smoke; not training or a performance result",
                        "CPU execution does not establish 8 GiB GPU memory fit or GPU inference",
                        "no optimizer step, checkpoint/model save, Hub transfer, Isaac, or robot execution"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False); stream.write("\n")
    print(json.dumps({"output": str(output), "success": True, "loss": report["model_smoke"]["loss"]}))


if __name__ == "__main__":
    main()
