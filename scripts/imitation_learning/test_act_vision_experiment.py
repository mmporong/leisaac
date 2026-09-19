"""CPU regression checks for the ACT training/validation experiment contract."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.imitation_learning.run_act_vision_experiment import (
    sha256, training_command, validate_offline_report, validate_split,
)
from scripts.imitation_learning.action_contract import (
    CONTRACT_FILENAME, atomic_json, build_split_contract,
)


class ActVisionExperimentTest(unittest.TestCase):
    def test_training_uses_separate_root_and_explicit_hyperparameters(self):
        command = training_command(Path("/train"), Path("/model"), "local/train", 30000, 8)
        for option in ("--dataset.root=/train", "--steps=30000", "--batch_size=8",
                       "--policy.optimizer_lr=0.0001", "--policy.chunk_size=30",
                       "--dataset.eval_split=0", "--dataset.use_imagenet_stats=true"):
            self.assertIn(option, command)
        self.assertFalse(any("policy.path" in option or "dataset.episodes" in option for option in command))
        for steps, batch in ((0, 8), (30000, 0)):
            with self.assertRaises(ValueError):
                training_command(Path("/train"), Path("/model"), "local/train", steps, batch)

    def test_offline_metrics_reject_nonfinite_and_wrong_counts(self):
        metric = {"samples": 10, "rmse": 0.1, "mae": 0.05, "per_joint_rmse": [0.1] * 6}
        report = {"episodes": 100, "frames": 50000, "first_frames": dict(metric), "uniform_frames": dict(metric)}
        valid = {"episode_count": 100, "frame_count": 50000}
        validate_offline_report(report, valid)
        report["first_frames"]["rmse"] = float("nan")
        with self.assertRaises(ValueError):
            validate_offline_report(report, valid)
        report["first_frames"]["rmse"] = 0.1
        report["episodes"] = 99
        with self.assertRaises(ValueError):
            validate_offline_report(report, valid)

    def test_uncontracted_opt_in_never_relabels_recorded_target_or_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = root / "source/meta/info.json"
            info.parent.mkdir(parents=True)
            info.write_text('{}')
            marker = {"schema_version": 1, "success": True, "source": str(root / "source"),
                      "source_info_sha256": sha256(info)}
            for source in ("recorded_target", "unknown", None):
                with self.subTest(source=source):
                    atomic_json(root / "split_provenance.json", marker | {"action_source": source})
                    with self.assertRaisesRegex(ValueError, "uncontracted legacy opt-in"):
                        validate_split(root, allow_legacy_action_source=True)

    def test_split_requires_disjoint_cover_and_unchanged_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source/meta").mkdir(parents=True)
            (root / "source/meta/info.json").write_text('{}')
            raw = root / "raw.hdf5"
            raw.write_bytes(b"raw")
            limits = root / "limits.json"
            limit_payload = {
                "joint_names": ["a", "b", "c", "d", "e", "f"],
                "joint_lower_limits_rad": [-1.0] * 6,
                "joint_upper_limits_rad": [1.0] * 6,
            }
            limits.write_text(json.dumps(limit_payload))
            limit_record = {
                "path": str(limits), "sha256": sha256(limits),
                **limit_payload,
            }
            conversion = root / "conversion_provenance.json"
            atomic_json(conversion, {
                "input_sha256": sha256(raw), "action_source": "recorded_target",
                "target_alignment": "action[t] = obs/joint_pos_target[t+1]",
                "episode_count": 500, "frame_count": 1000, "joint_limits": limit_record,
                "episodes": [{"name": f"demo_{index}", "frame_count": 2} for index in range(500)],
            })
            source_contract = {
                "schema_version": 1, "stage": "aggregate", "dataset_root": str(root / "source"),
                "action_source": "recorded_target",
                "alignment": "action[t] = obs/joint_pos_target[t+1]",
                "legacy_action_source": False,
                "joint_limits": limit_record,
                "episode_count": 500, "frame_count": 1000,
                "sources": [{
                    "raw_path": str(raw), "raw_sha256": sha256(raw),
                    "conversion_provenance_path": str(conversion),
                    "conversion_provenance_sha256": sha256(conversion),
                    "episode_count": 500, "frame_count": 1000,
                }],
                "episodes": [
                    {"aggregate_episode_index": index, "raw_path": str(raw), "raw_sha256": sha256(raw),
                     "raw_demo": f"demo_{index}", "frame_count": 2}
                    for index in range(500)
                ],
            }
            atomic_json(root / "source" / CONTRACT_FILENAME, source_contract)
            split_contract = build_split_contract(
                root / "source", source_contract,
                {"train": list(range(400)), "valid": list(range(400, 500))},
            )
            atomic_json(root / CONTRACT_FILENAME, split_contract)
            marker = {"schema_version": 1, "success": True, "source": str(root / "source"),
                      "source_info_sha256": sha256(root / "source/meta/info.json"),
                      "action_contract_sha256": sha256(root / CONTRACT_FILENAME),
                      "action_source": "recorded_target", "splits": {}}
            for name, indices in (("train", list(range(400))), ("valid", list(range(400, 500)))):
                meta = root / name / "meta"
                meta.mkdir(parents=True)
                (meta / "stats.json").write_text('{}')
                info = {"total_episodes": len(indices), "total_frames": len(indices) * 2,
                        "features": {f"observation.images.{cam}": {"shape": [224, 224, 3]}
                                     for cam in ("front", "wrist")}}
                (meta / "info.json").write_text(json.dumps(info))
                marker["splits"][name] = {"root": str(root / name), "original_episode_indices": indices,
                                          "episode_count": len(indices), "frame_count": len(indices) * 2,
                                          "stats_sha256": sha256(meta / "stats.json"),
                                          "stats_validation": {"episode_stats_aggregate": "passed",
                                                               "independent_full_frame_mean_std": "passed"}}
            path = root / "split_provenance.json"
            path.write_text(json.dumps(marker))
            validate_split(root)
            marker["splits"]["valid"]["original_episode_indices"][0] = 0
            path.write_text(json.dumps(marker))
            with self.assertRaises(ValueError):
                validate_split(root)
            marker["splits"]["valid"]["original_episode_indices"][0] = 400
            path.write_text(json.dumps(marker))
            (root / "train/meta/stats.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                validate_split(root)


if __name__ == "__main__":
    unittest.main()
