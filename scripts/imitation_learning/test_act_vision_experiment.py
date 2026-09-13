"""CPU regression checks for the ACT training/validation experiment contract."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.imitation_learning.run_act_vision_experiment import (
    sha256, training_command, validate_offline_report, validate_split,
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

    def test_split_requires_disjoint_cover_and_unchanged_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source/meta").mkdir(parents=True)
            (root / "source/meta/info.json").write_text('{}')
            marker = {"schema_version": 1, "success": True, "source": str(root / "source"),
                      "source_info_sha256": sha256(root / "source/meta/info.json"), "splits": {}}
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
