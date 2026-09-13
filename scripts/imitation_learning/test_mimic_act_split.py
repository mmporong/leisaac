"""CPU-only tests for deterministic Mimic ACT split planning and input guards."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.imitation_learning.prepare_mimic_act_split import (
    EXPECTED_EPISODES,
    build_split_indices,
    run_split,
    validate_split,
)


class MimicActSplitTest(unittest.TestCase):
    def test_seed43_split_is_deterministic_balanced_and_complete(self):
        first = build_split_indices(EXPECTED_EPISODES, seed=43, shard_size=25, valid_per_shard=5)
        second = build_split_indices(EXPECTED_EPISODES, seed=43, shard_size=25, valid_per_shard=5)
        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 400)
        self.assertEqual(len(first["valid"]), 100)
        self.assertFalse(set(first["train"]) & set(first["valid"]))
        self.assertEqual(sorted(first["train"] + first["valid"]), list(range(EXPECTED_EPISODES)))
        for start in range(0, EXPECTED_EPISODES, 25):
            self.assertEqual(sum(start <= index < start + 25 for index in first["valid"]), 5)
        self.assertEqual(first["valid"][:10], [1, 9, 10, 14, 24, 30, 37, 43, 45, 46])

    def test_seed_changes_selection(self):
        self.assertNotEqual(
            build_split_indices(EXPECTED_EPISODES, seed=43),
            build_split_indices(EXPECTED_EPISODES, seed=44),
        )

    def test_invalid_split_inputs_are_rejected(self):
        invalid = (
            (0, 43, 25, 5),
            (499, 43, 25, 5),
            (500, -1, 25, 5),
            (500, 43, 1, 0),
            (500, 43, 25, 0),
            (500, 43, 25, 25),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                build_split_indices(*arguments)

    def test_existing_output_is_rejected_before_dataset_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            (source / "meta").mkdir(parents=True)
            (source / "meta/info.json").write_text("{}")
            (source / "meta/stats.json").write_text("{}")
            output.mkdir()
            with patch(
                "scripts.imitation_learning.prepare_mimic_act_split.LeRobotDataset",
                side_effect=AssertionError("dataset loading must not occur"),
            ):
                with self.assertRaises(FileExistsError):
                    run_split(source, "local/test", output, 43, 25, 5)

    def test_missing_source_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            with self.assertRaises(FileNotFoundError):
                run_split(source, "local/test", root / "output", 43, 25, 5)

    def test_validated_split_record_contains_pipeline_fields(self):
        stats = {"action": {"count": np.array([5]), "mean": np.array([1.0]), "std": np.array([0.5])}}
        source = SimpleNamespace(meta=SimpleNamespace(episodes=[{"length": 2}, {"length": 3}]))
        split = SimpleNamespace(
            repo_id="local/source_train",
            meta=SimpleNamespace(
                total_episodes=2,
                total_frames=5,
                episodes=[{"episode_index": 0, "length": 2}, {"episode_index": 1, "length": 3}],
            ),
        )
        with (
            patch("scripts.imitation_learning.prepare_mimic_act_split.load_stats", return_value=stats),
            patch("scripts.imitation_learning.prepare_mimic_act_split._episode_stats", return_value=stats),
            patch("scripts.imitation_learning.prepare_mimic_act_split._frame_stats", return_value=stats),
            patch("scripts.imitation_learning.prepare_mimic_act_split.sha256", return_value="digest"),
        ):
            record = validate_split(source, split, [0, 1], Path("/tmp/train"))
        self.assertEqual(record["root"], "/tmp/train")
        self.assertEqual(record["repo_id"], "local/source_train")
        self.assertEqual(record["original_to_new_episode_index"], {"0": 0, "1": 1})
        self.assertEqual(record["episode_count"], 2)
        self.assertEqual(record["frame_count"], 5)
        self.assertEqual(record["stats_sha256"], "digest")
        self.assertEqual(record["stats_validation"]["episode_stats_aggregate"], "passed")


if __name__ == "__main__":
    unittest.main()
