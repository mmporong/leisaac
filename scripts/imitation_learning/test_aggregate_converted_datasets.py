"""CPU checks for reading converted LeRobot roots before aggregation."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.imitation_learning.aggregate_converted_datasets import converted_records


def write_root(root: Path, episodes: int, frames: int, info_frames: int | None = None, camera: str = "wrist"):
    (root / "meta").mkdir(parents=True)
    (root / "conversion_provenance.json").write_text(json.dumps({
        "input_path": f"/raw/{root.name}.hdf5", "episode_count": episodes, "frame_count": frames,
    }))
    (root / "meta/info.json").write_text(json.dumps({
        "total_episodes": episodes, "total_frames": frames if info_frames is None else info_frames,
        "features": {f"observation.images.{camera}": {"shape": [224, 224, 3]}},
    }))
    return root


class ConvertedRecordsTest(unittest.TestCase):
    def test_records_carry_raw_path_and_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            first = write_root(Path(folder) / "a", 10, 7000)
            second = write_root(Path(folder) / "b", 50, 35000)
            records = converted_records([first, second])
        self.assertEqual([record["raw_path"] for record in records], ["/raw/a.hdf5", "/raw/b.hdf5"])
        self.assertEqual([(record["episodes"], record["frames"]) for record in records], [(10, 7000), (50, 35000)])

    def test_count_or_feature_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "counts differ"):
                converted_records([write_root(Path(folder) / "a", 10, 7000, info_frames=6999)])
            wrist = write_root(Path(folder) / "b", 10, 7000)
            front = write_root(Path(folder) / "c", 10, 7000, camera="front")
            with self.assertRaisesRegex(ValueError, "feature layout differs"):
                converted_records([wrist, front])
            with self.assertRaises(ValueError):
                converted_records([])


if __name__ == "__main__":
    unittest.main()
