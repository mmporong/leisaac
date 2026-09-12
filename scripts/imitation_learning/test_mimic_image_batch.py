"""CPU checks for shard planning and success/image validation."""

from pathlib import Path
import json
import tempfile
import unittest

import h5py
import numpy as np

from scripts.imitation_learning.run_mimic_image_batch import (
    check_free_space, sha256, shard_plan, validate_conversion, validate_generation_manifest, validate_shard,
)


class MimicImageBatchTest(unittest.TestCase):
    def test_generation_manifest_rejects_wrong_seed_and_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            expected = {"completed": True, "datagen_seed": 6100,
                        "successful_demos": 25, "max_attempts": 250}
            manifest = {**expected, "failed_demos": 30, "attempts": 55}
            path.write_text(json.dumps(manifest))
            validate_generation_manifest(path, expected)
            for override in ({"datagen_seed": 6101}, {"attempts": 54}, {"completed": False}):
                path.write_text(json.dumps(manifest | override))
                with self.assertRaises(ValueError):
                    validate_generation_manifest(path, expected)
    def test_disk_threshold_rejects_nonfinite_values(self):
        for value in (float("nan"), float("inf"), -1, 0):
            with self.assertRaises(ValueError):
                check_free_space(Path("."), value)

    def test_conversion_requires_matching_final_provenance_and_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.hdf5"
            raw.write_bytes(b"test source")
            (root / "meta").mkdir()
            info = {
                "total_episodes": 3, "total_frames": 100,
                "features": {f"observation.images.{camera}": {"shape": [224, 224, 3]}
                             for camera in ("front", "wrist")},
            }
            (root / "meta/info.json").write_text(json.dumps(info))
            with self.assertRaises(FileNotFoundError):
                validate_conversion(root, raw, 3, 100, 224)
            provenance = {
                "input_sha256": sha256(raw), "action_source": "next_observed",
                "image_size": 224, "episode_count": 3, "frame_count": 100,
            }
            marker = root / "conversion_provenance.json"
            marker.write_text(json.dumps(provenance))
            validate_conversion(root, raw, 3, 100, 224)
            provenance["action_source"] = "stored"
            marker.write_text(json.dumps(provenance))
            with self.assertRaises(ValueError):
                validate_conversion(root, raw, 3, 100, 224)

    def test_plan_has_unique_seeds_and_exact_target(self):
        plan = shard_plan(500, 25, 6100)
        self.assertEqual(len(plan), 20)
        self.assertEqual(sum(item["episodes"] for item in plan), 500)
        self.assertEqual(len({item["seed"] for item in plan}), 20)
        self.assertEqual(shard_plan(7, 3, 10)[-1]["episodes"], 1)
        for total, chunk, seed in ((0, 25, 0), (500, 0, 0), (1, 1, -1)):
            with self.assertRaises(ValueError):
                shard_plan(total, chunk, seed)

    def test_shard_rejects_failure_wrong_count_and_image_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.hdf5"
            with h5py.File(path, "w") as hdf:
                demo = hdf.create_group("data/demo_0")
                demo.attrs["success"] = True
                demo.create_dataset("actions", data=np.zeros((3, 8)))
                demo.create_dataset("obs/joint_pos", data=np.zeros((3, 6)))
                for camera in ("front", "wrist"):
                    demo.create_dataset(f"obs/{camera}", data=np.zeros((3, 224, 224, 3), dtype=np.uint8))
            self.assertEqual(validate_shard(path, 1, 224), 3)
            with self.assertRaises(ValueError):
                validate_shard(path, 2, 224)
            with self.assertRaises(ValueError):
                validate_shard(path, 1, 84)
            with h5py.File(path, "a") as hdf:
                hdf["data/demo_0/actions"][0, 0] = np.nan
            with self.assertRaises(ValueError):
                validate_shard(path, 1, 224)
            with h5py.File(path, "a") as hdf:
                hdf["data/demo_0/actions"][0, 0] = 0
                hdf["data/demo_0"].attrs["success"] = False
            with self.assertRaises(ValueError):
                validate_shard(path, 1, 224)


if __name__ == "__main__":
    unittest.main()
