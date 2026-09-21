"""CPU checks for shard planning and success/image validation."""

from pathlib import Path
import json
import tempfile
import unittest

import h5py
import numpy as np

from scripts.imitation_learning.run_mimic_image_batch import (
    check_free_space, load_selection_manifest, sha256, shard_plan, validate_conversion,
    validate_generation_manifest, validate_shard,
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

    def test_generation_manifest_treats_missing_reset_render_frames_as_legacy_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            base = {"completed": True, "datagen_seed": 6100, "successful_demos": 25, "max_attempts": 250}
            legacy = {**base, "failed_demos": 30, "attempts": 55}
            path.write_text(json.dumps(legacy))
            validate_generation_manifest(path, {**base, "reset_render_frames": 0})
            with self.assertRaises(ValueError):
                validate_generation_manifest(path, {**base, "reset_render_frames": 4})
            path.write_text(json.dumps({**legacy, "reset_render_frames": 4}))
            validate_generation_manifest(path, {**base, "reset_render_frames": 4})
            with self.assertRaises(ValueError):
                validate_generation_manifest(path, {**base, "reset_render_frames": 0})

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
                validate_conversion(root, raw, 3, 100, 224, "next_observed")
            provenance = {
                "input_sha256": sha256(raw), "action_source": "next_observed",
                "image_size": 224, "episode_count": 3, "frame_count": 100,
            }
            marker = root / "conversion_provenance.json"
            marker.write_text(json.dumps(provenance))
            validate_conversion(root, raw, 3, 100, 224, "next_observed")
            provenance["action_source"] = "stored"
            marker.write_text(json.dumps(provenance))
            with self.assertRaises(ValueError):
                validate_conversion(root, raw, 3, 100, 224, "next_observed")

    def test_plan_has_unique_seeds_and_exact_target(self):
        plan = shard_plan(500, 25, 6100)
        self.assertEqual(len(plan), 20)
        self.assertEqual(sum(item["episodes"] for item in plan), 500)
        self.assertEqual(len({item["seed"] for item in plan}), 20)
        self.assertEqual(shard_plan(7, 3, 10)[-1]["episodes"], 1)
        for total, chunk, seed in ((0, 25, 0), (500, 0, 0), (1, 1, -1)):
            with self.assertRaises(ValueError):
                shard_plan(total, chunk, seed)

    def test_selection_requires_matching_raw_and_audit_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "shard_000.hdf5"
            raw.write_bytes(b"raw")
            audit = root / "audit.json"
            limits = {"sha256": "limits-for-test"}
            audit.write_text(json.dumps({
                "schema_version": 1, "max_action_step_norm": 1.0, "joint_limits": limits,
                "input_sha256": sha256(raw), "action_source": "recorded_target",
                "episode_count": 3, "accepted_count": 2, "rejected_count": 1,
                "episodes": [
                    {"name": "demo_0", "accepted": True},
                    {"name": "demo_1", "accepted": False},
                    {"name": "demo_2", "accepted": True},
                ],
            }))
            selection = root / "selection.json"
            selection.write_text(json.dumps({
                "schema_version": 1, "action_source": "recorded_target",
                "shards": [{
                    "filename": raw.name, "raw_sha256": sha256(raw),
                    "audit_path": str(audit), "audit_sha256": sha256(audit),
                    "selected_demo_names": ["demo_2", "demo_0"],
                }],
            }))
            plan = load_selection_manifest(selection, root, "recorded_target", 1.0, limits)
            self.assertEqual(plan[0]["selected_demo_names"], ["demo_2", "demo_0"])
            original_audit = json.loads(audit.read_text())
            original_selection = json.loads(selection.read_text())
            for override in ({"schema_version": 999}, {"max_action_step_norm": 999},
                             {"joint_limits": None}, {"accepted_count": 3}, {"episode_count": 999}):
                with self.subTest(override=override):
                    audit.write_text(json.dumps(original_audit | override))
                    modified = json.loads(json.dumps(original_selection))
                    modified["shards"][0]["audit_sha256"] = sha256(audit)
                    selection.write_text(json.dumps(modified))
                    with self.assertRaises(ValueError):
                        load_selection_manifest(selection, root, "recorded_target", 1.0, limits)
            audit.write_text(json.dumps(original_audit))
            selection.write_text(json.dumps(original_selection))
            single = original_audit | {"episodes": [{"name": "demo_0", "accepted": True}],
                                      "episode_count": 1, "accepted_count": 1, "rejected_count": 0}
            for key, value in (("schema_version", True), ("max_action_step_norm", True),
                               ("episode_count", True), ("accepted_count", True),
                               ("rejected_count", False)):
                with self.subTest(boolean_field=key):
                    audit.write_text(json.dumps(single | {key: value}))
                    modified = json.loads(json.dumps(original_selection))
                    modified["shards"][0].update(audit_sha256=sha256(audit), selected_demo_names=["demo_0"])
                    selection.write_text(json.dumps(modified))
                    with self.assertRaises(ValueError):
                        load_selection_manifest(selection, root, "recorded_target", 1.0, limits)
            audit.write_text(json.dumps(original_audit))
            selection.write_text(json.dumps(original_selection | {"schema_version": True}))
            with self.assertRaises(ValueError):
                load_selection_manifest(selection, root, "recorded_target", 1.0, limits)
            selection.write_text(json.dumps(original_selection))
            payload = json.loads(selection.read_text())
            payload["shards"][0]["selected_demo_names"] = ["demo_1"]
            selection.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "rejected"):
                load_selection_manifest(selection, root, "recorded_target", 1.0, limits)

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
