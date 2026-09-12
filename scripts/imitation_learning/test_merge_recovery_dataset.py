"""CPU-only tests for recovery dataset merging."""

from pathlib import Path
import tempfile
import unittest

import cv2
import h5py
import numpy as np

from merge_recovery_dataset import merge_recovery_dataset


def _write_demo(group: h5py.Group, joint_pos: np.ndarray, success: bool, seed: int, size: int) -> None:
    demo = group.create_group(f"demo_{len(group)}")
    demo.attrs.update(num_samples=len(joint_pos), success=success, seed=seed)
    obs = demo.create_group("obs")
    obs.create_dataset("joint_pos", data=joint_pos.astype(np.float32))
    obs.create_dataset("joint_vel", data=np.zeros_like(joint_pos, dtype=np.float32))
    obs.create_dataset("actions", data=np.zeros((len(joint_pos), 8), dtype=np.float32))
    image = np.arange(size * size * 3, dtype=np.uint8).reshape(size, size, 3)
    for camera in ("front", "wrist"):
        obs.create_dataset(camera, data=np.repeat(image[None], len(joint_pos), axis=0))
    demo.create_dataset("actions", data=np.zeros((len(joint_pos), 8), dtype=np.float32))
    demo.create_dataset("processed_actions", data=np.ones((len(joint_pos), 8), dtype=np.float32))


def _make_baseline(path: Path) -> None:
    with h5py.File(path, "w") as hdf:
        hdf.attrs["action_representation"] = "next_observed_joint_position"
        data = hdf.create_group("data")
        for index in range(3):
            joint_pos = np.arange(18, dtype=np.float32).reshape(3, 6) + index * 100
            demo = data.create_group(f"demo_{index}")
            demo.attrs.update(num_samples=3, success=True)
            obs = demo.create_group("obs")
            actions = np.concatenate((joint_pos[1:], joint_pos[-1:]))
            demo.create_dataset("actions", data=actions)
            obs.create_dataset("joint_pos", data=joint_pos)
            obs.create_dataset("joint_vel", data=np.zeros_like(joint_pos))
            obs.create_dataset("actions", data=np.zeros_like(joint_pos))
            for camera in ("front", "wrist"):
                obs.create_dataset(camera, data=np.zeros((3, 84, 84, 3), dtype=np.uint8))
        data.attrs["total"] = 9
        mask = hdf.create_group("mask")
        dtype = h5py.string_dtype("utf-8")
        mask.create_dataset("train", data=np.asarray(["demo_0", "demo_1"], dtype=dtype))
        mask.create_dataset("valid", data=np.asarray(["demo_2"], dtype=dtype))


class MergeRecoveryDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.baseline = self.root / "baseline.hdf5"
        _make_baseline(self.baseline)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_filters_failure_converts_labels_preserves_split_and_deduplicates(self) -> None:
        first = self.root / "recovery_seed3100_a.hdf5"
        second = self.root / "recovery_seed3100_b.hdf5"
        trajectory = np.arange(24, dtype=np.float32).reshape(4, 6) + 500
        with h5py.File(first, "w") as hdf:
            data = hdf.create_group("data")
            _write_demo(data, trajectory, True, 3100, 100)
            _write_demo(data, trajectory + 10, False, 3100, 100)
        with h5py.File(second, "w") as hdf:
            _write_demo(hdf.create_group("data"), trajectory, True, 3100, 84)

        output = self.root / "merged.hdf5"
        manifest_path = self.root / "merged.json"
        manifest = merge_recovery_dataset(
            self.baseline, [first, second], output, manifest_path, batch_size=2
        )

        self.assertEqual(len(manifest["accepted_recovery"]), 1)
        statuses = [
            episode
            for source in manifest["recovery_sources"]
            for episode in source["episodes"]
        ]
        self.assertEqual([entry["status"] for entry in statuses], ["accepted", "skipped", "skipped"])
        self.assertEqual(statuses[1]["reason"], "success_attr_is_not_true")
        self.assertEqual(statuses[2]["reason"], "duplicate_action_state")
        self.assertTrue(all(source["sha256"] for source in manifest["recovery_sources"]))
        self.assertEqual(manifest["accepted_recovery"][0]["seed"], 3100)

        with h5py.File(output, "r") as hdf:
            self.assertEqual(list(hdf["data"]), ["demo_0", "demo_1", "demo_2", "demo_3"])
            train = hdf["mask/train"].asstr()[...].tolist()
            valid = hdf["mask/valid"].asstr()[...].tolist()
            self.assertEqual(train, ["demo_0", "demo_1", "demo_3"])
            self.assertEqual(valid, ["demo_2"])
            self.assertFalse(set(train) & set(valid))
            demo = hdf["data/demo_3"]
            expected = np.concatenate((trajectory[1:], trajectory[-1:]))
            np.testing.assert_array_equal(demo["actions"], expected)
            np.testing.assert_array_equal(demo["obs/actions"][0], np.zeros(6))
            np.testing.assert_array_equal(demo["obs/actions"][1:], expected[:-1])
            self.assertEqual(demo["obs/front"].shape, (4, 84, 84, 3))
            source_image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
            np.testing.assert_array_equal(
                demo["obs/front"][0], cv2.resize(source_image, (84, 84), interpolation=cv2.INTER_AREA)
            )
            self.assertNotIn("processed_actions", demo)
            self.assertEqual(hdf["data"].attrs["total"], 13)

    def test_rejects_bad_shape_or_non_finite_successful_demo_atomically(self) -> None:
        for case in ("shape", "finite"):
            with self.subTest(case=case):
                recovery = self.root / f"bad_{case}.hdf5"
                output = self.root / f"bad_{case}_output.hdf5"
                manifest = self.root / f"bad_{case}_manifest.json"
                joint_pos = np.zeros((3, 5 if case == "shape" else 6), dtype=np.float32)
                if case == "finite":
                    joint_pos[1, 0] = np.nan
                with h5py.File(recovery, "w") as hdf:
                    _write_demo(hdf.create_group("data"), joint_pos, True, 3101, 84)
                with self.assertRaises(ValueError):
                    merge_recovery_dataset(self.baseline, [recovery], output, manifest)
                self.assertFalse(output.exists())
                self.assertFalse(manifest.exists())

    def test_rejects_zero_valid_recovery_and_existing_output(self) -> None:
        recovery = self.root / "failed.hdf5"
        with h5py.File(recovery, "w") as hdf:
            _write_demo(hdf.create_group("data"), np.zeros((2, 6)), False, 3102, 84)
        output = self.root / "output.hdf5"
        manifest = self.root / "manifest.json"
        with self.assertRaisesRegex(ValueError, "no valid"):
            merge_recovery_dataset(self.baseline, [recovery], output, manifest)
        self.assertFalse(output.exists())
        output.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            merge_recovery_dataset(self.baseline, [recovery], output, manifest)
        self.assertEqual(output.read_bytes(), b"keep")

    def test_rejects_wrong_baseline_action_representation(self) -> None:
        with h5py.File(self.baseline, "r+") as hdf:
            hdf.attrs["action_representation"] = "joint_position_target"
        recovery = self.root / "successful.hdf5"
        with h5py.File(recovery, "w") as hdf:
            _write_demo(hdf.create_group("data"), np.zeros((2, 6)), True, 3103, 84)
        with self.assertRaisesRegex(ValueError, "action_representation"):
            merge_recovery_dataset(
                self.baseline, [recovery], self.root / "output.hdf5", self.root / "manifest.json"
            )

    def test_import_does_not_run_cli(self) -> None:
        __import__("merge_recovery_dataset")

    def test_rejects_duplicate_mask_entries(self) -> None:
        with h5py.File(self.baseline, "r+") as hdf:
            hdf["mask/train"][1] = "demo_0"
        recovery = self.root / "successful.hdf5"
        with h5py.File(recovery, "w") as hdf:
            _write_demo(hdf.create_group("data"), np.zeros((2, 6)), True, 3103, 84)
        with self.assertRaisesRegex(ValueError, "duplicate or overlapping"):
            merge_recovery_dataset(self.baseline, [recovery], self.root / "output.hdf5")


if __name__ == "__main__":
    unittest.main()
