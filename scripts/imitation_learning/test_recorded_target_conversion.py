"""CPU-only regression tests for recorded actuator-target conversion."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("convert_hdf5_to_lerobot.py")


def load_converter():
    h5py = ModuleType("h5py")
    h5py.File = object
    h5py.Group = object
    lerobot = ModuleType("lerobot")
    datasets = ModuleType("lerobot.datasets")
    dataset_module = ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_module.LeRobotDataset = object
    previous = {
        name: sys.modules.get(name)
        for name in (
            "h5py",
            "lerobot",
            "lerobot.datasets",
            "lerobot.datasets.lerobot_dataset",
        )
    }
    sys.modules.update(
        {
            "h5py": h5py,
            "lerobot": lerobot,
            "lerobot.datasets": datasets,
            "lerobot.datasets.lerobot_dataset": dataset_module,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location("recorded_target_converter", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class ArrayDataset:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape = self.value.shape
        self.dtype = self.value.dtype

    def __len__(self):
        return len(self.value)

    def __getitem__(self, key):
        return self.value[key]


class Demo(dict):
    name = "/data/demo_0"
    attrs = {"success": True}


class RecordedTargetConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.converter = load_converter()
        cls.names = [name.removesuffix(".pos") for name in cls.converter.JOINT_NAMES]

    def write_limits(self, root: Path, **overrides) -> Path:
        payload = {
            "joint_names": self.names,
            "joint_lower_limits_rad": [-1.0] * 6,
            "joint_upper_limits_rad": [1.0] * 6,
        }
        payload.update(overrides)
        path = root / "evaluation.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def make_demo(self) -> Demo:
        joint_pos = np.arange(18, dtype=np.float32).reshape(3, 6) / 100
        targets = np.stack(
            (
                np.zeros(6, dtype=np.float32),
                np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=np.float32),
                np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
            )
        )
        return Demo(
            {
                "obs/joint_pos": ArrayDataset(joint_pos),
                "obs/joint_pos_target": ArrayDataset(targets),
                "actions": ArrayDataset(np.zeros((3, 8), dtype=np.float32)),
                "obs/front": ArrayDataset(np.zeros((3, 4, 4, 3), dtype=np.uint8)),
                "obs/wrist": ArrayDataset(np.zeros((3, 4, 4, 3), dtype=np.uint8)),
                "states/articulation/robot/joint_position": ArrayDataset(
                    np.concatenate((joint_pos[1:], joint_pos[-1:]), axis=0)
                ),
            }
        )

    def test_target_is_shifted_clamped_and_final_source_frame_is_removed(self):
        demo = self.make_demo()
        lower, upper = np.full(6, -1.0), np.full(6, 1.0)
        actions = self.converter.get_demo_actions(
            demo, "recorded_target", (lower, upper)
        )
        np.testing.assert_allclose(actions[0], [-1.0, -1.0, -0.5, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(actions[1], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        self.assertEqual(len(actions), 2)
        self.assertEqual(
            self.converter.validate_demo(
                demo, 3.0, 4, "recorded_target", (lower, upper)
            ),
            2,
        )

    def test_limits_require_exact_order_finite_bounds_and_strict_intervals(self):
        invalid_payloads = (
            {"joint_names": list(reversed(self.names))},
            {"joint_lower_limits_rad": [-1.0] * 5},
            {"joint_upper_limits_rad": [1.0] * 5 + [float("nan")]},
            {"joint_lower_limits_rad": [-1.0] * 5 + [1.0]},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, override in enumerate(invalid_payloads):
                path = self.write_limits(root, **override)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    self.converter.load_joint_limits(path)

    def test_limits_and_alignment_are_recorded_in_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_limits(Path(temporary))
            lower, upper, limits_provenance = self.converter.load_joint_limits(path)
            np.testing.assert_array_equal(lower, [-1.0] * 6)
            np.testing.assert_array_equal(upper, [1.0] * 6)
            self.assertEqual(len(limits_provenance["sha256"]), 64)
            provenance = self.converter.build_provenance(
                Path("/tmp/source.hdf5"),
                "a" * 64,
                "recorded_target",
                4,
                ["demo_0"],
                [2],
                limits_provenance,
                [
                    self.converter.recorded_target_clipping_stats(
                        self.make_demo(), (lower, upper)
                    )
                ],
            )
            self.assertEqual(provenance["frame_count"], 2)
            self.assertEqual(
                provenance["target_alignment"],
                "action[t] = obs/joint_pos_target[t+1]",
            )
            self.assertTrue(provenance["last_source_frame_removed"])
            self.assertIn(
                "no recorded target", provenance["last_source_frame_removal_reason"]
            )
            self.assertEqual(
                provenance["joint_limits"]["sha256"], limits_provenance["sha256"]
            )
            clipping = provenance["episodes"][0]["clipping"]
            self.assertEqual(clipping["clipped_frames"], 1)
            self.assertEqual(clipping["clipped_values"], 2)
            self.assertEqual(clipping["max_abs_correction_rad"], 1.0)
            self.assertEqual(
                clipping["by_joint"]["shoulder_pan.pos"],
                {"clipped_values": 1, "max_abs_correction_rad": 1.0},
            )
            self.assertEqual(
                provenance["action_clamping"]["aggregate"], clipping
            )

    def test_recorded_target_requires_limits_and_rejects_bad_alignment(self):
        demo = self.make_demo()
        with self.assertRaisesRegex(ValueError, "requires validated joint limits"):
            self.converter.get_demo_actions(demo, "recorded_target")
        demo["states/articulation/robot/joint_position"] = ArrayDataset(
            np.zeros((3, 6), dtype=np.float32)
        )
        with self.assertRaisesRegex(ValueError, "alignment does not match"):
            self.converter.validate_demo(
                demo,
                3.0,
                4,
                "recorded_target",
                (np.full(6, -1.0), np.full(6, 1.0)),
            )

    def test_recorded_target_requires_post_step_state(self):
        demo = self.make_demo()
        del demo["states/articulation/robot/joint_position"]
        with self.assertRaisesRegex(ValueError, "missing required post-step"):
            self.converter.validate_demo(
                demo,
                3.0,
                4,
                "recorded_target",
                (np.full(6, -1.0), np.full(6, 1.0)),
            )


if __name__ == "__main__":
    unittest.main()
