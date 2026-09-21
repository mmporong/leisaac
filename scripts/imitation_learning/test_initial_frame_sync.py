"""CPU-only tests for the camera/joint synchronization diagnostics."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "evaluation"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DuplicatePatternTest(unittest.TestCase):
    def test_30hz_camera_pattern_is_even_pairs_only(self):
        inspect = load_script("inspect_initial_frames")
        rng = np.random.default_rng(0)
        distinct = rng.integers(0, 255, size=(5, 4, 4, 3), dtype=np.uint8)
        # frames (0,1) (2,3) (4,5) (6,7) repeat, frame 8 alone: 9 frames
        images = np.repeat(distinct[:4], 2, axis=0)
        images = np.concatenate([images, distinct[4:5]])
        pattern = inspect.duplicate_pattern(images)
        self.assertEqual(pattern["consecutive_pairs"], 8)
        self.assertEqual(pattern["even_start_duplicates"], pattern["even_start_pairs"])
        self.assertEqual(pattern["odd_start_duplicates"], 0)
        self.assertIsNone(pattern["first_non_duplicate_even_pair"])

    def test_60hz_camera_has_no_duplicates(self):
        inspect = load_script("inspect_initial_frames")
        images = np.arange(6 * 4 * 4 * 3, dtype=np.uint8).reshape(6, 4, 4, 3)
        pattern = inspect.duplicate_pattern(images)
        self.assertEqual(pattern["duplicate_pairs"], 0)
        self.assertEqual(pattern["first_non_duplicate_even_pair"], 0)


class CompareFirstApproachTest(unittest.TestCase):
    def test_trace_aligned_to_target_t_plus_one_gives_zero_error(self):
        compare = load_script("compare_first_approach")
        frames = 12
        target = np.linspace(0.0, 1.0, frames * 6, dtype=np.float32).reshape(frames, 6)
        joint_pos = target - 0.01
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.hdf5"
            with h5py.File(raw, "w") as f:
                g = f.create_group("data/demo_0")
                g["obs/joint_pos_target"] = target
                g["obs/joint_pos"] = joint_pos
                cube = np.zeros((frames, 7), dtype=np.float32)
                cube[:, 2] = 0.06
                g["states/rigid_object/cube/root_pose"] = cube
                g["initial_state/rigid_object/cube/root_pose"] = cube[:1]
            trace = [
                {
                    "step": s,
                    "state_before": joint_pos[s - 1].tolist(),
                    "requested_action": target[s].tolist(),
                    "applied_action": target[s].tolist(),
                    "state_after": joint_pos[s].tolist(),
                    "cube_xyz": [0.0, 0.0, 0.06],
                }
                for s in range(1, frames)
            ]
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(trace), encoding="utf-8")
            output = Path(tmp) / "report.json"
            argv = sys.argv
            sys.argv = [
                "compare_first_approach.py", "--trace", str(trace_path), "--raw-path", str(raw),
                "--demo", "demo_0", "--output", str(output), "--windows", "1", "5", "11",
            ]
            try:
                compare.main()
            finally:
                sys.argv = argv
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["compared_steps"], frames - 1)
        for window in ("1", "5", "11"):
            self.assertAlmostEqual(report["windows"][window]["command_rmse_all_rad"], 0.0)
            self.assertAlmostEqual(report["windows"][window]["state_rmse_all_rad"], 0.0)
        self.assertEqual(report["initial_state_match"]["max_joint_abs_diff_rad"], 0.0)


if __name__ == "__main__":
    unittest.main()
