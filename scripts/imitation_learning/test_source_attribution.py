"""CPU-only tests for the source attribution and frame-transform contracts."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "evaluation/attribute_source_demo.py"
SPEC = importlib.util.spec_from_file_location("attribute_source_demo", SCRIPT)
attribute = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(attribute)


class SourceAttributionTest(unittest.TestCase):
    def test_first_close_ignores_stale_target_zero(self):
        targets = np.ones((5, 6), dtype=np.float32)
        targets[0, 5] = 0.2
        targets[3:, 5] = 0.4
        self.assertEqual(attribute.first_close_index(targets), 3)

    def test_close_one_is_a_real_degenerate_transition(self):
        targets = np.ones((4, 6), dtype=np.float32)
        targets[1:, 5] = 0.4
        self.assertEqual(attribute.first_close_index(targets), 1)

    def test_signal_margin_requires_injective_signatures(self):
        report = attribute.signal_margins({0: 10, 1: 14, 2: 30})
        self.assertEqual(report["minimum_margin"], 4)
        self.assertEqual(report["by_source"][0]["margin"], 4)
        with self.assertRaisesRegex(ValueError, "not injective"):
            attribute.signal_margins({0: 10, 1: 10})

    def test_strict_candidate_set_requires_exact_identity_equality(self):
        expected = {("shard_000.hdf5", "demo_1"), ("shard_001.hdf5", "demo_2")}
        attribute.require_exact_episode_set(expected, set(expected))
        with self.assertRaisesRegex(AssertionError, "missing=.*unexpected="):
            attribute.require_exact_episode_set(
                {("shard_000.hdf5", "demo_1")},
                expected,
            )

    def test_decision_gate_is_derived_from_measured_close_clipping(self):
        signals = {"length": {"minimum_margin": 16}, "close": {"minimum_margin": 4}}
        passing = attribute.decision_gate(
            signals,
            [{"close_clipped_frame_share": 0.0}, {"close_clipped_frame_share": 0.02}],
        )
        self.assertTrue(passing["proceed_to_2_2_when_other_prerequisites_pass"])
        failing = attribute.decision_gate(
            signals,
            [{"close_clipped_frame_share": 0.020001}],
        )
        self.assertFalse(failing["all_close_clipping_shares_lte_0_02"])
        self.assertFalse(failing["proceed_to_2_2_when_other_prerequisites_pass"])

    def test_three_term_object_frame_pose_uses_quaternion_conjugate(self):
        root_half = np.sqrt(0.5)
        world_object = np.array([1.0, 2.0, 0.0, root_half, 0.0, 0.0, root_half])
        world_robot = np.array([1.0, 3.0, 0.0, root_half, 0.0, 0.0, root_half])
        robot_eef = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        result = attribute.object_frame_eef_pose(world_robot, robot_eef, world_object)
        np.testing.assert_allclose(result[:3], [2.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(result[3:], [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_close_window_maps_thirty_lerobot_actions_to_raw_targets(self):
        close_index = 278
        raw_indices = list(range(close_index - 29, close_index + 1))
        self.assertEqual(len(raw_indices), 30)
        self.assertEqual(raw_indices[0], 249)
        self.assertEqual(raw_indices[-1], 278)


if __name__ == "__main__":
    unittest.main()
