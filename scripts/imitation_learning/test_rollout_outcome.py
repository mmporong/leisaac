"""CPU-only tests for ACT rollout outcome classification."""

import ast
import hashlib
from pathlib import Path
import unittest

import numpy as np
import torch


def load_classify_rollout_outcome():
    source = Path(__file__).resolve().parents[1] / "evaluation/lerobot_act_so101.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "classify_rollout_outcome"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["classify_rollout_outcome"]


class RolloutOutcomeTest(unittest.TestCase):
    def setUp(self):
        self.classify = load_classify_rollout_outcome()
        self.threshold = 0.02

    def test_success_has_priority_over_lift_observations(self):
        self.assertEqual(self.classify(True, 0.0, 0.0, self.threshold), "success")

    def test_no_lift_when_maximum_is_below_threshold(self):
        self.assertEqual(self.classify(False, 0.019, 0.019, self.threshold), "no_lift")

    def test_threshold_maximum_counts_as_lift_then_low(self):
        self.assertEqual(self.classify(False, self.threshold, 0.019, self.threshold), "low_after_lift")

    def test_final_at_threshold_is_lifted_not_completed(self):
        self.assertEqual(
            self.classify(False, self.threshold, self.threshold, self.threshold),
            "lifted_not_completed",
        )


class SceneStateHashTest(unittest.TestCase):
    def test_order_independent_and_detects_velocity_change(self):
        source = Path(__file__).resolve().parents[1] / "evaluation/lerobot_act_so101.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "scene_state_sha256"
        )
        namespace = {"hashlib": hashlib, "np": np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        hash_state = namespace["scene_state_sha256"]
        fields = {"pose": torch.tensor([[0.0, 1.0]]), "velocity": torch.tensor([[0.0, 0.0]])}
        state = {"rigid_object": {"cube": fields}}
        reordered = {"rigid_object": {"cube": dict(reversed(list(fields.items())))}}
        self.assertEqual(hash_state(state), hash_state(reordered))
        changed = {"rigid_object": {"cube": {**fields, "velocity": torch.tensor([[0.0, 1.0]])}}}
        self.assertNotEqual(hash_state(state), hash_state(changed))


if __name__ == "__main__":
    unittest.main()
