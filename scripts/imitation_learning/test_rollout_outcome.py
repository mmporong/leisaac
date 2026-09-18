"""CPU-only tests for ACT rollout outcome classification."""

import ast
import hashlib
from pathlib import Path
import unittest
from types import SimpleNamespace

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


class ResetImageDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "evaluation/lerobot_act_so101.py"
        names = {"scene_state_sha256", "refresh_reset_images", "joint_clip_diagnostics"}
        functions = [node for node in ast.parse(source.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"hashlib": hashlib, "np": np, "torch": torch}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        self.refresh = namespace["refresh_reset_images"]
        self.clip = namespace["joint_clip_diagnostics"]
        self.state = torch.zeros(1, 6)
        self.calls = []

        class Scene(dict):
            def get_state(scene, is_relative):
                return {"robot": {"arm": {"joint_pos": self.state}}}

        self.scene = Scene()
        for name in ("front", "wrist"):
            self.scene[name] = SimpleNamespace(
                reset=lambda name=name: self.calls.append((name, "reset")),
                update=lambda dt, force_recompute, name=name: self.calls.append((name, dt, force_recompute)),
                data=SimpleNamespace(output={"rgb": torch.ones(1, 2, 2, 4, dtype=torch.uint8)}),
            )
        self.env = SimpleNamespace(scene=self.scene, sim=SimpleNamespace(render=lambda: self.calls.append("render")))
        self.obs = {"policy": {"front": torch.zeros(1, 2, 2, 3), "wrist": torch.zeros(1, 2, 2, 3),
                               "joint_pos": self.state}, "other": "preserved"}

    def test_legacy_does_not_render_or_replace_observations(self):
        self.assertIs(self.refresh(self.env, self.obs, 0), self.obs)
        self.assertEqual(self.calls, [])

    def test_refresh_replaces_only_rgb_after_render(self):
        result = self.refresh(self.env, self.obs, 4)
        self.assertEqual(self.calls[:4], ["render"] * 4)
        self.assertEqual(self.calls[4:], [("front", "reset"), ("front", 0.0, True),
                                         ("wrist", "reset"), ("wrist", 0.0, True)])
        self.assertEqual(result["policy"]["front"].shape, (1, 2, 2, 3))
        self.assertTrue(result["policy"]["front"].all())
        self.assertFalse(self.obs["policy"]["front"].any())
        self.assertIs(result["policy"]["joint_pos"], self.state)
        self.assertEqual(result["other"], "preserved")

    def test_physics_change_is_rejected(self):
        self.env.sim.render = lambda: self.state.add_(1)
        with self.assertRaisesRegex(RuntimeError, "physical state changed"):
            self.refresh(self.env, self.obs, 1)

    def test_negative_frames_are_rejected(self):
        with self.assertRaises(ValueError):
            self.refresh(self.env, self.obs, -1)

    def test_clipping_reports_each_joint(self):
        action = torch.tensor([[-2., 0., 3.]])
        clipped, mask, correction = self.clip(action, torch.full((3,), -1.), torch.ones(3))
        torch.testing.assert_close(clipped, torch.tensor([[-1., 0., 1.]]))
        torch.testing.assert_close(mask, torch.tensor([[True, False, True]]))
        torch.testing.assert_close(correction, torch.tensor([[1., 0., 2.]]))
        torch.testing.assert_close(action, torch.tensor([[-2., 0., 3.]]))

    def test_nonfinite_action_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                self.clip(torch.tensor([[value]]), torch.tensor([-1.]), torch.tensor([1.]))


if __name__ == "__main__":
    unittest.main()
