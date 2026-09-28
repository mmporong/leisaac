"""Diagnostic command mixing must not change the ordinary policy path."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from action_intervention import (action_source_at_step, load_teacher_commands, substitute_action,
                                 validate_arm_prefix, validate_fixed_effort)


class ActionInterventionTests(unittest.TestCase):
    def test_arm_prefix_boundary_is_inclusive_and_preserves_gripper(self):
        policy = torch.arange(6).float().view(1, 6)
        teacher = policy + 10
        for step in (1, 239, 240):
            result = substitute_action(policy, teacher, "teacher_arm", completed_step=step, until_step=240)
            torch.testing.assert_close(result, torch.tensor([[10, 11, 12, 13, 14, 5]], dtype=policy.dtype))
        self.assertIs(substitute_action(policy, teacher, "teacher_arm", completed_step=241, until_step=240), policy)
        self.assertEqual(action_source_at_step("teacher_arm", 240, 240), "teacher_arm")
        self.assertEqual(action_source_at_step("teacher_arm", 241, 240), "policy")
        torch.testing.assert_close(policy, torch.arange(6).float().view(1, 6))

    def test_invalid_arm_prefix_is_rejected_and_defaults_unchanged(self):
        for source in ("policy", "teacher_all", "teacher_gripper"):
            self.assertIsNone(validate_arm_prefix(source, None, 675))
            with self.assertRaises(ValueError):
                validate_arm_prefix(source, 240, 675)
        for cutoff in (0, -1, 675, True, 1.5):
            with self.assertRaises(ValueError):
                validate_arm_prefix("teacher_arm", cutoff, 675)
        for step in (0, None, True, 1.5):
            with self.assertRaises(ValueError):
                action_source_at_step("teacher_arm", step, 240)

    def test_explicit_effort_is_opt_in_positive_finite_and_fixed_only(self):
        self.assertIsNone(validate_fixed_effort("task", None))
        self.assertIsNone(validate_fixed_effort("fixed", None))
        self.assertEqual(validate_fixed_effort("fixed", 0.06666666269302368), 0.06666666269302368)
        for mode, limit in (("task", .1), ("fixed", 0), ("fixed", -.1),
                            ("fixed", float("nan")), ("fixed", float("inf"))):
            with self.assertRaises(ValueError):
                validate_fixed_effort(mode, limit)

    def test_channel_substitution_and_default_identity(self):
        policy = torch.arange(6).float().view(1, 6)
        teacher = policy + 10
        self.assertIs(substitute_action(policy, None, "policy"), policy)
        expected = {"teacher_arm": [10, 11, 12, 13, 14, 5],
                    "teacher_gripper": [0, 1, 2, 3, 4, 15],
                    "teacher_all": [10, 11, 12, 13, 14, 15]}
        for mode, values in expected.items():
            result = substitute_action(policy, teacher, mode)
            torch.testing.assert_close(result, torch.tensor([values], dtype=policy.dtype))
            self.assertEqual(result.device, policy.device)
        torch.testing.assert_close(policy, torch.arange(6).float().view(1, 6))
        with self.assertRaises(ValueError):
            substitute_action(policy, teacher, "unknown")

    def test_recorded_targets_start_at_t_plus_one_without_extension(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.hdf5"
            targets = np.arange(60).reshape(10, 6).astype(np.float32)
            pre = targets.copy()
            post = np.concatenate([pre[1:], pre[-1:]])
            with h5py.File(path, "w") as file:
                group = file.create_group("data/demo_9")
                group.attrs["success"] = True
                group.create_dataset("obs/joint_pos_target", data=targets)
                group.create_dataset("obs/joint_pos", data=pre)
                group.create_dataset("states/articulation/robot/joint_position", data=post)
            commands, metadata = load_teacher_commands(path, "demo_9", 9)
            np.testing.assert_array_equal(commands, targets[1:])
            self.assertEqual(metadata["source_frames"], 10)
            self.assertEqual(metadata["training_eligibility"], "diagnostic_only")
            for horizon in [0, 10]:
                with self.assertRaises(ValueError):
                    load_teacher_commands(path, "demo_9", horizon)
            with h5py.File(path, "r+") as file:
                file["data/demo_9"].attrs["success"] = False
            with self.assertRaisesRegex(ValueError, "successful"):
                load_teacher_commands(path, "demo_9", 9)

    def test_evaluator_substitutes_after_policy_and_before_existing_clip(self):
        path = Path(__file__).resolve().parents[1] / "evaluation/lerobot_act_so101.py"
        tree = ast.parse(path.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
        mix = next(n for n in calls if ast.unparse(n.func) == "substitute_action")
        clip = next(n for n in calls if ast.unparse(n.func) == "joint_clip_diagnostics")
        physics = next(n for n in calls if ast.unparse(n.func) == "env.step")
        predict = max((n for n in calls if ast.unparse(n.func) == "send_request" and n.lineno < mix.lineno),
                      key=lambda n: n.lineno)
        self.assertLess(predict.lineno, mix.lineno)
        self.assertLess(mix.lineno, clip.lineno)
        self.assertLess(clip.lineno, physics.lineno)
        self.assertEqual(ast.unparse(mix.args[2]), "args_cli.diagnostic_action_source")
        self.assertIn("teacher_commands[step]", ast.unparse(main))


if __name__ == "__main__":
    unittest.main()
