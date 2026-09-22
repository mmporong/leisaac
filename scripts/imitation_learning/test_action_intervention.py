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
from action_intervention import load_teacher_commands, substitute_action


class ActionInterventionTests(unittest.TestCase):
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
