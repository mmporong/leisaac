"""CPU regression checks for pre-step recorder command alignment."""

import ast
from pathlib import Path
import unittest

import numpy as np


class CommandAlignmentTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "evaluation/replay_joint_contract.py"
        nodes = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == "aligned_commands"]
        namespace = {"np": np}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        self.align = namespace["aligned_commands"]
        self.q = np.arange(24, dtype=np.float32).reshape(4, 6)
        self.targets = self.q + 100
        self.actions = np.arange(32, dtype=np.float32).reshape(4, 8)
        self.post_states = self.q + 6

    def test_full_control_and_training_contract_preserve_last_transition(self):
        expected = {"next_observed": np.concatenate((self.q[1:], self.q[-1:])),
                    "recorded_target": self.targets[1:], "mimic_action": self.actions}
        for mode, commands in expected.items():
            actual, reference = self.align(self.q, self.targets, self.actions, self.post_states, mode)
            np.testing.assert_array_equal(actual, commands)
            np.testing.assert_array_equal(reference, self.post_states[:len(commands)])

    def test_invalid_length_and_mode_are_rejected(self):
        for q, target, actions, mode in (
            (self.q[:1], self.targets[:1], self.actions[:1], "next_observed"),
            (self.q, self.targets[:2], self.actions, "recorded_target"),
            (self.q, self.targets, self.actions[:2], "mimic_action"),
            (self.q, self.targets, self.actions, "invalid"),
        ):
            with self.assertRaises(ValueError):
                self.align(q, target, actions, self.post_states, mode)

    def test_nonfinite_commands_are_rejected(self):
        self.targets[1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.align(self.q, self.targets, self.actions, self.post_states, "recorded_target")

    def test_shifted_post_states_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "alignment"):
            self.align(self.q, self.targets, self.actions, self.post_states + 1, "mimic_action")


if __name__ == "__main__":
    unittest.main()
