"""CPU tests of temporal release success and annotation replay integration."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/leisaac/leisaac/tasks/pick_cube_into_box/mdp/release_state.py"
SPEC = importlib.util.spec_from_file_location("stable_release_test_target", SOURCE)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class StableReleaseTest(unittest.TestCase):
    def setUp(self):
        self.criteria = release.release_criteria_metadata()
        self.pos = torch.tensor([[0., 0., .02]])
        self.linear = torch.zeros(1, 3)
        self.angular = torch.zeros(1, 3)
        self.gripper = torch.tensor([.4])

    def candidate(self):
        return release.stable_release_candidate(self.pos, self.linear, self.angular, self.gripper, self.criteria)

    def test_flying_cube_is_not_success_candidate(self):
        self.linear[0, 2] = 2.0
        self.pos[0, 2] = .07
        self.assertFalse(self.candidate().item())

    def test_fast_rotation_closed_gripper_and_outside_are_rejected(self):
        self.angular[0, 0] = 1.
        self.assertFalse(self.candidate().item())
        self.angular.zero_()
        self.gripper[0] = .1
        self.assertFalse(self.candidate().item())
        self.gripper[0] = .4
        self.pos[0, 0] = .1
        self.assertFalse(self.candidate().item())

    def test_nonfinite_state_is_rejected(self):
        for tensor in (self.pos, self.linear, self.angular, self.gripper):
            original = tensor.clone()
            tensor.reshape(-1)[0] = float("nan")
            self.assertFalse(self.candidate().item())
            tensor.copy_(original)

    def test_requires_full_half_second_and_duplicate_queries_do_not_count(self):
        window = release.StableReleaseWindow(torch.zeros(1, dtype=torch.long))
        for step in range(1, 31):
            for _ in range(3):
                result = window.update(self.candidate(), torch.tensor([step]), step, 1 / 60, .5)
                self.assertEqual(result.item(), step == 30)

    def test_unstable_sample_breaks_hold_window(self):
        window = release.StableReleaseWindow(torch.zeros(1, dtype=torch.long))
        for step in range(1, 30):
            window.update(torch.tensor([True]), torch.tensor([step]), step, 1 / 60, .5)
        window.update(torch.tensor([False]), torch.tensor([30]), 30, 1 / 60, .5)
        self.assertFalse(window.update(torch.tensor([True]), torch.tensor([31]), 31, 1 / 60, .5).item())

    def test_partial_reset_does_not_reuse_previous_episode_hold(self):
        window = release.StableReleaseWindow(torch.zeros(2, dtype=torch.long))
        for step in range(1, 31):
            window.update(torch.tensor([True, True]), torch.tensor([step, step]), step, 1 / 60, .5)
        result = window.update(torch.tensor([True, True]), torch.tensor([1, 31]), 31, 1 / 60, .5)
        self.assertEqual(result.tolist(), [False, True])

    def test_reset_frame_and_missing_observations_do_not_accumulate(self):
        window = release.StableReleaseWindow(torch.zeros(1, dtype=torch.long))
        self.assertFalse(window.update(torch.tensor([True]), torch.tensor([0]), 0, .1, .5).item())
        window.update(torch.tensor([True]), torch.tensor([1]), 1, .1, .5)
        result = window.update(torch.tensor([True]), torch.tensor([20]), 20, .1, .5)
        self.assertFalse(result.item())
        self.assertEqual(window.count.item(), 1)

    def test_metadata_overrides_are_validated(self):
        self.assertEqual(release.release_criteria_metadata({"hold_time_s": 1.})["hold_time_s"], 1.)
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                release.release_criteria_metadata({"hold_time_s": value})
        with self.assertRaises(ValueError):
            release.release_criteria_metadata({"min_height_m": .2, "max_height_m": .1})

    def test_annotation_checks_every_control_step(self):
        source = ROOT / "scripts/mimic/annotate_demos.py"
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "replay_episode")
        namespace = {"torch": torch, "ManagerBasedRLMimicEnv": object, "EpisodeData": object,
                     "TerminationTermCfg": object, "skip_episode": False, "is_paused": False,
                     "task_type": "so101leader"}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        calls = []
        window = release.StableReleaseWindow(torch.zeros(1, dtype=torch.long))
        env = NS(seed=lambda _: None, sim=NS(reset=lambda: None), recorder_manager=NS(reset=lambda: None),
                 reset_to=lambda *a, **k: None, cfg=NS(dynamic_reset_gripper_effort_limit=False), counter=0)
        def step(_):
            env.counter += 1
        def success(env):
            calls.append(env.counter)
            return window.update(torch.tensor([True]), torch.tensor([env.counter]), env.counter, 1 / 60, .5)
        env.step = step
        episode = NS(seed=43, data={"initial_state": {}, "actions": torch.zeros(30, 6)})
        self.assertTrue(namespace["replay_episode"](env, episode, NS(func=success, params={})))
        self.assertEqual(calls, list(range(1, 31)) + [30])

    def test_manual_evaluators_observe_success_inside_control_loop(self):
        for name in ("lerobot_act_so101.py", "replay_joint_contract.py", "robomimic_so101.py"):
            source = ROOT / "scripts/evaluation" / name
            tree = ast.parse(source.read_text())
            steps = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "step"
                     and isinstance(node.func.value, ast.Name) and node.func.value.id == "env"]
            self.assertTrue(steps, name)
            for step in steps:
                loops = [node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))
                         and step in list(ast.walk(node))]
                control_loop = min(loops, key=lambda node: node.end_lineno - node.lineno)
                checks = [node for node in ast.walk(control_loop) if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Attribute) and node.func.attr == "func"
                          and isinstance(node.func.value, ast.Name) and node.func.value.id == "success_term"]
                self.assertTrue(any(node.lineno > step.lineno for node in checks), name)


if __name__ == "__main__":
    unittest.main()
