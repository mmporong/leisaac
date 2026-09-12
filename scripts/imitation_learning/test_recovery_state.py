"""CPU regression checks for recovery snapshots without starting Isaac Sim."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


def load_helpers():
    # The executable imports AppLauncher at module scope. Extract only the
    # pure helpers so this test does not launch a GPU simulator.
    source = Path(__file__).resolve().parents[1] / "datagen/state_machine/generate.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"_move_tensors", "_restore_position_control_state", "_on_episode_done", "_finalize_pending_episode"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class RecoveryStateTest(unittest.TestCase):
    def test_fixed_wrist_holds_open_before_grasp_without_changing_legacy_timing(self):
        source = Path(__file__).resolve().parents[2] / "source/leisaac/leisaac/datagen/state_machine/pick_cube_into_box.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        namespace = {"torch": torch, "math": math, "StateMachineBase": object, "_GRASP_OFFSET": (-0.012, 0.020, 0.090)}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
        machine_class = namespace[cls.name]
        legacy = machine_class()
        fixed = machine_class(grasp_alignment="fixed_wrist")
        self.assertEqual(legacy._max_steps, 810)
        self.assertEqual(fixed._max_steps, 930)
        for step in (210, 329):
            fixed._step_count = step
            self.assertEqual(fixed._phase()[1], "align_hold")
        fixed._step_count = 330
        self.assertEqual(fixed._phase()[1], "grasp")
        with self.assertRaises(ValueError):
            machine_class(grasp_alignment="fixed_wrist", grasp_offset=(0, 0, float("nan")))

    def test_episode_export_is_owned_by_reset_or_final_flush_never_both(self):
        events = []
        recorder = SimpleNamespace(exported_successful_episode_count=0, exported_failed_episode_count=0, success=False)

        def export(_):
            events.append("export")
            recorder.exported_successful_episode_count += int(recorder.success)
            recorder.exported_failed_episode_count += int(not recorder.success)

        recorder.record_pre_reset = export
        env = SimpleNamespace(recorder_manager=recorder)
        args = SimpleNamespace(record=True, num_demos=1)
        helpers = load_helpers()
        def set_success(env, success):
            recorder.success = success
            events.append(("success", success))

        helpers["auto_terminate"] = set_success
        result = helpers["_on_episode_done"](env, None, args, 0, 0, True, success=True)
        self.assertEqual(events, [("success", True)])
        self.assertTrue(result[2])
        helpers["_finalize_pending_episode"](env, True)
        self.assertEqual(events, [("success", True), "export"])
        self.assertEqual(recorder.exported_successful_episode_count, 1)
        self.assertEqual(recorder.exported_failed_episode_count, 0)
        # In a continuing loop reset_to owns the export, so final flush skips.
        events.clear()
        recorder.exported_successful_episode_count = 0
        recorder.record_pre_reset(None)
        helpers["_finalize_pending_episode"](env, False)
        self.assertEqual(events, ["export"])
        self.assertEqual(recorder.exported_successful_episode_count, 1)

    def test_measured_velocity_is_preserved_but_pd_target_is_cleared(self):
        velocity = torch.tensor([[0.3, -0.7]])
        joint_position = torch.tensor([[0.2, 1.1]])
        arm = SimpleNamespace(data=SimpleNamespace(joint_vel=None))
        arm.set_joint_velocity_target = lambda target: setattr(arm, "velocity_target", target.clone())
        calls = []

        def reset_to(state, env_ids, **kwargs):
            calls.append((state, env_ids, kwargs))
            arm.data.joint_vel = state["joint_velocity"].clone()
            arm.velocity_target = arm.data.joint_vel.clone()

        env = SimpleNamespace(
            device="cpu", reset_to=reset_to,
            cfg=SimpleNamespace(seed=999),
            scene=SimpleNamespace(articulations={"robot": arm}),
        )
        entry = {"seed": 3000, "state": {"joint_velocity": velocity, "joint_position": joint_position}}
        load_helpers()["_restore_position_control_state"](env, entry)
        torch.testing.assert_close(arm.data.joint_vel, velocity)
        torch.testing.assert_close(arm.velocity_target, torch.zeros_like(velocity))
        torch.testing.assert_close(calls[0][0]["joint_position"], joint_position)
        self.assertEqual(calls[0][2], {"seed": 3000, "is_relative": True})
        self.assertEqual(env.cfg.seed, 3000)


if __name__ == "__main__":
    unittest.main()
