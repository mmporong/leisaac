"""CPU fake-fixture tests for the action intervention runner."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "evaluation/run_action_intervention.py"
SPEC = importlib.util.spec_from_file_location("run_action_intervention", SCRIPT)
runner = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(runner)


class ActionInterventionRunnerTest(unittest.TestCase):
    def fixture(self, root, condition, *, corrupt=None, success=True):
        case = root / condition; case.mkdir()
        lower, upper = np.full(6, -2, np.float32), np.full(6, 2, np.float32)
        targets = np.arange(676 * 6, dtype=np.float32).reshape(676, 6) / 10000
        policy = np.full(6, 0.25, np.float32); trace = []
        teacher_idx = set(range(6)) if condition == "teacher_all" else ({5} if condition == "teacher_gripper" else (set(range(5)) if condition == "teacher_arm" else set()))
        for step in range(1, 4):
            requested = policy.copy(); requested[list(teacher_idx)] = targets[step, list(teacher_idx)]
            applied = np.clip(requested, lower, upper)
            row = {"step": step, "policy_action": policy.tolist(),
                   "requested_action": requested.tolist(), "applied_action": applied.tolist()}
            if condition != "policy": row["teacher_action"] = targets[step].tolist()
            trace.append(row)
        if corrupt == "teacher": trace[0]["requested_action"][next(iter(teacher_idx))] += .1
        if corrupt == "clip": trace[0]["applied_action"][0] += .1
        evaluation = {"task": runner.TASK, "checkpoint": "/model", "num_rollouts": 1,
                      "seed_start": 4101, "horizon": 675, "trace_steps": 675,
                      "n_action_steps": 30, "reset_render_frames": 4, "gripper_effort_mode": "task",
                      "server_seed": 0, "server_device": "cpu", "render_width": 640,
                      "render_height": 480, "policy_image_size": 224, "lift_threshold_m": .02,
                      "autonomous_policy_evaluation": condition == "policy",
                      "diagnostic_action_source": condition, "control_dt_s": 1/60, "success_criteria": {"version": "stable_release_v2"},
                      "joint_names": [str(i) for i in range(6)], "joint_lower_limits_rad": lower.tolist(),
                      "joint_upper_limits_rad": upper.tolist(), "initial_state_source": {"hdf5": "/raw", "demo": "demo_9"},
                      "teacher_command_source": None if condition == "policy" else {
                          "path":"/raw", "demo":"demo_9", "sha256":"raw", "source_frames":676,
                          "command_alignment":"loop t -> raw joint_pos_target[t+1]", "horizon":675,
                          "training_eligibility":"diagnostic_only"},
                      "results": [{"seed": 4101, "steps": 3, "success": success,
                                   "first_success_step": 3 if success else None,
                                   "outcome": "success" if success else "no_lift",
                                   "initial_scene_state_sha256": runner.EXPECTED_SCENE, "gripper_effort_limit_range": [1,1],
                                   "clipped_steps": 0, "max_clip_correction_rad": {str(i): 0 for i in range(6)},
                                   "max_cube_lift_m": .1 if success else 0, "first_lift_step": 2 if success else None,
                                   "initial_observation_sha256": {"front":"a","wrist":"b"}}]}
        (case/"evaluation.json").write_text(json.dumps(evaluation)); (case/"trace_001.json").write_text(json.dumps(trace))
        (case/f"rollout_001_{'success' if success else 'failure'}.mp4").write_bytes(b"video")
        preflight = {"model":"/model", "raw_path":"/raw", "demo":"demo_9", "scene_sha256":runner.EXPECTED_SCENE,
                     "control_dt_s":1/60, "success_criteria":{"version":"stable_release_v2"}, "joint_names":[str(i) for i in range(6)],
                     "lower":lower.tolist(), "upper":upper.tolist(), "effort":[1,1], "targets":targets,
                     "raw_sha256":"raw", "model_sha256":"model",
                     "implementation": {
                         "runner": {"path": str(SCRIPT), "sha256": runner.sha256_file(SCRIPT)},
                         "helper": {"path": str(SCRIPT), "sha256": runner.sha256_file(SCRIPT)}}}
        return case, preflight

    def audit(self, case, condition, preflight):
        old = runner.sha256_file
        runner.sha256_file = lambda path: "model" if str(path).endswith("model.safetensors") else ("raw" if str(path)=="/raw" else old(path))
        try: return runner.audit_case(case, condition, 4101, preflight)
        finally: runner.sha256_file = old

    def test_all_conditions_substitute_exact_channels_and_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            for condition in ("policy", "teacher_gripper", "teacher_arm", "teacher_all"):
                case, preflight = self.fixture(Path(folder), condition)
                self.assertTrue(self.audit(case, condition, preflight)["audit_pass"])

    def test_corrupt_teacher_or_applied_channel_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for index, corrupt in enumerate(("teacher", "clip")):
                root=Path(folder)/str(index); root.mkdir(); case,preflight=self.fixture(root,"teacher_all",corrupt=corrupt)
                with self.assertRaisesRegex(ValueError, "audit failed"): self.audit(case,"teacher_all",preflight)

    def test_missing_or_corrupt_teacher_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            case, preflight = self.fixture(Path(folder), "teacher_all")
            trace_path = case / "trace_001.json"
            trace = json.loads(trace_path.read_text()); trace[0]["teacher_action"][0] += .1
            trace_path.write_text(json.dumps(trace))
            with self.assertRaisesRegex(ValueError, "audit failed"):
                self.audit(case, "teacher_all", preflight)

    def test_teacher_success_and_audit_gate_followups(self):
        self.assertEqual(runner.allowed_followups({"audit_pass":True,"success":True}), runner.FOLLOWUPS)
        self.assertEqual(runner.allowed_followups({"audit_pass":True,"success":False}), ())
        self.assertEqual(runner.allowed_followups({"audit_pass":False,"success":True}), ())

    def test_gpu_occupancy_and_output_overlap_are_refused(self):
        self.assertEqual(runner.parse_compute_processes("\n"), [])
        self.assertEqual(len(runner.parse_compute_processes("123, python, 2 MiB\n")), 1)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError): runner.prepare_output_root(Path(folder))


if __name__ == "__main__": unittest.main()
