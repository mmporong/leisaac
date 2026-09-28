"""Recorded-command replay is diagnostic, exact, bounded, and non-trainable."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.imitation_learning import test_action_intervention_runner as fixtures

EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))
import serve_recorded_actions as playback
SPEC = importlib.util.spec_from_file_location("contact_replay_test", EVALUATION / "run_contact_replay.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ContactReplayTests(unittest.TestCase):
    def test_replay_reset_and_horizon_never_extend_or_hold(self):
        commands = np.arange(12, dtype=np.float32).reshape(2, 6)
        recorded = playback.RecordedCommands(commands)
        self.assertEqual(recorded.next(), commands[0].tolist())
        self.assertEqual(recorded.next(), commands[1].tolist())
        with self.assertRaisesRegex(ValueError, "extending or holding"):
            recorded.next()
        recorded.reset()
        self.assertEqual(recorded.next(), commands[0].tolist())

    def test_replay_load_requires_kind_hash_and_contiguous_finite_six_joints(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trace_path, config_path = root / "trace.json", root / "config.json"
            trace = [{"step": 1, "applied_action": [0]*6}, {"step": 2, "applied_action": [1]*6}]
            trace_path.write_text(json.dumps(trace))
            config = {"kind": "historical_applied_action_replay", "trace": str(trace_path),
                      "trace_sha256": playback.sha256_file(trace_path), "horizon": 2}
            config_path.write_text(json.dumps(config))
            np.testing.assert_array_equal(playback.load_commands(config_path), np.array([[0]*6,[1]*6], np.float32))
            config["kind"] = "model"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "explicit diagnostic"):
                playback.load_commands(config_path)
            config["kind"] = "historical_applied_action_replay"
            for corrupt in ({"step": 3, "applied_action": [1]*6},
                            {"step": 2, "applied_action": [float("nan")]*6},
                            {"step": 2, "applied_action": [1]*5}):
                trace[1] = corrupt
                trace_path.write_text(json.dumps(trace))
                config["trace_sha256"] = playback.sha256_file(trace_path)
                config_path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    playback.load_commands(config_path)

    def fixture(self, root):
        case, preflight = fixtures.ActionInterventionRunnerTest().fixture(root, "teacher_arm")
        trace_path, evaluation_path = case / "trace_001.json", case / "evaluation.json"
        trace = json.loads(trace_path.read_text())
        for row in trace:
            row["policy_action"] = row["applied_action"][:]
            row.update(gripper_effort_limit=1, applied_action_source="teacher_arm")
        trace_path.write_text(json.dumps(trace))
        history_path, history_eval = root / "historical_trace.json", root / "historical_eval.json"
        history_path.write_text(json.dumps(trace)); history_eval.write_text("{}")
        source = {"trace": str(history_path), "trace_sha256": runner.sha256_file(history_path),
                  "evaluation": str(history_eval), "evaluation_sha256": runner.sha256_file(history_eval),
                  "horizon": 3, "source_was_successful": True}
        parent_path = root / "parent_manifest.json"
        parent_path.write_text(json.dumps({"runs": [{"audit": {"audit_pass": True, "condition": "teacher_arm",
            "seed": 4101, "steps": 3, "success": True, "trace": str(history_path),
            "trace_sha256": source["trace_sha256"], "evaluation": str(history_eval),
            "evaluation_sha256": source["evaluation_sha256"]}}]}))
        source.update(parent_manifest=str(parent_path), parent_manifest_sha256=runner.sha256_file(parent_path),
                      source_seed=4101, source_condition="teacher_arm")
        config_path = root / "replay_source.json"
        config_path.write_text(json.dumps(source))
        evaluation = json.loads(evaluation_path.read_text())
        evaluation.update(checkpoint=str(config_path), horizon=3, trace_steps=3,
                          diagnostic_teacher_arm_until_step=None, gripper_effort_mode="fixed", gripper_effort_limit=1)
        evaluation_path.write_text(json.dumps(evaluation))
        return case, source, config_path, preflight

    def test_coherently_changed_historical_commands_cannot_be_reapproved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, source, _, _ = self.fixture(root)
            pin = {**source, "condition": "teacher_arm", "seed": 4101}
            original = runner.load_pinned_source(pin)
            self.assertEqual(original["trace_sha256"], source["trace_sha256"])
            trace_path = Path(source["trace"])
            trace = json.loads(trace_path.read_text()); trace[0]["applied_action"][5] += .001
            trace_path.write_text(json.dumps(trace))
            with self.assertRaisesRegex(ValueError, "pinned parent audit"):
                runner.load_pinned_source(pin)
            parent_path = Path(source["parent_manifest"])
            parent = json.loads(parent_path.read_text())
            parent["runs"][0]["audit"]["trace_sha256"] = runner.sha256_file(trace_path)
            parent_path.write_text(json.dumps(parent))
            with self.assertRaisesRegex(ValueError, "parent manifest hash"):
                runner.load_pinned_source(pin)

    def audit(self, case, source, config, preflight):
        original = runner.sha256_file
        def digest(path):
            if str(path)=="/raw": return "raw"
            if str(path).endswith("model.safetensors"): return "model"
            return original(path)
        with patch.object(runner, "sha256_file", side_effect=digest), \
                patch.object(runner, "audit_images_and_video", return_value={}):
            return runner.audit_replay(case, source, 4101, config, preflight)

    def test_exact_replay_is_non_autonomous_and_gripper_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            case, source, config, preflight = self.fixture(Path(folder))
            result = self.audit(case, source, config, preflight)
            self.assertTrue(result["audit_pass"])
            self.assertFalse(result["model_loaded"])
            trace_path = case / "trace_001.json"
            trace = json.loads(trace_path.read_text()); trace[1]["applied_action"][5] += .001
            trace_path.write_text(json.dumps(trace))
            with self.assertRaisesRegex(ValueError, "historical six-joint"):
                self.audit(case, source, config, preflight)

    def test_wrong_scene_effort_and_autonomy_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            case, source, config, preflight = self.fixture(Path(folder))
            path = case / "evaluation.json"
            original = json.loads(path.read_text())
            original["autonomous_policy_evaluation"] = True
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "evaluation contract"):
                self.audit(case, source, config, preflight)
            original["autonomous_policy_evaluation"] = False
            original["results"][0]["gripper_effort_limit_range"] = [1,2]
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "scene, effort"):
                self.audit(case, source, config, preflight)


if __name__ == "__main__":
    unittest.main()
