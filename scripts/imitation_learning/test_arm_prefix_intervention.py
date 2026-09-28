"""Prefix runner contracts: exact handoff, input dumps, and safe continuation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.imitation_learning import test_action_intervention_runner as fixtures

EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))
SPEC = importlib.util.spec_from_file_location("arm_prefix_test", EVALUATION / "run_arm_prefix_intervention.py")
prefix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prefix)
from policy_image_dump import write_policy_images


class FakeVideo:
    def __init__(self, steps=211):
        self.values = {prefix.cv2.CAP_PROP_FRAME_COUNT: steps, prefix.cv2.CAP_PROP_FRAME_WIDTH: 1280,
                       prefix.cv2.CAP_PROP_FRAME_HEIGHT: 480, prefix.cv2.CAP_PROP_FPS: 60}

    def get(self, key):
        return self.values[key]

    def read(self):
        return True, None

    def release(self):
        pass


class ArmPrefixRunnerTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(isaac_python=Path("/isaac"), evaluator=Path("/evaluator"),
                               checkpoint=Path("/model"), raw_path=Path("/raw"), demo="demo_9",
                               server_python=Path("/server"), gripper_effort_mode="fixed",
                               gripper_effort_limit=.1)

    def test_command_cutoff_is_opt_in_and_queue_configuration_is_unchanged(self):
        bounded = prefix.build_case_command(self.args(), "teacher_arm", 240, 4101, Path("/output"))
        unlimited = prefix.build_case_command(self.args(), "teacher_arm", None, 4101, Path("/output"))
        self.assertEqual(bounded[-2:], ["--diagnostic-teacher-arm-until-step", "240"])
        self.assertNotIn("--diagnostic-teacher-arm-until-step", unlimited)
        self.assertEqual(bounded[bounded.index("--n-action-steps") + 1], "30")
        self.assertEqual(float(bounded[bounded.index("--gripper-effort-limit") + 1]), .1)

    def image_fixture(self, root):
        state = np.arange(6, dtype=np.float32)
        pixels = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
        row = write_policy_images(root, 1, 211, state, {"front": pixels, "wrist": pixels.copy()})
        (root / "policy_images_001.json").write_text(json.dumps([row]))
        (root / "trace_001.json").write_text(json.dumps([{"state_before": state.tolist()}] * 211))
        (root / "evaluation.json").write_text(json.dumps({"dump_policy_images_every": 30,
                                                         "dump_policy_images_range": [211, 451]}))
        return {"steps": 211, "video": "unused"}, row

    def image_audit(self, root, audit):
        with patch.object(prefix.cv2, "VideoCapture", return_value=FakeVideo(audit["steps"])):
            return prefix.audit_images_and_video(root, audit)

    def test_lossless_input_state_and_video_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audit, _ = self.image_fixture(root)
            self.assertEqual(self.image_audit(root, audit)["input_dump_frames"], 1)

    def test_missing_image_step_and_misaligned_state_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audit, row = self.image_fixture(root)
            path = root / "policy_images_001.json"
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "steps changed"):
                self.image_audit(root, audit)
            row["state_before"][0] += 1
            path.write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, "pre-action"):
                self.image_audit(root, audit)

    def test_corrupt_image_file_and_pixel_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audit, row = self.image_fixture(root)
            row["front"]["pixel_sha256"] = "changed"
            (root / "policy_images_001.json").write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, "RGB input"):
                self.image_audit(root, audit)
            (root / row["front"]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "file hash"):
                self.image_audit(root, audit)

    def test_video_frame_count_is_not_a_success_placeholder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audit, _ = self.image_fixture(root)
            with patch.object(prefix.cv2, "VideoCapture", return_value=FakeVideo(210)):
                with self.assertRaisesRegex(ValueError, "video metadata"):
                    prefix.audit_images_and_video(root, audit)

    def test_prefix_handoff_statistics_require_a_real_policy_step(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = fixtures.ActionInterventionRunnerTest()
            case, preflight = fixture.fixture(root, "teacher_arm")
            trace_path = case / "trace_001.json"
            trace = json.loads(trace_path.read_text())
            for row in trace:
                row.update(state_before=[0] * 6, cube_xyz=[0] * 3)
            trace_path.write_text(json.dumps(trace))
            base_audit = {"steps": 3, "clipped_steps": 0}
            with patch.object(prefix, "audit_case", return_value=base_audit.copy()), \
                    patch.object(prefix, "audit_images_and_video", return_value={}):
                result = prefix.audit_prefix_case(case, "teacher_arm_until_2", "teacher_arm", 2, 4101, preflight)
                self.assertEqual(result["handoff"]["first_policy_step"], 3)
                self.assertTrue(np.isfinite(result["handoff"]["policy_teacher_arm_error_l2_rad"]))
                with self.assertRaisesRegex(ValueError, "execute the policy handoff"):
                    prefix.audit_prefix_case(case, "teacher_arm_until_3", "teacher_arm", 3, 4101, preflight)

    def saved_fixture(self, root, success=True):
        args = self.args()
        output = root / "seed_4101/teacher_all"
        output.mkdir(parents=True)
        (root / "logs").mkdir()
        log = root / "logs/seed_4101_teacher_all.log"
        log.write_text("completed")
        command = prefix.build_case_command(args, "teacher_all", None, 4101, output)
        audit = {"seed": 4101, "case": "teacher_all", "success": success}
        manifest = {"runs": [{"command": command,
            "command_sha256": hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest(),
            "log": str(log), "log_sha256": prefix.sha256_file(log), "audit": audit}]}
        return args, manifest, audit

    def test_saved_prefix_is_read_only_and_changed_log_or_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args, manifest, audit = self.saved_fixture(root)
            before = {p: prefix.sha256_file(p) for p in root.rglob("*") if p.is_file()}
            with patch.object(prefix, "audit_prefix_case", return_value=audit):
                prefix.validate_saved_runs(root, manifest, args, {})
                self.assertEqual(before, {p: prefix.sha256_file(p) for p in before})
                manifest["runs"][0]["command_sha256"] = "changed"
                with self.assertRaisesRegex(ValueError, "command or log"):
                    prefix.validate_saved_runs(root, manifest, args, {})
                manifest["runs"][0]["command_sha256"] = hashlib.sha256(
                    json.dumps(manifest["runs"][0]["command"], separators=(",", ":")).encode()).hexdigest()
                Path(manifest["runs"][0]["log"]).write_text("changed")
                with self.assertRaisesRegex(ValueError, "command or log"):
                    prefix.validate_saved_runs(root, manifest, args, {})
                manifest["runs"][0]["audit"]["case"] = "policy"
                with self.assertRaisesRegex(ValueError, "preregistered prefix"):
                    prefix.validate_saved_runs(root, manifest, args, {})

    def test_failed_positive_control_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args, manifest, audit = self.saved_fixture(root, success=False)
            with patch.object(prefix, "audit_prefix_case", return_value=audit):
                with self.assertRaisesRegex(ValueError, "positive control failed"):
                    prefix.validate_saved_runs(root, manifest, args, {})


if __name__ == "__main__":
    unittest.main()
