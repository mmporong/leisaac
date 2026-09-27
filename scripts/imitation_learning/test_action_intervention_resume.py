"""Continuation must reject changed evidence and non-prefix case histories."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.imitation_learning import test_action_intervention_runner as fixtures

EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))
SPEC = importlib.util.spec_from_file_location("resume_intervention_test", EVALUATION / "resume_action_intervention.py")
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


class ActionInterventionResumeTest(unittest.TestCase):
    def fixture(self, root, *, success=True):
        fixture = fixtures.ActionInterventionRunnerTest()
        seed_root = root / "seed_4101"
        seed_root.mkdir()
        (root / "logs").mkdir()
        case, preflight = fixture.fixture(seed_root, "teacher_all", success=success)
        args = SimpleNamespace(isaac_python=Path("/isaac"), evaluator=Path("/evaluator"),
                               checkpoint=Path("/model"), raw_path=Path("/raw"), demo="demo_9",
                               server_python=Path("/server"), gripper_effort_mode="task")
        command = resume.build_command(args, "teacher_all", 4101, case)
        log = root / "logs/seed_4101_teacher_all.log"
        log.write_text("completed")
        audit = fixture.audit(case, "teacher_all", preflight)
        run = {"command": command,
               "command_sha256": hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest(),
               "log": str(log), "log_sha256": resume.sha256_file(log), "audit": audit}
        return {"runs": [run]}, preflight, args, fixture

    def validate(self, root, manifest, preflight, args, fixture):
        with patch.object(resume, "audit_case", side_effect=lambda case, condition, seed, pre: fixture.audit(case, condition, pre)):
            resume.validate_saved_runs(root, manifest, preflight, args)

    def test_completed_prefix_is_audited_without_changing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, preflight, args, fixture = self.fixture(root)
            before = {p: resume.sha256_file(p) for p in root.rglob("*") if p.is_file()}
            self.validate(root, manifest, preflight, args, fixture)
            self.assertEqual(before, {p: resume.sha256_file(p) for p in before})

    def test_duplicate_or_missing_prefix_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, preflight, args, fixture = self.fixture(root)
            manifest["runs"].append(manifest["runs"][0])
            with self.assertRaisesRegex(ValueError, "preregistered prefix"):
                self.validate(root, manifest, preflight, args, fixture)

    def test_changed_command_or_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, preflight, args, fixture = self.fixture(root)
            original_hash = manifest["runs"][0]["command_sha256"]
            manifest["runs"][0]["command_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "saved command"):
                self.validate(root, manifest, preflight, args, fixture)
            manifest["runs"][0]["command_sha256"] = original_hash
            Path(manifest["runs"][0]["log"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "saved log"):
                self.validate(root, manifest, preflight, args, fixture)

    def test_failed_positive_control_cannot_be_continued(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, preflight, args, fixture = self.fixture(root, success=False)
            with self.assertRaisesRegex(ValueError, "failed positive control"):
                self.validate(root, manifest, preflight, args, fixture)
