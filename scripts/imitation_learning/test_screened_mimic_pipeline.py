"""CPU checks for the screened Mimic pipeline's fail-closed gates."""

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.imitation_learning.action_contract import sha256_file
from scripts.imitation_learning.run_screened_mimic_pipeline import (
    known_candidate_failures,
    group_has_live_processes,
    Pipeline,
    terminate_owned_group,
    main,
    manifest_identities,
    validate_candidate_contract,
    validate_replay_report,
    wait_for_replay,
)


LIMITS = {"path": "/limits.json", "sha256": "a" * 64, "joint_names": [f"j{i}" for i in range(6)],
          "joint_lower_limits_rad": [-1.0] * 6, "joint_upper_limits_rad": [1.0] * 6}
CRITERIA = {"version": "stable_release_v2", "hold_time_s": 0.5}


class ScreenedPipelineGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw.hdf5"
        self.raw.write_bytes(b"raw")
        self.manifest = self.root / "selection.json"
        self.manifest.write_text(json.dumps({
            "schema_version": 1, "action_source": "recorded_target", "shards": [{
                "filename": self.raw.name, "raw_path": str(self.raw.resolve()),
                "raw_sha256": sha256_file(self.raw), "selected_demo_names": ["demo_1", "demo_2"],
            }]}), encoding="utf-8")
        self.plan = {"success_criteria": CRITERIA, "control_dt_s": 1 / 60, "joint_limits": LIMITS}
        self.report = {
            "dataset": None, "selection_manifest": str(self.manifest.resolve()),
            "selection_manifest_sha256": sha256_file(self.manifest),
            "selection_action_source": "recorded_target", "mode": "recorded_target",
            "gripper_effort_mode": "task", "success_criteria": CRITERIA, "control_dt_s": 1 / 60,
            "joint_names": LIMITS["joint_names"],
            "joint_lower_limits_rad": LIMITS["joint_lower_limits_rad"],
            "joint_upper_limits_rad": LIMITS["joint_upper_limits_rad"],
            "results": [self._result("demo_1"), self._result("demo_2")],
        }
        self.report_path = self.root / "evaluation.json"

    def tearDown(self):
        self.temporary.cleanup()

    def _result(self, name):
        return {"episode": name, "dataset": str(self.raw.resolve()), "raw_sha256": sha256_file(self.raw),
                "mode": "recorded_target", "steps": 60, "common_horizon": 60,
                "source_frame_count": 61, "first_success_step": 30,
                "source_success": True, "omitted_final_transition": True,
                "success_by_common_horizon": True, "success": True, "final_success": True,
                "clipped_steps": 0, "initial_joint_max_error_rad": 0.0, "joint_rmse_rad": 0.01}

    def _validate(self, report=None):
        self.report_path.write_text(json.dumps(self.report if report is None else report), encoding="utf-8")
        return validate_replay_report(self.report_path, self.manifest, self.plan)

    def test_valid_replay_and_candidate_contract_pass(self):
        self._validate()
        ids, _ = manifest_identities(self.manifest, verify_raw=True)
        contract = {"action_source": "recorded_target", "episode_count": 2,
                    "joint_limits": LIMITS, "episodes": [
                        {"raw_path": item[0], "raw_sha256": item[1], "raw_demo": item[2]} for item in ids]}
        validate_candidate_contract(contract, ids, LIMITS)

    def test_settling_diagnostics_cannot_approve_training(self):
        self.report["settling_protocol"] = {"version": "final_target_hold_v1",
                                            "steps": 60, "requested_seconds": 1.0}
        with self.assertRaisesRegex(ValueError, "diagnostic only"):
            self._validate()
        self.report["settling_protocol"].update(steps=0, requested_seconds=0)
        self._validate()

    def test_replay_failure_missing_duplicate_and_wrong_hash_are_rejected(self):
        cases = []
        failed = copy.deepcopy(self.report)
        failed["results"][0]["final_success"] = False
        cases.append(failed)
        missing = copy.deepcopy(self.report)
        missing["results"].pop()
        cases.append(missing)
        duplicate = copy.deepcopy(self.report)
        duplicate["results"][1] = copy.deepcopy(duplicate["results"][0])
        cases.append(duplicate)
        wrong_hash = copy.deepcopy(self.report)
        wrong_hash["results"][0]["raw_sha256"] = "0" * 64
        cases.append(wrong_hash)
        for report in cases:
            with self.subTest(report=report), self.assertRaises(ValueError):
                self._validate(report)

    def test_clip_nonfinite_and_boolean_type_errors_are_rejected(self):
        for key, value in (("clipped_steps", 1), ("initial_joint_max_error_rad", float("nan")),
                           ("final_success", 1), ("success", False), ("steps", 0),
                           ("first_success_step", None), ("common_horizon", 59),
                           ("mode", "next_observed")):
            report = copy.deepcopy(self.report)
            report["results"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._validate(report)

    def test_manifest_duplicate_and_changed_raw_are_rejected(self):
        payload = json.loads(self.manifest.read_text())
        payload["shards"][0]["selected_demo_names"] = ["demo_1", "demo_1"]
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            manifest_identities(self.manifest, verify_raw=True)
        payload["shards"][0]["selected_demo_names"] = ["demo_1"]
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        self.raw.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            manifest_identities(self.manifest, verify_raw=True)

    def test_candidate_contract_missing_duplicate_and_limits_mismatch_are_rejected(self):
        ids, _ = manifest_identities(self.manifest, verify_raw=True)
        episodes = [{"raw_path": item[0], "raw_sha256": item[1], "raw_demo": item[2]} for item in ids]
        base = {"action_source": "recorded_target", "episode_count": 2,
                "joint_limits": LIMITS, "episodes": episodes}
        for mutate in (
            lambda value: value["episodes"].pop(),
            lambda value: value["episodes"].__setitem__(1, copy.deepcopy(value["episodes"][0])),
            lambda value: value.__setitem__("joint_limits", {}),
        ):
            contract = copy.deepcopy(base)
            mutate(contract)
            with self.assertRaises(ValueError):
                validate_candidate_contract(contract, ids, LIMITS)

    def test_known_broad_failure_blocks_overlapping_training_candidate(self):
        ids, _ = manifest_identities(self.manifest, verify_raw=True)
        results = [self._result("demo_1"), self._result("demo_2")]
        results[0]["final_success"] = False
        failures = known_candidate_failures({"results": results}, ids)
        self.assertEqual(failures, [{"dataset": str(self.raw.resolve()),
                                    "raw_sha256": sha256_file(self.raw),
                                    "episode": "demo_1", "final_success": False}])
        results[0]["final_success"] = 0
        with self.assertRaisesRegex(ValueError, "strict boolean"):
            known_candidate_failures({"results": results}, ids)

    def test_broad_wait_rejects_different_manifest_before_returning_complete(self):
        import time
        self.report["selection_manifest_sha256"] = "0" * 64
        self.report_path.write_text(json.dumps(self.report))
        with patch("scripts.imitation_learning.run_screened_mimic_pipeline.EXPECTED_REPLAY", 2):
            with self.assertRaisesRegex(ValueError, "broad replay manifest"):
                wait_for_replay(self.root, self.manifest, time.monotonic() + 1, self.root / "STOP", self.plan)

    def test_broad_wait_rejects_changed_physics_criteria(self):
        for key, value in (("success_criteria", {}), ("control_dt_s", .1), ("joint_names", [])):
            report = copy.deepcopy(self.report)
            report[key] = value
            self.report_path.write_text(json.dumps(report))
            with self.subTest(key=key), \
                    patch("scripts.imitation_learning.run_screened_mimic_pipeline.EXPECTED_REPLAY", 2), \
                    self.assertRaisesRegex(ValueError, "differ from audit plan"):
                wait_for_replay(self.root, self.manifest, time.monotonic() + 1,
                                self.root / "STOP", self.plan)

    def test_broad_cohort_cannot_omit_a_known_failure(self):
        payload = json.loads(self.manifest.read_text())
        payload["shards"][0]["selected_demo_names"] = [f"demo_{i}" for i in range(20)]
        report = copy.deepcopy(self.report)
        report["results"] = [self._result(f"demo_{i}") for i in range(20)]
        report["results"][-1]["final_success"] = False
        # A self-consistent 19-episode manifest/report must not hide the failed demo.
        payload["shards"][0]["selected_demo_names"].pop()
        report["results"].pop()
        self.manifest.write_text(json.dumps(payload))
        report["selection_manifest_sha256"] = sha256_file(self.manifest)
        self.report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            wait_for_replay(self.root, self.manifest, time.monotonic() + 1, self.root / "STOP", self.plan)

    def test_main_never_launches_split_or_train_for_known_failure(self):
        import sys
        ids, _ = manifest_identities(self.manifest, verify_raw=True)
        audit = {"plan": self.plan, "candidate_ids": ids, "replay_ids": ids,
                 "candidate_manifest": self.manifest, "replay_manifest": self.manifest,
                 "candidate_manifest_sha256": sha256_file(self.manifest),
                 "replay_manifest_sha256": sha256_file(self.manifest)}
        (self.root / "plan.json").write_text(json.dumps(self.plan))
        broad = copy.deepcopy(self.report)
        broad["results"][0]["final_success"] = False
        output = self.root / "pipeline"
        args = ["runner", "--audit-root", str(self.root), "--wait-replay-dir", str(self.root),
                "--wait-replay-pid", "99999999",
                "--wait-replay-manifest", str(self.manifest), "--batch-root", str(self.root),
                "--output-dir", str(output), "--isaac-python", sys.executable]
        module = "scripts.imitation_learning.run_screened_mimic_pipeline"
        with patch.object(sys, "argv", args), patch(f"{module}.validate_audit_root", return_value=audit), \
                patch(f"{module}.check_free_space"), patch(f"{module}.wait_for_replay", return_value=broad), \
                patch(f"{module}.validate_replay_report", return_value=self.report), \
                patch(f"{module}.Pipeline.run_child") as child:
            with self.assertRaisesRegex(ValueError, "already disproved"):
                main()
        self.assertEqual([call.args[0] for call in child.call_args_list], ["strict_replay"])
        status = json.loads((output / "progress.json").read_text())
        self.assertEqual(status["status"], "rejected")
        self.assertFalse(status["training_started"])
        self.assertEqual(status["known_candidate_failures"][0]["episode"], "demo_1")

    def test_exited_leader_cannot_leave_live_descendants(self):
        (self.root / "logs").mkdir()
        pipeline = Pipeline(SimpleNamespace(min_free_gib=8), self.root, {})
        command = [sys.executable, "-c",
                   "import subprocess,sys; "
                   "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                   "print(p.pid,flush=True)"]
        with patch("scripts.imitation_learning.run_screened_mimic_pipeline.check_free_space"):
            with self.assertRaisesRegex(RuntimeError, "live descendants"):
                pipeline.run_child("orphan_test", command)
        child = int((self.root / "logs/orphan_test.log").read_text().strip())
        stat = Path(f"/proc/{child}/stat")
        self.assertTrue(not stat.exists() or stat.read_text().rsplit(") ", 1)[1].split()[0] == "Z")

    def test_sigterm_cleans_owned_child_group(self):
        (self.root / "logs").mkdir()
        program = (
            "import sys; from pathlib import Path; from types import SimpleNamespace; "
            "import scripts.imitation_learning.run_screened_mimic_pipeline as m; "
            "m.check_free_space=lambda *a:None; m.install_termination_handlers(); "
            "p=m.Pipeline(SimpleNamespace(min_free_gib=8),Path(sys.argv[1]),{}); "
            "p.run_child('signal_test',[sys.executable,'-c','import time; time.sleep(30)'])"
        )
        parent = subprocess.Popen([sys.executable, "-c", program, str(self.root)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pgid = None
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    pgid = json.loads((self.root / "progress.json").read_text()).get("child_pid")
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                if pgid:
                    break
                time.sleep(.05)
            self.assertIsNotNone(pgid)
            os.kill(parent.pid, signal.SIGTERM)
            self.assertEqual(parent.wait(timeout=20), 128 + signal.SIGTERM)
            self.assertFalse(group_has_live_processes(pgid))
        finally:
            if pgid:
                terminate_owned_group(pgid, grace_s=1)
            if parent.poll() is None:
                parent.kill()
                parent.wait()

    def test_nested_act_stage_cannot_leave_descendants(self):
        from scripts.imitation_learning.run_act_vision_experiment import run_owned_stage
        (self.root / "logs").mkdir()
        status = {}
        command = [sys.executable, "-c",
                   "import subprocess,sys; "
                   "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                   "print(p.pid,flush=True)"]
        with patch("scripts.imitation_learning.run_act_vision_experiment.check_free_space"):
            with self.assertRaisesRegex(RuntimeError, "live descendants"):
                run_owned_stage(self.root, 8, status.update, "nested_test", command)
        child = int((self.root / "logs/nested_test.log").read_text().strip())
        stat = Path(f"/proc/{child}/stat")
        self.assertTrue(not stat.exists() or stat.read_text().rsplit(") ", 1)[1].split()[0] == "Z")
        self.assertIsNone(status["child_pid"])


if __name__ == "__main__":
    unittest.main()
