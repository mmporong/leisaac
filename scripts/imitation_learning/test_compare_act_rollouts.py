"""CPU-only tests for paired ACT rollout comparison."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.evaluation.compare_act_rollouts import EXPECTED_SUCCESS_CRITERIA, build_report


SCRIPT = Path(__file__).resolve().parents[1] / "evaluation/compare_act_rollouts.py"


def write_evaluation(
    root: Path,
    seed: int,
    *,
    checkpoint: str,
    success: bool,
    outcome: str,
    cube_xyz: list[float] | None = None,
    joint_pos: list[float] | None = None,
    horizon: int = 1200,
    front_hash: str = "front",
    wrist_hash: str = "wrist",
    scene_hash: str = "a" * 64,
    metadata_overrides: dict | None = None,
) -> None:
    directory = root / f"seed_{seed}"
    directory.mkdir(parents=True)
    evaluation = {
        "task": "LeIsaac-SO101-PickCubeIntoBox-v0",
        "checkpoint": checkpoint,
        "num_rollouts": 1,
        "seed_start": seed,
        "horizon": horizon,
        "server_seed": 0,
        "n_action_steps": None,
        "lift_threshold_m": 0.02,
        "server_device": "cpu",
        "render_width": 320,
        "render_height": 240,
        "policy_image_size": 84,
        "control_dt_s": 1.0 / 60.0,
        "reset_render_frames": 4,
        "gripper_effort_mode": "task",
        "success_criteria": dict(EXPECTED_SUCCESS_CRITERIA),
        "joint_names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        "joint_lower_limits_rad": [-1.0] * 6,
        "joint_upper_limits_rad": [1.0] * 6,
        "results": [
            {
                "seed": seed,
                "success": success,
                "outcome": outcome,
                "initial_cube_xyz": cube_xyz or [0.4, 0.0, 0.03],
                "initial_joint_pos": joint_pos or [0.0] * 6,
                "initial_scene_state_sha256": scene_hash,
                "initial_observation_sha256": {"front": front_hash, "wrist": wrist_hash},
            }
        ],
    }
    if metadata_overrides:
        evaluation.update(metadata_overrides)
    (directory / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")


class CompareActRolloutsTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.baseline = root / "baseline"
        self.recovery = root / "recovery"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_reports_every_paired_result_including_unfavorable_recovery(self):
        cases = [
            (4000, True, "success", True, "success"),
            (4001, False, "no_lift", False, "low_after_lift"),
            (4002, True, "success", False, "lifted_not_completed"),
            (4003, False, "no_lift", True, "success"),
        ]
        for seed, baseline_success, baseline_outcome, recovery_success, recovery_outcome in cases:
            write_evaluation(
                self.baseline,
                seed,
                checkpoint="baseline/checkpoint",
                success=baseline_success,
                outcome=baseline_outcome,
            )
            write_evaluation(
                self.recovery,
                seed,
                checkpoint="recovery/checkpoint",
                success=recovery_success,
                outcome=recovery_outcome,
                front_hash="different-front" if seed == 4002 else "front",
            )

        report = build_report(self.baseline, self.recovery, [4000, 4001, 4002, 4003])

        self.assertEqual(report["baseline"]["successes"], 2)
        self.assertEqual(report["recovery"]["successes"], 2)
        self.assertEqual(
            report["baseline"]["outcome_counts"],
            {"success": 2, "no_lift": 2, "low_after_lift": 0, "lifted_not_completed": 0},
        )
        self.assertEqual(
            report["recovery"]["outcome_counts"],
            {"success": 2, "no_lift": 0, "low_after_lift": 1, "lifted_not_completed": 1},
        )
        self.assertEqual(report["delta_percentage_points"], 0.0)
        self.assertEqual(
            report["paired"],
            {"both_success": 1, "both_failure": 1, "baseline_only": 1, "recovery_only": 1},
        )
        self.assertEqual(len(report["per_seed"]), 4)
        self.assertFalse(report["per_seed"][2]["initial_observation_sha256_equal"]["front"])
        self.assertTrue(report["per_seed"][2]["initial_state_equal"]["initial_cube_xyz"])
        self.assertEqual(report["conditions"]["server_device"], "cpu")
        self.assertEqual(report["conditions"]["policy_image_size"], 84)
        self.assertIn("4 paired rollout seeds", report["interpretation"])
        self.assertIn("does not establish statistical superiority", report["interpretation"])

    def test_rejects_duplicate_and_missing_seeds(self):
        with self.assertRaisesRegex(ValueError, "duplicate seeds"):
            build_report(self.baseline, self.recovery, [4000, 4000])
        with self.assertRaisesRegex(FileNotFoundError, "missing baseline evaluation"):
            build_report(self.baseline, self.recovery, [4000])

    def test_rejects_condition_mismatch(self):
        write_evaluation(
            self.baseline, 4000, checkpoint="baseline/checkpoint", success=False, outcome="no_lift"
        )
        write_evaluation(
            self.recovery,
            4000,
            checkpoint="recovery/checkpoint",
            success=False,
            outcome="no_lift",
            horizon=600,
        )
        with self.assertRaisesRegex(ValueError, "condition mismatch.*horizon"):
            build_report(self.baseline, self.recovery, [4000])

    def test_rejects_mismatch_for_each_strict_evaluation_condition(self):
        mismatches = {
            "control_dt_s": 1.0 / 30.0,
            "reset_render_frames": 0,
            "gripper_effort_mode": "fixed",
            "success_criteria": {**EXPECTED_SUCCESS_CRITERIA, "hold_time_s": 0.6},
            "joint_names": ["different", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
            "joint_lower_limits_rad": [-0.9] + [-1.0] * 5,
            "joint_upper_limits_rad": [0.9] + [1.0] * 5,
        }
        for key, recovery_value in mismatches.items():
            with self.subTest(key=key):
                root = Path(self.temporary_directory.name) / f"mismatch_{key}"
                baseline = root / "baseline"
                recovery = root / "recovery"
                write_evaluation(
                    baseline, 4000, checkpoint="baseline/checkpoint", success=False, outcome="no_lift"
                )
                write_evaluation(
                    recovery,
                    4000,
                    checkpoint="recovery/checkpoint",
                    success=False,
                    outcome="no_lift",
                    metadata_overrides={key: recovery_value},
                )
                with self.assertRaisesRegex(ValueError, f"condition mismatch.*{key}"):
                    build_report(baseline, recovery, [4000])

    def test_rejects_changed_or_missing_success_criteria_fields(self):
        for key, expected_value in EXPECTED_SUCCESS_CRITERIA.items():
            changed = dict(EXPECTED_SUCCESS_CRITERIA)
            changed[key] = "different" if isinstance(expected_value, str) else expected_value + 0.01
            missing = dict(EXPECTED_SUCCESS_CRITERIA)
            del missing[key]
            for case, criteria in (("changed", changed), ("missing", missing)):
                with self.subTest(key=key, case=case):
                    root = Path(self.temporary_directory.name) / f"criteria_{case}_{key}"
                    baseline = root / "baseline"
                    recovery = root / "recovery"
                    write_evaluation(
                        baseline,
                        4000,
                        checkpoint="baseline/checkpoint",
                        success=False,
                        outcome="no_lift",
                        metadata_overrides={"success_criteria": criteria},
                    )
                    write_evaluation(
                        recovery,
                        4000,
                        checkpoint="recovery/checkpoint",
                        success=False,
                        outcome="no_lift",
                    )
                    expected_error = (
                        "success_criteria.version"
                        if key == "version" and case == "changed"
                        else "success_criteria"
                    )
                    with self.assertRaisesRegex(ValueError, expected_error):
                        build_report(baseline, recovery, [4000])

    def test_rejects_nonfinite_success_criteria_numbers(self):
        numeric_keys = [
            key for key, value in EXPECTED_SUCCESS_CRITERIA.items() if isinstance(value, float)
        ]
        for key in numeric_keys:
            with self.subTest(key=key):
                root = Path(self.temporary_directory.name) / f"criteria_nonfinite_{key}"
                baseline = root / "baseline"
                recovery = root / "recovery"
                criteria = dict(EXPECTED_SUCCESS_CRITERIA)
                criteria[key] = float("nan")
                write_evaluation(
                    baseline,
                    4000,
                    checkpoint="baseline/checkpoint",
                    success=False,
                    outcome="no_lift",
                    metadata_overrides={"success_criteria": criteria},
                )
                write_evaluation(
                    recovery,
                    4000,
                    checkpoint="recovery/checkpoint",
                    success=False,
                    outcome="no_lift",
                )
                with self.assertRaisesRegex(ValueError, f"success_criteria.{key} must be finite and positive"):
                    build_report(baseline, recovery, [4000])

    def test_allows_same_valid_custom_success_criteria_in_both_arms(self):
        custom_criteria = {**EXPECTED_SUCCESS_CRITERIA, "hold_time_s": 1.0}
        write_evaluation(
            self.baseline,
            4000,
            checkpoint="baseline/checkpoint",
            success=False,
            outcome="no_lift",
            metadata_overrides={"success_criteria": custom_criteria},
        )
        write_evaluation(
            self.recovery,
            4000,
            checkpoint="recovery/checkpoint",
            success=False,
            outcome="no_lift",
            metadata_overrides={"success_criteria": custom_criteria},
        )

        report = build_report(self.baseline, self.recovery, [4000])

        self.assertEqual(report["conditions"]["success_criteria"]["hold_time_s"], 1.0)

    def test_rejects_nonpositive_and_reversed_success_criteria_bounds(self):
        invalid_criteria = {
            "nonpositive": {**EXPECTED_SUCCESS_CRITERIA, "hold_time_s": 0.0},
            "reversed_heights": {
                **EXPECTED_SUCCESS_CRITERIA,
                "min_height_m": EXPECTED_SUCCESS_CRITERIA["max_height_m"],
            },
        }
        for case, criteria in invalid_criteria.items():
            with self.subTest(case=case):
                root = Path(self.temporary_directory.name) / f"criteria_{case}"
                baseline = root / "baseline"
                recovery = root / "recovery"
                write_evaluation(
                    baseline,
                    4000,
                    checkpoint="baseline/checkpoint",
                    success=False,
                    outcome="no_lift",
                    metadata_overrides={"success_criteria": criteria},
                )
                write_evaluation(
                    recovery,
                    4000,
                    checkpoint="recovery/checkpoint",
                    success=False,
                    outcome="no_lift",
                )
                with self.assertRaisesRegex(ValueError, "success_criteria"):
                    build_report(baseline, recovery, [4000])

    def test_rejects_missing_strict_evaluation_metadata(self):
        for key in (
            "control_dt_s",
            "reset_render_frames",
            "gripper_effort_mode",
            "success_criteria",
            "joint_names",
            "joint_lower_limits_rad",
            "joint_upper_limits_rad",
        ):
            with self.subTest(key=key):
                root = Path(self.temporary_directory.name) / f"missing_{key}"
                baseline = root / "baseline"
                recovery = root / "recovery"
                write_evaluation(
                    baseline, 4000, checkpoint="baseline/checkpoint", success=False, outcome="no_lift"
                )
                path = baseline / "seed_4000" / "evaluation.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                del payload[key]
                path.write_text(json.dumps(payload), encoding="utf-8")
                write_evaluation(
                    recovery, 4000, checkpoint="recovery/checkpoint", success=False, outcome="no_lift"
                )
                with self.assertRaisesRegex(ValueError, key):
                    build_report(baseline, recovery, [4000])

    def test_rejects_invalid_strict_metadata_values(self):
        invalid_values = {
            "control_dt_s": float("nan"),
            "reset_render_frames": -1,
            "gripper_effort_mode": "unknown",
            "success_criteria": {**EXPECTED_SUCCESS_CRITERIA, "extra": 1},
            "joint_names": ["duplicate"] * 6,
            "joint_lower_limits_rad": [float("nan")] + [-1.0] * 5,
            "joint_upper_limits_rad": [float("inf")] + [1.0] * 5,
        }
        for key, value in invalid_values.items():
            with self.subTest(key=key):
                root = Path(self.temporary_directory.name) / f"invalid_{key}"
                baseline = root / "baseline"
                recovery = root / "recovery"
                write_evaluation(
                    baseline,
                    4000,
                    checkpoint="baseline/checkpoint",
                    success=False,
                    outcome="no_lift",
                    metadata_overrides={key: value},
                )
                write_evaluation(
                    recovery, 4000, checkpoint="recovery/checkpoint", success=False, outcome="no_lift"
                )
                with self.assertRaisesRegex(ValueError, key):
                    build_report(baseline, recovery, [4000])

    def test_rejects_nonpositive_control_dt(self):
        write_evaluation(
            self.baseline,
            4000,
            checkpoint="baseline/checkpoint",
            success=False,
            outcome="no_lift",
            metadata_overrides={"control_dt_s": 0.0},
        )
        write_evaluation(
            self.recovery, 4000, checkpoint="recovery/checkpoint", success=False, outcome="no_lift"
        )
        with self.assertRaisesRegex(ValueError, "control_dt_s must be finite and positive"):
            build_report(self.baseline, self.recovery, [4000])

    def test_rejects_initial_state_mismatch_with_exact_difference(self):
        write_evaluation(
            self.baseline, 4000, checkpoint="baseline/checkpoint", success=False, outcome="no_lift"
        )
        write_evaluation(
            self.recovery,
            4000,
            checkpoint="recovery/checkpoint",
            success=False,
            outcome="no_lift",
            cube_xyz=[0.4, 0.0, 0.030002],
        )
        with self.assertRaisesRegex(
            ValueError, r"initial_cube_xyz\[2\].*absolute_difference=.*tolerance=1e-06"
        ):
            build_report(self.baseline, self.recovery, [4000])

    def test_rejects_scene_state_hash_mismatch(self):
        write_evaluation(
            self.baseline, 4000, checkpoint="baseline/checkpoint", success=False, outcome="no_lift"
        )
        write_evaluation(
            self.recovery,
            4000,
            checkpoint="recovery/checkpoint",
            success=False,
            outcome="no_lift",
            scene_hash="b" * 64,
        )
        with self.assertRaisesRegex(ValueError, "initial scene state hash mismatch"):
            build_report(self.baseline, self.recovery, [4000])

    def test_rejects_invalid_scene_state_hash(self):
        write_evaluation(
            self.baseline,
            4000,
            checkpoint="baseline/checkpoint",
            success=False,
            outcome="no_lift",
            scene_hash="not-a-sha256",
        )
        write_evaluation(
            self.recovery, 4000, checkpoint="recovery/checkpoint", success=False, outcome="no_lift"
        )
        with self.assertRaisesRegex(ValueError, "64-character hexadecimal SHA-256"):
            build_report(self.baseline, self.recovery, [4000])

    def test_cli_refuses_to_overwrite_output(self):
        output = Path(self.temporary_directory.name) / "report.json"
        output.write_text("keep", encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--baseline-dir",
                str(self.baseline),
                "--recovery-dir",
                str(self.recovery),
                "--seeds",
                "4000",
                "--output",
                str(output),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("output already exists", completed.stderr)
        self.assertEqual(output.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
