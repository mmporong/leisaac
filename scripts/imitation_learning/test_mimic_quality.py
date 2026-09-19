"""CPU tests for raw-state quality screening and target-jump attribution."""

from pathlib import Path
import json
import tempfile
import unittest

import h5py
import numpy as np
import torch

from scripts.imitation_learning.audit_mimic_quality import (
    jump_diagnostics, recorded_release, release, release_run_summary, run_audit,
)
from scripts.imitation_learning.convert_hdf5_to_lerobot import JOINT_NAMES, load_joint_limits
from scripts.imitation_learning.run_mimic_image_batch import load_selection_manifest


class MimicQualityTest(unittest.TestCase):
    def test_holding_and_interruption_match_live_window(self):
        candidates = np.array([True] * 30 + [False] + [True] * 29, dtype=bool)
        result = release_run_summary(candidates, 1 / 60, .5)
        self.assertEqual(result["first_success_step"], 30)
        self.assertFalse(result["final_stable"])
        self.assertEqual(result["terminal_stable_steps"], 29)
        window = release.StableReleaseWindow(torch.zeros(1, dtype=torch.long))
        success_steps = []
        for step, value in enumerate(candidates, start=1):
            if window.update(torch.tensor([bool(value)]), torch.tensor([step]), step, 1 / 60, .5).item():
                success_steps.append(step)
        self.assertEqual(result["first_success_step"], success_steps[0])
        self.assertEqual(result["final_stable"], len(candidates) in success_steps)

    def test_empty_and_invalid_timing(self):
        self.assertFalse(release_run_summary(np.array([], dtype=bool), 1 / 60, .5)["ever_stable"])
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                release_run_summary(np.array([True]), value, .5)

    def make_demo(self, hdf, frames):
        demo = hdf.create_group("data/demo_0")
        pose = np.zeros((frames, 7), dtype=np.float32)
        pose[:, 3] = 1.
        demo.create_dataset("states/rigid_object/box_target/root_pose", data=pose)
        pose[:, 2] = .02
        demo.create_dataset("states/rigid_object/cube/root_pose", data=pose)
        demo.create_dataset("states/rigid_object/cube/root_velocity", data=np.zeros((frames, 6), np.float32))
        joint = np.zeros((frames, 6), np.float32)
        joint[:, -1] = .4
        for key in ("obs/joint_pos", "obs/joint_pos_target", "states/articulation/robot/joint_position"):
            demo.create_dataset(key, data=joint)
        demo.create_dataset("actions", data=np.zeros((frames, 8), np.float32))
        return demo

    def test_last_unlabelled_transition_cannot_supply_missing_hold_sample(self):
        with tempfile.TemporaryDirectory() as directory, h5py.File(Path(directory) / "raw.hdf5", "w") as hdf:
            demo = self.make_demo(hdf, 30)
            result = recorded_release(demo, release.release_criteria_metadata(), 1 / 60)
            self.assertEqual(result["terminal_stable_steps"], 29)
            self.assertFalse(result["final_stable"])

    def test_recorded_release_requires_valid_velocity(self):
        with tempfile.TemporaryDirectory() as directory, h5py.File(Path(directory) / "raw.hdf5", "w") as hdf:
            demo = self.make_demo(hdf, 31)
            self.assertTrue(recorded_release(demo, release.release_criteria_metadata(), 1 / 60)["final_stable"])
            demo["states/rigid_object/cube/root_velocity"][0, 2] = 2.
            self.assertFalse(recorded_release(demo, release.release_criteria_metadata(), 1 / 60)["final_stable"])
            demo["states/rigid_object/cube/root_velocity"][1, 0] = np.nan
            with self.assertRaises(ValueError):
                recorded_release(demo, release.release_criteria_metadata(), 1 / 60)

    def test_joint_jump_is_not_confused_with_observation_or_pose_jump(self):
        with tempfile.TemporaryDirectory() as directory, h5py.File(Path(directory) / "raw.hdf5", "w") as hdf:
            demo = self.make_demo(hdf, 4)
            demo["obs/joint_pos_target"][2, 2] = 2.
            result = jump_diagnostics(demo, (np.full(6, -3.), np.full(6, 3.)), 1.,
                                      ["pan", "lift", "elbow", "wrist", "roll", "gripper"])
            self.assertEqual(result["jump_count"], 2)
            self.assertEqual(result["clipped_frames"], 0)
            self.assertEqual(result["dominant_joint_counts"], {"elbow": 2})
            self.assertEqual(result["events"][0]["source_target_index"], 2)
            self.assertEqual(result["events"][0]["observed_joint_step_norm_rad"], 0.)
            self.assertEqual(result["events"][0]["eef_target_translation_step_m"], 0.)
            demo["states/articulation/robot/joint_position"][:, 0] = [.01, .04, .12, .30]
            demo["obs/joint_pos"][:, 0] = [0., .01, .04, .12]
            aligned = jump_diagnostics(demo, (np.full(6, -3.), np.full(6, 3.)), 1.,
                                       ["pan", "lift", "elbow", "wrist", "roll", "gripper"])
            self.assertAlmostEqual(aligned["events"][0]["observed_joint_step_norm_rad"], .03, places=6)
            self.assertAlmostEqual(aligned["events"][1]["observed_joint_step_norm_rad"], .08, places=6)

    def test_bad_image_size_is_configuration_error_not_episode_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for size in (0, 257):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    run_audit(root, root / "limits", root / "reference", root / "output", image_size=size)
            self.assertFalse((root / "output").exists())

    def test_full_audit_produces_hash_bound_selection_without_touching_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            source = raw / "shard_000.hdf5"
            with h5py.File(source, "w") as hdf:
                demo = self.make_demo(hdf, 31)
                demo.attrs["success"] = True
                for camera in ("front", "wrist"):
                    demo.create_dataset(f"obs/{camera}", data=np.zeros((31, 84, 84, 3), np.uint8))
            original = source.read_bytes()
            reference = root / "reference.json"
            reference.write_text(json.dumps({"control_dt_s": 1 / 60,
                                             "success_criteria": release.release_criteria_metadata()}))
            limits = root / "limits.json"
            limits.write_text(json.dumps({"joint_names": [n.removesuffix('.pos') for n in JOINT_NAMES],
                                         "joint_lower_limits_rad": [-3.] * 6,
                                         "joint_upper_limits_rad": [3.] * 6}))
            output = root / "audit"
            report = run_audit(raw, limits, reference, output, image_size=84)
            self.assertEqual(report["screen_pass"], 1)
            self.assertFalse(report["replay_executed"])
            selected = load_selection_manifest(output / "candidate_selection.json", raw,
                                               "recorded_target", 1., load_joint_limits(limits)[2])
            self.assertEqual(selected[0]["selected_demo_names"], ["demo_0"])
            self.assertEqual(original, source.read_bytes())
            with self.assertRaises(FileExistsError):
                run_audit(raw, limits, reference, output, image_size=84)
            with h5py.File(source, "a") as hdf:
                hdf["data/demo_0/obs/joint_pos_target"][1:, 2] = 3.01
            screened = run_audit(raw, limits, reference, root / "clipped_allowed", image_size=84)
            strict = run_audit(raw, limits, reference, root / "clipped_rejected", image_size=84,
                               require_unclipped_targets=True)
            self.assertEqual(screened["screen_pass"], 1)
            self.assertEqual(strict["screen_pass"], 0)


if __name__ == "__main__":
    unittest.main()
