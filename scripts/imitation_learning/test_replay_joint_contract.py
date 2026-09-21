"""CPU regression checks for pre-step recorder command alignment."""

import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch


class ControlSnapshotTest(unittest.TestCase):
    def test_snapshot_records_state_without_mutating_buffers(self):
        source = Path(__file__).resolve().parents[1] / "evaluation/replay_joint_contract.py"
        nodes = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == "control_snapshot"]
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        values = torch.arange(6, dtype=torch.float32).reshape(1, 6)
        robot = SimpleNamespace(body_names=["base", "gripper"], data=SimpleNamespace(
            joint_pos=values, joint_vel=values, joint_pos_target=values,
            joint_effort_limits=values, applied_torque=values,
            body_link_pos_w=torch.zeros(1, 2, 3)))
        cube = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[.1, 0., 0.]]),
            body_link_pos_w=torch.tensor([[[.1, 0., 0.]]]), default_mass=torch.tensor([[.01]])))
        class Scene(dict):
            rigid_objects = {"cube": cube}
        env = SimpleNamespace(scene=Scene(robot=robot))
        result = namespace["control_snapshot"](env)
        self.assertEqual(result["last_robot_link_name"], "gripper")
        self.assertAlmostEqual(result["objects"]["cube"]["distance_to_last_robot_link_m"], .1)
        result["joint_position_rad"][0] = 999
        self.assertEqual(float(robot.data.joint_pos[0, 0]), 0)


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


class ReplaySelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "evaluation/replay_selection.py"
        spec = importlib.util.spec_from_file_location("replay_selection_for_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.load_selection = staticmethod(module.load_replay_selection)
        cls.sha256_file = staticmethod(module.sha256_file)

    def _write_manifest(self, root, shards, **overrides):
        path = root / "selection.json"
        payload = {"schema_version": 1, "action_source": "recorded_target", "shards": shards}
        payload.update(overrides)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _shard(self, raw, names):
        return {"filename": raw.name, "raw_path": str(raw.resolve()),
                "raw_sha256": self.sha256_file(raw), "selected_demo_names": names}

    def test_dataset_mode_preserves_legacy_default_and_explicit_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw.hdf5"
            raw.write_bytes(b"raw")
            plan, metadata = self.load_selection(raw, None, None)
            self.assertEqual(plan[0]["selected_demo_names"], ["demo_0", "demo_1", "demo_2"])
            self.assertEqual(plan[0]["raw_sha256"], self.sha256_file(raw))
            self.assertEqual(metadata["dataset"], str(raw.resolve()))
            plan, _ = self.load_selection(raw, [7, 2], None)
            self.assertEqual(plan[0]["selected_demo_names"], ["demo_7", "demo_2"])

    def test_manifest_preserves_shard_and_demo_order_with_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_a, raw_b = root / "a.hdf5", root / "b.hdf5"
            raw_a.write_bytes(b"a")
            raw_b.write_bytes(b"b")
            audit = root / "audit.json"
            audit.write_text("{}", encoding="utf-8")
            first_shard = self._shard(raw_a, ["demo_4", "demo_1"])
            first_shard.update(audit_path=str(audit.resolve()), audit_sha256=self.sha256_file(audit))
            manifest = self._write_manifest(root, [
                first_shard,
                self._shard(raw_b, ["demo_0"]),
            ], purpose="CPU replay cohort")
            plan, metadata = self.load_selection(None, None, manifest)
            self.assertEqual([item["selected_demo_names"] for item in plan],
                             [["demo_4", "demo_1"], ["demo_0"]])
            self.assertEqual([item["manifest_shard_index"] for item in plan], [0, 1])
            self.assertEqual(metadata["selection_manifest"], str(manifest.resolve()))
            self.assertEqual(metadata["selection_manifest_sha256"], self.sha256_file(manifest))
            self.assertEqual(metadata["selection_action_source"], "recorded_target")

    def test_manifest_rejects_explicit_episodes_and_duplicate_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.hdf5"
            raw.write_bytes(b"same bytes")
            shard = self._shard(raw, ["demo_0"])
            manifest = self._write_manifest(root, [shard, dict(shard)])
            with self.assertRaisesRegex(ValueError, "episodes"):
                self.load_selection(None, [0], manifest)
            with self.assertRaisesRegex(ValueError, "duplicate raw/demo"):
                self.load_selection(None, None, manifest)

    def test_manifest_fails_closed_for_malformed_or_changed_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.hdf5"
            raw.write_bytes(b"before")
            valid_shard = self._shard(raw, ["demo_0"])
            malformed = [
                ({"schema_version": True}, "schema"),
                ({"action_source": "mimic_action"}, "action_source"),
                ({"unknown": 1}, "missing or unknown"),
            ]
            for override, message in malformed:
                manifest = self._write_manifest(root, [valid_shard], **override)
                with self.subTest(override=override), self.assertRaisesRegex(ValueError, message):
                    self.load_selection(None, None, manifest)
            for field, value, message in (
                ("filename", "other.hdf5", "filename does not match"),
                ("raw_path", "relative.hdf5", "absolute"),
                ("raw_sha256", "0" * 64, "hash mismatch"),
                ("selected_demo_names", ["demo_0", "demo_0"], "unique"),
                ("selected_demo_names", ["bad_name"], "demo_N"),
            ):
                shard = dict(valid_shard)
                shard[field] = value
                manifest = self._write_manifest(root, [shard])
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
                    self.load_selection(None, None, manifest)
            manifest = self._write_manifest(root, [valid_shard])
            raw.write_bytes(b"after")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.load_selection(None, None, manifest)

    def test_exactly_one_source_and_unique_nonnegative_episodes_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw.hdf5"
            raw.write_bytes(b"raw")
            for args in ((None, None, None), (raw, None, raw)):
                with self.subTest(args=args), self.assertRaisesRegex(ValueError, "exactly one"):
                    self.load_selection(*args)
            for episodes in ([], [0, 0], [-1]):
                with self.subTest(episodes=episodes), self.assertRaisesRegex(ValueError, "nonnegative and unique"):
                    self.load_selection(raw, episodes, None)


if __name__ == "__main__":
    unittest.main()
