"""Preregistered chunk-probe boundaries and degenerate-predictor checks."""
import sys
import unittest
import hashlib
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from probe_gripper_close_offline import (aligned_state, band_for, closed_loop_status, first_close,
    metric_row, null_chunks, predict_chunk, registered_selection, run_closed_loop, summarize, trace_index, verdict)


class ClosedLoopIntegrationTests(unittest.TestCase):
    def exercise(self, *, gate=True, checkpoint_match=True, retained_close=30, changed=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "train/meta").mkdir(parents=True)
            (root / "train/meta/stats.json").write_text(json.dumps({"action": {"mean": [1] * 6}}))
            (root / "model.safetensors").write_bytes(b"fake test model")
            manifest = root / "policy_images_001.json"
            manifest.write_text("[]")
            raw = root / "raw.hdf5"
            states = np.ones((400, 6), dtype=np.float32)
            targets = states.copy()
            targets[278:, 5] = 0
            with h5py.File(raw, "w") as file:
                obs = file.create_group("data/demo_9/obs")
                obs.create_dataset("joint_pos", data=states)
                obs.create_dataset("joint_pos_target", data=targets)
                for camera in ("front", "wrist"):
                    obs.create_dataset(camera, shape=(400, 224, 224, 3), dtype="uint8", fillvalue=0)
            inputs = {}
            for step in range(240, 341):
                state = states[step - 1].copy()
                if 249 + retained_close <= step <= 278:
                    state[5] += .21
                inputs[step] = {"state_before": state, **{
                    camera: {"pixels": np.zeros((224, 224, 3), dtype=np.uint8)} for camera in ("front", "wrist")}}
            entry = {"raw_path": str(raw), "raw_demo": "demo_9", "aggregate_episode_index": 24}
            contract = {"splits": {"train": {"episodes": [entry] * 21}}, "joint_limits": {
                "joint_lower_limits_rad": [-2] * 6, "joint_upper_limits_rad": [2] * 6}}
            args = SimpleNamespace(part="train", episode=20, input_path="raw", stride=1, max_samples=None,
                frame_range=None, closed_loop_dir=root, reference_dir=root, checkpoint=root,
                split_root=root, output=root / "result.json")
            comparison = {"strict_reproduction_gate": {"pass": gate},
                          "inputs": {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}}
            from contextlib import ExitStack
            with ExitStack() as stack:
                stack.enter_context(patch("compare_close_window_inputs.analyze", side_effect=[comparison,
                    {**comparison, "changed": True} if changed else comparison]))
                stack.enter_context(patch("compare_close_window_inputs.load_json", return_value={
                    "checkpoint": str(root if checkpoint_match else root / "different")}))
                stack.enter_context(patch("compare_close_window_inputs.load_trace", return_value=[]))
                stack.enter_context(patch("compare_close_window_inputs.validate_manifest", return_value=inputs))
                stack.enter_context(patch("probe_gripper_close_offline.predict_chunk", return_value=np.zeros((30, 6))))
                run_closed_loop(args, None, None, None, contract, {})
            return json.loads(args.output.read_text())

    def test_paired_counterfactual_alignment_and_nulls(self):
        result = self.exercise()
        self.assertEqual(result["samples"], 101)
        self.assertEqual(result["closed_loop_status"], "high")
        self.assertEqual([r["s"] for r in result["rows"]], list(range(239, 340)))
        self.assertEqual([r["s"] for r in result["paired_teacher_rows"]], list(range(239, 340)))
        self.assertEqual([r["step"] for r in result["rows"] if r["actual_chunk_query_boundary"]], [241, 271, 301, 331])
        rows = {r["s"]: r for r in result["rows"]}
        self.assertFalse(rows[247]["predictions"]["act"]["target_closed"])
        self.assertTrue(rows[248]["predictions"]["act"]["target_closed"])
        self.assertEqual(set(rows[248]["predictions"]), {"act", "mean", "always_closed", "state_copy"})

    def test_retained_floor_overrides_perfect_hit(self):
        result = self.exercise(retained_close=14)
        self.assertEqual(result["metrics"]["act"]["close"]["frames"], 14)
        self.assertEqual(result["closed_loop_status"], "unresolved")
        self.assertEqual(len(result["excluded"]), 16)
        self.assertEqual([r["s"] for r in result["rows"]], [r["s"] for r in result["paired_teacher_rows"]])

    def test_strict_gate_checkpoint_and_mutation_rejected(self):
        for options, message in (({"gate": False}, "replay gate"),
                                 ({"checkpoint_match": False}, "checkpoints differ"),
                                 ({"changed": True}, "inputs changed")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                self.exercise(**options)


class GripperCloseProbeTests(unittest.TestCase):
    def test_closed_loop_alignment_checks_all_six_joints(self):
        teacher = np.zeros(6)
        for joint in range(6):
            state = teacher.copy()
            state[joint] = .2
            self.assertTrue(aligned_state(state, teacher)[0])
            state[joint] = .20001
            self.assertFalse(aligned_state(state, teacher)[0])
        with self.assertRaises(ValueError):
            aligned_state(np.full(6, np.nan), teacher)
        self.assertEqual(closed_loop_status(17, .4), "low")
        self.assertEqual(closed_loop_status(17, .6), "partial")

    def test_aggregate_rejects_swapped_checkpoint_identity(self):
        from summarize_gripper_close_probes import EXPECTED_MODELS, REPO, check_identity
        model, split, digest = EXPECTED_MODELS["oldmodel"]
        report = {"checkpoint": str(REPO / "outputs" / model / "model/checkpoints/010000/pretrained_model"),
                  "split_root": str(REPO / "outputs" / split), "model_sha256": digest}
        check_identity(report, "oldmodel")
        with self.assertRaisesRegex(ValueError, "preregistered"):
            check_identity(report, "fixedmodel")

    def test_close_trace_distance_excludes_any_joint(self):
        from inspect_close_window_trace import inspect
        states = np.ones((400, 6))
        trace = [{"step": step, "state_before": states[step-1].tolist()} for step in range(240, 341)]
        for row in trace:
            if 249 <= row["step"] < 265:
                row["state_before"][0] += .21
        result = inspect(trace, states, 278)
        self.assertEqual(result["retained_close_steps"], 14)
        self.assertFalse(result["state_distance_gate_passed"])

    def test_partial_runs_are_not_formal_verdicts(self):
        self.assertTrue(registered_selection("valid", 1, None, None, None))
        self.assertTrue(registered_selection("train", 1, 20, [238, 338], None))
        self.assertFalse(registered_selection("valid", 1, None, None, 1))
        self.assertFalse(registered_selection("valid", 2, None, None, None))
        self.assertFalse(registered_selection("train", 1, 20, [250, 260], None))

    def test_target_zero_is_not_degenerate_c1(self):
        targets = np.ones((100, 6))
        targets[0, 5] = 0
        targets[70:, 5] = 0
        self.assertEqual(first_close(targets), 70)
        targets[1, 5] = 0
        self.assertEqual(first_close(targets), 1)

    def test_index_bands(self):
        c = 70
        targets = np.ones((150, 6))
        targets[c:, 5] = 0
        for s, index in ((c - 1, 0), (c - 30, 29)):
            chunk = targets[s+1:s+31]
            self.assertEqual(np.flatnonzero(chunk[:, 5] < .5)[0], index)
            self.assertEqual(band_for(s, c), "close")
        self.assertEqual(band_for(c-31, c), "pre_close")
        self.assertEqual(band_for(c+30, c), "post_close")
        self.assertEqual(band_for(c+31, c), "baseline")
        self.assertEqual(sum(s+30 > len(targets)-1 for s in range(len(targets)-1)), 29)

    def test_trace_and_retained_sample_floor(self):
        self.assertEqual(trace_index(249), 248)
        self.assertEqual(band_for(trace_index(249), 278), "close")
        self.assertEqual(band_for(trace_index(278), 278), "close")
        self.assertEqual(band_for(trace_index(279), 278), "post_close")
        self.assertEqual(closed_loop_status(14, 1.), "unresolved")
        self.assertEqual(closed_loop_status(15, .8), "high")
        self.assertEqual(closed_loop_status(30, None), "unresolved")

    def test_null_models_do_not_pass_gate(self):
        targets = np.ones((150, 6))
        targets[70:, 5] = 0
        rows = []
        for s in range(10, 70):
            truth = targets[s+1:s+31]
            predictions = {"act": truth, **null_chunks(targets[s], np.zeros(6))}
            rows.append({"band": band_for(s, 70), "clipped_target_share": 0.,
                         "predictions": {n: metric_row(p, truth) for n, p in predictions.items()}})
        summaries = {n: summarize(rows, n) for n in predictions}
        self.assertEqual(summaries["always_closed"]["close"]["chunk_close_hit"], 1.)
        self.assertEqual(summaries["always_closed"]["pre_close_false_alarm"], 1.)
        self.assertEqual(summaries["state_copy"]["close"]["no_prediction"], 30)
        self.assertEqual(verdict(summaries), "teacher_forcing_learned")
        for null in ("mean", "always_closed", "state_copy"):
            changed = {**summaries, "act": summaries[null]}
            self.assertNotEqual(verdict(changed), "teacher_forcing_learned")
        self.assertEqual(verdict({**summaries, "mean": summaries["act"]}), "partial_or_unresolved")

    def test_fake_act_uses_full_chunk_and_postprocessor(self):
        import torch
        class FakeACT:
            def reset(self):
                pass
            def predict_action_chunk(self, batch):
                return batch["observation.state"].unsqueeze(1).repeat(1, 30, 1)
            def select_action(self, batch):
                raise AssertionError("queued action must not be used")
        output = predict_chunk(FakeACT(), lambda x: x, lambda x: x * 2,
                               {"observation.state": torch.ones(6)})
        self.assertEqual(output.shape, (30, 6))
        np.testing.assert_array_equal(output, np.full((30, 6), 2.))


if __name__ == "__main__":
    unittest.main()
