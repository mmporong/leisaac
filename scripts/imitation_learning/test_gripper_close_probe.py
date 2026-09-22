"""Preregistered chunk-probe boundaries and degenerate-predictor checks."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from probe_gripper_close_offline import (band_for, closed_loop_status, first_close,
    metric_row, null_chunks, predict_chunk, registered_selection, summarize, trace_index, verdict)


class GripperCloseProbeTests(unittest.TestCase):
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
