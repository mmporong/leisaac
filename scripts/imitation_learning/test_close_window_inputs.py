"""CPU-only contract and malformed-input tests for close-window dump comparison."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "evaluation/compare_close_window_inputs.py"
SPEC = importlib.util.spec_from_file_location("compare_close_window_inputs", SCRIPT)
compare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compare)


def trace_rows(offset: float = 0.0):
    return [
        {
            "step": step,
            "state_before": [offset] * 6,
            "requested_action": [float(step)] * 6,
            "applied_action": [float(step)] * 6,
        }
        for step in range(1, compare.EXPECTED_TRACE_STEPS + 1)
    ]


class CloseWindowInputsTest(unittest.TestCase):
    def test_completed_step_maps_to_previous_raw_frame(self):
        self.assertEqual(compare.completed_step_to_raw_frame(240), 239)
        self.assertEqual(compare.completed_step_to_raw_frame(278), 277)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            compare.completed_step_to_raw_frame(0)

    def test_trace_comparison_reports_bit_exact_and_max_difference(self):
        reference = trace_rows()
        identical = compare.compare_trace(reference, trace_rows())
        self.assertTrue(identical["bit_identical"])
        changed = trace_rows()
        changed[239]["requested_action"][2] += 0.25
        result = compare.compare_trace(reference, changed)
        self.assertFalse(result["bit_identical"])
        self.assertEqual(result["fields"]["requested_action"]["max_abs_difference"], 0.25)
        self.assertEqual(result["fields"]["requested_action"]["mismatched_scalars"], 1)

    def test_required_config_rejects_missing_on_both_sides(self):
        complete = {field: f"value-{field}" for field in compare.CONFIG_FIELDS}
        matching = compare.compare_required_config(complete, dict(complete))
        self.assertTrue(all(record["equal"] for record in matching.values()))
        incomplete = dict(complete)
        incomplete.pop("server_device")
        rejected = compare.compare_required_config(incomplete, dict(incomplete))
        self.assertFalse(rejected["server_device"]["reference_present"])
        self.assertFalse(rejected["server_device"]["candidate_present"])
        self.assertFalse(rejected["server_device"]["equal"])

    def test_output_writer_refuses_to_replace_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            compare.write_json_exclusive(output, {"version": 1})
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                compare.write_json_exclusive(output, {"version": 2})
            self.assertEqual(output.read_bytes(), original)

    def make_manifest(self, root: Path, trace: list[dict]):
        rows = []
        image_dir = root / "policy_images/rollout_001"
        image_dir.mkdir(parents=True, exist_ok=True)
        for step in compare.EXPECTED_DUMP_STEPS:
            pixels = np.full((224, 224, 3), step % 256, dtype=np.uint8)
            row = {
                "step": step,
                "observation_index": step - 1,
                "state_before": list(trace[step - 1]["state_before"]),
            }
            for camera in ("front", "wrist"):
                relative = f"policy_images/rollout_001/step_{step:06d}_{camera}.png"
                path = root / relative
                Image.fromarray(pixels, mode="RGB").save(path)
                row[camera] = {
                    "path": relative,
                    "sha256": compare.sha256_file(path),
                    "pixel_sha256": compare.sha256_bytes(pixels.tobytes(order="C")),
                    "shape": [224, 224, 3],
                    "dtype": "uint8",
                }
            rows.append(row)
        return rows

    def test_manifest_validates_exact_trace_state_checksums_and_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = trace_rows()
            manifest = self.make_manifest(root, trace)
            validated = compare.validate_manifest(root, manifest, trace)
            self.assertEqual(sorted(validated), compare.EXPECTED_DUMP_STEPS)
            self.assertEqual(validated[240]["front"]["pixels"].shape, (224, 224, 3))

    def test_manifest_rejects_observation_index_off_by_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = trace_rows()
            manifest = self.make_manifest(root, trace)
            manifest[0]["observation_index"] = 240
            with self.assertRaisesRegex(ValueError, "step-1"):
                compare.validate_manifest(root, manifest, trace)

    def test_manifest_rejects_state_mismatch_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = trace_rows()
            manifest = self.make_manifest(root, trace)
            manifest[0]["state_before"][0] = 1.0
            with self.assertRaisesRegex(ValueError, "state_before"):
                compare.validate_manifest(root, manifest, trace)
            manifest = self.make_manifest(root, trace)
            manifest[0]["front"]["path"] = "../outside.png"
            with self.assertRaisesRegex(ValueError, "must be policy_images"):
                compare.validate_manifest(root, manifest, trace)

    def test_manifest_rejects_file_and_pixel_checksum_defects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = trace_rows()
            manifest = self.make_manifest(root, trace)
            manifest[0]["front"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "file sha256"):
                compare.validate_manifest(root, manifest, trace)
            manifest = self.make_manifest(root, trace)
            manifest[0]["wrist"]["pixel_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "pixel sha256"):
                compare.validate_manifest(root, manifest, trace)


if __name__ == "__main__":
    unittest.main()
