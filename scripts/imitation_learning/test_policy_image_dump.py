"""CPU tests for lossless, correctly indexed opt-in policy image recording."""
import hashlib
import ast
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from policy_image_dump import should_dump, validate_dump_options, write_policy_images


class PolicyImageDumpTests(unittest.TestCase):
    def test_evaluator_dump_is_same_pre_action_observation_as_trace(self):
        path = Path(__file__).resolve().parents[1] / "evaluation/lerobot_act_so101.py"
        tree = ast.parse(path.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        dump = next(node for node in calls if ast.unparse(node.func) == "write_policy_images")
        send = min((node for node in calls if ast.unparse(node.func) == "send_request"
                    and node.lineno > dump.lineno), key=lambda node: node.lineno)
        physics = next(node for node in calls if ast.unparse(node.func) == "env.step")
        self.assertLess(dump.lineno, send.lineno)
        self.assertLess(send.lineno, physics.lineno)
        self.assertEqual(ast.unparse(dump.args[2]), "step + 1")
        self.assertEqual(ast.unparse(dump.args[3]), "policy_obs['joint_pos'][0].detach().cpu().numpy()")
        self.assertEqual(ast.unparse(dump.args[4]), "{'front': front, 'wrist': wrist}")
        state = next(node for node in ast.walk(main) if isinstance(node, ast.Assign)
                     and any(ast.unparse(target) == "state_before" for target in node.targets))
        self.assertLess(state.lineno, physics.lineno)
        self.assertEqual(ast.unparse(state.value), "policy_obs['joint_pos'][0].detach().clone()")

    def test_non_224_dump_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                write_policy_images(Path(folder), 1, 1, np.zeros(6),
                                    {name: np.zeros((84, 84, 3), dtype=np.uint8) for name in ("front", "wrist")})

    def test_default_disabled_creates_no_interval(self):
        self.assertIsNone(validate_dump_options(0, None, 1200, 60))
        self.assertFalse(should_dump(240, 0, None))

    def test_interval_is_completed_step_not_zero_based_index(self):
        bounds = validate_dump_options(1, [240, 340], 1200, 420)
        steps = [step + 1 for step in range(1200) if should_dump(step + 1, 1, bounds)]
        self.assertEqual(steps, list(range(240, 341)))
        self.assertFalse(should_dump(239, 1, bounds))
        self.assertFalse(should_dump(341, 1, bounds))
        self.assertTrue(should_dump(240, 2, bounds))
        self.assertFalse(should_dump(241, 2, bounds))

    def test_invalid_options_rejected_before_isaac(self):
        for every, span, trace in ((-1, None, 420), (0, [240, 340], 420),
                                   (1, [0, 340], 420), (1, [340, 240], 420),
                                   (1, [240, 1201], 1201), (1, [240, 340], 300)):
            with self.assertRaises(ValueError):
                validate_dump_options(every, span, 1200, trace)

    def test_exact_rgb_bytes_state_and_filename_preserved(self):
        image = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
        state = np.arange(6, dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = write_policy_images(root, 1, 249, state, {"front": image, "wrist": image.copy()})
            self.assertEqual(row["step"], 249)
            self.assertEqual(row["observation_index"], 248)
            self.assertEqual(row["state_before"], state.tolist())
            for name in ("front", "wrist"):
                path = root / row[name]["path"]
                self.assertEqual(path.name, f"step_000249_{name}.png")
                with Image.open(path) as decoded:
                    np.testing.assert_array_equal(np.asarray(decoded), image)
                self.assertEqual(row[name]["pixel_sha256"], hashlib.sha256(image.tobytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                write_policy_images(root, 1, 249, state, {"front": image, "wrist": image})


if __name__ == "__main__":
    unittest.main()
