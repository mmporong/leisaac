"""CPU-only regression tests for the fine-tuning processor compatibility layer."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_act_fixed_normalizer import fixed_processor_kwargs, preserve_checkpoint_normalizers


class FixedNormalizerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.temp.name)
        for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
            (self.checkpoint / filename).write_text("{}")

    def tearDown(self):
        self.temp.cleanup()

    def kwargs(self):
        return {
            "policy_cfg": SimpleNamespace(type="act"),
            "pretrained_path": self.checkpoint,
            "dataset_stats": {"sentinel": 1},
            "preprocessor_overrides": {
                "device_processor": {"device": "cpu"},
                "normalizer_processor": {"stats": {"sentinel": 1}, "features": "keep"},
            },
            "postprocessor_overrides": {
                "unnormalizer_processor": {"stats": {"sentinel": 1}, "norm_map": "keep"},
            },
        }

    def test_removes_only_stats_without_mutating_input(self):
        original = self.kwargs()
        result = fixed_processor_kwargs(original)
        self.assertNotIn("dataset_stats", result)
        self.assertNotIn("stats", result["preprocessor_overrides"]["normalizer_processor"])
        self.assertNotIn("stats", result["postprocessor_overrides"]["unnormalizer_processor"])
        self.assertEqual(result["preprocessor_overrides"]["device_processor"], {"device": "cpu"})
        self.assertEqual(result["preprocessor_overrides"]["normalizer_processor"]["features"], "keep")
        self.assertEqual(result["postprocessor_overrides"]["unnormalizer_processor"]["norm_map"], "keep")
        self.assertIn("dataset_stats", original)
        self.assertIn("stats", original["preprocessor_overrides"]["normalizer_processor"])

    def test_requires_act_and_local_processor_files(self):
        for key, value in (("policy_cfg", SimpleNamespace(type="other")), ("pretrained_path", None)):
            kwargs = self.kwargs()
            kwargs[key] = value
            with self.assertRaises(ValueError):
                fixed_processor_kwargs(kwargs)
        (self.checkpoint / "policy_postprocessor.json").unlink()
        with self.assertRaises(ValueError):
            fixed_processor_kwargs(self.kwargs())

    def test_factory_restored_on_success_and_error(self):
        original = lambda **kwargs: kwargs
        module = SimpleNamespace(make_pre_post_processors=original)
        with preserve_checkpoint_normalizers(module):
            result = module.make_pre_post_processors(**self.kwargs())
            self.assertNotIn("dataset_stats", result)
        self.assertIs(module.make_pre_post_processors, original)
        with self.assertRaisesRegex(RuntimeError, "test error"):
            with preserve_checkpoint_normalizers(module):
                raise RuntimeError("test error")
        self.assertIs(module.make_pre_post_processors, original)


if __name__ == "__main__":
    unittest.main()
