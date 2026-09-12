"""CPU-only tests for configurable LeRobot ACT image sizes and conversion actions."""

import ast
import importlib.util
import pickle
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest

import cv2
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
CONVERTER_PATH = SCRIPT_DIR / "convert_hdf5_to_lerobot.py"
SERVER_PATH = SCRIPT_DIR / "serve_lerobot_act.py"
EVALUATOR_PATH = SCRIPT_DIR.parent / "evaluation/lerobot_act_so101.py"


def load_module_with_stubs(name: str, path: Path, stubs: dict[str, ModuleType]):
    previous = {module_name: sys.modules.get(module_name) for module_name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for module_name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module


def load_converter():
    h5py = ModuleType("h5py")
    h5py.File = object
    h5py.Group = object
    lerobot = ModuleType("lerobot")
    datasets = ModuleType("lerobot.datasets")
    dataset_module = ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_module.LeRobotDataset = object
    return load_module_with_stubs(
        "test_convert_hdf5_to_lerobot",
        CONVERTER_PATH,
        {
            "h5py": h5py,
            "lerobot": lerobot,
            "lerobot.datasets": datasets,
            "lerobot.datasets.lerobot_dataset": dataset_module,
        },
    )


def load_server():
    lerobot = ModuleType("lerobot")
    policies = ModuleType("lerobot.policies")
    policies.make_pre_post_processors = object
    act = ModuleType("lerobot.policies.act")
    act.ACTPolicy = object
    return load_module_with_stubs(
        "test_serve_lerobot_act",
        SERVER_PATH,
        {
            "lerobot": lerobot,
            "lerobot.policies": policies,
            "lerobot.policies.act": act,
        },
    )


def load_evaluator_helpers():
    tree = ast.parse(EVALUATOR_PATH.read_text(encoding="utf-8"))
    names = {"validate_policy_image_size", "resize_policy_image"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"MAX_POLICY_IMAGE_SIZE": 256, "cv2": cv2, "np": np, "torch": torch}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(EVALUATOR_PATH), "exec"),
        namespace,
    )
    return namespace


class ArrayDataset:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape = self.value.shape
        self.dtype = self.value.dtype

    def __len__(self):
        return len(self.value)

    def __getitem__(self, key):
        return self.value[key]


class Demo(dict):
    name = "/data/demo_0"
    attrs = {"success": True}


class ImageSizeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.converter = load_converter()
        cls.server = load_server()
        cls.evaluator = load_evaluator_helpers()

    def make_request(self, image_size: int, *, as_bytes: bool = True):
        image = np.arange(image_size * image_size * 3, dtype=np.uint8).reshape(
            image_size, image_size, 3
        )
        image_value = image.tobytes() if as_bytes else image
        return {"state": [0.0] * 6, "front": image_value, "wrist": image_value}

    def test_server_builds_84_and_224_observations(self):
        for image_size in (84, 224):
            observation = self.server.build_observation(
                self.make_request(image_size), torch.device("cpu"), image_size
            )
            self.assertEqual(observation["observation.state"].shape, (1, 6))
            self.assertEqual(
                observation["observation.images.front"].shape,
                (1, 3, image_size, image_size),
            )
            self.assertEqual(
                observation["observation.images.wrist"].shape,
                (1, 3, image_size, image_size),
            )

    def test_server_default_remains_84_and_rejects_bad_bytes_or_shape(self):
        observation = self.server.build_observation(self.make_request(84), torch.device("cpu"))
        self.assertEqual(observation["observation.images.front"].shape, (1, 3, 84, 84))
        request = self.make_request(224)
        request["front"] = request["front"][:-1]
        with self.assertRaisesRegex(ValueError, "invalid front byte count"):
            self.server.build_observation(request, torch.device("cpu"), 224)
        with self.assertRaisesRegex(ValueError, "invalid wrist image"):
            self.server.build_observation(
                {
                    **self.make_request(224, as_bytes=False),
                    "wrist": np.zeros((224, 223, 3), np.uint8),
                },
                torch.device("cpu"),
                224,
            )

    def test_checkpoint_image_shape_must_match_cli(self):
        config_84 = SimpleNamespace(
            input_features={
                "observation.images.front": SimpleNamespace(shape=(3, 84, 84)),
                "observation.images.wrist": {"shape": [3, 84, 84]},
            }
        )
        self.server.validate_checkpoint_image_size(config_84, 84)
        with self.assertRaisesRegex(ValueError, "does not match --image-size 224"):
            self.server.validate_checkpoint_image_size(config_84, 224)

    def test_image_sizes_are_positive_and_bounded_by_protocol_limit(self):
        for image_size in (84, 224, 256):
            self.assertEqual(self.converter.validate_image_size(image_size), image_size)
            self.assertEqual(self.server.validate_image_size(image_size), image_size)
            self.assertEqual(self.evaluator["validate_policy_image_size"](image_size), image_size)
        for image_size in (0, 257):
            with self.assertRaises(ValueError):
                self.converter.validate_image_size(image_size)
            with self.assertRaises(ValueError):
                self.server.validate_image_size(image_size)
            with self.assertRaises(ValueError):
                self.evaluator["validate_policy_image_size"](image_size)

        largest_request = {
            "command": "predict",
            "state": [0.0] * 6,
            "front": bytes(256 * 256 * 3),
            "wrist": bytes(256 * 256 * 3),
        }
        self.assertLess(
            len(pickle.dumps(largest_request, protocol=pickle.HIGHEST_PROTOCOL)),
            self.server.MAX_MESSAGE_BYTES,
        )

    def test_evaluator_resizes_native_image_to_224(self):
        native = (
            torch.arange(480 * 640 * 3, dtype=torch.int32)
            .remainder(256)
            .to(torch.uint8)
            .reshape(480, 640, 3)
        )
        resized = self.evaluator["resize_policy_image"](native, 224)
        self.assertEqual(resized.shape, (224, 224, 3))
        self.assertEqual(resized.dtype, np.uint8)

    def test_conversion_features_include_84_and_224_metadata(self):
        for image_size in (84, 224):
            features = self.converter.build_features(image_size)
            self.assertEqual(
                features["observation.images.front"]["shape"],
                (image_size, image_size, 3),
            )
            self.assertEqual(
                features["observation.images.wrist"]["shape"],
                (image_size, image_size, 3),
            )
        provenance = self.converter.build_provenance(
            Path("/tmp/input.hdf5"), "a" * 64, "next_observed", 224, ["demo_2"], [3]
        )
        self.assertEqual(provenance["image_size"], 224)
        self.assertEqual(provenance["action_source"], "next_observed")
        self.assertEqual(provenance["episode_names"], ["demo_2"])
        self.assertEqual(provenance["frame_counts"], [3])
        self.assertEqual(provenance["episodes"], [{"name": "demo_2", "frame_count": 3}])

    def test_stored_and_next_observed_actions(self):
        joint_pos = np.arange(18, dtype=np.float32).reshape(3, 6)
        stored = np.full((3, 6), 7.0, dtype=np.float32)
        demo = Demo(
            {
                "obs/joint_pos": ArrayDataset(joint_pos),
                "actions": ArrayDataset(stored),
            }
        )
        np.testing.assert_array_equal(self.converter.get_demo_actions(demo, "stored"), stored)
        expected_next = np.stack((joint_pos[1], joint_pos[2], joint_pos[2]))
        np.testing.assert_array_equal(
            self.converter.get_demo_actions(demo, "next_observed"), expected_next
        )

    def test_next_observed_requires_matching_raw_action_length(self):
        demo = Demo(
            {
                "obs/joint_pos": ArrayDataset(np.zeros((3, 6), dtype=np.float32)),
                "actions": ArrayDataset(np.zeros((2, 8), dtype=np.float32)),
            }
        )
        with self.assertRaisesRegex(ValueError, "actions length does not match observations"):
            self.converter.get_demo_actions(demo, "next_observed")

    def test_stored_mode_preserves_8d_rejection_and_224_validation(self):
        demo = Demo(
            {
                "obs/joint_pos": ArrayDataset(np.zeros((2, 6), dtype=np.float32)),
                "actions": ArrayDataset(np.zeros((2, 8), dtype=np.float32)),
                "obs/front": ArrayDataset(np.zeros((2, 224, 224, 3), dtype=np.uint8)),
                "obs/wrist": ArrayDataset(np.zeros((2, 224, 224, 3), dtype=np.uint8)),
            }
        )
        with self.assertRaisesRegex(ValueError, "unexpected actions shape"):
            self.converter.validate_demo(demo, 1.0, 224, "stored")
        self.assertEqual(self.converter.validate_demo(demo, 1.0, 224, "next_observed"), 2)
        for threshold in (float("nan"), float("inf"), 0):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                self.converter.validate_demo(demo, threshold, 224, "next_observed")
        demo["obs/joint_pos"] = ArrayDataset(np.zeros((1, 6), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "at least two frames"):
            self.converter.validate_demo(demo, 1.0, 224, "next_observed")

    def test_next_observed_rejects_nonfinite_raw_ik_actions(self):
        for value in (float("nan"), float("inf")):
            demo = Demo({"obs/joint_pos": ArrayDataset(np.zeros((2, 6))),
                         "actions": ArrayDataset(np.full((2, 8), value))})
            with self.assertRaisesRegex(ValueError, "non-finite raw actions"):
                self.converter.get_demo_actions(demo, "next_observed")


if __name__ == "__main__":
    unittest.main()
