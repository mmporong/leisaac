"""CPU-only regressions for state-machine stable-release observation wiring."""

import ast
import math
import os
from pathlib import Path
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_SOURCE = REPO_ROOT / "source/leisaac/leisaac/datagen/state_machine/base.py"
PICK_SOURCE = REPO_ROOT / "source/leisaac/leisaac/datagen/state_machine/pick_cube_into_box.py"
GENERATE_SOURCE = REPO_ROOT / "scripts/datagen/state_machine/generate.py"


def _method_calls(source: Path, method_name: str) -> list[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return [node.func.id for node in ast.walk(method) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]


def _load_pick_machine(released_results: list[bool]):
    tree = ast.parse(PICK_SOURCE.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    calls = []

    class SceneEntityCfg:
        def __init__(self, name):
            self.name = name

    def cube_released_in_box(env, **kwargs):
        calls.append(tuple(config.name for config in kwargs.values()))
        return torch.tensor([released_results[len(calls) - 1]])

    namespace = {
        "StateMachineBase": object,
        "SceneEntityCfg": SceneEntityCfg,
        "cube_released_in_box": cube_released_in_box,
        "cube_placed_in_box": lambda *args, **kwargs: torch.tensor([False]),
        "object_grasped": lambda *args, **kwargs: torch.tensor([False]),
        "math": math,
        "os": os,
        "torch": torch,
        "_GRASP_OFFSET": (-0.012, 0.020, 0.090),
    }
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(PICK_SOURCE), "exec"), namespace)
    return namespace[class_node.name](), calls


def _statement_call(statement) -> tuple[str, str] | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    function = statement.value.func
    if not isinstance(function, ast.Attribute) or not isinstance(function.value, ast.Name):
        return None
    return function.value.id, function.attr


class StateMachineSuccessObservationTest(unittest.TestCase):
    def test_generate_observes_each_environment_step_before_advancing(self):
        tree = ast.parse(GENERATE_SOURCE.read_text(encoding="utf-8"))
        step_paths = []
        for node in ast.walk(tree):
            for _, value in ast.iter_fields(node):
                if not isinstance(value, list):
                    continue
                for index, statement in enumerate(value):
                    if _statement_call(statement) == ("env", "step"):
                        step_paths.append([_statement_call(item) for item in value[index : index + 3]])
        self.assertEqual(step_paths, [[("env", "step"), ("sm", "observe_step"), ("sm", "advance")]])

    def test_observe_step_accumulates_and_end_check_approves_same_tick(self):
        machine, calls = _load_pick_machine([True, True])

        machine.observe_step(object())
        success = machine.check_success(object())

        self.assertTrue(success)
        self.assertEqual(
            calls,
            [
                ("cube", "box_target", "robot"),
                ("cube", "box_target", "robot"),
            ],
        )

    def test_episode_start_and_diagnostics_use_stateless_placement(self):
        self.assertIn("cube_placed_in_box", _method_calls(PICK_SOURCE, "begin_episode"))
        self.assertNotIn("cube_released_in_box", _method_calls(PICK_SOURCE, "begin_episode"))
        self.assertIn("cube_placed_in_box", _method_calls(PICK_SOURCE, "_print_diagnostics"))
        self.assertNotIn("cube_released_in_box", _method_calls(PICK_SOURCE, "_print_diagnostics"))
        self.assertEqual(_method_calls(PICK_SOURCE, "observe_step").count("cube_released_in_box"), 1)
        self.assertEqual(_method_calls(PICK_SOURCE, "check_success").count("cube_released_in_box"), 1)

    def test_base_class_documents_post_step_observation_hook(self):
        tree = ast.parse(BASE_SOURCE.read_text(encoding="utf-8"))
        class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        methods = {node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("observe_step", methods)
        self.assertIsNotNone(ast.get_docstring(methods["observe_step"]))


if __name__ == "__main__":
    unittest.main()
