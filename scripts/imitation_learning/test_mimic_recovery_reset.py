"""CPU regression tests for MimicGen recovery reset wiring."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimic"))
from recovery_reset import (  # noqa: E402
    SOURCE_TASK,
    TARGET_TASK,
    bounded_recovery_env_loop,
    load_recovery_snapshot,
    temporary_recovery_reset,
    validate_recovery_options,
)


def _state() -> dict:
    return {
        "articulation": {
            "robot": {
                "root_pose": torch.zeros((1, 7)),
                "root_velocity": torch.zeros((1, 6)),
                "joint_position": torch.ones((1, 6)),
                "joint_velocity": torch.full((1, 6), 0.25),
            }
        },
        "rigid_object": {
            name: {"root_pose": torch.zeros((1, 7)), "root_velocity": torch.zeros((1, 6))}
            for name in ("cube", "box_target")
        },
    }


def _write_snapshot(path: Path, *, task: str = SOURCE_TASK, states: list | None = None) -> None:
    torch.save(
        {
            "format_version": 1,
            "task": task,
            "states": states
            if states is not None
            else [{"seed": 3100, "step": 600, "state": _state()}],
        },
        path,
    )


class FakeArticulation:
    def __init__(self) -> None:
        self.data = SimpleNamespace(joint_vel=torch.full((1, 6), 0.25))
        self.velocity_target = None

    def set_joint_velocity_target(self, target: torch.Tensor) -> None:
        self.velocity_target = target.clone()


class FakeEnv:
    def __init__(self) -> None:
        self.device = "cpu"
        self.cfg = SimpleNamespace(seed=42)
        self.articulation = FakeArticulation()
        self.episode = SimpleNamespace(seed=None)
        self.recorder_manager = SimpleNamespace(
            get_episode=lambda env_id: self.episode if env_id == 0 else None
        )
        self.scene = SimpleNamespace(
            articulations={"robot": self.articulation},
            write_data_to_sim=self._write_data_to_sim,
        )
        self.calls = []
        self.write_count = 0
        self.num_envs = 1
        self.action_space = SimpleNamespace(shape=(1, 1))
        self.steps = []
        self.closed = False

    def reset(self, **kwargs):
        self.calls.append(("normal", kwargs))
        return "normal"

    def reset_to(self, state, env_ids, **kwargs):
        self.calls.append(("recovery", state, env_ids, kwargs))
        return "recovery"

    def _write_data_to_sim(self) -> None:
        self.write_count += 1

    def step(self, actions: torch.Tensor) -> None:
        self.steps.append(actions.clone())

    def close(self) -> None:
        self.closed = True


class MimicRecoveryResetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "snapshot.pt"
        _write_snapshot(self.path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_loads_exact_single_cube_snapshot_with_provenance(self) -> None:
        snapshot = load_recovery_snapshot(self.path, TARGET_TASK)
        self.assertEqual((snapshot.seed, snapshot.step), (3100, 600))
        self.assertEqual(snapshot.source_task, SOURCE_TASK)
        self.assertEqual(len(snapshot.sha256), 64)
        self.assertEqual(snapshot.path, self.path.resolve())
        metadata = snapshot.manifest_metadata(reset_calls=2)
        self.assertEqual(metadata["reset_calls"], 2)
        self.assertIsNone(metadata["reset_to_seed"])
        self.assertTrue(metadata["reused_for_each_reset"])

    def test_temporary_reset_reuses_snapshot_and_restores_original_method(self) -> None:
        snapshot = load_recovery_snapshot(self.path, TARGET_TASK)
        env = FakeEnv()
        original_reset = env.reset
        env_ids = torch.tensor([0])
        with temporary_recovery_reset(env, snapshot) as stats:
            self.assertEqual(env.reset(env_ids=env_ids), "recovery")
            self.assertEqual(env.reset(env_ids=env_ids), "recovery")
            self.assertEqual(stats["reset_calls"], 2)
            self.assertEqual(env.cfg.seed, 3100)
            self.assertEqual(env.episode.seed, 3100)
            torch.testing.assert_close(
                env.articulation.velocity_target,
                torch.zeros_like(env.articulation.data.joint_vel),
            )
            self.assertEqual(env.write_count, 2)
            self.assertNotIn("seed", env.calls[0][3])
            self.assertEqual(env.calls[0][3], {"is_relative": True})
        self.assertEqual(env.reset, original_reset)
        self.assertEqual(env.reset(marker=True), "normal")

    def test_restores_original_reset_when_recovery_or_body_raises(self) -> None:
        snapshot = load_recovery_snapshot(self.path, TARGET_TASK)
        for failure_site in ("reset", "body"):
            with self.subTest(failure_site=failure_site):
                env = FakeEnv()
                original_reset = env.reset
                if failure_site == "reset":
                    env.reset_to = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    with temporary_recovery_reset(env, snapshot):
                        if failure_site == "reset":
                            env.reset(env_ids=torch.tensor([0]))
                        else:
                            raise RuntimeError("boom")
                self.assertEqual(env.reset, original_reset)

    def test_rejects_invalid_snapshot_and_cli_combinations(self) -> None:
        validate_recovery_options(None, "successes", None, 2)
        invalid_options = [
            (self.path, "successes", 1, 1),
            (self.path, "attempts", 2, 1),
            (self.path, "attempts", 1, 2),
            (None, "attempts", 1, 1),
        ]
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_recovery_options(*options)

        bad_task = self.root / "bad_task.pt"
        _write_snapshot(bad_task, task="OtherTask-v0")
        with self.assertRaisesRegex(ValueError, "task mapping"):
            load_recovery_snapshot(bad_task, TARGET_TASK)
        multiple = self.root / "multiple.pt"
        entries = [{"seed": 1, "step": index, "state": _state()} for index in range(2)]
        _write_snapshot(multiple, states=entries)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_recovery_snapshot(multiple, TARGET_TASK)

        malformed = self.root / "malformed.pt"
        state = _state()
        state["articulation"]["robot"]["joint_position"] = torch.zeros((1, 5))
        _write_snapshot(malformed, states=[{"seed": 1, "step": 1, "state": state}])
        with self.assertRaisesRegex(ValueError, "joint_position"):
            load_recovery_snapshot(malformed, TARGET_TASK)

    def test_bounded_loop_stops_before_prefetched_second_reset(self) -> None:
        snapshot = load_recovery_snapshot(self.path, TARGET_TASK)
        env = FakeEnv()
        runtime = SimpleNamespace(num_attempts=0)
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        reset_queue = asyncio.Queue()
        action_queue = asyncio.Queue()

        async def generator() -> None:
            for _ in range(2):
                await reset_queue.put(0)
                await reset_queue.join()
                await action_queue.put((0, torch.tensor([1.0])))
                await action_queue.join()
                runtime.num_attempts += 1

        task = event_loop.create_task(generator())
        try:
            with temporary_recovery_reset(env, snapshot) as stats:
                bounded_recovery_env_loop(
                    env, reset_queue, action_queue, event_loop, [task], runtime
                )
            self.assertEqual(runtime.num_attempts, 1)
            self.assertEqual(stats["reset_calls"], 1)
            self.assertEqual(len(env.steps), 1)
            self.assertEqual(reset_queue.qsize(), 1)
            self.assertTrue(env.closed)
        finally:
            task.cancel()
            event_loop.run_until_complete(
                asyncio.gather(task, return_exceptions=True)
            )
            event_loop.close()
            asyncio.set_event_loop(None)

    def test_bounded_loop_propagates_generator_task_error(self) -> None:
        env = FakeEnv()
        runtime = SimpleNamespace(num_attempts=0)
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        reset_queue = asyncio.Queue()
        action_queue = asyncio.Queue()

        async def broken_generator() -> None:
            await asyncio.sleep(0)
            raise RuntimeError("generator failed")

        task = event_loop.create_task(broken_generator())
        try:
            with self.assertRaisesRegex(RuntimeError, "generator failed"):
                bounded_recovery_env_loop(
                    env, reset_queue, action_queue, event_loop, [task], runtime
                )
            self.assertTrue(env.closed)
        finally:
            event_loop.close()
            asyncio.set_event_loop(None)


if __name__ == "__main__":
    unittest.main()
