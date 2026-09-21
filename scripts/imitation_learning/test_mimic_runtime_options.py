"""CPU regression tests for opt-in MimicGen runtime controls."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimic"))
from runtime_options import (  # noqa: E402
    PICK_CUBE_MIMIC_TASK,
    _resize_uint8_nhwc,
    bounded_normal_env_loop,
    configure_successful_only,
    configure_visual_options,
    refresh_reset_observations,
    validate_runtime_options,
)


class FakeCamera:
    def __init__(self, env) -> None:
        self.env = env
        self.reset_calls = 0
        self.update_calls = []

    def reset(self) -> None:
        self.reset_calls += 1

    def update(self, dt, force_recompute=False) -> None:
        # record how many renders had happened when the buffer was forced, to pin render -> update order
        self.update_calls.append((dt, force_recompute, self.env.render_calls))


class FakeScene:
    def __init__(self, env) -> None:
        self.cameras = {"front": FakeCamera(env), "wrist": FakeCamera(env)}
        self.state_value = torch.zeros(1, 3)
        self.drift_on_render = False

    def __getitem__(self, name):
        return self.cameras[name]

    def get_state(self, is_relative: bool):
        return {"rigid_object": {"cube": {"root_pose": self.state_value.clone()}}}


class FakeEnv:
    def __init__(self) -> None:
        self.device = "cpu"
        self.num_envs = 1
        self.action_space = SimpleNamespace(shape=(1, 1))
        self.reset_calls = 0
        self.steps = []
        self.closed = False
        self.render_calls = 0
        self.scene = FakeScene(self)
        self.sim = SimpleNamespace(render=self._render)
        self.obs_buf = {"policy": {"front": torch.zeros(1)}}
        self.compute_calls = 0
        self.observation_manager = SimpleNamespace(compute=self._compute)
        self.refresh_log = []

    def _render(self) -> None:
        self.render_calls += 1
        if self.scene.drift_on_render:
            self.scene.state_value += 1.0

    def _compute(self):
        self.compute_calls += 1
        return {"policy": {"front": torch.full((1,), float(self.compute_calls))}}

    def reset(self, *, env_ids) -> None:
        self.reset_calls += 1
        self.refresh_log.append(("reset", self.render_calls))

    def step(self, actions) -> None:
        self.steps.append(actions.clone())
        self.refresh_log.append(("step", self.render_calls))

    def close(self) -> None:
        self.closed = True


class MimicRuntimeOptionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_defaults_preserve_unbounded_runtime(self) -> None:
        self.assertFalse(
            validate_runtime_options(
                render_width=None,
                render_height=None,
                observation_image_size=None,
                max_attempts=None,
                min_free_gib=0,
                progress_file=None,
                initial_state_file=None,
            )
        )

    def test_option_validation(self) -> None:
        base = dict(
            render_width=None,
            render_height=None,
            observation_image_size=None,
            max_attempts=None,
            min_free_gib=0,
            progress_file=None,
            initial_state_file=None,
        )
        for override in (
            {"render_width": 640},
            {"render_height": 480},
            {"render_width": 320, "render_height": 240},
            {"observation_image_size": 83},
            {"observation_image_size": 257},
            {"max_attempts": 0},
            {"min_free_gib": -1},
            {"min_free_gib": float("nan")},
            {"min_free_gib": float("inf")},
            {"max_attempts": 1, "initial_state_file": Path("snapshot.pt")},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_runtime_options(**(base | override))
        self.assertTrue(validate_runtime_options(**(base | {"max_attempts": 2})))
        self.assertFalse(
            validate_runtime_options(
                **(base | {"render_width": 640, "render_height": 480, "observation_image_size": 84})
            )
        )

    def test_visual_config_keeps_native_resolution_and_resizes_policy_terms(self) -> None:
        source = torch.arange(1 * 12 * 16 * 3, dtype=torch.uint8).reshape(1, 12, 16, 3)
        calls = []

        def observe(env, *, sensor_cfg, marker):
            calls.append((env, sensor_cfg, marker))
            return source

        cfg = SimpleNamespace(
            scene=SimpleNamespace(
                front=SimpleNamespace(width=640, height=480),
                wrist=SimpleNamespace(width=640, height=480),
            ),
            observations=SimpleNamespace(
                policy=SimpleNamespace(
                    front=SimpleNamespace(func=observe, params={"sensor_cfg": "front", "marker": 1}),
                    wrist=SimpleNamespace(func=observe, params={"sensor_cfg": "wrist", "marker": 2}),
                )
            ),
        )
        configure_visual_options(
            cfg,
            PICK_CUBE_MIMIC_TASK,
            render_width=None,
            render_height=None,
            observation_image_size=84,
        )
        self.assertEqual((cfg.scene.front.width, cfg.scene.front.height), (640, 480))
        expected = cv2.resize(source[0].numpy(), (84, 84), interpolation=cv2.INTER_AREA)
        for camera in ("front", "wrist"):
            resized = getattr(cfg.observations.policy, camera).func("env")
            self.assertEqual((resized.shape, resized.dtype, resized.device.type), ((1, 84, 84, 3), torch.uint8, "cpu"))
            np.testing.assert_array_equal(resized[0].numpy(), expected)
        self.assertEqual(calls, [("env", "front", 1), ("env", "wrist", 2)])

        configure_visual_options(
            cfg,
            PICK_CUBE_MIMIC_TASK,
            render_width=640,
            render_height=480,
            observation_image_size=None,
        )
        with self.assertRaisesRegex(ValueError, "only supported"):
            configure_visual_options(
                cfg,
                "OtherTask-v0",
                render_width=640,
                render_height=480,
                observation_image_size=None,
            )

    def test_resize_supports_dataset_and_policy_sizes(self) -> None:
        source = torch.arange(2 * 12 * 16 * 3, dtype=torch.uint8).reshape(2, 12, 16, 3)
        for size in (84, 224):
            with self.subTest(size=size):
                resized = _resize_uint8_nhwc(source, size)
                self.assertEqual(resized.shape, (2, size, size, 3))
                expected = cv2.resize(source[1].numpy(), (size, size), interpolation=cv2.INTER_AREA)
                np.testing.assert_array_equal(resized[1].numpy(), expected)

    def test_successful_only_updates_datagen_and_recorder_before_make(self) -> None:
        cfg = SimpleNamespace(
            datagen_config=SimpleNamespace(generation_keep_failed=True, keep_failed=True),
            recorders=SimpleNamespace(dataset_export_mode="separate"),
        )
        configure_successful_only(cfg, True, "success-only")
        self.assertFalse(cfg.datagen_config.generation_keep_failed)
        self.assertFalse(cfg.datagen_config.keep_failed)
        self.assertEqual(cfg.recorders.dataset_export_mode, "success-only")

    def _run_loop(self, generator_factory, *, runtime=None, **kwargs):
        env = FakeEnv()
        runtime = runtime or SimpleNamespace(num_success=0, num_failures=0, num_attempts=0)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        resets = asyncio.Queue()
        actions = asyncio.Queue()
        task = loop.create_task(generator_factory(resets, actions, runtime))
        try:
            result = bounded_normal_env_loop(
                env,
                resets,
                actions,
                loop,
                [task],
                runtime,
                target_successes=kwargs.pop("target_successes", 1),
                max_attempts=kwargs.pop("max_attempts", None),
                min_free_gib=kwargs.pop("min_free_gib", 0),
                output_path=self.root / "partial.hdf5",
                **kwargs,
            )
            return result, env, runtime, resets, task
        finally:
            task.cancel()
            loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
            loop.close()
            asyncio.set_event_loop(None)

    def test_target_stops_before_prefetched_reset_or_action(self) -> None:
        async def generator(resets, actions, runtime):
            for _ in range(2):
                await resets.put(0)
                await resets.join()
                await actions.put((0, torch.tensor([1.0])))
                await actions.join()
                runtime.num_success += 1
                runtime.num_attempts += 1

        reason, env, runtime, resets, _task = self._run_loop(generator)
        self.assertEqual(reason, "target_successes_reached")
        self.assertEqual((runtime.num_success, runtime.num_attempts), (1, 1))
        self.assertEqual((env.reset_calls, len(env.steps)), (1, 1))
        self.assertEqual(resets.qsize(), 1)
        self.assertTrue(env.closed)

    def test_max_attempts_finishes_current_episode_and_writes_progress(self) -> None:
        progress = self.root / "progress.json"

        async def generator(resets, actions, runtime):
            for _ in range(3):
                await resets.put(0)
                await resets.join()
                for _ in range(2):
                    await actions.put((0, torch.tensor([1.0])))
                    await actions.join()
                runtime.num_failures += 1
                runtime.num_attempts += 1

        reason, env, runtime, resets, _task = self._run_loop(
            generator,
            target_successes=5,
            max_attempts=2,
            progress_file=progress,
        )
        self.assertEqual(reason, "max_attempts_reached")
        self.assertEqual((runtime.num_attempts, len(env.steps), env.reset_calls), (2, 4, 2))
        self.assertEqual(resets.qsize(), 1)
        payload = json.loads(progress.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["stop_reason"], "max_attempts_reached")
        self.assertEqual(payload["attempts"], 2)

    def test_disk_threshold_keeps_partial_file_and_reports_failure(self) -> None:
        partial = self.root / "partial.hdf5"
        partial.write_bytes(b"partial")
        progress = self.root / "disk-progress.json"

        async def generator(resets, actions, runtime):
            await resets.put(0)
            await resets.join()
            await actions.put((0, torch.tensor([1.0])))
            await actions.join()

        env = FakeEnv()
        runtime = SimpleNamespace(num_success=0, num_failures=0, num_attempts=0)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        resets, actions = asyncio.Queue(), asyncio.Queue()
        task = loop.create_task(generator(resets, actions, runtime))
        try:
            with self.assertRaisesRegex(RuntimeError, "insufficient disk space"):
                bounded_normal_env_loop(
                    env,
                    resets,
                    actions,
                    loop,
                    [task],
                    runtime,
                    target_successes=1,
                    max_attempts=None,
                    min_free_gib=2,
                    output_path=partial,
                    progress_file=progress,
                    disk_usage=lambda _path: SimpleNamespace(free=1024**3),
                )
            self.assertEqual(partial.read_bytes(), b"partial")
            self.assertEqual(json.loads(progress.read_text())["status"], "failed")
            self.assertTrue(env.closed)
        finally:
            task.cancel()
            loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
            loop.close()
            asyncio.set_event_loop(None)

    def test_disk_space_is_checked_no_less_often_than_every_60_steps(self) -> None:
        checks = []

        async def generator(resets, actions, runtime):
            await resets.put(0)
            await resets.join()
            for _ in range(61):
                await actions.put((0, torch.tensor([1.0])))
                await actions.join()
            runtime.num_success += 1
            runtime.num_attempts += 1

        reason, env, _runtime, _resets, _task = self._run_loop(
            generator,
            min_free_gib=1,
            disk_usage=lambda path: checks.append(path) or SimpleNamespace(free=10 * 1024**3),
        )
        self.assertEqual(reason, "target_successes_reached")
        self.assertEqual(len(env.steps), 61)
        self.assertEqual(len(checks), 2)
        self.assertTrue(all(path == self.root for path in checks))

    def test_generator_error_and_clean_exit_are_not_busy_loops(self) -> None:
        async def broken(_resets, _actions, _runtime):
            await asyncio.sleep(0)
            raise RuntimeError("generator failed")

        with self.assertRaisesRegex(RuntimeError, "generator failed"):
            self._run_loop(broken)

        async def stopped(_resets, _actions, _runtime):
            return

        with self.assertRaisesRegex(RuntimeError, "stopped before"):
            self._run_loop(stopped)


class ResetRefreshTest(unittest.TestCase):
    def test_zero_frames_is_a_no_op(self) -> None:
        env = FakeEnv()
        original = env.obs_buf
        refresh_reset_observations(env, 0)
        self.assertEqual(env.render_calls, 0)
        self.assertIs(env.obs_buf, original)
        self.assertEqual(env.compute_calls, 0)

    def test_refresh_renders_forces_camera_update_and_recomputes_obs(self) -> None:
        env = FakeEnv()
        refresh_reset_observations(env, 4)
        self.assertEqual(env.render_calls, 4)
        for camera in env.scene.cameras.values():
            self.assertEqual(camera.reset_calls, 1)
            self.assertEqual(camera.update_calls, [(0.0, True, 4)])
        self.assertEqual(env.compute_calls, 1)
        self.assertEqual(float(env.obs_buf["policy"]["front"][0]), 1.0)

    def test_refresh_rejects_physical_state_change(self) -> None:
        env = FakeEnv()
        env.scene.drift_on_render = True
        original = env.obs_buf
        with self.assertRaises(RuntimeError):
            refresh_reset_observations(env, 1)
        self.assertIs(env.obs_buf, original)
        self.assertEqual(env.compute_calls, 0)
        with self.assertRaises(ValueError):
            refresh_reset_observations(env, -1)
        multi = FakeEnv()
        multi.num_envs = 2
        with self.assertRaises(ValueError):
            refresh_reset_observations(multi, 1)

    def test_reset_render_frames_makes_runtime_guarded(self) -> None:
        base = dict(
            render_width=None, render_height=None, observation_image_size=None,
            max_attempts=None, min_free_gib=0.0, progress_file=None, initial_state_file=None,
        )
        self.assertFalse(validate_runtime_options(**base))
        self.assertTrue(validate_runtime_options(**base, reset_render_frames=4))
        with self.assertRaises(ValueError):
            validate_runtime_options(**base, reset_render_frames=-1)
        # the recovery loop has no refresh hook, so the flag must not be silently ignored there
        with self.assertRaises(ValueError):
            validate_runtime_options(**(base | {"initial_state_file": Path("snapshot.pt")}), reset_render_frames=4)


class BoundedLoopResetRefreshTest(unittest.TestCase):
    setUp = MimicRuntimeOptionsTest.setUp
    tearDown = MimicRuntimeOptionsTest.tearDown
    _run_loop = MimicRuntimeOptionsTest._run_loop

    def test_loop_refreshes_after_every_reset_when_requested(self) -> None:
        async def generator(resets, actions, runtime):
            for _ in range(2):
                await resets.put(0)
                await resets.join()
                await actions.put((0, torch.tensor([1.0])))
                await actions.join()
                runtime.num_failures += 1
                runtime.num_attempts += 1

        reason, env, runtime, _resets, _task = self._run_loop(
            generator, target_successes=1, max_attempts=2, reset_render_frames=3,
        )
        self.assertEqual(reason, "max_attempts_reached")
        self.assertEqual(env.reset_calls, 2)
        self.assertEqual(env.render_calls, 6)
        self.assertEqual(env.compute_calls, 2)
        # every reset is followed by exactly 3 renders before the next step is applied
        self.assertEqual(env.refresh_log, [("reset", 0), ("step", 3), ("reset", 3), ("step", 6)])

    def test_loop_default_keeps_legacy_no_refresh(self) -> None:
        async def generator(resets, actions, runtime):
            await resets.put(0)
            await resets.join()
            await actions.put((0, torch.tensor([1.0])))
            await actions.join()
            runtime.num_success += 1
            runtime.num_attempts += 1

        _reason, env, _runtime, _resets, _task = self._run_loop(generator)
        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(env.render_calls, 0)
        self.assertEqual(env.compute_calls, 0)


if __name__ == "__main__":
    unittest.main()
