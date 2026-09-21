"""Opt-in runtime controls for bounded MimicGen dataset generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch


PICK_CUBE_MIMIC_TASK = "LeIsaac-SO101-PickCubeIntoBox-Mimic-v0"
CAMERA_NAMES = ("front", "wrist")


def validate_runtime_options(
    *,
    render_width: int | None,
    render_height: int | None,
    observation_image_size: int | None,
    max_attempts: int | None,
    min_free_gib: float,
    progress_file: Path | None,
    initial_state_file: Path | None,
    reset_render_frames: int = 0,
) -> bool:
    """Validate optional controls and return whether the bounded loop is needed."""
    if reset_render_frames < 0:
        raise ValueError("--reset-render-frames cannot be negative")
    if (render_width is None) != (render_height is None):
        raise ValueError("--render-width and --render-height must be provided together")
    if render_width is not None and (render_width, render_height) != (640, 480):
        raise ValueError("the safe render override is limited to 640x480")
    if observation_image_size is not None and not 84 <= observation_image_size <= 256:
        raise ValueError("--observation-image-size must be between 84 and 256")
    if max_attempts is not None and max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError("--min-free-gib must be finite and non-negative")

    guarded = (
        max_attempts is not None or min_free_gib > 0 or progress_file is not None or reset_render_frames > 0
    )
    if guarded and initial_state_file is not None:
        raise ValueError("bounded runtime guards cannot be combined with --initial-state-file")
    return guarded


def require_safe_task(env_name: str) -> None:
    """Reject runtime overrides outside the validated cube Mimic task."""
    if env_name != PICK_CUBE_MIMIC_TASK:
        raise ValueError(
            f"MimicGen runtime overrides are only supported for {PICK_CUBE_MIMIC_TASK}, "
            f"got {env_name}"
        )


def _resize_uint8_nhwc(images: torch.Tensor, size: int) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"camera observation must be a tensor, got {type(images).__name__}")
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            "camera observation must be uint8 NHWC with three channels, "
            f"got shape={tuple(images.shape)} dtype={images.dtype}"
        )
    device = images.device
    resized = [
        cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        for image in images.detach().cpu().numpy()
    ]
    return torch.from_numpy(np.stack(resized)).to(device=device)


def wrap_policy_camera_observations(env_cfg: Any, size: int) -> None:
    """Resize only front/wrist policy observations while native sensors stay unchanged."""
    for camera_name in CAMERA_NAMES:
        term = getattr(env_cfg.observations.policy, camera_name)
        original_func = term.func
        original_params = dict(term.params)

        def resized_observation(env: Any, _func=original_func, _params=original_params) -> torch.Tensor:
            return _resize_uint8_nhwc(_func(env, **_params), size)

        term.func = resized_observation
        term.params = {}


def configure_visual_options(
    env_cfg: Any,
    env_name: str,
    *,
    render_width: int | None,
    render_height: int | None,
    observation_image_size: int | None,
) -> None:
    """Apply the allow-listed native render and recorded observation sizes."""
    if render_width is None and observation_image_size is None:
        return
    require_safe_task(env_name)
    if render_width is not None:
        for camera_name in CAMERA_NAMES:
            camera_cfg = getattr(env_cfg.scene, camera_name)
            camera_cfg.width = render_width
            camera_cfg.height = render_height
    if observation_image_size is not None:
        wrap_policy_camera_observations(env_cfg, observation_image_size)


def configure_successful_only(env_cfg: Any, successful_only: bool, succeeded_only_mode: Any) -> None:
    """Keep datagen and recorder failed-episode settings consistent."""
    if not successful_only:
        return
    env_cfg.datagen_config.generation_keep_failed = False
    if hasattr(env_cfg.datagen_config, "keep_failed"):
        env_cfg.datagen_config.keep_failed = False
    env_cfg.recorders.dataset_export_mode = succeeded_only_mode


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON artifact atomically within its destination directory."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _raise_generator_task_error(tasks: list[Any]) -> None:
    for task in tasks:
        if not task.done():
            continue
        if task.cancelled():
            raise RuntimeError("MimicGen generator task was cancelled before generation completed")
        error = task.exception()
        if error is not None:
            raise error
        raise RuntimeError("MimicGen generator task stopped before generation completed")


def _counts(generation_runtime: Any) -> dict[str, int]:
    return {
        "successful_demos": int(generation_runtime.num_success),
        "failed_demos": int(generation_runtime.num_failures),
        "attempts": int(generation_runtime.num_attempts),
    }


def _physical_state_digest(env: Any) -> str:
    digest = hashlib.sha256()
    state = env.scene.get_state(is_relative=True)
    for group, entities in sorted(state.items()):
        for entity, fields in sorted(entities.items()):
            for field, value in sorted(fields.items()):
                array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype="<f4")
                digest.update(f"{group}/{entity}/{field}:{array.shape}".encode())
                digest.update(array.tobytes())
    return digest.hexdigest()


def refresh_reset_observations(env: Any, render_frames: int) -> None:
    """Re-render the reset scene without stepping physics and recompute ``env.obs_buf``.

    After ``env.reset`` Isaac Lab returns camera observations from the renderer's previous
    output (the last frame of the previous attempt). The pre-step recorder stores
    ``env.obs_buf`` verbatim, so without this refresh recorded frame 0 shows a stale scene.
    Mirrors ``refresh_reset_images`` in scripts/evaluation/lerobot_act_so101.py.
    """
    if render_frames < 0:
        raise ValueError("render_frames cannot be negative")
    if render_frames == 0:
        return
    if env.num_envs != 1:
        raise ValueError("reset refresh resets every camera timer; it supports exactly one environment")
    before = _physical_state_digest(env)
    for _ in range(render_frames):
        env.sim.render()
    for camera_name in CAMERA_NAMES:
        camera = env.scene[camera_name]
        camera.reset()
        camera.update(0.0, force_recompute=True)
    if _physical_state_digest(env) != before:
        raise RuntimeError("physical state changed during render-only reset refresh")
    env.obs_buf = env.observation_manager.compute()


def bounded_normal_env_loop(
    env: Any,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    asyncio_event_loop: asyncio.AbstractEventLoop,
    generator_tasks: list[Any],
    generation_runtime: Any,
    *,
    target_successes: int,
    max_attempts: int | None,
    min_free_gib: float,
    output_path: Path,
    progress_file: Path | None = None,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    reset_render_frames: int = 0,
) -> str:
    """Run one upstream async generator without prefetching beyond a stop condition."""
    if env.num_envs != 1:
        raise ValueError("bounded MimicGen runtime requires exactly one environment")
    if target_successes <= 0:
        raise ValueError("target_successes must be positive")
    if max_attempts is not None and max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    output_parent = output_path.expanduser().resolve().parent
    env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
    steps_since_disk_check = 60
    last_attempts = -1

    def publish(status: str, stop_reason: str | None = None) -> None:
        if progress_file is not None:
            atomic_write_json(
                progress_file,
                {"status": status, "stop_reason": stop_reason, **_counts(generation_runtime)},
            )

    def stop_reason() -> str | None:
        if generation_runtime.num_success >= target_successes:
            return "target_successes_reached"
        if max_attempts is not None and generation_runtime.num_attempts >= max_attempts:
            return "max_attempts_reached"
        return None

    publish("running")
    try:
        with torch.inference_mode():
            while True:
                reason = stop_reason()
                if reason is not None:
                    publish("completed", reason)
                    return reason

                while env_action_queue.qsize() != 1:
                    asyncio_event_loop.run_until_complete(asyncio.sleep(0))
                    reason = stop_reason()
                    if reason is not None:
                        publish("completed", reason)
                        return reason
                    _raise_generator_task_error(generator_tasks)
                    while not env_reset_queue.empty():
                        reason = stop_reason()
                        if reason is not None:
                            publish("completed", reason)
                            return reason
                        env_id_tensor[0] = env_reset_queue.get_nowait()
                        env.reset(env_ids=env_id_tensor)
                        refresh_reset_observations(env, reset_render_frames)
                        env_reset_queue.task_done()

                reason = stop_reason()
                if reason is not None:
                    publish("completed", reason)
                    return reason

                if min_free_gib > 0 and steps_since_disk_check >= 60:
                    free_gib = disk_usage(output_parent).free / (1024**3)
                    steps_since_disk_check = 0
                    if free_gib < min_free_gib:
                        raise RuntimeError(
                            f"insufficient disk space for partial dataset: {free_gib:.2f} GiB free "
                            f"under {output_parent}, requires at least {min_free_gib:.2f} GiB"
                        )

                env_id, action = asyncio_event_loop.run_until_complete(env_action_queue.get())
                actions = torch.zeros(env.action_space.shape)
                actions[env_id] = action
                env.step(actions)
                env_action_queue.task_done()
                steps_since_disk_check += 1

                if generation_runtime.num_attempts != last_attempts:
                    last_attempts = int(generation_runtime.num_attempts)
                    publish("running")
    except BaseException as error:
        publish("failed", type(error).__name__)
        raise
    finally:
        env.close()
