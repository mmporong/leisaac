"""Pure-Python recovery snapshot validation and temporary reset wiring."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterator

import torch


SOURCE_TASK = "LeIsaac-SO101-PickCubeIntoBox-v0"
TARGET_TASK = "LeIsaac-SO101-PickCubeIntoBox-Mimic-v0"


@dataclass(frozen=True)
class RecoverySnapshot:
    path: Path
    sha256: str
    source_task: str
    seed: int
    step: int
    state: dict[str, Any]

    def manifest_metadata(self, reset_calls: int) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "source_task": self.source_task,
            "seed": self.seed,
            "step": self.step,
            "reset_calls": reset_calls,
            "reused_for_each_reset": True,
            "reset_to_seed": None,
            "env_cfg_seed": self.seed,
        }


def validate_recovery_options(
    initial_state_file: Path | None,
    generation_mode: str,
    generation_num_trials: int | None,
    num_envs: int,
) -> None:
    if generation_mode not in {"successes", "attempts"}:
        raise ValueError(f"unsupported generation mode: {generation_mode}")
    if initial_state_file is None:
        if generation_mode != "successes":
            raise ValueError("attempts mode requires --initial-state-file")
        return
    if generation_mode != "attempts":
        raise ValueError("recovery snapshots require --generation-mode attempts")
    if num_envs != 1:
        raise ValueError("recovery snapshots require --num_envs 1")
    if generation_num_trials not in (None, 1):
        raise ValueError("a single recovery snapshot requires --generation_num_trials 1")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_shape(value: Any, expected: tuple[int, ...], name: str) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        raise ValueError(f"snapshot {name} must be a tensor with shape {expected}, got {shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"snapshot {name} contains non-finite values")


def load_recovery_snapshot(path: Path, target_task: str) -> RecoverySnapshot:
    path = path.expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported recovery snapshot format; expected format_version=1")
    if payload.get("task") != SOURCE_TASK or target_task != TARGET_TASK:
        raise ValueError(
            f"recovery snapshot task mapping must be {SOURCE_TASK!r} -> {TARGET_TASK!r}, "
            f"got {payload.get('task')!r} -> {target_task!r}"
        )
    entries = payload.get("states")
    if not isinstance(entries, list) or len(entries) != 1:
        count = len(entries) if isinstance(entries, list) else 0
        raise ValueError(f"recovery snapshot file must contain exactly one state, got {count}")
    entry = entries[0]
    if not isinstance(entry, dict) or not isinstance(entry.get("state"), dict):
        raise ValueError("recovery snapshot entry is malformed")
    try:
        seed = int(entry["seed"])
        step = int(entry["step"])
        state = entry["state"]
        robot = state["articulation"]["robot"]
        rigid_objects = state["rigid_object"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("recovery snapshot is missing seed, step, robot, cube, or box state") from error

    for key, shape in {
        "root_pose": (1, 7),
        "root_velocity": (1, 6),
        "joint_position": (1, 6),
        "joint_velocity": (1, 6),
    }.items():
        _require_shape(robot.get(key), shape, f"articulation/robot/{key}")
    for object_name in ("cube", "box_target"):
        if object_name not in rigid_objects:
            raise ValueError(f"recovery snapshot is missing rigid_object/{object_name}")
        _require_shape(
            rigid_objects[object_name].get("root_pose"),
            (1, 7),
            f"rigid_object/{object_name}/root_pose",
        )
        _require_shape(
            rigid_objects[object_name].get("root_velocity"),
            (1, 6),
            f"rigid_object/{object_name}/root_velocity",
        )
    return RecoverySnapshot(path, _sha256_file(path), SOURCE_TASK, seed, step, state)


def _move_tensors(value: Any, device: Any) -> Any:
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _raise_generator_task_error(tasks: list[Any]) -> None:
    for task in tasks:
        if not task.done():
            continue
        if task.cancelled():
            raise RuntimeError("recovery generator task was cancelled before the attempt completed")
        error = task.exception()
        if error is not None:
            raise error
        raise RuntimeError("recovery generator task stopped before the attempt completed")


def bounded_recovery_env_loop(
    env: Any,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    asyncio_event_loop: asyncio.AbstractEventLoop,
    generator_tasks: list[Any],
    generation_runtime: Any,
    max_attempts: int = 1,
) -> None:
    """Run exactly the requested recovery attempts without prefetching another reset."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
    try:
        with torch.inference_mode():
            while generation_runtime.num_attempts < max_attempts:
                while env_action_queue.qsize() != env.num_envs:
                    asyncio_event_loop.run_until_complete(asyncio.sleep(0))
                    if generation_runtime.num_attempts >= max_attempts:
                        return
                    _raise_generator_task_error(generator_tasks)
                    while not env_reset_queue.empty():
                        env_id_tensor[0] = env_reset_queue.get_nowait()
                        env.reset(env_ids=env_id_tensor)
                        env_reset_queue.task_done()

                actions = torch.zeros(env.action_space.shape)
                for _ in range(env.num_envs):
                    env_id, action = asyncio_event_loop.run_until_complete(env_action_queue.get())
                    actions[env_id] = action
                env.step(actions)
                for _ in range(env.num_envs):
                    env_action_queue.task_done()
    finally:
        env.close()


@contextmanager
def temporary_recovery_reset(env: Any, snapshot: RecoverySnapshot) -> Iterator[dict[str, int]]:
    """Temporarily route ``env.reset`` to one reusable recovery snapshot.

    ``reset_to`` deliberately receives no seed: MimicGen's source selection RNG
    remains controlled by its datagen seed. ``cfg.seed`` is changed only after
    restoration so recorder metadata identifies the ACT snapshot seed.
    """
    had_instance_reset = "reset" in vars(env)
    previous_instance_reset = vars(env).get("reset")
    stats = {"reset_calls": 0}

    def recovery_reset(*_args: Any, env_ids: Any = None, **_kwargs: Any) -> Any:
        result = env.reset_to(
            _move_tensors(snapshot.state, env.device),
            env_ids,
            is_relative=True,
        )
        env.cfg.seed = snapshot.seed
        env.recorder_manager.get_episode(0).seed = snapshot.seed
        for articulation in env.scene.articulations.values():
            articulation.set_joint_velocity_target(torch.zeros_like(articulation.data.joint_vel))
        env.scene.write_data_to_sim()
        stats["reset_calls"] += 1
        return result

    env.reset = recovery_reset
    try:
        yield stats
    finally:
        if had_instance_reset:
            env.reset = previous_instance_reset
        else:
            del env.reset
