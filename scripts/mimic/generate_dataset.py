# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.
"""


"""Launch Isaac Sim Simulator first."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)
parser.add_argument(
    "--datagen-seed",
    type=int,
    default=None,
    help="Override the MimicGen random seed so additional runs do not repeat an earlier dataset.",
)
parser.add_argument(
    "--generation-mode",
    choices=("successes", "attempts"),
    default="successes",
    help="Stop after the requested number of successes (default) or bounded attempts.",
)
parser.add_argument(
    "--initial-state-file",
    type=Path,
    default=None,
    help="A format-version 1 ACT failure snapshot containing exactly one cube-task state.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
)
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument(
    "--task_type",
    type=str,
    default=None,
    help=(
        "Specify task type. If your annotated dataset is recorded with keyboard, you should set it to 'keyboard',"
        " otherwise not to set it and keep default value None."
    ),
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument("--render-width", type=int, default=None, help="Opt-in front/wrist camera render width.")
parser.add_argument("--render-height", type=int, default=None, help="Opt-in front/wrist camera render height.")
parser.add_argument(
    "--observation-image-size",
    type=int,
    default=None,
    help="Opt-in square HDF5 policy image size (84 through 256).",
)
parser.add_argument(
    "--successful-only",
    action="store_true",
    help="Export successful demonstrations only.",
)
parser.add_argument(
    "--max-attempts",
    type=int,
    default=None,
    help="Finish the current episode and stop after this many total attempts.",
)
parser.add_argument(
    "--min-free-gib",
    type=float,
    default=0.0,
    help="Stop with the partial HDF5 intact when output storage falls below this threshold.",
)
parser.add_argument(
    "--progress-file",
    type=Path,
    default=None,
    help="Optional atomic JSON progress file for bounded generation.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import inspect
import random

import gymnasium as gym
import isaaclab_mimic.envs  # noqa: F401
import isaaclab_mimic.datagen.generation as generation_runtime
import numpy as np
import omni
import torch
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import DatasetExportMode

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_mimic.datagen.generation import (
    env_loop,
    setup_async_generation,
    setup_env_config,
)
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths
from leisaac.utils.env_utils import get_task_type
from leisaac.tasks.pick_cube_into_box.mdp.release_state import release_criteria_metadata
from leisaac.tasks.pick_cube_into_box.mdp.terminations import cube_released_in_box
from recovery_reset import (
    bounded_recovery_env_loop,
    load_recovery_snapshot,
    temporary_recovery_reset,
    validate_recovery_options,
)
from runtime_options import (
    atomic_write_json,
    bounded_normal_env_loop,
    configure_successful_only,
    configure_visual_options,
    require_safe_task,
    validate_runtime_options,
)

import leisaac  # noqa: F401


def main():
    num_envs = args_cli.num_envs
    guarded_runtime = validate_runtime_options(
        render_width=args_cli.render_width,
        render_height=args_cli.render_height,
        observation_image_size=args_cli.observation_image_size,
        max_attempts=args_cli.max_attempts,
        min_free_gib=args_cli.min_free_gib,
        progress_file=args_cli.progress_file,
        initial_state_file=args_cli.initial_state_file,
    )
    if guarded_runtime:
        num_envs = 1
    validate_recovery_options(
        args_cli.initial_state_file,
        args_cli.generation_mode,
        args_cli.generation_num_trials,
        num_envs,
    )

    output_path = Path(args_cli.output_file).expanduser().resolve()
    failed_output_path = output_path.with_name(f"{output_path.stem}_failed{output_path.suffix}")
    manifest_path = output_path.with_suffix(".generation.json")
    existing_outputs = [
        path for path in (output_path, failed_output_path, manifest_path) if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(f"refusing to overwrite existing datasets: {existing_outputs}")

    # Setup output paths and get env name
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task
    if task_name:
        task_name = args_cli.task.split(":")[-1]
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)
    runtime_options_requested = (
        guarded_runtime
        or args_cli.render_width is not None
        or args_cli.observation_image_size is not None
        or args_cli.successful_only
    )
    if runtime_options_requested:
        require_safe_task(env_name)

    recovery_snapshot = None
    if args_cli.initial_state_file is not None:
        recovery_snapshot = load_recovery_snapshot(args_cli.initial_state_file, env_name)

    # Configure environment
    generation_num_trials = 1 if recovery_snapshot is not None else args_cli.generation_num_trials
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=generation_num_trials,
    )
    env_cfg.datagen_config.generation_guarantee = args_cli.generation_mode == "successes"
    if args_cli.datagen_seed is not None:
        env_cfg.datagen_config.seed = args_cli.datagen_seed
        env_cfg.seed = args_cli.datagen_seed
    datagen_seed = int(env_cfg.datagen_config.seed)
    setattr(env_cfg, "task_type", get_task_type(task_name, args_cli.task_type))
    configure_visual_options(
        env_cfg,
        env_name,
        render_width=args_cli.render_width,
        render_height=args_cli.render_height,
        observation_image_size=args_cli.observation_image_size,
    )
    configure_successful_only(
        env_cfg,
        args_cli.successful_only,
        DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
    )

    # create environment
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    # check if the mimic API from this environment contains decprecated signatures
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        omni.log.warn(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    # set seed for generation
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # reset before starting
    env.reset()

    # Setup and run async data generation
    async_components = setup_async_generation(
        env=env,
        num_envs=num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
    )

    target_count = env.cfg.datagen_config.generation_num_trials
    reset_context = (
        temporary_recovery_reset(env, recovery_snapshot)
        if recovery_snapshot is not None
        else nullcontext({"reset_calls": 0})
    )
    stop_reason = None
    loop_error = None
    with reset_context as reset_stats:
        try:
            if recovery_snapshot is not None:
                bounded_recovery_env_loop(
                    env,
                    async_components["reset_queue"],
                    async_components["action_queue"],
                    async_components["event_loop"],
                    async_components["tasks"],
                    generation_runtime,
                    max_attempts=target_count,
                )
            elif guarded_runtime:
                stop_reason = bounded_normal_env_loop(
                    env,
                    async_components["reset_queue"],
                    async_components["action_queue"],
                    async_components["event_loop"],
                    async_components["tasks"],
                    generation_runtime,
                    target_successes=target_count,
                    max_attempts=args_cli.max_attempts,
                    min_free_gib=args_cli.min_free_gib,
                    output_path=output_path,
                    progress_file=args_cli.progress_file,
                )
            else:
                env_loop(
                    env,
                    async_components["reset_queue"],
                    async_components["action_queue"],
                    async_components["info_pool"],
                    async_components["event_loop"],
                )
        except BaseException as error:
            if not guarded_runtime:
                raise
            loop_error = error
        finally:
            for task in async_components["tasks"]:
                task.cancel()
            async_components["event_loop"].run_until_complete(
                asyncio.gather(*async_components["tasks"], return_exceptions=True)
            )

    completed_count = (
        generation_runtime.num_success
        if args_cli.generation_mode == "successes"
        else generation_runtime.num_attempts
    )
    completed = completed_count >= target_count and loop_error is None
    if stop_reason == "max_attempts_reached" and not completed:
        completed = False
    manifest = {
        "completed": completed,
        "task": env_name,
        "input_file": str(Path(args_cli.input_file).expanduser().resolve()),
        "output_file": str(output_path),
        "failed_output_file": str(failed_output_path),
        "generation_mode": args_cli.generation_mode,
        "generation_num_trials": target_count,
        "datagen_seed": datagen_seed,
        "num_envs": num_envs,
        "successful_demos": generation_runtime.num_success,
        "failed_demos": generation_runtime.num_failures,
        "attempts": generation_runtime.num_attempts,
        "source_snapshot": recovery_snapshot.manifest_metadata(reset_stats["reset_calls"])
        if recovery_snapshot is not None
        else None,
    }
    if success_term.func is cube_released_in_box:
        manifest["success_criteria"] = release_criteria_metadata(success_term.params)
    if guarded_runtime:
        manifest["stop_reason"] = stop_reason or (
            type(loop_error).__name__
            if loop_error is not None
            else "target_reached"
            if completed
            else "target_not_reached"
        )
        manifest["render_dimensions"] = {
            camera: [getattr(env_cfg.scene, camera).width, getattr(env_cfg.scene, camera).height]
            for camera in ("front", "wrist")
        }
        manifest["observation_image_size"] = args_cli.observation_image_size
        manifest["successful_only"] = args_cli.successful_only
        manifest["max_attempts"] = args_cli.max_attempts
        manifest["min_free_gib"] = args_cli.min_free_gib
        atomic_write_json(manifest_path, manifest)
    elif completed:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if loop_error is not None:
        raise loop_error
    if not completed:
        raise RuntimeError(
            f"generation stopped before target: {completed_count}/{target_count} "
            f"{args_cli.generation_mode}; stop_reason={manifest.get('stop_reason', 'target_not_reached')}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # close sim app
    simulation_app.close()
