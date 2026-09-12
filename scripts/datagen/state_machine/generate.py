"""Unified data generation script using state machines.

Selects the appropriate state machine based on --task and runs the recording loop.

Usage:
    python scripts/datagen/state_machine/generate.py \
        --task LeIsaac-SO101-PickOrange-v0 \
        --num_envs 1 --device cuda --enable_cameras \
        --record --dataset_file ./datasets/pick_orange.hdf5 --num_demos 50
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import os
import signal
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="State machine data generation for LeIsaac tasks.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument(
    "--initial_state_file",
    type=str,
    default=None,
    help="Optional torch file of policy failure states created by the ACT evaluator.",
)
parser.add_argument("--record", action="store_true", help="Whether to enable record function.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--resume", action="store_true", help="Whether to resume recording in the existing dataset file.")
parser.add_argument(
    "--num_demos", type=int, default=1, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--max_attempts",
    type=int,
    default=0,
    help="Maximum completed episodes, including failures. 0 means unlimited.",
)
parser.add_argument("--quality", action="store_true", help="Whether to enable quality render mode.")
parser.add_argument("--summary_file", type=str, default=None, help="Optional JSON report of every attempted episode.")
parser.add_argument(
    "--cube_grasp_alignment", choices=("live_jaw", "fixed_wrist"), default="live_jaw",
    help="Experimental oracle alignment; fixed_wrist has not demonstrated recovery generalization.",
)
parser.add_argument("--cube_grasp_offset", type=float, nargs=3, default=None, metavar=("X_M", "Y_M", "Z_M"))
parser.add_argument("--cube_grasp_rpy", type=float, nargs=3, default=None, metavar=("ROLL_RAD", "PITCH_RAD", "YAW_RAD"))
parser.add_argument("--use_lerobot_recorder", action="store_true", help="Whether to use lerobot recorder.")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None, help="Lerobot Dataset repository ID.")
parser.add_argument("--lerobot_dataset_fps", type=int, default=30, help="Lerobot Dataset frames per second.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac.tasks  # noqa: F401
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.datagen.state_machine import PickCubeIntoBoxStateMachine, PickOrangeStateMachine
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

# Maps gym task id → (StateMachineClass, device_type)
TASK_REGISTRY = {
    "LeIsaac-SO101-PickOrange-v0": (PickOrangeStateMachine, "so101_state_machine"),
    "LeIsaac-SO101-PickCubeIntoBox-v0": (PickCubeIntoBoxStateMachine, "so101_cube_state_machine"),
}


def _move_tensors(value, device):
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _load_initial_states(path: str | None, task_name: str):
    if path is None:
        return []
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported failure-state format: {payload.get('format_version')}")
    if payload.get("task") != task_name:
        raise ValueError(f"Failure states belong to {payload.get('task')!r}, not {task_name!r}")
    states = payload.get("states", [])
    if not states:
        raise ValueError(f"No failure states found in {path}")
    return states


def _restore_position_control_state(env, entry):
    env.reset_to(
        _move_tensors(entry["state"], env.device), None,
        seed=int(entry["seed"]), is_relative=True,
    )
    # Update after reset_to exports the preceding episode with its own seed.
    # StreamingRecorderManager uses cfg.seed for episode provenance.
    env.cfg.seed = int(entry["seed"])
    # Isaac Lab reset_to uses measured velocities as PD velocity targets.
    # Keep physical velocities but clear the targets for position control.
    for articulation in env.scene.articulations.values():
        articulation.set_joint_velocity_target(torch.zeros_like(articulation.data.joint_vel))


def _finalize_pending_episode(env, pending):
    """Flush only when the last completed episode has no following reset."""
    if pending:
        env.recorder_manager.record_pre_reset(None)


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def auto_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool):
    if hasattr(env, "termination_manager"):
        if "success" not in env.termination_manager.active_terms:
            return
        if success:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        else:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = success


def _configure_env_cfg(env_cfg, args_cli, is_direct_env, output_dir, output_file_name):
    """Configure termination and recorder settings on env_cfg."""
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.auto_terminate = True
    else:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None

    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        else:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
                assert os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file does not exist, please don't use '--resume' if you want to record a new dataset"
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                assert not os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file already exists, please use '--resume' to resume recording"
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if is_direct_env:
            env_cfg.return_success_status = False
        else:
            if not hasattr(env_cfg.terminations, "success"):
                setattr(env_cfg.terminations, "success", None)
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            )
    else:
        env_cfg.recorders = None


def _replace_recorder_manager(env, env_cfg, args_cli):
    """Replace the default recorder manager with streaming or lerobot recorder."""
    del env.recorder_manager
    if args_cli.use_lerobot_recorder:
        from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
        from leisaac.enhance.managers.lerobot_recorder_manager import (
            LeRobotRecorderManager,
        )

        dataset_cfg = LeRobotDatasetCfg(
            repo_id=args_cli.lerobot_dataset_repo_id,
            fps=args_cli.lerobot_dataset_fps,
        )
        env.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env)
    else:
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = "lzf"


def _on_episode_done(env, sm, args_cli, resume_recorded_demo_count, current_recorded_demo_count, start_record_state, success=None):
    """Handle end-of-episode logic. Returns (current_recorded_demo_count, start_record_state, should_break)."""
    try:
        if success is None:
            success = sm.check_success(env)
    except Exception as e:
        print("Success check failed:", e)
        success = False

    print("Episode success!" if success else "Episode failed!")

    if start_record_state:
        if args_cli.record:
            print("Stop Recording!!!")
        start_record_state = False

    if args_cli.record and success:
        auto_terminate(env, True)
        current_recorded_demo_count += 1
    else:
        auto_terminate(env, False)

    if (
        args_cli.record
        and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
        > current_recorded_demo_count
    ):
        current_recorded_demo_count = (
            env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
        )
        print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

    if (
        args_cli.record
        and args_cli.num_demos > 0
        and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count >= args_cli.num_demos
    ):
        print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
        return current_recorded_demo_count, start_record_state, True

    if args_cli.record and args_cli.num_demos > 0 and current_recorded_demo_count >= args_cli.num_demos:
        print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
        return current_recorded_demo_count, start_record_state, True

    return current_recorded_demo_count, start_record_state, False


def main():
    """Run a state machine in a LeIsaac manipulation environment."""
    if args_cli.max_attempts < 0 or args_cli.num_demos < 0 or args_cli.step_hz <= 0:
        raise ValueError("attempt/demo limits must be nonnegative and step_hz must be positive")
    if args_cli.initial_state_file and args_cli.num_envs != 1:
        raise ValueError("policy failure snapshots require num_envs=1")
    task_name = args_cli.task
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Task '{task_name}' is not registered in TASK_REGISTRY.\nAvailable tasks: {list(TASK_REGISTRY.keys())}"
        )
    SMClass, device = TASK_REGISTRY[task_name]

    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    initial_states = _load_initial_states(args_cli.initial_state_file, task_name)
    initial_state_index = 0

    is_direct_env = "Direct" in task_name
    _configure_env_cfg(env_cfg, args_cli, is_direct_env, output_dir, output_file_name)

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # disable gravity for every robot link prim
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    _stage = omni.usd.get_context().get_stage()
    for _prim in _stage.Traverse():
        if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)

    if args_cli.record:
        _replace_recorder_manager(env, env_cfg, args_cli)

    rate_limiter = RateLimiter(args_cli.step_hz)

    if hasattr(env, "initialize"):
        env.initialize()

    # one-time state machine setup (e.g. FK calibration)
    sm = (
        SMClass(
            grasp_alignment=args_cli.cube_grasp_alignment,
            grasp_offset=args_cli.cube_grasp_offset,
            grasp_rpy=args_cli.cube_grasp_rpy,
        )
        if SMClass is PickCubeIntoBoxStateMachine else SMClass()
    )
    sm.setup(env)

    def reset_episode() -> bool:
        nonlocal initial_state_index
        sm.reset()
        if initial_states:
            if initial_state_index >= len(initial_states):
                return False
            entry = initial_states[initial_state_index]
            initial_state_index += 1
            _restore_position_control_state(env, entry)
            print(
                f"Loaded policy failure state {initial_state_index}/{len(initial_states)} "
                f"(seed={entry['seed']}, step={entry['step']})."
            )
        else:
            env.reset()
        if hasattr(sm, "begin_episode"):
            sm.begin_episode(env)
        auto_terminate(env, False)
        return True

    reset_episode()

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count
    attempt_count = 0
    episode_results = []
    episode_needs_export = False

    start_record_state = False
    interrupted = False

    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) signal."""
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt (Ctrl+C) detected. Cleaning up resources...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        while simulation_app.is_running() and not simulation_app.is_exiting() and not interrupted:
            with torch.inference_mode():
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, device)

                if sm.is_episode_done:
                    episode_needs_export = args_cli.record
                    attempt_count += 1
                    episode_results.append({
                        "attempt": attempt_count,
                        "source_seed": initial_states[initial_state_index - 1]["seed"] if initial_states else env_cfg.seed,
                        "source_step": initial_states[initial_state_index - 1]["step"] if initial_states else None,
                        "success": sm.check_success(env),
                        "final_cube_xyz": env.scene["cube"].data.root_pos_w.detach().cpu().tolist()
                        if "cube" in env.scene.rigid_objects else None,
                    })
                    current_recorded_demo_count, start_record_state, should_break = _on_episode_done(
                        env, sm, args_cli, resume_recorded_demo_count, current_recorded_demo_count, start_record_state,
                        success=episode_results[-1]["success"],
                    )
                    if should_break:
                        break
                    if args_cli.max_attempts > 0 and attempt_count >= args_cli.max_attempts:
                        print(f"Reached max_attempts={args_cli.max_attempts}. Exiting the app.")
                        break
                    if not reset_episode():
                        print(f"All {len(initial_states)} policy failure states were attempted.")
                        break
                    episode_needs_export = False
                else:
                    if not start_record_state:
                        if args_cli.record:
                            print("Start Recording!!!")
                        start_record_state = True

                    sm.pre_step(env)
                    actions = sm.get_action(env)
                    env.step(actions)
                    sm.advance()

                if rate_limiter:
                    rate_limiter.sleep(env)

            if interrupted:
                break
    except Exception as e:
        import traceback

        print(f"\n[ERROR] An error occurred: {e}\n")
        traceback.print_exc()
        print("[INFO] Cleaning up resources...")
        raise
    finally:
        _finalize_pending_episode(env, episode_needs_export)
        if args_cli.summary_file:
            from pathlib import Path

            summary_path = Path(args_cli.summary_file).expanduser()
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps({
                "task": task_name,
                "initial_state_file": args_cli.initial_state_file,
                "cube_grasp_alignment": args_cli.cube_grasp_alignment,
                "cube_grasp_offset_m": args_cli.cube_grasp_offset,
                "cube_grasp_rpy_rad": args_cli.cube_grasp_rpy,
                "attempts": len(episode_results),
                "successes": sum(item["success"] for item in episode_results),
                "interrupted": interrupted,
                "results": episode_results,
            }, indent=2), encoding="utf-8")
        signal.signal(signal.SIGINT, original_sigint_handler)
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
