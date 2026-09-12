"""Evaluate a LeRobot ACT SO101 policy in Isaac Lab through a local inference server."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="LeIsaac-SO101-PickCubeIntoBox-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num-rollouts", type=int, default=5)
parser.add_argument("--horizon", type=int, default=1200)
parser.add_argument("--seed", type=int, default=2000)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--video-count", type=int, default=2)
parser.add_argument("--server-python", type=Path, default=Path.home() / "miniforge3/envs/lerobot/bin/python")
parser.add_argument(
    "--server-script",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "imitation_learning/serve_lerobot_act.py",
)
parser.add_argument("--server-host", default="127.0.0.1")
parser.add_argument("--server-port", type=int, default=5557)
parser.add_argument("--server-timeout", type=float, default=60.0)
parser.add_argument("--server-device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--server-seed", type=int, default=0)
parser.add_argument("--n-action-steps", type=int, default=None)
parser.add_argument("--render-width", type=int, default=320)
parser.add_argument("--render-height", type=int, default=240)
parser.add_argument("--policy-image-size", type=int, default=84)
parser.add_argument(
    "--lift-threshold-m",
    type=float,
    default=0.02,
    help="Cube lift threshold in meters used only to classify rollout outcomes.",
)
parser.add_argument(
    "--failure-state-file",
    type=Path,
    default=None,
    help="Optional torch file receiving simulator states sampled from failed rollouts.",
)
parser.add_argument(
    "--failure-state-interval",
    type=int,
    default=120,
    help="Step interval for failure-state snapshots. Snapshots are kept only when the rollout fails.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import hashlib
import json
import math
import pickle
import random
import socket
import struct
import subprocess
import time
import traceback

import gymnasium as gym
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

import leisaac  # noqa: F401


HEADER = struct.Struct("!Q")
MAX_MESSAGE_BYTES = 1_048_576


def classify_rollout_outcome(
    success: bool,
    max_cube_lift_m: float,
    final_cube_lift_m: float,
    lift_threshold_m: float,
) -> str:
    """Classify observed lift behavior without changing the task success criterion.

    ``low_after_lift`` only means the final observed lift is below the threshold
    after reaching it earlier; it does not prove that the cube was dropped.
    """
    if success:
        return "success"
    if max_cube_lift_m < lift_threshold_m:
        return "no_lift"
    if final_cube_lift_m < lift_threshold_m:
        return "low_after_lift"
    return "lifted_not_completed"


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("ACT server disconnected")
        chunks.extend(chunk)
    return bytes(chunks)


def send_request(connection: socket.socket, request: dict):
    payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(HEADER.pack(len(payload)) + payload)
    response_size = HEADER.unpack(receive_exact(connection, HEADER.size))[0]
    if response_size > MAX_MESSAGE_BYTES:
        raise ValueError(f"ACT server response is too large: {response_size} bytes")
    return pickle.loads(receive_exact(connection, response_size))


def connect_to_server(process: subprocess.Popen) -> socket.socket:
    deadline = time.monotonic() + args_cli.server_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ACT server exited with code {process.returncode}")
        try:
            connection = socket.create_connection(
                (args_cli.server_host, args_cli.server_port),
                timeout=1.0,
            )
            connection.settimeout(args_cli.server_timeout)
            return connection
        except OSError:
            time.sleep(0.25)
    raise TimeoutError("timed out waiting for ACT server")


def resize_policy_image(image: torch.Tensor) -> np.ndarray:
    image_array = image.detach().cpu().numpy()
    return cv2.resize(
        image_array,
        (args_cli.policy_image_size, args_cli.policy_image_size),
        interpolation=cv2.INTER_AREA,
    )


def tensors_to_cpu(value):
    if isinstance(value, dict):
        return {key: tensors_to_cpu(item) for key, item in value.items()}
    return value.detach().cpu()


def capture_state(env, seed: int, step: int) -> dict:
    return {
        "seed": seed,
        "step": step,
        "state": tensors_to_cpu(env.scene.get_state(is_relative=True)),
    }


def scene_state_sha256(state: dict) -> str:
    """Hash physical state, including orientations and velocities, in a stable order."""
    digest = hashlib.sha256()
    for group, entities in sorted(state.items()):
        for entity, fields in sorted(entities.items()):
            for field, value in sorted(fields.items()):
                array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype="<f4")
                digest.update(f"{group}/{entity}/{field}:{array.shape}".encode())
                digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    if args_cli.num_rollouts <= 0 or args_cli.horizon <= 0:
        raise ValueError("num-rollouts and horizon must be positive")
    if args_cli.video_count < 0:
        raise ValueError("video-count cannot be negative")
    if args_cli.render_width <= 0 or args_cli.render_height <= 0:
        raise ValueError("render dimensions must be positive")
    if args_cli.policy_image_size != 84:
        raise ValueError("this ACT server expects policy-image-size=84")
    if args_cli.n_action_steps is not None and args_cli.n_action_steps <= 0:
        raise ValueError("n-action-steps must be positive")
    if args_cli.failure_state_interval <= 0:
        raise ValueError("failure-state-interval must be positive")
    if not math.isfinite(args_cli.lift_threshold_m) or args_cli.lift_threshold_m <= 0:
        raise ValueError("lift-threshold-m must be a finite positive number")

    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    server_command = [
        str(args_cli.server_python.expanduser().resolve()),
        str(args_cli.server_script.expanduser().resolve()),
        "--checkpoint",
        str(args_cli.checkpoint.expanduser().resolve()),
        "--host",
        args_cli.server_host,
        "--port",
        str(args_cli.server_port),
        "--device",
        args_cli.server_device,
        "--seed",
        str(args_cli.server_seed),
    ]
    if args_cli.n_action_steps is not None:
        server_command.extend(("--n-action-steps", str(args_cli.n_action_steps)))
    server_process = subprocess.Popen(server_command)
    connection = None
    env = None
    try:
        connection = connect_to_server(server_process)
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.use_teleop_device("so101leader")
        for camera_name in ("front", "wrist"):
            camera_cfg = getattr(env_cfg.scene, camera_name)
            camera_cfg.width = args_cli.render_width
            camera_cfg.height = args_cli.render_height
        env_cfg.observations.policy.concatenate_terms = False
        env_cfg.recorders = None
        env_cfg.terminations.time_out = None
        env_cfg.seed = args_cli.seed
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        robot = env.scene["robot"]
        lower = robot.data.soft_joint_pos_limits[0, :, 0]
        upper = robot.data.soft_joint_pos_limits[0, :, 1]

        results = []
        failure_states = []
        for trial in range(args_cli.num_rollouts):
            trial_seed = args_cli.seed + trial
            torch.manual_seed(trial_seed)
            np.random.seed(trial_seed)
            random.seed(trial_seed)
            env.seed(trial_seed)
            observations, _ = env.reset()
            initial_scene_state_sha256 = scene_state_sha256(env.scene.get_state(is_relative=True))
            send_request(connection, {"command": "reset"})
            initial_cube_xyz = env.scene["cube"].data.root_pos_w[0, :3].detach().cpu().tolist()
            cube_xy = initial_cube_xyz[:2]
            max_cube_lift_m = 0.0
            initial_joint_pos = observations["policy"]["joint_pos"][0].detach().cpu().tolist()
            first_action = None
            action_min = None
            action_max = None
            initial_observation_sha256 = None
            video_path = output_dir / f"rollout_{trial + 1:03d}.mp4"
            video_writer = (
                imageio.get_writer(video_path, fps=60, codec="libx264", quality=7)
                if trial < args_cli.video_count
                else None
            )
            clipped_steps = 0
            success = False
            trial_state_snapshots = []
            try:
                for step in range(args_cli.horizon):
                    policy_obs = observations["policy"]
                    front = resize_policy_image(policy_obs["front"][0])
                    wrist = resize_policy_image(policy_obs["wrist"][0])
                    if initial_observation_sha256 is None:
                        initial_observation_sha256 = {
                            "front": hashlib.sha256(front.tobytes()).hexdigest(),
                            "wrist": hashlib.sha256(wrist.tobytes()).hexdigest(),
                        }
                    response = send_request(
                        connection,
                        {
                            "command": "predict",
                            "state": policy_obs["joint_pos"][0].detach().cpu().tolist(),
                            "front": front.tobytes(),
                            "wrist": wrist.tobytes(),
                        },
                    )
                    action = torch.as_tensor(
                        response["action"],
                        dtype=torch.float32,
                        device=env.device,
                    ).view(1, 6)
                    if first_action is None:
                        first_action = action[0].detach().cpu().tolist()
                        action_min = action[0].detach().clone()
                        action_max = action[0].detach().clone()
                    else:
                        action_min = torch.minimum(action_min, action[0])
                        action_max = torch.maximum(action_max, action[0])
                    clipped = action.clamp(lower, upper)
                    clipped_steps += int(not torch.equal(action, clipped))
                    observations, _, _, _, _ = env.step(clipped)
                    cube_z = float(env.scene["cube"].data.root_pos_w[0, 2].item())
                    max_cube_lift_m = max(max_cube_lift_m, cube_z - initial_cube_xyz[2])
                    completed_step = step + 1
                    if (
                        args_cli.failure_state_file is not None
                        and completed_step % args_cli.failure_state_interval == 0
                    ):
                        trial_state_snapshots.append(capture_state(env, trial_seed, completed_step))
                    if video_writer is not None:
                        front = observations["policy"]["front"][0].detach().cpu().numpy()
                        wrist = observations["policy"]["wrist"][0].detach().cpu().numpy()
                        video_writer.append_data(np.concatenate((front, wrist), axis=1))
                    if bool(success_term.func(env, **success_term.params)[0]):
                        success = True
                        break
            finally:
                if video_writer is not None:
                    video_writer.close()

            if trial < args_cli.video_count:
                video_path.rename(
                    output_dir / f"rollout_{trial + 1:03d}_{'success' if success else 'failure'}.mp4"
                )
            final_cube_xyz = env.scene["cube"].data.root_pos_w[0, :3].detach().cpu().tolist()
            final_cube_lift_m = final_cube_xyz[2] - initial_cube_xyz[2]
            outcome = classify_rollout_outcome(
                success,
                max_cube_lift_m,
                final_cube_lift_m,
                args_cli.lift_threshold_m,
            )
            result = {
                "trial": trial,
                "seed": trial_seed,
                "cube_xy": cube_xy,
                "success": success,
                "steps": step + 1,
                "clipped_steps": clipped_steps,
                "initial_joint_pos": initial_joint_pos,
                "initial_observation_sha256": initial_observation_sha256,
                "initial_scene_state_sha256": initial_scene_state_sha256,
                "first_action": first_action,
                "action_min": action_min.detach().cpu().tolist(),
                "action_max": action_max.detach().cpu().tolist(),
                "final_cube_xyz": final_cube_xyz,
                "initial_cube_xyz": initial_cube_xyz,
                "max_cube_lift_m": max_cube_lift_m,
                "final_cube_lift_m": final_cube_lift_m,
                "final_gripper_position_rad": float(observations["policy"]["joint_pos"][0, -1].item()),
                "outcome": outcome,
            }
            results.append(result)
            if not success and args_cli.failure_state_file is not None:
                final_step = step + 1
                if not trial_state_snapshots or trial_state_snapshots[-1]["step"] != final_step:
                    trial_state_snapshots.append(capture_state(env, trial_seed, final_step))
                failure_states.extend(trial_state_snapshots)
            print(json.dumps(result), flush=True)

        outcome_counts = {
            outcome: sum(result["outcome"] == outcome for result in results)
            for outcome in ("success", "no_lift", "low_after_lift", "lifted_not_completed")
        }
        summary = {
            "task": args_cli.task,
            "checkpoint": str(args_cli.checkpoint.expanduser().resolve()),
            "num_rollouts": len(results),
            "successes": sum(result["success"] for result in results),
            "success_rate": sum(result["success"] for result in results) / len(results),
            "seed_start": args_cli.seed,
            "horizon": args_cli.horizon,
            "server_seed": args_cli.server_seed,
            "server_device": args_cli.server_device,
            "n_action_steps": args_cli.n_action_steps,
            "render_width": args_cli.render_width,
            "render_height": args_cli.render_height,
            "policy_image_size": args_cli.policy_image_size,
            "lift_threshold_m": args_cli.lift_threshold_m,
            "outcome_counts": outcome_counts,
            "results": results,
        }
        (output_dir / "evaluation.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        if args_cli.failure_state_file is not None:
            failure_state_file = args_cli.failure_state_file.expanduser().resolve()
            failure_state_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format_version": 1,
                    "task": args_cli.task,
                    "checkpoint": str(args_cli.checkpoint.expanduser().resolve()),
                    "states": failure_states,
                },
                failure_state_file,
            )
        print(json.dumps(summary, indent=2))
    finally:
        if connection is not None:
            try:
                send_request(connection, {"command": "shutdown"})
            except (ConnectionError, OSError):
                pass
            connection.close()
        if env is not None:
            env.close()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait(timeout=5)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:  # Isaac Sim shutdown can otherwise hide the original traceback.
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
