"""Evaluate a Robomimic SO101 policy in Isaac Lab and save rollout evidence."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="LeIsaac-SO101-PickCubeIntoBox-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num-rollouts", type=int, default=20)
parser.add_argument("--horizon", type=int, default=1200)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--video-count", type=int, default=5)
parser.add_argument("--action-mode", choices=("joint_delta", "joint_target"), default="joint_delta")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import random

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as functional
from isaaclab_tasks.utils import parse_env_cfg
from robomimic.utils.file_utils import policy_from_checkpoint
from robomimic.utils.torch_utils import get_torch_device

import leisaac  # noqa: F401


def policy_observation(observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    def resize_image(image: torch.Tensor) -> torch.Tensor:
        chw = image.permute(0, 3, 1, 2).float().div(255.0).clamp(0.0, 1.0)
        return functional.interpolate(chw, size=(84, 84), mode="area").squeeze(0)

    front = resize_image(observations["front"])
    wrist = resize_image(observations["wrist"])
    joint_pos = observations["joint_pos"].squeeze(0)
    joint_vel = observations["joint_vel"].squeeze(0)
    previous_action = observations["actions"].squeeze(0)
    return {
        "front": front,
        "wrist": wrist,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "actions": previous_action,
    }


def main() -> None:
    if args_cli.num_rollouts <= 0:
        raise ValueError("num-rollouts must be positive")
    if args_cli.horizon <= 0:
        raise ValueError("horizon must be positive")
    if args_cli.video_count < 0:
        raise ValueError("video-count cannot be negative")

    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] output directory: {output_dir}", flush=True)

    print(f"[eval] loading task configuration: {args_cli.task}", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.use_teleop_device("so101leader")
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = None
    env_cfg.terminations.time_out = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    print("[eval] creating simulation environment", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    device = get_torch_device(try_to_use_cuda=True)
    print(f"[eval] loading checkpoint on {device}: {args_cli.checkpoint}", flush=True)
    policy, _ = policy_from_checkpoint(ckpt_path=str(args_cli.checkpoint.resolve()), device=device)
    robot = env.scene["robot"]
    lower = robot.data.soft_joint_pos_limits[0, :, 0]
    upper = robot.data.soft_joint_pos_limits[0, :, 1]

    results = []
    for trial in range(args_cli.num_rollouts):
        trial_seed = args_cli.seed + trial
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        random.seed(trial_seed)
        env.seed(trial_seed)
        observations, _ = env.reset()
        policy.start_episode()
        initial_joint_pos = observations["policy"]["joint_pos"][0].detach().cpu().tolist()
        first_action = None
        action_min = None
        action_max = None
        video_path = output_dir / f"rollout_{trial + 1:03d}.mp4"
        video_writer = (
            imageio.get_writer(video_path, fps=60, codec="libx264", quality=7)
            if trial < args_cli.video_count
            else None
        )
        clipped_steps = 0
        success = False

        cube_xy = env.scene["cube"].data.root_pos_w[0, :2].detach().cpu().tolist()
        for step in range(args_cli.horizon):
            action_np = policy(policy_observation(observations["policy"]))
            policy_output = torch.as_tensor(action_np, dtype=torch.float32, device=env.device).view(1, 6)
            action = (
                observations["policy"]["joint_pos"] + policy_output
                if args_cli.action_mode == "joint_delta"
                else policy_output
            )
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

            if video_writer is not None:
                front = observations["policy"]["front"][0].detach().cpu().numpy()
                wrist = observations["policy"]["wrist"][0].detach().cpu().numpy()
                video_writer.append_data(np.concatenate((front, wrist), axis=1))

            if bool(success_term.func(env, **success_term.params)[0]):
                success = True
                break

        if video_writer is not None:
            video_writer.close()
            video_path.rename(
                output_dir / f"rollout_{trial + 1:03d}_{'success' if success else 'failure'}.mp4"
            )

        result = {
            "trial": trial,
            "seed": trial_seed,
            "cube_xy": cube_xy,
            "success": success,
            "steps": step + 1,
            "clipped_steps": clipped_steps,
            "initial_joint_pos": initial_joint_pos,
            "first_action": first_action,
            "action_min": action_min.detach().cpu().tolist(),
            "action_max": action_max.detach().cpu().tolist(),
            "final_joint_pos": observations["policy"]["joint_pos"][0].detach().cpu().tolist(),
            "final_cube_xyz": env.scene["cube"].data.root_pos_w[0, :3].detach().cpu().tolist(),
        }
        results.append(result)
        print(json.dumps(result), flush=True)

    summary = {
        "task": args_cli.task,
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "action_mode": args_cli.action_mode,
        "num_rollouts": len(results),
        "successes": sum(result["success"] for result in results),
        "success_rate": sum(result["success"] for result in results) / len(results),
        "seed_start": args_cli.seed,
        "horizon": args_cli.horizon,
        "results": results,
    }
    summary_path = output_dir / "evaluation.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"[eval] failed: {error!r}", flush=True)
        raise
    finally:
        simulation_app.close()
