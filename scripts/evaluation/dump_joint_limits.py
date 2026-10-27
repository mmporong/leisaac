"""Write a task's simulated joint names and soft limits, the --joint-limits-file for recorded_target conversion.

The limits are read the same way as lerobot_act_so101.py does after gym.make. No physical hardware is used.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg

import leisaac  # noqa: F401


def main() -> None:
    if args_cli.output.exists():
        raise FileExistsError(f"refusing to overwrite {args_cli.output}")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.use_teleop_device("so101leader")
    env_cfg.recorders = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    try:
        robot = env.scene["robot"]
        limits = robot.data.soft_joint_pos_limits[0]
        payload = {
            "task": args_cli.task,
            "source": "robot.data.soft_joint_pos_limits[0] after gym.make (same as lerobot_act_so101.py)",
            "joint_names": robot.joint_names,
            "joint_lower_limits_rad": limits[:, 0].cpu().tolist(),
            "joint_upper_limits_rad": limits[:, 1].cpu().tolist(),
        }
    finally:
        env.close()
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
