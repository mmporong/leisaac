"""IK baseline for the black-box task: look once with the wrist camera, then run the joint-space oracle.

Each episode resets to the observation pose, renders a fresh wrist frame, estimates the red cube
pose from it (forward kinematics + fitted camera, the same path the physical arm would use) and
runs the oracle planned from that estimate. The simulator cube pose is read only to score the
estimate and the result. Simulation only; no physical hardware.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="LeIsaac-SO101-PickCubeIntoBlackBox-v0")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=7000)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--reset_render_frames", type=int, default=4)
parser.add_argument(
    "--disable_robot_gravity", action="store_true",
    help="Disable gravity on the robot links like scripts/datagen/state_machine/generate.py (default: gravity on).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import cv2
import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.math import matrix_from_quat
from isaaclab_tasks.utils import parse_env_cfg

import leisaac.tasks  # noqa: F401
from leisaac.datagen.state_machine import PerceivedPickCubeIntoBlackBoxStateMachine
from leisaac.datagen.state_machine.so101_kinematics import ARM_JOINT_NAMES
from leisaac.datagen.state_machine.wrist_cube_perception import (
    PinholeCamera,
    estimate_cube_pose,
    red_cube_mask,
    yaw_error_mod_90,
)
from leisaac.tasks.pick_cube_into_black_box.pick_cube_into_black_box_env_cfg import (
    WRIST_CAMERA_APERTURE_MM,
    WRIST_CAMERA_FOCAL_LENGTH_MM,
    WRIST_CAMERA_HEIGHT,
    WRIST_CAMERA_POS_M,
    WRIST_CAMERA_ROT_ROS_WXYZ,
    WRIST_CAMERA_WIDTH,
)
from leisaac.tasks.pick_cube_rot45.pick_cube_rot45_env_cfg import CUBE_HALF_EXTENT_M, TABLE_TOP_Z
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim


def fresh_wrist_frame(env, render_frames: int) -> np.ndarray:
    """Re-render after reset so the frame shows the new scene, not the previous episode."""
    for _ in range(render_frames):
        env.sim.render()
    camera = env.scene["wrist"]
    camera.reset()
    camera.update(0.0, force_recompute=True)
    return camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)


def wrist_camera_pose_w(env, sm) -> tuple[np.ndarray, np.ndarray, float]:
    """Camera pose from CPU forward kinematics of the measured joints, and the FK gripper error vs Isaac."""
    robot = env.scene["robot"]
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    pos_b, rot_b = sm._kinematics.forward(robot.data.joint_pos[:, arm_ids].double().cpu())
    root_rot = matrix_from_quat(robot.data.root_quat_w).double().cpu()
    root_pos = robot.data.root_pos_w.double().cpu()
    grip_pos = root_pos + torch.einsum("nij,nj->ni", root_rot, pos_b)
    grip_rot = root_rot @ rot_b
    isaac_pos = robot.data.body_link_pos_w[:, robot.find_bodies("gripper")[0][0]].double().cpu()
    fk_error = float(torch.linalg.vector_norm(grip_pos - isaac_pos, dim=-1)[0])
    mount_rot = matrix_from_quat(torch.tensor(WRIST_CAMERA_ROT_ROS_WXYZ, dtype=torch.float64))
    cam_pos = grip_pos[0] + grip_rot[0] @ torch.tensor(WRIST_CAMERA_POS_M, dtype=torch.float64)
    cam_rot = grip_rot[0] @ mount_rot
    return cam_pos.numpy(), cam_rot.numpy(), fk_error


def yaw_of(quat_wxyz) -> float:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> None:
    if args_cli.num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    output_dir = args_cli.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into a non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.use_teleop_device(PerceivedPickCubeIntoBlackBoxStateMachine.TELEOP_DEVICE)
    env_cfg.seed = args_cli.seed
    env_cfg.recorders = None
    env_cfg.terminations.time_out = None
    env_cfg.terminations.success = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if args_cli.disable_robot_gravity:
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        for prim in omni.usd.get_context().get_stage().Traverse():
            if "Robot" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)

    camera = PinholeCamera.from_aperture(
        WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT, WRIST_CAMERA_FOCAL_LENGTH_MM, WRIST_CAMERA_APERTURE_MM
    )
    sm = PerceivedPickCubeIntoBlackBoxStateMachine()
    sm.setup(env)

    results = []
    try:
        # Like generate.py, keep resets and steps in one inference mode so env tensors stay consistent.
        with torch.inference_mode():
            for episode in range(args_cli.num_episodes):
                sm.reset()
                env.reset()
                rgb = fresh_wrist_frame(env, args_cli.reset_render_frames)
                cam_pos, cam_rot, fk_error = wrist_camera_pose_w(env, sm)
                cube = env.scene["cube"].data
                true_pos = cube.root_pos_w[0].detach().cpu().numpy()
                true_yaw = yaw_of(cube.root_quat_w[0].detach().cpu())
                estimate = estimate_cube_pose(rgb, camera, cam_pos, cam_rot, TABLE_TOP_Z, 2 * CUBE_HALF_EXTENT_M)
                row = {"episode": episode, "true_cube_xyz": true_pos.tolist(), "true_yaw_rad": true_yaw,
                       "fk_gripper_error_m": fk_error}
                overlay = rgb.copy()
                mask = red_cube_mask(rgb)
                if mask is not None:
                    overlay[mask] = (0, 255, 0)
                cv2.imwrite(str(output_dir / f"wrist_{episode:03d}.png"),
                            cv2.cvtColor(np.concatenate((rgb, overlay), axis=1), cv2.COLOR_RGB2BGR))
                if estimate is None:
                    row.update({"outcome": "cube_not_seen", "success": False})
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    continue
                row.update({
                    "estimated_cube_xyz": estimate.position.tolist(),
                    "estimate_error_xy_m": float(np.linalg.norm(estimate.position[:2] - true_pos[:2])),
                    "estimate_yaw_error_rad": yaw_error_mod_90(estimate.yaw_mod_90, true_yaw),
                    "red_pixels": estimate.pixel_count,
                })
                sm.set_cube_estimate(
                    torch.tensor(estimate.position, dtype=torch.float32, device=env.device).unsqueeze(0),
                    torch.tensor([estimate.yaw_mod_90], dtype=torch.float32, device=env.device),
                )
                sm.begin_episode(env)
                steps = 0
                while not sm.is_episode_done:
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, sm.TELEOP_DEVICE)
                    sm.pre_step(env)
                    env.step(sm.get_action(env))
                    sm.observe_step(env)
                    sm.advance()
                    steps += 1
                success = sm.check_success(env)
                row.update({
                    "success": bool(success),
                    "outcome": "success" if success else "failure",
                    "steps": steps,
                    "final_cube_xyz": env.scene["cube"].data.root_pos_w[0].detach().cpu().tolist(),
                    "reach_error_m": float(sm.reach_error_m[0]),
                })
                results.append(row)
                print(json.dumps(row), flush=True)
    finally:
        errors = [r["estimate_error_xy_m"] for r in results if "estimate_error_xy_m" in r]
        summary = {
            "task": args_cli.task,
            "seed": args_cli.seed,
            "robot_gravity": not args_cli.disable_robot_gravity,
            "reset_render_frames": args_cli.reset_render_frames,
            "requested_episodes": args_cli.num_episodes,
            "episodes": len(results),
            "complete": len(results) == args_cli.num_episodes,
            "successes": sum(r["success"] for r in results),
            "cube_not_seen": sum(r["outcome"] == "cube_not_seen" for r in results),
            "estimate_error_xy_m": {"mean": float(np.mean(errors)), "max": float(np.max(errors))} if errors else None,
            "results": results,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("IK_BASELINE_SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
