"""Wrist-camera helpers shared by the black-box IK baseline and closed-loop evaluations (import after app launch)."""

import math

import numpy as np
import torch
from isaaclab.utils.math import matrix_from_quat

from leisaac.datagen.state_machine.so101_kinematics import ARM_JOINT_NAMES
from leisaac.tasks.pick_cube_into_black_box.pick_cube_into_black_box_env_cfg import (
    WRIST_CAMERA_POS_M,
    WRIST_CAMERA_ROT_ROS_WXYZ,
)


def fresh_wrist_frame(env, render_frames: int) -> np.ndarray:
    """Re-render without stepping physics so the frame shows the current scene, not an older render."""
    for _ in range(render_frames):
        env.sim.render()
    camera = env.scene["wrist"]
    camera.reset()
    camera.update(0.0, force_recompute=True)
    return camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)


def wrist_camera_pose_w(env, kinematics) -> tuple[np.ndarray, np.ndarray, float]:
    """Camera pose from CPU forward kinematics of the measured joints, and the FK gripper error vs Isaac."""
    robot = env.scene["robot"]
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    pos_b, rot_b = kinematics.forward(robot.data.joint_pos[:, arm_ids].double().cpu())
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
