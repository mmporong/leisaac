"""Tensor-only geometry and release predicates for the 45-degree placement task."""

import math

import torch


def rotate_xy_about_pan(cube_xy: torch.Tensor, pan_xy: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Rotate each cube center around its matching pan-axis center."""
    if cube_xy.shape != pan_xy.shape or cube_xy.ndim != 2 or cube_xy.shape[-1] != 2:
        raise ValueError("cube_xy and pan_xy must both have shape (N, 2)")
    if not math.isfinite(angle_rad):
        raise ValueError("angle_rad must be finite")
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    rotation = cube_xy.new_tensor(((cosine, -sine), (sine, cosine)))
    return pan_xy + (cube_xy - pan_xy) @ rotation.T


def rotate_quat_about_world_z(quaternion_wxyz: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Left-multiply each orientation by a world-Z rotation."""
    if quaternion_wxyz.ndim != 2 or quaternion_wxyz.shape[-1] != 4:
        raise ValueError("quaternion_wxyz must have shape (N, 4)")
    if not math.isfinite(angle_rad):
        raise ValueError("angle_rad must be finite")
    half = 0.5 * angle_rad
    qw, qz = math.cos(half), math.sin(half)
    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    return torch.stack((qw * w - qz * z, qw * x - qz * y, qw * y + qz * x, qw * z + qz * w), dim=-1)


def rotated_goal_pose(
    cube_pose_w: torch.Tensor,
    pan_center_w: torch.Tensor,
    table_top_z_w: torch.Tensor,
    *,
    cube_half_extent_m: float,
    angle_rad: float,
) -> torch.Tensor:
    """Rigidly rotate the initial cube pose about the pan axis and seat it on the table."""
    if cube_pose_w.ndim != 2 or cube_pose_w.shape[-1] != 7:
        raise ValueError("cube_pose_w must have shape (N, 7)")
    if pan_center_w.shape != (cube_pose_w.shape[0], 3):
        raise ValueError("pan_center_w must have shape (N, 3)")
    if table_top_z_w.shape != (cube_pose_w.shape[0],):
        raise ValueError("table_top_z_w must have shape (N,)")
    if not math.isfinite(cube_half_extent_m) or cube_half_extent_m <= 0:
        raise ValueError("cube_half_extent_m must be finite and positive")
    goal = torch.empty_like(cube_pose_w)
    goal[:, :2] = rotate_xy_about_pan(cube_pose_w[:, :2], pan_center_w[:, :2], angle_rad)
    goal[:, 2] = table_top_z_w + cube_half_extent_m
    goal[:, 3:] = rotate_quat_about_world_z(cube_pose_w[:, 3:], angle_rad)
    return goal


def projected_half_height_wxyz(quaternion_wxyz: torch.Tensor, half_extents_m: torch.Tensor) -> torch.Tensor:
    """Project an oriented cuboid's half extents onto the world Z axis."""
    if quaternion_wxyz.ndim != 2 or quaternion_wxyz.shape[-1] != 4:
        raise ValueError("quaternion_wxyz must have shape (N, 4)")
    if half_extents_m.shape not in {(3,), quaternion_wxyz.shape[:1] + (3,)}:
        raise ValueError("half_extents_m must have shape (3,) or (N, 3)")
    norm = torch.linalg.vector_norm(quaternion_wxyz, dim=-1, keepdim=True)
    finite = torch.isfinite(quaternion_wxyz).all(dim=-1) & (norm[:, 0] > 0)
    q = quaternion_wxyz / norm.clamp_min(torch.finfo(quaternion_wxyz.dtype).eps)
    w, x, y, z = q.unbind(dim=-1)
    world_z_row = torch.stack(
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), dim=-1
    )
    projected = (world_z_row.abs() * half_extents_m).sum(dim=-1)
    return torch.where(finite, projected, torch.full_like(projected, float("nan")))


def tabletop_release_candidate(
    cube_pos_w: torch.Tensor,
    cube_quat_wxyz: torch.Tensor,
    goal_pos_w: torch.Tensor,
    table_top_z_w: torch.Tensor,
    linear_velocity_w: torch.Tensor,
    angular_velocity_w: torch.Tensor,
    gripper_pos_rad: torch.Tensor,
    *,
    cube_half_extent_m: float,
    max_goal_xy_error_m: float,
    max_bottom_error_m: float,
    open_threshold_rad: float,
    max_linear_speed_m_s: float,
    max_angular_speed_rad_s: float,
) -> torch.Tensor:
    """Return candidates that are released, resting at the goal, and nearly stationary."""
    half_extents = cube_pos_w.new_full((3,), cube_half_extent_m)
    bottom_z = cube_pos_w[:, 2] - projected_half_height_wxyz(cube_quat_wxyz, half_extents)
    finite = (
        torch.isfinite(cube_pos_w).all(dim=-1)
        & torch.isfinite(goal_pos_w).all(dim=-1)
        & torch.isfinite(linear_velocity_w).all(dim=-1)
        & torch.isfinite(angular_velocity_w).all(dim=-1)
        & torch.isfinite(gripper_pos_rad)
        & torch.isfinite(bottom_z)
    )
    return (
        finite
        & (torch.linalg.vector_norm(cube_pos_w[:, :2] - goal_pos_w[:, :2], dim=-1) <= max_goal_xy_error_m)
        & ((bottom_z - table_top_z_w).abs() <= max_bottom_error_m)
        & (gripper_pos_rad > open_threshold_rad)
        & (torch.linalg.vector_norm(linear_velocity_w, dim=-1) <= max_linear_speed_m_s)
        & (torch.linalg.vector_norm(angular_velocity_w, dim=-1) <= max_angular_speed_rad_s)
    )
