"""Subtask signals for the rotation-based tabletop placement."""

import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from leisaac.tasks.lift_cube.mdp.observations import object_grasped

from .geometry import tabletop_release_candidate


def place_cube_at_goal(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg,
    goal_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    table_top_z_m: float,
    cube_half_extent_m: float,
    open_threshold_rad: float,
    max_goal_xy_error_m: float = 0.015,
    max_bottom_error_m: float = 0.003,
    max_linear_speed_m_s: float = 0.03,
    max_angular_speed_rad_s: float = 0.5,
) -> torch.Tensor:
    """Signal a released cube resting at the goal marker."""
    cube, goal, robot = (env.scene[cfg.name] for cfg in (cube_cfg, goal_cfg, robot_cfg))
    table_top = env.scene.env_origins[:, 2] + table_top_z_m
    return tabletop_release_candidate(
        cube.data.root_pos_w,
        cube.data.root_quat_w,
        goal.data.root_pos_w,
        table_top,
        cube.data.root_lin_vel_w,
        cube.data.root_ang_vel_w,
        robot.data.joint_pos[:, -1],
        cube_half_extent_m=cube_half_extent_m,
        max_goal_xy_error_m=max_goal_xy_error_m,
        max_bottom_error_m=max_bottom_error_m,
        open_threshold_rad=open_threshold_rad,
        max_linear_speed_m_s=max_linear_speed_m_s,
        max_angular_speed_rad_s=max_angular_speed_rad_s,
    )
