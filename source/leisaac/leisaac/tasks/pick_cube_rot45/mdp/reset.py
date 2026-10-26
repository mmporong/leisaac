"""Reset-time goal placement for the 45-degree tabletop task."""

import math

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from .geometry import rotated_goal_pose


def reset_goal_from_cube(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    cube_cfg: SceneEntityCfg,
    goal_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    pan_body_name: str,
    table_top_z_m: float,
    cube_half_extent_m: float,
    angle_rad: float = math.pi / 4,
) -> None:
    """Move the non-colliding kinematic marker for the reset environments only.

    The marker pose is the single source of truth for the goal, so a state
    restored with ``reset_to`` keeps its recorded goal. Must run after the cube
    reset events.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    cube, goal, robot = (env.scene[cfg.name] for cfg in (cube_cfg, goal_cfg, robot_cfg))
    body_ids, body_names = robot.find_bodies(pan_body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"pan body must resolve once, got {body_names}")
    pose = rotated_goal_pose(
        cube.data.root_pose_w[env_ids],
        robot.data.body_link_pos_w[env_ids, body_ids[0]],
        env.scene.env_origins[env_ids, 2] + table_top_z_m,
        cube_half_extent_m=cube_half_extent_m,
        angle_rad=angle_rad,
    )
    goal.write_root_pose_to_sim(pose, env_ids=env_ids)
    goal.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device, dtype=pose.dtype), env_ids=env_ids)
