"""Privileged-state oracle for the SO-101 45-degree tabletop placement task."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from leisaac.tasks.lift_cube.mdp import object_grasped
from leisaac.tasks.pick_cube_rot45.mdp import cube_released_at_goal, place_cube_at_goal
from leisaac.tasks.pick_cube_rot45.pick_cube_rot45_env_cfg import (
    GOAL_ANGLE_RAD,
    GRASP_JAW_DISTANCE_M,
    GRASPED_MAX_GRIPPER_RAD,
    RELEASE_HOLD_TIME_S,
    RELEASE_PARAMS,
)

from .joint_space_pick_place import GRIPPER_CLOSE_RAD, GRIPPER_OPEN_RAD, LIFT_HEIGHT_M, JointSpacePickPlaceOracle

__all__ = ["GRIPPER_CLOSE_RAD", "GRIPPER_OPEN_RAD", "PickCubeRot45StateMachine"]

# Release with the cube bottom slightly above the table to avoid pressing into it.
_PLACE_CLEARANCE_M = 0.004


class PickCubeRot45StateMachine(JointSpacePickPlaceOracle):
    """Set the cube down on the goal marker, 45 degrees left about the pan axis."""

    def _place_targets(self, env, cube_pos_w):
        goal_pos = env.scene["goal_marker"].data.root_pos_w
        place = goal_pos.clone()
        place[:, 2] += _PLACE_CLEARANCE_M
        above = goal_pos.clone()
        above[:, 2] += LIFT_HEIGHT_M
        return place, above, torch.full((env.num_envs,), GOAL_ANGLE_RAD, device=env.device)

    def _grasped(self, env) -> torch.Tensor:
        return object_grasped(
            env,
            robot_cfg=SceneEntityCfg("robot"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg("cube"),
            diff_threshold=GRASP_JAW_DISTANCE_M,
            grasp_threshold=GRASPED_MAX_GRIPPER_RAD,
        )

    def _placed(self, env) -> torch.Tensor:
        return place_cube_at_goal(env, **RELEASE_PARAMS)

    def _success(self, env) -> torch.Tensor:
        return cube_released_at_goal(env, **RELEASE_PARAMS, hold_time_s=RELEASE_HOLD_TIME_S)
