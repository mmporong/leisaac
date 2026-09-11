import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def cube_placed_in_box(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box_target"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    half_extent_xy: float = 0.045,
    min_height: float = 0.012,
    max_height: float = 0.075,
    open_threshold: float = 0.26,
) -> torch.Tensor:
    """큐브가 박스 내부에 있고 그리퍼가 열린 상태인지 판정한다."""
    cube: RigidObject = env.scene[cube_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    relative = cube.data.root_pos_w - box.data.root_pos_w
    inside_xy = torch.logical_and(relative[:, 0].abs() < half_extent_xy, relative[:, 1].abs() < half_extent_xy)
    inside_z = torch.logical_and(relative[:, 2] > min_height, relative[:, 2] < max_height)
    gripper_open = robot.data.joint_pos[:, -1] > open_threshold
    return inside_xy & inside_z & gripper_open
