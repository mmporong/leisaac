import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .observations import cube_placed_in_box


def cube_released_in_box(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """박스 안에 큐브를 놓고 그리퍼를 연 경우 태스크를 종료한다."""
    return cube_placed_in_box(env, cube_cfg=cube_cfg, box_cfg=box_cfg, robot_cfg=robot_cfg)
