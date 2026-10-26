"""Reset events for the black-box task."""

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def reset_joints_to_pose(
    env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, joint_pos: dict[str, float]
) -> None:
    """Place the named joints at a fixed pose and hold it, leaving the default joint pose untouched.

    The leader joint-position action adds the default joint pose as an offset, so the
    start pose is applied here instead of through ``init_state.joint_pos``.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene[asset_cfg.name]
    joint_ids, joint_names = robot.find_joints(list(joint_pos), preserve_order=True)
    position = robot.data.default_joint_pos[env_ids].clone()
    position[:, joint_ids] = torch.tensor([joint_pos[name] for name in joint_names], device=env.device)
    robot.write_joint_state_to_sim(position, torch.zeros_like(position), env_ids=env_ids)
    robot.set_joint_position_target(position, env_ids=env_ids)
