import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .release_state import StableReleaseWindow, release_criteria_metadata, stable_release_candidate


def cube_released_in_box(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    hold_time_s: float = 0.5,
    max_linear_speed_m_s: float = 0.03,
    max_angular_speed_rad_s: float = 0.5,
    half_extent_xy_m: float = 0.045,
    min_height_m: float = 0.012,
    max_height_m: float = 0.075,
    open_threshold_rad: float = 0.26,
) -> torch.Tensor:
    """그리퍼를 연 뒤 박스 내부의 큐브가 안정된 상태를 유지해야 종료한다."""
    criteria = release_criteria_metadata({
        "hold_time_s": hold_time_s, "max_linear_speed_m_s": max_linear_speed_m_s,
        "max_angular_speed_rad_s": max_angular_speed_rad_s, "half_extent_xy_m": half_extent_xy_m,
        "min_height_m": min_height_m, "max_height_m": max_height_m,
        "open_threshold_rad": open_threshold_rad,
    })
    cube, box, robot = (env.scene[config.name] for config in (cube_cfg, box_cfg, robot_cfg))
    candidate = stable_release_candidate(
        cube.data.root_pos_w - box.data.root_pos_w,
        cube.data.root_lin_vel_w, cube.data.root_ang_vel_w, robot.data.joint_pos[:, -1], criteria,
    )
    key = (cube_cfg.name, box_cfg.name, robot_cfg.name, tuple(criteria.items()))
    if getattr(env, "_stable_release_key", None) != key:
        env._stable_release_key = key
        env._stable_release_window = StableReleaseWindow(env.episode_length_buf)
    return env._stable_release_window.update(
        candidate, env.episode_length_buf, env.common_step_counter, env.step_dt, hold_time_s,
    )
