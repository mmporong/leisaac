"""Stable success termination for tabletop placement."""

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from leisaac.tasks.pick_cube_into_box.mdp.release_state import StableReleaseWindow

from .observations import place_cube_at_goal


def cube_released_at_goal(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg,
    goal_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    table_top_z_m: float,
    cube_half_extent_m: float,
    open_threshold_rad: float,
    hold_time_s: float = 0.5,
    max_goal_xy_error_m: float = 0.015,
    max_bottom_error_m: float = 0.003,
    max_linear_speed_m_s: float = 0.03,
    max_angular_speed_rad_s: float = 0.5,
):
    """Require the complete release predicate for consecutive control steps.

    Isaac Lab validates term params against this signature, so the criteria are
    listed explicitly instead of being forwarded through ``**kwargs``.
    """
    criteria = {
        "table_top_z_m": table_top_z_m,
        "cube_half_extent_m": cube_half_extent_m,
        "open_threshold_rad": open_threshold_rad,
        "max_goal_xy_error_m": max_goal_xy_error_m,
        "max_bottom_error_m": max_bottom_error_m,
        "max_linear_speed_m_s": max_linear_speed_m_s,
        "max_angular_speed_rad_s": max_angular_speed_rad_s,
    }
    candidate = place_cube_at_goal(env, cube_cfg, goal_cfg, robot_cfg, **criteria)
    key = (cube_cfg.name, goal_cfg.name, robot_cfg.name, hold_time_s, tuple(sorted(criteria.items())))
    if getattr(env, "_rot45_release_key", None) != key:
        env._rot45_release_key = key
        env._rot45_release_window = StableReleaseWindow(env.episode_length_buf)
    return env._rot45_release_window.update(
        candidate,
        env.episode_length_buf,
        env.common_step_counter,
        env.step_dt,
        hold_time_s,
    )
