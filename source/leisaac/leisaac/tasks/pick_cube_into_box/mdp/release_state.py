"""Tensor-only stable-release criteria shared by task and evaluation diagnostics."""

import math

import torch


def release_criteria_metadata(params=None):
    """Return the operational success specification, not a hardware calibration."""
    criteria = {
        "version": "stable_release_v2",
        "hold_time_s": 0.5,
        "max_linear_speed_m_s": 0.03,
        "max_angular_speed_rad_s": 0.5,
        "half_extent_xy_m": 0.045,
        "min_height_m": 0.012,
        "max_height_m": 0.075,
        "open_threshold_rad": 0.26,
    }
    for key, value in (params or {}).items():
        if key in criteria and key != "version":
            criteria[key] = value
    for key, value in criteria.items():
        if key != "version" and (isinstance(value, bool) or not isinstance(value, (int, float))
                                 or not math.isfinite(value)):
            raise ValueError(f"invalid stable release parameter: {key}")
    for key in criteria:
        if key != "version" and criteria[key] <= 0:
            raise ValueError(f"stable release parameter must be positive: {key}")
    if criteria["min_height_m"] >= criteria["max_height_m"]:
        raise ValueError("stable release height interval is empty")
    return criteria


def stable_release_candidate(relative_pos, linear_velocity, angular_velocity, gripper_pos, criteria):
    finite = (torch.isfinite(relative_pos).all(dim=-1)
              & torch.isfinite(linear_velocity).all(dim=-1)
              & torch.isfinite(angular_velocity).all(dim=-1)
              & torch.isfinite(gripper_pos))
    return (finite
            & (relative_pos[:, :2].abs() < criteria["half_extent_xy_m"]).all(dim=-1)
            & (relative_pos[:, 2] > criteria["min_height_m"])
            & (relative_pos[:, 2] < criteria["max_height_m"])
            & (gripper_pos > criteria["open_threshold_rad"])
            & (torch.linalg.vector_norm(linear_velocity, dim=-1) <= criteria["max_linear_speed_m_s"])
            & (torch.linalg.vector_norm(angular_velocity, dim=-1) <= criteria["max_angular_speed_rad_s"]))


class StableReleaseWindow:
    """Count only consecutive observed control steps, once per step and environment."""

    def __init__(self, prototype):
        self.count = torch.zeros_like(prototype, dtype=torch.long)
        self.last_episode_step = torch.full_like(self.count, -1)
        self.last_global_step = -1

    def update(self, candidate, episode_step, global_step, step_dt_s, hold_time_s):
        if not math.isfinite(step_dt_s) or step_dt_s <= 0:
            raise ValueError("step_dt_s must be finite and positive")
        if not math.isfinite(hold_time_s) or hold_time_s <= 0:
            raise ValueError("hold_time_s must be finite and positive")
        duplicate = (self.last_global_step == global_step) & (self.last_episode_step == episode_step)
        consecutive = ((self.last_global_step + 1 == global_step)
                       & (episode_step == self.last_episode_step + 1))
        count = torch.where(consecutive, self.count + 1, torch.ones_like(self.count))
        count = torch.where(candidate & (episode_step > 0), count, torch.zeros_like(count))
        self.count = torch.where(duplicate, self.count, count)
        self.last_episode_step = episode_step.clone()
        self.last_global_step = int(global_step)
        required_steps = math.ceil(hold_time_s / step_dt_s - 1e-9)
        return candidate & (episode_step > 0) & (self.count >= required_steps)
