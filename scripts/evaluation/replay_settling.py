"""Additive, bounded observation after replay; never relabel source frames."""

import math


def observation_steps(seconds_s, control_dt_s):
    if (not math.isfinite(seconds_s) or not 0 <= seconds_s <= 5
            or not math.isfinite(control_dt_s) or control_dt_s <= 0):
        raise ValueError("settling seconds must be within [0, 5] and control dt positive")
    return max(1, math.ceil(seconds_s / control_dt_s - 1e-9)) if seconds_s else 0


def settling_summary(source_success, observations, control_dt_s):
    """Success must still hold at the end; transient successes remain visible."""
    if type(source_success) is not bool:
        raise ValueError("source success must be boolean")
    if not math.isfinite(control_dt_s) or control_dt_s <= 0:
        raise ValueError("control dt must be finite and positive")
    if any(type(row.get("success")) is not bool for row in observations):
        raise ValueError("observation success must be boolean")
    first = next((i for i, row in enumerate(observations, 1) if row["success"]), None)
    final = observations[-1]["success"] if observations else None
    if not observations:
        category = "not_observed"
    elif final:
        category = "stable_at_source_horizon" if source_success else "settled_during_observation"
    elif source_success or first is not None:
        category = "unstable_during_observation"
    else:
        category = "unresolved_after_observation"
    return {"classification": category, "steps": len(observations),
            "duration_s": len(observations) * control_dt_s,
            "first_success_step": first, "final_success": final,
            "source_horizon_final_success": source_success}
