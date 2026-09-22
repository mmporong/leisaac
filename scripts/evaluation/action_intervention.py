"""Opt-in diagnostic substitution of recorded commands; never a learned policy."""
import hashlib
from pathlib import Path

import h5py
import numpy as np


SOURCES = ("policy", "teacher_arm", "teacher_gripper", "teacher_all")


def load_teacher_commands(path, demo, horizon):
    path = Path(path).resolve()
    with h5py.File(path, "r") as file:
        group = file[f"data/{demo}"]
        target = group["obs/joint_pos_target"][:]
        pre = group["obs/joint_pos"][:]
        post = group["states/articulation/robot/joint_position"][:]
        if target.ndim != 2 or target.shape[1] != 6 or target.shape != pre.shape or post.shape != pre.shape:
            raise ValueError("teacher requires matching six-joint recorded arrays")
        if not all(np.isfinite(value).all() for value in (target, pre, post)) or not np.allclose(post[:-1], pre[1:], atol=1e-6, rtol=0):
            raise ValueError("teacher trajectory is non-finite or violates pre/post-step alignment")
        if not bool(group.attrs.get("success", False)):
            raise ValueError("teacher diagnostic requires a successful source demonstration")
        if not 1 <= horizon <= len(target) - 1:
            raise ValueError("horizon exceeds recorded t+1 targets; holding the last target is not allowed")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return target[1:horizon + 1].astype(np.float32), {
        "path": str(path), "demo": demo, "sha256": digest,
        "source_frames": len(target), "command_alignment": "loop t -> raw joint_pos_target[t+1]",
        "horizon": horizon, "training_eligibility": "diagnostic_only",
    }


def substitute_action(policy_action, teacher_action, source):
    """Return a copy with selected channels replaced, preserving tensor device/dtype."""
    if source not in SOURCES:
        raise ValueError("unknown diagnostic action source")
    if policy_action.shape != (1, 6):
        raise ValueError("expected policy action shape (1,6)")
    if source == "policy":
        return policy_action
    if teacher_action.shape != policy_action.shape:
        raise ValueError("teacher action shape must match policy action")
    action = policy_action.clone()
    if source in ("teacher_arm", "teacher_all"):
        action[:, :5] = teacher_action[:, :5]
    if source in ("teacher_gripper", "teacher_all"):
        action[:, 5:] = teacher_action[:, 5:]
    return action
