"""Batched SO101 arm kinematics built from the robot USD joint frames.

Forward kinematics chains each revolute joint as ``T(local_pos0, local_rot0) * Rz(q)``
from the base link to the gripper link; every SO101 joint has an identity body1
frame and a local Z axis. The inverse solver prioritises a point on the gripper
and treats finger direction and closing axis as soft targets, because the 5-DOF
arm cannot reach every 6-DOF pose.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

ARM_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    w, x, y, z = quat.unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), dim=-1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), dim=-1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )


def _rot_z(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack(
        (torch.stack((c, -s, zero), -1), torch.stack((s, c, zero), -1), torch.stack((zero, zero, one), -1)), dim=-2
    )


class So101ArmKinematics:
    """Gripper-link pose in the base-link frame as a function of the five arm joints."""

    def __init__(self, joint_pos0: Sequence[Sequence[float]], joint_rot0_wxyz: Sequence[Sequence[float]]):
        self.joint_pos0 = torch.as_tensor(joint_pos0, dtype=torch.float64)
        self.joint_rot0 = quat_wxyz_to_matrix(torch.as_tensor(joint_rot0_wxyz, dtype=torch.float64))
        if self.joint_pos0.shape != (len(ARM_JOINT_NAMES), 3):
            raise ValueError("expected one local_pos0 per arm joint")

    @classmethod
    def from_stage(cls, stage, joints_root: str) -> So101ArmKinematics:
        """Read local frames from ``<joints_root>/<joint name>`` revolute joints and check the chain."""
        from pxr import UsdPhysics

        positions, rotations, previous_body1 = [], [], None
        for name in ARM_JOINT_NAMES:
            joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"{joints_root}/{name}"))
            if not joint:
                raise RuntimeError(f"revolute joint {joints_root}/{name} is missing")
            body0 = [str(t) for t in joint.GetBody0Rel().GetTargets()]
            body1 = [str(t) for t in joint.GetBody1Rel().GetTargets()]
            if previous_body1 is not None and body0 != previous_body1:
                raise RuntimeError(f"{name} does not follow the previous joint: {body0} != {previous_body1}")
            if joint.GetAxisAttr().Get() != "Z":
                raise RuntimeError(f"{name} axis is not local Z")
            pos1, rot1 = joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()
            if max(abs(v) for v in pos1) > 1e-6 or abs(abs(rot1.GetReal()) - 1.0) > 1e-6:
                raise RuntimeError(f"{name} body1 frame is not identity")
            rot0 = joint.GetLocalRot0Attr().Get()
            positions.append(list(joint.GetLocalPos0Attr().Get()))
            rotations.append([rot0.GetReal(), *rot0.GetImaginary()])
            previous_body1 = body1
        return cls(positions, rotations)

    def forward(self, joint_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return gripper position ``(N, 3)`` and rotation ``(N, 3, 3)`` in the base frame."""
        q = joint_pos.to(torch.float64)
        count = q.shape[0]
        rotation = torch.eye(3, dtype=torch.float64, device=q.device).expand(count, 3, 3)
        position = torch.zeros(count, 3, dtype=torch.float64, device=q.device)
        pos0, rot0 = self.joint_pos0.to(q.device), self.joint_rot0.to(q.device)
        for index in range(len(ARM_JOINT_NAMES)):
            position = position + rotation @ pos0[index]
            rotation = rotation @ rot0[index] @ _rot_z(q[:, index])
        return position, rotation

    def solve(
        self,
        initial_joint_pos: torch.Tensor,
        target_point: torch.Tensor,
        point_in_gripper: torch.Tensor,
        finger_dir: torch.Tensor,
        closing_axis: torch.Tensor,
        joint_limits: torch.Tensor,
        *,
        finger_weight_m: float = 0.05,
        closing_weight_m: float = 0.02,
        iterations: int = 30,
        damping: float = 1e-6,
        max_step_rad: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Levenberg-Marquardt IK in the base frame; returns joints and point error (m).

        ``finger_dir`` is the desired gripper -Z axis and ``closing_axis`` the desired
        gripper +X axis. Their weights convert unit-vector errors into metres.
        """
        q = initial_joint_pos.to(torch.float64).clone()
        lower, upper = joint_limits[..., 0].to(q), joint_limits[..., 1].to(q)
        target_point, point_in_gripper = target_point.to(q), point_in_gripper.to(q)
        finger_dir, closing_axis = finger_dir.to(q), closing_axis.to(q)

        def residual(joints):
            position, rotation = self.forward(joints)
            point = position + (rotation @ point_in_gripper.unsqueeze(-1)).squeeze(-1)
            return torch.cat(
                (
                    point - target_point,
                    finger_weight_m * (-rotation[..., 2] - finger_dir),
                    closing_weight_m * (rotation[..., 0] - closing_axis),
                ),
                dim=-1,
            )

        step = 1e-6
        eye = torch.eye(q.shape[-1], dtype=q.dtype, device=q.device)
        for _ in range(iterations):
            r = residual(q)
            jacobian = torch.stack(
                [(residual(q + step * eye[i]) - r) / step for i in range(q.shape[-1])], dim=-1
            )
            normal = jacobian.transpose(-1, -2) @ jacobian + damping * eye
            delta = torch.linalg.solve(normal, -(jacobian.transpose(-1, -2) @ r.unsqueeze(-1))).squeeze(-1)
            # Cap each iteration so a far seed walks to the nearest branch instead of jumping.
            delta = delta * (max_step_rad / delta.abs().amax(-1, keepdim=True).clamp_min(max_step_rad))
            q = torch.minimum(torch.maximum(q + delta, lower), upper)
        point_error = torch.linalg.vector_norm(residual(q)[..., :3], dim=-1)
        return q.to(initial_joint_pos.dtype), point_error.to(initial_joint_pos.dtype)
