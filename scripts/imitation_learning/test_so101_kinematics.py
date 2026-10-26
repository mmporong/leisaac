"""CPU regressions for the USD-derived SO101 arm kinematics used by the rot45 oracle."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/leisaac/leisaac/datagen/state_machine/so101_kinematics.py"
SPEC = importlib.util.spec_from_file_location("so101_kinematics_test_target", SOURCE)
kinematics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kinematics
SPEC.loader.exec_module(kinematics)

# local_pos0 / local_rot0 of the five arm joints in so101_follower.usd
# (sha256 64a877c3...ae58), read with pxr.
JOINT_POS0 = [
    [0.02079089917242527, -0.023074500262737274, 0.09488169848918915],
    [-0.030399199575185776, -0.01827779971063137, -0.05420000106096268],
    [-0.11257000267505646, -0.02800000086426735, 0.0],
    [-0.13490000367164612, 0.005200000014156103, 0.0],
    [0.0, -0.06109999865293503, 0.01810000091791153],
]
JOINT_ROT0 = [
    [8.963205004874908e-07, -0.7071055173873901, -0.7071080803871155, 8.963237405623659e-07],
    [0.49999821186065674, -0.5, -0.5, -0.500001847743988],
    [0.7071055173873901, 0.0, 0.0, 0.7071080803871155],
    [0.7071055173873901, 0.0, 0.0, -0.7071080803871155],
    [8.631541845716129e-07, 9.294899427914061e-07, 0.7071080803871155, 0.7071055173873901],
]
# Authored world translation of the gripper link with every joint at zero.
GRIPPER_AT_ZERO = [0.02061525173485279, -0.2774733006954193, 0.2668520510196686]
LIMITS = torch.tensor([[-1.92, 1.92], [-1.74, 1.74], [-1.74, 1.57], [-1.65, 1.65], [-2.79, 2.79]])


class So101KinematicsTest(unittest.TestCase):
    def setUp(self):
        self.arm = kinematics.So101ArmKinematics(JOINT_POS0, JOINT_ROT0)

    def test_zero_pose_matches_authored_gripper_link(self):
        position, rotation = self.arm.forward(torch.zeros(1, 5))
        self.assertTrue(torch.allclose(position[0], torch.tensor(GRIPPER_AT_ZERO, dtype=torch.float64), atol=1e-6))
        self.assertTrue(torch.allclose(rotation[0] @ rotation[0].T, torch.eye(3, dtype=torch.float64), atol=1e-6))

    def test_pan_keeps_points_on_its_axis_and_turns_others(self):
        q = torch.zeros(2, 5)
        q[1, 0] = 0.5
        position, _ = self.arm.forward(q)
        axis_xy = torch.tensor(JOINT_POS0[0][:2], dtype=torch.float64)
        radius = torch.linalg.vector_norm(position[:, :2] - axis_xy, dim=-1)
        self.assertAlmostEqual(radius[0].item(), radius[1].item(), places=6)
        self.assertGreater(torch.linalg.vector_norm(position[0] - position[1]).item(), 0.05)

    def test_inverse_recovers_reachable_point_and_direction(self):
        generator = torch.Generator().manual_seed(3)
        q_true = LIMITS[:, 0] + (LIMITS[:, 1] - LIMITS[:, 0]) * torch.rand(8, 5, generator=generator) * 0.6 + 0.2 * (
            LIMITS[:, 1] - LIMITS[:, 0]
        )
        q_true = torch.minimum(torch.maximum(q_true, LIMITS[:, 0]), LIMITS[:, 1])
        point_in_gripper = torch.tensor([0.016, 0.0, -0.088]).expand(8, 3)
        position, rotation = self.arm.forward(q_true)
        target = position + (rotation @ point_in_gripper.double().unsqueeze(-1)).squeeze(-1)
        start = torch.minimum(torch.maximum(q_true + 0.3, LIMITS[:, 0]), LIMITS[:, 1])
        q, error = self.arm.solve(
            start, target, point_in_gripper, -rotation[..., 2], rotation[..., 0], LIMITS.expand(8, 5, 2),
            iterations=60,
        )
        self.assertLess(error.max().item(), 1e-4)
        _, solved_rotation = self.arm.forward(q)
        alignment = (-solved_rotation[..., 2] * -rotation[..., 2]).sum(-1)
        self.assertGreater(alignment.min().item(), math.cos(math.radians(2.0)))

    def test_inverse_respects_joint_limits(self):
        target = torch.tensor([[2.0, 2.0, 2.0]])
        q, error = self.arm.solve(
            torch.zeros(1, 5), target, torch.zeros(1, 3), torch.tensor([[0.0, 0.0, -1.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]), LIMITS.expand(1, 5, 2),
        )
        self.assertTrue(((q >= LIMITS[:, 0] - 1e-9) & (q <= LIMITS[:, 1] + 1e-9)).all())
        self.assertGreater(error.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
