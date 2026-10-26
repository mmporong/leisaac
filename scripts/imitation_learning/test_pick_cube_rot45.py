"""CPU regressions for the 45-degree SO101 tabletop task."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/leisaac/leisaac/tasks/pick_cube_rot45/mdp/geometry.py"
SPEC = importlib.util.spec_from_file_location("pick_cube_rot45_geometry_test_target", SOURCE)
geometry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)


class PickCubeRot45GeometryTest(unittest.TestCase):
    def test_goal_rotates_left_45_degrees_at_same_radius(self):
        pan = torch.tensor([[0.0, 0.0], [2.0, -1.0]])
        cube = torch.tensor([[1.0, 0.0], [2.0, 1.0]])
        goal = geometry.rotate_xy_about_pan(cube, pan, math.pi / 4)
        expected = torch.tensor([[math.sqrt(0.5), math.sqrt(0.5)],
                                 [2.0 - math.sqrt(2.0), -1.0 + math.sqrt(2.0)]])
        self.assertTrue(torch.allclose(goal, expected, atol=1e-6))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(goal - pan, dim=-1),
                                       torch.linalg.vector_norm(cube - pan, dim=-1)))

    def test_positive_angle_turns_toward_robot_left(self):
        # The task robot faces world +Y, so its left side is world -X.
        goal = geometry.rotate_xy_about_pan(torch.tensor([[0.0, 0.25]]), torch.zeros(1, 2), math.pi / 4)
        self.assertLess(goal[0, 0].item(), 0.0)
        self.assertGreater(goal[0, 1].item(), 0.0)

    def test_goal_pose_rotates_cube_pose_rigidly_and_seats_on_table(self):
        yaw = 0.3
        cube = torch.tensor([[1.0, 0.0, 0.2, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])
        goal = geometry.rotated_goal_pose(
            cube, torch.zeros(1, 3), torch.tensor([0.04]), cube_half_extent_m=0.02, angle_rad=math.pi / 4
        )
        expected_yaw = yaw + math.pi / 4
        expected = torch.tensor([[math.sqrt(0.5), math.sqrt(0.5), 0.06,
                                  math.cos(expected_yaw / 2), 0.0, 0.0, math.sin(expected_yaw / 2)]])
        self.assertTrue(torch.allclose(goal, expected, atol=1e-6))

    def test_world_z_rotation_composes_with_tilted_orientation(self):
        about_x = torch.tensor([[math.cos(0.2), math.sin(0.2), 0.0, 0.0]])
        rotated = geometry.rotate_quat_about_world_z(about_x, math.pi / 2)
        self.assertAlmostEqual(torch.linalg.vector_norm(rotated).item(), 1.0, places=6)
        # A world-Z turn must not change how far the cube is tilted from vertical.
        half = torch.tensor([0.02, 0.02, 0.02])
        self.assertAlmostEqual(
            geometry.projected_half_height_wxyz(rotated, half).item(),
            geometry.projected_half_height_wxyz(about_x, half).item(),
            places=6,
        )

    def test_goal_pose_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            geometry.rotated_goal_pose(
                torch.zeros(2, 7), torch.zeros(1, 3), torch.zeros(2), cube_half_extent_m=0.02, angle_rad=0.1
            )

    def test_projected_half_height_accounts_for_cube_rotation(self):
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        angle = math.pi / 4
        about_x = torch.tensor([[math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0]])
        half = torch.tensor([0.02, 0.02, 0.02])
        self.assertAlmostEqual(geometry.projected_half_height_wxyz(identity, half).item(), 0.02, places=6)
        self.assertAlmostEqual(
            geometry.projected_half_height_wxyz(about_x, half).item(),
            0.02 * math.sqrt(2.0),
            places=6,
        )

    def _candidate(self, *, xy=(0.0, 0.0), center_z=0.0615, linear=0.0,
                   angular=0.0, gripper=0.4):
        return geometry.tabletop_release_candidate(
            torch.tensor([[xy[0], xy[1], center_z]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0615]]),
            torch.tensor([0.0415]),
            torch.tensor([[linear, 0.0, 0.0]]),
            torch.tensor([[angular, 0.0, 0.0]]),
            torch.tensor([gripper]),
            cube_half_extent_m=0.02,
            max_goal_xy_error_m=0.015,
            max_bottom_error_m=0.003,
            open_threshold_rad=0.26,
            max_linear_speed_m_s=0.03,
            max_angular_speed_rad_s=0.5,
        )

    def test_release_requires_table_contact_open_gripper_and_stability(self):
        self.assertTrue(self._candidate().item())
        self.assertFalse(self._candidate(xy=(0.016, 0.0)).item())
        self.assertFalse(self._candidate(center_z=0.0715).item())
        self.assertFalse(self._candidate(linear=0.031).item())
        self.assertFalse(self._candidate(angular=0.51).item())
        self.assertFalse(self._candidate(gripper=0.25).item())

    def test_nonfinite_state_is_rejected(self):
        result = self._candidate()
        self.assertTrue(result.item())
        self.assertFalse(self._candidate(xy=(float("nan"), 0.0)).item())


if __name__ == "__main__":
    unittest.main()
