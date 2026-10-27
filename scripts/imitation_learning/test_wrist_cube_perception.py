"""CPU checks for the wrist-camera red-cube estimate used by the IK baseline."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


SOURCE = Path(__file__).resolve().parents[2] / "source/leisaac/leisaac/datagen/state_machine/wrist_cube_perception.py"
SPEC = importlib.util.spec_from_file_location("wrist_cube_perception_test_target", SOURCE)
perception = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = perception
SPEC.loader.exec_module(perception)

CAMERA = perception.PinholeCamera.from_aperture(352, 288, 39.491, 36.83)
TABLE_Z, CUBE = 0.041, 0.04


def look_down(tilt_rad: float, height: float):
    """Camera above the origin pitched by tilt_rad about the world x axis, optical axis mostly down."""
    rot = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # columns: x right, y down, z down
    c, s = math.cos(tilt_rad), math.sin(tilt_rad)
    pitch = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return np.array([0.0, -0.05, TABLE_Z + height]), pitch @ rot


def project(points, cam_pos, cam_rot):
    pc = (np.asarray(points) - cam_pos) @ cam_rot
    return CAMERA.focal_px * pc[:, :2] / pc[:, 2:3] + np.array([CAMERA.width / 2, CAMERA.height / 2]) - 0.5


def draw_cube(center_xy, yaw, cam_pos, cam_rot):
    """Render the cube silhouette (convex hull of its projected corners) in red on a white table."""
    corners = []
    for dz in (0.0, CUBE):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            local = np.array([sx * CUBE / 2, sy * CUBE / 2])
            c, s = math.cos(yaw), math.sin(yaw)
            xy = np.array(center_xy) + np.array([[c, -s], [s, c]]) @ local
            corners.append([xy[0], xy[1], TABLE_Z + dz])
    uv = project(corners, cam_pos, cam_rot)
    image = np.full((CAMERA.height, CAMERA.width, 3), 235, np.uint8)
    hull = cv2.convexHull(np.round(uv * 16).astype(np.int32))
    cv2.fillConvexPoly(image, hull, (200, 20, 20), lineType=cv2.LINE_AA, shift=4)
    return image


class WristCubePerceptionTest(unittest.TestCase):
    def test_pixel_ray_lands_on_projected_point(self):
        cam_pos, cam_rot = look_down(math.radians(25), 0.25)
        point = np.array([[0.03, 0.02, TABLE_Z + 0.02]])
        u, v = project(point, cam_pos, cam_rot)[0]
        back = perception.pixels_to_plane(np.array([u]), np.array([v]), CAMERA, cam_pos, cam_rot, TABLE_Z + 0.02)
        np.testing.assert_allclose(back[0], point[0], atol=1e-9)

    def test_cube_center_and_yaw_from_an_oblique_view(self):
        cam_pos, cam_rot = look_down(math.radians(25), 0.25)
        for center, yaw in (((0.0, 0.0), 0.2), ((0.03, -0.03), -0.5), ((-0.04, 0.04), 0.7)):
            image = draw_cube(center, yaw, cam_pos, cam_rot)
            estimate = perception.estimate_cube_pose(image, CAMERA, cam_pos, cam_rot, TABLE_Z, CUBE)
            self.assertIsNotNone(estimate)
            self.assertLess(np.linalg.norm(estimate.position[:2] - center), 0.001)
            self.assertAlmostEqual(estimate.position[2], TABLE_Z + CUBE / 2)
            self.assertLess(perception.yaw_error_mod_90(estimate.yaw_mod_90, yaw), math.radians(4))

    def test_no_red_or_too_few_pixels_returns_none(self):
        cam_pos, cam_rot = look_down(math.radians(25), 0.25)
        blank = np.full((CAMERA.height, CAMERA.width, 3), 235, np.uint8)
        self.assertIsNone(perception.estimate_cube_pose(blank, CAMERA, cam_pos, cam_rot, TABLE_Z, CUBE))
        speck = blank.copy()
        speck[100:105, 100:105] = (200, 20, 20)
        self.assertIsNone(perception.estimate_cube_pose(speck, CAMERA, cam_pos, cam_rot, TABLE_Z, CUBE))

    def test_wrong_image_size_and_upward_rays_are_rejected(self):
        cam_pos, cam_rot = look_down(math.radians(25), 0.25)
        with self.assertRaises(ValueError):
            perception.estimate_cube_pose(np.zeros((10, 10, 3), np.uint8), CAMERA, cam_pos, cam_rot, TABLE_Z, CUBE)
        with self.assertRaises(ValueError):
            perception.pixels_to_plane(np.array([176.0]), np.array([144.0]), CAMERA, cam_pos, -cam_rot, TABLE_Z)

    def test_yaw_error_wraps_at_90_degrees(self):
        self.assertAlmostEqual(perception.yaw_error_mod_90(math.radians(89), math.radians(1)), math.radians(2))
        self.assertAlmostEqual(perception.yaw_error_mod_90(math.radians(46), math.radians(-44)), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
