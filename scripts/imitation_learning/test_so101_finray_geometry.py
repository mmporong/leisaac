"""CPU checks for the FinRay finger mount used by the black-box task (needs the local STL asset)."""

import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/leisaac/leisaac/assets/robots/so101_finray_geometry.py"
SPEC = importlib.util.spec_from_file_location("so101_finray_geometry_test_target", SOURCE)
geometry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)

ASSETS_ROOT = Path(os.environ.get("LEISAAC_ASSETS_ROOT", ROOT / "assets"))
STL = ASSETS_ROOT / "robots" / "so101_finray_80mm_lip3_tpu95a.stl"
# User measurements of the physical gripper (2026-10-26): pivot to moving tip 8 cm,
# closed tip gap 0, open tip gap 18 cm.
MEASURED_PIVOT_TO_TIP_M = 0.08


@unittest.skipUnless(STL.exists(), f"local FinRay STL not found at {STL}")
class FinRayMountTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.triangles = geometry.load_binary_stl(STL)
        cls.fixed = geometry.fixed_finger_in_gripper(cls.triangles)
        cls.moving_jaw = geometry.moving_finger_in_jaw(cls.triangles)

    def moving(self, gripper_rad):
        return geometry.moving_finger_in_gripper(self.moving_jaw, gripper_rad)

    def test_fixed_finger_replaces_stock_finger(self):
        points = self.fixed.reshape(-1, 3)
        self.assertAlmostEqual(points[:, 2].min(), geometry.GRIPPER_FIXED_TIP_Z_M, places=4)
        self.assertAlmostEqual(np.median(points[points[:, 0] > geometry.GRIPPER_FIXED_FACE_X_M - 1e-4][:, 0]),
                               geometry.GRIPPER_FIXED_FACE_X_M, places=4)
        self.assertLess(abs(points[:, 1].mean()), 0.002)

    def test_lips_touch_without_overlap_when_closed(self):
        closed = self.moving(geometry.FINRAY_CLOSED_GRIPPER_RAD).reshape(-1, 3)
        fixed = self.fixed.reshape(-1, 3)
        lip_x = geometry.lip_contact_x_m()
        self.assertAlmostEqual(fixed[:, 0].max(), lip_x, places=4)
        self.assertAlmostEqual(closed[:, 0].min(), lip_x, places=4)

    def test_pivot_to_moving_tip_matches_measurement(self):
        tip = self.moving(geometry.FINRAY_CLOSED_GRIPPER_RAD).reshape(-1, 3)
        pivot = np.asarray(geometry.GRIPPER_JOINT_POS0_M)
        radius = np.linalg.norm((tip - pivot)[:, [0, 2]], axis=1).max()
        self.assertLess(abs(radius - MEASURED_PIVOT_TO_TIP_M), 0.01)

    def test_opening_admits_a_40mm_cube(self):
        opened = self.moving(geometry.FINRAY_CLOSED_GRIPPER_RAD + math.radians(60.0)).reshape(-1, 3)
        fixed = self.fixed.reshape(-1, 3)
        tip_gap = np.linalg.norm(opened[opened[:, 2].argmin()] - fixed[fixed[:, 2].argmin()])
        self.assertGreater(tip_gap, 0.05)

    def test_hash_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            geometry.load_binary_stl(STL, expected_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
