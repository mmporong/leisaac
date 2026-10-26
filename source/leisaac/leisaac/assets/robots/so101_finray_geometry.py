"""Mount geometry for 80 mm TPU FinRay fingers on both SO101 gripper jaws (numpy only).

The finger STL is a local asset (license unverified), so only its hash and the
anchor values below live in this repository. Anchors come from the STL and the
SO101 USD collision meshes; see docs/provenance/so101_finray_mount_20261026.yaml.
The fixed finger replaces the stock fixed finger in place; the moving finger is its
mirror image, placed so the two tip lips touch at FINRAY_CLOSED_GRIPPER_RAD.
"""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path

import numpy as np

FINRAY_STL_SHA256 = "93d2fd90adc2b6b045f3cca8f153ed68d5431e301bcc370924c75d72f82b6373"

# STL anchors (mm): flat contact face plane, tip, width centre and lip height.
STL_CONTACT_FACE_Y_MM = 9.91
STL_TIP_X_MM = -70.496
STL_WIDTH_CENTER_Z_MM = -0.1255
STL_LIP_HEIGHT_MM = 3.0
# Stock fixed finger in the gripper link frame (m): inner face X and tip Z.
GRIPPER_FIXED_FACE_X_M = -0.0079
GRIPPER_FIXED_TIP_Z_M = -0.1044
# Gripper joint (body0 = gripper, body1 = jaw) frame in the gripper link frame.
GRIPPER_JOINT_POS0_M = (0.0201999992, 0.0187999997, -0.0233999994)
GRIPPER_JOINT_ROT0_WXYZ = (0.7071055174, 0.7071080804, 0.0, 0.0)
# LeIsaac maps the USD gripper lower limit (-10 deg) to gripper 0 %, which the
# follower calibration records with the two finger tips touching.
FINRAY_CLOSED_GRIPPER_RAD = -math.radians(10.0)

# STL axes in the gripper frame: length (STL x) along gripper z, contact normal
# (STL +y) toward the moving jaw (gripper +x), width (STL z) along gripper y.
_STL_TO_GRIPPER = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def load_binary_stl(path: str | Path, expected_sha256: str | None = FINRAY_STL_SHA256) -> np.ndarray:
    """Return triangles ``(T, 3, 3)`` in STL units after checking the file hash."""
    data = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"{path} does not match the expected FinRay STL sha256 {expected_sha256}")
    count = struct.unpack("<I", data[80:84])[0]
    if len(data) != 84 + 50 * count:
        raise ValueError(f"{path} is not a binary STL with {count} triangles")
    record = np.dtype([("normal", "<3f4"), ("vertices", "<9f4"), ("attribute", "<u2")])
    return np.frombuffer(data[84:], dtype=record)["vertices"].reshape(-1, 3, 3).astype(np.float64)


def _quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def fixed_finger_in_gripper(triangles_mm: np.ndarray) -> np.ndarray:
    """Fixed finger triangles in the gripper link frame (m)."""
    anchor = np.array([STL_TIP_X_MM, STL_CONTACT_FACE_Y_MM, STL_WIDTH_CENTER_Z_MM])
    target = np.array([GRIPPER_FIXED_FACE_X_M, 0.0, GRIPPER_FIXED_TIP_Z_M])
    return (triangles_mm - anchor) / 1000.0 @ _STL_TO_GRIPPER.T + target


def lip_contact_x_m() -> float:
    """Gripper-frame X where the closed fingers' lips meet."""
    return GRIPPER_FIXED_FACE_X_M + STL_LIP_HEIGHT_MM / 1000.0


def moving_finger_in_jaw(triangles_mm: np.ndarray) -> np.ndarray:
    """Moving finger triangles in the jaw link frame (m), winding kept outward after mirroring."""
    closed = fixed_finger_in_gripper(triangles_mm).copy()
    closed[..., 0] = 2.0 * lip_contact_x_m() - closed[..., 0]
    closed = closed[:, ::-1, :]
    rot0 = _quat_wxyz_to_matrix(GRIPPER_JOINT_ROT0_WXYZ)
    c, s = math.cos(FINRAY_CLOSED_GRIPPER_RAD), math.sin(FINRAY_CLOSED_GRIPPER_RAD)
    rot_closed = rot0 @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (closed - np.asarray(GRIPPER_JOINT_POS0_M)) @ rot_closed


def moving_finger_in_gripper(jaw_triangles_m: np.ndarray, gripper_rad: float) -> np.ndarray:
    """Place jaw-frame triangles in the gripper frame at a gripper joint angle."""
    rot0 = _quat_wxyz_to_matrix(GRIPPER_JOINT_ROT0_WXYZ)
    c, s = math.cos(gripper_rad), math.sin(gripper_rad)
    rot = rot0 @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return jaw_triangles_m @ rot.T + np.asarray(GRIPPER_JOINT_POS0_M)
