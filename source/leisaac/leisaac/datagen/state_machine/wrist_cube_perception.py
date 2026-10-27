"""Red-cube pose on the table from one wrist-camera frame (NumPy and OpenCV only, no Isaac imports).

The camera pose comes from forward kinematics plus the fitted wrist-camera mount, so the same code
can run on the physical arm. The red pixels are back-projected onto a horizontal plane; for a cube
seen obliquely from above, the plane through the cube center balances the top face against the
visible side faces (simulated check: docs/provenance/so101_ik_baseline_20261027.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    """Pinhole camera with square pixels and the principal point at the image center."""

    width: int
    height: int
    focal_px: float

    @classmethod
    def from_aperture(cls, width: int, height: int, focal_length_mm: float, horizontal_aperture_mm: float):
        return cls(width, height, width * focal_length_mm / horizontal_aperture_mm)


@dataclass(frozen=True)
class CubeEstimate:
    position: np.ndarray  # (3,) world position of the cube center
    yaw_mod_90: float  # rad in [0, pi/2); a cube's yaw is only defined modulo 90 degrees
    pixel_count: int


def red_cube_mask(rgb: np.ndarray, min_red: int = 90, ratio: float = 1.8) -> np.ndarray | None:
    """Largest connected red region of an RGB uint8 image, or None when nothing red is visible."""
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an RGB uint8 image, got shape={rgb.shape} dtype={rgb.dtype}")
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    mask = ((r > min_red) & (r > ratio * g) & (r > ratio * b)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def pixels_to_plane(
    u: np.ndarray, v: np.ndarray, camera: PinholeCamera, cam_pos: np.ndarray, cam_rot: np.ndarray, plane_z: float
) -> np.ndarray:
    """Intersect pixel-center rays with the plane z = plane_z.

    cam_rot columns are the camera x (image right), y (image down) and z (optical axis) axes in
    the world frame (ROS optical convention). Returns (N, 3) world points.
    """
    rays_cam = np.stack(
        (
            (np.asarray(u, float) + 0.5 - camera.width / 2) / camera.focal_px,
            (np.asarray(v, float) + 0.5 - camera.height / 2) / camera.focal_px,
            np.ones(np.shape(u)),
        ),
        axis=-1,
    )
    rays = rays_cam @ np.asarray(cam_rot).T
    if np.any(rays[:, 2] >= -1e-9):
        raise ValueError("some pixel rays do not point down toward the plane")
    scale = (plane_z - cam_pos[2]) / rays[:, 2]
    return np.asarray(cam_pos)[None] + rays * scale[:, None]


def cube_silhouette(
    center_xy, yaw: float, camera: PinholeCamera, cam_pos: np.ndarray, cam_rot: np.ndarray,
    table_z: float, cube_size: float, shape: tuple[int, int], offset=(0, 0),
) -> np.ndarray:
    """Boolean image of a cube on the table seen by the camera (convex hull of its projected corners)."""
    c, s = math.cos(yaw), math.sin(yaw)
    half = cube_size / 2
    local = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    xy = np.asarray(center_xy)[None] + local @ np.array([[c, s], [-s, c]])
    corners = np.concatenate([np.column_stack((xy, np.full(4, table_z + dz))) for dz in (0.0, cube_size)])
    pc = (corners - cam_pos) @ cam_rot
    uv = camera.focal_px * pc[:, :2] / pc[:, 2:3] + np.array([camera.width / 2, camera.height / 2]) - 0.5
    uv -= np.asarray(offset, float)
    image = np.zeros(shape, np.uint8)
    hull = cv2.convexHull(np.round(uv * 16).astype(np.int32))
    cv2.fillConvexPoly(image, hull, 1, lineType=cv2.LINE_8, shift=4)
    return image.astype(bool)


def _silhouette_iou(mask_roi, params, camera, cam_pos, cam_rot, table_z, cube_size, offset) -> float:
    model = cube_silhouette(params[:2], params[2], camera, cam_pos, cam_rot, table_z, cube_size, mask_roi.shape, offset)
    union = np.logical_or(model, mask_roi).sum()
    return float(np.logical_and(model, mask_roi).sum() / union) if union else 0.0


def estimate_cube_pose(
    rgb: np.ndarray,
    camera: PinholeCamera,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    table_z: float,
    cube_size: float,
    min_pixels: int = 200,
) -> CubeEstimate | None:
    """Estimate the cube center and yaw from the red blob; None when the cube is not clearly visible.

    The blob back-projected onto the cube-center plane gives a start; the pose is then refined so the
    projected cube silhouette best overlaps the blob (intersection over union), which removes the
    view-dependent bias of the side faces.
    """
    if rgb.shape[:2] != (camera.height, camera.width):
        raise ValueError(f"image {rgb.shape[:2]} does not match camera {(camera.height, camera.width)}")
    mask = red_cube_mask(rgb)
    if mask is None or int(mask.sum()) < min_pixels:
        return None
    cam_pos, cam_rot = np.asarray(cam_pos, float), np.asarray(cam_rot, float)
    v, u = np.nonzero(mask)
    center_z = table_z + cube_size / 2
    points = pixels_to_plane(u, v, camera, cam_pos, cam_rot, center_z)
    margin = 20
    top, left = max(int(v.min()) - margin, 0), max(int(u.min()) - margin, 0)
    roi = mask[top:int(v.max()) + margin + 1, left:int(u.max()) + margin + 1]
    args = (camera, cam_pos, cam_rot, table_z, cube_size, (left, top))
    start_xy = points[:, :2].mean(0)
    best = max(
        (np.array([*start_xy, math.radians(yaw_deg)]) for yaw_deg in range(0, 90, 10)),
        key=lambda p: _silhouette_iou(roi, p, *args),
    )
    best_iou = _silhouette_iou(roi, best, *args)
    steps = np.array([0.008, 0.008, math.radians(5)])
    while steps[0] > 0.0002:
        improved = False
        for axis in range(3):
            for sign in (1.0, -1.0):
                trial = best.copy()
                trial[axis] += sign * steps[axis]
                iou = _silhouette_iou(roi, trial, *args)
                if iou > best_iou:
                    best, best_iou, improved = trial, iou, True
        if not improved:
            steps /= 2
    return CubeEstimate(np.array([best[0], best[1], center_z]), float(best[2]) % (math.pi / 2), int(mask.sum()))


def yaw_error_mod_90(a: float, b: float) -> float:
    """Smallest absolute difference between two cube yaws, modulo 90 degrees."""
    d = (a - b) % (math.pi / 2)
    return min(d, math.pi / 2 - d)
