"""Refit the wrist-camera mount and field of view to physical frames. CPU only; no hardware access.

Stage 1 (mount): a frame with the cube held between the fingers. The cube pose in the gripper frame
is known up to how deep and how tilted it sits between the pads, so those are fitted alongside the
camera pose and field of view by maximizing the overlap of the projected cube with the red blob.
Stage 2 (check): frames of a cube left on the table seen from several arm poses. With the refitted
camera, the estimates should agree; the spread is reported for the joint mappings under test.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

import estimate_real_cube as erc

P = erc.perception
BASE_POS = np.array(erc.CAMERA_POS_M)
BASE_ROT = Rotation.from_quat([erc.CAMERA_ROT_ROS_WXYZ[1], erc.CAMERA_ROT_ROS_WXYZ[2],
                               erc.CAMERA_ROT_ROS_WXYZ[3], erc.CAMERA_ROT_ROS_WXYZ[0]])
# Cube center between the pads in the gripper frame (oracle grasp point) and the box the fit may move it in.
HELD_CENTER = np.array([0.019, 0.0, -0.088])
W, H = 352, 288


def camera_from(params: np.ndarray):
    """params[:3] rotation vector (deg) applied in the camera frame, [3:6] mm offset, [6] horizontal FOV (deg)."""
    rot = BASE_ROT * Rotation.from_rotvec(np.radians(params[:3]))
    pos = BASE_POS + params[3:6] / 1000.0
    camera = P.PinholeCamera(W, H, W / 2 / math.tan(math.radians(params[6]) / 2))
    return pos, rot.as_matrix(), camera


def held_cube_mask(params: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    """Projected silhouette of a 40 mm cube held between the pads; nuisance = (dx, dy, dz mm, roll about x deg)."""
    pos, rot, camera = camera_from(params)
    half = erc.CUBE_M / 2
    corners = np.array([[sx, sy, sz] for sx in (-half, half) for sy in (-half, half) for sz in (-half, half)])
    corners = corners @ Rotation.from_euler("x", nuisance[3], degrees=True).as_matrix().T
    corners = corners + HELD_CENTER + nuisance[:3] / 1000.0
    pc = (corners - pos) @ rot
    if np.any(pc[:, 2] <= 0.005):
        return np.zeros((H, W), bool)
    uv = camera.focal_px * pc[:, :2] / pc[:, 2:3] + np.array([W / 2, H / 2]) - 0.5
    image = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(image, cv2.convexHull(np.round(uv * 16).astype(np.int32)), 1, lineType=cv2.LINE_8, shift=4)
    return image.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def fit_mount(held_rgb: np.ndarray) -> dict:
    observed = P.red_cube_mask(held_rgb)

    def loss(x):
        params, nuisance = x[:7], x[7:]
        penalty = 0.0
        penalty += max(0.0, abs(nuisance[0]) - 6) + max(0.0, abs(nuisance[1]) - 15) + max(0.0, abs(nuisance[2]) - 25)
        penalty += max(0.0, abs(nuisance[3]) - 45) * 0.1 + max(0.0, abs(params[6] - 55) - 20)
        return 1.0 - iou(held_cube_mask(params, nuisance), observed) + 0.05 * penalty

    best = None
    for roll in (-30, 0, 30):
        for dz in (-15, 0, 15):
            start = np.array([0, 0, 0, 0, 0, 0, 50.0, 0, 0, dz, roll], float)
            result = minimize(loss, start, method="Powell", options={"maxiter": 20000, "xtol": 1e-3, "ftol": 1e-5})
            if best is None or result.fun < best.fun:
                best = result
    params, nuisance = best.x[:7], best.x[7:]
    return {"iou": 1.0 - loss(best.x), "params": params.tolist(), "nuisance": nuisance.tolist(),
            "pos_m": camera_from(params)[0].tolist(),
            "rot_ros_wxyz": np.roll((BASE_ROT * Rotation.from_rotvec(np.radians(params[:3]))).as_quat(), 1).tolist(),
            "hfov_deg": float(params[6])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held", type=Path, required=True, help="readback JSON whose frame shows the cube held in the fingers")
    parser.add_argument("--table-views", type=Path, nargs="*", default=[], help="readbacks of one cube left on the table")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    held = json.loads(args.held.read_text())
    held_rgb = cv2.cvtColor(cv2.imread(held["frame_file"]), cv2.COLOR_BGR2RGB)
    fit = fit_mount(held_rgb)
    params = np.array(fit["params"])
    over = held_rgb.copy()
    model = held_cube_mask(params, np.array(fit["nuisance"]))
    over[model] = (0.5 * over[model] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.imwrite(str(args.output.with_suffix(".held_overlay.png")), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
    print("MOUNT", json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)) for k, v in fit.items()}))

    checks = []
    if args.table_views:
        pos, rot, camera = camera_from(params)
        erc.CAMERA_POS_M = tuple(pos)
        quat = Rotation.from_matrix(rot).as_quat()
        erc.CAMERA_ROT_ROS_WXYZ = (quat[3], quat[0], quat[1], quat[2])
        erc.CAMERA = camera
        import search_joint_mapping as sjm

        arm = erc.usd_kinematics(erc.DEFAULT_USD_FRAMES)
        calibration = json.loads(Path(held["calibration_file"]).read_text())
        records = [json.loads(p.read_text()) for p in args.table_views]
        for pan_sign in (1, -1):
            for roll_offset in (0, 90, -90, 180):
                for roll_sign in (1, -1):
                    xy, yaw = [], []
                    for record in records:
                        rgb = cv2.cvtColor(cv2.imread(record["frame_file"]), cv2.COLOR_BGR2RGB)
                        q = sjm.joint_angles(record["median_raw"], calibration, (pan_sign, 1, 1, 1, roll_sign),
                                             ("2047", "2047", "2047", "2047", "2047"), roll_offset)
                        cam_pos, cam_rot = erc.camera_pose(arm, q)
                        try:
                            est = P.estimate_cube_pose(rgb, camera, cam_pos, cam_rot, erc.TABLE_Z_BASE_M, erc.CUBE_M)
                        except ValueError:
                            est = None
                        if est is None:
                            break
                        xy.append(est.position[:2])
                        yaw.append(est.yaw_mod_90)
                    if len(xy) < len(records):
                        continue
                    xy = np.array(xy)
                    checks.append({"pan_sign": pan_sign, "roll_sign": roll_sign, "roll_offset_deg": roll_offset,
                                   "spread_m": float(np.linalg.norm(xy - xy.mean(0), axis=1).max()),
                                   "yaw_spread_deg": math.degrees(max(P.yaw_error_mod_90(a, b) for a in yaw for b in yaw)),
                                   "mean_xy_m": xy.mean(0).round(4).tolist()})
        checks.sort(key=lambda c: c["spread_m"])
        for c in checks[:5]:
            print("CHECK", json.dumps(c))
    args.output.write_text(json.dumps({"mount_fit": fit, "table_view_checks": checks}, indent=2) + "\n")


if __name__ == "__main__":
    main()
