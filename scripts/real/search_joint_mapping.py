"""Search discrete sim<->real joint mappings that make the cube estimates from several arm poses agree.

Each arm joint maps as q_sim = sign * (raw - zero) * 2*pi/4096 with sign in {+1, -1} and zero either
2047 (calibration neutral) or the middle of the recorded range; wrist_roll may also be offset by
0, +-90 or 180 degrees. The cube must not move between the captures. Candidates are ranked by the
spread of the fast centroid estimate, and the best few are re-scored with the silhouette fit.
CPU only; no hardware access.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np

import estimate_real_cube as erc

ARM = erc.ARM
TICK = 2 * math.pi / 4096


def centroid_xy(mask, cam_pos, cam_rot):
    v, u = np.nonzero(mask)
    points = erc.perception.pixels_to_plane(u, v, erc.CAMERA, cam_pos, cam_rot, erc.TABLE_Z_BASE_M + erc.CUBE_M / 2)
    return points[:, :2].mean(0)


def joint_angles(raw, calibration, signs, zeros, roll_offset_deg):
    q = []
    for name, sign, zero in zip(ARM, signs, zeros):
        cal = calibration[name]
        center = 2047 if zero == "2047" else (cal["range_min"] + cal["range_max"]) / 2
        angle = sign * (raw[name] - center) * TICK
        if name == "wrist_roll":
            angle += math.radians(roll_offset_deg)
        q.append(angle)
    return q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("readbacks", type=Path, nargs="+")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arm = erc.usd_kinematics(erc.DEFAULT_USD_FRAMES)
    captures = []
    for path in args.readbacks:
        record = json.loads(path.read_text())
        rgb = cv2.cvtColor(cv2.imread(record["frame_file"]), cv2.COLOR_BGR2RGB)
        captures.append((record["median_raw"], rgb, erc.perception.red_cube_mask(rgb)))
    calibration = json.loads(Path(json.loads(args.readbacks[0].read_text())["calibration_file"]).read_text())

    scored = []
    for signs in itertools.product((1, -1), repeat=5):
        for zeros in itertools.product(("2047", "mid"), repeat=4):
            for roll in (0, 90, -90, 180):
                xys = []
                for raw, _, mask in captures:
                    cam_pos, cam_rot = erc.camera_pose(arm, joint_angles(raw, calibration, signs, (*zeros, "2047"), roll))
                    try:
                        xys.append(centroid_xy(mask, cam_pos, cam_rot))
                    except ValueError:
                        break
                if len(xys) != len(captures):
                    continue
                xys = np.array(xys)
                scored.append((float(np.linalg.norm(xys - xys.mean(0), axis=1).max()), signs, zeros, roll, xys.mean(0)))
    scored.sort(key=lambda s: s[0])

    results = []
    for spread, signs, zeros, roll, _ in scored[: args.top]:
        estimates = []
        for raw, rgb, _ in captures:
            cam_pos, cam_rot = erc.camera_pose(arm, joint_angles(raw, calibration, signs, (*zeros, "2047"), roll))
            est = erc.perception.estimate_cube_pose(rgb, erc.CAMERA, cam_pos, cam_rot, erc.TABLE_Z_BASE_M, erc.CUBE_M)
            estimates.append(None if est is None else (est.position[:2], est.yaw_mod_90))
        xy = np.array([e[0] for e in estimates if e is not None])
        yaws = [e[1] for e in estimates if e is not None]
        yaw_spread = max(erc.perception.yaw_error_mod_90(a, b) for a in yaws for b in yaws)
        results.append({
            "signs": dict(zip(ARM, signs)), "zeros": dict(zip(ARM, (*zeros, "2047"))), "wrist_roll_offset_deg": roll,
            "centroid_spread_m": spread,
            "silhouette_spread_m": float(np.linalg.norm(xy - xy.mean(0), axis=1).max()),
            "silhouette_mean_xy_m": xy.mean(0).round(4).tolist(),
            "yaw_spread_deg": math.degrees(yaw_spread),
            "per_pose_xy_m": [e[0].round(4).tolist() for e in estimates if e is not None],
        })
    args.output.write_text(json.dumps({"candidates_scored": len(scored), "top": results}, indent=2) + "\n")
    for r in results:
        print(json.dumps({k: r[k] for k in ("signs", "zeros", "wrist_roll_offset_deg", "centroid_spread_m",
                                             "silhouette_spread_m", "yaw_spread_deg", "silhouette_mean_xy_m")}))


if __name__ == "__main__":
    main()
