"""Estimate the red cube pose in the robot base frame from real joint readbacks and wrist frames.

For each readback JSON (scripts/real/read_so101_joints.py) and each candidate joint mapping, the
camera pose comes from the USD forward kinematics and the fitted wrist mount, and the cube is
estimated from the saved wrist frame. With the cube left in place, a correct mapping gives the same
cube position from every arm pose. CPU only; no hardware access.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
STATE_MACHINE = ROOT / "source/leisaac/leisaac/datagen/state_machine"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


perception = _load("wrist_cube_perception_real", STATE_MACHINE / "wrist_cube_perception.py")
kinematics = _load("so101_kinematics_real", STATE_MACHINE / "so101_kinematics.py")

# Wrist mount and camera fitted to a physical frame (docs/provenance/so101_finray_mount_20261026.yaml).
CAMERA_POS_M = (-0.00934, 0.04828, -0.07344)
CAMERA_ROT_ROS_WXYZ = (0.303987, 0.588847, 0.702262, -0.260153)
CAMERA = perception.PinholeCamera.from_aperture(352, 288, 39.491, 36.83)
# Table top in the base-link frame: the USD base mesh bottom is 0.0301 m above the base origin and the
# simulated table top sits 0.0309 m above it (table 0.0409 m, robot root 0.01 m).
TABLE_Z_BASE_M = 0.0309
CUBE_M = 0.04
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
MAPPINGS = ("leisaac_linear_rad", "ticks_from_2047_rad", "ticks_from_range_mid_rad")


def usd_kinematics(frames_json: Path):
    joints = json.loads(frames_json.read_text())["assets"][0]["joints"]
    by_name = {j["path"].rsplit("/", 1)[-1]: j for j in joints}
    return kinematics.So101ArmKinematics([by_name[n]["local_pos0"] for n in ARM],
                                         [by_name[n]["local_rot0_wxyz"] for n in ARM])


def camera_pose(arm, q5):
    pos, rot = arm.forward(torch.tensor([q5], dtype=torch.float64))
    pos, rot = pos.numpy()[0], rot.numpy()[0]
    mount = kinematics.quat_wxyz_to_matrix(torch.tensor(CAMERA_ROT_ROS_WXYZ, dtype=torch.float64)).numpy()
    return pos + rot @ np.array(CAMERA_POS_M), rot @ mount


DEFAULT_USD_FRAMES = Path.home() / ".config/so101-sim2real/tabletop_follower_20261026/sim_usd_frames.json"


def analyze(readbacks: list[Path], usd_frames: Path = DEFAULT_USD_FRAMES, overlay_dir: Path | None = None) -> dict:
    """Cube estimate per readback and mapping, and per mapping the spread of the estimates."""
    arm = usd_kinematics(usd_frames)
    rows = []
    for path in readbacks:
        record = json.loads(path.read_text())
        rgb = cv2.cvtColor(cv2.imread(record["frame_file"]), cv2.COLOR_BGR2RGB)
        for mapping in MAPPINGS:
            q5 = [record["angles"][n][mapping] for n in ARM]
            cam_pos, cam_rot = camera_pose(arm, q5)
            try:
                est = perception.estimate_cube_pose(rgb, CAMERA, cam_pos, cam_rot, TABLE_Z_BASE_M, CUBE_M)
            except ValueError as error:
                est, note = None, str(error)
            else:
                note = None
            rows.append({"label": record["label"], "mapping": mapping, "camera_z_m": float(cam_pos[2]),
                         "cube_xy_m": None if est is None else est.position[:2].round(4).tolist(),
                         "yaw_deg": None if est is None else round(math.degrees(est.yaw_mod_90), 1),
                         "red_pixels": None if est is None else est.pixel_count, "note": note})
            if overlay_dir is not None and est is not None:
                overlay_dir.mkdir(parents=True, exist_ok=True)
                model = perception.cube_silhouette(est.position[:2], est.yaw_mod_90, CAMERA, cam_pos, cam_rot,
                                                   TABLE_Z_BASE_M, CUBE_M, rgb.shape[:2])
                over = rgb.copy()
                over[model] = (0.5 * over[model] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
                cv2.imwrite(str(overlay_dir / f"{record['label']}_{mapping}.png"), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
    summaries = []
    for mapping in MAPPINGS:
        xy = np.array([r["cube_xy_m"] for r in rows if r["mapping"] == mapping and r["cube_xy_m"] is not None])
        spread = float(np.linalg.norm(xy - xy.mean(0), axis=1).max()) if len(xy) > 1 else None
        summaries.append({"mapping": mapping, "poses": len(xy),
                          "mean_xy_m": xy.mean(0).round(4).tolist() if len(xy) else None,
                          "max_deviation_from_mean_m": spread})
    return {"mappings": summaries, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("readbacks", type=Path, nargs="+")
    parser.add_argument("--usd-frames", type=Path, default=DEFAULT_USD_FRAMES)
    parser.add_argument("--overlay-dir", type=Path, default=None)
    args = parser.parse_args()
    result = analyze(args.readbacks, args.usd_frames, args.overlay_dir)
    for line in result["mappings"] + result["rows"]:
        print(json.dumps(line))


if __name__ == "__main__":
    main()
