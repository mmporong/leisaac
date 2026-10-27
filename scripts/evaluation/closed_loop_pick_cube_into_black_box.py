"""Closed-loop pick for the black-box task: hover, servo the cube's image position to a taught reference, descend.

teach: with the simulator cube pose, hover above the cube and record where the cube appears in the
       wrist image (reference pixel) and how that position moves when the hover shifts (image
       Jacobian, radial/tangential to the pan axis).
run:   coarse estimate from the observation frame (optionally with a deliberately wrong camera model),
       hover above it, correct the hover until the cube pixel matches the reference, then descend and
       place with the joint-space oracle. Only the image feedback steers the correction, so camera
       model errors do not enter the final placement. The simulator cube pose only scores the run.
Simulation only; no physical hardware.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("teach", "run"), required=True)
parser.add_argument("--task", default="LeIsaac-SO101-PickCubeIntoBlackBox-v0")
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=7200)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--approach_height", type=float, default=0.09, help="Hover distance from the cube center (m).")
parser.add_argument("--vertical_approach", action="store_true",
                    help="Hover straight above the cube instead of back along the tilted fingers.")
parser.add_argument("--reference", type=Path, default=None, help="teach summary.json (run mode).")
parser.add_argument("--reset_render_frames", type=int, default=4)
parser.add_argument("--jacobian_step", type=float, default=0.01, help="Hover shift (m) for the Jacobian in teach mode.")
parser.add_argument("--max_servo_iterations", type=int, default=5)
parser.add_argument("--pixel_tolerance", type=float, default=3.0)
parser.add_argument("--servo_axes", choices=("closing", "both"), default="closing",
                    help="closing: correct only along the grasp closing axis, which the image sees well; the normal "
                         "direction runs along the camera's line of sight at the hover and keeps the coarse estimate.")
parser.add_argument("--coarse_hfov_deg", type=float, default=50.0,
                    help="Field of view the coarse estimate assumes (the simulated camera has 50).")
parser.add_argument("--coarse_pitch_error_deg", type=float, default=0.0,
                    help="Rotation about the camera x axis added to the coarse estimate's camera model.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import cv2
import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

import leisaac.tasks  # noqa: F401
from leisaac.datagen.state_machine import PerceivedPickCubeIntoBlackBoxStateMachine
from leisaac.datagen.state_machine.wrist_cube_perception import PinholeCamera, estimate_cube_pose, red_cube_mask
from leisaac.tasks.pick_cube_into_black_box.pick_cube_into_black_box_env_cfg import WRIST_CAMERA_HEIGHT, WRIST_CAMERA_WIDTH
from leisaac.tasks.pick_cube_rot45.pick_cube_rot45_env_cfg import CUBE_HALF_EXTENT_M, TABLE_TOP_Z
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from wrist_view import fresh_wrist_frame, wrist_camera_pose_w, yaw_of

MAX_PHASE_STEPS = 2000


def run_until_paused_or_done(env, sm) -> int:
    steps = 0
    while not sm.is_paused and not sm.is_episode_done and steps < MAX_PHASE_STEPS:
        if env.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(env, sm.TELEOP_DEVICE)
        sm.pre_step(env)
        env.step(sm.get_action(env))
        sm.observe_step(env)
        sm.advance()
        steps += 1
    return steps


def cube_pixel(rgb: np.ndarray) -> dict | None:
    mask = red_cube_mask(rgb)
    if mask is None:
        return None
    v, u = np.nonzero(mask)
    border = bool(u.min() == 0 or v.min() == 0 or u.max() == rgb.shape[1] - 1 or v.max() == rgb.shape[0] - 1)
    return {"uv": np.array([u.mean(), v.mean()]), "pixels": int(mask.sum()), "touches_border": border}


def pan_frame(env, sm, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit radial (pan axis -> point) and tangential (counterclockwise) directions in the world xy plane."""
    pan = sm._pan_xy(env)[0].detach().cpu().numpy()
    radial = (xy - pan) / np.linalg.norm(xy - pan)
    return radial, np.array([-radial[1], radial[0]])


def grip_frame(sm) -> tuple[np.ndarray, np.ndarray]:
    """Unit closing axis of the planned grasp and its horizontal normal (world xy); the wrist image turns with them."""
    closing = sm._closing_axis_w[0, :2].detach().cpu().numpy().astype(float)
    closing /= np.linalg.norm(closing)
    return closing, np.array([-closing[1], closing[0]])


def set_estimate(env, sm, xy: np.ndarray, yaw: float) -> None:
    position = torch.tensor([[xy[0], xy[1], TABLE_TOP_Z + CUBE_HALF_EXTENT_M]], dtype=torch.float32, device=env.device)
    sm.set_cube_estimate(position, torch.tensor([yaw], dtype=torch.float32, device=env.device))


def hover_view(env, sm, output_dir: Path, name: str) -> dict | None:
    run_until_paused_or_done(env, sm)
    if not sm.is_paused:
        return None
    rgb = fresh_wrist_frame(env, args_cli.reset_render_frames)
    cv2.imwrite(str(output_dir / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return cube_pixel(rgb)


def optical_axis_hit(env, sm) -> np.ndarray | None:
    """Where the camera's optical axis meets the cube-center plane, from the current joints (world xy)."""
    cam_pos, cam_rot, _ = wrist_camera_pose_w(env, sm._kinematics)
    axis = cam_rot[:, 2]
    if axis[2] >= -1e-6:
        return None
    t = (TABLE_TOP_Z + CUBE_HALF_EXTENT_M - cam_pos[2]) / axis[2]
    return (cam_pos + t * axis)[:2]


def coarse_estimate(env, sm, rgb):
    cam_pos, cam_rot, _ = wrist_camera_pose_w(env, sm._kinematics)
    c, s = math.cos(math.radians(args_cli.coarse_pitch_error_deg)), math.sin(math.radians(args_cli.coarse_pitch_error_deg))
    cam_rot = cam_rot @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    camera = PinholeCamera(WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT,
                           WRIST_CAMERA_WIDTH / 2 / math.tan(math.radians(args_cli.coarse_hfov_deg) / 2))
    return estimate_cube_pose(rgb, camera, cam_pos, cam_rot, TABLE_TOP_Z, 2 * CUBE_HALF_EXTENT_M)


def teach_episode(env, sm, episode: int, output_dir: Path) -> dict:
    cube = env.scene["cube"].data
    true_xy = cube.root_pos_w[0, :2].detach().cpu().numpy().astype(float)
    true_yaw = yaw_of(cube.root_quat_w[0].detach().cpu())
    set_estimate(env, sm, true_xy, true_yaw)
    sm.begin_episode(env)
    base = hover_view(env, sm, output_dir, f"teach_{episode:03d}_base")
    hit = optical_axis_hit(env, sm) if sm.is_paused else None
    radial, tangent = pan_frame(env, sm, true_xy)
    look = None if hit is None else {"hit_xy_m": hit.tolist(), "offset_radial_m": float((hit - true_xy) @ radial),
                                     "offset_tangential_m": float((hit - true_xy) @ tangent)}
    row = {"episode": episode, "true_xy_m": true_xy.tolist(), "tilt_rad": float(sm._tilt[0]), "optical_axis_hit": look,
           "reach_error_m": float(sm.reach_error_m[0]), "base": None if base is None else
           {"uv": base["uv"].tolist(), "pixels": base["pixels"], "touches_border": base["touches_border"]}}
    if base is None:
        return row
    closing, normal = grip_frame(sm)
    columns = []
    for axis_name, axis in (("closing", closing), ("normal", normal)):
        set_estimate(env, sm, true_xy + args_cli.jacobian_step * axis, true_yaw)
        sm.replan_from_cube(env)
        shifted = hover_view(env, sm, output_dir, f"teach_{episode:03d}_{axis_name}")
        if shifted is None:
            return row
        columns.append((shifted["uv"] - base["uv"]) / args_cli.jacobian_step)
    row["jacobian_px_per_m"] = np.column_stack(columns).tolist()
    return row


def run_episode(env, sm, episode: int, output_dir: Path, reference: dict) -> dict:
    cube = env.scene["cube"].data
    true_xy = cube.root_pos_w[0, :2].detach().cpu().numpy().astype(float)
    true_yaw = yaw_of(cube.root_quat_w[0].detach().cpu())
    rgb = fresh_wrist_frame(env, args_cli.reset_render_frames)
    coarse = coarse_estimate(env, sm, rgb)
    row = {"episode": episode, "true_xy_m": true_xy.tolist()}
    if coarse is None:
        row.update(outcome="cube_not_seen", success=False)
        return row
    xy, yaw = coarse.position[:2].astype(float), coarse.yaw_mod_90
    row["coarse_error_m"] = float(np.linalg.norm(xy - true_xy))
    set_estimate(env, sm, xy, yaw)
    sm.begin_episode(env)
    reference_uv = np.array(reference["reference_uv"])
    jacobian = np.array(reference["jacobian_px_per_m"])
    jacobian_inv = np.linalg.inv(jacobian)
    closing_column = jacobian[:, 0]
    servo = []
    for iteration in range(args_cli.max_servo_iterations + 1):
        view = hover_view(env, sm, output_dir, f"run_{episode:03d}_hover_{iteration}")
        if view is None:
            break
        error_px = reference_uv - view["uv"]
        servo.append({"error_px": error_px.tolist(), "estimate_error_m": float(np.linalg.norm(xy - true_xy)),
                      "touches_border": view["touches_border"]})
        closing_error_px = float(abs(closing_column @ error_px) / np.linalg.norm(closing_column))
        servo[-1]["closing_error_px"] = closing_error_px
        done = closing_error_px if args_cli.servo_axes == "closing" else float(np.linalg.norm(error_px))
        if done <= args_cli.pixel_tolerance or iteration == args_cli.max_servo_iterations:
            break
        closing, normal = grip_frame(sm)
        if args_cli.servo_axes == "closing":
            correction = np.array([closing_column @ error_px / (closing_column @ closing_column), 0.0])
        else:
            correction = jacobian_inv @ error_px
        xy = xy + correction[0] * closing + correction[1] * normal
        set_estimate(env, sm, xy, yaw)
        sm.replan_from_cube(env)
    row["servo"] = servo
    row["final_estimate_error_m"] = float(np.linalg.norm(xy - true_xy))
    if not sm.is_paused:
        row.update(outcome="hover_failed", success=False)
        return row
    sm.resume()
    run_until_paused_or_done(env, sm)
    success = bool(sm.check_success(env))
    row.update(success=success, outcome="success" if success else "failure",
               final_cube_xyz=env.scene["cube"].data.root_pos_w[0].detach().cpu().tolist())
    return row


def main() -> None:
    if args_cli.mode == "run" and args_cli.reference is None:
        raise ValueError("run mode needs --reference from a teach run")
    output_dir = args_cli.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into a non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = json.loads(args_cli.reference.read_text()) if args_cli.reference else None

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.use_teleop_device(PerceivedPickCubeIntoBlackBoxStateMachine.TELEOP_DEVICE)
    env_cfg.seed = args_cli.seed
    env_cfg.recorders = None
    env_cfg.terminations.time_out = None
    env_cfg.terminations.success = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    sm = PerceivedPickCubeIntoBlackBoxStateMachine()
    sm.approach_height_m = args_cli.approach_height
    sm.approach_along_fingers = not args_cli.vertical_approach
    sm.pause_after_phase = "above_cube"
    sm.setup(env)

    rows = []
    try:
        with torch.inference_mode():
            for episode in range(args_cli.num_episodes):
                sm.reset()
                env.reset()
                if args_cli.mode == "teach":
                    row = teach_episode(env, sm, episode, output_dir)
                else:
                    row = run_episode(env, sm, episode, output_dir, reference)
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        summary = {"mode": args_cli.mode, "task": args_cli.task, "seed": args_cli.seed,
                   "approach_height_m": args_cli.approach_height, "approach_along_fingers": not args_cli.vertical_approach,
                   "requested_episodes": args_cli.num_episodes,
                   "episodes": len(rows), "rows": rows}
        if args_cli.mode == "teach":
            # A cube that touches the image border is cut the same way at the same hover, so it still teaches.
            good = [r for r in rows if r.get("jacobian_px_per_m")]
            if good:
                uv = np.array([r["base"]["uv"] for r in good])
                summary.update(reference_uv=uv.mean(0).tolist(),
                               reference_uv_spread_px=float(np.linalg.norm(uv - uv.mean(0), axis=1).max()),
                               jacobian_px_per_m=np.mean([r["jacobian_px_per_m"] for r in good], axis=0).tolist(),
                               taught_episodes=len(good))
        else:
            summary.update(servo_axes=args_cli.servo_axes, max_servo_iterations=args_cli.max_servo_iterations,
                       coarse_hfov_deg=args_cli.coarse_hfov_deg, coarse_pitch_error_deg=args_cli.coarse_pitch_error_deg,
                           reference=str(args_cli.reference), successes=sum(r.get("success", False) for r in rows))
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("CLOSED_LOOP_SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
