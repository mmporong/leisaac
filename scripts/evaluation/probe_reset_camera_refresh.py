"""Reproduce and measure the stale camera observation after ``env.reset`` in Isaac Lab.

Read-only with respect to datasets and models. One Isaac process runs:

1. replay one raw MimicGen demo from its stored initial state with the original
   8D Mimic actions, so the renderer ends on a realistic "previous attempt" scene;
2. a plain ``env.reset()`` (same call the data generator issues) and a check that
   the returned policy images are bit-identical to the last render before reset;
3. render-only refresh (``sim.render`` + camera ``reset``/``update(force)``) and a
   check that the physical state hash does not change while the image converges;
4. ``env.reset_to`` on stored per-step raw states followed by the same refresh,
   compared against the recorded raw frames, to show that missing/incorrect
   frames can be re-rendered from saved states instead of being dropped/shifted.

With ``--rerender-on-reset`` the same run uses the Isaac Lab config flag so the
built-in single ``sim.render()`` can be compared with the multi-frame refresh.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--raw-path", type=Path, required=True)
parser.add_argument("--demo", default="demo_9")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--refresh-frames", type=int, default=4)
parser.add_argument("--rerender-on-reset", action="store_true")
parser.add_argument("--state-frames", type=int, nargs="+", default=[1, 299, 300],
                    help="raw states[k] to re-render; recorded frame k+1 (and k+2 if k+1 is even) should match")
parser.add_argument("--observation-image-size", type=int, default=224)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import hashlib
import json
import sys

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

import leisaac  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimic"))
from runtime_options import wrap_policy_camera_observations  # noqa: E402

CAMERAS = ("front", "wrist")
TASK = "LeIsaac-SO101-PickCubeIntoBox-v0"


def physical_hash(state) -> str:
    digest = hashlib.sha256()
    for group, entities in sorted(state.items()):
        for entity, fields in sorted(entities.items()):
            for field, value in sorted(fields.items()):
                array = np.ascontiguousarray(value.cpu().numpy(), dtype="<f4")
                digest.update(f"{group}/{entity}/{field}:{array.shape}".encode())
                digest.update(array.tobytes())
    return digest.hexdigest()


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def policy_images(observations) -> dict[str, np.ndarray]:
    return {cam: observations["policy"][cam][0].detach().cpu().numpy().copy() for cam in CAMERAS}


def camera_images(env) -> dict[str, np.ndarray]:
    """Read the sensors' current data buffers without stepping or rendering."""
    return {cam: env.scene[cam].data.output["rgb"][0, ..., :3].detach().cpu().numpy().copy() for cam in CAMERAS}


def latest_render(env) -> dict[str, np.ndarray]:
    """Pull the annotator output into the camera buffer (no physics, no new render)."""
    for cam in CAMERAS:
        env.scene[cam].update(0.0, force_recompute=True)
    return camera_images(env)


def resize(images: dict[str, np.ndarray], size: int) -> dict[str, np.ndarray]:
    import cv2

    return {cam: cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA) for cam, img in images.items()}


def refresh_without_physics(env, frames: int, size: int) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Render ``frames`` times and refresh cameras; return per-frame convergence and the final images."""
    before = physical_hash(env.scene.get_state(is_relative=True))
    history = []
    previous = None
    current = None
    for index in range(frames):
        env.sim.render()
        for cam in CAMERAS:
            env.scene[cam].reset()
            env.scene[cam].update(0.0, force_recompute=True)
        current = resize(camera_images(env), size)
        entry = {"render_calls": index + 1}
        if previous is not None:
            entry["mae_vs_previous_render"] = {cam: mae(current[cam], previous[cam]) for cam in CAMERAS}
        history.append(entry)
        previous = current
    after = physical_hash(env.scene.get_state(is_relative=True))
    if before != after:
        raise RuntimeError("render-only refresh changed the physical state")
    return history, current


def state_from_raw(group: h5py.Group, key: str, index: int | None, device) -> dict:
    state: dict = {}
    for entity_type in group[key]:
        state[entity_type] = {}
        for entity in group[f"{key}/{entity_type}"]:
            state[entity_type][entity] = {}
            for field in group[f"{key}/{entity_type}/{entity}"]:
                data = group[f"{key}/{entity_type}/{entity}/{field}"]
                value = data[index] if index is not None else data[0]
                state[entity_type][entity][field] = torch.as_tensor(np.asarray(value), device=device).unsqueeze(0)
    return state


def main() -> None:
    output = args_cli.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    size = args_cli.observation_image_size
    report: dict = {
        "raw_path": str(args_cli.raw_path.resolve()),
        "demo": args_cli.demo,
        "seed": args_cli.seed,
        "rerender_on_reset": args_cli.rerender_on_reset,
        "refresh_frames": args_cli.refresh_frames,
        "observation_image_size": size,
    }

    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    cfg.use_teleop_device("mimic_so101leader")
    cfg.recorders = None
    cfg.terminations = {}
    cfg.observations.policy.concatenate_terms = False
    cfg.seed = args_cli.seed
    cfg.rerender_on_reset = args_cli.rerender_on_reset
    for cam in CAMERAS:
        getattr(cfg.scene, cam).width = 640
        getattr(cfg.scene, cam).height = 480
    wrap_policy_camera_observations(cfg, size)
    report["camera_update_period_s"] = {cam: getattr(cfg.scene, cam).update_period for cam in CAMERAS}
    report["sim_dt_s"] = cfg.sim.dt
    report["decimation"] = cfg.decimation

    env = gym.make(TASK, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    handler = HDF5DatasetFileHandler()
    handler.open(str(args_cli.raw_path))
    try:
        with h5py.File(args_cli.raw_path, "r") as dataset, torch.inference_mode():
            demo = dataset[f"data/{args_cli.demo}"]
            raw_actions = demo["actions"][:]
            raw_front = demo["obs/front"]
            raw_wrist = demo["obs/wrist"]
            raw_frame = lambda k: {"front": raw_front[k], "wrist": raw_wrist[k]}  # noqa: E731
            episode = handler.load_episode(args_cli.demo, env.device)

            # ---- step 0: first reset of the process (data generator does the same before generating)
            observations, _ = env.reset()
            first_reset_images = policy_images(observations)
            report["first_reset_after_launch"] = {
                "joint_pos": observations["policy"]["joint_pos"][0].cpu().tolist(),
                "image_sha256": {cam: sha(img) for cam, img in first_reset_images.items()},
                "mae_vs_raw_frame0": {cam: mae(first_reset_images[cam], raw_frame(0)[cam]) for cam in CAMERAS},
            }

            # ---- step 1: reset_to the demo's stored initial state; observation before any refresh
            env.reset_to(episode.get_initial_state(), torch.tensor([0], device=env.device), seed=args_cli.seed,
                         is_relative=True)
            robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
            env.scene.write_data_to_sim()
            stale_after_reset_to = resize(camera_images(env), size)
            initial_hash = physical_hash(env.scene.get_state(is_relative=True))
            history, refreshed_initial = refresh_without_physics(env, args_cli.refresh_frames, size)
            report["reset_to_initial_state"] = {
                "joint_pos": robot.data.joint_pos[0].cpu().tolist(),
                "raw_joint_pos_frame0": demo["obs/joint_pos"][0].tolist(),
                "physical_hash": initial_hash,
                "stale_image_mae_vs_raw_frame0": {cam: mae(stale_after_reset_to[cam], raw_frame(0)[cam]) for cam in CAMERAS},
                "stale_image_mae_vs_raw_frame2": {cam: mae(stale_after_reset_to[cam], raw_frame(2)[cam]) for cam in CAMERAS},
                "refresh_history": history,
                "refreshed_mae_vs_raw_frame0": {cam: mae(refreshed_initial[cam], raw_frame(0)[cam]) for cam in CAMERAS},
                "refreshed_mae_vs_raw_frame2": {cam: mae(refreshed_initial[cam], raw_frame(2)[cam]) for cam in CAMERAS},
            }
            for cam in CAMERAS:
                imageio.imwrite(output / f"initial_refreshed_{cam}.png", refreshed_initial[cam])
                imageio.imwrite(output / f"raw_frame0_{cam}.png", raw_frame(0)[cam])
                imageio.imwrite(output / f"raw_frame2_{cam}.png", raw_frame(2)[cam])

            # ---- step 2: replay the original Mimic actions; record images seen at frames 0..3 and parity pattern
            # A second reset_to without refresh mirrors the generator: obs_buf holds whatever the renderer
            # produced last (here: the refreshed initial scene), and the camera timer restarts from zero.
            env.reset_to(episode.get_initial_state(), torch.tensor([0], device=env.device), seed=args_cli.seed,
                         is_relative=True)
            robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
            env.scene.write_data_to_sim()
            if physical_hash(env.scene.get_state(is_relative=True)) != initial_hash:
                raise RuntimeError("second reset_to produced a different physical state")
            replay_images = []
            duplicate_flags = []
            previous_sha = None
            obs = env.obs_buf
            for step, command in enumerate(raw_actions):
                images = policy_images(obs)
                current_sha = {cam: sha(images[cam]) for cam in CAMERAS}
                duplicate_flags.append(previous_sha is not None and current_sha == previous_sha)
                previous_sha = current_sha
                if step < 4:
                    replay_images.append(images)
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, "so101leader")
                obs, _, _, _, _ = env.step(torch.tensor(command, device=env.device).unsqueeze(0))
            flags = np.array(duplicate_flags[1:])  # flags[k] == frame k+1 equals frame k
            report["replay"] = {
                "steps": int(len(raw_actions)),
                "final_cube_xyz": env.scene["cube"].data.root_pos_w[0].cpu().tolist(),
                "final_joint_pos": robot.data.joint_pos[0].cpu().tolist(),
                "frame2_mae_vs_raw_frame2": {cam: mae(replay_images[2][cam], raw_frame(2)[cam]) for cam in CAMERAS},
                "frame2_mae_vs_refreshed_initial": {cam: mae(replay_images[2][cam], refreshed_initial[cam]) for cam in CAMERAS},
                "frame1_equals_frame0": bool(duplicate_flags[1]),
                "frame2_equals_frame1": bool(duplicate_flags[2]),
                "frame3_equals_frame2": bool(duplicate_flags[3]),
                "even_start_pairs": int(len(flags[0::2])),
                "even_start_duplicates": int(flags[0::2].sum()),
                "odd_start_pairs": int(len(flags[1::2])),
                "odd_start_duplicates": int(flags[1::2].sum()),
            }

            # ---- step 3: plain reset like the generator; compare with the last render before reset
            last_render = resize(latest_render(env), size)
            pre_reset_hash = physical_hash(env.scene.get_state(is_relative=True))
            observations, _ = env.reset()
            stale = policy_images(observations)
            post_reset_hash = physical_hash(env.scene.get_state(is_relative=True))
            history, refreshed = refresh_without_physics(env, args_cli.refresh_frames, size)
            report["plain_reset_after_replay"] = {
                "obs_buf_is_returned_observation": observations is env.obs_buf,
                "physical_hash_changed": pre_reset_hash != post_reset_hash,
                "joint_pos_after_reset": observations["policy"]["joint_pos"][0].cpu().tolist(),
                "cube_xyz_after_reset": env.scene["cube"].data.root_pos_w[0].cpu().tolist(),
                "reset_image_identical_to_last_render_before_reset": {
                    cam: sha(stale[cam]) == sha(last_render[cam]) for cam in CAMERAS
                },
                "reset_image_mae_vs_last_render_before_reset": {cam: mae(stale[cam], last_render[cam]) for cam in CAMERAS},
                "reset_image_mae_vs_refreshed": {cam: mae(stale[cam], refreshed[cam]) for cam in CAMERAS},
                "refresh_history": history,
            }
            for cam in CAMERAS:
                imageio.imwrite(output / f"plain_reset_stale_{cam}.png", stale[cam])
                imageio.imwrite(output / f"plain_reset_refreshed_{cam}.png", refreshed[cam])

            # ---- step 4: re-render recorded frames from stored per-step states
            rerender = []
            n_frames = int(demo["obs/joint_pos"].shape[0])
            for k in args_cli.state_frames:
                if k >= n_frames:
                    continue
                state = state_from_raw(demo, "states", k, env.device)
                env.reset_to(state, torch.tensor([0], device=env.device), seed=args_cli.seed, is_relative=True)
                robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
                env.scene.write_data_to_sim()
                _, images = refresh_without_physics(env, args_cli.refresh_frames, size)
                entry = {
                    "state_index": k,
                    "joint_pos_error_vs_raw_states": float(np.abs(
                        robot.data.joint_pos[0].cpu().numpy() - demo["states/articulation/robot/joint_position"][k]
                    ).max()),
                    "mae_vs_recorded_frame": {},
                }
                for frame in (k, k + 1, k + 2, k + 3):
                    if frame < n_frames:
                        entry["mae_vs_recorded_frame"][frame] = {cam: mae(images[cam], raw_frame(frame)[cam]) for cam in CAMERAS}
                rerender.append(entry)
                if k == args_cli.state_frames[0]:
                    for cam in CAMERAS:
                        imageio.imwrite(output / f"rerender_state{k}_{cam}.png", images[cam])
            report["rerender_from_raw_states"] = {
                "note": "recorded frame k (k even) is the render after physics step k-1, i.e. states[k-1]; "
                        "frame k+1 repeats it. A state index j should therefore match frames j+1 and j+2 when j+1 is even.",
                "entries": rerender,
            }
    finally:
        handler.close()
        env.close()

    (output / "probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
