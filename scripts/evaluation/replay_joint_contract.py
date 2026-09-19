"""Compare recorded next-state and actuator-target commands without a learned policy."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--mode", choices=("next_observed", "recorded_target", "mimic_action"), required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--video-count", type=int, default=1)
parser.add_argument("--gripper-effort-mode", choices=("task", "fixed"), default="task")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import hashlib
import json

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from leisaac.tasks.pick_cube_into_box.mdp.release_state import release_criteria_metadata

import leisaac  # noqa: F401


def aligned_commands(joint_pos, joint_target, mimic_actions, post_step_joint_pos, mode):
    """obs[t] is pre-step; target[t+1] was applied during transition t -> t+1.

    Only recorded_target omits the final transition: its target was not recorded.
    next_observed reproduces the training converter's repeated final position.
    """
    if len(joint_pos) < 2 or joint_pos.shape != joint_target.shape:
        raise ValueError("expected matching joint trajectories with at least two frames")
    if len(mimic_actions) != len(joint_pos):
        raise ValueError("action and observation lengths differ")
    if post_step_joint_pos.shape != joint_pos.shape or not np.isfinite(post_step_joint_pos).all():
        raise ValueError("invalid post-step state trajectory")
    if not np.allclose(post_step_joint_pos[:-1], joint_pos[1:], rtol=0, atol=1e-6):
        raise ValueError("pre/post-step recorder alignment does not match")
    choices = {"next_observed": np.concatenate((joint_pos[1:], joint_pos[-1:])),
               "recorded_target": joint_target[1:], "mimic_action": mimic_actions}
    if mode not in choices:
        raise ValueError(f"unknown mode: {mode}")
    commands = choices[mode]
    if not np.isfinite(commands).all():
        raise ValueError("non-finite command")
    return commands, post_step_joint_pos[:len(commands)]


def physical_hash(state):
    digest = hashlib.sha256()
    for group, entities in sorted(state.items()):
        for entity, fields in sorted(entities.items()):
            for field, value in sorted(fields.items()):
                array = np.ascontiguousarray(value.cpu().numpy(), dtype="<f4")
                digest.update(f"{group}/{entity}/{field}:{array.shape}".encode())
                digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    output = args_cli.output_dir.resolve()
    if args_cli.video_count < 0 or len(set(args_cli.episodes)) != len(args_cli.episodes):
        raise ValueError("video-count must be nonnegative; episode indices must be unique")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    handler = HDF5DatasetFileHandler()
    handler.open(str(args_cli.dataset.resolve()))
    cfg = parse_env_cfg("LeIsaac-SO101-PickCubeIntoBox-v0", device=args_cli.device, num_envs=1)
    cfg.use_teleop_device("mimic_so101leader" if args_cli.mode == "mimic_action" else "so101leader")
    cfg.recorders = None
    success_term = cfg.terminations.success
    cfg.terminations = {}
    cfg.observations.policy.concatenate_terms = False
    cfg.seed = 43
    cfg.rerender_on_reset = True
    for name in ("front", "wrist"):
        getattr(cfg.scene, name).width = 640
        getattr(cfg.scene, name).height = 480
    env = gym.make("LeIsaac-SO101-PickCubeIntoBox-v0", cfg=cfg).unwrapped
    results = []
    try:
        env.reset()
        robot = env.scene["robot"]
        lower, upper = robot.data.soft_joint_pos_limits[0].unbind(-1)
        with h5py.File(args_cli.dataset) as dataset, torch.inference_mode():
            for trial, index in enumerate(args_cli.episodes):
                name = f"demo_{index}"
                demo = dataset[f"data/{name}"]
                commands, reference = aligned_commands(
                    demo["obs/joint_pos"][:], demo["obs/joint_pos_target"][:], demo["actions"][:],
                    demo["states/articulation/robot/joint_position"][:], args_cli.mode)
                episode = handler.load_episode(name, env.device)
                env.reset_to(episode.get_initial_state(), torch.tensor([0], device=env.device),
                             seed=43, is_relative=True)
                # reset_to restores measured velocity as a target; position control uses zero target velocity.
                robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
                env.scene.write_data_to_sim()
                initial_hash = physical_hash(env.scene.get_state(is_relative=True))
                for _ in range(4):
                    env.sim.render()
                if physical_hash(env.scene.get_state(is_relative=True)) != initial_hash:
                    raise RuntimeError("render-only warmup changed physical state")
                initial_cube = env.scene["cube"].data.root_pos_w[0].clone()
                initial_error = float((robot.data.joint_pos[0] - torch.tensor(demo["obs/joint_pos"][0], device=env.device)).abs().max())
                trajectory_error = []
                clipped_steps = 0
                success = False
                first_success_step = None
                max_lift = 0.0
                common_max_lift = 0.0
                effort_min, effort_max = float("inf"), 0.0
                writer = imageio.get_writer(output / f"{name}.mp4", fps=60, codec="libx264", quality=7) if trial < args_cli.video_count else None
                try:
                    for step, command in enumerate(commands):
                        action = torch.tensor(command, device=env.device).unsqueeze(0)
                        if args_cli.mode != "mimic_action":
                            bounded = action.clamp(lower, upper)
                            clipped_steps += int((bounded != action).any())
                            action = bounded
                        if args_cli.gripper_effort_mode == "task" and env.cfg.dynamic_reset_gripper_effort_limit:
                            dynamic_reset_gripper_effort_limit_sim(env, "so101leader")
                        effort = float(robot.data.joint_effort_limits[0, -1])
                        effort_min, effort_max = min(effort_min, effort), max(effort_max, effort)
                        observations, _, _, _, _ = env.step(action)
                        actual = robot.data.joint_pos[0].cpu().numpy()
                        trajectory_error.append((actual - reference[step]).tolist())
                        max_lift = max(max_lift, float(env.scene["cube"].data.root_pos_w[0, 2] - initial_cube[2]))
                        if step < len(demo["actions"]) - 1:
                            common_max_lift = max_lift
                        if bool(success_term.func(env, **success_term.params)[0]):
                            success = True
                            if first_success_step is None:
                                first_success_step = step + 1
                        if writer:
                            images = [observations["policy"][camera][0].cpu().numpy() for camera in ("front", "wrist")]
                            writer.append_data(np.concatenate(images, axis=1))
                finally:
                    if writer:
                        writer.close()
                error = np.asarray(trajectory_error)
                result = {"episode": name, "mode": args_cli.mode, "steps": len(commands),
                          "source_frame_count": len(demo["actions"]),
                          "source_success": bool(demo.attrs["success"]),
                          "omitted_final_transition": args_cli.mode == "recorded_target",
                          "common_horizon": len(demo["actions"]) - 1,
                          "success_by_common_horizon": first_success_step is not None and first_success_step < len(demo["actions"]),
                          "success": success, "first_success_step": first_success_step,
                          "final_success": bool(success_term.func(env, **success_term.params)[0]),
                          "initial_physical_sha256": initial_hash, "initial_joint_max_error_rad": initial_error,
                          "joint_rmse_rad": float(np.sqrt(np.mean(error ** 2))),
                          "common_horizon_joint_rmse_rad": float(np.sqrt(np.mean(error[:len(demo["actions"]) - 1] ** 2))),
                          "common_horizon_max_cube_lift_m": common_max_lift,
                          "joint_rmse_by_joint_rad": dict(zip(robot.joint_names, np.sqrt(np.mean(error ** 2, axis=0)).tolist())),
                          "clipped_steps": clipped_steps, "max_cube_lift_m": max_lift,
                          "gripper_effort_limit_range": [effort_min, effort_max],
                          "final_cube_xyz": env.scene["cube"].data.root_pos_w[0].cpu().tolist()}
                results.append(result)
                print(json.dumps(result), flush=True)
                (output / "evaluation.json").write_text(json.dumps({"dataset": str(args_cli.dataset.resolve()),
                    "mode": args_cli.mode, "gripper_effort_mode": args_cli.gripper_effort_mode,
                    "success_criteria": release_criteria_metadata(success_term.params),
                    "control_dt_s": float(env.step_dt),
                    "joint_names": robot.joint_names,
                    "joint_lower_limits_rad": lower.cpu().tolist(),
                    "joint_upper_limits_rad": upper.cpu().tolist(), "results": results}, indent=2))
    finally:
        env.close()
        handler.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
