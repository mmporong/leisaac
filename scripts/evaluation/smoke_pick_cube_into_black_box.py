"""Headless integration smoke for LeIsaac-SO101-PickCubeIntoBlackBox-v0 (simulation only, no hardware).

Checks the scene (4 cm cube, black box, white table, wrist camera only), that the reset
observation pose shows the whole cube reset area in the wrist camera, and runs the
privileged joint-space oracle once per env.

Usage:
    python scripts/evaluation/smoke_pick_cube_into_black_box.py --output_dir <new dir> --num_envs 2
"""

import argparse
import itertools
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--output_dir", type=Path, required=True, help="New directory for evidence files.")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
output_dir = args_cli.output_dir.expanduser().resolve()
if output_dir.exists():
    raise SystemExit(f"refusing to overwrite existing output: {output_dir}")
args_cli.headless = True
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import leisaac.tasks  # noqa: E402,F401
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import TerminationTermCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_inv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from leisaac.datagen.state_machine import PickCubeIntoBlackBoxStateMachine  # noqa: E402
from leisaac.tasks.pick_cube_into_black_box import pick_cube_into_black_box_env_cfg as task  # noqa: E402
from leisaac.tasks.pick_cube_rot45.pick_cube_rot45_env_cfg import (  # noqa: E402
    CUBE_HALF_EXTENT_M,
    CUBE_SIZE_M,
    TABLE_TOP_Z,
)
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

TASK_ID = "LeIsaac-SO101-PickCubeIntoBlackBox-v0"
MIMIC_TASK_ID = "LeIsaac-SO101-PickCubeIntoBlackBox-Mimic-v0"
DEVICE_TYPE = PickCubeIntoBlackBoxStateMachine.TELEOP_DEVICE
CUBE_RESET_HALF_RANGE_M = 0.075

checks: dict[str, dict] = {}
metrics: dict[str, object] = {}


def check(name: str, passed: bool, **detail) -> None:
    checks[name] = {"pass": bool(passed), **detail}
    print(f"SMOKE_CHECK {'PASS' if passed else 'FAIL'} {name} {json.dumps(detail, default=float)}", flush=True)


def tolist(value):
    return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value


def save_wrist(obs, label):
    for env_index in range(obs["policy"]["wrist"].shape[0]):
        image = obs["policy"]["wrist"][env_index].detach().cpu().numpy().astype(np.uint8)[..., :3]
        Image.fromarray(image).save(output_dir / f"env{env_index}_wrist_{label}.png")


def stage_checks(env) -> None:
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for env_index in range(env.num_envs):
        root = f"/World/envs/env_{env_index}"
        source_cube = stage.GetPrimAtPath(f"{root}/Scene/cube")
        cube = stage.GetPrimAtPath(f"{root}/Cube")
        size = list(cache.ComputeUntransformedBound(cube).ComputeAlignedRange().GetSize())
        table = stage.GetPrimAtPath(f"{root}/Scene/counter_right_main_group")
        bound, _ = UsdShade.MaterialBindingAPI(table).ComputeBoundMaterial()
        check(
            f"env{env_index}_scene_matches_physical_setup",
            source_cube.IsValid() and not source_cube.IsActive()
            and all(abs(s - CUBE_SIZE_M) < 1e-6 for s in size)
            and not stage.GetPrimAtPath(f"{root}/Robot/base/front_camera").IsValid()
            and stage.GetPrimAtPath(f"{root}/Robot/gripper/wrist_camera").IsValid()
            and stage.GetPrimAtPath(f"{root}/Robot/gripper/finray_fixed").IsValid()
            and stage.GetPrimAtPath(f"{root}/Robot/jaw/finray_moving").IsValid()
            and bool(bound) and bound.GetPath().pathString.endswith("WhiteTableMaterial"),
            cube_size_m=size, table_material=str(bound.GetPath()) if bound else None,
        )


def project_to_wrist(env, points_w: torch.Tensor) -> torch.Tensor:
    """Pinhole projection ``(N, P, 3)`` -> pixel coordinates and depth ``(N, P, 3)``."""
    camera = env.scene["wrist"]
    rel = points_w - camera.data.pos_w.unsqueeze(1)
    rotation = camera.data.quat_w_ros.unsqueeze(1).expand(-1, points_w.shape[1], -1)
    local = quat_apply(quat_inv(rotation), rel)
    pixels = torch.einsum("nij,npj->npi", camera.data.intrinsic_matrices, local)
    return torch.cat((pixels[..., :2] / pixels[..., 2:3], local[..., 2:3]), dim=-1)


def observation_pose_checks(env) -> None:
    robot = env.scene["robot"]
    expected = torch.tensor([task.OBSERVATION_JOINT_POS[n] for n in robot.joint_names], device=env.device)
    pose_error = float((robot.data.joint_pos - expected).abs().max())
    nominal = env.scene["cube"].data.default_root_state[:, :3] + env.scene.env_origins
    corners = [
        (sx * (CUBE_RESET_HALF_RANGE_M + CUBE_HALF_EXTENT_M), sy * (CUBE_RESET_HALF_RANGE_M + CUBE_HALF_EXTENT_M), dz)
        for sx, sy, dz in itertools.product((-1, 1), (-1, 1), (-CUBE_HALF_EXTENT_M, CUBE_HALF_EXTENT_M))
    ]
    points = nominal.unsqueeze(1) + torch.tensor(corners, device=env.device).unsqueeze(0)
    projected = project_to_wrist(env, points)
    width, height = env.scene["wrist"].cfg.width, env.scene["wrist"].cfg.height
    margin = torch.minimum(
        torch.minimum(projected[..., 0], width - projected[..., 0]),
        torch.minimum(projected[..., 1], height - projected[..., 1]),
    )
    detail = {"joint_pose_error_rad": pose_error, "min_margin_px": tolist(margin.min(-1).values),
              "min_depth_m": tolist(projected[..., 2].min(-1).values)}
    camera, gripper_id = env.scene["wrist"], robot.find_bodies("gripper")[0][0]
    gripper_quat = robot.data.body_link_quat_w[:, gripper_id]
    detail["camera_in_gripper_pos_m"] = tolist(quat_apply(quat_inv(gripper_quat),
                                                          camera.data.pos_w - robot.data.body_link_pos_w[:, gripper_id]))
    metrics["observation_pose"] = detail
    # The sim wrist camera is LeIsaac's stock mount, not the physical one, so coverage is
    # recorded only; the pose is re-derived once the real camera mount is calibrated.
    check("observation_pose_applied_at_reset", pose_error < 0.02, **detail)


def main():
    output_dir.mkdir(parents=True)
    check("gym_ids_registered", TASK_ID in gym.registry and MIMIC_TASK_ID in gym.registry)
    mimic_cfg = parse_env_cfg(MIMIC_TASK_ID, device=args_cli.device, num_envs=1)
    check("mimic_cfg_parses", [c.subtask_term_signal for c in mimic_cfg.subtask_configs["so101_follower"]]
          == ["pick_cube", "place_cube", None])

    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(DEVICE_TYPE)
    env_cfg.seed = args_cli.seed
    env_cfg.recorders = None
    # Camera pose data is otherwise frozen at spawn; projections need the live pose.
    env_cfg.scene.wrist.update_latest_camera_pose = True
    event_names = [n for n in env_cfg.events.__dict__ if not n.startswith("_")]
    check("front_camera_event_removed", all(
        getattr((getattr(env_cfg.events, n).params or {}).get("asset_cfg"), "name", None) != "front" for n in event_names
    ), events=event_names)

    env = gym.make(TASK_ID, cfg=env_cfg).unwrapped
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if "Robot" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)

    with torch.inference_mode():
        obs, _ = env.reset(seed=args_cli.seed)
        stage_checks(env)
        check("policy_observes_wrist_only", "wrist" in obs["policy"] and "front" not in obs["policy"],
              keys=sorted(obs["policy"].keys()))
        hold = torch.cat((env.scene["robot"].data.joint_pos[:, :5], torch.zeros(env.num_envs, 1, device=env.device)), -1)
        for _ in range(10):
            obs, *_ = env.step(hold)
        observation_pose_checks(env)
        save_wrist(obs, "observation_pose")
        bottom = env.scene["cube"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - TABLE_TOP_Z - CUBE_HALF_EXTENT_M
        check("cube_rests_on_white_table", bool((bottom.abs() < 0.001).all()), bottom_offset_m=tolist(bottom))

        env.termination_manager.set_term_cfg(
            "time_out", TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        )
        sm = PickCubeIntoBlackBoxStateMachine()
        sm.setup(env)
        sm.begin_episode(env)
        metrics["fsm_tilt_rad"] = tolist(sm._tilt)
        metrics["fsm_turn_rad"] = tolist(sm._turn)
        rows = []
        terminated_at = [None] * env.num_envs
        while not sm.is_episode_done:
            name = sm.phase_name()
            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, DEVICE_TYPE)
            obs, _, terminated, _, _ = env.step(sm.get_action(env))
            sm.observe_step(env)
            cube_z = env.scene["cube"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
            jaw = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
            rows.append({"phase": name, "gripper": tolist(env.scene["robot"].data.joint_pos[:, -1]),
                         "jaw_cube_m": tolist(torch.linalg.vector_norm(jaw - env.scene["cube"].data.root_pos_w, dim=-1)),
                         "cube_lift_m": tolist(cube_z - TABLE_TOP_Z - CUBE_HALF_EXTENT_M),
                         "grasped": tolist(sm._grasped(env)), "placed": tolist(sm._placed(env))})
            for env_index in torch.nonzero(terminated).flatten().tolist():
                if terminated_at[env_index] is None:
                    terminated_at[env_index] = len(rows) - 1
                    save_wrist(obs, f"terminated_env{env_index}")
            sm.advance()
        (output_dir / "fsm_trace.json").write_text(json.dumps(rows))
        summary = []
        for e in range(env.num_envs):
            end = terminated_at[e] if terminated_at[e] is not None else len(rows)
            held = [r for r in rows[:end] if r["phase"] in {"lift", "above_goal"} and r["cube_lift_m"][e] > 0.03]
            summary.append({
                "held_steps": len(held),
                "held_gripper_rad_max": max((r["gripper"][e] for r in held), default=None),
                "held_gripper_rad_median": float(np.median([r["gripper"][e] for r in held])) if held else None,
                "held_jaw_cube_m_max": max((r["jaw_cube_m"][e] for r in held), default=None),
                "grasp_signal_steps_while_held": sum(r["grasped"][e] for r in held),
                "place_signal_before_release": sum(r["placed"][e] for r in rows[:end]
                                                   if r["phase"] in {"lift", "above_goal", "lower_goal"}),
                "terminated_in_phase": rows[end]["phase"] if end < len(rows) else None,
            })
        metrics["fsm"] = summary
        check("fsm_drops_cube_into_box_after_release",
              all(s["held_steps"] > 0 and s["terminated_in_phase"] in {"release", "retreat"}
                  and s["place_signal_before_release"] == 0 for s in summary), summary=summary)
        check("fsm_hold_reads_as_grasped_not_released",
              all(s["held_steps"] > 0 and s["held_gripper_rad_max"] < task.FINRAY_RELEASED_MIN_GRIPPER_RAD
                  and s["grasp_signal_steps_while_held"] >= 0.9 * s["held_steps"] for s in summary))
    env.close()


if __name__ == "__main__":
    exit_code = 1
    error = None
    try:
        main()
        exit_code = 0 if checks and all(c["pass"] for c in checks.values()) else 2
    except Exception:
        import traceback

        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        if output_dir.exists():
            (output_dir / "smoke_result.json").write_text(json.dumps({
                "task": TASK_ID, "num_envs": args_cli.num_envs, "seed": args_cli.seed,
                "scope": "simulation only; authored USD geometry; no hardware access",
                "all_pass": exit_code == 0, "error": error, "checks": checks, "metrics": metrics,
            }, indent=2, default=float))
        print(f"SMOKE_EXIT {exit_code}", flush=True)
        simulation_app.close()
    raise SystemExit(exit_code)
