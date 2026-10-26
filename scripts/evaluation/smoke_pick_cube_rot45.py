"""Headless integration smoke for LeIsaac-SO101-PickCubeRot45-v0 (simulation only, no hardware).

Checks the scene build, goal geometry about the measured pan axis, resting height,
partial reset, the stable-release termination, and runs the privileged FSM once to
measure the gripper angle while a 4 cm cube is held.

Usage:
    python scripts/evaluation/smoke_pick_cube_rot45.py --output_dir <new dir> --num_envs 2
"""

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--output_dir", type=Path, required=True, help="New directory for evidence files.")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 2:
    raise SystemExit("--num_envs must be at least 2 for the partial reset check")
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
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from leisaac.datagen.state_machine import PickCubeRot45StateMachine  # noqa: E402
from leisaac.datagen.state_machine.pick_cube_rot45 import GRIPPER_CLOSE_RAD, GRIPPER_OPEN_RAD  # noqa: E402
from leisaac.datagen.state_machine.so101_kinematics import (  # noqa: E402
    ARM_JOINT_NAMES,
    So101ArmKinematics,
    quat_wxyz_to_matrix,
)
from leisaac.tasks.lift_cube.mdp import object_grasped  # noqa: E402
from leisaac.tasks.pick_cube_rot45 import pick_cube_rot45_env_cfg as task  # noqa: E402
from leisaac.tasks.pick_cube_rot45.mdp import place_cube_at_goal, rotated_goal_pose  # noqa: E402
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

TASK_ID = "LeIsaac-SO101-PickCubeRot45-v0"
MIMIC_TASK_ID = "LeIsaac-SO101-PickCubeRot45-Mimic-v0"
DEVICE_TYPE = PickCubeRot45StateMachine.TELEOP_DEVICE

checks: dict[str, dict] = {}
metrics: dict[str, object] = {}


def check(name: str, passed: bool, **detail) -> None:
    checks[name] = {"pass": bool(passed), **detail}
    print(f"SMOKE_CHECK {'PASS' if passed else 'FAIL'} {name} {json.dumps(detail, default=float)}", flush=True)


def tolist(value):
    return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value


def hold_action(env, gripper: torch.Tensor) -> torch.Tensor:
    """Hold the arm joints where they are and command the gripper joint target (rad)."""
    robot = env.scene["robot"]
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    return torch.cat((robot.data.joint_pos[:, arm_ids], gripper.view(-1, 1)), dim=-1)


def kinematics_checks(env) -> None:
    """Compare the USD-derived forward kinematics with simulated gripper poses at random joints."""
    robot = env.scene["robot"]
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    gripper_id, base_id = robot.find_bodies("gripper")[0][0], robot.find_bodies("base")[0][0]
    arm = So101ArmKinematics.from_stage(omni.usd.get_context().get_stage(), "/World/envs/env_0/Robot/joints")
    limits = robot.data.soft_joint_pos_limits[:, arm_ids]
    generator = torch.Generator(device="cpu").manual_seed(5)
    position_errors, rotation_errors = [], []
    for _ in range(3):
        fraction = torch.rand(env.num_envs, len(arm_ids), generator=generator).to(env.device) * 0.6 + 0.2
        arm_pos = limits[..., 0] + fraction * (limits[..., 1] - limits[..., 0])
        joints = robot.data.default_joint_pos.clone()
        joints[:, arm_ids] = arm_pos
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        robot.set_joint_position_target(joints)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        measured = robot.data.joint_pos[:, arm_ids]
        base_pose = robot.data.body_link_pose_w[:, base_id]
        sim_pos = quat_apply(quat_inv(base_pose[:, 3:]), robot.data.body_link_pos_w[:, gripper_id] - base_pose[:, :3])
        sim_quat = quat_mul(quat_inv(base_pose[:, 3:]), robot.data.body_link_quat_w[:, gripper_id])
        fk_pos, fk_rot = arm.forward(measured)
        position_errors.append(torch.linalg.vector_norm(fk_pos.float() - sim_pos, dim=-1))
        sim_rot = quat_wxyz_to_matrix(sim_quat.double())
        rotation_errors.append(torch.linalg.matrix_norm(fk_rot - sim_rot))
    detail = {"max_position_error_m": float(torch.cat(position_errors).max()),
              "max_rotation_error_fro": float(torch.cat(rotation_errors).max())}
    metrics["fk_vs_sim"] = detail
    check("usd_forward_kinematics_matches_sim", detail["max_position_error_m"] < 1e-4
          and detail["max_rotation_error_fro"] < 1e-3, **detail)


def yaw_of(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def shoulder_xy(env) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.body_link_pos_w[:, robot.find_bodies(task.PAN_BODY_NAME)[0][0], :2]


def goal_consistency(env, env_ids: torch.Tensor, cube_pose: torch.Tensor) -> dict:
    robot = env.scene["robot"]
    pan = robot.data.body_link_pos_w[env_ids, robot.find_bodies(task.PAN_BODY_NAME)[0][0]]
    expected = rotated_goal_pose(
        cube_pose[env_ids], pan, env.scene.env_origins[env_ids, 2] + task.TABLE_TOP_Z,
        cube_half_extent_m=task.CUBE_HALF_EXTENT_M, angle_rad=task.GOAL_ANGLE_RAD,
    )
    goal = env.scene["goal_marker"].data.root_pose_w[env_ids]
    radius_cube = torch.linalg.vector_norm(cube_pose[env_ids, :2] - pan[:, :2], dim=-1)
    radius_goal = torch.linalg.vector_norm(goal[:, :2] - pan[:, :2], dim=-1)
    turn = wrap(torch.atan2(goal[:, 1] - pan[:, 1], goal[:, 0] - pan[:, 0])
                - torch.atan2(cube_pose[env_ids, 1] - pan[:, 1], cube_pose[env_ids, 0] - pan[:, 0]))
    return {
        "position_error_m": tolist(torch.linalg.vector_norm(goal[:, :3] - expected[:, :3], dim=-1)),
        "radius_difference_m": tolist((radius_goal - radius_cube).abs()),
        "azimuth_turn_rad": tolist(turn),
        "yaw_turn_rad": tolist(wrap(yaw_of(goal[:, 3:]) - yaw_of(cube_pose[env_ids, 3:]))),
        "goal_height_above_table_m": tolist(goal[:, 2] - env.scene.env_origins[env_ids, 2] - task.TABLE_TOP_Z),
        "goal_local_xy": tolist(goal[:, :2] - env.scene.env_origins[env_ids, :2]),
    }


def goal_ok(result: dict) -> bool:
    return (max(result["position_error_m"]) < 1e-5 and max(result["radius_difference_m"]) < 1e-5
            and all(abs(t - task.GOAL_ANGLE_RAD) < 1e-4 for t in result["azimuth_turn_rad"])
            and all(abs(t - task.GOAL_ANGLE_RAD) < 1e-4 for t in result["yaw_turn_rad"])
            and all(abs(h - task.CUBE_HALF_EXTENT_M) < 1e-6 for h in result["goal_height_above_table_m"]))


def stage_checks(env) -> None:
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for env_index in range(env.num_envs):
        root = f"/World/envs/env_{env_index}"
        source_cube = stage.GetPrimAtPath(f"{root}/Scene/{task.SOURCE_CUBE_CHILD}")
        check(f"env{env_index}_source_30mm_cube_inactive", source_cube.IsValid() and not source_cube.IsActive(),
              path=str(source_cube.GetPath()))
        cube = stage.GetPrimAtPath(f"{root}/Cube")
        size = list(cache.ComputeUntransformedBound(cube).ComputeAlignedRange().GetSize())
        check(f"env{env_index}_cube_is_40mm", all(abs(s - task.CUBE_SIZE_M) < 1e-6 for s in size), size_m=size)
        marker = stage.GetPrimAtPath(f"{root}/GoalMarker")
        colliders = [p for p in Usd.PrimRange(marker) if p.HasAPI(UsdPhysics.CollisionAPI)]
        check(
            f"env{env_index}_goal_marker_invisible_kinematic_noncolliding",
            UsdGeom.Imageable(marker).ComputeVisibility() == UsdGeom.Tokens.invisible
            and bool(UsdPhysics.RigidBodyAPI(marker).GetKinematicEnabledAttr().Get())
            and all(not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() for p in colliders),
            colliders=[str(p.GetPath()) for p in colliders],
        )
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/World/envs/env_0/Robot/joints/shoulder_pan"))
    metrics["pan_joint_usd"] = {
        "axis": joint.GetAxisAttr().Get(),
        "local_pos0": list(joint.GetLocalPos0Attr().Get()),
        "local_rot0_wxyz": [joint.GetLocalRot0Attr().Get().GetReal(), *joint.GetLocalRot0Attr().Get().GetImaginary()],
        "body0": [str(t) for t in joint.GetBody0Rel().GetTargets()],
        "body1": [str(t) for t in joint.GetBody1Rel().GetTargets()],
    }


def pan_axis_checks(env) -> None:
    """Rotate shoulder_pan by a known angle and verify the gripper circles the shoulder link origin."""
    robot = env.scene["robot"]
    pan_id = robot.find_joints("shoulder_pan")[0][0]
    gripper_id = robot.find_bodies("gripper")[0][0]
    base_id = robot.find_bodies("base")[0][0]

    def settle(joint_pos):
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        robot.set_joint_position_target(joint_pos)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)

    joints = robot.data.default_joint_pos.clone()
    settle(joints)
    shoulder_0, gripper_0 = shoulder_xy(env).clone(), robot.data.body_link_pos_w[:, gripper_id, :2].clone()
    delta = 0.5
    joints[:, pan_id] += delta
    settle(joints)
    shoulder_1, gripper_1 = shoulder_xy(env).clone(), robot.data.body_link_pos_w[:, gripper_id, :2].clone()

    local = metrics["pan_joint_usd"]["local_pos0"]
    base_pose = robot.data.body_link_pose_w[:, base_id]
    analytic = base_pose[:, :3] + quat_apply(base_pose[:, 3:], torch.tensor(local, device=env.device).expand(env.num_envs, 3))
    radius_0 = torch.linalg.vector_norm(gripper_0 - shoulder_0, dim=-1)
    radius_1 = torch.linalg.vector_norm(gripper_1 - shoulder_0, dim=-1)
    world_turn = wrap(torch.atan2(*(gripper_1 - shoulder_0).flip(-1).unbind(-1))
                      - torch.atan2(*(gripper_0 - shoulder_0).flip(-1).unbind(-1)))
    detail = {
        "shoulder_shift_m": tolist(torch.linalg.vector_norm(shoulder_1 - shoulder_0, dim=-1)),
        "shoulder_vs_usd_joint_origin_m": tolist(torch.linalg.vector_norm(shoulder_0 - analytic[:, :2], dim=-1)),
        "gripper_radius_change_m": tolist((radius_1 - radius_0).abs()),
        "gripper_world_turn_rad": tolist(world_turn),
        "commanded_pan_delta_rad": delta,
        "shoulder_local_xy": tolist(shoulder_0 - env.scene.env_origins[:, :2]),
    }
    metrics["pan_axis"] = detail
    check(
        "pan_axis_is_shoulder_origin",
        max(detail["shoulder_shift_m"]) < 1e-4 and max(detail["shoulder_vs_usd_joint_origin_m"]) < 1e-4
        and max(detail["gripper_radius_change_m"]) < 1e-3
        and all(abs(abs(t) - delta) < 0.02 for t in detail["gripper_world_turn_rad"]),
        **detail,
    )
    # Positive joint motion turns the arm one way about world +Z; recorded for sim-to-real sign alignment.
    metrics["positive_shoulder_pan_world_z_sign"] = int(math.copysign(1, detail["gripper_world_turn_rad"][0]))


def run_teleport_window(env, label, offsets_xy, gripper_cmd, steps):
    """Teleport cubes onto their goals (with XY offsets) and record termination per step."""
    cube, goal = env.scene["cube"], env.scene["goal_marker"]
    pose = goal.data.root_pose_w.clone()
    pose[:, :2] += offsets_xy
    pose[:, 2] += 0.0005
    cube.write_root_pose_to_sim(pose)
    cube.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
    terminated_steps = [[] for _ in range(env.num_envs)]
    for step in range(steps):
        _, _, terminated, _, _ = env.step(hold_action(env, gripper_cmd))
        for env_index in torch.nonzero(terminated).flatten().tolist():
            terminated_steps[env_index].append(step + 1)
    metrics[f"teleport_{label}"] = {"offsets_xy_m": tolist(offsets_xy), "gripper_cmd": tolist(gripper_cmd),
                                    "terminated_steps": terminated_steps}
    return terminated_steps


def main():
    output_dir.mkdir(parents=True)
    required_steps = None
    check("gym_ids_registered", TASK_ID in gym.registry and MIMIC_TASK_ID in gym.registry)
    mimic_cfg = parse_env_cfg(MIMIC_TASK_ID, device=args_cli.device, num_envs=1)
    check("mimic_cfg_parses", [c.subtask_term_signal for c in mimic_cfg.subtask_configs["so101_follower"]]
          == ["pick_cube", "place_cube_at_goal", None],
          object_refs=[c.object_ref for c in mimic_cfg.subtask_configs["so101_follower"]])

    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(DEVICE_TYPE)
    env_cfg.seed = args_cli.seed
    env_cfg.recorders = None
    metrics["table_top_z_authored"] = task.authored_table_top_z()
    metrics["cube_init_local"] = list(env_cfg.scene.cube.init_state.pos)
    metrics["event_order"] = [name for name in env_cfg.events.__dict__ if not name.startswith("_")]
    check("goal_event_runs_after_cube_randomization",
          metrics["event_order"][-1] == "reset_goal_from_cube", order=metrics["event_order"])

    env = gym.make(TASK_ID, cfg=env_cfg).unwrapped
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if "Robot" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
    required_steps = math.ceil(task.RELEASE_HOLD_TIME_S / env.step_dt - 1e-9)
    metrics["step_dt_s"] = env.step_dt
    metrics["release_required_steps"] = required_steps
    open_cmd = torch.full((env.num_envs,), GRIPPER_OPEN_RAD, device=env.device)
    all_ids = torch.arange(env.num_envs, device=env.device)

    with torch.inference_mode():
        obs, _ = env.reset(seed=args_cli.seed)
        stage_checks(env)
        first = goal_consistency(env, all_ids, env.scene["cube"].data.root_pose_w.clone())
        metrics["goal_after_reset"] = first
        check("goal_is_rigid_45deg_left_turn_about_pan", goal_ok(first), **first)
        in_table = all(0.003 + 0.02 <= x <= 0.703 - 0.02 and -0.677 + 0.02 <= y <= -0.027 - 0.02
                       for x, y in first["goal_local_xy"])
        check("goal_on_authored_table", in_table, goal_local_xy=first["goal_local_xy"])

        pan_axis_checks(env)
        kinematics_checks(env)

        obs, _ = env.reset(seed=args_cli.seed + 1)
        goal_before = env.scene["goal_marker"].data.root_pose_w.clone()
        for _ in range(60):
            obs, _, terminated, truncated, _ = env.step(hold_action(env, open_cmd))
        cube = env.scene["cube"].data
        bottom = cube.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - task.TABLE_TOP_Z - task.CUBE_HALF_EXTENT_M
        finite = all(torch.isfinite(v).all().item() for k, v in obs["policy"].items() if v.dtype.is_floating_point)
        check("cube_rests_on_table", bool((bottom.abs() < 0.001).all() and torch.isfinite(bottom).all()),
              bottom_offset_m=tolist(bottom), lin_speed=tolist(torch.linalg.vector_norm(cube.root_lin_vel_w, dim=-1)))
        check("goal_marker_static_while_stepping",
              float((env.scene["goal_marker"].data.root_pose_w - goal_before).abs().max()) < 1e-6)
        check("policy_observations_finite", finite and not bool(terminated.any() or truncated.any()))
        for camera in ("front", "wrist"):
            image = obs["policy"][camera][0].detach().cpu().numpy().astype(np.uint8)
            Image.fromarray(image[..., :3]).save(output_dir / f"env0_{camera}_after_settle.png")
            metrics[f"{camera}_image_shape"] = list(image.shape)
            metrics[f"{camera}_image_std"] = float(image[..., :3].std())

        cube_before = env.scene["cube"].data.root_pose_w.clone()
        goal_before = env.scene["goal_marker"].data.root_pose_w.clone()
        env._reset_idx(torch.tensor([1], device=env.device))
        env.scene.write_data_to_sim()
        env.sim.forward()
        cube_after = env.scene["cube"].data.root_pose_w.clone()
        goal_after = env.scene["goal_marker"].data.root_pose_w.clone()
        partial = goal_consistency(env, torch.tensor([1], device=env.device), cube_after)
        check(
            "partial_reset_keeps_other_env_goal",
            float((goal_after[0] - goal_before[0]).abs().max()) == 0.0
            and float((cube_after[0] - cube_before[0]).abs().max()) == 0.0
            and float((cube_after[1, :2] - cube_before[1, :2]).abs().max()) > 1e-4
            and goal_ok(partial),
            env1_goal=partial,
        )

        for _ in range(20):
            env.step(hold_action(env, open_cmd))
        zero = torch.zeros((env.num_envs, 2), device=env.device)
        offsets = zero.clone()
        offsets[1, 0] = task.MAX_GOAL_XY_ERROR_M + 0.005
        window_a = run_teleport_window(env, "a_exact_vs_offset", offsets, open_cmd, required_steps + 15)
        # The teleported cube settles for a step or two before it is stationary.
        settle_window = range(required_steps, required_steps + 5)
        check("release_success_after_hold_time", window_a[0][:1] != [] and window_a[0][0] in settle_window,
              terminated_steps=window_a[0], required_steps=required_steps)
        check("xy_offset_never_succeeds", window_a[1] == [], terminated_steps=window_a[1])

        for _ in range(20):
            env.step(hold_action(env, open_cmd))
        gripper = open_cmd.clone()
        gripper[0] = GRIPPER_CLOSE_RAD
        for _ in range(20):
            env.step(hold_action(env, gripper))
        metrics["closed_gripper_joint_rad"] = tolist(env.scene["robot"].data.joint_pos[:, -1])
        after_reset = goal_consistency(env, all_ids, env.scene["cube"].data.root_pose_w.clone())
        window_b = run_teleport_window(env, "b_closed_vs_open", zero, gripper, required_steps + 15)
        check("closed_gripper_never_succeeds", window_b[0] == [], terminated_steps=window_b[0])
        check("env1_release_success_after_hold_time", window_b[1][:1] != [] and window_b[1][0] in settle_window,
              terminated_steps=window_b[1], required_steps=required_steps)
        metrics["goal_before_window_b"] = after_reset

        obs, _ = env.reset(seed=args_cli.seed + 2)
        # Like generate.py: no episode time limit for the oracle, leader-style gripper effort.
        env.termination_manager.set_term_cfg(
            "time_out", TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        )
        sm = PickCubeRot45StateMachine()
        sm.setup(env)
        sm.begin_episode(env)
        metrics["fsm_tilt_rad"] = tolist(sm._tilt)
        metrics["fsm_reach_error_m"] = tolist(sm.reach_error_m)
        trace = []
        robot = env.scene["robot"]
        while not sm.is_episode_done:
            name = sm.phase_name()
            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, DEVICE_TYPE)
            actions = sm.get_action(env)
            _, _, terminated, _, _ = env.step(actions)
            sm.observe_step(env)
            cube_pos = env.scene["cube"].data.root_pos_w
            jaw = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
            trace.append({
                "step": sm._step_count, "phase": name,
                "gripper_rad": tolist(robot.data.joint_pos[:, -1]),
                "cube_height_above_rest_m": tolist(cube_pos[:, 2] - env.scene.env_origins[:, 2]
                                                   - task.TABLE_TOP_Z - task.CUBE_HALF_EXTENT_M),
                "cube_goal_xy_m": tolist(torch.linalg.vector_norm(
                    cube_pos[:, :2] - env.scene["goal_marker"].data.root_pos_w[:, :2], dim=-1)),
                "jaw_cube_m": tolist(torch.linalg.vector_norm(jaw - cube_pos, dim=-1)),
                "grasped": tolist(object_grasped(env, SceneEntityCfg("robot"), SceneEntityCfg("ee_frame"),
                                                 SceneEntityCfg("cube"), diff_threshold=task.GRASP_JAW_DISTANCE_M,
                                                 grasp_threshold=task.GRASPED_MAX_GRIPPER_RAD)),
                "placed": tolist(place_cube_at_goal(env, **task.RELEASE_PARAMS)),
                "terminated": tolist(terminated),
            })
            sm.advance()
        (output_dir / "fsm_trace.json").write_text(json.dumps(trace))
        summary = []
        for env_index in range(env.num_envs):
            end = next((i for i, r in enumerate(trace) if r["terminated"][env_index]), len(trace))
            rows = trace[:end]
            held = [r for r in rows if r["phase"] in {"lift", "above_goal"} and r["cube_height_above_rest_m"][env_index] > 0.03]
            first_success = trace[end]["phase"] if end < len(trace) else None
            summary.append({
                "held_steps": len(held),
                "held_gripper_rad": {
                    "min": min(r["gripper_rad"][env_index] for r in held) if held else None,
                    "median": float(np.median([r["gripper_rad"][env_index] for r in held])) if held else None,
                    "max": max(r["gripper_rad"][env_index] for r in held) if held else None,
                },
                "held_jaw_cube_m_max": max(r["jaw_cube_m"][env_index] for r in held) if held else None,
                "grasp_signal_steps_while_held": sum(r["grasped"][env_index] for r in held),
                "place_signal_steps_before_release": sum(
                    r["placed"][env_index] for r in rows if r["phase"] in {"lift", "above_goal", "lower_goal"}),
                "terminated_at_step": trace[end]["step"] if end < len(trace) else None,
                "terminated_in_phase": first_success,
                "final_cube_goal_xy_m": rows[-1]["cube_goal_xy_m"][env_index] if rows else None,
                "final_cube_height_above_rest_m": rows[-1]["cube_height_above_rest_m"][env_index] if rows else None,
            })
        metrics["fsm"] = summary
        check("fsm_lifts_cube", all(s["held_steps"] > 0 for s in summary), summary=summary)
        check("fsm_hold_reads_as_grasped_not_released",
              all(s["held_steps"] > 0 and s["held_gripper_rad"]["max"] < task.RELEASED_MIN_GRIPPER_RAD
                  and s["grasp_signal_steps_while_held"] >= 0.9 * s["held_steps"] for s in summary))
        check("fsm_success_only_after_release",
              all(s["terminated_in_phase"] in {"release", "retreat"} and s["place_signal_steps_before_release"] == 0
                  for s in summary))
        check("fsm_cube_lands_on_goal",
              all(s["final_cube_goal_xy_m"] is not None and s["final_cube_goal_xy_m"] <= task.MAX_GOAL_XY_ERROR_M
                  and abs(s["final_cube_height_above_rest_m"]) <= task.MAX_BOTTOM_ERROR_M for s in summary))

    env.close()
    return checks, metrics


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
