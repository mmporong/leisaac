"""Joint-space privileged pick-and-place oracle shared by SO-101 cube tasks."""

from __future__ import annotations

import math
import os

import torch
from isaaclab.utils.math import quat_apply, quat_inv
from leisaac.tasks.pick_cube_rot45.pick_cube_rot45_env_cfg import PAN_BODY_NAME

from .base import StateMachineBase
from .so101_kinematics import ARM_JOINT_NAMES, So101ArmKinematics

# Gripper joint targets (rad) for the joint-position action used by leader teleoperation.
GRIPPER_OPEN_RAD = 1.0
GRIPPER_CLOSE_RAD = 0.0
# Cube center in the gripper link frame at grasp, from the SO101 collision meshes
# (docs/provenance/so101_pick_cube_rot45_20261026.yaml):
# the fixed finger's inner face is at x = -0.008 and both finger tips reach z = -0.104,
# so a 40 mm cube keeps 4 mm from the fixed finger and the tips stay 4 mm above the table.
GRASP_POINT_GRIPPER = (0.016, 0.0, -0.088)
# The finger tips lean away from the pan axis; a vertical gripper needs wrist_flex past
# its limit and far cubes need more lean, so the smallest reachable candidate is used.
GRASP_TILT_CANDIDATES_RAD = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
_REACH_TOLERANCE_M = 0.001
_APPROACH_HEIGHT_M = 0.10
LIFT_HEIGHT_M = 0.10
# Phases end only once every env's grasp point is within this distance of its
# waypoint, or after the extra wait expires.
_PHASE_POINT_TOLERANCE_M = {
    "above_cube": 0.005, "at_cube": 0.003, "lift": 0.005, "above_goal": 0.005, "lower_goal": 0.003,
}
_MAX_EXTRA_WAIT_STEPS = 180
# Long reconfigurations interpolate between solved joint targets; short moves track a
# Cartesian line with warm-started IK.
_JOINT_SPACE_PHASES = frozenset({"above_cube", "above_goal"})
_MAX_JOINT_STEP_RAD = 0.05
_WAYPOINT_IK_ITERATIONS = 150


def _rotate_about_z(vectors: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate ``(N, 3)`` vectors about +Z by per-env angles."""
    c, s = torch.cos(angle), torch.sin(angle)
    x, y, z = vectors.unbind(-1)
    return torch.stack((c * x - s * y, s * x + c * y, z), dim=-1)


class JointSpacePickPlaceOracle(StateMachineBase):
    """Pick the cube, carry it with a pan turn and release it, all as joint targets.

    Position-priority IK on the USD kinematics keeps the 5-DOF arm from fighting an
    unreachable 6-DOF pose. The carried orientation is the grasp orientation turned
    about the pan axis, so placing mostly changes shoulder_pan. Subclasses define the
    place target and the task signals. Actions are absolute joint targets like leader
    teleoperation. This oracle uses simulator poses and is not a deployable policy.
    """

    TELEOP_DEVICE = "so101leader"
    _PHASES = (
        ("above_cube", 120),
        ("at_cube", 90),
        ("grasp", 60),
        ("lift", 90),
        ("above_goal", 120),
        ("lower_goal", 90),
        ("release", 45),
        ("retreat", 60),
    )
    _CLOSED_PHASES = frozenset({"grasp", "lift", "above_goal", "lower_goal"})
    _LIVE_CUBE_PHASES = frozenset({"at_cube", "grasp"})

    def __init__(
        self,
        grasp_point_gripper=GRASP_POINT_GRIPPER,
        tilt_candidates_rad=GRASP_TILT_CANDIDATES_RAD,
        gripper_open_rad: float = GRIPPER_OPEN_RAD,
    ) -> None:
        self.gripper_open_rad = gripper_open_rad
        self.grasp_point_gripper = tuple(grasp_point_gripper)
        if len(self.grasp_point_gripper) != 3 or not all(math.isfinite(v) for v in self.grasp_point_gripper):
            raise ValueError("grasp_point_gripper must contain three finite values")
        self.tilt_candidates_rad = tuple(tilt_candidates_rad)
        if not self.tilt_candidates_rad:
            raise ValueError("tilt_candidates_rad must not be empty")
        self._phase_names = [name for name, _ in self._PHASES]
        self._phase_ends = []
        total = 0
        for _, duration in self._PHASES:
            total += duration
            self._phase_ends.append(total)
        self._max_steps = total
        self._kinematics: So101ArmKinematics | None = None
        self.reset()

    def setup(self, env) -> None:
        import omni.usd

        robot_path = env.scene["robot"].cfg.prim_path.replace("env_.*", "env_0")
        self._kinematics = So101ArmKinematics.from_stage(omni.usd.get_context().get_stage(), f"{robot_path}/joints")

    def reset(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._converged = False
        self._extra_wait = 0
        self._waypoints: dict[str, torch.Tensor] = {}
        self._joint_target: torch.Tensor | None = None

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    def advance(self) -> None:
        _, name, _, phase_end = self._phase()
        at_end = self._step_count == phase_end - 1
        if at_end and name in _PHASE_POINT_TOLERANCE_M and not self._converged and self._extra_wait < _MAX_EXTRA_WAIT_STEPS:
            self._extra_wait += 1
            return
        if at_end:
            self._extra_wait = 0
        self._step_count += 1
        self._episode_done = self._step_count >= self._max_steps

    def begin_episode(self, env) -> None:
        if self._kinematics is None:
            self.setup(env)
        robot = env.scene["robot"]
        self._arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
        self._limits = robot.data.soft_joint_pos_limits[:, self._arm_ids]
        cube = env.scene["cube"].data

        # Close across the cube face pair nearest the pan tangent.
        radial = cube.root_pos_w[:, :2] - self._pan_xy(env)
        self._radial = radial / torch.linalg.vector_norm(radial, dim=-1, keepdim=True)
        w, x, y, z = cube.root_quat_w.unbind(-1)
        cube_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        tangent_yaw = torch.atan2(self._radial[:, 1], self._radial[:, 0]) + math.pi / 2
        closing_yaw = cube_yaw + torch.round((tangent_yaw - cube_yaw) / (math.pi / 2)) * (math.pi / 2)
        self._closing_axis_w = torch.stack(
            (torch.cos(closing_yaw), torch.sin(closing_yaw), torch.zeros_like(closing_yaw)), dim=-1
        )

        up = torch.tensor((0.0, 0.0, 1.0), device=env.device)
        cube_start = cube.root_pos_w.clone()
        place, above_place, self._turn = self._place_targets(env, cube_start)
        self._waypoints = {
            "above_cube": cube_start + _APPROACH_HEIGHT_M * up,
            "at_cube": cube_start,
            "grasp": cube_start,
            "lift": cube_start + LIFT_HEIGHT_M * up,
            "above_goal": above_place,
            "lower_goal": place,
            "release": place,
            "retreat": above_place,
        }
        self._held_point = self._grasp_point(env).clone()
        self._joint_target = robot.data.joint_pos[:, self._arm_ids].clone()
        self._phase_start_joints = self._joint_target.clone()
        self._choose_tilt(env)

        if bool(self._placed(env).all().item()):
            start_phase = self._phase_names.index("retreat")
        elif bool(self._grasped(env).all().item()):
            start_phase = self._phase_names.index("lift")
        else:
            start_phase = 0
        self._step_count = 0 if start_phase == 0 else self._phase_ends[start_phase - 1]
        self._extra_wait = 0
        self._episode_done = False

    def get_action(self, env) -> torch.Tensor:
        if not self._waypoints:
            self.begin_episode(env)
        index, name, phase_start, phase_end = self._phase()
        first_step = self._step_count == phase_start and self._extra_wait == 0
        if first_step:
            self._print_diagnostics(env, f"start_{name}")
            self._phase_start_joints = self._joint_target.clone()
            if name == "lift":
                # Carry the cube where it actually sits in the hand.
                gripper_pos, gripper_quat = self._gripper_pose_w(env)
                cube_pos = env.scene["cube"].data.root_pos_w
                self._held_point = quat_apply(quat_inv(gripper_quat), cube_pos - gripper_pos)
                self._waypoints["grasp"] = cube_pos.clone()
            if name == "above_goal":
                # The goal pose is the carried pose turned about the pan axis; positive pan turns clockwise.
                seed = self._joint_target.clone()
                seed[:, 0] -= self._turn
                self._phase_goal_joints, _ = self._solve(
                    env, seed, self._waypoints["above_goal"], self._held_point, self._turn, _WAYPOINT_IK_ITERATIONS
                )

        alpha = min((self._step_count - phase_start + 1) / max(phase_end - phase_start, 1), 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        point = self._grasp_point(env) if index <= self._phase_names.index("grasp") else self._held_point
        if name == "above_goal":
            turn = alpha * self._turn
        elif index > self._phase_names.index("above_goal"):
            turn = self._turn
        else:
            turn = torch.zeros_like(self._turn)
        if name in _JOINT_SPACE_PHASES:
            goal_joints = self._above_cube_joints if name == "above_cube" else self._phase_goal_joints
            joint_target = torch.lerp(self._phase_start_joints, goal_joints, alpha)
            target_w = self._waypoints[name]
        else:
            target_w = torch.lerp(self._waypoints[self._phase_names[index - 1]], self._waypoints[name], alpha)
            if name in self._LIVE_CUBE_PHASES:
                target_w = env.scene["cube"].data.root_pos_w.clone()
                if name == "at_cube":
                    target_w[:, 2] += (1.0 - alpha) * _APPROACH_HEIGHT_M
            joint_target, _ = self._solve(env, self._joint_target, target_w, point, turn)
        step = (joint_target - self._joint_target).clamp(-_MAX_JOINT_STEP_RAD, _MAX_JOINT_STEP_RAD)
        self._joint_target = self._joint_target + step

        point_error = torch.linalg.vector_norm(self._grasp_point_w(env, point) - target_w, dim=-1)
        self._converged = bool((point_error < _PHASE_POINT_TOLERANCE_M.get(name, math.inf)).all().item())
        gripper = GRIPPER_CLOSE_RAD if name in self._CLOSED_PHASES else self.gripper_open_rad
        gripper_cmd = torch.full((env.num_envs, 1), gripper, device=env.device)
        return torch.cat((self._joint_target, gripper_cmd), dim=-1)

    def observe_step(self, env) -> None:
        self._success(env)

    def check_success(self, env) -> bool:
        success = self._success(env)
        self._print_diagnostics(env, "episode_end")
        return bool(success.all().item())

    def phase_name(self) -> str:
        return self._phase()[1]

    def _phase(self) -> tuple[int, str, int, int]:
        start = 0
        for index, ((name, _), end) in enumerate(zip(self._PHASES, self._phase_ends, strict=True)):
            if self._step_count < end:
                return index, name, start, end
            start = end
        return len(self._PHASES) - 1, self._PHASES[-1][0], self._phase_ends[-2], self._phase_ends[-1]

    def _choose_tilt(self, env) -> None:
        """Pick, per env, the smallest tilt that reaches above and at the cube."""
        point = self._grasp_point(env)
        self._tilt = torch.full((env.num_envs,), self.tilt_candidates_rad[-1], device=env.device)
        self._above_cube_joints = self._joint_target.clone()
        best_error = torch.full((env.num_envs,), math.inf, device=env.device)
        chosen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for tilt in self.tilt_candidates_rad:
            candidate = torch.full((env.num_envs,), tilt, device=env.device)
            above, above_error = self._solve(
                env, self._joint_target, self._waypoints["above_cube"], point, 0.0, _WAYPOINT_IK_ITERATIONS,
                tilt=candidate,
            )
            _, at_error = self._solve(
                env, above, self._waypoints["at_cube"], point, 0.0, _WAYPOINT_IK_ITERATIONS, tilt=candidate
            )
            error = torch.maximum(above_error, at_error)
            take = ~chosen & ((error < _REACH_TOLERANCE_M) | (error < best_error))
            self._tilt = torch.where(take, candidate, self._tilt)
            self._above_cube_joints = torch.where(take.unsqueeze(-1), above, self._above_cube_joints)
            best_error = torch.where(take, error, best_error)
            chosen |= error < _REACH_TOLERANCE_M
        self.reach_error_m = best_error

    def _solve(self, env, seed, target_w, point, turn, iterations: int = 30, tilt=None):
        robot = env.scene["robot"]
        tilt = self._tilt if tilt is None else tilt
        turn = torch.as_tensor(turn, device=env.device, dtype=seed.dtype).expand(env.num_envs)
        finger_dir_w = torch.stack(
            (torch.sin(tilt) * self._radial[:, 0], torch.sin(tilt) * self._radial[:, 1], -torch.cos(tilt)), dim=-1
        )
        to_base = quat_inv(robot.data.root_quat_w)
        return self._kinematics.solve(
            seed,
            quat_apply(to_base, target_w - robot.data.root_pos_w),
            point,
            quat_apply(to_base, _rotate_about_z(finger_dir_w, turn)),
            quat_apply(to_base, _rotate_about_z(self._closing_axis_w, turn)),
            self._limits,
            iterations=iterations,
        )

    def _pan_xy(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        return robot.data.body_link_pos_w[:, robot.find_bodies(PAN_BODY_NAME)[0][0], :2]

    def _gripper_pose_w(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        robot = env.scene["robot"]
        gripper_id = robot.find_bodies("gripper")[0][0]
        return robot.data.body_link_pos_w[:, gripper_id], robot.data.body_link_quat_w[:, gripper_id]

    def _grasp_point_w(self, env, point_in_gripper: torch.Tensor) -> torch.Tensor:
        gripper_pos, gripper_quat = self._gripper_pose_w(env)
        return gripper_pos + quat_apply(gripper_quat, point_in_gripper)

    def _grasp_point(self, env) -> torch.Tensor:
        return torch.tensor(self.grasp_point_gripper, device=env.device).expand(env.num_envs, 3)

    def _place_targets(self, env, cube_pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the release cube position, the position above it and the pan turn (rad, CCW)."""
        raise NotImplementedError

    def _grasped(self, env) -> torch.Tensor:
        raise NotImplementedError

    def _placed(self, env) -> torch.Tensor:
        raise NotImplementedError

    def _success(self, env) -> torch.Tensor:
        raise NotImplementedError

    def _print_diagnostics(self, env, boundary: str) -> None:
        if os.environ.get("LEISAAC_SM_DIAGNOSTICS") != "1":
            return
        print(
            "SM_DIAGNOSTIC",
            {
                "boundary": boundary,
                "step": self._step_count,
                "cube_xyz": env.scene["cube"].data.root_pos_w.detach().cpu().tolist(),
                "gripper_joint": env.scene["robot"].data.joint_pos[:, -1].detach().cpu().tolist(),
                "grasped": self._grasped(env).detach().cpu().tolist(),
                "placed": self._placed(env).detach().cpu().tolist(),
            },
            flush=True,
        )
