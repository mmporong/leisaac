"""Privileged-state recovery oracle for the SO-101 cube-into-box task."""

from __future__ import annotations

import os

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul
from leisaac.tasks.lift_cube.mdp import object_grasped
from leisaac.tasks.pick_cube_into_box.mdp import cube_released_in_box

from .base import StateMachineBase


_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_GRASP_OFFSET = (-0.012, 0.020, 0.090)


class PickCubeIntoBoxStateMachine(StateMachineBase):
    """Generate a pick/place recovery using privileged simulator poses.

    This oracle labels states visited by a learned policy. It is not a policy
    intended for deployment on the physical robot.
    """

    _PHASES = (
        ("above_cube", 120),
        ("at_cube", 90),
        ("grasp", 150),
        ("lift", 100),
        ("above_box", 120),
        ("lower_box", 90),
        ("release", 60),
        ("retreat", 80),
    )

    def __init__(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._phase_names = [name for name, _ in self._PHASES]
        self._phase_ends = []
        total = 0
        for _, duration in self._PHASES:
            total += duration
            self._phase_ends.append(total)
        self._max_steps = total
        self._waypoints: dict[str, torch.Tensor] = {}
        self._target_quat_w: torch.Tensor | None = None

    def setup(self, env) -> None:
        """Waypoints are initialized from each restored simulator state."""

    def begin_episode(self, env) -> None:
        """Build waypoints from the current state, including policy failures."""
        cube_pos = env.scene["cube"].data.root_pos_w.clone()
        box_pos = env.scene["box_target"].data.root_pos_w.clone()
        current_pos = env.scene["ee_frame"].data.target_pos_w[:, 0, :].clone()
        canonical_quat_w = quat_from_euler_xyz(
            torch.tensor(0.0, device=env.device),
            torch.tensor(0.0, device=env.device),
            torch.tensor(0.0, device=env.device),
        ).repeat(env.num_envs, 1)
        self._target_quat_w = canonical_quat_w

        above_cube = cube_pos.clone()
        above_cube[:, 0] += _GRASP_OFFSET[0]
        above_cube[:, 1] += _GRASP_OFFSET[1]
        above_cube[:, 2] += 0.22
        at_cube = cube_pos.clone()
        at_cube[:, 0] += _GRASP_OFFSET[0]
        at_cube[:, 1] += _GRASP_OFFSET[1]
        at_cube[:, 2] += _GRASP_OFFSET[2]
        lift = cube_pos.clone()
        lift[:, 0] += _GRASP_OFFSET[0]
        lift[:, 1] += _GRASP_OFFSET[1]
        lift[:, 2] += 0.27
        above_box = box_pos.clone()
        above_box[:, 0] += _GRASP_OFFSET[0]
        above_box[:, 1] += _GRASP_OFFSET[1]
        above_box[:, 2] += 0.25
        lower_box = box_pos.clone()
        lower_box[:, 0] += _GRASP_OFFSET[0]
        lower_box[:, 1] += _GRASP_OFFSET[1]
        lower_box[:, 2] += 0.15

        self._waypoints = {
            "start": current_pos,
            "above_cube": above_cube,
            "at_cube": at_cube,
            "grasp": at_cube,
            "lift": lift,
            "above_box": above_box,
            "lower_box": lower_box,
            "release": lower_box,
            "retreat": above_box,
        }

        placed = cube_released_in_box(
            env,
            cube_cfg=SceneEntityCfg("cube"),
            box_cfg=SceneEntityCfg("box_target"),
            robot_cfg=SceneEntityCfg("robot"),
        )
        grasped = object_grasped(
            env,
            robot_cfg=SceneEntityCfg("robot"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg("cube"),
        )
        if bool(placed.all().item()):
            start_phase = self._phase_names.index("retreat")
        elif bool(grasped.all().item()):
            start_phase = self._phase_names.index("lift")
        else:
            start_phase = 0

        self._step_count = 0 if start_phase == 0 else self._phase_ends[start_phase - 1]
        self._episode_done = False

    def check_success(self, env) -> bool:
        success = cube_released_in_box(
            env,
            cube_cfg=SceneEntityCfg("cube"),
            box_cfg=SceneEntityCfg("box_target"),
            robot_cfg=SceneEntityCfg("robot"),
        )
        self._print_diagnostics(env, "episode_end")
        return bool(success.all().item())

    def _print_diagnostics(self, env, boundary: str) -> None:
        if os.environ.get("LEISAAC_SM_DIAGNOSTICS") != "1":
            return
        jaw_pos = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
        cube_pos = env.scene["cube"].data.root_pos_w
        grasped = object_grasped(
            env,
            robot_cfg=SceneEntityCfg("robot"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg("cube"),
        )
        placed = cube_released_in_box(
            env,
            cube_cfg=SceneEntityCfg("cube"),
            box_cfg=SceneEntityCfg("box_target"),
            robot_cfg=SceneEntityCfg("robot"),
        )
        print(
            "SM_DIAGNOSTIC",
            {
                "boundary": boundary,
                "step": self._step_count,
                "jaw_cube_distance": torch.linalg.vector_norm(jaw_pos - cube_pos, dim=1).detach().cpu().tolist(),
                "cube_xyz": cube_pos.detach().cpu().tolist(),
                "jaw_xyz": jaw_pos.detach().cpu().tolist(),
                "gripper_joint": env.scene["robot"].data.joint_pos[:, -1].detach().cpu().tolist(),
                "joint_pos": env.scene["robot"].data.joint_pos.detach().cpu().tolist(),
                "joint_target": env.scene["robot"].data.joint_pos_target.detach().cpu().tolist(),
                "grasped": grasped.detach().cpu().tolist(),
                "placed": placed.detach().cpu().tolist(),
            },
            flush=True,
        )

    def _phase(self) -> tuple[int, str, int, int]:
        start = 0
        for index, ((name, _), end) in enumerate(zip(self._PHASES, self._phase_ends, strict=True)):
            if self._step_count < end:
                return index, name, start, end
            start = end
        return len(self._PHASES) - 1, self._PHASES[-1][0], self._phase_ends[-2], self._phase_ends[-1]

    def get_action(self, env) -> torch.Tensor:
        if not self._waypoints or self._target_quat_w is None:
            self.begin_episode(env)

        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)
        index, name, phase_start, phase_end = self._phase()
        if self._step_count == phase_start:
            self._print_diagnostics(env, f"start_{name}")
        previous_name = "start" if index == 0 else self._phase_names[index - 1]
        alpha = (self._step_count - phase_start + 1) / max(phase_end - phase_start, 1)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        target_pos_w = torch.lerp(self._waypoints[previous_name], self._waypoints[name], alpha)

        # The second frame is the jaw centre used by the task's own
        # ``object_grasped`` predicate.  Align that frame with the cube during
        # approach/closure instead of relying only on a hand-tuned wrist
        # offset.  This keeps the privileged oracle valid for policy states
        # whose wrist pose differs from the original demonstrations.
        if name in {"at_cube", "grasp"}:
            gripper_pos_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
            jaw_pos_w = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
            cube_pos_w = env.scene["cube"].data.root_pos_w
            target_pos_w = gripper_pos_w + (cube_pos_w - jaw_pos_w)

        robot_base_pos_w = robot.data.root_pos_w
        robot_base_quat_w = robot.data.root_quat_w
        target_pos_local = quat_apply(quat_inv(robot_base_quat_w), target_pos_w - robot_base_pos_w)
        target_quat_local = quat_mul(quat_inv(robot_base_quat_w), self._target_quat_w)
        gripper = _GRIPPER_CLOSE if name in {"grasp", "lift", "above_box", "lower_box"} else _GRIPPER_OPEN
        gripper_cmd = torch.full((env.num_envs, 1), gripper, device=env.device)
        return torch.cat((target_pos_local, target_quat_local, gripper_cmd), dim=-1)

    def advance(self) -> None:
        self._step_count += 1
        self._episode_done = self._step_count >= self._max_steps

    def reset(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._waypoints = {}
        self._target_quat_w = None

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done
