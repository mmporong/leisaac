"""Privileged-state oracle for dropping the 4 cm cube into the black box."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from leisaac.tasks.pick_cube_into_box.mdp import cube_placed_in_box, cube_released_in_box, object_grasped
from leisaac.tasks.pick_cube_into_box.pick_cube_into_box_env_cfg import BOX_WALL_Z
from leisaac.tasks.pick_cube_into_black_box.pick_cube_into_black_box_env_cfg import (
    FINRAY_GRASP_JAW_DISTANCE_M,
    FINRAY_GRASPED_MAX_GRIPPER_RAD,
    FINRAY_RELEASED_MIN_GRIPPER_RAD,
)
from leisaac.tasks.pick_cube_rot45.pick_cube_rot45_env_cfg import CUBE_HALF_EXTENT_M

from .joint_space_pick_place import JointSpacePickPlaceOracle

# Cube center in the gripper link frame at grasp: like GRASP_POINT_GRIPPER but 3 mm further
# from the fixed FinRay finger, whose 3 mm tip lip would otherwise touch the cube on descent.
FINRAY_GRASP_POINT_GRIPPER = (0.019, 0.0, -0.088)
# The FinRay hold sits near 0.7 rad, so open well past the release threshold.
FINRAY_GRIPPER_OPEN_RAD = 1.4
# Box walls are 0.07 m tall around BOX_WALL_Z; release the cube just clear of their top.
_BOX_WALL_TOP_Z = BOX_WALL_Z + 0.035
_DROP_CLEARANCE_M = 0.012
_ABOVE_DROP_M = 0.05
_BOX_PARAMS = {
    "cube_cfg": SceneEntityCfg("cube"),
    "box_cfg": SceneEntityCfg("box_target"),
    "robot_cfg": SceneEntityCfg("robot"),
}


class PickCubeIntoBlackBoxStateMachine(JointSpacePickPlaceOracle):
    """Carry the cube over the box center and drop it in, turning about the pan axis.

    The fitted wrist camera looks past the finger tips, about 47 degrees off the finger direction, so a
    straight-down approach with tilted fingers loses the cube from view. Approaching back along the
    fingers from 9 cm keeps the cube in the wrist frame until about 1 cm before the grasp
    (docs/provenance/so101_ik_baseline_20261027.yaml).
    """

    approach_along_fingers = True
    approach_height_m = 0.09

    def __init__(
        self, grasp_point_gripper=FINRAY_GRASP_POINT_GRIPPER, gripper_open_rad=FINRAY_GRIPPER_OPEN_RAD, **kwargs
    ) -> None:
        super().__init__(grasp_point_gripper=grasp_point_gripper, gripper_open_rad=gripper_open_rad, **kwargs)

    def _place_targets(self, env, cube_pos_w):
        box_pos = env.scene["box_target"].data.root_pos_w
        place = box_pos.clone()
        place[:, 2] = env.scene.env_origins[:, 2] + _BOX_WALL_TOP_Z + CUBE_HALF_EXTENT_M + _DROP_CLEARANCE_M
        above = place.clone()
        above[:, 2] += _ABOVE_DROP_M
        pan_xy = self._pan_xy(env)
        cube_dir, box_dir = cube_pos_w[:, :2] - pan_xy, box_pos[:, :2] - pan_xy
        turn = torch.atan2(box_dir[:, 1], box_dir[:, 0]) - torch.atan2(cube_dir[:, 1], cube_dir[:, 0])
        return place, above, torch.atan2(torch.sin(turn), torch.cos(turn))

    def _grasped(self, env) -> torch.Tensor:
        return object_grasped(
            env,
            robot_cfg=SceneEntityCfg("robot"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            object_cfg=SceneEntityCfg("cube"),
            diff_threshold=FINRAY_GRASP_JAW_DISTANCE_M,
            grasp_threshold=FINRAY_GRASPED_MAX_GRIPPER_RAD,
        )

    def _placed(self, env) -> torch.Tensor:
        return cube_placed_in_box(env, **_BOX_PARAMS, open_threshold=FINRAY_RELEASED_MIN_GRIPPER_RAD)

    def _success(self, env) -> torch.Tensor:
        return cube_released_in_box(env, **_BOX_PARAMS, open_threshold_rad=FINRAY_RELEASED_MIN_GRIPPER_RAD)


class PerceivedPickCubeIntoBlackBoxStateMachine(PickCubeIntoBlackBoxStateMachine):
    """The same oracle planned from a cube pose estimated once, e.g. from the wrist camera.

    Only the cube pose changes source: the approach, grasp and carried point use the estimate
    instead of the simulator state, so the motion is what the physical arm would run. Robot
    kinematics and the box position stay as configured. The starting-phase check and the success
    judgement still read the simulator; they evaluate the run and do not steer it.
    """

    def reset(self) -> None:
        super().reset()
        self._estimate: tuple[torch.Tensor, torch.Tensor] | None = None

    def set_cube_estimate(self, position_w: torch.Tensor, yaw: torch.Tensor) -> None:
        """Cube center (N, 3) in the world frame and yaw (N,) in rad; call after reset(), before begin_episode()."""
        half = yaw / 2
        zero = torch.zeros_like(half)
        self._estimate = (position_w.clone(), torch.stack((torch.cos(half), zero, zero, torch.sin(half)), dim=-1))

    def _cube_pose_w(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        if self._estimate is None:
            raise RuntimeError("set_cube_estimate() must be called before begin_episode()")
        return self._estimate

    def _cube_in_hand_w(self, env) -> torch.Tensor:
        # Assume the cube sits where it was estimated when the fingers closed.
        return self._waypoints["grasp"]
