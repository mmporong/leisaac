"""Pick a 4 cm red cube and place it 45 degrees to the left about the SO101 pan axis.

All geometry here is authored simulation geometry. It is not a measurement of
the physical tabletop, cube mass or friction.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage
from isaaclab.utils import configclass
from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_USD_PATH
from leisaac.tasks.lift_cube.lift_cube_env_cfg import (
    LiftCubeEnvCfg,
    LiftCubeSceneCfg,
    ObservationsCfg as LiftCubeObservationsCfg,
)
from leisaac.tasks.pick_cube_into_box.pick_cube_into_box_env_cfg import (
    POLICY_CAMERA_HEIGHT,
    POLICY_CAMERA_WIDTH,
    _wrist_camera_cfg,
)
from leisaac.tasks.template import SingleArmTerminationsCfg
from leisaac.tasks.template import mdp as template_mdp
from leisaac.utils.general_assets import get_stage
from pxr import Usd, UsdGeom

from . import mdp

# Authored scene prims. The USD cube is 30 mm, so it is deactivated and replaced.
SOURCE_TABLE_PRIM = "/world/counter_right_main_group"
SOURCE_CUBE_CHILD = "cube"
SOURCE_CUBE_SIZE_M = 0.03
SOURCE_CUBE_MASS_KG = 0.01

CUBE_SIZE_M = 0.04
CUBE_HALF_EXTENT_M = CUBE_SIZE_M / 2
# Same density as the authored cube; the physical cube mass is unmeasured.
CUBE_MASS_KG = SOURCE_CUBE_MASS_KG * (CUBE_SIZE_M / SOURCE_CUBE_SIZE_M) ** 3

GOAL_ANGLE_RAD = math.pi / 4
# The shoulder link origin sits on the shoulder_pan joint axis.
PAN_BODY_NAME = "shoulder"

# Success policy values, not physical measurements.
MAX_GOAL_XY_ERROR_M = 0.015
MAX_BOTTOM_ERROR_M = 0.003
MAX_LINEAR_SPEED_M_S = 0.03
MAX_ANGULAR_SPEED_RAD_S = 0.5
RELEASE_HOLD_TIME_S = 0.5
# Gripper split between holding the 40 mm cube and released, plus the jaw-frame distance
# for the grasp signal. Policy values chosen from the simulated hold measurement in
# docs/provenance/so101_pick_cube_rot45_20261026.yaml; the 30 mm task values (0.26 rad, 0.02 m)
# misread a 40 mm hold.
GRASPED_MAX_GRIPPER_RAD = 0.55
RELEASED_MIN_GRIPPER_RAD = 0.55
GRASP_JAW_DISTANCE_M = 0.03


# Top of SOURCE_TABLE_PRIM's collider, checked against the USD when the env cfg is built
# (docs/provenance/so101_pick_cube_rot45_20261026.yaml).
TABLE_TOP_Z = 0.040911
_TABLE_TOP_TOLERANCE_M = 1e-6


def authored_table_top_z(usd_path: str = TABLE_WITH_CUBE_USD_PATH) -> float:
    """Return the world-Z top of the authored table collider."""
    # Keep the stage referenced; a prim from a released stage is invalid.
    stage = get_stage(usd_path)
    prim = stage.GetPrimAtPath(SOURCE_TABLE_PRIM)
    if not prim.IsValid():
        raise RuntimeError(f"table prim {SOURCE_TABLE_PRIM} is missing from {usd_path}")
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim)
    return float(bounds.ComputeAlignedRange().GetMax()[2])


def spawn_table_without_source_cube(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the table scene and deactivate the authored 30 mm cube in every spawned copy."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = get_current_stage()
    cube_paths = find_matching_prim_paths(f"{prim_path}/{SOURCE_CUBE_CHILD}")
    if not cube_paths:
        raise RuntimeError(f"authored cube {prim_path}/{SOURCE_CUBE_CHILD} was not found")
    for path in cube_paths:
        stage.GetPrimAtPath(path).SetActive(False)
    return prim


def red_cube_40mm_cfg(source_pos: tuple[float, float, float]) -> RigidObjectCfg:
    """Red 40 mm cube resting on the authored table at the source cube XY."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(source_pos[0], source_pos[1], TABLE_TOP_Z + CUBE_HALF_EXTENT_M)),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE_M, CUBE_SIZE_M, CUBE_SIZE_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.95, dynamic_friction=0.95, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.05, 0.05), roughness=0.6),
        ),
    )


@configclass
class PickCubeRot45SceneCfg(LiftCubeSceneCfg):
    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_WITH_CUBE_USD_PATH, func=spawn_table_without_source_cube),
    )

    # Restored like PickCubeIntoBox because the real SO101 uses a wrist camera.
    wrist: TiledCameraCfg = _wrist_camera_cfg()

    # Invisible and non-colliding so cameras and contacts never see it. Its mass
    # matches the cube so the nearest-object gripper effort heuristic is unchanged.
    goal_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_TOP_Z + CUBE_HALF_EXTENT_M)),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE_M, CUBE_SIZE_M, CUBE_SIZE_M),
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.front.width = POLICY_CAMERA_WIDTH
        self.front.height = POLICY_CAMERA_HEIGHT
        self.wrist = _wrist_camera_cfg()


RELEASE_PARAMS = {
    "cube_cfg": SceneEntityCfg("cube"),
    "goal_cfg": SceneEntityCfg("goal_marker"),
    "robot_cfg": SceneEntityCfg("robot"),
    "table_top_z_m": TABLE_TOP_Z,
    "cube_half_extent_m": CUBE_HALF_EXTENT_M,
    "open_threshold_rad": RELEASED_MIN_GRIPPER_RAD,
    "max_goal_xy_error_m": MAX_GOAL_XY_ERROR_M,
    "max_bottom_error_m": MAX_BOTTOM_ERROR_M,
    "max_linear_speed_m_s": MAX_LINEAR_SPEED_M_S,
    "max_angular_speed_rad_s": MAX_ANGULAR_SPEED_RAD_S,
}


@configclass
class ObservationsCfg(LiftCubeObservationsCfg):
    @configclass
    class SubtaskCfg(ObsGroup):
        pick_cube = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube"),
                "diff_threshold": GRASP_JAW_DISTANCE_M,
                "grasp_threshold": GRASPED_MAX_GRIPPER_RAD,
            },
        )
        place_cube_at_goal = ObsTerm(func=mdp.place_cube_at_goal, params=dict(RELEASE_PARAMS))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()

    def __post_init__(self):
        super().__post_init__()
        self.policy.wrist = ObsTerm(
            func=template_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False},
        )


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):
    success = DoneTerm(
        func=mdp.cube_released_at_goal,
        params={**RELEASE_PARAMS, "hold_time_s": RELEASE_HOLD_TIME_S},
    )


@configclass
class PickCubeRot45EnvCfg(LiftCubeEnvCfg):
    scene: PickCubeRot45SceneCfg = PickCubeRot45SceneCfg(env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = (
        "Pick the red 4 cm cube and place it on the table 45 degrees to the left around the arm base, "
        "then open the gripper."
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        table_top_z = authored_table_top_z()
        if abs(table_top_z - TABLE_TOP_Z) > _TABLE_TOP_TOLERANCE_M:
            raise RuntimeError(f"authored table top {table_top_z} no longer matches TABLE_TOP_Z={TABLE_TOP_Z}")

        # LiftCube registered the authored 30 mm cube; keep its XY and swap the body.
        self.scene.cube = red_cube_40mm_cfg(self.scene.cube.init_state.pos)

        # Added after LiftCube's cube randomization so the goal follows the final cube pose.
        self.events.reset_goal_from_cube = EventTerm(
            func=mdp.reset_goal_from_cube,
            mode="reset",
            params={
                "cube_cfg": SceneEntityCfg("cube"),
                "goal_cfg": SceneEntityCfg("goal_marker"),
                "robot_cfg": SceneEntityCfg("robot"),
                "pan_body_name": PAN_BODY_NAME,
                "table_top_z_m": TABLE_TOP_Z,
                "cube_half_extent_m": CUBE_HALF_EXTENT_M,
                "angle_rad": GOAL_ANGLE_RAD,
            },
        )

        # This single-arm scene does not need IsaacLab's large-population PhysX buffer defaults.
        self.sim.physx.gpu_max_rigid_contact_count = 2**18
        self.sim.physx.gpu_max_rigid_patch_count = 2**13
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**16
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**16
        self.sim.physx.gpu_collision_stack_size = 2**22
        self.sim.physx.gpu_heap_capacity = 2**22
        self.sim.physx.gpu_temp_buffer_capacity = 2**20
