"""Pick a 4 cm red cube and drop it into a black box, seen only through the wrist camera.

This is the PickCubeIntoBox layout adapted to the physical setup: a 40 mm red cube,
a black box, a white tabletop, 80 mm TPU FinRay fingers on both jaws and no front
camera. Geometry, mass and friction are simulation values, not measurements of the
physical scene; the FinRay fingers are rigid here.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.utils import find_matching_prim_paths
from isaaclab.utils import configclass
from leisaac.assets.robots.so101_finray import SO101_FOLLOWER_FINRAY_CFG
from leisaac.assets.robots.so101_finray_geometry import FINRAY_CLOSED_GRIPPER_RAD
from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_USD_PATH
from leisaac.tasks.pick_cube_into_box import mdp as box_mdp
from leisaac.tasks.pick_cube_into_box.pick_cube_into_box_env_cfg import (
    BOX_FLOOR_Z,
    BOX_WALL_Z,
    BOX_X,
    BOX_Y,
    ObservationsCfg as PickCubeIntoBoxObservationsCfg,
    PickCubeIntoBoxEnvCfg,
    PickCubeIntoBoxSceneCfg,
    TerminationsCfg as PickCubeIntoBoxTerminationsCfg,
)
from leisaac.tasks.pick_cube_rot45.pick_cube_rot45_env_cfg import (
    SOURCE_TABLE_PRIM,
    TABLE_TOP_Z,
    authored_table_top_z,
    red_cube_40mm_cfg,
    spawn_table_without_source_cube,
)
from leisaac.utils.env_utils import delete_attribute

from . import mdp

# Wrist camera fitted to a physical wrist frame by matching the FinRay finger silhouette
# (pose in the gripper link frame, ROS convention); see docs/provenance/so101_finray_mount_20261026.yaml.
WRIST_CAMERA_POS_M = (-0.00934, 0.04828, -0.07344)
WRIST_CAMERA_ROT_ROS_WXYZ = (0.303987, 0.588847, 0.702262, -0.260153)
WRIST_CAMERA_FOCAL_LENGTH_MM = 39.491
WRIST_CAMERA_APERTURE_MM = 36.83
WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT = 352, 288

# The cube is reset within this XY half range around its nominal spot; with the fitted camera a
# wider area cannot be seen clear of the fingers from any pose that keeps wrist_roll near middle.
CUBE_RESET_HALF_RANGE_M = 0.04
# Joint pose (rad) applied at reset: the fitted wrist camera sees the whole cube reset area
# clear of the closed FinRay fingers (offline search, same provenance file).
OBSERVATION_JOINT_POS = {
    "shoulder_pan": -0.338,
    "shoulder_lift": -0.965,
    "elbow_flex": 0.38,
    "wrist_flex": 1.508,
    "wrist_roll": -0.238,
    "gripper": FINRAY_CLOSED_GRIPPER_RAD,
}

# Gripper split between holding the 40 mm cube with the FinRay fingers and released, and the
# jaw-frame distance for the grasp signal. Policy values above the simulated FinRay hold in
# docs/provenance/so101_finray_mount_20261026.yaml; the stock-finger values would read it as released.
FINRAY_GRASPED_MAX_GRIPPER_RAD = 0.85
FINRAY_RELEASED_MIN_GRIPPER_RAD = 0.85
FINRAY_GRASP_JAW_DISTANCE_M = 0.045

_WHITE_TABLE = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.92), roughness=0.8)
_BLACK_BOX = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.03, 0.03, 0.03), roughness=0.6)


def spawn_white_table_without_source_cube(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the table scene without the authored cube and paint the table white like the real tabletop."""
    prim = spawn_table_without_source_cube(prim_path, cfg, translation, orientation, **kwargs)
    table_child = SOURCE_TABLE_PRIM.rsplit("/", 1)[-1]
    for scene_path in find_matching_prim_paths(prim_path):
        material_path = f"{scene_path}/WhiteTableMaterial"
        _WHITE_TABLE.func(material_path, _WHITE_TABLE)
        sim_utils.bind_visual_material(f"{scene_path}/{table_child}", material_path)
    return prim


def _physical_wrist_camera_cfg() -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=WRIST_CAMERA_POS_M, rot=WRIST_CAMERA_ROT_ROS_WXYZ, convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=WRIST_CAMERA_FOCAL_LENGTH_MM,
            focus_distance=400.0,
            horizontal_aperture=WRIST_CAMERA_APERTURE_MM,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=WRIST_CAMERA_WIDTH,
        height=WRIST_CAMERA_HEIGHT,
        update_period=1 / 30.0,
    )


def _black_box_part(name: str, size: tuple[float, float, float], pos: tuple[float, float, float]):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Box{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=_BLACK_BOX,
        ),
    )


@configclass
class PickCubeIntoBlackBoxSceneCfg(PickCubeIntoBoxSceneCfg):
    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_WITH_CUBE_USD_PATH, func=spawn_white_table_without_source_cube),
    )

    robot: ArticulationCfg = SO101_FOLLOWER_FINRAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Same box geometry as PickCubeIntoBox, painted black.
    box_target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoxTarget",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(BOX_X, BOX_Y, BOX_FLOOR_Z)),
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.12, 0.008),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=_BLACK_BOX,
        ),
    )
    box_wall_left = _black_box_part("WallLeft", (0.012, 0.132, 0.07), (BOX_X - 0.066, BOX_Y, BOX_WALL_Z))
    box_wall_right = _black_box_part("WallRight", (0.012, 0.132, 0.07), (BOX_X + 0.066, BOX_Y, BOX_WALL_Z))
    box_wall_front = _black_box_part("WallFront", (0.12, 0.012, 0.07), (BOX_X, BOX_Y - 0.066, BOX_WALL_Z))
    box_wall_back = _black_box_part("WallBack", (0.12, 0.012, 0.07), (BOX_X, BOX_Y + 0.066, BOX_WALL_Z))

    def __post_init__(self):
        super().__post_init__()
        # The physical follower has only the wrist camera, mounted beside the fingers.
        delete_attribute(self, "front")
        self.wrist = _physical_wrist_camera_cfg()


_GRASP_PARAMS = {
    "robot_cfg": SceneEntityCfg("robot"),
    "ee_frame_cfg": SceneEntityCfg("ee_frame"),
    "object_cfg": SceneEntityCfg("cube"),
    "diff_threshold": FINRAY_GRASP_JAW_DISTANCE_M,
    "grasp_threshold": FINRAY_GRASPED_MAX_GRIPPER_RAD,
}
_BOX_PARAMS = {
    "cube_cfg": SceneEntityCfg("cube"),
    "box_cfg": SceneEntityCfg("box_target"),
    "robot_cfg": SceneEntityCfg("robot"),
}


@configclass
class ObservationsCfg(PickCubeIntoBoxObservationsCfg):
    @configclass
    class SubtaskCfg(ObsGroup):
        pick_cube = ObsTerm(func=box_mdp.object_grasped, params=dict(_GRASP_PARAMS))
        place_cube = ObsTerm(
            func=box_mdp.cube_placed_in_box, params={**_BOX_PARAMS, "open_threshold": FINRAY_RELEASED_MIN_GRIPPER_RAD}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()

    def __post_init__(self):
        super().__post_init__()
        delete_attribute(self.policy, "front")


@configclass
class TerminationsCfg(PickCubeIntoBoxTerminationsCfg):
    success = DoneTerm(
        func=box_mdp.cube_released_in_box,
        params={**_BOX_PARAMS, "open_threshold_rad": FINRAY_RELEASED_MIN_GRIPPER_RAD},
    )


@configclass
class PickCubeIntoBlackBoxEnvCfg(PickCubeIntoBoxEnvCfg):
    scene: PickCubeIntoBlackBoxSceneCfg = PickCubeIntoBlackBoxSceneCfg(env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "Pick the red 4 cm cube and drop it into the black box, then open the gripper."

    def __post_init__(self) -> None:
        super().__post_init__()

        table_top_z = authored_table_top_z()
        if abs(table_top_z - TABLE_TOP_Z) > 1e-6:
            raise RuntimeError(f"authored table top {table_top_z} no longer matches TABLE_TOP_Z={TABLE_TOP_Z}")
        self.scene.cube = red_cube_40mm_cfg(self.scene.cube.init_state.pos)
        # Added after reset_all, which restores the zero default pose used as the action offset.
        self.events.reset_robot_to_observation_pose = EventTerm(
            func=mdp.reset_joints_to_pose,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "joint_pos": dict(OBSERVATION_JOINT_POS)},
        )

        # LiftCube randomizes the front camera, which this task removes, and the cube over a wider area.
        for name, term in list(self.events.__dict__.items()):
            params = getattr(term, "params", None) or {}
            asset = getattr(params.get("asset_cfg"), "name", None)
            if asset == "front":
                delattr(self.events, name)
            elif asset == "cube":
                params["pose_range"]["x"] = (-CUBE_RESET_HALF_RANGE_M, CUBE_RESET_HALF_RANGE_M)
                params["pose_range"]["y"] = (-CUBE_RESET_HALF_RANGE_M, CUBE_RESET_HALF_RANGE_M)
