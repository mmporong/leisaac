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
from isaaclab.sim.utils import find_matching_prim_paths
from isaaclab.utils import configclass
from leisaac.assets.robots.so101_finray import SO101_FOLLOWER_FINRAY_CFG
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

# Provisional joint pose (rad) applied at reset, chosen for LeIsaac's stock wrist camera with
# wrist_roll near its calibrated middle. The physical camera sits elsewhere, so re-derive it
# after the real camera mount is calibrated (docs/provenance/so101_finray_mount_20261026.yaml).
OBSERVATION_JOINT_POS = {
    "shoulder_pan": -0.524,
    "shoulder_lift": -0.791,
    "elbow_flex": 0.409,
    "wrist_flex": 1.508,
    "wrist_roll": 0.083,
    "gripper": 0.0,
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
        # The physical follower has only the wrist camera.
        delete_attribute(self, "front")


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

        # LiftCube randomizes the front camera, which this task removes.
        for name, term in list(self.events.__dict__.items()):
            params = getattr(term, "params", None) or {}
            if getattr(params.get("asset_cfg"), "name", None) == "front":
                delattr(self.events, name)
