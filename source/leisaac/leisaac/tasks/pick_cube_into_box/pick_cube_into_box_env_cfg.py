import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from leisaac.tasks.lift_cube.lift_cube_env_cfg import (
    LiftCubeEnvCfg,
    LiftCubeSceneCfg,
    ObservationsCfg as LiftCubeObservationsCfg,
)
from leisaac.tasks.template import SingleArmTerminationsCfg

from . import mdp

BOX_X = 0.58
BOX_Y = -0.35
TABLE_TOP_Z = 0.0415
BOX_FLOOR_Z = TABLE_TOP_Z + 0.004
BOX_WALL_Z = TABLE_TOP_Z + 0.04


def _box_material():
    return sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.45, 0.85), roughness=0.65)


def _static_box_part(name: str, size: tuple[float, float, float], pos: tuple[float, float, float]):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Box{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=_box_material(),
        ),
    )


@configclass
class PickCubeIntoBoxSceneCfg(LiftCubeSceneCfg):
    box_target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoxTarget",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(BOX_X, BOX_Y, BOX_FLOOR_Z)),
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.12, 0.008),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=_box_material(),
        ),
    )
    box_wall_left = _static_box_part("WallLeft", (0.012, 0.132, 0.07), (BOX_X - 0.066, BOX_Y, BOX_WALL_Z))
    box_wall_right = _static_box_part("WallRight", (0.012, 0.132, 0.07), (BOX_X + 0.066, BOX_Y, BOX_WALL_Z))
    box_wall_front = _static_box_part("WallFront", (0.12, 0.012, 0.07), (BOX_X, BOX_Y - 0.066, BOX_WALL_Z))
    box_wall_back = _static_box_part("WallBack", (0.12, 0.012, 0.07), (BOX_X, BOX_Y + 0.066, BOX_WALL_Z))


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
            },
        )
        place_cube = ObsTerm(
            func=mdp.cube_placed_in_box,
            params={
                "cube_cfg": SceneEntityCfg("cube"),
                "box_cfg": SceneEntityCfg("box_target"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):
    success = DoneTerm(
        func=mdp.cube_released_in_box,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "box_cfg": SceneEntityCfg("box_target"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class PickCubeIntoBoxEnvCfg(LiftCubeEnvCfg):
    scene: PickCubeIntoBoxSceneCfg = PickCubeIntoBoxSceneCfg(env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "Pick the red cube and place it inside the blue box, then open the gripper."

    def __post_init__(self) -> None:
        super().__post_init__()

        # This single-arm scene does not need IsaacLab's large-population PhysX buffer defaults.
        self.sim.physx.gpu_max_rigid_contact_count = 2**18
        self.sim.physx.gpu_max_rigid_patch_count = 2**13
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**16
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**16
        self.sim.physx.gpu_collision_stack_size = 2**22
        self.sim.physx.gpu_heap_capacity = 2**22
        self.sim.physx.gpu_temp_buffer_capacity = 2**20
