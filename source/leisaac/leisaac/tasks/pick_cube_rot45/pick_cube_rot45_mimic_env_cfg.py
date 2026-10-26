from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .pick_cube_rot45_env_cfg import PickCubeRot45EnvCfg


@configclass
class PickCubeRot45MimicEnvCfg(PickCubeRot45EnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "pick_cube_rot45_leisaac_task_v0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.seed = 42
        self.subtask_configs["so101_follower"] = [
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal="pick_cube",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Pick cube",
                next_subtask_description="Place cube at rotated goal",
            ),
            SubTaskConfig(
                object_ref="goal_marker",
                subtask_term_signal="place_cube_at_goal",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.002,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place cube at goal",
                next_subtask_description="Release cube",
            ),
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0001,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
