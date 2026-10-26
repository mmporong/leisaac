import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-PickCubeIntoBlackBox-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.pick_cube_into_black_box_env_cfg:PickCubeIntoBlackBoxEnvCfg"},
)

gym.register(
    id="LeIsaac-SO101-PickCubeIntoBlackBox-Mimic-v0",
    entry_point="leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_cube_into_black_box_mimic_env_cfg:PickCubeIntoBlackBoxMimicEnvCfg"
    },
)
