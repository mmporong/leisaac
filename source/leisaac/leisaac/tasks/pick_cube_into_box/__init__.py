import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-PickCubeIntoBox-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.pick_cube_into_box_env_cfg:PickCubeIntoBoxEnvCfg"},
)

gym.register(
    id="LeIsaac-SO101-PickCubeIntoBox-Mimic-v0",
    entry_point="leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.pick_cube_into_box_mimic_env_cfg:PickCubeIntoBoxMimicEnvCfg"},
)
