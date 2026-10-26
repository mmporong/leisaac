import gymnasium as gym


gym.register(
    id="LeIsaac-SO101-PickCubeRot45-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.pick_cube_rot45_env_cfg:PickCubeRot45EnvCfg"},
)

gym.register(
    id="LeIsaac-SO101-PickCubeRot45-Mimic-v0",
    entry_point="leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.pick_cube_rot45_mimic_env_cfg:PickCubeRot45MimicEnvCfg"},
)
