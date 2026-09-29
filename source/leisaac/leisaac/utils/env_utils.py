import torch


def dynamic_reset_gripper_effort_limit_sim(env, teleop_device):
    need_to_set = []
    if "bi-so101leader" in teleop_device:
        need_to_set = [env.scene.articulations["left_arm"], env.scene.articulations["right_arm"]]
    elif "so101leader" in teleop_device or teleop_device in ["keyboard", "gamepad"]:
        need_to_set = [env.scene["robot"]]
    for arm in need_to_set:
        write_gripper_effort_limit_sim(env, arm)
    return


def write_gripper_effort_limit_sim(env, env_arm):
    gripper_pos = env_arm.data.body_link_pos_w[:, -1]  # [num_envs, 3]
    num_envs = gripper_pos.shape[0]

    object_positions = []
    object_masses = []
    object_names = []

    for name, obj in env.scene._rigid_objects.items():
        pos = obj.data.body_link_pos_w[:, 0]  # [num_envs, 3]
        object_positions.append(pos)
        object_masses.append(obj.data.default_mass)
        object_names.append(name)

    if not object_positions:
        return

    object_positions = torch.stack(object_positions)  # [num_objects, num_envs, 3]
    object_masses = torch.stack(object_masses)  # [num_objects, num_envs, 1]

    distances = torch.sqrt(torch.sum((object_positions - gripper_pos.unsqueeze(0)) ** 2, dim=2))

    min_distances, min_indices = torch.min(distances, dim=0)  # [num_envs]

    target_masses = object_masses[min_indices.cpu(), 0, 0]  # [num_envs]

    target_effort_limits = (target_masses / 0.15).to(env_arm._data.joint_effort_limits.device)

    current_effort_limit_sim = env_arm._data.joint_effort_limits[:, -1]  # [num_envs]
    need_update = torch.abs(target_effort_limits - current_effort_limit_sim) > 0.1

    if torch.any(need_update):
        new_limits = current_effort_limit_sim.clone()
        new_limits[need_update] = target_effort_limits[need_update]

        env_arm.write_joint_effort_limit_to_sim(limits=new_limits, joint_ids=[5 for _ in range(num_envs)])


def get_task_type(task: str, task_type: str | None = None) -> str:
    """
    Make sure the task type is in the supported teleop devices.
    """
    if task_type is not None:
        return task_type
    if "BiArm" in task:
        return "bi-so101leader"
    elif "LeKiwi" in task:
        return "lekiwi-leader"
    else:
        return "so101leader"


def delete_attribute(obj, attr_name):
    if hasattr(obj, attr_name):
        delattr(obj, attr_name)


# Render passes needed after a reset before the RTX cameras show the reset scene. With fewer passes
# the camera images still match the frame rendered before the reset (Isaac Sim 5.1, Isaac Lab 2.3.0).
RESET_RENDER_STEPS = 3


def use_isaaclab_reset_rerender(env_cfg, render_steps: int = RESET_RENDER_STEPS) -> bool:
    """Let Isaac Lab re-render inside ``reset()`` if it supports ``num_rerenders_on_reset``.

    Returns True if Isaac Lab handles it, or False if the caller has to call
    :func:`refresh_camera_obs_after_reset` after each reset (Isaac Lab 2.3.0 and older).
    """
    if not hasattr(env_cfg, "num_rerenders_on_reset"):
        return False
    env_cfg.num_rerenders_on_reset = max(env_cfg.num_rerenders_on_reset, render_steps)
    return True


def refresh_camera_obs_after_reset(env, env_ids, cameras, render_steps: int = RESET_RENDER_STEPS):
    """Render the reset scene again and recompute ``env.obs_buf`` with up-to-date camera images.

    Right after a reset the cameras still return the image rendered before the reset. This renders
    without stepping physics, marks the cameras of the reset environments as outdated so that
    their buffers are read again, and recomputes the observations.
    """
    if not cameras:
        return
    for _ in range(render_steps):
        env.sim.render()
    for camera in cameras:
        camera.reset(env_ids)
        camera.update(0.0, force_recompute=True)
    env.obs_buf = env.observation_manager.compute()
