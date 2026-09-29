"""Tests for refreshing the camera observations after a reset. They do not need Isaac Sim."""

from types import SimpleNamespace

import pytest
from leisaac.utils.env_utils import (
    RESET_RENDER_STEPS,
    refresh_camera_obs_after_reset,
    use_isaaclab_reset_rerender,
)


class FakeSim:
    def __init__(self, log):
        self.log = log

    def render(self):
        self.log.append("render")

    def step(self, render=True):
        raise AssertionError("refreshing the cameras must not step the physics")


class FakeCamera:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def reset(self, env_ids=None):
        self.log.append((self.name, "reset", env_ids))

    def update(self, dt, force_recompute=False):
        self.log.append((self.name, "update", dt, force_recompute))


class FakeObservationManager:
    def __init__(self, log):
        self.log = log

    def compute(self):
        self.log.append("compute")
        return {"policy": "after reset"}


def make_env(log):
    return SimpleNamespace(
        sim=FakeSim(log), observation_manager=FakeObservationManager(log), obs_buf={"policy": "before reset"}
    )


def test_refresh_renders_before_reading_the_cameras():
    log = []
    env = make_env(log)
    cameras = [FakeCamera("front", log), FakeCamera("wrist", log)]
    env_ids = [0]

    refresh_camera_obs_after_reset(env, env_ids, cameras)

    assert log == ["render"] * RESET_RENDER_STEPS + [
        ("front", "reset", env_ids),
        ("front", "update", 0.0, True),
        ("wrist", "reset", env_ids),
        ("wrist", "update", 0.0, True),
        "compute",
    ]
    assert env.obs_buf == {"policy": "after reset"}


def test_refresh_without_cameras_keeps_the_observations():
    log = []
    env = make_env(log)

    refresh_camera_obs_after_reset(env, [0], cameras=[])

    assert log == []
    assert env.obs_buf == {"policy": "before reset"}


def test_isaaclab_without_num_rerenders_on_reset_needs_a_manual_refresh():
    env_cfg = SimpleNamespace(rerender_on_reset=False)

    assert use_isaaclab_reset_rerender(env_cfg) is False
    assert not hasattr(env_cfg, "num_rerenders_on_reset")


@pytest.mark.parametrize(
    "configured, expected", [(0, RESET_RENDER_STEPS), (RESET_RENDER_STEPS + 2, RESET_RENDER_STEPS + 2)]
)
def test_isaaclab_with_num_rerenders_on_reset_renders_inside_reset(configured, expected):
    env_cfg = SimpleNamespace(num_rerenders_on_reset=configured)

    assert use_isaaclab_reset_rerender(env_cfg) is True
    assert env_cfg.num_rerenders_on_reset == expected
