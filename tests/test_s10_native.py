"""S10 regression checks; run with pytest in the repository's Isaac Lab environment."""

import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


@pytest.fixture(scope="module")
def runtime():
    # Isaac Sim must start before importing task/asset modules.
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, fast_shutdown=False)
    import gymnasium as gym
    import torch

    from isaaclab.managers import SceneEntityCfg
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import (
        DeeproboticsM20RoughEnvCfg,
    )
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_s10.agents.rsl_rl_ppo_cfg import (
        DeeproboticsS10RoughPPORunnerCfg,
    )
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_s10.rewards import (
        joint_pos_penalty_turn_side,
    )
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_s10.rough_env_cfg import (
        DeeproboticsS10RoughEnvCfg,
    )
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.rewards import joint_pos_penalty_except_turn_side_cmd

    yield SimpleNamespace(
        gym=gym, torch=torch, entity=SceneEntityCfg,
        s10=DeeproboticsS10RoughEnvCfg, m20=DeeproboticsM20RoughEnvCfg,
        runner=DeeproboticsS10RoughPPORunnerCfg,
        s10_reward=joint_pos_penalty_turn_side,
        shared_reward=joint_pos_penalty_except_turn_side_cmd,
    )
    launcher.app.close()


def test_registration_and_native_runner(runtime):
    for task in (
        "Rough-Deeprobotics-S10-v0", "Rough-Deeprobotics-M20-v0",
        "Rough-Deeprobotics-Lite3-v0", "Amp-Flat-Deeprobotics-DR02-v0",
    ):
        assert runtime.gym.spec(task) is not None
    assert runtime.gym.spec("Rough-Deeprobotics-S10-v0").entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    cfg = runtime.runner()
    cfg.validate()
    assert cfg.class_name == "OnPolicyRunner"
    assert cfg.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert cfg.actor["distribution_cfg"]["std_type"] == "log"
    assert cfg.actor["hidden_dims"] == cfg.critic["hidden_dims"] == [512, 256, 128]


def test_model_actions_and_reward_parameters(runtime):
    cfg = runtime.s10()
    cfg.validate()
    urdf_path = Path(cfg.scene.robot.spawn.asset_path)
    model = ET.parse(urdf_path).getroot()
    joints = {j.attrib["name"] for j in model.findall("joint") if j.attrib["type"] != "fixed"}
    assert len(joints) == 16
    assert set(cfg.joint_names) == joints
    assert cfg.actions.joint_pos.joint_names == cfg.leg_joint_names
    assert cfg.actions.joint_vel.joint_names == cfg.wheel_joint_names
    for mesh in model.findall(".//mesh"):
        assert (urdf_path.parent / mesh.attrib["filename"]).is_file()
    assert cfg.rewards.track_lin_vel_xy_exp.params["std"] == pytest.approx(math.sqrt(0.15))
    assert cfg.rewards.track_ang_vel_z_exp.params["std"] == pytest.approx(0.5)
    assert cfg.rewards.hipy_joint_pos_penalty.func is runtime.s10_reward
    assert cfg.rewards.knee_joint_pos_penalty.func is runtime.s10_reward


def test_s10_configuration_does_not_mutate_m20(runtime):
    before = runtime.m20()
    s10 = runtime.s10()
    after = runtime.m20()
    for cfg in (before, after):
        assert cfg.rewards.hipy_joint_pos_penalty.func is runtime.shared_reward
        assert cfg.rewards.hipy_joint_pos_penalty.weight == -1.5
        assert cfg.rewards.track_lin_vel_xy_exp.params["std"] == pytest.approx(math.sqrt(0.5))
        assert cfg.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range == (0.01, 0.16)
        assert cfg.scene.robot.spawn.usd_path.endswith("/M20/M20_usd/M20.usd")
    s10.rewards.hipy_joint_pos_penalty.params["high_terrain_penalty_scale"] = 0.9
    assert runtime.s10().rewards.hipy_joint_pos_penalty.params["high_terrain_penalty_scale"] == 0.1


@pytest.mark.parametrize("high_terrain", [False, True])
def test_s10_turn_side_penalty_and_invalid_rays(runtime, high_terrain):
    torch = runtime.torch
    # Standing, straight, pure turn, sideways, and mixed command cases.
    commands = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 1.], [0., 1., 0.], [1., 1., 1.]])
    robot = SimpleNamespace(data=SimpleNamespace(
        joint_pos=torch.tensor([[3., 4.]]).repeat(5, 1),
        default_joint_pos=torch.zeros(5, 2), root_lin_vel_b=torch.zeros(5, 3),
    ))
    origins = torch.full((5, 3), 7.)
    hits = torch.zeros(5, 3, 3)
    hits[:, :, 2] = torch.tensor([7.2 if high_terrain else 7., float("nan"), float("inf")])
    hits[4, :, 2] = float("inf")  # No valid ray: retain the ordinary penalty.

    class Scene(dict):
        env_origins = origins

    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_command=lambda _: commands),
        scene=Scene(robot=robot, height=SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits))),
    )
    kwargs = dict(
        command_name="base_velocity", asset_cfg=runtime.entity("robot", joint_ids=[0, 1]),
        stand_still_scale=2., velocity_threshold=0.5, command_threshold=0.1,
        sensor_cfg=runtime.entity("height"), terrain_height_threshold=0.06, high_terrain_penalty_scale=0.1,
    )
    expected = torch.tensor([10., 5., 1., 1., 5.])
    shared_expected = torch.tensor([10., 5., 0., 0., 5.])
    if high_terrain:
        expected[:4] *= 0.1
        shared_expected[:4] *= 0.1
    torch.testing.assert_close(runtime.s10_reward(env, **kwargs), expected)
    torch.testing.assert_close(runtime.shared_reward(env, **kwargs), shared_expected)
