# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Detailed validation script for FrozenLocomotionPolicy wrapper.

This script performs more thorough validation including:
- Velocity tracking verification
- Action distribution analysis
- Consistency checks across different commands
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Validate frozen policy wrapper.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
# append RSL-RL cli arguments (includes --checkpoint)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401

from rl_training.utils.frozen_policy import FrozenLocomotionPolicy, freeze_policy


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Validate frozen policy wrapper."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid
    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for testing
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels = None

    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    inference_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Freeze the policy
    freeze_policy(ppo_runner)

    # Create FrozenLocomotionPolicy wrapper
    low_level_policy = FrozenLocomotionPolicy(inference_policy, env.unwrapped)

    print("\n" + "="*80)
    print("DETAILED VALIDATION")
    print("="*80)

    # Reset environment
    obs, _ = env.reset()
    obs = env.get_observations()
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    # Test 1: Velocity tracking - does the robot actually move in the commanded direction?
    print("\n[Test 1] Velocity Tracking Verification")
    print("-" * 80)
    
    test_cmd = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(num_envs, -1)  # Forward
    
    # Get initial robot position and orientation
    robot = env.unwrapped.scene["robot"]
    initial_pos = robot.data.root_pos_w.clone()
    initial_yaw = torch.atan2(
        2 * (robot.data.root_quat_w[:, 0] * robot.data.root_quat_w[:, 3] + 
             robot.data.root_quat_w[:, 1] * robot.data.root_quat_w[:, 2]),
        1 - 2 * (robot.data.root_quat_w[:, 2]**2 + robot.data.root_quat_w[:, 3]**2)
    )
    
    # Step for multiple timesteps
    num_steps = 100
    for step in range(num_steps):
        actions = low_level_policy(test_cmd)
        step_result = env.step(actions)
        if len(step_result) == 4:
            obs, rewards, terminated, truncated = step_result
        else:
            obs, rewards, terminated, truncated, info = step_result
        obs = env.get_observations()
        
        # Handle termination/truncation
        if isinstance(terminated, torch.Tensor) and isinstance(truncated, torch.Tensor):
            if terminated.any() or truncated.any():
                reset_envs = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                if len(reset_envs) > 0:
                    obs, _ = env.reset(options={"env_ids": reset_envs})
                    obs = env.get_observations()
    
    # Get final position
    final_pos = robot.data.root_pos_w.clone()
    final_yaw = torch.atan2(
        2 * (robot.data.root_quat_w[:, 0] * robot.data.root_quat_w[:, 3] + 
             robot.data.root_quat_w[:, 1] * robot.data.root_quat_w[:, 2]),
        1 - 2 * (robot.data.root_quat_w[:, 2]**2 + robot.data.root_quat_w[:, 3]**2)
    )
    
    # Calculate displacement in robot frame
    displacement = final_pos - initial_pos
    displacement_2d = displacement[:, :2]  # x, y displacement
    
    # Rotate to robot's initial frame
    cos_yaw = torch.cos(initial_yaw)
    sin_yaw = torch.sin(initial_yaw)
    displacement_x = displacement_2d[:, 0] * cos_yaw + displacement_2d[:, 1] * sin_yaw
    displacement_y = -displacement_2d[:, 0] * sin_yaw + displacement_2d[:, 1] * cos_yaw
    
    avg_displacement_x = displacement_x.mean().item()
    avg_displacement_y = displacement_y.mean().item()
    avg_yaw_change = (final_yaw - initial_yaw).mean().item()
    
    print(f"  Command: [vx=1.0, vy=0.0, vyaw=0.0]")
    print(f"  Steps: {num_steps}")
    print(f"  Average X displacement (forward): {avg_displacement_x:.3f} m")
    print(f"  Average Y displacement (lateral): {avg_displacement_y:.3f} m")
    print(f"  Average yaw change: {avg_yaw_change:.3f} rad ({np.degrees(avg_yaw_change):.2f} deg)")
    
    # Expected: positive X displacement (forward), small Y, small yaw change
    if avg_displacement_x > 0.1:
        print(f"  ✅ Robot moved forward as commanded")
    else:
        print(f"  ⚠️  Robot forward movement is limited")
    
    if abs(avg_displacement_y) < 0.5:
        print(f"  ✅ Lateral drift is minimal")
    else:
        print(f"  ⚠️  Significant lateral drift detected")
    
    if abs(avg_yaw_change) < 0.2:
        print(f"  ✅ Yaw stability maintained")
    else:
        print(f"  ⚠️  Yaw drift detected")

    # Test 2: Action distribution analysis
    print("\n[Test 2] Action Distribution Analysis")
    print("-" * 80)
    
    test_commands = [
        ("Forward", torch.tensor([[1.0, 0.0, 0.0]], device=device)),
        ("Backward", torch.tensor([[-0.5, 0.0, 0.0]], device=device)),
        ("Rotate", torch.tensor([[0.0, 0.0, 0.5]], device=device)),
        ("Strafe", torch.tensor([[0.0, 0.5, 0.0]], device=device)),
    ]
    
    for cmd_name, cmd in test_commands:
        cmd_expanded = cmd.expand(num_envs, -1)
        actions = low_level_policy(cmd_expanded)
        
        print(f"\n  {cmd_name} command {cmd[0].tolist()}:")
        print(f"    Shape: {actions.shape}")
        print(f"    Range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
        print(f"    Mean: {actions.mean().item():.3f}, Std: {actions.std().item():.3f}")
        print(f"    Per-joint mean: {actions.mean(dim=0).abs().mean().item():.3f}")

    # Test 3: Consistency - same command should give similar actions (within noise)
    print("\n[Test 3] Action Consistency Check")
    print("-" * 80)
    
    test_cmd = torch.tensor([[0.5, 0.0, 0.0]], device=device).expand(num_envs, -1)
    
    actions_list = []
    for _ in range(5):
        actions = low_level_policy(test_cmd)
        actions_list.append(actions)
    
    actions_tensor = torch.stack(actions_list)  # [5, num_envs, num_joints]
    action_std = actions_tensor.std(dim=0).mean().item()  # Average std across envs and joints
    
    print(f"  Command: [0.5, 0.0, 0.0] (repeated 5 times)")
    print(f"  Average action std across repeats: {action_std:.4f}")
    
    if action_std < 0.01:
        print(f"  ✅ Actions are highly consistent (deterministic)")
    elif action_std < 0.1:
        print(f"  ✅ Actions are reasonably consistent")
    else:
        print(f"  ⚠️  Actions show significant variation (might be due to observation changes)")

    # Test 4: Command scaling - different command magnitudes
    print("\n[Test 4] Command Scaling Analysis")
    print("-" * 80)
    
    cmd_magnitudes = [0.25, 0.5, 1.0, 1.5]
    test_cmd_base = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    
    for mag in cmd_magnitudes:
        cmd = test_cmd_base * mag
        cmd_expanded = cmd.expand(num_envs, -1)
        actions = low_level_policy(cmd_expanded)
        action_magnitude = actions.abs().mean().item()
        
        print(f"  Command magnitude: {mag:.2f} -> Action magnitude: {action_magnitude:.3f}")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

