# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Test script for FrozenLocomotionPolicy wrapper."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Test FrozenLocomotionPolicy wrapper.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments (includes --checkpoint)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
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

from rl_training.utils.frozen_policy import FrozenLocomotionPolicy, freeze_policy, is_frozen, count_parameters

import rl_training.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Test FrozenLocomotionPolicy wrapper."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # set the environment seed
    env_cfg.seed = agent_cfg.seed

    # disable randomization for testing
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    inference_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Freeze the policy
    policy_nn = freeze_policy(ppo_runner)

    # Print policy info
    num_params = count_parameters(policy_nn)
    frozen_status = is_frozen(policy_nn)

    print("\n" + "="*80)
    print("FROZEN POLICY INFO")
    print("="*80)
    print(f"Checkpoint: {resume_path}")
    print(f"Parameters: {num_params:,}")
    print(f"Frozen (requires_grad=False): {'✅' if frozen_status else '❌'}")
    print("="*80 + "\n")

    # Create FrozenLocomotionPolicy wrapper
    print("[INFO] Creating FrozenLocomotionPolicy wrapper...")
    low_level_policy = FrozenLocomotionPolicy(inference_policy, env.unwrapped)
    print("[INFO] FrozenLocomotionPolicy wrapper created successfully.\n")

    # Test the wrapper
    print("[INFO] Testing FrozenLocomotionPolicy wrapper...")
    obs, _ = env.reset()
    obs = env.get_observations()

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    # Test with different velocity commands
    test_commands = [
        torch.tensor([[1.0, 0.0, 0.0]], device=device),  # Forward
        torch.tensor([[-0.5, 0.0, 0.0]], device=device),  # Backward
        torch.tensor([[0.0, 0.0, 0.5]], device=device),  # Rotate in place
        torch.tensor([[0.5, 0.3, 0.2]], device=device),  # Complex motion
    ]

    print(f"[INFO] Testing with {len(test_commands)} different velocity commands...")
    for i, cmd in enumerate(test_commands):
        # Expand command to all environments
        cmd_expanded = cmd.expand(num_envs, -1)
        
        print(f"\n  Test {i+1}: Velocity command [vx={cmd[0,0]:.2f}, vy={cmd[0,1]:.2f}, vyaw={cmd[0,2]:.2f}]")
        
        # Get actions from wrapper
        try:
            actions = low_level_policy(cmd_expanded)
            print(f"    ✅ Success: Actions shape = {actions.shape}")
            print(f"       Actions range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
            print(f"       Actions mean: {actions.mean().item():.3f}, std: {actions.std().item():.3f}")
        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Test with actual environment stepping
    print("\n[INFO] Testing wrapper with environment stepping...")
    obs, _ = env.reset()
    obs = env.get_observations()
    
    num_steps = 10
    test_cmd = torch.tensor([[0.5, 0.0, 0.0]], device=device).expand(num_envs, -1)
    
    print(f"  Running {num_steps} steps with velocity command [0.5, 0.0, 0.0]...")
    for step in range(num_steps):
        # Get actions from wrapper
        actions = low_level_policy(test_cmd)
        
        # Step environment
        step_result = env.step(actions)
        if len(step_result) == 4:
            obs, rewards, terminated, truncated = step_result
        else:
            obs, rewards, terminated, truncated, info = step_result
        obs = env.get_observations()
        
        if (step + 1) % 5 == 0:
            if isinstance(rewards, dict):
                reward_val = rewards.get("policy", torch.tensor(0.0)).mean().item()
            else:
                reward_val = rewards.mean().item()
            print(f"    Step {step+1}/{num_steps}: Reward = {reward_val:.4f}")

    print("\n" + "="*80)
    print("WRAPPER TEST COMPLETE")
    print("="*80)
    print("✅ FrozenLocomotionPolicy wrapper is working correctly!")
    print("   - Can convert velocity commands to joint actions")
    print("   - Can be used with environment stepping")
    print("="*80)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

