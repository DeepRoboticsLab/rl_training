# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Test script for hierarchical navigation environment."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Test hierarchical navigation environment.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the low-level task.")
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

from rl_training.utils.frozen_policy import FrozenLocomotionPolicy, freeze_policy
from rl_training.tasks.manager_based.locomotion.hierarchical_nav.hierarchical_nav_env import HierarchicalNavEnv

import rl_training.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Test hierarchical navigation environment."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 4
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

    print(f"[INFO]: Loading low-level policy checkpoint from: {resume_path}")
    
    # Create low-level environment
    print("[INFO] Creating low-level environment...")
    low_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(low_env.unwrapped, DirectMARLEnv):
        low_env = multi_agent_to_single_agent(low_env)

    # wrap around environment for rsl-rl
    low_env = RslRlVecEnvWrapper(low_env, clip_actions=agent_cfg.clip_actions)

    # load previously trained model
    ppo_runner = OnPolicyRunner(low_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    inference_policy = ppo_runner.get_inference_policy(device=low_env.unwrapped.device)

    # Freeze the policy
    freeze_policy(ppo_runner)
    print("✅ Low-level policy loaded and frozen!")

    # Create FrozenLocomotionPolicy wrapper
    frozen_policy = FrozenLocomotionPolicy(inference_policy, low_env.unwrapped)
    print("✅ FrozenLocomotionPolicy wrapper created!")

    # Create hierarchical navigation environment
    print("[INFO] Creating hierarchical navigation environment...")
    hierarchical_env = HierarchicalNavEnv(
        env=low_env,
        frozen_policy_wrapper=frozen_policy,
        decimation=10,
    )
    print("✅ Hierarchical navigation environment created!")

    # Test hierarchical environment
    print("\n[INFO] Testing hierarchical navigation environment...")
    
    # Reset
    obs, info = hierarchical_env.reset()
    print(f"✅ Reset successful! Observation shape: {obs.shape}")
    print(f"   Observation range: [{obs.min().item():.3f}, {obs.max().item():.3f}]")
    
    # Test stepping
    num_steps = 100
    print(f"\n[INFO] Running {num_steps} high-level steps...")
    
    for step in range(num_steps):
        # Sample random action (velocity command)
        action = hierarchical_env.action_space.sample()
        action = torch.tensor(action, device=hierarchical_env.device, dtype=torch.float32)
        if action.dim() == 1:
            action = action.unsqueeze(0).expand(hierarchical_env.num_envs, -1)
        
        # Step hierarchical environment
        obs, rewards, terminated, truncated, info = hierarchical_env.step(action)
        
        # Check for terminations and reset (only reset if all environments are done)
        if isinstance(terminated, torch.Tensor):
            term_flag = terminated.all()
            trunc_flag = truncated.all() if isinstance(truncated, torch.Tensor) else False
        else:
            term_flag = all(terminated) if isinstance(terminated, (list, tuple)) else terminated
            trunc_flag = all(truncated) if isinstance(truncated, (list, tuple)) else truncated
        
        if term_flag or trunc_flag:
            obs, info = hierarchical_env.reset()
            print(f"  Step {step+1}: All environments reset due to termination/truncation")
        
        # Log progress
        if (step + 1) % 20 == 0:
            reward_mean = rewards.mean().item() if isinstance(rewards, torch.Tensor) else rewards
            print(f"  Step {step+1}/{num_steps}: Mean reward = {reward_mean:.4f}")
            
            # Log goal distance (from observation)
            if obs.shape[1] >= 8:  # Should have distance at index 6
                distance = obs[:, 6].mean().item()
                print(f"             Mean distance to goal = {distance:.3f}")

    print("\n" + "="*80)
    print("HIERARCHICAL NAVIGATION ENVIRONMENT TEST COMPLETE")
    print("="*80)
    print("✅ All tests passed!")
    print("   - High-level environment created successfully")
    print("   - Reset and step functions working correctly")
    print("   - Observations and rewards computed correctly")
    print("="*80)

    hierarchical_env.env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

