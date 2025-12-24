# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Training script for hierarchical navigation environment."""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train hierarchical navigation policy.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the low-level task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
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
from isaaclab.utils.io import dump_yaml

from rl_training.utils.frozen_policy import FrozenLocomotionPolicy, freeze_policy
from rl_training.tasks.manager_based.locomotion.hierarchical_nav.hierarchical_nav_env import HierarchicalNavEnv

import rl_training.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train hierarchical navigation policy."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if hasattr(args_cli, 'distributed') and args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", "hierarchical_nav")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Load low-level policy checkpoint (required for hierarchical training)
    if not args_cli.checkpoint:
        raise ValueError("--checkpoint argument is required for hierarchical navigation training. Specify the path to the low-level policy checkpoint.")
    low_level_checkpoint = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO]: Loading low-level policy checkpoint from: {low_level_checkpoint}")

    # Create low-level environment
    print("[INFO] Creating low-level environment...")
    render_mode = "rgb_array" if (hasattr(args_cli, 'video') and args_cli.video) else None
    low_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(low_env.unwrapped, DirectMARLEnv):
        low_env = multi_agent_to_single_agent(low_env)

    # wrap around environment for rsl-rl
    low_env = RslRlVecEnvWrapper(low_env, clip_actions=agent_cfg.clip_actions)

    # load low-level policy
    ppo_runner = OnPolicyRunner(low_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(low_level_checkpoint)

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
    print(f"   Action space: {hierarchical_env.action_space}")
    print(f"   Observation space: {hierarchical_env.observation_space}")

    # Update agent config for hierarchical environment
    # Action space: 3D (vx, vy, vyaw)
    # Observation space: 8D (robot_pos_2d, robot_yaw, goal_pos_2d, distance, direction)
    agent_cfg.experiment_name = "hierarchical_nav"
    # Adjust network sizes if needed
    # The default config should work, but we can adjust based on observation/action spaces
    
    # Wrap hierarchical environment for RSL-RL
    # We need to create a wrapper that provides the VecEnv interface for RSL-RL
    from rsl_rl.env import VecEnv
    from tensordict import TensorDict
    import gymnasium.vector.utils
    
    class HierarchicalVecEnvWrapper(VecEnv):
        """Wrapper to make HierarchicalNavEnv compatible with RSL-RL VecEnv interface."""
        
        def __init__(self, hierarchical_env: HierarchicalNavEnv, clip_actions: float | None = None):
            self.env = hierarchical_env
            self.clip_actions = clip_actions
            self.num_envs = hierarchical_env.num_envs
            self.device = hierarchical_env.device
            # Use a reasonable episode length (30 seconds at high-level, with decimation=10)
            # High-level step dt = 10 * low_level_dt (0.005), so episode_length should be adjusted
            # For 30 seconds: 30.0 / (10 * 0.005) = 600 high-level steps
            self.max_episode_length = 600
            
            # Store action/observation space info for RSL-RL
            self.num_actions = gym.spaces.flatdim(hierarchical_env.action_space)
            
            # Store single action space for compatibility
            self.single_action_space = hierarchical_env.action_space
            self.single_observation_space = hierarchical_env.observation_space
            
            # Modify action space if clipping is needed
            if self.clip_actions is not None:
                self.single_action_space = gym.spaces.Box(
                    low=-self.clip_actions,
                    high=self.clip_actions,
                    shape=(self.num_actions,),
                    dtype=hierarchical_env.action_space.dtype,
                )
                self.env.action_space = gym.vector.utils.batch_space(
                    self.single_action_space, self.num_envs
                )
            
            # Episode length buffer (needed for RSL-RL)
            self._episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            
            # Reset at the start since RSL-RL runner does not call reset
            self.reset()
        
        @property
        def observation_space(self) -> gym.Space:
            """Returns the observation space."""
            return self.env.observation_space
        
        @property
        def action_space(self) -> gym.Space:
            """Returns the action space."""
            return self.env.action_space
        
        @property
        def episode_length_buf(self) -> torch.Tensor:
            """The episode length buffer."""
            return self._episode_length_buf
        
        @episode_length_buf.setter
        def episode_length_buf(self, value: torch.Tensor):
            """Set the episode length buffer."""
            self._episode_length_buf = value
        
        def reset(self):
            """Reset the environment."""
            obs, info = self.env.reset()
            return TensorDict({"policy": obs}, batch_size=[self.num_envs]), info
        
        def step(self, actions: torch.Tensor):
            """Step the environment."""
            # Clip actions if needed
            if self.clip_actions is not None:
                actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
            
            # Step hierarchical environment
            obs, rewards, terminated, truncated, info = self.env.step(actions)
            
            # Compute dones for compatibility with RSL-RL
            dones = (terminated | truncated).to(dtype=torch.long)
            
            # Ensure info is a dict and contains high-level metrics
            if not isinstance(info, dict):
                info = {}
            
            # Make sure high-level metrics are in info (they should already be added in HierarchicalNavEnv.step())
            # But ensure they're present in case they weren't added
            if "Episode_Termination/goal_reached" not in info:
                # Calculate metrics from current state if not already in info
                low_env = self.env.unwrapped
                import rl_training.tasks.manager_based.locomotion.hierarchical_nav.mdp as mdp
                robot_pos_2d = mdp.robot_position_2d(low_env)
                distance_vec = self.env._goal_positions - robot_pos_2d
                distance = torch.norm(distance_vec, dim=1)
                goal_reached = (distance < 0.5).float().mean().item()
                
                std = 0.5
                goal_reward = torch.exp(-distance / std**2).mean().item()
                
                info["Episode_Termination/goal_reached"] = goal_reached
                info["Episode_Reward/goal_reaching"] = goal_reward
                info["hierarchical/distance_to_goal"] = distance.mean().item()
            
            # Return in RSL-RL format: (obs_dict, rew, dones, extras)
            # RSL-RL may read Episode_Reward/ and Episode_Termination/ from extras
            return TensorDict({"policy": obs}, batch_size=[self.num_envs]), rewards, dones, info
        
        def get_observations(self) -> TensorDict:
            """Returns the current observations of the environment."""
            obs = self.env._get_high_level_observations()
            return TensorDict({"policy": obs}, batch_size=[self.num_envs])
        
        def close(self):
            """Close the environment."""
            return self.env.env.close()
        
        @property
        def unwrapped(self):
            # Return hierarchical env directly
            # Note: RSL-RL will still access low-level env's reward_manager for logging
            # High-level metrics are added to info dict in step() method
            return self.env
    
    # Wrap hierarchical environment
    vec_env = HierarchicalVecEnvWrapper(hierarchical_env, clip_actions=agent_cfg.clip_actions)
    
    # Update observation/action space dimensions in agent config
    # This is needed for RSL-RL to initialize the policy networks correctly
    # The runner will infer these from the environment, but we can set them explicitly
    # Actually, RSL-RL will get these from the environment's observation_space and action_space
    
    # Create runner for high-level policy
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # Skip git state logging (optional, uncomment if needed for debugging)
    # runner.add_git_repo_to_log(__file__)
    
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "low_level_checkpoint.yaml"), {"path": low_level_checkpoint})
    
    # run training
    print(f"\n[INFO] Starting training with {env_cfg.scene.num_envs} environments...")
    print(f"[INFO] High-level action space: {hierarchical_env.action_space}")
    print(f"[INFO] High-level observation space: {hierarchical_env.observation_space}")
    print(f"\n[NOTE] High-level navigation metrics (goal_reached, distance_to_goal) are computed")
    print(f"       but may not appear in RSL-RL logs. Check 'Mean reward' for high-level reward.\n")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    hierarchical_env.env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

