# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

# Use this checkout even when m20_wzh also has an older rl_training installation.
_LOCAL_SOURCE = Path(__file__).resolve().parents[3] / "source" / "rl_training"
sys.path.insert(0, str(_LOCAL_SOURCE))

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--init_actor_from", type=str, default=None,
    help="Initialize only the actor from an RSL-RL checkpoint; critic, optimizer, and iteration start fresh.",
)
parser.add_argument("--init_critic_from", type=str, default=None,
                    help="Initialize critic weights with --init_actor_from, keeping a fresh optimizer.")
parser.add_argument("--preserve_actor_from", type=str, default=None,
                    help="Freeze a task-capable reference for StairGuardedPPO (also persisted in checkpoints).")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.init_actor_from and args_cli.resume:
    parser.error("--init_actor_from and --resume cannot be used together")
if args_cli.init_critic_from and (not args_cli.init_actor_from or args_cli.resume):
    parser.error("--init_critic_from requires --init_actor_from and cannot be used with --resume")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# suppress noisy omni.usd warnings (e.g. unresolved visual prim references)
import carb
carb.logging.acquire_logging().set_level_threshold_for_source(
    "omni.usd", carb.logging.LogSettingBehavior.OVERRIDE, carb.logging.LEVEL_ERROR
)

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "5.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    cmd = [sys.executable, "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_callable

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401
from rl_training.envs.amp_locomotion_env import AmpLocomotionEnv, AmpRslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    runner_class_name = getattr(agent_cfg, "class_name", "OnPolicyRunner")
    is_custom_runner = runner_class_name not in ("OnPolicyRunner", "DistillationRunner")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # A file path permits full resume across experiment directories, while a
    # filename/regex retains the original --load_run lookup behavior.
    _algo_class_name = getattr(getattr(agent_cfg, "algorithm", None), "class_name", None)
    if agent_cfg.resume or _algo_class_name == "Distillation":
        checkpoint = Path(agent_cfg.load_checkpoint).expanduser()
        if checkpoint.is_absolute() or os.sep in agent_cfg.load_checkpoint:
            checkpoint = checkpoint.resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
            resume_path = str(checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    if isinstance(env.unwrapped, AmpLocomotionEnv):
        env = AmpRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # convert config to dict and create runner
    train_cfg = agent_cfg.to_dict()
    if version.parse(installed_version) >= version.parse("5.0.0"):
        train_cfg = cli_args.convert_rsl_rl_cfg_dict(train_cfg)
    runner_class = resolve_callable(runner_class_name) if is_custom_runner else OnPolicyRunner
    runner = runner_class(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    if args_cli.init_actor_from:
        init_path = Path(args_cli.init_actor_from).expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"Actor initialization checkpoint does not exist: {init_path}")
        saved = torch.load(init_path, map_location="cpu", weights_only=True)
        actor_state = saved.get("actor_state_dict")
        if not isinstance(actor_state, dict):
            raise ValueError(f"Checkpoint has no actor_state_dict: {init_path}")
        runner.alg.actor.load_state_dict(actor_state, strict=True)
        print(f"[INFO]: Initialized actor only from: {init_path}")

    if args_cli.init_critic_from:
        init_path = Path(args_cli.init_critic_from).expanduser().resolve()
        saved = torch.load(init_path, map_location="cpu", weights_only=True)
        runner.alg.critic.load_state_dict(saved["critic_state_dict"], strict=True)
        print(f"[INFO]: Initialized critic weights (fresh optimizer) from: {init_path}")
    reference_path = args_cli.preserve_actor_from or args_cli.init_actor_from
    if hasattr(runner.alg, "set_reference") and reference_path:
        reference = torch.load(Path(reference_path).expanduser().resolve(), map_location="cpu", weights_only=True)
        runner.alg.set_reference(reference["actor_state_dict"])
        print(f"[INFO]: Frozen task preservation actor from: {reference_path}")
    elif args_cli.preserve_actor_from:
        raise ValueError("--preserve_actor_from requires the StairGuardedPPO algorithm")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "cli.yaml"), vars(args_cli))
    
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[INFO]: Training completed: {log_dir}", flush=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Kit fast shutdown can suppress an unwinding exception and exit 0.
        # Emit it and return a real failure before simulator teardown.
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
    finally:
        simulation_app.close()
