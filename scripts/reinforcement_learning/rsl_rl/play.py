# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections import deque
import math
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# import after SimulationApp is created to avoid early Omniverse/pxr imports
from rl_utils import camera_follow

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform
from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

import isaaclab.utils.math as math_utils

try:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
except Exception:
    try:
        import omni.isaac.debug_draw._debug_draw as omni_debug_draw
    except Exception:
        omni_debug_draw = None

from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50

    # handle deprecated configurations (convert old policy format to new actor/critic format)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels = None

    keyboard_command_state = None
    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1]/2,
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)

        def _keyboard_obs_term(env):
            nonlocal keyboard_command_state
            keyboard_command_state = torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device)
            return keyboard_command_state

        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=_keyboard_obs_term,
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    # convert config to dict and create runner
    train_cfg = agent_cfg.to_dict()
    ppo_runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # Use runner-native exporters for rsl-rl >= 4.0.0
        ppo_runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        ppo_runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        policy_nn = None
    else:
        # Fallback for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = ppo_runner.alg.policy
        else:
            policy_nn = ppo_runner.alg.actor_critic

        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        else:
            normalizer = None

        export_policy_as_onnx(
            policy=policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )
        export_policy_as_jit(
            policy=policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.pt",
        )

    dt = env.unwrapped.step_dt
    # reset environment
    obs, _ = env.reset()
    
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # env stepping
            obs, _, _, _ = env.step(actions)

        if (
            VIS_ENABLEd
            and draw_interface is not None
            and foot_ids is not None
            and phase_offsets is not None
            and cycle_time is not None
            and gait_span is not None
            and gait_psi is not None
            and gait_delta is not None
            and x_offset is not None
            and stance_span is not None
            and cmd_threshold is not None
            and stand_ref_z_offset is not None
            and cmd_hist is not None
            and act_hist is not None
        ):
            local_foot_ids = foot_ids
            robot = env.unwrapped.scene["robot"]
            root_pos = robot.data.root_pos_w[0]
            root_quat = robot.data.root_quat_w[0].unsqueeze(0)

            # Initialize base-fixed stand reference once from current posture.
            if stand_ref_body is None:
                rel_init = robot.data.body_pos_w[0, local_foot_ids, :] - root_pos.unsqueeze(0)
                stand_ref_body = math_utils.quat_apply_inverse(root_quat.expand(len(local_foot_ids), -1), rel_init)
                stand_ref_body[:, 2] += stand_ref_z_offset

            elapsed_t = float(env.unwrapped.common_step_counter) * dt
            phase_s = torch.remainder((2.0 * elapsed_t / max(cycle_time, 1e-6)) + phase_offsets, 2.0)
            cmd_local = _mujoco_phase_traj_body(
                phase_s=phase_s,
                gait_span=gait_span,
                gait_psi=gait_psi,
                gait_delta=gait_delta,
                x_offset=x_offset,
                stance_span=stance_span,
            )
            ref_body = stand_ref_body + cmd_local
            ref_world = root_pos.unsqueeze(0) + math_utils.quat_apply(root_quat.expand(len(local_foot_ids), -1), ref_body)

            actual_world = robot.data.body_pos_w[0, local_foot_ids, :]
            for i in range(4):
                cmd_hist[i].append(ref_world[i].detach().cpu().tolist())
                act_hist[i].append(actual_world[i].detach().cpu().tolist())

            if args_cli.keyboard and keyboard_command_state is not None:
                cmd_vec = keyboard_command_state[0, :3]
            else:
                cmd_vec = env.unwrapped.command_manager.get_command("base_velocity")[0, :3]

            cmd_norm = torch.linalg.norm(cmd_vec).item()
            gate_on = cmd_norm > cmd_threshold

            if args_cli.keyboard:
                ref_gate_on = cmd_norm > 0.1
            else:
                ref_gate_on = gate_on

            draw_interface.clear_lines()
            starts = []
            ends = []
            colors = []
            widths = []

            ref_alpha = 0.95 if gate_on else 0.35
            act_alpha = 0.35 if gate_on else 0.20

            if not phase_vis_z_printed:
                print(
                    "[INFO] phase_foot_trajectory_exp z check: "
                    f"ref_z_mean={ref_world[:, 2].mean().item():.4f}, "
                    f"act_z_mean={actual_world[:, 2].mean().item():.4f}, "
                    f"stand_ref_z_offset={stand_ref_z_offset:.4f}"
                )
                phase_vis_z_printed = True

            for i in range(4):
                act_pts = list(act_hist[i])
                for j in range(1, len(act_pts)):
                    starts.append(act_pts[j - 1])
                    ends.append(act_pts[j])
                    colors.append([0.0, 0.0, 0.0, act_alpha])
                    widths.append(1.5)

                cmd_pts = list(cmd_hist[i])
                if VIS_REF_ENABLE and ref_gate_on:
                    for j in range(1, len(cmd_pts)):
                        starts.append(cmd_pts[j - 1])
                        ends.append(cmd_pts[j])
                        color = color_palette[i].copy()
                        color[3] = ref_alpha
                        colors.append(color)
                        widths.append(2.8)

            if starts:
                draw_interface.draw_lines(starts, ends, colors, widths)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()