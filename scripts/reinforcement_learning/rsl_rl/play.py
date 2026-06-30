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
import math
import time
from collections import deque

import torch

import isaaclab.utils.math as math_utils

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


def _phase_traj_body(
    phase_s: torch.Tensor,
    cycle_time: float,
    gait_span: float,
    gait_psi: float,
    gait_delta: float,
    x_offset: float,
    stance_span: float,
) -> torch.Tensor:
    """MuJoCo-style local body-frame foot offsets, shape [N, 3].

    Mirrors `phase_foot_trajectory_exp` reward in rl_training rewards.py.
    `phase_s` is per-foot phase in [0, 2). Returns (q + x_offset, 0, z) in body frame.
    """
    tau = float(gait_span)
    psi = float(gait_psi)
    delta = float(gait_delta)
    stance_span = min(max(float(stance_span), 1e-6), 2.0 - 1e-6)

    S = phase_s
    q = torch.zeros_like(S)
    z = torch.zeros_like(S)

    stance_mask = S < stance_span
    if stance_mask.any():
        s_stance = S / stance_span
        q = torch.where(stance_mask, tau * (1.0 - 2.0 * s_stance), q)
        z = torch.where(stance_mask, torch.full_like(S, delta), z)

    swing_mask = ~stance_mask
    if swing_mask.any():
        t_b = torch.clamp((S - stance_span) / (2.0 - stance_span), 0.0, 1.0)
        ctrl = torch.tensor(
            [
                [-tau, 0.0],
                [-0.95 * tau, 0.80 * psi],
                [-0.55 * tau, 1.00 * psi],
                [0.55 * tau, 1.00 * psi],
                [0.95 * tau, 0.80 * psi],
                [tau, 0.0],
            ],
            device=S.device,
            dtype=S.dtype,
        )
        n = ctrl.shape[0] - 1
        out = torch.zeros(*t_b.shape, 2, device=S.device, dtype=S.dtype)
        for k in range(n + 1):
            coeff = float(math.comb(n, k))
            basis = coeff * (1.0 - t_b) ** (n - k) * t_b ** k
            out = out + basis.unsqueeze(-1) * ctrl[k]
        q = torch.where(swing_mask, out[..., 0], q)
        z = torch.where(swing_mask, out[..., 1] + delta, z)

    return torch.stack([q + float(x_offset), torch.zeros_like(q), z], dim=-1)


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
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
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

    # ====== Visualization init ======
    VIS_ENABLED = True
    draw_interface = None
    foot_ids = None
    act_hist = None
    cmd_hist = None
    stand_ref_body = None

    # Pull gait reference parameters from env config (matches training reward).
    gait_params = {
        "cycle_time": 0.425,
        "phase_offsets": (0.0, 1.0, 1.0, 0.0),
        "gait_span": 0.0,
        "gait_psi": 0.05,
        "gait_delta": 0.02,
        "x_offset": 0.0,
        "stance_span": 0.0,
        "stand_ref_z_offset": 0.0,
        "command_threshold": 0.1,
    }
    has_ref_trajectory = False
    try:
        cfg_params = env_cfg.rewards.phase_foot_trajectory_exp.params
        for key in gait_params:
            if key in cfg_params:
                gait_params[key] = cfg_params[key]
        has_ref_trajectory = True
    except Exception:
        pass

    # Per-foot reference trail color [r, g, b, a].
    color_palette = [
        [1.0, 0.2, 0.2, 1.0],   # foot 0: red
        [0.2, 1.0, 0.2, 1.0],   # foot 1: green
        [0.2, 0.4, 1.0, 1.0],   # foot 2: blue
        [1.0, 0.85, 0.0, 1.0],  # foot 3: yellow
    ]

    try:
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
        draw_interface = omni_debug_draw.acquire_debug_draw_interface()
        if draw_interface is None:
            VIS_ENABLED = False
    except Exception:
        VIS_ENABLED = False

    if VIS_ENABLED:
        robot = env.unwrapped.scene["robot"]
        # Resolve foot/wheel bodies. Prefer the env_cfg.foot_link_name regex
        # (Lite3: ".*_FOOT", M20: ".*_wheel"); otherwise fall back to a
        # name-substring heuristic so this script works on robots that don't
        # expose foot_link_name.
        foot_pattern = getattr(env_cfg, "foot_link_name", None)
        if foot_pattern:
            ids, names = robot.find_bodies(foot_pattern)
        else:
            names = sorted(
                [n for n in robot.body_names if "foot" in n.lower() or "wheel" in n.lower()]
            )
            ids = [robot.find_bodies(n)[0][0] for n in names] if names else []

        if len(ids) >= 4:
            foot_ids = list(ids[:4])
            foot_names = list(names[:4])
            act_hist = [deque(maxlen=100) for _ in range(4)]
            cmd_hist = [deque(maxlen=100) for _ in range(4)]
            print(f"[INFO] Using foot bodies: {foot_names}")
            print(f"[INFO] Gait reference params: {gait_params}")
        else:
            print(
                f"[WARN] Found {len(ids)} foot/wheel bodies (need >= 4). "
                "Disabling foot trajectory visualization."
            )
            VIS_ENABLED = False

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

        # ----- Draw foot trajectories -----
        if VIS_ENABLED and draw_interface and foot_ids and act_hist and cmd_hist:
            robot = env.unwrapped.scene["robot"]
            root_pos = robot.data.root_pos_w[0]
            root_quat = robot.data.root_quat_w[0].unsqueeze(0)
            actual_world = robot.data.body_pos_w[0, foot_ids, :]

            # Initialize body-fixed stand reference once from current posture.
            if has_ref_trajectory and stand_ref_body is None:
                rel_init = actual_world - root_pos.unsqueeze(0)
                stand_ref_body = math_utils.quat_apply_inverse(
                    root_quat.expand(len(foot_ids), -1), rel_init
                )
                stand_ref_body[:, 2] += float(gait_params["stand_ref_z_offset"])

            # Phase-driven local foot trajectory in body frame (only if reward exists).
            if has_ref_trajectory:
                elapsed_t = float(env.unwrapped.common_step_counter) * dt
                phase_offsets_t = torch.tensor(
                    list(gait_params["phase_offsets"]),
                    device=root_pos.device,
                    dtype=torch.float32,
                )
                phase_s = torch.remainder(
                    (2.0 * elapsed_t / max(float(gait_params["cycle_time"]), 1e-6)) + phase_offsets_t,
                    2.0,
                )
                cmd_local = _phase_traj_body(
                    phase_s,
                    cycle_time=gait_params["cycle_time"],
                    gait_span=gait_params["gait_span"],
                    gait_psi=gait_params["gait_psi"],
                    gait_delta=gait_params["gait_delta"],
                    x_offset=gait_params["x_offset"],
                    stance_span=gait_params["stance_span"],
                )
                ref_body = stand_ref_body + cmd_local
                ref_world = root_pos.unsqueeze(0) + math_utils.quat_apply(
                    root_quat.expand(len(foot_ids), -1), ref_body
                )

            # Gate the reference trail on commanded velocity magnitude.
            try:
                cmd_vec = env.unwrapped.command_manager.get_command("base_velocity")[0, :3]
                gate_on = torch.linalg.norm(cmd_vec).item() > float(gait_params["command_threshold"])
            except Exception:
                gate_on = True

            for i in range(4):
                act_hist[i].append(actual_world[i].detach().cpu().tolist())
            if has_ref_trajectory:
                for i in range(4):
                    cmd_hist[i].append(ref_world[i].detach().cpu().tolist())

            draw_interface.clear_lines()
            starts, ends, colors, widths = [], [], [], []

            # Actual foot trail (black)
            for i in range(4):
                pts = list(act_hist[i])
                for j in range(1, len(pts)):
                    starts.append(pts[j - 1])
                    ends.append(pts[j])
                    colors.append([0.0, 0.0, 0.0, 0.6])
                    widths.append(1.5)

            # Reference trail (per-foot color), only when commanded and trajectory exists
            if has_ref_trajectory and gate_on:
                for i in range(4):
                    pts = list(cmd_hist[i])
                    for j in range(1, len(pts)):
                        starts.append(pts[j - 1])
                        ends.append(pts[j])
                        colors.append(color_palette[i])
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
