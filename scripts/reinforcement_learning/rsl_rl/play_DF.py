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

installed_version = metadata.version("rsl-rl-lib")

# ------------------------------Trajectory Visualization Switches------------------------------
# Master switch for phase foot trajectory visualization in play.
VIS_ENABLE = True
# Switch for drawing reference trajectory (actual trajectory can still be drawn).
VIS_REF_ENABLE = True
# Which phase reward term to visualize: auto | phase_foot_trajectory_exp | phase_wheel_trajectory_exp
PHASE_VIS_REWARD_TERM = "phase_wheel_trajectory_exp"


def _bernstein_torch(n: int, k: int, t: torch.Tensor) -> torch.Tensor:
    coeff = float(math.comb(n, k))
    return coeff * (1.0 - t) ** (n - k) * t**k


def _bezier_curve_torch(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    n = control_points.shape[0] - 1
    out = torch.zeros(*t.shape, 2, device=t.device, dtype=t.dtype)
    for k in range(n + 1):
        out = out + _bernstein_torch(n, k, t).unsqueeze(-1) * control_points[k]
    return out


def _mujoco_phase_traj_body(
    phase_s: torch.Tensor,
    gait_span: float,
    gait_psi: float,
    gait_delta: float,
    x_offset: float,
    stance_span: float,
) -> torch.Tensor:
    """MuJoCo-style local trajectory on XZ plane for phase S in [0, 2)."""
    stance_span = min(max(float(stance_span), 1e-6), 2.0 - 1e-6)
    tau = float(gait_span)
    psi = float(gait_psi)
    delta = float(gait_delta)

    q = torch.zeros_like(phase_s)
    z = torch.zeros_like(phase_s)

    stance_mask = phase_s < stance_span
    if stance_mask.any():
        s_stance = phase_s / stance_span
        q_stance = tau * (1.0 - 2.0 * s_stance)
        z_stance = torch.full_like(phase_s, delta)
        q = torch.where(stance_mask, q_stance, q)
        z = torch.where(stance_mask, z_stance, z)

    swing_mask = ~stance_mask
    if swing_mask.any():
        t_bezier = torch.clamp((phase_s - stance_span) / (2.0 - stance_span), 0.0, 1.0)
        ctrl = torch.tensor(
            [
                [-tau, 0.0],
                [-0.95 * tau, 0.80 * psi],
                [-0.55 * tau, 1.00 * psi],
                [0.55 * tau, 1.00 * psi],
                [0.95 * tau, 0.80 * psi],
                [tau, 0.0],
            ],
            device=phase_s.device,
            dtype=phase_s.dtype,
        )
        qz_swing = _bezier_curve_torch(ctrl, t_bezier)
        q = torch.where(swing_mask, qz_swing[..., 0], q)
        z = torch.where(swing_mask, qz_swing[..., 1] + delta, z)

    return torch.stack([q + float(x_offset), torch.zeros_like(q), z], dim=-1)


def _append_arrow_lines(
    starts: list,
    ends: list,
    colors: list,
    widths: list,
    origin: list[float],
    vec_xy: list[float],
    color: list[float],
    width: float = 3.0,
    z_offset: float = 0.05,
    min_len: float = 0.15,
    max_len: float = 1.2,
):
    """Append a 2D XY arrow (with head) in world frame."""
    vx, vy = float(vec_xy[0]), float(vec_xy[1])
    mag = math.sqrt(vx * vx + vy * vy)
    if mag <= 1e-6:
        return

    dx, dy = vx / mag, vy / mag
    arrow_len = max(min_len, min(max_len, mag))
    start = [origin[0], origin[1], origin[2] + z_offset]
    end = [start[0] + arrow_len * dx, start[1] + arrow_len * dy, start[2]]
    starts.append(start)
    ends.append(end)
    colors.append(color)
    widths.append(width)

    head_len = min(0.2, 0.35 * arrow_len)
    nx, ny = -dy, dx
    back = [end[0] - head_len * dx, end[1] - head_len * dy, end[2]]
    left = [back[0] + 0.5 * head_len * nx, back[1] + 0.5 * head_len * ny, back[2]]
    right = [back[0] - 0.5 * head_len * nx, back[1] - 0.5 * head_len * ny, back[2]]
    starts.append(end); ends.append(left); colors.append(color); widths.append(max(1.0, width - 1.0))
    starts.append(end); ends.append(right); colors.append(color); widths.append(max(1.0, width - 1.0))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1

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
    is_waypoint_command = False
    # Detect if base_velocity is a WaypointPositionCommand
    _base_vel_cfg = getattr(env_cfg.commands, "base_velocity", None)
    if _base_vel_cfg is not None:
        _cfg_cls_name = type(_base_vel_cfg).__name__
        is_waypoint_command = "Waypoint" in _cfg_cls_name or "waypoint" in _cfg_cls_name

    if args_cli.keyboard and is_waypoint_command:
        print("[WARN] --keyboard is not compatible with WaypointPositionCommand. "
              "Waypoints are auto-generated. Keyboard control disabled.")
        args_cli.keyboard = False
    elif args_cli.keyboard:
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

    try:
        ppo_runner.load(resume_path)
    except RuntimeError as exc:
        raise

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

    # phase trajectory visualization (MuJoCo style), synchronized with selected reward params.
    draw_interface = None
    VIS_ENABLEd = False
    active_phase_reward_name = None
    phase_vis_printed = False
    phase_vis_z_printed = False
    foot_ids = None
    phase_offsets = None
    cycle_time = None
    gait_span = None
    gait_psi = None
    gait_delta = None
    x_offset = None
    stance_span = None
    cmd_threshold = None
    stand_ref_z_offset = None
    cmd_hist = None
    act_hist = None
    stand_ref_body = None
    color_palette = [
        [0.12, 0.47, 0.71, 0.85],
        [1.00, 0.50, 0.05, 0.85],
        [0.17, 0.63, 0.17, 0.85],
        [0.84, 0.15, 0.16, 0.85],
    ]

    if PHASE_VIS_REWARD_TERM == "auto":
        for candidate_name in ("phase_wheel_trajectory_exp", "phase_foot_trajectory_exp"):
            candidate_cfg = getattr(env_cfg.rewards, candidate_name, None)
            if candidate_cfg is not None and getattr(candidate_cfg, "weight", 0.0) != 0.0:
                active_phase_reward_name = candidate_name
                phase_cfg = candidate_cfg
                break
        else:
            phase_cfg = None
    else:
        active_phase_reward_name = PHASE_VIS_REWARD_TERM
        phase_cfg = getattr(env_cfg.rewards, active_phase_reward_name, None)

    if VIS_ENABLE and phase_cfg is not None and getattr(phase_cfg, "weight", 0.0) != 0.0:
        if omni_debug_draw is None:
            print("[WARN] Debug draw extension is unavailable. Phase trajectory visualization is disabled.")
            phase_cfg = None

    if VIS_ENABLE and phase_cfg is not None and getattr(phase_cfg, "weight", 0.0) != 0.0:
        params = phase_cfg.params
        robot = env.unwrapped.scene["robot"]
        phase_asset_cfg = params["asset_cfg"]

        raw_body_ids = getattr(phase_asset_cfg, "body_ids", None)
        body_names = getattr(phase_asset_cfg, "body_names", None)

        # Prefer resolving by body_names regex to avoid unresolved body_ids=slice(None)
        # being interpreted as all robot bodies.
        if body_names not in (None, "", []):
            foot_ids, foot_names = robot.find_bodies(body_names)
            print(f"[INFO] {active_phase_reward_name} body match: {foot_names}")
        elif isinstance(raw_body_ids, slice):
            total_bodies = int(robot.data.body_pos_w.shape[1])
            foot_ids = list(range(total_bodies))[raw_body_ids]
        elif isinstance(raw_body_ids, int):
            foot_ids = [raw_body_ids]
        elif raw_body_ids is None:
            foot_ids, _ = robot.find_bodies(phase_asset_cfg.body_names)
        else:
            try:
                foot_ids = list(raw_body_ids)
            except TypeError:
                foot_ids, _ = robot.find_bodies(phase_asset_cfg.body_names)

        if len(foot_ids) == 4:
            cycle_time = float(params.get("cycle_time", 0.4))
            phase_offsets = torch.tensor(
                params.get("phase_offsets", (0.0, 1.0, 1.0, 0.0)),
                device=env.unwrapped.device,
                dtype=torch.float32,
            )
            gait_span = float(params.get("gait_span", -0.008))
            gait_psi = float(params.get("gait_psi", 0.15))
            gait_delta = float(params.get("gait_delta", 0.03))
            x_offset = float(params.get("x_offset", 0.0))
            stance_span = float(params.get("stance_span", 0.20))
            cmd_threshold = float(params.get("command_threshold", 0.1))
            stand_ref_z_offset = float(params.get("stand_ref_z_offset", -0.2))

            cmd_hist = [deque(maxlen=80) for _ in range(4)]
            act_hist = [deque(maxlen=80) for _ in range(4)]
            if omni_debug_draw is not None:
                draw_interface = omni_debug_draw.acquire_debug_draw_interface()
            else:
                draw_interface = None
            VIS_ENABLEd = True
            print(f"[INFO] {active_phase_reward_name} visualization enabled.")
        else:
            print(f"[WARN] {active_phase_reward_name} expects 4 feet, got {len(foot_ids)}. Visualization disabled.")
    # print(dt, "dt")

    # ---- Waypoint visualization setup ----
    wp_command_term = None
    wp_vis_enabled = True
    try:
        for _cmd_name, _cmd_term in env.unwrapped.command_manager._terms.items():
            if hasattr(_cmd_term, "wp_pos") and hasattr(_cmd_term, "wp_idx"):
                wp_command_term = _cmd_term
                if omni_debug_draw is not None:
                    wp_vis_enabled = True
                    # Reuse draw_interface if phase vis already acquired it, else acquire now
                    if draw_interface is None:
                        draw_interface = omni_debug_draw.acquire_debug_draw_interface()
                    print(f"[INFO] Waypoint visualization enabled for command: '{_cmd_name}'")
                else:
                    print("[WARN] Debug draw unavailable. Waypoint visualization disabled.")
                break
    except Exception as _e:
        print(f"[WARN] Could not find WaypointPositionCommand: {_e}")

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

        # ---- Waypoint visualization (env 0) ----
        if wp_vis_enabled and draw_interface is not None and wp_command_term is not None:
            env_id = 0
            wp_count = wp_command_term.wp_count[env_id].item()
            wp_idx = wp_command_term.wp_idx[env_id].item()
            env_origin = env.unwrapped.scene.env_origins[env_id]
            robot_wp = env.unwrapped.scene["robot"]
            robot_z = robot_wp.data.root_pos_w[env_id, 2].item()

            # Build world-frame waypoint positions (slightly above ground)
            wp_world = []
            for _wi in range(wp_count):
                _lx = wp_command_term.wp_pos[env_id, _wi, 0].item()
                _ly = wp_command_term.wp_pos[env_id, _wi, 1].item()
                wp_world.append([
                    env_origin[0].item() + _lx,
                    env_origin[1].item() + _ly,
                    robot_z + 0.05,
                ])

            wp_s, wp_e, wp_c, wp_w = [], [], [], []

            # Path lines between consecutive waypoints
            for _wi in range(1, wp_count):
                wp_s.append(wp_world[_wi - 1])
                wp_e.append(wp_world[_wi])
                if _wi < wp_idx:
                    wp_c.append([0.5, 0.5, 0.5, 0.4])   # passed: gray
                    wp_w.append(2.0)
                elif _wi == wp_idx:
                    wp_c.append([0.0, 1.0, 0.0, 1.0])   # current segment: green
                    wp_w.append(4.0)
                else:
                    wp_c.append([0.2, 0.6, 1.0, 0.6])   # future: blue
                    wp_w.append(2.0)

            # Robot -> current target (red)
            _rp = robot_wp.data.root_pos_w[env_id].detach().cpu().tolist()
            _tp = wp_command_term.pos_command_w[env_id].detach().cpu().tolist()
            wp_s.append(_rp); wp_e.append(_tp)
            wp_c.append([1.0, 0.2, 0.0, 1.0]); wp_w.append(3.0)

            # Robot body linear velocity arrow (magenta): root_lin_vel_w xy
            _v_body_xy = robot_wp.data.root_lin_vel_w[env_id, :2].detach().cpu().tolist()
            _append_arrow_lines(
                wp_s,
                wp_e,
                wp_c,
                wp_w,
                origin=_rp,
                vec_xy=_v_body_xy,
                color=[1.0, 0.0, 1.0, 1.0],
                width=4.0,
                z_offset=0.08,
                min_len=0.18,
                max_len=1.5,
            )

            # Current target expected velocity arrow (yellow)
            if 0 <= wp_idx < wp_count:
                _vxy = wp_command_term.wp_vel[env_id, wp_idx].detach().cpu()
                _vmag = torch.norm(_vxy).item()
                if _vmag > 1e-6:
                    _vdir = (_vxy / _vmag).tolist()
                    # Arrow length represents speed magnitude with a minimum visible size.
                    _arrow_len = max(0.15, _vmag)
                    _p = wp_world[wp_idx]
                    _arrow_end = [
                        _p[0] + _arrow_len * _vdir[0],
                        _p[1] + _arrow_len * _vdir[1],
                        _p[2],
                    ]
                    wp_s.append(_p)
                    wp_e.append(_arrow_end)
                    wp_c.append([1.0, 1.0, 0.0, 1.0])
                    wp_w.append(4.0)
                    # Small V-shaped arrow head.
                    _head_len = min(0.2, 0.35 * _arrow_len)
                    _nx, _ny = -_vdir[1], _vdir[0]
                    _back = [_arrow_end[0] - _head_len * _vdir[0], _arrow_end[1] - _head_len * _vdir[1], _arrow_end[2]]
                    _left = [_back[0] + 0.5 * _head_len * _nx, _back[1] + 0.5 * _head_len * _ny, _back[2]]
                    _right = [_back[0] - 0.5 * _head_len * _nx, _back[1] - 0.5 * _head_len * _ny, _back[2]]
                    wp_s.append(_arrow_end); wp_e.append(_left)
                    wp_c.append([1.0, 1.0, 0.0, 1.0]); wp_w.append(3.0)
                    wp_s.append(_arrow_end); wp_e.append(_right)
                    wp_c.append([1.0, 1.0, 0.0, 1.0]); wp_w.append(3.0)

            # Current target self-motion velocity arrow (cyan)
            if 0 <= wp_idx < wp_count and hasattr(wp_command_term, "wp_move_vel"):
                _mvxy = wp_command_term.wp_move_vel[env_id, wp_idx].detach().cpu()
                _mvmag = torch.norm(_mvxy).item()
                if _mvmag > 1e-6:
                    _mvdir = (_mvxy / _mvmag).tolist()
                    _arrow_len = max(0.15, _mvmag)
                    _p = wp_world[wp_idx]
                    _arrow_end = [
                        _p[0] + _arrow_len * _mvdir[0],
                        _p[1] + _arrow_len * _mvdir[1],
                        _p[2] + 0.03,
                    ]
                    wp_s.append(_p)
                    wp_e.append(_arrow_end)
                    wp_c.append([0.0, 1.0, 1.0, 1.0])
                    wp_w.append(4.0)
                    _head_len = min(0.2, 0.35 * _arrow_len)
                    _nx, _ny = -_mvdir[1], _mvdir[0]
                    _back = [_arrow_end[0] - _head_len * _mvdir[0], _arrow_end[1] - _head_len * _mvdir[1], _arrow_end[2]]
                    _left = [_back[0] + 0.5 * _head_len * _nx, _back[1] + 0.5 * _head_len * _ny, _back[2]]
                    _right = [_back[0] - 0.5 * _head_len * _nx, _back[1] - 0.5 * _head_len * _ny, _back[2]]
                    wp_s.append(_arrow_end); wp_e.append(_left)
                    wp_c.append([0.0, 1.0, 1.0, 1.0]); wp_w.append(3.0)
                    wp_s.append(_arrow_end); wp_e.append(_right)
                    wp_c.append([0.0, 1.0, 1.0, 1.0]); wp_w.append(3.0)

            # Cross markers at each waypoint
            _cross = 0.15
            for _wi in range(wp_count):
                _p = wp_world[_wi]
                if _wi == wp_idx:
                    _mc, _mw = [0.0, 1.0, 0.0, 1.0], 5.0    # current: green
                elif _wi < wp_idx:
                    _mc, _mw = [0.5, 0.5, 0.5, 0.4], 2.0    # passed: gray
                else:
                    _mc, _mw = [0.2, 0.6, 1.0, 0.8], 3.0    # future: blue
                # horizontal cross (X direction)
                wp_s.append([_p[0] - _cross, _p[1], _p[2]])
                wp_e.append([_p[0] + _cross, _p[1], _p[2]])
                wp_c.append(_mc); wp_w.append(_mw)
                # horizontal cross (Y direction)
                wp_s.append([_p[0], _p[1] - _cross, _p[2]])
                wp_e.append([_p[0], _p[1] + _cross, _p[2]])
                wp_c.append(_mc); wp_w.append(_mw)

            # If phase vis did NOT run this frame, clear stale lines first
            if not VIS_ENABLEd:
                draw_interface.clear_lines()
            if wp_s:
                draw_interface.draw_lines(wp_s, wp_e, wp_c, wp_w)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # if args_cli.keyboard:
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
