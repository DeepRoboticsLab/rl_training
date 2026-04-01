# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Fast ONNX export for Deep Robotics policy — no Isaac Sim required.

Reconstructs the actor MLP directly from a checkpoint and exports it to ONNX,
optionally embedding robot metadata as ONNX model properties.

Examples:
    # Lite3
    python scripts/tools/export_onnx_fast.py \\
        --checkpoint_path logs/rsl_rl/deeprobotics_lite3_rough/2025-01-01_12-00-00/model_5000.pt \\
        --robot lite3 \\
        --output_path exported/lite3_policy.onnx

    # M20
    python scripts/tools/export_onnx_fast.py \\
        --checkpoint_path logs/rsl_rl/deeprobotics_m20_rough/2025-01-01_12-00-00/model_5000.pt \\
        --robot m20 \\
        --output_path exported/m20_policy.onnx
"""

import argparse
import os

import onnx
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Robot constants (joint order matches USD traversal / env config)
# ---------------------------------------------------------------------------

# fmt: off
_LITE3_JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]

_LITE3_LINK_NAMES = [
    "TORSO",
    "FL_HIP", "FR_HIP", "HL_HIP", "HR_HIP",
    "FL_THIGH", "FR_THIGH", "HL_THIGH", "HR_THIGH",
    "FL_SHANK", "FR_SHANK", "HL_SHANK", "HR_SHANK",
    "FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT",
]

# HipX: stiffness=30, damping=1 | HipY: stiffness=30, damping=1 | Knee: stiffness=30, damping=1
_LITE3_STIFFNESS  = [30.0, 30.0, 30.0] * 4
_LITE3_DAMPING    = [1.0,  1.0,  1.0 ] * 4
# Default init pose from DEEPROBOTICS_LITE3_CFG
_LITE3_DEFAULT_POS = [0.0, -0.8, 1.6] * 4
# Action scale: HipX=0.125, HipY=0.25, Knee=0.25
_LITE3_ACTION_SCALE = [0.125, 0.25, 0.25] * 4

LITE3_CFG = {
    "joint_names":       _LITE3_JOINT_NAMES,
    "link_names":        _LITE3_LINK_NAMES,
    "base_link":         "TORSO",
    "stiffness":         _LITE3_STIFFNESS,
    "damping":           _LITE3_DAMPING,
    "default_joint_pos": _LITE3_DEFAULT_POS,
    "action_scale":      _LITE3_ACTION_SCALE,
}

# M20: 12 leg joints + 4 wheel joints = 16 total
_M20_LEG_JOINT_NAMES = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
]
_M20_WHEEL_JOINT_NAMES = [
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
]
_M20_JOINT_NAMES = _M20_LEG_JOINT_NAMES + _M20_WHEEL_JOINT_NAMES

# Leg: stiffness=80, damping=2 | Wheel: stiffness=0, damping=0.6
_M20_STIFFNESS  = [80.0] * 12 + [0.0] * 4
_M20_DAMPING    = [2.0 ] * 12 + [0.6] * 4
# Default init pose from DEEPROBOTICS_M20_CFG
_M20_DEFAULT_POS = [
    0.0, -0.6,  1.0,   # fl: hipx, hipy, knee
    0.0, -0.6,  1.0,   # fr: hipx, hipy, knee
    0.0,  0.6, -1.0,   # hl: hipx, hipy, knee
    0.0,  0.6, -1.0,   # hr: hipx, hipy, knee
    0.0,  0.0,  0.0,  0.0,  # wheels
]
# Action scale: leg hipx=0.125, leg other=0.25, wheel velocity scale=5.0
_M20_ACTION_SCALE = [0.125, 0.25, 0.25] * 4 + [5.0] * 4

M20_CFG = {
    "joint_names":       _M20_JOINT_NAMES,
    "link_names":        None,   # not specified in env config
    "base_link":         "base_link",
    "stiffness":         _M20_STIFFNESS,
    "damping":           _M20_DAMPING,
    "default_joint_pos": _M20_DEFAULT_POS,
    "action_scale":      _M20_ACTION_SCALE,
}
# fmt: on

ROBOT_CONFIGS = {
    "lite3": LITE3_CFG,
    "m20":   M20_CFG,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_actor(state_dict: dict) -> nn.Sequential:
    """Reconstruct actor MLP from checkpoint weights.

    RSL-RL stores actor layers at even indices (0, 2, 4, ...) with ELU
    activations between hidden layers.
    """
    layers: list[nn.Module] = []
    i = 0
    while f"actor.{i}.weight" in state_dict:
        w = state_dict[f"actor.{i}.weight"]
        b = state_dict[f"actor.{i}.bias"]
        linear = nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data.copy_(w)
        linear.bias.data.copy_(b)
        layers.append(linear)
        if f"actor.{i + 2}.weight" in state_dict:  # not the last layer
            layers.append(nn.ELU())
        i += 2
    return nn.Sequential(*layers)


def _csv(values, decimals: int = 4) -> str:
    fmt = f"{{:.{decimals}f}}"
    return ",".join(fmt.format(v) if isinstance(v, (int, float)) else str(v) for v in values)


def _attach_metadata(onnx_path: str, robot_cfg: dict, checkpoint_path: str) -> None:
    metadata: dict[str, str] = {
        "checkpoint_path":   checkpoint_path,
        "base_link_name":    robot_cfg["base_link"],
        "joint_names":       _csv(robot_cfg["joint_names"]),
        "joint_stiffness":   _csv(robot_cfg["stiffness"]),
        "joint_damping":     _csv(robot_cfg["damping"]),
        "default_joint_pos": _csv(robot_cfg["default_joint_pos"]),
        "action_scale":      _csv(robot_cfg["action_scale"]),
    }
    if robot_cfg["link_names"] is not None:
        metadata["link_names"] = _csv(robot_cfg["link_names"])

    model = onnx.load(onnx_path)
    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = v
        model.metadata_props.append(entry)
    onnx.save(model, onnx_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Deep Robotics RSL-RL policy to ONNX without Isaac Sim."
    )
    parser.add_argument("--checkpoint_path", required=True, help="Path to .pt checkpoint file.")
    parser.add_argument(
        "--robot",
        required=True,
        choices=list(ROBOT_CONFIGS.keys()),
        help="Robot type: 'lite3' or 'm20'.",
    )
    parser.add_argument("--output_path", required=True, help="Output .onnx file path.")
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        default=False,
        help="Skip attaching robot metadata to the ONNX file.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    # --- load checkpoint ---
    print(f"[INFO] Loading: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)

    # rsl-rl-lib >=3.1.0 stores actor separately with "mlp." prefix;
    # older versions store everything in "model_state_dict" with "actor." prefix.
    if "actor_state_dict" in ckpt:
        raw_sd = ckpt["actor_state_dict"]
        # Remap "mlp.X.weight/bias" -> "actor.X.weight/bias" so _build_actor works
        sd = {}
        for k, v in raw_sd.items():
            if k.startswith("mlp."):
                sd["actor." + k[len("mlp."):]] = v
            elif k.startswith("distribution."):
                # "distribution.log_std_param" -> "log_std"
                sd["log_std"] = v
            else:
                sd[k] = v
    else:
        sd = ckpt["model_state_dict"]

    obs_dim    = sd["actor.0.weight"].shape[1]
    std_key    = "log_std" if "log_std" in sd else "std"
    action_dim = sd[std_key].shape[0]
    print(f"[INFO] obs_dim={obs_dim}  action_dim={action_dim}")

    # --- build & load actor ---
    actor = _build_actor(sd)
    actor.eval()

    # --- export to ONNX ---
    dummy_obs = torch.zeros(1, obs_dim)
    torch.onnx.export(
        actor,
        dummy_obs,
        args.output_path,
        export_params=True,
        opset_version=11,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(f"[INFO] ONNX saved: {args.output_path}")

    # --- attach metadata ---
    if not args.no_metadata:
        robot_cfg = ROBOT_CONFIGS[args.robot]
        _attach_metadata(args.output_path, robot_cfg, args.checkpoint_path)
        print("[INFO] Robot metadata attached.")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
