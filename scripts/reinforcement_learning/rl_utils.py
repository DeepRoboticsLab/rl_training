# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import torch

import isaaclab.utils.math as math_utils


def camera_follow(env):
    if not hasattr(camera_follow, "smooth_camera_positions"):
        camera_follow.smooth_camera_positions = []
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]
    camera_offset = torch.tensor([-3.0, 0.0, 0.5], dtype=torch.float32, device=env.device)
    camera_pos = math_utils.transform_points(
        camera_offset.unsqueeze(0), pos=robot_pos.unsqueeze(0), quat=robot_quat.unsqueeze(0)
    ).squeeze(0)
    # camera_pos[2] = torch.clamp(camera_pos[2], min=0.1)
    window_size = 50
    camera_follow.smooth_camera_positions.append(camera_pos)
    if len(camera_follow.smooth_camera_positions) > window_size:
        camera_follow.smooth_camera_positions.pop(0)
    smooth_camera_pos = torch.mean(torch.stack(camera_follow.smooth_camera_positions), dim=0)
    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.cpu().numpy(), lookat=robot_pos.cpu().numpy()
    )




import os
import pickle
from typing import Any

def reindex_feet(vec):
    return vec[:, [1, 0, 3, 2]]

def reindex(vec):
    return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]


def load_pickle(filename: str) -> Any:
    """Loads an input PKL file safely.

    Args:
        filename: The path to pickled file.

    Raises:
        FileNotFoundError: When the specified file does not exist.

    Returns:
        The data read from the input file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def dump_pickle(filename: str, data: Any):
    """Saves data into a pickle file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save.
    """
    # check ending
    if not filename.endswith("pkl"):
        filename += ".pkl"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # save data
    with open(filename, "wb") as f:
        pickle.dump(data, f)
