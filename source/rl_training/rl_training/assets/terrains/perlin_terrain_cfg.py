# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import warnings
from dataclasses import MISSING
# from typing import Literal

# import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
import rl_training.assets.terrains.perlin_terrain as perlin_terrain
# import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field import hf_terrains

"""
Different trimesh terrain configurations.
"""

######################
@configclass
class HfPerlinTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a Perlin noise height field terrain."""

    function = perlin_terrain.perlin_terrain

    frequency: float = 10.0
    """Frequency of the Perlin noise pattern."""

    fractal_octaves: int = 2
    """Number of noise octaves used in fractal Perlin noise generation."""

    fractal_lacunarity: float = 2.0
    """Lacunarity value used in fractal Perlin noise generation."""

    fractal_gain: float = 0.25
    """Gain value used in fractal Perlin noise generation."""

    z_scale: float = 0.23
    """Vertical scaling factor for the terrain elevation."""
