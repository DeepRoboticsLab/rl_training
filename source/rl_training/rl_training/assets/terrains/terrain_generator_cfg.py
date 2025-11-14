# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

# import isaaclab.terrains as terrain_gen
import rl_training.assets.terrains as terrain_gen

# from ..terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

PERLIN_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(9.05, 9.05),
    border_width=1.0,
    num_rows=2,
    num_cols=2,
    horizontal_scale=0.05,
    # vertical_scale=0.05,
    # slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "perlin": terrain_gen.MeshPerlinTerrainCfg(
            border_width=0.5,
            z_scale=1.0,
        ),
    },
)
"""Rough terrains configuration."""
