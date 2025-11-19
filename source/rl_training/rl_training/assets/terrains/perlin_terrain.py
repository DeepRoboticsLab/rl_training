# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate height fields for perlin noise terrain."""

from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import perlin_terrain_cfg

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    # print("shape:", shape, "res:", res)
    delta = (res[0] / shape[0], res[1] / shape[1]) # 0.125 0.1
    d = (shape[0] // res[0], shape[1] // res[1])    # 8 10
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    # gradients = np.random.standard_normal((2, res[0], res[1]))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1) * 0.5 + 0.5

def generate_fractal_noise_2d(xSize=20, ySize=20, xSamples=1600, ySamples=1600, \
    frequency=10, fractalOctaves=2, fractalLacunarity = 2.0, fractalGain=0.25, zScale = 0.23):
    xScale = int(frequency * xSize)
    yScale = int(frequency * ySize)
    amplitude = 1
    shape = (xSamples, ySamples)
    noise = np.zeros(shape)
    for _ in range(fractalOctaves):
        noise += amplitude * generate_perlin_noise_2d((xSamples, ySamples), (xScale, yScale)) * zScale
        amplitude *= fractalGain
        xScale, yScale = int(fractalLacunarity * xScale), int(fractalLacunarity * yScale)

    return noise

@height_field_to_mesh
def perlin_terrain(
    difficulty: float, cfg: perlin_terrain_cfg.HfPerlinTerrainCfg
) -> np.ndarray:
    """Generate a Perlin noise terrain.

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the downsampled scale is smaller than the horizontal scale.
    """
    """
    xSize must be an integer, otherwise it will cause an error of inconsistent array size.
     Therefore, cfg.size - (2* border_width + horizontal) must be set as an integer in cfg.
     It will be corrected in the future
    """
    # print("cfg.size[0]:", cfg.size[0], "cfg.size[1]:", cfg.size[1], "cfg.horizontal_scale:", cfg.horizontal_scale)

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_scale = int(cfg.z_scale / cfg.vertical_scale)
    # print("width_pixels:", width_pixels, "length_pixels:", length_pixels)

    # hf_raw =np.zeros((width_pixels, length_pixels))
    hf_noise = generate_fractal_noise_2d(xSize=int(cfg.size[0]), ySize=int(cfg.size[1]),xSamples=width_pixels, \
                ySamples=length_pixels, frequency=int(cfg.frequency), fractalOctaves=cfg.fractal_octaves, \
                fractalLacunarity=cfg.fractal_lacunarity, fractalGain=cfg.fractal_gain, zScale=height_scale)

    # hf_raw += hf_noise
    heightmap = np.rint(hf_noise).astype(np.int16)
    return heightmap
