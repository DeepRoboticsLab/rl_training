# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for 3D pose conversion."""
import math
import numpy as np


def _quaternion_multiply(q1, q2):
    """Multiply two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)


def _quaternion_inverse(q):
    """Return the inverse of a quaternion [x, y, z, w]."""
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)
    return q_conj / np.dot(q, q)


VECTOR3_0 = np.zeros(3, dtype=np.float64)
VECTOR3_1 = np.ones(3, dtype=np.float64)
VECTOR3_X = np.array([1, 0, 0], dtype=np.float64)
VECTOR3_Y = np.array([0, 1, 0], dtype=np.float64)
VECTOR3_Z = np.array([0, 0, 1], dtype=np.float64)

QUATERNION_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def Vector3RandomNormal(sigma, mu=VECTOR3_0):
    random_v3 = np.random.normal(scale=sigma, size=3) + mu
    return random_v3


def Vector3RandomUniform(low=VECTOR3_0, high=VECTOR3_1):
    random_x = np.random.uniform(low=low[0], high=high[0])
    random_y = np.random.uniform(low=low[1], high=high[1])
    random_z = np.random.uniform(low=low[2], high=high[2])
    return np.array([random_x, random_y, random_z])


def Vector3RandomUnit():
    longitude = np.random.uniform(low=-math.pi, high=math.pi)
    sin_latitude = np.random.uniform(low=-1.0, high=1.0)
    cos_latitude = math.sqrt(1.0 - sin_latitude * sin_latitude)
    x = math.cos(longitude) * cos_latitude
    y = math.sin(longitude) * cos_latitude
    z = sin_latitude
    return np.array([x, y, z], dtype=np.float64)


def QuaternionNormalize(q):
    """Normalizes the quaternion to length 1."""
    q_norm = np.linalg.norm(q)
    if np.isclose(q_norm, 0.0):
        raise ValueError(
            'Quaternion may not be zero in QuaternionNormalize: |q| = %f, q = %s' %
            (q_norm, q))
    return q / q_norm


def QuaternionFromAxisAngle(axis, angle):
    """Returns a quaternion that generates the given axis-angle rotation."""
    if len(axis) != 3:
        raise ValueError('Axis vector should have three components: %s' % axis)
    axis_norm = np.linalg.norm(axis)
    if np.isclose(axis_norm, 0.0):
        raise ValueError('Axis vector may not have zero length: |v| = %f, v = %s' %
                         (axis_norm, axis))
    half_angle = angle * 0.5
    q = np.zeros(4, dtype=np.float64)
    q[0:3] = axis
    q[0:3] *= math.sin(half_angle) / axis_norm
    q[3] = math.cos(half_angle)
    return q


def QuaternionToAxisAngle(quat, default_axis=VECTOR3_Z, direction_axis=None):
    """Calculates axis and angle of rotation performed by a quaternion."""
    if len(quat) != 4:
        raise ValueError(
            'Quaternion should have four components [x, y, z, w]: %s' % quat)
    if not np.isclose(1.0, np.linalg.norm(quat)):
        raise ValueError('Quaternion should have unit length: |q| = %f, q = %s' %
                         (np.linalg.norm(quat), quat))
    axis = quat[:3].copy()
    axis_norm = np.linalg.norm(axis)
    min_axis_norm = 1e-8
    if axis_norm < min_axis_norm:
        axis = default_axis
        if len(default_axis) != 3:
            raise ValueError('Axis vector should have three components: %s' % axis)
        if not np.isclose(np.linalg.norm(axis), 1.0):
            raise ValueError('Axis vector should have unit length: |v| = %f, v = %s' %
                             (np.linalg.norm(axis), axis))
    else:
        axis /= axis_norm
    sin_half_angle = axis_norm
    if direction_axis is not None and np.inner(axis, direction_axis) < 0:
        sin_half_angle = -sin_half_angle
        axis = -axis
    cos_half_angle = quat[3]
    half_angle = math.atan2(sin_half_angle, cos_half_angle)
    angle = half_angle * 2
    return axis, angle


def QuaternionRandomRotation(max_angle=math.pi):
    angle = np.random.uniform(low=0, high=max_angle)
    axis = Vector3RandomUnit()
    return QuaternionFromAxisAngle(axis, angle)


def QuaternionRotatePoint(point, quat):
    """Performs a rotation by quaternion."""
    q_point = np.array([point[0], point[1], point[2], 0.0])
    quat_inverse = _quaternion_inverse(quat)
    q_point_rotated = _quaternion_multiply(
        _quaternion_multiply(quat, q_point), quat_inverse)
    return q_point_rotated[:3]


def IsRotationMatrix(m):
    """Returns true if the 3x3 submatrix represents a rotation."""
    if len(m.shape) != 2 or m.shape[0] < 3 or m.shape[1] < 3:
        raise ValueError('Matrix should be 3x3 or 4x4: %s\n %s' % (m.shape, m))
    rot = m[:3, :3]
    eye = np.matmul(rot, np.transpose(rot))
    return np.isclose(eye, np.identity(3), atol=1e-4).all()
