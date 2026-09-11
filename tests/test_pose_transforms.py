from __future__ import annotations

import math

import numpy as np

from igris_teleop.core.math.transforms import (
    apply_common_frame_relative_pose,
    common_frame_relative_pose,
)


def _rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_common_frame_mapping_cancels_local_axis_offset() -> None:
    source_home = np.eye(4, dtype=np.float64)
    source_home[:3, :3] = _rotation_x(70.0)
    source_home[:3, 3] = [1.0, 2.0, 3.0]

    world_motion = _rotation_z(20.0)
    source_now = source_home.copy()
    source_now[:3, :3] = world_motion @ source_home[:3, :3]
    source_now[:3, 3] += [0.03, -0.02, 0.01]

    robot_anchor = np.eye(4, dtype=np.float64)
    robot_anchor[:3, :3] = _rotation_y(-35.0)
    robot_anchor[:3, 3] = [0.2, -0.1, 1.4]

    relative = common_frame_relative_pose(source_now, source_home)
    target = apply_common_frame_relative_pose(relative, robot_anchor)

    np.testing.assert_allclose(relative[:3, :3], world_motion, atol=1e-9)
    np.testing.assert_allclose(target[:3, :3], world_motion @ robot_anchor[:3, :3], atol=1e-9)
    np.testing.assert_allclose(target[:3, 3], [0.23, -0.12, 1.41], atol=1e-9)
