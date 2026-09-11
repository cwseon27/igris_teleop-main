from __future__ import annotations

import math

import numpy as np
import pytest

from igris_teleop.teleop_devices.unity.hmd_pose import (
    openxr_hmd_pose_to_robot,
)


def _axis_angle_rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    unit_axis = np.asarray(axis, dtype=np.float64)
    unit_axis /= np.linalg.norm(unit_axis)
    x, y, z = unit_axis
    angle = math.radians(degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ],
        ],
        dtype=np.float64,
    )


def _pose(rotation: np.ndarray, translation=None) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    if translation is not None:
        pose[:3, 3] = translation
    return pose


@pytest.mark.parametrize(
    ("openxr_translation", "robot_translation"),
    [
        ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, -1.0], [1.0, 0.0, 0.0]),
    ],
    ids=["right", "up", "forward"],
)
def test_openxr_hmd_translation_basis(
    openxr_translation: list[float],
    robot_translation: list[float],
) -> None:
    converted = openxr_hmd_pose_to_robot(
        _pose(np.eye(3), openxr_translation)
    )

    np.testing.assert_allclose(converted[:3, 3], robot_translation, atol=1e-12)


@pytest.mark.parametrize(
    ("source_rotation", "expected_rotation"),
    [
        (
            _axis_angle_rotation([1.0, 0.0, 0.0], -18.0),
            _axis_angle_rotation([0.0, 1.0, 0.0], 18.0),
        ),
        (
            _axis_angle_rotation([0.0, 0.0, -1.0], 18.0),
            _axis_angle_rotation([1.0, 0.0, 0.0], 18.0),
        ),
        (
            _axis_angle_rotation([0.0, 1.0, 0.0], 18.0),
            _axis_angle_rotation([0.0, 0.0, 1.0], 18.0),
        ),
    ],
    ids=["pitch-down", "roll-about-forward", "yaw-left"],
)
def test_hmd_axis_directions(
    source_rotation: np.ndarray,
    expected_rotation: np.ndarray,
) -> None:
    converted = openxr_hmd_pose_to_robot(_pose(source_rotation))

    np.testing.assert_allclose(
        converted[:3, :3],
        expected_rotation,
        atol=1e-12,
    )


def test_observed_floor_facing_hmd_maps_robot_forward_down() -> None:
    # Live /hmd/pose sample captured while the headset faced the floor.
    qx, qy, qz, qw = (
        -0.6433801651,
        -0.0115136802,
        -0.00586412847,
        0.7654378414,
    )
    source_rotation = _axis_angle_rotation(
        [qx, qy, qz],
        math.degrees(2.0 * math.acos(qw)),
    )

    converted = openxr_hmd_pose_to_robot(_pose(source_rotation))
    robot_forward = converted[:3, :3] @ np.array([1.0, 0.0, 0.0])

    assert robot_forward[2] < -0.98
