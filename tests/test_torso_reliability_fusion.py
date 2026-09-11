from __future__ import annotations

import math

import numpy as np
import pytest

from igris_teleop.teleop_devices.unity.reliability_fusion import (
    DEFAULT_CONTROLLER_CONFIDENCE_FALLBACK,
    DEFAULT_CONTROLLER_TRACKED_CONFIDENCE_FLOOR,
    DEFAULT_CONTROLLER_UNTRACKED_POSE_CONFIDENCE_FLOOR,
    ReliabilityAwareTorsoFusion,
    select_tracking_confidence,
    transform_from_xyz_xyzw,
)


def _pose(x: float = 0.0, yaw_rad: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array(
        [
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pose[0, 3] = x
    return pose


def _pose_xyz_quat(position, quaternion_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    pose[:3, 3] = np.asarray(position, dtype=np.float64)
    return pose


def _fusion(**overrides) -> ReliabilityAwareTorsoFusion:
    values = {
        "left_to_chest": transform_from_xyz_xyzw([0.16, 0, 0, 0, 0, 0, 1]),
        "right_to_chest": transform_from_xyz_xyzw([-0.16, 0, 0, 0, 0, 0, 1]),
        "alpha_rate_up_per_sec": 1000.0,
        "alpha_rate_down_per_sec": 1000.0,
        "nominal_rate_hz": 60.0,
    }
    values.update(overrides)
    return ReliabilityAwareTorsoFusion(**values)


def test_local_controller_offsets_recover_shared_chest_target() -> None:
    result = _fusion().update(
        left_controller=_pose(-0.16),
        right_controller=_pose(0.16),
        left_confidence=1.0,
        right_confidence=1.0,
        now=1.0,
    )

    assert result.ready is True
    assert result.alpha == pytest.approx(1.0)
    assert result.consistency_gate == pytest.approx(1.0)
    assert result.target is not None
    np.testing.assert_allclose(result.target[:3, 3], np.zeros(3), atol=1e-10)


def test_calibrated_offsets_accept_recorded_forward_facing_controller_pose() -> None:
    fusion = _fusion(
        left_to_chest=transform_from_xyz_xyzw([0.064, 0, 0, 0, 0, 0, 1]),
        right_to_chest=transform_from_xyz_xyzw([-0.064, 0, 0, 0, 0, 0, 1]),
    )
    result = fusion.update(
        left_controller=_pose_xyz_quat(
            [-0.057996, 0.995476, -0.147064],
            [-0.129713, 0.075605, -0.002039, 0.988663],
        ),
        right_controller=_pose_xyz_quat(
            [0.069120, 0.996742, -0.157426],
            [-0.123964, 0.001226, 0.006322, 0.992266],
        ),
        left_confidence=1.0,
        right_confidence=1.0,
        now=1.0,
    )

    assert result.ready is True
    assert result.alpha > 0.9
    assert result.position_disagreement_m < 0.003
    assert math.degrees(result.rotation_disagreement_rad) < 9.0


def test_inconsistent_confident_controllers_suppress_torso_task() -> None:
    result = _fusion(position_sigma_m=0.1).update(
        left_controller=_pose(-0.16),
        right_controller=_pose(1.16),
        left_confidence=1.0,
        right_confidence=1.0,
        now=1.0,
    )

    assert result.position_disagreement_m == pytest.approx(1.0)
    assert result.consistency_gate < 1e-10
    assert result.raw_alpha < 1e-10
    assert result.ready is False
    assert result.target is None


def test_alpha_has_asymmetric_rate_limits_and_stale_target_is_held() -> None:
    fusion = _fusion(
        alpha_rate_up_per_sec=1.0,
        alpha_rate_down_per_sec=5.0,
        nominal_rate_hz=10.0,
    )
    first = fusion.update(
        left_controller=_pose(),
        right_controller=None,
        left_confidence=1.0,
        right_confidence=0.0,
        now=1.0,
    )
    second = fusion.update(
        left_controller=_pose(),
        right_controller=None,
        left_confidence=1.0,
        right_confidence=0.0,
        now=1.1,
    )
    stale = fusion.update(
        left_controller=None,
        right_controller=None,
        left_confidence=1.0,
        right_confidence=1.0,
        now=1.2,
    )

    assert first.alpha == pytest.approx(0.1)
    assert second.alpha == pytest.approx(0.2)
    assert stale.alpha == pytest.approx(0.0)
    assert stale.ready is False
    assert stale.target is not None


def test_default_alpha_reaches_reliable_chest_in_under_200_ms() -> None:
    fusion = ReliabilityAwareTorsoFusion(
        left_to_chest=transform_from_xyz_xyzw([0.16, 0, 0, 0, 0, 0, 1]),
        right_to_chest=transform_from_xyz_xyzw([-0.16, 0, 0, 0, 0, 0, 1]),
        nominal_rate_hz=60.0,
    )

    result = None
    for frame in range(10):
        result = fusion.update(
            left_controller=_pose(-0.16),
            right_controller=_pose(0.16),
            left_confidence=1.0,
            right_confidence=1.0,
            now=1.0 + frame / 60.0,
        )

    assert result is not None
    assert result.alpha == pytest.approx(1.0)


def test_single_reliable_controller_can_fully_activate_task() -> None:
    result = _fusion().update(
        left_controller=_pose(),
        right_controller=_pose(),
        left_confidence=1.0,
        right_confidence=0.0,
        now=1.0,
    )
    assert result.raw_alpha == pytest.approx(1.0)


def test_zero_confidence_pose_cannot_move_held_target_or_become_ready() -> None:
    fusion = _fusion(minimum_confidence_sum=0.05)
    reliable = fusion.update(
        left_controller=_pose(-0.16),
        right_controller=None,
        left_confidence=1.0,
        right_confidence=0.0,
        now=1.0,
    )
    unreliable = fusion.update(
        left_controller=_pose(2.0),
        right_controller=_pose(3.0),
        left_confidence=0.0,
        right_confidence=0.0,
        now=1.1,
    )

    assert reliable.target is not None
    assert unreliable.target is not None
    assert unreliable.ready is False
    assert unreliable.raw_alpha == pytest.approx(0.0)
    np.testing.assert_allclose(unreliable.target, reliable.target)


def test_tracked_fresh_pose_gets_floor_when_policy_returns_zero() -> None:
    confidence = select_tracking_confidence(
        pose_fresh=True,
        tracked=True,
        confidence_fresh=True,
        confidence=0.0,
        fallback=1.0,
        tracked_floor=0.25,
        untracked_pose_floor=0.10,
    )

    assert confidence == pytest.approx(0.25)


def test_default_controller_confidence_requires_reliability_publisher() -> None:
    without_policy = select_tracking_confidence(
        pose_fresh=True,
        tracked=True,
        confidence_fresh=False,
        confidence=0.0,
        fallback=DEFAULT_CONTROLLER_CONFIDENCE_FALLBACK,
        tracked_floor=DEFAULT_CONTROLLER_TRACKED_CONFIDENCE_FLOOR,
        untracked_pose_floor=DEFAULT_CONTROLLER_UNTRACKED_POSE_CONFIDENCE_FLOOR,
    )
    with_policy = select_tracking_confidence(
        pose_fresh=True,
        tracked=False,
        confidence_fresh=True,
        confidence=0.8,
        fallback=DEFAULT_CONTROLLER_CONFIDENCE_FALLBACK,
        tracked_floor=DEFAULT_CONTROLLER_TRACKED_CONFIDENCE_FLOOR,
        untracked_pose_floor=DEFAULT_CONTROLLER_UNTRACKED_POSE_CONFIDENCE_FLOOR,
    )

    assert without_policy == 0.0
    assert with_policy == pytest.approx(0.8)


def test_untracked_fresh_pose_gets_lower_floor_but_stale_pose_is_rejected() -> None:
    common = {
        "confidence_fresh": True,
        "confidence": 1.0,
        "fallback": 1.0,
        "tracked_floor": 0.25,
        "untracked_pose_floor": 0.10,
    }

    assert select_tracking_confidence(pose_fresh=True, tracked=False, **common) == 1.0
    common["confidence"] = 0.0
    assert select_tracking_confidence(pose_fresh=True, tracked=False, **common) == pytest.approx(0.10)
    assert select_tracking_confidence(pose_fresh=False, tracked=True, **common) == 0.0
