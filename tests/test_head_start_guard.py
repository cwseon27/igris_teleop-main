from __future__ import annotations

import math

import numpy as np

from igris_teleop.head_start_guard import (
    HEAD_REASON_OK,
    HEAD_REASON_POS_AND_ROT_EXCEEDED,
    HEAD_REASON_POS_EXCEEDED,
    HEAD_REASON_ROT_EXCEEDED,
    HEAD_REASON_WAITING_HOME_REFERENCE,
    build_candidate_head_target,
    evaluate_head_start_guard,
    format_head_guard_message,
    is_valid_pose_matrix,
    rotation_error_deg,
    signed_yaw_error_deg,
)


def _pose(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    out[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return out


def test_is_valid_pose_matrix_accepts_homogeneous_pose() -> None:
    assert is_valid_pose_matrix(_pose())


def test_is_valid_pose_matrix_rejects_zero_matrix() -> None:
    assert not is_valid_pose_matrix(np.zeros((4, 4), dtype=np.float64))


def test_rotation_error_deg_matches_geodesic_angle() -> None:
    err = rotation_error_deg(_pose(yaw_deg=20.0)[:3, :3], _pose(yaw_deg=0.0)[:3, :3])
    assert np.isclose(err, 20.0, atol=1e-6)


def test_signed_yaw_error_deg_is_positive_for_left_turn() -> None:
    err = signed_yaw_error_deg(_pose(yaw_deg=20.0)[:3, :3], _pose(yaw_deg=0.0)[:3, :3])
    assert np.isclose(err, 20.0, atol=1e-6)


def test_build_candidate_head_target_uses_robot_home_translation_offset() -> None:
    candidate = build_candidate_head_target(
        _pose(tx=0.03, ty=-0.02, tz=0.01, yaw_deg=10.0),
        _pose(),
        _pose(tx=1.0, ty=2.0, tz=3.0, yaw_deg=0.0),
    )
    assert np.allclose(candidate[:3, 3], np.array([1.03, 1.98, 3.01], dtype=np.float64))
    assert np.allclose(candidate[:3, :3], _pose(yaw_deg=10.0)[:3, :3])


def test_build_candidate_head_target_cancels_static_vr_robot_axis_offset() -> None:
    vr_home = _pose(tx=1.0, ty=2.0, tz=3.0, yaw_deg=60.0)
    robot_home = _pose(tx=0.2, ty=-0.1, tz=1.4, yaw_deg=-25.0)

    at_home = build_candidate_head_target(vr_home, vr_home, robot_home)
    np.testing.assert_allclose(at_home, robot_home, atol=1e-9)

    moved = _pose(tx=1.03, ty=1.98, tz=3.01, yaw_deg=70.0)
    candidate = build_candidate_head_target(moved, vr_home, robot_home)
    np.testing.assert_allclose(candidate[:3, 3], [0.23, -0.12, 1.41], atol=1e-9)
    np.testing.assert_allclose(candidate[:3, :3], _pose(yaw_deg=-15.0)[:3, :3], atol=1e-9)


def test_head_guard_is_zero_at_home_even_when_absolute_axes_differ() -> None:
    vr_home = _pose(tx=0.3, ty=-0.4, tz=1.6, yaw_deg=95.0)
    status = evaluate_head_start_guard(
        vr_home,
        vr_home,
        _pose(tx=0.1, ty=0.0, tz=1.3, yaw_deg=-35.0),
    )

    assert status["head_guard_ok"] == 1.0
    assert np.isclose(status["head_pos_err_m"], 0.0, atol=1e-12)
    assert np.isclose(status["head_rot_err_deg"], 0.0, atol=1e-9)


def test_evaluate_head_start_guard_passes_inside_thresholds() -> None:
    status = evaluate_head_start_guard(
        _pose(tx=0.049, yaw_deg=14.9),
        _pose(),
        _pose(),
    )
    assert status["head_guard_ready"] == 1.0
    assert status["head_guard_ok"] == 1.0
    assert status["head_reason_code"] == float(HEAD_REASON_OK)


def test_evaluate_head_start_guard_blocks_when_home_reference_missing() -> None:
    status = evaluate_head_start_guard(_pose(), None, _pose())
    assert status["head_guard_ready"] == 0.0
    assert status["head_guard_ok"] == 0.0
    assert status["head_reason_code"] == float(HEAD_REASON_WAITING_HOME_REFERENCE)


def test_evaluate_head_start_guard_sets_pos_reason() -> None:
    status = evaluate_head_start_guard(
        _pose(tx=0.051, yaw_deg=0.0),
        _pose(),
        _pose(),
    )
    assert status["head_guard_ok"] == 0.0
    assert status["head_reason_code"] == float(HEAD_REASON_POS_EXCEEDED)
    assert status["head_pos_err_m"] > 0.05


def test_evaluate_head_start_guard_sets_rot_reason() -> None:
    status = evaluate_head_start_guard(
        _pose(tx=0.0, yaw_deg=15.1),
        _pose(),
        _pose(),
    )
    assert status["head_guard_ok"] == 0.0
    assert status["head_reason_code"] == float(HEAD_REASON_ROT_EXCEEDED)
    assert status["head_rot_err_deg"] > 15.0
    assert status["head_yaw_correction_deg"] < 0.0


def test_format_head_guard_message_includes_turn_direction() -> None:
    status = evaluate_head_start_guard(
        _pose(tx=0.0, yaw_deg=20.0),
        _pose(),
        _pose(),
    )
    msg = format_head_guard_message(status)
    assert "turn right 20.0 deg" in msg


def test_evaluate_head_start_guard_sets_combined_reason() -> None:
    status = evaluate_head_start_guard(
        _pose(tx=0.051, yaw_deg=15.1),
        _pose(),
        _pose(),
    )
    assert status["head_guard_ok"] == 0.0
    assert status["head_reason_code"] == float(HEAD_REASON_POS_AND_ROT_EXCEEDED)
