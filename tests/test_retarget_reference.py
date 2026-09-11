from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from igris_teleop.hand_control import retarget_reference as adapter
from igris_teleop.teleop_devices.unity.constants import (
    grd_yup2grd_zup,
    lefthand2igris,
    righthand2igris,
)


def robot_shaped_landmarks(side: str) -> np.ndarray:
    """Synthetic anatomical landmarks with real URDF neutral MCP positions."""
    path = Path(adapter.__file__).parent / "hand_urdf" / f"{side}_hand_igris_c.urdf"
    root = ET.parse(path).getroot()
    points = np.zeros((21, 3))
    for index, joint_name in zip(adapter.MEDIAPIPE_MCP_INDICES, adapter._MCP_JOINT_NAMES):
        element = root.find(f"joint[@name='{joint_name}']/origin")
        points[index] = np.fromstring(element.get("xyz"), sep=" ")
        points[index + 1] = points[index] + (0.0, 0.0, 0.025)
        points[index + 2] = points[index] + (0.0, 0.0, 0.05)
        points[index + 3] = points[index] + (0.0, 0.0, 0.075)
    points[1:5] = ((0.025, 0, 0.03), (0.04, 0, 0.04), (0.06, 0, 0.05), (0.075, 0, 0.06))
    return points


@pytest.mark.parametrize("side,hand_basis", [("left", lefthand2igris), ("right", righthand2igris)])
def test_vr_geometry_is_exact_existing_transform(side, hand_basis):
    tips = np.random.default_rng(3).normal(size=(5, 3)) * 0.1
    homogeneous = np.vstack((tips.T, np.ones((1, 5))))
    expected = (hand_basis.T @ (grd_yup2grd_zup @ homogeneous))[:3].T
    assert np.array_equal(adapter.openxr_fingertips_to_retarget_reference(tips, side), expected)


@pytest.mark.parametrize("side", ["left", "right"])
def test_robot_palm_geometry_aligns_with_urdf_base_not_guessed_axes(side):
    points = robot_shaped_landmarks(side)
    expected = points[list(adapter.MEDIAPIPE_TIP_INDICES)]
    actual = adapter.mediapipe_landmarks_to_retarget_reference(points, side)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    basis = adapter._robot_palm_basis(side)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-14)
    assert np.linalg.det(basis) == pytest.approx(1.0)
    assert basis[0, 0] > 0.99  # index is at +X, little is at -X in both URDFs
    assert basis[2, 2] > 0.99  # MCPs extend along +Z in both URDFs


@pytest.mark.parametrize("side", ["left", "right"])
def test_mediapipe_reference_is_rigid_camera_pose_invariant_and_metric(side):
    points = robot_shaped_landmarks(side)
    points[list(adapter.MEDIAPIPE_TIP_INDICES), 1] += 0.015 if side == "left" else -0.015
    expected = adapter.mediapipe_landmarks_to_retarget_reference(points, side)
    rng = np.random.default_rng(91)
    for _ in range(20):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotation[:, 0] *= np.linalg.det(rotation)
        shifted = points @ rotation.T + rng.normal(size=3)
        actual = adapter.mediapipe_landmarks_to_retarget_reference(shifted, side)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), np.linalg.norm(expected, axis=1), atol=1e-14)


def test_left_right_mirror_keeps_urdf_opposite_flexion_directions():
    left = robot_shaped_landmarks("left")
    left[list(adapter.MEDIAPIPE_TIP_INDICES), 1] += 0.04
    right = left * (1.0, -1.0, 1.0)
    left_ref = adapter.mediapipe_landmarks_to_retarget_reference(left, "left")
    right_ref = adapter.mediapipe_landmarks_to_retarget_reference(right, "right")
    np.testing.assert_allclose(right_ref, left_ref * (1.0, -1.0, 1.0), atol=1e-14)
    assert np.all(left_ref[:, 1] > 0.0)
    assert np.all(right_ref[:, 1] < 0.0)


@pytest.mark.parametrize("kind", ["zero", "collinear", "nan", "inf", "tip_at_wrist", "wrong_count"])
def test_invalid_mediapipe_geometry_is_rejected_not_converted_to_open_hand(kind):
    points = robot_shaped_landmarks("left")
    if kind == "zero":
        points[:] = 0.0
    elif kind == "collinear":
        points[list(adapter.MEDIAPIPE_MCP_INDICES)] = np.arange(4)[:, None] * np.array([0.01, 0.02, 0.03])
    elif kind == "nan":
        points[9, 0] = np.nan
    elif kind == "inf":
        points[7, 2] = np.inf
    elif kind == "tip_at_wrist":
        points[4] = points[0]
    elif kind == "wrong_count":
        points = points[:6]
    with pytest.raises(ValueError):
        adapter.mediapipe_landmarks_to_retarget_reference(points, "left")


@pytest.mark.parametrize("bad", [np.zeros((5, 3)), np.full((5, 3), np.nan), np.zeros((6, 3))])
def test_invalid_vr_geometry_is_rejected(bad):
    with pytest.raises(ValueError):
        adapter.openxr_fingertips_to_retarget_reference(bad, "right")


@pytest.mark.parametrize("function,points", [
    (adapter.openxr_fingertips_to_retarget_reference, np.ones((5, 3))),
    (adapter.mediapipe_landmarks_to_retarget_reference, np.ones((21, 3))),
])
def test_unknown_side_is_rejected(function, points):
    with pytest.raises(ValueError, match="hand_side"):
        function(points, "unknown")
