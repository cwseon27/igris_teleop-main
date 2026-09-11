"""Source geometry adapters for the existing IGRIS vector retargeter.

OpenXR fingertip positions already follow the project's wrist-local convention;
their historical transform is intentionally kept unchanged. MediaPipe's world
landmarks have metric units, but their axes follow the camera, not that wrist
frame. A translation alone therefore cannot make them equivalent. Recover a
palm-attached frame from the four MCP landmarks and align it with the same
frame in the *actual retargeting hand URDF*. No camera extrinsic is required.

This is rigid orientation alignment, not a hand-size calibration or a correction
for image mirroring/perspective distortion. The upstream anatomical hand label
and metric landmark geometry must be correct. Invalid observations are rejected
so the caller can keep its last valid command instead of commanding an open hand.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from igris_teleop.teleop_devices.unity.constants import (
    grd_yup2grd_zup,
    lefthand2igris,
    righthand2igris,
)


MEDIAPIPE_MCP_INDICES = (5, 9, 13, 17)
MEDIAPIPE_TIP_INDICES = (4, 8, 12, 16, 20)
_MCP_JOINT_NAMES = (
    "3_Joint_Index_Middle",
    "5_Joint_Middle_Middle",
    "7_Joint_Ring_Middle",
    "9_Joint_Little_Middle",
)
_MIN_PALM_AXIS_M = 1.0e-4


def _side(hand_side: str) -> str:
    side = str(hand_side).strip().lower()
    if side not in ("left", "right"):
        raise ValueError("hand_side must be left or right")
    return side


def _points(value, shape: tuple[int, int], name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.shape != shape or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return points


def _palm_basis(wrist: np.ndarray, mcps: np.ndarray) -> np.ndarray:
    """Return orthonormal [radial, normal, distal] column vectors.

    Radial points little MCP -> index MCP; distal points wrist -> MCP centre,
    orthogonalized against radial. normal = distal x radial. In both IGRIS
    URDFs the radial/distal axes are approximately +X/+Z. The resulting frame
    is always a proper rotation; reflection of a left hand into a right hand
    reverses normal displacement, retaining their opposite flexion directions.
    """
    radial = mcps[0] - mcps[3]
    radial_length = float(np.linalg.norm(radial))
    if radial_length < _MIN_PALM_AXIS_M:
        raise ValueError("degenerate palm: index and little MCP coincide")
    radial = radial / radial_length
    distal = np.mean(mcps, axis=0) - wrist
    distal = distal - radial * np.dot(distal, radial)
    distal_length = float(np.linalg.norm(distal))
    if distal_length < _MIN_PALM_AXIS_M:
        raise ValueError("degenerate palm: wrist and MCPs are collinear")
    distal = distal / distal_length
    normal = np.cross(distal, radial)
    return np.column_stack((radial, normal, distal))


@lru_cache(maxsize=2)
def _robot_palm_basis(hand_side: str) -> np.ndarray:
    """Use base-frame neutral MCP origins, not guessed per-hand axis signs."""
    path = Path(__file__).parent / "hand_urdf" / f"{hand_side}_hand_igris_c.urdf"
    robot = ET.parse(path).getroot()
    origins = []
    for name in _MCP_JOINT_NAMES:
        joint = robot.find(f"joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"missing retargeting palm joint {name} in {path}")
        parent, origin = joint.find("parent"), joint.find("origin")
        if parent is None or parent.get("link") != "base_link" or origin is None:
            raise ValueError(f"{name} must have an origin directly in base_link")
        xyz = np.asarray([float(value) for value in origin.get("xyz", "").split()])
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError(f"invalid retargeting palm origin for {name}")
        origins.append(xyz)
    result = _palm_basis(np.zeros(3), np.asarray(origins))
    result.setflags(write=False)
    return result


def openxr_fingertips_to_retarget_reference(fingertips, hand_side: str) -> np.ndarray:
    """Convert five existing VR wrist-local tips exactly as the Unity bridge."""
    side = _side(hand_side)
    points = _points(fingertips, (5, 3), "OpenXR fingertips")
    if np.allclose(points, 0.0):
        raise ValueError("OpenXR fingertips are all zero")
    hand_basis = lefthand2igris if side == "left" else righthand2igris
    transform = hand_basis.T @ grd_yup2grd_zup
    # Both historical matrices have zero translation. Use homogeneous points
    # nonetheless to preserve the existing bridge expression exactly.
    homogeneous = np.vstack((points.T, np.ones((1, 5))))
    return (transform @ homogeneous)[:3].T.copy()


def mediapipe_landmarks_to_retarget_reference(landmarks, hand_side: str) -> np.ndarray:
    """Convert 21 metric MediaPipe landmarks into five URDF-base fingertip vectors.

    Wrist subtraction is safe whether upstream already made the landmarks
    wrist-relative or supplied MediaPipe's hand-centred world coordinates.
    No lengths are changed and a rigid camera rotation/translation does not
    change the reference passed to the retargeter.
    """
    side = _side(hand_side)
    points = _points(landmarks, (21, 3), "MediaPipe landmarks")
    wrist = points[0]
    source_basis = _palm_basis(wrist, points[list(MEDIAPIPE_MCP_INDICES)])
    target_basis = _robot_palm_basis(side)
    tips = points[list(MEDIAPIPE_TIP_INDICES)] - wrist
    if np.any(np.linalg.norm(tips, axis=1) < _MIN_PALM_AXIS_M):
        raise ValueError("degenerate hand: fingertip coincides with wrist")
    return tips @ source_basis @ target_basis.T
