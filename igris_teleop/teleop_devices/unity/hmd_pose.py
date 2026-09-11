from __future__ import annotations

import numpy as np

from .constants import T_robot_openxr


_T_OPENXR_ROBOT = np.linalg.inv(
    np.asarray(T_robot_openxr, dtype=np.float64)
)


def openxr_hmd_pose_to_robot(openxr_pose: np.ndarray) -> np.ndarray:
    """Express an OpenXR HMD pose in the robot coordinate basis.

    OpenXR +x/right, +y/up, -z/forward map to robot -y/right, +z/up,
    +x/forward respectively. This is a basis change, not a sign correction
    applied after HOME-relative pose calculation.
    """
    pose = np.asarray(openxr_pose, dtype=np.float64).reshape(4, 4)
    return (
        np.asarray(T_robot_openxr, dtype=np.float64)
        @ pose
        @ _T_OPENXR_ROBOT
    )
