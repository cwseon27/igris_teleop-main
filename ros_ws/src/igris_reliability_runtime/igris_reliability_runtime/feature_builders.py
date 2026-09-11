from __future__ import annotations

import numpy as np

from train_lib.pose_utils import (
    canonicalize_quaternions,
    quaternion_angular_velocity,
    split_pose_flat,
    validate_vector,
)


FEATURE_MODE_CONTROLLER_MOTION = "controller_motion_tracking_v1"


def controller_motion_features(
    pose_flat,
    dt: float | None,
    previous_state: dict | None,
) -> tuple[np.ndarray, dict]:
    """Match the feature order used by collect_controller_dataset.py."""
    positions, orientations = split_pose_flat(pose_flat)
    previous_orientations = None
    if previous_state is not None:
        previous_orientations = previous_state.get("orientations")
    orientations = canonicalize_quaternions(orientations, previous_orientations)

    if (
        previous_state is None
        or dt is None
        or dt <= 1e-8
        or previous_state.get("positions", np.empty((0, 3))).shape != positions.shape
    ):
        linear_velocity = np.zeros_like(positions, dtype=np.float32)
        angular_velocity = np.zeros((orientations.shape[0], 3), dtype=np.float32)
    else:
        linear_velocity = ((positions - previous_state["positions"]) / float(dt)).astype(
            np.float32
        )
        angular_velocity = quaternion_angular_velocity(
            previous_state["orientations"],
            orientations,
            float(dt),
        )

    feature = np.concatenate(
        [
            positions.reshape(-1),
            orientations.reshape(-1),
            linear_velocity.reshape(-1),
            angular_velocity.reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)
    state = {
        "positions": positions.astype(np.float32, copy=True),
        "orientations": orientations.astype(np.float32, copy=True),
    }
    return validate_vector(feature), state


class ControllerMotionFeatureBuilder:
    def __init__(self, pose_stale_sec: float = 0.2):
        self.pose_stale_sec = float(pose_stale_sec)
        self.previous_timestamp = None
        self.previous_left_state = None
        self.previous_right_state = None

    def availability_features(self, is_tracked: bool, pose_age_sec: float | None) -> np.ndarray:
        pose_age = 0.0 if pose_age_sec is None else max(0.0, float(pose_age_sec))
        if self.pose_stale_sec > 1e-8:
            pose_age_norm = min(pose_age / self.pose_stale_sec, 1.0)
            pose_valid = 1.0 if pose_age <= self.pose_stale_sec else 0.0
        else:
            pose_age_norm = 0.0
            pose_valid = 1.0
        return np.asarray(
            [1.0 if bool(is_tracked) else 0.0, pose_valid, pose_age_norm],
            dtype=np.float32,
        )

    def build(
        self,
        left_pose_flat,
        right_pose_flat,
        timestamp: float | None,
        left_is_tracked: bool,
        right_is_tracked: bool,
        left_pose_age_sec: float | None,
        right_pose_age_sec: float | None,
    ) -> np.ndarray:
        dt = None
        if timestamp is not None and self.previous_timestamp is not None:
            dt = float(timestamp) - float(self.previous_timestamp)

        left_feature, left_state = controller_motion_features(
            left_pose_flat, dt, self.previous_left_state
        )
        left_feature = np.concatenate(
            [left_feature, self.availability_features(left_is_tracked, left_pose_age_sec)]
        ).astype(np.float32)

        right_feature, right_state = controller_motion_features(
            right_pose_flat, dt, self.previous_right_state
        )
        right_feature = np.concatenate(
            [right_feature, self.availability_features(right_is_tracked, right_pose_age_sec)]
        ).astype(np.float32)

        self.previous_timestamp = None if timestamp is None else float(timestamp)
        self.previous_left_state = left_state
        self.previous_right_state = right_state
        return validate_vector(np.concatenate([left_feature, right_feature], axis=0))
