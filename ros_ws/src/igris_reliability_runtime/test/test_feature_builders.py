from __future__ import annotations

import numpy as np

from igris_reliability_runtime.feature_builders import ControllerMotionFeatureBuilder


def test_controller_feature_order_and_dimension_match_policy_contract() -> None:
    builder = ControllerMotionFeatureBuilder(pose_stale_sec=0.2)
    left = np.array([1, 2, 3, 0, 0, 0, 1], dtype=np.float32)
    right = np.array([4, 5, 6, 0, 0, 0, 1], dtype=np.float32)

    first = builder.build(
        left,
        right,
        timestamp=1.0,
        left_is_tracked=True,
        right_is_tracked=False,
        left_pose_age_sec=0.0,
        right_pose_age_sec=0.1,
    )
    second = builder.build(
        left + np.array([0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        right,
        timestamp=1.1,
        left_is_tracked=True,
        right_is_tracked=True,
        left_pose_age_sec=0.0,
        right_pose_age_sec=0.3,
    )

    assert first.shape == (32,)
    np.testing.assert_allclose(first[0:3], [1, 2, 3])
    np.testing.assert_allclose(first[7:10], [0, 0, 0])
    np.testing.assert_allclose(first[13:16], [1, 1, 0])
    np.testing.assert_allclose(first[29:32], [0, 1, 0.5])
    np.testing.assert_allclose(second[7], np.float32(1.0), rtol=1e-6)
    np.testing.assert_allclose(second[29:32], [1, 0, 1])
