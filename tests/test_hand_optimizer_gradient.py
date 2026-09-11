"""Numerical consistency checks using the configured IGRIS hand kinematics."""

from __future__ import annotations

import numpy as np
import pytest

retargeting_module = pytest.importorskip("igris_teleop.hand_control.hand_retargeting")


@pytest.mark.parametrize("side", ["left", "right"])
def test_vector_optimizer_gradient_matches_returned_objective(side: str) -> None:
    retargeter = retargeting_module.HandRetargeting()
    sequential = getattr(retargeter, f"{side}_retargeting")
    optimizer = sequential.optimizer
    assert optimizer.retargeting_type == "VECTOR"
    assert optimizer.norm_delta > 0.0

    # Stay inside all joint bounds and deliberately differ from last_qpos;
    # evaluating only x == last_qpos would conceal the missing scalar penalty.
    bounds = sequential.joint_limits.astype(np.float64)
    previous = bounds.mean(axis=1)
    x = previous + 0.08 * (bounds[:, 1] - bounds[:, 0])
    target_vectors = np.asarray(
        [
            [0.020, 0.035, 0.080],
            [0.010, 0.045, 0.120],
            [0.005, 0.025, 0.140],
            [-0.010, 0.010, 0.120],
            [-0.025, -0.005, 0.090],
        ],
        dtype=np.float64,
    )
    fixed_qpos = np.zeros(len(optimizer.idx_pin2fixed), dtype=np.float64)
    objective = optimizer.get_objective_function(target_vectors, fixed_qpos, previous)
    analytic = np.empty_like(x)
    value = objective(x, analytic)
    assert np.isfinite(value)
    np.testing.assert_allclose(value, objective(x, np.empty(0)), rtol=0.0, atol=1e-14)

    epsilon = 1e-6
    numerical = np.empty_like(x)
    for index in range(x.size):
        delta = np.zeros_like(x)
        delta[index] = epsilon
        numerical[index] = (
            objective(x + delta, np.empty(0)) - objective(x - delta, np.empty(0))
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(analytic, numerical, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize("side", ["left", "right"])
def test_vector_objective_includes_previous_pose_regularization(side: str) -> None:
    retargeter = retargeting_module.HandRetargeting()
    sequential = getattr(retargeter, f"{side}_retargeting")
    optimizer = sequential.optimizer
    x = sequential.joint_limits.mean(axis=1).astype(np.float64)
    previous = x + 0.1
    target_vectors = np.full((5, 3), 0.08, dtype=np.float64)
    fixed_qpos = np.zeros(len(optimizer.idx_pin2fixed), dtype=np.float64)

    current_anchor = optimizer.get_objective_function(target_vectors, fixed_qpos, x)
    previous_anchor = optimizer.get_objective_function(target_vectors, fixed_qpos, previous)
    increment = previous_anchor(x, np.empty(0)) - current_anchor(x, np.empty(0))

    assert increment == pytest.approx(optimizer.norm_delta * np.sum((x - previous) ** 2))
