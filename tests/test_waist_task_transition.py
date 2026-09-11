from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from igris_teleop.robot_control.kinematics.ik.waist_task_transition import (
    WaistTaskTransition,
    WaistTransitionParams,
    compute_task_activation,
)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.0, 0.0),
        (0.1, 0.0),
        (0.2, 0.0),
        (0.5, 0.5),
        (0.8, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
    ],
)
def test_compute_task_activation_uses_smoothstep(
    confidence: float,
    expected: float,
) -> None:
    assert compute_task_activation(confidence, 0.2, 0.8) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (-0.1, 0.8),
        (0.2, 0.2),
        (0.8, 0.2),
        (0.2, 1.1),
    ],
)
def test_compute_task_activation_rejects_invalid_thresholds(
    low: float,
    high: float,
) -> None:
    with pytest.raises(ValueError):
        compute_task_activation(0.5, low, high)


@pytest.mark.parametrize(
    ("confidence", "target_valid"),
    [
        (0.0, True),
        (-0.1, True),
        (float("nan"), True),
        (float("inf"), True),
        (1.0, False),
    ],
)
def test_invalid_or_exact_zero_forces_immediate_disable(
    confidence: float,
    target_valid: bool,
) -> None:
    transition = WaistTaskTransition(WaistTransitionParams())
    transition.reset(np.array([0.1, -0.2, 0.3]), activation=1.0, timestamp=1.0)

    result = transition.update_activation(
        confidence,
        target_valid=target_valid,
        now=1.02,
    )

    assert result.forced_disable is True
    assert result.raw_activation == 0.0
    assert result.filtered_activation == 0.0
    assert transition.previous_activation == 0.0


def test_valid_positive_confidence_decrease_uses_fall_rate() -> None:
    params = WaistTransitionParams(activation_fall_rate=4.0)
    transition = WaistTaskTransition(params)
    transition.reset(np.zeros(3), activation=1.0, timestamp=1.0)

    result = transition.update_activation(
        0.1,
        target_valid=True,
        now=1.02,
    )

    assert result.forced_disable is False
    assert result.raw_activation == 0.0
    assert result.filtered_activation == pytest.approx(0.92)


def test_positive_confidence_recovers_with_rise_rate_after_forced_disable() -> None:
    params = WaistTransitionParams(activation_rise_rate=2.0)
    transition = WaistTaskTransition(params)
    transition.reset(np.zeros(3), activation=1.0, timestamp=1.0)
    transition.update_activation(0.0, target_valid=True, now=1.02)

    result = transition.update_activation(
        1.0,
        target_valid=True,
        now=1.04,
    )

    assert result.forced_disable is False
    assert result.raw_activation == 1.0
    assert result.filtered_activation == pytest.approx(0.04)


def test_transition_clamps_large_dt() -> None:
    params = WaistTransitionParams(
        activation_rise_rate=2.0,
        max_dt_sec=0.1,
    )
    transition = WaistTaskTransition(params)
    transition.reset(np.zeros(3), timestamp=1.0)

    result = transition.update_activation(
        1.0,
        target_valid=True,
        now=11.0,
    )

    assert result.dt == pytest.approx(0.1)
    assert result.filtered_activation == pytest.approx(0.2)


def test_previous_published_command_is_copied_and_committed_explicitly() -> None:
    transition = WaistTaskTransition(WaistTransitionParams())
    command = np.array([0.1, 0.2, 0.3])
    transition.reset(command, timestamp=1.0)
    command[:] = 9.0

    np.testing.assert_array_equal(
        transition.previous_published_waist,
        [0.1, 0.2, 0.3],
    )

    next_command = np.array([-0.1, -0.2, -0.3])
    transition.commit_published_command(next_command)
    next_command[:] = 9.0

    np.testing.assert_array_equal(
        transition.previous_published_waist,
        [-0.1, -0.2, -0.3],
    )


def test_default_delta_limit_inherits_solver_limit() -> None:
    transition = WaistTaskTransition(WaistTransitionParams())

    assert transition.activated_delta_limit(
        0.5,
        fallback_limit=0.05,
    ) == pytest.approx(0.025)


def test_head_translation_scaling_is_configurable_and_off_by_default() -> None:
    default_transition = WaistTaskTransition(WaistTransitionParams())
    scaled_transition = WaistTaskTransition(
        replace(
            WaistTransitionParams(),
            scale_head_translation_with_activation=True,
            low_conf_head_translation_weight=0.0,
        )
    )

    assert default_transition.head_translation_weight(
        0.0,
        base_weight=1.0,
    ) == pytest.approx(1.0)
    assert scaled_transition.head_translation_weight(
        0.0,
        base_weight=1.0,
    ) == pytest.approx(0.0)
    assert scaled_transition.head_translation_weight(
        0.5,
        base_weight=1.0,
    ) == pytest.approx(0.5)
    assert scaled_transition.head_translation_weight(
        1.0,
        base_weight=1.0,
    ) == pytest.approx(1.0)
