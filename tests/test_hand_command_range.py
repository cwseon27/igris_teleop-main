from __future__ import annotations

import numpy as np
import pytest

from igris_teleop.hand_control.command_range import (
    apply_finger_close_overrides,
    finger_close_to_hand_motors,
)


def test_finger_close_expands_thumb_to_both_thumb_motors() -> None:
    command = finger_close_to_hand_motors(
        [0.2, 0.3, 0.4, 0.5, 0.6],
        close_gain=1.0,
    )
    assert command == pytest.approx([0.2, 0.3, 0.4, 0.5, 0.6, 0.2])


def test_direct_override_preserves_invalid_side_retargeting() -> None:
    base = np.linspace(0.0, 1.0, 12)
    command = apply_finger_close_overrides(
        base,
        left_close=np.zeros(5),
        right_close=np.ones(5),
        left_valid=False,
        right_valid=True,
        close_gain=1.0,
    )
    assert command[:6] == pytest.approx(np.ones(6))
    assert command[6:] == pytest.approx(base[6:])


def test_invalid_finger_close_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="length must be 5"):
        finger_close_to_hand_motors([0.0, 1.0])
