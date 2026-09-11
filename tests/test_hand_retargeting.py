from __future__ import annotations

import numpy as np
import pytest

from igris_teleop.hand_control.command_range import HAND_RETARGET_MAX_RAD

retargeting_module = pytest.importorskip("igris_teleop.hand_control.hand_retargeting")


class FakeDexRetargeting:
    def __init__(self, result) -> None:
        self.result = np.asarray(result, dtype=np.float64)

    def retarget(self, _reference):
        return self.result.copy()


def _retargeter(left_result, right_result):
    instance = retargeting_module.HandRetargeting.__new__(retargeting_module.HandRetargeting)
    instance.left_retargeting = FakeDexRetargeting(left_result)
    instance.right_retargeting = FakeDexRetargeting(right_result)
    instance.left_dex_retargeting_to_hardware = list(range(6))
    instance.right_dex_retargeting_to_hardware = list(range(6))
    return instance


def test_dual_hand_retargeting_outputs_right_then_left_normalized_bend() -> None:
    left_result = HAND_RETARGET_MAX_RAD / 3.0
    right_result = HAND_RETARGET_MAX_RAD / 3.0
    right_result[-1] *= -1.0
    retargeter = _retargeter(left_result, right_result)
    valid_reference = np.ones((5, 3), dtype=np.float64)

    command = retargeter.retarget_normalized(valid_reference, valid_reference)

    assert command.shape == (12,)
    assert np.allclose(command[:6], 0.5)
    assert np.allclose(command[6:], 0.5)


def test_each_hand_can_retarget_independently() -> None:
    retargeter = _retargeter(HAND_RETARGET_MAX_RAD, HAND_RETARGET_MAX_RAD)
    valid_reference = np.ones((5, 3), dtype=np.float64)
    missing_reference = np.zeros((5, 3), dtype=np.float64)

    command = retargeter.retarget_normalized(missing_reference, valid_reference)

    assert np.allclose(command[:6], 1.0)
    assert np.allclose(command[6:], 0.0)
