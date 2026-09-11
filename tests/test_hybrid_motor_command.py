from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from igris_teleop.hand_control.command_range import apply_motor_command_overrides
from igris_teleop.teleop_devices.unity.hybrid_motor_command import HybridMotorCommand
from igris_teleop.workers.worker_hand import HandWorker


def test_startup_untracked_zero_does_not_activate_and_fresh_command_has_no_filter_delay():
    stream = HybridMotorCommand()
    stream.update_tracking(False, 1.0)
    assert not stream.update_command(np.zeros(6), 1.01)
    assert not stream.snapshot()[1]
    stream.update_tracking(True, 2.0)
    assert not stream.snapshot()[1]  # Earlier placeholder is never promoted.
    target = np.linspace(0.1, 0.6, 6)
    assert stream.update_command(target, 2.001)
    np.testing.assert_array_equal(stream.snapshot()[0], target)
    assert stream.snapshot()[1]


def test_activated_stream_holds_after_tracking_loss_topic_dropout_and_malformed_samples():
    stream = HybridMotorCommand()
    stream.update_tracking(True, 1.0)
    target = np.linspace(0.1, 0.6, 6)
    stream.update_command(target, 1.001)
    # No heartbeat: reject incoming new values rather than trusting stale tracking.
    assert not stream.update_command(np.ones(6), 1.3)
    stream.update_tracking(False, 2.0)
    assert not stream.update_command(np.zeros(6), 2.01)
    held, valid = stream.snapshot()
    assert valid  # Still overrides the other retargeter even if the publisher stops.
    np.testing.assert_array_equal(held, target)
    stream.update_tracking(True, 100.0)
    for bad in ([np.nan] * 6, [np.inf] * 6, [0.5] * 5, [-0.01] * 6, [1.01] * 6):
        assert not stream.update_command(bad, 100.001)
        np.testing.assert_array_equal(stream.snapshot()[0], target)
    recovered = target / 2
    assert stream.update_command(recovered, 100.002)
    np.testing.assert_array_equal(stream.snapshot()[0], recovered)


def test_final_motor_override_preserves_two_different_thumb_motors_and_does_not_regain():
    right = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    left = right + 0.1
    result = apply_motor_command_overrides(
        np.ones(12), left_motor=left, right_motor=right, left_valid=True, right_valid=True,
    )
    np.testing.assert_array_equal(result, np.r_[right, left])
    assert result[0] != result[5] and result[6] != result[11]


@pytest.mark.parametrize("bad", ([1.1] * 6, [np.nan] * 6, [0.5] * 5))
def test_worker_rejects_invalid_motor_input(bad):
    left, _, valid, _ = HandWorker._normalized_motor_inputs(
        {"left_hand_motor": bad, "left_hand_motor_valid": 1.0}
    )
    assert not valid
    np.testing.assert_array_equal(left, np.zeros(6))


def test_real_control_loop_prefers_final_motors_over_legacy_without_actuating_hardware():
    pytest.importorskip("pinocchio", reason="controller dependencies are installed in .venv-ik")
    from igris_teleop.hand_control.robot_hand import IgrisHandController

    controller = IgrisHandController.__new__(IgrisHandController)
    controller.rate = SimpleNamespace(tick_hz=lambda: 0, sleep=lambda: None)
    controller.hand_interface = SimpleNamespace(
        get_present_position=lambda: [0.0] * 12, stop=lambda: None,
    )
    controller.hand_retargeting = SimpleNamespace(
        retarget_normalized=lambda *args: pytest.fail("must not retarget final motor commands twice")
    )
    sent = []

    def capture_only(command):
        sent.append(command.copy())
        controller.running = False

    controller.ctrl_dual_hand = capture_only
    left, right = np.linspace(0.2, 0.7, 6), np.linspace(0.1, 0.6, 6)
    controller.control_process(
        np.zeros(15), np.zeros(15),
        hybrid_hand_command_lock=threading.Lock(),
        left_hand_close_array=np.ones(5), right_hand_close_array=np.ones(5),
        hand_close_valid_array=np.ones(2),
        left_hand_motor_array=left, right_hand_motor_array=right,
        hand_motor_valid_array=np.ones(2),
    )
    assert len(sent) == 1
    np.testing.assert_array_equal(sent[0], np.r_[right, left])
