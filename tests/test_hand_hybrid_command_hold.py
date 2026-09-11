from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("rclpy")
from std_msgs.msg import Bool, Float32MultiArray

from igris_teleop.hand_control.command_range import apply_finger_close_overrides
from igris_teleop.teleop_devices.unity.unity_ros_interface import TelevisionROSInterface
from igris_teleop.teleop_devices.unity.hybrid_motor_command import HybridMotorCommand


def _interface() -> TelevisionROSInterface:
    node = object.__new__(TelevisionROSInterface)
    node._lock = threading.Lock()
    node._hybrid_hand_close = {side: np.zeros(5) for side in ("left", "right")}
    node._hybrid_hand_close_time = {side: None for side in ("left", "right")}
    node._has_hybrid_hand_close = {side: False for side in ("left", "right")}
    node._hybrid_hand_command_activated = {side: False for side in ("left", "right")}
    node._hybrid_hand_tracked = {side: None for side in ("left", "right")}
    node._hybrid_hand_tracked_time = {side: None for side in ("left", "right")}
    node._hybrid_hand_pose_stale_sec = 0.25
    return node


def test_held_normalized_vr_commands_keep_precedence_over_retarget_pose_on_tracking_loss() -> None:
    node = _interface()
    for side, value in (("left", 0.2), ("right", 0.3)):
        node._on_hybrid_hand_close(side, Float32MultiArray(data=[value] * 5))
        node._on_hybrid_hand_tracked(side, Bool(data=True))
        # Strict always_1 fusion keeps publishing its last VR command while its
        # tracked flag is false. The receiver must not fall back to a different
        # pose retargeter for these continuously refreshed hold messages.
        node._on_hybrid_hand_tracked(side, Bool(data=False))
        node._on_hybrid_hand_close(side, Float32MultiArray(data=[value] * 5))

    left, right, left_valid, right_valid = node.get_hybrid_hand_commands()
    assert left_valid and right_valid
    command = apply_finger_close_overrides(
        np.ones(12),
        left_close=left,
        right_close=right,
        left_valid=left_valid,
        right_valid=right_valid,
        close_gain=1.0,
    )
    assert command == pytest.approx([0.3] * 6 + [0.2] * 6)


def test_placeholder_before_first_valid_vr_command_does_not_activate_override() -> None:
    node = _interface()
    for side in ("left", "right"):
        node._on_hybrid_hand_close(side, Float32MultiArray(data=[0.0] * 5))
        node._on_hybrid_hand_tracked(side, Bool(data=False))
    _, _, left_valid, right_valid = node.get_hybrid_hand_commands()
    assert not left_valid
    assert not right_valid


def test_motor_ros_callbacks_are_independent_of_legacy_and_hold_when_publishers_stop(monkeypatch):
    node = _interface()
    node._hybrid_hand_motor = {side: HybridMotorCommand() for side in ("left", "right")}
    monkeypatch.setattr("igris_teleop.teleop_devices.unity.unity_ros_interface.time.monotonic", lambda: 1.0)
    node._on_hybrid_hand_motor("left", Float32MultiArray(data=[0.0] * 6))
    assert not node.get_hybrid_hand_motor_commands()[2]
    node._on_hybrid_hand_motor_tracked("left", Bool(data=True))
    node._on_hybrid_hand_motor("left", Float32MultiArray(data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    node._on_hybrid_hand_tracked("left", Bool(data=False))  # Legacy tracking is unrelated.
    monkeypatch.setattr("igris_teleop.teleop_devices.unity.unity_ros_interface.time.monotonic", lambda: 100.0)
    left, _, left_valid, right_valid = node.get_hybrid_hand_motor_commands()
    assert left_valid and not right_valid
    assert left == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
