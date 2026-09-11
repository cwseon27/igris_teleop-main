from __future__ import annotations

import logging
import sys
import threading
import time
import types

import numpy as np

if "logging_mp" not in sys.modules:
    logging_mp = types.ModuleType("logging_mp")
    logging_mp.INFO = logging.INFO
    logging_mp.get_logger = lambda *args, **kwargs: logging.getLogger("test-masterarm-guard")
    sys.modules["logging_mp"] = logging_mp

if "rclpy" not in sys.modules:
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *args, **kwargs: None
    rclpy.spin = lambda *args, **kwargs: None
    rclpy.ok = lambda: False
    rclpy.shutdown = lambda: None
    rclpy.try_shutdown = lambda: None
    sys.modules["rclpy"] = rclpy

if "rclpy.node" not in sys.modules:
    rclpy_node = types.ModuleType("rclpy.node")

    class _Node:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def create_subscription(self, *args, **kwargs):
            return None

    rclpy_node.Node = _Node
    sys.modules["rclpy.node"] = rclpy_node

if "sensor_msgs" not in sys.modules:
    sensor_msgs = types.ModuleType("sensor_msgs")
    sys.modules["sensor_msgs"] = sensor_msgs

if "sensor_msgs.msg" not in sys.modules:
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")

    class _JointState:
        def __init__(self) -> None:
            self.name = []
            self.position = []

    sensor_msgs_msg.JointState = _JointState
    sys.modules["sensor_msgs.msg"] = sensor_msgs_msg

from igris_teleop.workers.worker_master_arm_bridge import (
    MASTERARM_ARM_ALL_ZERO_REASON,
    MASTERARM_ARM_INVALID_REASON,
    MASTERARM_ARM_STALE_REASON,
    MASTERARM_ARM_WAITING_REASON,
    MasterarmRosridgeWorker,
    extract_obs_arm_hold_target,
    load_masterarm_joint_calibration,
    resolve_masterarm_bridge_hz,
    resolve_masterarm_arm_target,
)
from igris_teleop.core.events import EventSnapshot
from igris_teleop.core.state_machine import ModeState, TransitionResult
from igris_teleop.robot_control.interfaces.master_arm_ros_interface import (
    ARM_SRC_NAMES,
    MasterArmROSInterface,
)


def _bare_masterarm_interface() -> MasterArmROSInterface:
    iface = MasterArmROSInterface.__new__(MasterArmROSInterface)
    iface._present_position_lock = threading.Lock()
    iface._hand_position_lock = threading.Lock()
    iface._arm_update_condition = threading.Condition()
    iface._present_position = [0.0] * 14
    iface._hand_position = [0.0] * 4
    iface._has_position = threading.Event()
    iface._has_hand_position = threading.Event()
    iface._arm_seq = 0
    iface._hand_seq = 0
    iface._last_arm_update_monotonic = None
    iface._last_hand_update_monotonic = None
    return iface


def _arm_joint_state(values: np.ndarray):
    msg = sys.modules["sensor_msgs.msg"].JointState()
    msg.name = list(ARM_SRC_NAMES)
    msg.position = np.asarray(values, dtype=np.float64).tolist()
    return msg


def test_extract_obs_arm_hold_target_accepts_valid_obs_arm() -> None:
    obs_arm = np.linspace(-0.5, 0.5, 14, dtype=np.float64)
    target = extract_obs_arm_hold_target({"obs_arm": obs_arm})
    assert target is not None
    assert np.allclose(target, obs_arm)


def test_resolve_masterarm_arm_target_prefers_leader_input_when_fresh() -> None:
    present = np.linspace(-1.0, 1.0, 14, dtype=np.float64)
    obs_arm = np.full(14, 9.0, dtype=np.float64)

    target, reason = resolve_masterarm_arm_target(
        present,
        {"has_position": 1.0, "age_s": 0.01, "is_all_zero": 0.0},
        {"obs_arm": obs_arm},
        stale_timeout_s=0.3,
    )

    assert reason is None
    assert target is not None
    assert np.allclose(target, present)


def test_resolve_masterarm_arm_target_holds_current_pose_before_first_sample() -> None:
    obs_arm = np.linspace(0.1, 1.4, 14, dtype=np.float64)

    target, reason = resolve_masterarm_arm_target(
        None,
        {"has_position": 0.0, "age_s": float("inf"), "is_all_zero": 1.0},
        {"obs_arm": obs_arm},
        stale_timeout_s=0.3,
    )

    assert reason == MASTERARM_ARM_WAITING_REASON
    assert target is not None
    assert np.allclose(target, obs_arm)


def test_resolve_masterarm_arm_target_holds_current_pose_for_all_zero_input() -> None:
    obs_arm = np.linspace(-0.2, 0.7, 14, dtype=np.float64)

    target, reason = resolve_masterarm_arm_target(
        np.zeros(14, dtype=np.float64),
        {"has_position": 1.0, "age_s": 0.01, "is_all_zero": 1.0},
        {"obs_arm": obs_arm},
        stale_timeout_s=0.3,
    )

    assert reason == MASTERARM_ARM_ALL_ZERO_REASON
    assert target is not None
    assert np.allclose(target, obs_arm)


def test_resolve_masterarm_arm_target_holds_current_pose_for_stale_input() -> None:
    present = np.linspace(-1.0, 1.0, 14, dtype=np.float64)
    obs_arm = np.linspace(1.0, -1.0, 14, dtype=np.float64)

    target, reason = resolve_masterarm_arm_target(
        present,
        {"has_position": 1.0, "age_s": 0.5, "is_all_zero": 0.0},
        {"obs_arm": obs_arm},
        stale_timeout_s=0.3,
    )

    assert reason == MASTERARM_ARM_STALE_REASON
    assert target is not None
    assert np.allclose(target, obs_arm)


def test_resolve_masterarm_arm_target_returns_none_when_invalid_and_no_hold_source() -> None:
    target, reason = resolve_masterarm_arm_target(
        [1.0, 2.0, 3.0],
        {"has_position": 1.0, "age_s": 0.01, "is_all_zero": 0.0},
        {},
        stale_timeout_s=0.3,
    )

    assert reason == MASTERARM_ARM_INVALID_REASON
    assert target is None


def test_masterarm_bridge_rate_uses_bounded_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("IGRIS_MASTERARM_BRIDGE_HZ", "350")
    assert resolve_masterarm_bridge_hz() == 350.0

    monkeypatch.setenv("IGRIS_MASTERARM_BRIDGE_HZ", "5")
    assert resolve_masterarm_bridge_hz() == 20.0


def test_masterarm_joint_calibration_env_offset(monkeypatch) -> None:
    offset = np.linspace(-0.13, 0.13, 14, dtype=np.float64)
    scale = np.ones(14, dtype=np.float64)
    monkeypatch.setenv("IGRIS_MASTERARM_ARM_OFFSET_RAD", ",".join(str(float(v)) for v in offset))
    monkeypatch.setenv("IGRIS_MASTERARM_ARM_SCALE", ",".join(str(float(v)) for v in scale))

    calibration = load_masterarm_joint_calibration()

    assert calibration["enabled"] is True
    assert np.allclose(calibration["scale"], scale)
    assert np.allclose(calibration["offset"], offset)


def test_worker_applies_masterarm_joint_calibration() -> None:
    worker = MasterarmRosridgeWorker.__new__(MasterarmRosridgeWorker)
    worker._joint_calibration = {
        "enabled": True,
        "scale": np.full(14, 2.0, dtype=np.float64),
        "offset": np.linspace(-0.1, 0.1, 14, dtype=np.float64),
        "source": "test",
    }
    raw = np.linspace(-1.0, 1.0, 14, dtype=np.float64)

    calibrated = worker._apply_joint_calibration(raw)

    assert np.allclose(calibrated, raw * 2.0 + worker._joint_calibration["offset"])


def test_worker_wait_uses_interface_fresh_sample_wait() -> None:
    calls: list[tuple[int, float]] = []

    class _Interface:
        def wait_for_arm_update(self, after_seq: int, timeout: float) -> int:
            calls.append((after_seq, timeout))
            return after_seq + 1

    worker = MasterarmRosridgeWorker.__new__(MasterarmRosridgeWorker)
    worker.iface = _Interface()
    worker.ctx = types.SimpleNamespace(stop_event=None)
    worker._watchdog_wait_event = threading.Event()

    seq = worker._wait_for_arm_sample_or_watchdog(7, 0.005)

    assert seq == 8
    assert calls == [(7, 0.005)]


def test_worker_run_keeps_configured_periodic_watchdog_rate() -> None:
    worker = MasterarmRosridgeWorker.__new__(MasterarmRosridgeWorker)
    stop_event = threading.Event()
    worker.ctx = types.SimpleNamespace(
        name="master_arm_ros_bridge",
        stop_event=stop_event,
        runtime_diagnostics=None,
    )
    worker.hz = 125.0
    worker._state_lock = threading.Lock()
    worker._state = ModeState.WAIT_CONNECT
    worker.on_start = lambda: None
    worker.on_stop = lambda: None
    worker.poll = lambda: (
        EventSnapshot(level={}),
        TransitionResult(ModeState.WAIT_CONNECT, "test"),
    )
    step_count = 0
    observed_timeouts: list[float] = []

    def _step_once(*_args) -> None:
        nonlocal step_count
        step_count += 1
        if step_count == 2:
            stop_event.set()

    def _wait(after_seq: int, timeout_s: float) -> int:
        observed_timeouts.append(timeout_s)
        return after_seq

    worker.step_once = _step_once
    worker._wait_for_arm_sample_or_watchdog = _wait

    worker.run()

    assert step_count == 2
    assert observed_timeouts == [1.0 / 125.0, 1.0 / 125.0]


def test_arm_joint_state_mapping_preserves_leader_joint_values() -> None:
    iface = _bare_masterarm_interface()
    expected = np.linspace(-0.7, 0.6, 14, dtype=np.float64)
    msg = _arm_joint_state(expected)

    iface._on_joint_state(msg)

    assert iface._has_position.is_set()
    np.testing.assert_allclose(iface.get_present_position(), expected)


def test_arm_snapshot_keeps_position_status_and_sequence_atomic() -> None:
    iface = _bare_masterarm_interface()
    expected = np.linspace(0.2, 1.5, 14, dtype=np.float64)

    iface._on_joint_state(_arm_joint_state(expected))
    present, status = iface.get_arm_snapshot()

    np.testing.assert_allclose(present, expected)
    assert status["has_position"] == 1.0
    assert status["seq"] == 1.0
    assert 0.0 <= status["age_s"] < 0.5
    assert status["is_all_zero"] == 0.0


def test_arm_update_wait_does_not_lose_notification_before_wait() -> None:
    iface = _bare_masterarm_interface()
    iface._on_joint_state(_arm_joint_state(np.ones(14, dtype=np.float64)))

    started_at = time.monotonic()
    seq = iface.wait_for_arm_update(after_seq=0, timeout=0.5)
    elapsed = time.monotonic() - started_at

    assert seq == 1
    assert elapsed < 0.1
    # The existing event remains the one-shot first-sample indicator.
    assert iface._has_position.is_set()


def test_arm_update_wait_wakes_when_callback_commits_fresh_sample() -> None:
    iface = _bare_masterarm_interface()
    result: dict[str, float | int] = {}
    waiter_started = threading.Event()

    def _waiter() -> None:
        waiter_started.set()
        started_at = time.monotonic()
        result["seq"] = iface.wait_for_arm_update(after_seq=0, timeout=0.5)
        result["elapsed"] = time.monotonic() - started_at

    thread = threading.Thread(target=_waiter)
    thread.start()
    assert waiter_started.wait(timeout=0.2)
    time.sleep(0.01)
    iface._on_joint_state(_arm_joint_state(np.full(14, 0.4, dtype=np.float64)))
    thread.join(timeout=0.2)

    assert not thread.is_alive()
    assert result["seq"] == 1
    assert float(result["elapsed"]) < 0.2


def test_arm_update_wait_times_out_for_periodic_stale_watchdog() -> None:
    iface = _bare_masterarm_interface()

    started_at = time.monotonic()
    seq = iface.wait_for_arm_update(after_seq=0, timeout=0.03)
    elapsed = time.monotonic() - started_at

    assert seq == 0
    assert elapsed >= 0.02
    assert elapsed < 0.2


def test_masterarm_hand_mapping_matches_observed_polarity() -> None:
    iface = MasterArmROSInterface.__new__(MasterArmROSInterface)
    iface._hand_position_lock = threading.Lock()

    iface._hand_position = [1.6, -1.47, -1.57, 1.50]

    act = np.asarray(iface.get_act_hand_12(), dtype=np.float64)

    assert np.allclose(act[:5], [1.0, 1.0, 1.0, 1.0, 1.0])
    assert act[5] == 1.0
    assert np.allclose(act[6:11], [1.0, 1.0, 1.0, 1.0, 1.0])
    assert act[11] == 1.0

    iface._hand_position = [-0.03, 0.03, 0.05, -0.01]
    act = np.asarray(iface.get_act_hand_12(), dtype=np.float64)

    assert np.allclose(act, np.zeros(12))
