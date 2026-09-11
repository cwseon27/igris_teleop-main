from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from igris_teleop.robot_control.controller import core as core_module
from igris_teleop.robot_control.controller.core import ControllerCore
from igris_teleop.robot_control.kinematics.joints import NUM_MOTORS


def _controller_for_stop_test(control_backend: str) -> tuple[ControllerCore, list[tuple]]:
    ctrl = ControllerCore.__new__(ControllerCore)
    ctrl._poses = {
        "waypoint_3": object(),
        "waypoint_2": object(),
        "waypoint_1": object(),
        "zero_pos": object(),
    }
    ctrl._control_backend = control_backend
    calls: list[tuple] = []

    def move_to_pose(pose, **kwargs):
        calls.append(("move_to_pose", pose, kwargs))

    def move_through_poses(pose_sequence, **kwargs):
        calls.append(("move_through_poses", list(pose_sequence), kwargs))
        return True

    def prepare_service_shutdown(**kwargs):
        calls.append(("prepare_service_shutdown", kwargs))

    def run_blocking_shutdown_call(label, fn, *, timeout_s):
        calls.append(("run_blocking_shutdown_call", label, timeout_s))
        fn()
        return True

    def service_shutdown(**kwargs):
        calls.append(("service_shutdown", kwargs))

    def run_service_shutdown_helper(**kwargs):
        calls.append(("run_service_shutdown_helper", kwargs))

    def cleanup_dds():
        calls.append(("cleanup_dds",))

    ctrl.move_to_pose = move_to_pose
    ctrl.move_through_poses = move_through_poses
    ctrl._prepare_service_shutdown = prepare_service_shutdown
    ctrl._run_blocking_shutdown_call = run_blocking_shutdown_call
    ctrl._service_shutdown = service_shutdown
    ctrl._run_service_shutdown_helper = run_service_shutdown_helper
    ctrl._cleanup_dds = cleanup_dds
    return ctrl, calls


def test_stop_runs_shutdown_pose_sequence_before_ros2_torque_off_service() -> None:
    ctrl, calls = _controller_for_stop_test("ros2")

    ctrl.stop(
        shutdown_timeout_ms=5000,
        service_call_timeout_s=12.0,
        ctrl_thread_join_timeout_s=5.0,
    )

    assert [call[1] for call in calls if call[0] == "move_through_poses"] == [
        [
            ("waypoint_3", 2.0),
            ("waypoint_2", 2.0),
            ("waypoint_1", 2.0),
            ("zero_pos", 2.0),
        ]
    ]
    sequence_call = next(call for call in calls if call[0] == "move_through_poses")
    assert sequence_call[2]["use_publisher_timing"] is True
    assert sequence_call[2]["start_from_measured"] is True
    assert sequence_call[2]["arrival_velocity_tolerance_rad_s"] == pytest.approx(0.12)
    service_idx = next(idx for idx, call in enumerate(calls) if call[0] == "service_shutdown")
    prepare_idx = next(idx for idx, call in enumerate(calls) if call[0] == "prepare_service_shutdown")
    last_move_idx = next(idx for idx, call in enumerate(calls) if call[0] == "move_through_poses")
    assert last_move_idx < prepare_idx < service_idx
    assert any(call[0] == "run_blocking_shutdown_call" for call in calls)
    assert not any(call[0] == "run_service_shutdown_helper" for call in calls)
    assert calls[-1] == ("cleanup_dds",)


def test_stop_keeps_sdk_backend_on_shutdown_helper_path() -> None:
    ctrl, calls = _controller_for_stop_test("sdk")

    ctrl.stop(
        shutdown_timeout_ms=5000,
        service_call_timeout_s=12.0,
        ctrl_thread_join_timeout_s=5.0,
    )

    assert any(call[0] == "run_service_shutdown_helper" for call in calls)
    assert not any(call[0] == "run_blocking_shutdown_call" for call in calls)
    assert calls[-1] == ("cleanup_dds",)


def test_stop_keeps_ros2_client_open_when_service_timeout_needs_abort_fallback() -> None:
    ctrl, calls = _controller_for_stop_test("ros2")

    def timeout_shutdown_call(label, fn, *, timeout_s):
        calls.append(("run_blocking_shutdown_call", label, timeout_s))
        return False

    ctrl._run_blocking_shutdown_call = timeout_shutdown_call

    with pytest.raises(TimeoutError, match="did not complete"):
        ctrl.stop()

    assert not any(call[0] == "cleanup_dds" for call in calls)


def test_move_to_pose_does_not_skip_command_samples_when_scheduler_is_overloaded(monkeypatch) -> None:
    ctrl = ControllerCore.__new__(ControllerCore)
    ctrl.control_dt = 1.0 / 300.0
    ctrl._ctrl_lock = threading.Lock()
    ctrl._stop_event = threading.Event()
    ctrl._default_q = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_q = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._target_dq = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_tau = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._kp_default = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kd_default = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kp = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kd = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._poses = {"goal": np.full(NUM_MOTORS, 2.0, dtype=np.float32)}
    ctrl._dbg_publish_logged = True
    ctrl._collect_joint_indices = lambda **kwargs: [0]
    # Generic moves preserve command continuity unless measured-start is
    # explicitly requested by a safety transition.
    ctrl.get_joint_q = lambda: np.zeros(NUM_MOTORS, dtype=np.float32)

    clock = {"now": 0.0}
    targets_seen_before_sleep: list[float] = []

    monkeypatch.setattr(core_module.time, "monotonic", lambda: clock["now"])

    def overloaded_sleep(requested_s: float) -> None:
        targets_seen_before_sleep.append(float(ctrl._target_q[0]))
        # Model the real ROS-loaded machine where short Python sleeps overshoot.
        clock["now"] += max(float(requested_s), 0.05)

    monkeypatch.setattr(core_module.time, "sleep", overloaded_sleep)

    ctrl.move_to_pose("goal", duration=0.2)

    assert targets_seen_before_sleep[0] == pytest.approx(1.0)
    assert ctrl._target_q[0] == pytest.approx(2.0)
    assert targets_seen_before_sleep == sorted(targets_seen_before_sleep)
    adjacent_steps = np.diff(np.asarray(targets_seen_before_sleep, dtype=np.float64))
    assert float(np.max(adjacent_steps)) < 0.06
    # Oversubscribed execution stretches time instead of jumping over samples.
    assert clock["now"] > 0.2


def test_move_through_poses_keeps_velocity_continuous_at_waypoint(monkeypatch) -> None:
    ctrl = ControllerCore.__new__(ControllerCore)
    ctrl.control_dt = 1.0 / 300.0
    ctrl._ctrl_lock = threading.Lock()
    ctrl._stop_event = threading.Event()
    ctrl._default_q = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_q = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_dq = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_tau = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._kp_default = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kd_default = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kp = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kd = np.ones(NUM_MOTORS, dtype=np.float32)
    pose_1 = np.zeros(NUM_MOTORS, dtype=np.float32)
    pose_2 = np.zeros(NUM_MOTORS, dtype=np.float32)
    pose_1[0] = 1.0
    pose_2[0] = 2.0
    ctrl._poses = {"one": pose_1, "two": pose_2}
    ctrl._dbg_publish_logged = True
    ctrl._collect_joint_indices = lambda **kwargs: [0]
    ctrl.get_joint_q = lambda: pose_2.copy()

    clock = {"now": 0.0}
    samples: list[tuple[float, float]] = []
    monkeypatch.setattr(core_module.time, "monotonic", lambda: clock["now"])

    def deterministic_sleep(requested_s: float) -> None:
        samples.append((clock["now"], float(ctrl._target_q[0])))
        clock["now"] += float(requested_s)

    monkeypatch.setattr(core_module.time, "sleep", deterministic_sleep)

    ctrl.move_through_poses([("one", 1.0), ("two", 1.0)])

    by_time = {round(timestamp, 2): target for timestamp, target in samples}
    left_velocity = (by_time[1.0] - by_time[0.99]) / 0.01
    right_velocity = (by_time[1.01] - by_time[1.0]) / 0.01
    left_acceleration = (by_time[1.0] - 2.0 * by_time[0.99] + by_time[0.98]) / 0.01**2
    right_acceleration = (by_time[1.02] - 2.0 * by_time[1.01] + by_time[1.0]) / 0.01**2
    assert by_time[1.0] == pytest.approx(1.0, abs=1e-6)
    assert left_velocity > 0.5
    assert right_velocity > 0.5
    assert left_velocity == pytest.approx(right_velocity, rel=0.05)
    assert abs(left_acceleration) < 0.3
    assert abs(right_acceleration) < 0.3
    assert min(target for _, target in samples) >= 0.0
    assert max(target for _, target in samples) <= 2.0
    assert ctrl._target_q[0] == pytest.approx(2.0)


def test_move_through_poses_can_be_driven_by_control_publisher_thread() -> None:
    ctrl = ControllerCore.__new__(ControllerCore)
    ctrl.control_dt = 1.0 / 300.0
    ctrl._ctrl_lock = threading.Lock()
    ctrl._stop_event = threading.Event()
    ctrl._active_pose_trajectory = None
    ctrl._default_q = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_q = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_dq = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_tau = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._kp_default = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kd_default = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kp = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kd = np.ones(NUM_MOTORS, dtype=np.float32)
    goal = np.zeros(NUM_MOTORS, dtype=np.float32)
    goal[0] = 1.0
    ctrl._poses = {"goal": goal}
    ctrl._dbg_publish_logged = True
    ctrl._collect_joint_indices = lambda **kwargs: [0]
    ctrl.get_joint_q = lambda: goal.copy()

    publisher_stop = threading.Event()

    def publisher_driver() -> None:
        while not publisher_stop.is_set():
            with ctrl._ctrl_lock:
                trajectory = ctrl._active_pose_trajectory
                if trajectory is not None:
                    ctrl._update_pose_trajectory_locked(trajectory, time.monotonic())
            time.sleep(0.001)

    publisher_thread = threading.Thread(target=publisher_driver)
    ctrl._ctrl_thread = publisher_thread
    publisher_thread.start()
    try:
        ctrl.move_through_poses([("goal", 0.03)], use_publisher_timing=True)
    finally:
        publisher_stop.set()
        publisher_thread.join(timeout=1.0)

    assert ctrl._active_pose_trajectory is None
    assert ctrl._target_q[0] == pytest.approx(1.0)


def test_measured_start_interpolates_nonzero_waist_to_zero_without_jump(monkeypatch) -> None:
    ctrl = ControllerCore.__new__(ControllerCore)
    ctrl.control_dt = 1.0 / 300.0
    ctrl._ctrl_lock = threading.Lock()
    ctrl._stop_event = threading.Event()
    ctrl._default_q = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_q = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_dq = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_tau = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._kp_default = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kd_default = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kp = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kd = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._poses = {"zero": np.zeros(NUM_MOTORS, dtype=np.float32)}
    ctrl._dbg_publish_logged = True
    ctrl._collect_joint_indices = lambda **kwargs: [0]
    measured = np.zeros(NUM_MOTORS, dtype=np.float32)
    measured[0] = 0.6
    ctrl.get_joint_q = lambda: measured.copy()

    clock = {"now": 0.0}
    commands: list[float] = []
    monkeypatch.setattr(core_module.time, "monotonic", lambda: clock["now"])

    def deterministic_sleep(requested_s: float) -> None:
        commands.append(float(ctrl._target_q[0]))
        clock["now"] += float(requested_s)

    monkeypatch.setattr(core_module.time, "sleep", deterministic_sleep)

    completed = ctrl.move_through_poses(
        [("zero", 0.05)],
        start_from_measured=True,
    )

    assert completed is True
    assert commands[0] == pytest.approx(0.6)
    assert min(commands) >= -1e-7
    assert max(commands) <= 0.6 + 1e-7
    assert commands == sorted(commands, reverse=True)
    assert ctrl._target_q[0] == pytest.approx(0.0)


def test_tracking_governor_pauses_on_state_stall_and_resumes(monkeypatch) -> None:
    ctrl = ControllerCore.__new__(ControllerCore)
    ctrl.control_dt = 1.0 / 300.0
    ctrl._ctrl_lock = threading.Lock()
    ctrl._stop_event = threading.Event()
    ctrl._default_q = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_q = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_dq = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_tau = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._kp_default = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kd_default = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kp = np.ones(NUM_MOTORS, dtype=np.float32)
    ctrl._kd = np.ones(NUM_MOTORS, dtype=np.float32)
    goal = np.zeros(NUM_MOTORS, dtype=np.float32)
    goal[0] = 1.0
    ctrl._poses = {"goal": goal}
    ctrl._dbg_publish_logged = True
    ctrl._collect_joint_indices = lambda **kwargs: [0]

    clock = {"now": 0.0}
    commands: list[tuple[float, float]] = []

    def measured_q() -> np.ndarray:
        if clock["now"] < 0.2:
            return np.zeros(NUM_MOTORS, dtype=np.float32)
        return ctrl._target_q.copy()

    ctrl.get_joint_q = measured_q
    monkeypatch.setattr(core_module.time, "monotonic", lambda: clock["now"])

    def deterministic_sleep(requested_s: float) -> None:
        commands.append((clock["now"], float(ctrl._target_q[0])))
        clock["now"] += float(requested_s)

    monkeypatch.setattr(core_module.time, "sleep", deterministic_sleep)

    completed = ctrl.move_through_poses(
        [("goal", 0.05)],
        tracking_error_soft_rad=0.01,
        tracking_error_hard_rad=0.05,
        trajectory_timeout_s=1.0,
    )

    assert completed is True
    stalled_commands = [value for timestamp, value in commands if 0.05 <= timestamp < 0.2]
    assert len(stalled_commands) > 5
    assert max(stalled_commands) - min(stalled_commands) < 1e-9
    assert clock["now"] >= 0.2
    assert ctrl._target_q[0] == pytest.approx(1.0)


def test_tracking_governor_can_exclude_persistent_neck_residual() -> None:
    ctrl = ControllerCore.__new__(ControllerCore)
    ctrl._target_q = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_dq = np.zeros(NUM_MOTORS, dtype=np.float32)
    ctrl._target_tau = np.zeros(NUM_MOTORS, dtype=np.float32)
    trajectory = {
        "last_update_at": 0.0,
        "max_phase_step_s": 0.01,
        "tracking_enabled": True,
        "tracking_joint_ids": [0],
        "joint_ids": [0, 1],
        "tracking_soft_error": 0.01,
        "tracking_hard_error": 0.05,
        "max_tracking_error": 0.0,
        "phase_s": 0.0,
        "total_duration": 1.0,
        "segment_durations": np.array([1.0]),
        "segment_ends": np.array([1.0]),
        "q_points": np.zeros((2, NUM_MOTORS), dtype=np.float32),
        "tangents": np.zeros((2, NUM_MOTORS), dtype=np.float64),
        "preserve_gains": True,
        "updates": 0,
        "complete_event": threading.Event(),
    }
    measured = np.zeros(NUM_MOTORS, dtype=np.float32)
    measured[1] = 0.4  # excluded neck-like residual above the hard threshold

    ctrl._update_pose_trajectory_locked(trajectory, 0.01, measured)

    assert trajectory["phase_s"] == pytest.approx(0.01)
    assert trajectory["max_tracking_error"] == pytest.approx(0.0)
