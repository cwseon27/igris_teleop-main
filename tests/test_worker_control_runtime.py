import igris_c_sdk as igc_sdk
import numpy as np
import types

from igris_teleop.workers.worker_control import ControlWorker, resolve_control_kinematic_mode
from igris_teleop.robot_control.controller.igris_controller import BaseController
from igris_teleop.robot_control.kinematics.joints import (
    ARM_INDICES,
    LEG_INDICES,
    NECK_INDICES,
    WAIST_INDICES,
    JointIndex,
    NUM_MOTORS,
)
from igris_teleop.robot_control.kinematics.pr2ab import default_pr2ab_pairs


def test_simulation_uses_pjs_without_pr2ab_calibration() -> None:
    for runtime_environment in ("sim", "simulator", "simulation", "mujoco"):
        assert resolve_control_kinematic_mode(runtime_environment) == igc_sdk.KinematicMode.PJS


def test_real_robot_keeps_ms_calibration_path() -> None:
    for runtime_environment in (None, "", "real"):
        assert resolve_control_kinematic_mode(runtime_environment) == igc_sdk.KinematicMode.MS


def test_real_robot_ms_command_flips_only_neck_pitch_motor_direction() -> None:
    ctrl = BaseController.__new__(BaseController)
    ctrl._pr2ab_pairs = default_pr2ab_pairs()
    ctrl._warned_unconfigured_ms = True

    q_pjs = np.zeros(NUM_MOTORS, dtype=np.float32)
    q_pjs[int(JointIndex.NECK_YAW)] = 0.25
    q_pjs[int(JointIndex.NECK_PITCH)] = 0.35

    q_ms = ctrl._build_command_q(q_pjs, igc_sdk.KinematicMode.MS)

    assert q_ms[int(JointIndex.NECK_YAW)] == q_pjs[int(JointIndex.NECK_YAW)]
    assert q_ms[int(JointIndex.NECK_PITCH)] == -q_pjs[int(JointIndex.NECK_PITCH)]
    assert q_pjs[int(JointIndex.NECK_PITCH)] == np.float32(0.35)


def test_pjs_command_keeps_neck_pitch_direction_for_sim_and_visualizer_convention() -> None:
    ctrl = BaseController.__new__(BaseController)
    q_pjs = np.zeros(NUM_MOTORS, dtype=np.float32)
    q_pjs[int(JointIndex.NECK_PITCH)] = 0.35

    q_cmd = ctrl._build_command_q(q_pjs, igc_sdk.KinematicMode.PJS)

    assert q_cmd[int(JointIndex.NECK_PITCH)] == q_pjs[int(JointIndex.NECK_PITCH)]


def test_entry_gate_mode_uses_leader_joint_for_masterarm_devices() -> None:
    worker = ControlWorker.__new__(ControlWorker)
    for teleop_device in ("masterarm", "vr_masterarm"):
        worker.ctx = types.SimpleNamespace(run_config=types.SimpleNamespace(teleop_device=teleop_device))
        assert worker._entry_gate_mode() == "leader_joint"


def test_entry_gate_mode_uses_ee_for_vr_only_devices() -> None:
    worker = ControlWorker.__new__(ControlWorker)
    for teleop_device in ("unity", "unity_hybrid"):
        worker.ctx = types.SimpleNamespace(run_config=types.SimpleNamespace(teleop_device=teleop_device))
        assert worker._entry_gate_mode() == "vr_ee"


def test_vr_ee_gate_checks_position_and_rotation_error() -> None:
    class _Shm:
        def __init__(self, data):
            self._data = data

        def read_data(self):
            return self._data

    worker = ControlWorker.__new__(ControlWorker)
    worker._ee_target_max_pos_error = 0.02
    worker._ee_target_max_rot_error = np.deg2rad(5.0)
    identity = np.eye(4, dtype=np.float64)
    target = np.eye(4, dtype=np.float64)
    target[:3, 3] = [0.01, 0.0, 0.0]
    worker.ee_shm = _Shm({"left_wrist_mat": identity, "right_wrist_mat": identity})
    worker.ik_target_shm = _Shm(
        {
            "target_valid": 1.0,
            "left_wrist_mat": target,
            "right_wrist_mat": target,
        }
    )

    within, pos_error, rot_error = worker._ee_target_within_error()

    assert within is True
    assert np.isclose(pos_error, 0.01)
    assert np.isclose(rot_error, 0.0)


def test_vr_entry_waits_for_ik_target_published_after_start() -> None:
    class _Shm:
        def __init__(self, seq: float):
            self.seq = seq

        def read_data(self):
            return {"target_seq": self.seq}

    worker = ControlWorker.__new__(ControlWorker)
    worker.ik_target_shm = _Shm(10.0)
    worker._entry_target_seq_at_start = 10.0

    assert worker._vr_entry_target_is_fresh() is False

    worker.ik_target_shm.seq = 10.1
    assert worker._vr_entry_target_is_fresh() is True


def test_leader_pass_indices_are_arm_only_but_ramp_covers_upper_body() -> None:
    worker = ControlWorker.__new__(ControlWorker)
    worker.ctx = types.SimpleNamespace(
        run_config=types.SimpleNamespace(teleop_device="vr_masterarm")
    )
    worker._leader_gate_indices = tuple(int(idx) for idx in ARM_INDICES)
    worker._entry_ramp_indices = tuple(
        int(idx)
        for group in (WAIST_INDICES, ARM_INDICES, NECK_INDICES)
        for idx in group
    )

    ramp = set(worker._active_gate_indices())

    assert ramp.issuperset(int(idx) for idx in WAIST_INDICES)
    assert ramp.issuperset(int(idx) for idx in ARM_INDICES)
    assert ramp.issuperset(int(idx) for idx in NECK_INDICES)
    assert ramp.isdisjoint(int(idx) for idx in LEG_INDICES)
    assert set(worker._leader_gate_indices) == set(int(idx) for idx in ARM_INDICES)


def test_leader_joint_pass_condition_ignores_waist_error() -> None:
    worker = ControlWorker.__new__(ControlWorker)
    worker.ctrl = object()
    current = np.zeros(NUM_MOTORS, dtype=np.float64)
    target = np.zeros(NUM_MOTORS, dtype=np.float64)
    target[list(WAIST_INDICES)] = 1.0
    target[list(ARM_INDICES)] = 0.1

    within, max_error = worker._joint_target_within_error(
        target,
        current,
        tuple(int(idx) for idx in ARM_INDICES),
        0.2,
    )

    assert within is True
    assert np.isclose(max_error, 0.1)


def test_home_moves_directly_to_default_without_ready_or_shutdown_waypoints() -> None:
    calls = []

    class _Controller:
        def move_to_pose(self, pose, **kwargs):
            calls.append((pose, kwargs))
            return True

        def move_through_poses(self, *args, **kwargs):
            raise AssertionError("HOME must not use the multi-waypoint Ready path")

    worker = ControlWorker.__new__(ControlWorker)
    worker.ctrl = _Controller()
    worker.ctx = types.SimpleNamespace(stop_event=types.SimpleNamespace(is_set=lambda: False))

    assert worker._move_directly_to_default_pose() is True
    assert len(calls) == 1
    pose, kwargs = calls[0]
    assert pose == "default_pos"
    assert kwargs["start_from_measured"] is True
    assert set(kwargs["tracking_joint_ids"]) == (
        set(int(idx) for idx in WAIST_INDICES)
        | set(int(idx) for idx in ARM_INDICES)
    )


def test_teleop_conditioner_bounds_upper_body_handover(monkeypatch) -> None:
    worker = ControlWorker.__new__(ControlWorker)
    worker.fast_hz = 100.0
    worker.ctrl = None
    worker._entry_ramp_indices = tuple(
        int(idx)
        for group in (WAIST_INDICES, ARM_INDICES, NECK_INDICES)
        for idx in group
    )
    worker._teleop_cmd_q = None
    worker._teleop_cmd_dq = np.zeros(NUM_MOTORS, dtype=np.float64)
    worker._teleop_limiter_last_t = None
    worker._teleop_velocity_limit = np.full(NUM_MOTORS, np.inf, dtype=np.float64)
    worker._teleop_acceleration_limit = np.full(NUM_MOTORS, np.inf, dtype=np.float64)
    worker._teleop_command_lead_limit = np.full(NUM_MOTORS, np.inf, dtype=np.float64)
    active = list(worker._entry_ramp_indices)
    worker._teleop_velocity_limit[active] = 1.0
    worker._teleop_acceleration_limit[active] = 2.0
    worker._teleop_command_lead_limit[active] = 0.2

    clock = {"now": 0.0}
    monkeypatch.setattr(
        "igris_teleop.workers.worker_control.time.monotonic",
        lambda: clock["now"],
    )
    current = np.zeros(NUM_MOTORS, dtype=np.float64)
    target = np.ones(NUM_MOTORS, dtype=np.float64)
    target[list(LEG_INDICES)] = 0.0

    commands = []
    for _ in range(20):
        commands.append(worker._condition_teleop_target(target, current).copy())
        clock["now"] += 0.01

    commands = np.asarray(commands)
    increments = np.diff(commands[:, active], axis=0)
    assert np.all(commands[:, active] >= 0.0)
    assert np.all(commands[:, active] < 1.0)
    assert float(np.max(np.abs(increments))) <= 0.011
    assert float(np.max(commands[:, list(WAIST_INDICES)])) <= 0.2
    assert np.allclose(commands[:, list(LEG_INDICES)], 0.0)
