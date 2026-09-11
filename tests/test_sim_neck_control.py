from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from igris_teleop.robot_control.kinematics.joints import NUM_MOTORS  # noqa: E402
from igris_teleop.workers.worker_simulator import (  # noqa: E402
    ARM_INDEX_ARRAY,
    NECK_INDEX_ARRAY,
    MujocoSimulationWorker,
)


def test_neck_gravity_compensation_adds_only_neck_bias_torque() -> None:
    worker = MujocoSimulationWorker.__new__(MujocoSimulationWorker)
    bias = np.linspace(-2.0, 2.0, NUM_MOTORS, dtype=np.float32)
    worker.data = SimpleNamespace(qfrc_bias=bias.copy())
    worker._joint_dof_adrs = np.arange(NUM_MOTORS, dtype=np.int32)
    ctrl = np.zeros(NUM_MOTORS, dtype=np.float32)

    compensated = worker._apply_neck_gravity_compensation(ctrl)

    expected = np.zeros(NUM_MOTORS, dtype=np.float32)
    expected[NECK_INDEX_ARRAY] = bias[NECK_INDEX_ARRAY]
    np.testing.assert_allclose(compensated, expected)


def test_upper_body_gravity_compensation_adds_arm_and_neck_bias_torque() -> None:
    worker = MujocoSimulationWorker.__new__(MujocoSimulationWorker)
    bias = np.linspace(-2.0, 2.0, NUM_MOTORS, dtype=np.float32)
    worker.data = SimpleNamespace(qfrc_bias=bias.copy())
    worker._joint_dof_adrs = np.arange(NUM_MOTORS, dtype=np.int32)
    ctrl = np.zeros(NUM_MOTORS, dtype=np.float32)

    compensated = worker._apply_upper_body_gravity_compensation(ctrl)

    expected = np.zeros(NUM_MOTORS, dtype=np.float32)
    expected[ARM_INDEX_ARRAY] = bias[ARM_INDEX_ARRAY]
    expected[NECK_INDEX_ARRAY] = bias[NECK_INDEX_ARRAY]
    np.testing.assert_allclose(compensated, expected)


def test_sim_arm_gain_override_applies_only_arm_indices() -> None:
    worker = MujocoSimulationWorker.__new__(MujocoSimulationWorker)
    worker._sim_arm_kp = np.linspace(100.0, 230.0, ARM_INDEX_ARRAY.size, dtype=np.float32)
    worker._sim_arm_kd = np.linspace(1.0, 2.3, ARM_INDEX_ARRAY.size, dtype=np.float32)
    kp = np.zeros(NUM_MOTORS, dtype=np.float32)
    kd = np.zeros(NUM_MOTORS, dtype=np.float32)

    worker._apply_sim_arm_gains(kp, kd)

    np.testing.assert_allclose(kp[ARM_INDEX_ARRAY], worker._sim_arm_kp)
    np.testing.assert_allclose(kd[ARM_INDEX_ARRAY], worker._sim_arm_kd)
    assert np.count_nonzero(kp) == ARM_INDEX_ARRAY.size
    assert np.count_nonzero(kd) == ARM_INDEX_ARRAY.size
