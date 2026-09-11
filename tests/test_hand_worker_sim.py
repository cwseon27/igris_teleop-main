from __future__ import annotations

import numpy as np

from igris_teleop.workers.worker_hand import HandWorker


class FakeShm:
    def __init__(self, data=None) -> None:
        self.data = dict(data or {})
        self.writes: list[dict] = []

    def read_data(self):
        return dict(self.data)

    def write_data(self, **kwargs) -> None:
        self.writes.append(kwargs)
        self.data.update(kwargs)


class FakeRetargeting:
    def __init__(self, target) -> None:
        self.target = np.asarray(target, dtype=np.float64)
        self.references = None

    def retarget_normalized(self, left, right):
        self.references = (np.asarray(left).copy(), np.asarray(right).copy())
        return self.target.copy()


def test_sim_vr_hand_target_is_written_to_act_shm_as_normalized_bend() -> None:
    television_shm = FakeShm(
        {
            "left_hand": np.arange(15, dtype=np.float64).reshape(5, 3),
            "right_hand": np.arange(15, 30, dtype=np.float64).reshape(5, 3),
        }
    )
    act_shm = FakeShm()
    retargeting = FakeRetargeting(np.linspace(-0.2, 1.2, 12))
    worker = HandWorker.__new__(HandWorker)
    worker.sim_hand_retargeting = retargeting
    worker.television_shm = television_shm
    worker.act_shm = act_shm
    worker._last_sim_retarget_log_at = float("inf")
    worker._last_sim_retarget_error_at = 0.0

    worker._publish_sim_vr_hand_target()

    expected = np.clip(np.linspace(-0.2, 1.2, 12), 0.0, 1.0)
    assert np.allclose(act_shm.data["act_hand"], expected)
    assert retargeting.references[0].shape == (5, 3)
    assert retargeting.references[1].shape == (5, 3)


def test_sim_hybrid_hand_uses_direct_normalized_command_without_retargeting() -> None:
    television_shm = FakeShm(
        {
            "left_hand": np.zeros((5, 3), dtype=np.float64),
            "right_hand": np.zeros((5, 3), dtype=np.float64),
            "left_hand_close": np.array([0.0, 0.2, 0.4, 0.6, 0.8]),
            "right_hand_close": np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
            "left_hand_close_valid": 1.0,
            "right_hand_close_valid": 1.0,
        }
    )
    act_shm = FakeShm()
    retargeting = FakeRetargeting(np.ones(12))
    worker = HandWorker.__new__(HandWorker)
    worker.sim_hand_retargeting = retargeting
    worker.television_shm = television_shm
    worker.act_shm = act_shm
    worker._last_sim_retarget_log_at = float("inf")
    worker._last_sim_retarget_error_at = 0.0

    worker._publish_sim_vr_hand_target()

    expected_right = np.clip(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.1]) * 1.5, 0.0, 1.0)
    expected_left = np.clip(np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.0]) * 1.5, 0.0, 1.0)
    assert np.allclose(act_shm.data["act_hand"], np.concatenate((expected_right, expected_left)))
    assert retargeting.references is None


def test_sim_hybrid_hand_can_override_one_side_only() -> None:
    television_shm = FakeShm(
        {
            "left_hand": np.zeros((5, 3), dtype=np.float64),
            "right_hand": np.ones((5, 3), dtype=np.float64),
            "left_hand_close": np.zeros(5),
            "right_hand_close": np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
            "left_hand_close_valid": 0.0,
            "right_hand_close_valid": 1.0,
        }
    )
    act_shm = FakeShm()
    retargeted = np.linspace(0.0, 0.55, 12)
    retargeting = FakeRetargeting(retargeted)
    worker = HandWorker.__new__(HandWorker)
    worker.sim_hand_retargeting = retargeting
    worker.television_shm = television_shm
    worker.act_shm = act_shm
    worker._last_sim_retarget_log_at = float("inf")
    worker._last_sim_retarget_error_at = 0.0

    worker._publish_sim_vr_hand_target()

    expected_right = np.clip(np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.2]) * 1.5, 0.0, 1.0)
    assert np.allclose(act_shm.data["act_hand"][:6], expected_right)
    assert np.allclose(act_shm.data["act_hand"][6:], retargeted[6:])
    assert retargeting.references is not None


def test_sim_final_motor_commands_take_precedence_without_duplicate_gain_or_thumb() -> None:
    left, right = np.linspace(0.1, 0.6, 6), np.linspace(0.2, 0.7, 6)
    worker = HandWorker.__new__(HandWorker)
    worker.sim_hand_retargeting = FakeRetargeting(np.ones(12))
    worker.television_shm = FakeShm({
        "left_hand": np.zeros((5, 3)), "right_hand": np.zeros((5, 3)),
        "left_hand_close": np.ones(5), "right_hand_close": np.ones(5),
        "left_hand_close_valid": 1.0, "right_hand_close_valid": 1.0,
        "left_hand_motor": left, "right_hand_motor": right,
        "left_hand_motor_valid": 1.0, "right_hand_motor_valid": 1.0,
    })
    worker.act_shm = FakeShm()
    worker._last_sim_retarget_log_at = float("inf")
    worker._last_sim_retarget_error_at = 0.0
    worker._publish_sim_vr_hand_target()
    np.testing.assert_array_equal(worker.act_shm.data["act_hand"], np.r_[right, left])
    assert worker.sim_hand_retargeting.references is None
