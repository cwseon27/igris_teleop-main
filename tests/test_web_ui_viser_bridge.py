from __future__ import annotations

import numpy as np

from igris_teleop.web_ui.viser_bridge import WebUIViserBridge


class FakeShm:
    def __init__(self, data) -> None:
        self.data = dict(data)

    def read_data(self):
        return dict(self.data)


def test_read_q_from_shm_reorders_waist_for_urdf_cfg() -> None:
    waist_yaw_roll_pitch = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    leg = np.arange(12, dtype=np.float64) + 100.0
    arm = np.arange(14, dtype=np.float64) + 200.0
    neck = np.arange(2, dtype=np.float64) + 300.0

    bridge = WebUIViserBridge(
        {
            "act_shm": FakeShm(
                {
                    "act_waist": waist_yaw_roll_pitch,
                    "act_leg": leg,
                    "act_arm": arm,
                    "act_neck": neck,
                }
            )
        }
    )

    q = bridge._read_q_from_shm("act_shm", "act")

    assert q is not None
    np.testing.assert_array_equal(q[:3], np.array([30.0, 20.0, 10.0]))
    np.testing.assert_array_equal(q[3:15], leg)
    np.testing.assert_array_equal(q[15:29], arm)
    np.testing.assert_array_equal(q[29:31], neck)


def test_read_target_frames_requires_valid_flag_and_matrices() -> None:
    left = np.eye(4, dtype=np.float64)
    left[:3, 3] = np.array([0.1, 0.2, 0.3])
    right = np.eye(4, dtype=np.float64)
    head = np.eye(4, dtype=np.float64)
    torso = np.eye(4, dtype=np.float64)
    torso[:3, 3] = np.array([0.4, 0.0, 0.2])
    chest = np.eye(4, dtype=np.float64)
    chest[:3, 3] = np.array([0.55, 0.0, 0.2])
    invalid = np.zeros((4, 4), dtype=np.float64)

    bridge = WebUIViserBridge(
        {
            "ik_target_shm": FakeShm(
                {
                    "target_valid": np.array(1.0, dtype=np.float64),
                    "target_seq": np.array(123.0, dtype=np.float64),
                    "torso_target_valid": np.array(1.0, dtype=np.float64),
                    "torso_alpha": np.array(0.8, dtype=np.float64),
                    "chest_target_valid": np.array(1.0, dtype=np.float64),
                    "chest_alpha": np.array(0.7, dtype=np.float64),
                    "left_wrist_mat": left,
                    "right_wrist_mat": right,
                    "head_mat": head,
                    "torso_mat": torso,
                    "chest_mat": chest,
                }
            )
        }
    )

    frames = bridge._read_target_frames()

    np.testing.assert_array_equal(frames["left_wrist_mat"], left)
    np.testing.assert_array_equal(frames["right_wrist_mat"], right)
    np.testing.assert_array_equal(frames["head_mat"], head)
    np.testing.assert_array_equal(frames["torso_mat"], torso)
    np.testing.assert_array_equal(frames["chest_mat"], chest)

    bridge.shared_memory["ik_target_shm"].data["target_valid"] = np.array(0.0, dtype=np.float64)
    assert bridge._read_target_frames() == {}

    bridge.shared_memory["ik_target_shm"].data["target_valid"] = np.array(1.0, dtype=np.float64)
    bridge.shared_memory["ik_target_shm"].data["torso_target_valid"] = np.array(0.0, dtype=np.float64)
    frames = bridge._read_target_frames()
    assert "torso_mat" not in frames
    assert "chest_mat" in frames

    bridge.shared_memory["ik_target_shm"].data["chest_target_valid"] = np.array(0.0, dtype=np.float64)
    frames = bridge._read_target_frames()
    assert "chest_mat" not in frames

    bridge.shared_memory["ik_target_shm"].data["torso_target_valid"] = np.array(1.0, dtype=np.float64)
    bridge.shared_memory["ik_target_shm"].data["right_wrist_mat"] = invalid
    frames = bridge._read_target_frames()
    assert "right_wrist_mat" not in frames
    assert "left_wrist_mat" in frames


def test_rotation_matrix_to_wxyz() -> None:
    q_identity = WebUIViserBridge._rotation_matrix_to_wxyz(np.eye(3, dtype=np.float64))
    np.testing.assert_allclose(q_identity, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-9)

    rot_z_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    q_z_90 = WebUIViserBridge._rotation_matrix_to_wxyz(rot_z_90)
    np.testing.assert_allclose(
        q_z_90,
        np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
        atol=1e-9,
    )
