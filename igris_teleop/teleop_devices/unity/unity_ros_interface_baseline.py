import threading
import time
from typing import Tuple

import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped

from .constants_baseline import (
    T_robot_openxr,
    T_to_unitree_left_wrist,
    T_to_unitree_right_wrist,
    grd_yup2grd_zup,
    lefthand2igris,
    righthand2igris,
    const_head_vuer_mat,
    const_left_wrist_vuer_mat,
    const_right_wrist_vuer_mat,
)
from ...sharedmemory.shm_schema import TELEVISION
from ...core.math.transforms import mat_update, fast_mat_inv


FINGERTIP_COUNT = 5

T_OPENXR_ROBOT = fast_mat_inv(T_robot_openxr)



def television_dtype() -> np.dtype:
    return np.dtype(TELEVISION)


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion(x,y,z,w) -> 3x3 rotation matrix (float64)."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q /= n
    x, y, z, w = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _pose_to_mat(pose) -> np.ndarray:
    """geometry_msgs/Pose -> 4x4 homogeneous transform (float64)."""
    px = float(pose.position.x)
    py = float(pose.position.y)
    pz = float(pose.position.z)

    qx = float(pose.orientation.x)
    qy = float(pose.orientation.y)
    qz = float(pose.orientation.z)
    qw = float(pose.orientation.w)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
    T[:3, 3] = np.array([px, py, pz], dtype=np.float64)
    return T


def _pad_or_trunc_hand(xyz: np.ndarray, count: int = FINGERTIP_COUNT) -> np.ndarray:
    """
    xyz: (N,3) -> (count,3)로 맞춤.
    부족하면 0 패딩, 많으면 앞에서 count개만 사용.
    """
    out = np.zeros((count, 3), dtype=np.float64)
    if xyz.size == 0:
        return out
    n = min(xyz.shape[0], count)
    out[:n, :] = xyz[:n, :]
    return out


class TelevisionROSInterface(Node):
    """
    TELEVISION SHM 스키마에 바로 쓸 수 있도록
    내부 상태를 (4x4,4x4,4x4,4x4,5x3,5x3) 고정 shape numpy 배열로 유지.
    """

    def __init__(
        self,
        qos_depth: int = 10,
    ) -> None:
        super().__init__("television_bridge")

        # -------------------- subscribers (fixed topics) --------------------
        self.create_subscription(PoseStamped, "/hmd/pose", self._on_head_pose, qos_depth)
        self.create_subscription(PoseStamped, "/right_controller/pose", self._on_torso_pose, qos_depth)
        self.create_subscription(PoseArray, "/left_hand/poses", self._on_left_wrist_poses, qos_depth)
        self.create_subscription(PoseArray, "/right_hand/poses", self._on_right_wrist_poses, qos_depth)

        # -------------------- locks/events --------------------
        self._lock = threading.Lock()

        self._has_head = threading.Event()
        self._has_left_wrist = threading.Event()
        self._has_right_wrist = threading.Event()
        self._has_left_hand = threading.Event()
        self._has_right_hand = threading.Event()

        self._prefer_combined_right = False

        # -------------------- internal buffers (raw OpenXR-like) --------------------
        self._head_vuer_mat = const_head_vuer_mat.copy()
        self._torso_vuer_mat = np.eye(4, dtype=np.float64)
        self._left_wrist_vuer_mat = const_left_wrist_vuer_mat.copy()
        self._right_wrist_vuer_mat = const_right_wrist_vuer_mat.copy()

        self._left_hand_vuer = np.zeros((FINGERTIP_COUNT, 3), dtype=np.float64)
        self._right_hand_vuer = np.zeros((FINGERTIP_COUNT, 3), dtype=np.float64)
        self._last_left_hand = np.zeros((FINGERTIP_COUNT, 3), dtype=np.float64)
        self._last_right_hand = np.zeros((FINGERTIP_COUNT, 3), dtype=np.float64)
        self._last_left_wrist = np.eye(4, dtype=np.float64)
        self._last_right_wrist = np.eye(4, dtype=np.float64)
        self._last_wrist_log_ts = 0.0
        self._wrist_log_interval = 1.0

    # ----------------------------- callbacks -----------------------------
    def _on_head_pose(self, msg: PoseStamped) -> None:
        T = _pose_to_mat(msg.pose)
        with self._lock:
            self._head_vuer_mat = T
        self._has_head.set()

    def _on_torso_pose(self, msg: PoseStamped) -> None:
        T = _pose_to_mat(msg.pose)
        with self._lock:
            self._torso_vuer_mat = T


    def _on_left_wrist_poses(self, msg: PoseArray) -> None:
        poses = msg.poses if msg.poses is not None else []
        if len(poses) == 0:
            return

        wrist_pose = poses[0]
        T = _pose_to_mat(wrist_pose)

        xyz_raw = np.array(
            [[float(p.position.x), float(p.position.y), float(p.position.z)] for p in poses[1:]],
            dtype=np.float64,
        )
        xyz = _pad_or_trunc_hand(xyz_raw, FINGERTIP_COUNT)

        with self._lock:
            self._left_wrist_vuer_mat = T
            self._left_hand_vuer = xyz

        self._has_left_wrist.set()
        if len(poses) > 1:
            self._has_left_hand.set()

    def _on_right_wrist_poses(self, msg: PoseArray) -> None:
        poses = msg.poses if msg.poses is not None else []
        if len(poses) == 0:
            return
        self._prefer_combined_right = True

        wrist_pose = poses[0]
        T = _pose_to_mat(wrist_pose)

        xyz_raw = np.array(
            [[float(p.position.x), float(p.position.y), float(p.position.z)] for p in poses[1:]],
            dtype=np.float64,
        )
        xyz = _pad_or_trunc_hand(xyz_raw, FINGERTIP_COUNT)

        with self._lock:
            self._right_wrist_vuer_mat = T
            self._right_hand_vuer = xyz

        self._has_right_wrist.set()
        if len(poses) > 1:
            self._has_right_hand.set()



    # ----------------------------- waiters -----------------------------
    def wait_for_all(self, timeout: float = 5.0) -> bool:
        """
        head + left/right wrist + left/right hand 모두 최소 1회 수신될 때까지 대기.
        """
        ok_head = self._has_head.wait(timeout=timeout)
        ok_lw = self._has_left_wrist.wait(timeout=timeout)
        ok_rw = self._has_right_wrist.wait(timeout=timeout)
        ok_lh = self._has_left_hand.wait(timeout=timeout)
        ok_rh = self._has_right_hand.wait(timeout=timeout)
        return bool(ok_head and ok_lw and ok_rw and ok_lh and ok_rh)

    def wait_for_control_streams(self, timeout: float = 5.0) -> bool:
        """Wait for pose streams without making hand tracking a startup lock.

        OpenXR may publish a hand only after it becomes tracked.  Requiring
        both fingertip streams before forwarding any Unity data stranded the
        bridge when one hand was temporarily untracked; hand samples can join
        later through the existing callbacks.
        """
        ok_head = self._has_head.wait(timeout=timeout)
        ok_lw = self._has_left_wrist.wait(timeout=timeout)
        ok_rw = self._has_right_wrist.wait(timeout=timeout)
        return bool(ok_head and ok_lw and ok_rw)

    # ----------------------------- SHM-friendly getters -----------------------------
    def get_television_payload(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        TELEVISION 스키마 순서 그대로 tuple 반환:
        (head_mat, torso_mat, left_wrist_mat, right_wrist_mat, left_hand, right_hand)
        """
        with self._lock:
            head_vuer_mat = self._head_vuer_mat.copy()
            torso_vuer_mat = self._torso_vuer_mat.copy()
            left_wrist_vuer_mat = self._left_wrist_vuer_mat.copy()
            right_wrist_vuer_mat = self._right_wrist_vuer_mat.copy()
            left_hand_vuer = self._left_hand_vuer.copy()
            right_hand_vuer = self._right_hand_vuer.copy()

        head_vuer_mat, _ = mat_update(const_head_vuer_mat, head_vuer_mat)
        left_wrist_vuer_mat, left_wrist_flag = mat_update(const_left_wrist_vuer_mat, left_wrist_vuer_mat)
        right_wrist_vuer_mat, right_wrist_flag = mat_update(const_right_wrist_vuer_mat, right_wrist_vuer_mat)

        now = time.monotonic()
        if (now - self._last_wrist_log_ts) >= self._wrist_log_interval:
            self._last_wrist_log_ts = now
            left_str = np.array2string(left_wrist_vuer_mat, precision=3, suppress_small=True)
            right_str = np.array2string(right_wrist_vuer_mat, precision=3, suppress_small=True)
            # self.get_logger().info(
            #     f"[television_bridge] wrist pose (L={left_wrist_flag} R={right_wrist_flag}). "
            #     f"left_wrist_vuer_mat={left_str} right_wrist_vuer_mat={right_str}"
            # )

        head_mat = T_robot_openxr @ head_vuer_mat @ T_OPENXR_ROBOT
        torso_mat = T_robot_openxr @ torso_vuer_mat @ T_OPENXR_ROBOT
        left_wrist_mat = T_robot_openxr @ left_wrist_vuer_mat @ T_OPENXR_ROBOT
        right_wrist_mat = T_robot_openxr @ right_wrist_vuer_mat @ T_OPENXR_ROBOT

        unitree_left_wrist = left_wrist_mat @ (
            T_to_unitree_left_wrist if left_wrist_flag else np.eye(4, dtype=np.float64)
        )
        unitree_right_wrist = right_wrist_mat @ (
            T_to_unitree_right_wrist if right_wrist_flag else np.eye(4, dtype=np.float64)
        )

        left_hand_vuer_mat = np.concatenate(
            [left_hand_vuer.T, np.ones((1, left_hand_vuer.shape[0]), dtype=np.float64)]
        )
        right_hand_vuer_mat = np.concatenate(
            [right_hand_vuer.T, np.ones((1, right_hand_vuer.shape[0]), dtype=np.float64)]
        )

        left_hand_mat = grd_yup2grd_zup @ left_hand_vuer_mat
        right_hand_mat = grd_yup2grd_zup @ right_hand_vuer_mat

        # /left_hand/fingertips_rel, /right_hand/fingertips_rel are already wrist-relative.
        left_hand_mat_wb = left_hand_mat
        right_hand_mat_wb = right_hand_mat

        unitree_left_hand = (lefthand2igris.T @ left_hand_mat_wb)[:3, :].T
        unitree_right_hand = (righthand2igris.T @ right_hand_mat_wb)[:3, :].T

        # 손 유효성은 hand 값이 아니라 wrist pose로 판단
        # Unity가 추적을 잃으면 wrist가 identity(=const_*)로 들어오는 케이스가 있어 이를 invalid로 처리
        left_is_default = np.allclose(left_wrist_vuer_mat, const_left_wrist_vuer_mat, atol=1e-6) or np.allclose(
            left_wrist_vuer_mat, np.eye(4, dtype=np.float64), atol=1e-6
        )
        right_is_default = np.allclose(right_wrist_vuer_mat, const_right_wrist_vuer_mat, atol=1e-6) or np.allclose(
            right_wrist_vuer_mat, np.eye(4, dtype=np.float64), atol=1e-6
        )
        left_valid = bool(left_wrist_flag) and (not left_is_default)
        right_valid = bool(right_wrist_flag) and (not right_is_default)
        with self._lock:
            if left_valid:
                self._last_left_wrist = unitree_left_wrist.copy()
            else:
                unitree_left_wrist = self._last_left_wrist.copy()
            if right_valid:
                self._last_right_wrist = unitree_right_wrist.copy()
            else:
                unitree_right_wrist = self._last_right_wrist.copy()
            if left_valid:
                self._last_left_hand = unitree_left_hand.copy()
            else:
                unitree_left_hand = self._last_left_hand.copy()
            if right_valid:
                self._last_right_hand = unitree_right_hand.copy()
            else:
                unitree_right_hand = self._last_right_hand.copy()

        return (
            head_mat,
            torso_mat,
            unitree_left_wrist,
            unitree_right_wrist,
            unitree_left_hand,
            unitree_right_hand,
        )

    def get_television_struct(self) -> np.ndarray:
        """
        np.dtype(TELEVISION) 기반 structured array (1,) 반환.
        SHM writer가 dtype 동일하게 잡아두면 out 전체를 바로 assign/memcpy 가능.
        """
        head_mat, torso_mat, left_wrist_mat, right_wrist_mat, left_hand, right_hand = self.get_television_payload()
        dt = television_dtype()
        out = np.zeros((1,), dtype=dt)
        out["head_mat"][0] = head_mat
        out["torso_mat"][0] = torso_mat
        out["left_wrist_mat"][0] = left_wrist_mat
        out["right_wrist_mat"][0] = right_wrist_mat
        out["left_hand"][0] = left_hand
        out["right_hand"][0] = right_hand
        return out
