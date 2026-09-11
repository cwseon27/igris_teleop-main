import os
import threading
import time
from typing import Tuple

import numpy as np

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Bool, Float32, Float32MultiArray

from .constants import (
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
from .reliability_fusion import (
    DEFAULT_CONTROLLER_CONFIDENCE_FALLBACK,
    DEFAULT_CONTROLLER_TRACKED_CONFIDENCE_FLOOR,
    DEFAULT_CONTROLLER_UNTRACKED_POSE_CONFIDENCE_FLOOR,
    ReliabilityAwareTorsoFusion,
    select_tracking_confidence,
    transform_from_xyz_xyzw,
)
from .pose_source import should_accept_pose_source
from .hand_source import hybrid_hand_pose_is_usable
from .hybrid_motor_command import HybridMotorCommand
from .hmd_pose import openxr_hmd_pose_to_robot


FINGERTIP_COUNT = 5

DEFAULT_LEFT_CONTROLLER_POSE_TOPIC = "/left_controller/poses"
DEFAULT_RIGHT_CONTROLLER_POSE_TOPIC = "/right_controller/poses"
LEGACY_LEFT_CONTROLLER_POSE_TOPIC = "/left_controller/pose"
LEGACY_RIGHT_CONTROLLER_POSE_TOPIC = "/right_controller/pose"

T_OPENXR_ROBOT = fast_mat_inv(T_robot_openxr)


def _transform_parameter_default(name: str, default: list[float]) -> list[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    values = [float(value.strip()) for value in raw.split(",")]
    if len(values) != 7:
        raise ValueError(
            f"{name} must contain x,y,z,qx,qy,qz,qw (7 comma-separated values)"
        )
    return values


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
        *,
        hybrid_torso_enabled: bool = False,
    ) -> None:
        super().__init__("television_bridge")
        self._hybrid_torso_enabled = bool(hybrid_torso_enabled)

        self.declare_parameter(
            "left_controller_pose_topic", DEFAULT_LEFT_CONTROLLER_POSE_TOPIC
        )
        self.declare_parameter(
            "right_controller_pose_topic", DEFAULT_RIGHT_CONTROLLER_POSE_TOPIC
        )
        self.declare_parameter(
            "left_controller_confidence_topic", "/left_controller/tracking_confidence"
        )
        self.declare_parameter(
            "right_controller_confidence_topic", "/right_controller/tracking_confidence"
        )
        self.declare_parameter("left_controller_tracked_topic", "/left_controller/is_tracked")
        self.declare_parameter("right_controller_tracked_topic", "/right_controller/is_tracked")
        self.declare_parameter(
            "left_controller_to_chest",
            _transform_parameter_default(
                "IGRIS_LEFT_CONTROLLER_TO_CHEST",
                [0.064, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ),
        )
        self.declare_parameter(
            "right_controller_to_chest",
            _transform_parameter_default(
                "IGRIS_RIGHT_CONTROLLER_TO_CHEST",
                [-0.064, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ),
        )
        self.declare_parameter("controller_pose_stale_sec", 0.2)
        self.declare_parameter("controller_confidence_stale_sec", 0.5)
        self.declare_parameter("hybrid_hand_pose_stale_sec", 0.25)
        self.declare_parameter(
            "controller_confidence_fallback",
            DEFAULT_CONTROLLER_CONFIDENCE_FALLBACK,
        )
        self.declare_parameter("controller_confidence_power", 1.0)
        self.declare_parameter(
            "controller_tracked_confidence_floor",
            DEFAULT_CONTROLLER_TRACKED_CONFIDENCE_FLOOR,
        )
        self.declare_parameter(
            "controller_untracked_pose_confidence_floor",
            DEFAULT_CONTROLLER_UNTRACKED_POSE_CONFIDENCE_FLOOR,
        )
        self.declare_parameter("controller_minimum_confidence_sum", 0.05)
        self.declare_parameter("controller_consistency_position_sigma_m", 0.15)
        self.declare_parameter("controller_consistency_rotation_sigma_deg", 25.0)
        self.declare_parameter("torso_alpha_rate_up_per_sec", 6.0)
        self.declare_parameter("torso_alpha_rate_down_per_sec", 10.0)

        left_controller_pose_topic = str(self.get_parameter("left_controller_pose_topic").value)
        right_controller_pose_topic = str(self.get_parameter("right_controller_pose_topic").value)
        left_confidence_topic = str(self.get_parameter("left_controller_confidence_topic").value)
        right_confidence_topic = str(self.get_parameter("right_controller_confidence_topic").value)
        left_tracked_topic = str(self.get_parameter("left_controller_tracked_topic").value)
        right_tracked_topic = str(self.get_parameter("right_controller_tracked_topic").value)
        controller_pose_topics = {
            "left": tuple(
                dict.fromkeys(
                    (
                        left_controller_pose_topic,
                        DEFAULT_LEFT_CONTROLLER_POSE_TOPIC,
                        LEGACY_LEFT_CONTROLLER_POSE_TOPIC,
                    )
                )
            ),
            "right": tuple(
                dict.fromkeys(
                    (
                        right_controller_pose_topic,
                        DEFAULT_RIGHT_CONTROLLER_POSE_TOPIC,
                        LEGACY_RIGHT_CONTROLLER_POSE_TOPIC,
                    )
                )
            ),
        }
        self._controller_preferred_topic = {
            "left": left_controller_pose_topic,
            "right": right_controller_pose_topic,
        }
        self._controller_topic_last_seen = {
            side: {topic: None for topic in topics}
            for side, topics in controller_pose_topics.items()
        }
        self._controller_active_topic = {"left": None, "right": None}
        self._controller_ignored_count = {"left": 0, "right": 0}
        self._source_frame_id = {
            "head": None,
            "left_controller": None,
            "right_controller": None,
        }
        stream_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, int(qos_depth)),
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._controller_pose_stale_sec = max(
            0.0, float(self.get_parameter("controller_pose_stale_sec").value)
        )
        self._controller_confidence_stale_sec = max(
            0.0, float(self.get_parameter("controller_confidence_stale_sec").value)
        )
        self._hybrid_hand_pose_stale_sec = max(
            0.0, float(self.get_parameter("hybrid_hand_pose_stale_sec").value)
        )
        self._controller_confidence_fallback = max(
            0.0,
            min(1.0, float(self.get_parameter("controller_confidence_fallback").value)),
        )
        self._controller_tracked_confidence_floor = max(
            0.0,
            min(
                1.0,
                float(self.get_parameter("controller_tracked_confidence_floor").value),
            ),
        )
        self._controller_untracked_pose_confidence_floor = max(
            0.0,
            min(
                1.0,
                float(
                    self.get_parameter("controller_untracked_pose_confidence_floor").value
                ),
            ),
        )
        self._torso_fusion = ReliabilityAwareTorsoFusion(
            left_to_chest=transform_from_xyz_xyzw(
                self.get_parameter("left_controller_to_chest").value
            ),
            right_to_chest=transform_from_xyz_xyzw(
                self.get_parameter("right_controller_to_chest").value
            ),
            confidence_power=float(self.get_parameter("controller_confidence_power").value),
            minimum_confidence_sum=float(
                self.get_parameter("controller_minimum_confidence_sum").value
            ),
            position_sigma_m=float(
                self.get_parameter("controller_consistency_position_sigma_m").value
            ),
            rotation_sigma_rad=np.deg2rad(
                float(self.get_parameter("controller_consistency_rotation_sigma_deg").value)
            ),
            alpha_rate_up_per_sec=float(
                self.get_parameter("torso_alpha_rate_up_per_sec").value
            ),
            alpha_rate_down_per_sec=float(
                self.get_parameter("torso_alpha_rate_down_per_sec").value
            ),
            nominal_rate_hz=60.0,
        )
        self.get_logger().info(
            f"chest hybrid={self._hybrid_torso_enabled} "
            f"confidence_fallback={self._controller_confidence_fallback:.3f} "
            f"tracked_floor={self._controller_tracked_confidence_floor:.3f} "
            f"untracked_pose_floor={self._controller_untracked_pose_confidence_floor:.3f} "
            f"controller_topics={controller_pose_topics}"
        )
        self.get_logger().info(
            "pose coordinates: raw OpenXR/Unity -> robot basis once "
            "(robot xyz = -OpenXR z, -OpenXR x, +OpenXR y); "
            "controller fusion runs before this basis conversion"
        )

        # -------------------- subscribers --------------------
        self.create_subscription(PoseStamped, "/hmd/pose", self._on_head_pose, stream_qos)
        for side, topics in controller_pose_topics.items():
            for topic in topics:
                self.create_subscription(
                    PoseStamped,
                    topic,
                    lambda msg, controller_side=side, source_topic=topic: self._on_controller_pose(
                        controller_side, msg, source_topic
                    ),
                    stream_qos,
                )
        self.create_subscription(
            Float32,
            left_confidence_topic,
            lambda msg: self._on_controller_confidence("left", msg),
            stream_qos,
        )
        self.create_subscription(
            Float32,
            right_confidence_topic,
            lambda msg: self._on_controller_confidence("right", msg),
            stream_qos,
        )
        self.create_subscription(
            Bool,
            left_tracked_topic,
            lambda msg: self._on_controller_tracked("left", msg),
            stream_qos,
        )
        self.create_subscription(
            Bool,
            right_tracked_topic,
            lambda msg: self._on_controller_tracked("right", msg),
            stream_qos,
        )
        self.create_subscription(PoseArray, "/left_hand/poses", self._on_left_wrist_poses, stream_qos)
        self.create_subscription(PoseArray, "/right_hand/poses", self._on_right_wrist_poses, stream_qos)
        self.create_subscription(
            PoseArray, "/left_hybrid_hand/poses", self._on_left_hybrid_hand_poses, stream_qos
        )
        self.create_subscription(
            PoseArray, "/right_hybrid_hand/poses", self._on_right_hybrid_hand_poses, stream_qos
        )
        self.create_subscription(
            Float32MultiArray,
            "/left_hybrid_hand/finger_normalized",
            lambda msg: self._on_hybrid_hand_close("left", msg),
            stream_qos,
        )
        self.create_subscription(
            Float32MultiArray,
            "/right_hybrid_hand/finger_normalized",
            lambda msg: self._on_hybrid_hand_close("right", msg),
            stream_qos,
        )
        self.create_subscription(
            Bool,
            "/left_hybrid_hand/is_tracked",
            lambda msg: self._on_hybrid_hand_tracked("left", msg),
            stream_qos,
        )
        self.create_subscription(
            Bool,
            "/right_hybrid_hand/is_tracked",
            lambda msg: self._on_hybrid_hand_tracked("right", msg),
            stream_qos,
        )

        # -------------------- locks/events --------------------
        self._lock = threading.Lock()
        self._hybrid_hand_motor = {
            side: HybridMotorCommand(self._hybrid_hand_pose_stale_sec)
            for side in ("left", "right")
        }
        for side in ("left", "right"):
            self.create_subscription(
                Float32MultiArray, f"/{side}_hybrid_hand/motor_normalized",
                lambda msg, side=side: self._on_hybrid_hand_motor(side, msg), stream_qos,
            )
            self.create_subscription(
                Bool, f"/{side}_hybrid_hand/motor_tracked",
                lambda msg, side=side: self._on_hybrid_hand_motor_tracked(side, msg), stream_qos,
            )

        self._has_head = threading.Event()
        self._has_left_wrist = threading.Event()
        self._has_right_wrist = threading.Event()
        self._has_left_hand = threading.Event()
        self._has_right_hand = threading.Event()

        self._prefer_combined_right = False

        # -------------------- internal buffers (raw OpenXR-like) --------------------
        self._head_vuer_mat = const_head_vuer_mat.copy()
        self._chest_vuer_mat = np.eye(4, dtype=np.float64)
        self._controller_pose = {"left": None, "right": None}
        self._controller_pose_time = {"left": None, "right": None}
        self._controller_confidence = {"left": 0.0, "right": 0.0}
        self._controller_confidence_time = {"left": None, "right": None}
        self._controller_tracked = {"left": None, "right": None}
        self._controller_tracked_time = {"left": None, "right": None}
        self._chest_alpha = 0.0
        self._chest_left_confidence = 0.0
        self._chest_right_confidence = 0.0
        self._chest_pose_ready = False
        self._chest_last_log_ts = 0.0
        self._left_wrist_vuer_mat = const_left_wrist_vuer_mat.copy()
        self._right_wrist_vuer_mat = const_right_wrist_vuer_mat.copy()

        self._left_hand_vuer = np.zeros((FINGERTIP_COUNT, 3), dtype=np.float64)
        self._right_hand_vuer = np.zeros((FINGERTIP_COUNT, 3), dtype=np.float64)
        self._left_hybrid_hand_vuer = np.zeros((FINGERTIP_COUNT, 3), dtype=np.float64)
        self._right_hybrid_hand_vuer = np.zeros((FINGERTIP_COUNT, 3), dtype=np.float64)
        self._has_left_hybrid_hand = False
        self._has_right_hybrid_hand = False
        self._hybrid_hand_pose_time = {"left": None, "right": None}
        self._hybrid_hand_tracked = {"left": None, "right": None}
        self._hybrid_hand_tracked_time = {"left": None, "right": None}
        self._hybrid_hand_close = {
            "left": np.zeros(FINGERTIP_COUNT, dtype=np.float64),
            "right": np.zeros(FINGERTIP_COUNT, dtype=np.float64),
        }
        self._hybrid_hand_close_time = {"left": None, "right": None}
        self._has_hybrid_hand_close = {"left": False, "right": False}
        self._hybrid_hand_command_activated = {"left": False, "right": False}
        self._active_hand_source = {"left": "openxr", "right": "openxr"}
        self._last_left_hand = np.zeros((FINGERTIP_COUNT, 3), dtype=np.float64)
        self._last_right_hand = np.zeros((FINGERTIP_COUNT, 3), dtype=np.float64)
        self._last_left_wrist = np.eye(4, dtype=np.float64)
        self._last_right_wrist = np.eye(4, dtype=np.float64)
        self._stream_counts = {
            "head": 0,
            "left_controller": 0,
            "right_controller": 0,
            "left_wrist": 0,
            "right_wrist": 0,
            "left_hybrid_hand": 0,
            "right_hybrid_hand": 0,
        }
        self._stream_times = {key: None for key in self._stream_counts}
        self._last_wrist_log_ts = 0.0
        self._wrist_log_interval = 1.0

    # ----------------------------- callbacks -----------------------------
    def _record_source_frame(self, key: str, frame_id: str) -> None:
        normalized = str(frame_id or "<empty>")
        with self._lock:
            previous = self._source_frame_id.get(key)
            if previous == normalized:
                return
            self._source_frame_id[key] = normalized
        if previous is None:
            self.get_logger().info(f"pose frame {key}={normalized}")
        else:
            self.get_logger().warning(
                f"pose frame changed for {key}: {previous} -> {normalized}"
            )

    def _on_head_pose(self, msg: PoseStamped) -> None:
        T = _pose_to_mat(msg.pose)
        self._record_source_frame("head", msg.header.frame_id)
        with self._lock:
            self._head_vuer_mat = T
            self._stream_counts["head"] += 1
            self._stream_times["head"] = time.monotonic()
        self._has_head.set()

    def _on_controller_pose(
        self,
        side: str,
        msg: PoseStamped,
        source_topic: str | None = None,
    ) -> None:
        T = _pose_to_mat(msg.pose)
        now = time.monotonic()
        topic = str(source_topic or self._controller_preferred_topic[side])
        source_changed = False
        with self._lock:
            topic_last_seen = self._controller_topic_last_seen[side]
            topic_last_seen[topic] = now
            if not should_accept_pose_source(
                preferred_topic=self._controller_preferred_topic[side],
                incoming_topic=topic,
                topic_last_seen=topic_last_seen,
                now=now,
                stale_after_sec=self._controller_pose_stale_sec,
            ):
                self._controller_ignored_count[side] += 1
                return
            source_changed = self._controller_active_topic[side] != topic
            self._controller_active_topic[side] = topic
            self._controller_pose[side] = T
            self._controller_pose_time[side] = now
            key = f"{side}_controller"
            if key in self._stream_counts:
                self._stream_counts[key] += 1
                self._stream_times[key] = self._controller_pose_time[side]
        self._record_source_frame(f"{side}_controller", msg.header.frame_id)
        if source_changed:
            self.get_logger().info(
                f"controller pose source {side}={topic} "
                f"(preferred={self._controller_preferred_topic[side]})"
            )

    def _on_torso_pose(self, msg: PoseStamped) -> None:
        """Backward-compatible alias for the former right-controller callback."""
        self._on_controller_pose("right", msg, self._controller_preferred_topic["right"])

    def _on_controller_confidence(self, side: str, msg: Float32) -> None:
        with self._lock:
            self._controller_confidence[side] = float(msg.data)
            self._controller_confidence_time[side] = time.monotonic()

    def _on_controller_tracked(self, side: str, msg: Bool) -> None:
        with self._lock:
            self._controller_tracked[side] = bool(msg.data)
            self._controller_tracked_time[side] = time.monotonic()


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
            self._stream_counts["left_wrist"] += 1
            self._stream_times["left_wrist"] = time.monotonic()

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
            self._stream_counts["right_wrist"] += 1
            self._stream_times["right_wrist"] = time.monotonic()

        self._has_right_wrist.set()
        if len(poses) > 1:
            self._has_right_hand.set()

    def _on_left_hybrid_hand_poses(self, msg: PoseArray) -> None:
        poses = msg.poses if msg.poses is not None else []
        if len(poses) < 2:
            return
        now = time.monotonic()
        xyz_raw = np.asarray(
            [[float(p.position.x), float(p.position.y), float(p.position.z)] for p in poses[1:]],
            dtype=np.float64,
        )
        with self._lock:
            self._left_hybrid_hand_vuer = _pad_or_trunc_hand(xyz_raw, FINGERTIP_COUNT)
            self._has_left_hybrid_hand = True
            self._hybrid_hand_pose_time["left"] = now
            self._stream_counts["left_hybrid_hand"] += 1
            self._stream_times["left_hybrid_hand"] = now

    def _on_right_hybrid_hand_poses(self, msg: PoseArray) -> None:
        poses = msg.poses if msg.poses is not None else []
        if len(poses) < 2:
            return
        now = time.monotonic()
        xyz_raw = np.asarray(
            [[float(p.position.x), float(p.position.y), float(p.position.z)] for p in poses[1:]],
            dtype=np.float64,
        )
        with self._lock:
            self._right_hybrid_hand_vuer = _pad_or_trunc_hand(xyz_raw, FINGERTIP_COUNT)
            self._has_right_hybrid_hand = True
            self._hybrid_hand_pose_time["right"] = now
            self._stream_counts["right_hybrid_hand"] += 1
            self._stream_times["right_hybrid_hand"] = now

    def _on_hybrid_hand_tracked(self, side: str, msg: Bool) -> None:
        now = time.monotonic()
        with self._lock:
            self._hybrid_hand_tracked[side] = bool(msg.data)
            self._hybrid_hand_tracked_time[side] = now
            if bool(msg.data):
                self._hybrid_hand_command_activated[side] = True

    def _on_hybrid_hand_close(self, side: str, msg: Float32MultiArray) -> None:
        close = np.asarray(msg.data, dtype=np.float64).reshape(-1)
        if close.size != FINGERTIP_COUNT or not np.all(np.isfinite(close)):
            self.get_logger().warning(
                f"dropping invalid {side} normalized hand command shape={close.shape}"
            )
            return
        now = time.monotonic()
        with self._lock:
            self._hybrid_hand_close[side] = np.clip(close, 0.0, 1.0)
            self._hybrid_hand_close_time[side] = now
            self._has_hybrid_hand_close[side] = True

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

    def has_head_stream(self) -> bool:
        return self._has_head.is_set()

    def get_stream_status(self) -> dict[str, tuple[int, float | None]]:
        now = time.monotonic()
        with self._lock:
            counts = dict(self._stream_counts)
            times = dict(self._stream_times)
        return {
            key: (
                int(counts.get(key, 0)),
                None if times.get(key) is None else float(now - float(times[key])),
            )
            for key in counts
        }

    # ----------------------------- SHM-friendly getters -----------------------------
    def get_television_payload(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        TELEVISION 스키마 순서 그대로 tuple 반환:
        (head_mat, chest_mat, left_wrist_mat, right_wrist_mat, left_hand, right_hand)
        """
        with self._lock:
            head_vuer_mat = self._head_vuer_mat.copy()
            chest_vuer_mat = self._chest_vuer_mat.copy()
            left_wrist_vuer_mat = self._left_wrist_vuer_mat.copy()
            right_wrist_vuer_mat = self._right_wrist_vuer_mat.copy()
            left_hand_vuer = self._left_hand_vuer.copy()
            right_hand_vuer = self._right_hand_vuer.copy()
            left_hybrid_hand_vuer = self._left_hybrid_hand_vuer.copy()
            right_hybrid_hand_vuer = self._right_hybrid_hand_vuer.copy()
            hybrid_hand_received = {
                "left": self._has_left_hybrid_hand,
                "right": self._has_right_hybrid_hand,
            }
            hybrid_hand_pose_time = self._hybrid_hand_pose_time.copy()
            hybrid_hand_tracked = self._hybrid_hand_tracked.copy()
            hybrid_hand_tracked_time = self._hybrid_hand_tracked_time.copy()
            controller_pose = {
                side: None if self._controller_pose[side] is None else self._controller_pose[side].copy()
                for side in ("left", "right")
            }
            controller_pose_time = self._controller_pose_time.copy()
            controller_confidence = self._controller_confidence.copy()
            controller_confidence_time = self._controller_confidence_time.copy()
            controller_tracked = self._controller_tracked.copy()
            controller_tracked_time = self._controller_tracked_time.copy()

        now_mono = time.monotonic()
        hybrid_hand_ready = {
            side: hybrid_hand_pose_is_usable(
                pose_received=hybrid_hand_received[side],
                pose_timestamp=hybrid_hand_pose_time[side],
                tracked=hybrid_hand_tracked[side],
                tracked_timestamp=hybrid_hand_tracked_time[side],
                now=now_mono,
                stale_after_sec=self._hybrid_hand_pose_stale_sec,
            )
            for side in ("left", "right")
        }
        if hybrid_hand_ready["left"]:
            left_hand_vuer = left_hybrid_hand_vuer
        if hybrid_hand_ready["right"]:
            right_hand_vuer = right_hybrid_hand_vuer

        hand_source_changes: list[tuple[str, str]] = []
        with self._lock:
            for side in ("left", "right"):
                source = "hybrid" if hybrid_hand_ready[side] else "openxr"
                if source != self._active_hand_source[side]:
                    self._active_hand_source[side] = source
                    hand_source_changes.append((side, source))
        for side, source in hand_source_changes:
            self.get_logger().info(f"hand pose source {side}={source}")

        effective_pose = {}
        effective_confidence = {}
        for side in ("left", "right"):
            pose_stamp = controller_pose_time[side]
            pose_fresh = (
                pose_stamp is not None
                and now_mono - float(pose_stamp) <= self._controller_pose_stale_sec
            )
            confidence_stamp = controller_confidence_time[side]
            confidence_fresh = (
                confidence_stamp is not None
                and now_mono - float(confidence_stamp) <= self._controller_confidence_stale_sec
            )
            tracked_stamp = controller_tracked_time[side]
            tracked_fresh = (
                tracked_stamp is not None
                and now_mono - float(tracked_stamp) <= self._controller_confidence_stale_sec
            )
            tracked_state = controller_tracked[side] if tracked_fresh else None
            effective_pose[side] = controller_pose[side] if pose_fresh else None
            effective_confidence[side] = select_tracking_confidence(
                pose_fresh=pose_fresh,
                tracked=tracked_state,
                confidence_fresh=confidence_fresh,
                confidence=float(controller_confidence[side]),
                fallback=self._controller_confidence_fallback,
                tracked_floor=self._controller_tracked_confidence_floor,
                untracked_pose_floor=self._controller_untracked_pose_confidence_floor,
            )

        chest_result = self._torso_fusion.update(
            left_controller=effective_pose["left"],
            right_controller=effective_pose["right"],
            left_confidence=effective_confidence["left"],
            right_confidence=effective_confidence["right"],
            now=now_mono,
        )
        if chest_result.target is not None:
            chest_vuer_mat = chest_result.target
        with self._lock:
            self._chest_vuer_mat = chest_vuer_mat.copy()
            self._chest_alpha = float(chest_result.alpha)
            self._chest_left_confidence = float(chest_result.left_confidence)
            self._chest_right_confidence = float(chest_result.right_confidence)
            self._chest_pose_ready = bool(chest_result.ready)

        if now_mono - self._chest_last_log_ts >= 1.0:
            self._chest_last_log_ts = now_mono
            stream_status = self.get_stream_status()
            with self._lock:
                active_topics = self._controller_active_topic.copy()
                ignored_counts = self._controller_ignored_count.copy()
            self.get_logger().info(
                "chest reliability: alpha=%.3f raw=%.3f gate=%.3f c_left=%.3f "
                "c_right=%.3f disagreement=(%.3fm, %.1fdeg) ready=%s "
                "pose_rx=(%d,%d) tracked=(%s,%s) source=(%s,%s) ignored=(%d,%d)"
                % (
                    chest_result.alpha,
                    chest_result.raw_alpha,
                    chest_result.consistency_gate,
                    chest_result.left_confidence,
                    chest_result.right_confidence,
                    chest_result.position_disagreement_m,
                    np.rad2deg(chest_result.rotation_disagreement_rad),
                    chest_result.ready,
                    stream_status["left_controller"][0],
                    stream_status["right_controller"][0],
                    controller_tracked["left"],
                    controller_tracked["right"],
                    active_topics["left"],
                    active_topics["right"],
                    ignored_counts["left"],
                    ignored_counts["right"],
                )
            )

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

        head_mat = openxr_hmd_pose_to_robot(head_vuer_mat)
        chest_mat = T_robot_openxr @ chest_vuer_mat @ T_OPENXR_ROBOT
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

        # Raw OpenXR fingers follow wrist validity. Hybrid fingers have their
        # own freshness/tracked gate so leader-arm operation does not discard them.
        left_is_default = np.allclose(left_wrist_vuer_mat, const_left_wrist_vuer_mat, atol=1e-6) or np.allclose(
            left_wrist_vuer_mat, np.eye(4, dtype=np.float64), atol=1e-6
        )
        right_is_default = np.allclose(right_wrist_vuer_mat, const_right_wrist_vuer_mat, atol=1e-6) or np.allclose(
            right_wrist_vuer_mat, np.eye(4, dtype=np.float64), atol=1e-6
        )
        left_wrist_valid = bool(left_wrist_flag) and (not left_is_default)
        right_wrist_valid = bool(right_wrist_flag) and (not right_is_default)
        left_hand_valid = left_wrist_valid or hybrid_hand_ready["left"]
        right_hand_valid = right_wrist_valid or hybrid_hand_ready["right"]
        with self._lock:
            if left_wrist_valid:
                self._last_left_wrist = unitree_left_wrist.copy()
            else:
                unitree_left_wrist = self._last_left_wrist.copy()
            if right_wrist_valid:
                self._last_right_wrist = unitree_right_wrist.copy()
            else:
                unitree_right_wrist = self._last_right_wrist.copy()
            if left_hand_valid:
                self._last_left_hand = unitree_left_hand.copy()
            else:
                unitree_left_hand = self._last_left_hand.copy()
            if right_hand_valid:
                self._last_right_hand = unitree_right_hand.copy()
            else:
                unitree_right_hand = self._last_right_hand.copy()

        return (
            head_mat,
            chest_mat,
            unitree_left_wrist,
            unitree_right_wrist,
            unitree_left_hand,
            unitree_right_hand,
        )

    def get_chest_reliability(self) -> tuple[float, float, float, bool]:
        with self._lock:
            return (
                float(self._chest_alpha),
                float(self._chest_left_confidence),
                float(self._chest_right_confidence),
                bool(self._chest_pose_ready),
            )

    def get_hybrid_hand_commands(
        self,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        now = time.monotonic()
        with self._lock:
            close = {
                side: self._hybrid_hand_close[side].copy()
                for side in ("left", "right")
            }
            received = dict(self._has_hybrid_hand_close)
            close_time = self._hybrid_hand_close_time.copy()
            activated = dict(self._hybrid_hand_command_activated)

        ready = {
            side: hybrid_hand_pose_is_usable(
                pose_received=received[side],
                pose_timestamp=close_time[side],
                # Once a valid source has activated the command stream, tracked=false
                # intentionally means "hold the last normalized command".
                tracked=activated[side],
                tracked_timestamp=close_time[side],
                now=now,
                stale_after_sec=self._hybrid_hand_pose_stale_sec,
            )
            for side in ("left", "right")
        }
        return close["left"], close["right"], ready["left"], ready["right"]

    def _on_hybrid_hand_motor(self, side: str, msg: Float32MultiArray) -> None:
        with self._lock:
            self._hybrid_hand_motor[side].update_command(msg.data, time.monotonic())

    def _on_hybrid_hand_motor_tracked(self, side: str, msg: Bool) -> None:
        with self._lock:
            self._hybrid_hand_motor[side].update_tracking(msg.data, time.monotonic())

    def get_hybrid_hand_motor_commands(self) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        with self._lock:
            left, left_valid = self._hybrid_hand_motor["left"].snapshot()
            right, right_valid = self._hybrid_hand_motor["right"].snapshot()
        return left, right, left_valid, right_valid

    def get_torso_reliability(self) -> tuple[float, float, float, bool]:
        """Compatibility alias for callers predating the separate chest target."""
        return self.get_chest_reliability()

    def get_television_struct(self) -> np.ndarray:
        """
        np.dtype(TELEVISION) 기반 structured array (1,) 반환.
        SHM writer가 dtype 동일하게 잡아두면 out 전체를 바로 assign/memcpy 가능.
        """
        head_mat, chest_mat, left_wrist_mat, right_wrist_mat, left_hand, right_hand = self.get_television_payload()
        dt = television_dtype()
        out = np.zeros((1,), dtype=dt)
        out["head_mat"][0] = head_mat
        out["chest_mat"][0] = chest_mat
        out["left_wrist_mat"][0] = left_wrist_mat
        out["right_wrist_mat"][0] = right_wrist_mat
        out["left_hand"][0] = left_hand
        out["right_hand"][0] = right_hand
        left_close, right_close, left_close_valid, right_close_valid = (
            self.get_hybrid_hand_commands()
        )
        out["left_hand_close"][0] = left_close
        out["right_hand_close"][0] = right_close
        out["left_hand_close_valid"][0] = 1.0 if left_close_valid else 0.0
        out["right_hand_close_valid"][0] = 1.0 if right_close_valid else 0.0
        left_motor, right_motor, left_valid, right_valid = self.get_hybrid_hand_motor_commands()
        out["left_hand_motor"][0] = left_motor
        out["right_hand_motor"][0] = right_motor
        out["left_hand_motor_valid"][0] = float(left_valid)
        out["right_hand_motor_valid"][0] = float(right_valid)
        return out
