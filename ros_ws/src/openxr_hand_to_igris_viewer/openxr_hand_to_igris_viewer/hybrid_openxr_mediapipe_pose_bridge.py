#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Bool, Float32, Float32MultiArray

from .hand_fusion import ReliabilityAwareHandCommandFusion


FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
FINGER_TIP_INDEX = {
    "thumb": 1,
    "index": 2,
    "middle": 3,
    "ring": 4,
    "little": 5,
}
def _default_calibration_dir() -> Path:
    try:
        package_share = Path(get_package_share_directory("igris_reliability_runtime"))
        return package_share / "config" / "hand"
    except Exception:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "ros_ws" / "src" / "igris_reliability_runtime" / "config" / "hand"
            if candidate.is_dir():
                return candidate
        return Path(__file__).resolve().parent


DEFAULT_CALIB_DIR = _default_calibration_dir()
DEFAULT_OPENXR_CALIB = DEFAULT_CALIB_DIR / "openxr_hand_calibration.json"
DEFAULT_MEDIAPIPE_CALIB = DEFAULT_CALIB_DIR / "mediapipe_hand_calibration.json"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def copy_pose(pose: Pose) -> Pose:
    out = Pose()
    out.position.x = float(pose.position.x)
    out.position.y = float(pose.position.y)
    out.position.z = float(pose.position.z)
    out.orientation.x = float(pose.orientation.x)
    out.orientation.y = float(pose.orientation.y)
    out.orientation.z = float(pose.orientation.z)
    out.orientation.w = float(pose.orientation.w)
    return out


class HybridOpenXRMediaPipePoseBridge(Node):
    def __init__(self):
        super().__init__("hybrid_openxr_mediapipe_pose_bridge")

        self.declare_parameter("hand_side", "right")
        self.declare_parameter("openxr_pose_topic", "")
        self.declare_parameter("openxr_tracked_topic", "")
        self.declare_parameter("openxr_confidence_topic", "")
        self.declare_parameter("mediapipe_pose_topic", "")
        self.declare_parameter("mediapipe_tracked_topic", "")
        self.declare_parameter("output_pose_topic", "")
        self.declare_parameter("output_tracked_topic", "")
        self.declare_parameter("output_normalized_topic", "")
        self.declare_parameter("apply_calibration_remap", True)
        self.declare_parameter("openxr_calibration_json", "")
        self.declare_parameter("mediapipe_calibration_json", "")
        self.declare_parameter("output_calibration_json", "")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("pose_stale_sec", 0.2)
        self.declare_parameter("confidence_stale_sec", 0.5)
        self.declare_parameter("tracked_stale_sec", 0.5)
        self.declare_parameter("min_pose_count", 6)
        self.declare_parameter("default_confidence", 0.0)
        self.declare_parameter("confidence_gamma", 1.0)
        self.declare_parameter("confidence_min", 0.0)
        self.declare_parameter("confidence_max", 1.0)
        self.declare_parameter("previous_command_weight", 0.0)
        self.declare_parameter("close_rate_per_sec", 15.0)
        self.declare_parameter("open_rate_per_sec", 20.0)
        self.declare_parameter("openxr_only_confidence_threshold", 0.6)
        self.declare_parameter("source_dropout_grace_sec", 0.15)
        self.declare_parameter("openxr_only", False)
        self.declare_parameter("debug_log", False)

        self.hand_side = str(self.get_parameter("hand_side").value).lower()
        if self.hand_side not in ("right", "left"):
            raise ValueError("hand_side must be 'right' or 'left'")

        self.openxr_pose_topic = str(self.get_parameter("openxr_pose_topic").value) or f"/{self.hand_side}_hand/poses"
        self.openxr_tracked_topic = (
            str(self.get_parameter("openxr_tracked_topic").value) or f"/{self.hand_side}_hand/is_tracked"
        )
        self.openxr_confidence_topic = (
            str(self.get_parameter("openxr_confidence_topic").value)
            or f"/{self.hand_side}_hand/openxr_confidence"
        )
        self.mediapipe_pose_topic = (
            str(self.get_parameter("mediapipe_pose_topic").value)
            or f"/{self.hand_side}_mediapipe_hand/poses"
        )
        self.mediapipe_tracked_topic = (
            str(self.get_parameter("mediapipe_tracked_topic").value)
            or f"/{self.hand_side}_mediapipe_hand/is_tracked"
        )
        self.output_pose_topic = (
            str(self.get_parameter("output_pose_topic").value)
            or f"/{self.hand_side}_hybrid_hand/poses"
        )
        self.output_tracked_topic = (
            str(self.get_parameter("output_tracked_topic").value)
            or f"/{self.hand_side}_hybrid_hand/is_tracked"
        )
        self.output_normalized_topic = (
            str(self.get_parameter("output_normalized_topic").value)
            or f"/{self.hand_side}_hybrid_hand/finger_normalized"
        )

        self.apply_calibration_remap = bool(self.get_parameter("apply_calibration_remap").value)
        self.openxr_calibration_path = Path(
            str(self.get_parameter("openxr_calibration_json").value) or str(DEFAULT_OPENXR_CALIB)
        ).expanduser()
        self.mediapipe_calibration_path = Path(
            str(self.get_parameter("mediapipe_calibration_json").value) or str(DEFAULT_MEDIAPIPE_CALIB)
        ).expanduser()
        output_calibration_param = str(self.get_parameter("output_calibration_json").value)
        self.output_calibration_path = Path(output_calibration_param).expanduser() if output_calibration_param else self.openxr_calibration_path

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.pose_stale_sec = float(self.get_parameter("pose_stale_sec").value)
        self.confidence_stale_sec = float(self.get_parameter("confidence_stale_sec").value)
        self.tracked_stale_sec = float(self.get_parameter("tracked_stale_sec").value)
        self.min_pose_count = int(self.get_parameter("min_pose_count").value)
        self.default_confidence = float(self.get_parameter("default_confidence").value)
        self.confidence_gamma = max(0.01, float(self.get_parameter("confidence_gamma").value))
        self.confidence_min = float(self.get_parameter("confidence_min").value)
        self.confidence_max = float(self.get_parameter("confidence_max").value)
        self.previous_command_weight = max(
            0.0, float(self.get_parameter("previous_command_weight").value)
        )
        self.close_rate_per_sec = max(
            0.0, float(self.get_parameter("close_rate_per_sec").value)
        )
        self.open_rate_per_sec = max(
            0.0, float(self.get_parameter("open_rate_per_sec").value)
        )
        self.openxr_only_confidence_threshold = clamp(
            float(self.get_parameter("openxr_only_confidence_threshold").value), 0.0, 1.0
        )
        self.source_dropout_grace_sec = max(
            0.0, float(self.get_parameter("source_dropout_grace_sec").value)
        )
        self.openxr_only = bool(self.get_parameter("openxr_only").value)
        self.debug_log = bool(self.get_parameter("debug_log").value)

        self.openxr_calib = self.load_calibration(self.openxr_calibration_path, "OpenXR")
        self.mediapipe_calib = self.load_calibration(self.mediapipe_calibration_path, "MediaPipe")
        if self.output_calibration_path == self.openxr_calibration_path:
            self.output_calib = self.openxr_calib
        elif self.output_calibration_path == self.mediapipe_calibration_path:
            self.output_calib = self.mediapipe_calib
        else:
            self.output_calib = self.load_calibration(self.output_calibration_path, "output")

        self.openxr_pose: Optional[PoseArray] = None
        self.openxr_pose_time: Optional[float] = None
        self.mediapipe_pose: Optional[PoseArray] = None
        self.mediapipe_pose_time: Optional[float] = None
        self.openxr_tracked: Optional[bool] = None
        self.openxr_tracked_time: Optional[float] = None
        self.mediapipe_tracked: Optional[bool] = None
        self.mediapipe_tracked_time: Optional[float] = None
        self.confidence: Optional[float] = None
        self.confidence_time: Optional[float] = None
        self.last_source = None
        self.last_log_t = time.monotonic()
        self.last_close_amounts: Optional[List[float]] = [0.0] * len(FINGER_NAMES)
        self.last_output_pose: Optional[PoseArray] = None
        self.command_fusion = ReliabilityAwareHandCommandFusion(
            confidence_gamma=self.confidence_gamma,
            previous_command_weight=self.previous_command_weight,
            close_rate_per_sec=self.close_rate_per_sec,
            open_rate_per_sec=self.open_rate_per_sec,
            openxr_only_confidence_threshold=self.openxr_only_confidence_threshold,
            source_dropout_grace_sec=self.source_dropout_grace_sec,
            nominal_rate_hz=self.publish_rate_hz,
            openxr_only=self.openxr_only,
        )

        qos = QoSProfile(depth=10)
        self.create_subscription(PoseArray, self.openxr_pose_topic, self.on_openxr_pose, qos)
        self.create_subscription(Bool, self.openxr_tracked_topic, self.on_openxr_tracked, qos)
        if not self.openxr_only:
            self.create_subscription(Float32, self.openxr_confidence_topic, self.on_confidence, qos)
            self.create_subscription(PoseArray, self.mediapipe_pose_topic, self.on_mediapipe_pose, qos)
            self.create_subscription(Bool, self.mediapipe_tracked_topic, self.on_mediapipe_tracked, qos)

        self.pose_pub = self.create_publisher(PoseArray, self.output_pose_topic, qos)
        self.tracked_pub = self.create_publisher(Bool, self.output_tracked_topic, qos)
        self.normalized_pub = self.create_publisher(
            Float32MultiArray, self.output_normalized_topic, qos
        )
        self.timer = self.create_timer(1.0 / max(self.publish_rate_hz, 1.0), self.publish_hybrid)

        self.get_logger().info("Hybrid OpenXR/MediaPipe pose bridge started.")
        self.get_logger().info(f"hand_side               : {self.hand_side}")
        self.get_logger().info(
            f"hand command mode       : {'openxr_only (hold on tracking loss)' if self.openxr_only else 'hybrid'}"
        )
        self.get_logger().info(f"openxr_pose_topic       : {self.openxr_pose_topic}")
        self.get_logger().info(f"openxr_tracked_topic    : {self.openxr_tracked_topic}")
        self.get_logger().info(f"openxr_confidence_topic : {self.openxr_confidence_topic}")
        self.get_logger().info(f"mediapipe_pose_topic    : {self.mediapipe_pose_topic}")
        self.get_logger().info(f"mediapipe_tracked_topic : {self.mediapipe_tracked_topic}")
        self.get_logger().info(f"output_pose_topic       : {self.output_pose_topic}")
        self.get_logger().info(f"output_tracked_topic    : {self.output_tracked_topic}")
        self.get_logger().info(f"output_normalized_topic : {self.output_normalized_topic}")
        self.get_logger().info(f"apply_calibration_remap : {self.apply_calibration_remap}")
        self.get_logger().info(f"openxr_calibration_json : {self.openxr_calibration_path}")
        self.get_logger().info(f"mediapipe_calibration_json: {self.mediapipe_calibration_path}")
        self.get_logger().info(f"output_calibration_json : {self.output_calibration_path}")

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def load_calibration(self, path: Path, label: str) -> Optional[Dict[str, float]]:
        if not self.apply_calibration_remap:
            return None
        if not path.exists():
            self.get_logger().warn(f"{label} calibration JSON not found: {path}. Raw poses will be used for that source.")
            return None

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        required = []
        for finger in FINGER_NAMES:
            required.append(f"{finger}_max")
            required.append(f"{finger}_min")

        missing = [key for key in required if key not in data]
        if missing:
            self.get_logger().warn(
                f"{label} calibration JSON is missing keys {missing}. Raw poses will be used for that source."
            )
            return None

        calib = {key: float(data[key]) for key in required}
        self.get_logger().info(f"Loaded {label} calibration: {path}")
        return calib

    def is_fresh(self, stamp: Optional[float], limit_sec: float) -> bool:
        if stamp is None:
            return False
        return (self.now_sec() - float(stamp)) <= max(0.0, float(limit_sec))

    def on_openxr_pose(self, msg: PoseArray):
        self.openxr_pose = msg
        self.openxr_pose_time = self.now_sec()

    def on_mediapipe_pose(self, msg: PoseArray):
        self.mediapipe_pose = msg
        self.mediapipe_pose_time = self.now_sec()

    def on_openxr_tracked(self, msg: Bool):
        self.openxr_tracked = bool(msg.data)
        self.openxr_tracked_time = self.now_sec()

    def on_mediapipe_tracked(self, msg: Bool):
        self.mediapipe_tracked = bool(msg.data)
        self.mediapipe_tracked_time = self.now_sec()

    def on_confidence(self, msg: Float32):
        self.confidence = float(msg.data)
        self.confidence_time = self.now_sec()

    def pose_valid(self, msg: Optional[PoseArray], stamp: Optional[float]) -> bool:
        return (
            msg is not None
            and len(msg.poses) >= self.min_pose_count
            and self.is_fresh(stamp, self.pose_stale_sec)
        )

    def tracked_valid(self, tracked: Optional[bool], stamp: Optional[float]) -> bool:
        return bool(tracked) and self.is_fresh(stamp, self.tracked_stale_sec)

    def current_confidence(self) -> float:
        if self.openxr_only:
            return 1.0
        if self.confidence is None or not self.is_fresh(self.confidence_time, self.confidence_stale_sec):
            confidence = self.default_confidence
        else:
            confidence = self.confidence
        confidence = clamp(float(confidence), self.confidence_min, self.confidence_max)
        if self.confidence_max - self.confidence_min > 1e-9:
            confidence = (confidence - self.confidence_min) / (self.confidence_max - self.confidence_min)
        confidence = clamp(confidence, 0.0, 1.0)
        return confidence

    def close_amounts_from_message(
        self,
        msg: Optional[PoseArray],
        calib: Optional[Dict[str, float]],
    ) -> Optional[List[float]]:
        if msg is None or calib is None or len(msg.poses) < self.min_pose_count:
            return None
        values = []
        for finger in FINGER_NAMES:
            value = self.close_amount_from_pose(msg.poses[FINGER_TIP_INDEX[finger]], finger, calib)
            if value is None:
                return None
            values.append(value)
        return values

    def close_amount_from_pose(
        self,
        pose: Pose,
        finger: str,
        calib: Optional[Dict[str, float]],
    ) -> Optional[float]:
        if calib is None:
            return None

        position = pose.position
        distance = math.sqrt(
            position.x * position.x
            + position.y * position.y
            + position.z * position.z
        )

        max_dist = calib[f"{finger}_max"]
        min_dist = calib[f"{finger}_min"]
        if not all(math.isfinite(value) for value in (distance, max_dist, min_dist)):
            return None
        denom = max_dist - min_dist
        if abs(denom) < 1e-9:
            return None

        return clamp((max_dist - distance) / denom, 0.0, 1.0)

    def distance_from_close_amount(self, finger: str, close_amount: float) -> Optional[float]:
        if self.output_calib is None:
            return None

        max_dist = self.output_calib[f"{finger}_max"]
        min_dist = self.output_calib[f"{finger}_min"]
        close_amount = clamp(close_amount, 0.0, 1.0)
        return max_dist - close_amount * (max_dist - min_dist)

    def pose_direction(self, pose: Pose) -> Optional[tuple[float, float, float]]:
        position = pose.position
        x = float(position.x)
        y = float(position.y)
        z = float(position.z)
        norm = math.sqrt(x * x + y * y + z * z)
        if norm < 1e-9:
            return None
        return x / norm, y / norm, z / norm

    def blended_direction(self, openxr_pose: Pose, mediapipe_pose: Pose, theta: float) -> tuple[float, float, float]:
        theta = clamp(theta, 0.0, 1.0)
        beta = 1.0 - theta
        x = theta * float(openxr_pose.position.x) + beta * float(mediapipe_pose.position.x)
        y = theta * float(openxr_pose.position.y) + beta * float(mediapipe_pose.position.y)
        z = theta * float(openxr_pose.position.z) + beta * float(mediapipe_pose.position.z)
        norm = math.sqrt(x * x + y * y + z * z)
        if norm >= 1e-9:
            return x / norm, y / norm, z / norm

        preferred = self.pose_direction(openxr_pose if theta >= 0.5 else mediapipe_pose)
        if preferred is not None:
            return preferred

        fallback = self.pose_direction(mediapipe_pose if theta >= 0.5 else openxr_pose)
        if fallback is not None:
            return fallback

        return 1.0, 0.0, 0.0

    def set_pose_distance(self, pose: Pose, direction: tuple[float, float, float], distance: float) -> None:
        pose.position.x = float(direction[0] * distance)
        pose.position.y = float(direction[1] * distance)
        pose.position.z = float(direction[2] * distance)

    def publish_tracked(self, tracked: bool):
        msg = Bool()
        msg.data = bool(tracked)
        self.tracked_pub.publish(msg)

    def publish_normalized(self, values: List[float] | tuple[float, ...]) -> None:
        msg = Float32MultiArray()
        msg.data = [float(clamp(value, 0.0, 1.0)) for value in values]
        self.normalized_pub.publish(msg)

    def pose_list_from_command(
        self,
        command: List[float] | tuple[float, ...],
        openxr_msg: Optional[PoseArray],
        mediapipe_msg: Optional[PoseArray],
        theta: float,
    ) -> Optional[List[Pose]]:
        if self.output_calib is None:
            return None
        source = openxr_msg if openxr_msg is not None else mediapipe_msg
        if source is None or len(source.poses) < self.min_pose_count:
            return None
        count = len(source.poses)
        if openxr_msg is not None and mediapipe_msg is not None:
            count = min(len(openxr_msg.poses), len(mediapipe_msg.poses))
        poses = [copy_pose(source.poses[i]) for i in range(count)]
        for finger_idx, finger in enumerate(FINGER_NAMES):
            idx = FINGER_TIP_INDEX[finger]
            if idx >= count:
                return None
            target_distance = self.distance_from_close_amount(finger, command[finger_idx])
            if target_distance is None:
                return None
            if openxr_msg is not None and mediapipe_msg is not None:
                direction = self.blended_direction(
                    openxr_msg.poses[idx], mediapipe_msg.poses[idx], theta
                )
            else:
                direction = self.pose_direction(source.poses[idx]) or (1.0, 0.0, 0.0)
            self.set_pose_distance(poses[idx], direction, target_distance)
        return poses

    def publish_hybrid(self):
        openxr_ready = self.pose_valid(self.openxr_pose, self.openxr_pose_time) and self.tracked_valid(
            self.openxr_tracked, self.openxr_tracked_time
        )
        mediapipe_ready = (
            not self.openxr_only
            and self.pose_valid(self.mediapipe_pose, self.mediapipe_pose_time)
            and self.tracked_valid(self.mediapipe_tracked, self.mediapipe_tracked_time)
        )

        confidence = self.current_confidence()
        openxr_close = self.close_amounts_from_message(
            self.openxr_pose if openxr_ready else None, self.openxr_calib
        )
        mediapipe_close = self.close_amounts_from_message(
            self.mediapipe_pose if mediapipe_ready else None, self.mediapipe_calib
        )
        result = self.command_fusion.update(
            openxr_close=openxr_close,
            mediapipe_close=mediapipe_close,
            confidence=confidence,
            openxr_ready=openxr_ready,
            mediapipe_ready=mediapipe_ready,
            now=self.now_sec(),
        )
        self.last_close_amounts = list(result.command)
        self.publish_normalized(result.command)
        self.publish_tracked(result.tracked)

        # Use only sources with valid finger geometry for the compatibility
        # pose output too; a fresh/tracked message can still contain NaNs.
        openxr_for_pose = self.openxr_pose if openxr_ready and openxr_close is not None else None
        mediapipe_for_pose = (
            self.mediapipe_pose if mediapipe_ready and mediapipe_close is not None else None
        )
        pose_list = self.pose_list_from_command(
            result.command, openxr_for_pose, mediapipe_for_pose, result.theta
        )
        if pose_list is not None:
            out = PoseArray()
            out.header.stamp = self.get_clock().now().to_msg()
            if openxr_for_pose is not None:
                out.header.frame_id = openxr_for_pose.header.frame_id
            elif mediapipe_for_pose is not None:
                out.header.frame_id = mediapipe_for_pose.header.frame_id
            out.poses = pose_list
            self.last_output_pose = out
        elif self.last_output_pose is not None:
            out = self.last_output_pose
            out.header.stamp = self.get_clock().now().to_msg()
        else:
            out = None

        if out is not None:
            self.pose_pub.publish(out)

        source = result.source
        theta = result.theta

        if source != self.last_source:
            self.last_source = source
            self.get_logger().info(f"hybrid source: {source}")

        if self.debug_log and time.monotonic() - self.last_log_t > 1.0:
            self.last_log_t = time.monotonic()
            close_text = "n/a"
            if self.last_close_amounts:
                close_text = ",".join(f"{value:.2f}" for value in self.last_close_amounts)
            self.get_logger().info(
                f"theta={theta:.3f}, confidence={result.confidence:.3f}, "
                f"openxr_ready={openxr_ready}, mediapipe_ready={mediapipe_ready}, "
                f"close=[{close_text}]"
            )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = HybridOpenXRMediaPipePoseBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
