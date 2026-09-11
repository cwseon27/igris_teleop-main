from __future__ import annotations

from collections import deque
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Bool, Float32

from train_lib.pose_utils import (
    FEATURE_MODE_HMD_RELATIVE_MOTION,
    HmdRelativeMotionFeatureBuilder,
    pose_to_flat_from_fields,
    validate_vector,
)

from .feature_builders import ControllerMotionFeatureBuilder, FEATURE_MODE_CONTROLLER_MOTION
from .model_runtime import ALWAYS_ONE_VARIANT, load_policy_bundle


TYPE_MAP = {
    "geometry_msgs/msg/Pose": Pose,
    "geometry_msgs/msg/PoseStamped": PoseStamped,
    "geometry_msgs/msg/PoseArray": PoseArray,
}
POSE_TYPES = tuple(TYPE_MAP)


def extract_pose_flat(msg) -> np.ndarray:
    if isinstance(msg, Pose):
        return pose_to_flat_from_fields(msg.position, msg.orientation)
    if isinstance(msg, PoseStamped):
        return pose_to_flat_from_fields(msg.pose.position, msg.pose.orientation)
    if isinstance(msg, PoseArray):
        parts = [pose_to_flat_from_fields(p.position, p.orientation) for p in msg.poses]
        if not parts:
            raise ValueError("PoseArray has zero poses")
        return validate_vector(np.concatenate(parts, axis=0))
    raise TypeError(f"unsupported pose message type: {type(msg)!r}")


def _finite_confidence(value: float, default: float = 0.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        value = float(default)
    return float(np.clip(value, 0.0, 1.0))


class PairConfidenceNode(Node):
    def __init__(self, kind: str):
        node_name = f"{kind}_tracking_confidence_inference"
        super().__init__(node_name)
        self.kind = kind
        self.qos = QoSProfile(depth=10)

        self.declare_parameter("model_variant", "rnn")
        self.declare_parameter("fallback_variant", "histgb")
        self.declare_parameter("model_path", "")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("pose_stale_sec", 0.0)
        self.declare_parameter("tracked_stale_sec", 0.5)
        self.declare_parameter("publish_when_not_ready", True)
        self.declare_parameter("default_confidence", 0.0)

        if kind == "hand":
            self._declare_hand_parameters()
        elif kind == "controller":
            self._declare_controller_parameters()
        else:
            raise ValueError(f"unsupported inference kind {kind!r}")

        variant = str(self.get_parameter("model_variant").value).strip().lower()
        fallback_variant = str(self.get_parameter("fallback_variant").value).strip().lower()
        model_override = str(self.get_parameter("model_path").value).strip()
        self.bundle, self.model_path, self.loaded_variant = load_policy_bundle(
            kind,
            variant=variant,
            override=model_override,
            fallback_variant=fallback_variant,
        )
        self.always_one = self.loaded_variant == ALWAYS_ONE_VARIANT
        self.window_size = int(self.bundle["window_size"])
        self.feature_dim = int(self.bundle["feature_dim"])
        self.left_model = self.bundle["left_model"]
        self.right_model = self.bundle["right_model"]
        configured_stale = float(self.get_parameter("pose_stale_sec").value)
        self.pose_stale_sec = (
            configured_stale
            if configured_stale > 0.0
            else float(self.bundle.get("pose_stale_sec", 0.2))
        )
        self.tracked_stale_sec = max(
            0.0, float(self.get_parameter("tracked_stale_sec").value)
        )
        self.rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        self.publish_when_not_ready = bool(self.get_parameter("publish_when_not_ready").value)
        self.default_confidence = _finite_confidence(
            self.get_parameter("default_confidence").value
        )

        self.latest_pose: dict[str, np.ndarray | None] = {
            key: None for key in self.pose_topics
        }
        self.latest_pose_time: dict[str, float | None] = {
            key: None for key in self.pose_topics
        }
        self.latest_bool: dict[str, bool | None] = {"left": None, "right": None}
        self.latest_bool_time: dict[str, float | None] = {"left": None, "right": None}
        self.pose_subscriptions = {}
        self.pose_array_counts = {}
        self.frame_buffer = deque(maxlen=self.window_size)
        self.last_warning_time: dict[str, float] = {}
        self.last_status_time = time.monotonic()
        self.last_published = (self.default_confidence, self.default_confidence)

        self.left_pub = self.create_publisher(Float32, self.output_topics["left"], self.qos)
        self.right_pub = self.create_publisher(Float32, self.output_topics["right"], self.qos)
        if self.always_one:
            self.feature_builder = None
            self.discovery_timer = None
            self.infer_timer = self.create_timer(1.0 / self.rate_hz, self._on_infer_timer)
            self.get_logger().warning(
                f"DEBUG override enabled for {kind}: publishing confidence=1.0 "
                "without model inference or tracking gates"
            )
            return

        self.create_subscription(
            Bool,
            self.tracked_topics["left"],
            lambda msg: self._on_tracked("left", msg),
            self.qos,
        )
        self.create_subscription(
            Bool,
            self.tracked_topics["right"],
            lambda msg: self._on_tracked("right", msg),
            self.qos,
        )

        if kind == "hand":
            self.feature_builder = HmdRelativeMotionFeatureBuilder(
                pose_stale_sec=self.pose_stale_sec
            )
            expected_mode = FEATURE_MODE_HMD_RELATIVE_MOTION
        else:
            self.feature_builder = ControllerMotionFeatureBuilder(
                pose_stale_sec=self.pose_stale_sec
            )
            expected_mode = FEATURE_MODE_CONTROLLER_MOTION

        model_mode = str(self.bundle.get("feature_mode", ""))
        if model_mode != expected_mode:
            raise ValueError(
                f"policy feature_mode={model_mode!r}, expected {expected_mode!r} for {kind}"
            )

        self.discovery_timer = self.create_timer(1.0, self._discover_pose_topics)
        self.infer_timer = self.create_timer(1.0 / self.rate_hz, self._on_infer_timer)
        self._discover_pose_topics()
        self.get_logger().info(
            f"loaded {kind} policy variant={self.loaded_variant} path={self.model_path} "
            f"window={self.window_size} feature_dim={self.feature_dim}"
        )

    def _declare_hand_parameters(self) -> None:
        self.declare_parameter("hmd_pose_topic", "/hmd/pose")
        self.declare_parameter("left_pose_topic", "/left_hand/poses")
        self.declare_parameter("right_pose_topic", "/right_hand/poses")
        self.declare_parameter("left_tracked_topic", "/left_hand/is_tracked")
        self.declare_parameter("right_tracked_topic", "/right_hand/is_tracked")
        self.declare_parameter("left_output_topic", "/left_hand/openxr_confidence")
        self.declare_parameter("right_output_topic", "/right_hand/openxr_confidence")
        self.pose_topics = {
            "hmd": str(self.get_parameter("hmd_pose_topic").value),
            "left": str(self.get_parameter("left_pose_topic").value),
            "right": str(self.get_parameter("right_pose_topic").value),
        }
        self.expected_dims = {
            "hmd": int(self.bundle_value("hmd_pose_dim", 7)),
            "left": int(self.bundle_value("left_hand_pose_dim", 42)),
            "right": int(self.bundle_value("right_hand_pose_dim", 42)),
        }
        self._finish_topic_parameters()

    def _declare_controller_parameters(self) -> None:
        self.declare_parameter("left_pose_topic", "/left_controller/poses")
        self.declare_parameter("right_pose_topic", "/right_controller/poses")
        self.declare_parameter("left_tracked_topic", "/left_controller/is_tracked")
        self.declare_parameter("right_tracked_topic", "/right_controller/is_tracked")
        self.declare_parameter(
            "left_output_topic", "/left_controller/tracking_confidence"
        )
        self.declare_parameter(
            "right_output_topic", "/right_controller/tracking_confidence"
        )
        self.pose_topics = {
            "left": str(self.get_parameter("left_pose_topic").value),
            "right": str(self.get_parameter("right_pose_topic").value),
        }
        self.expected_dims = {
            "left": int(self.bundle_value("left_controller_pose_dim", 7)),
            "right": int(self.bundle_value("right_controller_pose_dim", 7)),
        }
        self._finish_topic_parameters()

    def bundle_value(self, key: str, default: int) -> int:
        # Topic parameters are declared before the bundle is loaded. Bundle dimensions
        # are refreshed after loading in _expected_dim().
        del key
        return default

    def _finish_topic_parameters(self) -> None:
        self.tracked_topics = {
            "left": str(self.get_parameter("left_tracked_topic").value),
            "right": str(self.get_parameter("right_tracked_topic").value),
        }
        self.output_topics = {
            "left": str(self.get_parameter("left_output_topic").value),
            "right": str(self.get_parameter("right_output_topic").value),
        }

    def _expected_dim(self, key: str) -> int:
        if self.kind == "hand":
            bundle_key = {
                "hmd": "hmd_pose_dim",
                "left": "left_hand_pose_dim",
                "right": "right_hand_pose_dim",
            }[key]
        else:
            bundle_key = f"{key}_controller_pose_dim"
        return int(self.bundle.get(bundle_key, self.expected_dims[key]) or self.expected_dims[key])

    def _warn_limited(self, key: str, message: str, period_sec: float = 5.0) -> None:
        now = time.monotonic()
        if now - self.last_warning_time.get(key, 0.0) >= period_sec:
            self.get_logger().warning(message)
            self.last_warning_time[key] = now

    def _discover_pose_topics(self) -> None:
        discovered = dict(self.get_topic_names_and_types())
        for key, topic in self.pose_topics.items():
            if key in self.pose_subscriptions:
                continue
            available_types = discovered.get(topic, [])
            chosen_type = next((name for name in POSE_TYPES if name in available_types), None)
            if chosen_type is None:
                self._warn_limited(
                    f"wait:{topic}",
                    f"waiting for pose topic {topic}; discovered types={available_types}",
                )
                continue
            self.pose_subscriptions[key] = self.create_subscription(
                TYPE_MAP[chosen_type],
                topic,
                lambda msg, pose_key=key: self._on_pose(pose_key, msg),
                self.qos,
            )
            self.get_logger().info(f"subscribed {topic} as {chosen_type}")

    def _on_pose(self, key: str, msg) -> None:
        try:
            if isinstance(msg, PoseArray):
                count = len(msg.poses)
                previous = self.pose_array_counts.setdefault(key, count)
                if previous != count:
                    raise ValueError(
                        f"PoseArray count changed from {previous} to {count}"
                    )
            value = extract_pose_flat(msg)
            expected = self._expected_dim(key)
            if expected > 0 and value.size != expected:
                raise ValueError(f"pose dim {value.size} != expected {expected}")
            self.latest_pose[key] = value
            self.latest_pose_time[key] = self.get_clock().now().nanoseconds * 1e-9
        except Exception as exc:
            self.latest_pose[key] = None
            self.latest_pose_time[key] = None
            self._warn_limited(f"pose:{key}", f"dropping invalid {key} pose: {exc}")

    def _on_tracked(self, side: str, msg: Bool) -> None:
        self.latest_bool[side] = bool(msg.data)
        self.latest_bool_time[side] = self.get_clock().now().nanoseconds * 1e-9

    def _publish(self, left: float, right: float) -> None:
        left_msg = Float32()
        right_msg = Float32()
        left_msg.data = _finite_confidence(left, self.default_confidence)
        right_msg.data = _finite_confidence(right, self.default_confidence)
        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)
        self.last_published = (float(left_msg.data), float(right_msg.data))

    def _publish_default(self) -> None:
        if self.publish_when_not_ready:
            self._publish(self.default_confidence, self.default_confidence)

    def _inputs_ready(self) -> bool:
        return all(value is not None for value in self.latest_pose.values()) and all(
            value is not None for value in self.latest_bool.values()
        )

    def _build_feature(self, now_sec: float) -> tuple[np.ndarray, bool, bool, float, float]:
        left_age = now_sec - float(self.latest_pose_time["left"])
        right_age = now_sec - float(self.latest_pose_time["right"])
        left_tracked_age = now_sec - float(self.latest_bool_time["left"])
        right_tracked_age = now_sec - float(self.latest_bool_time["right"])
        left_tracked = bool(self.latest_bool["left"]) and (
            left_tracked_age <= self.tracked_stale_sec
        )
        right_tracked = bool(self.latest_bool["right"]) and (
            right_tracked_age <= self.tracked_stale_sec
        )
        if self.kind == "hand":
            hmd_age = now_sec - float(self.latest_pose_time["hmd"])
            if hmd_age > self.pose_stale_sec:
                raise ValueError(f"HMD pose is stale ({hmd_age:.3f}s)")
            feature = self.feature_builder.build(
                validate_vector(self.latest_pose["hmd"]),
                validate_vector(self.latest_pose["left"]),
                validate_vector(self.latest_pose["right"]),
                timestamp=now_sec,
                left_is_tracked=left_tracked,
                right_is_tracked=right_tracked,
                left_pose_age_sec=left_age,
                right_pose_age_sec=right_age,
            )
        else:
            feature = self.feature_builder.build(
                validate_vector(self.latest_pose["left"]),
                validate_vector(self.latest_pose["right"]),
                timestamp=now_sec,
                left_is_tracked=left_tracked,
                right_is_tracked=right_tracked,
                left_pose_age_sec=left_age,
                right_pose_age_sec=right_age,
            )
        return feature, left_tracked, right_tracked, left_age, right_age

    def _on_infer_timer(self) -> None:
        now_mono = time.monotonic()
        if self.always_one:
            self._publish(1.0, 1.0)
            if now_mono - self.last_status_time >= 5.0:
                self.last_status_time = now_mono
                self.get_logger().info(
                    f"{self.kind} confidence DEBUG always_1 "
                    f"left={self.last_published[0]:.3f} right={self.last_published[1]:.3f}"
                )
            return

        if now_mono - self.last_status_time >= 5.0:
            self.last_status_time = now_mono
            self.get_logger().info(
                f"{self.kind} confidence buffer={len(self.frame_buffer)}/{self.window_size} "
                f"left={self.last_published[0]:.3f} right={self.last_published[1]:.3f}"
            )
        if not self._inputs_ready():
            self._publish_default()
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        try:
            feature, left_tracked, right_tracked, left_age, right_age = self._build_feature(
                now_sec
            )
        except Exception as exc:
            self._warn_limited("feature", f"cannot build {self.kind} feature: {exc}", 2.0)
            self.frame_buffer.clear()
            self._publish_default()
            return
        if feature.size != self.feature_dim:
            self._warn_limited(
                "feature-dim",
                f"feature dim {feature.size} != policy feature dim {self.feature_dim}",
                2.0,
            )
            self.frame_buffer.clear()
            self._publish_default()
            return

        self.frame_buffer.append(feature.astype(np.float32, copy=True))
        if len(self.frame_buffer) < self.window_size:
            self._publish_default()
            return

        window = np.stack(self.frame_buffer, axis=0).astype(np.float32)[None, :, :]
        try:
            left = float(np.asarray(self.left_model.predict_confidence(window)).reshape(-1)[0])
            right = float(np.asarray(self.right_model.predict_confidence(window)).reshape(-1)[0])
        except Exception as exc:
            self._warn_limited("predict", f"{self.kind} policy inference failed: {exc}", 2.0)
            self._publish_default()
            return

        if not left_tracked or left_age > self.pose_stale_sec:
            left = 0.0
        if not right_tracked or right_age > self.pose_stale_sec:
            right = 0.0
        self._publish(left, right)


def _spin(kind: str, args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = PairConfidenceNode(kind)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main_hand(args=None) -> None:
    _spin("hand", args=args)


def main_controller(args=None) -> None:
    _spin("controller", args=args)
