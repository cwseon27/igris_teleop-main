"""Reliability hand backend: OpenXR / MediaPipe -> Dex -> six motor commands.

This node never publishes robot SDK topics or calls hand-init/torque services.
The existing local hand worker owns actuation and applies the six-motor output.
"""
from __future__ import annotations

import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Bool, Float32, Float32MultiArray

from .hand_retarget_runtime import RetargetedHandFusion


class ReliabilityHandRetargetNode(Node):
    def __init__(self):
        super().__init__("reliability_hand_retarget")
        defaults = {
            "openxr_only": False, "publish_rate_hz": 30.0,
            "pose_stale_sec": 0.2, "tracked_stale_sec": 0.5,
            "confidence_stale_sec": 0.5, "confidence_gamma": 1.0,
            "previous_command_weight": 0.0, "close_rate_per_sec": 15.0,
            "open_rate_per_sec": 20.0, "source_dropout_grace_sec": 0.15,
            "openxr_only_confidence_threshold": 0.6,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.options = {name: self.get_parameter(name).value for name in defaults}
        self.openxr_only = bool(self.options["openxr_only"])
        sources = ("openxr",) if self.openxr_only else ("openxr", "mediapipe")
        self.poses = {}
        self.tracking = {}
        self.confidence = {}
        self.last_source = {}
        self.last_warning = 0.0
        self.engines = {}
        self.publishers_by_side = {}
        fusion_options = {name: self.options[name] for name in (
            "confidence_gamma", "previous_command_weight", "close_rate_per_sec",
            "open_rate_per_sec", "source_dropout_grace_sec", "openxr_only_confidence_threshold",
        )}
        for side in ("left", "right"):
            self.engines[side] = RetargetedHandFusion(
                side, openxr_only=self.openxr_only,
                nominal_rate_hz=self.options["publish_rate_hz"], **fusion_options,
            )
            self.publishers_by_side[side] = (
                self.create_publisher(Float32MultiArray, f"/{side}_hybrid_hand/motor_normalized", 1),
                self.create_publisher(Bool, f"/{side}_hybrid_hand/motor_tracked", 1),
            )
            for source in sources:
                base = f"/{side}_hand" if source == "openxr" else f"/{side}_mediapipe_hand"
                pose_topic = base + ("/poses" if source == "openxr" else "/all_poses")
                key = (side, source)
                self.create_subscription(
                    PoseArray, pose_topic, lambda msg, key=key: self.on_pose(key, msg), qos_profile_sensor_data,
                )
                self.create_subscription(
                    Bool, base + "/is_tracked",
                    lambda msg, key=key: self.tracking.update({key: (bool(msg.data), time.monotonic())}),
                    qos_profile_sensor_data,
                )
            if not self.openxr_only:
                self.create_subscription(
                    Float32, f"/{side}_hand/openxr_confidence",
                    lambda msg, side=side: self.confidence.update({side: (float(msg.data), time.monotonic())}),
                    qos_profile_sensor_data,
                )
        self.timer = self.create_timer(1.0 / max(1.0, self.options["publish_rate_hz"]), self.publish_commands)
        self.get_logger().info(
            "Hand backend: shared DexRetargeting, independent sensor histories, 6 motors/hand; "
            + ("VR ONLY; hold on loss" if self.openxr_only else "VR + MediaPipe motor-space reliability fusion")
        )

    def on_pose(self, key, msg):
        expected = 6 if key[1] == "openxr" else 21
        if len(msg.poses) != expected:
            self.poses.pop(key, None)
            return
        poses = msg.poses[1:] if key[1] == "openxr" else msg.poses
        points = np.asarray([[p.position.x, p.position.y, p.position.z] for p in poses], dtype=np.float64)
        self.poses[key] = (points, time.monotonic())

    def observation(self, key, now):
        item = self.poses.get(key)
        tracked, stamp = self.tracking.get(key, (False, float("-inf")))
        if item is None or not tracked:
            return None
        if not 0 <= now - stamp <= self.options["tracked_stale_sec"]:
            return None
        if not 0 <= now - item[1] <= self.options["pose_stale_sec"]:
            return None
        return item

    def publish_commands(self):
        now = time.monotonic()
        for side, engine in self.engines.items():
            confidence, stamp = self.confidence.get(side, (0.0, float("-inf")))
            if not np.isfinite(confidence) or not 0 <= now - stamp <= self.options["confidence_stale_sec"]:
                confidence = 0.0
            result = engine.update(
                openxr=self.observation((side, "openxr"), now),
                mediapipe=self.observation((side, "mediapipe"), now),
                confidence=confidence, now=now,
            )
            command_pub, tracked_pub = self.publishers_by_side[side]
            # tracked is deliberately distinct from motor command validity: after
            # activation, missing observations continue the last command, not zero.
            tracked_pub.publish(Bool(data=result.tracked))
            command_pub.publish(Float32MultiArray(data=list(result.command)))
            if self.last_source.get(side) != result.source:
                self.last_source[side] = result.source
                self.get_logger().info(f"{side} Dex motor source: {result.source}")
            if engine.last_error and now - self.last_warning > 2.0:
                self.last_warning = now
                self.get_logger().warning(f"{side} rejected hand reference: {engine.last_error}")


def main():
    rclpy.init()
    node = None
    try:
        node = ReliabilityHandRetargetNode()
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
