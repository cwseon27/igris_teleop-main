from __future__ import annotations

import time
from typing import Mapping

import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage

from ...sim.stereo_camera import (
    ROS_STEREO_LEFT_COMPRESSED_TOPIC,
    ROS_STEREO_LEFT_INFO_TOPIC,
    ROS_STEREO_RIGHT_COMPRESSED_TOPIC,
    ROS_STEREO_RIGHT_INFO_TOPIC,
    STEREO_CAMERA_BASELINE_M,
    STEREO_CAMERA_HEIGHT,
    STEREO_CAMERA_WIDTH,
    STEREO_LEFT_FRAME_ID,
    STEREO_RIGHT_FRAME_ID,
    pinhole_intrinsics,
    stereo_projection,
)


def build_sim_stereo_qos(depth: int = 1) -> QoSProfile:
    """Match the reliable stereo QoS expected by ROS-TCP-Endpoint/Unity."""
    return QoSProfile(
        depth=max(1, int(depth)),
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        durability=DurabilityPolicy.VOLATILE,
    )


class SimStereoCameraPublisher(Node):
    """Publish MuJoCo RGB stereo SHM frames on the existing ROS camera topics."""

    def __init__(self, *, jpeg_quality: int = 90) -> None:
        super().__init__("igris_sim_stereo_camera")
        self._jpeg_quality = max(1, min(100, int(jpeg_quality)))
        stereo_qos = build_sim_stereo_qos()
        self._left_pub = self.create_publisher(
            CompressedImage,
            ROS_STEREO_LEFT_COMPRESSED_TOPIC,
            stereo_qos,
        )
        self._right_pub = self.create_publisher(
            CompressedImage,
            ROS_STEREO_RIGHT_COMPRESSED_TOPIC,
            stereo_qos,
        )
        self._left_info_pub = self.create_publisher(
            CameraInfo,
            ROS_STEREO_LEFT_INFO_TOPIC,
            stereo_qos,
        )
        self._right_info_pub = self.create_publisher(
            CameraInfo,
            ROS_STEREO_RIGHT_INFO_TOPIC,
            stereo_qos,
        )
        self._last_warning_at = 0.0

    @staticmethod
    def _valid_frame(frame: np.ndarray) -> bool:
        return (
            frame.dtype == np.uint8
            and frame.shape == (STEREO_CAMERA_HEIGHT, STEREO_CAMERA_WIDTH, 3)
            and bool(np.any(frame))
        )

    def _encode(self, frame_rgb: np.ndarray, *, frame_id: str, stamp) -> CompressedImage | None:
        frame_bgr = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
        )
        if not ok:
            return None
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        return msg

    @staticmethod
    def _camera_info(*, frame_id: str, stamp, right: bool, baseline_m: float) -> CameraInfo:
        fx, fy, cx, cy = pinhole_intrinsics()
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.width = STEREO_CAMERA_WIDTH
        msg.height = STEREO_CAMERA_HEIGHT
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0] * 5
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = list(stereo_projection(baseline_m=baseline_m, right=right))
        return msg

    def publish_frames(
        self,
        camera_data: Mapping[str, np.ndarray],
        *,
        baseline_m: float = STEREO_CAMERA_BASELINE_M,
    ) -> bool:
        left = np.asarray(camera_data.get("stereo_left"))
        right = np.asarray(camera_data.get("stereo_right"))
        if not self._valid_frame(left) or not self._valid_frame(right):
            return False

        stamp = self.get_clock().now().to_msg()
        left_msg = self._encode(left, frame_id=STEREO_LEFT_FRAME_ID, stamp=stamp)
        right_msg = self._encode(right, frame_id=STEREO_RIGHT_FRAME_ID, stamp=stamp)
        if left_msg is None or right_msg is None:
            now = time.monotonic()
            if now - self._last_warning_at >= 5.0:
                self._last_warning_at = now
                self.get_logger().warning("Failed to JPEG-encode simulated stereo frames")
            return False

        self._left_pub.publish(left_msg)
        self._right_pub.publish(right_msg)
        self._left_info_pub.publish(
            self._camera_info(
                frame_id=STEREO_LEFT_FRAME_ID,
                stamp=stamp,
                right=False,
                baseline_m=baseline_m,
            )
        )
        self._right_info_pub.publish(
            self._camera_info(
                frame_id=STEREO_RIGHT_FRAME_ID,
                stamp=stamp,
                right=True,
                baseline_m=baseline_m,
            )
        )
        return True
