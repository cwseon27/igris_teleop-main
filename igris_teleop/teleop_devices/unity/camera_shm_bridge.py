from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from multiprocessing import Lock as MpLock

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

from ...sharedmemory.shmManager import SharedMemoryManager
from ...sharedmemory.shm_schema import CAMERA


# ------------------------- CAMERA schema helpers -------------------------
CAMERA_SHAPES = {name: shape for name, shape, _ in CAMERA}


def _prepare_frame(frame: np.ndarray, shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """Resize/cast frame to target shape (H, W, C)."""
    if frame is None:
        return None
    try:
        target_h, target_w, target_c = shape
        if frame.ndim != 3:
            return None
        if frame.shape[2] != target_c:
            return None
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            frame = cv2.resize(frame, (target_w, target_h))
        if frame.dtype != np.uint8:
            frame = np.asarray(frame, dtype=np.uint8)
        return frame
    except Exception:
        return None


class CameraShmBridge:
    """ROS2 compressed image topics -> CAMERA SHM."""

    def __init__(
        self,
        node: Node,
        *,
        camera_shm_name: str = "camera_shm",
        camera_lock: Optional[MpLock] = None,
        stereo_left_topic: str = "/stereo_left/image_rect/compressed",
        stereo_right_topic: str = "/stereo_right/image_rect/compressed",
        realsense_head_topic: str = "/realsense_head/image_rect/compressed",
        realsense_wrist_left_topic: str = "/realsense_wrist_left/image_rect/compressed",
        realsense_wrist_right_topic: str = "/realsense_wrist_right/image_rect/compressed",
    ) -> None:
        self._node = node
        self._camera_shm: Optional[SharedMemoryManager] = None
        self._stereo_left_shape: Optional[Tuple[int, int, int]] = None
        self._stereo_right_shape: Optional[Tuple[int, int, int]] = None
        self._realsense_head_shape: Optional[Tuple[int, int, int]] = None
        self._realsense_wrist_left_shape: Optional[Tuple[int, int, int]] = None
        self._realsense_wrist_right_shape: Optional[Tuple[int, int, int]] = None

        self._stereo_left_shape = CAMERA_SHAPES.get("stereo_left")
        self._stereo_right_shape = CAMERA_SHAPES.get("stereo_right")
        self._realsense_head_shape = CAMERA_SHAPES.get("realsense_head")
        self._realsense_wrist_left_shape = CAMERA_SHAPES.get("realsense_wrist_left")
        self._realsense_wrist_right_shape = CAMERA_SHAPES.get("realsense_wrist_right")

        cam_lock = camera_lock if camera_lock is not None else MpLock()
        self._camera_shm = SharedMemoryManager(CAMERA, cam_lock, camera_shm_name)

        if stereo_left_topic:
            node.create_subscription(
                CompressedImage,
                stereo_left_topic,
                self._on_left_image,
                qos_profile_sensor_data,
            )
        if stereo_right_topic:
            node.create_subscription(
                CompressedImage,
                stereo_right_topic,
                self._on_right_image,
                qos_profile_sensor_data,
            )
        if realsense_head_topic:
            node.create_subscription(
                CompressedImage,
                realsense_head_topic,
                self._on_head_image,
                qos_profile_sensor_data,
            )
        if realsense_wrist_left_topic:
            node.create_subscription(
                CompressedImage,
                realsense_wrist_left_topic,
                self._on_wrist_left_image,
                qos_profile_sensor_data,
            )
        if realsense_wrist_right_topic:
            node.create_subscription(
                CompressedImage,
                realsense_wrist_right_topic,
                self._on_wrist_right_image,
                qos_profile_sensor_data,
            )

    # ----------------------------- camera callbacks -----------------------------
    def _decode_compressed_to_rgb(self, msg: CompressedImage) -> Optional[np.ndarray]:
        """sensor_msgs/CompressedImage -> RGB uint8 (H,W,3)."""
        try:
            if msg is None or not msg.data:
                return None
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        except Exception:
            return None

    def _on_left_image(self, msg: CompressedImage) -> None:
        shm = self._camera_shm
        shape = self._stereo_left_shape
        if shm is None or shape is None:
            return
        frame = self._decode_compressed_to_rgb(msg)
        if frame is None:
            return
        frame = _prepare_frame(frame, shape)
        if frame is None:
            return
        try:
            shm.write_data(stereo_left=frame)
        except Exception:
            # throttle_duration_sec는 rclpy 버전에 따라 지원이 다를 수 있어 기본 로그로 처리
            self._node.get_logger().debug("failed to write left camera frame to SHM")

    def _on_right_image(self, msg: CompressedImage) -> None:
        shm = self._camera_shm
        shape = self._stereo_right_shape
        if shm is None or shape is None:
            return
        frame = self._decode_compressed_to_rgb(msg)
        if frame is None:
            return
        frame = _prepare_frame(frame, shape)
        if frame is None:
            return
        try:
            shm.write_data(stereo_right=frame)
        except Exception:
            self._node.get_logger().debug("failed to write right camera frame to SHM")

    def _on_head_image(self, msg: CompressedImage) -> None:
        shm = self._camera_shm
        shape = self._realsense_head_shape
        if shm is None or shape is None:
            return
        frame = self._decode_compressed_to_rgb(msg)
        if frame is None:
            return
        frame = _prepare_frame(frame, shape)
        if frame is None:
            return
        try:
            shm.write_data(realsense_head=frame)
        except Exception:
            self._node.get_logger().debug("failed to write head realsense frame to SHM")

    def _on_wrist_left_image(self, msg: CompressedImage) -> None:
        shm = self._camera_shm
        shape = self._realsense_wrist_left_shape
        if shm is None or shape is None:
            return
        frame = self._decode_compressed_to_rgb(msg)
        if frame is None:
            return
        frame = _prepare_frame(frame, shape)
        if frame is None:
            return
        try:
            shm.write_data(realsense_wrist_left=frame)
        except Exception:
            self._node.get_logger().debug("failed to write left wrist realsense frame to SHM")

    def _on_wrist_right_image(self, msg: CompressedImage) -> None:
        shm = self._camera_shm
        shape = self._realsense_wrist_right_shape
        if shm is None or shape is None:
            return
        frame = self._decode_compressed_to_rgb(msg)
        if frame is None:
            return
        frame = _prepare_frame(frame, shape)
        if frame is None:
            return
        try:
            shm.write_data(realsense_wrist_right=frame)
        except Exception:
            self._node.get_logger().debug("failed to write right wrist realsense frame to SHM")
