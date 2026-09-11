import threading
from typing import Dict, Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage

from ...sharedmemory.shm_schema import CAMERA


# ------------------------- CAMERA schema helpers -------------------------
CAMERA_SHAPES = {name: shape for name, shape, _ in CAMERA}


def build_camera_qos(qos_depth: int = 1) -> QoSProfile:
    return QoSProfile(
        depth=qos_depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
    )


def prepare_camera_frame(frame: np.ndarray, shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """Resize/cast frame to target shape (H, W, C)."""
    if frame is None:
        return None
    try:
        target_h, target_w, target_c = shape
        if frame.ndim != 3 or frame.shape[2] != target_c:
            return None
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            frame = cv2.resize(frame, (target_w, target_h))
        if frame.dtype != np.uint8:
            frame = np.asarray(frame, dtype=np.uint8)
        return frame
    except Exception:
        return None


def _decode_compressed_to_rgb(msg: CompressedImage) -> Optional[np.ndarray]:
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


class CameraROSInterface(Node):
    """ROS2 camera subscriber (compressed) -> latest RGB frames."""

    def __init__(
        self,
        *,
        stereo_left_topic: str = "/left/image_rect/compressed",
        stereo_right_topic: str = "/right/image_rect/compressed",
        realsense_head_topic: str = "/rs_comp/cam_213622075556/color/image/compressed",
        realsense_wrist_left_topic: str = "/rs_comp/cam_335122271161/color/image/compressed",
        realsense_wrist_right_topic: str = "/rs_comp/cam_335122271403/color/image/compressed",
        qos_depth: int = 1,
    ) -> None:
        super().__init__("camera_ros_bridge")
        qos = build_camera_qos(qos_depth=qos_depth)

        self._lock = threading.Lock()
        self._topic_by_key: Dict[str, str] = {}
        self._shapes: Dict[str, Optional[Tuple[int, int, int]]] = {
            "stereo_left": CAMERA_SHAPES.get("stereo_left"),
            "stereo_right": CAMERA_SHAPES.get("stereo_right"),
            "realsense_head": CAMERA_SHAPES.get("realsense_head"),
            "realsense_wrist_left": CAMERA_SHAPES.get("realsense_wrist_left"),
            "realsense_wrist_right": CAMERA_SHAPES.get("realsense_wrist_right"),
        }
        self._frames: Dict[str, Optional[np.ndarray]] = {k: None for k in self._shapes}
        self._events: Dict[str, threading.Event] = {k: threading.Event() for k in self._shapes}

        if stereo_left_topic:
            self._topic_by_key["stereo_left"] = stereo_left_topic
            self.create_subscription(
                CompressedImage,
                stereo_left_topic,
                self._make_cb("stereo_left"),
                qos,
            )
        if stereo_right_topic:
            self._topic_by_key["stereo_right"] = stereo_right_topic
            self.create_subscription(
                CompressedImage,
                stereo_right_topic,
                self._make_cb("stereo_right"),
                qos,
            )
        if realsense_head_topic:
            self._topic_by_key["realsense_head"] = realsense_head_topic
            self.create_subscription(
                CompressedImage,
                realsense_head_topic,
                self._make_cb("realsense_head"),
                qos,
            )
        if realsense_wrist_left_topic:
            self._topic_by_key["realsense_wrist_left"] = realsense_wrist_left_topic
            self.create_subscription(
                CompressedImage,
                realsense_wrist_left_topic,
                self._make_cb("realsense_wrist_left"),
                qos,
            )
        if realsense_wrist_right_topic:
            self._topic_by_key["realsense_wrist_right"] = realsense_wrist_right_topic
            self.create_subscription(
                CompressedImage,
                realsense_wrist_right_topic,
                self._make_cb("realsense_wrist_right"),
                qos,
            )

    def _make_cb(self, key: str):
        def _cb(msg: CompressedImage) -> None:
            self._on_image(msg, key)
        return _cb

    # ----------------------------- callbacks -----------------------------
    def _on_image(self, msg: CompressedImage, key: str) -> None:
        frame = _decode_compressed_to_rgb(msg)
        if frame is None:
            return
        shape = self._shapes.get(key)
        if shape is not None:
            frame = prepare_camera_frame(frame, shape)
        if frame is None:
            return
        with self._lock:
            self._frames[key] = frame
        ev = self._events.get(key)
        if ev is not None:
            ev.set()

    # ----------------------------- getters -----------------------------
    def wait_for_frames(self, timeout: float = 5.0, keys: Optional[Tuple[str, ...]] = None) -> bool:
        """Wait until selected frames are received at least once."""
        if keys is None:
            keys = ("stereo_left", "stereo_right")
        ok = True
        for key in keys:
            ev = self._events.get(key)
            if ev is None:
                ok = False
                continue
            ok = ev.wait(timeout=timeout) and ok
        return bool(ok)

    def get_frames(self) -> Dict[str, Optional[np.ndarray]]:
        """Return latest frames as copies."""
        with self._lock:
            frames = {k: (None if v is None else v.copy()) for k, v in self._frames.items()}
        return frames

    def get_missing_frame_topics(self) -> Dict[str, str]:
        with self._lock:
            missing_keys = [k for k, v in self._frames.items() if v is None]
        return {key: self._topic_by_key[key] for key in missing_keys if key in self._topic_by_key}

    def count_publishers_for_key(self, key: str) -> Optional[int]:
        topic = self._topic_by_key.get(key)
        if not topic:
            return None
        try:
            return int(self.count_publishers(topic))
        except Exception:
            return None
