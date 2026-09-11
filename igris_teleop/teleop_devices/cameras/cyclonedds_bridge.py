from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import igris_c_sdk as igc_sdk
except ImportError:
    igc_sdk = None

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage

from ...core.project_paths import REPO_ROOT
from ...core.worker_base import DEFAULT_CAMERA_DOMAIN_ID
from .ros_interface import CAMERA_SHAPES, prepare_camera_frame

import logging_mp

logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)


DDS_D435_COLOR = "igris_c/sensor/d435_color"
DDS_D435_DEPTH = "igris_c/sensor/d435_depth"
DDS_EYES_STEREO = "igris_c/sensor/eyes_stereo"
DDS_LEFT_HAND = "igris_c/sensor/left_hand"
DDS_RIGHT_HAND = "igris_c/sensor/right_hand"

ROS_HEAD_COLOR = "/rs_comp/cam_213622075556/color/image/compressed"
ROS_LEFT_WRIST_COLOR = "/rs_comp/cam_335122271161/color/image/compressed"
ROS_RIGHT_WRIST_COLOR = "/rs_comp/cam_335122271403/color/image/compressed"
ROS_HEAD_DEPTH = "/rs_comp/cam_213622075556/depth/image/compressed"
ROS_STEREO_LEFT = "/left/image_rect/compressed"
ROS_STEREO_RIGHT = "/right/image_rect/compressed"
ROS_STEREO_LEFT_INFO = "/left/camera_info"
ROS_STEREO_RIGHT_INFO = "/right/camera_info"
ROS_COMBINED_COLOR = "/rs_comp/combined/color/image/compressed"

DEFAULT_STEREO_SWAP = False
DEFAULT_STEREO_OUTPUT_WIDTH = 640
DEFAULT_STEREO_OUTPUT_HEIGHT = 480
DEFAULT_STEREO_JPEG_QUALITY = 85
DEFAULT_COMBINED_RESIZE_SCALE = 0.5
DEFAULT_COMBINED_JPEG_QUALITY = 85
DEFAULT_STATUS_LOG_PERIOD_S = 5.0
DEFAULT_WARN_THROTTLE_PERIOD_S = 5.0
DEFAULT_STEREO_LEFT_FRAME_ID = "left_camera"
DEFAULT_STEREO_RIGHT_FRAME_ID = "right_camera"

DEFAULT_STEREO_MAP_PATH = (
    REPO_ROOT
    / "ros_ws"
    / "src"
    / "stereo_sbs_cam_pub"
    / "config"
    / "stereo_rectify_maps_tuned.npz"
)

_COLOR_STREAM_TO_SHM_KEY: dict[str, str] = {
    DDS_D435_COLOR: "realsense_head",
    DDS_LEFT_HAND: "realsense_wrist_left",
    DDS_RIGHT_HAND: "realsense_wrist_right",
}

_ROTATE_180_COLOR_TOPICS: frozenset[str] = frozenset({DDS_RIGHT_HAND})


@dataclass(frozen=True)
class _DDSHeaderSnapshot:
    sec: int = 0
    nanosec: int = 0
    frame_id: str = ""
    seq: int = 0


@dataclass(frozen=True)
class _DDSMessageSnapshot:
    payload: bytes
    format: str
    header: _DDSHeaderSnapshot


@dataclass
class _StreamState:
    name: str
    fps: float = 0.0
    bytes_per_sec: float = 0.0
    last_time: float = 0.0
    last_message: _DDSMessageSnapshot | None = None
    sequence: int = 0
    processed_sequence: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, message: _DDSMessageSnapshot) -> None:
        now = time.perf_counter()
        with self.lock:
            if self.last_time > 0.0:
                elapsed = now - self.last_time
                if elapsed > 0.0:
                    inst_fps = 1.0 / elapsed
                    inst_bps = len(message.payload) / elapsed
                    alpha = 0.15
                    if self.fps <= 0.0:
                        self.fps = inst_fps
                        self.bytes_per_sec = inst_bps
                    else:
                        self.fps = (1.0 - alpha) * self.fps + alpha * inst_fps
                        self.bytes_per_sec = (1.0 - alpha) * self.bytes_per_sec + alpha * inst_bps
            self.last_time = now
            self.last_message = message
            self.sequence += 1

    def get_unprocessed(self) -> tuple[int, _DDSMessageSnapshot | None]:
        with self.lock:
            if self.sequence == self.processed_sequence:
                return self.sequence, None
            return self.sequence, self.last_message

    def mark_processed(self, sequence: int) -> None:
        with self.lock:
            if sequence > self.processed_sequence:
                self.processed_sequence = sequence

    def snapshot(self) -> tuple[float, float]:
        with self.lock:
            return self.fps, self.bytes_per_sec


def _build_reliable_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        durability=DurabilityPolicy.VOLATILE,
    )


def _decode_payload(payload: bytes, flags: int) -> Optional[np.ndarray]:
    if not payload:
        return None
    raw = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(raw, flags)
    if frame is None or frame.size == 0:
        return None
    return frame


def _bgr_to_rgb(frame: np.ndarray) -> Optional[np.ndarray]:
    try:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def _normalize_compressed_format(value: str | None) -> Optional[str]:
    fmt = str(value or "").strip().lower()
    if not fmt:
        return None
    if "jpeg" in fmt or "jpg" in fmt:
        return "jpeg"
    if "png" in fmt:
        return "png"
    return None


def _encoding_extension_for_format(fmt: str) -> str:
    normalized = _normalize_compressed_format(fmt)
    if normalized == "jpeg":
        return ".jpg"
    if normalized == "png":
        return ".png"
    raise ValueError(f"unsupported compressed format: {fmt!r}")


def _resize_by_scale(frame: np.ndarray, scale: float) -> np.ndarray:
    if abs(float(scale) - 1.0) <= 1e-9:
        return frame.copy()
    new_w = max(1, int(round(frame.shape[1] * float(scale))))
    new_h = max(1, int(round(frame.shape[0] * float(scale))))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _scale_projection_matrix(projection: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = np.asarray(projection, dtype=np.float64).copy()
    scaled[0, :] *= float(scale_x)
    scaled[1, :] *= float(scale_y)
    return scaled


def _should_rotate_color_topic(topic: str) -> bool:
    return topic in _ROTATE_180_COLOR_TOPICS


class CycloneDDSCameraBridge(Node):
    def __init__(
        self,
        *,
        domain_id: int = DEFAULT_CAMERA_DOMAIN_ID,
        stereo_map_path: Path = DEFAULT_STEREO_MAP_PATH,
        status_log_period_s: float = DEFAULT_STATUS_LOG_PERIOD_S,
    ) -> None:
        if igc_sdk is None:
            raise RuntimeError("igris_c_sdk is not available. Cannot use camera_pub mode.")

        super().__init__("camera_cyclonedds_bridge")

        self._status_log_period_s = max(1.0, float(status_log_period_s))
        self._last_status_log_ts = 0.0
        self._warn_log_ts: dict[str, float] = {}
        self._subscribers: list[object] = []
        self._subscriber_callbacks: list[object] = []
        self._stopped = False

        self._streams = {
            DDS_D435_COLOR: _StreamState(DDS_D435_COLOR),
            DDS_D435_DEPTH: _StreamState(DDS_D435_DEPTH),
            DDS_EYES_STEREO: _StreamState(DDS_EYES_STEREO),
            DDS_LEFT_HAND: _StreamState(DDS_LEFT_HAND),
            DDS_RIGHT_HAND: _StreamState(DDS_RIGHT_HAND),
        }

        self._combined_resize_scale = DEFAULT_COMBINED_RESIZE_SCALE
        self._combined_jpeg_quality = DEFAULT_COMBINED_JPEG_QUALITY
        self._combined_frames: dict[str, np.ndarray | None] = {
            DDS_LEFT_HAND: None,
            DDS_D435_COLOR: None,
            DDS_RIGHT_HAND: None,
        }
        self._combined_lock = threading.Lock()

        self._stereo_swap = DEFAULT_STEREO_SWAP
        self._stereo_output_width = DEFAULT_STEREO_OUTPUT_WIDTH
        self._stereo_output_height = DEFAULT_STEREO_OUTPUT_HEIGHT
        self._stereo_jpeg_quality = DEFAULT_STEREO_JPEG_QUALITY

        self._map1x: np.ndarray | None = None
        self._map1y: np.ndarray | None = None
        self._map2x: np.ndarray | None = None
        self._map2y: np.ndarray | None = None
        self._stereo_p1: np.ndarray | None = None
        self._stereo_p2: np.ndarray | None = None
        self._stereo_rect_width = 0
        self._stereo_rect_height = 0
        self._load_stereo_maps(stereo_map_path)

        direct_image_qos = qos_profile_sensor_data
        reliable_image_qos = _build_reliable_qos()
        info_qos = _build_reliable_qos()
        self._ros_publishers = {
            DDS_D435_COLOR: self.create_publisher(CompressedImage, ROS_HEAD_COLOR, direct_image_qos),
            DDS_LEFT_HAND: self.create_publisher(CompressedImage, ROS_LEFT_WRIST_COLOR, direct_image_qos),
            DDS_RIGHT_HAND: self.create_publisher(CompressedImage, ROS_RIGHT_WRIST_COLOR, direct_image_qos),
            DDS_EYES_STEREO + ":left": self.create_publisher(CompressedImage, ROS_STEREO_LEFT, reliable_image_qos),
            DDS_EYES_STEREO + ":right": self.create_publisher(CompressedImage, ROS_STEREO_RIGHT, reliable_image_qos),
            DDS_D435_DEPTH: self.create_publisher(CompressedImage, ROS_HEAD_DEPTH, direct_image_qos),
            "combined": self.create_publisher(CompressedImage, ROS_COMBINED_COLOR, reliable_image_qos),
        }
        self._stereo_left_info_pub = self.create_publisher(CameraInfo, ROS_STEREO_LEFT_INFO, info_qos)
        self._stereo_right_info_pub = self.create_publisher(CameraInfo, ROS_STEREO_RIGHT_INFO, info_qos)

        self._channel_factory = igc_sdk.ChannelFactory.Instance()
        if self._channel_factory.IsInitialized():
            current_domain = int(self._channel_factory.GetDomainId())
            if current_domain != int(domain_id):
                logger.warning(
                    "[CycloneDDSCameraBridge] ChannelFactory already initialized on domain %d; requested domain %d.",
                    current_domain,
                    int(domain_id),
                )
        self._channel_factory.Init(int(domain_id))

        for topic in self._streams:
            subscriber = igc_sdk.CompressedMessageSubscriber(topic)
            callback = lambda msg, topic=topic: self._on_message(topic, msg)
            ok = subscriber.init(callback)
            if ok is False:
                raise RuntimeError(f"Failed to init CompressedMessageSubscriber({topic})")
            self._subscribers.append(subscriber)
            self._subscriber_callbacks.append(callback)

        logger.info(
            "[CycloneDDSCameraBridge] initialized domain_id=%d stereo_maps=%s",
            int(domain_id),
            str(stereo_map_path),
        )

    def _warning_throttle(self, key: str, message: str, *args: object) -> None:
        now = time.monotonic()
        last_ts = self._warn_log_ts.get(key, 0.0)
        if now - last_ts < DEFAULT_WARN_THROTTLE_PERIOD_S:
            return
        self._warn_log_ts[key] = now
        logger.warning(message, *args)

    def _load_stereo_maps(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Stereo rectification map is missing: {path}")

        data = np.load(str(path))
        if "mapL1" in data:
            self._map1x = np.asarray(data["mapL1"])
            self._map1y = np.asarray(data["mapL2"])
            self._map2x = np.asarray(data["mapR1"])
            self._map2y = np.asarray(data["mapR2"])
        elif "map1x" in data:
            self._map1x = np.asarray(data["map1x"])
            self._map1y = np.asarray(data["map1y"])
            self._map2x = np.asarray(data["map2x"])
            self._map2y = np.asarray(data["map2y"])
        else:
            raise RuntimeError(f"Unsupported stereo map keys in {path}")

        if "P1" not in data or "P2" not in data:
            raise RuntimeError(f"Stereo projection matrices are missing in {path}")
        self._stereo_p1 = np.asarray(data["P1"], dtype=np.float64)
        self._stereo_p2 = np.asarray(data["P2"], dtype=np.float64)

        size_key = "img_size" if "img_size" in data else "image_size" if "image_size" in data else None
        if size_key is not None:
            flat = np.asarray(data[size_key]).reshape(-1)
            if flat.size >= 2:
                self._stereo_rect_width = int(round(float(flat[0])))
                self._stereo_rect_height = int(round(float(flat[1])))

        if self._stereo_rect_width <= 0 or self._stereo_rect_height <= 0:
            self._stereo_rect_height, self._stereo_rect_width = self._map1x.shape[:2]

        if self._map1x is None or self._map1y is None or self._map2x is None or self._map2y is None:
            raise RuntimeError(f"Stereo rectification maps are incomplete in {path}")
        if self._stereo_rect_width <= 0 or self._stereo_rect_height <= 0:
            raise RuntimeError(f"Stereo rectification size is invalid in {path}")

    def _on_message(self, topic: str, msg: object) -> None:
        stream = self._streams.get(topic)
        if stream is None:
            return

        try:
            header_obj = msg.header()
            header = _DDSHeaderSnapshot(
                sec=int(header_obj.sec()),
                nanosec=int(header_obj.nanosec()),
                frame_id=str(header_obj.frame_id() or ""),
                seq=int(header_obj.seq()),
            )
            snapshot = _DDSMessageSnapshot(
                payload=bytes(msg.image_data()),
                format=str(msg.format() or ""),
                header=header,
            )
        except Exception as exc:
            self._warning_throttle(
                f"snapshot:{topic}",
                "[CycloneDDSCameraBridge] failed to snapshot DDS message from %s: %s",
                topic,
                exc,
            )
            return

        stream.update(snapshot)

    def _header_to_stamp(self, header: _DDSHeaderSnapshot):
        if header.sec == 0 and header.nanosec == 0:
            return self.get_clock().now().to_msg()
        stamp = self.get_clock().now().to_msg()
        stamp.sec = int(header.sec)
        stamp.nanosec = int(header.nanosec)
        return stamp

    def _to_passthrough_ros_image(self, snapshot: _DDSMessageSnapshot) -> CompressedImage:
        msg = CompressedImage()
        msg.header.stamp = self._header_to_stamp(snapshot.header)
        msg.header.frame_id = snapshot.header.frame_id
        msg.format = snapshot.format
        msg.data = snapshot.payload
        return msg

    def _encode_compressed(
        self,
        frame: np.ndarray,
        *,
        stamp,
        frame_id: str,
        fmt: str,
        jpeg_quality: int,
    ) -> Optional[CompressedImage]:
        try:
            encode_ext = _encoding_extension_for_format(fmt)
        except ValueError:
            self._warning_throttle(
                f"format:{fmt}",
                "[CycloneDDSCameraBridge] unsupported output compressed format %s",
                fmt,
            )
            return None

        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)] if fmt == "jpeg" else []
        ok, encoded = cv2.imencode(encode_ext, frame, params)
        if not ok:
            logger.warning("[CycloneDDSCameraBridge] failed to encode %s image", fmt)
            return None

        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.format = fmt
        msg.data = encoded.tobytes()
        return msg

    def _make_camera_info(self, *, stamp, frame_id: str, projection: np.ndarray, size: tuple[int, int]) -> CameraInfo:
        projection64 = np.asarray(projection, dtype=np.float64)
        width, height = int(size[0]), int(size[1])

        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.width = width
        msg.height = height
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.k = [
            float(projection64[0, 0]),
            float(projection64[0, 1]),
            float(projection64[0, 2]),
            float(projection64[1, 0]),
            float(projection64[1, 1]),
            float(projection64[1, 2]),
            float(projection64[2, 0]),
            float(projection64[2, 1]),
            float(projection64[2, 2]),
        ]
        msg.p = [
            float(projection64[0, 0]),
            float(projection64[0, 1]),
            float(projection64[0, 2]),
            float(projection64[0, 3]),
            float(projection64[1, 0]),
            float(projection64[1, 1]),
            float(projection64[1, 2]),
            float(projection64[1, 3]),
            float(projection64[2, 0]),
            float(projection64[2, 1]),
            float(projection64[2, 2]),
            float(projection64[2, 3]),
        ]
        return msg

    def poll(self) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {}

        for topic in (DDS_D435_COLOR, DDS_LEFT_HAND, DDS_RIGHT_HAND):
            frame_payload = self._process_color_stream(topic)
            if frame_payload:
                payload.update(frame_payload)

        stereo_payload = self._process_stereo_stream()
        if stereo_payload:
            payload.update(stereo_payload)

        self._process_depth_stream()
        self._maybe_log_status()
        return payload

    def _process_color_stream(self, topic: str) -> Optional[dict[str, np.ndarray]]:
        stream = self._streams[topic]
        sequence, snapshot = stream.get_unprocessed()
        if snapshot is None:
            return None

        try:
            bgr = _decode_payload(snapshot.payload, cv2.IMREAD_COLOR)
            if bgr is None:
                self._warning_throttle(
                    f"decode:{topic}",
                    "[CycloneDDSCameraBridge] failed to decode color payload from %s",
                    topic,
                )
                return None

            if _should_rotate_color_topic(topic):
                bgr = cv2.rotate(bgr, cv2.ROTATE_180)
                output_format = _normalize_compressed_format(snapshot.format) or "jpeg"
                msg = self._encode_compressed(
                    bgr,
                    stamp=self._header_to_stamp(snapshot.header),
                    frame_id=snapshot.header.frame_id,
                    fmt=output_format,
                    jpeg_quality=self._combined_jpeg_quality,
                )
                if msg is None:
                    self._warning_throttle(
                        f"publish:{topic}",
                        "[CycloneDDSCameraBridge] failed to publish rotated color payload for %s",
                        topic,
                    )
                    return None
                self._ros_publishers[topic].publish(msg)
            else:
                self._ros_publishers[topic].publish(self._to_passthrough_ros_image(snapshot))

            self._update_combined_color_output(topic, bgr, snapshot)

            rgb = _bgr_to_rgb(bgr)
            if rgb is None:
                self._warning_throttle(
                    f"rgb:{topic}",
                    "[CycloneDDSCameraBridge] failed to convert payload to RGB from %s",
                    topic,
                )
                return None

            shm_key = _COLOR_STREAM_TO_SHM_KEY[topic]
            shape = CAMERA_SHAPES.get(shm_key)
            if shape is None:
                return None

            prepared = prepare_camera_frame(rgb, shape)
            if prepared is None:
                self._warning_throttle(
                    f"shape:{shm_key}",
                    "[CycloneDDSCameraBridge] failed to shape frame for %s",
                    shm_key,
                )
                return None
            return {shm_key: prepared}
        finally:
            stream.mark_processed(sequence)

    def _update_combined_color_output(self, topic: str, frame: np.ndarray, snapshot: _DDSMessageSnapshot) -> None:
        if topic not in self._combined_frames:
            return

        resized = _resize_by_scale(frame, self._combined_resize_scale)
        with self._combined_lock:
            self._combined_frames[topic] = resized
            if any(self._combined_frames[key] is None for key in (DDS_LEFT_HAND, DDS_D435_COLOR, DDS_RIGHT_HAND)):
                return

            images = [
                self._combined_frames[DDS_LEFT_HAND],
                self._combined_frames[DDS_D435_COLOR],
                self._combined_frames[DDS_RIGHT_HAND],
            ]
            output_height = max(int(img.shape[0]) for img in images if img is not None)
            output_width = sum(int(img.shape[1]) for img in images if img is not None)
            combined = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            offset_x = 0
            for img in images:
                if img is None:
                    return
                offset_y = max(0, (output_height - int(img.shape[0])) // 2)
                combined[offset_y : offset_y + img.shape[0], offset_x : offset_x + img.shape[1]] = img
                offset_x += int(img.shape[1])

        stamp = self._header_to_stamp(snapshot.header)
        msg = self._encode_compressed(
            combined,
            stamp=stamp,
            frame_id=snapshot.header.frame_id,
            fmt="jpeg",
            jpeg_quality=self._combined_jpeg_quality,
        )
        if msg is not None:
            self._ros_publishers["combined"].publish(msg)

    def _process_depth_stream(self) -> None:
        stream = self._streams[DDS_D435_DEPTH]
        sequence, snapshot = stream.get_unprocessed()
        if snapshot is None:
            return

        try:
            self._ros_publishers[DDS_D435_DEPTH].publish(self._to_passthrough_ros_image(snapshot))
        finally:
            stream.mark_processed(sequence)

    def _process_stereo_stream(self) -> dict[str, np.ndarray]:
        stream = self._streams[DDS_EYES_STEREO]
        sequence, snapshot = stream.get_unprocessed()
        if snapshot is None:
            return {}

        try:
            output_format = _normalize_compressed_format(snapshot.format)
            if output_format is None:
                self._warning_throttle(
                    "stereo:format",
                    "[CycloneDDSCameraBridge] unsupported stereo compressed format %r; skipping transformed stereo frame",
                    snapshot.format,
                )
                return {}

            frame = _decode_payload(snapshot.payload, cv2.IMREAD_COLOR)
            if frame is None:
                self._warning_throttle(
                    "stereo:decode",
                    "[CycloneDDSCameraBridge] failed to decode stereo payload",
                )
                return {}

            width = int(frame.shape[1])
            if width <= 1 or width % 2 != 0:
                self._warning_throttle(
                    "stereo:width",
                    "[CycloneDDSCameraBridge] invalid stereo frame width=%d",
                    width,
                )
                return {}

            # The robot publishes the SBS frame upright. Preserve that
            # orientation and rectify each split half directly.
            half_width = width // 2
            left_raw = frame[:, :half_width].copy()
            right_raw = frame[:, half_width:].copy()

            if self._stereo_swap:
                left_raw, right_raw = right_raw, left_raw

            rectify_size = (self._stereo_rect_width, self._stereo_rect_height)
            if left_raw.shape[1] != self._stereo_rect_width or left_raw.shape[0] != self._stereo_rect_height:
                left_raw = cv2.resize(left_raw, rectify_size, interpolation=cv2.INTER_LINEAR)
            if right_raw.shape[1] != self._stereo_rect_width or right_raw.shape[0] != self._stereo_rect_height:
                right_raw = cv2.resize(right_raw, rectify_size, interpolation=cv2.INTER_LINEAR)

            left_rect = cv2.remap(left_raw, self._map1x, self._map1y, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_raw, self._map2x, self._map2y, cv2.INTER_LINEAR)

            output_size = (self._stereo_output_width, self._stereo_output_height)
            if left_rect.shape[1] != self._stereo_output_width or left_rect.shape[0] != self._stereo_output_height:
                left_rect = cv2.resize(left_rect, output_size, interpolation=cv2.INTER_AREA)
            if right_rect.shape[1] != self._stereo_output_width or right_rect.shape[0] != self._stereo_output_height:
                right_rect = cv2.resize(right_rect, output_size, interpolation=cv2.INTER_AREA)

            stamp = self._header_to_stamp(snapshot.header)
            left_msg = self._encode_compressed(
                left_rect,
                stamp=stamp,
                frame_id=DEFAULT_STEREO_LEFT_FRAME_ID,
                fmt=output_format,
                jpeg_quality=self._stereo_jpeg_quality,
            )
            right_msg = self._encode_compressed(
                right_rect,
                stamp=stamp,
                frame_id=DEFAULT_STEREO_RIGHT_FRAME_ID,
                fmt=output_format,
                jpeg_quality=self._stereo_jpeg_quality,
            )
            if left_msg is None or right_msg is None:
                return {}

            self._ros_publishers[DDS_EYES_STEREO + ":left"].publish(left_msg)
            self._ros_publishers[DDS_EYES_STEREO + ":right"].publish(right_msg)

            scale_x = float(self._stereo_output_width) / float(self._stereo_rect_width)
            scale_y = float(self._stereo_output_height) / float(self._stereo_rect_height)
            left_projection = _scale_projection_matrix(self._stereo_p1, scale_x, scale_y)
            right_projection = _scale_projection_matrix(self._stereo_p2, scale_x, scale_y)
            self._stereo_left_info_pub.publish(
                self._make_camera_info(
                    stamp=stamp,
                    frame_id=DEFAULT_STEREO_LEFT_FRAME_ID,
                    projection=left_projection,
                    size=output_size,
                )
            )
            self._stereo_right_info_pub.publish(
                self._make_camera_info(
                    stamp=stamp,
                    frame_id=DEFAULT_STEREO_RIGHT_FRAME_ID,
                    projection=right_projection,
                    size=output_size,
                )
            )

            left_rgb = _bgr_to_rgb(left_rect)
            right_rgb = _bgr_to_rgb(right_rect)
            if left_rgb is None or right_rgb is None:
                self._warning_throttle(
                    "stereo:rgb",
                    "[CycloneDDSCameraBridge] failed to convert stereo frame to RGB",
                )
                return {}

            stereo_left = prepare_camera_frame(left_rgb, CAMERA_SHAPES["stereo_left"])
            stereo_right = prepare_camera_frame(right_rgb, CAMERA_SHAPES["stereo_right"])
            if stereo_left is None or stereo_right is None:
                self._warning_throttle(
                    "stereo:shape",
                    "[CycloneDDSCameraBridge] failed to shape stereo frame",
                )
                return {}

            return {
                "stereo_left": stereo_left,
                "stereo_right": stereo_right,
            }
        finally:
            stream.mark_processed(sequence)

    def _maybe_log_status(self) -> None:
        now = time.perf_counter()
        if now - self._last_status_log_ts < self._status_log_period_s:
            return
        self._last_status_log_ts = now

        metrics: list[str] = []
        for topic in (
            DDS_D435_COLOR,
            DDS_LEFT_HAND,
            DDS_RIGHT_HAND,
            DDS_EYES_STEREO,
            DDS_D435_DEPTH,
        ):
            fps, bps = self._streams[topic].snapshot()
            metrics.append(f"{topic}: {fps:.1f} FPS {self._format_rate(bps)}")
        logger.info("[CycloneDDSCameraBridge] %s", " | ".join(metrics))

    @staticmethod
    def _format_rate(bytes_per_sec: float) -> str:
        if bytes_per_sec > 1024.0 * 1024.0:
            return f"{bytes_per_sec / (1024.0 * 1024.0):.1f} MB/s"
        if bytes_per_sec > 1024.0:
            return f"{bytes_per_sec / 1024.0:.1f} KB/s"
        return f"{bytes_per_sec:.0f} B/s"

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        for subscriber in self._subscribers:
            try:
                subscriber.stop()
            except Exception:
                logger.debug("[CycloneDDSCameraBridge] failed to stop subscriber", exc_info=True)

        try:
            self._channel_factory.Release()
        except Exception:
            logger.debug("[CycloneDDSCameraBridge] failed to release ChannelFactory", exc_info=True)
