#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
import math
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage


@dataclass
class DeviceCtx:
    serial: str
    cam_name: str
    pipeline: object
    cfg: object
    running: bool = False
    thread: Optional[threading.Thread] = None


class MultiRealSenseCompressedPub(Node):
    """
    - 연결된 RealSense를 자동 감지
    - 각 디바이스에서 color / infra1 / infra2 스트림을 받아
    - JPEG로 압축한 CompressedImage만 publish
    """

    def __init__(self):
        super().__init__("realsense_multi_compressed_pub")

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("base_ns", "/rs_comp")
        self.declare_parameter("enable_color", True)
        self.declare_parameter("enable_infra", True)

        # profile: "WIDTHxHEIGHTxFPS"
        self.declare_parameter("color_profile", "640x480x30")
        self.declare_parameter("infra_profile", "640x480x30")

        self.declare_parameter("jpeg_quality", 80)

        # 로그 스팸 방지: 같은 에러는 N초에 1번만 출력
        self.declare_parameter("log_throttle_sec", 5.0)

        # disconnect 시 재시도 주기
        self.declare_parameter("reconnect_wait_sec", 2.0)

        # publish 옵션
        self.declare_parameter("publish_individual", True)
        self.declare_parameter("publish_combined", True)
        self.declare_parameter("combined_stream", "color")  # color | infra1 | infra2
        self.declare_parameter("combined_publish_hz", 30.0)
        self.declare_parameter(
            "combined_serial_order",
            ["335122271161", "213622075556", "335122271403"],
        )
        self.declare_parameter(
            "combined_rotate_180_serials",
            ["335122271403"],
        )

        self.base_ns = self.get_parameter("base_ns").value.rstrip("/")
        self.enable_color = bool(self.get_parameter("enable_color").value)
        self.enable_infra = bool(self.get_parameter("enable_infra").value)
        self.color_profile = self.get_parameter("color_profile").value
        self.infra_profile = self.get_parameter("infra_profile").value
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.log_throttle = float(self.get_parameter("log_throttle_sec").value)
        self.reconnect_wait = float(self.get_parameter("reconnect_wait_sec").value)

        self.publish_individual = bool(self.get_parameter("publish_individual").value)
        self.publish_combined = bool(self.get_parameter("publish_combined").value)
        self.combined_stream = str(self.get_parameter("combined_stream").value)
        self.combined_publish_hz = float(self.get_parameter("combined_publish_hz").value)
        order_param = self.get_parameter("combined_serial_order").value
        rotate_param = self.get_parameter("combined_rotate_180_serials").value
        if isinstance(order_param, (list, tuple)):
            self.combined_serial_order = [str(s) for s in order_param]
        else:
            self.combined_serial_order = []
        if isinstance(rotate_param, (list, tuple)):
            self.combined_rotate_180_serials = {str(s) for s in rotate_param}
        else:
            self.combined_rotate_180_serials = set()

        self.declare_parameter("enable_infra2", False)
        self.enable_infra2 = bool(self.get_parameter("enable_infra2").value)


        self._last_log: Dict[str, float] = {}
        self._stop = False

        # ----------------------------
        # RealSense import & detect devices
        # ----------------------------
        try:
            import pyrealsense2 as rs
        except Exception as e:
            raise RuntimeError(
                "pyrealsense2 import 실패. (pip/apt로 pyrealsense2 설치 필요)\n"
                f"원인: {e}"
            )

        self.rs = rs
        serials = self._detect_serials()
        if not serials:
            self.get_logger().error("RealSense 디바이스를 찾지 못했습니다.")
            raise RuntimeError("No RealSense devices found")

        # 안정적인 순서를 위해 정렬
        serials = sorted(serials)
        self.serials = serials
        self.combined_serials = self._resolve_combined_serials(serials)

        self.get_logger().info(f"Detected {len(serials)} device(s): {serials}")

        # ----------------------------
        # Publishers per device/stream
        # ----------------------------
        self.pubs: Dict[str, Dict[str, object]] = {}  # pubs[serial][stream] = publisher

        # device contexts
        self.devices: List[DeviceCtx] = []

        for sn in serials:
            cam_name = f"cam_{sn}"  # 토큰이 숫자로 시작하면 ROS 네이밍에서 불리하니 cam_ prefix
            self.pubs[sn] = {}

            if self.enable_color:
                topic = f"{self.base_ns}/{cam_name}/color/image/compressed"
                self.pubs[sn]["color"] = self.create_publisher(
                    CompressedImage, topic, qos_profile_sensor_data
                )

            if self.enable_infra:
                topic_l = f"{self.base_ns}/{cam_name}/infra1/image/compressed"
                topic_r = f"{self.base_ns}/{cam_name}/infra2/image/compressed"
                self.pubs[sn]["infra1"] = self.create_publisher(
                    CompressedImage, topic_l, qos_profile_sensor_data
                )
                if self.enable_infra2:

                    self.pubs[sn]["infra2"] = self.create_publisher(
                        CompressedImage, topic_r, qos_profile_sensor_data
                    )

            # Create pipeline/config per device
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(sn)

            # enable streams
            if self.enable_color:
                cw, ch, cfps = self._parse_profile(self.color_profile)
                # RealSense는 보통 RGB8로 나옴 -> 아래에서 BGR로 변환 후 JPEG 인코딩
                cfg.enable_stream(rs.stream.color, cw, ch, rs.format.rgb8, cfps)

            if self.enable_infra:
                iw, ih, ifps = self._parse_profile(self.infra_profile)
                cfg.enable_stream(rs.stream.infrared, 1, iw, ih, rs.format.y8, ifps)
                if self.enable_infra2:
                    cfg.enable_stream(rs.stream.infrared, 2, iw, ih, rs.format.y8, ifps)
                    
            self.devices.append(DeviceCtx(serial=sn, cam_name=cam_name, pipeline=pipeline, cfg=cfg))

        # ----------------------------
        # Combined publisher (optional)
        # ----------------------------
        self.latest_frames: Dict[str, Dict[str, np.ndarray]] = {}
        self._frames_lock = threading.Lock()
        self.combined_pub = None

        if self.publish_combined:
            valid_streams = {"color", "infra1", "infra2"}
            if self.combined_stream not in valid_streams:
                self.get_logger().warn(
                    f"combined_stream '{self.combined_stream}'는 지원하지 않습니다. 'color'로 변경합니다."
                )
                self.combined_stream = "color"

            if self.combined_stream == "color" and not self.enable_color:
                self.get_logger().warn("enable_color=False 상태입니다. combined publish를 비활성화합니다.")
                self.publish_combined = False
            if self.combined_stream in ("infra1", "infra2") and not self.enable_infra:
                self.get_logger().warn("enable_infra=False 상태입니다. combined publish를 비활성화합니다.")
                self.publish_combined = False
            if self.combined_stream == "infra2" and not self.enable_infra2:
                self.get_logger().warn("enable_infra2=False 상태입니다. combined publish를 비활성화합니다.")
                self.publish_combined = False

        if self.publish_combined:
            if self.combined_publish_hz <= 0:
                self.get_logger().warn(
                    f"combined_publish_hz={self.combined_publish_hz}는 유효하지 않습니다. 30.0으로 변경합니다."
                )
                self.combined_publish_hz = 30.0

            combined_topic = f"{self.base_ns}/combined/{self.combined_stream}/image/compressed"
            self.combined_pub = self.create_publisher(
                CompressedImage, combined_topic, qos_profile_sensor_data
            )
            self.latest_frames = {sn: {} for sn in serials}
            self._combined_timer = self.create_timer(
                1.0 / self.combined_publish_hz, self._publish_combined
            )

        # ----------------------------
        # Start threads
        # ----------------------------
        for dev in self.devices:
            dev.thread = threading.Thread(target=self._device_loop, args=(dev,), daemon=True)
            dev.thread.start()

        self.get_logger().info(
            f"Started. Publishing ONLY CompressedImage under base_ns={self.base_ns}"
        )
        if self.publish_combined and self.combined_pub is not None:
            self.get_logger().info(
                f"Combined publish enabled: stream={self.combined_stream} -> {self.base_ns}/combined/{self.combined_stream}/image/compressed"
            )

    def _detect_serials(self) -> List[str]:
        rs = self.rs
        ctx = rs.context()
        serials = []
        for dev in ctx.query_devices():
            try:
                serials.append(dev.get_info(rs.camera_info.serial_number))
            except Exception:
                pass
        return serials

    @staticmethod
    def _parse_profile(s: str):
        # "640x480x30"
        w, h, fps = s.lower().split("x")
        return int(w), int(h), int(fps)

    def _log_throttled(self, key: str, msg: str, level: str = "warn"):
        now = time.time()
        last = self._last_log.get(key, 0.0)
        if now - last < self.log_throttle:
            return
        self._last_log[key] = now
        if level == "error":
            self.get_logger().error(msg)
        else:
            self.get_logger().warn(msg)

    def _store_latest(self, serial: str, stream: str, img: np.ndarray):
        if not self.publish_combined:
            return
        if stream != self.combined_stream:
            return
        with self._frames_lock:
            if serial not in self.latest_frames:
                self.latest_frames[serial] = {}
            self.latest_frames[serial][stream] = img.copy()

    def _combined_target(self, stream: str):
        if stream == "color":
            w, h, _fps = self._parse_profile(self.color_profile)
            channels = 3
        else:
            w, h, _fps = self._parse_profile(self.infra_profile)
            channels = 1
        tw = max(1, int(w // 2))
        th = max(1, int(h // 2))
        return tw, th, channels

    def _resolve_combined_serials(self, serials: List[str]) -> List[str]:
        if not self.combined_serial_order:
            return list(serials)

        ordered: List[str] = []
        missing: List[str] = []
        for sn in self.combined_serial_order:
            if sn in serials and sn not in ordered:
                ordered.append(sn)
            else:
                missing.append(sn)

        extras = [sn for sn in serials if sn not in ordered]

        if missing:
            self.get_logger().warn(
                f"combined_serial_order에 있으나 미연결된 디바이스: {missing}"
            )
        if extras:
            self.get_logger().warn(
                f"combined_serial_order에 없는 디바이스가 있어 뒤에 추가: {extras}"
            )

        return ordered + extras

    @staticmethod
    def _grid_shape(n: int):
        cols = max(1, int(n))
        rows = 1
        return rows, cols

    def _publish_combined(self):
        if not self.publish_combined or self.combined_pub is None:
            return

        stream = self.combined_stream
        target_w, target_h, channels = self._combined_target(stream)

        with self._frames_lock:
            frames = [
                self.latest_frames.get(sn, {}).get(stream)
                for sn in self.combined_serials
            ]

        if not any(f is not None for f in frames):
            return

        tiles = []
        for sn, frame in zip(self.combined_serials, frames):
            if frame is None:
                if channels == 3:
                    tile = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                else:
                    tile = np.zeros((target_h, target_w), dtype=np.uint8)
            else:
                if sn in self.combined_rotate_180_serials:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                if channels == 3 and frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if channels == 1 and frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if frame.shape[1] != target_w or frame.shape[0] != target_h:
                    interp = cv2.INTER_AREA
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=interp)
                tile = frame
            tiles.append(tile)

        rows, cols = self._grid_shape(len(tiles))
        if channels == 3:
            canvas = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)
        else:
            canvas = np.zeros((rows * target_h, cols * target_w), dtype=np.uint8)

        for idx, tile in enumerate(tiles):
            r = idx // cols
            c = idx % cols
            y0 = r * target_h
            x0 = c * target_w
            if channels == 3:
                canvas[y0:y0 + target_h, x0:x0 + target_w, :] = tile
            else:
                canvas[y0:y0 + target_h, x0:x0 + target_w] = tile

        self._publish_jpeg(self.combined_pub, canvas, frame_id=f"combined_{stream}")

    def _start_pipeline(self, dev: DeviceCtx) -> bool:
        try:
            dev.pipeline.start(dev.cfg)
            dev.running = True
            self.get_logger().info(f"[{dev.serial}] pipeline started")
            return True
        except Exception as e:
            dev.running = False
            self._log_throttled(f"start_{dev.serial}", f"[{dev.serial}] start 실패: {e}", "warn")
            return False

    def _stop_pipeline(self, dev: DeviceCtx):
        try:
            dev.pipeline.stop()
        except Exception:
            pass
        dev.running = False

    def _publish_jpeg(self, pub, img: np.ndarray, frame_id: str):
        # img: BGR (color) or GRAY (infra)
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.format = "jpeg"

        ok, enc = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        )
        if not ok:
            return
        msg.data = enc.tobytes()
        pub.publish(msg)

    def _device_loop(self, dev: DeviceCtx):
        rs = self.rs

        while rclpy.ok() and not self._stop:
            if not dev.running:
                if not self._start_pipeline(dev):
                    time.sleep(self.reconnect_wait)
                    continue

            try:
                frames = dev.pipeline.wait_for_frames(timeout_ms=1000)

                # color
                if self.enable_color and "color" in self.pubs[dev.serial]:
                    cf = frames.get_color_frame()
                    if cf:
                        rgb = np.asanyarray(cf.get_data())
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        if self.publish_individual:
                            self._publish_jpeg(
                                self.pubs[dev.serial]["color"],
                                bgr,
                                frame_id=f"{dev.cam_name}_color",
                            )
                        self._store_latest(dev.serial, "color", bgr)

                # infra1/infra2
                if self.enable_infra:
                    if "infra1" in self.pubs[dev.serial]:
                        f1 = frames.get_infrared_frame(1)
                        if f1:
                            ir1 = np.asanyarray(f1.get_data())  # GRAY8
                            if self.publish_individual:
                                self._publish_jpeg(
                                    self.pubs[dev.serial]["infra1"],
                                    ir1,
                                    frame_id=f"{dev.cam_name}_infra1",
                                )
                            self._store_latest(dev.serial, "infra1", ir1)
                    if self.enable_infra and "infra2" in self.pubs[dev.serial]:
                        f2 = frames.get_infrared_frame(2)
                        if f2:
                            ir2 = np.asanyarray(f2.get_data())
                            if self.publish_individual:
                                self._publish_jpeg(
                                    self.pubs[dev.serial]["infra2"],
                                    ir2,
                                    frame_id=f"{dev.cam_name}_infra2",
                                )
                            self._store_latest(dev.serial, "infra2", ir2)

            except Exception as e:
                # 디바이스 끊김/timeout/USB 이슈 등
                self._log_throttled(f"run_{dev.serial}", f"[{dev.serial}] frame loop 에러: {e}", "warn")
                self._stop_pipeline(dev)
                time.sleep(self.reconnect_wait)

        # shutdown
        self._stop_pipeline(dev)

    def destroy_node(self):
        self._stop = True
        for dev in self.devices:
            if dev.thread and dev.thread.is_alive():
                dev.thread.join(timeout=2.0)
            self._stop_pipeline(dev)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MultiRealSenseCompressedPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
