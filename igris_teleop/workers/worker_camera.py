from __future__ import annotations

import logging_mp
import numpy as np
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import cv2

from ..core.events import EventSnapshot
from ..core.state_machine import TransitionResult
from ..core.worker_base import SingleRateWorker, WorkerContext
from ..teleop_devices.cameras.threaded_camera import (
    StereoSbsCalibrator,
    ThreadedRealSense,
    ThreadedOpenCVCam,
    _parse_flip_env,
    discover_usb_candidates,
    device_supports_sizes,
    looks_like_sbs,
    _wait_for_frame,
    _split_sbs,
    _apply_flip,
    _prepare_frame,
    HAS_REALSENSE,
    CAMERA_SHAPES,
)

logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)

import pyrealsense2 as rs


@dataclass
class CameraSourceInfo:
    source: Any
    stereo: bool
    flip_code: Optional[int]
    stereo_calib: Optional[StereoSbsCalibrator]
    desc: str


class BaseCameraWorker(SingleRateWorker, ABC):
    """모든 카메라 워커에서 공유하는 단일 루프 로직."""

    def __init__(self, ctx: WorkerContext, hz: float = 30.0) -> None:
        super().__init__(ctx, hz=hz)
        self._shared_memory = ctx.shared_memory
        self._owns_shared_memory = False

        self.camera_shm = self._shared_memory.get("camera_shm")

        self._source = None
        self._stereo = False
        self._stereo_calib: Optional[StereoSbsCalibrator] = None
        self._flip_code: Optional[int] = None
        self._source_name = "uninitialized"
        self._usb_flip_override = _parse_flip_env("USB_CAMERA_FLIP")
        self._rs_flip_override = _parse_flip_env("RS_CAMERA_FLIP")
        self._last_init_try = 0.0

        self._init_camera()

    def on_start(self) -> None:
        logger.info(f"[{self.ctx.name}] start (single-rate {self.hz} Hz) source={self._source_name}")

    def _init_camera(self) -> None:
        self._close_source()
        self._source = None
        self._stereo = False
        self._stereo_calib = None
        self._flip_code = None
        self._source_name = "none"

        info = self._open_source()
        if info is None:
            logger.warning(f"[{self.ctx.name}] no available camera found.")
            return

        self._source = info.source
        self._stereo = info.stereo
        self._flip_code = info.flip_code
        self._stereo_calib = info.stereo_calib
        self._source_name = info.desc

        logger.info(
            "[CameraWorker] using %s stereo=%s source=%s", info.desc, info.stereo, getattr(info.source, "name", None)
        )
        if self._flip_code is not None:
            logger.info(f"[CameraWorker] applying flip_code={self._flip_code} for {info.desc} source")

    def _close_source(self) -> None:
        src = self._source
        self._source = None
        if src is not None:
            try:
                src.close()
            except Exception:
                logger.exception("[CameraWorker] failed to close camera source.")

    def _build_payload(self, frame: np.ndarray) -> dict:
        payload = {}

        if self._stereo:
            left_raw, right_raw = _split_sbs(frame)
            if left_raw is None or right_raw is None:
                return payload

            if self._stereo_calib is not None:
                try:
                    left, right = self._stereo_calib.apply(left_raw, right_raw)
                except Exception:
                    logger.exception("[CameraWorker][Calib] apply failed. fallback to raw split")
                    left, right = left_raw, right_raw
            else:
                left, right = left_raw, right_raw

            left = _apply_flip(left, self._flip_code)
            right = _apply_flip(right, self._flip_code)

            left_shape = CAMERA_SHAPES.get("stereo_left")
            right_shape = CAMERA_SHAPES.get("stereo_right")
            color_shape = CAMERA_SHAPES.get("realsense_head")
            if left is not None and left_shape is not None:
                prepared_left = _prepare_frame(left, left_shape)
                if prepared_left is not None:
                    payload["stereo_left"] = prepared_left
            if right is not None and right_shape is not None:
                prepared_right = _prepare_frame(right, right_shape)
                if prepared_right is not None:
                    payload["stereo_right"] = prepared_right
        else:
            frame = _apply_flip(frame, self._flip_code)
            frame_shape = CAMERA_SHAPES.get("realsense_head")
            if self._should_export_color() and frame_shape is not None:
                color_prepared = _prepare_frame(frame, frame_shape)
                if color_prepared is not None:
                    payload["realsense_head"] = color_prepared

        return payload

    def _should_export_color(self) -> bool:
        return False

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        if self._source is None:
            now = time.time()
            if now - self._last_init_try > 2.0:
                self._last_init_try = now
                self._init_camera()
            time.sleep(0.05)
            return

        frame = self._source.read_bgr()
        if frame is None:
            return

        # 내부 처리와 공유 메모리에 RGB 순서가 필요하므로 BGR->RGB 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        payload = self._build_payload(frame)
        if payload:
            try:
                self.camera_shm.write_data(**payload)
            except Exception:
                logger.exception("[CameraWorker] failed to write camera frame to SHM.")

    def on_stop(self) -> None:
        if self._shared_memory:
            for key, mgr in self._shared_memory.items():
                try:
                    mgr.worker_close()
                except Exception:
                    logger.exception(f"[{self.ctx.name}] failed to close shared memory {key}")

        logger.info(f"[{self.ctx.name}] stop")

    @abstractmethod
    def _open_source(self) -> Optional[CameraSourceInfo]:
        ...


class StereoCameraWorker(BaseCameraWorker):
    """USB/Stereo 카메라용 워커."""

    def _open_source(self) -> Optional[CameraSourceInfo]:
        usb_nodes = discover_usb_candidates()
        logger.info("[StereoCameraWorker] USB candidates with formats: %s", usb_nodes)

        for dev_path in usb_nodes:
            try:
                prefer_sbs = device_supports_sizes(dev_path, "2560x720", "1280x720")
                width, height = (2560, 720) if prefer_sbs else (1280, 720)
                cam = ThreadedOpenCVCam(dev_path, width=width, height=height, fps=30, prefer="MJPG")
                frame = _wait_for_frame(cam.read_bgr, timeout=2.0)
                if frame is None:
                    cam.close()
                    continue
                is_sbs = prefer_sbs and looks_like_sbs(frame)
                flip_code = self._usb_flip_override if self._usb_flip_override is not None else -1

                stereo_calib: Optional[StereoSbsCalibrator] = None
                if is_sbs:
                    h_full, w_full = frame.shape[:2]
                    mid = w_full // 2
                    half_w, half_h = mid, h_full
                    try:
                        stereo_calib = StereoSbsCalibrator(half_w, half_h)
                    except Exception:
                        stereo_calib = None
                        logger.exception(
                            "[StereoCameraWorker][Calib] init failed (half=%dx%d). fallback to raw split",
                            half_w,
                            half_h,
                        )

                return CameraSourceInfo(
                    source=cam,
                    stereo=is_sbs,
                    flip_code=flip_code,
                    stereo_calib=stereo_calib,
                    desc=f"USB {dev_path}",
                )
            except Exception:
                logger.exception("[StereoCameraWorker] failed to open %s", dev_path)
                continue

        return None


class RealSenseCameraWorker(BaseCameraWorker):
    """RealSense 카메라용 워커."""

    def _open_source(self) -> Optional[CameraSourceInfo]:
        if not HAS_REALSENSE:
            return None

        ctx = rs.context()
        for dev in ctx.query_devices():
            try:
                serial = dev.get_info(rs.camera_info.serial_number)
                cam = ThreadedRealSense(serial, width=640, height=480, fps=30)
                frame = _wait_for_frame(cam.read_bgr, timeout=2.0)
                if frame is None:
                    cam.close()
                    continue

                return CameraSourceInfo(
                    source=cam,
                    stereo=False,
                    flip_code=self._rs_flip_override,
                    stereo_calib=None,
                    desc=f"RealSense {serial}",
                )
            except Exception:
                logger.exception("[RealSenseCameraWorker] failed to open RealSense")
                continue

        return None

    def _should_export_color(self) -> bool:
        return True


CameraWorker = StereoCameraWorker
