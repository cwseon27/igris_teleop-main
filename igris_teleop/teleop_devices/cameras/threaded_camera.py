import glob
import os
import subprocess
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


import logging_mp
logger_mp = logging_mp.get_logger(__name__, level=logging_mp.INFO)

from ...sharedmemory.shmManager import SharedMemoryManager
from ...sharedmemory.shm_schema import CAMERA

cv2.setNumThreads(1)

try:
    import pyrealsense2 as rs

    HAS_REALSENSE = True
except Exception:
    HAS_REALSENSE = False

# (field_name -> shape)
CAMERA_SHAPES = {name: shape for name, shape, _ in CAMERA}

# -----------------------------
# USB Stereo (SBS) Calibration (AS-IS from sample)
# -----------------------------
IMG_WIDTH = 1280
IMG_HEIGHT = 720

K_LEFT = np.array([
    [775.41759739,   0.0,         643.80314126],
    [0.0,            778.11204444, 352.48404218],
    [0.0,              0.0,         1.0]
], dtype=np.float32)

DIST_LEFT = np.array([[
    -0.32010433,
    -0.15772447,
    0.00699062,
    -0.00321917,
    0.17565373
]], dtype=np.float32)

K_RIGHT = np.array([
    [637.91669645,   0.0,          621.58351688],
    [0.0,            642.77528872, 369.66446855],
    [0.0,              0.0,          1.0]
], dtype=np.float32)

DIST_RIGHT = np.array([[
    -3.50821744e-01,
    9.53275302e-02,
    3.21910131e-04,
    9.66375599e-03,
    -8.05632456e-03
]], dtype=np.float32)

ALPHA = 0.4



def _parse_flip_env(var_name: str) -> Optional[int]:
    """
    Read an env var and return cv2 flip code (-1, 0, 1) or None if unset/invalid.
    """
    val = os.environ.get(var_name)
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    try:
        code = int(val)
    except ValueError:
        return None
    return code if code in (-1, 0, 1) else None


def _flag(obj) -> bool:
    """Best-effort check for Event-like or value-like stop flags."""
    if obj is None:
        return False
    is_set = getattr(obj, "is_set", None)
    if callable(is_set):
        try:
            return bool(is_set())
        except Exception:
            return False
    if hasattr(obj, "value"):
        try:
            return bool(obj.value)
        except Exception:
            return False
    return bool(obj)


def fourcc(code4: str) -> int:
    return cv2.VideoWriter_fourcc(*code4)


def v4l2_list_formats_ext(devnode: str) -> str:
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "-d", devnode, "--list-formats-ext"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2.0,
        )
        return out
    except Exception:
        return ""


def v4l2_has_formats(devnode: str) -> bool:
    """
    Check whether v4l2-ctl reports at least one format.
    Returns False for metadata/dummy nodes with no formats.
    """
    out = v4l2_list_formats_ext(devnode)
    return "[0]:" in out


def device_supports_sizes(devnode: str, *sizes: str) -> bool:
    """
    Verify that all given sizes (e.g., "2560x720") appear in --list-formats-ext output.
    """
    out = v4l2_list_formats_ext(devnode)
    if not out:
        return False
    return all(s in out for s in sizes)


def discover_usb_candidates():
    """
    Strategy:
    1) Prefer /dev/v4l/by-id (stable even when device numbers change)
    2) Fallback: parse "v4l2-ctl --list-devices" and keep USB Camera blocks
    3) Return only nodes that expose at least one format
    """
    candidates = []

    # (1) Prefer by-id paths
    by_id_paths = sorted(glob.glob("/dev/v4l/by-id/*"))
    for p in by_id_paths:
        low = p.lower()
        # Exclude RealSense (add keywords here if needed)
        if "realsense" in low or "intel" in low:
            continue
        candidates.append(p)

    # (2) Fallback: parse v4l2-ctl --list-devices
    if not candidates:
        try:
            out = subprocess.check_output(
                ["v4l2-ctl", "--list-devices"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=2.0,
            )
        except Exception:
            out = ""

        blocks = [b.strip() for b in out.split("\n\n") if b.strip()]
        for b in blocks:
            header = b.splitlines()[0].lower()
            if "usb" not in header:
                continue
            if "realsense" in header or "intel" in header:
                continue

            for line in b.splitlines()[1:]:
                line = line.strip()
                if line.startswith("/dev/video"):
                    candidates.append(line)

    # Drop duplicates while keeping order
    uniq = []
    seen = set()
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    # (3) Keep only nodes that expose at least one format
    usable = [c for c in uniq if v4l2_has_formats(c)]
    return usable


def looks_like_sbs(frame: np.ndarray) -> bool:
    """
    Heuristic check for side-by-side stereo frames.
    - even width
    - high aspect ratio compared to typical mono (e.g., 16:9)
    """
    if frame is None:
        return False
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return False
    if (w % 2) != 0:
        return False
    aspect = w / float(h)
    return aspect >= 2.2  # 2560x720=3.56, 1280x480=2.67 are typical SBS ratios


# class StereoSbsCalibrator:
#     """
#     Sample code logic as-is:
#       - assume each half is 1280x720
#       - newK/roi from getOptimalNewCameraMatrix(alpha=0.4)
#       - initUndistortRectifyMap + remap
#       - ROI crop (only if all roi components > 0)
#       - resize to (IMG_WIDTH, IMG_HEIGHT) == (1280,720)
#     """

#     def __init__(self, half_w: int, half_h: int):
#         self.half_w = int(half_w)
#         self.half_h = int(half_h)

#         if self.half_w != IMG_WIDTH or self.half_h != IMG_HEIGHT:
#             raise ValueError(
#                 f"Stereo calibration expects half size {IMG_WIDTH}x{IMG_HEIGHT}, "
#                 f"but got {self.half_w}x{self.half_h}"
#             )

#         self.newK_left, self.roi_left = cv2.getOptimalNewCameraMatrix(
#             K_LEFT, DIST_LEFT, (self.half_w, self.half_h), ALPHA, (self.half_w, self.half_h)
#         )
#         self.newK_right, self.roi_right = cv2.getOptimalNewCameraMatrix(
#             K_RIGHT, DIST_RIGHT, (self.half_w, self.half_h), ALPHA, (self.half_w, self.half_h)
#         )

#         self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
#             K_LEFT, DIST_LEFT, None, self.newK_left, (self.half_w, self.half_h), cv2.CV_16SC2
#         )
#         self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
#             K_RIGHT, DIST_RIGHT, None, self.newK_right, (self.half_w, self.half_h), cv2.CV_16SC2
#         )

#         logger_mp.info(
#             "[CameraWorker][Calib] stereo maps ready half=%dx%d alpha=%.3f",
#             self.half_w, self.half_h, ALPHA
#         )

#     @staticmethod
#     def _crop_with_roi(img: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
#         x, y, w, h = roi
#         # sample's condition 그대로: x,y,w,h 모두 > 0일 때만 crop
#         if all(v > 0 for v in [x, y, w, h]):
#             return img[y:y + h, x:x + w]
#         return img

#     def apply(self, left_raw: np.ndarray, right_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         left_undist = cv2.remap(left_raw, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
#         right_undist = cv2.remap(right_raw, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

#         left_crop = self._crop_with_roi(left_undist, self.roi_left)
#         right_crop = self._crop_with_roi(right_undist, self.roi_right)

#         left_out = cv2.resize(left_crop, (IMG_WIDTH, IMG_HEIGHT))
#         right_out = cv2.resize(right_crop, (IMG_WIDTH, IMG_HEIGHT))
#         return left_out, right_out


class StereoSbsCalibrator:
    def __init__(self, half_w: int = 1280, half_h: int = 720):
        npz_path = "igris_teleop/teleop_devices/cameras/stereo_rectify_maps_tuned.npz"
        d = np.load(npz_path)

        # 해상도 일치 확인
        w, h = map(int, d["img_size"])
        if (w, h) != (half_w, half_h):
            raise ValueError(f"NPZ img_size={w}x{h} != expected {half_w}x{half_h}")

        # OpenCV remap에 바로 넣을 수 있는 float32 맵 (x,y 분리형)
        self.map_left_x  = d["map1x"].astype(np.float32)
        self.map_left_y  = d["map1y"].astype(np.float32)
        self.map_right_x = d["map2x"].astype(np.float32)
        self.map_right_y = d["map2y"].astype(np.float32)

        # 필요 시 보관 (깊이/3D 재투영 등에 사용)
        self.Q  = d.get("Q", None)
        self.P1 = d.get("P1", None)
        self.P2 = d.get("P2", None)

    def apply(self, left_raw, right_raw):
        left_rect = cv2.remap(left_raw,  self.map_left_x,  self.map_left_y,  cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        return left_rect, right_rect


def _decode_fourcc(v: float) -> str:
    v = int(v)
    return "".join([chr((v >> 8*i) & 0xFF) for i in range(4)])


class ThreadedOpenCVCam:
    def __init__(self, dev_path: str, width=1280, height=720, fps=30, prefer="MJPG"):
        self.name = f"USB {dev_path}"
        self.dev_path = dev_path
        self.req_width, self.req_height, self.req_fps = width, height, fps
        self.prefer = prefer

        # 먼저 멤버를 안전하게 초기화 (중요)
        self._lock = threading.Lock()
        self._last_bgr = None
        self._running = False
        self._thread = None
        self.cap = None
        self.active_fmt = None

        cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open {dev_path}")
        self.cap = cap

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        fmt_order = [prefer, "YUYV" if prefer != "YUYV" else "MJPG"]
        for fmt in fmt_order:
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc(fmt))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

            # warm-up
            for _ in range(10):
                self.cap.read()
            ok, frame = self.cap.read()
            if ok and frame is not None and frame.size > 0:
                self.active_fmt = fmt
                with self._lock:
                    self._last_bgr = frame
                break

        if self.active_fmt is None:
            self.cap.release()
            self.cap = None
            raise RuntimeError(f"{dev_path}: cannot get frames with {fmt_order}")

        # 모든 준비가 끝난 뒤에 스레드 시작
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afps = self.cap.get(cv2.CAP_PROP_FPS)

        logger_mp.info(
            "[USB] opened %s fmt=%s req=%dx%d@%s act=%dx%d@%.3f",
            dev_path, self.active_fmt, width, height, fps, aw, ah, afps
        )

    def _loop(self):
        while self._running:
            ok, frame_bgr = self.cap.read()
            if not ok or frame_bgr is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._last_bgr = frame_bgr

    def read_bgr(self):
        with self._lock:
            return None if self._last_bgr is None else self._last_bgr.copy()

    def close(self):
        self._running = False
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass



class ThreadedRealSense:
    def __init__(self, serial: str, width=1280, height=720, fps=30):
        self.name = f"RealSense {serial}"
        self.serial = serial

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(cfg)

        self._lock = threading.Lock()
        self._last_bgr = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        logger_mp.info("[RS] opened %s %dx%d@%d", serial, width, height, fps)

    def _loop(self):
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=200)
            except Exception:
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            bgr = np.asanyarray(color.get_data())
            with self._lock:
                self._last_bgr = bgr

    def read_bgr(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._last_bgr is None else self._last_bgr.copy()

    def close(self):
        self._running = False
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.pipeline.stop()
        except Exception:
            pass


def _wait_for_frame(reader, timeout: float = 2.0) -> Optional[np.ndarray]:
    end = time.time() + timeout
    while time.time() < end:
        frame = reader()
        if frame is not None and getattr(frame, "size", 0) > 0:
            return frame
        time.sleep(0.01)
    return None


def _split_sbs(frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if frame is None:
        return None, None
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0 or (w % 2) != 0:
        return None, None
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]


def _prepare_frame(frame: np.ndarray, shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """Resize/cast frame to target shape (H, W, C)."""
    if frame is None:
        return None
    try:
        target_h, target_w, _ = shape
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            frame = cv2.resize(frame, (target_w, target_h))
        if frame.dtype != np.uint8:
            frame = np.asarray(frame, dtype=np.uint8)
        return frame
    except Exception:
        return None


def _apply_flip(frame: Optional[np.ndarray], flip_code: Optional[int]) -> Optional[np.ndarray]:
    if frame is None or flip_code is None:
        return frame
    try:
        return cv2.flip(frame, flip_code)
    except Exception:
        return frame
