"""V4L2 capture that survives USB reconnects without changing camera identity."""

import time
from typing import Union

import cv2


def normalize_camera_device(value) -> Union[int, str]:
    """Accept old camera indices and stable udev paths without resolving symlinks."""
    if isinstance(value, bool):
        raise ValueError('camera device must be a nonnegative index or /dev/ path')
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.isdecimal():
            return int(value)
        if value.startswith('/dev/') and len(value) > len('/dev/'):
            # Preserve by-path/udev aliases: their target may change on reconnect.
            return value
    raise ValueError('camera device must be a nonnegative index or /dev/ path')


class RecoveringCameraCapture:
    """Retry the exact configured device after open/read failures.

    A missing camera is not fatal to the ROS node. Returning no frame lets the
    caller publish is_tracked=False while this class waits for the same USB port
    to return. No other index, camera, or backend is selected as a fallback.
    """

    def __init__(
        self,
        device,
        width,
        height,
        *,
        retry_interval_s=1.0,
        warning_interval_s=5.0,
        capture_factory=None,
        clock=time.monotonic,
        logger=None,
    ):
        self.device = normalize_camera_device(device)
        self.width = int(width)
        self.height = int(height)
        self.retry_interval_s = max(0.1, float(retry_interval_s))
        self.warning_interval_s = max(self.retry_interval_s, float(warning_interval_s))
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._clock = clock
        self._logger = logger
        self._cap = None
        self._next_retry = 0.0
        self._last_warning = float('-inf')
        self._connected = False

    def _failed(self, reason):
        self.release()
        now = self._clock()
        self._next_retry = now + self.retry_interval_s
        if self._logger is not None and now - self._last_warning >= self.warning_interval_s:
            self._logger.warning(
                f'Camera {self.device} unavailable ({reason}); tracking invalid; '
                f'retrying the same device every {self.retry_interval_s:.1f}s'
            )
            self._last_warning = now

    def _open(self):
        self._cap = self._capture_factory(self.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._failed('open failed')
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def read(self):
        if self._cap is None and self._clock() < self._next_retry:
            return False, None
        try:
            if self._cap is None and not self._open():
                return False, None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._failed('read failed')
                return False, None
        except (cv2.error, OSError) as exc:
            self._failed(str(exc))
            return False, None
        if not self._connected:
            self._connected = True
            if self._logger is not None:
                self._logger.info(f'Camera {self.device} connected; receiving frames')
        return True, frame

    def release(self):
        cap, self._cap = self._cap, None
        self._connected = False
        if cap is not None:
            try:
                cap.release()
            except (cv2.error, OSError):
                pass
