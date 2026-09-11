from pathlib import Path
import sys

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mediapipe_hand_pose_bridge.camera_capture import (  # noqa: E402
    RecoveringCameraCapture,
    normalize_camera_device,
)


STABLE_DEVICE = '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:5.3.1:1.0-video-index0'


class FakeCapture:
    def __init__(self, *, opened=True, reads=()):
        self.opened = opened
        self.reads = list(reads)
        self.released = False
        self.properties = {}

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.properties[prop] = value
        return True

    def read(self):
        result = self.reads.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def release(self):
        self.released = True


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, message):
        self.warnings.append(message)

    def info(self, message):
        self.infos.append(message)


@pytest.mark.parametrize('value, expected', [(0, 0), (3, 3), ('3', 3), (' 2 ', 2), (STABLE_DEVICE, STABLE_DEVICE)])
def test_device_accepts_legacy_indices_and_preserves_stable_path(value, expected):
    assert normalize_camera_device(value) == expected


@pytest.mark.parametrize('value', [True, -1, '-1', '', '/dev/', 'camera.mp4', 'http://camera', 1.5])
def test_device_rejects_ambiguous_or_non_camera_sources(value):
    with pytest.raises(ValueError):
        normalize_camera_device(value)


def test_usb_disconnect_reopens_stable_alias_and_never_changes_camera():
    now = [0.0]
    frame_before, frame_after = object(), object()
    old_device = FakeCapture(reads=[(True, frame_before), (False, None)])
    missing_device = FakeCapture(opened=False)
    # The udev alias now points to a newly enumerated /dev/videoN.
    new_device = FakeCapture(reads=[(True, frame_after)])
    captures = iter([old_device, missing_device, new_device])
    calls = []
    logger = FakeLogger()

    def factory(device, backend):
        calls.append((device, backend))
        return next(captures)

    camera = RecoveringCameraCapture(
        STABLE_DEVICE, 640, 480, capture_factory=factory, clock=lambda: now[0], logger=logger,
    )
    assert camera.read() == (True, frame_before)
    assert camera.read() == (False, None)
    assert old_device.released
    now[0] = 0.9
    assert camera.read() == (False, None)
    assert len(calls) == 1
    now[0] = 1.0
    assert camera.read() == (False, None)
    assert missing_device.released
    now[0] = 1.9
    assert camera.read() == (False, None)
    assert len(calls) == 2
    now[0] = 2.0
    assert camera.read() == (True, frame_after)
    assert calls == [(STABLE_DEVICE, cv2.CAP_V4L2)] * 3
    assert len(logger.warnings) == 1
    assert len(logger.infos) == 2
    for cap in (old_device, new_device):
        assert cap.properties[cv2.CAP_PROP_FRAME_WIDTH] == 640
        assert cap.properties[cv2.CAP_PROP_FRAME_HEIGHT] == 480
        assert cap.properties[cv2.CAP_PROP_BUFFERSIZE] == 1
    camera.release()
    assert new_device.released


def test_missing_camera_at_startup_retries_with_throttled_warnings():
    now = [0.0]
    logger = FakeLogger()
    calls = []

    def factory(device, backend):
        cap = FakeCapture(opened=False)
        calls.append(cap)
        return cap

    camera = RecoveringCameraCapture(
        STABLE_DEVICE, 640, 480, capture_factory=factory, clock=lambda: now[0], logger=logger,
    )
    for tick in range(51):
        now[0] = tick / 10
        assert camera.read() == (False, None)
    assert len(calls) == 6
    assert all(cap.released for cap in calls)
    assert len(logger.warnings) == 2
    assert not logger.infos


def test_camera_absent_at_startup_returns_without_restarting_node():
    now = [0.0]
    frame = object()
    missing = FakeCapture(opened=False)
    present = FakeCapture(reads=[(True, frame)])
    captures = iter([missing, present])
    calls = []

    def factory(device, backend):
        calls.append((device, backend))
        return next(captures)

    camera = RecoveringCameraCapture(
        STABLE_DEVICE, 640, 480, capture_factory=factory, clock=lambda: now[0],
    )
    assert camera.read() == (False, None)
    assert missing.released
    now[0] = 1.0
    assert camera.read() == (True, frame)
    assert calls == [(STABLE_DEVICE, cv2.CAP_V4L2)] * 2


def test_repeated_read_failures_release_each_handle_until_frames_return():
    now = [0.0]
    frame = object()
    failing = [FakeCapture(reads=[(False, None)]) for _ in range(3)]
    captures = iter([*failing, FakeCapture(reads=[(True, frame)])])
    calls = []

    def factory(device, backend):
        calls.append((device, backend))
        return next(captures)

    camera = RecoveringCameraCapture(
        STABLE_DEVICE, 640, 480, capture_factory=factory, clock=lambda: now[0],
    )
    for tick, cap in enumerate(failing):
        now[0] = float(tick)
        assert camera.read() == (False, None)
        assert cap.released
    now[0] = 3.0
    assert camera.read() == (True, frame)
    assert calls == [(STABLE_DEVICE, cv2.CAP_V4L2)] * 4


@pytest.mark.parametrize('failure', [OSError('camera disconnected'), cv2.error('camera disconnected')])
def test_read_exception_releases_handle_and_recovers(failure):
    now = [0.0]
    broken = FakeCapture(reads=[failure])
    frame = object()
    recovered = FakeCapture(reads=[(True, frame)])
    captures = iter([broken, recovered])
    camera = RecoveringCameraCapture(
        '2', 640, 480, capture_factory=lambda *_: next(captures), clock=lambda: now[0],
    )
    assert camera.read() == (False, None)
    assert broken.released
    now[0] = 1.0
    assert camera.read() == (True, frame)


def test_open_exception_retries_same_stable_device():
    now = [0.0]
    frame = object()
    outcomes = iter([OSError('not present'), FakeCapture(reads=[(True, frame)])])
    calls = []

    def factory(device, backend):
        calls.append(device)
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    camera = RecoveringCameraCapture(
        STABLE_DEVICE, 640, 480, capture_factory=factory, clock=lambda: now[0],
    )
    assert camera.read() == (False, None)
    now[0] = 1.0
    assert camera.read() == (True, frame)
    assert calls == [STABLE_DEVICE, STABLE_DEVICE]
