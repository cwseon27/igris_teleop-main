import math
import time
import threading
import subprocess
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

cv2.setNumThreads(1)

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except Exception:
    HAS_REALSENSE = False


def fourcc(code4: str) -> int:
    return cv2.VideoWriter_fourcc(*code4)


def v4l2_list_formats_ext(devnode: str) -> str:
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "-d", devnode, "--list-formats-ext"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2.0
        )
        return out
    except Exception:
        return ""


def v4l2_has_formats(devnode: str) -> bool:
    """
    v4l2-ctl로 포맷이 1개 이상 존재하는지 확인.
    video 노드(메타/더미)처럼 포맷이 없으면 False.
    """
    out = v4l2_list_formats_ext(devnode)
    return "[0]:" in out


def device_supports_sizes(devnode: str, *sizes: str) -> bool:
    """
    --list-formats-ext 출력에 sizes(예: "2560x720")가 모두 포함되는지 확인.
    """
    out = v4l2_list_formats_ext(devnode)
    if not out:
        return False
    return all(s in out for s in sizes)


def discover_usb_candidates():
    """
    방법 B:
    1) /dev/v4l/by-id 우선(번호 바뀌어도 안정적)
    2) 없으면 v4l2-ctl --list-devices에서 USB Camera 블록만 파싱
    3) 후보 중 '포맷 존재' 노드만 반환
    """
    candidates = []

    # (1) by-id 우선
    by_id_paths = sorted(glob.glob("/dev/v4l/by-id/*"))
    for p in by_id_paths:
        low = p.lower()
        # RealSense 등 제외(필요하면 키워드 추가)
        if "realsense" in low or "intel" in low:
            continue
        candidates.append(p)

    # (2) fallback: v4l2-ctl --list-devices 파싱
    if not candidates:
        try:
            out = subprocess.check_output(
                ["v4l2-ctl", "--list-devices"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=2.0
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

    # 중복 제거(표시 순서 유지)
    uniq = []
    seen = set()
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    # (3) 포맷 존재 노드만 남김
    usable = [c for c in uniq if v4l2_has_formats(c)]
    return usable


class ThreadedOpenCVCam:
    def __init__(self, dev_path: str, width=1280, height=720, fps=30, prefer="MJPG"):
        self.name = f"USB {dev_path}"
        self.dev_path = dev_path
        self.req_width, self.req_height, self.req_fps = width, height, fps
        self.prefer = prefer

        self.cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {dev_path}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        fmt_order = [prefer, "YUYV" if prefer != "YUYV" else "MJPG"]
        self.active_fmt = None

        for fmt in fmt_order:
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc(fmt))
            # 워밍업
            for _ in range(10):
                self.cap.read()
            ok, frame = self.cap.read()
            if ok and frame is not None and frame.size > 0:
                self.active_fmt = fmt
                break

        if self.active_fmt is None:
            self.cap.release()
            raise RuntimeError(f"{dev_path}: cannot get frames with {fmt_order}")

        self._lock = threading.Lock()
        self._last_rgb = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        # 실제 적용된 파라미터 출력(중요)
        real = os.path.realpath(dev_path)
        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afps = self.cap.get(cv2.CAP_PROP_FPS)
        print(
            f"[USB] opened {dev_path} -> {real} fmt={self.active_fmt} "
            f"req={width}x{height}@{fps} act={aw}x{ah}@{afps:.3f}"
        )

    def _loop(self):
        while self._running:
            ok, frame_bgr = self.cap.read()
            if not ok or frame_bgr is None:
                time.sleep(0.005)
                continue
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._last_rgb = rgb

    def read_rgb(self):
        with self._lock:
            return None if self._last_rgb is None else self._last_rgb.copy()

    def close(self):
        self._running = False
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass


class ThreadedRealSense:
    def __init__(self, serial: str, width=640, height=480, fps=30):
        self.name = f"RealSense {serial}"
        self.serial = serial

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(cfg)

        self._lock = threading.Lock()
        self._last_rgb = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        print(f"[RS] opened {serial} {width}x{height}@{fps}")

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
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._last_rgb = rgb

    def read_rgb(self):
        with self._lock:
            return None if self._last_rgb is None else self._last_rgb.copy()

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


class StereoView:
    """
    하나의 base 소스 프레임(예: 2560x720)을 좌/우로 잘라서 별도 소스처럼 제공.
    base는 ThreadedOpenCVCam(실제 디바이스 핸들)이고, StereoView는 display 전용.
    """
    def __init__(self, base, side: str):
        assert side in ("L", "R")
        self.base = base
        self.side = side
        self.name = f"{base.name} [{side}]"

    def read_rgb(self):
        rgb = self.base.read_rgb()
        if rgb is None:
            return None
        h, w, _ = rgb.shape
        if w < 2 or (w % 2) != 0:
            return rgb
        mid = w // 2
        return rgb[:, :mid] if self.side == "L" else rgb[:, mid:]

    def close(self):
        # base는 별도로 한 번만 close해야 하므로 여기서는 no-op
        pass


def make_grid(n):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def looks_like_sbs(rgb: np.ndarray) -> bool:
    """
    Side-by-Side(좌/우 붙은 프레임) 추정.
    - 가로가 짝수
    - 가로/세로 비가 충분히 큼(일반 mono 16:9(1.77)보다 훨씬 큰 값)
    """
    if rgb is None:
        return False
    h, w = rgb.shape[:2]
    if w <= 0 or h <= 0:
        return False
    if (w % 2) != 0:
        return False
    aspect = w / float(h)
    return aspect >= 2.2  # 2560x720=3.56, 1280x480=2.67 등 SBS에서 보통 만족


def main():
    display_sources = []   # 실제로 그리드에 띄울 소스(스테레오면 2개)
    close_targets = []     # 실제 디바이스를 여는 객체만(중복 close 방지)

    # 1) RealSense (SDK)
    if HAS_REALSENSE:
        ctx = rs.context()
        for dev in ctx.query_devices():
            try:
                serial = dev.get_info(rs.camera_info.serial_number)
                rs_cam = ThreadedRealSense(serial)
                display_sources.append(rs_cam)
                close_targets.append(rs_cam)
            except Exception as e:
                print("[RS] skip:", e)

    # 2) USB Camera (UVC): 후보 노드 자동 선택
    usb_nodes = discover_usb_candidates()
    print("[USB] candidates with formats:", usb_nodes)

    for dev_path in usb_nodes:
        try:
            # 스테레오(SBS) 가능성이 높으면 2560x720 우선, 아니면 기존처럼 1280x720
            prefer_sbs = device_supports_sizes(dev_path, "2560x720", "1280x720")
            if prefer_sbs:
                cam = ThreadedOpenCVCam(dev_path, width=2560, height=720, fps=30, prefer="MJPG")
            else:
                cam = ThreadedOpenCVCam(dev_path, width=1280, height=720, fps=30, prefer="MJPG")

            close_targets.append(cam)

            # 첫 프레임 대기 후 SBS면 split해서 2개로 등록
            rgb = None
            t0 = time.time()
            while rgb is None and (time.time() - t0) < 2.0:
                rgb = cam.read_rgb()
                time.sleep(0.01)

            if rgb is not None and prefer_sbs and looks_like_sbs(rgb):
                display_sources.append(StereoView(cam, "L"))
                display_sources.append(StereoView(cam, "R"))
                print(f"[USB] SBS stereo detected: frame={rgb.shape} -> split L/R")
            else:
                display_sources.append(cam)
                if rgb is not None:
                    print(f"[USB] mono/unknown: frame={rgb.shape}")

        except Exception as e:
            print(f"[USB] open failed {dev_path}:", e)

    if not display_sources:
        print("열 수 있는 카메라가 없습니다.")
        return

    n = len(display_sources)
    rows, cols = make_grid(n)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax in axes[n:]:
        ax.axis("off")

    ims = []
    for i, src in enumerate(display_sources):
        ax = axes[i]
        ax.set_title(src.name)
        ax.axis("off")

        rgb = None
        t0 = time.time()
        while rgb is None and (time.time() - t0) < 2.0:
            rgb = src.read_rgb()
            time.sleep(0.01)

        if rgb is None:
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        ims.append(ax.imshow(rgb))

    plt.tight_layout()
    plt.ion()

    try:
        while plt.fignum_exists(fig.number):
            for i, src in enumerate(display_sources):
                rgb = src.read_rgb()
                if rgb is not None:
                    ims[i].set_data(rgb)
            fig.canvas.draw_idle()
            plt.pause(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        for dev in close_targets:
            dev.close()
        plt.close(fig)


if __name__ == "__main__":
    main()
