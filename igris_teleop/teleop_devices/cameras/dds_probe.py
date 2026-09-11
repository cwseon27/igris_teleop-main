from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from dataclasses import dataclass, field

try:
    import igris_c_sdk as igc_sdk
except ImportError:
    igc_sdk = None


DEFAULT_TOPICS = [
    "igris_c/sensor/d435_color",
    "igris_c/sensor/d435_depth",
    "igris_c/sensor/eyes_stereo",
    "igris_c/sensor/left_hand",
    "igris_c/sensor/right_hand",
]


def _format_rate(bytes_per_sec: float) -> str:
    if bytes_per_sec >= 1024.0 * 1024.0:
        return f"{bytes_per_sec / (1024.0 * 1024.0):.1f} MB/s"
    if bytes_per_sec >= 1024.0:
        return f"{bytes_per_sec / 1024.0:.1f} KB/s"
    return f"{bytes_per_sec:.0f} B/s"


def _format_bytes(num_bytes: float) -> str:
    if num_bytes >= 1024.0 * 1024.0:
        return f"{num_bytes / (1024.0 * 1024.0):.1f} MB"
    if num_bytes >= 1024.0:
        return f"{num_bytes / 1024.0:.1f} KB"
    return f"{num_bytes:.0f} B"


def _format_age(age_s: float | None) -> str:
    if age_s is None:
        return "never"
    if age_s < 1.0:
        return f"{age_s * 1000.0:.0f} ms ago"
    return f"{age_s:.1f} s ago"


@dataclass
class StreamState:
    name: str
    fps: float = 0.0
    bytes_per_sec: float = 0.0
    last_time: float = 0.0
    last_size: int = 0
    total_messages: int = 0
    total_bytes: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, payload: bytes) -> None:
        now = time.perf_counter()
        payload_size = len(payload)
        with self.lock:
            if self.last_time > 0.0:
                elapsed = now - self.last_time
                if elapsed > 0.0:
                    inst_fps = 1.0 / elapsed
                    inst_bps = payload_size / elapsed
                    alpha = 0.15
                    if self.fps <= 0.0:
                        self.fps = inst_fps
                        self.bytes_per_sec = inst_bps
                    else:
                        self.fps = (1.0 - alpha) * self.fps + alpha * inst_fps
                        self.bytes_per_sec = (1.0 - alpha) * self.bytes_per_sec + alpha * inst_bps
            self.last_time = now
            self.last_size = payload_size
            self.total_messages += 1
            self.total_bytes += payload_size

    def snapshot(self, now: float) -> tuple[float, float, int, int, int, float | None]:
        with self.lock:
            age_s = None if self.last_time <= 0.0 else max(0.0, now - self.last_time)
            return (
                self.fps,
                self.bytes_per_sec,
                self.last_size,
                self.total_messages,
                self.total_bytes,
                age_s,
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe CycloneDDS compressed camera topics.")
    parser.add_argument(
        "--domain-id",
        type=int,
        default=10,
        help="Cyclone DDS domain id (default: 10)",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=0.5,
        help="status print interval in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="optional probe duration in seconds; 0 means run until interrupted",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="do not clear the terminal between status updates",
    )
    parser.add_argument(
        "topics",
        nargs="*",
        help="override default compressed DDS topics to subscribe to",
    )
    return parser


def _print_status(streams: list[StreamState], *, clear_screen: bool) -> None:
    now = time.perf_counter()
    if clear_screen:
        sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.write("=== CycloneDDS Camera Probe ===\n")
    sys.stdout.write("Press Ctrl-C to quit.\n\n")

    total_bps = 0.0
    total_msgs = 0
    seen_topics = 0
    for stream in streams:
        fps, bps, last_size, total_messages, total_bytes, age_s = stream.snapshot(now)
        total_bps += bps
        total_msgs += total_messages
        if age_s is not None:
            seen_topics += 1
        status = "OK" if age_s is not None and age_s < 2.0 else "STALE"
        sys.stdout.write(
            f"{stream.name}\n"
            f"  status={status} fps={fps:.1f} rate={_format_rate(bps)} "
            f"last={last_size} B seen={_format_age(age_s)} total={total_messages} msgs / {_format_bytes(float(total_bytes))}\n"
        )

    sys.stdout.write("\n")
    sys.stdout.write(
        f"topics_seen={seen_topics}/{len(streams)} total_rate={_format_rate(total_bps)} total_messages={total_msgs}\n"
    )
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    if igc_sdk is None:
        raise SystemExit("igris_c_sdk is not available.")

    args = _build_arg_parser().parse_args(argv)
    topics = args.topics or DEFAULT_TOPICS
    status_interval = max(0.1, float(args.status_interval))
    duration_s = max(0.0, float(args.duration))
    clear_screen = not bool(args.no_clear)

    stop_event = threading.Event()
    streams = [StreamState(name=topic) for topic in topics]
    callbacks: list[object] = []
    subscribers: list[object] = []

    def _handle_signal(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    channel_factory = igc_sdk.ChannelFactory.Instance()
    if channel_factory.IsInitialized():
        current_domain = int(channel_factory.GetDomainId())
        if current_domain != int(args.domain_id):
            sys.stdout.write(
                f"[camera_dds_probe] ChannelFactory already initialized on domain {current_domain}; "
                f"requested domain {int(args.domain_id)}.\n"
            )
            sys.stdout.flush()
    channel_factory.Init(int(args.domain_id))

    for stream in streams:
        subscriber = igc_sdk.CompressedMessageSubscriber(stream.name)
        callback = lambda msg, stream=stream: stream.update(bytes(msg.image_data()))
        ok = subscriber.init(callback)
        if ok is False:
            raise RuntimeError(f"Failed to init CompressedMessageSubscriber({stream.name})")
        callbacks.append(callback)
        subscribers.append(subscriber)

    start_ts = time.perf_counter()
    last_status_ts = 0.0

    try:
        while not stop_event.is_set():
            now = time.perf_counter()
            if now - last_status_ts >= status_interval:
                _print_status(streams, clear_screen=clear_screen)
                last_status_ts = now
            if duration_s > 0.0 and (now - start_ts) >= duration_s:
                break
            time.sleep(0.05)
    finally:
        stop_event.set()
        for subscriber in subscribers:
            try:
                subscriber.stop()
            except Exception:
                pass
        try:
            channel_factory.Release()
        except Exception:
            pass
        callbacks.clear()

    _print_status(streams, clear_screen=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
