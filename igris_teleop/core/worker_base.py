from __future__ import annotations

import multiprocessing as mp
import threading
import time
import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional, TYPE_CHECKING

from .events import EventBus, EventSnapshot
from .rate import Rate
from .state_machine import ModeState, TransitionResult, step

if TYPE_CHECKING:
    from igris_teleop.sharedmemory.shmManager import SharedMemoryManager

CAMERA_MODE_SUB = "camera_sub"
CAMERA_MODE_PUB = "camera_pub"
CAMERA_MODE_CHOICES: tuple[str, str] = (CAMERA_MODE_SUB, CAMERA_MODE_PUB)
DEFAULT_CAMERA_MODE = CAMERA_MODE_PUB
DEFAULT_CAMERA_DOMAIN_ID = 1
DEFAULT_ROBOT_DDS_DOMAIN_ID = 0
SIM_ROBOT_DDS_DOMAIN_ID = 99


@dataclass(frozen=True)
class RunConfig:
    mode: Optional[str]         # "teleop" | "walking" | "inference" | "replay" | None
    teleop_device: Optional[str]  # "unity" | "vr_masterarm" | "masterarm" | None
    teleop_hand_source: Optional[str] = None  # "vr" | "masterarm" | None
    runtime_environment: Optional[str] = None  # "sim" | "real" | None
    walking_policy_profile: Optional[str] = None
    walking_policy_path: Optional[str] = None
    walking_startup_blend_enabled: Optional[bool] = None
    inference_dataset_folder: Optional[str] = None
    inference_pretrained_rel: Optional[str] = None
    inference_policy: Optional[str] = None
    inference_chunk_size: Optional[int] = None
    inference_horizon: Optional[int] = None
    inference_n_action_step: Optional[int] = None
    inference_use_dataset_state: Optional[bool] = None
    inference_use_dataset_tau: Optional[bool] = None
    inference_use_dataset_camera: Optional[bool] = None
    inference_instruction: Optional[str] = None
    replay_dataset_folder: Optional[str] = None
    collect_dataset_repo_id: Optional[str] = None
    camera_mode: Optional[str] = DEFAULT_CAMERA_MODE
    camera_domain_id: Optional[int] = DEFAULT_CAMERA_DOMAIN_ID

    def __post_init__(self) -> None:
        camera_mode = DEFAULT_CAMERA_MODE if self.camera_mode is None else str(self.camera_mode)
        if camera_mode not in CAMERA_MODE_CHOICES:
            raise ValueError(
                f"camera_mode={camera_mode!r} is not supported. "
                f"Allowed: {list(CAMERA_MODE_CHOICES)}"
            )

        camera_domain_id = DEFAULT_CAMERA_DOMAIN_ID if self.camera_domain_id is None else int(self.camera_domain_id)
        object.__setattr__(self, "camera_mode", camera_mode)
        object.__setattr__(self, "camera_domain_id", camera_domain_id)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray | None) -> "RunConfig":
        if not payload:
            return cls(mode=None, teleop_device=None)
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("RunConfig JSON must decode to an object")
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in raw.items() if key in allowed}
        return cls(**filtered)


def resolve_robot_dds_domain_id(runtime_environment: Optional[str]) -> int:
    value = str(runtime_environment or "").strip().lower()
    if value in {"sim", "simulator", "simulation", "mujoco"}:
        return SIM_ROBOT_DDS_DOMAIN_ID
    return DEFAULT_ROBOT_DDS_DOMAIN_ID


_TELEOP_HAND_SOURCE_BY_DEVICE: dict[str, tuple[str, ...]] = {
    "unity": ("vr",),
    "unity_hybrid": ("vr",),
    "vr_masterarm": ("vr", "masterarm"),
    "masterarm": ("masterarm",),
}
_DEFAULT_TELEOP_HAND_SOURCE_BY_DEVICE: dict[str, str] = {
    "unity": "vr",
    "unity_hybrid": "vr",
    # vr_masterarm means leader arms plus the same VR/hybrid hand path used by
    # unity_hybrid. The leader hand remains available as an explicit override.
    "vr_masterarm": "vr",
    "masterarm": "masterarm",
}
HYBRID_TORSO_TELEOP_DEVICES = frozenset({"unity_hybrid", "vr_masterarm"})


def teleop_uses_hybrid_torso(teleop_device: Optional[str]) -> bool:
    return teleop_device in HYBRID_TORSO_TELEOP_DEVICES


def allowed_teleop_hand_sources(teleop_device: Optional[str]) -> tuple[str, ...]:
    if teleop_device is None:
        return ()
    return _TELEOP_HAND_SOURCE_BY_DEVICE.get(str(teleop_device), ())


def resolve_teleop_hand_source(
    teleop_device: Optional[str],
    teleop_hand_source: Optional[str],
) -> Optional[str]:
    if teleop_device is None:
        if teleop_hand_source is None:
            return None
        raise ValueError("teleop_hand_source requires a teleop_device")

    allowed_sources = allowed_teleop_hand_sources(teleop_device)
    if not allowed_sources:
        raise ValueError(f"Unknown teleop_device={teleop_device!r}")

    if teleop_hand_source is None:
        return _DEFAULT_TELEOP_HAND_SOURCE_BY_DEVICE[str(teleop_device)]

    resolved = str(teleop_hand_source)
    if resolved not in allowed_sources:
        raise ValueError(
            f"teleop_hand_source={resolved!r} is not supported for teleop_device={teleop_device!r}. "
            f"Allowed: {list(allowed_sources)}"
        )
    return resolved

@dataclass
class WorkerContext:
    name: str
    bus: EventBus
    run_config: RunConfig        
    stop_event: Optional[Any] = None
    log_queue: Optional[Any] = None
    
    shared_lock: Optional[dict[str, Any]] = None
    shm_name: Optional[dict[str, str]] = None
    shared_memory: Optional[dict[str, "SharedMemoryManager"]] = None
    runtime_diagnostics: Optional[Any] = None


class LoopDiagnostics:
    """Low-overhead loop timing sampler published to the Web UI."""

    def __init__(
        self,
        *,
        ctx: WorkerContext,
        loop: str,
        target_hz: float,
        publish_interval_s: float = 1.0,
    ) -> None:
        self._sink = ctx.runtime_diagnostics
        self.worker = str(ctx.name)
        self.loop = str(loop)
        self.key = self.worker if self.loop == "main" else f"{self.worker}:{self.loop}"
        self.target_hz = max(1e-9, float(target_hz))
        self.target_period_s = 1.0 / self.target_hz
        self.publish_interval_s = max(0.2, float(publish_interval_s))

        self._last_start_s: float | None = None
        self._last_publish_s = 0.0
        self._count = 0
        self._actual_hz: float | None = None
        self._period_ms: float | None = None
        self._jitter_ms: float | None = None
        self._latency_ms: float | None = None
        self._max_jitter_ms = 0.0
        self._max_latency_ms = 0.0

    @staticmethod
    def _ema(previous: float | None, value: float, alpha: float = 0.18) -> float:
        if previous is None:
            return value
        return previous * (1.0 - alpha) + value * alpha

    @staticmethod
    def _round(value: float | None, digits: int = 3) -> float | None:
        if value is None:
            return None
        return round(float(value), digits)

    def observe(self, *, start_s: float, end_s: float) -> None:
        if self._sink is None:
            return

        self._count += 1
        latency_ms = max(0.0, (end_s - start_s) * 1000.0)
        self._latency_ms = self._ema(self._latency_ms, latency_ms)
        self._max_latency_ms = max(self._max_latency_ms, latency_ms)

        if self._last_start_s is not None:
            period_s = max(1e-9, start_s - self._last_start_s)
            period_ms = period_s * 1000.0
            actual_hz = 1.0 / period_s
            jitter_ms = abs(period_s - self.target_period_s) * 1000.0
            self._actual_hz = self._ema(self._actual_hz, actual_hz)
            self._period_ms = self._ema(self._period_ms, period_ms)
            self._jitter_ms = self._ema(self._jitter_ms, jitter_ms)
            self._max_jitter_ms = max(self._max_jitter_ms, jitter_ms)

        self._last_start_s = start_s
        if end_s - self._last_publish_s >= self.publish_interval_s:
            self._publish(stopped=False)
            self._last_publish_s = end_s

    def stop(self) -> None:
        self._publish(stopped=True)

    def _publish(self, *, stopped: bool) -> None:
        if self._sink is None:
            return
        payload = {
            "worker": self.worker,
            "loop": self.loop,
            "key": self.key,
            "target_hz": self._round(self.target_hz),
            "actual_hz": self._round(self._actual_hz),
            "period_ms": self._round(self._period_ms),
            "jitter_ms": self._round(self._jitter_ms),
            "latency_ms": self._round(self._latency_ms),
            "max_jitter_ms": self._round(self._max_jitter_ms),
            "max_latency_ms": self._round(self._max_latency_ms),
            "samples": int(self._count),
            "updated_at": time.time(),
            "stopped": bool(stopped),
        }
        try:
            self._sink[self.key] = payload
        except Exception:
            pass


class BaseWorker:
    """모든 워커의 공통 베이스: 이벤트 읽기 + 상태 전이를 획일화."""

    def __init__(self, ctx: WorkerContext) -> None:
        self.ctx = ctx
        self._state_lock = threading.Lock()
        self._state: ModeState = ModeState.WAIT_CONNECT

    @property
    def state(self) -> ModeState:
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: ModeState) -> None:
        with self._state_lock:
            self._state = new_state

    def should_stop(self) -> bool:
        stop_event = self.ctx.stop_event
        if stop_event is None:
            return False
        try:
            return bool(stop_event.is_set())
        except Exception:
            return False

    def poll(self) -> tuple[EventSnapshot, TransitionResult]:
        """모든 워커에서 동일하게 호출하는 poll 함수.

        1) 이벤트 스냅샷 읽기
        2) 상태 전이 수행
        """
        ev = self.ctx.bus.read_snapshot()
        with self._state_lock:
            tr = step(self._state, ev)
            self._state = tr.state
        return ev, tr

    # ---- Lifecycle hooks --------------------------------------------------
    def on_start(self) -> None:
        pass

    def on_stop(self) -> None:
        pass


class SingleRateWorker(BaseWorker):
    """단일 루프(단일 주기) 워커 템플릿."""

    def __init__(self, ctx: WorkerContext, hz: float) -> None:
        super().__init__(ctx)
        self.hz = float(hz)

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        """사용자가 구현: 틱마다 수행할 작업."""
        raise NotImplementedError

    def run(self) -> None:
        self.on_start()
        rate = Rate(self.hz)
        diagnostics = LoopDiagnostics(ctx=self.ctx, loop="main", target_hz=self.hz)
        try:
            while True:
                if self.should_stop():
                    break
                loop_start = time.perf_counter()
                ev, tr = self.poll()
                if self.state == ModeState.EXIT:
                    break
                self.step_once(ev, tr)
                diagnostics.observe(start_s=loop_start, end_s=time.perf_counter())
                rate.sleep()
        finally:
            diagnostics.stop()
            self.on_stop()


class DualRateWorker(BaseWorker):
    """느린 루프+빠른 루프(2 스레드) 워커 템플릿."""

    def __init__(self, ctx: WorkerContext, slow_hz: float, fast_hz: float) -> None:
        super().__init__(ctx)
        self.slow_hz = float(slow_hz)
        self.fast_hz = float(fast_hz)

        self._stop_threads = threading.Event()
        self._t_slow: Optional[threading.Thread] = None
        self._t_fast: Optional[threading.Thread] = None

    def do_slow(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        raise NotImplementedError

    def do_fast(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        raise NotImplementedError

    def _slow_loop(self) -> None:
        rate = Rate(self.slow_hz)
        diagnostics = LoopDiagnostics(ctx=self.ctx, loop="slow", target_hz=self.slow_hz)
        try:
            while not self._stop_threads.is_set():
                if self.should_stop():
                    self._stop_threads.set()
                    break
                loop_start = time.perf_counter()
                ev, tr = self.poll()
                if self.state == ModeState.EXIT:
                    self._stop_threads.set()
                    break
                self.do_slow(ev, tr)
                diagnostics.observe(start_s=loop_start, end_s=time.perf_counter())
                rate.sleep()
        finally:
            diagnostics.stop()

    def _fast_loop(self) -> None:
        rate = Rate(self.fast_hz)
        diagnostics = LoopDiagnostics(ctx=self.ctx, loop="fast", target_hz=self.fast_hz)
        try:
            while not self._stop_threads.is_set():
                if self.should_stop():
                    self._stop_threads.set()
                    break
                loop_start = time.perf_counter()
                ev, tr = self.poll()
                if self.state == ModeState.EXIT:
                    self._stop_threads.set()
                    break
                self.do_fast(ev, tr)
                diagnostics.observe(start_s=loop_start, end_s=time.perf_counter())
                rate.sleep()
        finally:
            diagnostics.stop()

    def run(self) -> None:
        self.on_start()
        try:
            self._t_slow = threading.Thread(target=self._slow_loop, name=f"{self.ctx.name}-slow", daemon=True)
            self._t_fast = threading.Thread(target=self._fast_loop, name=f"{self.ctx.name}-fast", daemon=True)
            self._t_slow.start()
            self._t_fast.start()

            # 메인 스레드는 join만 수행
            while not self._stop_threads.is_set():
                if self.should_stop():
                    self._stop_threads.set()
                    break
                time.sleep(0.1)
        finally:
            self._stop_threads.set()
            try:
                join_timeout = float(getattr(self, "thread_join_timeout_s", 1.0))
            except Exception:
                join_timeout = 1.0
            if join_timeout < 0.0:
                join_timeout = 1.0
            if self._t_slow is not None:
                self._t_slow.join(timeout=join_timeout)
            if self._t_fast is not None:
                self._t_fast.join(timeout=join_timeout)
            self.on_stop()
