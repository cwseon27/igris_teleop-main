from __future__ import annotations

import time

import logging_mp
from ..core.events import EventSnapshot
from ..core.state_machine import ModeState, TransitionResult
from ..core.worker_base import DualRateWorker, WorkerContext


logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)


class DualExampleWorker(DualRateWorker):
    """Dual-rate 워커 예제:
    - slow: 관측(telemetry) 업데이트
    - fast: 제어(명령) 업데이트
    """

    def __init__(self, ctx: WorkerContext, slow_hz: float = 10.0, fast_hz: float = 10.0) -> None:
        super().__init__(ctx, slow_hz=slow_hz, fast_hz=fast_hz)
        self._obs = 0.0
        self._cmd = 0.0
        self._fast_ticks = 0
        self._shared_memory = ctx.shared_memory
        self._owns_shared_memory = False
        
    def on_start(self) -> None:
        logger.info(f"[{self.ctx.name}] start (dual-rate slow={self.slow_hz}Hz fast={self.fast_hz}Hz)")

    def do_slow(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        # 예: 센서 관측값 업데이트 (여기서는 단순히 시간 기반 값)
        # RUN일 때만 명령 생성/전송
        if self.state == ModeState.RUN:
            act_shm = self._shared_memory.get("act_shm")
            # act_shm.write_data()
            self._cmd = self._cmd + 0.01
        else:
            self._cmd = 0.0

        logger.info(f"[{self.ctx.name}][slow] state={self.state.name} reason={tr.reason} cmd={self._cmd:.3f}")
            
    def do_fast(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        self._fast_ticks += 1

        # RUN일 때만 명령 생성/전송
        if self.state == ModeState.RUN:
            act_shm = self._shared_memory.get("act_shm")
            # act_shm.write_data()
            
            self._cmd = self._cmd + 0.01
        else:
            self._cmd = 0.0


        logger.info(f"[{self.ctx.name}][fast] state={self.state.name} reason={tr.reason} cmd={self._cmd:.3f}")

    def on_stop(self) -> None:
        if self._shared_memory:
            for key, mgr in self._shared_memory.items():
                try:
                    mgr.worker_close()
                except Exception:
                    logger.exception(f"[{self.ctx.name}] failed to close shared memory {key}")

        
        logger.info(f"[{self.ctx.name}] stop")
