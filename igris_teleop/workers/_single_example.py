from __future__ import annotations

import logging_mp


from ..core.events import EventSnapshot
from ..core.state_machine import ModeState, TransitionResult
from ..core.worker_base import SingleRateWorker, WorkerContext


logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)



class SingleExampleWorker(SingleRateWorker):
    """Single-rate 워커 예제: 상태에 따라 카운터를 업데이트."""

    def __init__(self, ctx: WorkerContext, hz: float = 10.0) -> None:
        super().__init__(ctx, hz=hz)
        self._counter = 0
        self._shared_memory = ctx.shared_memory
        self._owns_shared_memory = False


    def on_start(self) -> None:
        logger.info(f"[{self.ctx.name}] start (single-rate {self.hz} Hz)")

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        st = self.state

        # 예시: RUN 상태에서만 카운터 증가
        if st == ModeState.RUN:
            self._counter += 1
            act_shm = self._shared_memory.get("act_shm")
            # act_shm.write_data()

        logger.info(f"[{self.ctx.name}] state={st.name} reason={tr.reason} counter={self._counter}")

    def on_stop(self) -> None:
        if self._shared_memory:
            for key, mgr in self._shared_memory.items():
                try:
                    mgr.worker_close()
                except Exception:
                    logger.exception(f"[{self.ctx.name}] failed to close shared memory {key}")

        logger.info(f"[{self.ctx.name}] stop")
