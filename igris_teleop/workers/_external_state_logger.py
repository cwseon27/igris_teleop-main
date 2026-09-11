from __future__ import annotations

from typing import Optional

import logging_mp
import numpy as np

from ..core.events import EventSnapshot
from ..core.state_machine import TransitionResult
from ..core.worker_base import SingleRateWorker, WorkerContext
from ..sharedmemory.shm_schema import MODE_LAYOUT


logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)

MODE_FIELD_NAMES = tuple(name for (name, _, _) in MODE_LAYOUT)


class ExternalStateLoggerWorker(SingleRateWorker):
    def __init__(self, ctx: WorkerContext, hz: float = 5.0) -> None:
        super().__init__(ctx, hz=hz)
        self._shared_memory = ctx.shared_memory or {}
        self._mode_shm = self._shared_memory.get("mode_shm")
        self._last_state_name: Optional[str] = None
        self._last_mode_snapshot: Optional[dict[str, bool]] = None
        self._mode_warned = False

    def on_start(self) -> None:
        logger.info("[%s] external state logger start (single-rate %.1f Hz)", self.ctx.name, self.hz)
        if self._mode_shm is None:
            self._mode_warned = True
            logger.warning("[%s] mode_shm unavailable; mode change logs disabled", self.ctx.name)

    def _read_mode_snapshot(self) -> Optional[dict[str, bool]]:
        if self._mode_shm is None:
            return None
        try:
            data = self._mode_shm.read_data()
        except Exception:
            if not self._mode_warned:
                self._mode_warned = True
                logger.warning("[%s] failed to read mode_shm", self.ctx.name, exc_info=True)
            return None

        self._mode_warned = False
        snapshot: dict[str, bool] = {}
        for name in MODE_FIELD_NAMES:
            try:
                snapshot[name] = bool(np.asarray(data.get(name, False)).reshape(()).item())
            except Exception:
                snapshot[name] = False
        return snapshot

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        del ev
        state_name = self.state.name
        mode_snapshot = self._read_mode_snapshot()

        state_changed = state_name != self._last_state_name
        mode_changed = mode_snapshot != self._last_mode_snapshot
        if state_changed or mode_changed:
            logger.info(
                "[%s] worker_state=%s reason=%s mode_shm=%s",
                self.ctx.name,
                state_name,
                tr.reason,
                mode_snapshot,
            )
        self._last_state_name = state_name
        self._last_mode_snapshot = mode_snapshot

    def on_stop(self) -> None:
        for key, mgr in self._shared_memory.items():
            try:
                mgr.worker_close()
            except Exception:
                logger.exception("[%s] failed to close shared memory %s", self.ctx.name, key)
        logger.info("[%s] external state logger stop", self.ctx.name)
