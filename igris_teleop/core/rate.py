from __future__ import annotations

import time


class Rate:
    """고정 주기 루프 헬퍼(원 프로젝트의 Rate와 유사한 형태)."""

    def __init__(self, hz: float) -> None:
        if hz <= 0:
            raise ValueError("hz must be > 0")
        self.hz = float(hz)
        self.dt = 1.0 / self.hz
        self.next_t = time.perf_counter() + self.dt
        self._last_tick = None

    def sleep(self) -> None:
        now = time.perf_counter()
        remaining = self.next_t - now
        if remaining > 0:
            # A Python busy-wait holds the GIL.  With the real controller's
            # 300 Hz publisher plus two 100 Hz worker loops, the former 0.5 ms
            # spin window starved pose interpolation down to roughly 30 Hz.
            # time.sleep() releases the GIL and Linux monotonic timers provide
            # sufficient precision for this user-space command publisher.
            time.sleep(remaining)
        self.next_t += self.dt
        # If the loop overran by one or more periods, skip missed deadlines.
        # Keeping next_t in the past makes the caller run an unbounded catch-up
        # loop without sleeping, which can monopolize the GIL and starve the
        # real-robot shutdown trajectory updater.
        now = time.perf_counter()
        if self.next_t <= now:
            missed_periods = int((now - self.next_t) / self.dt) + 1
            self.next_t += missed_periods * self.dt

    def tick_hz(self) -> float:
        now = time.perf_counter_ns()
        if self._last_tick is None:
            self._last_tick = now
            return 0.0
        dt = now - self._last_tick
        self._last_tick = now
        return 1e9 / dt if dt > 0 else 0.0
