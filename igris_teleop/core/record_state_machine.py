from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class RecordState(IntEnum):
    WAIT = 0
    RECORD_START = 1
    RECORD_DONE = 2
    RECORD_RESET = 3


@dataclass(frozen=True)
class RecordEventSnapshot:
    """Simple snapshot of record-related events."""

    record_start: bool = False
    record_done: bool = False
    record_reset: bool = False


@dataclass(frozen=True)
class RecordTransitionResult:
    state: RecordState
    reason: str


def step_record(state: RecordState, ev: RecordEventSnapshot) -> RecordTransitionResult:
    """Record-only state machine that works independently of EventBus/ModeState."""

    if state == RecordState.WAIT:
        if ev.record_reset:
            return RecordTransitionResult(RecordState.RECORD_RESET, "reset_requested")
        if ev.record_start:
            return RecordTransitionResult(RecordState.RECORD_START, "record_start")
        return RecordTransitionResult(state, "waiting")

    if state == RecordState.RECORD_START:
        if ev.record_reset:
            return RecordTransitionResult(RecordState.RECORD_RESET, "reset_requested")
        if ev.record_done:
            return RecordTransitionResult(RecordState.RECORD_DONE, "record_done")
        return RecordTransitionResult(state, "recording")

    if state == RecordState.RECORD_DONE:
        if ev.record_reset:
            return RecordTransitionResult(RecordState.RECORD_RESET, "reset_requested")
        if ev.record_start:
            return RecordTransitionResult(RecordState.RECORD_START, "restart_record")
        return RecordTransitionResult(state, "done")

    if state == RecordState.RECORD_RESET:
        if not ev.record_reset:
            return RecordTransitionResult(RecordState.WAIT, "reset_complete")
        return RecordTransitionResult(state, "resetting")

    # Defensive fallback: reset to WAIT if state becomes invalid
    return RecordTransitionResult(RecordState.WAIT, "unknown_state")
