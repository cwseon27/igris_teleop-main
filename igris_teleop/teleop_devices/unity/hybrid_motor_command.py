"""Latched final-motor stream, deliberately separate from legacy finger-close data."""
from __future__ import annotations

import numpy as np

from ...hand_control.command_range import validated_motor_command


class HybridMotorCommand:
    """Hold last activated command on tracking loss, malformed input or topic stalls.

    A fresh tracking=true heartbeat followed by a valid motor sample is required
    to activate. Startup untracked placeholders never take control of the hand.
    There is no automatic retargeter/source switch when this stream disappears.
    """

    def __init__(self, stale_after_sec: float = 0.25) -> None:
        self.command = np.zeros(6, dtype=np.float64)
        self.activated = False
        self.tracked = False
        self.tracked_at = None
        self.command_at = None
        self.stale_after_sec = max(0.0, float(stale_after_sec))

    def update_tracking(self, tracked: bool, now: float) -> None:
        self.tracked = bool(tracked)
        self.tracked_at = float(now)

    def update_command(self, values, now: float) -> bool:
        if not self.tracked or self.tracked_at is None:
            return False
        age = float(now) - self.tracked_at
        if age < 0.0 or age > self.stale_after_sec:
            return False
        try:
            command = validated_motor_command(values)
        except (ValueError, TypeError):
            return False
        self.command = command
        self.command_at = float(now)
        self.activated = True
        return True

    def snapshot(self) -> tuple[np.ndarray, bool]:
        return self.command.copy(), self.activated
