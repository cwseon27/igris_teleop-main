from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


FINGER_COUNT = 5


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _close_vector(values: Iterable[float] | None, count: int = FINGER_COUNT) -> tuple[float, ...] | None:
    if values is None:
        return None
    raw = tuple(float(value) for value in values)
    if len(raw) != count or not all(math.isfinite(value) for value in raw):
        return None
    return tuple(clamp(value) for value in raw)


@dataclass(frozen=True)
class HandFusionResult:
    command: tuple[float, ...]
    theta: float
    confidence: float
    source: str
    tracked: bool


class ReliabilityAwareHandCommandFusion:
    """Blend hand sources only in normalized finger-close command space."""

    def __init__(
        self,
        *,
        confidence_gamma: float = 1.0,
        previous_command_weight: float = 0.0,
        close_rate_per_sec: float = 15.0,
        open_rate_per_sec: float = 20.0,
        openxr_only_confidence_threshold: float = 0.6,
        source_dropout_grace_sec: float = 0.15,
        nominal_rate_hz: float = 30.0,
        openxr_only: bool = False,
        command_size: int = FINGER_COUNT,
    ) -> None:
        # A source override is distinct from a confidence value: even confidence=1
        # normally permits MediaPipe fallback when OpenXR tracking disappears.
        self.openxr_only = bool(openxr_only)
        if command_size not in (5, 6):
            raise ValueError("command_size must be 5 fingers or 6 retargeted motors")
        self.command_size = command_size
        self.confidence_gamma = max(0.01, float(confidence_gamma))
        self.previous_command_weight = max(0.0, float(previous_command_weight))
        self.close_rate_per_sec = max(0.0, float(close_rate_per_sec))
        self.open_rate_per_sec = max(0.0, float(open_rate_per_sec))
        self.openxr_only_confidence_threshold = clamp(openxr_only_confidence_threshold)
        self.source_dropout_grace_sec = max(0.0, float(source_dropout_grace_sec))
        self.nominal_dt = 1.0 / max(1.0, float(nominal_rate_hz))
        self.previous = (0.0,) * self.command_size
        self.last_time: float | None = None
        self.last_mediapipe: tuple[float, ...] | None = None
        self.last_mediapipe_time: float | None = None

    def _with_dropout_grace(
        self,
        current: tuple[float, ...] | None,
        previous: tuple[float, ...] | None,
        previous_time: float | None,
        now: float,
    ) -> tuple[float, ...] | None:
        if current is not None:
            return current
        if previous is None or previous_time is None:
            return None
        age = float(now) - float(previous_time)
        if 0.0 <= age <= self.source_dropout_grace_sec:
            return previous
        return None

    def update(
        self,
        *,
        openxr_close: Iterable[float] | None,
        mediapipe_close: Iterable[float] | None,
        confidence: float,
        openxr_ready: bool,
        mediapipe_ready: bool,
        now: float,
    ) -> HandFusionResult:
        now = float(now)
        current_openxr = _close_vector(openxr_close, self.command_size) if openxr_ready else None
        current_mediapipe = (
            _close_vector(mediapipe_close, self.command_size)
            if mediapipe_ready and not self.openxr_only
            else None
        )
        if current_mediapipe is not None:
            self.last_mediapipe = current_mediapipe
            self.last_mediapipe_time = now

        openxr = current_openxr
        mediapipe = (
            self._with_dropout_grace(
                current_mediapipe, self.last_mediapipe, self.last_mediapipe_time, now
            )
            if not self.openxr_only
            else None
        )
        confidence = (
            1.0
            if self.openxr_only
            else clamp(confidence if math.isfinite(float(confidence)) else 0.0)
        )
        theta = confidence**self.confidence_gamma

        raw: tuple[float, ...] | None = None
        tracked = False
        if self.openxr_only:
            if openxr is not None:
                raw = openxr
                source = "openxr_only"
                tracked = True
            else:
                source = "hold_openxr_missing"
        elif openxr is not None and mediapipe is not None:
            raw = tuple(
                theta * openxr[i] + (1.0 - theta) * mediapipe[i]
                for i in range(self.command_size)
            )
            source = "hybrid"
            tracked = True
        elif mediapipe is not None:
            raw = mediapipe
            theta = 0.0
            source = "mediapipe_only"
            tracked = True
        elif openxr is not None and confidence >= self.openxr_only_confidence_threshold:
            raw = openxr
            theta = 1.0
            source = "openxr_only"
            tracked = True
        elif openxr is not None:
            source = "hold_openxr_unreliable"
        else:
            source = "hold_no_source"

        dt = self.nominal_dt
        if self.last_time is not None:
            dt = clamp(now - self.last_time, 0.0, 0.1)
        self.last_time = now

        if raw is not None:
            weight = self.previous_command_weight
            desired = tuple(
                clamp((raw[i] + weight * self.previous[i]) / (1.0 + weight))
                for i in range(self.command_size)
            )
            command = []
            for previous, target in zip(self.previous, desired):
                delta = target - previous
                lower = -self.open_rate_per_sec * dt
                upper = self.close_rate_per_sec * dt
                command.append(clamp(previous + clamp(delta, lower, upper)))
            self.previous = tuple(command)

        return HandFusionResult(
            command=self.previous,
            theta=clamp(theta),
            confidence=confidence,
            source=source,
            tracked=tracked,
        )
