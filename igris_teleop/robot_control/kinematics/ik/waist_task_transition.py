from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class WaistTransitionParams:
    low_threshold: float = 0.2
    high_threshold: float = 0.8
    activation_rise_rate: float = 2.0
    activation_fall_rate: float = 4.0
    max_dt_sec: float = 0.1
    base_max_delta_per_step: float | None = None
    base_hold_weight: float = 1.0
    scale_head_translation_with_activation: bool = False
    low_conf_head_translation_weight: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.low_threshold,
            self.high_threshold,
            self.activation_rise_rate,
            self.activation_fall_rate,
            self.max_dt_sec,
            self.base_hold_weight,
            self.low_conf_head_translation_weight,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("waist transition parameters must be finite")
        if not 0.0 <= self.low_threshold < self.high_threshold <= 1.0:
            raise ValueError(
                "waist transition thresholds must satisfy "
                "0 <= low_threshold < high_threshold <= 1"
            )
        if self.activation_rise_rate < 0.0 or self.activation_fall_rate < 0.0:
            raise ValueError("waist activation rates must be nonnegative")
        if self.max_dt_sec <= 0.0:
            raise ValueError("max_dt_sec must be positive")
        if self.base_max_delta_per_step is not None:
            limit = float(self.base_max_delta_per_step)
            if not math.isfinite(limit) or limit <= 0.0:
                raise ValueError("base_max_delta_per_step must be finite and positive")
        if self.base_hold_weight < 0.0:
            raise ValueError("base_hold_weight must be nonnegative")
        if self.low_conf_head_translation_weight < 0.0:
            raise ValueError("low_conf_head_translation_weight must be nonnegative")


def compute_task_activation(
    confidence: float,
    low_threshold: float,
    high_threshold: float,
) -> float:
    low = float(low_threshold)
    high = float(high_threshold)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("activation thresholds must be finite")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(
            "activation thresholds must satisfy "
            "0 <= low_threshold < high_threshold <= 1"
        )

    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0

    value = min(1.0, max(0.0, value))
    x = min(1.0, max(0.0, (value - low) / (high - low)))
    return x * x * (3.0 - 2.0 * x)


@dataclass(frozen=True)
class WaistTransitionResult:
    confidence: float
    target_valid: bool
    forced_disable: bool
    raw_activation: float
    filtered_activation: float
    dt: float
    previous_published_waist: np.ndarray | None

    def __post_init__(self) -> None:
        if self.previous_published_waist is None:
            return
        waist = np.asarray(
            self.previous_published_waist,
            dtype=np.float64,
        ).reshape(-1).copy()
        waist.setflags(write=False)
        object.__setattr__(self, "previous_published_waist", waist)


class WaistTaskTransition:
    def __init__(
        self,
        params: WaistTransitionParams,
        *,
        waist_dof: int = 3,
    ) -> None:
        if waist_dof <= 0:
            raise ValueError("waist_dof must be positive")
        self.params = params
        self._waist_dof = int(waist_dof)
        self._previous_activation = 0.0
        self._previous_published_waist: np.ndarray | None = None
        self._last_update_time: float | None = None

    @property
    def previous_activation(self) -> float:
        return float(self._previous_activation)

    @property
    def previous_published_waist(self) -> np.ndarray | None:
        if self._previous_published_waist is None:
            return None
        return self._previous_published_waist.copy()

    def _validated_waist(self, command: np.ndarray) -> np.ndarray:
        waist = np.asarray(command, dtype=np.float64).reshape(-1)
        if waist.shape != (self._waist_dof,):
            raise ValueError(
                f"waist command shape {waist.shape} != ({self._waist_dof},)"
            )
        if not np.all(np.isfinite(waist)):
            raise ValueError("waist command must be finite")
        return waist.copy()

    @staticmethod
    def _validated_timestamp(timestamp: float | None) -> float | None:
        if timestamp is None:
            return None
        value = float(timestamp)
        if not math.isfinite(value):
            raise ValueError("transition timestamp must be finite")
        return value

    def reset(
        self,
        published_waist_command: np.ndarray,
        *,
        activation: float = 0.0,
        timestamp: float | None = None,
    ) -> None:
        activation_value = float(activation)
        if not math.isfinite(activation_value):
            raise ValueError("activation must be finite")
        self._previous_published_waist = self._validated_waist(
            published_waist_command
        )
        self._previous_activation = min(1.0, max(0.0, activation_value))
        self._last_update_time = self._validated_timestamp(timestamp)

    def commit_published_command(
        self,
        published_waist_command: np.ndarray,
    ) -> None:
        self._previous_published_waist = self._validated_waist(
            published_waist_command
        )

    def _compute_dt(self, now: float) -> float:
        if self._last_update_time is None:
            return 0.0
        raw_dt = now - self._last_update_time
        if not math.isfinite(raw_dt) or raw_dt <= 0.0:
            return 0.0
        return min(raw_dt, float(self.params.max_dt_sec))

    def update_activation(
        self,
        confidence: float,
        *,
        target_valid: bool,
        now: float | None = None,
    ) -> WaistTransitionResult:
        update_time = time.monotonic() if now is None else float(now)
        if not math.isfinite(update_time):
            update_time = (
                self._last_update_time
                if self._last_update_time is not None
                else time.monotonic()
            )
        dt = self._compute_dt(update_time)

        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = float("nan")

        forced_disable = (
            not bool(target_valid)
            or not math.isfinite(confidence_value)
            or confidence_value <= 0.0
        )
        if forced_disable:
            sanitized_confidence = 0.0
            raw_activation = 0.0
            filtered_activation = 0.0
        else:
            sanitized_confidence = min(1.0, confidence_value)
            raw_activation = compute_task_activation(
                sanitized_confidence,
                self.params.low_threshold,
                self.params.high_threshold,
            )
            activation_delta = raw_activation - self._previous_activation
            if activation_delta >= 0.0:
                max_delta = float(self.params.activation_rise_rate) * dt
            else:
                max_delta = float(self.params.activation_fall_rate) * dt
            limited_delta = min(max_delta, max(-max_delta, activation_delta))
            filtered_activation = min(
                1.0,
                max(0.0, self._previous_activation + limited_delta),
            )

        self._previous_activation = filtered_activation
        self._last_update_time = update_time
        return WaistTransitionResult(
            confidence=sanitized_confidence,
            target_valid=bool(target_valid),
            forced_disable=forced_disable,
            raw_activation=raw_activation,
            filtered_activation=filtered_activation,
            dt=dt,
            previous_published_waist=self._previous_published_waist,
        )

    def resolved_base_delta_limit(self, fallback_limit: float) -> float:
        configured = self.params.base_max_delta_per_step
        value = float(fallback_limit if configured is None else configured)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("resolved waist delta limit must be finite and positive")
        return value

    def activated_delta_limit(
        self,
        activation: float,
        *,
        fallback_limit: float,
    ) -> float:
        activation_value = min(1.0, max(0.0, float(activation)))
        return activation_value * self.resolved_base_delta_limit(fallback_limit)

    def waist_hold_weight(self, activation: float) -> float:
        activation_value = min(1.0, max(0.0, float(activation)))
        return (1.0 - activation_value) * float(self.params.base_hold_weight)

    def head_translation_weight(
        self,
        activation: float,
        *,
        base_weight: float,
    ) -> float:
        base = max(0.0, float(base_weight))
        if not self.params.scale_head_translation_with_activation:
            return base
        activation_value = min(1.0, max(0.0, float(activation)))
        low = float(self.params.low_conf_head_translation_weight)
        return low + activation_value * (base - low)
