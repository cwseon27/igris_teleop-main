from __future__ import annotations

from typing import Iterable

import numpy as np


def make_future_label_from_bool_history(
    history: Iterable[bool],
    future_horizon: int,
    stable_threshold: float,
) -> tuple[int, float]:
    values = np.asarray([1.0 if bool(v) else 0.0 for v in history], dtype=np.float32)
    if values.size < int(future_horizon):
        raise ValueError(
            f"need at least future_horizon={future_horizon} values, got {values.size}"
        )
    values = values[: int(future_horizon)]
    future_score = float(np.mean(values))
    label = 1 if future_score >= float(stable_threshold) else 0
    return label, future_score

