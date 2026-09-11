from __future__ import annotations

import numpy as np

HAND_COMMAND_MIN = 0.0
HAND_COMMAND_MAX = 1.0
HAND_CLOSE_GAIN = 1.5


def compress_normalized_hand_command(
    values,
    *,
    lower: float = HAND_COMMAND_MIN,
    upper: float = HAND_COMMAND_MAX,
    close_gain: float = HAND_CLOSE_GAIN,
    dtype=None,
) -> np.ndarray:
    """Boost human hand closing so the robot reaches full close earlier."""
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    boosted = np.clip(clipped * float(close_gain), 0.0, 1.0)
    compressed = float(lower) + boosted * float(upper - lower)
    if dtype is not None:
        return compressed.astype(dtype, copy=False)
    return compressed
