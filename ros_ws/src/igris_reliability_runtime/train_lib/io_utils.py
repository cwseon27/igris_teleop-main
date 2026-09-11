from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def current_iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_dataset_npz(
    path: str | Path,
    X,
    y_left,
    y_right,
    future_score_left,
    future_score_right,
    timestamps,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "X": np.asarray(X, dtype=np.float32),
        "y_left": np.asarray(y_left, dtype=np.int64),
        "y_right": np.asarray(y_right, dtype=np.int64),
        "future_score_left": np.asarray(future_score_left, dtype=np.float32),
        "future_score_right": np.asarray(future_score_right, dtype=np.float32),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }

    for key, value in metadata.items():
        if value is None:
            arrays[key] = np.asarray("")
        elif isinstance(value, (str, int, float, bool, np.integer, np.floating)):
            arrays[key] = np.asarray(value)
        else:
            arrays[key] = np.asarray(json.dumps(value, sort_keys=True))

    np.savez_compressed(path, **arrays)


def scalar_from_npz(value):
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return item
    return arr.tolist()

