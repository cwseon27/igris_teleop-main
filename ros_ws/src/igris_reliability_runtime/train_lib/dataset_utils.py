from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io_utils import scalar_from_npz


KNOWN_DATASET_ARRAYS = {
    "X",
    "y_left",
    "y_right",
    "future_score_left",
    "future_score_right",
    "timestamps",
    "metadata_json",
    "source_id",
    "original_index",
    "scene_id",
    "split_id",
    "joint_label",
}


def load_dataset(path: str | Path):
    data = np.load(path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y_left = np.asarray(data["y_left"], dtype=np.int64)
    y_right = np.asarray(data["y_right"], dtype=np.int64)

    metadata = {}
    if "metadata_json" in data:
        raw_metadata = scalar_from_npz(data["metadata_json"])
        if raw_metadata:
            metadata.update(json.loads(raw_metadata))

    for key in data.files:
        if key not in KNOWN_DATASET_ARRAYS:
            metadata[key] = scalar_from_npz(data[key])

    if X.ndim != 3:
        raise ValueError(f"expected X shape (N, window_size, feature_dim), got {X.shape}")
    if y_left.shape[0] != X.shape[0] or y_right.shape[0] != X.shape[0]:
        raise ValueError("X, y_left, and y_right have inconsistent sample counts")

    metadata.setdefault("window_size", int(X.shape[1]))
    metadata.setdefault("feature_dim", int(X.shape[2]))
    return X, y_left, y_right, metadata


def flatten_for_sklearn(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X
    if X.ndim != 3:
        raise ValueError(f"expected X shape (N, T, C) or (N, T*C), got {X.shape}")
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


def to_sequence_channels_first(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"expected X shape (N, T, C), got {X.shape}")
    return np.transpose(X, (0, 2, 1))


def _split_count(n_samples: int, ratio: float, reserve_train: int = 1) -> int:
    if ratio <= 0.0 or n_samples <= reserve_train:
        return 0
    count = int(round(n_samples * ratio))
    if count == 0 and ratio > 0.0:
        count = 1
    return min(count, max(0, n_samples - reserve_train))


def _stratify_or_none(indices: np.ndarray, y: np.ndarray, holdout_count: int):
    y_subset = np.asarray(y)[indices]
    classes, counts = np.unique(y_subset, return_counts=True)
    n_classes = classes.size
    if n_classes < 2:
        return None
    if holdout_count < n_classes:
        return None
    if indices.size - holdout_count < n_classes:
        return None
    if np.min(counts) < 2:
        return None
    return y_subset


def _split_once(indices: np.ndarray, y: np.ndarray, holdout_count: int, random_state: int):
    indices = np.asarray(indices, dtype=np.int64)
    if holdout_count <= 0:
        return indices, np.asarray([], dtype=np.int64)

    rng = np.random.default_rng(random_state)
    stratify = _stratify_or_none(indices, y, holdout_count)
    if stratify is None:
        shuffled = np.array(indices, copy=True)
        rng.shuffle(shuffled)
        holdout_idx = np.sort(shuffled[:holdout_count])
        train_idx = np.sort(shuffled[holdout_count:])
        return train_idx, holdout_idx

    y_subset = np.asarray(y)[indices]
    classes, counts = np.unique(y_subset, return_counts=True)
    allocations = {}
    for klass, count in zip(classes, counts):
        raw = int(round((count / indices.size) * holdout_count))
        allocations[int(klass)] = min(max(raw, 1), int(count) - 1)

    def allocated_total():
        return int(sum(allocations.values()))

    while allocated_total() > holdout_count:
        candidates = [klass for klass, value in allocations.items() if value > 1]
        if not candidates:
            break
        klass = max(candidates, key=lambda k: allocations[k])
        allocations[klass] -= 1

    while allocated_total() < holdout_count:
        candidates = []
        for klass, count in zip(classes, counts):
            klass = int(klass)
            if allocations[klass] < int(count) - 1:
                candidates.append(klass)
        if not candidates:
            break
        klass = min(candidates, key=lambda k: allocations[k])
        allocations[klass] += 1

    holdout_parts = []
    train_parts = []
    for klass in classes:
        klass_indices = indices[y_subset == klass]
        klass_indices = np.array(klass_indices, copy=True)
        rng.shuffle(klass_indices)
        n_holdout = allocations[int(klass)]
        holdout_parts.append(klass_indices[:n_holdout])
        train_parts.append(klass_indices[n_holdout:])

    holdout_idx = np.sort(np.concatenate(holdout_parts).astype(np.int64))
    train_idx = np.sort(np.concatenate(train_parts).astype(np.int64))
    return train_idx, holdout_idx


def split_indices(
    y,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y)
    n_samples = y.shape[0]
    if n_samples == 0:
        raise ValueError("cannot split an empty dataset")

    all_indices = np.arange(n_samples)
    test_count = _split_count(n_samples, float(test_ratio), reserve_train=1)
    train_val_idx, test_idx = _split_once(all_indices, y, test_count, random_state)

    val_count = _split_count(n_samples, float(val_ratio), reserve_train=1)
    val_count = min(val_count, max(0, train_val_idx.size - 1))
    train_idx, val_idx = _split_once(train_val_idx, y, val_count, random_state + 1)

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def chronological_split_indices(
    n_samples: int,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    split_gap: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("cannot split an empty dataset")

    test_count = _split_count(n_samples, float(test_ratio), reserve_train=1)
    val_count = _split_count(n_samples - test_count, float(val_ratio), reserve_train=1)
    gap_count = max(0, int(split_gap))

    if val_count > 0 and test_count > 0:
        max_gap = max(0, (n_samples - test_count - val_count - 1) // 2)
        gap_count = min(gap_count, max_gap)
        train_end = n_samples - test_count - val_count - 2 * gap_count
        val_start = train_end + gap_count
        val_end = val_start + val_count
        test_start = val_end + gap_count
    elif test_count > 0:
        max_gap = max(0, n_samples - test_count - 1)
        gap_count = min(gap_count, max_gap)
        train_end = n_samples - test_count - gap_count
        val_start = train_end
        val_end = train_end
        test_start = train_end + gap_count
    else:
        train_end = n_samples
        val_start = n_samples
        val_end = n_samples
        test_start = n_samples

    train_idx = np.arange(0, max(1, train_end), dtype=np.int64)
    val_idx = np.arange(val_start, val_end, dtype=np.int64)
    test_idx = np.arange(test_start, n_samples, dtype=np.int64)
    return train_idx, val_idx, test_idx


def split_dataset(
    X,
    y,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    random_state: int = 42,
):
    train_idx, val_idx, test_idx = split_indices(y, test_ratio, val_ratio, random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    return (
        X[train_idx],
        X[val_idx],
        X[test_idx],
        y[train_idx],
        y[val_idx],
        y[test_idx],
    )
