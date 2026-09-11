from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def class_distribution(y) -> dict[str, int]:
    y = np.asarray(y, dtype=np.int64)
    values, counts = np.unique(y, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def evaluate_binary_model(model, X_test, y_test, name: str) -> dict:
    y_test = np.asarray(y_test, dtype=np.int64)
    metrics = {
        "name": name,
        "n_samples": int(y_test.size),
        "class_distribution": class_distribution(y_test),
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
    }

    print(f"\n[{name}] class distribution: {metrics['class_distribution']}")
    if y_test.size == 0:
        print(f"[{name}] no test samples; metrics skipped")
        return metrics

    confidence = np.asarray(model.predict_confidence(X_test), dtype=np.float32)
    y_pred = (confidence >= 0.5).astype(np.int64)

    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
    metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))

    if np.unique(y_test).size >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, confidence))

    print(f"[{name}] accuracy={metrics['accuracy']:.4f}")
    print(f"[{name}] precision={metrics['precision']:.4f}")
    print(f"[{name}] recall={metrics['recall']:.4f}")
    print(f"[{name}] f1={metrics['f1']:.4f}")
    if metrics["roc_auc"] is None:
        print(f"[{name}] roc_auc=skipped (single class in y_true)")
    else:
        print(f"[{name}] roc_auc={metrics['roc_auc']:.4f}")

    return metrics


def benchmark_predict_latency(model, X, name: str, repeats: int = 100) -> dict:
    repeats = int(repeats)
    metrics = {
        "name": name,
        "repeats": repeats,
        "single_window_ms": None,
    }
    X = np.asarray(X, dtype=np.float32)
    if repeats <= 0 or X.shape[0] == 0:
        return metrics

    sample = X[:1]
    model.predict_confidence(sample)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict_confidence(sample)
    elapsed = time.perf_counter() - start
    single_window_ms = (elapsed / repeats) * 1000.0
    metrics["single_window_ms"] = float(single_window_ms)
    print(f"[{name}] predict latency: {single_window_ms:.4f} ms/window ({repeats} repeats)")
    return metrics
