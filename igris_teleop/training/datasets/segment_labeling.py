from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import numpy as np


LABEL_FEATURE_KEY = "annotation.segment_label_id"
LABEL_FEATURE_SPEC = {
    "dtype": "int64",
    "shape": [1],
    "names": "segment_label_id",
}
LABEL_SIDECAR_NAME = "annotation_segment_labels.json"
UNASSIGNED_CLASS_ID = -1
DEFAULT_LABEL_CLASSES = (
    {"id": 0, "name": "success"},
    {"id": 1, "name": "failure"},
    {"id": 2, "name": "recovery"},
)


def get_labeling_backend_status() -> tuple[bool, str]:
    missing: list[str] = []

    try:
        import pandas  # noqa: F401
    except Exception:
        missing.append("pandas")

    try:
        import pyarrow.parquet  # noqa: F401
    except Exception:
        missing.append("pyarrow")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    except Exception:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
        except Exception:
            missing.append("lerobot")

    if missing:
        deps = ", ".join(missing)
        return False, f"Label saving requires: {deps}"
    return True, "Label saving is available."


def default_label_classes() -> list[dict[str, Any]]:
    return [dict(item) for item in DEFAULT_LABEL_CLASSES]


def label_name_for_id(classes: Iterable[Mapping[str, Any]], class_id: int | None) -> str:
    if class_id is None or int(class_id) < 0:
        return "unassigned"
    target = int(class_id)
    for item in classes:
        try:
            if int(item.get("id", -1)) == target:
                name = str(item.get("name", "")).strip()
                if name:
                    return name
        except Exception:
            continue
    return f"class_{target}"


def normalize_classes(raw_classes: Any) -> list[dict[str, Any]]:
    classes: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    if isinstance(raw_classes, list):
        for item in raw_classes:
            if not isinstance(item, Mapping):
                continue
            try:
                class_id = int(item.get("id", -1))
            except Exception:
                continue
            if class_id < 0 or class_id in seen_ids:
                continue
            name = str(item.get("name", "")).strip() or f"class_{class_id}"
            classes.append({"id": class_id, "name": name})
            seen_ids.add(class_id)

    if not classes:
        classes = default_label_classes()

    classes.sort(key=lambda item: int(item["id"]))
    return classes


def ensure_classes_cover_ids(
    classes: Iterable[Mapping[str, Any]],
    class_ids: Iterable[int | np.integer],
) -> list[dict[str, Any]]:
    out = normalize_classes(list(classes))
    seen_ids = {int(item["id"]) for item in out}
    for raw_id in class_ids:
        class_id = int(raw_id)
        if class_id < 0 or class_id in seen_ids:
            continue
        out.append({"id": class_id, "name": f"class_{class_id}"})
        seen_ids.add(class_id)
    out.sort(key=lambda item: int(item["id"]))
    return out


def _normalize_class_id(value: Any) -> int:
    try:
        class_id = int(value)
    except Exception:
        return UNASSIGNED_CLASS_ID
    if class_id < 0:
        return UNASSIGNED_CLASS_ID
    return class_id


def segments_from_label_ids(frame_numbers: np.ndarray, label_ids: np.ndarray) -> list[dict[str, int]]:
    frames = np.asarray(frame_numbers, dtype=np.int64).reshape(-1)
    labels = np.asarray(label_ids, dtype=np.int64).reshape(-1)
    if frames.size == 0:
        return []
    if labels.size != frames.size:
        raise ValueError("frame_numbers and label_ids size mismatch")

    segments: list[dict[str, int]] = []
    start_idx = 0
    current_label = int(labels[0])
    for idx in range(1, frames.size):
        next_label = int(labels[idx])
        if next_label == current_label:
            continue
        segments.append(
            {
                "start_frame": int(frames[start_idx]),
                "end_frame": int(frames[idx - 1]),
                "class_id": int(current_label),
            }
        )
        start_idx = idx
        current_label = next_label

    segments.append(
        {
            "start_frame": int(frames[start_idx]),
            "end_frame": int(frames[-1]),
            "class_id": int(current_label),
        }
    )
    return segments


def default_segments_for_episode(frame_numbers: np.ndarray) -> list[dict[str, int]]:
    frames = np.asarray(frame_numbers, dtype=np.int64).reshape(-1)
    if frames.size == 0:
        return []
    label_ids = np.full((frames.size,), UNASSIGNED_CLASS_ID, dtype=np.int64)
    return segments_from_label_ids(frames, label_ids)


def segments_to_label_ids(
    segments: Iterable[Mapping[str, Any]],
    frame_numbers: np.ndarray,
) -> np.ndarray:
    frames = np.asarray(frame_numbers, dtype=np.int64).reshape(-1)
    label_ids = np.full((frames.size,), UNASSIGNED_CLASS_ID, dtype=np.int64)
    if frames.size == 0:
        return label_ids

    for raw_segment in segments:
        if not isinstance(raw_segment, Mapping):
            continue
        try:
            start_frame = int(raw_segment.get("start_frame", frames[0]))
            end_frame = int(raw_segment.get("end_frame", frames[-1]))
        except Exception:
            continue
        if end_frame < start_frame:
            continue
        class_id = _normalize_class_id(raw_segment.get("class_id", UNASSIGNED_CLASS_ID))
        mask = (frames >= start_frame) & (frames <= end_frame)
        if np.any(mask):
            label_ids[mask] = class_id

    return label_ids


def normalize_episode_segments(
    raw_segments: Any,
    frame_numbers: np.ndarray,
) -> list[dict[str, int]]:
    frames = np.asarray(frame_numbers, dtype=np.int64).reshape(-1)
    if frames.size == 0:
        return []
    if not isinstance(raw_segments, list) or not raw_segments:
        return default_segments_for_episode(frames)
    label_ids = segments_to_label_ids(raw_segments, frames)
    return segments_from_label_ids(frames, label_ids)


def normalize_boundary_segments(
    raw_segments: Any,
    frame_numbers: np.ndarray,
) -> list[dict[str, int]]:
    frames = np.asarray(frame_numbers, dtype=np.int64).reshape(-1)
    if frames.size == 0:
        return []
    if not isinstance(raw_segments, list) or not raw_segments:
        return default_segments_for_episode(frames)

    parsed_segments: list[tuple[int, int, int]] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, Mapping):
            continue
        try:
            start_frame = int(raw_segment.get("start_frame", frames[0]))
            end_frame = int(raw_segment.get("end_frame", frames[-1]))
        except Exception:
            continue
        if end_frame < start_frame:
            continue
        mask = (frames >= start_frame) & (frames <= end_frame)
        idxs = np.flatnonzero(mask)
        if idxs.size <= 0:
            continue
        parsed_segments.append(
            (
                int(idxs[0]),
                int(idxs[-1]),
                _normalize_class_id(raw_segment.get("class_id", UNASSIGNED_CLASS_ID)),
            )
        )

    if not parsed_segments:
        return default_segments_for_episode(frames)

    parsed_segments.sort(key=lambda item: (item[0], item[1]))
    segments: list[dict[str, int]] = []
    cursor = 0
    for start_idx, end_idx, class_id in parsed_segments:
        if end_idx < cursor:
            continue
        start_idx = max(start_idx, cursor)
        if start_idx > cursor:
            segments.append(
                {
                    "start_frame": int(frames[cursor]),
                    "end_frame": int(frames[start_idx - 1]),
                    "class_id": UNASSIGNED_CLASS_ID,
                }
            )
        segments.append(
            {
                "start_frame": int(frames[start_idx]),
                "end_frame": int(frames[end_idx]),
                "class_id": int(class_id),
            }
        )
        cursor = end_idx + 1

    if cursor < frames.size:
        segments.append(
            {
                "start_frame": int(frames[cursor]),
                "end_frame": int(frames[-1]),
                "class_id": UNASSIGNED_CLASS_ID,
            }
        )

    return segments


def episode_has_unassigned_segments(segments: Iterable[Mapping[str, Any]]) -> bool:
    for raw_segment in segments:
        if not isinstance(raw_segment, Mapping):
            continue
        if _normalize_class_id(raw_segment.get("class_id", UNASSIGNED_CLASS_ID)) < 0:
            return True
    return False


def load_annotation_sidecar(dataset_dir: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_dir).expanduser().resolve()
    sidecar_path = dataset_path / "meta" / LABEL_SIDECAR_NAME
    payload: dict[str, Any] = {
        "feature_key": LABEL_FEATURE_KEY,
        "classes": default_label_classes(),
        "episodes": {},
    }
    if not sidecar_path.is_file():
        return payload

    with sidecar_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return payload

    payload["feature_key"] = str(raw.get("feature_key", LABEL_FEATURE_KEY)).strip() or LABEL_FEATURE_KEY
    payload["classes"] = normalize_classes(raw.get("classes"))

    raw_episodes = raw.get("episodes", {})
    episodes: dict[int, list[dict[str, int]]] = {}
    if isinstance(raw_episodes, Mapping):
        for raw_ep, raw_segments in raw_episodes.items():
            try:
                episode_idx = int(raw_ep)
            except Exception:
                continue
            if not isinstance(raw_segments, list):
                continue
            segments: list[dict[str, int]] = []
            for segment in raw_segments:
                if not isinstance(segment, Mapping):
                    continue
                try:
                    start_frame = int(segment.get("start_frame"))
                    end_frame = int(segment.get("end_frame"))
                except Exception:
                    continue
                if end_frame < start_frame:
                    continue
                segments.append(
                    {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "class_id": _normalize_class_id(segment.get("class_id", UNASSIGNED_CLASS_ID)),
                    }
                )
            if segments:
                episodes[episode_idx] = segments

    payload["episodes"] = episodes
    return payload


def serialize_annotation_sidecar(
    *,
    classes: Iterable[Mapping[str, Any]],
    episodes: Mapping[int, Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    out_episodes: dict[str, list[dict[str, int]]] = {}
    for raw_episode, raw_segments in sorted(episodes.items(), key=lambda item: int(item[0])):
        episode_idx = int(raw_episode)
        serialized_segments: list[dict[str, int]] = []
        for raw_segment in raw_segments:
            if not isinstance(raw_segment, Mapping):
                continue
            serialized_segments.append(
                {
                    "start_frame": int(raw_segment.get("start_frame", 0)),
                    "end_frame": int(raw_segment.get("end_frame", 0)),
                    "class_id": _normalize_class_id(raw_segment.get("class_id", UNASSIGNED_CLASS_ID)),
                }
            )
        if serialized_segments:
            out_episodes[str(episode_idx)] = serialized_segments

    return {
        "feature_key": LABEL_FEATURE_KEY,
        "classes": normalize_classes(list(classes)),
        "episodes": out_episodes,
    }


def _compute_scalar_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        arr = np.asarray([UNASSIGNED_CLASS_ID], dtype=np.float64)
        count = 0
    else:
        count = int(arr.size)

    quantiles = np.quantile(arr, [0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        "min": [int(np.min(arr))],
        "max": [int(np.max(arr))],
        "mean": [float(np.mean(arr))],
        "std": [float(np.std(arr))],
        "count": [count],
        "q01": [float(quantiles[0])],
        "q10": [float(quantiles[1])],
        "q50": [float(quantiles[2])],
        "q90": [float(quantiles[3])],
        "q99": [float(quantiles[4])],
    }


def _timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")


def _find_existing_backup_dir(dataset_path: Path) -> Path | None:
    backup_candidates = sorted(
        path
        for path in dataset_path.parent.glob(f"{dataset_path.name}__label_backup_*")
        if path.is_dir()
    )
    if not backup_candidates:
        return None
    return backup_candidates[0]


def rewrite_dataset_segment_labels_in_place(
    dataset_dir: str | Path,
    *,
    classes: Iterable[Mapping[str, Any]],
    episode_segments: Mapping[int, Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    ready, message = get_labeling_backend_status()
    if not ready:
        raise RuntimeError(message)

    import pandas as pd

    dataset_path = Path(dataset_dir).expanduser().resolve()
    data_dir = dataset_path / "data"
    meta_dir = dataset_path / "meta"
    info_path = meta_dir / "info.json"
    stats_path = meta_dir / "stats.json"
    sidecar_path = meta_dir / LABEL_SIDECAR_NAME

    if not info_path.is_file():
        raise FileNotFoundError(f"Missing info.json: {info_path}")

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    with info_path.open("r", encoding="utf-8") as f:
        info_payload = json.load(f)
    if not isinstance(info_payload, dict):
        raise ValueError(f"Invalid info payload: {info_path}")

    if stats_path.is_file():
        with stats_path.open("r", encoding="utf-8") as f:
            stats_payload = json.load(f)
        if not isinstance(stats_payload, dict):
            stats_payload = {}
    else:
        stats_payload = {}

    normalized_classes = normalize_classes(list(classes))
    normalized_segments = {
        int(raw_episode): [
            {
                "start_frame": int(segment.get("start_frame", 0)),
                "end_frame": int(segment.get("end_frame", 0)),
                "class_id": _normalize_class_id(segment.get("class_id", UNASSIGNED_CLASS_ID)),
            }
            for segment in raw_segments
            if isinstance(segment, Mapping)
        ]
        for raw_episode, raw_segments in episode_segments.items()
    }

    temp_root = dataset_path.parent / f".{dataset_path.name}__label_tmp_{_timestamp()}"
    backup_root = _find_existing_backup_dir(dataset_path)
    create_backup = backup_root is None
    if backup_root is None:
        backup_root = dataset_path.parent / f"{dataset_path.name}__label_backup_{_timestamp()}"
    label_value_chunks: list[np.ndarray] = []
    replaced_rel_paths: list[Path] = []

    try:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        temp_root.mkdir(parents=True, exist_ok=False)

        backup_targets = [info_path, stats_path]
        if sidecar_path.exists():
            backup_targets.append(sidecar_path)
        backup_targets.extend(parquet_files)

        if create_backup:
            for source_path in backup_targets:
                if not source_path.exists():
                    continue
                rel_path = source_path.relative_to(dataset_path)
                backup_path = backup_root / rel_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, backup_path)

        for source_path in parquet_files:
            df = pd.read_parquet(source_path).reset_index(drop=True)
            if "episode_index" not in df.columns or "frame_index" not in df.columns:
                raise KeyError(f"{source_path} is missing episode_index/frame_index columns")

            episode_index = df["episode_index"].to_numpy(dtype=np.int64, copy=False)
            frame_index = df["frame_index"].to_numpy(dtype=np.int64, copy=False)
            if LABEL_FEATURE_KEY in df.columns:
                label_values = df[LABEL_FEATURE_KEY].to_numpy(dtype=np.int64, copy=True)
            else:
                label_values = np.full((len(df),), UNASSIGNED_CLASS_ID, dtype=np.int64)

            for episode_idx in np.unique(episode_index):
                episode_segments_for_idx = normalized_segments.get(int(episode_idx))
                if episode_segments_for_idx is None:
                    continue
                mask = episode_index == int(episode_idx)
                label_values[mask] = segments_to_label_ids(episode_segments_for_idx, frame_index[mask])

            df[LABEL_FEATURE_KEY] = label_values.astype(np.int64, copy=False)
            label_value_chunks.append(np.asarray(label_values, dtype=np.int64))

            rel_path = source_path.relative_to(dataset_path)
            temp_path = temp_root / rel_path
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(temp_path, index=False)
            replaced_rel_paths.append(rel_path)

        all_label_values = np.concatenate(label_value_chunks, axis=0) if label_value_chunks else np.zeros((0,), dtype=np.int64)
        features = info_payload.get("features", {})
        if not isinstance(features, dict):
            features = {}
        info_payload["features"] = dict(features)
        info_payload["features"][LABEL_FEATURE_KEY] = dict(LABEL_FEATURE_SPEC)

        stats_payload[LABEL_FEATURE_KEY] = _compute_scalar_stats(all_label_values)
        sidecar_payload = serialize_annotation_sidecar(
            classes=normalized_classes,
            episodes=normalized_segments,
        )

        temp_info_path = temp_root / "meta" / "info.json"
        temp_stats_path = temp_root / "meta" / "stats.json"
        temp_sidecar_path = temp_root / "meta" / LABEL_SIDECAR_NAME
        temp_info_path.parent.mkdir(parents=True, exist_ok=True)

        temp_info_path.write_text(json.dumps(info_payload, indent=4), encoding="utf-8")
        temp_stats_path.write_text(json.dumps(stats_payload, indent=4), encoding="utf-8")
        temp_sidecar_path.write_text(json.dumps(sidecar_payload, indent=2), encoding="utf-8")
        replaced_rel_paths.extend(
            [
                Path("meta") / "info.json",
                Path("meta") / "stats.json",
                Path("meta") / LABEL_SIDECAR_NAME,
            ]
        )

        for rel_path in replaced_rel_paths:
            source_path = temp_root / rel_path
            dest_path = dataset_path / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.replace(dest_path)

    except Exception:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
        raise
    else:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)

    return {
        "dataset_dir": str(dataset_path),
        "backup_dir": str(backup_root),
        "feature_key": LABEL_FEATURE_KEY,
        "rewritten_files": len(replaced_rel_paths),
    }
