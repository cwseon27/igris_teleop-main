# 변경사항
# igris_teleop/config/robot_control/init_setting.yaml 파일에 데이터셋 초기 상태 요약 정보 저장하게 추가함
# worker_control에서 init_setting.yaml 파일 읽어서 데이터셋 초기 상태 정보 활용할 수 있도록 수정함

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from igris_teleop.core.project_paths import DATASETS_ROOT, INIT_SETTING_PATH


DEFAULT_DATASET_ROOT = DATASETS_ROOT
DEFAULT_DATASET_DIR = (
    DEFAULT_DATASET_ROOT
    / "0304_실증과제_dataset_train"
    / "0304_lifting"
).resolve()
DEFAULT_OUTPUT_YAML = INIT_SETTING_PATH

HAND_LABELS = [f"hand_{idx}" for idx in range(12)]
ARM_LABELS = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow_pitch",
    "l_wrist_yaw",
    "l_wrist_roll",
    "l_wrist_pitch",
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow_pitch",
    "r_wrist_yaw",
    "r_wrist_roll",
    "r_wrist_pitch",
]
NECK_LABELS = ["neck_yaw", "neck_pitch"]
WAIST_LABELS = ["waist_yaw", "waist_roll", "waist_pitch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the initial state of each episode in a LeRobot dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset root containing data/ and meta/ directories.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="observation.state",
        choices=("observation.state", "action", "observation.torque"),
        help="Feature to summarize from the first frame of each episode.",
    )
    parser.add_argument(
        "--stat",
        type=str,
        default="mean",
        choices=("min", "max", "mean", "std"),
        help="Primary statistic to print per dimension.",
    )
    parser.add_argument(
        "--show-all-stats",
        action="store_true",
        help="Print min/max/mean/std for every dimension instead of only the selected stat.",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=DEFAULT_OUTPUT_YAML,
        help="YAML file used to persist dataset init-state summaries.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing the selected statistic into the YAML file.",
    )
    return parser.parse_args()


def load_dataset_rows(dataset_dir: Path, feature: str):
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required to run this script.") from exc

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to read the dataset parquet files.") from exc

    parquets = sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}")

    required_columns = ["episode_index", "frame_index", "timestamp", feature]
    frames = [pq.read_table(path, columns=required_columns).to_pandas() for path in parquets]
    df = pd.concat(frames, ignore_index=True)
    return df


def read_expected_feature_dim(dataset_dir: Path, feature: str) -> int | None:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return None

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    feature_info = info.get("features", {}).get(feature)
    if not feature_info:
        return None

    shape = feature_info.get("shape")
    if not shape:
        return None

    first_dim = int(shape[0])
    return first_dim


def select_first_episode_rows(df):
    df_sorted = df.sort_values(["episode_index", "frame_index", "timestamp"])
    return df_sorted.groupby("episode_index", as_index=False).first()


def stack_feature_vectors(first_rows, feature: str) -> np.ndarray:
    vectors = [np.asarray(value, dtype=np.float64).reshape(-1) for value in first_rows[feature].to_numpy()]
    if not vectors:
        raise ValueError(f"No rows found for feature {feature}")

    dims = {vec.size for vec in vectors}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent feature dimensions detected: {sorted(dims)}")

    return np.stack(vectors, axis=0)


def resolve_feature_labels(feature: str, dim: int) -> list[str]:
    if feature in ("observation.state", "action"):
        if dim == 28:
            return HAND_LABELS + ARM_LABELS + NECK_LABELS
        if dim == 31:
            return HAND_LABELS + ARM_LABELS + NECK_LABELS + WAIST_LABELS

    if feature == "observation.torque" and dim == 14:
        return ARM_LABELS.copy()

    return [f"{feature}[{idx}]" for idx in range(dim)]


def compute_stats(values: np.ndarray) -> dict[str, np.ndarray | int]:
    return {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": int(values.shape[0]),
    }


def resolve_dataset_name(dataset_dir: Path, dataset_root: Path) -> str:
    try:
        return dataset_dir.relative_to(dataset_root).as_posix()
    except ValueError:
        return dataset_dir.as_posix()


def _normalize_payload(payload: object) -> dict[str, object]:
    if payload is None:
        return {"datasets": {}}
    if not isinstance(payload, dict):
        raise ValueError("init_setting.yaml must contain a mapping at the top level.")

    datasets = payload.get("datasets")
    if datasets is None:
        payload["datasets"] = {}
    elif not isinstance(datasets, dict):
        raise ValueError("'datasets' in init_setting.yaml must be a mapping.")
    return payload


def save_dataset_summary(
    output_yaml: Path,
    dataset_name: str,
    dataset_dir: Path,
    feature: str,
    stat_name: str,
    labels: list[str],
    stats: dict[str, np.ndarray | int],
    rows_loaded: int,
) -> None:
    joint_values = [round(float(value), 9) for value in np.asarray(stats[stat_name], dtype=np.float64)]

    payload: dict[str, object]
    if output_yaml.exists():
        with output_yaml.open("r", encoding="utf-8") as f:
            payload = _normalize_payload(yaml.safe_load(f))
    else:
        payload = {"datasets": {}}

    datasets = payload["datasets"]
    assert isinstance(datasets, dict)
    datasets[dataset_name] = {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "feature": feature,
        "stat": stat_name,
        "episodes": int(stats["count"]),
        "feature_dim": len(labels),
        "rows_loaded": int(rows_loaded),
        "joint_names": labels,
        "joint_values": joint_values,
    }

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def print_selected_stat(labels: list[str], stats: dict[str, np.ndarray | int], stat_name: str) -> None:
    values = np.asarray(stats[stat_name], dtype=np.float64)
    for label, value in zip(labels, values):
        print(f"{label}: {value:.9f}")


def print_all_stats(labels: list[str], stats: dict[str, np.ndarray | int]) -> None:
    mins = np.asarray(stats["min"], dtype=np.float64)
    maxs = np.asarray(stats["max"], dtype=np.float64)
    means = np.asarray(stats["mean"], dtype=np.float64)
    stds = np.asarray(stats["std"], dtype=np.float64)

    for idx, label in enumerate(labels):
        print(
            f"{idx:02d} {label}: "
            f"min={mins[idx]:.9f}, "
            f"max={maxs[idx]:.9f}, "
            f"mean={means[idx]:.9f}, "
            f"std={stds[idx]:.9f}"
        )


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_yaml = args.output_yaml.expanduser().resolve()

    df = load_dataset_rows(dataset_dir, args.feature)
    first_rows = select_first_episode_rows(df)
    init_values = stack_feature_vectors(first_rows, args.feature)

    expected_dim = read_expected_feature_dim(dataset_dir, args.feature)
    actual_dim = int(init_values.shape[1])
    if expected_dim is not None and expected_dim != actual_dim:
        raise ValueError(
            f"Feature dimension mismatch for {args.feature}: expected {expected_dim}, got {actual_dim}"
        )

    labels = resolve_feature_labels(args.feature, actual_dim)
    stats = compute_stats(init_values)
    dataset_name = resolve_dataset_name(dataset_dir, DEFAULT_DATASET_ROOT)

    print(f"dataset_dir: {dataset_dir}")
    print(f"dataset_name: {dataset_name}")
    print(f"feature: {args.feature}")
    print(f"episodes: {stats['count']}")
    print(f"feature_dim: {actual_dim}")
    print(f"rows_loaded: {len(df)}")
    print("")

    if args.show_all_stats:
        print_all_stats(labels, stats)
    else:
        print_selected_stat(labels, stats, args.stat)

    if not args.no_save:
        save_dataset_summary(
            output_yaml=output_yaml,
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            feature=args.feature,
            stat_name=args.stat,
            labels=labels,
            stats=stats,
            rows_loaded=len(df),
        )
        print("")
        print(f"saved_to: {output_yaml}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
