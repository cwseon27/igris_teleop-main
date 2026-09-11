import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections.abc import Callable

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.utils import DATA_DIR, DEFAULT_DATA_PATH
from lerobot.datasets.dataset_tools import _copy_videos, _write_parquet, _copy_episodes_metadata_and_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from igris_teleop.core.project_paths import DATASETS_ROOT

DEFAULT_DATASET_PATH = DATASETS_ROOT


def _apply_custom_funcion(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    custom_function: Callable,
    modify_feature: str,
) -> None:
    """Copy data while adding or removing features."""
    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    frame_idx = 0

    for src_path in tqdm(parquet_files, desc="Processing data files"):
        df = pd.read_parquet(src_path).reset_index(drop=True)

        relative_path = src_path.relative_to(dataset.root)
        chunk_dir = relative_path.parts[1]
        file_name = relative_path.parts[2]

        chunk_idx = int(chunk_dir.split("-")[1])
        file_idx = int(file_name.split("-")[1].split(".")[0])

        end_idx = frame_idx + len(df)
        
        print(new_meta.features.items())
        
        try:
            for feature_name, (values, _) in new_meta.features.items():
                if modify_feature == feature_name:
                    if callable(values):
                        feature_values = []
                        for _, row in df.iterrows():
                            ep_idx = row["episode_index"]
                            frame_in_ep = row["frame_index"]
                            value = values(row.to_dict(), ep_idx, frame_in_ep)
                            if isinstance(value, np.ndarray) and value.size == 1:
                                value = value.item()
                                value = custom_function(value)
                            feature_values.append(value)
                        df[feature_name] = feature_values
                else:
                    feature_slice = values[frame_idx:end_idx]
                    if len(feature_slice.shape) > 1 and feature_slice.shape[1] == 1:
                        df[feature_name] = feature_slice.flatten()
                    else:
                        df[feature_name] = feature_slice
        except Exception as e:
            raise RuntimeError(
                f"Error processing feature '{type(modify_feature)}' {type(feature_name)}in file '{src_path}': {e}"
            ) from e
                    
        frame_idx = end_idx

        # Write using the same chunk/file structure as source
        dst_path = new_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        _write_parquet(df, dst_path, new_meta)

    _copy_episodes_metadata_and_stats(dataset, new_meta)

def modify_features(
    dataset: LeRobotDataset,
    custom_function: Callable,
    modify_feature: dict[str, tuple[np.ndarray | torch.Tensor | Callable, dict]],
    output_folder_name: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Modify features of a LeRobotDataset."""
    
    if modify_feature is None:
        raise ValueError("Must specify at least one of modify_feature")

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"
        
    base_root = (DEFAULT_DATASET_PATH.parent / Path(DEFAULT_DATASET_PATH.name).expanduser()).resolve()
    ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    output_dir = base_root / dataset.root.split('/')[-1] / f"{output_folder_name}_{ts}"
    print(dataset.root, "!!!!!!!!!!!")

    new_features = dataset.meta.features.copy()
    
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir
    )

    _apply_custom_funcion(
        dataset=dataset,
        new_meta=new_meta,
        custom_function=custom_function,
        modify_feature=modify_feature
    )

    if new_meta.video_keys:
        _copy_videos(dataset, new_meta)

    new_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

    return new_dataset

if __name__ == "__main__":
    
    dataset_folder_name = "IGRIS_C_20251230_165442_insert_test"
    base_root = (DEFAULT_DATASET_PATH.parent / Path(DEFAULT_DATASET_PATH.name).expanduser()).resolve()
    dataset_root = Path(os.path.join(base_root, dataset_folder_name))

    dataset = LeRobotDataset(dataset_root)

    import cv2
    
    def custom_func(value):
        return value * 2

    modify_feature = "observation.image.realsense_head"

    new_dataset = modify_features(
        dataset,
        custom_function=custom_func,
        modify_feature=modify_feature,
        output_folder_name="modified_dataset",
        repo_id="example_dataset_modified"
    )

    print(f"Modified dataset created at {new_dataset.root}")

