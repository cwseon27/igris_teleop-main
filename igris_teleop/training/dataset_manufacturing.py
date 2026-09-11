import os
import cv2
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections.abc import Callable
from torchcodec.decoders import VideoDecoder

from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_DATA_PATH,
    load_episodes,
    write_stats,
    write_info
)
from lerobot.datasets.video_utils import (
    get_video_info
)
from lerobot.datasets.dataset_tools import (
    _copy_videos,
    _write_parquet,
    _copy_episodes_metadata_and_stats,
)
from lerobot.datasets.compute_stats import (
    compute_episode_stats,
    aggregate_stats,
)
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from igris_teleop.core.project_paths import DATASETS_ROOT

DEFAULT_DATASET_PATH = DATASETS_ROOT

def modify_vector_feature_only(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    custom_function: Callable,
    modify_feature: str,
):
    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    for src_path in tqdm(parquet_files, desc=f"Modify vector: {modify_feature}"):
        df = pd.read_parquet(src_path).reset_index(drop=True)

        if modify_feature not in df.columns:
            raise KeyError(f"{modify_feature} not in parquet")

        df[modify_feature] = df[modify_feature].apply(custom_function)

        rel = src_path.relative_to(dataset.root)
        chunk_idx = int(rel.parts[1].split("-")[1])
        file_idx  = int(rel.parts[2].split("-")[1].split(".")[0])

        dst_path = new_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(df, dst_path, new_meta)

    _copy_episodes_metadata_and_stats(dataset, new_meta)
    recompute_stats_for_modified_feature(dataset, new_meta, new_meta.root, modify_feature)

def recompute_stats_for_modified_feature(
    src_dataset,
    dst_meta,
    dst_root: Path,
    feature_key: str,
):
    if dst_meta.episodes is None:
        dst_meta.episodes = load_episodes(dst_root)

    all_episode_stats = []

    for ep_idx in tqdm(range(dst_meta.total_episodes), desc=f"Stats {feature_key}"):
        ep = dst_meta.episodes[ep_idx]
        chunk_idx = ep["data/chunk_index"]
        file_idx = ep["data/file_index"]

        data_path = dst_root / DEFAULT_DATA_PATH.format(
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        df = pd.read_parquet(data_path)

        ep_df = df[df["episode_index"] == ep_idx]
        arr = np.stack(ep_df[feature_key].to_list(), axis=0)

        ep_stats = compute_episode_stats(
            {feature_key: arr},
            dst_meta.features,
        )[feature_key]

        all_episode_stats.append({feature_key: ep_stats})

    aggregated = aggregate_stats(all_episode_stats)

    new_stats = dict(src_dataset.meta.stats or {})
    new_stats[feature_key] = aggregated[feature_key]
    write_stats(new_stats, dst_root)

def rewrite_video_file_ffmpeg(
    src_path: Path,
    dst_path: Path,
    vf: str,
):
    """
    Rewrite a video file using ffmpeg while preserving the original temporal structure.

    This function applies spatial video filters (e.g., crop, scale, flip) to an
    existing video file **without trimming or changing episode boundaries**.
    It is designed for datasets where videos are stored as large chunked MP4 files
    (e.g., ~200MB per file) and episodes reference subranges via timestamps.

    The function:
      - Processes the **entire input video file** (no -ss / -to trimming)
      - Applies the given ffmpeg video filter string (`vf`)
      - Preserves frame ordering and timing (no FPS resampling)
      - Writes a re-encoded MP4 file at `dst_path`

    Parameters
    ----------
    src_path : pathlib.Path
        Path to the source video file (chunk MP4).
        This is typically shared by multiple episodes.

    dst_path : pathlib.Path
        Output path for the rewritten video file.
        The directory will be created if it does not exist.

    vf : str
        ffmpeg video filter string applied to the entire video.
        Multiple filters can be chained using commas.

        Commonly used filters include:

        1) Crop
           Extract a rectangular region from each frame.

           Format:
               "crop=w:h:x:y"

           Example:
               "crop=400:240:550:480"

           Meaning:
               - w, h : width and height of the crop
               - x, y : top-left corner of the crop region

        2) Scale / Resize
           Resize frames to a fixed resolution.

           Format:
               "scale=width:height"

           Example:
               "scale=640:480"

        3) Horizontal / Vertical Flip
           Flip frames along an axis.

           Horizontal flip:
               "hflip"

           Vertical flip:
               "vflip"

        4) Crop + Scale (chained)
           Filters are applied left-to-right.

           Example:
               "crop=400:240:550:480,scale=640:480"

        5) Transpose (rotate)
           Rotate frames in 90-degree increments.

           Examples:
               "transpose=1"  # 90° clockwise
               "transpose=2"  # 90° counter-clockwise

        6) Padding
           Add borders to reach a target resolution.

           Format:
               "pad=width:height:x:y:color"

           Example:
               "pad=640:480:0:0:black"

        7) Color / format adjustments (advanced)
           Typically not needed for robot datasets, but supported:

               "format=yuv420p"
               "eq=brightness=0.05:contrast=1.2"

        Notes on `vf` usage:
        --------------------
        - Do NOT include FPS filters (e.g., "fps=30") unless you explicitly want
          to resample frames. Frame count preservation is usually desired.
        - Do NOT include trimming filters (e.g., "trim=start:end").
          Episode boundaries are handled by metadata, not by video slicing.
        - If `vf` is an empty string or None, the video will be re-encoded as-is.

    Behavior and Assumptions
    ------------------------
    - The original FPS and frame count are preserved (no temporal resampling).
    - Audio streams are removed (-an).
    - Output is encoded using H.264 (libx264) with:
        - preset: veryfast
        - crf: 18 (high visual quality)
        - pixel format: yuv420p (broad compatibility)

    Typical Use Cases
    -----------------
    - Cropping stereo camera images while keeping episode timestamps intact
    - Resizing RGB streams for training efficiency
    - Flipping images for data augmentation at the video level
    - Normalizing video resolution across datasets

    This function is intended to be used at the **video-chunk level**, not per episode.
    """
    
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",     # 에러 메시지만 출력
        "-i", str(src_path),      # 입력 파일
        "-vf", vf,                # 비디오 필터
        "-an",                    # 오디오 없음
        "-fps_mode", "passthrough",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(dst_path),
    ]

    subprocess.run(cmd, check=True)

def rewrite_dataset_video_ffmpeg(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    video_key: str,
    vf: str,
):
    unique_rels = []
    seen = set()
    for ep_idx in range(dataset.meta.total_episodes):
        rel = dataset.meta.get_video_file_path(ep_idx, video_key)
        if rel not in seen:
            seen.add(rel)
            unique_rels.append(rel)

    for rel in tqdm(unique_rels, desc=f"ffmpeg rewrite chunks {video_key}"):
        src_path = dataset.root / rel
        dst_path = new_meta.root / rel
        rewrite_video_file_ffmpeg(src_path, dst_path, vf=vf)

def modify_video_feature_only_ffmpeg(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    modify_video_feature: str,
    vf: str,
):
    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.root)

    rewrite_dataset_video_ffmpeg(
        dataset=dataset,
        new_meta=new_meta,
        video_key=modify_video_feature,
        vf=vf,
    )
    
    _copy_episodes_metadata_and_stats(dataset, new_meta)

    video_path = new_meta.root / new_meta.video_path.format(video_key=modify_video_feature, chunk_index=0, file_index=0)
    new_meta.info["features"][modify_video_feature]["info"] = get_video_info(video_path)
    
    feat = new_meta.info["features"][modify_video_feature]
    shape = list(feat["shape"])
    shape[0] = int(feat["info"]["video.height"])
    shape[1] = int(feat["info"]["video.width"])
    feat["shape"] = tuple(shape)

    write_info(new_meta.info, new_meta.root)

    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    for src_path in tqdm(parquet_files, desc=f"Modify vector: {modify_video_feature}"):
        df = pd.read_parquet(src_path).reset_index(drop=True)

        rel = src_path.relative_to(dataset.root)
        chunk_idx = int(rel.parts[1].split("-")[1])
        file_idx  = int(rel.parts[2].split("-")[1].split(".")[0])

        dst_path = new_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(df, dst_path, new_meta)

    _copy_videos(dataset, new_meta, exclude_keys=[modify_video_feature])

def modify_features(
    dataset: LeRobotDataset,
    modify_feature: str,
    custom_function,
    output_folder_name: str,
    repo_id: str | None = None,
) -> LeRobotDataset:

    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"

    ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    output_dir = DEFAULT_DATASET_PATH / f"{dataset.root.name}_{output_folder_name}_{ts}"

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=dataset.meta.features.copy(),
        robot_type=dataset.meta.robot_type,
        root=output_dir,
    )

    if modify_feature in new_meta.video_keys:
        _copy_videos(dataset, new_meta, exclude_keys=[modify_feature])
        modify_video_feature_only_ffmpeg(
            dataset,
            new_meta,
            modify_feature,
            vf=custom_function,  # string
        )
    else:
        modify_vector_feature_only(
            dataset,
            new_meta,
            custom_function,
            modify_feature,
        )
        _copy_videos(dataset, new_meta)

    return LeRobotDataset(
        repo_id=repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )

def check_episode_frame_index_safety(
    dataset: LeRobotDataset,
    video_keys: list[str] | None = None,
):
    """
    ✅ OK	       절대 문제 없음
    ⚠️ BORDERLINE	지금은 OK지만 rounding / tolerance에 따라 터질 수 있음
    ❌ INVALID	   torchcodec에서 반드시 터짐
    """
    fps = float(dataset.meta.fps)

    if video_keys is None:
        video_keys = dataset.meta.video_keys

    # 모든 parquet 로드 (timestamp 기준)
    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise RuntimeError("No parquet files found")

    df_all = pd.concat(
        [pd.read_parquet(p) for p in parquet_files],
        ignore_index=True,
    )

    print(f"\n📊 FPS = {fps}")
    print(f"🎥 Video keys = {video_keys}")
    print("=" * 80)

    problems = []

    for ep_idx, ep in tqdm(
        enumerate(dataset.meta.episodes),
        total=dataset.meta.total_episodes,
        desc="Checking episodes",
    ):
        ep_df = df_all[df_all["episode_index"] == ep_idx]
        if ep_df.empty:
            continue

        ts_values = ep_df["timestamp"].to_numpy()
        requested_max = max(int(round(ts * fps)) for ts in ts_values)

        for vk in video_keys:
            video_rel = dataset.meta.get_video_file_path(ep_idx, vk)
            video_path = dataset.root / video_rel

            if not video_path.exists():
                print(f"⚠️ missing video: {video_path}")
                continue

            decoder = VideoDecoder(str(video_path))
            frame_count = decoder._num_frames

            status = "OK"
            if requested_max >= frame_count:
                status = "❌ INVALID"
                problems.append((ep_idx, vk, requested_max, frame_count))
            elif requested_max == frame_count - 1:
                status = "⚠️ BORDERLINE"

            print(
                f"[EP {ep_idx:03d}] {vk:<35} "
                f"requested_max={requested_max:5d} "
                f"frames={frame_count:5d} "
                f"→ {status}"
            )

    print("\n" + "=" * 80)
    if problems:
        print("❌ Found INVALID episodes:")
        for ep_idx, vk, req, fc in problems:
            print(
                f"  - EP {ep_idx}, {vk}: requested {req}, frames {fc}"
            )
    else:
        print("✅ All episodes are frame-safe.")

    return problems

if __name__ == "__main__":
    
    repo_id = "IGRIS_C"
    dataset_folder_name = "IGRIS_C_20251230_165442_insert"
    base_root = (DEFAULT_DATASET_PATH.parent / Path(DEFAULT_DATASET_PATH.name).expanduser()).resolve()
    dataset_root = Path(os.path.join(base_root, dataset_folder_name))

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    
    # video_key = "observation.image.realsense_head"
    # for ep_idx, ep in tqdm(
    #     enumerate(dataset.meta.episodes),
    #     total=dataset.meta.total_episodes,
    #     desc=f"ffmpeg rewrite {video_key}",
    # ):
    #     src_rel = dataset.meta.get_video_file_path(ep_idx, video_key)
        
    #     start_ts = ep[f"videos/{video_key}/from_timestamp"]
    #     end_ts   = ep[f"videos/{video_key}/to_timestamp"]
        
    #     print(f"EP {ep_idx}: start_ts={start_ts}, end_ts={end_ts}, fps={dataset.meta.fps}")
    
    # check_episode_frame_index_safety(dataset)
    
    def normalize_hand(v):
        v = np.asarray(v, dtype=np.float64)
        v[:12] /= np.array([5397,3920,5608,5299,3447,2543,2368,4295,4626,4371,5223,2591])
        return v
    
    modify_obs_hand_dataset = modify_features(
        dataset,
        "observation.state",
        normalize_hand,
        "hand_norm",
        repo_id
    )

    modify_left_video_dataset = modify_features(
        modify_obs_hand_dataset,
        "observation.image.stereo_left",
        "crop=400:240:550:480",
        "left_crop",
        repo_id
    )
    
    modify_right_video_dataset = modify_features(
        modify_left_video_dataset,
        "observation.image.stereo_right",
        "crop=400:240:285:480",
        "right_crop",
        repo_id
    )

    modify_realsense_video_dataset = modify_features(
        modify_right_video_dataset,
        "observation.image.realsense_head",
        "scale=640:480",
        "realsense_resize",
        repo_id
    )

    check_episode_frame_index_safety(modify_realsense_video_dataset)

    print("All modifications completed.")
    print(f"Hand normalized dataset at: {modify_obs_hand_dataset.root.name}")
    print(f"Left stereo cropped dataset at: {modify_left_video_dataset.root.name}")
    print(f"Right stereo cropped dataset at: {modify_right_video_dataset.root.name}")
    print(f"Realsense dataset at: {modify_realsense_video_dataset.root.name}")
