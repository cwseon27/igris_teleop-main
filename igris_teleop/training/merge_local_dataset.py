from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
from typing import Iterable, Sequence

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets


def _unique_repo_id_from_path(ds_path: Path) -> str:
    """
    LeRobotDataset(repo_id, root)에서 repo_id가 root 아래 폴더명으로 쓰이므로,
    입력 경로가 서로 다른데 폴더명이 같은 경우를 대비해 유니크 ID 생성.
    """
    p = ds_path.resolve()
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:8]
    return f"{p.name}__{h}"


def merge_lerobot_datasets_from_paths(
    dataset_paths: Sequence[str | Path],
    output_dir: str | Path,
    output_repo_id: str | None = None,
    backup_existing: bool = True,
) -> LeRobotDataset:
    """
    여러 로컬 LeRobotDataset(절대 경로)을 읽어 하나로 병합해 output_dir에 저장.

    Args:
        dataset_paths: 병합할 데이터셋들의 절대 경로 리스트
                      예) ["/data/lerobot/a", "/data/lerobot/b"]
        output_dir: 병합 결과를 저장할 절대 경로
                    예) "/data/lerobot/merged_out"
        output_repo_id: 병합된 데이터셋의 repo_id (기본: output_dir 폴더명)
        backup_existing: output_dir이 이미 있으면 _old_타임스탬프로 백업 후 진행

    Returns:
        병합된 LeRobotDataset (root=output_dir)
    """
    if not dataset_paths:
        raise ValueError("dataset_paths is empty.")

    # Path 정리 + 존재 확인
    ds_paths: list[Path] = []
    for p in dataset_paths:
        pp = Path(p).expanduser().resolve()
        if not pp.is_absolute():
            raise ValueError(f"Path must be absolute: {pp}")
        if not pp.exists():
            raise FileNotFoundError(f"Dataset path not found: {pp}")
        if not pp.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {pp}")
        ds_paths.append(pp)

    out_dir = Path(output_dir).expanduser().resolve()
    if not out_dir.is_absolute():
        raise ValueError(f"output_dir must be absolute: {out_dir}")

    if output_repo_id is None:
        output_repo_id = out_dir.name

    # output_dir 처리 (이미 존재하면 백업)
    if out_dir.exists():
        if backup_existing:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = out_dir.with_name(out_dir.name + f"_old_{ts}")
            logging.warning(f"Output dir already exists. Moving to backup: {backup_dir}")
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.move(str(out_dir), str(backup_dir))
        else:
            raise FileExistsError(f"Output dir already exists: {out_dir}")

    # out_dir.mkdir(parents=True, exist_ok=True)

    # 입력 데이터셋 로드
    datasets: list[LeRobotDataset] = []
    # for p in ds_paths:
    #     # LeRobotDataset은 (root / repo_id)를 기준으로 읽는 패턴이므로,
    #     # root=p.parent, repo_id=p.name 형태로 로드하면 경로가 정확히 맞습니다.
    #     # 단, 폴더명이 같은 경로들이 있을 수 있으니 repo_id는 유니크하게 만들어줍니다.
    #     repo_id = _unique_repo_id_from_path(p)
    #     root = p.parent
    #     expected = (root / p.name).resolve()
    #     if expected != p:
    #         # 일반적으로는 동일해야 합니다. 혹시 심볼릭 링크 등 케이스 방어
    #         logging.warning(f"Path normalization changed: expected={expected}, given={p}")

    #     ds = LeRobotDataset(repo_id=p.name, root=root)  # 실제 로드는 root/p.name
    #     # repo_id를 유니크하게 쓰고 싶으면 아래처럼 덮어써도 됨(내부 식별용)
    #     ds.repo_id = repo_id
    #     datasets.append(ds)

    for p in ds_paths:
        info_path = p / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing meta/info.json: {info_path}")

        # 핵심 수정: root는 p (Padding_1 폴더 자체)
        ds = LeRobotDataset(repo_id=_unique_repo_id_from_path(p), root=p)
        datasets.append(ds)

    logging.info(f"Loaded {len(datasets)} datasets:")
    for d, p in zip(datasets, ds_paths):
        logging.info(f" - {p} (episodes={d.meta.total_episodes}, frames={d.meta.total_frames})")

    # 병합
    logging.info(f"Merging into: {out_dir} (output_repo_id={output_repo_id})")
    merged = merge_datasets(
        datasets,
        output_repo_id=output_repo_id,
        output_dir=out_dir,
    )

    logging.info(
        f"Done. episodes={merged.meta.total_episodes}, frames={merged.meta.total_frames}, saved_to={out_dir}, data size={merged.meta.data_files_size_in_mb} MB, video size={merged.meta.video_files_size_in_mb} MB"
    )

    # 병합된 데이터셋 다시 로드해서 반환(경로 확실히)
    return LeRobotDataset(repo_id=output_repo_id, root=out_dir)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    
    merged = merge_lerobot_datasets_from_paths(
        dataset_paths=[
            "/home/dam/igris_teleop_v4/igris_teleop/AutoBagger_dataset_origin/IGRIS_C_20260211_180516_episode2",
            "/home/dam/igris_teleop_v4/igris_teleop/AutoBagger_dataset_origin/IGRIS_C_20260211_195724_episode1",
            "/home/dam/igris_teleop_v4/igris_teleop/AutoBagger_dataset_origin/IGRIS_C_20260211_201911_episode14",
            "/home/dam/igris_teleop_v4/igris_teleop/AutoBagger_dataset_origin/IGRIS_C_20260211_212035_episode6",
            "/home/dam/igris_teleop_v4/igris_teleop/AutoBagger_dataset_origin/IGRIS_C_20260211_214554_episode1",
            "/home/dam/igris_teleop_v4/igris_teleop/AutoBagger_dataset_origin/IGRIS_C_20260211_220052_episode1",
            
            "/home/dam/igris_teleop_v4/igris_teleop/AutoBagger_dataset_origin/IGRIS_C_20260212_014436",
            "/home/dam/igris_teleop_v4/igris_teleop/AutoBagger_dataset_origin/IGRIS_C_20260212_021514",
            "/home/dam/igris_teleop_v4/igris_teleop/AutoBagger_dataset_origin/IGRIS_C_20260212_023522",
            "/home/dam/igris_teleop_v4/igris_teleop/AutoBagger_dataset_origin/IGRIS_C_20260212_031048",
        ],
        output_dir="/home/dam/igris_teleop_v4/igris_teleop/dataset/AutoBagger",
        output_repo_id='IGRIS_C'
    )
