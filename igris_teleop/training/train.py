#!/usr/bin/env python
from __future__ import annotations

import os
import re
import tqdm
import traceback
import numpy as np
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Tuple, Dict, Any

import torch
from torchvision.transforms import v2

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.factory import resolve_delta_timestamps

try:
    from .image_transforms import XYXYCrop, BottomCenterCrop
    from .loss_logger import LossLogger
except ImportError:  # Allow direct script execution.
    from image_transforms import XYXYCrop, BottomCenterCrop
    from loss_logger import LossLogger
from igris_teleop.core.project_paths import CHECKPOINTS_ROOT, DATASETS_ROOT

DEFAULT_DATASET_PATH = DATASETS_ROOT
DEFAULT_POLICY_PATH  = CHECKPOINTS_ROOT

_EPOCH_DIR_RE = re.compile(r"^epoch_(\d{6})$")


# -----------------------------
# Resume / checkpoint utilities
# -----------------------------
def _checkpoint_has_model(ckpt_dir: Path) -> bool:
    # safetensors를 우선으로 확인. (환경에 따라 pytorch_model.bin 등일 수도 있어 같이 허용)
    return (ckpt_dir / "model.safetensors").exists() or (ckpt_dir / "pytorch_model.bin").exists()


def _find_resume_checkpoint(run_dir: Path) -> Optional[Path]:
    """
    LeRobot train.py 스타일:
    - "epoch_latest"가 있으면 우선
    - 없으면 epoch_XXXXXX 중 가장 큰 것
    단, 모델 파일이 실제로 있는 폴더만 후보로 인정.
    """
    if not run_dir.exists() or not run_dir.is_dir():
        return None

    latest = run_dir / "epoch_latest"
    if latest.exists() and latest.is_dir() and _checkpoint_has_model(latest):
        return latest

    candidates = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        m = _EPOCH_DIR_RE.match(p.name)
        if m and _checkpoint_has_model(p):
            candidates.append((int(m.group(1)), p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _save_training_state(path: Path, *, epoch: int, global_step: int,
                        optimizer: torch.optim.Optimizer,
                        lr_scheduler: Optional[Any] = None) -> None:
    """
    LeRobot의 load_training_state / save_checkpoint 철학을 따라
    optimizer/scheduler/step을 별도 파일로 저장.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "optimizer": optimizer.state_dict(),
    }
    if lr_scheduler is not None:
        obj["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(obj, path)


def _load_training_state(path: Path, optimizer: torch.optim.Optimizer,
                        lr_scheduler: Optional[Any] = None) -> Tuple[int, int]:
    """
    Returns (start_epoch, global_step).
    저장된 epoch는 "마지막으로 완료한 epoch"로 간주하고, 다음 epoch부터 이어서.
    """
    if not path.exists():
        return 0, 0
    obj = torch.load(path, map_location="cpu")
    if "optimizer" in obj:
        optimizer.load_state_dict(obj["optimizer"])
    if lr_scheduler is not None and obj.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(obj["lr_scheduler"])
    last_epoch = int(obj.get("epoch", 0))
    global_step = int(obj.get("global_step", 0))
    return last_epoch + 1, global_step


def _safe_save_pretrained(obj, save_dir: Path) -> None:
    """
    save_pretrained가 safe_serialization 인자를 받는 경우 safetensors를 강제,
    안 받으면 기본 save_pretrained 호출.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        obj.save_pretrained(save_dir, safe_serialization=True)
    except TypeError:
        obj.save_pretrained(save_dir)


def _safe_load_processor(processor_obj, ckpt_dir: Path):
    """
    LeRobot 0.4.2에서 processor 저장/로드 포맷이 케이스에 따라 달라질 수 있어:
    - from_pretrained 시도
    - 실패하면 None 반환 (caller에서 make_pre_post_processors로 재생성)
    """
    try:
        return processor_obj.__class__.from_pretrained(ckpt_dir)
    except Exception:
        return None


# -----------------------------
# Main
# -----------------------------
def main():
    try:
        STATE_ACTION_DIM = 28
        KEEP = slice(0, STATE_ACTION_DIM)

        policy_type = "ACT"  # "Diffusion" or "ACT"
        dataset_repo_id = "IGRIS_C"
        dataset_folder_name = "AutoBagger"

        # -----------------------------
        # Resume control (사용자가 지정)
        # -----------------------------
        # 1) 새 학습: resume_dir = None
        # 2) 재개: 기존 output run 폴더 경로를 넣기
        resume_dir: Optional[Path] = None
        # 예) resume_dir = DEFAULT_POLICY_PATH / "IGRIS_C_20260111_040919_insert_manufacturing""

        # optimizer state까지 이어서 학습하려면 True 권장
        SAVE_TRAINING_STATE = True

        safe_repo = dataset_repo_id.replace("/", "__")
        ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")

        if resume_dir is None:
            policy_root = DEFAULT_POLICY_PATH / f"{safe_repo}_{ts}"
        else:
            policy_root = Path(resume_dir)

        dataset_root = Path(os.path.join(DEFAULT_DATASET_PATH, dataset_folder_name))

        output_directory = Path(policy_root)
        output_directory.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_directory}")

        device = torch.device("cuda")

        training_epochs = 30000
        checkpoint_epochs = 1000
        if training_epochs < checkpoint_epochs:
            raise ValueError("training_epochs must be greater than or equal to checkpoint_epochs")

        # -----------------------------
        # Dataset metadata / features / stats
        # -----------------------------
        dataset_metadata = LeRobotDatasetMetadata(repo_id=dataset_repo_id, root=dataset_root)

        # (1) feature spec 축소
        features_spec = deepcopy(dataset_metadata.features)
        features_spec["observation.state"]["shape"] = [STATE_ACTION_DIM]
        features_spec["action"]["shape"] = [STATE_ACTION_DIM]

        features = dataset_to_policy_features(features_spec)
        output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features  = {k: ft for k, ft in features.items() if k not in output_features}
        
        print("\n=============== Input features (Observation) ===============")
        for k, v in input_features.items():
            print(f"{k:<40} {tuple(v.shape)}")
            
        print("\n=============== Output features (ACTION) ===================")
        for k, v in output_features.items():
            print(f"{k:<40} {tuple(v.shape)}")



        # (2) stats 축소
        dataset_stats = deepcopy(dataset_metadata.stats)

        def _slice_stats(stats_dict, key, keep):
            if key not in stats_dict:
                return
            for stat_name, v in stats_dict[key].items():
                if isinstance(v, torch.Tensor):
                    stats_dict[key][stat_name] = v[..., keep].clone()
                else:
                    arr = np.asarray(v)
                    stats_dict[key][stat_name] = arr[..., keep]

        _slice_stats(dataset_stats, "observation.state", KEEP)
        _slice_stats(dataset_stats, "action", KEEP)
        
        _slice_stats(dataset_stats, "observation.torque", KEEP)

        # -----------------------------
        # Build policy (or load if resuming)
        # -----------------------------
        if policy_type == "ACT":
            cfg = ACTConfig(input_features=input_features, output_features=output_features, chunk_size=100, n_action_steps=100)
            policy = ACTPolicy(cfg)
        elif policy_type == "Diffusion":
            cfg = DiffusionConfig(input_features=input_features, output_features=output_features, horizon=16, n_action_steps=8)
            policy = DiffusionPolicy(cfg)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        # pre/post는 일단 생성 (resume 시 로드 성공하면 덮어씀)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_stats)

        # -----------------------------
        # Resume: find checkpoint and load
        # -----------------------------
        resume_ckpt = _find_resume_checkpoint(output_directory)
        start_epoch = 0
        global_step = 0

        if resume_ckpt is not None:
            print(f"[RESUME] Found checkpoint: {resume_ckpt}")

            # (A) policy 로드 (model.safetensors 기반)
            if policy_type == "ACT":
                policy = ACTPolicy.from_pretrained(resume_ckpt)
            else:
                policy = DiffusionPolicy.from_pretrained(resume_ckpt)

        # device / mode
        policy.train()
        policy.to(device, non_blocking=True)

        # AMP 유지
        try:
            policy.config.use_amp = True
        except Exception:
            pass

        # optimizer는 반드시 policy 로드 이후 생성
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

        # (B) optimizer/epoch/step 로드 (lerobot load_training_state 대응)
        if resume_ckpt is not None and SAVE_TRAINING_STATE:
            state_path = resume_ckpt / "training_state.pt"
            start_epoch, global_step = _load_training_state(state_path, optimizer, lr_scheduler=None)
            print(f"[RESUME] start_epoch={start_epoch}, global_step={global_step}")

        # (C) pre/post 로드 시도 (없으면 재생성 유지)
        if resume_ckpt is not None:
            loaded_pre = _safe_load_processor(preprocessor, resume_ckpt)
            loaded_post = _safe_load_processor(postprocessor, resume_ckpt)
            if loaded_pre is not None and loaded_post is not None:
                preprocessor, postprocessor = loaded_pre, loaded_post
                print("[RESUME] Loaded pre/post processors from checkpoint.")
            else:
                # lerobot_train.py처럼 "stats 기반으로 재생성"을 fallback으로 둠
                preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_stats)
                print("[RESUME] Could not load processors; recreated from dataset_stats.")

        # -----------------------------
        # Dataset / loader
        # -----------------------------
        delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

        image_transforms = v2.Compose([
            v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)], p=0.01),
            # v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.01),
            # v2.RandomApply([v2.RandomAffine(degrees=3, translate=(0.03, 0.03))], p=0.01),
        ])

        dataset = LeRobotDataset(
            repo_id=dataset_repo_id,
            root=dataset_root,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=3,
            batch_size=16,
            shuffle=True,
            pin_memory=device.type != "cpu",
            persistent_workers=True
        )

        logging_step = 100
        logger = LossLogger(output_directory, run_name="train", keep_last=5000)

        # -----------------------------
        # Training loop (resume-aware)
        # -----------------------------
        tq_epoch = tqdm.tqdm(range(start_epoch, training_epochs), desc="[epoch]", dynamic_ncols=True)

        for epoch in tq_epoch:
            tq_batch = tqdm.tqdm(dataloader, desc=f"[batch] epoch={epoch}", leave=False, dynamic_ncols=True)

            for batch in tq_batch:
                # preprocessor 전에 차원 축소
                batch["observation.state"] = batch["observation.state"][..., KEEP]
                batch["action"]            = batch["action"][..., KEEP]

                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
                batch = preprocessor(batch)

                loss, _ = policy.forward(batch)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % logging_step == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    loss_val = float(loss.detach().item())
                    tq_batch.set_description_str(
                        f"[train] epoch={epoch} loss={loss_val:.3f} lr={lr:.6f} step={global_step}",
                        refresh=True
                    )

            tq_epoch.set_description_str(
                f"[epoch] epoch={epoch} last_loss={loss_val:.3f} lr={lr:.6f} step={global_step}",
                refresh=True
            )
            logger.log(epoch=epoch, loss=loss_val, lr=lr, split="train")

            # -----------------------------
            # Checkpoint save (lerobot 스타일로 latest도 갱신)
            # -----------------------------
            if epoch % checkpoint_epochs == 0:
                save_dir = output_directory / f"epoch_{epoch:06d}"
                latest_dir = output_directory / "epoch_latest"
                save_dir.mkdir(parents=True, exist_ok=True)
                latest_dir.mkdir(parents=True, exist_ok=True)

                # (1) policy/processor는 save_pretrained만 사용 -> model.safetensors 생성 기대
                _safe_save_pretrained(policy, save_dir)
                _safe_save_pretrained(preprocessor, save_dir)
                _safe_save_pretrained(postprocessor, save_dir)

                _safe_save_pretrained(policy, latest_dir)
                _safe_save_pretrained(preprocessor, latest_dir)
                _safe_save_pretrained(postprocessor, latest_dir)

                # (2) optimizer/step 상태 저장 (lerobot load_training_state 대응)
                if SAVE_TRAINING_STATE:
                    _save_training_state(save_dir / "training_state.pt",
                                        epoch=epoch, global_step=global_step,
                                        optimizer=optimizer, lr_scheduler=None)
                    _save_training_state(latest_dir / "training_state.pt",
                                        epoch=epoch, global_step=global_step,
                                        optimizer=optimizer, lr_scheduler=None)

                logger.export_csv(output_directory / "loss_log.csv")
                logger.save_plot(save_dir / "loss_recent.png", x_key="epoch")

        # finalize: epoch_latest에 최종 저장
        latest_dir = output_directory / "epoch_latest"
        latest_dir.mkdir(parents=True, exist_ok=True)

        _safe_save_pretrained(policy, latest_dir)
        _safe_save_pretrained(preprocessor, latest_dir)
        _safe_save_pretrained(postprocessor, latest_dir)

        if SAVE_TRAINING_STATE:
            _save_training_state(latest_dir / "training_state.pt",
                                epoch=training_epochs - 1, global_step=global_step,
                                optimizer=optimizer, lr_scheduler=None)

        logger.export_csv(output_directory / "loss_log.csv")
        logger.save_plot(output_directory / "loss_log_recent.png", x_key="epoch")
        logger.close()

    except Exception:
        print("An error occurred during training:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
