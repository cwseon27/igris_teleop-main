from __future__ import annotations

import os
import tqdm
import traceback
import numpy as np
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo

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
DEFAULT_POLICY_PATH = CHECKPOINTS_ROOT

def main():
    try:        
        STATE_ACTION_DIM = 28
        KEEP = slice(0, STATE_ACTION_DIM)
        # KEEP = [0,1,2,3, 6,7,8, ...]             # 특정 인덱스만 쓰려면 이렇게 리스트로

        policy_type = "ACT"  # "Diffusion" or "ACT"
        dataset_repo_id = "IGRIS_C"
        dataset_folder_name = "IGRIS_C_20251230_165442_insert"
        
        safe_repo = dataset_repo_id.replace("/", "__")
        ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
        policy_root = DEFAULT_POLICY_PATH / f"{safe_repo}_{ts}"
        
        dataset_root = Path(os.path.join(DEFAULT_DATASET_PATH, dataset_folder_name))
        
        output_directory = Path(policy_root)
        output_directory.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_directory}")

        device = torch.device("cuda")

        training_epochs = 5000
        checkpoint_epochs = 10
        if training_epochs < checkpoint_epochs:
            raise ValueError("training_epochs must be greater than or equal to checkpoint_epochs")

        dataset_metadata = LeRobotDatasetMetadata(repo_id=dataset_repo_id, root=dataset_root)
        features = dataset_to_policy_features(dataset_metadata.features)
        
        ####################################################################
        # 1) feature spec(shape) 축소
        features_spec = deepcopy(dataset_metadata.features)
        features_spec["observation.state"]["shape"] = [STATE_ACTION_DIM]
        features_spec["action"]["shape"] = [STATE_ACTION_DIM]

        features = dataset_to_policy_features(features_spec)
        output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features  = {k: ft for k, ft in features.items() if k not in output_features}

        # 2) stats 축소 (정규화 차원 불일치 방지)
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
        ####################################################################
        
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}
        
        # remove features example code
        # input_features.pop("observation.image")

        if policy_type == "ACT":
            cfg = ACTConfig(input_features=input_features, output_features=output_features)
            policy = ACTPolicy(cfg)
        elif policy_type == "Diffusion":
            cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
            policy = DiffusionPolicy(cfg)
        else:
            raise ValueError(f"Unknown policy type: {policy}")
        
        policy.train()
        policy.to(device, non_blocking=True)

        policy.config.use_amp = True
        
        use_amp = True
        amp_dtype = torch.bfloat16

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_stats)

        delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
        
        image_transforms = v2.Compose([
                                XYXYCrop(x1=350, y1=270, x2=1050, y2=720),
                                # BottomCenterCrop(width=640, height=480, pad_if_needed=False),
                                v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)], p=0.1),
                                v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.1),
                                v2.RandomApply([v2.RandomAffine(
                                    degrees=3,                 # 회전 ±3도
                                    translate=(0.03, 0.03),    # 좌우/상하 최대 3% 평행이동
                                )], p=0.1),
                            ])
        
        dataset = LeRobotDataset(repo_id=dataset_repo_id, 
                                 root=dataset_root,
                                 delta_timestamps=delta_timestamps,
                                 image_transforms=image_transforms)

        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=4,
            batch_size=32,
            shuffle=True,
            pin_memory=device.type != "cpu",
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        
        logger = LossLogger(output_directory, run_name="train", keep_last=5000)
        
        tq_epoch = tqdm.tqdm(range(training_epochs), desc="[epoch]", dynamic_ncols=True)
        
        printed = False
        for epoch in tq_epoch:
            tq_batch = tqdm.tqdm(dataloader, desc=f"[batch] epoch={epoch}", leave=False, dynamic_ncols=True)

            for batch in tq_batch:
                batch["observation.state"] = batch["observation.state"][..., KEEP]
                batch["action"]            = batch["action"][..., KEEP]

                batch = preprocessor(batch)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    if not printed:
                        print("autocast enabled (inside):", torch.is_autocast_enabled())
                        print("autocast dtype (inside):", torch.get_autocast_dtype("cuda"))
                        printed = True
                        
                    loss, _ = policy.forward(batch)
                
                loss.backward()
                optimizer.step()

                lr = optimizer.param_groups[0]["lr"]
                loss_val = float(loss.detach().item())
                tq_batch.set_description_str(f"[train] epoch={epoch} loss={loss_val:.3f} lr={lr:.6f}", refresh=True)

                
            tq_epoch.set_description_str(f"[epoch] epoch={epoch} last_loss={loss_val:.3f} lr={lr:.6f}", refresh=True)
            logger.log(epoch=epoch, loss=loss_val, lr=lr, split="train")
                
            if epoch % checkpoint_epochs == 0:
                save_dir = output_directory / f"epoch_{epoch:06d}"
                save_dir.mkdir(parents=True, exist_ok=True)

                policy.save_pretrained(save_dir)
                preprocessor.save_pretrained(save_dir)
                postprocessor.save_pretrained(save_dir)

                logger.export_csv(output_directory / "loss_log.csv")
                logger.save_plot(save_dir / "loss_recent.png", x_key="epoch")

        save_dir = output_directory / "epoch_latest"
        save_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(save_dir)

        policy.save_pretrained(save_dir)
        preprocessor.save_pretrained(save_dir)
        postprocessor.save_pretrained(save_dir)
        
        logger.export_csv(output_directory / "loss_log.csv")
        logger.save_plot(output_directory / "loss_log_recent.png", x_key="epoch")
        logger.close()
        
    except Exception as e:
        print("An error occurred during training:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
