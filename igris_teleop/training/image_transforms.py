from torchvision.transforms.v2 import functional as F
from torch.utils.data import DataLoader
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

class XYXYCrop:
    """
    좌상단(x1,y1) ~ 우하단(x2,y2) 영역만 crop.
    - 좌표계: (0,0)=좌상단, x→오른쪽, y→아래
    - (x2, y2)는 exclusive (파이썬 슬라이싱처럼)로 해석
      => width = x2 - x1, height = y2 - y1
    """

    def __init__(
        self,
        x1: int, # Left upper
        y1: int, # Left upper
        x2: int, # Right lower
        y2: int, # Right lower
        pad_if_needed: bool = False,
        strict: bool = True,
        pad_fill: int | float = 0,
    ):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.pad_if_needed = bool(pad_if_needed)
        self.strict = bool(strict)
        self.pad_fill = pad_fill

        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"Invalid box: (x1,y1)=({self.x1},{self.y1}), (x2,y2)=({self.x2},{self.y2})")

    def __call__(self, img):
        
        if isinstance(img, np.ndarray):
            # cv2 입력은 보통 HWC(BGR) 입니다. 학습이 RGB였으면 아래 한 줄을 켜세요.
            # img = img[..., ::-1].copy()

            img = torch.from_numpy(img)
            if img.ndim == 3:  # HWC -> CHW
                img = img.permute(2, 0, 1).contiguous()
            img = img.to(torch.float32)
            if img.max() > 1.0:
                img = img / 255.0
        
        # 텐서/TVTensor: 보통 (C,H,W). (H,W,C)도 들어올 수 있어 방어적으로 처리.
        if hasattr(img, "shape") and len(img.shape) >= 2:
            if len(img.shape) >= 3 and img.shape[-3] in (1, 3, 4):
                # (C,H,W) 가정
                h, w = int(img.shape[-2]), int(img.shape[-1])
            else:
                # (H,W) 또는 (H,W,C) 가정
                h, w = int(img.shape[-3]), int(img.shape[-2]) if len(img.shape) >= 3 else (int(img.shape[-2]), int(img.shape[-1]))
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2

        # 필요 시 패딩으로 박스를 이미지 내부로 만들기
        if self.pad_if_needed:
            pad_l = max(-x1, 0)
            pad_t = max(-y1, 0)
            pad_r = max(x2 - w, 0)
            pad_b = max(y2 - h, 0)
            if pad_l or pad_t or pad_r or pad_b:
                # torchvision v2: F.pad(img, [left, top, right, bottom], ...)
                img = F.pad(img, [pad_l, pad_t, pad_r, pad_b], fill=self.pad_fill)
                # 패딩 후 좌표 이동
                x1 += pad_l; x2 += pad_l
                y1 += pad_t; y2 += pad_t
                h += pad_t + pad_b
                w += pad_l + pad_r

        # strict 모드에서는 범위 체크
        if self.strict:
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                raise ValueError(
                    f"Crop box out of bounds: box=({x1},{y1})-({x2},{y2}), image (H,W)=({h},{w}). "
                    f"Use pad_if_needed=True or set strict=False."
                )
        else:
            # 비-strict 모드에서는 클램프
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Clamped box became empty: ({x1},{y1})-({x2},{y2})")

        return F.crop(img, top=y1, left=x1, height=(y2 - y1), width=(x2 - x1))


class BottomCenterCrop:
    """가로 중앙 + 하단 기준으로 (width, height) 크롭"""
    def __init__(self, width: int, height: int, pad_if_needed: bool = False):
        self.width = int(width)
        self.height = int(height)
        self.pad_if_needed = bool(pad_if_needed)

    def __call__(self, img):
        # v2.ToImage() 이후라면 대부분 (C,H,W) 텐서/TVTensor로 들어옵니다.
        h, w = int(img.shape[-2]), int(img.shape[-1])

        if self.pad_if_needed:
            pad_l = max((self.width - w) // 2, 0)
            pad_r = max(self.width - w - pad_l, 0)
            pad_t = max(self.height - h, 0)  # 하단 기준이므로 위쪽으로 패딩
            pad_b = 0
            if pad_l or pad_r or pad_t or pad_b:
                img = F.pad(img, [pad_l, pad_t, pad_r, pad_b])
                h, w = int(img.shape[-2]), int(img.shape[-1])

        if w < self.width or h < self.height:
            raise ValueError(f"Image too small: got (H,W)=({h},{w}), need ({self.height},{self.width})")

        top = h - self.height
        left = (w - self.width) // 2
        return F.crop(img, top=top, left=left, height=self.height, width=self.width)

def chw_to_hwc_uint8(x: torch.Tensor):
    """
    x: (C,H,W), float(0~1) or uint8
    return: (H,W,C) uint8 numpy
    """
    t = x.detach().cpu()
    if t.dtype != torch.uint8:
        t = t.clamp(0, 1)
        t = (t * 255.0).to(torch.uint8)
    # (C,H,W) -> (H,W,C)
    return t.permute(1, 2, 0).numpy()

def main():
    repo_id = "IGRIS_C"
    root = Path("/home/dam/igris_teleop_v4/igris_teleop/dataset/IGRIS_C_20251230_165442_insert")

    dataset = LeRobotDataset(repo_id=repo_id, root=root, episodes=[0])
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    batch = next(iter(loader))

    # image_key = "observation.image.realsense_head"
    image_key = "observation.image.stereo_left"
    
    img = batch[image_key]          # (B,C,H,W)
    img0 = img[0]                   # (C,H,W)

    x1, y1, x2, y2 = 350, 270, 1050, 720
    crop = XYXYCrop(x1=x1, y1=y1, x2=x2, y2=y2)

    cropped = crop(img0)            # (C,h,w)

    # --- 시각화용 변환 ---
    orig_vis = chw_to_hwc_uint8(img0)
    crop_vis = chw_to_hwc_uint8(cropped)

    # 1) 원본 + crop 박스 표시
    plt.figure(figsize=(10, 6))
    plt.imshow(orig_vis)
    plt.title(f"Original: {image_key}  shape={tuple(img0.shape)}")
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, fill=False)
    plt.gca().add_patch(rect)
    plt.axis("off")
    plt.show()

    # 2) 크롭 결과 표시
    plt.figure(figsize=(10, 6))
    plt.imshow(crop_vis)
    plt.title(f"Cropped: ({x1},{y1})-({x2},{y2})  shape={tuple(cropped.shape)}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
