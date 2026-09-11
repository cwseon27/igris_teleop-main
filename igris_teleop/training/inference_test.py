import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch

# --- LeRobot ---
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION
from lerobot.policies.act.modeling_act import ACTPolicy

try:
    from PIL import Image
except Exception:
    Image = None


# -------------------------
# 이미지 전처리: crop + resize + tensor(CHW, float32, 0~1)
# -------------------------
def img_to_tensor(img, crop_xyxy=(350, 270, 1050, 720), out_hw=(224, 224)):
    """
    img: HWC uint8 numpy / PIL.Image / torch tensor 가능
    crop_xyxy: (x1, y1, x2, y2)
    out_hw: (H, W)
    """
    # -> numpy uint8 HWC
    if isinstance(img, torch.Tensor):
        x = img.detach().cpu().numpy()
        if x.ndim == 3 and x.shape[0] in (1, 3):  # CHW 추정
            x = np.transpose(x, (1, 2, 0))
        img = x

    if Image is not None and isinstance(img, Image.Image):
        img = np.array(img)

    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unexpected image shape: {img.shape}")

    # 알파 채널 제거
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    x1, y1, x2, y2 = crop_xyxy
    H, W = img.shape[:2]
    x1 = int(np.clip(x1, 0, W - 1))
    x2 = int(np.clip(x2, x1 + 1, W))
    y1 = int(np.clip(y1, 0, H - 1))
    y2 = int(np.clip(y2, y1 + 1, H))

    crop = img[y1:y2, x1:x2, :]

    # resize (opencv가 없으면 PIL로 대체)
    oh, ow = out_hw
    try:
        import cv2
        crop = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_AREA)
    except Exception:
        if Image is None:
            raise RuntimeError("cv2 또는 PIL 중 하나는 필요합니다.")
        crop = Image.fromarray(crop).resize((ow, oh))
        crop = np.array(crop)

    # -> torch tensor CHW float32 [0,1]
    t = torch.from_numpy(crop).permute(2, 0, 1).contiguous().float() / 255.0
    return t


# -------------------------
# dataset frame에서 obs/state 및 image 키를 최대한 견고하게 찾아서 배치 구성
# -------------------------
def extract_obs_state(frame, hand_len=12, arm_len=14, neck_len=2):
    """
    inference 코드와 동일한 순서: [hand(12), arm(14), neck(2)] = 28
    dataset에 obs_hand/obs_arm/obs_neck가 있으면 그걸 우선 사용.
    아니면 observation.state에서 split.
    """
    # 1) 분리된 키가 있는 경우
    if "obs_hand" in frame and "obs_arm" in frame and "obs_neck" in frame:
        obs_hand = np.asarray(frame["obs_hand"], dtype=np.float32).reshape(-1)
        obs_arm = np.asarray(frame["obs_arm"], dtype=np.float32).reshape(-1)
        obs_neck = np.asarray(frame["obs_neck"], dtype=np.float32).reshape(-1)
        obs = np.concatenate([obs_hand, obs_arm, obs_neck], axis=0)
        return obs

    # 2) LeRobot 표준 키(가능성이 높은 후보들)
    candidates = [
        "observation.state",
        "observation_state",
        "state",
        "observation",
    ]
    key = None
    for k in candidates:
        if k in frame:
            key = k
            break
    if key is None:
        raise KeyError(f"Cannot find observation state key in frame. tried={candidates}, keys={list(frame.keys())}")

    obs = np.asarray(frame[key], dtype=np.float32).reshape(-1)
    if obs.size < (hand_len + arm_len + neck_len):
        raise ValueError(f"observation.state too short: {obs.size}")
    obs = obs[: (hand_len + arm_len + neck_len)]
    return obs


def extract_images(frame):
    """
    inference 코드에서 요구하는 키로 최대한 맞춤:
      - observation.image.stereo_right
      - observation.image.stereo_left
      - observation.image.realsense_head
    데이터셋에 따라 키 이름이 다를 수 있어 후보를 몇 개 둠.
    """
    def pick(keys):
        for k in keys:
            if k in frame:
                return frame[k], k
        return None, None

    cam_right, kr = pick([
        "observation.image.stereo_right",
        "observation.image.right",
        "stereo_right",
    ])
    cam_left, kl = pick([
        "observation.image.stereo_left",
        "observation.image.left",
        "stereo_left",
    ])
    cam_head, kh = pick([
        "observation.image.realsense_head",
        "observation.image.head",
        "realsense_head",
        "camera_head",
    ])

    if cam_right is None or cam_left is None or cam_head is None:
        missing = []
        if cam_right is None: missing.append("stereo_right")
        if cam_left is None:  missing.append("stereo_left")
        if cam_head is None:  missing.append("realsense_head")
        raise KeyError(
            f"Missing image(s): {missing}. frame keys sample={list(frame.keys())[:30]}"
        )

    return (cam_right, cam_left, cam_head), (kr, kl, kh)


def build_batch(frame, device, crop_xyxy, img_hw, hand_len=12, arm_len=14, neck_len=2):
    obs_state = extract_obs_state(frame, hand_len, arm_len, neck_len)  # (28,)
    (cam_r, cam_l, cam_h), _ = extract_images(frame)

    batch = {
        "observation.state": torch.from_numpy(obs_state).unsqueeze(0).to(device, non_blocking=True),
        "observation.image.stereo_right": img_to_tensor(cam_r, crop_xyxy, img_hw).unsqueeze(0).to(device, non_blocking=True),
        "observation.image.stereo_left": img_to_tensor(cam_l, crop_xyxy, img_hw).unsqueeze(0).to(device, non_blocking=True),
        "observation.image.realsense_head": img_to_tensor(cam_h, crop_xyxy, img_hw).unsqueeze(0).to(device, non_blocking=True),
    }
    return batch, obs_state


# -------------------------
# 오프라인 평가 루프
# -------------------------
def run_offline_compare(
    dataset_root,
    episode_idx=0,
    max_steps=2000,
    stride=1,
    checkpoint_path=None,
    use_temporal_ensemble=False,
    device_str=None,
    crop_xyxy=(350, 270, 1050, 720),
    img_hw=(224, 224),
    dims_to_plot=(0, 12, 13, 14, 15),  # 예시: 일부 차원만
    save_prefix=None,
):
    device = torch.device(device_str if device_str else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    # dataset
    dataset = LeRobotDataset("IGRIS_C", root=dataset_root, episodes=[episode_idx])
    frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_idx)
    episode_rows = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_idx)
    episode_indices = list(episode_rows["index"])  # parquet에 있는 index 컬럼 사용

    # policy
    if checkpoint_path is None:
        raise ValueError("checkpoint_path를 지정하세요. (ACTPolicy.from_pretrained 경로)")
    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.reset()
    policy.eval()

    ts = []
    obs_list = []
    gt_list = []
    pred_list = []

    # 루프
    n = min(len(frames), max_steps * stride)
    step_count = 0
    for step_i, idx in enumerate(episode_indices[::stride]):
        frame = dataset[int(idx)]   # <-- 중요: hf row가 아니라 LeRobotDataset item

        batch, obs_state = build_batch(frame, device, crop_xyxy, img_hw)

        # gt action
        if ACTION not in frame:
            raise KeyError(f"Frame does not contain ACTION key='{ACTION}'. keys={list(frame.keys())}")
        gt = np.asarray(frame[ACTION], dtype=np.float32).reshape(-1)
        gt = gt[:28]  # 28-dim 가정(당신 코드 기준)

        with torch.inference_mode():
            if use_temporal_ensemble:
                act = policy.select_action(batch)  # (1, action_dim)
                pred = act.detach().cpu().numpy().reshape(-1)[:28].astype(np.float32)
            else:
                chunk = policy.predict_action_chunk(batch)  # (1, T, action_dim)
                # 현재 스텝 비교용으로 "첫 action"을 사용 (필요시 0 대신 k로 변경)
                pred = chunk.detach().cpu().numpy()[0, 0, :28].astype(np.float32)

        ts.append(step_count)
        obs_list.append(obs_state.astype(np.float32))
        gt_list.append(gt)
        pred_list.append(pred)

        step_count += 1

    ts = np.asarray(ts)
    obs_arr = np.vstack(obs_list)    # (T, 28)
    gt_arr = np.vstack(gt_list)      # (T, 28)
    pred_arr = np.vstack(pred_list)  # (T, 28)

    # metrics
    err = pred_arr - gt_arr
    rmse = np.sqrt(np.mean(err**2, axis=0))
    mae = np.mean(np.abs(err), axis=0)

    print("=== Per-dim metrics (pred vs gt_action) ===")
    for d in dims_to_plot:
        print(f"d={d:>2} | RMSE={rmse[d]:.6g} | MAE={mae[d]:.6g}")

    # ---- plots: obs_state vs gt_action vs pred_action (selected dims) ----
    dims = [int(d) for d in dims_to_plot if 0 <= int(d) < 28]
    if len(dims) == 0:
        raise ValueError("dims_to_plot이 비어있거나 범위를 벗어났습니다. (0~27)")

    # 1) time series
    fig1 = plt.figure()
    ax1 = plt.gca()
    for d in dims:
        ax1.plot(ts, obs_arr[:, d], label=f"obs[d={d}]")
        ax1.plot(ts, gt_arr[:, d], linestyle="--", label=f"gt_action[d={d}]")
        ax1.plot(ts, pred_arr[:, d], linestyle=":", label=f"pred_action[d={d}]")
    ax1.set_title("Offline compare (obs vs gt_action vs pred_action)")
    ax1.set_xlabel("step")
    ax1.set_ylabel("value")
    ax1.grid(True)
    ax1.legend(ncol=2, fontsize=9)

    # 2) error over time
    fig2 = plt.figure()
    ax2 = plt.gca()
    for d in dims:
        ax2.plot(ts, err[:, d], label=f"pred-gt[d={d}]")
    ax2.set_title("Error over time (pred_action - gt_action)")
    ax2.set_xlabel("step")
    ax2.set_ylabel("error")
    ax2.grid(True)
    ax2.legend(ncol=2, fontsize=9)

    # 3) scatter per dim
    scatter_figs = []
    for d in dims:
        fig = plt.figure()
        ax = plt.gca()
        ax.scatter(gt_arr[:, d], pred_arr[:, d], s=10)
        vmin = float(min(gt_arr[:, d].min(), pred_arr[:, d].min()))
        vmax = float(max(gt_arr[:, d].max(), pred_arr[:, d].max()))
        ax.plot([vmin, vmax], [vmin, vmax], linestyle="--")
        ax.set_title(f"Scatter d={d} | RMSE={rmse[d]:.4g}, MAE={mae[d]:.4g}")
        ax.set_xlabel("gt_action")
        ax.set_ylabel("pred_action")
        ax.grid(True)
        scatter_figs.append(fig)

    # 4) pooled error histogram
    fig3 = plt.figure()
    ax3 = plt.gca()
    pooled = err[:, dims].reshape(-1)
    ax3.hist(pooled, bins=60)
    ax3.set_title(f"Error histogram (dims={dims})")
    ax3.set_xlabel("pred - gt")
    ax3.set_ylabel("count")
    ax3.grid(True)

    # save
    if save_prefix:
        fig1.savefig(f"{save_prefix}_timeseries.png", dpi=200, bbox_inches="tight")
        fig2.savefig(f"{save_prefix}_error_time.png", dpi=200, bbox_inches="tight")
        fig3.savefig(f"{save_prefix}_error_hist.png", dpi=200, bbox_inches="tight")
        for fig, d in zip(scatter_figs, dims):
            fig.savefig(f"{save_prefix}_scatter_d{d}.png", dpi=200, bbox_inches="tight")

    plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True, help="예: ./dataset/IGRIS_C_20251215_231430/")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--stride", type=int, default=1, help="프레임 샘플링 간격")
    p.add_argument("--checkpoint", type=str, required=True, help="ACTPolicy.from_pretrained 경로")
    p.add_argument("--temporal-ensemble", action="store_true", help="policy.select_action 사용")
    p.add_argument("--device", type=str, default=None, help="예: cuda:0 또는 cpu")
    p.add_argument("--crop", type=int, nargs=4, default=[350, 270, 1050, 720], help="x1 y1 x2 y2")
    p.add_argument("--img-hw", type=int, nargs=2, default=[224, 224], help="H W")
    p.add_argument("--dims", type=int, nargs="+", default=[0, 12, 13, 14, 15], help="플롯할 action/state 차원들(0~27)")
    p.add_argument("--save-prefix", type=str, default=None, help="저장 파일 prefix (없으면 저장 안함)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_offline_compare(
        dataset_root=args.dataset_root,
        episode_idx=args.episode,
        max_steps=args.max_steps,
        stride=args.stride,
        checkpoint_path=args.checkpoint,
        use_temporal_ensemble=args.temporal_ensemble,
        device_str=args.device,
        crop_xyxy=tuple(args.crop),
        img_hw=tuple(args.img_hw),
        dims_to_plot=tuple(args.dims),
        save_prefix=args.save_prefix,
    )
