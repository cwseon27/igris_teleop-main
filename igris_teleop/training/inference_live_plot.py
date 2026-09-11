import argparse
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION
from lerobot.policies.factory import make_pre_post_processors

ARM_DIMS = list(range(12, 26))  # arm only: 14 dims


def img_to_tensor(img, crop_xyxy=(0, 0, 1280, 720), out_hw=(224, 224)):
    """
    img: HWC uint8 numpy / PIL.Image / torch tensor 가능
    return: torch.float32 CHW, [0,1]
    """
    # -> numpy HWC
    if isinstance(img, torch.Tensor):
        x = img.detach().cpu().numpy()
        if x.ndim == 3 and x.shape[0] in (1, 3):  # CHW
            x = np.transpose(x, (1, 2, 0))
        img = x

    try:
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
    except Exception:
        pass

    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    # RGBA -> RGB
    if img.shape[2] == 4:
        img = img[:, :, :3]
    # Gray -> RGB
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    H, W = img.shape[:2]
    x1, y1, x2, y2 = crop_xyxy
    x1 = int(np.clip(x1, 0, W - 1))
    x2 = int(np.clip(x2, x1 + 1, W))
    y1 = int(np.clip(y1, 0, H - 1))
    y2 = int(np.clip(y2, y1 + 1, H))
    crop = img[y1:y2, x1:x2, :]

    oh, ow = out_hw
    try:
        import cv2
        crop = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image
        crop = np.array(Image.fromarray(crop).resize((ow, oh)))

    t = torch.from_numpy(crop).permute(2, 0, 1).contiguous().float() / 255.0
    return t


def build_batch(frame, device, crop_xyxy, default_img_hw, expected_state_dim, expected_torque_dim, img_hw_by_key):
    # 원본 state/torque
    obs_state_full = np.asarray(frame["observation.state"], dtype=np.float32).reshape(-1)
    obs_torque_full = np.asarray(frame["observation.torque"], dtype=np.float32).reshape(-1)

    # --- 모델이 기대하는 차원으로 맞춤 (기본: 앞에서부터 expected_dim만 사용) ---
    if obs_state_full.size < expected_state_dim:
        raise ValueError(f"observation.state too short: {obs_state_full.size} < {expected_state_dim}")
    if obs_torque_full.size < expected_torque_dim:
        raise ValueError(f"observation.torque too short: {obs_torque_full.size} < {expected_torque_dim}")

    obs_state = obs_state_full[:expected_state_dim]
    obs_torque = obs_torque_full[:expected_torque_dim]

    # 이미지
    # cam_l = frame["observation.image.stereo_left"]
    # cam_r = frame["observation.image.stereo_right"]
    cam_h = frame["observation.image.realsense_head"]

    # 키별 기대 해상도가 있으면 그걸 사용, 없으면 default 사용

    hw_h = img_hw_by_key.get("observation.image.realsense_head", default_img_hw)

    batch = {
        "observation.state": torch.from_numpy(obs_state).unsqueeze(0).to(device, non_blocking=True),
        "observation.torque": torch.from_numpy(obs_torque).unsqueeze(0).to(device, non_blocking=True),
        # "observation.image.stereo_left": img_to_tensor(cam_l, crop_xyxy, hw_l).unsqueeze(0).to(device, non_blocking=True),
        # "observation.image.stereo_right": img_to_tensor(cam_r, crop_xyxy, hw_r).unsqueeze(0).to(device, non_blocking=True),
        "observation.image.realsense_head": img_to_tensor(cam_h, crop_xyxy, hw_h).unsqueeze(0).to(device, non_blocking=True),
    }
    return batch, obs_state



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--stride", type=int, default=1)

    p.add_argument("--hz", type=float, default=30.0, help="플롯/루프를 스트리밍처럼 돌릴 속도(유사 실시간)")
    p.add_argument("--win-sec", type=float, default=10.0, help="화면에 보일 시간 창")
    p.add_argument("--plot-dt", type=float, default=0.05, help="플롯 갱신 주기(초)")

    p.add_argument("--dims", type=int, nargs="+", help="비교할 차원(0~27). 기본은 arm 14 dims (12~25)")
    p.add_argument("--plot-all", action="store_true", help="모든 차원(0~27)을 플롯")
    p.add_argument("--use-select-action", action="store_true", help="temporal ensemble 방식이면 select_action 사용 권장")

    p.add_argument("--crop", type=int, nargs=4, default=[0, 0, 1280, 720])
    p.add_argument("--img-hw", type=int, nargs=2, default=[224, 224])
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    # dataset
    dataset = LeRobotDataset("IGRIS_C", root=args.dataset_root, episodes=[args.episode])

    # episode row -> global index list (중요: 실제 샘플은 dataset[idx]로 꺼내야 videos에서 이미지가 디코딩됨)
    episode_rows = dataset.hf_dataset.filter(lambda x: x["episode_index"] == args.episode)
    episode_indices = list(episode_rows["index"])
    if len(episode_indices) == 0:
        raise RuntimeError(f"Empty episode: {args.episode}")

    # policy
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.reset()
    policy.eval()
    device_str = str(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": device_str},
            "normalizer_processor": {"device": device_str},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {"device": device_str},
            "device_processor": {"device": "cpu"},
        },
    )

    state_shape = getattr(policy.config.input_features.get("observation.state", None), "shape", None)
    torque_shape = getattr(policy.config.input_features.get("observation.torque", None), "shape", None)

    EXPECTED_STATE_DIM = int(state_shape[0]) if state_shape is not None else 28
    EXPECTED_TORQUE_DIM = int(torque_shape[0]) if torque_shape is not None else EXPECTED_STATE_DIM

    # 이미지 입력 shape도 가능하면 모델 기준으로 사용 (없으면 args.img_hw 유지)
    IMG_HW_BY_KEY = {}
    for k in policy.config.input_features.keys():
        if k.startswith("observation.image."):
            ft = policy.config.input_features[k]
            shp = getattr(ft, "shape", None)
            # (C,H,W) 또는 (H,W,C) 가능성 대응
            if shp is not None and len(shp) == 3:
                if shp[0] in (1,3):   # CHW
                    IMG_HW_BY_KEY[k] = (int(shp[1]), int(shp[2]))
                elif shp[2] in (1,3): # HWC
                    IMG_HW_BY_KEY[k] = (int(shp[0]), int(shp[1]))

    print(f"[model] expected state dim={EXPECTED_STATE_DIM}, torque dim={EXPECTED_TORQUE_DIM}")
    if IMG_HW_BY_KEY:
        print(f"[model] expected image HW by key: {IMG_HW_BY_KEY}")
    
    if args.plot_all:
        dims = list(range(min(28, EXPECTED_STATE_DIM)))
    elif args.dims is None:
        dims = ARM_DIMS
    else:
        dims = [int(d) for d in args.dims if 0 <= int(d) < 28]
        if not dims:
            raise ValueError("dims가 비어있습니다. (0~27)")
    print(f"[plot] plotting dims: {dims}")

    # buffers
    maxlen = int(args.win_sec * args.hz) + 200
    t_buf = deque(maxlen=maxlen)
    obs_buf = {d: deque(maxlen=maxlen) for d in dims}
    gt_buf = {d: deque(maxlen=maxlen) for d in dims}
    pred_buf = {d: deque(maxlen=maxlen) for d in dims}

    # plot init
    plt.ion()
    n_dims = len(dims)
    ncols = 2 if n_dims > 1 else 1
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=(12, 2.5 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    fig.suptitle("Arm observation vs dataset action vs inference action")

    # lines per dim
    lines = {}
    for i, d in enumerate(dims):
        ax = axes[i]
        (l_obs,) = ax.plot([], [], label="obs", linewidth=1.2)
        (l_gt,) = ax.plot([], [], linestyle="--", label="dataset action", linewidth=1.2)
        (l_pred,) = ax.plot([], [], linestyle=":", label="inference action", linewidth=1.2)
        ax.set_title(f"Arm dim {d}")
        ax.grid(True)
        lines[d] = (l_obs, l_gt, l_pred)

    # 빈 축이 있으면 숨김
    for ax in axes[n_dims:]:
        ax.set_visible(False)

    # 범례는 첫 번째 축에만 표시
    if axes.size > 0:
        axes[0].legend(fontsize=9, loc="upper right")
    for ax in axes[-ncols:]:
        ax.set_xlabel("t [s]")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    start_wall = time.perf_counter()
    last_plot_wall = 0.0
    dt_target = 1.0 / max(1e-6, args.hz)

    # main loop
    n = min(len(episode_indices[::args.stride]), args.max_steps)
    for step_i, idx in enumerate(episode_indices[::args.stride][:n]):
        loop_t0 = time.perf_counter()

        frame = dataset[int(idx)]  # <-- 중요: 여기서 videos mp4에서 이미지 디코딩됨
        gt = np.asarray(frame[ACTION], dtype=np.float32).reshape(-1)[:28]

        batch, obs_state = build_batch(
            frame,
            device,
            tuple(args.crop),
            tuple(args.img_hw),
            expected_state_dim=EXPECTED_STATE_DIM,
            expected_torque_dim=EXPECTED_TORQUE_DIM,
            img_hw_by_key=IMG_HW_BY_KEY,
        )
        batch = preprocessor(batch)
        with torch.inference_mode():
            if args.use_select_action:
                act = policy.select_action(batch)
                act = postprocessor(act)
                pred = act.detach().cpu().numpy().reshape(-1)[:28].astype(np.float32)
            else:
                chunk = policy.predict_action_chunk(batch)
                chunk = postprocessor(chunk)
                pred = chunk.detach().cpu().numpy()[0, 0, :28].astype(np.float32)

        t = time.perf_counter() - start_wall
        t_buf.append(t)
        for d in dims:
            o = float(obs_state[d]) if obs_state is not None and obs_state.size > d else np.nan
            g = float(gt[d])
            p_ = float(pred[d])

            obs_buf[d].append(o)
            gt_buf[d].append(g)
            pred_buf[d].append(p_)

        # plot update (저주기)
        now = time.perf_counter()
        if now - last_plot_wall >= args.plot_dt and len(t_buf) > 2:
            last_plot_wall = now
            tt = list(t_buf)

            # update lines
            for d in dims:
                l_obs, l_gt, l_pred = lines[d]
                l_obs.set_data(tt, list(obs_buf[d]))
                l_gt.set_data(tt, list(gt_buf[d]))
                l_pred.set_data(tt, list(pred_buf[d]))

            # xlim window
            tmax = tt[-1]
            for ax in axes[:n_dims]:
                ax.set_xlim(max(0.0, tmax - args.win_sec), tmax)

            # y-lims (각 subplot별)
            def set_ylim(ax, series_lists):
                vals = []
                for s in series_lists:
                    vals.extend([v for v in s if np.isfinite(v)])
                if not vals:
                    return
                vmin, vmax = min(vals), max(vals)
                margin = max(1e-3, 0.1 * (vmax - vmin))
                ax.set_ylim(vmin - margin, vmax + margin)

            for i, d in enumerate(dims):
                set_ylim(axes[i], [list(obs_buf[d]), list(gt_buf[d]), list(pred_buf[d])])

            fig.canvas.draw_idle()
            plt.pause(0.001)

            if not plt.fignum_exists(fig.number):
                break

        # 유사 실시간: hz에 맞춰 sleep
        elapsed = time.perf_counter() - loop_t0
        remain = dt_target - elapsed
        if remain > 0:
            time.sleep(remain)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
