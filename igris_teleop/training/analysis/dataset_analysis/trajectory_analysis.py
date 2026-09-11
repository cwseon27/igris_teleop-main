#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("pandas가 필요합니다. `pip install pandas pyarrow`를 설치하세요.") from exc

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


EE_KEYS = ("head_mat", "left_wrist_mat", "right_wrist_mat")
EE_COL = "observation.ee_pose"


def default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[3] / "dataset" / "AutoBagger"


def load_total_episodes(info_path: Path) -> int | None:
    try:
        data = json.loads(info_path.read_text())
        return int(data.get("total_episodes"))
    except Exception:
        return None


def parse_episodes_arg(episodes_arg: str | None, default_ep: int, total_episodes: int | None) -> list[int] | None:
    if episodes_arg is None:
        return [default_ep]

    arg = episodes_arg.strip().lower()
    if arg in ("all", "*"):
        if total_episodes is None:
            return None
        return list(range(total_episodes))

    eps: set[int] = set()
    for part in episodes_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":", 1)
            start = int(a) if a else 0
            if b:
                end = int(b)
            else:
                if total_episodes is None:
                    raise ValueError("episodes range end required when total_episodes is unknown.")
                end = total_episodes
            eps.update(range(start, end))
        else:
            eps.add(int(part))
    return sorted(eps)


def parquet_paths(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"dataset data dir not found: {data_dir}")
    return sorted(data_dir.glob("chunk-*/*.parquet"))


def pick_mat_order(sample_vec: np.ndarray) -> str:
    target = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    mats_c = np.asarray(sample_vec, dtype=np.float64).reshape(3, 4, 4, order="C")
    score_c = float(np.mean(np.abs(mats_c[:, 3, :] - target)))
    mats_f = np.asarray(sample_vec, dtype=np.float64).reshape(3, 4, 4, order="F")
    score_f = float(np.mean(np.abs(mats_f[:, 3, :] - target)))
    return "F" if score_f < score_c else "C"


def collect_episode_poses(
    paths: list[Path],
    episodes: set[int] | None,
    stride: int,
) -> dict[int, np.ndarray]:
    data_by_ep: dict[int, list[np.ndarray]] = {}
    cols = ["episode_index", "frame_index", EE_COL]

    for path in paths:
        df = pd.read_parquet(path, columns=cols).reset_index(drop=True)
        if episodes is not None:
            df = df[df["episode_index"].isin(episodes)]
        if df.empty:
            continue

        df = df.sort_values(["episode_index", "frame_index"], kind="mergesort")
        for ep, group in df.groupby("episode_index", sort=True):
            arr = np.stack(group[EE_COL].to_list(), axis=0)
            if stride > 1:
                arr = arr[::stride]
            data_by_ep.setdefault(int(ep), []).append(arr)

    merged: dict[int, np.ndarray] = {}
    for ep, chunks in data_by_ep.items():
        merged[ep] = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
    return merged


def positions_from_poses(
    poses: np.ndarray,
    mat_order: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if poses.ndim != 2 or poses.shape[1] != 48:
        raise ValueError(f"Expected poses shape (T, 48), got {poses.shape}")

    order = mat_order if mat_order in ("C", "F") else pick_mat_order(poses[0])
    mats = poses.reshape(-1, 3, 4, 4, order=order)
    mask = np.isfinite(mats).all(axis=(1, 2, 3))
    mats = mats[mask]
    head = mats[:, 0, :3, 3]
    left = mats[:, 1, :3, 3]
    right = mats[:, 2, :3, 3]
    return head, left, right


def set_equal_axes(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius == 0.0:
        radius = 1e-6
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_trajectory(
    ax,
    pts: np.ndarray,
    title: str,
    color: str,
    label: str | None = None,
    mark_ends: bool = True,
) -> None:
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=1.0, label=label)
    if mark_ends and pts.shape[0] >= 1:
        ax.scatter([pts[0, 0]], [pts[0, 1]], [pts[0, 2]], color="black", s=12, marker="o")
        ax.scatter([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]], color="black", s=16, marker="x")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_equal_axes(ax, pts)


def resolve_save_path(save_arg: str | None, episode: int, multi: bool, combined: bool) -> Path | None:
    if save_arg is None:
        return None
    path = Path(save_arg)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf", ".svg"):
        if multi and not combined:
            return path.with_name(f"{path.stem}_ep{episode:03d}{path.suffix}")
        return path
    path.mkdir(parents=True, exist_ok=True)
    if combined:
        return path / "trajectory_combined.png"
    return path / f"trajectory_ep{episode:03d}.png"


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot EE pose trajectories from dataset episodes.")
    parser.add_argument("--dataset-root", type=str, default=str(default_dataset_root()))
    parser.add_argument("--episode", type=int, default=0, help="single episode index")
    parser.add_argument("--episodes", type=str, default=None, help="comma list/range (e.g., 0,1,2 or 0:10) or 'all'")
    parser.add_argument("--stride", type=int, default=1, help="subsample every N frames")
    parser.add_argument("--mat-order", type=str, choices=["auto", "C", "F"], default="auto")
    parser.add_argument("--save", type=str, default=None, help="save path or directory")
    parser.add_argument("--combine", action="store_true", help="plot multiple episodes in one figure")
    parser.add_argument("--separate", action="store_true", help="plot each episode separately")
    parser.add_argument("--no-show", action="store_true", help="do not call plt.show()")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    total_episodes = load_total_episodes(info_path) if info_path.exists() else None

    selected = parse_episodes_arg(args.episodes, args.episode, total_episodes)
    episodes_set = set(selected) if selected is not None else None

    paths = parquet_paths(dataset_root)
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    data_by_ep = collect_episode_poses(paths, episodes_set, args.stride)
    if not data_by_ep:
        print("No matching episode data found.", file=sys.stderr)
        return 1

    multi = len(data_by_ep) > 1
    combined = args.combine or (multi and not args.separate)
    mat_order = None if args.mat_order == "auto" else args.mat_order

    if combined:
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")

        episodes_sorted = sorted(data_by_ep.keys())
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N) for i in range(len(episodes_sorted))]

        all_head = []
        all_left = []
        all_right = []

        for i, ep in enumerate(episodes_sorted):
            poses = data_by_ep[ep]
            head, left, right = positions_from_poses(poses, mat_order)
            color = colors[i]
            label = f"ep{ep:03d}"
            plot_trajectory(ax1, head, "head_mat", color, label=label, mark_ends=False)
            plot_trajectory(ax2, left, "left_wrist_mat", color, label=label, mark_ends=False)
            plot_trajectory(ax3, right, "right_wrist_mat", color, label=label, mark_ends=False)
            all_head.append(head)
            all_left.append(left)
            all_right.append(right)

        if all_head:
            set_equal_axes(ax1, np.concatenate(all_head, axis=0))
        if all_left:
            set_equal_axes(ax2, np.concatenate(all_left, axis=0))
        if all_right:
            set_equal_axes(ax3, np.concatenate(all_right, axis=0))

        ax1.legend(loc="upper left", fontsize=8)
        fig.suptitle("EE Trajectories (Combined Episodes)")
        fig.tight_layout()

        save_path = resolve_save_path(args.save, 0, multi, combined=True)
        if save_path is not None:
            fig.savefig(save_path, dpi=200)
            print(f"saved: {save_path}")
    else:
        for ep in sorted(data_by_ep.keys()):
            poses = data_by_ep[ep]
            head, left, right = positions_from_poses(poses, mat_order)

            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(1, 3, 1, projection="3d")
            ax2 = fig.add_subplot(1, 3, 2, projection="3d")
            ax3 = fig.add_subplot(1, 3, 3, projection="3d")

            plot_trajectory(ax1, head, "head_mat", "green")
            plot_trajectory(ax2, left, "left_wrist_mat", "blue")
            plot_trajectory(ax3, right, "right_wrist_mat", "red")

            fig.suptitle(f"Episode {ep} EE Trajectories")
            fig.tight_layout()

            save_path = resolve_save_path(args.save, ep, multi, combined=False)
            if save_path is not None:
                fig.savefig(save_path, dpi=200)
                print(f"saved: {save_path}")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
