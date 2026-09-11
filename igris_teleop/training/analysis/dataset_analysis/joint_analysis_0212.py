#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("pandas/pyarrow가 필요합니다. `pip install pandas pyarrow`를 설치하세요.") from exc

import matplotlib.pyplot as plt


STATE_KEY = "observation.state"

DEFAULT_STATE_ORDER = ["obs_hand", "obs_arm", "obs_neck", "obs_waist", "obs_leg"]
DEFAULT_STATE_DIMS = {
    "obs_hand": 12,
    "obs_arm": 14,
    "obs_neck": 2,
    "obs_waist": 3,
    "obs_leg": 12,
}

GROUPS = {
    "arm": "obs_arm",
    "neck": "obs_neck",
    "waist": "obs_waist",
    "full": "obs_arm,obs_neck,obs_waist",
}


def default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[3] / "dataset" / "AutoBagger"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "outputs"


def load_info(info_path: Path) -> dict:
    try:
        return json.loads(info_path.read_text())
    except Exception:
        return {}


def parse_range_list(text: str, total: int | None = None) -> list[int]:
    items: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":", 1)
            start = int(a) if a else 0
            if b:
                end = int(b)
            else:
                if total is None:
                    raise ValueError("Range end required when total is unknown.")
                end = total
            items.update(range(start, end))
        else:
            items.add(int(part))
    return sorted(items)


def parse_split(info: dict, split: str | None) -> list[int] | None:
    if not split:
        return None
    splits = info.get("splits", {}) if isinstance(info, dict) else {}
    if split not in splits:
        raise KeyError(f"Split '{split}' not found in meta/info.json")
    spec = splits[split]
    if isinstance(spec, str):
        return parse_range_list(spec, total=info.get("total_episodes"))
    if isinstance(spec, (list, tuple)):
        eps: set[int] = set()
        for s in spec:
            if isinstance(s, int):
                eps.add(int(s))
            elif isinstance(s, str):
                eps.update(parse_range_list(s, total=info.get("total_episodes")))
        return sorted(eps)
    return None


def parquet_paths(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"dataset data dir not found: {data_dir}")
    return sorted(data_dir.glob("chunk-*/*.parquet"))


def build_state_slices(order: list[str], dims: dict[str, int]) -> tuple[dict[str, slice], int]:
    idx = 0
    slices: dict[str, slice] = {}
    for name in order:
        if name not in dims:
            raise KeyError(f"Missing dimension for '{name}'")
        d = int(dims[name])
        slices[name] = slice(idx, idx + d)
        idx += d
    return slices, idx


def collect_episode_states(paths: list[Path], episodes: set[int] | None) -> dict[int, np.ndarray]:
    data_by_ep: dict[int, list[np.ndarray]] = {}
    cols = ["episode_index", "frame_index", STATE_KEY]

    for path in paths:
        df = pd.read_parquet(path, columns=cols).reset_index(drop=True)
        if episodes is not None:
            df = df[df["episode_index"].isin(episodes)]
        if df.empty:
            continue
        df = df.sort_values(["episode_index", "frame_index"], kind="mergesort")
        for ep, group in df.groupby("episode_index", sort=True):
            arr = np.stack(group[STATE_KEY].to_list(), axis=0)
            data_by_ep.setdefault(int(ep), []).append(arr)

    merged: dict[int, np.ndarray] = {}
    for ep, chunks in data_by_ep.items():
        merged[ep] = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
    return merged


def split_groups(states: dict[int, np.ndarray], slices: dict[str, slice]) -> dict[str, dict[int, np.ndarray]]:
    groups: dict[str, dict[int, np.ndarray]] = {}
    for group_name, fields in GROUPS.items():
        names = [n.strip() for n in fields.split(",")]
        grp: dict[int, np.ndarray] = {}
        for ep, arr in states.items():
            parts = [arr[:, slices[name]] for name in names]
            grp[ep] = np.concatenate(parts, axis=1)
        groups[group_name] = grp
    return groups


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    if seq.shape[0] == target_len:
        return seq
    x_old = np.linspace(0.0, 1.0, seq.shape[0])
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, seq.shape[1]), dtype=seq.dtype)
    for j in range(seq.shape[1]):
        out[:, j] = np.interp(x_new, x_old, seq[:, j])
    return out


def preprocess_sequences(
    seqs: list[np.ndarray],
    unwrap: bool,
    relative: bool,
    norm: str,
) -> list[np.ndarray]:
    processed: list[np.ndarray] = []
    for seq in seqs:
        x = np.asarray(seq, dtype=np.float64)
        if unwrap:
            x = np.unwrap(x, axis=0)
        if relative and x.shape[0] > 0:
            x = x - x[0]
        processed.append(x)

    if norm == "none":
        return processed

    all_data = np.concatenate(processed, axis=0)
    if norm == "zscore":
        mean = all_data.mean(axis=0)
        std = all_data.std(axis=0)
        std[std == 0] = 1.0
        return [(x - mean) / std for x in processed]
    if norm == "minmax":
        minv = all_data.min(axis=0)
        maxv = all_data.max(axis=0)
        denom = maxv - minv
        denom[denom == 0] = 1.0
        return [(x - minv) / denom for x in processed]
    raise ValueError(f"Unknown norm: {norm}")


def parse_band(value: float | None, n: int, m: int) -> int | None:
    if value is None:
        return None
    if value <= 0:
        return None
    if value <= 1.0:
        return int(math.ceil(value * max(n, m)))
    return int(value)


def dtw_distance(x: np.ndarray, y: np.ndarray, band: float | None = None) -> float:
    n, m = x.shape[0], y.shape[0]
    if n == 0 or m == 0:
        return float("inf")
    b = parse_band(band, n, m)
    inf = 1e18
    dp = np.full((n + 1, m + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = 1
        j_end = m
        if b is not None:
            j_start = max(1, i - b)
            j_end = min(m, i + b)
        for j in range(j_start, j_end + 1):
            cost = np.sum((x[i - 1] - y[j - 1]) ** 2)
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    path_len = n + m
    return math.sqrt(dp[n, m] / max(path_len, 1))


def dtw_path(x: np.ndarray, y: np.ndarray, band: float | None = None) -> list[tuple[int, int]]:
    n, m = x.shape[0], y.shape[0]
    if n == 0 or m == 0:
        return []
    b = parse_band(band, n, m)
    inf = 1e18
    dp = np.full((n + 1, m + 1), inf, dtype=np.float64)
    phi = np.full((n + 1, m + 1), -1, dtype=np.int8)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = 1
        j_end = m
        if b is not None:
            j_start = max(1, i - b)
            j_end = min(m, i + b)
        for j in range(j_start, j_end + 1):
            cost = np.sum((x[i - 1] - y[j - 1]) ** 2)
            choices = [dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]]
            k = int(np.argmin(choices))
            dp[i, j] = cost + choices[k]
            phi[i, j] = k  # 0:diag, 1:up, 2:left

    i, j = n, m
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = phi[i, j]
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        elif step == 2:
            j -= 1
        else:
            break
    path.reverse()
    return path


def soft_dtw_distance(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    n, m = x.shape[0], y.shape[0]
    if n == 0 or m == 0:
        return float("inf")
    if gamma <= 0:
        raise ValueError("gamma must be > 0 for soft-DTW")

    D = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        diff = y - x[i]
        D[i] = np.sum(diff * diff, axis=1)

    R = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    R[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r0 = -R[i - 1, j - 1] / gamma
            r1 = -R[i - 1, j] / gamma
            r2 = -R[i, j - 1] / gamma
            rmax = max(r0, r1, r2)
            softmin = -gamma * (rmax + math.log(math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)))
            R[i, j] = D[i - 1, j - 1] + softmin
    return float(R[n, m])


def pairwise_distance_matrix(
    seqs: list[np.ndarray],
    method: str,
    gamma: float,
    band: float | None,
) -> np.ndarray:
    n = len(seqs)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if method == "softdtw":
                d = soft_dtw_distance(seqs[i], seqs[j], gamma=gamma)
            else:
                d = dtw_distance(seqs[i], seqs[j], band=band)
            D[i, j] = d
            D[j, i] = d
    return D


def medoid_index(D: np.ndarray) -> int:
    if D.shape[0] == 1:
        return 0
    mean_dist = (np.sum(D, axis=1) - np.diag(D)) / (D.shape[0] - 1)
    return int(np.argmin(mean_dist))


def dba_barycenter(
    seqs: list[np.ndarray],
    init: np.ndarray,
    iters: int = 5,
    band: float | None = None,
) -> np.ndarray:
    bary = np.asarray(init, dtype=np.float64)
    for _ in range(iters):
        acc: list[list[np.ndarray]] = [[] for _ in range(bary.shape[0])]
        for seq in seqs:
            path = dtw_path(bary, seq, band=band)
            for i, j in path:
                acc[i].append(seq[j])
        new_bary = bary.copy()
        for i, pts in enumerate(acc):
            if pts:
                new_bary[i] = np.mean(pts, axis=0)
        bary = new_bary
    return bary


def classical_mds(D: np.ndarray, n_components: int = 2) -> np.ndarray:
    n = D.shape[0]
    if n == 1:
        return np.zeros((1, n_components), dtype=np.float64)
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals = np.maximum(eigvals[:n_components], 0.0)
    coords = eigvecs[:, :n_components] * np.sqrt(eigvals)
    return coords


def average_linkage(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    if n <= 1:
        return np.zeros((0, 4), dtype=np.float64)

    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    active = list(range(n))
    linkage = []
    next_idx = n

    def cluster_distance(a: int, b: int) -> float:
        members_a = clusters[a]
        members_b = clusters[b]
        total = 0.0
        count = 0
        for i in members_a:
            for j in members_b:
                total += D[i, j]
                count += 1
        return total / max(count, 1)

    while len(active) > 1:
        best = (None, None, float("inf"))
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                d = cluster_distance(a, b)
                if d < best[2]:
                    best = (a, b, d)
        a, b, d = best
        members = clusters[a] + clusters[b]
        linkage.append([a, b, d, len(members)])
        clusters[next_idx] = members
        active = [x for x in active if x not in (a, b)]
        active.append(next_idx)
        next_idx += 1

    return np.asarray(linkage, dtype=np.float64)


@dataclass
class _Node:
    x: float
    y: float
    left: int | None
    right: int | None


def plot_dendrogram(linkage: np.ndarray, labels: list[str], out_path: Path) -> None:
    n = len(labels)
    nodes: dict[int, _Node] = {}
    for i in range(n):
        nodes[i] = _Node(x=float(i), y=0.0, left=None, right=None)

    for i, row in enumerate(linkage):
        a, b, dist, _cnt = row
        a = int(a)
        b = int(b)
        left = nodes[a]
        right = nodes[b]
        x = (left.x + right.x) / 2.0
        y = float(dist)
        nodes[n + i] = _Node(x=x, y=y, left=a, right=b)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, row in enumerate(linkage):
        a, b, dist, _cnt = row
        a = int(a)
        b = int(b)
        left = nodes[a]
        right = nodes[b]
        x1, y1 = left.x, left.y
        x2, y2 = right.x, right.y
        y = float(dist)
        ax.plot([x1, x1], [y1, y], color="black")
        ax.plot([x2, x2], [y2, y], color="black")
        ax.plot([x1, x2], [y, y], color="black")
        nodes[n + i].y = y

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("distance")
    ax.set_title("Average Linkage Dendrogram")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_heatmap(D: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(D, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mds(coords: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords[:, 0], coords[:, 1], s=30, color="tab:blue")
    for i, lbl in enumerate(labels):
        ax.text(coords[i, 0], coords[i, 1], lbl, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_trajectory_overlay(
    seqs: list[np.ndarray],
    bary: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
    max_episodes: int | None = None,
) -> None:
    if not seqs:
        return
    n_eps = len(seqs)
    if max_episodes is not None and n_eps > max_episodes:
        seqs = seqs[:max_episodes]
        labels = labels[:max_episodes]
    joints = bary.shape[1]
    ncols = min(4, joints)
    nrows = int(math.ceil(joints / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2), squeeze=False)
    t_b = np.arange(bary.shape[0])
    for j in range(joints):
        r = j // ncols
        c = j % ncols
        ax = axes[r][c]
        for seq in seqs:
            t = np.arange(seq.shape[0])
            ax.plot(t, seq[:, j], color="tab:blue", alpha=0.2, linewidth=0.8)
        ax.plot(t_b, bary[:, j], color="black", linewidth=1.8)
        ax.set_title(f"joint {j}", fontsize=8)
    for k in range(joints, nrows * ncols):
        r = k // ncols
        c = k % ncols
        axes[r][c].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def align_sequences_to_barycenter(
    seqs: list[np.ndarray],
    bary: np.ndarray,
    band: float | None,
) -> np.ndarray:
    if not seqs:
        return np.zeros((0, bary.shape[0], bary.shape[1]), dtype=np.float64)
    aligned = []
    T = bary.shape[0]
    for seq in seqs:
        acc = np.zeros((T, seq.shape[1]), dtype=np.float64)
        counts = np.zeros((T,), dtype=np.int64)
        path = dtw_path(bary, seq, band=band)
        for i, j in path:
            acc[i] += seq[j]
            counts[i] += 1
        mask = counts > 0
        if np.any(mask):
            acc[mask] = acc[mask] / counts[mask, None]
        acc[~mask] = bary[~mask]
        aligned.append(acc)
    return np.stack(aligned, axis=0)


def segment_edges(length: int, segments: int) -> list[tuple[int, int]]:
    if segments <= 1 or length <= 1:
        return [(0, length)]
    edges = np.linspace(0, length, segments + 1)
    edges = np.unique(np.round(edges).astype(int))
    if edges[0] != 0:
        edges = np.insert(edges, 0, 0)
    if edges[-1] != length:
        edges = np.append(edges, length)
    pairs = []
    for i in range(len(edges) - 1):
        start = int(edges[i])
        end = int(edges[i + 1])
        if end > start:
            pairs.append((start, end))
    if not pairs:
        pairs = [(0, length)]
    return pairs


def icc_a1(Y: np.ndarray) -> float:
    n, k = Y.shape
    if n < 2 or k < 2:
        return float("nan")
    mean_t = Y.mean(axis=1, keepdims=True)
    mean_r = Y.mean(axis=0, keepdims=True)
    grand = Y.mean()
    ssr = k * np.sum((mean_t - grand) ** 2)
    ssc = n * np.sum((mean_r - grand) ** 2)
    sse = np.sum((Y - mean_t - mean_r + grand) ** 2)
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + (k * (msc - mse) / n)
    if denom == 0:
        return float("nan")
    return float((msr - mse) / denom)


def icc_3k(Y: np.ndarray) -> float:
    n, k = Y.shape
    if n < 2 or k < 2:
        return float("nan")
    mean_t = Y.mean(axis=1, keepdims=True)
    mean_r = Y.mean(axis=0, keepdims=True)
    grand = Y.mean()
    ssr = k * np.sum((mean_t - grand) ** 2)
    sse = np.sum((Y - mean_t - mean_r + grand) ** 2)
    msr = ssr / (n - 1)
    mse = sse / ((n - 1) * (k - 1))
    if msr == 0:
        return float("nan")
    return float((msr - mse) / msr)


def plot_matrix_heatmap(
    mat: np.ndarray,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    xlabels: list[str] | None = None,
    ylabels: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlabels:
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
    if ylabels:
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_feature_arrays(aligned: np.ndarray, settle_tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if aligned.size == 0:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0, 0), dtype=np.int64),
        )
    n_eps, T, J = aligned.shape
    rom = aligned.max(axis=1) - aligned.min(axis=1)
    peak = np.max(np.abs(aligned), axis=1)
    settle_idx = np.zeros((n_eps, J), dtype=np.int64)
    for i in range(n_eps):
        seq = aligned[i]
        for j in range(J):
            target = seq[-1, j]
            tol = settle_tol * (rom[i, j] if rom[i, j] > 1e-9 else 1.0)
            idx = T - 1
            for t in range(T):
                if np.all(np.abs(seq[t:, j] - target) <= tol):
                    idx = t
                    break
            settle_idx[i, j] = idx
    return rom, peak, settle_idx


def compute_feature_rows(
    aligned: np.ndarray,
    episodes: list[int],
    group_name: str,
    fps: float | None,
    settle_tol: float,
) -> list[dict]:
    rows: list[dict] = []
    if aligned.size == 0:
        return rows
    n_eps, _T, J = aligned.shape
    rom, peak, settle_idx = compute_feature_arrays(aligned, settle_tol=settle_tol)
    if fps and fps > 0:
        settle_time = settle_idx.astype(np.float64) / fps
    else:
        settle_time = settle_idx.astype(np.float64)
    for i in range(n_eps):
        for j in range(J):
            rows.append(
                {
                    "episode": episodes[i],
                    "group": group_name,
                    "joint": j,
                    "rom": float(rom[i, j]),
                    "peak": float(peak[i, j]),
                    "settling_idx": int(settle_idx[i, j]),
                    "settling_time": float(settle_time[i, j]),
                }
            )
    return rows

def compute_outliers(distances: np.ndarray) -> tuple[list[int], list[int], np.ndarray]:
    n = distances.shape[0]
    if n == 0:
        return [], [], np.zeros((0,), dtype=np.float64)
    med = np.median(distances)
    mad = np.median(np.abs(distances - med))
    if mad == 0:
        robust_z = np.zeros_like(distances)
    else:
        robust_z = 0.6745 * (distances - med) / mad
    robust_outliers = np.where(robust_z > 3.0)[0].tolist()
    k = max(1, int(math.ceil(0.05 * n)))
    top_idx = np.argsort(distances)[::-1][:k].tolist()
    return top_idx, robust_outliers, robust_z


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Episode-wise joint similarity analysis.")
    parser.add_argument("--dataset-root", type=str, default=str(default_dataset_root()))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--episodes", type=str, default=None, help="override split: e.g., 0,1,2 or 0:10")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--band", type=float, default=None, help="DTW band (int or ratio <=1)")
    parser.add_argument("--resample-len", type=int, default=None)
    parser.add_argument("--method", type=str, choices=["auto", "softdtw", "dtw"], default="auto")
    parser.add_argument("--norm", type=str, choices=["zscore", "minmax", "none"], default="zscore")
    parser.add_argument("--no-unwrap", action="store_true")
    parser.add_argument("--no-relative", action="store_true")
    parser.add_argument("--dba-iters", type=int, default=5)
    parser.add_argument("--segments", type=int, default=4, help="segments for ICC over time")
    parser.add_argument("--settle-tol", type=float, default=0.05, help="settling tolerance ratio of ROM")
    parser.add_argument("--embed", type=str, choices=["mds", "none"], default="mds")
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir()))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--state-order", type=str, default=",".join(DEFAULT_STATE_ORDER))
    parser.add_argument("--state-dims", type=str, default=",".join(str(DEFAULT_STATE_DIMS[n]) for n in DEFAULT_STATE_ORDER))
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    info = load_info(dataset_root / "meta" / "info.json")
    fps = info.get("fps", None)
    try:
        fps = float(fps) if fps is not None else None
    except Exception:
        fps = None

    if args.episodes:
        eps = parse_range_list(args.episodes, total=info.get("total_episodes"))
    else:
        eps = parse_split(info, args.split)
    if eps is not None and args.max_episodes is not None:
        eps = eps[: args.max_episodes]

    order = [x.strip() for x in args.state_order.split(",") if x.strip()]
    dims_list = [int(x.strip()) for x in args.state_dims.split(",") if x.strip()]
    if len(order) != len(dims_list):
        raise ValueError("state-order and state-dims length mismatch")
    dims = {name: dims_list[i] for i, name in enumerate(order)}
    slices, total_dim = build_state_slices(order, dims)

    paths = parquet_paths(dataset_root)
    if not paths:
        print("No parquet files found.", file=sys.stderr)
        return 1

    episodes_set = set(eps) if eps is not None else None
    states = collect_episode_states(paths, episodes_set)
    if not states:
        print("No matching episode data found.", file=sys.stderr)
        return 1

    # validate state dim
    any_state = next(iter(states.values()))
    if any_state.shape[1] < total_dim:
        raise ValueError(f"observation.state dim too small: {any_state.shape[1]} < {total_dim}")

    groups = split_groups(states, slices)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir).expanduser().resolve() / run_id
    ensure_dir(out_root)

    method = args.method
    if method == "auto":
        method = "softdtw"

    summary = {
        "dataset_root": str(dataset_root),
        "episodes": sorted(states.keys()),
        "method": method,
        "gamma": args.gamma,
        "band": args.band,
        "resample_len": args.resample_len,
        "norm": args.norm,
        "unwrap": not args.no_unwrap,
        "relative": not args.no_relative,
        "state_order": order,
        "state_dims": dims,
    }

    metrics_rows = []
    outliers_lines = []
    icc_rows_all = []
    feature_rows_all = []
    feature_icc_rows = []
    feature_stats_rows = []

    for group_name, ep_map in groups.items():
        episodes_sorted = sorted(ep_map.keys())
        seqs = [ep_map[ep] for ep in episodes_sorted]
        seqs = preprocess_sequences(seqs, unwrap=not args.no_unwrap, relative=not args.no_relative, norm=args.norm)

        lengths = [s.shape[0] for s in seqs]
        target_len = args.resample_len
        if target_len is None and len(set(lengths)) > 1:
            target_len = int(np.median(lengths))
        if target_len is not None:
            seqs_rs = [resample_sequence(s, target_len) for s in seqs]
        else:
            seqs_rs = seqs

        # distance matrix
        D = pairwise_distance_matrix(seqs_rs, method=method, gamma=args.gamma, band=args.band)
        np.save(out_root / f"distance_matrix_{group_name}.npy", D)

        # medoid + DBA barycenter
        medoid = medoid_index(D)
        bary = dba_barycenter(seqs_rs, init=seqs_rs[medoid], iters=args.dba_iters, band=args.band)
        np.save(out_root / f"barycenter_{group_name}.npy", bary)

        # distances to barycenter
        d_to_bary = []
        for seq in seqs_rs:
            if method == "softdtw":
                d = soft_dtw_distance(seq, bary, gamma=args.gamma)
            else:
                d = dtw_distance(seq, bary, band=args.band)
            d_to_bary.append(d)
        d_to_bary = np.asarray(d_to_bary, dtype=np.float64)

        # mean pairwise distance per episode
        if D.shape[0] > 1:
            mean_pairwise = (np.sum(D, axis=1) - np.diag(D)) / (D.shape[0] - 1)
        else:
            mean_pairwise = np.zeros((1,), dtype=np.float64)

        top_idx, robust_idx, robust_z = compute_outliers(d_to_bary)
        outliers_lines.append(f"[{group_name}] top5%: {[episodes_sorted[i] for i in top_idx]}")
        outliers_lines.append(f"[{group_name}] robust_z>3: {[episodes_sorted[i] for i in robust_idx]}")

        dispersion_mean = float(np.mean(d_to_bary)) if d_to_bary.size else float("nan")
        dispersion_median = float(np.median(d_to_bary)) if d_to_bary.size else float("nan")

        # save plots
        labels = [f"ep{ep:03d}" for ep in episodes_sorted]
        plot_heatmap(D, labels, out_root / f"distance_heatmap_{group_name}.png", f"{group_name} distance heatmap")
        linkage = average_linkage(D)
        plot_dendrogram(linkage, labels, out_root / f"dendrogram_{group_name}.png")
        if args.embed == "mds":
            coords = classical_mds(D, n_components=2)
            plot_mds(coords, labels, out_root / f"mds_{group_name}.png", f"{group_name} MDS")
        plot_trajectory_overlay(
            seqs_rs,
            bary,
            labels,
            out_root / f"trajectory_overlay_{group_name}.png",
            f"{group_name} trajectories + barycenter",
        )

        # alignment to barycenter for ICC/feature analysis
        aligned = align_sequences_to_barycenter(seqs_rs, bary, band=args.band)
        if aligned.size > 0:
            var_tp = np.var(aligned, axis=0)  # (T, J)
            np.save(out_root / f"variance_time_{group_name}.npy", var_tp)
            joint_labels = [f"j{j}" for j in range(var_tp.shape[1])]
            plot_matrix_heatmap(
                var_tp.T,
                out_root / f"variance_heatmap_{group_name}.png",
                f"{group_name} timepoint variance",
                xlabel="time",
                ylabel="joint",
                ylabels=joint_labels,
            )

            segments = segment_edges(aligned.shape[1], args.segments)
            seg_labels = [f"s{i}({s}-{e})" for i, (s, e) in enumerate(segments)]
            J = aligned.shape[2]
            S = len(segments)
            icc_a1_mat = np.full((J, S), np.nan, dtype=np.float64)
            icc_3k_mat = np.full((J, S), np.nan, dtype=np.float64)
            var_seg_mat = np.full((J, S), np.nan, dtype=np.float64)

            for j in range(J):
                for s_idx, (start, end) in enumerate(segments):
                    if end <= start:
                        continue
                    Y = aligned[:, start:end, j].T  # (time, episode)
                    icc_a1_mat[j, s_idx] = icc_a1(Y)
                    icc_3k_mat[j, s_idx] = icc_3k(Y)
                    var_seg_mat[j, s_idx] = np.var(aligned[:, start:end, j], axis=0).mean()
                    icc_rows_all.append(
                        {
                            "group": group_name,
                            "joint": j,
                            "segment": s_idx,
                            "start": start,
                            "end": end,
                            "icc_a1": float(icc_a1_mat[j, s_idx]),
                            "icc_3k": float(icc_3k_mat[j, s_idx]),
                            "var_mean": float(var_seg_mat[j, s_idx]),
                        }
                    )

            plot_matrix_heatmap(
                icc_a1_mat,
                out_root / f"icc_a1_{group_name}.png",
                f"{group_name} ICC(A,1)",
                xlabel="segment",
                ylabel="joint",
                xlabels=seg_labels,
                ylabels=joint_labels,
            )
            plot_matrix_heatmap(
                icc_3k_mat,
                out_root / f"icc_3k_{group_name}.png",
                f"{group_name} ICC(3,k)",
                xlabel="segment",
                ylabel="joint",
                xlabels=seg_labels,
                ylabels=joint_labels,
            )

            # feature analysis + ICC
            feature_rows = compute_feature_rows(
                aligned,
                episodes_sorted,
                group_name,
                fps=fps,
                settle_tol=args.settle_tol,
            )
            feature_rows_all.extend(feature_rows)

            rom, peak, settle_idx = compute_feature_arrays(aligned, settle_tol=args.settle_tol)
            if fps and fps > 0:
                settle_time = settle_idx.astype(np.float64) / fps
            else:
                settle_time = settle_idx.astype(np.float64)

            feature_mats = {
                "rom": rom,
                "peak": peak,
                "settling_time": settle_time,
            }
            for fname, mat in feature_mats.items():
                if mat.size == 0:
                    continue
                Y = mat.T  # joints x episodes
                feature_icc_rows.append(
                    {
                        "group": group_name,
                        "feature": fname,
                        "icc_a1": icc_a1(Y),
                        "icc_3k": icc_3k(Y),
                    }
                )
                for j in range(mat.shape[1]):
                    vals = mat[:, j]
                    mean = float(np.mean(vals))
                    std = float(np.std(vals))
                    cv = std / (abs(mean) + 1e-9)
                    feature_stats_rows.append(
                        {
                            "group": group_name,
                            "joint": j,
                            "feature": fname,
                            "mean": mean,
                            "std": std,
                            "cv": cv,
                        }
                    )

        # append metrics
        for i, ep in enumerate(episodes_sorted):
            metrics_rows.append(
                {
                    "episode": ep,
                    "group": group_name,
                    "length": int(seqs_rs[i].shape[0]),
                    "barycenter_distance": float(d_to_bary[i]),
                    "mean_pairwise_distance": float(mean_pairwise[i]),
                    "robust_z": float(robust_z[i]),
                }
            )

        summary[f"{group_name}_dispersion_mean"] = dispersion_mean
        summary[f"{group_name}_dispersion_median"] = dispersion_median

    # save metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_root / "metrics.csv", index=False)
    (out_root / "metrics.json").write_text(json.dumps(metrics_rows, indent=2))
    if icc_rows_all:
        pd.DataFrame(icc_rows_all).to_csv(out_root / "icc_timeseries.csv", index=False)
    if feature_rows_all:
        pd.DataFrame(feature_rows_all).to_csv(out_root / "features.csv", index=False)
    if feature_icc_rows:
        pd.DataFrame(feature_icc_rows).to_csv(out_root / "icc_features.csv", index=False)
    if feature_stats_rows:
        pd.DataFrame(feature_stats_rows).to_csv(out_root / "feature_joint_stats.csv", index=False)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_root / "outliers.txt").write_text("\n".join(outliers_lines))
    (out_root / "episodes.json").write_text(json.dumps(summary["episodes"], indent=2))

    print(f"Saved outputs to: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
