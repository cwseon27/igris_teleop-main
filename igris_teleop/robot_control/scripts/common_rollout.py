from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _append_sample(store: Dict[str, List[np.ndarray]], prefix: str, sample: Dict[str, object]) -> None:
    for key, value in sample.items():
        full_key = f"{prefix}{key}"
        if value is None:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            arr = np.asarray([value], dtype=np.float32)
        else:
            arr = np.asarray(value)
        store.setdefault(full_key, []).append(arr.copy())


def _finalize_store(store: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.stack(v, axis=0) if v else np.empty((0,), dtype=np.float32) for k, v in store.items()}


def _slice_joint_dims(arr: Optional[np.ndarray], joint_indices: Optional[Sequence[int]]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if joint_indices is None or arr.ndim < 2:
        return arr
    dims = [int(idx) for idx in joint_indices]
    return arr[:, dims]


def make_hold_profile(q_ref: np.ndarray) -> Callable[[float], Dict[str, np.ndarray]]:
    q_ref = np.asarray(q_ref, dtype=np.float32).copy()
    zeros = np.zeros_like(q_ref)
    def _fn(_t: float) -> Dict[str, np.ndarray]:
        return {"q": q_ref, "dq": zeros, "tau": zeros}
    return _fn


def make_sine_profile(q_ref: np.ndarray, joint_indices: Iterable[int], amplitude: float = 0.1, frequency: float = 0.5) -> Callable[[float], Dict[str, np.ndarray]]:
    q_ref = np.asarray(q_ref, dtype=np.float32).copy()
    joint_indices = [int(i) for i in joint_indices]
    zeros = np.zeros_like(q_ref)
    omega = 2.0 * math.pi * float(frequency)

    def _fn(t: float) -> Dict[str, np.ndarray]:
        q = q_ref.copy()
        dq = zeros.copy()
        phase = float(amplitude) * math.sin(omega * t)
        vel = float(amplitude) * omega * math.cos(omega * t)
        for idx in joint_indices:
            q[idx] += phase
            dq[idx] = vel
        return {"q": q, "dq": dq, "tau": zeros}

    return _fn


def make_chirp_profile(q_ref: np.ndarray, joint_indices: Iterable[int], amplitude: float = 0.1, f0: float = 0.2, f1: float = 2.0, duration: float = 10.0) -> Callable[[float], Dict[str, np.ndarray]]:
    q_ref = np.asarray(q_ref, dtype=np.float32).copy()
    joint_indices = [int(i) for i in joint_indices]
    zeros = np.zeros_like(q_ref)
    duration = max(float(duration), 1e-3)

    def _fn(t: float) -> Dict[str, np.ndarray]:
        q = q_ref.copy()
        dq = zeros.copy()
        tau = zeros.copy()
        tt = min(max(float(t), 0.0), duration)
        inst_freq = f0 + (f1 - f0) * (tt / duration)
        phase = 2.0 * math.pi * (f0 * tt + 0.5 * (f1 - f0) * (tt ** 2) / duration)
        pos = float(amplitude) * math.sin(phase)
        vel = float(amplitude) * 2.0 * math.pi * inst_freq * math.cos(phase)
        for idx in joint_indices:
            q[idx] += pos
            dq[idx] = vel
        return {"q": q, "dq": dq, "tau": tau}

    return _fn


def _run_rollout_impl(
    controller,
    trajectory_fn: Callable[[float], Dict[str, np.ndarray]],
    duration: float,
    record_hz: float = 100.0,
    set_targets: bool = True,
    interruptible: bool = False,
) -> Tuple[Dict[str, np.ndarray], bool]:
    dt = 1.0 / float(record_hz)
    start = time.monotonic()
    next_tick = start
    store: Dict[str, List[np.ndarray]] = {"t": []}
    interrupted = False
    while True:
        try:
            now = time.monotonic()
            elapsed = now - start
            if elapsed > duration:
                break
            target = trajectory_fn(elapsed)
            if set_targets:
                controller.set_joint_targets(
                    q=target.get("q"),
                    dq=target.get("dq"),
                    tau=target.get("tau"),
                )
            cmd = controller.get_command_snapshot()
            obs = controller.get_observation_snapshot()
            store["t"].append(np.asarray([elapsed], dtype=np.float64))
            _append_sample(store, "cmd_", cmd)
            _append_sample(store, "obs_", obs)
            sleep_time = next_tick + dt - time.monotonic()
            next_tick += dt
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            if not interruptible:
                raise
            interrupted = True
            break
    return _finalize_store(store), interrupted


def run_rollout(controller, trajectory_fn: Callable[[float], Dict[str, np.ndarray]], duration: float, record_hz: float = 100.0, set_targets: bool = True) -> Dict[str, np.ndarray]:
    log, _ = _run_rollout_impl(
        controller,
        trajectory_fn=trajectory_fn,
        duration=duration,
        record_hz=record_hz,
        set_targets=set_targets,
        interruptible=False,
    )
    return log


def run_rollout_interruptible(
    controller,
    trajectory_fn: Callable[[float], Dict[str, np.ndarray]],
    duration: float,
    record_hz: float = 100.0,
    set_targets: bool = True,
) -> Tuple[Dict[str, np.ndarray], bool]:
    return _run_rollout_impl(
        controller,
        trajectory_fn=trajectory_fn,
        duration=duration,
        record_hz=record_hz,
        set_targets=set_targets,
        interruptible=True,
    )


def compute_tracking_metrics(
    log: Dict[str, np.ndarray],
    joint_indices: Optional[Sequence[int]] = None,
    cost_mode: str = "tracking",
) -> Dict[str, float]:
    q_cmd = _slice_joint_dims(log.get("cmd_q_target_pjs"), joint_indices)
    q_obs = _slice_joint_dims(log.get("obs_q_joint"), joint_indices)
    dq_cmd = _slice_joint_dims(log.get("cmd_dq_target"), joint_indices)
    dq_obs = _slice_joint_dims(log.get("obs_dq_joint"), joint_indices)
    tau_cmd = _slice_joint_dims(log.get("cmd_tau_target"), joint_indices)
    tau_obs = _slice_joint_dims(log.get("obs_tau_joint_filt"), joint_indices)
    metrics: Dict[str, float] = {}
    if q_cmd is not None and q_obs is not None and q_cmd.size and q_obs.size:
        q_err = q_cmd - q_obs
        metrics["mae_q"] = float(np.mean(np.abs(q_err)))
        metrics["rmse_q"] = float(np.sqrt(np.mean(np.square(q_err))))
        metrics["max_abs_q"] = float(np.max(np.abs(q_err)))
    if dq_cmd is not None and dq_obs is not None and dq_cmd.size and dq_obs.size:
        dq_err = dq_cmd - dq_obs
        metrics["mae_dq"] = float(np.mean(np.abs(dq_err)))
        metrics["rmse_dq"] = float(np.sqrt(np.mean(np.square(dq_err))))
    if tau_cmd is not None and tau_obs is not None and tau_cmd.size and tau_obs.size:
        tau_err = tau_cmd - tau_obs
        metrics["mae_tau"] = float(np.mean(np.abs(tau_err)))
    tau_smooth = 0.0
    if tau_obs is not None and tau_obs.shape[0] > 1:
        tau_smooth = float(np.mean(np.abs(np.diff(tau_obs, axis=0))))
    metrics["smooth_tau"] = tau_smooth

    if cost_mode == "tracking":
        metrics["cost"] = (
            1.0 * metrics.get("mae_q", 0.0)
            + 0.5 * metrics.get("rmse_q", 0.0)
            + 0.25 * metrics.get("max_abs_q", 0.0)
            + 0.2 * metrics.get("mae_dq", 0.0)
            + 0.1 * metrics.get("rmse_dq", 0.0)
        )
    elif cost_mode == "balanced":
        metrics["cost"] = (
            1.0 * metrics.get("mae_q", 0.0)
            + 0.3 * metrics.get("mae_dq", 0.0)
            + 0.05 * metrics.get("mae_tau", 0.0)
            + 0.005 * metrics.get("smooth_tau", 0.0)
        )
    elif cost_mode == "effort":
        metrics["cost"] = (
            1.0 * metrics.get("mae_q", 0.0)
            + 0.3 * metrics.get("mae_dq", 0.0)
            + 0.1 * metrics.get("mae_tau", 0.0)
            + 0.01 * metrics.get("smooth_tau", 0.0)
        )
    else:
        raise ValueError(f"Unsupported cost_mode: {cost_mode}")
    metrics["cost_mode"] = cost_mode
    return metrics


def save_tracking_plot(
    out_path: Path,
    log: Dict[str, np.ndarray],
    joint_indices: Sequence[int],
    joint_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> Optional[Path]:
    out_path = Path(out_path)
    t = np.asarray(log.get("t", np.empty((0,), dtype=np.float32))).reshape(-1)
    if t.size == 0:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    q_cmd = _slice_joint_dims(log.get("cmd_q_target_pjs"), joint_indices)
    q_obs = _slice_joint_dims(log.get("obs_q_joint"), joint_indices)
    dq_cmd = _slice_joint_dims(log.get("cmd_dq_target"), joint_indices)
    dq_obs = _slice_joint_dims(log.get("obs_dq_joint"), joint_indices)
    tau_cmd = _slice_joint_dims(log.get("cmd_tau_target"), joint_indices)
    tau_obs = _slice_joint_dims(log.get("obs_tau_joint_filt"), joint_indices)

    if joint_labels is None:
        joint_labels = [f"joint_{int(idx)}" for idx in joint_indices]
    joint_labels = [str(label) for label in joint_labels]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 9))
    series_specs = [
        ("q [rad]", q_cmd, q_obs),
        ("dq [rad/s]", dq_cmd, dq_obs),
        ("tau [Nm]", tau_cmd, tau_obs),
    ]
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(joint_labels), 1)))

    for ax, (ylabel, ref_arr, obs_arr) in zip(axes, series_specs):
        if ref_arr is None and obs_arr is None:
            ax.set_visible(False)
            continue
        for idx, label in enumerate(joint_labels):
            color = colors[idx % len(colors)]
            if ref_arr is not None and ref_arr.size:
                ax.plot(t, ref_arr[:, idx], linestyle="--", color=color, linewidth=1.2, label=f"{label} cmd")
            if obs_arr is not None and obs_arr.size:
                ax.plot(t, obs_arr[:, idx], color=color, linewidth=1.2, label=f"{label} obs")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8, loc="upper right")

    axes[-1].set_xlabel("t [s]")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_episode_npz(out_path: Path, log: Dict[str, np.ndarray], metadata: Optional[Dict[str, object]] = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **log)
    if metadata is not None:
        meta_path = out_path.with_suffix('.json')
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
