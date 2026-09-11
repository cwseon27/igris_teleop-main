#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_LOG_DIR = (Path(__file__).resolve().parents[1] / "outputs" / "inference_logs").resolve()

HAND_LEN = 12
ARM_LEN = 14
NECK_LEN = 2
WAIST_LEN = 3
DEFAULT_N_ACTION_STEP = 50
DEFAULT_BLENDING_HORIZON = 10
DEFAULT_INFERENCE_DELAY_STEPS = 5

ARM_JOINT_LABELS = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow_pitch",
    "l_wrist_yaw",
    "l_wrist_roll",
    "l_wrist_pitch",
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow_pitch",
    "r_wrist_yaw",
    "r_wrist_roll",
    "r_wrist_pitch",
]
NECK_JOINT_LABELS = ["neck_yaw", "neck_pitch"]
WAIST_JOINT_LABELS = ["waist_yaw", "waist_roll", "waist_pitch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize an inference action log saved as .npz."
    )
    parser.add_argument(
        "npz",
        nargs="?",
        help="Path to the .npz log file. If omitted, the latest file in outputs/inference_logs is used.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional output image path. If omitted and --show is not set, a PNG is saved next to the input file.",
        default=False,
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in a window.",
    )
    parser.add_argument(
        "--dims",
        type=str,
        default=None,
        help="Comma-separated action indices to plot, for example: 0,1,12,26",
    )
    parser.add_argument(
        "--joints",
        type=str,
        default=None,
        help=(
            "Comma-separated joint selectors. Supports labels like "
            "l_wrist_roll, neck_pitch, waist_yaw, group selectors like "
            "left_arm/right_arm, or local selectors like hand:3, arm:5, "
            "neck:1, waist:2."
        ),
    )
    parser.add_argument(
        "--list-joints",
        action="store_true",
        help="Print available action joint labels for the selected log and exit.",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=None,
        help="Minimum time in seconds to include.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Maximum time in seconds to include.",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "all"),
        default="auto",
        help="Use body-part grouping for 28/31-dim actions, or a single axis for all dimensions.",
    )
    parser.add_argument(
        "--hide-schedule",
        action="store_true",
        help="Do not overlay inferred/saved inference start and blending spans.",
    )
    parser.add_argument(
        "--n-action-step",
        type=int,
        default=None,
        help="Override the action chunk length used to infer schedule overlays for older logs.",
    )
    parser.add_argument(
        "--blend-steps",
        type=int,
        default=None,
        help="Override the blending horizon used to infer schedule overlays for older logs.",
    )
    parser.add_argument(
        "--delay-steps",
        type=int,
        default=None,
        help="Override the LiPo time-delay steps used to infer schedule overlays for older logs.",
    )
    parser.add_argument(
        "--inference-dt",
        type=float,
        default=None,
        help="Override the per-step inference dt in seconds used to infer schedule overlays for older logs.",
    )
    return parser.parse_args()


def resolve_npz_path(raw_path: str | None) -> Path:
    if raw_path:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"NPZ file not found: {path}")
        return path

    candidates = sorted(DEFAULT_LOG_DIR.glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No .npz files found in {DEFAULT_LOG_DIR}")
    return candidates[-1]


def load_npz_log(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}

    chunk_time = arrays.get("chunk_time", arrays.get("policy_time", np.empty((0,), dtype=np.float32)))
    chunk_action = arrays.get("chunk_action", arrays.get("policy_action", np.empty((0, 0), dtype=np.float32)))
    publish_time = arrays.get("publish_time", np.empty((0,), dtype=np.float32))
    publish_action = arrays.get("publish_action", np.empty((0, 0), dtype=np.float32))

    if chunk_action.ndim == 1:
        chunk_action = chunk_action.reshape(-1, 1)
    if publish_action.ndim == 1:
        publish_action = publish_action.reshape(-1, 1)

    action_dim_arr = arrays.get("action_dim")
    if action_dim_arr is not None and action_dim_arr.size > 0:
        action_dim = int(action_dim_arr.reshape(-1)[0])
    elif chunk_action.ndim == 2 and chunk_action.shape[1] > 0:
        action_dim = int(chunk_action.shape[1])
    elif publish_action.ndim == 2 and publish_action.shape[1] > 0:
        action_dim = int(publish_action.shape[1])
    else:
        raise ValueError(f"Could not infer action_dim from {path}")

    return {
        "chunk_time": np.asarray(chunk_time, dtype=np.float32).reshape(-1),
        "chunk_action": np.asarray(chunk_action, dtype=np.float32),
        "publish_time": np.asarray(publish_time, dtype=np.float32).reshape(-1),
        "publish_action": np.asarray(publish_action, dtype=np.float32),
        "action_dim": np.asarray([action_dim], dtype=np.int32),
        "chunk_apply_time": np.asarray(
            arrays.get("chunk_apply_time", np.empty((0,), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1),
        "chunk_apply_step": np.asarray(
            arrays.get("chunk_apply_step", np.empty((0,), dtype=np.int32)),
            dtype=np.int32,
        ).reshape(-1),
        "n_action_step": np.asarray(
            arrays.get("n_action_step", np.empty((0,), dtype=np.int32)),
            dtype=np.int32,
        ).reshape(-1),
        "blending_horizon": np.asarray(
            arrays.get("blending_horizon", np.empty((0,), dtype=np.int32)),
            dtype=np.int32,
        ).reshape(-1),
        "request_step": np.asarray(
            arrays.get("request_step", np.empty((0,), dtype=np.int32)),
            dtype=np.int32,
        ).reshape(-1),
        "inference_delay_steps": np.asarray(
            arrays.get("inference_delay_steps", np.empty((0,), dtype=np.int32)),
            dtype=np.int32,
        ).reshape(-1),
        "inference_dt": np.asarray(
            arrays.get("inference_dt", np.empty((0,), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1),
    }


def apply_time_window(
    time_values: np.ndarray,
    action_values: np.ndarray,
    t_min: float | None,
    t_max: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if time_values.size == 0 or action_values.size == 0:
        return time_values, action_values

    mask = np.ones_like(time_values, dtype=bool)
    if t_min is not None:
        mask &= time_values >= t_min
    if t_max is not None:
        mask &= time_values <= t_max
    return time_values[mask], action_values[mask]


def parse_dims(raw_dims: str | None, action_dim: int) -> list[int] | None:
    if not raw_dims:
        return None

    dims: list[int] = []
    for chunk in raw_dims.split(","):
        token = chunk.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0 or idx >= action_dim:
            raise ValueError(f"Dimension index out of range: {idx} (action_dim={action_dim})")
        if idx not in dims:
            dims.append(idx)
    if not dims:
        raise ValueError("No valid dimensions were provided.")
    return dims


def normalize_selector(token: str) -> str:
    return token.strip().lower().replace("-", "_").replace(" ", "_")


def build_joint_parts(action_dim: int) -> list[tuple[str, list[str]]]:
    base_dim = HAND_LEN + ARM_LEN + NECK_LEN
    if action_dim == base_dim:
        return [
            ("hand", [f"hand_{idx}" for idx in range(HAND_LEN)]),
            ("arm", ARM_JOINT_LABELS),
            ("neck", NECK_JOINT_LABELS),
        ]

    if action_dim == base_dim + WAIST_LEN:
        return [
            ("hand", [f"hand_{idx}" for idx in range(HAND_LEN)]),
            ("arm", ARM_JOINT_LABELS),
            ("neck", NECK_JOINT_LABELS),
            ("waist", WAIST_JOINT_LABELS),
        ]

    return [("all", [f"dim_{idx}" for idx in range(action_dim)])]


def build_joint_index(action_dim: int) -> tuple[list[str], dict[str, int], dict[int, str]]:
    labels: list[str] = []
    selector_to_dim: dict[str, int] = {}
    dim_to_label: dict[int, str] = {}

    dim = 0
    for part_name, part_labels in build_joint_parts(action_dim):
        for local_idx, label in enumerate(part_labels):
            labels.append(label)
            dim_to_label[dim] = label

            aliases = {
                str(dim),
                f"d{dim}",
                f"dim_{dim}",
                label,
            }
            if part_name != "all":
                aliases.add(f"{part_name}:{local_idx}")

            for alias in aliases:
                selector_to_dim[normalize_selector(alias)] = dim
            dim += 1

    if dim != action_dim:
        raise ValueError(f"Joint label mapping mismatch: expected {action_dim}, got {dim}")

    return labels, selector_to_dim, dim_to_label


def build_selector_groups(action_dim: int) -> dict[str, tuple[str, list[int]]]:
    groups: dict[str, tuple[str, list[int]]] = {}

    def register_group(display_name: str, dims: list[int], *aliases: str) -> None:
        if not dims:
            return
        entry = (display_name, dims)
        for alias in aliases:
            groups[normalize_selector(alias)] = entry

    base_dim = HAND_LEN + ARM_LEN + NECK_LEN
    if action_dim not in (base_dim, base_dim + WAIST_LEN):
        register_group("all", list(range(action_dim)), "all")
        return groups

    hand_dims = list(range(0, HAND_LEN))
    arm_start = HAND_LEN
    arm_dims = list(range(arm_start, arm_start + ARM_LEN))
    neck_start = arm_start + ARM_LEN
    neck_dims = list(range(neck_start, neck_start + NECK_LEN))

    register_group("hand", hand_dims, "hand")
    register_group("arm", arm_dims, "arm")
    register_group("neck", neck_dims, "neck")

    left_arm_dims = [
        arm_start + local_idx
        for local_idx, label in enumerate(ARM_JOINT_LABELS)
        if label.startswith("l_")
    ]
    right_arm_dims = [
        arm_start + local_idx
        for local_idx, label in enumerate(ARM_JOINT_LABELS)
        if label.startswith("r_")
    ]
    register_group("left_arm", left_arm_dims, "left_arm", "leftarm", "l_arm", "arm_left")
    register_group("right_arm", right_arm_dims, "right_arm", "rightarm", "r_arm", "arm_right")

    if action_dim == base_dim + WAIST_LEN:
        waist_start = neck_start + NECK_LEN
        waist_dims = list(range(waist_start, waist_start + WAIST_LEN))
        register_group("waist", waist_dims, "waist")

    return groups


def parse_joint_selectors(
    raw_joints: str | None,
    action_dim: int,
    selector_to_dim: dict[str, int],
    selector_groups: dict[str, tuple[str, list[int]]],
) -> list[int] | None:
    if not raw_joints:
        return None

    dims: list[int] = []
    for chunk in raw_joints.split(","):
        token = normalize_selector(chunk)
        if not token:
            continue
        group_entry = selector_groups.get(token)
        if group_entry is not None:
            _, group_dims = group_entry
            for dim in group_dims:
                if dim not in dims:
                    dims.append(dim)
            continue
        if token not in selector_to_dim:
            raise ValueError(f"Unknown joint selector: {chunk.strip()}")
        dim = selector_to_dim[token]
        if dim not in dims:
            dims.append(dim)

    if not dims:
        raise ValueError("No valid joints were provided.")
    return dims


def merge_selected_dims(dims_a: list[int] | None, dims_b: list[int] | None) -> list[int] | None:
    merged: list[int] = []
    for dim_list in (dims_a, dims_b):
        if dim_list is None:
            continue
        for dim in dim_list:
            if dim not in merged:
                merged.append(dim)
    return merged or None


def format_joint_listing(
    action_dim: int,
    dim_to_label: dict[int, str],
    selector_groups: dict[str, tuple[str, list[int]]],
) -> str:
    lines = [f"Available action joints for action_dim={action_dim}:"]
    for dim in range(action_dim):
        label = dim_to_label.get(dim, f"dim_{dim}")
        lines.append(f"  {dim:>2}: {label}")

    unique_groups: dict[str, list[int]] = {}
    for display_name, dims in selector_groups.values():
        unique_groups.setdefault(display_name, dims)

    if unique_groups:
        lines.append("")
        lines.append("Group selectors:")
        for display_name, dims in sorted(unique_groups.items()):
            dim_summary = ", ".join(f"{dim}:{dim_to_label.get(dim, f'd{dim}')}" for dim in dims)
            lines.append(f"  {display_name}: {dim_summary}")
    return "\n".join(lines)


def build_group_specs(action_dim: int, layout: str, dims: list[int] | None) -> list[tuple[str, list[int]]]:
    if dims is not None:
        return [("selected", dims)]

    if layout == "all":
        return [("all", list(range(action_dim)))]

    if action_dim == HAND_LEN + ARM_LEN + NECK_LEN:
        return [
            ("hand", list(range(0, HAND_LEN))),
            ("arm", list(range(HAND_LEN, HAND_LEN + ARM_LEN))),
            ("neck", list(range(HAND_LEN + ARM_LEN, action_dim))),
        ]

    if action_dim == HAND_LEN + ARM_LEN + NECK_LEN + WAIST_LEN:
        return [
            ("hand", list(range(0, HAND_LEN))),
            ("arm", list(range(HAND_LEN, HAND_LEN + ARM_LEN))),
            ("neck", list(range(HAND_LEN + ARM_LEN, HAND_LEN + ARM_LEN + NECK_LEN))),
            ("waist", list(range(HAND_LEN + ARM_LEN + NECK_LEN, action_dim))),
        ]

    return [("all", list(range(action_dim)))]


def _read_optional_scalar(values: np.ndarray, cast_type):
    if values.size == 0:
        return None
    return cast_type(values.reshape(-1)[0])


def _infer_inference_dt(
    chunk_time: np.ndarray,
    publish_time: np.ndarray,
    override: float | None,
    saved_dt: np.ndarray,
) -> float | None:
    if override is not None:
        return float(override)

    stored = _read_optional_scalar(saved_dt, float)
    if stored is not None and stored > 0.0:
        return stored

    for source in (chunk_time, publish_time):
        if source.size < 2:
            continue
        diffs = np.diff(source)
        diffs = diffs[np.isfinite(diffs) & (diffs > 1e-5)]
        if diffs.size > 0:
            return float(np.median(diffs))
    return None


def resolve_schedule_overlay(
    log_data: dict[str, np.ndarray],
    *,
    n_action_step_override: int | None,
    blend_steps_override: int | None,
    delay_steps_override: int | None,
    inference_dt_override: float | None,
) -> dict[str, float | int | np.ndarray] | None:
    n_action_step = n_action_step_override
    if n_action_step is None:
        n_action_step = _read_optional_scalar(log_data["n_action_step"], int)
    if n_action_step is None:
        n_action_step = DEFAULT_N_ACTION_STEP

    blend_steps = blend_steps_override
    if blend_steps is None:
        blend_steps = _read_optional_scalar(log_data["blending_horizon"], int)
    if blend_steps is None:
        blend_steps = DEFAULT_BLENDING_HORIZON

    request_step = _read_optional_scalar(log_data["request_step"], int)
    if request_step is None:
        request_step = max(1, int(n_action_step) - int(blend_steps))

    delay_steps = delay_steps_override
    if delay_steps is None:
        delay_steps = _read_optional_scalar(log_data["inference_delay_steps"], int)
    if delay_steps is None:
        delay_steps = DEFAULT_INFERENCE_DELAY_STEPS

    inference_dt = _infer_inference_dt(
        log_data["chunk_time"],
        log_data["publish_time"],
        inference_dt_override,
        log_data["inference_dt"],
    )
    if inference_dt is None or inference_dt <= 0.0:
        return None

    start_times = np.asarray(log_data["chunk_apply_time"], dtype=np.float32).reshape(-1)
    if start_times.size == 0:
        chunk_time = np.asarray(log_data["chunk_time"], dtype=np.float32).reshape(-1)
        if chunk_time.size == 0:
            return None
        start_times = chunk_time[:: max(int(request_step), 1)].copy()

    return {
        "start_times": start_times,
        "n_action_step": int(n_action_step),
        "blend_steps": int(blend_steps),
        "delay_steps": int(delay_steps),
        "request_step": int(request_step),
        "inference_dt": float(inference_dt),
        "delay_sec": float(delay_steps) * float(inference_dt),
        "blend_sec": float(blend_steps) * float(inference_dt),
        "chunk_stride_sec": float(request_step) * float(inference_dt),
    }


def _clip_interval(
    start: float,
    end: float,
    *,
    t_min: float | None,
    t_max: float | None,
) -> tuple[float, float] | None:
    clipped_start = start if t_min is None else max(start, t_min)
    clipped_end = end if t_max is None else min(end, t_max)
    if clipped_end <= clipped_start:
        return None
    return clipped_start, clipped_end


def overlay_schedule(
    ax,
    schedule: dict[str, float | int | np.ndarray] | None,
    *,
    t_min: float | None,
    t_max: float | None,
) -> bool:
    if schedule is None:
        return False

    start_times = np.asarray(schedule["start_times"], dtype=np.float32).reshape(-1)
    if start_times.size == 0:
        return False

    delay_sec = float(schedule["delay_sec"])
    blend_sec = float(schedule["blend_sec"])
    drew_anything = False
    start_label_drawn = False
    delay_label_drawn = False
    blend_label_drawn = False

    for start_time in start_times:
        if t_max is not None and start_time > t_max:
            continue
        if t_min is not None and (start_time + max(delay_sec, blend_sec, 0.0)) < t_min:
            continue

        ax.axvline(
            x=float(start_time),
            color="#6c757d",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            label="inference start" if not start_label_drawn else None,
        )
        start_label_drawn = True
        drew_anything = True

        if delay_sec > 0.0:
            clipped_delay = _clip_interval(
                float(start_time),
                float(start_time + delay_sec),
                t_min=t_min,
                t_max=t_max,
            )
            if clipped_delay is not None:
                ax.axvspan(
                    clipped_delay[0],
                    clipped_delay[1],
                    color="#9aa0a6",
                    alpha=0.14,
                    label="inference delay" if not delay_label_drawn else None,
                )
                delay_label_drawn = True
                drew_anything = True

        if blend_sec > delay_sec:
            clipped_blend = _clip_interval(
                float(start_time + delay_sec),
                float(start_time + blend_sec),
                t_min=t_min,
                t_max=t_max,
            )
            if clipped_blend is not None:
                ax.axvspan(
                    clipped_blend[0],
                    clipped_blend[1],
                    color="#ffd166",
                    alpha=0.18,
                    label="blending zone" if not blend_label_drawn else None,
                )
                blend_label_drawn = True
                drew_anything = True

    return drew_anything


def format_schedule_summary(schedule: dict[str, float | int | np.ndarray] | None) -> str | None:
    if schedule is None:
        return None
    return (
        "schedule: "
        f"step={int(schedule['n_action_step'])}, "
        f"request={int(schedule['request_step'])}, "
        f"blend={int(schedule['blend_steps'])}, "
        f"delay={int(schedule['delay_steps'])}, "
        f"dt={float(schedule['inference_dt']):.4f}s"
    )


def make_title(
    path: Path,
    action_dim: int,
    chunk_count: int,
    publish_count: int,
    schedule: dict[str, float | int | np.ndarray] | None,
) -> str:
    base = (
        f"{path.name}\n"
        f"action_dim={action_dim} | chunk_samples={chunk_count} | publish_samples={publish_count}"
    )
    schedule_text = format_schedule_summary(schedule)
    if schedule_text:
        return f"{base}\n{schedule_text}"
    return base


def plot_log(
    npz_path: Path,
    log_data: dict[str, np.ndarray],
    *,
    dim_to_label: dict[int, str],
    dims: list[int] | None,
    layout: str,
    t_min: float | None,
    t_max: float | None,
    show_schedule: bool,
    n_action_step_override: int | None,
    blend_steps_override: int | None,
    delay_steps_override: int | None,
    inference_dt_override: float | None,
    save_path: Path | None,
    show: bool,
) -> Path | None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    action_dim = int(log_data["action_dim"][0])
    chunk_time, chunk_action = apply_time_window(
        log_data["chunk_time"], log_data["chunk_action"], t_min, t_max
    )
    publish_time, publish_action = apply_time_window(
        log_data["publish_time"], log_data["publish_action"], t_min, t_max
    )
    schedule = None
    if show_schedule:
        schedule = resolve_schedule_overlay(
            log_data,
            n_action_step_override=n_action_step_override,
            blend_steps_override=blend_steps_override,
            delay_steps_override=delay_steps_override,
            inference_dt_override=inference_dt_override,
        )

    group_specs = build_group_specs(action_dim, layout, dims)
    fig, axes = plt.subplots(len(group_specs), 1, sharex=True, figsize=(15, 3.2 * len(group_specs)))
    axes = np.atleast_1d(axes)

    for ax, (group_name, group_dims) in zip(axes, group_specs):
        colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(group_dims), 1)))
        show_dim_legend = len(group_dims) <= 6
        drew_anything = False

        for color, dim in zip(colors, group_dims):
            joint_label = dim_to_label.get(dim, f"d{dim}")
            if chunk_time.size > 0 and chunk_action.size > 0:
                ax.plot(
                    chunk_time,
                    chunk_action[:, dim],
                    linestyle="--",
                    marker="o",
                    markersize=2.2,
                    linewidth=0.9,
                    alpha=0.9,
                    color=color,
                    label=f"chunk {joint_label}" if show_dim_legend else None,
                )
                drew_anything = True
            if publish_time.size > 0 and publish_action.size > 0:
                ax.plot(
                    publish_time,
                    publish_action[:, dim],
                    linewidth=0.9,
                    alpha=0.55,
                    color=color,
                    label=f"publish {joint_label}" if show_dim_legend else None,
                )
                drew_anything = True

        if not drew_anything:
            ax.text(0.5, 0.5, "No samples in selected range", ha="center", va="center", transform=ax.transAxes)

        dim_text = ",".join(dim_to_label.get(dim, str(dim)) for dim in group_dims[:8])
        if len(group_dims) > 8:
            dim_text += ",..."
        ax.set_title(f"{group_name} [{dim_text}]")
        schedule_drawn = overlay_schedule(ax, schedule, t_min=t_min, t_max=t_max)
        ax.set_ylabel("value")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        if (show_dim_legend and drew_anything) or schedule_drawn:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("time [s]")
    fig.suptitle(
        make_title(npz_path, action_dim, chunk_time.size, publish_time.size, schedule),
        fontsize=11,
    )
    fig.tight_layout()

    written_path: Path | None = None
    if save_path is not None or not show:
        written_path = save_path or npz_path.with_name(f"{npz_path.stem}_viewer.png")
        written_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(written_path, dpi=180)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return written_path


def main() -> int:
    args = parse_args()
    npz_path = resolve_npz_path(args.npz)
    log_data = load_npz_log(npz_path)
    action_dim = int(log_data["action_dim"][0])
    _, selector_to_dim, dim_to_label = build_joint_index(action_dim)
    selector_groups = build_selector_groups(action_dim)

    if args.list_joints:
        print(format_joint_listing(action_dim, dim_to_label, selector_groups))
        return 0

    dims = merge_selected_dims(
        parse_dims(args.dims, action_dim),
        parse_joint_selectors(args.joints, action_dim, selector_to_dim, selector_groups),
    )

    output_path = plot_log(
        npz_path,
        log_data,
        dim_to_label=dim_to_label,
        dims=dims,
        layout=args.layout,
        t_min=args.t_min,
        t_max=args.t_max,
        show_schedule=not args.hide_schedule,
        n_action_step_override=args.n_action_step,
        blend_steps_override=args.blend_steps,
        delay_steps_override=args.delay_steps,
        inference_dt_override=args.inference_dt,
        save_path=args.save,
        show=args.show,
    )

    print(f"Loaded: {npz_path}")
    print(
        "Summary: "
        f"action_dim={action_dim}, "
        f"chunk_samples={log_data['chunk_time'].size}, "
        f"publish_samples={log_data['publish_time'].size}"
    )
    if dims is not None:
        selected = ", ".join(f"{dim}:{dim_to_label.get(dim, f'd{dim}')}" for dim in dims)
        print(f"Selected joints: {selected}")
    if not args.hide_schedule:
        schedule = resolve_schedule_overlay(
            log_data,
            n_action_step_override=args.n_action_step,
            blend_steps_override=args.blend_steps,
            delay_steps_override=args.delay_steps,
            inference_dt_override=args.inference_dt,
        )
        schedule_text = format_schedule_summary(schedule)
        if schedule_text is not None:
            print(schedule_text)
    if output_path is not None:
        print(f"Saved plot: {output_path}")
    elif args.show:
        print("Displayed plot window.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
