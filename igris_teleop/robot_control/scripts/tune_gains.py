from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from igris_teleop.core.project_paths import TUNING_RUNS_ROOT
from igris_teleop.robot_control.kinematics.joints import (
    ARM_INDICES,
    LEG_INDICES,
    L_ARM_INDICES,
    L_ELBOW_INDICES,
    L_SHOULDER_INDICES,
    L_WRIST_INDICES,
    NECK_INDICES,
    JointIndex,
    R_ARM_INDICES,
    R_ELBOW_INDICES,
    R_SHOULDER_INDICES,
    R_WRIST_INDICES,
    WAIST_INDICES,
)
from igris_teleop.robot_control.scripts.apply_tuned_gains import (
    apply_single_axis_best,
    default_joint_profile_path,
    load_joint_profile_yaml,
)
from igris_teleop.robot_control.scripts.common_rollout import (
    compute_tracking_metrics,
    make_chirp_profile,
    make_sine_profile,
    run_rollout_interruptible,
    save_episode_npz,
    save_tracking_plot,
)


def resolve_joint_group(name: str):
    groups = {
        "waist": list(WAIST_INDICES),
        "leg": list(LEG_INDICES),
        "arm": list(ARM_INDICES),
        "l_arm": list(L_ARM_INDICES),
        "r_arm": list(R_ARM_INDICES),
        "l_wrist": list(L_WRIST_INDICES),
        "r_wrist": list(R_WRIST_INDICES),
        "l_shoulder": list(L_SHOULDER_INDICES),
        "r_shoulder": list(R_SHOULDER_INDICES),
        "l_elbow": list(L_ELBOW_INDICES),
        "r_elbow": list(R_ELBOW_INDICES),
        "neck": list(NECK_INDICES),
        "all": list(sorted(set(WAIST_INDICES) | set(LEG_INDICES) | set(ARM_INDICES) | set(NECK_INDICES))),
    }
    if name not in groups:
        raise KeyError(f"Unknown joint group: {name}")
    return groups[name]


def parse_scales(raw: str):
    return [float(x) for x in raw.split(',') if x.strip()]


def joint_name(joint_index: int) -> str:
    try:
        return JointIndex(int(joint_index)).name.lower()
    except ValueError:
        return f"joint_{int(joint_index)}"


def resolve_joint_mode(requested_mode: str, joint_group: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    if joint_group == "waist":
        return "single+pairs"
    return "together"


def build_target_sets(joint_group: str, joint_indices, joint_mode: str):
    joint_indices = [int(idx) for idx in joint_indices]
    if joint_mode == "together":
        subsets = [joint_indices]
    elif joint_mode == "single":
        subsets = [[idx] for idx in joint_indices]
    elif joint_mode == "pairs":
        subsets = [list(pair) for pair in itertools.combinations(joint_indices, 2)]
    elif joint_mode == "single+pairs":
        subsets = [[idx] for idx in joint_indices]
        subsets.extend(list(pair) for pair in itertools.combinations(joint_indices, 2))
    elif joint_mode == "single+pairs+all":
        subsets = [[idx] for idx in joint_indices]
        subsets.extend(list(pair) for pair in itertools.combinations(joint_indices, 2))
        subsets.append(joint_indices)
    else:
        raise ValueError(f"Unsupported joint mode: {joint_mode}")

    if not subsets:
        raise ValueError(f"Joint mode '{joint_mode}' produced no target sets for group '{joint_group}'.")

    target_sets = []
    for subset in subsets:
        name = "+".join(joint_name(idx) for idx in subset)
        target_sets.append(
            {
                "name": name,
                "slug": name.replace("+", "__"),
                "indices": [int(idx) for idx in subset],
            }
        )
    return target_sets


def wait_for_start(auto_start: bool) -> None:
    if auto_start:
        return
    prompt = "Type 's' then Enter to start tuning, or Ctrl+C to abort: "
    while True:
        try:
            raw = input(prompt).strip().lower()
        except EOFError as exc:
            raise RuntimeError("Interactive start requested but stdin is unavailable. Use --auto-start.") from exc
        if raw == "s":
            return
        if raw in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        print("Not started. Type 's' then Enter when ready.")


def clip_scale(value: float, min_scale: float, max_scale: float) -> float:
    return float(min(max(float(value), float(min_scale)), float(max_scale)))


def _float_slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def build_motion_spec(
    profile: str,
    *,
    amplitude: float,
    duration: float,
    frequency: float | None = None,
    f0: float | None = None,
    f1: float | None = None,
    name: str | None = None,
) -> dict:
    spec = {
        "profile": str(profile),
        "amplitude": float(amplitude),
        "duration": float(duration),
    }
    if profile == "sine":
        if frequency is None:
            raise ValueError("sine motion requires frequency")
        spec["frequency"] = float(frequency)
    elif profile == "chirp":
        if f0 is None or f1 is None:
            raise ValueError("chirp motion requires f0 and f1")
        spec["f0"] = float(f0)
        spec["f1"] = float(f1)
    else:
        raise ValueError(f"Unsupported motion profile: {profile}")

    if name is None:
        if profile == "sine":
            name = (
                f"sine_a{_float_slug(spec['amplitude'])}"
                f"_f{_float_slug(spec['frequency'])}"
                f"_d{_float_slug(spec['duration'])}"
            )
        else:
            name = (
                f"chirp_a{_float_slug(spec['amplitude'])}"
                f"_f{_float_slug(spec['f0'])}_{_float_slug(spec['f1'])}"
                f"_d{_float_slug(spec['duration'])}"
            )
    spec["name"] = str(name)
    spec["slug"] = str(name).replace("+", "__")
    return spec


def parse_motion_spec(raw: str, default_duration: float) -> dict:
    parts = [part.strip() for part in str(raw).split(":") if part.strip()]
    if not parts:
        raise ValueError("Empty --motion-spec")
    profile = parts[0]
    if profile == "sine":
        if len(parts) not in (3, 4):
            raise ValueError("sine motion spec must be sine:<amplitude>:<frequency>[:duration]")
        duration = float(parts[3]) if len(parts) == 4 else float(default_duration)
        return build_motion_spec(
            "sine",
            amplitude=float(parts[1]),
            frequency=float(parts[2]),
            duration=duration,
        )
    if profile == "chirp":
        if len(parts) not in (4, 5):
            raise ValueError("chirp motion spec must be chirp:<amplitude>:<f0>:<f1>[:duration]")
        duration = float(parts[4]) if len(parts) == 5 else float(default_duration)
        return build_motion_spec(
            "chirp",
            amplitude=float(parts[1]),
            f0=float(parts[2]),
            f1=float(parts[3]),
            duration=duration,
        )
    raise ValueError(f"Unsupported motion profile in --motion-spec: {profile}")


def motion_preset_specs(name: str, default_duration: float) -> list[dict]:
    duration = float(default_duration)
    presets = {
        "elbow": [
            build_motion_spec("sine", amplitude=0.02, frequency=0.2, duration=duration, name="sine_small_slow"),
            build_motion_spec("sine", amplitude=0.05, frequency=0.2, duration=duration, name="sine_mid_slow"),
            build_motion_spec("sine", amplitude=0.05, frequency=0.5, duration=duration, name="sine_mid_fast"),
            build_motion_spec("chirp", amplitude=0.03, f0=0.1, f1=0.8, duration=duration, name="chirp_small"),
        ],
        "wrist": [
            build_motion_spec("sine", amplitude=0.01, frequency=0.2, duration=duration, name="sine_small_slow"),
            build_motion_spec("sine", amplitude=0.03, frequency=0.2, duration=duration, name="sine_mid_slow"),
            build_motion_spec("sine", amplitude=0.03, frequency=0.5, duration=duration, name="sine_mid_fast"),
            build_motion_spec("chirp", amplitude=0.02, f0=0.1, f1=0.8, duration=duration, name="chirp_small"),
        ],
        "arm": [
            build_motion_spec("sine", amplitude=0.02, frequency=0.2, duration=duration, name="sine_small_slow"),
            build_motion_spec("sine", amplitude=0.05, frequency=0.2, duration=duration, name="sine_mid_slow"),
            build_motion_spec("sine", amplitude=0.05, frequency=0.4, duration=duration, name="sine_mid_fast"),
            build_motion_spec("chirp", amplitude=0.03, f0=0.1, f1=0.8, duration=duration, name="chirp_small"),
        ],
        "waist": [
            build_motion_spec("sine", amplitude=0.02, frequency=0.2, duration=duration, name="sine_small_slow"),
            build_motion_spec("sine", amplitude=0.04, frequency=0.2, duration=duration, name="sine_mid_slow"),
            build_motion_spec("sine", amplitude=0.04, frequency=0.35, duration=duration, name="sine_mid_fast"),
            build_motion_spec("chirp", amplitude=0.03, f0=0.05, f1=0.6, duration=duration, name="chirp_small"),
        ],
    }
    if name == "none":
        return []
    if name not in presets:
        raise ValueError(f"Unsupported motion preset: {name}")
    return [dict(spec) for spec in presets[name]]


def resolve_motion_specs(args) -> list[dict]:
    specs = motion_preset_specs(args.motion_preset, args.duration)
    specs.extend(parse_motion_spec(raw, args.duration) for raw in args.motion_spec)
    if specs:
        return specs
    return [
        build_motion_spec(
            args.profile,
            amplitude=args.amplitude,
            duration=args.duration,
            frequency=args.frequency,
            f0=args.f0,
            f1=args.f1,
        )
    ]


def build_profile_from_motion_spec(q_ref: np.ndarray, active_indices, motion_spec: dict):
    if motion_spec["profile"] == "sine":
        return make_sine_profile(
            q_ref,
            active_indices,
            amplitude=motion_spec["amplitude"],
            frequency=motion_spec["frequency"],
        )
    return make_chirp_profile(
        q_ref,
        active_indices,
        amplitude=motion_spec["amplitude"],
        f0=motion_spec["f0"],
        f1=motion_spec["f1"],
        duration=motion_spec["duration"],
    )


def aggregate_motion_metrics(motion_rows: list[dict], motion_aggregate: str) -> dict:
    if not motion_rows:
        return {"motion_count": 0, "cost": float("inf"), "motion_aggregate": motion_aggregate}

    out = {
        "motion_count": len(motion_rows),
        "motion_names": [row["motion_name"] for row in motion_rows],
        "motion_results": motion_rows,
        "motion_aggregate": motion_aggregate,
        "samples": int(sum(int(row.get("samples", 0)) for row in motion_rows)),
    }

    mean_keys = ("mae_q", "rmse_q", "mae_dq", "rmse_dq", "mae_tau", "smooth_tau")
    for key in mean_keys:
        values = [float(row[key]) for row in motion_rows if key in row]
        if values:
            out[key] = float(np.mean(values))

    max_abs_q_values = [float(row["max_abs_q"]) for row in motion_rows if "max_abs_q" in row]
    if max_abs_q_values:
        out["max_abs_q"] = float(max(max_abs_q_values))

    cost_values = [float(row.get("cost", float("inf"))) for row in motion_rows]
    mean_cost = float(np.mean(cost_values))
    max_cost = float(np.max(cost_values))
    min_cost = float(np.min(cost_values))
    out["motion_cost_mean"] = mean_cost
    out["motion_cost_max"] = max_cost
    out["motion_cost_min"] = min_cost
    out["motion_cost_std"] = float(np.std(cost_values))

    if motion_aggregate == "mean":
        out["cost"] = mean_cost
    elif motion_aggregate == "max":
        out["cost"] = max_cost
    elif motion_aggregate == "meanmax":
        out["cost"] = 0.7 * mean_cost + 0.3 * max_cost
    else:
        raise ValueError(f"Unsupported motion_aggregate: {motion_aggregate}")
    return out


def execute_tuning_run(
    args,
    ctrl,
    run_idx: int,
    joint_group: str,
    joint_mode: str,
    search_mode: str,
    all_group_indices,
    active_indices,
    subset_name: str,
    subset_slug: str,
    kp_default: np.ndarray,
    kd_default: np.ndarray,
    kp_scale: float,
    kd_scale: float,
    motion_specs,
    motion_aggregate: str,
    extra_metadata=None,
):
    active_indices = [int(idx) for idx in active_indices]
    base_kp_subset = [float(kp_default[idx]) for idx in active_indices]
    base_kd_subset = [float(kd_default[idx]) for idx in active_indices]
    applied_kp_subset = [float(kp_default[idx] * kp_scale) for idx in active_indices]
    applied_kd_subset = [float(kd_default[idx] * kd_scale) for idx in active_indices]
    motion_rows = []
    motion_logs = []
    run_interrupted = False

    for motion_idx, motion_spec in enumerate(motion_specs):
        if args.move_to_default:
            ctrl.move_to_pose("default_pos", duration=args.reset_duration)
            time.sleep(args.settle_sec)

        cmd_ref = ctrl.get_command_snapshot()
        q_ref = cmd_ref.get("q_target_pjs")
        if q_ref is None:
            raise RuntimeError("No command target available before rollout")

        ctrl.set_joint_gains(all_group_indices, kp=kp_default[all_group_indices], kd=kd_default[all_group_indices])
        ctrl.set_joint_gains(
            active_indices,
            kp=kp_default[active_indices] * kp_scale,
            kd=kd_default[active_indices] * kd_scale,
        )

        profile = build_profile_from_motion_spec(q_ref, active_indices, motion_spec)
        log, motion_interrupted = run_rollout_interruptible(
            ctrl,
            trajectory_fn=profile,
            duration=motion_spec["duration"],
            record_hz=args.record_hz,
        )
        samples = int(np.asarray(log.get("t", np.empty((0,), dtype=np.float32))).shape[0])
        metrics = (
            compute_tracking_metrics(log, joint_indices=active_indices, cost_mode=args.cost_mode)
            if samples > 0
            else {}
        )
        motion_row = {
            "motion_idx": motion_idx,
            "motion_name": motion_spec["name"],
            "motion_slug": motion_spec["slug"],
            "motion_profile": motion_spec["profile"],
            "motion_center_source": "target",
            "motion_amplitude": float(motion_spec["amplitude"]),
            "motion_duration": float(motion_spec["duration"]),
            "samples": samples,
            "interrupted": motion_interrupted,
            **metrics,
        }
        if motion_spec["profile"] == "sine":
            motion_row["motion_frequency"] = float(motion_spec["frequency"])
        else:
            motion_row["motion_f0"] = float(motion_spec["f0"])
            motion_row["motion_f1"] = float(motion_spec["f1"])
        motion_rows.append(motion_row)
        motion_logs.append(log)
        if motion_interrupted:
            run_interrupted = True
            break

    metrics = aggregate_motion_metrics(motion_rows, motion_aggregate)

    row = {
        "run_idx": run_idx,
        "joint_group": joint_group,
        "joint_mode": joint_mode,
        "search_mode": search_mode,
        "subset_name": subset_name,
        "subset_indices": active_indices,
        "subset_joint_names": [joint_name(idx) for idx in active_indices],
        "kp_scale": float(kp_scale),
        "kd_scale": float(kd_scale),
        "base_kp_subset": base_kp_subset,
        "base_kd_subset": base_kd_subset,
        "applied_kp_subset": applied_kp_subset,
        "applied_kd_subset": applied_kd_subset,
        "motion_specs": motion_specs,
        "interrupted": run_interrupted,
        **metrics,
    }
    if extra_metadata:
        row.update(extra_metadata)

    return row, motion_logs, run_interrupted


def save_best_run_artifacts(args, out_dir: Path, target, row, log) -> None:
    if row is None or log is None:
        return

    subset_slug = target["slug"]
    motion_results = row.get("motion_results", [])
    logs = log if isinstance(log, list) else [log]

    if len(logs) == 1:
        run_path = out_dir / f"best_{subset_slug}.npz"
        row["log_path"] = str(run_path)
        title = f"{row['subset_name']} | kp={row['kp_scale']:.3g}, kd={row['kd_scale']:.3g}"
        if motion_results:
            title = f"{title} | {motion_results[0]['motion_name']}"
        if args.save_plot:
            plot_path = save_tracking_plot(
                out_dir / f"best_{subset_slug}.png",
                logs[0],
                joint_indices=row["subset_indices"],
                joint_labels=row["subset_joint_names"],
                title=title,
            )
            if plot_path is not None:
                row["plot_path"] = str(plot_path)
                if motion_results:
                    motion_results[0]["plot_path"] = str(plot_path)
        if motion_results:
            motion_results[0]["log_path"] = str(run_path)
        save_episode_npz(run_path, logs[0], metadata=row)
        return

    row["log_paths"] = []
    row["plot_paths"] = []
    summary_path = out_dir / f"best_{subset_slug}.json"
    row["summary_path"] = str(summary_path)

    for motion_idx, (motion_result, motion_log) in enumerate(zip(motion_results, logs)):
        motion_slug = motion_result.get("motion_slug", f"motion_{motion_idx:02d}")
        run_path = out_dir / f"best_{subset_slug}__{motion_slug}.npz"
        motion_result["log_path"] = str(run_path)
        row["log_paths"].append(str(run_path))
        if args.save_plot:
            plot_path = save_tracking_plot(
                out_dir / f"best_{subset_slug}__{motion_slug}.png",
                motion_log,
                joint_indices=row["subset_indices"],
                joint_labels=row["subset_joint_names"],
                title=(
                    f"{row['subset_name']} | {motion_result['motion_name']} | "
                    f"kp={row['kp_scale']:.3g}, kd={row['kd_scale']:.3g}"
                ),
            )
            if plot_path is not None:
                motion_result["plot_path"] = str(plot_path)
                row["plot_paths"].append(str(plot_path))
        save_episode_npz(
            run_path,
            motion_log,
            metadata={
                **row,
                "motion_result": motion_result,
            },
        )

    if not row["plot_paths"]:
        row.pop("plot_paths", None)
    summary_path.write_text(json.dumps(row, indent=2, ensure_ascii=False))


def apply_best_to_joint_profile(summary: dict, joint_profile_path: Path) -> tuple[list[dict], list[str]]:
    joint_profile = load_joint_profile_yaml(joint_profile_path)
    updated, applied, warnings = apply_single_axis_best(
        summary=summary,
        joint_profile=joint_profile,
        allow_scale_fallback=False,
    )
    if not applied:
        raise RuntimeError(
            "No single-axis best results were available to apply. "
            "Run with --joint-mode single, or include single-axis targets."
        )
    joint_profile_path.parent.mkdir(parents=True, exist_ok=True)
    with joint_profile_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(updated, file_obj, sort_keys=False)
    return applied, warnings


def run_grid_search(
    args,
    ctrl,
    out_dir: Path,
    start_run_idx: int,
    joint_group: str,
    joint_mode: str,
    all_group_indices,
    target,
    kp_default: np.ndarray,
    kd_default: np.ndarray,
    kp_scales,
    kd_scales,
    motion_specs,
    motion_aggregate: str,
):
    rows = []
    best = None
    best_log = None
    run_idx = start_run_idx
    interrupted = False
    active_indices = [int(idx) for idx in target["indices"]]

    for kp_scale, kd_scale in itertools.product(kp_scales, kd_scales):
        row, log, run_interrupted = execute_tuning_run(
            args=args,
            ctrl=ctrl,
            run_idx=run_idx,
            joint_group=joint_group,
            joint_mode=joint_mode,
            search_mode="grid",
            all_group_indices=all_group_indices,
            active_indices=active_indices,
            subset_name=target["name"],
            subset_slug=target["slug"],
            kp_default=kp_default,
            kd_default=kd_default,
            kp_scale=kp_scale,
            kd_scale=kd_scale,
            motion_specs=motion_specs,
            motion_aggregate=motion_aggregate,
            extra_metadata={"search_probe": "grid"},
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if best is None or row.get("cost", float("inf")) < best.get("cost", float("inf")):
            best = row
            best_log = log
        run_idx += 1
        if run_interrupted:
            interrupted = True
            break

    return rows, best, best_log, run_idx, interrupted


def run_adaptive_search(
    args,
    ctrl,
    out_dir: Path,
    start_run_idx: int,
    joint_group: str,
    joint_mode: str,
    all_group_indices,
    target,
    kp_default: np.ndarray,
    kd_default: np.ndarray,
    motion_specs,
    motion_aggregate: str,
):
    rows = []
    best = None
    best_log = None
    run_idx = start_run_idx
    interrupted = False
    active_indices = [int(idx) for idx in target["indices"]]
    visited = {}

    def evaluate_candidate(kp_scale: float, kd_scale: float, search_iter: int, search_probe: str):
        nonlocal run_idx, best, best_log, interrupted
        key = (round(float(kp_scale), 6), round(float(kd_scale), 6))
        if key in visited:
            return visited[key]

        row, log, run_interrupted = execute_tuning_run(
            args=args,
            ctrl=ctrl,
            run_idx=run_idx,
            joint_group=joint_group,
            joint_mode=joint_mode,
            search_mode="adaptive",
            all_group_indices=all_group_indices,
            active_indices=active_indices,
            subset_name=target["name"],
            subset_slug=target["slug"],
            kp_default=kp_default,
            kd_default=kd_default,
            kp_scale=kp_scale,
            kd_scale=kd_scale,
            motion_specs=motion_specs,
            motion_aggregate=motion_aggregate,
            extra_metadata={
                "search_iter": search_iter,
                "search_probe": search_probe,
            },
        )
        visited[key] = row
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if best is None or row.get("cost", float("inf")) < best.get("cost", float("inf")):
            best = row
            best_log = log
        run_idx += 1
        if run_interrupted:
            interrupted = True
        return row

    current = evaluate_candidate(1.0, 1.0, search_iter=0, search_probe="baseline")
    if interrupted:
        return rows, best, best_log, run_idx, interrupted

    delta_kp = float(args.adaptive_kp_delta)
    delta_kd = float(args.adaptive_kd_delta)
    current_kp = float(current["kp_scale"])
    current_kd = float(current["kd_scale"])

    for search_iter in range(1, int(args.adaptive_iters) + 1):
        candidates = []
        kp_up = clip_scale(current_kp * (1.0 + delta_kp), args.kp_min_scale, args.kp_max_scale)
        kp_down = clip_scale(current_kp * (1.0 - delta_kp), args.kp_min_scale, args.kp_max_scale)
        kd_up = clip_scale(current_kd * (1.0 + delta_kd), args.kd_min_scale, args.kd_max_scale)
        kd_down = clip_scale(current_kd * (1.0 - delta_kd), args.kd_min_scale, args.kd_max_scale)

        if not np.isclose(kp_up, current_kp):
            candidates.append(("kp_up", kp_up, current_kd))
        if not np.isclose(kp_down, current_kp):
            candidates.append(("kp_down", kp_down, current_kd))
        if not np.isclose(kd_up, current_kd):
            candidates.append(("kd_up", current_kp, kd_up))
        if not np.isclose(kd_down, current_kd):
            candidates.append(("kd_down", current_kp, kd_down))

        if not candidates:
            break

        local_best = current
        for search_probe, kp_scale, kd_scale in candidates:
            row = evaluate_candidate(kp_scale, kd_scale, search_iter=search_iter, search_probe=search_probe)
            if interrupted:
                break
            if row.get("cost", float("inf")) < local_best.get("cost", float("inf")):
                local_best = row
        if interrupted:
            break

        improved = local_best is not current
        current = local_best
        current_kp = float(current["kp_scale"])
        current_kd = float(current["kd_scale"])

        if not improved:
            delta_kp *= float(args.adaptive_shrink)
            delta_kd *= float(args.adaptive_shrink)
            if delta_kp < float(args.adaptive_min_delta) and delta_kd < float(args.adaptive_min_delta):
                break

    return rows, best, best_log, run_idx, interrupted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--control-hz", type=float, default=300.0)
    parser.add_argument("--record-hz", type=float, default=100.0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--joint-group", type=str, default="waist")
    parser.add_argument("--search-mode", choices=["grid", "adaptive"], default="grid")
    parser.add_argument(
        "--joint-mode",
        choices=["auto", "together", "single", "pairs", "single+pairs", "single+pairs+all"],
        default="auto",
    )
    parser.add_argument("--profile", choices=["sine", "chirp"], default="sine")
    parser.add_argument("--cost-mode", choices=["tracking", "balanced", "effort"], default="tracking")
    parser.add_argument("--motion-preset", choices=["none", "elbow", "wrist", "arm", "waist"], default="none")
    parser.add_argument(
        "--motion-spec",
        action="append",
        default=[],
        help="Repeatable motion spec: sine:<amplitude>:<frequency>[:duration] or chirp:<amplitude>:<f0>:<f1>[:duration]",
    )
    parser.add_argument("--motion-aggregate", choices=["mean", "max", "meanmax"], default="meanmax")
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--frequency", type=float, default=0.5)
    parser.add_argument("--f0", type=float, default=0.2)
    parser.add_argument("--f1", type=float, default=2.0)
    parser.add_argument("--kp-scale", type=str, default="0.5,0.8,1.0,1.2,1.5")
    parser.add_argument("--kd-scale", type=str, default="0.5,0.8,1.0,1.2")
    parser.add_argument("--move-to-default", action="store_true")
    parser.add_argument("--reset-duration", type=float, default=2.0)
    parser.add_argument("--settle-sec", type=float, default=0.5)
    parser.add_argument("--adaptive-iters", type=int, default=6)
    parser.add_argument("--adaptive-kp-delta", type=float, default=0.2)
    parser.add_argument("--adaptive-kd-delta", type=float, default=0.2)
    parser.add_argument("--adaptive-shrink", type=float, default=0.5)
    parser.add_argument("--adaptive-min-delta", type=float, default=0.03)
    parser.add_argument("--kp-min-scale", type=float, default=0.25)
    parser.add_argument("--kp-max-scale", type=float, default=2.0)
    parser.add_argument("--kd-min-scale", type=float, default=0.25)
    parser.add_argument("--kd-max-scale", type=float, default=2.0)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--no-plot", action="store_false", dest="save_plot")
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="Write single-axis best gains directly to the joint profile after tuning completes",
    )
    parser.add_argument(
        "--apply-joint-profile",
        type=str,
        default=str(default_joint_profile_path()),
        help="Joint profile YAML path to overwrite when --apply-best is set",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.set_defaults(save_plot=True)
    args = parser.parse_args()

    from igris_teleop.robot_control.controller.igris_controller import IgrisController

    kp_scales = parse_scales(args.kp_scale)
    kd_scales = parse_scales(args.kd_scale)
    joint_indices = [int(idx) for idx in resolve_joint_group(args.joint_group)]
    joint_mode = resolve_joint_mode(args.joint_mode, args.joint_group)
    target_sets = build_target_sets(args.joint_group, joint_indices, joint_mode)
    motion_specs = resolve_motion_specs(args)

    ctrl = None
    kp_default = None
    kd_default = None
    interrupted = False
    emergency_abort = False
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else TUNING_RUNS_ROOT / time.strftime("%Y%m%d_%H%M%S")
    )
    try:
        ctrl = IgrisController(domain_id=args.domain_id, control_hz=args.control_hz)
        if not ctrl.wait_for_state(timeout=5.0):
            raise RuntimeError("LowState timeout")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"output_dir: {out_dir}")
        print(f"search_mode: {args.search_mode}")
        print(f"cost_mode: {args.cost_mode}")
        print(f"joint_mode: {joint_mode}")
        print(f"motion_aggregate: {args.motion_aggregate}")
        print(f"motions: {[motion['name'] for motion in motion_specs]}")
        print(f"targets: {[target['name'] for target in target_sets]}")
        print("Press Ctrl+C at any time to abort quickly.")
        wait_for_start(args.auto_start)

        if args.move_to_default:
            ctrl.move_to_pose("default_pos", duration=args.reset_duration)
            time.sleep(args.settle_sec)
        q_ref = ctrl.get_joint_q()
        if q_ref is None:
            raise RuntimeError("No joint state available")
        base_cmd = ctrl.get_command_snapshot()
        kp_default = base_cmd["kp"].copy()
        kd_default = base_cmd["kd"].copy()

        results = []
        best = None
        best_by_target = {}
        run_idx = 0
        for target in target_sets:
            if args.search_mode == "adaptive":
                target_rows, target_best, target_best_log, run_idx, target_interrupted = run_adaptive_search(
                    args=args,
                    ctrl=ctrl,
                    out_dir=out_dir,
                    start_run_idx=run_idx,
                    joint_group=args.joint_group,
                    joint_mode=joint_mode,
                    all_group_indices=joint_indices,
                    target=target,
                    kp_default=kp_default,
                    kd_default=kd_default,
                    motion_specs=motion_specs,
                    motion_aggregate=args.motion_aggregate,
                )
            else:
                target_rows, target_best, target_best_log, run_idx, target_interrupted = run_grid_search(
                    args=args,
                    ctrl=ctrl,
                    out_dir=out_dir,
                    start_run_idx=run_idx,
                    joint_group=args.joint_group,
                    joint_mode=joint_mode,
                    all_group_indices=joint_indices,
                    target=target,
                    kp_default=kp_default,
                    kd_default=kd_default,
                    kp_scales=kp_scales,
                    kd_scales=kd_scales,
                    motion_specs=motion_specs,
                    motion_aggregate=args.motion_aggregate,
                )

            results.extend(target_rows)
            if target_best is not None:
                save_best_run_artifacts(args, out_dir, target, target_best, target_best_log)
                best_by_target[target["name"]] = target_best
                if best is None or target_best.get("cost", float("inf")) < best.get("cost", float("inf")):
                    best = target_best
            if target_interrupted:
                interrupted = True
                emergency_abort = True
            if interrupted:
                break

        summary = {
            "best": best,
            "best_by_target": best_by_target,
            "results": results,
            "search_mode": args.search_mode,
            "cost_mode": args.cost_mode,
            "joint_group": args.joint_group,
            "joint_mode_requested": args.joint_mode,
            "joint_mode_resolved": joint_mode,
            "motion_center_source": "target",
            "targets": target_sets,
            "motion_preset": args.motion_preset,
            "motion_specs": motion_specs,
            "motion_aggregate": args.motion_aggregate,
            "profile": args.profile,
            "amplitude": args.amplitude,
            "frequency": args.frequency,
            "f0": args.f0,
            "f1": args.f1,
            "duration": args.duration,
            "record_hz": args.record_hz,
            "control_hz": args.control_hz,
            "save_plot": args.save_plot,
            "interrupted": interrupted,
            "grid_kp_scales": kp_scales,
            "grid_kd_scales": kd_scales,
            "adaptive": {
                "iters": args.adaptive_iters,
                "kp_delta": args.adaptive_kp_delta,
                "kd_delta": args.adaptive_kd_delta,
                "shrink": args.adaptive_shrink,
                "min_delta": args.adaptive_min_delta,
                "kp_scale_range": [args.kp_min_scale, args.kp_max_scale],
                "kd_scale_range": [args.kd_min_scale, args.kd_max_scale],
            },
        }
        if args.apply_best:
            if interrupted:
                summary["apply_warnings"] = ["apply-best skipped because tuning was interrupted"]
                print("apply-best skipped because tuning was interrupted.")
            else:
                joint_profile_path = Path(args.apply_joint_profile).expanduser()
                try:
                    applied_rows, apply_warnings = apply_best_to_joint_profile(summary, joint_profile_path)
                    summary["applied_joint_profile"] = str(joint_profile_path)
                    summary["applied_rows"] = applied_rows
                    if apply_warnings:
                        summary["apply_warnings"] = apply_warnings
                    print(f"applied_joint_profile: {joint_profile_path}")
                    for row in applied_rows:
                        print(
                            "applied: "
                            f"{row['joint_name']} -> kp={row['new_kp']:.4f}, kd={row['new_kd']:.4f}"
                        )
                    for warning in apply_warnings:
                        print(f"apply_warning: {warning}")
                except Exception as exc:
                    summary["apply_error"] = str(exc)
                    print(f"apply-best failed: {exc}")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"best: {json.dumps(best, ensure_ascii=False)}")
        if interrupted:
            print("Interrupted by user. Partial results were saved.")
    except KeyboardInterrupt:
        interrupted = True
        emergency_abort = True
        print("KeyboardInterrupt received. Aborting immediately.")
    finally:
        if ctrl is None:
            return
        if kp_default is not None and kd_default is not None:
            try:
                ctrl.set_joint_gains(joint_indices, kp=kp_default[joint_indices], kd=kd_default[joint_indices])
            except Exception:
                pass
        if emergency_abort:
            ctrl.abort()
        else:
            ctrl.stop()


if __name__ == "__main__":
    main()
