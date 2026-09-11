from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from igris_teleop.core.project_paths import JOINT_SETTING_PATH
from igris_teleop.robot_control.kinematics.joints import JointIndex, NUM_MOTORS


def _robot_control_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def default_joint_profile_path() -> Path:
    return JOINT_SETTING_PATH


def default_output_path(joint_profile_path: Path) -> Path:
    return joint_profile_path.with_name(f"{joint_profile_path.stem}_autotuned{joint_profile_path.suffix}")


def load_joint_profile_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    for key in ("kp", "kd", "default_dof_pos", "waypoint_1", "waypoint_2"):
        if key not in data:
            raise KeyError(f"Missing key in joint profile: {key}")
        values = list(data[key])
        if len(values) != NUM_MOTORS:
            raise ValueError(f"{key} length {len(values)} != {NUM_MOTORS}")
        data[key] = values
    return data


def joint_name(index: int) -> str:
    try:
        return JointIndex(int(index)).name.lower()
    except ValueError:
        return f"joint_{int(index)}"


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _subset_floats(row: dict, key: str, expected_len: int) -> list[float] | None:
    values = row.get(key)
    if values is None:
        return None
    values = [float(value) for value in values]
    if len(values) != expected_len:
        raise ValueError(f"{row.get('subset_name', '<unknown>')}: {key} length {len(values)} != {expected_len}")
    return values


def resolve_applied_subset_gains(
    row: dict,
    joint_profile: dict,
    allow_scale_fallback: bool,
    warnings: list[str],
) -> tuple[list[float], list[float]]:
    indices = [int(idx) for idx in row.get("subset_indices", [])]
    if not indices:
        raise ValueError(f"{row.get('subset_name', '<unknown>')}: missing subset_indices")

    applied_kp_subset = _subset_floats(row, "applied_kp_subset", len(indices))
    applied_kd_subset = _subset_floats(row, "applied_kd_subset", len(indices))
    if applied_kp_subset is not None and applied_kd_subset is not None:
        return applied_kp_subset, applied_kd_subset

    base_kp_subset = _subset_floats(row, "base_kp_subset", len(indices))
    base_kd_subset = _subset_floats(row, "base_kd_subset", len(indices))
    if base_kp_subset is not None and base_kd_subset is not None:
        kp_scale = float(row["kp_scale"])
        kd_scale = float(row["kd_scale"])
        return (
            [float(value * kp_scale) for value in base_kp_subset],
            [float(value * kd_scale) for value in base_kd_subset],
        )

    if not allow_scale_fallback:
        subset_name = row.get("subset_name", "<unknown>")
        raise RuntimeError(
            f"{subset_name}: summary.json does not contain baseline/applied gain metadata. "
            "Re-run tune_gains.py with the updated code, or pass --allow-scale-fallback to reuse current YAML gains."
        )

    kp_scale = float(row["kp_scale"])
    kd_scale = float(row["kd_scale"])
    warnings.append(
        f"{row.get('subset_name', '<unknown>')}: summary lacks exact gain metadata, "
        "so current joint profile values were rescaled"
    )
    return (
        [float(joint_profile["kp"][idx]) * kp_scale for idx in indices],
        [float(joint_profile["kd"][idx]) * kd_scale for idx in indices],
    )


def apply_single_axis_best(
    summary: dict,
    joint_profile: dict,
    allow_scale_fallback: bool,
) -> tuple[dict, list[dict], list[str]]:
    kp = list(joint_profile["kp"])
    kd = list(joint_profile["kd"])
    applied = []
    warnings = []

    adaptive_cfg = summary.get("adaptive", {})
    kp_min_scale, kp_max_scale = adaptive_cfg.get("kp_scale_range", [None, None])
    kd_min_scale, kd_max_scale = adaptive_cfg.get("kd_scale_range", [None, None])

    for subset_name, row in summary.get("best_by_target", {}).items():
        indices = [int(idx) for idx in row.get("subset_indices", [])]
        if len(indices) != 1:
            continue

        idx = indices[0]
        kp_scale = float(row["kp_scale"])
        kd_scale = float(row["kd_scale"])
        base_kp = float(kp[idx])
        base_kd = float(kd[idx])
        applied_kp_subset, applied_kd_subset = resolve_applied_subset_gains(
            row=row,
            joint_profile=joint_profile,
            allow_scale_fallback=allow_scale_fallback,
            warnings=warnings,
        )
        new_kp = float(applied_kp_subset[0])
        new_kd = float(applied_kd_subset[0])

        kp[idx] = new_kp
        kd[idx] = new_kd

        applied.append(
            {
                "subset_name": subset_name,
                "joint_index": idx,
                "joint_name": joint_name(idx),
                "base_kp": base_kp,
                "base_kd": base_kd,
                "kp_scale": kp_scale,
                "kd_scale": kd_scale,
                "new_kp": new_kp,
                "new_kd": new_kd,
                "cost": row.get("cost"),
            }
        )

        if kp_max_scale is not None and abs(kp_scale - float(kp_max_scale)) < 1e-6:
            warnings.append(f"{subset_name}: kp_scale hit upper bound ({kp_scale})")
        if kp_min_scale is not None and abs(kp_scale - float(kp_min_scale)) < 1e-6:
            warnings.append(f"{subset_name}: kp_scale hit lower bound ({kp_scale})")
        if kd_max_scale is not None and abs(kd_scale - float(kd_max_scale)) < 1e-6:
            warnings.append(f"{subset_name}: kd_scale hit upper bound ({kd_scale})")
        if kd_min_scale is not None and abs(kd_scale - float(kd_min_scale)) < 1e-6:
            warnings.append(f"{subset_name}: kd_scale hit lower bound ({kd_scale})")

    updated = dict(joint_profile)
    updated["kp"] = kp
    updated["kd"] = kd
    return updated, applied, warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=str, help="Path to tune_gains summary.json")
    parser.add_argument("--joint-profile", type=str, default=str(default_joint_profile_path()))
    parser.add_argument("--output", type=str, default=None, help="Output YAML path. Default: <joint_profile>_autotuned.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the input joint profile path instead of writing a new file")
    parser.add_argument(
        "--allow-scale-fallback",
        action="store_true",
        help="If summary lacks exact baseline/applied gains, rescale the current joint profile values instead",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary).expanduser()
    joint_profile_path = Path(args.joint_profile).expanduser()
    output_path = joint_profile_path if args.overwrite else (Path(args.output).expanduser() if args.output else default_output_path(joint_profile_path))

    summary = load_summary(summary_path)
    joint_profile = load_joint_profile_yaml(joint_profile_path)
    updated, applied, warnings = apply_single_axis_best(
        summary=summary,
        joint_profile=joint_profile,
        allow_scale_fallback=args.allow_scale_fallback,
    )

    if not applied:
        raise RuntimeError("No single-axis best results were found in summary.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(updated, file_obj, sort_keys=False)

    print(f"summary: {summary_path}")
    print(f"joint_profile_in: {joint_profile_path}")
    print(f"joint_profile_out: {output_path}")
    print("applied:")
    for row in applied:
        print(
            f"  {row['joint_name']}: "
            f"kp {row['base_kp']:.4f} * {row['kp_scale']:.4f} -> {row['new_kp']:.4f}, "
            f"kd {row['base_kd']:.4f} * {row['kd_scale']:.4f} -> {row['new_kd']:.4f}"
        )
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
