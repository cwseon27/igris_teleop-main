#!/usr/bin/env python3
"""
Visualize why the waist motor limits can be more restrictive than the pelvis
joint-space limits when the current PR->AB transform is applied.

Inputs:
- Joint/motor limits from `third_party/igris_c_sdk_public/examples/sdk_gui_client.cpp`
- PR->AB transform from a PR2AB calibration YAML

Outputs:
- Left subplot: PR space (roll/pitch) with the joint rectangle and the region
  that remains feasible under the motor limits.
- Right subplot: AB space (waist_l/waist_r) with the motor-limit box and the
  mapped joint-limit rectangle.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GUI_PATH = REPO_ROOT / "third_party" / "igris_c_sdk_public" / "examples" / "sdk_gui_client.cpp"
DEFAULT_PR2AB_PATH = REPO_ROOT / "igris_artifacts" / "logs" / "robot_control" / "pr2ab_calibration.yaml"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "igris_teleop" / "robot_control" / "diagnostics" / "outputs" / "waist_limit_explanation.png"


def _parse_array(source: str, array_name: str) -> list[float]:
    pattern = re.compile(
        rf"{re.escape(array_name)}\s*=\s*\{{(?P<body>.*?)\}};",
        re.DOTALL,
    )
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"Array not found in GUI source: {array_name}")

    body = re.sub(r"//.*", "", match.group("body"))
    values: list[float] = []
    for chunk in body.split(","):
        token = chunk.strip()
        if not token:
            continue
        token = token.rstrip("f")
        values.append(float(token))
    return values


def load_gui_limits(gui_path: Path) -> dict[str, np.ndarray]:
    source = gui_path.read_text(encoding="utf-8")
    joint_pos_max = np.asarray(_parse_array(source, "JOINT_POS_MAX"), dtype=np.float64)
    joint_pos_min = np.asarray(_parse_array(source, "JOINT_POS_MIN"), dtype=np.float64)
    motor_pos_max = np.asarray(_parse_array(source, "MOTOR_POS_MAX"), dtype=np.float64)
    motor_pos_min = np.asarray(_parse_array(source, "MOTOR_POS_MIN"), dtype=np.float64)
    if joint_pos_max.size != 31 or joint_pos_min.size != 31:
        raise ValueError("Unexpected joint limit array length in GUI source")
    if motor_pos_max.size != 31 or motor_pos_min.size != 31:
        raise ValueError("Unexpected motor limit array length in GUI source")
    return {
        "joint_roll": np.array([joint_pos_min[1], joint_pos_max[1]], dtype=np.float64),
        "joint_pitch": np.array([joint_pos_min[2], joint_pos_max[2]], dtype=np.float64),
        "motor_l": np.array([motor_pos_min[1], motor_pos_max[1]], dtype=np.float64),
        "motor_r": np.array([motor_pos_min[2], motor_pos_max[2]], dtype=np.float64),
    }


def load_waist_pair(calibration_path: Path) -> dict[str, np.ndarray]:
    payload = yaml.safe_load(calibration_path.read_text(encoding="utf-8")) or {}
    pairs = payload.get("pairs", payload)
    item = pairs["waist_rp"]
    return {
        "M": np.asarray(item["M"], dtype=np.float64).reshape(2, 2),
        "pr_center": np.asarray(item.get("pr_center", [0.0, 0.0]), dtype=np.float64).reshape(2),
        "ab_center": np.asarray(item.get("ab_center", [0.0, 0.0]), dtype=np.float64).reshape(2),
    }


def pr_to_ab(pr: np.ndarray, M: np.ndarray, pr_center: np.ndarray, ab_center: np.ndarray) -> np.ndarray:
    return ab_center + (M @ (pr - pr_center).T).T


def build_joint_rectangle(roll_limits: np.ndarray, pitch_limits: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [roll_limits[0], pitch_limits[0]],
            [roll_limits[0], pitch_limits[1]],
            [roll_limits[1], pitch_limits[1]],
            [roll_limits[1], pitch_limits[0]],
            [roll_limits[0], pitch_limits[0]],
        ],
        dtype=np.float64,
    )


def plot_pr_space(
    ax,
    roll_limits: np.ndarray,
    pitch_limits: np.ndarray,
    motor_l_limits: np.ndarray,
    motor_r_limits: np.ndarray,
    M: np.ndarray,
    pr_center: np.ndarray,
    ab_center: np.ndarray,
) -> None:
    roll_margin = 0.08
    pitch_margin = 0.08
    roll_vals = np.linspace(roll_limits[0] - roll_margin, roll_limits[1] + roll_margin, 501)
    pitch_vals = np.linspace(pitch_limits[0] - pitch_margin, pitch_limits[1] + pitch_margin, 501)
    grid_roll, grid_pitch = np.meshgrid(roll_vals, pitch_vals)
    pr_grid = np.stack((grid_roll, grid_pitch), axis=-1).reshape(-1, 2)
    ab_grid = pr_to_ab(pr_grid, M=M, pr_center=pr_center, ab_center=ab_center)
    feasible = (
        (ab_grid[:, 0] >= motor_l_limits[0])
        & (ab_grid[:, 0] <= motor_l_limits[1])
        & (ab_grid[:, 1] >= motor_r_limits[0])
        & (ab_grid[:, 1] <= motor_r_limits[1])
    ).reshape(grid_roll.shape)

    ax.contourf(
        grid_roll,
        grid_pitch,
        feasible.astype(np.float32),
        levels=[-0.1, 0.5, 1.1],
        colors=["#ffffff", "#cfe8ff"],
        alpha=0.9,
    )

    rect = build_joint_rectangle(roll_limits, pitch_limits)
    ax.fill(rect[:, 0], rect[:, 1], facecolor="#ffb266", alpha=0.2, edgecolor="#cc6d00", linewidth=2.0)
    ax.plot(rect[:, 0], rect[:, 1], color="#cc6d00", linewidth=2.0, label="Pelvis joint limits")

    samples = np.array(
        [
            [roll_limits[0], pitch_limits[0]],
            [roll_limits[0], pitch_limits[1]],
            [roll_limits[1], pitch_limits[0]],
            [roll_limits[1], pitch_limits[1]],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    sample_ab = pr_to_ab(samples, M=M, pr_center=pr_center, ab_center=ab_center)
    sample_inside = (
        (sample_ab[:, 0] >= motor_l_limits[0])
        & (sample_ab[:, 0] <= motor_l_limits[1])
        & (sample_ab[:, 1] >= motor_r_limits[0])
        & (sample_ab[:, 1] <= motor_r_limits[1])
    )
    for idx, point in enumerate(samples):
        color = "#1976d2" if sample_inside[idx] else "#d32f2f"
        ax.scatter(point[0], point[1], color=color, s=28, zorder=5)

    ax.set_title("PR Space: roll / pitch")
    ax.set_xlabel("waist_roll [rad]")
    ax.set_ylabel("waist_pitch [rad]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    ax.text(
        0.02,
        0.98,
        "Blue area: PR commands whose mapped AB stays inside\n"
        "sdk_gui_client.cpp motor limits",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#dddddd"},
    )


def plot_ab_space(
    ax,
    joint_rect_pr: np.ndarray,
    motor_l_limits: np.ndarray,
    motor_r_limits: np.ndarray,
    M: np.ndarray,
    pr_center: np.ndarray,
    ab_center: np.ndarray,
) -> None:
    joint_rect_ab = pr_to_ab(joint_rect_pr[:-1], M=M, pr_center=pr_center, ab_center=ab_center)
    joint_rect_ab = np.vstack((joint_rect_ab, joint_rect_ab[0]))

    motor_box = np.array(
        [
            [motor_l_limits[0], motor_r_limits[0]],
            [motor_l_limits[0], motor_r_limits[1]],
            [motor_l_limits[1], motor_r_limits[1]],
            [motor_l_limits[1], motor_r_limits[0]],
            [motor_l_limits[0], motor_r_limits[0]],
        ],
        dtype=np.float64,
    )

    ax.fill(motor_box[:, 0], motor_box[:, 1], facecolor="#cfe8ff", alpha=0.5, edgecolor="#1976d2", linewidth=2.0)
    ax.plot(motor_box[:, 0], motor_box[:, 1], color="#1976d2", linewidth=2.0, label="Motor limits")
    ax.fill(joint_rect_ab[:, 0], joint_rect_ab[:, 1], facecolor="#ffb266", alpha=0.2, edgecolor="#cc6d00", linewidth=2.0)
    ax.plot(joint_rect_ab[:, 0], joint_rect_ab[:, 1], color="#cc6d00", linewidth=2.0, label="Mapped pelvis joint rectangle")

    corner_labels = ["LL", "LU", "RU", "RL"]
    for label, point in zip(corner_labels, joint_rect_ab[:-1], strict=True):
        ax.scatter(point[0], point[1], color="#cc6d00", s=30, zorder=5)
        ax.text(point[0] + 0.02, point[1] + 0.02, label, fontsize=8, color="#7a3b00")

    ax.set_title("AB Space: Waist_L / Waist_R")
    ax.set_xlabel("waist_l [rad]")
    ax.set_ylabel("waist_r [rad]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=Path, default=DEFAULT_GUI_PATH)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_PR2AB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    gui_limits = load_gui_limits(args.gui)
    waist_pair = load_waist_pair(args.calibration)

    joint_rect_pr = build_joint_rectangle(gui_limits["joint_roll"], gui_limits["joint_pitch"])

    fig, (ax_pr, ax_ab) = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    plot_pr_space(
        ax_pr,
        roll_limits=gui_limits["joint_roll"],
        pitch_limits=gui_limits["joint_pitch"],
        motor_l_limits=gui_limits["motor_l"],
        motor_r_limits=gui_limits["motor_r"],
        M=waist_pair["M"],
        pr_center=waist_pair["pr_center"],
        ab_center=waist_pair["ab_center"],
    )
    plot_ab_space(
        ax_ab,
        joint_rect_pr=joint_rect_pr,
        motor_l_limits=gui_limits["motor_l"],
        motor_r_limits=gui_limits["motor_r"],
        M=waist_pair["M"],
        pr_center=waist_pair["pr_center"],
        ab_center=waist_pair["ab_center"],
    )

    M = waist_pair["M"]
    pr_center = waist_pair["pr_center"]
    ab_center = waist_pair["ab_center"]
    fig.suptitle(
        "Waist joint limits vs motor limits under current PR->AB transform\n"
        f"ab = ab_center + M @ (pr - pr_center), "
        f"M=[[{M[0,0]:+.3f}, {M[0,1]:+.3f}], [{M[1,0]:+.3f}, {M[1,1]:+.3f}]], "
        f"pr_center=[{pr_center[0]:+.3f}, {pr_center[1]:+.3f}], "
        f"ab_center=[{ab_center[0]:+.3f}, {ab_center[1]:+.3f}]",
        fontsize=11,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
