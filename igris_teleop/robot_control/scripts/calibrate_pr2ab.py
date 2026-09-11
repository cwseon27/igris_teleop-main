"""
2-DOF parallel pair PR->AB transform calibration script.

Supported pairs:
  - waist_rp
  - l_ankle_pr, r_ankle_pr
  - l_wrist_rp, r_wrist_rp

Model:
  [ab1]   = M @ [pr1] + [b1]
  [ab2]         [pr2]   [b2]

Equivalently, the calibration YAML stores:
  pr_center = [0, 0]
  ab_center = [b1, b2]

Run command
python igris_teleop/robot_control/scripts/calibrate_pr2ab.py --pair all
python igris_teleop/robot_control/scripts/calibrate_pr2ab.py --pair all --task verify
"""

import math
import signal
import sys
import time
import pathlib
import argparse
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add repository root so `igris_teleop.*` package imports resolve in script mode.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ROBOT_CONTROL_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import igris_c_sdk as igc_sdk
from igris_teleop.core.project_paths import ROBOT_CONTROL_LOGS_ROOT
from igris_teleop.robot_control.controller.igris_controller import BaseController
from igris_teleop.robot_control.kinematics.joints import (
    JointIndex,
    MotorIndex,
    ARM_INDICES,
    WAIST_INDICES,
    LEG_INDICES,
)

DEFAULT_OUTPUT_DIR = ROBOT_CONTROL_LOGS_ROOT


@dataclass(frozen=True)
class PairSpec:
    name: str
    group: str
    pr_indices: tuple[int, int]
    ab_indices: tuple[int, int]
    pr_joint_names: tuple[str, str]
    default_amp: tuple[float, float]


PAIR_SPECS = {
    "waist_rp": PairSpec(
        name="waist_rp",
        group="waist",
        pr_indices=(int(JointIndex.WAIST_ROLL), int(JointIndex.WAIST_PITCH)),
        ab_indices=(int(MotorIndex.WAIST_L), int(MotorIndex.WAIST_R)),
        pr_joint_names=("1_Joint_Waist_Roll", "0_Joint_Waist_Pitch"),
        default_amp=(0.20, 0.20),
    ),
    "l_ankle_pr": PairSpec(
        name="l_ankle_pr",
        group="leg",
        pr_indices=(int(JointIndex.L_ANKLE_PITCH), int(JointIndex.L_ANKLE_ROLL)),
        ab_indices=(int(MotorIndex.ANKLE_OUT_L), int(MotorIndex.ANKLE_IN_L)),
        pr_joint_names=("7_Joint_Ankle_Pitch_Left", "8_Joint_Ankle_Roll_Left"),
        default_amp=(0.20, 0.15),
    ),
    "r_ankle_pr": PairSpec(
        name="r_ankle_pr",
        group="leg",
        pr_indices=(int(JointIndex.R_ANKLE_PITCH), int(JointIndex.R_ANKLE_ROLL)),
        ab_indices=(int(MotorIndex.ANKLE_OUT_R), int(MotorIndex.ANKLE_IN_R)),
        pr_joint_names=("13_Joint_Ankle_Pitch_Right", "14_Joint_Ankle_Roll_Right"),
        default_amp=(0.20, 0.15),
    ),
    "l_wrist_rp": PairSpec(
        name="l_wrist_rp",
        group="arm",
        pr_indices=(int(JointIndex.L_WRIST_ROLL), int(JointIndex.L_WRIST_PITCH)),
        ab_indices=(int(MotorIndex.WRIST_FRONT_L), int(MotorIndex.WRIST_BACK_L)),
        pr_joint_names=("20_Joint_Wrist_Roll_Left", "21_Joint_Wrist_Pitch_Left"),
        default_amp=(0.50, 0.45),
    ),
    "r_wrist_rp": PairSpec(
        name="r_wrist_rp",
        group="arm",
        pr_indices=(int(JointIndex.R_WRIST_ROLL), int(JointIndex.R_WRIST_PITCH)),
        ab_indices=(int(MotorIndex.WRIST_FRONT_R), int(MotorIndex.WRIST_BACK_R)),
        pr_joint_names=("27_Joint_Wrist_Roll_Right", "28_Joint_Wrist_Pitch_Right"),
        default_amp=(0.50, 0.45),
    ),
}


def _get_group_context(ctrl: BaseController, group: str, q_now: np.ndarray):
    if group == "arm":
        group_indices = list(ARM_INDICES)
        setter = ctrl.ctrl_arm
    elif group == "waist":
        group_indices = list(WAIST_INDICES)
        setter = ctrl.ctrl_waist
    elif group == "leg":
        group_indices = list(LEG_INDICES)
        setter = ctrl.ctrl_leg
    else:
        raise ValueError(f"Unknown group: {group}")

    group_base = [float(q_now[i]) for i in group_indices]
    return group_indices, group_base, setter


def _upsert_calib_file(
    calib_path: pathlib.Path,
    pair_name: str,
    M_est: np.ndarray,
    pr_center: tuple[float, float],
    pr_limits: tuple[tuple[float, float], tuple[float, float]],
    ab_center: tuple[float, float],
    sample_count: int,
):
    payload = {}
    if calib_path.is_file():
        with calib_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    pairs = payload.setdefault("pairs", {})
    pairs[pair_name] = {
        "fit_model": "affine_intercept",
        "M": np.asarray(M_est, dtype=float).tolist(),
        "pr_center": [float(pr_center[0]), float(pr_center[1])],
        "pr_limits": [[float(pr_limits[0][0]), float(pr_limits[0][1])], [float(pr_limits[1][0]), float(pr_limits[1][1])]],
        "ab_center": [float(ab_center[0]), float(ab_center[1])],
        "ab_at_pr_zero": [float(ab_center[0]), float(ab_center[1])],
        "sample_count": int(sample_count),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    with calib_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True, allow_unicode=False)


def _upsert_verify_file(
    verify_path: pathlib.Path,
    pair_name: str,
    result: dict,
):
    payload = {}
    if verify_path.is_file():
        with verify_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

    pairs = payload.setdefault("pairs", {})
    item = pairs.setdefault(pair_name, {})
    history = item.setdefault("history", [])
    history.append(result)
    item["last"] = result
    item["updated_at"] = result.get("updated_at", time.strftime("%Y-%m-%d %H:%M:%S"))

    verify_path.parent.mkdir(parents=True, exist_ok=True)
    with verify_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True, allow_unicode=False)


def _load_joint_limits_from_urdf(urdf_path: pathlib.Path) -> dict[str, tuple[float, float]]:
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = limit.get("lower")
        upper = limit.get("upper")
        if lower is None or upper is None:
            continue
        try:
            limits[name] = (float(lower), float(upper))
        except ValueError:
            continue
    return limits


def _pair_limits_from_urdf(
    pair: PairSpec,
    joint_limits: dict[str, tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    j1, j2 = pair.pr_joint_names
    if j1 not in joint_limits:
        raise KeyError(f"Joint limit not found in URDF: {j1}")
    if j2 not in joint_limits:
        raise KeyError(f"Joint limit not found in URDF: {j2}")
    return joint_limits[j1], joint_limits[j2]


def _load_pair_calibration(
    calib_path: pathlib.Path,
    pair_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not calib_path.is_file():
        raise FileNotFoundError(f"Calibration YAML not found: {calib_path}")

    with calib_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    pairs = payload.get("pairs", payload)
    if pair_name not in pairs:
        raise KeyError(f"Pair '{pair_name}' not found in calibration file: {calib_path}")

    item = pairs[pair_name]
    M = np.asarray(item["M"], dtype=np.float64).reshape(2, 2)
    pr_center = np.asarray(item.get("pr_center", [0.0, 0.0]), dtype=np.float64).reshape(2)
    ab_center = np.asarray(item.get("ab_center", [0.0, 0.0]), dtype=np.float64).reshape(2)
    return M, pr_center, ab_center


def _move_pair_to_center(
    ctrl: BaseController,
    pair: PairSpec,
    q_now: np.ndarray,
    center_1: float,
    center_2: float,
    duration: float,
):
    q_goal = np.asarray(q_now, dtype=np.float32).copy()
    q_goal[pair.pr_indices[0]] = float(center_1)
    q_goal[pair.pr_indices[1]] = float(center_2)

    ctrl.move_to_pose(
        q_goal,
        duration=duration,
        leg=(pair.group == "leg"),
        waist=(pair.group == "waist"),
        arm=(pair.group == "arm"),
        neck=False,
    )
    ctrl.default_pos_state(
        pose=q_goal,
        leg=(pair.group == "leg"),
        waist=(pair.group == "waist"),
        arm=(pair.group == "arm"),
        neck=False,
    )


def wrist_rp_to_motor_cmd(
    roll_des: float,
    pitch_des: float,
    transform: np.ndarray,
    wrist_center=(0.0, 0.0),
    motor_center=(0.0, 0.0),
    motor_limits=None,
):
    """
    Convert IK wrist (roll, pitch) target to differential motor commands (qf, qb).

    Model (delta form):
        [qf - qf0]   = M @ [roll  - roll0 ]
        [qb - qb0]         [pitch - pitch0]
    """
    M = np.asarray(transform, dtype=np.float64).reshape(2, 2)
    rp_des = np.asarray([roll_des, pitch_des], dtype=np.float64)
    rp0 = np.asarray(wrist_center, dtype=np.float64).reshape(2)
    q0 = np.asarray(motor_center, dtype=np.float64).reshape(2)

    q_cmd = q0 + M @ (rp_des - rp0)

    if motor_limits is not None:
        lo = np.asarray([motor_limits[0][0], motor_limits[1][0]], dtype=np.float64)
        hi = np.asarray([motor_limits[0][1], motor_limits[1][1]], dtype=np.float64)
        q_cmd = np.clip(q_cmd, lo, hi)

    return float(q_cmd[0]), float(q_cmd[1])


def motor_cmd_to_wrist_rp(
    qf: float,
    qb: float,
    transform: np.ndarray,
    wrist_center=(0.0, 0.0),
    motor_center=(0.0, 0.0),
):
    """
    Inverse mapping for debug/verification: (qf, qb) -> (roll, pitch).
    """
    M = np.asarray(transform, dtype=np.float64).reshape(2, 2)
    q = np.asarray([qf, qb], dtype=np.float64)
    rp0 = np.asarray(wrist_center, dtype=np.float64).reshape(2)
    q0 = np.asarray(motor_center, dtype=np.float64).reshape(2)

    rp = rp0 + np.linalg.solve(M, q - q0)
    return float(rp[0]), float(rp[1])


def wait_for_key(key: str, stop_flag_fn) -> bool:
    """
    터미널에서 특정 키 입력을 대기.
    stop_flag_fn() 이 True 가 되면 False 반환 (중단).
    key 입력 시 True 반환.
    """
    import tty, termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print(f"\n[대기] '{key}' 키를 누르면 sweep 시작, Ctrl+C 는 언제든 종료")
    try:
        tty.setraw(fd)
        while not stop_flag_fn():
            # non-blocking read
            import select
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                ch = sys.stdin.read(1)
                if ch.lower() == key:
                    return True
                # Ctrl+C (ETX)
                if ch == "\x03":
                    raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return False   # stop 요청으로 빠져나옴


def sweep_sin(t, center, amp, freq):
    """sin 파형으로 center 기준 ±amp 왕복"""
    return center + amp * math.sin(2.0 * math.pi * freq * t)


def run_phase(
    ctrl,
    group_setter,
    group_base,
    group_pos_1,
    group_pos_2,
    center_1,
    center_2,
    amp_1,
    amp_2,
    freq_1,
    freq_2,
    read_pr_idx_1,
    read_pr_idx_2,
    read_ab_idx_1,
    read_ab_idx_2,
    cmd_limits,
    duration,
    dt,
    stop_flag_fn,
    use_measured_pr=False,
):
    """
    한 Phase 동안 sin 구동하면서 (x_1, x_2, ab_1, ab_2) 수집.
    - use_measured_pr=False: x는 command (cmd_1, cmd_2)
    - use_measured_pr=True : x는 get_joint_q()로 읽은 PR 실측값
    Returns: np.ndarray shape (N, 4)
    """
    data = []
    start = time.perf_counter()

    while True:
        t = time.perf_counter() - start
        if t >= duration or stop_flag_fn():
            break

        cmd_1 = sweep_sin(t, center_1, amp_1, freq_1)
        cmd_2 = sweep_sin(t, center_2, amp_2, freq_2)

        if cmd_limits is not None:
            cmd_1 = float(np.clip(cmd_1, cmd_limits[0][0], cmd_limits[0][1]))
            cmd_2 = float(np.clip(cmd_2, cmd_limits[1][0], cmd_limits[1][1]))
        else:
            cmd_1 = float(cmd_1)
            cmd_2 = float(cmd_2)

        group_cmd = list(group_base)
        group_cmd[group_pos_1] = cmd_1
        group_cmd[group_pos_2] = cmd_2
        group_setter(group_cmd, apply_clip=False)

        q_joint = ctrl.get_joint_q()
        q_motor = ctrl.get_motor_q()
        if q_motor is not None and (not use_measured_pr or q_joint is not None):
            if use_measured_pr:
                x_1 = float(q_joint[read_pr_idx_1])
                x_2 = float(q_joint[read_pr_idx_2])
            else:
                x_1 = cmd_1
                x_2 = cmd_2

            ab_1 = float(q_motor[read_ab_idx_1])
            ab_2 = float(q_motor[read_ab_idx_2])
            data.append([x_1, x_2, ab_1, ab_2])

            if use_measured_pr:
                print(
                    f"  [t={t:.2f}s] cmd=({cmd_1:+.3f},{cmd_2:+.3f}) "
                    f"pr_meas=({x_1:+.3f},{x_2:+.3f}) "
                    f"| ab=({ab_1:+.3f},{ab_2:+.3f})"
                )
            else:
                print(
                    f"  [t={t:.2f}s] cmd_1={cmd_1:+.3f} cmd_2={cmd_2:+.3f} "
                    f"| ab_1={ab_1:+.3f} ab_2={ab_2:+.3f}"
                )

        time.sleep(dt)

    return np.array(data) if data else np.zeros((0, 4))


def estimate_transform(data):
    """
    data: (N,4) -> x_1, x_2, ab_1, ab_2
    최소자승으로 affine 추정:
        [ab_1, ab_2]^T = M @ [x_1, x_2]^T + b
    즉, pr=0일 때 ms!=0 인 경우를 허용한다.
    Returns: M (2x2), b (2,), residuals
    """
    X = data[:, :2]   # (N,2) PR input for regression (cmd or measured PR)
    Y = data[:, 2:]   # (N,2) ab_1, ab_2

    # Solve Y = X @ M^T + b using an augmented design matrix.
    X_aug = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], axis=1)
    theta, res, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
    M = theta[:2, :].T
    b = theta[2, :]
    return M, b, res


def plot_results(data_phases, phase_names, M_est, ab_offset, pair_name, save_path):
    """결과 플롯: 각 Phase scatter + 추정행렬 vs 이론행렬"""
    M_theory = np.array([[1.0, 1.0],
                         [1.0, -1.0]])

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = ["steelblue", "tomato", "seagreen"]

    # ── 상단 2행: Phase별 ab_1/ab_2 vs x_1 / x_2 ──────────────
    for pi, (data, name, col) in enumerate(zip(data_phases, phase_names, colors)):
        if len(data) == 0:
            continue
        cmd_1 = data[:, 0]
        cmd_2 = data[:, 1]
        ab_1 = data[:, 2]
        ab_2 = data[:, 3]

        ax1 = fig.add_subplot(gs[pi // 1, 0] if pi == 0 else
                              gs[0, pi] if pi < 3 else gs[1, pi - 3])
        # 위치 재정의 (간단하게)
        ax_qf = fig.add_subplot(gs[0, pi])
        ax_qb = fig.add_subplot(gs[1, pi])

        ax_qf.scatter(cmd_1, ab_1,  s=5, c=col, alpha=0.5, label="ab_1 vs x_1")
        ax_qf.scatter(cmd_2, ab_1, s=5, c="orange", alpha=0.5, label="ab_1 vs x_2")
        ax_qf.set_title(f"{name}\nab_1 response", fontsize=9)
        ax_qf.set_xlabel("x [rad]", fontsize=8)
        ax_qf.set_ylabel("ab_1 [rad]",  fontsize=8)
        ax_qf.legend(fontsize=7)
        ax_qf.grid(True)

        ax_qb.scatter(cmd_1,  ab_2, s=5, c=col,      alpha=0.5, label="ab_2 vs x_1")
        ax_qb.scatter(cmd_2, ab_2, s=5, c="orange", alpha=0.5, label="ab_2 vs x_2")
        ax_qb.set_title(f"{name}\nab_2 response", fontsize=9)
        ax_qb.set_xlabel("x [rad]", fontsize=8)
        ax_qb.set_ylabel("ab_2 [rad]",  fontsize=8)
        ax_qb.legend(fontsize=7)
        ax_qb.grid(True)

    # ── 하단: 추정 M vs 이론 M ────────────────────────────────────────
    ax_mat = fig.add_subplot(gs[2, :])
    ax_mat.axis("off")

    err   = M_est - M_theory
    table_data = [
        ["", "col: x_1", "col: x_2", "", "col: x_1", "col: x_2", "", "col: x_1", "col: x_2"],
        ["row: ab_1",
         f"{M_est[0,0]:+.4f}", f"{M_est[0,1]:+.4f}",
         "",
         f"{M_theory[0,0]:+.4f}", f"{M_theory[0,1]:+.4f}",
         "",
         f"{err[0,0]:+.4f}", f"{err[0,1]:+.4f}"],
        ["row: ab_2",
         f"{M_est[1,0]:+.4f}", f"{M_est[1,1]:+.4f}",
         "",
         f"{M_theory[1,0]:+.4f}", f"{M_theory[1,1]:+.4f}",
         "",
         f"{err[1,0]:+.4f}", f"{err[1,1]:+.4f}"],
    ]

    col_labels = ["", "M_est[0]", "M_est[1]", "",
                      "M_theory[0]", "M_theory[1]", "",
                      "Error[0]", "Error[1]"]

    tbl = ax_mat.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    ax_mat.set_title(
        "Estimated linear term M_est vs theory M_theory = [[1,1],[1,-1]]",
        fontsize=11, pad=8
    )

    fig.text(
        0.5,
        0.02,
        f"Affine offset (ms at pr=0): [{ab_offset[0]:+.4f}, {ab_offset[1]:+.4f}]",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    fig.suptitle(f"{pair_name}: PR->AB Transform Identification", fontsize=13, y=0.98)

    save_path = pathlib.Path(save_path).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"\n[결과 저장] {save_path}")


def _execute_pair_task(
    ctrl: BaseController,
    args,
    pair_name: str,
    joint_limits: dict[str, tuple[float, float]],
    save_dir: pathlib.Path,
    calib_path: pathlib.Path,
    verify_path: pathlib.Path,
    stop_flag_fn,
) -> int:
    pair = PAIR_SPECS[pair_name]
    cmd_limits = _pair_limits_from_urdf(pair, joint_limits)
    zero_center_1 = float(np.clip(0.0, cmd_limits[0][0], cmd_limits[0][1]))
    zero_center_2 = float(np.clip(0.0, cmd_limits[1][0], cmd_limits[1][1]))
    print(
        f"[limit] {pair.name}: {pair.pr_joint_names[0]}={cmd_limits[0]}, "
        f"{pair.pr_joint_names[1]}={cmd_limits[1]}"
    )
    print(f"[시작] pair={pair.name} sweep 시작\n")

    q_now = ctrl.get_joint_q()
    q_motor_now = ctrl.get_motor_q()
    if q_now is None:
        print("[error] Missing joint state after init")
        return 1
    if q_motor_now is None:
        print("[error] Missing motor state after init")
        return 1

    if args.task == "calib":
        print(
            f"[center] calib sweep center -> ({zero_center_1:+.3f}, {zero_center_2:+.3f}) "
            f"(pair 2축만 zero 기준으로 이동)"
        )
        _move_pair_to_center(
            ctrl=ctrl,
            pair=pair,
            q_now=q_now,
            center_1=zero_center_1,
            center_2=zero_center_2,
            duration=min(args.move_duration, 2.0),
        )
        time.sleep(0.3)
        q_now = ctrl.get_joint_q()
        q_motor_now = ctrl.get_motor_q()
        if q_now is None or q_motor_now is None:
            print("[error] Missing state after centering move")
            return 1

    group_indices, group_base, group_setter = _get_group_context(ctrl, pair.group, q_now)
    pr_idx_1, pr_idx_2 = pair.pr_indices
    ab_idx_1, ab_idx_2 = pair.ab_indices

    group_pos_1 = group_indices.index(pr_idx_1)
    group_pos_2 = group_indices.index(pr_idx_2)

    center_1 = zero_center_1 if args.task == "calib" else float(q_now[pr_idx_1])
    center_2 = zero_center_2 if args.task == "calib" else float(q_now[pr_idx_2])

    dt = args.dt
    if args.task == "verify":
        M_cfg, pr_center_cfg, ab_center_cfg = _load_pair_calibration(calib_path, pair.name)
        des_center_1_raw = center_1 if args.axis1_des is None else float(args.axis1_des)
        des_center_2_raw = center_2 if args.axis2_des is None else float(args.axis2_des)
        des_center_1 = float(np.clip(des_center_1_raw, cmd_limits[0][0], cmd_limits[0][1]))
        des_center_2 = float(np.clip(des_center_2_raw, cmd_limits[1][0], cmd_limits[1][1]))
        if des_center_1 != des_center_1_raw or des_center_2 != des_center_2_raw:
            print(
                f"[verify] center target clipped by URDF limit: "
                f"({des_center_1_raw:+.3f}, {des_center_2_raw:+.3f}) -> "
                f"({des_center_1:+.3f}, {des_center_2:+.3f})"
            )
        traj_amp_1 = float(args.axis1_amp) if args.axis1_amp is not None else float(pair.default_amp[0])
        traj_amp_2 = float(args.axis2_amp) if args.axis2_amp is not None else float(pair.default_amp[1])

        print("\n[verify] loaded calibration")
        print(f"  pair={pair.name}")
        print(f"  M={M_cfg.tolist()}")
        print(f"  pr_center={pr_center_cfg.tolist()}")
        print(f"  ab_center={ab_center_cfg.tolist()}")
        M_inv_cfg = np.linalg.inv(M_cfg)
        verify_result = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "verify",
            "pair": pair.name,
            "mode": "phase" if args.verify_phases else ("trajectory" if args.verify_traj else "point"),
            "sample_dt": float(dt),
            "desired_center": [float(des_center_1), float(des_center_2)],
            "pr_center_cfg": [float(pr_center_cfg[0]), float(pr_center_cfg[1])],
            "ab_center_cfg": [float(ab_center_cfg[0]), float(ab_center_cfg[1])],
            "M": np.asarray(M_cfg, dtype=float).tolist(),
        }
        if args.verify_phases:
            print(
                f"  mode=phase1/2/3, center=({des_center_1:+.3f},{des_center_2:+.3f}), "
                f"amp=({traj_amp_1:+.3f},{traj_amp_2:+.3f}), freq={args.sweep_freq:.3f}Hz"
            )

            f = args.sweep_freq
            dur = args.phase_duration
            phases = [
                (traj_amp_1, 0.0, f, 0.0, "Phase1: axis1 only"),
                (0.0, traj_amp_2, 0.0, f, "Phase2: axis2 only"),
                (traj_amp_1, traj_amp_2, f, f * 0.7, "Phase3: axis1+axis2"),
            ]
            all_des = []
            all_rec = []
            phase_stats = []
            for phase_amp_1, phase_amp_2, phase_freq_1, phase_freq_2, name in phases:
                if stop_flag_fn():
                    break
                print(f"\n{'='*50}")
                print(f"[verify sweep] {name} ({dur}s)")
                print(f"{'='*50}")
                phase_data = run_phase(
                    ctrl=ctrl,
                    group_setter=group_setter,
                    group_base=group_base,
                    group_pos_1=group_pos_1,
                    group_pos_2=group_pos_2,
                    center_1=des_center_1,
                    center_2=des_center_2,
                    amp_1=phase_amp_1,
                    amp_2=phase_amp_2,
                    freq_1=phase_freq_1,
                    freq_2=phase_freq_2,
                    read_pr_idx_1=pr_idx_1,
                    read_pr_idx_2=pr_idx_2,
                    read_ab_idx_1=ab_idx_1,
                    read_ab_idx_2=ab_idx_2,
                    cmd_limits=cmd_limits,
                    duration=dur,
                    dt=dt,
                    stop_flag_fn=stop_flag_fn,
                    use_measured_pr=False,
                )
                if len(phase_data) == 0:
                    print("  -> no samples")
                    continue
                des_phase = np.asarray(phase_data[:, :2], dtype=np.float64)
                ab_phase = np.asarray(phase_data[:, 2:], dtype=np.float64)
                rec_phase = pr_center_cfg + (ab_phase - ab_center_cfg) @ M_inv_cfg.T
                mae_phase = np.mean(np.abs(rec_phase - des_phase), axis=0)
                rmse_phase = np.sqrt(np.mean((rec_phase - des_phase) ** 2, axis=0))
                print(
                    f"  -> samples={len(phase_data)}, "
                    f"MAE=({mae_phase[0]:.4f}, {mae_phase[1]:.4f}) rad, "
                    f"RMSE=({rmse_phase[0]:.4f}, {rmse_phase[1]:.4f}) rad"
                )
                phase_stats.append(
                    {
                        "name": name,
                        "sample_count": int(len(phase_data)),
                        "mae": [float(mae_phase[0]), float(mae_phase[1])],
                        "rmse": [float(rmse_phase[0]), float(rmse_phase[1])],
                    }
                )
                all_des.append(des_phase)
                all_rec.append(rec_phase)

            if not all_rec:
                print("[error] verify-phase 샘플이 수집되지 않았습니다.")
                return 1

            des_arr = np.vstack(all_des)
            rec_arr = np.vstack(all_rec)
            mae_des = np.mean(np.abs(rec_arr - des_arr), axis=0)
            rmse_des = np.sqrt(np.mean((rec_arr - des_arr) ** 2, axis=0))
            print("\n[verify result] recovered(PR) vs desired(PR) across phase1/2/3")
            print(f"  mean abs error: axis1={mae_des[0]:.4f} rad, axis2={mae_des[1]:.4f} rad")
            print(f"  rmse error    : axis1={rmse_des[0]:.4f} rad, axis2={rmse_des[1]:.4f} rad")
            print(f"  last recovered: axis1={rec_arr[-1,0]:+.4f}, axis2={rec_arr[-1,1]:+.4f}")
            verify_result.update(
                {
                    "verify_phases": True,
                    "phase_duration": float(dur),
                    "sweep_freq": float(args.sweep_freq),
                    "trajectory_amp": [float(traj_amp_1), float(traj_amp_2)],
                    "phase_stats": phase_stats,
                    "sample_count": int(len(rec_arr)),
                    "mae_desired": [float(mae_des[0]), float(mae_des[1])],
                    "rmse_desired": [float(rmse_des[0]), float(rmse_des[1])],
                    "last_recovered": [float(rec_arr[-1, 0]), float(rec_arr[-1, 1])],
                }
            )

        else:
            if args.verify_traj:
                print(
                    f"  mode=trajectory(sin), center=({des_center_1:+.3f},{des_center_2:+.3f}), "
                    f"amp=({traj_amp_1:+.3f},{traj_amp_2:+.3f}), freq={args.sweep_freq:.3f}Hz"
                )
            else:
                print(f"  mode=point-hold, target=({des_center_1:+.3f},{des_center_2:+.3f})")

            group_cmd = list(group_base)
            rec_list = []
            joint_list = []
            des_list = []
            t_end = time.perf_counter() + max(args.hold_sec, 0.2)
            last_log_t = 0.0
            t0 = time.perf_counter()
            while time.perf_counter() < t_end and not stop_flag_fn():
                t = time.perf_counter() - t0
                if args.verify_traj:
                    des_1 = sweep_sin(t, des_center_1, traj_amp_1, args.sweep_freq)
                    des_2 = sweep_sin(t, des_center_2, traj_amp_2, args.sweep_freq * 0.7)
                    des_1 = float(np.clip(des_1, cmd_limits[0][0], cmd_limits[0][1]))
                    des_2 = float(np.clip(des_2, cmd_limits[1][0], cmd_limits[1][1]))
                else:
                    des_1 = des_center_1
                    des_2 = des_center_2

                group_cmd[group_pos_1] = des_1
                group_cmd[group_pos_2] = des_2
                group_setter(group_cmd, apply_clip=False)
                q_motor = ctrl.get_motor_q()
                q_joint_now = ctrl.get_joint_q()
                if q_motor is not None:
                    ab_1 = float(q_motor[ab_idx_1])
                    ab_2 = float(q_motor[ab_idx_2])
                    rec_1, rec_2 = motor_cmd_to_wrist_rp(
                        ab_1,
                        ab_2,
                        transform=M_cfg,
                        wrist_center=pr_center_cfg,
                        motor_center=ab_center_cfg,
                    )
                    des_list.append([des_1, des_2])
                    rec_list.append([rec_1, rec_2])
                    if q_joint_now is not None:
                        joint_1 = float(q_joint_now[pr_idx_1])
                        joint_2 = float(q_joint_now[pr_idx_2])
                        joint_list.append([joint_1, joint_2])
                    now = time.perf_counter()
                    if now - last_log_t >= max(args.verify_log_interval, 0.05):
                        print(
                            f"[verify] des=({des_1:+.3f},{des_2:+.3f}) "
                            f"motor=({ab_1:+.3f},{ab_2:+.3f}) "
                            f"rec=({rec_1:+.3f},{rec_2:+.3f})"
                        )
                        last_log_t = now
                time.sleep(dt)

            if not rec_list:
                print("[error] verify 샘플이 수집되지 않았습니다.")
                return 1

            rec_arr = np.asarray(rec_list, dtype=np.float64)
            des_arr = np.asarray(des_list, dtype=np.float64)
            mae_des = np.mean(np.abs(rec_arr - des_arr), axis=0)
            rmse_des = np.sqrt(np.mean((rec_arr - des_arr) ** 2, axis=0))
            print("\n[verify result] recovered(PR) vs desired(PR)")
            print(f"  mean abs error: axis1={mae_des[0]:.4f} rad, axis2={mae_des[1]:.4f} rad")
            print(f"  rmse error    : axis1={rmse_des[0]:.4f} rad, axis2={rmse_des[1]:.4f} rad")
            print(f"  last recovered: axis1={rec_arr[-1,0]:+.4f}, axis2={rec_arr[-1,1]:+.4f}")

            joint_mae = None
            if joint_list:
                joint_arr = np.asarray(joint_list, dtype=np.float64)
                mae_joint = np.mean(np.abs(rec_arr[: len(joint_arr)] - joint_arr), axis=0)
                joint_mae = [float(mae_joint[0]), float(mae_joint[1])]
                print(
                    f"  recovered(PR) vs joint_state(PR) mean abs error: "
                    f"axis1={mae_joint[0]:.4f} rad, axis2={mae_joint[1]:.4f} rad"
                )
            verify_result.update(
                {
                    "verify_phases": False,
                    "verify_traj": bool(args.verify_traj),
                    "hold_sec": float(args.hold_sec),
                    "sweep_freq": float(args.sweep_freq),
                    "trajectory_amp": [float(traj_amp_1), float(traj_amp_2)],
                    "sample_count": int(len(rec_arr)),
                    "mae_desired": [float(mae_des[0]), float(mae_des[1])],
                    "rmse_desired": [float(rmse_des[0]), float(rmse_des[1])],
                    "last_recovered": [float(rec_arr[-1, 0]), float(rec_arr[-1, 1])],
                }
            )
            if joint_mae is not None:
                verify_result["mae_joint_state"] = joint_mae

        _upsert_verify_file(
            verify_path=verify_path,
            pair_name=pair.name,
            result=verify_result,
        )
        print(f"[verify 저장] {verify_path}")

    else:
        f = args.sweep_freq
        dur = args.phase_duration
        amp_1 = float(args.axis1_amp) if args.axis1_amp is not None else float(pair.default_amp[0])
        amp_2 = float(args.axis2_amp) if args.axis2_amp is not None else float(pair.default_amp[1])

        phases = [
            (amp_1, 0.0, f, 0.0, "Phase1: axis1 only"),
            (0.0, amp_2, 0.0, f, "Phase2: axis2 only"),
            (amp_1, amp_2, f, f * 0.7, "Phase3: axis1+axis2"),
        ]

        all_data = []
        all_labels = []

        for phase_amp_1, phase_amp_2, phase_freq_1, phase_freq_2, name in phases:
            if stop_flag_fn():
                break
            print(f"\n{'='*50}")
            print(f"[sweep] {name}  ({dur}s)")
            print(f"{'='*50}")
            time.sleep(1.0)

            data = run_phase(
                ctrl=ctrl,
                group_setter=group_setter,
                group_base=group_base,
                group_pos_1=group_pos_1,
                group_pos_2=group_pos_2,
                center_1=center_1,
                center_2=center_2,
                amp_1=phase_amp_1,
                amp_2=phase_amp_2,
                freq_1=phase_freq_1,
                freq_2=phase_freq_2,
                read_pr_idx_1=pr_idx_1,
                read_pr_idx_2=pr_idx_2,
                read_ab_idx_1=ab_idx_1,
                read_ab_idx_2=ab_idx_2,
                cmd_limits=cmd_limits,
                duration=dur,
                dt=dt,
                stop_flag_fn=stop_flag_fn,
                use_measured_pr=True,
            )
            all_data.append(data)
            all_labels.append(name)
            print(f"  → {len(data)} samples 수집")

        valid_data = [d for d in all_data if len(d) > 0]
        if not valid_data:
            print("[error] 수집된 데이터가 없습니다.")
            return 1

        combined = np.vstack(valid_data)
        print(f"\n[lstsq] pair={pair.name}, 전체 {len(combined)} samples 로 M 추정 중...")

        M_est, ab_offset, _ = estimate_transform(combined)

        M_theory = np.array([[1.0, 1.0], [1.0, -1.0]])

        print("\n" + "=" * 50)
        print(f"  pair: {pair.name}")
        print("  affine model: ab = M @ pr + b")
        print("  추정 변환행렬 M_est:")
        print(f"    [[{M_est[0,0]:+.4f}, {M_est[0,1]:+.4f}],")
        print(f"     [{M_est[1,0]:+.4f}, {M_est[1,1]:+.4f}]]")
        print(f"  추정 offset b (pr=0 -> ms): [{ab_offset[0]:+.4f}, {ab_offset[1]:+.4f}]")
        print("\n  이론값 M_theory:")
        print(f"    [[{M_theory[0,0]:+.4f}, {M_theory[0,1]:+.4f}],")
        print(f"     [{M_theory[1,0]:+.4f}, {M_theory[1,1]:+.4f}]]")
        print("\n  오차 (M_est - M_theory):")
        err = M_est - M_theory
        print(f"    [[{err[0,0]:+.4f}, {err[0,1]:+.4f}],")
        print(f"     [{err[1,0]:+.4f}, {err[1,1]:+.4f}]]")
        print("=" * 50)

        M_inv = np.linalg.inv(M_est)
        print(f"\n  M_est 역행렬 (ab_1,ab_2 → pr_1,pr_2):")
        print(f"    [[{M_inv[0,0]:+.4f}, {M_inv[0,1]:+.4f}],")
        print(f"     [{M_inv[1,0]:+.4f}, {M_inv[1,1]:+.4f}]]")
        print("  (이론 역행렬: [[0.5, 0.5], [0.5, -0.5]])")
        print(
            f"  inverse model: pr = M^-1 @ (ab - b), "
            f"b=[{ab_offset[0]:+.4f}, {ab_offset[1]:+.4f}]"
        )

        plot_path = save_dir / f"{pair.name}_transform_result.png"
        plot_results(all_data, all_labels, M_est, ab_offset, pair.name, plot_path)
        _upsert_calib_file(
            calib_path=calib_path,
            pair_name=pair.name,
            M_est=M_est,
            pr_center=(0.0, 0.0),
            pr_limits=cmd_limits,
            ab_center=(float(ab_offset[0]), float(ab_offset[1])),
            sample_count=len(combined),
        )

        print("\n[다음 적용] controller/igris_controller.py")
        print(f"  ctrl.set_pr2ab_transform('{pair.name}', M=np.array({M_est.tolist()}),")
        print("                          pr_center=[0.0, 0.0],")
        print(f"                          ab_center={[float(ab_offset[0]), float(ab_offset[1])]})")
        print(f"[캘리브레이션 저장] {calib_path}")

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--hz", type=float, default=300.0)
    parser.add_argument("--state-timeout", type=float, default=5.0)
    parser.add_argument("--move-duration", type=float, default=5.0)
    parser.add_argument(
        "--task",
        choices=["calib", "verify"],
        default="calib",
        help="calib: 변환행렬 추정, verify: 추정행렬 검증",
    )
    parser.add_argument("--mode", choices=["pjs", "ms"], default="pjs")
    parser.add_argument("--pair", choices=sorted(PAIR_SPECS.keys()) + ["all"], default="r_wrist_rp")
    parser.add_argument("--dt", type=float, default=0.033, help="샘플 주기 (s)")
    parser.add_argument("--phase-duration", type=float, default=10.0, help="각 Phase 지속시간 (s)")
    parser.add_argument("--axis1-amp", type=float, default=None, help="axis1 sweep 진폭 (rad)")
    parser.add_argument("--axis2-amp", type=float, default=None, help="axis2 sweep 진폭 (rad)")
    parser.add_argument("--axis1-des", type=float, default=None, help="verify: axis1 목표 각도 중심값 (rad)")
    parser.add_argument("--axis2-des", type=float, default=None, help="verify: axis2 목표 각도 중심값 (rad)")
    parser.add_argument("--hold-sec", type=float, default=3.0, help="verify: 목표 자세 유지 시간 (s)")
    parser.add_argument("--verify-traj", action="store_true", help="verify: 단일점 대신 sin 연속 궤적 검증")
    parser.add_argument("--verify-phases", action="store_true", help="verify: phase1/2/3 자동 검증")
    parser.add_argument("--verify-log-interval", type=float, default=0.2, help="verify: 로그 출력 주기 (s)")
    parser.add_argument("--sweep-freq", type=float, default=0.15, help="sweep sin 주파수 (Hz)")
    parser.add_argument(
        "--urdf-path",
        type=str,
        default=str(REPO_ROOT / "igris_teleop" / "robot_control" / "asset" / "urdf" / "igris_c_v2_pelvis.urdf"),
        help="Joint limit 조회용 URDF 경로",
    )
    parser.add_argument("--save-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="결과 plot 저장 폴더")
    parser.add_argument(
        "--calib-path",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "pr2ab_calibration.yaml"),
        help="PR->AB 캘리브레이션 결과 YAML 경로",
    )
    parser.add_argument(
        "--verify-path",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "pr2ab_verify.yaml"),
        help="verify 결과 YAML 경로",
    )
    args = parser.parse_args()

    if args.task == "verify" and args.mode != "ms":
        print("[info] verify task는 --mode ms로 강제됩니다.")
        args.mode = "ms"
    if args.task == "verify" and (not args.verify_phases) and (not args.verify_traj):
        # Keep CLI simple: adding only '--task verify' runs familiar phase1/2/3 validation.
        args.verify_phases = True
    if args.task == "calib" and args.mode != "pjs":
        print("[info] calib task는 PR(joint) 실측을 위해 --mode pjs로 강제됩니다.")
        args.mode = "pjs"

    pair_names = list(PAIR_SPECS.keys()) if args.pair == "all" else [args.pair]
    kinematic_mode = igc_sdk.KinematicMode.PJS if args.mode == "pjs" else igc_sdk.KinematicMode.MS
    use_motor_state = args.mode == "ms"
    stop_requested = False
    save_dir = pathlib.Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    calib_path = pathlib.Path(args.calib_path).expanduser()
    verify_path = pathlib.Path(args.verify_path).expanduser()
    urdf_path = pathlib.Path(args.urdf_path).expanduser()
    joint_limits = _load_joint_limits_from_urdf(urdf_path)
    if args.pair == "all":
        print(f"[pair] all mode: {', '.join(pair_names)}")
    print(f"[limit] URDF source: {urdf_path}")

    def _handle_signal(signum, _frame):
        nonlocal stop_requested
        print(f"\n[signal] {signum} → 안전 종료 중...")
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            pass

    ctrl = BaseController(
        domain_id=args.domain,
        control_hz=args.hz,
        kinematic_mode=kinematic_mode,
        use_motor_state=use_motor_state,
        auto_service_init=True,
    )

    try:
        if not ctrl.wait_for_state(timeout=args.state_timeout):
            print(f"[error] No LowState within {args.state_timeout}s")
            return 1

        ctrl.move_to_pose("default_pos", duration=args.move_duration)
        ctrl.default_pos_state()
        time.sleep(0.5)

        ready = wait_for_key("s", stop_flag_fn=lambda: stop_requested)
        if not ready or stop_requested:
            print("[중단] sweep 시작 전 종료")
            return 0
        run_rc = 0
        for i, pair_name in enumerate(pair_names, start=1):
            if stop_requested:
                break
            print(f"\n{'#'*60}")
            print(f"[run] pair {i}/{len(pair_names)}: {pair_name}")
            print(f"{'#'*60}")
            rc = _execute_pair_task(
                ctrl=ctrl,
                args=args,
                pair_name=pair_name,
                joint_limits=joint_limits,
                save_dir=save_dir,
                calib_path=calib_path,
                verify_path=verify_path,
                stop_flag_fn=lambda: stop_requested,
            )
            if rc != 0:
                run_rc = rc
                print(f"[error] pair={pair_name} failed (code={rc})")
                break

            if i < len(pair_names) and not stop_requested:
                print("[info] 다음 pair 실행 전 default_pos로 복귀합니다.")
                try:
                    ctrl.move_to_pose("default_pos", duration=min(args.move_duration, 2.0))
                    ctrl.default_pos_state()
                    time.sleep(0.3)
                except Exception as e:
                    print(f"[경고] pair 간 복귀 중 오류: {e}")

        if args.pair == "all" and run_rc == 0 and not stop_requested:
            print("\n[완료] 모든 pair 작업이 끝났습니다.")
        return run_rc

    finally:
        print("\n[종료] default_pos 복귀 중...")
        try:
            ctrl.move_to_pose("default_pos", duration=2.0)
            ctrl.default_pos_state()
            time.sleep(0.3)
        except Exception as e:
            print(f"[경고] 복귀 중 오류: {e}")
        print("[종료] controller 정지")
        ctrl.stop()


if __name__ == "__main__":
    sys.exit(main())
