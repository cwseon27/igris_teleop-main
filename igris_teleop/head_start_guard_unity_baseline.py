from __future__ import annotations

from typing import Mapping

import numpy as np


HEAD_POS_THRESH_M = 0.05
HEAD_ROT_THRESH_DEG = 15.0
HEAD_YAW_HINT_DEADBAND_DEG = 1.0

HEAD_REASON_INACTIVE = 0
HEAD_REASON_WAITING_VR_STREAM = 1
HEAD_REASON_WAITING_HOME_REFERENCE = 2
HEAD_REASON_WAITING_ROBOT_HEAD_REFERENCE = 3
HEAD_REASON_OK = 4
HEAD_REASON_POS_EXCEEDED = 5
HEAD_REASON_ROT_EXCEEDED = 6
HEAD_REASON_POS_AND_ROT_EXCEEDED = 7

HEAD_REASON_MESSAGES: dict[int, str] = {
    HEAD_REASON_INACTIVE: "VR head start guard inactive",
    HEAD_REASON_WAITING_VR_STREAM: "VR head stream not ready",
    HEAD_REASON_WAITING_HOME_REFERENCE: "HOME head reference not captured",
    HEAD_REASON_WAITING_ROBOT_HEAD_REFERENCE: "Robot HOME head pose not available",
    HEAD_REASON_OK: "VR head aligned",
    HEAD_REASON_POS_EXCEEDED: "Head position mismatch exceeds limit",
    HEAD_REASON_ROT_EXCEEDED: "Head rotation mismatch exceeds limit",
    HEAD_REASON_POS_AND_ROT_EXCEEDED: "Head position and rotation mismatch exceed limits",
}


def _scalar(value: object, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(np.asarray(value, dtype=np.float64).reshape(()).item())
    except Exception:
        return float(default)


def inactive_head_guard_status(*, seq: float = 0.0) -> dict[str, float]:
    return {
        "seq": float(seq),
        "guard_active": 0.0,
        "head_guard_ready": 0.0,
        "head_guard_ok": 0.0,
        "head_pos_err_m": 0.0,
        "head_rot_err_deg": 0.0,
        "head_yaw_correction_deg": 0.0,
        "head_pos_thresh_m": float(HEAD_POS_THRESH_M),
        "head_rot_thresh_deg": float(HEAD_ROT_THRESH_DEG),
        "head_reason_code": float(HEAD_REASON_INACTIVE),
    }


def is_valid_pose_matrix(mat: object) -> bool:
    try:
        arr = np.asarray(mat, dtype=np.float64)
    except Exception:
        return False
    if arr.shape != (4, 4):
        return False
    if not np.all(np.isfinite(arr)):
        return False
    if not np.any(arr):
        return False
    if not np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), atol=1e-6):
        return False
    return True


def rotation_error_deg(target_rot: np.ndarray, reference_rot: np.ndarray) -> float:
    target = np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
    reference = np.asarray(reference_rot, dtype=np.float64).reshape(3, 3)
    delta = reference.T @ target
    cos_theta = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def signed_yaw_error_deg(target_rot: np.ndarray, reference_rot: np.ndarray) -> float:
    target = np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
    reference = np.asarray(reference_rot, dtype=np.float64).reshape(3, 3)
    delta = reference.T @ target
    return float(np.degrees(np.arctan2(delta[1, 0], delta[0, 0])))


def yaw_correction_deg(target_rot: np.ndarray, reference_rot: np.ndarray) -> float:
    return float(-signed_yaw_error_deg(target_rot, reference_rot))


def build_candidate_head_target(
    vr_head_now: np.ndarray,
    vr_head_home: np.ndarray,
    robot_head_home: np.ndarray,
) -> np.ndarray:
    now = np.asarray(vr_head_now, dtype=np.float64).reshape(4, 4)
    home = np.asarray(vr_head_home, dtype=np.float64).reshape(4, 4)
    robot_home = np.asarray(robot_head_home, dtype=np.float64).reshape(4, 4)
    target = robot_home.copy()
    target[:3, :3] = now[:3, :3]
    target[:3, 3] = robot_home[:3, 3] + (now[:3, 3] - home[:3, 3])
    return target


def evaluate_head_start_guard(
    vr_head_now: object,
    vr_head_home: object,
    robot_head_home: object,
    *,
    seq: float = 0.0,
    pos_thresh_m: float = HEAD_POS_THRESH_M,
    rot_thresh_deg: float = HEAD_ROT_THRESH_DEG,
) -> dict[str, float]:
    status = inactive_head_guard_status(seq=seq)
    status["guard_active"] = 1.0
    status["head_pos_thresh_m"] = float(pos_thresh_m)
    status["head_rot_thresh_deg"] = float(rot_thresh_deg)

    if not is_valid_pose_matrix(vr_head_now):
        status["head_reason_code"] = float(HEAD_REASON_WAITING_VR_STREAM)
        return status
    if not is_valid_pose_matrix(vr_head_home):
        status["head_reason_code"] = float(HEAD_REASON_WAITING_HOME_REFERENCE)
        return status
    if not is_valid_pose_matrix(robot_head_home):
        status["head_reason_code"] = float(HEAD_REASON_WAITING_ROBOT_HEAD_REFERENCE)
        return status

    candidate = build_candidate_head_target(
        np.asarray(vr_head_now, dtype=np.float64),
        np.asarray(vr_head_home, dtype=np.float64),
        np.asarray(robot_head_home, dtype=np.float64),
    )
    robot_home = np.asarray(robot_head_home, dtype=np.float64).reshape(4, 4)

    pos_err_m = float(np.linalg.norm(candidate[:3, 3] - robot_home[:3, 3]))
    rot_err_deg = float(rotation_error_deg(candidate[:3, :3], robot_home[:3, :3]))
    yaw_correction = float(yaw_correction_deg(candidate[:3, :3], robot_home[:3, :3]))

    status["head_guard_ready"] = 1.0
    status["head_pos_err_m"] = pos_err_m
    status["head_rot_err_deg"] = rot_err_deg
    status["head_yaw_correction_deg"] = yaw_correction

    pos_bad = pos_err_m > float(pos_thresh_m)
    rot_bad = rot_err_deg > float(rot_thresh_deg)
    if pos_bad and rot_bad:
        reason = HEAD_REASON_POS_AND_ROT_EXCEEDED
    elif pos_bad:
        reason = HEAD_REASON_POS_EXCEEDED
    elif rot_bad:
        reason = HEAD_REASON_ROT_EXCEEDED
    else:
        reason = HEAD_REASON_OK

    status["head_reason_code"] = float(reason)
    status["head_guard_ok"] = 1.0 if reason == HEAD_REASON_OK else 0.0
    return status


def head_guard_reason_message(reason_code: object) -> str:
    reason = int(round(_scalar(reason_code, default=HEAD_REASON_INACTIVE)))
    return HEAD_REASON_MESSAGES.get(reason, HEAD_REASON_MESSAGES[HEAD_REASON_INACTIVE])


def head_guard_is_blocking(status: Mapping[str, object] | None) -> bool:
    if not status:
        return False
    return _scalar(status.get("guard_active")) == 1.0 and _scalar(status.get("head_guard_ok")) != 1.0


def format_head_guard_message(status: Mapping[str, object] | None) -> str:
    if not status:
        return "VR head start guard status unavailable"

    reason = head_guard_reason_message(status.get("head_reason_code"))
    pos_err = _scalar(status.get("head_pos_err_m"))
    rot_err = _scalar(status.get("head_rot_err_deg"))
    yaw_correction = _scalar(status.get("head_yaw_correction_deg"))
    pos_thresh = _scalar(status.get("head_pos_thresh_m"), default=HEAD_POS_THRESH_M)
    rot_thresh = _scalar(status.get("head_rot_thresh_deg"), default=HEAD_ROT_THRESH_DEG)
    rotation_hint = ""
    yaw_mag = abs(yaw_correction)
    if rot_err >= HEAD_YAW_HINT_DEADBAND_DEG:
        if yaw_mag >= HEAD_YAW_HINT_DEADBAND_DEG:
            direction = "left" if yaw_correction > 0.0 else "right"
            rotation_hint = f", turn {direction} {yaw_mag:.1f} deg"
        else:
            rotation_hint = ", adjust head tilt/roll"

    return (
        f"{reason} "
        f"(pos {pos_err:.3f}/{pos_thresh:.3f} m, "
        f"rot {rot_err:.1f}/{rot_thresh:.1f} deg{rotation_hint})"
    )


def head_guard_reason_label(reason_code: object) -> str:
    reason = int(round(_scalar(reason_code, default=HEAD_REASON_INACTIVE)))
    labels = {
        HEAD_REASON_INACTIVE: "inactive",
        HEAD_REASON_WAITING_VR_STREAM: "waiting_vr_stream",
        HEAD_REASON_WAITING_HOME_REFERENCE: "waiting_home_reference",
        HEAD_REASON_WAITING_ROBOT_HEAD_REFERENCE: "waiting_robot_head_reference",
        HEAD_REASON_OK: "ok",
        HEAD_REASON_POS_EXCEEDED: "pos_exceeded",
        HEAD_REASON_ROT_EXCEEDED: "rot_exceeded",
        HEAD_REASON_POS_AND_ROT_EXCEEDED: "pos_and_rot_exceeded",
    }
    return labels.get(reason, "inactive")
