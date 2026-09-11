from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from igris_teleop.sim.hand_physics import (
    HAND_ACTUATOR_TORQUE_LIMIT_NM,
    HAND_SERVO_KD,
    HAND_SERVO_KP,
)
from igris_teleop.sim.stereo_camera import (
    STEREO_CAMERA_BASELINE_M,
    STEREO_CAMERA_HEIGHT,
    STEREO_CAMERA_WIDTH,
    STEREO_LEFT_CAMERA_NAME,
    STEREO_RIGHT_CAMERA_NAME,
)


MODEL_PATH = Path(__file__).resolve().parent / "robot" / "mujoco" / "igris_c_v2_with_hand.xml"
HAND_JOINT_NAMES = (
    "Joint_Thumb_Middle_Right",
    "Joint_Index_Middle_Right",
    "Joint_Middle_Middle_Right",
    "Joint_Ring_Middle_Right",
    "Joint_Little_Middle_Right",
    "Joint_Thumb_Proximal_Right",
    "Joint_Thumb_Middle_Left",
    "Joint_Index_Middle_Left",
    "Joint_Middle_Middle_Left",
    "Joint_Ring_Middle_Left",
    "Joint_Little_Middle_Left",
    "Joint_Thumb_Proximal_Left",
)
def _id(model: mujoco.MjModel, obj_type, name: str) -> int:
    value = mujoco.mj_name2id(model, obj_type, name)
    if value < 0:
        raise AssertionError(f"missing MuJoCo object: {name}")
    return int(value)


def validate_stereo(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    left_id = _id(model, mujoco.mjtObj.mjOBJ_CAMERA, STEREO_LEFT_CAMERA_NAME)
    right_id = _id(model, mujoco.mjtObj.mjOBJ_CAMERA, STEREO_RIGHT_CAMERA_NAME)
    mujoco.mj_forward(model, data)

    baseline = float(np.linalg.norm(data.cam_xpos[left_id] - data.cam_xpos[right_id]))
    if not np.isclose(baseline, STEREO_CAMERA_BASELINE_M, atol=1e-6):
        raise AssertionError(f"stereo baseline={baseline:.6f}, expected={STEREO_CAMERA_BASELINE_M:.6f}")
    if model.vis.global_.offwidth < STEREO_CAMERA_WIDTH or model.vis.global_.offheight < STEREO_CAMERA_HEIGHT:
        raise AssertionError("MuJoCo offscreen framebuffer is smaller than the stereo output")

    for camera_id in (left_id, right_id):
        forward = -np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)[:, 2]
        if float(forward[0]) < 0.999:
            raise AssertionError(f"stereo optical axis does not face robot +X: {forward}")

    print(
        f"[validate] stereo OK: cameras=2 baseline={baseline:.3f} m "
        f"resolution={STEREO_CAMERA_WIDTH}x{STEREO_CAMERA_HEIGHT}"
    )


def validate_hand(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    qpos_adrs: list[int] = []
    dof_adrs: list[int] = []
    actuator_ids: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    for name in HAND_JOINT_NAMES:
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
        dof_adrs.append(int(model.jnt_dofadr[joint_id]))
        actuator_ids.append(actuator_id)
        lower.append(float(model.jnt_range[joint_id, 0]))
        upper.append(float(model.jnt_range[joint_id, 1]))

    qpos_idx = np.asarray(qpos_adrs, dtype=np.int32)
    dof_idx = np.asarray(dof_adrs, dtype=np.int32)
    actuator_idx = np.asarray(actuator_ids, dtype=np.int32)
    ctrl_limits = np.max(
        np.abs(model.actuator_ctrlrange[actuator_idx]),
        axis=1,
    )
    if float(np.min(ctrl_limits)) < HAND_ACTUATOR_TORQUE_LIMIT_NM:
        raise AssertionError(f"hand actuator torque limit too low: {ctrl_limits}")
    open_q = np.asarray(lower, dtype=np.float64)
    closed_q = np.asarray(upper, dtype=np.float64)
    reversed_idx = HAND_JOINT_NAMES.index("Joint_Thumb_Proximal_Right")
    open_q[reversed_idx], closed_q[reversed_idx] = closed_q[reversed_idx], open_q[reversed_idx]

    model.opt.gravity[:] = 0.0
    data.qpos[qpos_idx] = open_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(4500):
        ctrl = (
            HAND_SERVO_KP * (closed_q - data.qpos[qpos_idx])
            - HAND_SERVO_KD * data.qvel[dof_idx]
        )
        data.ctrl[:] = 0.0
        data.ctrl[actuator_idx] = np.clip(ctrl, -ctrl_limits, ctrl_limits)
        mujoco.mj_step(model, data)

    normalized = (data.qpos[qpos_idx] - open_q) / (closed_q - open_q)
    if float(np.min(normalized)) < 0.90:
        raise AssertionError(f"hand close target not reached: {normalized}")

    mimic_errors: list[float] = []
    for side in ("Left", "Right"):
        for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
            middle_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, f"Joint_{finger}_Middle_{side}")
            distal_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, f"Joint_{finger}_Distal_{side}")
            middle_q = float(data.qpos[int(model.jnt_qposadr[middle_id])])
            distal_q = float(data.qpos[int(model.jnt_qposadr[distal_id])])
            mimic_errors.append(abs(middle_q - distal_q))
    if max(mimic_errors) > 1e-3:
        raise AssertionError(f"distal mimic error too large: {max(mimic_errors):.6f} rad")

    print(
        f"[validate] hand OK: active=12 mimic=10 "
        f"torque_limit={float(np.min(ctrl_limits)):.1f} Nm "
        f"min_close={float(np.min(normalized)):.3f} max_mimic_error={max(mimic_errors):.6f} rad"
    )


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    validate_stereo(model, data)
    validate_hand(model, data)
    print(f"[validate] model OK: {MODEL_PATH}")


if __name__ == "__main__":
    main()
