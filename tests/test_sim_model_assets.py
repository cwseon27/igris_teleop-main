from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from igris_teleop.core.project_paths import SIM_ARM_GAIN_PATH, SIM_NECK_GAIN_PATH
from igris_teleop.sim.stereo_camera import (
    STEREO_CAMERA_BASELINE_M,
    STEREO_CAMERA_HEIGHT,
    STEREO_CAMERA_VERTICAL_FOV_DEG,
    STEREO_CAMERA_WIDTH,
    pinhole_intrinsics,
    stereo_projection,
    validate_stereo_baseline,
)


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "igris_teleop"
    / "sim"
    / "robot"
    / "mujoco"
    / "igris_c_v2_with_hand.xml"
)
ACTIVE_HAND_JOINTS = {
    f"Joint_{finger}_{segment}_{side}"
    for side in ("Left", "Right")
    for finger, segment in (
        ("Thumb", "Proximal"),
        ("Thumb", "Middle"),
        ("Index", "Middle"),
        ("Middle", "Middle"),
        ("Ring", "Middle"),
        ("Little", "Middle"),
    )
}
PROXIMAL_ARM_JOINTS = {
    f"Joint_{joint}_{side}"
    for side in ("Left", "Right")
    for joint in (
        "Shoulder_Pitch",
        "Shoulder_Roll",
        "Shoulder_Yaw",
        "Elbow_Pitch",
    )
}
WRIST_JOINTS = {
    f"Joint_Wrist_{axis}_{side}"
    for side in ("Left", "Right")
    for axis in ("Yaw", "Roll", "Pitch")
}


def test_sim_model_contains_actuated_hands_and_distal_couplings() -> None:
    root = ET.parse(MODEL_PATH).getroot()
    joint_names = {element.get("name") for element in root.findall(".//joint")}
    motor_names = {element.get("name") for element in root.findall("./actuator/motor")}
    equalities = root.findall("./equality/joint")
    hand_collisions = [
        element
        for element in root.findall(".//geom")
        if str(element.get("name", "")).endswith("_collision")
        and any(
            token in str(element.get("name", ""))
            for token in ("Palm", "Thumb", "Index", "Middle", "Ring", "Little")
        )
    ]

    assert ACTIVE_HAND_JOINTS <= joint_names
    assert ACTIVE_HAND_JOINTS <= motor_names
    assert len([element for element in equalities if str(element.get("name", "")).startswith("Eq_")]) == 10
    assert len(hand_collisions) == 26
    assert {(element.get("contype"), element.get("conaffinity")) for element in hand_collisions} == {
        ("2", "12")
    }
    assert {element.get("margin") for element in hand_collisions} == {"0"}
    assert {element.get("gap") for element in hand_collisions} == {"0"}


def test_sim_arm_motors_have_bimanual_holding_headroom() -> None:
    root = ET.parse(MODEL_PATH).getroot()
    motors = {
        str(element.get("joint")): element
        for element in root.findall("./actuator/motor")
    }
    defaults = {
        str(element.get("class")): element
        for element in root.findall("./default/default")
    }

    assert {motors[name].get("class") for name in PROXIMAL_ARM_JOINTS} == {
        "sim_arm_motor_90"
    }
    assert {motors[name].get("class") for name in WRIST_JOINTS} == {
        "sim_wrist_motor_20"
    }
    assert {motors[name].get("class") for name in ACTIVE_HAND_JOINTS} == {
        "sim_hand_motor_15"
    }
    assert defaults["sim_arm_motor_90"].find("motor").get("ctrlrange") == "-90.0 90.0"
    assert defaults["sim_wrist_motor_20"].find("motor").get("ctrlrange") == "-20.0 20.0"
    assert defaults["sim_hand_motor_15"].find("motor").get("ctrlrange") == "-15.0 15.0"


def test_sim_contact_solver_uses_benchmarked_stable_profile() -> None:
    root = ET.parse(MODEL_PATH).getroot()
    option = root.find("./option")

    assert option is not None
    assert option.get("integrator") == "Euler"
    assert option.get("solver") == "Newton"
    assert option.get("cone") == "elliptic"
    assert option.get("noslip_iterations") == "100"


def test_sim_arm_gain_override_resists_bimanual_contact_loads() -> None:
    config = yaml.safe_load(SIM_ARM_GAIN_PATH.read_text(encoding="utf-8"))

    assert config["default"]["kp"] == [
        500.0,
        400.0,
        300.0,
        600.0,
        360.0,
        360.0,
        360.0,
    ] * 2
    assert config["default"]["kd"] == [
        12.0,
        12.0,
        8.0,
        12.0,
        6.0,
        6.0,
        6.0,
    ] * 2


def test_sim_neck_gain_override_can_hold_head_against_gravity() -> None:
    config = yaml.safe_load(SIM_NECK_GAIN_PATH.read_text(encoding="utf-8"))

    assert config["default"]["kp"] == [30.0, 60.0]
    assert config["default"]["kd"] == [3.0, 6.0]
    assert config["walking"] == config["default"]


def test_sim_stereo_mount_has_zed_like_baseline_and_forward_optical_axes() -> None:
    root = ET.parse(MODEL_PATH).getroot()
    cameras = {element.get("name"): element for element in root.findall(".//camera")}
    left = cameras["head_stereo_left"]
    right = cameras["head_stereo_right"]
    left_pos = np.fromstring(str(left.get("pos")), sep=" ")
    right_pos = np.fromstring(str(right.get("pos")), sep=" ")

    assert np.isclose(np.linalg.norm(left_pos - right_pos), STEREO_CAMERA_BASELINE_M)
    assert left.get("xyaxes") == "0 -1 0 0 0 1"
    assert right.get("xyaxes") == "0 -1 0 0 0 1"
    assert float(left.get("fovy", "0")) == STEREO_CAMERA_VERTICAL_FOV_DEG
    assert float(right.get("fovy", "0")) == STEREO_CAMERA_VERTICAL_FOV_DEG


def test_sim_camera_intrinsics_match_mujoco_vertical_fov() -> None:
    fx, fy, cx, cy = pinhole_intrinsics()
    recovered_fov = math.degrees(2.0 * math.atan(0.5 * STEREO_CAMERA_HEIGHT / fy))

    assert fx == fy
    assert cx == 0.5 * (STEREO_CAMERA_WIDTH - 1)
    assert cy == 0.5 * (STEREO_CAMERA_HEIGHT - 1)
    assert np.isclose(recovered_fov, STEREO_CAMERA_VERTICAL_FOV_DEG)


def test_sim_stereo_projection_changes_translation_only() -> None:
    baseline_m = validate_stereo_baseline(0.064)
    left = np.asarray(stereo_projection(baseline_m=baseline_m, right=False))
    right = np.asarray(stereo_projection(baseline_m=baseline_m, right=True))
    fx, _fy, _cx, _cy = pinhole_intrinsics()

    assert left[3] == 0.0
    assert np.isclose(right[3], -fx * baseline_m)
    assert np.array_equal(np.delete(left, 3), np.delete(right, 3))
