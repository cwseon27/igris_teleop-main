from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np


def _parse_vec3(text: str | None, *, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if text is None or str(text).strip() == "":
        return np.asarray(default, dtype=np.float64)
    values = [float(token) for token in str(text).replace(",", " ").split()]
    if len(values) != 3:
        raise ValueError(f"expected 3 values, got {values!r}")
    return np.asarray(values, dtype=np.float64)


def _rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3)
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12 or abs(angle) <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    if rotation is not None:
        out[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if translation is not None:
        out[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return out


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray

    @property
    def origin_transform(self) -> np.ndarray:
        return _transform(_rpy_matrix(self.origin_rpy), self.origin_xyz)

    def motion_transform(self, value: float) -> np.ndarray:
        if self.joint_type in {"revolute", "continuous"}:
            return _transform(_axis_angle_matrix(self.axis, float(value)), np.zeros(3, dtype=np.float64))
        if self.joint_type == "prismatic":
            return _transform(np.eye(3, dtype=np.float64), np.asarray(self.axis, dtype=np.float64) * float(value))
        return np.eye(4, dtype=np.float64)


class IGRISUpperBodyFK:
    """Dependency-light upper-body FK for masterarm ee_pose publishing."""

    _WAIST_CHAIN = (
        "0_Joint_Waist_Pitch",
        "1_Joint_Waist_Roll",
        "2_Joint_Waist_Yaw",
    )
    _LEFT_CHAIN = _WAIST_CHAIN + (
        "15_Joint_Shoulder_Pitch_Left",
        "16_Joint_Shoulder_Roll_Left",
        "17_Joint_Shoulder_Yaw_Left",
        "18_Joint_Elbow_Pitch_Left",
        "19_Joint_Wrist_Yaw_Left",
        "20_Joint_Wrist_Roll_Left",
        "21_Joint_Wrist_Pitch_Left",
    )
    _RIGHT_CHAIN = _WAIST_CHAIN + (
        "22_Joint_Shoulder_Pitch_Right",
        "23_Joint_Shoulder_Roll_Right",
        "24_Joint_Shoulder_Yaw_Right",
        "25_Joint_Elbow_Pitch_Right",
        "26_Joint_Wrist_Yaw_Right",
        "27_Joint_Wrist_Roll_Right",
        "28_Joint_Wrist_Pitch_Right",
    )
    _HEAD_CHAIN = _WAIST_CHAIN + (
        "29_Joint_Neck_Yaw",
        "30_Joint_Neck_Pitch",
    )
    _ALL_REQUIRED = frozenset(_LEFT_CHAIN + _RIGHT_CHAIN + _HEAD_CHAIN)

    def __init__(self, urdf_path: str | Path | None = None) -> None:
        if urdf_path is None:
            base_dir = Path(__file__).resolve().parents[2]
            urdf_path = base_dir / "asset" / "urdf" / "igris_c_v2_pelvis.urdf"
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self._joint_specs = self._load_joint_specs(self.urdf_path)
        self._left_offset = _transform(np.eye(3, dtype=np.float64), np.asarray([0.0, 0.0, -0.05], dtype=np.float64))
        self._right_offset = _transform(np.eye(3, dtype=np.float64), np.asarray([0.0, 0.0, -0.05], dtype=np.float64))
        self._head_offset = _transform(np.eye(3, dtype=np.float64), np.asarray([0.05, 0.0, 0.15], dtype=np.float64))

    @classmethod
    def _load_joint_specs(cls, urdf_path: Path) -> dict[str, JointSpec]:
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        root = ET.parse(urdf_path).getroot()
        specs: dict[str, JointSpec] = {}
        for joint_el in root.findall("joint"):
            name = joint_el.get("name")
            if name not in cls._ALL_REQUIRED:
                continue
            origin_el = joint_el.find("origin")
            axis_el = joint_el.find("axis")
            spec = JointSpec(
                name=str(name),
                joint_type=str(joint_el.get("type", "fixed")),
                origin_xyz=_parse_vec3(origin_el.get("xyz") if origin_el is not None else None),
                origin_rpy=_parse_vec3(origin_el.get("rpy") if origin_el is not None else None),
                axis=_parse_vec3(axis_el.get("xyz") if axis_el is not None else None, default=(0.0, 0.0, 1.0)),
            )
            specs[spec.name] = spec

        missing = sorted(cls._ALL_REQUIRED.difference(specs))
        if missing:
            raise ValueError(f"missing required upper-body joints in URDF: {missing}")
        return specs

    @staticmethod
    def _joint_value_map(q: np.ndarray) -> dict[str, float]:
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.size == 31:
            waist = arr[0:3]
            left_arm = arr[15:22]
            right_arm = arr[22:29]
            neck = arr[29:31]
        elif arr.size == 19:
            waist = arr[0:3]
            left_arm = arr[3:10]
            right_arm = arr[10:17]
            neck = arr[17:19]
        else:
            raise ValueError(f"expected upper-body q with 19 or 31 values, got {arr.size}")

        if not np.all(np.isfinite(arr)):
            raise ValueError("q contains non-finite values")

        return {
            "0_Joint_Waist_Pitch": float(waist[2]),
            "1_Joint_Waist_Roll": float(waist[1]),
            "2_Joint_Waist_Yaw": float(waist[0]),
            "15_Joint_Shoulder_Pitch_Left": float(left_arm[0]),
            "16_Joint_Shoulder_Roll_Left": float(left_arm[1]),
            "17_Joint_Shoulder_Yaw_Left": float(left_arm[2]),
            "18_Joint_Elbow_Pitch_Left": float(left_arm[3]),
            "19_Joint_Wrist_Yaw_Left": float(left_arm[4]),
            "20_Joint_Wrist_Roll_Left": float(left_arm[5]),
            "21_Joint_Wrist_Pitch_Left": float(left_arm[6]),
            "22_Joint_Shoulder_Pitch_Right": float(right_arm[0]),
            "23_Joint_Shoulder_Roll_Right": float(right_arm[1]),
            "24_Joint_Shoulder_Yaw_Right": float(right_arm[2]),
            "25_Joint_Elbow_Pitch_Right": float(right_arm[3]),
            "26_Joint_Wrist_Yaw_Right": float(right_arm[4]),
            "27_Joint_Wrist_Roll_Right": float(right_arm[5]),
            "28_Joint_Wrist_Pitch_Right": float(right_arm[6]),
            "29_Joint_Neck_Yaw": float(neck[0]),
            "30_Joint_Neck_Pitch": float(neck[1]),
        }

    def _chain_transform(self, chain: tuple[str, ...], joint_values: dict[str, float]) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        for joint_name in chain:
            spec = self._joint_specs[joint_name]
            pose = pose @ spec.origin_transform @ spec.motion_transform(joint_values[joint_name])
        return pose

    def get_ee_pose_mats(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint_values = self._joint_value_map(q)
        left = self._chain_transform(self._LEFT_CHAIN, joint_values) @ self._left_offset
        right = self._chain_transform(self._RIGHT_CHAIN, joint_values) @ self._right_offset
        head = self._chain_transform(self._HEAD_CHAIN, joint_values) @ self._head_offset
        return (
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
            np.asarray(head, dtype=np.float64),
        )
