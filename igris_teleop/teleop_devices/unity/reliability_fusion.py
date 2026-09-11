from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


DEFAULT_CONTROLLER_CONFIDENCE_FALLBACK = 0.0
DEFAULT_CONTROLLER_TRACKED_CONFIDENCE_FLOOR = 0.0
DEFAULT_CONTROLLER_UNTRACKED_POSE_CONFIDENCE_FLOOR = 0.0


def clamp01(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def select_tracking_confidence(
    *,
    pose_fresh: bool,
    tracked: bool | None,
    confidence_fresh: bool,
    confidence: float,
    fallback: float,
    tracked_floor: float,
    untracked_pose_floor: float,
) -> float:
    """Resolve a fresh controller pose's effective reliability confidence."""
    if not pose_fresh:
        return 0.0
    if not confidence_fresh:
        return clamp01(fallback)

    value = clamp01(confidence)
    if tracked is True:
        value = max(value, clamp01(tracked_floor))
    elif tracked is False:
        value = max(value, clamp01(untracked_pose_floor))
    return value


def valid_transform(value) -> np.ndarray | None:
    try:
        transform = np.asarray(value, dtype=np.float64).reshape(4, 4).copy()
    except Exception:
        return None
    if not np.all(np.isfinite(transform)):
        return None
    rotation = transform[:3, :3]
    try:
        u, _, vt = np.linalg.svd(rotation)
    except np.linalg.LinAlgError:
        return None
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    transform[:3, :3] = rotation
    transform[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return transform


def transform_from_xyz_xyzw(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != 7 or not np.all(np.isfinite(values)):
        raise ValueError("controller-to-chest transform must be [x,y,z,qx,qy,qz,qw]")
    x, y, z, qx, qy, qz, qw = values.tolist()
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError("controller-to-chest quaternion has zero norm")
    qx, qy, qz, qw = (quat / norm).tolist()
    rotation = np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return transform


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (r[2, 1] - r[1, 2]) / scale,
                (r[0, 2] - r[2, 0]) / scale,
                (r[1, 0] - r[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(r)))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) * 2.0
            quat = np.array(
                [0.25 * scale, (r[0, 1] + r[1, 0]) / scale, (r[0, 2] + r[2, 0]) / scale, (r[2, 1] - r[1, 2]) / scale]
            )
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + r[1, 1] - r[0, 0] - r[2, 2])) * 2.0
            quat = np.array(
                [(r[0, 1] + r[1, 0]) / scale, 0.25 * scale, (r[1, 2] + r[2, 1]) / scale, (r[0, 2] - r[2, 0]) / scale]
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + r[2, 2] - r[0, 0] - r[1, 1])) * 2.0
            quat = np.array(
                [(r[0, 2] + r[2, 0]) / scale, (r[1, 2] + r[2, 1]) / scale, 0.25 * scale, (r[1, 0] - r[0, 1]) / scale]
            )
    norm = float(np.linalg.norm(quat))
    return quat / max(norm, 1e-12)


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_slerp(left: np.ndarray, right: np.ndarray, right_weight: float) -> np.ndarray:
    q0 = np.asarray(left, dtype=np.float64).reshape(4)
    q1 = np.asarray(right, dtype=np.float64).reshape(4)
    q0 /= max(float(np.linalg.norm(q0)), 1e-12)
    q1 /= max(float(np.linalg.norm(q1)), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    amount = clamp01(right_weight)
    if dot > 0.9995:
        result = q0 + amount * (q1 - q0)
        return result / max(float(np.linalg.norm(result)), 1e-12)
    angle = math.acos(max(-1.0, min(1.0, dot)))
    sin_angle = math.sin(angle)
    return (
        math.sin((1.0 - amount) * angle) / sin_angle * q0
        + math.sin(amount * angle) / sin_angle * q1
    )


@dataclass(frozen=True)
class TorsoFusionResult:
    target: np.ndarray | None
    alpha: float
    raw_alpha: float
    consistency_gate: float
    left_confidence: float
    right_confidence: float
    ready: bool
    position_disagreement_m: float
    rotation_disagreement_rad: float


class ReliabilityAwareTorsoFusion:
    def __init__(
        self,
        *,
        left_to_chest: np.ndarray,
        right_to_chest: np.ndarray,
        confidence_power: float = 1.0,
        minimum_confidence_sum: float = 0.05,
        position_sigma_m: float = 0.15,
        rotation_sigma_rad: float = math.radians(25.0),
        alpha_rate_up_per_sec: float = 6.0,
        alpha_rate_down_per_sec: float = 10.0,
        nominal_rate_hz: float = 60.0,
    ) -> None:
        self.left_to_chest = valid_transform(left_to_chest)
        self.right_to_chest = valid_transform(right_to_chest)
        if self.left_to_chest is None or self.right_to_chest is None:
            raise ValueError("controller-to-chest transforms must be valid SE(3) matrices")
        self.confidence_power = max(0.01, float(confidence_power))
        self.minimum_confidence_sum = max(0.0, float(minimum_confidence_sum))
        self.position_sigma_m = max(1e-6, float(position_sigma_m))
        self.rotation_sigma_rad = max(1e-6, float(rotation_sigma_rad))
        self.alpha_rate_up_per_sec = max(0.0, float(alpha_rate_up_per_sec))
        self.alpha_rate_down_per_sec = max(0.0, float(alpha_rate_down_per_sec))
        self.nominal_dt = 1.0 / max(1.0, float(nominal_rate_hz))
        self.alpha = 0.0
        self.last_time: float | None = None
        self.last_target: np.ndarray | None = None

    def update(
        self,
        *,
        left_controller: np.ndarray | None,
        right_controller: np.ndarray | None,
        left_confidence: float,
        right_confidence: float,
        now: float,
    ) -> TorsoFusionResult:
        left = valid_transform(left_controller)
        right = valid_transform(right_controller)
        left_candidate = None if left is None else left @ self.left_to_chest
        right_candidate = None if right is None else right @ self.right_to_chest
        c_left = clamp01(left_confidence) if left_candidate is not None else 0.0
        c_right = clamp01(right_confidence) if right_candidate is not None else 0.0

        pos_error = 0.0
        rot_error = 0.0
        gate = 1.0
        if left_candidate is not None and right_candidate is not None:
            pos_error = float(
                np.linalg.norm(left_candidate[:3, 3] - right_candidate[:3, 3])
            )
            q_left = _rotation_to_quaternion(left_candidate[:3, :3])
            q_right = _rotation_to_quaternion(right_candidate[:3, :3])
            dot = min(1.0, abs(float(np.dot(q_left, q_right))))
            rot_error = 2.0 * math.acos(dot)
            normalized_error_sq = (
                (pos_error / self.position_sigma_m) ** 2
                + (rot_error / self.rotation_sigma_rad) ** 2
            )
            gate = math.exp(-0.5 * c_left * c_right * normalized_error_sq)

        confidence_sum = c_left + c_right
        if confidence_sum < self.minimum_confidence_sum:
            raw_alpha = 0.0
        else:
            raw_alpha = (1.0 - (1.0 - c_left) * (1.0 - c_right)) * gate
        raw_alpha = clamp01(raw_alpha)

        candidates = [
            candidate for candidate in (left_candidate, right_candidate) if candidate is not None
        ]
        target = None
        activation_floor = max(self.minimum_confidence_sum, 1e-12)
        target_is_reliable = (
            confidence_sum >= self.minimum_confidence_sum
            and raw_alpha >= activation_floor
        )
        if target_is_reliable and left_candidate is not None and right_candidate is None:
            target = left_candidate.copy()
        elif target_is_reliable and right_candidate is not None and left_candidate is None:
            target = right_candidate.copy()
        elif target_is_reliable and len(candidates) == 2:
            powered_left = c_left**self.confidence_power
            powered_right = c_right**self.confidence_power
            denominator = powered_left + powered_right
            right_weight = 0.5 if denominator <= 1e-12 else powered_right / denominator
            target = np.eye(4, dtype=np.float64)
            target[:3, 3] = (
                (1.0 - right_weight) * left_candidate[:3, 3]
                + right_weight * right_candidate[:3, 3]
            )
            q_left = _rotation_to_quaternion(left_candidate[:3, :3])
            q_right = _rotation_to_quaternion(right_candidate[:3, :3])
            target[:3, :3] = _quaternion_to_rotation(
                quaternion_slerp(q_left, q_right, right_weight)
            )
        if target is not None:
            self.last_target = target.copy()
        elif self.last_target is not None:
            target = self.last_target.copy()

        dt = self.nominal_dt
        if self.last_time is not None:
            dt = max(0.0, min(0.1, float(now) - self.last_time))
        self.last_time = float(now)
        delta = raw_alpha - self.alpha
        lower = -self.alpha_rate_down_per_sec * dt
        upper = self.alpha_rate_up_per_sec * dt
        self.alpha = clamp01(self.alpha + max(lower, min(upper, delta)))

        return TorsoFusionResult(
            target=target,
            alpha=self.alpha,
            raw_alpha=raw_alpha,
            consistency_gate=clamp01(gate),
            left_confidence=c_left,
            right_confidence=c_right,
            ready=bool(target_is_reliable and candidates),
            position_disagreement_m=pos_error,
            rotation_disagreement_rad=rot_error,
        )
