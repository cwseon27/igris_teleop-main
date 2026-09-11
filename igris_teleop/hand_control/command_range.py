from __future__ import annotations

import numpy as np

HAND_COMMAND_MIN = 0.0
HAND_COMMAND_MAX = 1.0
HAND_CLOSE_GAIN = 1.5
HAND_RETARGET_MAX_RAD = np.asarray((1.23, 1.58, 1.58, 1.58, 1.58, 1.74), dtype=np.float64)
FINGER_CLOSE_COUNT = 5
HAND_MOTOR_COUNT = 6


def validated_motor_command(values) -> np.ndarray:
    """Validate final six-motor commands; these are already normalized/gain-adjusted."""
    motors = np.asarray(values, dtype=np.float64).reshape(-1)
    if motors.size != HAND_MOTOR_COUNT or not np.all(np.isfinite(motors)):
        raise ValueError("motor command must contain six finite normalized values")
    if np.any(motors < 0.0) or np.any(motors > 1.0):
        raise ValueError("motor command must be in [0, 1]")
    return motors.copy()


def apply_motor_command_overrides(
    base_command, *, left_motor=None, right_motor=None,
    left_valid: bool = False, right_valid: bool = False, dtype=None,
) -> np.ndarray:
    """Overlay final [right 6, left 6] commands without gain or thumb duplication."""
    command = np.asarray(base_command, dtype=np.float64).reshape(-1).copy()
    if command.size != 2 * HAND_MOTOR_COUNT:
        raise ValueError("dual hand command must contain twelve values")
    if right_valid:
        command[:HAND_MOTOR_COUNT] = validated_motor_command(right_motor)
    if left_valid:
        command[HAND_MOTOR_COUNT:] = validated_motor_command(left_motor)
    return command.astype(dtype, copy=False) if dtype is not None else command


def normalize_retargeted_hand_joints(values, *, right_hand: bool = False) -> np.ndarray:
    """Convert six DexRetargeting joint angles to 0=open, 1=closed bend values."""
    joints = np.asarray(values, dtype=np.float64).reshape(-1)
    if joints.size != HAND_RETARGET_MAX_RAD.size:
        raise ValueError(
            f"retargeted hand joint length must be {HAND_RETARGET_MAX_RAD.size}, got {joints.size}"
        )
    if not np.all(np.isfinite(joints)):
        return np.zeros(HAND_RETARGET_MAX_RAD.size, dtype=np.float64)

    joints = joints.copy()
    if right_hand:
        joints[5] = abs(joints[5])
    return np.clip(joints / HAND_RETARGET_MAX_RAD, 0.0, 1.0)


def compress_normalized_hand_command(
    values,
    *,
    lower: float = HAND_COMMAND_MIN,
    upper: float = HAND_COMMAND_MAX,
    close_gain: float = HAND_CLOSE_GAIN,
    dtype=None,
) -> np.ndarray:
    """Boost human hand closing so the robot reaches full close earlier."""
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    boosted = np.clip(clipped * float(close_gain), 0.0, 1.0)
    compressed = float(lower) + boosted * float(upper - lower)
    if dtype is not None:
        return compressed.astype(dtype, copy=False)
    return compressed


def finger_close_to_hand_motors(
    values,
    *,
    close_gain: float = HAND_CLOSE_GAIN,
    dtype=None,
) -> np.ndarray:
    """Expand [thumb, index, middle, ring, little] to six hand motor commands."""
    close = np.asarray(values, dtype=np.float64).reshape(-1)
    if close.size != FINGER_CLOSE_COUNT:
        raise ValueError(
            f"normalized finger-close length must be {FINGER_CLOSE_COUNT}, got {close.size}"
        )
    if not np.all(np.isfinite(close)):
        raise ValueError("normalized finger-close command contains NaN or inf")

    expanded = np.concatenate((close, close[:1]))
    if expanded.size != HAND_MOTOR_COUNT:
        raise RuntimeError(f"expanded hand command has unexpected size {expanded.size}")
    return compress_normalized_hand_command(
        expanded,
        close_gain=close_gain,
        dtype=dtype,
    )


def apply_finger_close_overrides(
    base_command,
    *,
    left_close=None,
    right_close=None,
    left_valid: bool = False,
    right_valid: bool = False,
    close_gain: float = HAND_CLOSE_GAIN,
    dtype=None,
) -> np.ndarray:
    """Overlay direct normalized commands on the [right 6, left 6] hand layout."""
    command = np.asarray(base_command, dtype=np.float64).reshape(-1).copy()
    if command.size != 2 * HAND_MOTOR_COUNT:
        raise ValueError(
            f"dual hand command length must be {2 * HAND_MOTOR_COUNT}, got {command.size}"
        )
    if right_valid:
        command[:HAND_MOTOR_COUNT] = finger_close_to_hand_motors(
            right_close,
            close_gain=close_gain,
        )
    if left_valid:
        command[HAND_MOTOR_COUNT:] = finger_close_to_hand_motors(
            left_close,
            close_gain=close_gain,
        )
    command = np.clip(command, HAND_COMMAND_MIN, HAND_COMMAND_MAX)
    if dtype is not None:
        return command.astype(dtype, copy=False)
    return command
