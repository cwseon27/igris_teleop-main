from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from ...core.project_paths import (
    REPO_ROOT,
    WALKING_JOINT_SETTING_PATH,
    WALKING_POLICIES_ROOT,
    WALKING_V2_JOINT_SETTING_PATH,
    resolve_under_root,
)


WALKING_MODE_NAME = "walking"
WALKING_STANCE_NAME = "walking_stance"

WALKING_POLICY_PROFILE_V1 = "v1"
WALKING_POLICY_PROFILE_V2_FAST_SAC = "v2_fast_sac"
DEFAULT_WALKING_POLICY_PROFILE = WALKING_POLICY_PROFILE_V1

WALKING_POLICY_PROFILE_CODES: dict[str, int] = {
    WALKING_POLICY_PROFILE_V1: 1,
    WALKING_POLICY_PROFILE_V2_FAST_SAC: 2,
}

WALKING_V1_DEFAULT_POLICY_PATH = (WALKING_POLICIES_ROOT / "policy_1.pt").resolve()
WALKING_V2_DEFAULT_POLICY_PATH = (WALKING_POLICIES_ROOT / "v2" / "model_0030000.onnx").resolve()


@dataclass(frozen=True)
class WalkingPolicyProfileSpec:
    name: str
    backend: str
    default_policy_path: Path
    joint_profile_path: Path
    loop_hz: float
    frame_stack: int
    single_obs_dim: int
    cycle_time: float
    action_scale: float
    action_clip: float
    obs_clip: float
    command_limits: dict[str, tuple[float, float]]
    command_step: dict[str, float]
    uses_projected_gravity: bool


WALKING_POLICY_PROFILES: dict[str, WalkingPolicyProfileSpec] = {
    WALKING_POLICY_PROFILE_V1: WalkingPolicyProfileSpec(
        name=WALKING_POLICY_PROFILE_V1,
        backend="torchscript",
        default_policy_path=WALKING_V1_DEFAULT_POLICY_PATH,
        joint_profile_path=WALKING_JOINT_SETTING_PATH,
        loop_hz=100.0,
        frame_stack=15,
        single_obs_dim=47,
        cycle_time=0.6,
        action_scale=0.25,
        action_clip=18.0,
        obs_clip=18.0,
        command_limits={
            "vx": (-0.3, 1.0),
            "vy": (-0.3, 0.3),
            "dyaw": (-0.3, 0.3),
        },
        command_step={
            "vx": 0.05,
            "vy": 0.05,
            "dyaw": 0.05,
        },
        uses_projected_gravity=False,
    ),
    WALKING_POLICY_PROFILE_V2_FAST_SAC: WalkingPolicyProfileSpec(
        name=WALKING_POLICY_PROFILE_V2_FAST_SAC,
        backend="onnx",
        default_policy_path=WALKING_V2_DEFAULT_POLICY_PATH,
        joint_profile_path=WALKING_V2_JOINT_SETTING_PATH,
        loop_hz=50.0,
        frame_stack=1,
        single_obs_dim=49,
        cycle_time=1.0,
        action_scale=0.25,
        action_clip=100.0,
        obs_clip=100.0,
        command_limits={
            "vx": (-0.8, 0.8),
            "vy": (-0.4, 0.4),
            "dyaw": (-0.8, 0.8),
        },
        command_step={
            "vx": 0.05,
            "vy": 0.05,
            "dyaw": 0.05,
        },
        uses_projected_gravity=True,
    ),
}

WALKING_PROFILE_CHOICES: tuple[str, ...] = tuple(WALKING_POLICY_PROFILES.keys())
WALKING_MAX_SINGLE_OBS_DIM = max(spec.single_obs_dim for spec in WALKING_POLICY_PROFILES.values())

DEFAULT_WALKING_POLICY_PATH = WALKING_POLICY_PROFILES[DEFAULT_WALKING_POLICY_PROFILE].default_policy_path
WALKING_FRAME_STACK = WALKING_POLICY_PROFILES[DEFAULT_WALKING_POLICY_PROFILE].frame_stack
WALKING_SINGLE_OBS_DIM = WALKING_POLICY_PROFILES[DEFAULT_WALKING_POLICY_PROFILE].single_obs_dim
WALKING_ACTION_DIM = 12
WALKING_POLICY_HZ = WALKING_POLICY_PROFILES[DEFAULT_WALKING_POLICY_PROFILE].loop_hz
WALKING_POLICY_DT = 1.0 / WALKING_POLICY_HZ
WALKING_ACTION_SCALE = WALKING_POLICY_PROFILES[DEFAULT_WALKING_POLICY_PROFILE].action_scale
WALKING_CYCLE_TIME = WALKING_POLICY_PROFILES[DEFAULT_WALKING_POLICY_PROFILE].cycle_time
WALKING_OBS_CLIP = WALKING_POLICY_PROFILES[DEFAULT_WALKING_POLICY_PROFILE].obs_clip
WALKING_ACTION_CLIP = WALKING_POLICY_PROFILES[DEFAULT_WALKING_POLICY_PROFILE].action_clip

WALKING_PREP_LEG_WAIST_DURATION = 3.0
WALKING_PREP_ARM_NECK_DURATION = 3.0
WALKING_START_ZERO_CMD_EPS = 1e-2
WALKING_START_POSE_TOL = 0.08
WALKING_START_DQ_TOL = 0.20
WALKING_START_GYRO_TOL = 0.40
WALKING_START_BASE_ROLL_TOL = 0.05
WALKING_START_BASE_PITCH_TOL = 0.05
WALKING_START_SETTLE_TIME = 0.30
WALKING_START_TIMEOUT = 5.0
WALKING_START_BLEND_TIME = 1.0
WALKING_START_REASON_LOG_INTERVAL = 1.0
WALKING_OBS_STALE_TIMEOUT = 0.25
# Startup stance gating ignores ankle joints. On hardware the ankles often sit
# at a static offset from the policy default stance even when the rest of the
# lower body is settled, and that should not block policy arming.
WALKING_START_POSE_CHECK_INDICES = np.array([0, 1, 2, 3, 6, 7, 8, 9], dtype=np.int64)

WALKING_OBS_SCALES = {
    "lin_vel": 2.0,
    "ang_vel": 1.0,
    "dof_pos": 1.0,
    "dof_vel": 0.05,
}

WALKING_CMD_LIMITS = dict(WALKING_POLICY_PROFILES[DEFAULT_WALKING_POLICY_PROFILE].command_limits)
WALKING_CMD_STEP = dict(WALKING_POLICY_PROFILES[DEFAULT_WALKING_POLICY_PROFILE].command_step)

WALKING_LEG_DEFAULT_Q = np.array(
    [-0.05, 0.0, 0.0, 0.36, -0.25, 0.0, -0.05, 0.0, 0.0, 0.36, -0.25, 0.0],
    dtype=np.float64,
)

WALKING_NEUTRAL_FULL_Q = np.array(
    [
        0.0,
        0.0,
        0.0,
        -0.05,
        0.0,
        0.0,
        0.36,
        -0.25,
        0.0,
        -0.05,
        0.0,
        0.0,
        0.36,
        -0.25,
        0.0,
        0.13,
        0.3,
        -0.3,
        0.0,
        0.0,
        0.0,
        0.0,
        0.13,
        -0.3,
        -0.3,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)

WALKING_NEUTRAL_HAND_Q = np.zeros((12,), dtype=np.float64)


def resolve_walking_policy_profile(raw_profile: str | None) -> str:
    profile = (raw_profile or "").strip()
    if not profile:
        return DEFAULT_WALKING_POLICY_PROFILE
    if profile not in WALKING_POLICY_PROFILES:
        raise ValueError(
            f"Unknown walking policy profile: {profile} (choices: {', '.join(WALKING_PROFILE_CHOICES)})"
        )
    return profile


def get_walking_policy_profile(raw_profile: str | None = None) -> WalkingPolicyProfileSpec:
    profile = resolve_walking_policy_profile(raw_profile)
    return WALKING_POLICY_PROFILES[profile]


def walking_policy_profile_code(raw_profile: str | None = None) -> int:
    return int(WALKING_POLICY_PROFILE_CODES[resolve_walking_policy_profile(raw_profile)])


def resolve_walking_policy_profile_code(raw_code: object) -> str:
    try:
        code = int(float(raw_code))
    except Exception:
        return DEFAULT_WALKING_POLICY_PROFILE
    for profile, value in WALKING_POLICY_PROFILE_CODES.items():
        if value == code:
            return profile
    return DEFAULT_WALKING_POLICY_PROFILE


def walking_cmd_limits(raw_profile: str | None = None) -> dict[str, tuple[float, float]]:
    return dict(get_walking_policy_profile(raw_profile).command_limits)


def walking_cmd_step(raw_profile: str | None = None) -> dict[str, float]:
    return dict(get_walking_policy_profile(raw_profile).command_step)


def walking_command_is_zero(cmd: Mapping[str, float] | None) -> bool:
    if not cmd:
        return True
    vx = float(cmd.get("vx", 0.0))
    vy = float(cmd.get("vy", 0.0))
    dyaw = float(cmd.get("dyaw", 0.0))
    planar_speed = float(np.hypot(vx, vy))
    return planar_speed < WALKING_START_ZERO_CMD_EPS and abs(dyaw) < WALKING_START_ZERO_CMD_EPS


def clamp_walking_command(field: str, value: float, *, profile: str | None = None) -> float:
    limits = walking_cmd_limits(profile)
    if field not in limits:
        raise KeyError(f"Unknown walking command field: {field}")
    lower, upper = limits[field]
    return float(np.clip(float(value), lower, upper))


def default_walking_policy_path(raw_profile: str | None = None) -> Path:
    return get_walking_policy_profile(raw_profile).default_policy_path.resolve()


def default_walking_joint_profile_path(raw_profile: str | None = None) -> Path:
    return get_walking_policy_profile(raw_profile).joint_profile_path.resolve()


def walking_policy_file_dialog_filter(raw_profile: str | None = None) -> str:
    profile = get_walking_policy_profile(raw_profile)
    if profile.backend == "onnx":
        return "ONNX (*.onnx);;All files (*)"
    return "TorchScript (*.pt);;All files (*)"


def walking_policy_path_placeholder(raw_profile: str | None = None) -> str:
    profile = get_walking_policy_profile(raw_profile)
    suffix = "igris_artifacts/walking" if profile.name == WALKING_POLICY_PROFILE_V1 else "igris_artifacts/walking/v2"
    return f"walking {profile.backend} model path (relative to {suffix})"


def resolve_walking_policy_path(raw_path: str | None, *, profile: str | None = None) -> Path:
    spec = get_walking_policy_profile(profile)
    txt = (raw_path or "").strip()
    if not txt:
        return spec.default_policy_path

    candidate = Path(txt).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    if candidate.parts and candidate.parts[0] == "igris_artifacts":
        return resolve_under_root(REPO_ROOT, candidate)
    return resolve_under_root(spec.default_policy_path.parent, candidate)
