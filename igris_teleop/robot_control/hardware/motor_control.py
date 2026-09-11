
import sys
import pathlib

# Ensure local package imports resolve when this file is executed directly.
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import threading
import time
import yaml

import igris_c_sdk as igc_sdk
from igris_teleop.core.project_paths import JOINT_SETTING_PATH, PR2AB_CALIBRATION_CONFIG_PATH, ROBOT_CONTROL_OUTPUTS_ROOT
from ...core.rate import Rate
from ..filters.torque_kalman import TorqueKalmanBank

import logging_mp
logger_mp = logging_mp.get_logger(__name__, level=logging_mp.INFO)

# DDS topic names (match current client.py defaults)
kTopicLowCommand = "rt/lowcmd"
kTopicLowState = "rt/lowstate"

NUM_MOTORS = 31  # == N_JOINTS (fixed)
DEFAULT_VEL_LIMIT = 5.0
RESERVED_JOINT_PROFILE_KEYS = frozenset(
    {"kp", "kd", "default_dof_pos", "waypoint_1", "waypoint_2", "joint_order", "joint_groups"}
)


# ---------------------------------------------------------------------- #
# Indices / ordering
#
# IMPORTANT:
# - LowCmd.motors()[i] ordering is FIXED (31 length).
# - Interpretation depends on KinematicMode:
#   * PJS: motors()[i].q is "Joint target" (PR space) in the "Joint Name" column.
#   * MS : motors()[i].q is "Motor target" (AB space) in the "Motor Name" column.
#
# The numeric order is the same; only semantic meaning changes for parallel mechanisms.
# ---------------------------------------------------------------------- #

class JointIndex(IntEnum):
    """PJS (joint/PR space) semantic names; 0-based index matches motors()[i] order."""

    # Waist
    WAIST_YAW   = 0   # Motor ID 1 : Motor=Waist_Yaw     | Joint=Waist_Yaw
    WAIST_ROLL  = 1   # Motor ID 2 : Motor=Waist_L       | Joint=Waist_Roll
    WAIST_PITCH = 2   # Motor ID 3 : Motor=Waist_R       | Joint=Waist_Pitch

    # Left leg
    L_HIP_PITCH   = 3   # Motor ID 4 : Motor=Hip_Pitch_L   | Joint=Hip_Pitch_L
    L_HIP_ROLL    = 4   # Motor ID 5 : Motor=Hip_Roll_L    | Joint=Hip_Roll_L
    L_HIP_YAW     = 5   # Motor ID 6 : Motor=Hip_Yaw_L     | Joint=Hip_Yaw_L
    L_KNEE_PITCH  = 6   # Motor ID 7 : Motor=Knee_Pitch_L  | Joint=Knee_Pitch_L
    L_ANKLE_PITCH = 7   # Motor ID 8 : Motor=Ankle_Out_L   | Joint=Ankle_Pitch_L
    L_ANKLE_ROLL  = 8   # Motor ID 9 : Motor=Ankle_In_L    | Joint=Ankle_Roll_L

    # Right leg
    R_HIP_PITCH   = 9    # Motor ID 10: Motor=Hip_Pitch_R   | Joint=Hip_Pitch_R
    R_HIP_ROLL    = 10   # Motor ID 11: Motor=Hip_Roll_R    | Joint=Hip_Roll_R
    R_HIP_YAW     = 11   # Motor ID 12: Motor=Hip_Yaw_R     | Joint=Hip_Yaw_R
    R_KNEE_PITCH  = 12   # Motor ID 13: Motor=Knee_Pitch_R  | Joint=Knee_Pitch_R
    R_ANKLE_PITCH = 13   # Motor ID 14: Motor=Ankle_Out_R   | Joint=Ankle_Pitch_R
    R_ANKLE_ROLL  = 14   # Motor ID 15: Motor=Ankle_In_R    | Joint=Ankle_Roll_R

    # Left arm
    L_SHOULDER_PITCH = 15  # Motor ID 16: Motor=Shoulder_Pitch_L | Joint=Shoulder_Pitch_L
    L_SHOULDER_ROLL  = 16  # Motor ID 17: Motor=Shoulder_Roll_L  | Joint=Shoulder_Roll_L
    L_SHOULDER_YAW   = 17  # Motor ID 18: Motor=Shoulder_Yaw_L   | Joint=Shoulder_Yaw_L
    L_ELBOW_PITCH    = 18  # Motor ID 19: Motor=Elbow_Pitch_L    | Joint=Elbow_Pitch_L
    L_WRIST_YAW      = 19  # Motor ID 20: Motor=Wrist_Yaw_L      | Joint=Wrist_Yaw_L
    L_WRIST_ROLL     = 20  # Motor ID 21: Motor=Wrist_Front_L    | Joint=Wrist_Roll_L
    L_WRIST_PITCH    = 21  # Motor ID 22: Motor=Wrist_Back_L     | Joint=Wrist_Pitch_L

    # Right arm
    R_SHOULDER_PITCH = 22  # Motor ID 23: Motor=Shoulder_Pitch_R | Joint=Shoulder_Pitch_R
    R_SHOULDER_ROLL  = 23  # Motor ID 24: Motor=Shoulder_Roll_R  | Joint=Shoulder_Roll_R
    R_SHOULDER_YAW   = 24  # Motor ID 25: Motor=Shoulder_Yaw_R   | Joint=Shoulder_Yaw_R
    R_ELBOW_PITCH    = 25  # Motor ID 26: Motor=Elbow_Pitch_R    | Joint=Elbow_Pitch_R
    R_WRIST_YAW      = 26  # Motor ID 27: Motor=Wrist_Yaw_R      | Joint=Wrist_Yaw_R
    R_WRIST_ROLL     = 27  # Motor ID 28: Motor=Wrist_Front_R    | Joint=Wrist_Roll_R
    R_WRIST_PITCH    = 28  # Motor ID 29: Motor=Wrist_Back_R     | Joint=Wrist_Pitch_R

    # Neck
    NECK_YAW   = 29        # Motor ID 30: Motor=Neck_Yaw      | Joint=Neck_Yaw
    NECK_PITCH = 30        # Motor ID 31: Motor=Neck_Pitch    | Joint=Neck_Pitch


class MotorIndex(IntEnum):
    """MS (motor/AB space) semantic names; 0-based index matches motors()[i] order."""

    WAIST_YAW = 0
    WAIST_L   = 1
    WAIST_R   = 2

    HIP_PITCH_L  = 3
    HIP_ROLL_L   = 4
    HIP_YAW_L    = 5
    KNEE_PITCH_L = 6
    ANKLE_OUT_L  = 7
    ANKLE_IN_L   = 8

    HIP_PITCH_R  = 9
    HIP_ROLL_R   = 10
    HIP_YAW_R    = 11
    KNEE_PITCH_R = 12
    ANKLE_OUT_R  = 13
    ANKLE_IN_R   = 14

    SHOULDER_PITCH_L = 15
    SHOULDER_ROLL_L  = 16
    SHOULDER_YAW_L   = 17
    ELBOW_PITCH_L    = 18
    WRIST_YAW_L      = 19
    WRIST_FRONT_L    = 20
    WRIST_BACK_L     = 21

    SHOULDER_PITCH_R = 22
    SHOULDER_ROLL_R  = 23
    SHOULDER_YAW_R   = 24
    ELBOW_PITCH_R    = 25
    WRIST_YAW_R      = 26
    WRIST_FRONT_R    = 27
    WRIST_BACK_R     = 28

    NECK_YAW   = 29
    NECK_PITCH = 30


# Group indices (PJS 기준: 상위 제어에서 사용하는 인덱스)
LEG_INDICES = (
    JointIndex.L_HIP_PITCH,
    JointIndex.L_HIP_ROLL,
    JointIndex.L_HIP_YAW,
    JointIndex.L_KNEE_PITCH,
    JointIndex.L_ANKLE_PITCH,
    JointIndex.L_ANKLE_ROLL,
    JointIndex.R_HIP_PITCH,
    JointIndex.R_HIP_ROLL,
    JointIndex.R_HIP_YAW,
    JointIndex.R_KNEE_PITCH,
    JointIndex.R_ANKLE_PITCH,
    JointIndex.R_ANKLE_ROLL,
)

WAIST_INDICES = (
    JointIndex.WAIST_YAW,
    JointIndex.WAIST_ROLL,
    JointIndex.WAIST_PITCH,
)

ARM_INDICES = (
    JointIndex.L_SHOULDER_PITCH,
    JointIndex.L_SHOULDER_ROLL,
    JointIndex.L_SHOULDER_YAW,
    JointIndex.L_ELBOW_PITCH,
    JointIndex.L_WRIST_YAW,
    JointIndex.L_WRIST_ROLL,
    JointIndex.L_WRIST_PITCH,
    JointIndex.R_SHOULDER_PITCH,
    JointIndex.R_SHOULDER_ROLL,
    JointIndex.R_SHOULDER_YAW,
    JointIndex.R_ELBOW_PITCH,
    JointIndex.R_WRIST_YAW,
    JointIndex.R_WRIST_ROLL,
    JointIndex.R_WRIST_PITCH,
)

NECK_INDICES = (
    JointIndex.NECK_YAW,
    JointIndex.NECK_PITCH,
)


# ---------------------------------------------------------------------- #
# PR <-> AB transform support (placeholder; user can refine M/centers/limits)
# ---------------------------------------------------------------------- #

@dataclass
class PR2ABPair:
    """
    A 2-DOF parallel mechanism pair.
      pr = [q_pr1, q_pr2] in PJS (joint space)
      ab = [q_ab1, q_ab2] in MS  (motor space)

    Mapping (delta form):
      ab = ab0 + M @ (pr - pr0)

    Notes:
    - pr indices should reference JointIndex.
    - ab indices should reference MotorIndex.
    - M/pr0/ab0/limits are placeholders; fill with identified values.
    """
    name: str
    pr_i1: int
    pr_i2: int
    ab_i1: int
    ab_i2: int
    M: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float64))
    pr_center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    ab_center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    ab_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    enabled: bool = False

    def configured(self) -> bool:
        return bool(self.enabled)


class LowStateBuffer:
    """Thread-safe buffer to store the latest LowState."""

    def __init__(self):
        self._data = None
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._data

    def set(self, data):
        with self._lock:
            self._data = data


def _load_joint_profile(cfg_path: str):
    """Load kp/kd/default_q from yaml, requiring length == NUM_MOTORS."""
    cfg_path = Path(cfg_path)
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent

    candidates = (
        [cfg_path] if cfg_path.is_absolute() else [base_dir / cfg_path, project_root / cfg_path]
    )
    cfg_file = next((p for p in candidates if p.is_file()), None)
    if cfg_file is None:
        searched = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Joint profile not found. Searched: {searched}")

    cfg = _load_joint_profile_payload(str(cfg_file))

    def _require(seq, name):
        arr = np.asarray(seq, dtype=np.float32).reshape(-1)
        if arr.size != NUM_MOTORS:
            raise ValueError(f"{name} length {arr.size} != {NUM_MOTORS}")
        return arr

    try:
        kp = _require(cfg["kp"], "kp")
        kd = _require(cfg["kd"], "kd")
        default_q = _require(cfg["default_dof_pos"], "default_dof_pos")
        waypoint_1 = _require(cfg["waypoint_1"], "waypoint_1")
        waypoint_2 = _require(cfg["waypoint_2"], "waypoint_2")
    except KeyError as exc:
        raise KeyError(f"Missing key in joint profile: {exc}") from exc

    return kp, kd, default_q, waypoint_1, waypoint_2


def _load_joint_profile_payload(cfg_path: str) -> dict:
    cfg_file = Path(cfg_path)
    with cfg_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Joint profile must be a mapping: {cfg_file}")
    return cfg


def _load_optional_joint_poses(cfg_path: str) -> Dict[str, np.ndarray]:
    cfg = _load_joint_profile_payload(cfg_path)
    poses: Dict[str, np.ndarray] = {}
    for key, value in cfg.items():
        if key in RESERVED_JOINT_PROFILE_KEYS or not isinstance(value, (list, tuple)):
            continue
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            continue
        if arr.size != NUM_MOTORS:
            continue
        poses[key] = arr
    return poses


class BaseController:
    """
    Low-level controller wrapper.

    Design (your requested behavior):
      - Upper-level (whole-body) control operates in PJS (joint/PR space).
      - Feedback for the upper-level control should be joint_state().q (PJS).
      - When publishing LowCmd:
          * If kinematic_mode == PJS: send q as-is.
          * If kinematic_mode == MS : convert selected PR pairs to AB motor commands, then send.

    You can fill the PR<->AB transforms via set_pr2ab_transform().
    """

    def __init__(
        self,
        domain_id: int = 0,
        control_hz: float = 300.0,
        kinematic_mode=igc_sdk.KinematicMode.PJS,
        use_motor_state: bool = False,
        auto_service_init: bool = True,
        bms_init_type=igc_sdk.BmsInitType.BMS_AND_MOTOR_INIT,
        torque_type=igc_sdk.TorqueType.TORQUE_ON,
        control_mode=igc_sdk.ControlMode.CONTROL_MODE_LOW_LEVEL,
        service_timeout_ms: int = 30000,
        service_init_delay: float = 0.5,
        pr2ab_config_path: Optional[str] = None,
    ):
        logger_mp.info("[BaseController] Initializing (domain_id=%s)...", domain_id)

        self.control_rate = Rate(control_hz)
        self.control_dt = 1.0 / control_hz
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._ctrl_lock = threading.Lock()
        self._dds_cleaned = False
        self.lowstate_sub = None
        self.lowcmd_pub = None

        # NOTE: for your intended architecture, keep this False (use joint_state as feedback)
        self._use_motor_state = use_motor_state

        self.channel_factory = igc_sdk.ChannelFactory.Instance()
        self.channel_factory.Init(int(domain_id))

        self.client = igc_sdk.IgrisC_Client()
        self.client.Init()
        self.client.SetTimeout(service_timeout_ms / 1000.0)

        kp_default, kd_default, default_q, waypoint_1, waypoint_2 = _load_joint_profile(
            str(JOINT_SETTING_PATH)
        )
        optional_poses = _load_optional_joint_poses(str(JOINT_SETTING_PATH))

        # Gains
        self._kp_default = kp_default.copy()
        self._kd_default = kd_default.copy()
        self._kp = kp_default.copy()
        self._kd = kd_default.copy()

        # Poses (PJS)
        self._default_q = default_q.astype(np.float32)
        self._waypoint_1 = waypoint_1.astype(np.float32)
        self._waypoint_2 = waypoint_2.astype(np.float32)

        self._poses: Dict[str, np.ndarray] = {}
        self.register_pose("default_pos", self._default_q)
        self.register_pose("zero_pos", np.zeros_like(self._default_q))
        self.register_pose("waypoint_1", self._waypoint_1)
        self.register_pose("waypoint_2", self._waypoint_2)
        for pose_name, pose_q in optional_poses.items():
            self.register_pose(pose_name, pose_q)
        if optional_poses:
            logger_mp.info("[BaseController] registered extra poses: %s", sorted(optional_poses))

        # Targets (kept in PJS semantics)
        self._target_q = np.zeros_like(self._default_q)
        self._target_dq = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._target_tau = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._initial_q = None

        # Velocity limits (PJS)
        self.leg_velocity_limit = DEFAULT_VEL_LIMIT
        self.waist_velocity_limit = DEFAULT_VEL_LIMIT
        self.arm_velocity_limit = DEFAULT_VEL_LIMIT
        self.neck_velocity_limit = DEFAULT_VEL_LIMIT

        # Torque filter (based on joint_state().tau_est)
        self._tau_kf = TorqueKalmanBank(n_joints=NUM_MOTORS, q=3.0, r=20.0, P0=1e3)

        self._dbg_publish_logged = False
        self._warned_unconfigured_ms = False

        # PR->AB transforms (placeholders)
        self._pr2ab_pairs: Dict[str, PR2ABPair] = {
            # Waist roll/pitch -> Waist_L/Waist_R
            "waist_rp": PR2ABPair(
                "waist_rp",
                int(JointIndex.WAIST_ROLL), int(JointIndex.WAIST_PITCH),
                int(MotorIndex.WAIST_L), int(MotorIndex.WAIST_R),
            ),
            # Left ankle pitch/roll -> Ankle_Out_L/Ankle_In_L
            "l_ankle_pr": PR2ABPair(
                "l_ankle_pr",
                int(JointIndex.L_ANKLE_PITCH), int(JointIndex.L_ANKLE_ROLL),
                int(MotorIndex.ANKLE_OUT_L), int(MotorIndex.ANKLE_IN_L),
            ),
            # Right ankle pitch/roll -> Ankle_Out_R/Ankle_In_R
            "r_ankle_pr": PR2ABPair(
                "r_ankle_pr",
                int(JointIndex.R_ANKLE_PITCH), int(JointIndex.R_ANKLE_ROLL),
                int(MotorIndex.ANKLE_OUT_R), int(MotorIndex.ANKLE_IN_R),
            ),
            # Left wrist roll/pitch -> Wrist_Front_L/Wrist_Back_L
            "l_wrist_rp": PR2ABPair(
                "l_wrist_rp",
                int(JointIndex.L_WRIST_ROLL), int(JointIndex.L_WRIST_PITCH),
                int(MotorIndex.WRIST_FRONT_L), int(MotorIndex.WRIST_BACK_L),
            ),
            # Right wrist roll/pitch -> Wrist_Front_R/Wrist_Back_R
            "r_wrist_rp": PR2ABPair(
                "r_wrist_rp",
                int(JointIndex.R_WRIST_ROLL), int(JointIndex.R_WRIST_PITCH),
                int(MotorIndex.WRIST_FRONT_R), int(MotorIndex.WRIST_BACK_R),
            ),
        }
        self._pr2ab_config_path = self._resolve_pr2ab_config_path(pr2ab_config_path)
        self.load_pr2ab_transforms_from_yaml(self._pr2ab_config_path)

        self.state_buffer = LowStateBuffer()

        # LowStateSubscriber
        logger_mp.info("[BaseController] lowstate_sub")
        self.lowstate_sub = igc_sdk.LowStateSubscriber(kTopicLowState)
        try:
            self.lowstate_sub.init(self._on_low_state)
        except Exception as e:
            self._cleanup_dds()
            raise RuntimeError("Failed to init LowStateSubscriber") from e

        # Optional auto-init services
        logger_mp.info("[BaseController] auto_service_init")
        if auto_service_init:
            if service_init_delay > 0.0:
                time.sleep(service_init_delay)
            self._service_bootstrap(
                bms_init_type=bms_init_type,
                torque_type=torque_type,
                control_mode=control_mode,
                timeout_ms=service_timeout_ms,
            )

        # LowCmdPublisher
        self.lowcmd_pub = igc_sdk.LowCmdPublisher(kTopicLowCommand)
        try:
            self.lowcmd_pub.init()
        except Exception as e:
            self._cleanup_dds()
            raise RuntimeError("Failed to init LowCmdPublisher") from e

        self._kinematic_mode = kinematic_mode

        self._ctrl_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._ctrl_thread.start()

        logger_mp.info("[BaseController] Init OK, waiting for first LowState...")

    def _cleanup_dds(self) -> None:
        if self._dds_cleaned:
            return
        self._dds_cleaned = True

        if self.lowstate_sub is not None:
            try:
                self.lowstate_sub.stop()
            except Exception:
                logger_mp.debug("[BaseController] failed to stop LowStateSubscriber", exc_info=True)

        if self.lowcmd_pub is not None:
            try:
                self.lowcmd_pub.stop()
            except Exception:
                logger_mp.debug("[BaseController] failed to stop LowCmdPublisher", exc_info=True)

        try:
            self.channel_factory.Release()
        except Exception:
            logger_mp.debug("[BaseController] failed to release ChannelFactory", exc_info=True)

    # -------------------------- transform API -------------------------- #

    @staticmethod
    def _normalize_limits(raw_limits):
        if raw_limits is None:
            return None
        if len(raw_limits) != 2:
            raise ValueError("ab_limits must have length 2")
        a = raw_limits[0]
        b = raw_limits[1]
        return ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))

    @staticmethod
    def _resolve_pr2ab_config_path(raw_path: Optional[str]) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        if raw_path is None:
            if PR2AB_CALIBRATION_CONFIG_PATH.is_file():
                return PR2AB_CALIBRATION_CONFIG_PATH
            return ROBOT_CONTROL_OUTPUTS_ROOT / "pr2ab_calibration.yaml"
        p = Path(raw_path).expanduser()
        if p.is_absolute():
            return p
        return repo_root / p

    def load_pr2ab_transforms_from_yaml(self, cfg_path: Path) -> None:
        if not cfg_path.is_file():
            logger_mp.info("[BaseController] PR2AB config not found: %s (placeholders kept)", cfg_path)
            return

        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            logger_mp.warning("[BaseController] Failed to read PR2AB config '%s': %s", cfg_path, exc)
            return

        pairs_cfg = cfg.get("pairs", cfg)
        if not isinstance(pairs_cfg, dict):
            logger_mp.warning("[BaseController] Invalid PR2AB config format in '%s'", cfg_path)
            return

        loaded = []
        for name, item in pairs_cfg.items():
            if name not in self._pr2ab_pairs:
                logger_mp.warning("[BaseController] Unknown PR2AB pair in config '%s': %s", cfg_path, name)
                continue
            if not isinstance(item, dict):
                logger_mp.warning("[BaseController] Invalid PR2AB entry for '%s': expected mapping", name)
                continue
            try:
                self.set_pr2ab_transform(
                    name=name,
                    M=np.asarray(item["M"], dtype=np.float64).reshape(2, 2),
                    pr_center=item.get("pr_center"),
                    ab_center=item.get("ab_center"),
                    ab_limits=self._normalize_limits(item.get("ab_limits")),
                )
                loaded.append(name)
            except Exception as exc:
                logger_mp.warning("[BaseController] Failed to load PR2AB pair '%s': %s", name, exc)
                continue

        if loaded:
            logger_mp.info("[BaseController] Loaded PR2AB transforms from %s: %s", cfg_path, loaded)
        else:
            logger_mp.info("[BaseController] No valid PR2AB entries loaded from %s", cfg_path)

    def set_pr2ab_transform(
        self,
        name: str,
        M: np.ndarray,
        pr_center: Optional[Sequence[float]] = None,
        ab_center: Optional[Sequence[float]] = None,
        ab_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> None:
        """
        Update/override a PR->AB transform.

        Example:
          ctrl.set_pr2ab_transform(
              "r_wrist_rp",
              M=np.array([[1, 1],[1, -1]]),
              pr_center=[roll0, pitch0],
              ab_center=[qf0, qb0],
              ab_limits=((-2.0, 2.0), (-2.0, 2.0))
          )
        """
        if name not in self._pr2ab_pairs:
            raise KeyError(f"Unknown PR2AB pair name '{name}'. Available: {list(self._pr2ab_pairs.keys())}")

        pair = self._pr2ab_pairs[name]
        pair.M = np.asarray(M, dtype=np.float64).reshape(2, 2)
        if pr_center is not None:
            pair.pr_center = np.asarray(pr_center, dtype=np.float64).reshape(2)
        if ab_center is not None:
            pair.ab_center = np.asarray(ab_center, dtype=np.float64).reshape(2)
        pair.ab_limits = self._normalize_limits(ab_limits)
        pair.enabled = True

    def _apply_pr2ab_pair(self, q_pjs: np.ndarray, pair: PR2ABPair) -> np.ndarray:
        pr = np.array([q_pjs[pair.pr_i1], q_pjs[pair.pr_i2]], dtype=np.float64)
        ab = pair.ab_center + pair.M @ (pr - pair.pr_center)

        if pair.ab_limits is not None:
            lo = np.array([pair.ab_limits[0][0], pair.ab_limits[1][0]], dtype=np.float64)
            hi = np.array([pair.ab_limits[0][1], pair.ab_limits[1][1]], dtype=np.float64)
            ab = np.clip(ab, lo, hi)

        return ab.astype(np.float32)

    def _pjs_to_ms_q(self, q_pjs: np.ndarray) -> np.ndarray:
        """
        Convert (31,) PJS target q -> (31,) MS target q.
        - Copies q_pjs then overwrites motor indices for configured PR2AB pairs.
        """
        q_ms = np.array(q_pjs, dtype=np.float32, copy=True)
        unconfigured = []

        for name, pair in self._pr2ab_pairs.items():
            if not pair.configured():
                unconfigured.append(name)
                # still apply identity mapping only if you explicitly want that; by default skip
                continue
            ab = self._apply_pr2ab_pair(q_pjs, pair)
            q_ms[pair.ab_i1] = ab[0]
            q_ms[pair.ab_i2] = ab[1]

        if unconfigured and not self._warned_unconfigured_ms:
            logger_mp.warning(
                "[BaseController] MS mode but some PR2AB transforms are not configured (skipped): %s",
                unconfigured,
            )
            self._warned_unconfigured_ms = True

        return q_ms

    # -------------------------- services -------------------------- #

    def _call_service_with_retry(self, fn, name: str, attempts: int = 3, delay_sec: float = 1.0):
        last_res = None
        for i in range(1, attempts + 1):
            res = fn()
            last_res = res
            if res.success():
                logger_mp.info("[BaseController] %s succeeded: %s", name, res.message())
                return res
            logger_mp.warning("[BaseController] %s failed (%d/%d): %s", name, i, attempts, res.message())
            if i < attempts:
                time.sleep(delay_sec)
        raise RuntimeError(f"{name} failed after {attempts} attempts: {last_res.message()}")

    def _service_bootstrap(self, bms_init_type, torque_type, control_mode, timeout_ms: int):
        logger_mp.info("[BaseController] InitBms start (%s, timeout=%d ms)", bms_init_type, timeout_ms)
        self._call_service_with_retry(
            lambda: self.client.InitBms(bms_init_type, timeout_ms),
            f"InitBms({bms_init_type})",
        )
        logger_mp.info("[BaseController] SetTorque start (%s, timeout=%d ms)", torque_type, timeout_ms)
        self._call_service_with_retry(
            lambda: self.client.SetTorque(torque_type, timeout_ms),
            f"SetTorque({torque_type})",
        )
        logger_mp.info("[BaseController] SetControlMode start (%s, timeout=%d ms)", control_mode, timeout_ms)
        self._call_service_with_retry(
            lambda: self.client.SetControlMode(control_mode, timeout_ms),
            f"SetControlMode({control_mode})",
        )

    def _service_shutdown(
        self,
        torque_type=igc_sdk.TorqueType.TORQUE_OFF,
        control_mode=igc_sdk.ControlMode.CONTROL_MODE_HIGH_LEVEL,
        timeout_ms: int = 30000,
    ):
        if control_mode is not None:
            logger_mp.info("[BaseController] SetControlMode (shutdown) start -> %s", control_mode)
            res = self.client.SetControlMode(control_mode, timeout_ms)
            if not res.success():
                logger_mp.warning("[BaseController] SetControlMode (shutdown) failed: %s", res.message())

        logger_mp.info("[BaseController] Torque OFF (shutdown) start")
        res = self.client.SetTorque(torque_type, timeout_ms)
        if not res.success():
            logger_mp.warning("[BaseController] Torque OFF failed: %s", res.message())

    # -------------------------- subscription -------------------------- #

    def _extract_state_q(self, state: igc_sdk.LowState) -> np.ndarray:
        # if self._use_motor_state:
        #     return np.array([ms.q() for ms in state.motor_state()], dtype=np.float32)
        return np.array([js.q() for js in state.joint_state()], dtype=np.float32)

    def _on_low_state(self, state: igc_sdk.LowState):
        if not self._ready_event.is_set():
            self._initial_q = self._extract_state_q(state)

            first_joint_tau_est = np.array([js.tau_est() for js in state.joint_state()], dtype=np.float32)
            self._tau_kf.reset_with_measurement(first_joint_tau_est)

            motor_q = np.array([ms.q() for ms in state.motor_state()], dtype=np.float32)
            logger_mp.info(
                "[BaseController] First LowState q stats (joint min/max=%.3f/%.3f, motor min/max=%.3f/%.3f)",
                float(self._initial_q.min()),
                float(self._initial_q.max()),
                float(motor_q.min()),
                float(motor_q.max()),
            )

            with self._ctrl_lock:
                self._target_q = self._initial_q.copy()
                self._target_dq[:] = 0.0
                self._target_tau[:] = 0.0

            self._ready_event.set()
            logger_mp.info("[BaseController] First LowState received.")

        self.state_buffer.set(state)

    def wait_for_state(self, timeout: Optional[float] = 5.0) -> bool:
        return self._ready_event.wait(timeout=timeout)

    # -------------------------- control loop -------------------------- #

    def _publish_loop(self):
        while not self._stop_event.is_set():
            if self.wait_for_state(timeout=0.1):
                break

        while not self._stop_event.is_set():
            with self._ctrl_lock:
                q_pjs = self._target_q.copy()
                dq = self._target_dq.copy()
                tau = self._target_tau.copy()
                kp = self._kp.copy()
                kd = self._kd.copy()
                mode = self._kinematic_mode

            # 핵심: MS면 q만 PJS->MS 변환해서 넣음
            if mode == igc_sdk.KinematicMode.MS:
                q_cmd = self._pjs_to_ms_q(q_pjs)
            else:
                q_cmd = q_pjs

            low_cmd_msg = igc_sdk.LowCmd()
            low_cmd_msg.kinematic_mode(mode)

            motors = low_cmd_msg.motors()
            for idx in range(NUM_MOTORS):
                mc = motors[idx]
                mc.id(idx)
                mc.q(float(q_cmd[idx]))
                mc.dq(float(dq[idx]))
                mc.tau(float(tau[idx]))
                mc.kp(float(kp[idx]))
                mc.kd(float(kd[idx]))

            if not self._dbg_publish_logged:
                logger_mp.info(
                    "[BaseController] First publish targets: mode=%s q min/max=%.3f/%.3f",
                    str(mode),
                    float(q_cmd.min()),
                    float(q_cmd.max()),
                )
                self._dbg_publish_logged = True

            self.lowcmd_pub.write(low_cmd_msg)
            self.control_rate.sleep()

    # -------------------------- public APIs -------------------------- #

    @staticmethod
    def _pad_array(arr):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.size < NUM_MOTORS:
            arr = np.pad(arr, (0, NUM_MOTORS - arr.size))
        return arr[:NUM_MOTORS]

    def register_pose(self, name: str, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        if not hasattr(self, "_default_q"):
            raise RuntimeError("Default q is not initialized yet.")
        if q.shape != self._default_q.shape:
            raise ValueError(f"Pose '{name}' has invalid shape {q.shape}, expected {self._default_q.shape}.")
        self._poses[name] = q.copy()

    def set_joint_targets(self, q=None, dq=None, tau=None, kp=None, kd=None):
        with self._ctrl_lock:
            if q is not None:
                self._target_q = self._pad_array(q)
            if dq is not None:
                self._target_dq = self._pad_array(dq)
            if tau is not None:
                self._target_tau = self._pad_array(tau)
            if kp is not None:
                self._kp = self._pad_array(kp)
            if kd is not None:
                self._kd = self._pad_array(kd)

    def hold_current_position(self):
        state = self.state_buffer.get()
        if state is None:
            return
        q_now = np.array([js.q() for js in state.joint_state()], dtype=np.float32)
        self.set_joint_targets(q=q_now, dq=np.zeros(NUM_MOTORS), tau=np.zeros(NUM_MOTORS))

    def set_kinematic_mode(self, mode):
        with self._ctrl_lock:
            self._kinematic_mode = mode
        # reset one-time debug prints
        self._dbg_publish_logged = False
        self._warned_unconfigured_ms = False

    def _collect_joint_indices(self, leg: bool = True, waist: bool = True, arm: bool = True, neck: bool = True):
        indices = []
        if leg:
            indices.extend(list(LEG_INDICES))
        if waist:
            indices.extend(list(WAIST_INDICES))
        if arm:
            indices.extend(list(ARM_INDICES))
        if neck:
            indices.extend(list(NECK_INDICES))
        return [int(i) for i in sorted(set(indices))]

    def zero_torque(self):
        self.set_joint_targets(
            tau=np.zeros(NUM_MOTORS, dtype=np.float32),
            kp=np.zeros(NUM_MOTORS, dtype=np.float32),
            kd=np.zeros(NUM_MOTORS, dtype=np.float32),
        )

    def _clip_targets(self, target_q, current_q, velocity_limit):
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        return current_q + delta / max(motion_scale, 1.0)

    def _set_group(self, indices, q_target, tau_target=None, apply_clip=True, vel_limit=DEFAULT_VEL_LIMIT):
        q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)
        if len(indices) != q_target.size:
            raise ValueError(f"q_target length {q_target.size} != len(indices) {len(indices)}")
        if tau_target is None:
            tau_target = np.zeros_like(q_target)
        tau_target = np.asarray(tau_target, dtype=np.float32).reshape(-1)
        if tau_target.size != q_target.size:
            raise ValueError("tau_target length mismatch")

        current_q = self.get_joint_q()
        if current_q is None:
            return

        current_group_q = current_q[list(indices)]
        if apply_clip:
            q_target = self._clip_targets(q_target, current_group_q, vel_limit)

        with self._ctrl_lock:
            for idx, jid in enumerate(indices):
                self._target_q[jid] = q_target[idx]
                self._target_tau[jid] = tau_target[idx]
        self._dbg_publish_logged = False

    def ctrl_waist(self, q_target, tau_target=None, apply_clip=True):
        self._set_group(WAIST_INDICES, q_target, tau_target, apply_clip, self.waist_velocity_limit)

    def ctrl_leg(self, q_target, tau_target=None, apply_clip=True):
        self._set_group(LEG_INDICES, q_target, tau_target, apply_clip, self.leg_velocity_limit)

    def ctrl_arm(self, q_target, tau_target=None, apply_clip=True):
        self._set_group(ARM_INDICES, q_target, tau_target, apply_clip, self.arm_velocity_limit)

    def ctrl_neck(self, q_target, tau_target=None, apply_clip=True):
        self._set_group(NECK_INDICES, q_target, tau_target, apply_clip, self.neck_velocity_limit)

    def get_joint_q(self):
        state = self.state_buffer.get()
        if state is None:
            return None
        return self._extract_state_q(state)

    def get_joint_dq(self):
        state = self.state_buffer.get()
        if state is None:
            return None
        return np.array([js.dq() for js in state.joint_state()], dtype=np.float32)

    def get_joint_tau(self):
        state = self.state_buffer.get()
        if state is None:
            return None
        tau_raw = np.array([js.tau_est() for js in state.joint_state()], dtype=np.float32)
        tau_filt = self._tau_kf.step(tau_raw) if getattr(self, "_tau_kf", None) is not None else tau_raw
        return tau_raw, tau_filt

    def get_motor_q(self):
        state = self.state_buffer.get()
        if state is None:
            return None
        return np.array([ms.q() for ms in state.motor_state()], dtype=np.float32)

    def get_motor_dq(self):
        state = self.state_buffer.get()
        if state is None:
            return None
        return np.array([ms.dq() for ms in state.motor_state()], dtype=np.float32)

    def get_imu(self):
        state = self.state_buffer.get()
        if state is None:
            return None, None, None
        imu = state.imu_state()
        return (
            np.array(imu.quaternion(), dtype=np.float32),
            np.array(imu.gyroscope(), dtype=np.float32),
            np.array(imu.rpy(), dtype=np.float32),
        )

    def move_to_pose(
        self,
        pose: Union[str, np.ndarray],
        duration: float = 2.0,
        leg: bool = True,
        waist: bool = True,
        arm: bool = True,
        neck: bool = True,
    ) -> None:
        if duration <= 0.0:
            duration = self.control_dt

        if isinstance(pose, str):
            if pose not in self._poses:
                raise KeyError(f"Unknown pose name: {pose}. Available: {list(self._poses.keys())}")
            q_goal = self._poses[pose]
            pose_name = pose
        else:
            q_goal = np.asarray(pose, dtype=float)
            if q_goal.shape != self._default_q.shape:
                raise ValueError(f"Pose array has invalid shape {q_goal.shape}, expected {self._default_q.shape}.")
            pose_name = "custom"

        with self._ctrl_lock:
            q_start = self._target_q.copy()

        joint_ids = self._collect_joint_indices(leg=leg, waist=waist, arm=arm, neck=neck)
        steps = max(1, int(duration / self.control_dt))

        logger_mp.info("[BaseController] move_to_pose(%s) steps=%d duration=%.2fs", pose_name, steps, duration)

        for step in range(1, steps + 1):
            if self._stop_event.is_set():
                break

            alpha = step / steps
            with self._ctrl_lock:
                for jid in joint_ids:
                    q0 = q_start[jid]
                    q1 = q_goal[jid]
                    self._target_q[jid] = float((1.0 - alpha) * q0 + alpha * q1)
                    self._target_dq[jid] = 0.0
                    self._target_tau[jid] = 0.0
                    self._kp[jid] = float(self._kp_default[jid])
                    self._kd[jid] = float(self._kd_default[jid])

            time.sleep(self.control_dt)

        self._dbg_publish_logged = False

    def default_pos_state(
        self,
        pose: Union[str, np.ndarray, None] = None,
        leg: bool = True,
        waist: bool = True,
        arm: bool = True,
        neck: bool = True,
    ):
        if pose is None:
            q_hold = self._default_q
            pose_name = "default_pos"
        elif isinstance(pose, str):
            if pose not in self._poses:
                raise KeyError(f"Unknown pose name: {pose}. Available: {list(self._poses.keys())}")
            q_hold = self._poses[pose]
            pose_name = pose
        else:
            q_hold = np.asarray(pose, dtype=float)
            if q_hold.shape != self._default_q.shape:
                raise ValueError(f"Pose array has invalid shape {q_hold.shape}, expected {self._default_q.shape}.")
            pose_name = "custom"

        joint_ids = self._collect_joint_indices(leg=leg, waist=waist, arm=arm, neck=neck)
        with self._ctrl_lock:
            for jid in joint_ids:
                self._target_q[jid] = float(q_hold[jid])
                self._target_dq[jid] = 0.0
                self._target_tau[jid] = 0.0
                self._kp[jid] = float(self._kp_default[jid])
                self._kd[jid] = float(self._kd_default[jid])
        self._dbg_publish_logged = False
        logger_mp.info(
            "[BaseController] default_pos_state set: pose=%s, target q min/max=%.3f/%.3f",
            pose_name,
            float(self._target_q.min()),
            float(self._target_q.max()),
        )

    # -------------------------- service wrappers -------------------------- #

    def init_bms(self, init_type=igc_sdk.BmsInitType.BMS_AND_MOTOR_INIT, timeout_ms: int = 30000):
        return self.client.InitBms(init_type, timeout_ms)

    def torque_on(self, timeout_ms: int = 30000):
        return self.client.SetTorque(igc_sdk.TorqueType.TORQUE_ON, timeout_ms)

    def torque_off(self, timeout_ms: int = 30000):
        return self.client.SetTorque(igc_sdk.TorqueType.TORQUE_OFF, timeout_ms)

    def set_control_mode(self, mode=igc_sdk.ControlMode.CONTROL_MODE_LOW_LEVEL, timeout_ms: int = 30000):
        return self.client.SetControlMode(mode, timeout_ms)

    def shutdown_services(
        self,
        torque_type=igc_sdk.TorqueType.TORQUE_OFF,
        control_mode=igc_sdk.ControlMode.CONTROL_MODE_HIGH_LEVEL,
        timeout_ms: int = 30000,
    ):
        return self._service_shutdown(torque_type=torque_type, control_mode=control_mode, timeout_ms=timeout_ms)

    def stop(self):
        # Optional: move to safe poses before shutdown, if desired by your stack.
        try:
            # 필요에 따라 duration 조정 (예: 2.0초)        
            self.move_to_pose("waypoint_2", duration=2.0, leg=True, waist=True, arm=True, neck=True)
            self.move_to_pose("waypoint_1", duration=2.0, leg=True, waist=True, arm=True, neck=True)
            self.move_to_pose("zero_pos", duration=2.0, leg=True, waist=True, arm=True, neck=True)
        except Exception as exc:
            logger_mp.warning("[BaseController] move_to_zero_pos in stop() failed: %s", exc)

        
        
        self._stop_event.set()
        try:
            self._service_shutdown(
                torque_type=igc_sdk.TorqueType.TORQUE_OFF,
                control_mode=igc_sdk.ControlMode.CONTROL_MODE_HIGH_LEVEL,
            )
        except Exception as exc:
            logger_mp.warning("[BaseController] Service shutdown failed: %s", exc)
        self._ctrl_thread.join(timeout=30.0)
        self._cleanup_dds()
        logger_mp.info("[BaseController] stop() done")
