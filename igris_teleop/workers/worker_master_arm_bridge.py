from __future__ import annotations

import os
import threading
import time
import numpy as np
import yaml

try:
    import rclpy
except ImportError:
    rclpy = None

from ..core.events import EventSnapshot
from ..core.state_machine import ModeState, TransitionResult
from ..core.worker_base import (
    LoopDiagnostics,
    SingleRateWorker,
    WorkerContext,
    resolve_teleop_hand_source,
)



import logging_mp
logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)


from ..core.project_paths import ROBOT_CONTROL_CONFIG_ROOT
from ..robot_control.interfaces.master_arm_ros_interface import JOINT_NAMES, MasterArmROSInterface
from ..hand_control.command_range import compress_normalized_hand_command
from ..robot_control.kinematics.joints import ARM_INDICES, NECK_INDICES, WAIST_INDICES


MASTERARM_ARM_ZERO_TOL = 1e-6
MASTERARM_ARM_WAITING_REASON = "waiting_first_sample"
MASTERARM_ARM_INVALID_REASON = "invalid_input"
MASTERARM_ARM_ALL_ZERO_REASON = "all_zero_input"
MASTERARM_ARM_STALE_REASON = "stale_input"
MASTERARM_BRIDGE_DEFAULT_HZ = 200.0
MASTERARM_JOINT_CALIBRATION_CONFIG_PATH = (
    ROBOT_CONTROL_CONFIG_ROOT / "masterarm_joint_calibration.yaml"
).resolve()
MASTERARM_CALIBRATION_WARN_INTERVAL_S = 2.0


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw.strip())
    except ValueError:
        return float(default)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_vector(name: str, expected_size: int) -> np.ndarray | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        values = [float(token.strip()) for token in raw.replace(";", ",").split(",")]
    except Exception:
        logger.warning("[ROSBridge] ignoring invalid %s=%r", name, raw)
        return None
    if len(values) != expected_size:
        logger.warning(
            "[ROSBridge] ignoring %s: expected %d comma-separated values, got %d",
            name,
            int(expected_size),
            int(len(values)),
        )
        return None
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        logger.warning("[ROSBridge] ignoring %s: non-finite value present", name)
        return None
    return arr


def resolve_masterarm_bridge_hz() -> float:
    """Return a bounded latest-sample bridge rate for leader-arm teleoperation."""
    requested = _env_float("IGRIS_MASTERARM_BRIDGE_HZ", MASTERARM_BRIDGE_DEFAULT_HZ)
    return max(20.0, min(500.0, requested))


def _coerce_arm_block(arr, expected_size: int = len(ARM_INDICES)) -> np.ndarray | None:
    try:
        out = np.asarray(arr, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if out.size != expected_size:
        return None
    if not np.all(np.isfinite(out)):
        return None
    return out.copy()


def _coerce_calibration_vector(raw, expected_size: int, default: float) -> np.ndarray:
    if raw is None:
        return np.full(expected_size, float(default), dtype=np.float64)
    try:
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    except Exception:
        return np.full(expected_size, float(default), dtype=np.float64)
    if arr.size != expected_size or not np.all(np.isfinite(arr)):
        return np.full(expected_size, float(default), dtype=np.float64)
    return arr.copy()


def load_masterarm_joint_calibration(expected_size: int = len(ARM_INDICES)) -> dict[str, np.ndarray | bool | str | None]:
    """Load affine leader-joint calibration.

    The leader ROS node publishes angles in robot joint order, but the physical
    Dynamixel zero of the leader is not guaranteed to match the robot's PJS zero.
    This bridge applies a late, transparent affine correction:

        calibrated = raw * scale + offset

    Values can come from config/robot_control/masterarm_joint_calibration.yaml or
    from environment variables for quick field tuning.
    """
    path = os.getenv("IGRIS_MASTERARM_JOINT_CALIBRATION_PATH")
    calib_path = (
        MASTERARM_JOINT_CALIBRATION_CONFIG_PATH
        if path is None or path.strip() == ""
        else os.path.abspath(os.path.expanduser(path.strip()))
    )

    payload = {}
    source: str | None = None
    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            payload = loaded
            source = str(calib_path)
    except FileNotFoundError:
        payload = {}
    except Exception as exc:
        logger.warning("[ROSBridge] failed to load masterarm calibration %s: %s", calib_path, exc)
        payload = {}

    enabled = bool(payload.get("enabled", True))
    scale = _coerce_calibration_vector(payload.get("scale"), expected_size, 1.0)
    offset = _coerce_calibration_vector(payload.get("offset"), expected_size, 0.0)

    env_scale = _env_vector("IGRIS_MASTERARM_ARM_SCALE", expected_size)
    env_offset = _env_vector("IGRIS_MASTERARM_ARM_OFFSET_RAD", expected_size)
    if env_scale is not None:
        scale = env_scale
        source = "env:IGRIS_MASTERARM_ARM_SCALE"
    if env_offset is not None:
        offset = env_offset
        source = (
            "env:IGRIS_MASTERARM_ARM_OFFSET_RAD"
            if source is None
            else f"{source}+IGRIS_MASTERARM_ARM_OFFSET_RAD"
        )

    enabled = _env_flag("IGRIS_MASTERARM_JOINT_CALIBRATION_ENABLED", enabled)
    return {
        "enabled": enabled,
        "scale": scale,
        "offset": offset,
        "source": source,
    }


def extract_obs_arm_hold_target(obs: dict | None, expected_size: int = len(ARM_INDICES)) -> np.ndarray | None:
    if not isinstance(obs, dict):
        return None
    return _coerce_arm_block(obs.get("obs_arm"), expected_size=expected_size)


def resolve_masterarm_arm_target(
    present,
    arm_status: dict[str, float] | None,
    obs: dict | None,
    stale_timeout_s: float,
    expected_size: int = len(ARM_INDICES),
) -> tuple[np.ndarray | None, str | None]:
    hold_arm = extract_obs_arm_hold_target(obs, expected_size=expected_size)
    present_arr = _coerce_arm_block(present, expected_size=expected_size)

    if arm_status is None:
        arm_status = {}
    has_position = bool(float(arm_status.get("has_position", 1.0 if present_arr is not None else 0.0)))
    is_all_zero = bool(float(arm_status.get("is_all_zero", 0.0)))
    try:
        age_s = float(arm_status.get("age_s", 0.0))
    except Exception:
        age_s = 0.0

    if not has_position:
        return hold_arm, MASTERARM_ARM_WAITING_REASON
    if present_arr is None:
        return hold_arm, MASTERARM_ARM_INVALID_REASON
    if is_all_zero or np.linalg.norm(present_arr) < MASTERARM_ARM_ZERO_TOL:
        return hold_arm, MASTERARM_ARM_ALL_ZERO_REASON
    if stale_timeout_s > 0.0 and np.isfinite(age_s) and age_s > stale_timeout_s:
        return hold_arm, MASTERARM_ARM_STALE_REASON
    return present_arr, None


class MasterarmRosridgeWorker(SingleRateWorker):
    """Single-rate 워커 예제: 상태에 따라 카운터를 업데이트."""

    def __init__(self, ctx: WorkerContext, hz: float | None = None) -> None:
        super().__init__(ctx, hz=resolve_masterarm_bridge_hz() if hz is None else float(hz))
        
        global rclpy
        if rclpy is None:
            raise RuntimeError("rclpy is not available. Please source ROS2 and install rclpy.")

        self._shared_memory = ctx.shared_memory
        self._owns_shared_memory = False

        self.teleop_device = ctx.run_config.teleop_device
        self.teleop_hand_source = resolve_teleop_hand_source(
            self.teleop_device,
            ctx.run_config.teleop_hand_source,
        )
        self._direct_masterarm = self.teleop_device == "masterarm"
        self._masterarm_hand_source = self.teleop_hand_source == "masterarm"
        self._direct_body_seeded = False

        self.act_shm = self._shared_memory.get("act_shm")
        self.obs_shm = self._shared_memory.get("obs_shm")
        self.ee_shm = self._shared_memory.get("ee_shm")
        self.iface: MasterArmROSInterface | None = None
        self._spin_thread: threading.Thread | None = None
        self._fk_solver = None
        self._fk_solver_kind = ""
        self._fk_init_failed = False
        self._last_valid_upper_q: np.ndarray | None = None
        self._arm_stale_timeout_s = max(0.0, _env_float("IGRIS_MASTERARM_ARM_STALE_TIMEOUT_S", 0.3))
        self._last_arm_guard_state: str | None = None
        self._joint_calibration = load_masterarm_joint_calibration()
        self._last_calibration_warning_t = 0.0
        self._last_calibration_debug_t = 0.0
        self._suspicious_target_abs_rad = _env_float("IGRIS_MASTERARM_SUSPICIOUS_TARGET_ABS_RAD", 2.8)
        self._calibration_debug_interval_s = _env_float(
            "IGRIS_MASTERARM_CALIBRATION_DEBUG_INTERVAL_S",
            2.0,
        )
        self._capture_joint_calibration_once = _env_flag(
            "IGRIS_MASTERARM_CAPTURE_JOINT_CALIBRATION_ONCE",
            False,
        )
        self._joint_calibration_captured = False
        # Used only before the ROS interface exists or with an older compatible
        # interface that lacks per-sample waiting. Event.wait is a sleeping
        # fallback, never a busy-spin.
        self._watchdog_wait_event = threading.Event()
        calib_enabled = bool(self._joint_calibration.get("enabled", True))
        calib_source = self._joint_calibration.get("source")
        if calib_source:
            logger.info(
                "[ROSBridge] masterarm joint calibration enabled=%s source=%s",
                calib_enabled,
                calib_source,
            )
        else:
            logger.info(
                "[ROSBridge] masterarm joint calibration enabled=%s source=identity "
                "(create %s for absolute leader alignment)",
                calib_enabled,
                MASTERARM_JOINT_CALIBRATION_CONFIG_PATH,
            )
        if self._capture_joint_calibration_once:
            logger.warning(
                "[ROSBridge] one-shot masterarm joint calibration capture is ENABLED. "
                "The next valid obs_arm/raw leader pair will overwrite %s.",
                MASTERARM_JOINT_CALIBRATION_CONFIG_PATH,
            )

    def _spin_ros(self) -> None:
        if self.iface is None:
            return
        try:
            rclpy.spin(self.iface)
        except Exception:
            logger.exception("[ROSBridge] spin failed.")

    def _create_ros_interface(self) -> None:
        logger.info("[ROSBridge] creating MasterArmROSInterface...")
        if self.iface is not None:
            return
        try:
            rclpy.init(args=None)
        except RuntimeError:
            pass

        try:
            self.iface = MasterArmROSInterface()
        except Exception:
            logger.exception("[ROSBridge] failed to create MasterArmROSInterface")
            self.iface = None
            return

        self._spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self._spin_thread.start()

    def _ensure_ros_ready(self, ev: EventSnapshot) -> bool:
        if self.iface is not None:
            return True
        if not ev.level.get("ready", False):
            return False
        self._create_ros_interface()
        return self.iface is not None
                

    def on_start(self) -> None:
        logger.info(f"[{self.ctx.name}] start (single-rate {self.hz} Hz)")

    def _wait_for_arm_sample_or_watchdog(self, after_seq: int, timeout_s: float) -> int:
        """Sleep until a fresh arm sample or the periodic watchdog deadline."""
        iface = self.iface
        if iface is not None and hasattr(iface, "wait_for_arm_update"):
            try:
                return int(iface.wait_for_arm_update(after_seq, timeout=timeout_s))
            except Exception:
                if not self.should_stop():
                    logger.debug(
                        "[ROSBridge] arm update wait failed; using periodic fallback.",
                        exc_info=True,
                    )

        stop_event = self.ctx.stop_event
        if stop_event is not None and hasattr(stop_event, "wait"):
            try:
                stop_event.wait(timeout_s)
            except Exception:
                self._watchdog_wait_event.wait(timeout_s)
        else:
            self._watchdog_wait_event.wait(timeout_s)

        if iface is not None and hasattr(iface, "get_arm_status"):
            try:
                return int(float(iface.get_arm_status().get("seq", after_seq)))
            except Exception:
                pass
        return int(after_seq)

    def run(self) -> None:
        """Process fresh leader samples immediately, with a periodic watchdog."""
        self.on_start()
        if self.hz <= 0.0:
            # Match the validation previously performed by Rate(self.hz).
            raise ValueError("hz must be > 0")
        watchdog_period_s = 1.0 / self.hz
        diagnostics = LoopDiagnostics(ctx=self.ctx, loop="main", target_hz=self.hz)
        observed_arm_seq = 0
        try:
            while True:
                if self.should_stop():
                    break
                loop_start = time.perf_counter()
                ev, tr = self.poll()
                if self.state == ModeState.EXIT:
                    break
                self.step_once(ev, tr)
                diagnostics.observe(start_s=loop_start, end_s=time.perf_counter())
                observed_arm_seq = self._wait_for_arm_sample_or_watchdog(
                    observed_arm_seq,
                    watchdog_period_s,
                )
        finally:
            diagnostics.stop()
            self.on_stop()

    def _seed_direct_body_targets(self) -> bool:
        if not self._direct_masterarm or self._direct_body_seeded:
            return True
        if self.act_shm is None or self.obs_shm is None:
            return False

        try:
            obs = self.obs_shm.read_data()
            act_leg = np.asarray(obs.get("obs_leg"), dtype=float).reshape(-1)
            act_waist = np.asarray(obs.get("obs_waist"), dtype=float).reshape(-1)
            act_neck = np.asarray(obs.get("obs_neck"), dtype=float).reshape(-1)
        except Exception:
            logger.debug("[ROSBridge] Failed to read obs_shm for masterarm seed.", exc_info=True)
            return False

        if act_leg.size != 12 or act_waist.size != 3 or act_neck.size != 2:
            logger.debug(
                "[ROSBridge] Waiting for valid obs seed in masterarm mode (leg=%d waist=%d neck=%d).",
                int(act_leg.size),
                int(act_waist.size),
                int(act_neck.size),
            )
            return False

        try:
            # Direct masterarm bypasses IK, so keep untouched body targets at the current robot pose.
            self.act_shm.write_data(
                act_leg=act_leg,
                act_waist=act_waist,
                act_neck=act_neck,
            )
        except Exception:
            logger.error("[ROSBridge] Failed to seed body targets for masterarm.", exc_info=True)
            return False

        self._direct_body_seeded = True
        logger.info("[ROSBridge] masterarm mode seeded act_leg/act_waist/act_neck from obs_shm")
        return True

    def _ensure_fk_solver(self):
        if not self._direct_masterarm or self.ee_shm is None:
            return None
        if self._fk_solver is not None:
            return self._fk_solver
        if self._fk_init_failed:
            return None
        try:
            from ..robot_control.kinematics.fk.upper_body_fk import IGRISUpperBodyFK

            self._fk_solver = IGRISUpperBodyFK()
            self._fk_solver_kind = "urdf_numpy"
            logger.info("[ROSBridge] Using lightweight URDF FK for masterarm ee_pose publishing.")
            return self._fk_solver
        except Exception:
            logger.exception("[ROSBridge] Failed to initialize lightweight FK solver for masterarm ee_pose publishing.")

        try:
            from ..robot_control.kinematics.ik.prox_ik_pelvis_env import IGRIS_C_UpperIK, IKConfig

            self._fk_solver = IGRIS_C_UpperIK(cfg=IKConfig.from_sources(profile=self.teleop_device))
            self._fk_solver_kind = "igris_upper_ik"
        except Exception:
            self._fk_init_failed = True
            logger.exception("[ROSBridge] Failed to initialize FK solver for masterarm ee_pose publishing.")
            return None
        return self._fk_solver

    @staticmethod
    def _is_finite_joint_block(arr: np.ndarray | None, expected_size: int) -> bool:
        if arr is None:
            return False
        if arr.shape[0] != expected_size:
            return False
        if not np.all(np.isfinite(arr)):
            return False
        return True

    @staticmethod
    def _has_joint_signal(arr: np.ndarray | None) -> bool:
        if arr is None:
            return False
        return bool(np.linalg.norm(arr) >= 1e-6)

    def _compose_upper_q(self, obs: dict, arm_q: np.ndarray | None) -> np.ndarray | None:
        try:
            waist = np.asarray(obs.get("obs_waist"), dtype=np.float64).reshape(-1)
            obs_arm = np.asarray(obs.get("obs_arm"), dtype=np.float64).reshape(-1)
            neck = np.asarray(obs.get("obs_neck"), dtype=np.float64).reshape(-1)
        except Exception:
            return self._last_valid_upper_q.copy() if self._last_valid_upper_q is not None else None

        if not self._is_finite_joint_block(waist, len(WAIST_INDICES)):
            return self._last_valid_upper_q.copy() if self._last_valid_upper_q is not None else None
        if not self._is_finite_joint_block(neck, len(NECK_INDICES)):
            return self._last_valid_upper_q.copy() if self._last_valid_upper_q is not None else None

        if self._is_finite_joint_block(arm_q, len(ARM_INDICES)) and self._has_joint_signal(arm_q):
            arm = arm_q
        else:
            arm = obs_arm

        if not self._is_finite_joint_block(arm, len(ARM_INDICES)):
            return self._last_valid_upper_q.copy() if self._last_valid_upper_q is not None else None

        return np.concatenate((waist, arm, neck), axis=0)

    def _publish_direct_masterarm_ee(self, obs: dict, arm_q: np.ndarray | None) -> None:
        solver = self._ensure_fk_solver()
        if solver is None or self.ee_shm is None:
            return

        q = self._compose_upper_q(obs, arm_q)
        if q is None:
            return

        try:
            q = np.asarray(q, dtype=np.float64).reshape(-1)
            if hasattr(solver, "get_ee_pose_mats"):
                l_mat, r_mat, h_mat = solver.get_ee_pose_mats(q)
            else:
                q = np.asarray(solver._clip_q_to_limits(q), dtype=np.float64).reshape(-1)
                l_pose, r_pose, h_pose = solver.get_ee_poses(q)
                l_mat = np.asarray(l_pose.homogeneous, dtype=np.float64)
                r_mat = np.asarray(r_pose.homogeneous, dtype=np.float64)
                h_mat = np.asarray(h_pose.homogeneous, dtype=np.float64)
        except Exception:
            logger.debug("[ROSBridge] Failed to compute ee_pose from masterarm FK.", exc_info=True)
            return

        try:
            self.ee_shm.write_data(
                left_wrist_mat=np.asarray(l_mat, dtype=np.float64),
                right_wrist_mat=np.asarray(r_mat, dtype=np.float64),
                head_mat=np.asarray(h_mat, dtype=np.float64),
            )
            self._last_valid_upper_q = q.copy()
        except Exception:
            logger.debug("[ROSBridge] Failed to write ee_pose shared memory in masterarm mode.", exc_info=True)

    def _log_arm_guard_state(self, reason: str | None, has_hold_target: bool) -> None:
        state = "ok" if reason is None else (reason if has_hold_target else f"{reason}:no_hold")
        prev_state = self._last_arm_guard_state
        if state == self._last_arm_guard_state:
            return
        self._last_arm_guard_state = state

        if reason is None:
            if prev_state not in {None, "ok"}:
                logger.info("[ROSBridge] masterarm arm input recovered; following leader arm again.")
            return

        reason_messages = {
            MASTERARM_ARM_WAITING_REASON: "masterarm arm input not received yet",
            MASTERARM_ARM_INVALID_REASON: "masterarm arm input invalid",
            MASTERARM_ARM_ALL_ZERO_REASON: "masterarm arm input is all zero",
            MASTERARM_ARM_STALE_REASON: (
                f"masterarm arm input stale (> {self._arm_stale_timeout_s:.2f}s without fresh sample)"
            ),
        }
        base = reason_messages.get(reason, f"masterarm arm input fallback ({reason})")
        if has_hold_target:
            logger.warning("[ROSBridge] %s -> holding current robot arm pose from obs_arm.", base)
        else:
            logger.warning("[ROSBridge] %s -> obs_arm unavailable, skipping act_arm update.", base)

    def _apply_joint_calibration(self, raw_arm_target: np.ndarray) -> np.ndarray:
        arm = np.asarray(raw_arm_target, dtype=np.float64).reshape(-1)
        if arm.size != len(ARM_INDICES):
            return arm.copy()
        if not bool(self._joint_calibration.get("enabled", True)):
            return arm.copy()
        scale = np.asarray(self._joint_calibration.get("scale"), dtype=np.float64).reshape(-1)
        offset = np.asarray(self._joint_calibration.get("offset"), dtype=np.float64).reshape(-1)
        if scale.size != arm.size or offset.size != arm.size:
            return arm.copy()
        calibrated = arm * scale + offset
        return calibrated.astype(np.float64, copy=False)

    def _warn_if_suspicious_masterarm_target(
        self,
        raw_arm_target: np.ndarray,
        calibrated_arm_target: np.ndarray,
    ) -> None:
        threshold = float(self._suspicious_target_abs_rad)
        if threshold <= 0.0:
            return
        try:
            raw = np.asarray(raw_arm_target, dtype=np.float64).reshape(-1)
            calibrated = np.asarray(calibrated_arm_target, dtype=np.float64).reshape(-1)
        except Exception:
            return
        if raw.size != len(JOINT_NAMES) or calibrated.size != len(JOINT_NAMES):
            return
        max_abs = float(np.max(np.abs(calibrated)))
        if max_abs < threshold:
            return
        now = time.monotonic()
        if now - self._last_calibration_warning_t < MASTERARM_CALIBRATION_WARN_INTERVAL_S:
            return
        self._last_calibration_warning_t = now
        worst = int(np.argmax(np.abs(calibrated)))
        logger.warning(
            "[ROSBridge] suspicious masterarm target after calibration: %s=%.3f rad "
            "(raw=%.3f rad, threshold=%.3f). If the physical leader pose does not match, "
            "set offsets in %s or IGRIS_MASTERARM_ARM_OFFSET_RAD.",
            JOINT_NAMES[worst],
            float(calibrated[worst]),
            float(raw[worst]),
            threshold,
            MASTERARM_JOINT_CALIBRATION_CONFIG_PATH,
        )

    def _log_masterarm_calibration_debug(
        self,
        raw_arm_target: np.ndarray,
        calibrated_arm_target: np.ndarray,
    ) -> None:
        interval = float(self._calibration_debug_interval_s)
        if interval <= 0.0:
            return
        now = time.monotonic()
        if now - self._last_calibration_debug_t < interval:
            return
        self._last_calibration_debug_t = now
        try:
            raw = np.asarray(raw_arm_target, dtype=np.float64).reshape(-1)
            calibrated = np.asarray(calibrated_arm_target, dtype=np.float64).reshape(-1)
        except Exception:
            return
        if raw.size != len(JOINT_NAMES) or calibrated.size != len(JOINT_NAMES):
            return

        tokens = []
        for idx, name in enumerate(JOINT_NAMES):
            short = (
                name.replace("Shoulder_", "Sh_")
                .replace("Elbow_", "El_")
                .replace("Wrist_", "Wr_")
                .replace("Pitch", "P")
                .replace("Roll", "R")
                .replace("Yaw", "Y")
            )
            tokens.append(f"{idx}:{short} raw={raw[idx]:+.3f} cal={calibrated[idx]:+.3f}")
        logger.info("[ROSBridge] masterarm calibration debug | %s", " | ".join(tokens))

    def _maybe_capture_joint_calibration(self, obs: dict, raw_arm_target: np.ndarray) -> None:
        if not self._capture_joint_calibration_once or self._joint_calibration_captured:
            return
        obs_arm = extract_obs_arm_hold_target(obs)
        raw = _coerce_arm_block(raw_arm_target)
        if obs_arm is None or raw is None:
            return
        if np.linalg.norm(raw) < MASTERARM_ARM_ZERO_TOL:
            return

        scale = np.ones(len(ARM_INDICES), dtype=np.float64)
        offset = obs_arm - raw
        payload = {
            "enabled": True,
            "scale": [float(v) for v in scale],
            "offset": [float(v) for v in offset],
            "joint_names": list(JOINT_NAMES),
            "captured_unix_time": float(time.time()),
            "capture_rule": "offset = current obs_arm - current raw_leader_arm",
        }
        try:
            MASTERARM_JOINT_CALIBRATION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = MASTERARM_JOINT_CALIBRATION_CONFIG_PATH.with_suffix(
                MASTERARM_JOINT_CALIBRATION_CONFIG_PATH.suffix + ".tmp"
            )
            with tmp_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
            tmp_path.replace(MASTERARM_JOINT_CALIBRATION_CONFIG_PATH)
        except Exception:
            logger.exception("[ROSBridge] failed to capture masterarm joint calibration")
            self._joint_calibration_captured = True
            return

        self._joint_calibration = {
            "enabled": True,
            "scale": scale,
            "offset": offset,
            "source": str(MASTERARM_JOINT_CALIBRATION_CONFIG_PATH),
        }
        self._joint_calibration_captured = True
        logger.warning(
            "[ROSBridge] captured masterarm joint calibration to %s. "
            "Restart is not required; applying captured offset immediately.",
            MASTERARM_JOINT_CALIBRATION_CONFIG_PATH,
        )

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        if not self._ensure_ros_ready(ev):
            return

        # present_position 콜백에서 act_shm 갱신
        if self.iface is None or self.act_shm is None:
            return
        if not self._seed_direct_body_targets():
            return

        try:
            obs = self.obs_shm.read_data() if self.obs_shm is not None else {}
        except Exception:
            logger.debug("[ROSBridge] Failed to read obs_shm for masterarm bridge.", exc_info=True)
            obs = {}

        present = None
        arm_status = None
        try:
            if hasattr(self.iface, "get_arm_snapshot"):
                present, arm_status = self.iface.get_arm_snapshot()
            else:
                present = self.iface.get_present_position()
            if arm_status is None and hasattr(self.iface, "get_arm_status"):
                arm_status = self.iface.get_arm_status()
        except Exception:
            logger.debug("[ROSBridge] Failed to read leader arm position from ROS interface.", exc_info=True)

        arm_target, arm_guard_reason = resolve_masterarm_arm_target(
            present,
            arm_status,
            obs,
            stale_timeout_s=self._arm_stale_timeout_s,
        )
        self._log_arm_guard_state(arm_guard_reason, has_hold_target=arm_target is not None)
        if arm_target is None:
            return

        raw_arm_target = np.asarray(arm_target, dtype=np.float64).reshape(-1)
        if arm_guard_reason is None:
            # Fresh leader samples are in master-arm coordinates and must be
            # converted into robot PJS coordinates before ControlWorker follows.
            self._maybe_capture_joint_calibration(obs, raw_arm_target)
            arm_target = self._apply_joint_calibration(raw_arm_target)
            self._log_masterarm_calibration_debug(raw_arm_target, arm_target)
            self._warn_if_suspicious_masterarm_target(raw_arm_target, arm_target)
        else:
            # Guard fallback uses obs_arm as a hold target. obs_arm is already
            # a robot PJS vector, so applying master-arm calibration here would
            # corrupt the safe hold pose whenever leader samples are absent,
            # all-zero, invalid, or stale.
            arm_target = raw_arm_target

        try:
            self.act_shm.write_data(act_arm=np.asarray(arm_target, dtype=float))
        except Exception:
            logger.error("[ROSBridge] Failed to write act_arm.", exc_info=True)
            return

        if self._direct_masterarm:
            self._publish_direct_masterarm_ee(obs, arm_target)

        if self._masterarm_hand_source:
            try:
                present_hand = compress_normalized_hand_command(
                    self.iface.get_act_hand_12(),
                    dtype=np.float64,
                )
                self.act_shm.write_data(act_hand=np.asarray(present_hand, dtype=float))
            except Exception:
                logger.error("[ROSBridge] Failed to write act_hand.", exc_info=True)
    


        
    def on_stop(self) -> None:
        if self._shared_memory:
            for key, mgr in self._shared_memory.items():
                try:
                    mgr.worker_close()
                except Exception:
                    logger.exception(f"[{self.ctx.name}] failed to close shared memory {key}")

        if self.iface is not None:
            try:
                self.iface.destroy_node()
            except Exception:
                logger.exception("[ROSBridge] failed to destroy ROS node.")

        try:
            # Avoid noisy shutdown errors when ROS was never initialized
            # (e.g., user quits before "ready" state).
            if hasattr(rclpy, "try_shutdown"):
                rclpy.try_shutdown()
            elif rclpy.ok():
                rclpy.shutdown()
        except Exception:
            logger.exception("[ROSBridge] failed to shutdown rclpy.")

        if getattr(self, "_spin_thread", None):
            self._spin_thread.join(timeout=1.0)

        logger.info(f"[{self.ctx.name}] stop")
