from __future__ import annotations

import threading
import time

import numpy as np
import igris_c_sdk as igc_sdk
import yaml

import logging_mp
from ..core.events import EventSnapshot
from ..core.state_machine import ModeState, TransitionResult
from ..core.worker_base import DualRateWorker, WorkerContext, resolve_robot_dds_domain_id


# from ..robot_control.controller.igris_controller import BaseController

from ..robot_control.controller.igris_controller import BaseController
from ..robot_control.controller.core import load_joint_profile
from ..robot_control.kinematics.joints import (
    ARM_INDICES,
    JointIndex,
    LEG_INDICES,
    NECK_INDICES,
    NUM_MOTORS,
    WAIST_INDICES,
)
from ..core.project_paths import INIT_SETTING_PATH, ROBOT_CONTROL_LOGS_ROOT, WALKING_JOINT_SETTING_PATH
from ..policies.walking import (
    DEFAULT_WALKING_POLICY_PROFILE,
    WALKING_PROFILE_CHOICES,
    WALKING_MODE_NAME,
    WALKING_NEUTRAL_FULL_Q,
    WALKING_NEUTRAL_HAND_Q,
    WALKING_PREP_ARM_NECK_DURATION,
    WALKING_PREP_LEG_WAIST_DURATION,
    WALKING_STANCE_NAME,
    default_walking_joint_profile_path,
    resolve_walking_policy_profile_code,
)

logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)
HAND_TARGET_DIM = 12
HAND_MOVE_DURATION = 2.0
HAND_MOVE_HZ = 50.0
DEFAULT_HAND_TARGET = np.zeros(HAND_TARGET_DIM, dtype=np.float64)
CONTROL_STOP_SERVICE_TIMEOUT_MS = 5000
CONTROL_STOP_SERVICE_CALL_TIMEOUT_S = 12.0
CONTROL_STOP_THREAD_JOIN_TIMEOUT_S = 5.0
CONTROL_ABORT_SERVICE_TIMEOUT_MS = 2000
CONTROL_ABORT_SERVICE_CALL_TIMEOUT_S = 6.0
CONTROL_ABORT_THREAD_JOIN_TIMEOUT_S = 2.0
CONTROL_MS_GUARD_STATE_TIMEOUT_S = 5.0
DEFAULT_PR2AB_LOG_PATH = ROBOT_CONTROL_LOGS_ROOT / "pr2ab_calibration.yaml"
SIM_RUNTIME_ENVIRONMENTS = frozenset({"sim", "simulator", "simulation", "mujoco"})
INIT_POSE_JOINT_INDEX = {
    "waist_yaw": int(JointIndex.WAIST_YAW),
    "waist_roll": int(JointIndex.WAIST_ROLL),
    "waist_pitch": int(JointIndex.WAIST_PITCH),
    "l_shoulder_pitch": int(JointIndex.L_SHOULDER_PITCH),
    "l_shoulder_roll": int(JointIndex.L_SHOULDER_ROLL),
    "l_shoulder_yaw": int(JointIndex.L_SHOULDER_YAW),
    "l_elbow_pitch": int(JointIndex.L_ELBOW_PITCH),
    "l_wrist_yaw": int(JointIndex.L_WRIST_YAW),
    "l_wrist_roll": int(JointIndex.L_WRIST_ROLL),
    "l_wrist_pitch": int(JointIndex.L_WRIST_PITCH),
    "r_shoulder_pitch": int(JointIndex.R_SHOULDER_PITCH),
    "r_shoulder_roll": int(JointIndex.R_SHOULDER_ROLL),
    "r_shoulder_yaw": int(JointIndex.R_SHOULDER_YAW),
    "r_elbow_pitch": int(JointIndex.R_ELBOW_PITCH),
    "r_wrist_yaw": int(JointIndex.R_WRIST_YAW),
    "r_wrist_roll": int(JointIndex.R_WRIST_ROLL),
    "r_wrist_pitch": int(JointIndex.R_WRIST_PITCH),
    "neck_yaw": int(JointIndex.NECK_YAW),
    "neck_pitch": int(JointIndex.NECK_PITCH),
}
INIT_POSE_HAND_INDEX = {f"hand_{idx}": idx for idx in range(HAND_TARGET_DIM)}


def _env_float(name: str, default: float) -> float:
    raw = None
    try:
        import os

        raw = os.getenv(name)
    except Exception:
        raw = None
    if raw is None:
        return float(default)
    try:
        value = float(str(raw).strip())
    except Exception:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(value)


def resolve_control_kinematic_mode(runtime_environment: str | None):
    value = str(runtime_environment or "").strip().lower()
    if value in SIM_RUNTIME_ENVIRONMENTS:
        return igc_sdk.KinematicMode.PJS
    return igc_sdk.KinematicMode.MS


class ControlWorker(DualRateWorker):
    """Dual-rate 워커 예제:
    - slow: 관측(telemetry) 업데이트
    - fast: 제어(명령) 업데이트
    """

    def __init__(self, ctx: WorkerContext, slow_hz: float = 100.0, fast_hz: float = 100.0) -> None:
        super().__init__(ctx, slow_hz=slow_hz, fast_hz=fast_hz)
        self.thread_join_timeout_s = _env_float("IGRIS_CONTROL_WORKER_THREAD_JOIN_TIMEOUT_S", 30.0)

        self._shared_memory = ctx.shared_memory
        self._owns_shared_memory = False
        self._homing_active = False
        self._home_request = False
        self._ctrl_init_lock = threading.Lock()
        self._body_motion_active = threading.Event()
        self._body_motion_last_telemetry_t: float | None = None
        self.ctrl: BaseController | None = None
        runtime_environment = getattr(ctx.run_config, "runtime_environment", None)
        self._body_domain_id = resolve_robot_dds_domain_id(runtime_environment)
        self._controller_init_blocked_reason: str | None = None
        
        self.obs_shm = self._shared_memory.get("obs_shm")
        self.act_shm = self._shared_memory.get("act_shm")
        self.tau_shm = self._shared_memory.get("tau_shm")
        self.walking_debug_shm = self._shared_memory.get("walking_debug_shm")
        self.walking_cmd_shm = self._shared_memory.get("walking_cmd_shm")
        self.mode_shm = self._shared_memory.get("mode_shm")
        self.pose_request_shm = self._shared_memory.get("pose_request_shm")
        self.ee_shm = self._shared_memory.get("ee_shm")
        self.ik_target_shm = self._shared_memory.get("ik_target_shm")

        walking_kp, walking_kd, _walking_q, _walking_wp1, _walking_wp2 = load_joint_profile(
            WALKING_JOINT_SETTING_PATH
        )
        self._walking_kp = np.asarray(walking_kp, dtype=np.float32).reshape(-1)
        self._walking_kd = np.asarray(walking_kd, dtype=np.float32).reshape(-1)
        self._walking_gain_profiles = self._load_walking_gain_profiles()
        self._walking_pose_q = np.asarray(WALKING_NEUTRAL_FULL_Q, dtype=np.float32).reshape(-1)
        self._walking_profile_active = False
        self._walking_profile_active_name: str | None = None
        self._walking_pose_pending = False
        self._prev_walking_mode = False

        # MuJoCo exposes PJS joints directly and must not depend on a real-robot
        # motor-to-joint calibration file. Hardware keeps the calibrated MS path.
        self._kinematic_mode = resolve_control_kinematic_mode(runtime_environment)
        self._use_motor_state = False
        # Teleop entry gate:
        # - leader/masterarm arm source: wait until each arm joint is close.
        # - VR-only arm source: wait until commanded EE poses are close to robot EE poses.
        # Set the thresholds to <0 to disable the corresponding gate.
        arm_gate_error = _env_float("IGRIS_TELEOP_ENTRY_ARM_JOINT_ERR_RAD", 0.2)
        self._arm_target_max_error: float | None = arm_gate_error if arm_gate_error >= 0.0 else None
        ee_pos_gate_error = _env_float("IGRIS_TELEOP_ENTRY_EE_POS_ERR_M", 0.06)
        self._ee_target_max_pos_error: float | None = (
            ee_pos_gate_error if ee_pos_gate_error >= 0.0 else None
        )
        ee_rot_gate_error_deg = _env_float("IGRIS_TELEOP_ENTRY_EE_ROT_ERR_DEG", 20.0)
        self._ee_target_max_rot_error: float | None = (
            np.deg2rad(ee_rot_gate_error_deg) if ee_rot_gate_error_deg >= 0.0 else None
        )
        self._arm_gate_passed = False
        # Gate smoothing (receding-horizon interpolation) parameters
        # duration = clamp(min, max, gain * max_error)
        self._gate_interp_duration_max: float = _env_float("IGRIS_TELEOP_ENTRY_RAMP_MAX_S", 5.0)
        self._gate_interp_duration_min: float = _env_float("IGRIS_TELEOP_ENTRY_RAMP_MIN_S", 0.8)
        self._gate_interp_duration_gain: float = _env_float("IGRIS_TELEOP_ENTRY_RAMP_GAIN_S_PER_RAD", 2.0)
        # If seed (_target_q) is too far from current, fall back to current.
        self._gate_seed_max_delta: float = 0.5  # rad
        self._gate_cmd_q: np.ndarray | None = None
        # Force-pass if error doesn't go below threshold within this time (seconds).
        # Set to None to disable force-pass behavior.
        force_pass_timeout = _env_float("IGRIS_TELEOP_ENTRY_FORCE_PASS_S", -1.0)
        self._gate_force_pass_timeout: float | None = (
            force_pass_timeout if force_pass_timeout > 0.0 else None
        )
        # Target stillness threshold for force-pass eligibility (rad/s).
        self._gate_target_vel_th: float = 0.1
        self._gate_target_vel_ema_alpha: float = 0.2
        self._gate_target_vel_ema: float | None = None
        self._gate_target_vel_eps: float = 1e-3
        # If act_shm suddenly changes by several tenths of a radian after the
        # entry gate already passed, re-arm the gate. This catches the common
        # case where the bridge holds obs_arm before the first leader sample,
        # then receives a real leader target after the gate has already passed.
        self._gate_rearm_target_jump_rad: float = _env_float(
            "IGRIS_TELEOP_ENTRY_REARM_TARGET_JUMP_RAD",
            0.35,
        )
        self._gate_min_error: float | None = None
        self._gate_last_improve_time: float | None = None
        self._gate_target_last_q: np.ndarray | None = None
        self._gate_target_last_t: float | None = None
        self._gate_last_step_t: float | None = None
        self._entry_last_target_sub: np.ndarray | None = None
        self._entry_target_seq_at_start: float | None = None
        # Periodic gate status log (seconds). Set to None to disable.
        self._gate_status_log_interval: float | None = 1.0
        self._gate_status_last_log_time: float | None = None
        self._gate_status_last_value: bool | None = None
        # Gate interpolation indices. The pass condition is selected separately:
        # leader uses arm joint error; VR-only uses EE pose error.
        seen = set()
        gate_indices: list[int] = []
        for group in (WAIST_INDICES, LEG_INDICES, ARM_INDICES, NECK_INDICES):
            for idx in group:
                idx = int(idx)
                if idx in seen:
                    continue
                seen.add(idx)
                gate_indices.append(idx)
        self._gate_indices: tuple[int, ...] = tuple(gate_indices)
        self._leader_gate_indices: tuple[int, ...] = tuple(int(idx) for idx in ARM_INDICES)
        self._vr_gate_indices: tuple[int, ...] = tuple(
            int(idx)
            for group in (WAIST_INDICES, ARM_INDICES, NECK_INDICES)
            for idx in group
        )
        # Passing the gate and interpolating the takeover are separate jobs.
        # Leader pass/error is arm-only, while every teleop source must ramp the
        # complete commanded upper body so waist/neck never bypass Start.
        self._entry_ramp_indices: tuple[int, ...] = self._vr_gate_indices
        self._teleop_cmd_q: np.ndarray | None = None
        self._teleop_cmd_dq = np.zeros(NUM_MOTORS, dtype=np.float64)
        self._teleop_limiter_last_t: float | None = None
        self._teleop_velocity_limit = np.full(NUM_MOTORS, np.inf, dtype=np.float64)
        self._teleop_acceleration_limit = np.full(NUM_MOTORS, np.inf, dtype=np.float64)
        self._teleop_command_lead_limit = np.full(NUM_MOTORS, np.inf, dtype=np.float64)
        for indices, velocity, acceleration, lead in (
            (
                WAIST_INDICES,
                _env_float("IGRIS_TELEOP_WAIST_MAX_VEL_RAD_S", 1.0),
                _env_float("IGRIS_TELEOP_WAIST_MAX_ACCEL_RAD_S2", 3.0),
                _env_float("IGRIS_TELEOP_WAIST_MAX_LEAD_RAD", 0.15),
            ),
            (
                ARM_INDICES,
                _env_float("IGRIS_TELEOP_ARM_MAX_VEL_RAD_S", 2.0),
                _env_float("IGRIS_TELEOP_ARM_MAX_ACCEL_RAD_S2", 6.0),
                _env_float("IGRIS_TELEOP_ARM_MAX_LEAD_RAD", 0.25),
            ),
            (
                NECK_INDICES,
                _env_float("IGRIS_TELEOP_NECK_MAX_VEL_RAD_S", 1.5),
                _env_float("IGRIS_TELEOP_NECK_MAX_ACCEL_RAD_S2", 5.0),
                _env_float("IGRIS_TELEOP_NECK_MAX_LEAD_RAD", 0.20),
            ),
        ):
            active = list(indices)
            self._teleop_velocity_limit[active] = max(0.01, float(velocity))
            self._teleop_acceleration_limit[active] = max(0.01, float(acceleration))
            self._teleop_command_lead_limit[active] = max(0.01, float(lead))
        # Seed act_shm on startup when targets are still zero.
        self._act_shm_seeded: bool = False
        self._act_shm_seed_tol: float = 1e-6
        # teleop 모드에서만 gate를 적용하기 위한 상태
        self._prev_teleop_mode: bool = False
        self._teleop_fixed_leg_q: np.ndarray | None = None
        self._last_pose_request_seq: int = 0
        self._pending_init_pose_dataset_key: str | None = None
        self._obs_seq: float = 0.0


    def on_start(self) -> None:
        logger.info(
            "[%s] start (dual-rate slow=%.1fHz fast=%.1fHz runtime=%s domain_id=%d kinematic_mode=%s)",
            self.ctx.name,
            self.slow_hz,
            self.fast_hz,
            getattr(self.ctx.run_config, "runtime_environment", None),
            self._body_domain_id,
            self._kinematic_mode,
        )

    def _load_walking_gain_profiles(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for profile in WALKING_PROFILE_CHOICES:
            kp, kd, _default_q, _waypoint_1, _waypoint_2 = load_joint_profile(
                default_walking_joint_profile_path(profile)
            )
            profiles[profile] = (
                np.asarray(kp, dtype=np.float32).reshape(-1),
                np.asarray(kd, dtype=np.float32).reshape(-1),
            )
        return profiles

    def _read_hand_target_from_shm(self, shm_mgr, field_name: str) -> np.ndarray | None:
        if shm_mgr is None:
            return None
        try:
            data = shm_mgr.read_data()
            target = np.asarray(data.get(field_name), dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if target.size != HAND_TARGET_DIM:
            return None
        return target.copy()

    def _resolve_hand_base_target(self) -> np.ndarray:
        hand_target = self._read_hand_target_from_shm(self.obs_shm, "obs_hand")
        if hand_target is not None:
            return hand_target
        hand_target = self._read_hand_target_from_shm(self.act_shm, "act_hand")
        if hand_target is not None:
            return hand_target
        return DEFAULT_HAND_TARGET.copy()

    def _write_hand_target(self, hand_target: np.ndarray) -> bool:
        if self.act_shm is None:
            return False
        target = np.asarray(hand_target, dtype=np.float64).reshape(-1)
        if target.size != HAND_TARGET_DIM:
            logger.warning(
                "[ControlWorker] invalid hand target shape %s, expected (%d,)",
                target.shape,
                HAND_TARGET_DIM,
            )
            return False
        try:
            self.act_shm.write_data(act_hand=target)
        except Exception as exc:
            logger.warning("[ControlWorker] failed to write act_hand target: %s", exc)
            return False
        return True

    def _interpolate_hand_target(
        self,
        hand_target: np.ndarray,
        duration: float = HAND_MOVE_DURATION,
        hz: float = HAND_MOVE_HZ,
    ) -> None:
        goal = np.asarray(hand_target, dtype=np.float64).reshape(-1)
        if goal.size != HAND_TARGET_DIM:
            logger.warning(
                "[ControlWorker] invalid interpolated hand target shape %s, expected (%d,)",
                goal.shape,
                HAND_TARGET_DIM,
            )
            return

        if duration <= 0.0 or hz <= 0.0:
            self._write_hand_target(goal)
            return

        start = self._resolve_hand_base_target()
        steps = max(1, int(round(duration * hz)))
        period = duration / float(steps)
        for step in range(1, steps + 1):
            alpha = step / steps
            target = (1.0 - alpha) * start + alpha * goal
            if not self._write_hand_target(target):
                return
            time.sleep(period)

    def _run_body_motion_with_hand(self, body_motion, hand_target: np.ndarray | None):
        hand_goal = None
        hand_thread: threading.Thread | None = None
        if hand_target is not None:
            hand_goal = np.asarray(hand_target, dtype=np.float64).reshape(-1)
            if hand_goal.size != HAND_TARGET_DIM:
                logger.warning(
                    "[ControlWorker] invalid hand goal shape %s, expected (%d,)",
                    hand_goal.shape,
                    HAND_TARGET_DIM,
                )
                hand_goal = None
            elif self.act_shm is not None:
                hand_thread = threading.Thread(
                    target=self._interpolate_hand_target,
                    args=(hand_goal, HAND_MOVE_DURATION, HAND_MOVE_HZ),
                    daemon=True,
                )
                hand_thread.start()

        self._body_motion_active.set()
        try:
            return body_motion()
        finally:
            if hand_thread is not None:
                hand_thread.join()
            if hand_goal is not None:
                self._write_hand_target(hand_goal)
            self._body_motion_active.clear()

    def _execute_default_pose_sequence(self, body_motion) -> bool:
        if self.ctrl is None:
            return False
        hand_target = DEFAULT_HAND_TARGET.copy()
        completed = self._run_body_motion_with_hand(body_motion, hand_target)
        if completed is False:
            logger.warning(
                "[ControlWorker] default pose trajectory did not complete; "
                "holding the last bounded command"
            )
            return False
        self.ctrl.default_pos_state("default_pos")
        self._sync_action_shm_with_goal(
            np.asarray(self.ctrl._default_q, dtype=np.float64).reshape(-1).copy(),
            hand_target=hand_target,
        )
        return True

    def _move_through_default_pose_sequence(self) -> bool:
        if self.ctrl is None:
            return False
        return self.ctrl.move_through_poses(
            [
                ("zero_pos", 2.0),
                ("waypoint_1", 2.0),
                ("waypoint_2", 2.0),
                ("waypoint_3", 2.0),
                ("default_pos", 3.0),
            ],
            arrival_tolerance_rad=0.06,
            arrival_timeout_s=8.0,
            arrival_velocity_tolerance_rad_s=0.15,
            arrival_settle_time_s=0.20,
            arrival_joint_ids=tuple(WAIST_INDICES) + tuple(ARM_INDICES),
            use_publisher_timing=True,
            start_from_measured=True,
            # Neck pitch feedback retains a hardware zero/sign residual on the
            # real robot. Keep commanding the neck, but do not let that one
            # residual freeze or falsely fail the arm/waist Ready path.
            tracking_joint_ids=tuple(WAIST_INDICES) + tuple(ARM_INDICES),
            tracking_error_soft_rad=0.08,
            tracking_error_hard_rad=0.25,
            trajectory_timeout_s=55.0,
            cancel_event=self.ctx.stop_event,
        )

    def _move_directly_to_default_pose(self) -> bool:
        """HOME is a direct, bounded current-to-default transition.

        The multi-waypoint sequence belongs to initial Ready only. Reusing it
        for HOME made a Home request first travel toward zero and visually
        resemble the reverse shutdown route.
        """
        if self.ctrl is None:
            return False
        return self.ctrl.move_to_pose(
            "default_pos",
            duration=3.0,
            arrival_tolerance_rad=0.06,
            arrival_timeout_s=8.0,
            arrival_velocity_tolerance_rad_s=0.15,
            arrival_settle_time_s=0.20,
            arrival_joint_ids=tuple(WAIST_INDICES) + tuple(ARM_INDICES),
            use_publisher_timing=True,
            start_from_measured=True,
            tracking_joint_ids=tuple(WAIST_INDICES) + tuple(ARM_INDICES),
            tracking_error_soft_rad=0.08,
            tracking_error_hard_rad=0.25,
            trajectory_timeout_s=24.0,
            cancel_event=self.ctx.stop_event,
        )


    def _go_home(self) -> None:
        if self._homing_active:
            return
        self._homing_active = True
        try:
            logger.info("[ControlWorker] go_home start")
            if self.ctrl is None:
                return
            if self._is_walking_mode():
                self._apply_walking_profile()
                self._move_to_walking_stance()
                logger.info("[ControlWorker] go_home complete, holding walking stance")
            else:
                if self._execute_default_pose_sequence(self._move_directly_to_default_pose):
                    logger.info("[ControlWorker] go_home complete, holding default pose")
        except Exception:
            logger.exception("[ControlWorker] go_home failed")
        finally:
            self._homing_active = False

    def _wait_ctrl_ready_and_home(self) -> None:
        """
        첫 LowState를 받을 때까지 기다린 뒤 초기 제어 상태를 준비합니다.
        """
        if self.ctrl is None:
            return
        if self.ctrl.wait_for_state(timeout=None):
            logger.info("[ControlWorker] First LowState received.")
            try:
                if self._is_walking_mode():
                    self._apply_walking_profile()
                    logger.info("[ControlWorker] Walking mode ready. Waiting for start to move to walking stance.")
                else:
                    if self._execute_default_pose_sequence(self._move_through_default_pose_sequence):
                        logger.info("[ControlWorker] Default pose reached and held.")
            except Exception as exc:
                logger.warning("[ControlWorker] initial pose sequence failed: %s", exc)

    def _create_controller(self) -> None:
        # DualRateWorker는 slow/fast 2개 스레드가 동시에 _ensure_controller_ready()
        # 를 호출할 수 있으므로, 컨트롤러 생성은 반드시 한 번만 되도록 락으로 보호한다.
        with self._ctrl_init_lock:
            if self.ctrl is not None or self._controller_init_blocked_reason is not None:
                return
            logger.info("[ControlWorker] initializing BaseController...")
            ctrl: BaseController | None = None
            try:
                if (
                    self._kinematic_mode == igc_sdk.KinematicMode.MS
                    and not DEFAULT_PR2AB_LOG_PATH.is_file()
                ):
                    raise RuntimeError(
                        "MS start blocked: PR2AB calibration file not found at "
                        f"{DEFAULT_PR2AB_LOG_PATH}"
                    )

                ctrl = BaseController(
                    domain_id=self._body_domain_id,
                    control_hz=300.0,
                    kinematic_mode=self._kinematic_mode,
                    use_motor_state=self._use_motor_state,
                    auto_service_init=True,
                    pr2ab_config_path=str(DEFAULT_PR2AB_LOG_PATH),
                    start_control_loop=False,
                )
                ctrl.ensure_ms_start_guard(state_timeout=CONTROL_MS_GUARD_STATE_TIMEOUT_S)
                ctrl.start_control_loop()
                self.ctrl = ctrl
                logger.info(
                    "[ControlWorker] BaseController initialized with "
                    f"domain_id={self._body_domain_id}, "
                    f"kinematic_mode={self._kinematic_mode}, use_motor_state={self._use_motor_state}, "
                    f"pr2ab_config_path={DEFAULT_PR2AB_LOG_PATH}"
                )
                self.ctrl.register_pose(WALKING_STANCE_NAME, self._walking_pose_q)
            except Exception as exc:
                self._controller_init_blocked_reason = str(exc)
                self.ctrl = None
                if ctrl is not None:
                    try:
                        ctrl.abort(
                            timeout_ms=CONTROL_ABORT_SERVICE_TIMEOUT_MS,
                            service_call_timeout_s=CONTROL_ABORT_SERVICE_CALL_TIMEOUT_S,
                            ctrl_thread_join_timeout_s=CONTROL_ABORT_THREAD_JOIN_TIMEOUT_S,
                        )
                    except Exception:
                        logger.exception("[ControlWorker] ctrl.abort() failed after start guard error")
                if isinstance(exc, RuntimeError):
                    logger.error("[ControlWorker] controller initialization blocked: %s", exc)
                else:
                    logger.exception("[ControlWorker] failed to initialize BaseController")
                return

        self._wait_ctrl_ready_and_home()

    def _reset_gate_filter(self) -> None:
        self._gate_cmd_q = None
        self._gate_min_error = None
        self._gate_last_improve_time = None
        self._gate_target_last_q = None
        self._gate_target_vel_ema = None
        self._gate_target_last_t = None
        self._gate_last_step_t = None

    def _reset_entry_target_tracking(self) -> None:
        self._entry_last_target_sub = None
        self._entry_target_seq_at_start = None

    def _read_ik_target_seq(self) -> float | None:
        if self.ik_target_shm is None:
            return None
        try:
            value = float(self.ik_target_shm.read_data().get("target_seq", np.nan))
        except Exception:
            return None
        return value if np.isfinite(value) and value > 0.0 else None

    def _vr_entry_target_is_fresh(self) -> bool:
        baseline = self._entry_target_seq_at_start
        current = self._read_ik_target_seq()
        if baseline is None:
            return current is not None
        return current is not None and current > baseline

    def _reset_teleop_limiter(self) -> None:
        self._teleop_cmd_q = None
        self._teleop_cmd_dq[:] = 0.0
        self._teleop_limiter_last_t = None

    def _condition_teleop_target(
        self,
        target_q: np.ndarray,
        current_q: np.ndarray | None = None,
    ) -> np.ndarray:
        """Continuously bound teleop velocity, acceleration and command lead."""
        target = np.asarray(target_q, dtype=np.float64).reshape(-1)
        if target.size != NUM_MOTORS or not np.all(np.isfinite(target)):
            raise ValueError("teleop target must be a finite full-body joint vector")

        if current_q is None and self.ctrl is not None:
            current_q = self.ctrl.get_joint_q()
        measured = None
        if current_q is not None:
            candidate = np.asarray(current_q, dtype=np.float64).reshape(-1)
            if candidate.size == NUM_MOTORS and np.all(np.isfinite(candidate)):
                measured = candidate

        now = time.monotonic()
        if self._teleop_cmd_q is None or self._teleop_cmd_q.shape != target.shape:
            seed = measured if measured is not None else target
            self._teleop_cmd_q = np.asarray(seed, dtype=np.float64).copy()
            self._teleop_cmd_dq[:] = 0.0
            self._teleop_limiter_last_t = now

        if self._teleop_limiter_last_t is None:
            dt = 1.0 / max(float(self.fast_hz), 1e-6)
        else:
            dt = now - self._teleop_limiter_last_t
            if not np.isfinite(dt) or dt <= 0.0:
                dt = 1.0 / max(float(self.fast_hz), 1e-6)
        # A stalled worker must not turn one delayed tick into a large command
        # jump. The limiter catches up over subsequent bounded ticks instead.
        dt = float(np.clip(dt, 1e-4, 0.02))
        self._teleop_limiter_last_t = now

        active = list(self._entry_ramp_indices)
        q_prev = self._teleop_cmd_q.copy()
        error = target[active] - q_prev[active]
        vmax = self._teleop_velocity_limit[active]
        amax = self._teleop_acceleration_limit[active]
        braking_velocity = np.sqrt(np.maximum(0.0, 2.0 * amax * np.abs(error)))
        desired_velocity = np.sign(error) * np.minimum(vmax, braking_velocity)
        previous_velocity = self._teleop_cmd_dq[active]
        next_velocity = previous_velocity + np.clip(
            desired_velocity - previous_velocity,
            -amax * dt,
            amax * dt,
        )
        next_velocity = np.clip(next_velocity, -vmax, vmax)
        step = next_velocity * dt
        crossed = (error == 0.0) | (np.sign(error - step) != np.sign(error))
        q_next = q_prev[active] + step
        q_next[crossed] = target[active][crossed]
        next_velocity[crossed] = 0.0

        if measured is not None:
            lead = self._teleop_command_lead_limit[active]
            bounded = measured[active] + np.clip(q_next - measured[active], -lead, lead)
            lead_limited = np.abs(bounded - q_next) > 1e-12
            q_next = bounded
            next_velocity[lead_limited] = np.clip(
                (q_next[lead_limited] - q_prev[active][lead_limited]) / dt,
                -vmax[lead_limited],
                vmax[lead_limited],
            )

        command = target.copy()
        command[active] = q_next
        self._teleop_cmd_q = command.copy()
        self._teleop_cmd_dq[:] = 0.0
        self._teleop_cmd_dq[active] = next_velocity
        return command

    def _maybe_rearm_entry_gate_on_target_jump(
        self,
        gate_mode: str,
        target_q: np.ndarray,
        gate_indices: tuple[int, ...],
    ) -> None:
        if not self._arm_gate_passed:
            return
        if gate_mode != "leader_joint":
            return
        if not gate_indices:
            return
        threshold = float(self._gate_rearm_target_jump_rad)
        if threshold <= 0.0:
            return
        target_sub = np.asarray(target_q[list(gate_indices)], dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(target_sub)):
            return
        prev = self._entry_last_target_sub
        self._entry_last_target_sub = target_sub.copy()
        if prev is None or prev.shape != target_sub.shape:
            return
        jump = float(np.max(np.abs(target_sub - prev)))
        if jump <= threshold:
            return
        logger.warning(
            "[ControlWorker] leader target jumped by %.3f rad after entry gate passed; "
            "re-arming entry interpolation",
            jump,
        )
        self._arm_gate_passed = False
        self._reset_gate_filter()

    def _gate_step_dt(self, now: float) -> float:
        last = self._gate_last_step_t
        self._gate_last_step_t = float(now)
        if last is None:
            return 1.0 / max(float(self.fast_hz), 1e-6)
        dt = float(now) - float(last)
        if not np.isfinite(dt) or dt <= 0.0:
            return 1.0 / max(float(self.fast_hz), 1e-6)
        return float(np.clip(dt, 1e-4, 0.1))

    def _gate_filter(
        self,
        target_q: np.ndarray,
        current_q: np.ndarray | None,
        dt: float,
        indices: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        # Receding-horizon interpolation: one step toward target each cycle
        if dt <= 0.0:
            dt = 1.0 / self.fast_hz

        target_q = np.asarray(target_q, dtype=np.float64).reshape(-1)
        if current_q is None:
            current_q = target_q
        current_q = np.asarray(current_q, dtype=np.float64).reshape(-1)

        if self._gate_cmd_q is None or self._gate_cmd_q.shape != target_q.shape:
            seed_q = None
            if self.ctrl is not None:
                try:
                    with self.ctrl._ctrl_lock:
                        base_q = self.ctrl._target_q.copy()
                    if base_q is not None:
                        if base_q.shape == target_q.shape:
                            seed_q = base_q
                        else:
                            idx = self._gate_indices if indices is None else indices
                            if target_q.size == len(idx) and base_q.size >= len(idx):
                                seed_q = base_q[list(idx)]
                except Exception:
                    seed_q = None
            if seed_q is None:
                seed_q = current_q
            else:
                try:
                    max_seed_delta = float(np.max(np.abs(seed_q - current_q)))
                except Exception:
                    max_seed_delta = float("inf")
                if max_seed_delta > float(self._gate_seed_max_delta):
                    seed_q = current_q
            self._gate_cmd_q = np.asarray(seed_q, dtype=np.float64).reshape(-1).copy()
        max_error = float(np.max(np.abs(target_q - self._gate_cmd_q)))
        duration = float(self._gate_interp_duration_gain) * max_error
        duration = float(np.clip(
            duration,
            float(self._gate_interp_duration_min),
            float(self._gate_interp_duration_max),
        ))
        steps = max(1, int(duration / dt)) if duration > 0.0 else 1
        alpha = 1.0 / steps  # one step toward target per cycle
        q_cmd = self._gate_cmd_q + alpha * (target_q - self._gate_cmd_q)
        self._gate_cmd_q = q_cmd
        return q_cmd

    def _apply_gate_targets(self, target_q: np.ndarray) -> None:
        if self.ctrl is None:
            return
        self.ctrl.ctrl_leg(target_q[list(LEG_INDICES)], apply_clip=True)
        self.ctrl.ctrl_waist(target_q[list(WAIST_INDICES)], apply_clip=True)
        self.ctrl.ctrl_arm(target_q[list(ARM_INDICES)], apply_clip=True)
        self.ctrl.ctrl_neck(target_q[list(NECK_INDICES)], apply_clip=True)

    def _ensure_controller_ready(self, ev: EventSnapshot) -> bool:
        if self.ctrl is not None:
            try:
                return bool(self.ctrl.wait_for_state(timeout=0.0))
            except Exception:
                return True
        if self._controller_init_blocked_reason is not None:
            return False
        if not ev.level.get("ready", False):
            return False
        self._create_controller()
        if self.ctrl is None:
            return False
        try:
            return bool(self.ctrl.wait_for_state(timeout=0.0))
        except Exception:
            return True

    def _is_teleop_mode(self) -> bool:
        # mode_shm가 있으면 UI 선택 모드를 우선 사용
        if self.mode_shm is not None:
            try:
                mode_data = self.mode_shm.read_data()
                return bool(mode_data.get("teleop", False))
            except Exception:
                pass
        # fallback: 런처 run_config
        return bool(getattr(self.ctx.run_config, "mode", None) == "teleop")

    def _is_walking_mode(self) -> bool:
        if self.mode_shm is not None:
            try:
                mode_data = self.mode_shm.read_data()
                return bool(mode_data.get(WALKING_MODE_NAME, False))
            except Exception:
                pass
        return bool(getattr(self.ctx.run_config, "mode", None) == WALKING_MODE_NAME)

    def _selected_walking_profile_name(self) -> str:
        if self.walking_cmd_shm is None:
            return DEFAULT_WALKING_POLICY_PROFILE
        try:
            cmd_data = self.walking_cmd_shm.read_data()
        except Exception:
            return DEFAULT_WALKING_POLICY_PROFILE
        return resolve_walking_policy_profile_code(cmd_data.get("profile_code", 0.0))

    def _apply_walking_profile(self) -> None:
        if self.ctrl is None:
            return
        profile_name = self._selected_walking_profile_name()
        gain_profile = self._walking_gain_profiles.get(profile_name)
        if gain_profile is None:
            profile_name = DEFAULT_WALKING_POLICY_PROFILE
            gain_profile = self._walking_gain_profiles[profile_name]
        if self._walking_profile_active and self._walking_profile_active_name == profile_name:
            return

        walking_kp, walking_kd = gain_profile
        self._walking_kp = np.asarray(walking_kp, dtype=np.float32).reshape(-1)
        self._walking_kd = np.asarray(walking_kd, dtype=np.float32).reshape(-1)

        self.ctrl.set_joint_gains(range(NUM_MOTORS), kp=self._walking_kp, kd=self._walking_kd)
        self._walking_profile_active = True
        self._walking_profile_active_name = profile_name
        logger.info("[ControlWorker] applied walking gains for profile=%s", profile_name)

    def _restore_default_profile(self) -> None:
        if self.ctrl is None or not self._walking_profile_active:
            return
        self.ctrl.set_joint_gains(
            range(NUM_MOTORS),
            kp=np.asarray(self.ctrl._kp_default, dtype=np.float32).reshape(-1),
            kd=np.asarray(self.ctrl._kd_default, dtype=np.float32).reshape(-1),
        )
        self._walking_profile_active = False
        self._walking_profile_active_name = None

    def _move_to_walking_stance(self) -> None:
        if self.ctrl is None:
            return
        hand_target = WALKING_NEUTRAL_HAND_Q.copy()
        self._run_body_motion_with_hand(
            lambda: (
                self.ctrl.move_to_pose(
                    WALKING_STANCE_NAME,
                    duration=WALKING_PREP_LEG_WAIST_DURATION,
                    leg=True,
                    waist=True,
                    arm=False,
                    neck=False,
                    preserve_gains=True,
                ),
                self.ctrl.default_pos_state(
                    WALKING_STANCE_NAME,
                    leg=True,
                    waist=True,
                    arm=False,
                    neck=False,
                    preserve_gains=True,
                ),
                self.ctrl.move_to_pose(
                    WALKING_STANCE_NAME,
                    duration=WALKING_PREP_ARM_NECK_DURATION,
                    leg=False,
                    waist=False,
                    arm=True,
                    neck=True,
                    preserve_gains=True,
                ),
                self.ctrl.default_pos_state(
                    WALKING_STANCE_NAME,
                    leg=False,
                    waist=False,
                    arm=True,
                    neck=True,
                    preserve_gains=True,
                ),
            ),
            hand_target,
        )
        self.ctrl.default_pos_state(WALKING_STANCE_NAME, preserve_gains=True)
        self._sync_action_shm_with_goal(self._walking_pose_q.copy(), hand_target=hand_target)
        self._walking_pose_pending = False

    def _read_action_target(self):
        """ROBOT_ACTION SHM에서 부위별 타겟을 읽어 full joint q 타겟으로 조합."""
        try:
            act = self.act_shm.read_data()
        except Exception as exc:
            logger.warning("[ControlWorker] act_shm read failed: %s", exc)
            return None

        try:
            # SHM은 부위별(act_leg/act_waist/act_arm/act_neck)로 저장되어 있음.
            # motor_control.BaseController는 JointIndex(Global) 순서를 기대하므로, 해당 인덱스에 맞춰 재배치.
            tgt = np.zeros(NUM_MOTORS, dtype=np.float64)
            tgt[list(WAIST_INDICES)] = act["act_waist"]
            if self._is_teleop_mode():
                if self._teleop_fixed_leg_q is None:
                    current_q = self.ctrl.get_joint_q() if self.ctrl is not None else None
                    if current_q is None:
                        return None
                    self._teleop_fixed_leg_q = np.asarray(
                        current_q[list(LEG_INDICES)], dtype=np.float64
                    ).copy()
                    logger.info("[ControlWorker] latched fixed teleop leg target")
                tgt[list(LEG_INDICES)] = self._teleop_fixed_leg_q
            else:
                self._teleop_fixed_leg_q = None
                tgt[list(LEG_INDICES)] = act["act_leg"]
            tgt[list(ARM_INDICES)] = act["act_arm"]
            tgt[list(NECK_INDICES)] = act["act_neck"]
            # act_hand는 별도 컨트롤러에서 사용하므로 여기서는 무시
            return tgt
        except Exception as exc:
            logger.warning(
                "[ControlWorker] act_shm shape mismatch "
                "(expect leg/waist/neck/arm slices matching JointIndex): %s",
                exc,
            )
            return None

    def _entry_gate_mode(self) -> str:
        teleop_device = str(getattr(self.ctx.run_config, "teleop_device", "") or "").strip().lower()
        if teleop_device in {"masterarm", "vr_masterarm"}:
            return "leader_joint"
        return "vr_ee"

    def _active_gate_indices(self) -> tuple[int, ...]:
        return self._entry_ramp_indices

    def _joint_target_within_error(
        self,
        target_q: np.ndarray,
        current_q: np.ndarray | None,
        indices: tuple[int, ...],
        threshold: float | None,
    ) -> tuple[bool, float | None]:
        if threshold is None:
            return True, None
        if self.ctrl is None:
            return False, None
        if current_q is None:
            current_q = self.ctrl.get_joint_q()
        if current_q is None:
            return False, None
        if not indices:
            return True, 0.0
        idx = list(indices)
        max_error = float(np.max(np.abs(target_q[idx] - current_q[idx])))
        return max_error <= float(threshold), max_error

    @staticmethod
    def _valid_transform_matrix(value) -> np.ndarray | None:
        try:
            mat = np.asarray(value, dtype=np.float64).reshape(4, 4)
        except Exception:
            return None
        if not np.all(np.isfinite(mat)):
            return None
        return mat

    @staticmethod
    def _rotation_error_rad(target_mat: np.ndarray, actual_mat: np.ndarray) -> float:
        r_target = np.asarray(target_mat[:3, :3], dtype=np.float64)
        r_actual = np.asarray(actual_mat[:3, :3], dtype=np.float64)
        r_err = r_target @ r_actual.T
        cos_theta = (float(np.trace(r_err)) - 1.0) * 0.5
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        return float(np.arccos(cos_theta))

    def _ee_target_within_error(self) -> tuple[bool, float | None, float | None]:
        if self._ee_target_max_pos_error is None and self._ee_target_max_rot_error is None:
            return True, None, None
        if self.ee_shm is None or self.ik_target_shm is None:
            return False, float("inf"), float("inf")
        try:
            ee = self.ee_shm.read_data()
            target = self.ik_target_shm.read_data()
        except Exception:
            return False, float("inf"), float("inf")

        try:
            if float(target.get("target_valid", 0.0)) <= 0.5:
                return False, float("inf"), float("inf")
        except Exception:
            return False, float("inf"), float("inf")

        max_pos_error = 0.0
        max_rot_error = 0.0
        for key in ("left_wrist_mat", "right_wrist_mat"):
            target_mat = self._valid_transform_matrix(target.get(key))
            actual_mat = self._valid_transform_matrix(ee.get(key))
            if target_mat is None or actual_mat is None:
                return False, float("inf"), float("inf")
            pos_error = float(np.linalg.norm(target_mat[:3, 3] - actual_mat[:3, 3]))
            rot_error = self._rotation_error_rad(target_mat, actual_mat)
            max_pos_error = max(max_pos_error, pos_error)
            max_rot_error = max(max_rot_error, rot_error)

        pos_ok = (
            True
            if self._ee_target_max_pos_error is None
            else max_pos_error <= float(self._ee_target_max_pos_error)
        )
        rot_ok = (
            True
            if self._ee_target_max_rot_error is None
            else max_rot_error <= float(self._ee_target_max_rot_error)
        )
        return bool(pos_ok and rot_ok), max_pos_error, max_rot_error

    def _entry_target_within_error(
        self, target_q: np.ndarray, current_q: np.ndarray | None
    ) -> tuple[bool, float | None, float | None, str]:
        mode = self._entry_gate_mode()
        if mode == "leader_joint":
            within, max_error = self._joint_target_within_error(
                target_q,
                current_q,
                self._leader_gate_indices,
                self._arm_target_max_error,
            )
            return within, max_error, self._arm_target_max_error, "rad"

        within, max_pos_error, max_rot_error = self._ee_target_within_error()
        if self._ee_target_max_pos_error is None:
            metric = max_rot_error
            threshold = self._ee_target_max_rot_error
            unit = "rad"
        else:
            metric = max_pos_error
            threshold = self._ee_target_max_pos_error
            unit = "m"
        return within, metric, threshold, unit

    def _maybe_seed_action_shm(self) -> None:
        if self._act_shm_seeded:
            return
        if self.act_shm is None or self.ctrl is None:
            return
        try:
            act = self.act_shm.read_data()
        except Exception as exc:
            logger.debug("[ControlWorker] act_shm read failed for seed: %s", exc)
            return

        try:
            act_leg = np.asarray(act.get("act_leg"), dtype=np.float64).reshape(-1)
            act_waist = np.asarray(act.get("act_waist"), dtype=np.float64).reshape(-1)
            act_arm = np.asarray(act.get("act_arm"), dtype=np.float64).reshape(-1)
            act_neck = np.asarray(act.get("act_neck"), dtype=np.float64).reshape(-1)
        except Exception:
            return

        act_concat = np.concatenate((act_leg, act_waist, act_arm, act_neck), axis=0)
        if np.all(np.abs(act_concat) <= float(self._act_shm_seed_tol)):
            current_q = self.ctrl.get_joint_q()
            if current_q is None:
                return
            self.act_shm.write_data(
                act_leg=current_q[list(LEG_INDICES)],
                act_waist=current_q[list(WAIST_INDICES)],
                act_arm=current_q[list(ARM_INDICES)],
                act_neck=current_q[list(NECK_INDICES)],
            )
        self._act_shm_seeded = True

    @staticmethod
    def _decode_shared_string(raw) -> str:
        if raw is None:
            return ""
        try:
            buf = np.asarray(raw, dtype=np.uint8).reshape(-1).tobytes()
        except Exception:
            return ""
        return buf.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()

    def _read_pose_request_dataset_key(self) -> str | None:
        if self.pose_request_shm is None:
            return None
        try:
            data = self.pose_request_shm.read_data()
        except Exception as exc:
            logger.warning("[ControlWorker] pose_request_shm read failed: %s", exc)
            return None

        try:
            request_seq = int(np.asarray(data.get("request_seq", 0), dtype=np.uint8).reshape(()).item())
        except Exception as exc:
            logger.warning("[ControlWorker] pose request sequence decode failed: %s", exc)
            return None

        if request_seq == self._last_pose_request_seq:
            return None
        self._last_pose_request_seq = request_seq

        dataset_key = self._decode_shared_string(data.get("dataset_key"))
        if not dataset_key:
            logger.warning("[ControlWorker] received empty init pose request")
            return None
        return dataset_key

    def _load_init_pose_entry(self, dataset_key: str) -> dict | None:
        try:
            with INIT_SETTING_PATH.open("r", encoding="utf-8") as f:
                payload = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("[ControlWorker] failed to load init pose config: %s", exc)
            return None

        datasets = payload.get("datasets", {})
        if not isinstance(datasets, dict):
            logger.warning("[ControlWorker] invalid init pose config: 'datasets' must be a mapping")
            return None

        entry = datasets.get(dataset_key)
        if not isinstance(entry, dict):
            logger.warning("[ControlWorker] no init pose entry for dataset=%s", dataset_key)
            return None
        return entry

    def _resolve_pose_base_q(self) -> np.ndarray:
        if self.ctrl is None:
            raise RuntimeError("controller is not ready")

        try:
            with self.ctrl._ctrl_lock:
                target_q = np.asarray(self.ctrl._target_q, dtype=np.float64).reshape(-1)
            if target_q.size == NUM_MOTORS:
                return target_q.copy()
        except Exception:
            pass

        current_q = self.ctrl.get_joint_q()
        if current_q is not None:
            current_q = np.asarray(current_q, dtype=np.float64).reshape(-1)
            if current_q.size == NUM_MOTORS:
                return current_q.copy()

        return np.asarray(self.ctrl._default_q, dtype=np.float64).reshape(-1).copy()

    def _build_pose_targets_from_dataset(self, dataset_key: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        entry = self._load_init_pose_entry(dataset_key)
        if entry is None:
            return None, None

        joint_names = entry.get("joint_names")
        joint_values = entry.get("joint_values")
        if not isinstance(joint_names, list) or not isinstance(joint_values, list):
            logger.warning("[ControlWorker] invalid init pose entry for dataset=%s", dataset_key)
            return None, None
        if len(joint_names) != len(joint_values):
            logger.warning(
                "[ControlWorker] init pose entry length mismatch for dataset=%s: names=%d values=%d",
                dataset_key,
                len(joint_names),
                len(joint_values),
            )
            return None, None

        goal_q = self._resolve_pose_base_q()
        body_applied = 0
        hand_applied = 0
        hand_invalid = False
        hand_target = np.full(HAND_TARGET_DIM, np.nan, dtype=np.float64)
        ignored: list[str] = []
        for joint_name, joint_value in zip(joint_names, joint_values):
            name = str(joint_name).strip()
            hand_idx = INIT_POSE_HAND_INDEX.get(name)
            if hand_idx is not None:
                try:
                    hand_target[hand_idx] = float(joint_value)
                    hand_applied += 1
                except Exception as exc:
                    hand_invalid = True
                    logger.warning(
                        "[ControlWorker] invalid hand joint value for dataset=%s joint=%s: %s",
                        dataset_key,
                        name,
                        exc,
                    )
                continue

            joint_idx = INIT_POSE_JOINT_INDEX.get(name)
            if joint_idx is None:
                ignored.append(name)
                continue
            try:
                goal_q[joint_idx] = float(joint_value)
            except Exception as exc:
                logger.warning(
                    "[ControlWorker] invalid joint value for dataset=%s joint=%s: %s",
                    dataset_key,
                    name,
                    exc,
                )
                return None, None
            body_applied += 1

        if ignored:
            logger.info(
                "[ControlWorker] ignored unknown init pose joints for dataset=%s: %s",
                dataset_key,
                ", ".join(ignored),
            )

        if body_applied <= 0:
            logger.warning("[ControlWorker] no controllable joints found for dataset=%s", dataset_key)
            return None, None

        resolved_hand_target: np.ndarray | None = None
        if hand_applied == 0 and not hand_invalid:
            logger.warning("[ControlWorker] no hand init pose joints found for dataset=%s", dataset_key)
        elif hand_invalid or hand_applied != HAND_TARGET_DIM or np.any(np.isnan(hand_target)):
            missing = [f"hand_{idx}" for idx, value in enumerate(hand_target) if not np.isfinite(value)]
            logger.warning(
                "[ControlWorker] incomplete hand init pose for dataset=%s; keeping current hand target (missing=%s)",
                dataset_key,
                ", ".join(missing) if missing else "n/a",
            )
        else:
            resolved_hand_target = hand_target.copy()

        return goal_q, resolved_hand_target

    def _sync_action_shm_with_goal(self, goal_q: np.ndarray, hand_target: np.ndarray | None = None) -> None:
        if self.act_shm is None:
            return
        goal_q = np.asarray(goal_q, dtype=np.float64).reshape(-1)
        if goal_q.size != NUM_MOTORS:
            return
        kwargs = {
            "act_leg": goal_q[list(LEG_INDICES)],
            "act_waist": goal_q[list(WAIST_INDICES)],
            "act_arm": goal_q[list(ARM_INDICES)],
            "act_neck": goal_q[list(NECK_INDICES)],
        }
        if hand_target is not None:
            hand_goal = np.asarray(hand_target, dtype=np.float64).reshape(-1)
            if hand_goal.size != HAND_TARGET_DIM:
                logger.warning(
                    "[ControlWorker] invalid hand sync target shape %s, expected (%d,)",
                    hand_goal.shape,
                    HAND_TARGET_DIM,
                )
            else:
                kwargs["act_hand"] = hand_goal
        try:
            self.act_shm.write_data(**kwargs)
            self._act_shm_seeded = True
        except Exception as exc:
            logger.warning("[ControlWorker] failed to sync act_shm with init pose target: %s", exc)

    def _publish_walking_debug_gains(self) -> None:
        if self.walking_debug_shm is None or self.ctrl is None:
            return
        try:
            snapshot = self.ctrl.get_command_snapshot()
        except Exception as exc:
            logger.debug("[ControlWorker] failed to read command snapshot for walking debug: %s", exc)
            return

        try:
            kp = np.asarray(snapshot["kp"], dtype=np.float64).reshape(-1)
            kd = np.asarray(snapshot["kd"], dtype=np.float64).reshape(-1)
            self.walking_debug_shm.write_data(
                active_kp_leg=kp[list(LEG_INDICES)],
                active_kd_leg=kd[list(LEG_INDICES)],
                active_kp_waist=kp[list(WAIST_INDICES)],
                active_kd_waist=kd[list(WAIST_INDICES)],
                active_kp_arm=kp[list(ARM_INDICES)],
                active_kd_arm=kd[list(ARM_INDICES)],
                active_kp_neck=kp[list(NECK_INDICES)],
                active_kd_neck=kd[list(NECK_INDICES)],
            )
        except Exception as exc:
            logger.debug("[ControlWorker] failed to publish walking debug gains: %s", exc)

    def _execute_init_pose_request(self, dataset_key: str) -> None:
        if self.ctrl is None:
            return

        goal_q, hand_target = self._build_pose_targets_from_dataset(dataset_key)
        if goal_q is None:
            return

        try:
            logger.info("[ControlWorker] moving to init pose for dataset=%s", dataset_key)
            self._run_body_motion_with_hand(
                lambda: self.ctrl.move_to_pose(goal_q, duration=HAND_MOVE_DURATION),
                hand_target,
            )
            self.ctrl.default_pos_state(pose=goal_q)
            self._sync_action_shm_with_goal(goal_q, hand_target=hand_target)
            logger.info("[ControlWorker] init pose move complete for dataset=%s", dataset_key)
        except Exception:
            logger.exception("[ControlWorker] init pose move failed for dataset=%s", dataset_key)



    def do_slow(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        """
        관측(waist/leg/arm/neck)을 읽어서 ROBOT_OBS SHM에 기록.
        """
        
        if not self._ensure_controller_ready(ev):
            return

        if self._body_motion_active.is_set():
            now = time.monotonic()
            if (
                self._body_motion_last_telemetry_t is not None
                and now - self._body_motion_last_telemetry_t < (1.0 / 30.0)
            ):
                return
            self._body_motion_last_telemetry_t = now
        else:
            self._body_motion_last_telemetry_t = None

        if tr.reason == "home_set":
            self._home_request = True

        q = self.ctrl.get_joint_q()
        dq = self.ctrl.get_joint_dq()
        tau_pair = self.ctrl.get_joint_tau()
        quat, gyro, rpy = self.ctrl.get_imu()
        if q is None:
            return
        if dq is None:
            dq = np.zeros((NUM_MOTORS,), dtype=np.float32)
        if tau_pair is None:
            tau_raw = np.zeros((NUM_MOTORS,), dtype=np.float32)
            tau = np.zeros((NUM_MOTORS,), dtype=np.float32)
        else:
            tau_raw, tau = tau_pair
        if quat is None:
            quat = np.zeros((4,), dtype=np.float32)
        if gyro is None:
            gyro = np.zeros((3,), dtype=np.float32)
        if rpy is None:
            rpy = np.zeros((3,), dtype=np.float32)

        # 관측을 부위별로 분리 (JointIndex 순서를 WAIST/LEG/ARM/NECK 슬라이스로 나눔)
        obs_waist = q[list(WAIST_INDICES)]
        obs_leg = q[list(LEG_INDICES)]
        obs_leg_dq = dq[list(LEG_INDICES)]
        obs_arm = q[list(ARM_INDICES)]
        obs_neck = q[list(NECK_INDICES)]
        self._obs_seq += 1.0

        self.obs_shm.write_data(
            obs_seq=np.float64(self._obs_seq),
            obs_leg=obs_leg,
            obs_leg_dq=obs_leg_dq,
            obs_waist=obs_waist,
            obs_neck=obs_neck,
            obs_arm=obs_arm,
            obs_imu_quat=np.asarray(quat, dtype=np.float64).reshape(-1),
            obs_imu_gyro=np.asarray(gyro, dtype=np.float64).reshape(-1),
            obs_imu_rpy=np.asarray(rpy, dtype=np.float64).reshape(-1),
        )
        self.tau_shm.write_data(
            tau_est_leg=tau[list(LEG_INDICES)],
            tau_est_waist=tau[list(WAIST_INDICES)],
            tau_est_neck=tau[list(NECK_INDICES)],
            tau_est_arm=tau[list(ARM_INDICES)],
        )
        self._publish_walking_debug_gains()
        
    def do_fast(self, ev: EventSnapshot, tr: TransitionResult) -> None:

        if not self._ensure_controller_ready(ev):
            return

        if self._body_motion_active.is_set():
            return

        walking_mode = self._is_walking_mode()
        if walking_mode != self._prev_walking_mode:
            if walking_mode:
                logger.info("[ControlWorker] walking mode entered -> apply walking gains/stance")
                self._walking_pose_pending = True
                self._apply_walking_profile()
            else:
                logger.info("[ControlWorker] walking mode exited -> restore default gains")
                self._walking_pose_pending = False
                self._restore_default_profile()
            self._act_shm_seeded = False
            self._prev_walking_mode = walking_mode

        if walking_mode and not self._walking_profile_active:
            self._apply_walking_profile()

        if tr.reason == "home_set":
            self._home_request = True
            self._act_shm_seeded = False
            self._teleop_fixed_leg_q = None

        if tr.reason in {"start_set", "home_cleared_start_set"} and self._is_teleop_mode():
            logger.info("[ControlWorker] teleop start -> body entry ramp reset")
            self._arm_gate_passed = False
            self._reset_gate_filter()
            self._reset_entry_target_tracking()
            self._reset_teleop_limiter()
            self._act_shm_seeded = False
            self._teleop_fixed_leg_q = None

        dataset_key = self._read_pose_request_dataset_key()
        if dataset_key is not None:
            self._pending_init_pose_dataset_key = dataset_key
            logger.info("[ControlWorker] queued init pose request for dataset=%s", dataset_key)

        if self.state == ModeState.HOME and self._home_request and not self._homing_active:
            logger.info("[ControlWorker] starting homing procedure.")
            self._home_request = False
            self._arm_gate_passed = False
            self._reset_gate_filter()
            self._reset_entry_target_tracking()
            self._reset_teleop_limiter()
            self._go_home()
            if walking_mode:
                logger.info("[ControlWorker] Homing complete, holding walking stance.")
            else:
                logger.info("[ControlWorker] Homing complete, holding default pose.")

        if (
            walking_mode
            and self._walking_pose_pending
            and self.state == ModeState.RUN
            and not self._home_request
            and not self._homing_active
            and self._pending_init_pose_dataset_key is None
        ):
            self._move_to_walking_stance()
            return

        if (
            self._pending_init_pose_dataset_key is not None
            and self.state != ModeState.RUN
            and not self._home_request
            and not self._homing_active
        ):
            dataset_key = self._pending_init_pose_dataset_key
            self._pending_init_pose_dataset_key = None
            self._execute_init_pose_request(dataset_key)
            return
            
        # RUN일 때만 명령 생성/전송
        if self.state == ModeState.RUN:
            self._maybe_seed_action_shm()
            if self._gate_status_log_interval is not None:
                now = time.monotonic()
                last = self._gate_status_last_log_time
                gate_changed = self._gate_status_last_value is None or self._gate_status_last_value != self._arm_gate_passed
                periodic_due = (
                    not self._arm_gate_passed
                    and (last is None or (now - last) >= float(self._gate_status_log_interval))
                )
                if gate_changed or periodic_due:
                    logger.info(
                        "[ControlWorker] entry gate passed=%s mode=%s",
                        self._arm_gate_passed,
                        self._entry_gate_mode(),
                    )
                    self._gate_status_last_log_time = now
                    self._gate_status_last_value = self._arm_gate_passed
            target_q = self._read_action_target()
            if target_q is None:
                return

            teleop_mode = self._is_teleop_mode()
            if teleop_mode != self._prev_teleop_mode:
                if teleop_mode:
                    logger.info(
                        "[ControlWorker] teleop mode entered -> entry gate enabled (mode=%s)",
                        self._entry_gate_mode(),
                    )
                    self._arm_gate_passed = False
                else:
                    logger.info("[ControlWorker] teleop mode exited -> entry gate bypassed")
                    self._arm_gate_passed = True
                self._reset_gate_filter()
                self._reset_entry_target_tracking()
                if teleop_mode and self._entry_gate_mode() == "vr_ee":
                    self._entry_target_seq_at_start = self._read_ik_target_seq()
                self._reset_teleop_limiter()
                self._act_shm_seeded = False
            self._prev_teleop_mode = teleop_mode

            if not teleop_mode:
                self._arm_gate_passed = True
                self._reset_gate_filter()
                self._reset_entry_target_tracking()
                self._reset_teleop_limiter()
                self.ctrl.set_joint_targets(q=target_q.astype(np.float32, copy=False))
                return

            gate_mode = self._entry_gate_mode()
            gate_indices = self._active_gate_indices()
            self._maybe_rearm_entry_gate_on_target_jump(
                gate_mode,
                target_q,
                self._leader_gate_indices,
            )
            
            if not self._arm_gate_passed:
                current_q = self.ctrl.get_joint_q()
                if current_q is None:
                    return
                if gate_mode == "vr_ee" and not self._vr_entry_target_is_fresh():
                    # HOME publishes an IK target equal to the measured pose.
                    # Do not let that stale sample pass the Start gate before
                    # the Unity IK worker has produced its first RUN target.
                    hold_q = self._condition_teleop_target(current_q, current_q)
                    self.ctrl.set_joint_targets(q=hold_q.astype(np.float32, copy=False))
                    return
                within_error, max_error, threshold, threshold_unit = self._entry_target_within_error(
                    target_q,
                    current_q,
                )
                if not within_error:
                    now = time.monotonic()
                    dt = self._gate_step_dt(now)
                    idx = gate_indices
                    target_sub = target_q[list(idx)]
                    current_sub = current_q[list(idx)]
                    # If target is still moving fast, don't allow force-pass timer to elapse.
                    if self._gate_target_last_q is None or self._gate_target_last_q.shape != target_sub.shape:
                        self._gate_target_last_q = target_sub.copy()
                        self._gate_target_last_t = now
                    else:
                        max_delta = float(np.max(np.abs(target_sub - self._gate_target_last_q)))
                        if max_delta <= float(self._gate_target_vel_eps):
                            target_vel = 0.0
                        else:
                            dt_target = (
                                now - self._gate_target_last_t
                                if self._gate_target_last_t is not None
                                else dt
                            )
                            target_vel = max_delta / max(dt_target, 1e-6)
                            self._gate_target_last_q = target_sub.copy()
                            self._gate_target_last_t = now
                        if self._gate_target_vel_ema is None:
                            self._gate_target_vel_ema = float(target_vel)
                        else:
                            alpha = float(self._gate_target_vel_ema_alpha)
                            self._gate_target_vel_ema = (
                                alpha * float(target_vel)
                                + (1.0 - alpha) * float(self._gate_target_vel_ema)
                            )
                        if self._gate_target_vel_ema > float(self._gate_target_vel_th):
                            self._gate_last_improve_time = now
                    if max_error is not None:
                        if self._gate_min_error is None or max_error < self._gate_min_error:
                            self._gate_min_error = max_error
                            self._gate_last_improve_time = now
                        if (
                            self._gate_force_pass_timeout is not None
                            and self._gate_last_improve_time is not None
                            and (now - self._gate_last_improve_time)
                            >= float(self._gate_force_pass_timeout)
                        ):
                            logger.warning(
                                "[ControlWorker] entry gate (%s) still outside threshold after %.2fs; "
                                "continuing ramp without direct target jump "
                                "(current=%.4f %s, min=%.4f %s, threshold=%.4f %s)",
                                gate_mode,
                                now - self._gate_last_improve_time,
                                float(max_error),
                                threshold_unit,
                                float(self._gate_min_error)
                                if self._gate_min_error is not None
                                else float("nan"),
                                threshold_unit,
                                float(threshold)
                                if threshold is not None
                                else float("nan"),
                                threshold_unit,
                            )
                            self._gate_last_improve_time = now
                    cmd_sub = self._gate_filter(target_sub, current_sub, dt, indices=idx)
                    cmd_q = target_q.copy()
                    cmd_q[list(idx)] = cmd_sub
                    bounded_q = self._condition_teleop_target(cmd_q, current_q)
                    self.ctrl.set_joint_targets(q=bounded_q.astype(np.float32, copy=False))
                    return
                if threshold is None:
                    logger.info("[ControlWorker] entry gate (%s) passed (gate disabled)", gate_mode)
                else:
                    logger.info(
                        "[ControlWorker] entry gate (%s) passed (max_error=%.4f %s <= %.4f %s)",
                        gate_mode,
                        float(max_error) if max_error is not None else float("nan"),
                        threshold_unit,
                        float(threshold),
                        threshold_unit,
                    )
                self._arm_gate_passed = True
                self._reset_gate_filter()
            
            bounded_q = self._condition_teleop_target(target_q)
            self.ctrl.set_joint_targets(q=bounded_q.astype(np.float32, copy=False))

        elif self.state == ModeState.HOME:
            pass

        # logger.info(f"[{self.ctx.name}][fast] state={self.state.name} reason={tr.reason} cmd={self._cmd:.3f}")

    def on_stop(self) -> None:
        
        if self.ctrl is not None:
            try:
                self.ctrl.stop(
                    shutdown_timeout_ms=CONTROL_STOP_SERVICE_TIMEOUT_MS,
                    service_call_timeout_s=CONTROL_STOP_SERVICE_CALL_TIMEOUT_S,
                    ctrl_thread_join_timeout_s=CONTROL_STOP_THREAD_JOIN_TIMEOUT_S,
                )
            except Exception:
                logger.exception("[ControlWorker] ctrl.stop() failed during shutdown, fallback to abort")
                try:
                    self.ctrl.abort(
                        timeout_ms=CONTROL_ABORT_SERVICE_TIMEOUT_MS,
                        service_call_timeout_s=CONTROL_ABORT_SERVICE_CALL_TIMEOUT_S,
                        ctrl_thread_join_timeout_s=CONTROL_ABORT_THREAD_JOIN_TIMEOUT_S,
                    )
                except Exception:
                    logger.exception("[ControlWorker] ctrl.abort() fallback failed during shutdown")

        if self._shared_memory:
            for key, mgr in self._shared_memory.items():
                try:
                    mgr.worker_close()
                except Exception:
                    logger.exception(f"[{self.ctx.name}] failed to close shared memory {key}")

        
        logger.info(f"[{self.ctx.name}] stop")
