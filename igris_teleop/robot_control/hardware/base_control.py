import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))


import numpy as np
import threading
from enum import IntEnum
from typing import Union, Dict, Optional
from pathlib import Path
import yaml
import time

import igris_c_sdk as igc_sdk
from igris_teleop.core.project_paths import JOINT_SETTING_PATH
from ...core.rate import Rate
from ..filters.torque_kalman import TorqueKalmanBank


import logging_mp
logger_mp = logging_mp.get_logger(__name__, level=logging_mp.INFO)


# DDS topic names (match current client.py defaults)
kTopicLowCommand = "rt/lowcmd"
kTopicLowState = "rt/lowstate"
NUM_MOTORS = 31  # == N_JOINTS (PJS joint DOF 개수)
DEFAULT_VEL_LIMIT = 5.0
RESERVED_JOINT_PROFILE_KEYS = frozenset(
    {"kp", "kd", "default_dof_pos", "waypoint_1", "waypoint_2", "joint_order", "joint_groups"}
)


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
    """Load kp/kd/default_q from yaml, requiring length == NUM_MOTORS (joints)."""
    cfg_path = Path(cfg_path)
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent

    candidates = (
        [cfg_path]
        if cfg_path.is_absolute()
        else [base_dir / cfg_path, project_root / cfg_path]
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
    Minimal low-level controller using IGRIS python binding (PJS joint space).

    - Subscribes to LowState and keeps the latest state in a thread-safe buffer
    - Publishes LowCmd at a fixed rate (default 300 Hz) using the current joint targets
    - Provides helpers to update joint targets and call service APIs (BMS/Torque/Mode)
    """

    def __init__(
        self,
        domain_id: int = 0,
        control_hz: float = 300.0,
        # 기본 모드: PJS (조인트 공간)
        kinematic_mode=igc_sdk.KinematicMode.PJS,
        use_motor_state: bool = False,
        auto_service_init: bool = True,
        bms_init_type=igc_sdk.BmsInitType.BMS_AND_MOTOR_INIT,
        torque_type=igc_sdk.TorqueType.TORQUE_ON,
        control_mode=igc_sdk.ControlMode.CONTROL_MODE_LOW_LEVEL,
        service_timeout_ms: int = 30000,
        service_init_delay: float = 0.5,
    ):
        logger_mp.info("[BaseController] Initializing (domain_id=%s)...", domain_id)

        self.control_rate = Rate(control_hz)
        self.control_dt = 1.0 / control_hz
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._ctrl_lock = threading.Lock()
        self._use_motor_state = use_motor_state
        self._dds_cleaned = False
        self.lowstate_sub = None
        self.lowcmd_pub = None

        self.channel_factory = igc_sdk.ChannelFactory.Instance()
        self.channel_factory.Init(int(domain_id))

        self.client = igc_sdk.IgrisC_Client()
        self.client.Init()
        self.client.SetTimeout(service_timeout_ms / 1000.0)
        kp_default, kd_default, default_q, waypoint_1, waypoint_2 = _load_joint_profile(str(JOINT_SETTING_PATH))
        optional_poses = _load_optional_joint_poses(str(JOINT_SETTING_PATH))

        # Keep original gains for easy restore after zero/damping modes
        self._kp_default = kp_default.copy()
        self._kd_default = kd_default.copy()

        self._kp = kp_default.copy()
        self._kd = kd_default.copy()

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

        
        
        self._target_q = np.zeros_like(self._default_q)
        self._target_dq = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._target_tau = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._initial_q = None
        
        
        self.leg_velocity_limit = DEFAULT_VEL_LIMIT
        self.waist_velocity_limit = DEFAULT_VEL_LIMIT
        self.arm_velocity_limit = DEFAULT_VEL_LIMIT
        self.neck_velocity_limit = DEFAULT_VEL_LIMIT

        self._tau_kf = TorqueKalmanBank(
            n_joints=NUM_MOTORS,
            q=3.0,     # 튜닝 포인트
            r=20.0,    # 튜닝 포인트
            P0=1e3
        )


        # Debug flag for one-time publish logging
        self._dbg_publish_logged = False

        self.state_buffer = LowStateBuffer()

        # LowStateSubscriber
        logger_mp.info("[BaseController] lowstate_sub")
        self.lowstate_sub = igc_sdk.LowStateSubscriber(kTopicLowState)
        try:
            self.lowstate_sub.init(self._on_low_state)
        except Exception as e:
            self._cleanup_dds()
            raise RuntimeError("Failed to init LowStateSubscriber") from e

        
        logger_mp.info("[BaseController] auto_service_init")
        if auto_service_init:
            if service_init_delay > 0.0:
                time.sleep(service_init_delay)  # allow DDS discovery like the menu pause in client.py
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

        self.lowcmd_msg = igc_sdk.LowCmd()
        self._kinematic_mode = kinematic_mode
        self.lowcmd_msg.kinematic_mode(self._kinematic_mode)

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

    def _call_service_with_retry(self, fn, name: str, attempts: int = 3, delay_sec: float = 1.0):
        """
        Call a service function with simple retries to avoid initial discovery misses.
        """
        last_res = None
        for i in range(1, attempts + 1):
            res = fn()
            last_res = res
            if res.success():
                logger_mp.info("[BaseController] %s succeeded: %s", name, res.message())
                return res
            logger_mp.warning(
                "[BaseController] %s failed (%d/%d): %s", name, i, attempts, res.message()
            )
            if i < attempts:
                time.sleep(delay_sec)
        raise RuntimeError(f"{name} failed after {attempts} attempts: {last_res.message()}")

    def _service_bootstrap(self, bms_init_type, torque_type, control_mode, timeout_ms: int):
        # 동일한 순서/호출 구조는 python client 예제와 맞춘다.
        logger_mp.info("[BaseController] InitBms start (%s, timeout=%d ms)", bms_init_type, timeout_ms)
        res = self._call_service_with_retry(
            lambda: self.client.InitBms(bms_init_type, timeout_ms),
            f"InitBms({bms_init_type})",
        )
        logger_mp.info("[BaseController] SetTorque start (%s, timeout=%d ms)", torque_type, timeout_ms)
        res = self._call_service_with_retry(
            lambda: self.client.SetTorque(torque_type, timeout_ms),
            f"SetTorque({torque_type})",
        )
        logger_mp.info("[BaseController] SetControlMode start (%s, timeout=%d ms)", control_mode, timeout_ms)
        res = self._call_service_with_retry(
            lambda: self.client.SetControlMode(control_mode, timeout_ms),
            f"SetControlMode({control_mode})",
        )

        logger_mp.info(
            "[BaseController] Services done: BMS=%s, Torque=%s, Mode=%s",
            bms_init_type,
            torque_type,
            control_mode,
        )

    def _service_shutdown(
        self,
        torque_type=igc_sdk.TorqueType.TORQUE_OFF,
        bms_shutdown_type=igc_sdk.BmsInitType.BMS_INIT_NONE,
        control_mode=igc_sdk.ControlMode.CONTROL_MODE_HIGH_LEVEL,
        timeout_ms: int = 30000,
    ):
        """
        Graceful shutdown sequence opposite to _service_bootstrap.
        - optionally switch control mode (e.g., HIGH_LEVEL) before powering down
        - torque OFF
        - BMS shutdown (BMS_INIT_NONE)
        """
        if control_mode is not None:
            logger_mp.info("[BaseController] SetControlMode (shutdown) start -> %s", control_mode)
            res = self.client.SetControlMode(control_mode, timeout_ms)
            if res.success():
                logger_mp.info("[BaseController] SetControlMode (shutdown) -> %s", control_mode)
            else:
                logger_mp.warning("[BaseController] SetControlMode (shutdown) failed: %s", res.message())

        logger_mp.info("[BaseController] Torque OFF (shutdown) start")
        res = self.client.SetTorque(torque_type, timeout_ms)
        if res.success():
            logger_mp.info("[BaseController] Torque OFF (shutdown)")
        else:
            logger_mp.warning("[BaseController] Torque OFF failed: %s", res.message())

        # if bms_shutdown_type is not None:
        #     logger_mp.info("[BaseController] InitBms (shutdown %s) start", bms_shutdown_type)
        #     res = self.client.InitBms(bms_shutdown_type, timeout_ms)
        #     if res.success():
        #         logger_mp.info("[BaseController] BMS shutdown (%s) OK", bms_shutdown_type)
        #     else:
        #         logger_mp.warning("[BaseController] BMS shutdown failed: %s", res.message())



    def register_pose(self, name: str, q: np.ndarray) -> None:
        """포즈 등록: name -> q (shape, dtype 검증 포함 권장)"""
        q = np.asarray(q, dtype=float)
        if not hasattr(self, "_default_q"):
            raise RuntimeError("Default q is not initialized yet.")
        if q.shape != self._default_q.shape:
            raise ValueError(f"Pose '{name}' has invalid shape {q.shape}, expected {self._default_q.shape}.")
        self._poses[name] = q.copy()

    # ------------------------------------------------------------------ #
    # Subscription / buffer helpers
    def _extract_state_q(self, state: igc_sdk.LowState):
        """Return q array from joint_state or motor_state based on configuration."""
        if self._use_motor_state:
            return np.array([ms.q() for ms in state.motor_state()], dtype=np.float32)
        return np.array([js.q() for js in state.joint_state()], dtype=np.float32)

    def _on_low_state(self, state: igc_sdk.LowState):
        # JointState (PJS 공간) 기준으로 초기 자세 읽기
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
            logger_mp.info(
                "[BaseController] First LowState joint q sample: %s | motor q sample: %s",
                np.round(self._initial_q[:10], 3).tolist(),
                np.round(motor_q[:10], 3).tolist(),
            )
            if np.allclose(self._initial_q, 0.0) and not np.allclose(motor_q, 0.0):
                logger_mp.warning(
                    "[BaseController] joint_state is all zeros but motor_state has data (check publisher/kinematic mode)"
                )
            with self._ctrl_lock:
                # 첫 상태 수신 직후에는 목표를 항상 현재 자세로 동기화
                self._target_q = self._initial_q.copy()
                self._target_dq[:] = 0.0
                self._target_tau[:] = 0.0
            self._ready_event.set()
            logger_mp.info("[BaseController] First LowState received.")

        self.state_buffer.set(state)

    def wait_for_state(self, timeout: Optional[float] = 5.0) -> bool:
        """Block until first LowState arrives."""
        return self._ready_event.wait(timeout=timeout)

    # ------------------------------------------------------------------ #
    # Control loop
    def _publish_loop(self):
        while not self._stop_event.is_set():
            if self.wait_for_state(timeout=0.1):
                break

        while not self._stop_event.is_set():
            with self._ctrl_lock:
                q = self._target_q.copy()
                dq = self._target_dq.copy()
                tau = self._target_tau.copy()
                kp = self._kp.copy()
                kd = self._kd.copy()
                mode = self._kinematic_mode

            low_cmd_msg = igc_sdk.LowCmd()
            low_cmd_msg.kinematic_mode(mode)

            motors = low_cmd_msg.motors()
            for idx in range(NUM_MOTORS):
                motor_cmd = motors[idx]
                motor_cmd.id(idx)
                motor_cmd.q(float(q[idx]))
                motor_cmd.dq(float(dq[idx]))
                motor_cmd.tau(float(tau[idx]))
                motor_cmd.kp(float(kp[idx]))
                motor_cmd.kd(float(kd[idx]))

            if not self._dbg_publish_logged:
                logger_mp.info(
                    "[BaseController] First publish targets: q min/max=%.3f/%.3f, "
                    "kp min/max=%.3f/%.3f, kd min/max=%.3f/%.3f",
                    float(q.min()), float(q.max()),
                    float(kp.min()), float(kp.max()),
                    float(kd.min()), float(kd.max()),
                )
                self._dbg_publish_logged = True

            self.lowcmd_pub.write(low_cmd_msg)
            self.control_rate.sleep()

    # ------------------------------------------------------------------ #
    # Public APIs
    @staticmethod
    def _pad_array(arr):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.size < NUM_MOTORS:
            arr = np.pad(arr, (0, NUM_MOTORS - arr.size))
        return arr[:NUM_MOTORS]

    def set_joint_targets(self, q=None, dq=None, tau=None, kp=None, kd=None):
        """Update desired joint targets (arrays of len NUM_MOTORS)."""
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
        """Latch current measured q as targets (useful after torque ON)."""
        state = self.state_buffer.get()
        if state is None:
            return
        q_now = np.array([js.q() for js in state.joint_state()], dtype=np.float32)
        self.set_joint_targets(q=q_now, dq=np.zeros(NUM_MOTORS), tau=np.zeros(NUM_MOTORS))

    def set_kinematic_mode(self, mode):
        # 필요 시 MS로 바꿀 수는 있지만, 기본은 PJS 사용 권장
        with self._ctrl_lock:
            self._kinematic_mode = mode


    def _collect_joint_indices(
        self,
        leg: bool = True,
        waist: bool = True,
        arm: bool = True,
        neck: bool = True,
    ):
        """
        제어 대상이 될 조인트 인덱스 리스트를 생성한다.
        leg/waist/arm/neck 플래그로 그룹별 선택 가능.
        반환값은 오름차순으로 정렬된 int 리스트.
        """
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
        """Set tau/kp/kd to zero (broadcasted until updated)."""
        self.set_joint_targets(
            tau=np.zeros(NUM_MOTORS, dtype=np.float32),
            kp=np.zeros(NUM_MOTORS, dtype=np.float32),
            kd=np.zeros(NUM_MOTORS, dtype=np.float32),
        )

    # ------------------------------------------------------------------ #
    # Group-wise helpers (leg / waist / arm / neck) with velocity clip
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
            for idx, joint_id in enumerate(indices):
                self._target_q[joint_id] = q_target[idx]
                self._target_tau[joint_id] = tau_target[idx]

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

    # def get_joint_tau(self):
    #     state = self.state_buffer.get()
    #     if state is None:
    #         return None
    #     return np.array([js.tau_est() for js in state.joint_state()], dtype=np.float32)


    def get_joint_tau(self):
        state = self.state_buffer.get()
        if state is None:
            return None

        tau_raw = np.array([js.tau_est() for js in state.joint_state()], dtype=np.float32)

        # Kalman filtering
        if getattr(self, "_tau_kf", None) is None:
            return tau_raw

        tau_filt = self._tau_kf.step(tau_raw)
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
        """
        선택한 조인트 그룹(leg/waist/arm/neck)을
        현재 자세 -> 지정한 pose 로 duration 초 동안 선형 보간 이동.
        pose는 "default_pos" 같은 이름(str) 또는 q 벡터(np.ndarray) 지원.
        """
        if duration <= 0.0:
            duration = self.control_dt

        # 목표 q 결정
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

        # 현재 자세 읽기
        # q_start = self.get_joint_q()
        
        with self._ctrl_lock:
            q_start = self._target_q.copy()
            
        if q_start is None:
            return

        logger_mp.info(
            "[BaseController] move_to_pose(%s) start: current q min/max=%.3f/%.3f, goal min/max=%.3f/%.3f, duration=%.2fs",
            pose_name,
            float(q_start.min()),
            float(q_start.max()),
            float(q_goal.min()),
            float(q_goal.max()),
            duration,
        )

        joint_ids = self._collect_joint_indices(leg=leg, waist=waist, arm=arm, neck=neck)
        steps = max(1, int(duration / self.control_dt))

        # for step in range(steps + 1):
        for step in range(1, steps + 1):  # 0 제외

            if self._stop_event.is_set():
                break

            alpha = step / steps  # 0 -> 1
            with self._ctrl_lock:
                for jid in joint_ids:
                    q0 = q_start[jid]
                    q1 = q_goal[jid]
                    q_target = (1.0 - alpha) * q0 + alpha * q1

                    self._target_q[jid] = float(q_target)
                    self._target_dq[jid] = 0.0
                    self._target_tau[jid] = 0.0  # 기존 zero에서 하던 것도 통일

                    self._kp[jid] = float(self._kp_default[jid])
                    self._kd[jid] = float(self._kd_default[jid])

            time.sleep(self.control_dt)

        with self._ctrl_lock:
            delta = self._target_q - q_start
        self._dbg_publish_logged = False
        logger_mp.info(
            "[BaseController] move_to_pose(%s) done: target q min/max=%.3f/%.3f, max|delta|=%.3f",
            pose_name,
            float(self._target_q.min()),
            float(self._target_q.max()),
            float(np.max(np.abs(delta))),
        )

    def default_pos_state(
        self,
        pose: Union[str, np.ndarray, None] = None,
        leg: bool = True,
        waist: bool = True,
        arm: bool = True,
        neck: bool = True,
    ):
        """
        선택한 조인트 그룹의 타겟을 지정 pose로 즉시 세팅하고 유지한다.
        pose가 None이면 기본 자세(self._default_q)를 사용한다.

        G1_29_ArmController의 default_pos_state에 대응하지만,
        여기서는 리모컨 입력 대기는 하지 않는다.
        """
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
                self._kp[jid] = float(self._kp_default[jid])
                self._kd[jid] = float(self._kd_default[jid])
                self._target_tau[jid] = 0.0
        self._dbg_publish_logged = False
        logger_mp.info(
            "[BaseController] default_pos_state set: pose=%s, target q min/max=%.3f/%.3f",
            pose_name,
            float(self._target_q.min()),
            float(self._target_q.max()),
        )


    # ------------------------------------------------------------------ #
    # Service helpers (blocking)
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
        bms_shutdown_type=igc_sdk.BmsInitType.BMS_INIT_NONE,
        control_mode=igc_sdk.ControlMode.CONTROL_MODE_HIGH_LEVEL,
        timeout_ms: int = 30000,
    ):
        """Public wrapper for graceful shutdown (torque off → BMS off, optional mode switch)."""
        return self._service_shutdown(
            torque_type=torque_type,
            bms_shutdown_type=bms_shutdown_type,
            control_mode=control_mode,
            timeout_ms=timeout_ms,
        )

    # ------------------------------------------------------------------ #
    def stop(self):
        
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
                bms_shutdown_type=igc_sdk.BmsInitType.BMS_INIT_NONE,
                control_mode=igc_sdk.ControlMode.CONTROL_MODE_HIGH_LEVEL,
            )
        except Exception as exc:
            logger_mp.warning("[BaseController] Service shutdown failed: %s", exc)

        self._ctrl_thread.join(timeout=30.0)
        self._cleanup_dds()
        logger_mp.info("[BaseController] stop() called: move_to_zero_pos then shutdown")


# ---------------------------------------------------------------------- #
# Joint index: IGRIS-C Joint Motor Order (0-based index)
# Motor ID(1-based) / Joint Name 순서와 정확히 일치시킴.
class JointIndex(IntEnum):
    # Waist
    WAIST_YAW = 0          # Motor ID 1: Waist_Yaw
    WAIST_ROLL = 1         # Motor ID 2: Waist_Roll
    WAIST_PITCH = 2        # Motor ID 3: Waist_Pitch

    # Left leg
    L_HIP_PITCH = 3        # Motor ID 4: Hip_Pitch_L
    L_HIP_ROLL = 4         # Motor ID 5: Hip_Roll_L
    L_HIP_YAW = 5          # Motor ID 6: Hip_Yaw_L
    L_KNEE_PITCH = 6       # Motor ID 7: Knee_Pitch_L
    L_ANKLE_PITCH = 7      # Motor ID 8: Ankle_Pitch_L
    L_ANKLE_ROLL = 8       # Motor ID 9: Ankle_Roll_L

    # Right leg
    R_HIP_PITCH = 9        # Motor ID 10: Hip_Pitch_R
    R_HIP_ROLL = 10        # Motor ID 11: Hip_Roll_R
    R_HIP_YAW = 11         # Motor ID 12: Hip_Yaw_R
    R_KNEE_PITCH = 12      # Motor ID 13: Knee_Pitch_R
    R_ANKLE_PITCH = 13     # Motor ID 14: Ankle_Pitch_R
    R_ANKLE_ROLL = 14      # Motor ID 15: Ankle_Roll_R

    # Left arm
    L_SHOULDER_PITCH = 15  # Motor ID 16: Shoulder_Pitch_L
    L_SHOULDER_ROLL = 16   # Motor ID 17: Shoulder_Roll_L
    L_SHOULDER_YAW = 17    # Motor ID 18: Shoulder_Yaw_L
    L_ELBOW_PITCH = 18     # Motor ID 19: Elbow_Pitch_L
    L_WRIST_YAW = 19       # Motor ID 20: Wrist_Yaw_L
    L_WRIST_ROLL = 20      # Motor ID 21: Wrist_Roll_L
    L_WRIST_PITCH = 21     # Motor ID 22: Wrist_Pitch_L

    # Right arm
    R_SHOULDER_PITCH = 22  # Motor ID 23: Shoulder_Pitch_R
    R_SHOULDER_ROLL = 23   # Motor ID 24: Shoulder_Roll_R
    R_SHOULDER_YAW = 24    # Motor ID 25: Shoulder_Yaw_R
    R_ELBOW_PITCH = 25     # Motor ID 26: Elbow_Pitch_R
    R_WRIST_YAW = 26       # Motor ID 27: Wrist_Yaw_R
    R_WRIST_ROLL = 27      # Motor ID 28: Wrist_Roll_R
    R_WRIST_PITCH = 28     # Motor ID 29: Wrist_Pitch_R

    # Neck
    NECK_YAW = 29          # Motor ID 30: Neck_Yaw
    NECK_PITCH = 30        # Motor ID 31: Neck_Pitch

# Group indices (조인트 공간 기준)
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
