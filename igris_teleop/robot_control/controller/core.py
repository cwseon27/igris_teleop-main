import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np
import yaml

import igris_c_sdk as igc_sdk
import logging_mp
from enum import IntEnum

from igris_teleop.core.project_paths import JOINT_SETTING_PATH
from ...core.rate import Rate
from ..filters.torque_kalman import TorqueKalmanBank
from .kinematics import (
    ARM_INDICES,
    LEG_INDICES,
    NECK_INDICES,
    NUM_MOTORS,
    WAIST_INDICES,
    JointIndex,
)


logger_mp = logging_mp.get_logger(__name__, level=logging_mp.INFO)


if not hasattr(igc_sdk, "KinematicMode"):
    class _CompatKinematicMode(IntEnum):
        MS = 0
        PJS = 1

    class _CompatBmsInitType(IntEnum):
        BMS_INIT_NONE = 0
        BMS_INIT = 1
        MOTOR_INIT = 2
        BMS_AND_MOTOR_INIT = 3
        BMS_OFF = 4

    class _CompatTorqueType(IntEnum):
        TORQUE_NONE = 0
        TORQUE_ON = 1
        TORQUE_OFF = 2

    class _CompatControlMode(IntEnum):
        CONTROL_MODE_HIGH_LEVEL = 0
        CONTROL_MODE_LOW_LEVEL = 1

    igc_sdk.KinematicMode = _CompatKinematicMode
    igc_sdk.BmsInitType = _CompatBmsInitType
    igc_sdk.TorqueType = _CompatTorqueType
    igc_sdk.ControlMode = _CompatControlMode

kTopicLowCommand = "rt/lowcmd"
kTopicLowState = "rt/lowstate"
DEFAULT_ROBOT_ROS_NAMESPACE = "igris_c_IG05"
DEFAULT_VEL_LIMIT = 5.0
POSE_INTERPOLATION_UPDATE_HZ = 200.0
POSE_MAX_PHASE_STEP_S = 1.0 / POSE_INTERPOLATION_UPDATE_HZ
POSE_TRACKING_SOFT_ERROR_RAD = float(os.getenv("IGRIS_POSE_TRACKING_SOFT_ERROR_RAD", "0.08"))
POSE_TRACKING_HARD_ERROR_RAD = float(os.getenv("IGRIS_POSE_TRACKING_HARD_ERROR_RAD", "0.25"))
POSE_MEASURED_START_MAX_AGE_S = float(os.getenv("IGRIS_POSE_STATE_MAX_AGE_S", "0.25"))
LOWLEVEL_TORQUE_TO_MODE_DELAY_S = 2.0
LOWLEVEL_MODE_SETTLE_DELAY_S = 2.0
ROS2_HOME_SETTLE_DELAY_S = float(os.getenv("IGRIS_ROS2_HOME_SETTLE_S", "8.0"))
ROS2_BMS_INIT_SETTLE_DELAY_S = float(os.getenv("IGRIS_ROS2_BMS_INIT_SETTLE_S", "8.0"))
RESERVED_JOINT_PROFILE_KEYS = frozenset(
    {"kp", "kd", "default_dof_pos", "waypoint_1", "waypoint_2", "waypoint_3", "joint_order", "joint_groups"}
)


def _clear_real_robot_dds_profile_for_nonzero_domain(domain_id: int) -> None:
    """Prevent the real-robot LAN CycloneDDS profile from breaking sim DDS."""
    if int(domain_id) == 0:
        return
    uri = os.environ.get("CYCLONEDDS_URI", "")
    if "cyclonedds_igris_lan.xml" not in uri:
        return
    logger_mp.info(
        "[ControllerCore] clearing real-robot CYCLONEDDS_URI for non-real domain_id=%s",
        domain_id,
    )
    os.environ.pop("CYCLONEDDS_URI", None)


class LowStateBuffer:
    """Thread-safe buffer for the latest LowState."""

    def __init__(self):
        self._data = None
        self._updated_at: float | None = None
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._data

    def age_s(self) -> float | None:
        """Return the local age of the latest LowState sample."""
        with self._lock:
            updated_at = self._updated_at
        if updated_at is None:
            return None
        return max(0.0, time.monotonic() - updated_at)

    def set(self, data):
        with self._lock:
            self._data = data
            self._updated_at = time.monotonic()


class _ServiceResult:
    def __init__(self, success: bool, message: str = "", error_code: int = 0) -> None:
        self._success = bool(success)
        self._message = str(message)
        self._error_code = int(error_code)

    def success(self) -> bool:
        return self._success

    def message(self) -> str:
        return self._message

    def error_code(self) -> int:
        return self._error_code


def _enum_value(value, fallback: int = 0) -> int:
    raw = getattr(value, "value", value)
    try:
        return int(raw)
    except Exception:
        return int(fallback)


def _ros_namespace() -> str:
    ns = os.getenv("IGRIS_ROBOT_ROS_NAMESPACE") or os.getenv("IGRIS_ROBOT_DDS_NAMESPACE")
    return (ns or DEFAULT_ROBOT_ROS_NAMESPACE).strip().strip("/")


def _is_noop_success(result: _ServiceResult) -> bool:
    message = result.message().lower()
    return "already" in message and "no update needed" in message


class _Ros2LowCmdPublisher:
    def __init__(
        self,
        node,
        topic: str,
        qos_profile,
        LowCmd,
        *,
        relay_process: subprocess.Popen | None = None,
        fallback_topic: str | None = None,
        stop_relay: Callable[[], None] | None = None,
    ) -> None:
        self._publisher = node.create_publisher(LowCmd, topic, qos_profile)
        self._fallback_publisher = (
            None
            if fallback_topic is None
            else node.create_publisher(LowCmd, fallback_topic, qos_profile)
        )
        self._LowCmd = LowCmd
        self._relay_process = relay_process
        self._stop_relay = stop_relay
        self._relay_failure_logged = False

    def write(self, msg) -> None:
        if self._relay_process is not None and self._relay_process.poll() is not None:
            if not self._relay_failure_logged:
                logger_mp.error(
                    "[ControllerCore] native LowCmd relay exited with code %s; "
                    "falling back to direct Python publication",
                    self._relay_process.returncode,
                )
                self._relay_failure_logged = True
            if self._fallback_publisher is not None:
                self._fallback_publisher.publish(msg)
                return
        self._publisher.publish(msg)

    def stop(self) -> None:
        if self._stop_relay is not None:
            self._stop_relay()


class _Ros2LowStateSubscriber:
    def __init__(self, node, topic: str, qos_profile, LowState, callback) -> None:
        self._subscription = node.create_subscription(LowState, topic, callback, qos_profile)

    def stop(self) -> None:
        return


class _Ros2RobotClient:
    def __init__(self, *, domain_id: int, namespace: str) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
        from igris_c_sdk.msg import LowCmd, LowState
        from igris_c_sdk.srv import BmsInitCmd, ControlModeCommandRequest, TorqueCmd

        self.rclpy = rclpy
        self.LowCmd = LowCmd
        self.LowState = LowState
        self.BmsInitCmd = BmsInitCmd
        self.TorqueCmd = TorqueCmd
        self.ControlModeCommandRequest = ControlModeCommandRequest
        self._namespace = namespace.strip("/")
        self._domain_id = int(domain_id)
        self._relay_process: subprocess.Popen | None = None
        self._relay_log_thread: threading.Thread | None = None
        self._context = rclpy.Context()
        rclpy.init(args=None, context=self._context, domain_id=int(domain_id))
        self.node = rclpy.create_node(
            f"igris_teleop_control_{os.getpid()}",
            context=self._context,
        )
        self.qos_best_effort = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        # A single executor worker is sufficient for depth-1 LowState and the
        # bounded service calls.  Two Python executor workers only added GIL
        # contention with the command and ControlWorker loops.
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="igris-ros2-control-spin",
            daemon=True,
        )
        self._spin_thread.start()
        self._bms_client = self.node.create_client(
            BmsInitCmd,
            self._topic("rt/service/bms_init"),
        )
        self._torque_client = self.node.create_client(
            TorqueCmd,
            self._topic("rt/service/torque"),
        )
        self._mode_client = self.node.create_client(
            ControlModeCommandRequest,
            self._topic("rt/service/control_mode"),
        )

    def _topic(self, suffix: str) -> str:
        return f"/{self._namespace}/{suffix.lstrip('/')}"

    def create_lowstate_subscriber(self, callback):
        return _Ros2LowStateSubscriber(
            self.node,
            self._topic(kTopicLowState),
            self.qos_best_effort,
            self.LowState,
            callback,
        )

    def create_lowcmd_publisher(self):
        robot_topic = self._topic(kTopicLowCommand)
        relay_process = self._start_lowcmd_relay(robot_topic)
        if relay_process is None:
            return _Ros2LowCmdPublisher(
                self.node,
                robot_topic,
                self.qos_best_effort,
                self.LowCmd,
            )
        desired_topic = f"/igris_teleop/control_{os.getpid()}/lowcmd_desired"
        return _Ros2LowCmdPublisher(
            self.node,
            desired_topic,
            self.qos_best_effort,
            self.LowCmd,
            relay_process=relay_process,
            fallback_topic=robot_topic,
            stop_relay=self._stop_lowcmd_relay,
        )

    @staticmethod
    def _relay_enabled() -> bool:
        return os.getenv("IGRIS_NATIVE_LOWCMD_RELAY", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _default_relay_executable() -> Path:
        return (
            Path(__file__).resolve().parents[3]
            / "ros_ws"
            / "install"
            / "igris_lowcmd_relay"
            / "lib"
            / "igris_lowcmd_relay"
            / "lowcmd_relay_node"
        )

    def _start_lowcmd_relay(self, robot_topic: str) -> subprocess.Popen | None:
        if not self._relay_enabled():
            logger_mp.warning("[ControllerCore] native LowCmd relay disabled by environment")
            return None
        raw_executable = os.getenv("IGRIS_LOWCMD_RELAY_EXECUTABLE")
        executable = Path(raw_executable).expanduser() if raw_executable else self._default_relay_executable()
        if not executable.is_file():
            logger_mp.warning(
                "[ControllerCore] native LowCmd relay not installed at %s; "
                "using direct Python publication",
                executable,
            )
            return None

        desired_topic = f"/igris_teleop/control_{os.getpid()}/lowcmd_desired"
        command = [
            str(executable),
            "--ros-args",
            "-p",
            f"input_topic:={desired_topic}",
            "-p",
            f"output_topic:={robot_topic}",
            "-p",
            f"parent_pid:={os.getpid()}",
        ]
        try:
            relay_env = os.environ.copy()
            relay_env["ROS_DOMAIN_ID"] = str(self._domain_id)
            process = subprocess.Popen(
                command,
                env=relay_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            logger_mp.warning("[ControllerCore] failed to start native LowCmd relay: %s", exc)
            return None

        self._relay_process = process

        def _pump_output() -> None:
            stream = process.stdout
            if stream is None:
                return
            for line in stream:
                line = line.rstrip()
                if line:
                    logger_mp.info("[LowCmdRelay] %s", line)

        self._relay_log_thread = threading.Thread(
            target=_pump_output,
            name="igris-lowcmd-relay-log",
            daemon=True,
        )
        self._relay_log_thread.start()
        time.sleep(0.05)
        if process.poll() is not None:
            logger_mp.error(
                "[ControllerCore] native LowCmd relay failed during startup (code=%s)",
                process.returncode,
            )
            self._stop_lowcmd_relay()
            return None
        logger_mp.info(
            "[ControllerCore] native 300Hz LowCmd relay started pid=%d input=%s output=%s",
            process.pid,
            desired_topic,
            robot_topic,
        )
        return process

    def _stop_lowcmd_relay(self) -> None:
        process = self._relay_process
        self._relay_process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        if process is not None and process.stdout is not None:
            try:
                process.stdout.close()
            except Exception:
                pass
        log_thread = self._relay_log_thread
        self._relay_log_thread = None
        if log_thread is not None and log_thread is not threading.current_thread():
            log_thread.join(timeout=0.5)

    def _call(self, client, request, timeout_ms: int) -> _ServiceResult:
        deadline = time.monotonic() + max(0.1, float(timeout_ms) / 1000.0)
        while not client.wait_for_service(timeout_sec=0.25):
            if time.monotonic() >= deadline:
                return _ServiceResult(False, "Request timeout", -1)
        future = client.call_async(request)
        while not future.done():
            if time.monotonic() >= deadline:
                return _ServiceResult(False, "Request timeout", -1)
            time.sleep(0.01)
        try:
            response = future.result()
        except Exception as exc:
            return _ServiceResult(False, str(exc), -1)
        return _ServiceResult(
            bool(getattr(response, "success", False)),
            str(getattr(response, "message", "")),
            int(getattr(response, "error_code", 0)),
        )

    def Init(self) -> None:
        return

    def SetTimeout(self, _timeout_s: float) -> None:
        return

    def InitBms(self, init_type, timeout_ms: int):
        req = self.BmsInitCmd.Request()
        req.request_id = f"igris_teleop_{uuid.uuid4().hex}"
        req.init = _enum_value(init_type)
        return self._call(self._bms_client, req, timeout_ms)

    def SetTorque(self, torque_type, timeout_ms: int):
        req = self.TorqueCmd.Request()
        req.request_id = f"igris_teleop_{uuid.uuid4().hex}"
        req.torque = _enum_value(torque_type)
        return self._call(self._torque_client, req, timeout_ms)

    def _start_home_preset(self, timeout_ms: int) -> _ServiceResult:
        req = self.ControlModeCommandRequest.Request()
        req.request_id = f"igris_teleop_home_{uuid.uuid4().hex}"
        req.command_type = self.ControlModeCommandRequest.Request.CONTROL_MODE_CMD_MOTION_PRESET
        req.preset_id = "HOME"
        req.is_cyclic = False
        return self._call(self._mode_client, req, timeout_ms)

    def SetControlMode(self, control_mode, timeout_ms: int):
        req = self.ControlModeCommandRequest.Request()
        req.request_id = f"igris_teleop_{uuid.uuid4().hex}"
        mode_value = _enum_value(control_mode)
        is_low_level = mode_value == _enum_value(igc_sdk.ControlMode.CONTROL_MODE_LOW_LEVEL)
        if is_low_level:
            req.command_type = self.ControlModeCommandRequest.Request.CONTROL_MODE_CMD_LOW_LEVEL_JOINT_CONTROL
        else:
            req.command_type = self.ControlModeCommandRequest.Request.CONTROL_MODE_CMD_HIGH_LEVEL_JOINT_CONTROL
        req.preset_id = ""
        req.is_cyclic = False
        result = self._call(self._mode_client, req, timeout_ms)
        stopped_state = "stopped state" in result.message().lower()
        home_on_stopped = os.getenv("IGRIS_ROS2_HOME_ON_STOPPED", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if _is_noop_success(result):
            return _ServiceResult(True, result.message(), result.error_code())
        if not (is_low_level and stopped_state and home_on_stopped):
            return result

        logger_mp.warning(
            "[ControllerCore] LOW_LEVEL rejected because robot is stopped; starting HOME preset before retry"
        )
        home_result = self._start_home_preset(timeout_ms)
        if not home_result.success():
            return _ServiceResult(
                False,
                f"{result.message()} HOME preset failed: {home_result.message()}",
                home_result.error_code(),
            )
        logger_mp.info("[ControllerCore] HOME preset accepted: %s", home_result.message())
        if ROS2_HOME_SETTLE_DELAY_S > 0.0:
            logger_mp.info(
                "[ControllerCore] waiting %.2fs after HOME preset before LOW_LEVEL retry",
                ROS2_HOME_SETTLE_DELAY_S,
            )
            time.sleep(ROS2_HOME_SETTLE_DELAY_S)
        req.request_id = f"igris_teleop_{uuid.uuid4().hex}"
        retry_result = self._call(self._mode_client, req, timeout_ms)
        if _is_noop_success(retry_result):
            return _ServiceResult(True, retry_result.message(), retry_result.error_code())
        return retry_result

    def close(self) -> None:
        self._stop_lowcmd_relay()
        try:
            self._executor.shutdown(timeout_sec=1.0)
        except Exception:
            pass
        try:
            self._spin_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            self.rclpy.shutdown(context=self._context)
        except Exception:
            pass


def _robot_control_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_joint_profile_path(raw_path: Optional[str]) -> Path:
    default_path = JOINT_SETTING_PATH
    if raw_path is None:
        return default_path

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    candidates = (
        _robot_control_dir() / raw_path,
        _repo_root() / raw_path,
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _load_joint_profile_payload(cfg_path: Path) -> dict:
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Joint profile not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as file_obj:
        cfg = yaml.safe_load(file_obj) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Joint profile must be a mapping: {cfg_path}")
    return cfg


def _require_joint_profile_array(cfg: dict, key: str) -> np.ndarray:
    arr = np.asarray(cfg[key], dtype=np.float32).reshape(-1)
    if arr.size != NUM_MOTORS:
        raise ValueError(f"{key} length {arr.size} != {NUM_MOTORS}")
    return arr


def _optional_joint_profile_array(cfg: dict, key: str) -> np.ndarray | None:
    if key not in cfg:
        return None
    return _require_joint_profile_array(cfg, key)


def _extract_optional_joint_poses(cfg: dict) -> Dict[str, np.ndarray]:
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


def load_joint_profile(cfg_path: Path):
    cfg = _load_joint_profile_payload(cfg_path)

    try:
        kp = _require_joint_profile_array(cfg, "kp")
        kd = _require_joint_profile_array(cfg, "kd")
        default_q = _require_joint_profile_array(cfg, "default_dof_pos")
        waypoint_1 = _require_joint_profile_array(cfg, "waypoint_1")
        waypoint_2 = _require_joint_profile_array(cfg, "waypoint_2")
    except KeyError as exc:
        raise KeyError(f"Missing key in joint profile: {exc}") from exc

    return kp, kd, default_q, waypoint_1, waypoint_2


def load_joint_profile_waypoint_3(cfg_path: Path) -> np.ndarray | None:
    cfg = _load_joint_profile_payload(cfg_path)
    return _optional_joint_profile_array(cfg, "waypoint_3")


def load_optional_joint_poses(cfg_path: Path) -> Dict[str, np.ndarray]:
    return _extract_optional_joint_poses(_load_joint_profile_payload(cfg_path))


class ControllerCore:
    """DDS/service/state/publish-loop base controller."""

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
        joint_profile_path: Optional[str] = None,
        start_thread: bool = True,
        control_backend: str | None = None,
        robot_namespace: str | None = None,
    ):
        logger_mp.info("[ControllerCore] Initializing (domain_id=%s)...", domain_id)

        self._control_hz = float(control_hz)
        self.control_rate = Rate(self._control_hz)
        self.control_dt = 1.0 / self._control_hz
        self._domain_id = int(domain_id)
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._ctrl_lock = threading.Lock()
        self._use_motor_state = use_motor_state
        self._kinematic_mode = kinematic_mode
        self._dbg_publish_logged = False
        self._ctrl_thread = None
        self._active_pose_trajectory: dict | None = None
        self._dds_cleaned = False
        self.lowstate_sub = None
        self.lowcmd_pub = None
        self._ros2_client = None
        self._control_backend = (control_backend or os.getenv("IGRIS_ROBOT_CONTROL_BACKEND") or "").strip().lower()
        if not self._control_backend:
            self._control_backend = "ros2" if int(domain_id) == 0 else "sdk"
        self._robot_namespace = (robot_namespace or _ros_namespace()).strip().strip("/")

        if self._control_backend == "ros2":
            self.channel_factory = None
            self._ros2_client = _Ros2RobotClient(
                domain_id=int(domain_id),
                namespace=self._robot_namespace,
            )
            self.client = self._ros2_client
            logger_mp.info(
                "[ControllerCore] ROS2 robot backend enabled namespace=/%s",
                self._robot_namespace,
            )
        else:
            _clear_real_robot_dds_profile_for_nonzero_domain(int(domain_id))
            self.channel_factory = igc_sdk.ChannelFactory.Instance()
            self.channel_factory.Init(int(domain_id))
            self.client = igc_sdk.IgrisC_Client()
            self.client.Init()
            self.client.SetTimeout(service_timeout_ms / 1000.0)

        profile_path = resolve_joint_profile_path(joint_profile_path)
        kp_default, kd_default, default_q, waypoint_1, waypoint_2 = load_joint_profile(profile_path)
        waypoint_3 = load_joint_profile_waypoint_3(profile_path)
        optional_poses = load_optional_joint_poses(profile_path)

        self._kp_default = kp_default.copy()
        self._kd_default = kd_default.copy()
        self._kp = kp_default.copy()
        self._kd = kd_default.copy()

        self._default_q = default_q.astype(np.float32)
        self._waypoint_1 = waypoint_1.astype(np.float32)
        self._waypoint_2 = waypoint_2.astype(np.float32)
        self._waypoint_3 = None if waypoint_3 is None else waypoint_3.astype(np.float32)

        self._poses: Dict[str, np.ndarray] = {}
        self.register_pose("default_pos", self._default_q)
        self.register_pose("zero_pos", np.zeros_like(self._default_q))
        self.register_pose("waypoint_1", self._waypoint_1)
        self.register_pose("waypoint_2", self._waypoint_2)
        if self._waypoint_3 is not None:
            self.register_pose("waypoint_3", self._waypoint_3)
        for pose_name, pose_q in optional_poses.items():
            self.register_pose(pose_name, pose_q)
        if optional_poses:
            logger_mp.info("[ControllerCore] registered extra poses: %s", sorted(optional_poses))

        self._target_q = np.zeros_like(self._default_q)
        self._target_dq = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._target_tau = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._initial_q = None

        self.leg_velocity_limit = DEFAULT_VEL_LIMIT
        self.waist_velocity_limit = DEFAULT_VEL_LIMIT
        self.arm_velocity_limit = DEFAULT_VEL_LIMIT
        self.neck_velocity_limit = DEFAULT_VEL_LIMIT

        self._tau_kf = TorqueKalmanBank(n_joints=NUM_MOTORS, q=3.0, r=20.0, P0=1e3)

        self.state_buffer = LowStateBuffer()

        if auto_service_init:
            if service_init_delay > 0.0:
                time.sleep(service_init_delay)
            self._service_bootstrap(
                bms_init_type=bms_init_type,
                torque_type=torque_type,
                control_mode=control_mode,
                timeout_ms=service_timeout_ms,
            )

        if self._control_backend == "ros2":
            self.lowstate_sub = self._ros2_client.create_lowstate_subscriber(self._on_low_state)
        else:
            self.lowstate_sub = igc_sdk.LowStateSubscriber(kTopicLowState)
            try:
                # Initialize service APIs before attaching the Python LowState callback.
                # In the current SDK build, blocking service calls can time out if a high-rate
                # Python subscriber callback is already active on the same participant.
                self.lowstate_sub.init(self._on_low_state)
            except Exception as exc:
                self._cleanup_dds()
                raise RuntimeError("Failed to init LowStateSubscriber") from exc

        if self._control_backend != "ros2":
            self.lowcmd_pub = igc_sdk.LowCmdPublisher(kTopicLowCommand)
            try:
                self.lowcmd_pub.init()
            except Exception as exc:
                self._cleanup_dds()
                raise RuntimeError("Failed to init LowCmdPublisher") from exc
        else:
            self.lowcmd_pub = self._ros2_client.create_lowcmd_publisher()

        if start_thread:
            self._start_control_loop()

        logger_mp.info("[ControllerCore] Init OK, waiting for first LowState...")

    def _cleanup_dds(self) -> None:
        if self._dds_cleaned:
            return
        self._dds_cleaned = True

        if self.lowstate_sub is not None:
            try:
                self.lowstate_sub.stop()
            except Exception:
                logger_mp.debug("[ControllerCore] failed to stop LowStateSubscriber", exc_info=True)

        if self.lowcmd_pub is not None:
            try:
                self.lowcmd_pub.stop()
            except Exception:
                logger_mp.debug("[ControllerCore] failed to stop LowCmdPublisher", exc_info=True)

        if self._ros2_client is not None:
            try:
                self._ros2_client.close()
            except Exception:
                logger_mp.debug("[ControllerCore] failed to close ROS2 robot backend", exc_info=True)
        elif self.channel_factory is not None:
            try:
                self.channel_factory.Release()
            except Exception:
                logger_mp.debug("[ControllerCore] failed to release ChannelFactory", exc_info=True)

    def _start_control_loop(self) -> None:
        if self._ctrl_thread is not None:
            return
        # Rate may have been constructed before blocking service bootstrap.
        # Reset its deadline here so the publisher does not run a long
        # catch-up loop and starve the shutdown interpolation thread.
        self.control_rate = Rate(self._control_hz)
        self._ctrl_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._ctrl_thread.start()

    def _prepare_service_shutdown(
        self,
        *,
        ctrl_thread_join_timeout_s: float,
        context: str,
    ) -> None:
        self._stop_event.set()

        if self._ctrl_thread is not None:
            self._ctrl_thread.join(timeout=max(0.0, float(ctrl_thread_join_timeout_s)))
            if self._ctrl_thread.is_alive():
                logger_mp.warning(
                    "[ControllerCore] control thread still alive before %s join timeout %.2fs",
                    context,
                    float(ctrl_thread_join_timeout_s),
                )

        time.sleep(min(0.1, max(self.control_dt * 2.0, 0.02)))

    def _run_service_shutdown_helper(
        self,
        *,
        timeout_ms: int,
        service_call_timeout_s: float | None,
        context: str,
    ) -> None:
        started_at = time.monotonic()
        cmd = [
            sys.executable,
            "-m",
            "igris_teleop.robot_control.controller.shutdown_helper",
            "--domain-id",
            str(self._domain_id),
            "--timeout-ms",
            str(int(timeout_ms)),
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(_repo_root()),
                capture_output=True,
                text=True,
                timeout=None if service_call_timeout_s is None else float(service_call_timeout_s),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{context} timed out after {float(service_call_timeout_s):.2f}s"
            ) from exc

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        payload = None
        if stdout:
            last_line = stdout.splitlines()[-1].strip()
            try:
                payload = json.loads(last_line)
            except json.JSONDecodeError:
                payload = None

        if result.returncode != 0:
            detail = None
            if isinstance(payload, dict):
                detail = payload.get("error") or payload.get("summary")
            if detail is None:
                detail = stderr or stdout or f"helper exited with code {result.returncode}"
            raise RuntimeError(str(detail))

        if isinstance(payload, dict) and not bool(payload.get("mode_success", True)):
            logger_mp.warning(
                "[ControllerCore] SetControlMode (shutdown) failed in helper, but Torque OFF succeeded: %s",
                payload.get("mode_message", "<no message>"),
            )
            logger_mp.info(
                "[ControllerCore] %s completed via helper in %.2fs: Torque OFF=%s",
                context,
                time.monotonic() - started_at,
                payload.get("torque_message", "<no message>"),
            )
            return

        if isinstance(payload, dict):
            logger_mp.info(
                "[ControllerCore] %s succeeded via helper in %.2fs: %s",
                context,
                time.monotonic() - started_at,
                payload.get("summary", "Torque OFF succeeded"),
            )
            return

        logger_mp.info(
            "[ControllerCore] %s succeeded via helper in %.2fs",
            context,
            time.monotonic() - started_at,
        )

    def _map_command_q(self, q_pjs: np.ndarray) -> np.ndarray:
        return np.asarray(q_pjs, dtype=np.float32)

    def _clip_ms_command_q(self, q_ms: np.ndarray) -> np.ndarray:
        return np.asarray(q_ms, dtype=np.float32)

    def _build_command_q(self, q_target_pjs: np.ndarray, mode) -> np.ndarray:
        q_target_pjs = np.asarray(q_target_pjs, dtype=np.float32)
        if mode != igc_sdk.KinematicMode.MS:
            return q_target_pjs.copy()
        q_ms = self._map_command_q(q_target_pjs)
        return self._clip_ms_command_q(q_ms)

    def _call_service_with_retry(self, fn, name: str, attempts: int = 3, delay_sec: float = 1.0):
        last_res = None
        for attempt in range(1, attempts + 1):
            res = fn()
            last_res = res
            if res.success():
                logger_mp.info("[ControllerCore] %s succeeded: %s", name, res.message())
                return res
            logger_mp.warning(
                "[ControllerCore] %s failed (%d/%d): %s",
                name,
                attempt,
                attempts,
                res.message(),
            )
            if attempt < attempts:
                time.sleep(delay_sec)
        raise RuntimeError(f"{name} failed after {attempts} attempts: {last_res.message()}")

    def _service_bootstrap(self, bms_init_type, torque_type, control_mode, timeout_ms: int):
        skip_ros2_bms = (
            self._control_backend == "ros2"
            and os.getenv("IGRIS_ROS2_SKIP_BMS_INIT", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if skip_ros2_bms:
            logger_mp.info(
                "[ControllerCore] skipping InitBms(%s) on ROS2 backend; using Torque/ControlMode services only",
                bms_init_type,
            )
        else:
            self._call_service_with_retry(
                lambda: self.client.InitBms(bms_init_type, timeout_ms),
                f"InitBms({bms_init_type})",
            )
        try:
            self._call_service_with_retry(
                lambda: self.client.SetTorque(torque_type, timeout_ms),
                f"SetTorque({torque_type})",
            )
        except RuntimeError:
            if not skip_ros2_bms or torque_type != igc_sdk.TorqueType.TORQUE_ON:
                raise
            logger_mp.warning(
                "[ControllerCore] SetTorque(%s) failed after skipped InitBms; running InitBms(%s) then retrying",
                torque_type,
                bms_init_type,
            )
            init_res = self.client.InitBms(bms_init_type, timeout_ms)
            if init_res.success():
                logger_mp.info("[ControllerCore] InitBms(%s) succeeded: %s", bms_init_type, init_res.message())
            else:
                logger_mp.warning(
                    "[ControllerCore] InitBms(%s) reported failure during fallback: %s; waiting for robot state anyway",
                    bms_init_type,
                    init_res.message(),
                )
            if ROS2_BMS_INIT_SETTLE_DELAY_S > 0.0:
                logger_mp.info(
                    "[ControllerCore] waiting %.2fs after InitBms fallback before Torque ON retry",
                    ROS2_BMS_INIT_SETTLE_DELAY_S,
                )
                time.sleep(ROS2_BMS_INIT_SETTLE_DELAY_S)
            self._call_service_with_retry(
                lambda: self.client.SetTorque(torque_type, timeout_ms),
                f"SetTorque({torque_type})",
            )
        if (
            torque_type == igc_sdk.TorqueType.TORQUE_ON
            and control_mode == igc_sdk.ControlMode.CONTROL_MODE_LOW_LEVEL
            and LOWLEVEL_TORQUE_TO_MODE_DELAY_S > 0.0
        ):
            logger_mp.info(
                "[ControllerCore] waiting %.2fs after Torque ON before LOW_LEVEL",
                LOWLEVEL_TORQUE_TO_MODE_DELAY_S,
            )
            time.sleep(LOWLEVEL_TORQUE_TO_MODE_DELAY_S)
        self._call_service_with_retry(
            lambda: self.client.SetControlMode(control_mode, timeout_ms),
            f"SetControlMode({control_mode})",
        )
        if (
            control_mode == igc_sdk.ControlMode.CONTROL_MODE_LOW_LEVEL
            and LOWLEVEL_MODE_SETTLE_DELAY_S > 0.0
        ):
            logger_mp.info(
                "[ControllerCore] waiting %.2fs after LOW_LEVEL mode switch",
                LOWLEVEL_MODE_SETTLE_DELAY_S,
            )
            time.sleep(LOWLEVEL_MODE_SETTLE_DELAY_S)

    def _service_shutdown(
        self,
        torque_type=igc_sdk.TorqueType.TORQUE_OFF,
        control_mode=igc_sdk.ControlMode.CONTROL_MODE_HIGH_LEVEL,
        timeout_ms: int = 30000,
    ):
        shutdown_mode_error: str | None = None
        if control_mode is not None:
            res = self.client.SetControlMode(control_mode, timeout_ms)
            if not res.success():
                shutdown_mode_error = str(res.message())
                logger_mp.warning(
                    "[ControllerCore] SetControlMode (shutdown) failed: %s",
                    shutdown_mode_error,
                )

        res = self.client.SetTorque(torque_type, timeout_ms)
        if not res.success():
            torque_error = str(res.message())
            if shutdown_mode_error is None:
                raise RuntimeError(f"Torque OFF failed: {torque_error}")
            raise RuntimeError(
                "Shutdown service calls failed: "
                f"SetControlMode (shutdown): {shutdown_mode_error}; "
                f"Torque OFF: {torque_error}"
            )

        logger_mp.info("[ControllerCore] Torque OFF succeeded: %s", res.message())

        if shutdown_mode_error is not None:
            logger_mp.warning(
                "[ControllerCore] Torque OFF succeeded even though SetControlMode (shutdown) failed: %s",
                shutdown_mode_error,
            )

    def _run_blocking_shutdown_call(
        self,
        label: str,
        fn: Callable[[], None],
        *,
        timeout_s: float | None,
    ) -> bool:
        if timeout_s is None:
            fn()
            return True

        timeout_s = float(timeout_s)
        if timeout_s <= 0.0:
            fn()
            return True

        done = threading.Event()
        errors: list[BaseException] = []

        def _worker() -> None:
            try:
                fn()
            except BaseException as exc:  # pragma: no cover - defensive for SDK calls
                errors.append(exc)
            finally:
                done.set()

        t = threading.Thread(target=_worker, name=f"controller-{label}", daemon=True)
        t.start()
        t.join(timeout=timeout_s)
        if not done.is_set():
            logger_mp.warning(
                "[ControllerCore] %s timed out after %.2fs; continuing shutdown",
                label,
                timeout_s,
            )
            return False

        if errors:
            raise errors[0]

        return True

    def register_pose(self, name: str, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        if not hasattr(self, "_default_q"):
            raise RuntimeError("Default q is not initialized yet.")
        if q.shape != self._default_q.shape:
            raise ValueError(f"Pose '{name}' has invalid shape {q.shape}, expected {self._default_q.shape}.")
        self._poses[name] = q.copy()

    @staticmethod
    def _msg_field(obj, name: str):
        value = getattr(obj, name)
        return value() if callable(value) else value

    def _state_joint_seq(self, state) -> list:
        return list(self._msg_field(state, "joint_state"))

    def _state_motor_seq(self, state) -> list:
        return list(self._msg_field(state, "motor_state"))

    def _extract_state_q(self, state) -> np.ndarray:
        if self._use_motor_state:
            return np.array([self._msg_field(ms, "q") for ms in self._state_motor_seq(state)], dtype=np.float32)
        return np.array([self._msg_field(js, "q") for js in self._state_joint_seq(state)], dtype=np.float32)

    def _on_low_state(self, state):
        self.state_buffer.set(state)

        if not self._ready_event.is_set():
            self._initial_q = self._extract_state_q(state)
            first_joint_tau_est = np.array(
                [self._msg_field(js, "tau_est") for js in self._state_joint_seq(state)],
                dtype=np.float32,
            )
            self._tau_kf.reset_with_measurement(first_joint_tau_est)

            with self._ctrl_lock:
                self._target_q = self._initial_q.copy()
                self._target_dq[:] = 0.0
                self._target_tau[:] = 0.0

            self._ready_event.set()
            logger_mp.info("[ControllerCore] First LowState received.")

    def wait_for_state(self, timeout: Optional[float] = 5.0) -> bool:
        return self._ready_event.wait(timeout=timeout)

    def _publish_loop(self):
        while not self._stop_event.is_set():
            if self.wait_for_state(timeout=0.1):
                break

        if self._control_backend == "ros2":
            low_cmd_msg = self._ros2_client.LowCmd()
            motors = low_cmd_msg.motors
            for idx, motor_cmd in enumerate(motors):
                motor_cmd.id = int(idx)
        else:
            low_cmd_msg = igc_sdk.LowCmd()
            motors = low_cmd_msg.motors()
            for idx, motor_cmd in enumerate(motors):
                motor_cmd.id(idx)

        sequence = 0
        diag_started_at = time.perf_counter()
        previous_publish_at: float | None = None
        previous_q_pjs: np.ndarray | None = None
        diag_publishes = 0
        diag_max_gap_s = 0.0
        diag_max_q_step = 0.0
        while not self._stop_event.is_set():
            with self._ctrl_lock:
                active_trajectory = self._active_pose_trajectory
            measured_q = None
            if active_trajectory is not None and active_trajectory.get("tracking_enabled", False):
                measured_q = self._fresh_measured_joint_q()

            with self._ctrl_lock:
                if self._active_pose_trajectory is active_trajectory and active_trajectory is not None:
                    self._update_pose_trajectory_locked(
                        active_trajectory,
                        time.monotonic(),
                        measured_q,
                    )
                q_pjs = self._target_q.copy()
                dq = self._target_dq.copy()
                tau = self._target_tau.copy()
                kp = self._kp.copy()
                kd = self._kd.copy()
                mode = self._kinematic_mode

            q_cmd = self._build_command_q(q_pjs, mode)

            if self._control_backend == "ros2":
                mode_value = _enum_value(mode)
                low_cmd_msg.kinematic_modes[:] = [mode_value] * len(low_cmd_msg.kinematic_modes)
                now_ns = time.time_ns()
                low_cmd_msg.header.seq = int(sequence & 0xFFFFFFFF)
                low_cmd_msg.header.sec = int(now_ns // 1_000_000_000)
                low_cmd_msg.header.nanosec = int(now_ns % 1_000_000_000)
            else:
                low_cmd_msg.kinematic_mode(mode)
            for idx in range(NUM_MOTORS):
                motor_cmd = motors[idx]
                if self._control_backend == "ros2":
                    motor_cmd.q = float(q_cmd[idx])
                    motor_cmd.dq = float(dq[idx])
                    motor_cmd.tau = float(tau[idx])
                    motor_cmd.kp = float(kp[idx])
                    motor_cmd.kd = float(kd[idx])
                else:
                    motor_cmd.q(float(q_cmd[idx]))
                    motor_cmd.dq(float(dq[idx]))
                    motor_cmd.tau(float(tau[idx]))
                    motor_cmd.kp(float(kp[idx]))
                    motor_cmd.kd(float(kd[idx]))

            if not self._dbg_publish_logged:
                logger_mp.info(
                    "[ControllerCore] First publish targets: mode=%s q min/max=%.3f/%.3f",
                    str(mode),
                    float(q_cmd.min()),
                    float(q_cmd.max()),
                )
                self._dbg_publish_logged = True

            self.lowcmd_pub.write(low_cmd_msg)
            sequence += 1

            published_at = time.perf_counter()
            if previous_publish_at is not None:
                diag_max_gap_s = max(diag_max_gap_s, published_at - previous_publish_at)
            if previous_q_pjs is not None:
                diag_max_q_step = max(
                    diag_max_q_step,
                    float(np.max(np.abs(q_pjs - previous_q_pjs))),
                )
            previous_publish_at = published_at
            previous_q_pjs = q_pjs
            diag_publishes += 1
            diag_elapsed_s = published_at - diag_started_at
            if diag_elapsed_s >= 5.0:
                age_getter = getattr(self.state_buffer, "age_s", None)
                state_age_s = age_getter() if callable(age_getter) else None
                logger_mp.info(
                    "[ControllerCore] desired LowCmd source %.1fHz max_gap=%.2fms "
                    "max_q_step=%.5frad lowstate_age=%s",
                    diag_publishes / max(diag_elapsed_s, 1e-9),
                    1000.0 * diag_max_gap_s,
                    diag_max_q_step,
                    "n/a" if state_age_s is None else f"{1000.0 * state_age_s:.1f}ms",
                )
                diag_started_at = published_at
                diag_publishes = 0
                diag_max_gap_s = 0.0
                diag_max_q_step = 0.0
            self.control_rate.sleep()

    @staticmethod
    def _pad_array(arr):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.size < NUM_MOTORS:
            arr = np.pad(arr, (0, NUM_MOTORS - arr.size))
        return arr[:NUM_MOTORS]

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

    def set_joint_gains(self, indices, kp=None, kd=None):
        indices = [int(idx) for idx in indices]
        with self._ctrl_lock:
            if kp is not None:
                kp_arr = np.asarray(kp, dtype=np.float32).reshape(-1)
                if kp_arr.size == 1:
                    kp_arr = np.full(len(indices), float(kp_arr[0]), dtype=np.float32)
                if kp_arr.size != len(indices):
                    raise ValueError(f"kp length {kp_arr.size} != len(indices) {len(indices)}")
                for joint_id, value in zip(indices, kp_arr):
                    self._kp[joint_id] = float(value)
            if kd is not None:
                kd_arr = np.asarray(kd, dtype=np.float32).reshape(-1)
                if kd_arr.size == 1:
                    kd_arr = np.full(len(indices), float(kd_arr[0]), dtype=np.float32)
                if kd_arr.size != len(indices):
                    raise ValueError(f"kd length {kd_arr.size} != len(indices) {len(indices)}")
                for joint_id, value in zip(indices, kd_arr):
                    self._kd[joint_id] = float(value)

    def hold_current_position(self):
        state = self.state_buffer.get()
        if state is None:
            return
        q_now = np.array([self._msg_field(js, "q") for js in self._state_joint_seq(state)], dtype=np.float32)
        self.set_joint_targets(q=q_now, dq=np.zeros(NUM_MOTORS), tau=np.zeros(NUM_MOTORS))

    def set_kinematic_mode(self, mode):
        with self._ctrl_lock:
            self._kinematic_mode = mode
        self._dbg_publish_logged = False

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
        return [int(idx) for idx in sorted(set(indices))]

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
            for offset, joint_id in enumerate(indices):
                self._target_q[joint_id] = q_target[offset]
                self._target_tau[joint_id] = tau_target[offset]
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

    def _fresh_measured_joint_q(self) -> np.ndarray | None:
        state_buffer = getattr(self, "state_buffer", None)
        age_getter = getattr(state_buffer, "age_s", None)
        if callable(age_getter):
            age_s = age_getter()
            if age_s is None or age_s > POSE_MEASURED_START_MAX_AGE_S:
                return None
        current_q = self.get_joint_q()
        if current_q is None:
            return None
        current_q = np.asarray(current_q, dtype=np.float32).reshape(-1)
        if current_q.shape != self._target_q.shape or not np.all(np.isfinite(current_q)):
            return None
        return current_q.copy()

    def get_command_snapshot(self):
        with self._ctrl_lock:
            q_target_pjs = self._target_q.copy()
            dq_target = self._target_dq.copy()
            tau_target = self._target_tau.copy()
            kp = self._kp.copy()
            kd = self._kd.copy()
            mode = self._kinematic_mode
        q_cmd = self._build_command_q(q_target_pjs, mode)
        return {
            "q_target_pjs": q_target_pjs,
            "q_cmd": q_cmd,
            "dq_target": dq_target,
            "tau_target": tau_target,
            "kp": kp,
            "kd": kd,
            "mode": mode,
        }

    def get_joint_dq(self):
        state = self.state_buffer.get()
        if state is None:
            return None
        return np.array([self._msg_field(js, "dq") for js in self._state_joint_seq(state)], dtype=np.float32)

    def get_joint_tau(self):
        state = self.state_buffer.get()
        if state is None:
            return None
        tau_raw = np.array([self._msg_field(js, "tau_est") for js in self._state_joint_seq(state)], dtype=np.float32)
        tau_filt = self._tau_kf.step(tau_raw) if self._tau_kf is not None else tau_raw
        return tau_raw, tau_filt

    def get_motor_q(self):
        state = self.state_buffer.get()
        if state is None:
            return None
        return np.array([self._msg_field(ms, "q") for ms in self._state_motor_seq(state)], dtype=np.float32)

    def get_motor_dq(self):
        state = self.state_buffer.get()
        if state is None:
            return None
        return np.array([self._msg_field(ms, "dq") for ms in self._state_motor_seq(state)], dtype=np.float32)

    def get_imu(self):
        state = self.state_buffer.get()
        if state is None:
            return None, None, None
        imu = self._msg_field(state, "imu_state")
        return (
            np.array(self._msg_field(imu, "quaternion"), dtype=np.float32),
            np.array(self._msg_field(imu, "gyroscope"), dtype=np.float32),
            np.array(self._msg_field(imu, "rpy"), dtype=np.float32),
        )

    def get_observation_snapshot(self):
        q_joint = self.get_joint_q()
        dq_joint = self.get_joint_dq()
        tau_pair = self.get_joint_tau()
        q_motor = self.get_motor_q()
        dq_motor = self.get_motor_dq()
        imu_quat, imu_gyro, imu_rpy = self.get_imu()
        tau_joint_raw = None if tau_pair is None else tau_pair[0]
        tau_joint_filt = None if tau_pair is None else tau_pair[1]
        return {
            "q_joint": q_joint,
            "dq_joint": dq_joint,
            "tau_joint_raw": tau_joint_raw,
            "tau_joint_filt": tau_joint_filt,
            "q_motor": q_motor,
            "dq_motor": dq_motor,
            "imu_quat": imu_quat,
            "imu_gyro": imu_gyro,
            "imu_rpy": imu_rpy,
        }

    def _wait_for_pose_arrival(
        self,
        *,
        q_goal: np.ndarray,
        joint_ids: list[int],
        pose_name: str,
        tolerance_rad: float | None,
        velocity_tolerance_rad_s: float | None = None,
        settle_time_s: float = 0.0,
        timeout_s: float,
        update_dt: float,
        cancel_event=None,
    ) -> bool:
        tolerance = None if tolerance_rad is None else max(0.0, float(tolerance_rad))
        velocity_tolerance = (
            None
            if velocity_tolerance_rad_s is None
            else max(0.0, float(velocity_tolerance_rad_s))
        )
        settle_time_s = max(0.0, float(settle_time_s))
        timeout_s = max(0.0, float(timeout_s))
        if tolerance is None or timeout_s <= 0.0 or not joint_ids:
            return True

        arrival_started_at = time.monotonic()
        settled_since: float | None = None
        max_error = float("inf")
        max_velocity = float("inf")
        group_errors: dict[str, float] = {}
        largest_joint_errors: list[tuple[str, float]] = []
        q_goal = np.asarray(q_goal, dtype=np.float32).reshape(-1)
        while not self._stop_event.is_set():
            if cancel_event is not None:
                try:
                    if cancel_event.is_set():
                        return False
                except Exception:
                    pass
            current_q = self.get_joint_q()
            if current_q is not None:
                current_q = np.asarray(current_q, dtype=np.float32).reshape(-1)
                if current_q.shape == q_goal.shape and np.all(np.isfinite(current_q)):
                    max_error = float(np.max(np.abs(current_q[joint_ids] - q_goal[joint_ids])))
                    group_errors = self._joint_group_errors(current_q, q_goal, joint_ids)
                    joint_error_values = np.abs(current_q[joint_ids] - q_goal[joint_ids])
                    worst_order = np.argsort(joint_error_values)[::-1][:3]
                    largest_joint_errors = [
                        (
                            JointIndex(int(joint_ids[int(local_idx)])).name,
                            round(float(joint_error_values[int(local_idx)]), 5),
                        )
                        for local_idx in worst_order
                    ]
                    velocity_ok = True
                    if velocity_tolerance is not None:
                        current_dq = self.get_joint_dq()
                        if current_dq is None:
                            velocity_ok = False
                        else:
                            current_dq = np.asarray(current_dq, dtype=np.float32).reshape(-1)
                            if current_dq.shape != q_goal.shape or not np.all(np.isfinite(current_dq)):
                                velocity_ok = False
                            else:
                                max_velocity = float(np.max(np.abs(current_dq[joint_ids])))
                                velocity_ok = max_velocity <= velocity_tolerance
                    now = time.monotonic()
                    if max_error <= tolerance and velocity_ok:
                        if settled_since is None:
                            settled_since = now
                        if now - settled_since < settle_time_s:
                            time.sleep(min(0.02, max(update_dt, 0.001)))
                            continue
                        logger_mp.info(
                            "[ControllerCore] %s settled max_error=%.4frad "
                            "max_velocity=%.4frad/s groups=%s",
                            pose_name,
                            max_error,
                            max_velocity,
                            group_errors,
                        )
                        return True
                    settled_since = None
            if time.monotonic() - arrival_started_at >= timeout_s:
                break
            time.sleep(min(0.02, max(update_dt, 0.001)))

        logger_mp.warning(
            "[ControllerCore] %s arrival timeout after %.2fs "
            "max_error=%.4frad max_velocity=%.4frad/s groups=%s worst_joints=%s "
            "(continuing while holding target)",
            pose_name,
            time.monotonic() - arrival_started_at,
            max_error,
            max_velocity,
            group_errors,
            largest_joint_errors,
        )
        return False

    @staticmethod
    def _joint_group_errors(
        q_actual: np.ndarray,
        q_goal: np.ndarray,
        joint_ids: list[int],
    ) -> dict[str, float]:
        selected = set(int(idx) for idx in joint_ids)
        result: dict[str, float] = {}
        for name, indices in (
            ("waist", WAIST_INDICES),
            ("leg", LEG_INDICES),
            ("arm", ARM_INDICES),
            ("neck", NECK_INDICES),
        ):
            active = [int(idx) for idx in indices if int(idx) in selected]
            if active:
                result[name] = round(
                    float(np.max(np.abs(q_actual[active] - q_goal[active]))),
                    5,
                )
        return result

    @staticmethod
    def _sample_pose_trajectory(trajectory: dict, elapsed_s: float) -> np.ndarray:
        segment_durations = trajectory["segment_durations"]
        segment_ends = trajectory["segment_ends"]
        q_points = trajectory["q_points"]
        tangents = trajectory["tangents"]
        total_duration = float(trajectory["total_duration"])
        elapsed_s = min(total_duration, max(0.0, float(elapsed_s)))
        segment_idx = min(
            int(np.searchsorted(segment_ends, elapsed_s, side="right")),
            len(segment_durations) - 1,
        )
        segment_start_s = 0.0 if segment_idx == 0 else float(segment_ends[segment_idx - 1])
        segment_h = float(segment_durations[segment_idx])
        u = min(1.0, max(0.0, (elapsed_s - segment_start_s) / segment_h))
        # Quintic Hermite basis with zero acceleration at every waypoint.
        # With one shared velocity tangent per waypoint this makes position,
        # velocity and acceleration continuous (C2), unlike the former cubic
        # PCHIP whose acceleration jumped at every waypoint.
        u2 = u * u
        u3 = u2 * u
        u4 = u3 * u
        u5 = u4 * u
        h00 = 1.0 - 10.0 * u3 + 15.0 * u4 - 6.0 * u5
        h01 = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
        h10 = u - 6.0 * u3 + 8.0 * u4 - 3.0 * u5
        h11 = -4.0 * u3 + 7.0 * u4 - 3.0 * u5
        return (
            h00 * q_points[segment_idx]
            + h10 * segment_h * tangents[segment_idx]
            + h01 * q_points[segment_idx + 1]
            + h11 * segment_h * tangents[segment_idx + 1]
        )

    @staticmethod
    def _trajectory_tracking_scale(max_error: float, soft_error: float, hard_error: float) -> float:
        if not np.isfinite(max_error) or max_error >= hard_error:
            return 0.0
        if max_error <= soft_error:
            return 1.0
        return float((hard_error - max_error) / max(hard_error - soft_error, 1e-9))

    def _update_pose_trajectory_locked(
        self,
        trajectory: dict,
        now_s: float,
        measured_q: np.ndarray | None = None,
    ) -> bool:
        now_s = float(now_s)
        raw_dt = max(0.0, now_s - float(trajectory["last_update_at"]))
        trajectory["last_update_at"] = now_s
        phase_dt = min(raw_dt, float(trajectory["max_phase_step_s"]))

        tracking_scale = 0.0 if trajectory["tracking_enabled"] else 1.0
        if trajectory["tracking_enabled"] and measured_q is not None:
            measured_q = np.asarray(measured_q, dtype=np.float32).reshape(-1)
            if measured_q.shape == self._target_q.shape and np.all(np.isfinite(measured_q)):
                tracking_joint_ids = trajectory.get(
                    "tracking_joint_ids", trajectory["joint_ids"]
                )
                max_error = float(
                    np.max(
                        np.abs(
                            measured_q[tracking_joint_ids]
                            - self._target_q[tracking_joint_ids]
                        )
                    )
                )
                trajectory["max_tracking_error"] = max(
                    float(trajectory["max_tracking_error"]),
                    max_error,
                )
                tracking_scale = self._trajectory_tracking_scale(
                    max_error,
                    float(trajectory["tracking_soft_error"]),
                    float(trajectory["tracking_hard_error"]),
                )

        trajectory["phase_s"] = min(
            float(trajectory["total_duration"]),
            float(trajectory["phase_s"]) + phase_dt * tracking_scale,
        )
        q_sample = self._sample_pose_trajectory(trajectory, float(trajectory["phase_s"]))
        for joint_id in trajectory["joint_ids"]:
            self._target_q[joint_id] = float(q_sample[joint_id])
            self._target_dq[joint_id] = 0.0
            self._target_tau[joint_id] = 0.0
            if not trajectory["preserve_gains"]:
                self._kp[joint_id] = float(self._kp_default[joint_id])
                self._kd[joint_id] = float(self._kd_default[joint_id])
        trajectory["updates"] += 1
        completed = float(trajectory["phase_s"]) >= float(trajectory["total_duration"])
        if completed:
            trajectory["complete_event"].set()
        return completed

    def move_through_poses(
        self,
        pose_sequence: list[tuple[Union[str, np.ndarray], float]],
        *,
        leg: bool = True,
        waist: bool = True,
        arm: bool = True,
        neck: bool = True,
        preserve_gains: bool = False,
        arrival_tolerance_rad: float | None = None,
        arrival_timeout_s: float = 0.0,
        arrival_velocity_tolerance_rad_s: float | None = None,
        arrival_settle_time_s: float = 0.0,
        arrival_joint_ids: list[int] | tuple[int, ...] | None = None,
        use_publisher_timing: bool = False,
        start_from_measured: bool = False,
        tracking_joint_ids: list[int] | tuple[int, ...] | None = None,
        tracking_error_soft_rad: float | None = None,
        tracking_error_hard_rad: float | None = None,
        trajectory_timeout_s: float | None = None,
        cancel_event=None,
    ) -> bool:
        """Move through waypoints with bounded phase advance and C2 commands.

        Each duration belongs to the segment ending at its associated pose.
        Runtime stalls never skip trajectory samples: phase advances by at most
        one 200 Hz sample per published command.  Optional feedback governing
        slows or pauses phase while the physical robot is behind the command.
        """
        if not pose_sequence:
            return True

        joint_ids = self._collect_joint_indices(leg=leg, waist=waist, arm=arm, neck=neck)
        commanded_joint_ids = {int(joint_id) for joint_id in joint_ids}

        def _resolve_checked_joint_ids(
            raw_joint_ids: list[int] | tuple[int, ...] | None,
            *,
            label: str,
        ) -> list[int]:
            if raw_joint_ids is None:
                return list(joint_ids)
            resolved: list[int] = []
            seen: set[int] = set()
            for raw_joint_id in raw_joint_ids:
                joint_id = int(raw_joint_id)
                if joint_id not in commanded_joint_ids:
                    raise ValueError(
                        f"{label} joint {joint_id} is not part of the commanded trajectory"
                    )
                if joint_id in seen:
                    continue
                seen.add(joint_id)
                resolved.append(joint_id)
            if not resolved:
                raise ValueError(f"{label}_joint_ids must contain at least one commanded joint")
            return resolved

        resolved_arrival_joint_ids = _resolve_checked_joint_ids(
            arrival_joint_ids,
            label="arrival",
        )
        if tracking_joint_ids is None:
            resolved_tracking_joint_ids = list(joint_ids)
        else:
            resolved_tracking_joint_ids = _resolve_checked_joint_ids(
                tracking_joint_ids,
                label="tracking",
            )
        with self._ctrl_lock:
            q_start = self._target_q.copy()
        if start_from_measured:
            measured_start = self._fresh_measured_joint_q()
            if measured_start is None:
                logger_mp.warning(
                    "[ControllerCore] measured trajectory start unavailable/stale; "
                    "using last command target"
                )
            else:
                q_start[joint_ids] = measured_start[joint_ids]
                # Synchronize the outgoing target to feedback before phase starts.
                # This removes stored command error without requesting a physical
                # jump, then interpolates measured waist/arms toward the first pose.
                with self._ctrl_lock:
                    self._target_q[joint_ids] = q_start[joint_ids]
                    self._target_dq[joint_ids] = 0.0
                    self._target_tau[joint_ids] = 0.0

        pose_names: list[str] = []
        durations: list[float] = []
        resolved_poses: list[np.ndarray] = [q_start]
        for pose, duration in pose_sequence:
            duration = float(duration)
            if not np.isfinite(duration) or duration <= 0.0:
                raise ValueError(f"Pose segment duration must be positive and finite, got {duration!r}")
            if isinstance(pose, str):
                if pose not in self._poses:
                    raise KeyError(f"Unknown pose name: {pose}. Available: {list(self._poses.keys())}")
                q_pose = np.asarray(self._poses[pose], dtype=np.float32)
                pose_name = pose
            else:
                q_pose = np.asarray(pose, dtype=np.float32)
                if q_pose.shape != self._default_q.shape:
                    raise ValueError(
                        f"Pose array has invalid shape {q_pose.shape}, expected {self._default_q.shape}."
                    )
                pose_name = "custom"
            resolved_poses.append(q_pose.copy())
            pose_names.append(pose_name)
            durations.append(duration)

        q_points = np.stack(resolved_poses, axis=0)
        segment_durations = np.asarray(durations, dtype=np.float64)
        segment_ends = np.cumsum(segment_durations)
        total_duration = float(segment_ends[-1])
        secants = np.diff(q_points, axis=0) / segment_durations[:, None]
        tangents = np.zeros_like(q_points, dtype=np.float64)

        # Shape-preserving velocity estimates are shared by both adjacent
        # quintic segments.  Every waypoint acceleration is zero, so q/dq/ddq
        # are continuous. A direction reversal receives zero velocity.
        for waypoint_idx in range(1, len(resolved_poses) - 1):
            previous_slope = secants[waypoint_idx - 1]
            next_slope = secants[waypoint_idx]
            same_direction = previous_slope * next_slope > 0.0
            if not np.any(same_direction):
                continue
            previous_h = segment_durations[waypoint_idx - 1]
            next_h = segment_durations[waypoint_idx]
            weight_1 = 2.0 * next_h + previous_h
            weight_2 = next_h + 2.0 * previous_h
            tangents[waypoint_idx, same_direction] = (weight_1 + weight_2) / (
                weight_1 / previous_slope[same_direction]
                + weight_2 / next_slope[same_direction]
            )

        update_dt = max(self.control_dt, 1.0 / POSE_INTERPOLATION_UPDATE_HZ)
        started_at = time.monotonic()
        trajectory_name = " -> ".join(pose_names)
        tracking_enabled = (
            tracking_error_soft_rad is not None or tracking_error_hard_rad is not None
        )
        tracking_soft_error = (
            POSE_TRACKING_SOFT_ERROR_RAD
            if tracking_error_soft_rad is None
            else max(0.0, float(tracking_error_soft_rad))
        )
        tracking_hard_error = (
            POSE_TRACKING_HARD_ERROR_RAD
            if tracking_error_hard_rad is None
            else max(0.0, float(tracking_error_hard_rad))
        )
        if tracking_hard_error <= tracking_soft_error:
            raise ValueError(
                "tracking_error_hard_rad must be greater than tracking_error_soft_rad"
            )
        if trajectory_timeout_s is None:
            trajectory_timeout_s = max(total_duration * 6.0, total_duration + 5.0)
        trajectory_timeout_s = max(total_duration, float(trajectory_timeout_s))
        trajectory = {
            "started_at": started_at,
            "last_update_at": started_at,
            "phase_s": 0.0,
            "max_phase_step_s": POSE_MAX_PHASE_STEP_S,
            "segment_durations": segment_durations,
            "segment_ends": segment_ends,
            "total_duration": total_duration,
            "q_points": q_points,
            "tangents": tangents,
            "joint_ids": joint_ids,
            "tracking_joint_ids": resolved_tracking_joint_ids,
            "preserve_gains": bool(preserve_gains),
            "tracking_enabled": tracking_enabled,
            "tracking_soft_error": tracking_soft_error,
            "tracking_hard_error": tracking_hard_error,
            "max_tracking_error": 0.0,
            "updates": 0,
            "complete_event": threading.Event(),
        }
        logger_mp.info(
            "[ControllerCore] bounded C2 pose sequence start nominal=%.2fs "
            "tracking=%s measured_start=%s poses=%s",
            total_duration,
            tracking_enabled,
            start_from_measured,
            trajectory_name,
        )

        ctrl_thread = getattr(self, "_ctrl_thread", None)
        publisher_driven = (
            bool(use_publisher_timing)
            and ctrl_thread is not None
            and ctrl_thread.is_alive()
        )
        completed = False
        timed_out = False

        def cancelled() -> bool:
            if self._stop_event.is_set():
                return True
            if cancel_event is None:
                return False
            try:
                return bool(cancel_event.is_set())
            except Exception:
                return False

        if publisher_driven:
            with self._ctrl_lock:
                self._active_pose_trajectory = trajectory
            while not cancelled():
                if trajectory["complete_event"].wait(timeout=0.02):
                    completed = True
                    break
                if time.monotonic() - started_at >= trajectory_timeout_s:
                    timed_out = True
                    break
            with self._ctrl_lock:
                if self._active_pose_trajectory is trajectory:
                    self._active_pose_trajectory = None
        else:
            # Deterministic fallback for controllers without a running publisher
            # (including unit tests and offline tools).
            next_update_at = started_at
            while not cancelled():
                measured_q = self._fresh_measured_joint_q() if tracking_enabled else None
                with self._ctrl_lock:
                    completed = self._update_pose_trajectory_locked(
                        trajectory,
                        time.monotonic(),
                        measured_q,
                    )
                if completed:
                    break
                if time.monotonic() - started_at >= trajectory_timeout_s:
                    timed_out = True
                    break
                next_update_at += update_dt
                sleep_s = next_update_at - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                else:
                    next_update_at = time.monotonic()

        elapsed_s = time.monotonic() - started_at
        if not completed:
            logger_mp.warning(
                "[ControllerCore] bounded C2 pose sequence %s after %.2fs "
                "phase=%.2f/%.2fs updates=%d max_tracking_error=%.4frad poses=%s",
                "timed out" if timed_out else "interrupted",
                elapsed_s,
                float(trajectory["phase_s"]),
                total_duration,
                int(trajectory["updates"]),
                float(trajectory["max_tracking_error"]),
                trajectory_name,
            )
            self._dbg_publish_logged = False
            return False

        logger_mp.info(
            "[ControllerCore] bounded C2 pose command complete in %.2fs "
            "(%d updates, max_tracking_error=%.4frad): %s",
            elapsed_s,
            int(trajectory["updates"]),
            float(trajectory["max_tracking_error"]),
            trajectory_name,
        )
        arrived = self._wait_for_pose_arrival(
            q_goal=q_points[-1],
            joint_ids=resolved_arrival_joint_ids,
            pose_name=f"continuous pose sequence final ({pose_names[-1]})",
            tolerance_rad=arrival_tolerance_rad,
            velocity_tolerance_rad_s=arrival_velocity_tolerance_rad_s,
            settle_time_s=arrival_settle_time_s,
            timeout_s=arrival_timeout_s,
            update_dt=update_dt,
            cancel_event=cancel_event,
        )
        self._dbg_publish_logged = False
        return arrived

    def move_to_pose(
        self,
        pose: Union[str, np.ndarray],
        duration: float = 2.0,
        leg: bool = True,
        waist: bool = True,
        arm: bool = True,
        neck: bool = True,
        preserve_gains: bool = False,
        arrival_tolerance_rad: float | None = None,
        arrival_timeout_s: float = 0.0,
        arrival_velocity_tolerance_rad_s: float | None = None,
        arrival_settle_time_s: float = 0.0,
        arrival_joint_ids: list[int] | tuple[int, ...] | None = None,
        start_from_measured: bool = False,
        tracking_joint_ids: list[int] | tuple[int, ...] | None = None,
        tracking_error_soft_rad: float | None = None,
        tracking_error_hard_rad: float | None = None,
        trajectory_timeout_s: float | None = None,
        use_publisher_timing: bool = True,
        cancel_event=None,
    ) -> bool:
        if duration <= 0.0:
            duration = self.control_dt
        return self.move_through_poses(
            [(pose, float(duration))],
            leg=leg,
            waist=waist,
            arm=arm,
            neck=neck,
            preserve_gains=preserve_gains,
            arrival_tolerance_rad=arrival_tolerance_rad,
            arrival_timeout_s=arrival_timeout_s,
            arrival_velocity_tolerance_rad_s=arrival_velocity_tolerance_rad_s,
            arrival_settle_time_s=arrival_settle_time_s,
            arrival_joint_ids=arrival_joint_ids,
            use_publisher_timing=use_publisher_timing,
            start_from_measured=start_from_measured,
            tracking_joint_ids=tracking_joint_ids,
            tracking_error_soft_rad=tracking_error_soft_rad,
            tracking_error_hard_rad=tracking_error_hard_rad,
            trajectory_timeout_s=trajectory_timeout_s,
            cancel_event=cancel_event,
        )

    def default_pos_state(
        self,
        pose: Union[str, np.ndarray, None] = None,
        leg: bool = True,
        waist: bool = True,
        arm: bool = True,
        neck: bool = True,
        preserve_gains: bool = False,
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
            for joint_id in joint_ids:
                self._target_q[joint_id] = float(q_hold[joint_id])
                self._target_dq[joint_id] = 0.0
                self._target_tau[joint_id] = 0.0
                if not preserve_gains:
                    self._kp[joint_id] = float(self._kp_default[joint_id])
                    self._kd[joint_id] = float(self._kd_default[joint_id])
        self._dbg_publish_logged = False
        logger_mp.info(
            "[ControllerCore] default_pos_state set: pose=%s, target q min/max=%.3f/%.3f",
            pose_name,
            float(self._target_q.min()),
            float(self._target_q.max()),
        )

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

    def abort(
        self,
        timeout_ms: int = 2000,
        service_call_timeout_s: float | None = 6.0,
        ctrl_thread_join_timeout_s: float = 5.0,
    ):
        shutdown_error: BaseException | None = None
        try:
            self.hold_current_position()
            time.sleep(min(0.1, max(self.control_dt * 2.0, 0.02)))
        except Exception as exc:
            logger_mp.warning("[ControllerCore] hold_current_position in abort() failed: %s", exc)

        try:
            self._prepare_service_shutdown(
                ctrl_thread_join_timeout_s=ctrl_thread_join_timeout_s,
                context="abort() pre-shutdown",
            )
            if self._control_backend == "ros2":
                completed = self._run_blocking_shutdown_call(
                    "service shutdown (abort)",
                    lambda: self._service_shutdown(timeout_ms=timeout_ms),
                    timeout_s=service_call_timeout_s,
                )
                if not completed:
                    raise TimeoutError("service shutdown (abort) did not complete")
            else:
                self._run_service_shutdown_helper(
                    timeout_ms=timeout_ms,
                    service_call_timeout_s=service_call_timeout_s,
                    context="service shutdown (abort)",
                )
        except Exception as exc:
            shutdown_error = exc
            logger_mp.warning("[ControllerCore] Service shutdown failed in abort(): %s", exc)
        finally:
            self._cleanup_dds()
        logger_mp.warning("[ControllerCore] abort() done")
        if shutdown_error is not None:
            raise shutdown_error

    def stop(
        self,
        shutdown_timeout_ms: int = 5000,
        service_call_timeout_s: float | None = 12.0,
        ctrl_thread_join_timeout_s: float = 5.0,
    ):
        shutdown_error: BaseException | None = None
        try:
            logger_mp.info("[ControllerCore] shutdown pose sequence start")
            shutdown_poses: list[tuple[Union[str, np.ndarray], float]] = []
            if "waypoint_3" in self._poses:
                shutdown_poses.append(("waypoint_3", 2.0))
            shutdown_poses.extend(
                [
                    ("waypoint_2", 2.0),
                    ("waypoint_1", 2.0),
                    ("zero_pos", 2.0),
                ]
            )
            pose_arrived = self.move_through_poses(
                shutdown_poses,
                leg=True,
                waist=True,
                arm=True,
                neck=True,
                arrival_tolerance_rad=0.06,
                arrival_timeout_s=8.0,
                arrival_velocity_tolerance_rad_s=0.12,
                arrival_settle_time_s=0.25,
                arrival_joint_ids=tuple(WAIST_INDICES) + tuple(ARM_INDICES),
                use_publisher_timing=True,
                start_from_measured=True,
                # The real neck pitch has a known persistent feedback-zero
                # residual. Keep commanding it through every waypoint, but do
                # not let that residual freeze or falsely fail the arm/waist
                # safety transition before Torque OFF.
                tracking_joint_ids=tuple(WAIST_INDICES) + tuple(ARM_INDICES),
                tracking_error_soft_rad=0.08,
                tracking_error_hard_rad=0.25,
                trajectory_timeout_s=42.0,
            )
            if pose_arrived:
                logger_mp.info(
                    "[ControllerCore] shutdown pose sequence settled; "
                    "zero pose will be held through Torque OFF"
                )
            else:
                logger_mp.warning(
                    "[ControllerCore] shutdown pose sequence did not settle within its "
                    "bounded timeout; proceeding to Torque OFF while holding the last safe command"
                )
        except Exception as exc:
            logger_mp.warning("[ControllerCore] shutdown pose sequence in stop() failed: %s", exc)

        try:
            self._prepare_service_shutdown(
                ctrl_thread_join_timeout_s=ctrl_thread_join_timeout_s,
                context="stop() pre-shutdown",
            )
            if self._control_backend == "ros2":
                completed = self._run_blocking_shutdown_call(
                    "service shutdown (stop)",
                    lambda: self._service_shutdown(timeout_ms=shutdown_timeout_ms),
                    timeout_s=service_call_timeout_s,
                )
                if not completed:
                    raise TimeoutError("service shutdown (stop) did not complete")
            else:
                self._run_service_shutdown_helper(
                    timeout_ms=shutdown_timeout_ms,
                    service_call_timeout_s=service_call_timeout_s,
                    context="service shutdown (stop)",
                )
        except Exception as exc:
            shutdown_error = exc
            logger_mp.warning("[ControllerCore] Service shutdown failed: %s", exc)
        finally:
            # Keep the ROS2 client alive when raising so ControlWorker.abort()
            # can make its bounded fallback Torque OFF attempt.
            if shutdown_error is None:
                self._cleanup_dds()

        if shutdown_error is not None:
            raise shutdown_error
        logger_mp.info("[ControllerCore] stop() done")
