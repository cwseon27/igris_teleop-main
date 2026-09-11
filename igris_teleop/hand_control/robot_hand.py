import threading
import time
import traceback
import os
import subprocess
from pathlib import Path

import numpy as np

from .command_range import apply_finger_close_overrides, apply_motor_command_overrides
from .hand_retargeting import HandRetargeting
from ..core.rate import Rate

import logging_mp

logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)

HAND_TARGET_LENGTH = 12
HAND_MOTOR_IDS = (11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26)
DEFAULT_HAND_DOMAIN_ID = 0
HAND_INIT_TRIGGER_ID = 99
HAND_INIT_COMMAND_GAP_S = 0.1
HAND_INIT_MOTION_START_TIMEOUT_S = 3.0
HAND_INIT_STABLE_TIMEOUT_S = 12.0
HAND_INIT_STABLE_WINDOW_S = 1.0
HAND_INIT_MOVE_THRESHOLD = 0.02
HAND_INIT_STABLE_DELTA_THRESHOLD = 0.005
HAND_STATE_POLL_S = 0.05
DEFAULT_HAND_STATE_TOPIC = "rt/handstate"
DEFAULT_HAND_CMD_TOPIC = "rt/handcmd"
DEFAULT_HAND_DDS_NAMESPACE = "igris_c_IG05"

try:
    import igris_c_sdk as igc_sdk
except ImportError:
    igc_sdk = None
    logger.error("igris_c_sdk is not available. Please install the SDK wheel.")


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _scoped_dds_topic(namespace: str, topic: str) -> str:
    """Return the physical DDS topic used by the robot-side SDK.

    The Python SDK on the operator PC has no ChannelFactory namespace API, while
    the robot initializes its participant with the ``igris_c_IG05`` namespace.
    Its logical ``rt/handcmd`` topic is therefore visible on the wire as
    ``igris_c_IG05/rt/handcmd``.  Prefixing here makes both SDK versions address
    the same physical topic.
    """
    clean_topic = str(topic).strip().strip("/")
    clean_namespace = str(namespace).strip().strip("/")
    if not clean_namespace or clean_topic == clean_namespace or clean_topic.startswith(clean_namespace + "/"):
        return clean_topic
    return f"{clean_namespace}/{clean_topic}"


class IgrisHandDDSInterface:
    """Direct DDS interface for the 12-DOF IGRIS hand."""

    def __init__(
        self,
        domain_id: int = DEFAULT_HAND_DOMAIN_ID,
        dds_namespace: str = DEFAULT_HAND_DDS_NAMESPACE,
    ) -> None:
        if igc_sdk is None:
            raise RuntimeError("igris_c_sdk is not available. Cannot use IgrisHandDDSInterface.")

        self._lock = threading.Lock()
        self._has_state = threading.Event()
        self._latest_position = np.zeros(HAND_TARGET_LENGTH, dtype=np.float64)
        self._stopped = False
        self._dds_cleaned = False
        self._state_sub = None
        self._cmd_pub = None

        self._channel_factory = igc_sdk.ChannelFactory.Instance()
        if self._channel_factory.IsInitialized():
            current_domain = int(self._channel_factory.GetDomainId())
            if current_domain != int(domain_id):
                logger.warning(
                    "[IgrisHandDDSInterface] ChannelFactory already initialized on domain %d; requested domain %d.",
                    current_domain,
                    int(domain_id),
                )
        self._channel_factory.Init(int(domain_id))

        namespace_override = os.getenv("IGRIS_HAND_DDS_NAMESPACE")
        if namespace_override is None:
            namespace_override = os.getenv("IGRIS_ROBOT_DDS_NAMESPACE")
        self._dds_namespace = (
            str(dds_namespace) if namespace_override is None else namespace_override
        ).strip().strip("/")
        state_leaf = os.getenv("IGRIS_HAND_STATE_TOPIC") or DEFAULT_HAND_STATE_TOPIC
        cmd_leaf = os.getenv("IGRIS_HAND_CMD_TOPIC") or DEFAULT_HAND_CMD_TOPIC
        self._state_topic = _scoped_dds_topic(self._dds_namespace, state_leaf)
        self._cmd_topic = _scoped_dds_topic(self._dds_namespace, cmd_leaf)

        self._state_sub = igc_sdk.HandStateSubscriber(self._state_topic)
        if not self._state_sub.init(self._on_hand_state):
            self._cleanup_dds()
            raise RuntimeError(f"Failed to init HandStateSubscriber({self._state_topic})")

        self._cmd_pub = igc_sdk.HandCmdPublisher(self._cmd_topic)
        if not self._cmd_pub.init():
            self._cleanup_dds()
            raise RuntimeError(f"Failed to init HandCmdPublisher({self._cmd_topic})")

        logger.info(
            "[IgrisHandDDSInterface] initialized domain=%d namespace=%s state=%s command=%s",
            int(domain_id),
            self._dds_namespace or "<none>",
            self._state_topic,
            self._cmd_topic,
        )

    def _on_hand_state(self, msg) -> None:
        motor_state = list(msg.motor_state())
        position = np.zeros(HAND_TARGET_LENGTH, dtype=np.float64)
        count = min(HAND_TARGET_LENGTH, len(motor_state))
        for idx in range(count):
            position[idx] = float(motor_state[idx].q())

        with self._lock:
            self._latest_position = position
        self._has_state.set()

    def wait_for_first_state(self, timeout: float = 5.0) -> bool:
        return self._has_state.wait(timeout=timeout)

    @property
    def state_topic(self) -> str:
        return self._state_topic

    @property
    def command_topic(self) -> str:
        return self._cmd_topic

    def get_present_position(self) -> np.ndarray:
        with self._lock:
            return self._latest_position.copy()

    def send_targets(self, targets) -> None:
        target_arr = np.asarray(targets, dtype=np.float64).reshape(-1)
        if target_arr.size != HAND_TARGET_LENGTH:
            raise ValueError(
                f"targets length must be {HAND_TARGET_LENGTH}, got {target_arr.size}"
            )

        motor_cmd = []
        for motor_id, target in zip(HAND_MOTOR_IDS, target_arr.tolist()):
            cmd = igc_sdk.MotorCmd()
            cmd.id(int(motor_id))
            cmd.q(float(_clamp01(target)))
            cmd.dq(0.0)
            cmd.tau(0.0)
            cmd.kp(0.0)
            cmd.kd(0.0)
            motor_cmd.append(cmd)

        hand_cmd = igc_sdk.HandCmd()
        hand_cmd.motor_cmd(motor_cmd)
        self._cmd_pub.write(hand_cmd)

    def send_init_command(self) -> bool:
        init_trigger = igc_sdk.MotorCmd()
        init_trigger.id(HAND_INIT_TRIGGER_ID)
        init_trigger.q(0.0)
        init_trigger.dq(0.0)
        init_trigger.tau(0.0)
        init_trigger.kp(0.0)
        init_trigger.kd(0.0)

        hand_cmd = igc_sdk.HandCmd()
        hand_cmd.motor_cmd([init_trigger])
        return bool(self._cmd_pub.write(hand_cmd))

    def _cleanup_dds(self) -> None:
        if self._dds_cleaned:
            return
        self._dds_cleaned = True

        if self._state_sub is not None:
            try:
                self._state_sub.stop()
            except Exception:
                logger.debug("[IgrisHandDDSInterface] failed to stop HandStateSubscriber.", exc_info=True)

        if self._cmd_pub is not None:
            try:
                self._cmd_pub.stop()
            except Exception:
                logger.debug("[IgrisHandDDSInterface] failed to stop HandCmdPublisher.", exc_info=True)

        try:
            self._channel_factory.Release()
        except Exception:
            logger.debug("[IgrisHandDDSInterface] failed to release ChannelFactory.", exc_info=True)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._cleanup_dds()


class IgrisHandROSBridgeInterface:
    """ROS front-end for the robot-schema-compatible native DDS hand bridge."""

    DEFAULT_COMMAND_TOPIC = "/igris_teleop/hand/command"
    DEFAULT_STATE_TOPIC = "/igris_teleop/hand/state"
    DEFAULT_INIT_SERVICE = "/igris_teleop/hand/init"

    def __init__(self, domain_id: int = DEFAULT_HAND_DOMAIN_ID, dds_namespace: str = DEFAULT_HAND_DDS_NAMESPACE) -> None:
        import rclpy
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Float32MultiArray
        from std_srvs.srv import Trigger

        self._rclpy = rclpy
        self._Float32MultiArray = Float32MultiArray
        self._Trigger = Trigger
        self._lock = threading.Lock()
        self._has_state = threading.Event()
        self._latest_position = np.zeros(HAND_TARGET_LENGTH, dtype=np.float64)
        self._stopped = False
        self._owns_rclpy = not rclpy.ok()
        self._bridge_process = None

        repo_root = Path(__file__).resolve().parents[2]
        default_executable = (
            repo_root
            / "ros_ws"
            / "install"
            / "igris_c_hand"
            / "lib"
            / "igris_c_hand"
            / "igris_c_hand_bridge_node"
        )
        executable = Path(os.getenv("IGRIS_HAND_BRIDGE_EXECUTABLE") or default_executable)
        if not executable.is_file():
            raise RuntimeError(f"Robot-compatible hand bridge is missing: {executable}")

        self._state_topic = (os.getenv("IGRIS_HAND_ROS_STATE_TOPIC") or self.DEFAULT_STATE_TOPIC).strip()
        self._cmd_topic = (os.getenv("IGRIS_HAND_ROS_COMMAND_TOPIC") or self.DEFAULT_COMMAND_TOPIC).strip()
        self._init_service = (os.getenv("IGRIS_HAND_ROS_INIT_SERVICE") or self.DEFAULT_INIT_SERVICE).strip()
        namespace_override = os.getenv("IGRIS_HAND_DDS_NAMESPACE")
        namespace = (dds_namespace if namespace_override is None else namespace_override).strip().strip("/")

        command = [
            str(executable),
            "--ros-args",
            "-p",
            f"domain_id:={int(domain_id)}",
            "-p",
            f"dds_namespace:={namespace}",
            "-p",
            f"ros_command_topic:={self._cmd_topic}",
            "-p",
            f"ros_state_topic:={self._state_topic}",
            "-p",
            f"ros_init_service:={self._init_service}",
        ]
        self._bridge_process = subprocess.Popen(command, cwd=str(repo_root))

        if self._owns_rclpy:
            rclpy.init(args=None)
        self._node = rclpy.create_node(f"igris_hand_client_{os.getpid()}")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._cmd_pub = self._node.create_publisher(Float32MultiArray, self._cmd_topic, qos)
        self._state_sub = self._node.create_subscription(Float32MultiArray, self._state_topic, self._on_hand_state, qos)
        self._init_client = self._node.create_client(Trigger, self._init_service)
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._spin_thread.start()
        logger.info(
            "[IgrisHandROSBridgeInterface] started pid=%d DDS domain=%d namespace=%s ROS state=%s command=%s",
            self._bridge_process.pid,
            int(domain_id),
            namespace or "<none>",
            self._state_topic,
            self._cmd_topic,
        )

    def _on_hand_state(self, msg) -> None:
        position = np.asarray(msg.data, dtype=np.float64).reshape(-1)
        if position.size < HAND_TARGET_LENGTH or not np.all(np.isfinite(position[:HAND_TARGET_LENGTH])):
            return
        with self._lock:
            self._latest_position = position[:HAND_TARGET_LENGTH].copy()
        self._has_state.set()

    @property
    def state_topic(self) -> str:
        return self._state_topic

    @property
    def command_topic(self) -> str:
        return self._cmd_topic

    def wait_for_first_state(self, timeout: float = 5.0) -> bool:
        return self._has_state.wait(timeout=timeout)

    def get_present_position(self) -> np.ndarray:
        with self._lock:
            return self._latest_position.copy()

    def send_targets(self, targets) -> None:
        target_arr = np.asarray(targets, dtype=np.float64).reshape(-1)
        if target_arr.size != HAND_TARGET_LENGTH:
            raise ValueError(f"targets length must be {HAND_TARGET_LENGTH}, got {target_arr.size}")
        message = self._Float32MultiArray()
        message.data = np.clip(target_arr, 0.0, 1.0).astype(np.float32).tolist()
        self._cmd_pub.publish(message)

    def send_init_command(self) -> bool:
        if not self._init_client.wait_for_service(timeout_sec=2.0):
            return False
        future = self._init_client.call_async(self._Trigger.Request())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not future.done():
            time.sleep(0.02)
        if not future.done():
            return False
        response = future.result()
        return bool(response is not None and response.success)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self._node.destroy_node()
        except Exception:
            logger.debug("[IgrisHandROSBridgeInterface] failed to destroy ROS node", exc_info=True)
        if self._owns_rclpy and self._rclpy.ok():
            try:
                self._rclpy.shutdown()
            except Exception:
                logger.debug("[IgrisHandROSBridgeInterface] failed to shutdown rclpy", exc_info=True)
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        if self._bridge_process is not None and self._bridge_process.poll() is None:
            self._bridge_process.terminate()
            try:
                self._bridge_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._bridge_process.kill()
                self._bridge_process.wait(timeout=1.0)


class IgrisHandController:
    """
    Teleop hand controller that retargets VR hand references and talks to IGRIS hand DDS directly.
    """

    def __init__(
        self,
        shm_name,
        shared_lock,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        hybrid_hand_command_lock=None,
        left_hand_close_array=None,
        right_hand_close_array=None,
        hand_close_valid_array=None,
        fps: float = 100.0,
        Unit_Test: bool = False,
        domain_id: int = DEFAULT_HAND_DOMAIN_ID,
        dds_namespace: str = DEFAULT_HAND_DDS_NAMESPACE,
        transport: str = "dds",
        start_control_thread: bool = True,
        auto_initialize: bool = True,
        left_hand_motor_array=None,
        right_hand_motor_array=None,
        hand_motor_valid_array=None,
    ):
        del shm_name, shared_lock

        logger.info("Initialize IgrisHandController...")

        self.rate = Rate(fps)
        self.Unit_Test = Unit_Test
        self.running = False
        self._domain_id = int(domain_id)
        self._hand_init_lock = threading.Lock()
        self._hand_initializing = threading.Event()

        if str(transport).strip().lower() == "ros_bridge":
            self.hand_interface = IgrisHandROSBridgeInterface(
                domain_id=self._domain_id,
                dds_namespace=dds_namespace,
            )
        else:
            self.hand_interface = IgrisHandDDSInterface(
                domain_id=self._domain_id,
                dds_namespace=dds_namespace,
            )
        self.hand_retargeting = HandRetargeting()

        try:
            if not self.hand_interface.wait_for_first_state(timeout=5.0):
                logger.warning(
                    "[IgrisHandController] Waiting for %s timed out. Check that the hand controller is running.",
                    self.hand_interface.state_topic,
                )
            elif auto_initialize:
                self._auto_initialize_hand()
        except Exception:
            logger.warning("[IgrisHandController] Failed to wait for first hand state.", exc_info=True)

        self._hand_control_thread = None
        if start_control_thread:
            hand_control_thread = threading.Thread(
                target=self.control_process,
                args=(
                    left_hand_array,
                    right_hand_array,
                    dual_hand_data_lock,
                    dual_hand_state_array,
                    dual_hand_action_array,
                    hybrid_hand_command_lock,
                    left_hand_close_array,
                    right_hand_close_array,
                    hand_close_valid_array,
                    left_hand_motor_array,
                    right_hand_motor_array,
                    hand_motor_valid_array,
                ),
                daemon=True,
            )
            hand_control_thread.start()
            self._hand_control_thread = hand_control_thread

        logger.info("Initialize IgrisHandController OK!\n")

    def _wait_for_position_change(
        self,
        baseline: np.ndarray,
        timeout_s: float,
        threshold: float = HAND_INIT_MOVE_THRESHOLD,
    ) -> bool:
        deadline = time.perf_counter() + float(timeout_s)
        baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
        while time.perf_counter() < deadline:
            current = self.hand_interface.get_present_position()
            if current.size == baseline.size and np.max(np.abs(current - baseline)) >= float(threshold):
                return True
            time.sleep(HAND_STATE_POLL_S)
        return False

    def _wait_until_position_stable(
        self,
        timeout_s: float = HAND_INIT_STABLE_TIMEOUT_S,
        stable_window_s: float = HAND_INIT_STABLE_WINDOW_S,
        delta_threshold: float = HAND_INIT_STABLE_DELTA_THRESHOLD,
    ) -> bool:
        deadline = time.perf_counter() + float(timeout_s)
        previous = self.hand_interface.get_present_position()
        stable_since = None

        while time.perf_counter() < deadline:
            time.sleep(HAND_STATE_POLL_S)
            current = self.hand_interface.get_present_position()
            if current.size != previous.size:
                previous = current
                stable_since = None
                continue

            delta = float(np.max(np.abs(current - previous))) if current.size else 0.0
            previous = current

            if delta <= float(delta_threshold):
                if stable_since is None:
                    stable_since = time.perf_counter()
                elif (time.perf_counter() - stable_since) >= float(stable_window_s):
                    return True
            else:
                stable_since = None

        return False

    def _auto_initialize_hand(self) -> bool:
        baseline = self.hand_interface.get_present_position()
        logger.info(
            "[IgrisHandController] Sending hand init command (id=%d)...",
            HAND_INIT_TRIGGER_ID,
        )
        try:
            if not self.hand_interface.send_init_command():
                logger.warning("[IgrisHandController] Automatic hand init command was rejected.")
                return False
        except Exception:
            logger.warning("[IgrisHandController] Failed to send automatic hand init command.", exc_info=True)
            return False

        time.sleep(HAND_INIT_COMMAND_GAP_S)
        if not self._wait_for_position_change(
            baseline,
            timeout_s=HAND_INIT_MOTION_START_TIMEOUT_S,
        ):
            logger.warning(
                "[IgrisHandController] Hand init trigger was sent, but no position change was observed within %.1f seconds.",
                HAND_INIT_MOTION_START_TIMEOUT_S,
            )
            return False

        logger.info("[IgrisHandController] Hand init motion detected. Waiting for motion to settle...")
        if self._wait_until_position_stable():
            logger.info("[IgrisHandController] Hand init motion settled. Starting control loop.")
            return True
        else:
            logger.warning(
                "[IgrisHandController] Hand init motion did not settle within %.1f seconds. Starting control loop anyway.",
                HAND_INIT_STABLE_TIMEOUT_S,
            )
            return False

    def initialize_hand(self, wait_for_state: bool = True) -> bool:
        with self._hand_init_lock:
            if wait_for_state and not self.hand_interface.wait_for_first_state(timeout=5.0):
                logger.warning(
                    "[IgrisHandController] Hand init requested, but %s was not received.",
                    self.hand_interface.state_topic,
                )
                return False
            self._hand_initializing.set()
            try:
                return self._auto_initialize_hand()
            finally:
                self._hand_initializing.clear()

    def ctrl_dual_hand(self, action_data: np.ndarray) -> None:
        if self._hand_initializing.is_set():
            return
        try:
            self.hand_interface.send_targets(action_data.tolist())
        except Exception as exc:
            logger.error(f"[IgrisHandController] Failed to send targets: {exc}")

    def control_process(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        hybrid_hand_command_lock=None,
        left_hand_close_array=None,
        right_hand_close_array=None,
        hand_close_valid_array=None,
        left_hand_motor_array=None,
        right_hand_motor_array=None,
        hand_motor_valid_array=None,
    ):
        self.running = True

        last_rate_log = time.perf_counter()
        loop_cnt = 0
        try:
            while self.running:
                hz_now = self.rate.tick_hz()
                loop_cnt += 1
                now = time.perf_counter()
                if (now - last_rate_log) >= 1.0 and hz_now > 0:
                    last_rate_log = now
                    loop_cnt = 0

                left_hand_mat = np.array(left_hand_array[:], dtype=float).reshape(5, 3).copy()
                right_hand_mat = np.array(right_hand_array[:], dtype=float).reshape(5, 3).copy()

                present = np.array(self.hand_interface.get_present_position(), dtype=float)
                if present.size == HAND_TARGET_LENGTH:
                    state_data = present
                else:
                    state_data = np.zeros(HAND_TARGET_LENGTH, dtype=float)

                left_close = None
                right_close = None
                left_close_valid = False
                right_close_valid = False
                if (
                    hybrid_hand_command_lock is not None
                    and left_hand_close_array is not None
                    and right_hand_close_array is not None
                    and hand_close_valid_array is not None
                ):
                    with hybrid_hand_command_lock:
                        left_close = np.asarray(left_hand_close_array[:], dtype=np.float64)
                        right_close = np.asarray(right_hand_close_array[:], dtype=np.float64)
                        close_valid = np.asarray(hand_close_valid_array[:], dtype=np.float64)
                    left_close_valid = bool(
                        close_valid.size >= 1 and close_valid[0] > 0.5
                    )
                    right_close_valid = bool(
                        close_valid.size >= 2 and close_valid[1] > 0.5
                    )

                left_motor = right_motor = None
                left_motor_valid = right_motor_valid = False
                if (
                    hybrid_hand_command_lock is not None
                    and left_hand_motor_array is not None
                    and right_hand_motor_array is not None
                    and hand_motor_valid_array is not None
                ):
                    with hybrid_hand_command_lock:
                        left_motor = np.asarray(left_hand_motor_array[:], dtype=np.float64)
                        right_motor = np.asarray(right_hand_motor_array[:], dtype=np.float64)
                        motor_valid = np.asarray(hand_motor_valid_array[:], dtype=np.float64)
                    left_motor_valid = bool(motor_valid.size >= 1 and motor_valid[0] > 0.5)
                    right_motor_valid = bool(motor_valid.size >= 2 and motor_valid[1] > 0.5)

                if (left_motor_valid or left_close_valid) and (right_motor_valid or right_close_valid):
                    action_data = np.zeros(HAND_TARGET_LENGTH, dtype=np.float64)
                else:
                    action_data = self.hand_retargeting.retarget_normalized(
                        left_hand_mat,
                        right_hand_mat,
                    )
                action_data = apply_finger_close_overrides(
                    action_data,
                    left_close=left_close,
                    right_close=right_close,
                    left_valid=left_close_valid and not left_motor_valid,
                    right_valid=right_close_valid and not right_motor_valid,
                    dtype=np.float64,
                )
                action_data = apply_motor_command_overrides(
                    action_data, left_motor=left_motor, right_motor=right_motor,
                    left_valid=left_motor_valid, right_valid=right_motor_valid,
                    dtype=np.float64,
                )

                if (
                    dual_hand_data_lock is not None
                    and dual_hand_state_array is not None
                    and dual_hand_action_array is not None
                ):
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self.ctrl_dual_hand(action_data)
                self.rate.sleep()

        except KeyboardInterrupt:
            logger.error("KeyboardInterrupt, exiting IgrisHandController...")
        except Exception as exc:
            logger.error(f"[IgrisHandController Main Error] {exc}")
            traceback.print_exc()
        finally:
            logger.info("IgrisHandController has been closed.")
            self.hand_interface.stop()

    def close(self, timeout_s: float = 1.0) -> None:
        self.running = False

        hand_control_thread = getattr(self, "_hand_control_thread", None)
        if hand_control_thread is not None and hand_control_thread.is_alive():
            try:
                hand_control_thread.join(timeout=max(0.1, float(timeout_s)))
            except Exception:
                logger.debug("[IgrisHandController] failed to join hand control thread.", exc_info=True)

        self.hand_interface.stop()
