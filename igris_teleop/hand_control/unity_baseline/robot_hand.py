import threading
import time
import traceback

import numpy as np

from .hand_retargeting import HandRetargeting
from .command_range import compress_normalized_hand_command
from ...core.rate import Rate

import logging_mp

logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)

IGRIS_Num_Motors = 6
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

try:
    import igris_c_sdk as igc_sdk
except ImportError:
    igc_sdk = None
    logger.error("igris_c_sdk is not available. Please install the SDK wheel.")


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class IgrisHandDDSInterface:
    """Direct DDS interface for the 12-DOF IGRIS hand."""

    def __init__(self, domain_id: int = DEFAULT_HAND_DOMAIN_ID) -> None:
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

        self._state_sub = igc_sdk.HandStateSubscriber("rt/handstate")
        if not self._state_sub.init(self._on_hand_state):
            self._cleanup_dds()
            raise RuntimeError("Failed to init HandStateSubscriber(rt/handstate)")

        self._cmd_pub = igc_sdk.HandCmdPublisher("rt/handcmd")
        if not self._cmd_pub.init():
            self._cleanup_dds()
            raise RuntimeError("Failed to init HandCmdPublisher(rt/handcmd)")

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
        fps: float = 100.0,
        Unit_Test: bool = False,
        domain_id: int = DEFAULT_HAND_DOMAIN_ID,
        start_control_thread: bool = True,
        auto_initialize: bool = True,
    ):
        del shm_name, shared_lock

        logger.info("Initialize IgrisHandController...")

        self.rate = Rate(fps)
        self.Unit_Test = Unit_Test
        self.running = False
        self._domain_id = int(domain_id)
        self._hand_init_lock = threading.Lock()
        self._hand_initializing = threading.Event()

        self.hand_interface = IgrisHandDDSInterface(domain_id=self._domain_id)
        self.hand_retargeting = HandRetargeting()

        try:
            if not self.hand_interface.wait_for_first_state(timeout=5.0):
                logger.warning(
                    "[IgrisHandController] Waiting for rt/handstate timed out. Check that the hand controller is running."
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
                    "[IgrisHandController] Hand init requested, but rt/handstate was not received."
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
    ):
        self.running = True

        left_q_target = np.full(IGRIS_Num_Motors, 0.0, dtype=float)
        right_q_target = np.full(IGRIS_Num_Motors, 0.0, dtype=float)

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

                if (
                    not np.all(right_hand_mat == 0.0)
                    and not np.all(left_hand_mat[4] == np.array([0.15, 0.8, -0.3]))
                ):
                    ref_left_value = left_hand_mat
                    ref_right_value = right_hand_mat

                    left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[
                        self.hand_retargeting.left_dex_retargeting_to_hardware
                    ]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[
                        self.hand_retargeting.right_dex_retargeting_to_hardware
                    ]

                    right_q_target[5] = abs(right_q_target[5])

                    def normalize(val, min_val, max_val):
                        return np.clip((val - min_val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(IGRIS_Num_Motors):
                        if idx == 0:
                            left_q_target[idx] = normalize(left_q_target[idx], 0.0, 1.23)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.23)
                        elif 1 <= idx <= 4:
                            left_q_target[idx] = normalize(left_q_target[idx], 0.0, 1.58)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.58)
                        elif idx == 5:
                            left_q_target[idx] = normalize(left_q_target[idx], 0.0, 1.74)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.74)
                else:
                    left_q_target = np.full(IGRIS_Num_Motors, 0.0, dtype=float)
                    right_q_target = np.full(IGRIS_Num_Motors, 0.0, dtype=float)

                action_data = compress_normalized_hand_command(
                    np.concatenate((right_q_target, left_q_target)),
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
