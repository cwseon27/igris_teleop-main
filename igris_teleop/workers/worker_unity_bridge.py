from __future__ import annotations

import threading
import time
import numpy as np

try:
    import rclpy
except ImportError:
    rclpy = None

from ..core.events import EventSnapshot
from ..core.math.transforms import common_frame_relative_pose
from ..core.state_machine import ModeState, TransitionResult
from ..core.worker_base import SingleRateWorker, WorkerContext, teleop_uses_hybrid_torso
from ..head_start_guard import (
    evaluate_head_start_guard,
    format_head_guard_message,
    head_guard_reason_label,
    inactive_head_guard_status,
    is_valid_pose_matrix,
)

import logging_mp
logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)

# 기존:
# from ..robot_control.interfaces.master_arm_ros_interface import MasterArmROSInterface
# 변경: Pose 기반 인터페이스 import (프로젝트 경로/파일명에 맞게 수정)
from ..teleop_devices.unity.unity_ros_interface import TelevisionROSInterface


class UnityRosridgeWorker(SingleRateWorker):
    """
    ROS Pose 토픽(/hmd/pose, /right_controller/poses, /left_hand/poses, /right_hand/poses)을 받아
    TELEVISION shm에 기록하며, HOME/RUN 상태에 따라 rel pose를 적용한다.
    """

    def __init__(self, ctx: WorkerContext, hz: float = 50.0) -> None:
        super().__init__(ctx, hz=hz)

        global rclpy
        if rclpy is None:
            raise RuntimeError("rclpy is not available. Please source ROS2 and install rclpy.")

        self._shared_memory = ctx.shared_memory
        self._owns_shared_memory = False

        # teleop pose를 television_shm에 기록한다.
        self.television_shm = self._shared_memory.get("television_shm")
        self.ee_shm = self._shared_memory.get("ee_shm")
        self.teleop_guard_shm = self._shared_memory.get("teleop_guard_shm")

        self.iface: TelevisionROSInterface | None = None
        self._spin_thread: threading.Thread | None = None
        self._guard_enabled = (
            getattr(ctx.run_config, "mode", None) == "teleop"
            and getattr(ctx.run_config, "teleop_device", None) in {"unity", "unity_hybrid", "vr_masterarm"}
        )
        self._hybrid_torso_enabled = teleop_uses_hybrid_torso(
            getattr(ctx.run_config, "teleop_device", None)
        )
        self._head_guard_seq = 0.0

        # HOME 기준 포즈(translation 상대값 계산용)
        self._home_left_wrist = None
        self._home_right_wrist = None
        self._home_head_pose = None
        self._home_chest_pose = None
        self._home_chest_ready = False
        self._robot_home_head_pose = None
        self._robot_home_head_pose_is_fallback = False
        self._guard_last_reason_code: int | None = None
        self._initial_home_wait_logged = False
        self._home_refresh_logged = False
        self._head_stream_wait_logged = False
        self._fallback_home_logged = False
        self._head_stream_wait_last_log_t: float | None = None

        # 초기 수신이 되었는지(기본값 identity/zeros로 HOME 캡처되는 것 방지)
        self._has_minimum_stream = False

    def _spin_ros(self) -> None:
        if self.iface is None:
            return
        try:
            rclpy.spin(self.iface)
        except Exception:
            logger.exception("[ROSBridge] spin failed.")

    def _create_ros_interface(self) -> None:
        logger.info("[ROSBridge] creating TelevisionROSInterface...")
        if self.iface is not None:
            return

        try:
            rclpy.init(args=None)
        except RuntimeError:
            pass

        try:
            self.iface = TelevisionROSInterface(
                hybrid_torso_enabled=self._hybrid_torso_enabled,
            )
        except Exception:
            logger.exception("[ROSBridge] failed to create TelevisionROSInterface")
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

    def _capture_home(
        self,
        head_mat: np.ndarray,
        chest_mat: np.ndarray,
        left_wrist: np.ndarray,
        right_wrist: np.ndarray,
        chest_ready: bool = False,
    ) -> None:
        self._home_left_wrist = left_wrist.copy()
        self._home_right_wrist = right_wrist.copy()
        self._home_head_pose = head_mat.copy()
        if chest_ready:
            self._home_chest_pose = chest_mat.copy()
            self._home_chest_ready = True

    def _read_robot_head_pose(self) -> np.ndarray | None:
        if self.ee_shm is None:
            return None
        try:
            data = self.ee_shm.read_data()
        except Exception:
            return None
        head_mat = data.get("head_mat")
        if not is_valid_pose_matrix(head_mat):
            return None
        return np.asarray(head_mat, dtype=np.float64).reshape(4, 4).copy()

    def _write_head_guard_status(self, status: dict[str, float]) -> None:
        if self.teleop_guard_shm is None:
            return
        self._head_guard_seq += 1.0
        payload = dict(status)
        payload["seq"] = float(self._head_guard_seq)
        try:
            self.teleop_guard_shm.write_data(**payload)
        except Exception:
            logger.debug("[ROSBridge] Failed to write teleop_guard_shm.", exc_info=True)

    def _clear_head_guard_status(self) -> None:
        self._write_head_guard_status(inactive_head_guard_status())
        self._guard_last_reason_code = None

    def _format_stream_status(self) -> str:
        if self.iface is None or not hasattr(self.iface, "get_stream_status"):
            return "stream_status=unavailable"
        try:
            status = self.iface.get_stream_status()
        except Exception:
            return "stream_status=unavailable"
        parts = []
        for key, (count, age) in status.items():
            age_text = "never" if age is None else f"{age:.2f}s"
            parts.append(f"{key}:{count}@{age_text}")
        return " ".join(parts)

    def _maybe_seed_initial_home_reference(
        self,
        head_mat: np.ndarray,
        chest_mat: np.ndarray,
        left_wrist: np.ndarray,
        right_wrist: np.ndarray,
        chest_ready: bool = False,
    ) -> None:
        if chest_ready:
            self._home_chest_pose = chest_mat.copy()
            self._home_chest_ready = True

        robot_head_pose = self._read_robot_head_pose()
        if robot_head_pose is not None and (
            self._robot_home_head_pose is None or self._robot_home_head_pose_is_fallback
        ):
            replacing_fallback = self._robot_home_head_pose_is_fallback
            self._robot_home_head_pose = robot_head_pose
            self._robot_home_head_pose_is_fallback = False
            self._initial_home_wait_logged = False
            if replacing_fallback:
                self._capture_home(
                    head_mat,
                    chest_mat,
                    left_wrist,
                    right_wrist,
                    chest_ready=chest_ready,
                )
                logger.info(
                    "[ROSBridge] robot HOME head reference captured from ee_shm; "
                    "VR HOME references rebased atomically"
                )
            else:
                logger.info("[ROSBridge] robot HOME head reference captured from ee_shm")

        if self._home_head_pose is not None and self._robot_home_head_pose is not None:
            return

        self._capture_home(
            head_mat,
            chest_mat,
            left_wrist,
            right_wrist,
            chest_ready=chest_ready,
        )
        if self._robot_home_head_pose is None:
            self._robot_home_head_pose = head_mat.copy()
            self._robot_home_head_pose_is_fallback = True
            if not self._fallback_home_logged:
                logger.info(
                    "[ROSBridge] initial HOME captured from HMD stream; "
                    "using HMD self-reference until ee_shm head pose is available"
                )
                self._fallback_home_logged = True
        else:
            logger.info("[ROSBridge] initial set_home captured before VR head start guard")

    def _update_head_guard(self, head_mat: np.ndarray | None) -> None:
        if not self._guard_enabled:
            return
        if self.state == ModeState.RUN:
            self._clear_head_guard_status()
            return
        status = evaluate_head_start_guard(
            head_mat,
            self._home_head_pose,
            self._robot_home_head_pose,
        )
        reason_code = int(round(float(status["head_reason_code"])))
        if reason_code != self._guard_last_reason_code:
            logger.info(
                "[ROSBridge] VR head start guard: %s | %s",
                head_guard_reason_label(reason_code),
                format_head_guard_message(status),
            )
            self._guard_last_reason_code = reason_code
        self._write_head_guard_status(status)

    def on_start(self) -> None:
        logger.info(f"[{self.ctx.name}] start (single-rate {self.hz} Hz)")
        if self._guard_enabled:
            self._clear_head_guard_status()

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        if not self._ensure_ros_ready(ev):
            if self._guard_enabled:
                self._clear_head_guard_status()
            return
        if self.iface is None:
            self._update_head_guard(None)
            return

        st = self.state
        try:
            # 인터페이스가 “고정 shape (4x4,4x4,4x4,4x4,5x3,5x3)”로 반환한다고 가정
            head_mat, chest_mat, left_wrist, right_wrist, left_hand, right_hand = self.iface.get_television_payload()
            chest_alpha, left_controller_confidence, right_controller_confidence, chest_ready = (
                self.iface.get_chest_reliability()
            )
            (
                left_hand_close,
                right_hand_close,
                left_hand_close_valid,
                right_hand_close_valid,
            ) = self.iface.get_hybrid_hand_commands()
            (
                left_hand_motor, right_hand_motor,
                left_hand_motor_valid, right_hand_motor_valid,
            ) = self.iface.get_hybrid_hand_motor_commands()
        except Exception:
            logger.error("[ROSBridge] Failed to read television payload from ROS iface.", exc_info=True)
            self._update_head_guard(None)
            return

        # dtype/shape 안정화(특히 shm writer가 엄격한 경우 대비)
        head_mat = np.asarray(head_mat, dtype=np.float64)
        chest_mat = np.asarray(chest_mat, dtype=np.float64)
        left_wrist = np.asarray(left_wrist, dtype=np.float64)
        right_wrist = np.asarray(right_wrist, dtype=np.float64)
        left_hand = np.asarray(left_hand, dtype=np.float64)
        right_hand = np.asarray(right_hand, dtype=np.float64)
        left_hand_close = np.asarray(left_hand_close, dtype=np.float64).reshape(5)
        right_hand_close = np.asarray(right_hand_close, dtype=np.float64).reshape(5)

        # 2) 최소 1회 수신 확인 (초기 default 값으로 HOME 캡처 방지)
        #    인터페이스에 wait_for_all()이 있다면 그것을 이용해 “초기 수신 완료”를 판정.
        if not self._has_minimum_stream:
            try:
                # Head-only upper-body tests should not be blocked by wrist/hand topics.
                has_head_stream = (
                    self.iface.has_head_stream()
                    if hasattr(self.iface, "has_head_stream")
                    else self.iface.wait_for_all(timeout=0.0)
                )
                if not has_head_stream:
                    now_mono = time.monotonic()
                    if (
                        not self._head_stream_wait_logged
                        or self._head_stream_wait_last_log_t is None
                        or now_mono - self._head_stream_wait_last_log_t >= 1.0
                    ):
                        logger.info(
                            "[ROSBridge] waiting for /hmd/pose before HOME capture (%s)",
                            self._format_stream_status(),
                        )
                        self._head_stream_wait_logged = True
                        self._head_stream_wait_last_log_t = now_mono
                    self._update_head_guard(None)
                    return
                self._has_minimum_stream = True
                self._head_stream_wait_logged = False
            except Exception:
                # wait_for_all이 없거나 예외면, 그대로 진행(보수적으로는 return이 맞지만 현장 편의상 진행)
                self._has_minimum_stream = True

        # 3) HOME 기준 캡처 로직
        if st == ModeState.HOME:
            if tr.reason == "home_set":
                self._home_chest_pose = None
                self._home_chest_ready = False
            self._capture_home(
                head_mat,
                chest_mat,
                left_wrist,
                right_wrist,
                chest_ready=chest_ready,
            )
            robot_head_pose = self._read_robot_head_pose()
            if robot_head_pose is not None:
                self._robot_home_head_pose = robot_head_pose
                self._robot_home_head_pose_is_fallback = False
                if tr.reason == "home_set" or not self._home_refresh_logged:
                    logger.info("[ROSBridge] HOME updated set_home reference for VR head start guard")
                    self._home_refresh_logged = True
            else:
                self._robot_home_head_pose = head_mat.copy()
                self._robot_home_head_pose_is_fallback = True
                if tr.reason == "home_set" or not self._home_refresh_logged:
                    logger.info(
                        "[ROSBridge] HOME updated from HMD stream; "
                        "ee_shm head pose unavailable, using self-reference guard"
                    )
                    self._home_refresh_logged = True
        elif st in (ModeState.WAIT_START, ModeState.PAUSE):
            self._home_refresh_logged = False
            self._maybe_seed_initial_home_reference(
                head_mat,
                chest_mat,
                left_wrist,
                right_wrist,
                chest_ready=chest_ready,
            )
        elif st == ModeState.RUN and (
            self._home_left_wrist is None
            or self._home_right_wrist is None
            or self._home_head_pose is None
        ):
            # RUN으로 바로 시작해도 한 번은 HOME처럼 기준 포즈를 캡처
            self._capture_home(
                head_mat,
                chest_mat,
                left_wrist,
                right_wrist,
                chest_ready=chest_ready,
            )

        if st == ModeState.RUN and chest_ready and not self._home_chest_ready:
            # A controller may appear after RUN starts. Anchor its first reliable
            # chest candidate before allowing the chest task to activate.
            self._home_chest_pose = chest_mat.copy()
            self._home_chest_ready = True
            chest_alpha = 0.0
            logger.info("[ROSBridge] captured late controller/chest HOME reference")

        # 4) rel pose 적용: home 대비 상대값
        if (
            self._home_left_wrist is not None
            and self._home_right_wrist is not None
            and self._home_head_pose is not None
        ):
            rel_left_wrist = common_frame_relative_pose(
                left_wrist, self._home_left_wrist
            )
            rel_right_wrist = common_frame_relative_pose(
                right_wrist, self._home_right_wrist
            )
            rel_head_pose = common_frame_relative_pose(
                head_mat, self._home_head_pose
            )
        else:
            rel_left_wrist = left_wrist
            rel_right_wrist = right_wrist
            rel_head_pose = head_mat

        if self._home_chest_ready and self._home_chest_pose is not None:
            rel_chest_pose = common_frame_relative_pose(
                chest_mat, self._home_chest_pose
            )
        else:
            rel_chest_pose = np.eye(4, dtype=np.float64)
            chest_alpha = 0.0

        self._update_head_guard(head_mat)

        # 5) RUN일 때만 television_shm write
        if st == ModeState.RUN:
            try:
                self.television_shm.write_data(
                    head_mat=rel_head_pose,
                    torso_mat=np.zeros((4, 4), dtype=np.float64),
                    chest_mat=rel_chest_pose,
                    left_wrist_mat=rel_left_wrist,
                    right_wrist_mat=rel_right_wrist,
                    left_hand=left_hand,
                    right_hand=right_hand,
                    left_hand_close=left_hand_close,
                    right_hand_close=right_hand_close,
                    left_hand_close_valid=1.0 if left_hand_close_valid else 0.0,
                    right_hand_close_valid=1.0 if right_hand_close_valid else 0.0,
                    left_hand_motor=left_hand_motor,
                    right_hand_motor=right_hand_motor,
                    left_hand_motor_valid=float(left_hand_motor_valid),
                    right_hand_motor_valid=float(right_hand_motor_valid),
                    torso_alpha=0.0,
                    chest_alpha=float(np.clip(chest_alpha, 0.0, 1.0)),
                    left_controller_confidence=float(
                        np.clip(left_controller_confidence, 0.0, 1.0)
                    ),
                    right_controller_confidence=float(
                        np.clip(right_controller_confidence, 0.0, 1.0)
                    ),
                    torso_source_valid=0.0,
                    chest_source_valid=(
                        1.0
                        if self._home_chest_ready and (chest_ready or chest_alpha > 0.0)
                        else 0.0
                    ),
                )
            except Exception:
                logger.error("[ROSBridge] Failed to write television_shm.", exc_info=True)

        # HOME에서는 캡처만, 기타 상태에서는 write 하지 않음(필요시 정책 변경)

    def on_stop(self) -> None:
        if self._guard_enabled:
            self._clear_head_guard_status()

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
            if hasattr(rclpy, "try_shutdown"):
                rclpy.try_shutdown()
            elif rclpy.ok():
                rclpy.shutdown()
        except Exception:
            logger.exception("[ROSBridge] failed to shutdown rclpy.")

        if getattr(self, "_spin_thread", None):
            self._spin_thread.join(timeout=1.0)

        logger.info(f"[{self.ctx.name}] stop")
