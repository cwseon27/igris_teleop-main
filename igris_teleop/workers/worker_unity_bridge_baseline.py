from __future__ import annotations

import threading
import numpy as np

try:
    import rclpy
except ImportError:
    rclpy = None

from ..core.events import EventSnapshot
from ..core.state_machine import ModeState, TransitionResult
from ..core.worker_base import SingleRateWorker, WorkerContext
from ..head_start_guard_unity_baseline import (
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
from ..teleop_devices.unity.unity_ros_interface_baseline import TelevisionROSInterface


class UnityRosridgeWorker(SingleRateWorker):
    """
    ROS Pose 토픽(/hmd/pose, /right_controller/pose, /left_hand/poses, /right_hand/poses)을 받아
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
            and getattr(ctx.run_config, "teleop_device", None) in {"unity", "vr_masterarm"}
        )
        self._head_guard_seq = 0.0

        # HOME 기준 포즈(translation 상대값 계산용)
        self._home_left_wrist = None
        self._home_right_wrist = None
        self._home_head_pose = None
        self._home_torso_pose = None
        self._robot_home_head_pose = None
        self._guard_last_reason_code: int | None = None
        self._initial_home_wait_logged = False
        self._home_refresh_logged = False

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
            self.iface = TelevisionROSInterface()
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
        torso_mat: np.ndarray,
        left_wrist: np.ndarray,
        right_wrist: np.ndarray,
    ) -> None:
        self._home_left_wrist = left_wrist.copy()
        self._home_right_wrist = right_wrist.copy()
        self._home_head_pose = head_mat.copy()
        self._home_torso_pose = torso_mat.copy()

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

    def _maybe_seed_initial_home_reference(
        self,
        head_mat: np.ndarray,
        torso_mat: np.ndarray,
        left_wrist: np.ndarray,
        right_wrist: np.ndarray,
    ) -> None:
        if self._home_head_pose is not None and self._robot_home_head_pose is not None:
            return

        robot_head_pose = self._read_robot_head_pose()
        if robot_head_pose is None:
            if not self._initial_home_wait_logged:
                logger.info("[ROSBridge] waiting for ee_shm head pose before initial set_home/guard")
                self._initial_home_wait_logged = True
            return

        self._capture_home(head_mat, torso_mat, left_wrist, right_wrist)
        self._robot_home_head_pose = robot_head_pose
        self._initial_home_wait_logged = False
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

    def _relative_pose(self, cur: np.ndarray, home: np.ndarray) -> np.ndarray:
        """
        Compute relative pose: T_rel = inv(T_home) * T_cur
        cur, home: 4x4 homogeneous transforms.

        Returns:
            4x4 relative transform (home frame 기준).
        """
        cur = np.asarray(cur, dtype=np.float64)
        home = np.asarray(home, dtype=np.float64)

        # (선택) 최소한의 shape 체크
        if cur.shape != (4, 4) or home.shape != (4, 4):
            raise ValueError(f"Expected (4,4) transforms, got cur={cur.shape}, home={home.shape}")

        R0 = home[:3, :3]
        t0 = home[:3, 3]
        R  = cur[:3, :3]
        t  = cur[:3, 3]

        rel = np.eye(4, dtype=cur.dtype)
        rel[:3, :3] = R0.T @ R
        rel[:3, 3]  = R0.T @ (t - t0)
        return rel


    def on_start(self) -> None:
        logger.info(f"[{self.ctx.name}] start (single-rate {self.hz} Hz)")
        if self._guard_enabled:
            self._write_head_guard_status(
                evaluate_head_start_guard(
                    None,
                    self._home_head_pose,
                    self._robot_home_head_pose,
                )
            )

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        if not self._ensure_ros_ready(ev):
            self._update_head_guard(None)
            return
        if self.iface is None:
            self._update_head_guard(None)
            return

        st = self.state
        try:
            # 인터페이스가 “고정 shape (4x4,4x4,4x4,4x4,5x3,5x3)”로 반환한다고 가정
            head_mat, torso_mat, left_wrist, right_wrist, left_hand, right_hand = self.iface.get_television_payload()
        except Exception:
            logger.error("[ROSBridge] Failed to read television payload from ROS iface.", exc_info=True)
            self._update_head_guard(None)
            return

        # dtype/shape 안정화(특히 shm writer가 엄격한 경우 대비)
        head_mat = np.asarray(head_mat, dtype=np.float64)
        torso_mat = np.asarray(torso_mat, dtype=np.float64)
        left_wrist = np.asarray(left_wrist, dtype=np.float64)
        right_wrist = np.asarray(right_wrist, dtype=np.float64)
        left_hand = np.asarray(left_hand, dtype=np.float64)
        right_hand = np.asarray(right_hand, dtype=np.float64)

        # 2) 최소 1회 수신 확인 (초기 default 값으로 HOME 캡처 방지)
        #    인터페이스에 wait_for_all()이 있다면 그것을 이용해 “초기 수신 완료”를 판정.
        if not self._has_minimum_stream:
            try:
                # timeout=0.0: 블로킹 없이 현재 수신여부만 체크
                waiter = getattr(self.iface, "wait_for_control_streams", None)
                if waiter is None:
                    waiter = getattr(self.iface, "wait_for_all", None)
                if callable(waiter) and not waiter(timeout=0.0):
                    self._update_head_guard(None)
                    return
                self._has_minimum_stream = True
            except Exception:
                # wait_for_all이 없거나 예외면, 그대로 진행(보수적으로는 return이 맞지만 현장 편의상 진행)
                self._has_minimum_stream = True

        # 3) HOME 기준 캡처 로직
        if st == ModeState.HOME:
            self._capture_home(head_mat, torso_mat, left_wrist, right_wrist)
            robot_head_pose = self._read_robot_head_pose()
            if robot_head_pose is not None:
                self._robot_home_head_pose = robot_head_pose
                if tr.reason == "home_set" or not self._home_refresh_logged:
                    logger.info("[ROSBridge] HOME updated set_home reference for VR head start guard")
                    self._home_refresh_logged = True
        elif st in (ModeState.WAIT_START, ModeState.PAUSE):
            self._home_refresh_logged = False
            self._maybe_seed_initial_home_reference(head_mat, torso_mat, left_wrist, right_wrist)
        elif st == ModeState.RUN and (
            self._home_left_wrist is None
            or self._home_right_wrist is None
            or self._home_head_pose is None
            or self._home_torso_pose is None
        ):
            # RUN으로 바로 시작해도 한 번은 HOME처럼 기준 포즈를 캡처
            self._capture_home(head_mat, torso_mat, left_wrist, right_wrist)

        # 4) rel pose 적용: home 대비 상대값
        if (
            self._home_left_wrist is not None
            and self._home_right_wrist is not None
            and self._home_head_pose is not None
            and self._home_torso_pose is not None
        ):
            # rotation은 그대로, translation만 home 대비 상대값
            rel_left_wrist = left_wrist.copy()
            rel_right_wrist = right_wrist.copy()
            rel_head_pose = head_mat.copy()
            rel_torso_pose = torso_mat.copy()

            rel_left_wrist[:3, 3] = left_wrist[:3, 3] - self._home_left_wrist[:3, 3]
            rel_right_wrist[:3, 3] = right_wrist[:3, 3] - self._home_right_wrist[:3, 3]
            rel_head_pose[:3, 3] = head_mat[:3, 3] - self._home_head_pose[:3, 3]
            rel_torso_pose[:3, 3] = torso_mat[:3, 3] - self._home_torso_pose[:3, 3]
        else:
            rel_left_wrist = left_wrist
            rel_right_wrist = right_wrist
            rel_head_pose = head_mat
            rel_torso_pose = torso_mat

        self._update_head_guard(head_mat)

        # 5) RUN일 때만 television_shm write
        if st == ModeState.RUN:
            try:
                self.television_shm.write_data(
                    head_mat=rel_head_pose,
                    torso_mat=rel_torso_pose,
                    left_wrist_mat=rel_left_wrist,
                    right_wrist_mat=rel_right_wrist,
                    left_hand=left_hand,
                    right_hand=right_hand,
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
