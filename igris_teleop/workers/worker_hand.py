from __future__ import annotations

import time
import os
import shlex
import subprocess
import uuid
from pathlib import Path
import numpy as np
from multiprocessing import Array, Lock

from ..core.events import EventSnapshot
from ..core.state_machine import ModeState, TransitionResult
from ..core.worker_base import (
    SingleRateWorker,
    WorkerContext,
    resolve_robot_dds_domain_id,
    resolve_teleop_hand_source,
)
from ..hand_control.command_range import (
    apply_finger_close_overrides, apply_motor_command_overrides, validated_motor_command,
)


import logging_mp
logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)

HAND_INIT_EVENT_NAME = "hand_init"
SIM_RUNTIME_ENVIRONMENTS = frozenset({"sim", "simulator", "simulation", "mujoco"})
DEFAULT_HAND_INIT_SERVICE = "/igris_c_IG05/rt/service/hand_init"
DEFAULT_HAND_DDS_NAMESPACE = "igris_c_IG05"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class HandWorker(SingleRateWorker):
    """Single-rate 워커 예제: 상태에 따라 카운터를 업데이트."""

    def __init__(self, ctx: WorkerContext, hz: float = 10.0) -> None:
        super().__init__(ctx, hz=hz)

        self._shared_memory = ctx.shared_memory
        self._shm_name = ctx.shm_name
        self._shared_lock = ctx.shared_lock
        self._owns_shared_memory = False

        self.television_shm = self._shared_memory.get("television_shm")
        self.obs_shm = self._shared_memory.get("obs_shm")
        self.act_shm = self._shared_memory.get("act_shm")
        self.teleop_device = ctx.run_config.teleop_device
        self.teleop_hand_source = resolve_teleop_hand_source(
            self.teleop_device,
            ctx.run_config.teleop_hand_source,
        )
        self._direct_hand_target_mode = self.teleop_hand_source == "masterarm"
        runtime_environment = str(getattr(ctx.run_config, "runtime_environment", None) or "").lower()
        self._simulation_mode = runtime_environment in SIM_RUNTIME_ENVIRONMENTS
        self._sim_vr_retarget_mode = self._simulation_mode and self.teleop_hand_source == "vr"
        self._hand_domain_id = resolve_robot_dds_domain_id(runtime_environment)
        # The simulator publishes unscoped rt/* topics.  The physical robot's
        # SDK applies its AP namespace, which resolves to igris_c_IG05.
        self._hand_dds_namespace = "" if self._simulation_mode else DEFAULT_HAND_DDS_NAMESPACE
        self._waiting_for_body_obs = False
        
        # retarget 입력/출력 버퍼
        self.left_hand_array = Array("d", 5 * 3, lock=True)
        self.right_hand_array = Array("d", 5 * 3, lock=True)
        self.dual_hand_data_lock = Lock()
        self.dual_hand_state_array = Array("d", 12, lock=False)
        self.dual_hand_action_array = Array("d", 12, lock=False)
        self.hybrid_hand_command_lock = Lock()
        self.left_hand_close_array = Array("d", 5, lock=False)
        self.right_hand_close_array = Array("d", 5, lock=False)
        self.hand_close_valid_array = Array("d", 2, lock=False)
        self.left_hand_motor_array = Array("d", 6, lock=False)
        self.right_hand_motor_array = Array("d", 6, lock=False)
        self.hand_motor_valid_array = Array("d", 2, lock=False)

        self.hand_ctrl = None
        self.sim_hand_retargeting = None
        self._last_sim_retarget_init_attempt_at = 0.0
        self._last_sim_retarget_log_at = 0.0
        self._last_sim_retarget_error_at = 0.0

    def on_start(self) -> None:
        logger.info(f"[{self.ctx.name}] start (single-rate {self.hz} Hz)")
        if self._direct_hand_target_mode:
            logger.info(
                "[HandWorker] direct hand target mode enabled for teleop_device=%s hand_source=%s",
                self.teleop_device,
                self.teleop_hand_source,
            )
        if self._simulation_mode:
            logger.info(
                "[HandWorker] simulation SHM hand path enabled (hand_source=%s, dex_retarget=%s)",
                self.teleop_hand_source,
                self._sim_vr_retarget_mode,
            )

    def _init_sim_retargeter(self) -> bool:
        if not self._sim_vr_retarget_mode or self.sim_hand_retargeting is not None:
            return True
        now = time.monotonic()
        if now - self._last_sim_retarget_init_attempt_at < 2.0:
            return False
        self._last_sim_retarget_init_attempt_at = now
        try:
            from ..hand_control.hand_retargeting import HandRetargeting

            self.sim_hand_retargeting = HandRetargeting()
            logger.info("[HandWorker] simulation DexRetargeting initialized.")
            return True
        except Exception:
            logger.exception("[HandWorker] simulation DexRetargeting init failed.")
            return False

    def _init_controller(self) -> bool:
        if self.hand_ctrl is not None:
            return True
        try:
            from ..hand_control.robot_hand import IgrisHandController

            self.hand_ctrl = IgrisHandController(
                shm_name=self._shm_name,
                shared_lock=self._shared_lock,
                left_hand_array=self.left_hand_array,
                right_hand_array=self.right_hand_array,
                dual_hand_data_lock=self.dual_hand_data_lock,
                dual_hand_state_array=self.dual_hand_state_array,
                dual_hand_action_array=self.dual_hand_action_array,
                hybrid_hand_command_lock=self.hybrid_hand_command_lock,
                left_hand_close_array=self.left_hand_close_array,
                right_hand_close_array=self.right_hand_close_array,
                hand_close_valid_array=self.hand_close_valid_array,
                left_hand_motor_array=self.left_hand_motor_array,
                right_hand_motor_array=self.right_hand_motor_array,
                hand_motor_valid_array=self.hand_motor_valid_array,
                fps=100.0,
                Unit_Test=False,
                domain_id=self._hand_domain_id,
                dds_namespace=self._hand_dds_namespace,
                transport="ros_bridge",
                start_control_thread=not self._direct_hand_target_mode,
                auto_initialize=False,
            )
            logger.info("[HandWorker] Hand controller initialized.")
            return True
        except Exception:
            logger.exception("[HandWorker] Hand init failed.")
            return False

    def _body_controller_ready(self) -> bool:
        if self.obs_shm is None:
            return False
        try:
            data = self.obs_shm.read_data()
            obs_seq = float(np.asarray(data.get("obs_seq", np.nan), dtype=np.float64).reshape(()).item())
        except Exception:
            return False
        return bool(np.isfinite(obs_seq) and obs_seq > 0.0)

    def _ensure_controller_ready(self, ev: EventSnapshot) -> bool:
        if self._simulation_mode:
            return self._init_sim_retargeter()
        if self.hand_ctrl is not None:
            return True
        if not ev.level.get("ready", False):
            self._waiting_for_body_obs = False
            return False
        # Hand transport runs in its own external process and no longer shares
        # the body SDK ChannelFactory.  Waiting on obs_seq here can strand the
        # hand worker even after Ready has completed, preventing both bridge
        # startup and hand-init event handling.
        self._waiting_for_body_obs = False
        return self._init_controller()

    def _publish_hand_data(self) -> None:
        """현재 핸드 상태/명령을 SHM에 반영."""
        with self.dual_hand_data_lock:
            state = np.asarray(self.dual_hand_state_array[:], dtype=np.float64)
            action = np.asarray(self.dual_hand_action_array[:], dtype=np.float64)
        if self.obs_shm is not None:
            self.obs_shm.write_data(obs_hand=state)
        if self.act_shm is not None:
            self.act_shm.write_data(act_hand=action)

    def _read_direct_hand_target(self) -> np.ndarray | None:
        if self.act_shm is None:
            return None
        try:
            data = self.act_shm.read_data()
            target = np.asarray(data.get("act_hand"), dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if target.size != 12:
            return None
        if not np.all(np.isfinite(target)):
            return None
        return target.copy()

    def _publish_direct_hand_snapshot(self, target: np.ndarray | None = None) -> None:
        if self.hand_ctrl is None:
            return

        try:
            state = np.asarray(self.hand_ctrl.hand_interface.get_present_position(), dtype=np.float64).reshape(-1)
        except Exception:
            logger.debug("[HandWorker] failed to read present hand position.", exc_info=True)
            return

        if state.size == 12 and self.obs_shm is not None:
            try:
                self.obs_shm.write_data(obs_hand=state)
            except Exception:
                logger.debug("[HandWorker] failed to write obs_hand in direct mode.", exc_info=True)

        if target is None:
            target = state

    @staticmethod
    def _normalized_close_inputs(
        data: dict,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        left = np.asarray(data.get("left_hand_close", np.zeros(5)), dtype=np.float64).reshape(-1)
        right = np.asarray(data.get("right_hand_close", np.zeros(5)), dtype=np.float64).reshape(-1)
        try:
            left_valid = float(np.asarray(data.get("left_hand_close_valid", 0.0)).reshape(())) > 0.5
            right_valid = float(np.asarray(data.get("right_hand_close_valid", 0.0)).reshape(())) > 0.5
        except Exception:
            left_valid = False
            right_valid = False
        left_valid = bool(left_valid and left.size == 5 and np.all(np.isfinite(left)))
        right_valid = bool(right_valid and right.size == 5 and np.all(np.isfinite(right)))
        if left.size != 5:
            left = np.zeros(5, dtype=np.float64)
        if right.size != 5:
            right = np.zeros(5, dtype=np.float64)
        return (
            np.clip(left, 0.0, 1.0),
            np.clip(right, 0.0, 1.0),
            left_valid,
            right_valid,
        )

    @staticmethod
    def _normalized_motor_inputs(data: dict) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        commands = []
        flags = []
        for side in ("left", "right"):
            try:
                valid = float(np.asarray(data.get(f"{side}_hand_motor_valid", 0.0)).reshape(())) > 0.5
                command = validated_motor_command(data.get(f"{side}_hand_motor", np.zeros(6)))
            except (ValueError, TypeError):
                command, valid = np.zeros(6), False
            commands.append(command)
            flags.append(valid)
        return commands[0], commands[1], flags[0], flags[1]

    def _write_normalized_close_inputs(self, data: dict) -> None:
        left, right, left_valid, right_valid = self._normalized_close_inputs(data)
        left_motor, right_motor, left_motor_valid, right_motor_valid = self._normalized_motor_inputs(data)
        with self.hybrid_hand_command_lock:
            self.left_hand_close_array[:] = left
            self.right_hand_close_array[:] = right
            self.hand_close_valid_array[:] = (
                1.0 if left_valid else 0.0,
                1.0 if right_valid else 0.0,
            )
            self.left_hand_motor_array[:] = left_motor
            self.right_hand_motor_array[:] = right_motor
            self.hand_motor_valid_array[:] = (float(left_motor_valid), float(right_motor_valid))

    def _publish_sim_vr_hand_target(self) -> None:
        if self.sim_hand_retargeting is None or self.television_shm is None or self.act_shm is None:
            return
        try:
            data = self.television_shm.read_data()
            left_hand = np.asarray(data.get("left_hand"), dtype=np.float64).reshape(5, 3)
            right_hand = np.asarray(data.get("right_hand"), dtype=np.float64).reshape(5, 3)
            left_close, right_close, left_close_valid, right_close_valid = (
                self._normalized_close_inputs(data)
            )
            left_motor, right_motor, left_motor_valid, right_motor_valid = self._normalized_motor_inputs(data)
            if (left_motor_valid or left_close_valid) and (right_motor_valid or right_close_valid):
                base_target = np.zeros(12, dtype=np.float64)
            else:
                base_target = np.asarray(
                    self.sim_hand_retargeting.retarget_normalized(left_hand, right_hand),
                    dtype=np.float64,
                ).reshape(-1)
            target = apply_finger_close_overrides(
                base_target,
                left_close=left_close,
                right_close=right_close,
                left_valid=left_close_valid and not left_motor_valid,
                right_valid=right_close_valid and not right_motor_valid,
                dtype=np.float64,
            )
            target = apply_motor_command_overrides(
                target, left_motor=left_motor, right_motor=right_motor,
                left_valid=left_motor_valid, right_valid=right_motor_valid,
                dtype=np.float64,
            )
            if target.size != 12 or not np.all(np.isfinite(target)):
                raise ValueError(f"invalid normalized hand target shape/value: {target}")
            target = np.clip(target, 0.0, 1.0)
            self.act_shm.write_data(act_hand=target)
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_sim_retarget_error_at >= 2.0:
                self._last_sim_retarget_error_at = now
                logger.warning("[HandWorker] simulation hand retarget failed: %s", exc)
            return

        now = time.monotonic()
        if now - self._last_sim_retarget_log_at >= 2.0:
            self._last_sim_retarget_log_at = now
            logger.info(
                "[HandWorker] sim act_hand bend min=%.3f max=%.3f values=%s",
                float(np.min(target)),
                float(np.max(target)),
                np.array2string(target, precision=2, suppress_small=True),
            )

    def _handle_hand_init_request(self, ev: EventSnapshot) -> bool:
        if not ev.level.get(HAND_INIT_EVENT_NAME, False):
            return False

        try:
            self.ctx.bus.clear_level(HAND_INIT_EVENT_NAME)
        except Exception:
            logger.debug("[HandWorker] failed to clear hand init event.", exc_info=True)

        if self._simulation_mode:
            if self.act_shm is not None:
                self.act_shm.write_data(act_hand=np.zeros(12, dtype=np.float64))
            logger.info("[HandWorker] simulation hand target reset to open.")
            return True

        logger.info("[HandWorker] hand init requested.")
        # Keep initialization on the robot ROS service that is known to drive
        # the hardware correctly.  The native bridge is reserved for continuous
        # HandCmd traffic.  This also permits init before the local controller
        # has finished starting.
        ok = self._call_robot_hand_init_service()
        if not ok and self.hand_ctrl is not None:
            try:
                ok = bool(self.hand_ctrl.initialize_hand())
            except Exception:
                logger.exception("[HandWorker] hand init bridge fallback failed.")
                ok = False

        if ok:
            logger.info("[HandWorker] hand init completed.")
        else:
            logger.warning("[HandWorker] hand init command did not complete cleanly.")
        return True

    def _call_robot_hand_init_service(self) -> bool:
        service_name = (
            os.getenv("IGRIS_ROS2_HAND_INIT_SERVICE") or DEFAULT_HAND_INIT_SERVICE
        ).strip()
        if not service_name:
            return False

        repo_root = _repo_root()
        ros_setup = Path("/opt/ros/jazzy/setup.bash")
        ws_setup = repo_root / "ros_ws" / "install" / "setup.bash"
        if not ros_setup.is_file() or not ws_setup.is_file():
            logger.warning(
                "[HandWorker] ROS2 hand init fallback unavailable: missing %s or %s",
                ros_setup,
                ws_setup,
            )
            return False

        request_id = f"igris_teleop_hand_{uuid.uuid4().hex}"
        payload = "{request_id: " + request_id + "}"
        command = " && ".join(
            (
                f"source {shlex.quote(str(ros_setup))}",
                f"source {shlex.quote(str(ws_setup))}",
                "export ROS_DOMAIN_ID=0",
                "ros2 service call "
                f"{shlex.quote(service_name)} "
                "igris_c_sdk/srv/HandInitRequest "
                f"{shlex.quote(payload)}",
            )
        )
        logger.info("[HandWorker] trying ROS2 hand init fallback: %s", service_name)
        try:
            proc = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(repo_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=float(os.getenv("IGRIS_ROS2_HAND_INIT_TIMEOUT_S", "20.0")),
                check=False,
            )
        except Exception:
            logger.exception("[HandWorker] ROS2 hand init fallback failed to execute.")
            return False

        output = f"{proc.stdout}\n{proc.stderr}".strip()
        if proc.returncode == 0 and "success=True" in output:
            logger.info("[HandWorker] ROS2 hand init fallback succeeded.")
            return True

        logger.warning(
            "[HandWorker] ROS2 hand init fallback failed rc=%s output=%s",
            proc.returncode,
            output[-1000:],
        )
        return False

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        st = self.state

        # Hand initialization is an independent robot service and must not be
        # blocked by body observation or local bridge readiness.
        if ev.level.get(HAND_INIT_EVENT_NAME, False):
            self._handle_hand_init_request(ev)
            return

        if not self._ensure_controller_ready(ev):
            return

        if self._simulation_mode:
            if st == ModeState.RUN and self._sim_vr_retarget_mode:
                self._publish_sim_vr_hand_target()
            return

        if self._direct_hand_target_mode:
            target = self._read_direct_hand_target()
            if st != ModeState.RUN:
                self._publish_direct_hand_snapshot(target=target)
                return
            if target is None:
                return
            try:
                self.hand_ctrl.ctrl_dual_hand(target)
            except Exception:
                logger.exception("[HandWorker] failed to send direct hand targets")
                return
            self._publish_direct_hand_snapshot(target=target)
            return

        if st != ModeState.RUN:
            return

        data = self.television_shm.read_data()

        left_hand = np.asarray(data.get("left_hand"), dtype=np.float64).reshape(-1)
        right_hand = np.asarray(data.get("right_hand"), dtype=np.float64).reshape(-1)
        self._write_normalized_close_inputs(data)

        with self.left_hand_array.get_lock():
            self.left_hand_array[:] = left_hand
        with self.right_hand_array.get_lock():
            self.right_hand_array[:] = right_hand

        self._publish_hand_data()
        
        
    def on_stop(self) -> None:
        if self.hand_ctrl is not None:
            try:
                self.hand_ctrl.close(timeout_s=1.0)
            except Exception:
                logger.exception(f"[{self.ctx.name}] failed to close hand controller")

        if self._shared_memory:
            for key, mgr in self._shared_memory.items():
                try:
                    mgr.worker_close()
                except Exception:
                    logger.exception(f"[{self.ctx.name}] failed to close shared memory {key}")

        logger.info(f"[{self.ctx.name}] stop")
