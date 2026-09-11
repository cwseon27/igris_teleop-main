from __future__ import annotations

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


import logging_mp
logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)

from ..hand_control.unity_baseline.robot_hand import IgrisHandController

HAND_INIT_EVENT_NAME = "hand_init"


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
        self._hand_domain_id = resolve_robot_dds_domain_id(getattr(ctx.run_config, "runtime_environment", None))
        self._waiting_for_body_obs = False
        
        # retarget 입력/출력 버퍼
        self.left_hand_array = Array("d", 5 * 3, lock=True)
        self.right_hand_array = Array("d", 5 * 3, lock=True)
        self.dual_hand_data_lock = Lock()
        self.dual_hand_state_array = Array("d", 12, lock=False)
        self.dual_hand_action_array = Array("d", 12, lock=False)

        self.hand_ctrl = None

    def on_start(self) -> None:
        logger.info(f"[{self.ctx.name}] start (single-rate {self.hz} Hz)")
        if self._direct_hand_target_mode:
            logger.info(
                "[HandWorker] direct hand target mode enabled for teleop_device=%s hand_source=%s",
                self.teleop_device,
                self.teleop_hand_source,
            )

    def _init_controller(self) -> bool:
        if self.hand_ctrl is not None:
            return True
        try:
            self.hand_ctrl = IgrisHandController(
                shm_name=self._shm_name,
                shared_lock=self._shared_lock,
                left_hand_array=self.left_hand_array,
                right_hand_array=self.right_hand_array,
                dual_hand_data_lock=self.dual_hand_data_lock,
                dual_hand_state_array=self.dual_hand_state_array,
                dual_hand_action_array=self.dual_hand_action_array,
                fps=100.0,
                Unit_Test=False,
                domain_id=self._hand_domain_id,
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
        if self.hand_ctrl is not None:
            return True
        if not ev.level.get("ready", False):
            self._waiting_for_body_obs = False
            return False
        if not self._body_controller_ready():
            if not self._waiting_for_body_obs:
                logger.info("[HandWorker] waiting for body controller observation before hand DDS init")
                self._waiting_for_body_obs = True
            return False
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

    def _handle_hand_init_request(self, ev: EventSnapshot) -> bool:
        if not ev.level.get(HAND_INIT_EVENT_NAME, False):
            return False

        try:
            self.ctx.bus.clear_level(HAND_INIT_EVENT_NAME)
        except Exception:
            logger.debug("[HandWorker] failed to clear hand init event.", exc_info=True)

        if self.hand_ctrl is None:
            logger.warning("[HandWorker] hand init requested before controller initialization.")
            return True

        logger.info("[HandWorker] hand init requested.")
        try:
            ok = bool(self.hand_ctrl.initialize_hand())
        except Exception:
            logger.exception("[HandWorker] hand init request failed.")
            return True

        if ok:
            logger.info("[HandWorker] hand init completed.")
        else:
            logger.warning("[HandWorker] hand init command did not complete cleanly.")
        return True

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        st = self.state

        if not self._ensure_controller_ready(ev):
            return

        if self._handle_hand_init_request(ev):
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
