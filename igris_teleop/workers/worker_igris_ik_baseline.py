from __future__ import annotations

from dataclasses import dataclass, replace
import logging_mp
import time


from ..core.events import EventSnapshot
from ..core.state_machine import ModeState, TransitionResult
from ..core.worker_base import SingleRateWorker, WorkerContext

logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)

import numpy as np
from ..robot_control.kinematics.joints import LEG_INDICES, WAIST_INDICES, ARM_INDICES, NECK_INDICES
from ..robot_control.kinematics.ik.prox_ik_pelvis_env_unity_baseline import (
    IGRIS_C_UpperIK,
    IKConfig,
)


VR_MASTERARM_ARM_LOCK_WEIGHT = 1000.0
HMD_HEAD_ROTATION_WEIGHT = 6.0
IK_COMMAND_LEAD_LIMIT_WAIST = 0.20
IK_COMMAND_LEAD_LIMIT_ARM = 0.15
IK_COMMAND_LEAD_LIMIT_NECK = 0.35
WAIST_IK_TO_CONTROLLER_ORDER = np.asarray((2, 1, 0), dtype=np.intp)
WAIST_CONTROLLER_TO_IK_ORDER = WAIST_IK_TO_CONTROLLER_ORDER
IK_RATE_LIMIT_REFERENCE_HZ = 50.0


@dataclass(frozen=True)
class IKModeProfile:
    use_hand_targets: bool
    publish_arm: bool
    freeze_arms: bool


IK_MODE_PROFILES: dict[str, IKModeProfile] = {
    "unity": IKModeProfile(
        use_hand_targets=True,
        publish_arm=True,
        freeze_arms=False,
    ),
    "vr_masterarm": IKModeProfile(
        use_hand_targets=False,
        publish_arm=False,
        freeze_arms=True,
    ),
}


class IGRISIKWorker(SingleRateWorker):
    """Single-rate 워커 예제: 상태에 따라 카운터를 업데이트."""

    def __init__(self, ctx: WorkerContext, hz: float = 10.0) -> None:
        super().__init__(ctx, hz=hz)
        self._counter = 0
        self._shared_memory = ctx.shared_memory
        self._owns_shared_memory = False

        self.mode = self.ctx.run_config.mode
        self.teleop_device = self.ctx.run_config.teleop_device
        self._profile = self._resolve_profile(self.mode, self.teleop_device)
        source_cfg = IKConfig.from_sources(profile=self.teleop_device)
        dt_scale = IK_RATE_LIMIT_REFERENCE_HZ / float(self.hz)
        effective_cfg = replace(
            source_cfg,
            rate_limit=replace(
                source_cfg.rate_limit,
                dq_max=float(source_cfg.rate_limit.dq_max) * dt_scale,
                ddq_max=float(source_cfg.rate_limit.ddq_max) * dt_scale**2,
                dddq_max=float(source_cfg.rate_limit.dddq_max) * dt_scale**3,
            ),
        )
        self.ik_solver = IGRIS_C_UpperIK(cfg=effective_cfg)


        self.television_shm = self._shared_memory.get("television_shm")
        self.obs_shm = self._shared_memory.get("obs_shm")
        self.act_shm = self._shared_memory.get("act_shm")
        self.ee_shm = self._shared_memory.get("ee_shm")
        self.ik_target_shm = self._shared_memory.get("ik_target_shm")

        # 홈 위치 관련 변수
        self._home_set = False
        
        self._vr_left_base_mat = None
        self._vr_right_base_mat = None
        self._vr_head_base_mat = None

        self._robot_left_init = None
        self._robot_right_init = None
        self._robot_head_init = None
        self._auto_home_wait_logged = False
        
        self._last_sol_q = self._default_home_q()
        # HOME -> RUN 블렌딩(완만한 수렴) 파라미터
        self._run_blend_active = False
        self._run_blend_start_t: float | None = None
        self._run_blend_duration = 1.0  # seconds
        self._run_blend_q0: np.ndarray | None = None
        self._was_running = False
        # RUN 중 급격한 변화 감지 시 블렌딩 시작 기준 (rad)
        self._run_blend_trigger_max_delta: float | None = 0.6
        # 블렌딩 디버그 로그
        self._blend_log_interval: float = 1.0  # seconds
        self._blend_last_log_t: float | None = None
        self._ik_info_log_interval: float = 1.0
        self._ik_info_last_log_t: float | None = None
        
    def on_start(self) -> None:
        logger.info(
            "[%s] start device=%s use_hand_targets=%s publish_arm=%s freeze_arms=%s",
            self.ctx.name,
            self.teleop_device,
            self._profile.use_hand_targets,
            self._profile.publish_arm,
            self._profile.freeze_arms,
        )
        logger.info(f"[{self.ctx.name}] start (single-rate {self.hz} Hz)")
        logger.info(
            "[%s] HMD-only upper-body IK: head rotation weight=%.1f, "
            "waist/neck command-continuity enabled",
            self.ctx.name,
            HMD_HEAD_ROTATION_WEIGHT,
        )

    @staticmethod
    def _resolve_profile(mode: str | None, teleop_device: str | None) -> IKModeProfile:
        if mode != "teleop":
            raise ValueError(f"IGRISIKWorker requires teleop mode, got mode={mode!r}")
        if teleop_device not in IK_MODE_PROFILES:
            raise ValueError(
                f"Unsupported teleop_device for IGRISIKWorker: {teleop_device!r}. "
                f"Supported: {sorted(IK_MODE_PROFILES)}"
            )
        return IK_MODE_PROFILES[teleop_device]

    @staticmethod
    def _default_home_q() -> np.ndarray:
        return np.array([
            0.0, 0.0, 0.0,               # Waist
            -0.17098, 0.38123, 0.33176, -1.19044, 0.25468, -0.42747, -0.09709,  # Left arm
            -0.17098, -0.38123, -0.33176, -1.19044, -0.25468, 0.42747, -0.09709,    # Right arm
            0.0, 0.0                # Neck
        ], dtype=float)

    def _read_observation_q(self, obs: dict) -> np.ndarray | None:
        try:
            waist = np.asarray(obs["obs_waist"], dtype=np.float64).reshape(-1)
            arm = np.asarray(obs["obs_arm"], dtype=np.float64).reshape(-1)
            head = np.asarray(obs["obs_neck"], dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if waist.size != len(WAIST_INDICES) or arm.size != len(ARM_INDICES) or head.size != len(NECK_INDICES):
            return None
        waist = waist[WAIST_CONTROLLER_TO_IK_ORDER]

        q = np.concatenate((waist, arm, head))
        if not np.all(np.isfinite(q)):
            return None
        if np.linalg.norm(q) < 1e-6:
            return None
        return q


    def _sanitize_current_q(self, q: np.ndarray | None) -> np.ndarray:
        if self._last_sol_q is None:
            self._last_sol_q = self._default_home_q()

        if q is None:
            return self._last_sol_q.copy()

        q = np.asarray(q, dtype=np.float64).reshape(-1)

        if not np.all(np.isfinite(q)):
            return self._last_sol_q.copy()

        if np.linalg.norm(q) < 1e-6:   # all-zero(또는 near-zero)
            return self._last_sol_q.copy()

        return q

    def _resolve_ik_seed_q(self, current_q: np.ndarray) -> np.ndarray:
        current_q = self._sanitize_current_q(current_q)

        if self._last_sol_q is None:
            return current_q.copy()

        last_q = np.asarray(self._last_sol_q, dtype=np.float64).reshape(-1)
        if last_q.shape != current_q.shape:
            return current_q.copy()
        if not np.all(np.isfinite(last_q)):
            return current_q.copy()

        limits = np.concatenate(
            (
                np.full(len(WAIST_INDICES), IK_COMMAND_LEAD_LIMIT_WAIST),
                np.full(len(ARM_INDICES), IK_COMMAND_LEAD_LIMIT_ARM),
                np.full(len(NECK_INDICES), IK_COMMAND_LEAD_LIMIT_NECK),
            )
        )
        return current_q + np.clip(last_q - current_q, -limits, limits)

    def _maybe_log_ik_solve_info(self) -> None:
        getter = getattr(self.ik_solver, "get_last_solve_info", None)
        if not callable(getter):
            return
        now = time.monotonic()
        if self._ik_info_last_log_t is not None and (now - self._ik_info_last_log_t) < self._ik_info_log_interval:
            return
        self._ik_info_last_log_t = now
        try:
            info = getter()
        except Exception:
            logger.debug("[G1IK] Failed to read IK solve info.", exc_info=True)
            return
        if not info:
            return
        logger.info(
            "[G1IK] prox ik status=%s success=%s sqp=%s iter=%s wall=%.2fms "
            "dq=%.4f waist=%.4f arm=%.4f neck=%.4f slack=%.4g "
            "err_t=%.4f->%.4f err_r=%.4f->%.4f col=%s margin=%s",
            info.get("status"),
            info.get("success"),
            info.get("sqp_iters"),
            info.get("iter"),
            1000.0 * float(info.get("wall_time_s", 0.0) or 0.0),
            float(info.get("max_returned_delta", 0.0) or 0.0),
            float(info.get("max_waist_delta", 0.0) or 0.0),
            float(info.get("max_arm_delta", 0.0) or 0.0),
            float(info.get("max_neck_delta", 0.0) or 0.0),
            float(info.get("max_collision_slack", 0.0) or 0.0),
            float(info.get("max_trans_error_before", 0.0) or 0.0),
            float(info.get("max_trans_error_after", 0.0) or 0.0),
            float(info.get("max_rot_error_before", 0.0) or 0.0),
            float(info.get("max_rot_error_after", 0.0) or 0.0),
            info.get("active_collision_constraints"),
            info.get("min_collision_margin_after"),
        )

    # def _maybe_set_home(self, vr_data: dict, current_q: np.ndarray | None = None) -> bool:
    #     if self._home_set:
    #         return True

    #     if current_q is None:
    #         if self._last_sol_q is not None:
    #             current_q = self._last_sol_q
    #         else:
    #             current_q = self._default_home_q()

    #     current_q = np.asarray(current_q, dtype=np.float64).reshape(-1)

    #     try:
    #         l_pose, r_pose, h_pose = self.ik_solver.get_ee_poses(current_q)
    #     except Exception:
    #         logger.exception("[G1IK] Failed to compute initial EE pose.")
    #         return False

    #     self._vr_left_base_mat  = vr_data["left_wrist_mat"].copy()
    #     self._vr_right_base_mat = vr_data["right_wrist_mat"].copy()
    #     self._vr_head_base_mat  = vr_data["head_mat"].copy()

    #     # ✅ robot init EE: 기존처럼 저장
    #     self._robot_left_init = l_pose.homogeneous
    #     self._robot_right_init = r_pose.homogeneous
    #     self._robot_head_init = h_pose.homogeneous

    #     self._home_set = True
    
    #     logger.info("[G1IK] Home reference set and ready.")
    #     return True
    
    
    def _maybe_set_home(self, vr_data: dict, current_q: np.ndarray | None = None) -> bool:
        # ✅ 여기서 home_set은 "로봇 anchor 설정됨" 의미로 사용
        if self._home_set:
            return True

        if current_q is None:
            current_q = self._last_sol_q if self._last_sol_q is not None else self._default_home_q()

        current_q = np.asarray(current_q, dtype=np.float64).reshape(-1)

        try:
            l_pose, r_pose, h_pose = self.ik_solver.get_ee_poses(current_q)
        except Exception:
            logger.exception("[G1IK] Failed to compute initial EE pose.")
            return False

        # ✅ robot_init EE: HOME 순간 1회만 저장
        self._robot_left_init  = np.asarray(l_pose.homogeneous, dtype=np.float64)
        self._robot_right_init = np.asarray(r_pose.homogeneous, dtype=np.float64)
        self._robot_head_init  = np.asarray(h_pose.homogeneous, dtype=np.float64)

        # VR base가 아직 없다면 HOME 시점의 데이터(없으면 identity)로 채움
        if self._vr_left_base_mat is None:
            left_base = self._as_valid_mat4(vr_data.get("left_wrist_mat"))
            self._vr_left_base_mat = left_base if left_base is not None else np.eye(4, dtype=np.float64)
        if self._vr_right_base_mat is None:
            right_base = self._as_valid_mat4(vr_data.get("right_wrist_mat"))
            self._vr_right_base_mat = right_base if right_base is not None else np.eye(4, dtype=np.float64)
        if self._vr_head_base_mat is None:
            head_base = self._as_valid_mat4(vr_data.get("head_mat"))
            self._vr_head_base_mat = head_base if head_base is not None else np.eye(4, dtype=np.float64)

        self._home_set = True
        logger.info("[G1IK] Robot EE anchor set (HOME).")
        return True

    def _clear_home_reference(self) -> None:
        self._home_set = False
        self._vr_left_base_mat = None
        self._vr_right_base_mat = None
        self._vr_head_base_mat = None
        self._robot_left_init = None
        self._robot_right_init = None
        self._robot_head_init = None

    def _reset_ik_motion_state(self, q: np.ndarray) -> None:
        resetter = getattr(self.ik_solver, "reset_motion_state", None)
        if not callable(resetter):
            return
        try:
            resetter(np.asarray(q, dtype=np.float64).reshape(-1))
        except Exception:
            logger.debug("[G1IK] Failed to reset IK motion state.", exc_info=True)

    
    def _compose_current_q(self, obs: dict) -> np.ndarray | None:
        q_obs = self._read_observation_q(obs)
        if q_obs is None:
            if self._last_sol_q is not None:
                logger.debug("[G1IK] Observation missing, fallback to last_sol_q.")
                return self._last_sol_q.copy()
            return None
        waist_len = len(WAIST_INDICES)
        arm_len = len(ARM_INDICES)
        waist = q_obs[:waist_len].copy()
        arm = q_obs[waist_len : waist_len + arm_len].copy()
        head = q_obs[waist_len + arm_len :].copy()

        act_arm = self._get_valid_act_arm()
        if act_arm is not None:
            arm = act_arm

        return np.concatenate((waist, arm, head))

    def _get_valid_act_arm(self) -> np.ndarray | None:
        if self.act_shm is None:
            return None
        try:
            act = self.act_shm.read_data()
            arm = np.asarray(act.get("act_arm"), dtype=np.float64).reshape(-1)
        except Exception:
            return None

        if arm.shape[0] != len(ARM_INDICES):
            return None
        if not np.all(np.isfinite(arm)):
            return None
        if np.linalg.norm(arm) < 1e-6:
            return None
        return arm

    
    def _update_vr_base_mats(self, vr_data: dict) -> None:
        """HOME 동안 매 tick 호출해서 VR base(4x4)를 최신으로 유지."""
        try:
            left_base = self._as_valid_mat4(vr_data.get("left_wrist_mat"))
            right_base = self._as_valid_mat4(vr_data.get("right_wrist_mat"))
            head_base = self._as_valid_mat4(vr_data.get("head_mat"))

            if left_base is not None:
                self._vr_left_base_mat = left_base.copy()
            if right_base is not None:
                self._vr_right_base_mat = right_base.copy()
            if head_base is not None:
                self._vr_head_base_mat = head_base.copy()
        except Exception:
            logger.debug("[G1IK] Failed to update VR base mats.", exc_info=True)

    def _has_valid_home_seed_inputs(self, vr_data: dict) -> bool:
        left_base = self._as_valid_mat4(vr_data.get("left_wrist_mat"))
        right_base = self._as_valid_mat4(vr_data.get("right_wrist_mat"))
        head_base = self._as_valid_mat4(vr_data.get("head_mat"))
        return left_base is not None and right_base is not None and head_base is not None

    def _maybe_auto_set_home_before_run(self, vr_data: dict, current_q: np.ndarray) -> None:
        if self._home_set:
            return
        if not self._has_valid_home_seed_inputs(vr_data):
            if not self._auto_home_wait_logged:
                logger.info("[G1IK] waiting for valid VR wrist/head poses before auto set_home")
                self._auto_home_wait_logged = True
            return
        if self._maybe_set_home(vr_data, current_q):
            self._auto_home_wait_logged = False
            logger.info("[G1IK] Auto set_home completed before RUN/guard.")

    
    # @staticmethod
    # def _build_target(human_mat: np.ndarray, base_pos: np.ndarray, robot_init: np.ndarray) -> np.ndarray:
    #     target = robot_init.copy()
    #     target[:3, :3] = human_mat[:3, :3]
    #     target[:3, 3] = robot_init[:3, 3] + (human_mat[:3, 3] - base_pos)
    #     return target
    
    @staticmethod
    def _as_valid_mat4(mat: np.ndarray | None) -> np.ndarray | None:
        if mat is None:
            return None
        mat = np.asarray(mat, dtype=np.float64)
        if mat.shape != (4, 4):
            return None
        if not np.all(np.isfinite(mat)):
            return None
        if not np.any(mat):  # all-zero => invalid
            return None
        return mat

    @staticmethod
    def _build_target_unity(vr_now: np.ndarray, vr_base: np.ndarray, robot_init: np.ndarray) -> np.ndarray:
        """
        Unity용 타깃 구성:
        - rotation: VR에서 받은 값 그대로
        - translation: Unity bridge가 HOME 기준 상대값으로 쓴 translation을 그대로 더함
        """
        del vr_base
        vr_now_mat = IGRISIKWorker._as_valid_mat4(vr_now)
        if vr_now_mat is None:
            robot_init_mat = IGRISIKWorker._as_valid_mat4(robot_init)
            if robot_init_mat is not None:
                return robot_init_mat.copy()
            return np.eye(4, dtype=np.float64)

        robot_init_mat = IGRISIKWorker._as_valid_mat4(robot_init)
        if robot_init_mat is None:
            return np.eye(4, dtype=np.float64)

        target = robot_init_mat.copy()
        target[:3, :3] = vr_now_mat[:3, :3]
        target[:3, 3] = robot_init_mat[:3, 3] + vr_now_mat[:3, 3]
        return target
        
    def _publish_ee_poses(self, q: np.ndarray | None) -> None:
        if q is None or self.ee_shm is None:
            return
        try:
            l_pose, r_pose, h_pose = self.ik_solver.get_ee_poses(q)
        except Exception:
            logger.debug("[G1IK] Failed to compute EE poses.", exc_info=True)
            return
        try:
            self.ee_shm.write_data(
                left_wrist_mat=np.asarray(l_pose.homogeneous, dtype=np.float64),
                right_wrist_mat=np.asarray(r_pose.homogeneous, dtype=np.float64),
                head_mat=np.asarray(h_pose.homogeneous, dtype=np.float64),
            )
        except Exception:
            logger.debug("[G1IK] Failed to write EE poses to shm.", exc_info=True)

    def _publish_ik_targets(
        self,
        left_target: np.ndarray | None,
        right_target: np.ndarray | None,
        head_target: np.ndarray | None,
    ) -> None:
        if self.ik_target_shm is None:
            return
        left = self._as_valid_mat4(left_target)
        right = self._as_valid_mat4(right_target)
        head = self._as_valid_mat4(head_target)
        if left is None or right is None or head is None:
            try:
                self.ik_target_shm.write_data(target_valid=0.0, target_seq=time.monotonic())
            except Exception:
                logger.debug("[G1IK] Failed to clear IK target shm.", exc_info=True)
            return
        try:
            self.ik_target_shm.write_data(
                target_valid=1.0,
                target_seq=time.monotonic(),
                left_wrist_mat=left,
                right_wrist_mat=right,
                head_mat=head,
            )
        except Exception:
            logger.debug("[G1IK] Failed to write IK targets to shm.", exc_info=True)


    def _publish_targets(self, q: np.ndarray | None) -> None:
        if q is None:
            return

        waist_len = len(WAIST_INDICES)
        arm_len = len(ARM_INDICES)
        head_len = len(NECK_INDICES)

        expected_len = waist_len + arm_len + head_len
        if q.shape[0] < expected_len:
            logger.warning(
                "[G1IK] IK result length %d shorter than expected %d", q.shape[0], expected_len
            )
            return

        # IK/URDF waist order is (Pitch, Roll, Yaw); controller order is (Yaw, Roll, Pitch).
        waist_q = np.asarray(q[:waist_len], dtype=np.float64)
        if waist_len >= 3:
            waist_q = waist_q[WAIST_IK_TO_CONTROLLER_ORDER]

        neck_q = np.asarray(
            q[waist_len + arm_len : waist_len + arm_len + head_len],
            dtype=np.float64
        )

        kwargs = dict(
            act_leg=np.zeros(len(LEG_INDICES), dtype=np.float64),
            act_waist=waist_q,
            act_neck=neck_q,
        )

        if self._profile.publish_arm:
            kwargs["act_arm"] = np.asarray(
                q[waist_len : waist_len + arm_len],
                dtype=np.float64
            )


        self.act_shm.write_data(**kwargs)

    def _reset_run_blend(self) -> None:
        self._run_blend_active = False
        self._run_blend_start_t = None
        self._run_blend_q0 = None
        self._blend_last_log_t = None

    def _start_run_blend(self, q0: np.ndarray) -> None:
        self._run_blend_active = True
        self._run_blend_start_t = time.monotonic()
        self._run_blend_q0 = np.asarray(q0, dtype=np.float64).copy()

    def _apply_run_blend(self, sol_q: np.ndarray) -> np.ndarray:
        if not self._run_blend_active:
            return sol_q
        if self._run_blend_start_t is None or self._run_blend_q0 is None:
            self._run_blend_active = False
            return sol_q
        if self._run_blend_duration <= 0.0:
            self._run_blend_active = False
            return sol_q
        if self._run_blend_q0.shape != sol_q.shape:
            self._run_blend_active = False
            return sol_q
        elapsed = time.monotonic() - self._run_blend_start_t
        alpha = float(np.clip(elapsed / float(self._run_blend_duration), 0.0, 1.0))
        if alpha >= 1.0:
            self._run_blend_active = False
            return sol_q
        return (1.0 - alpha) * self._run_blend_q0 + alpha * sol_q

    def _maybe_trigger_sudden_blend(self, sol_q: np.ndarray) -> None:
        if self._run_blend_active:
            return
        if self._run_blend_trigger_max_delta is None:
            return
        if self._last_sol_q is None:
            return
        if self._last_sol_q.shape != sol_q.shape:
            return
        try:
            max_delta = float(np.max(np.abs(sol_q - self._last_sol_q)))
        except Exception:
            return
        if max_delta >= float(self._run_blend_trigger_max_delta):
            logger.info(
                "[G1IK] Sudden jump detected (max_delta=%.4f rad). Start blending.",
                max_delta,
            )
            self._start_run_blend(self._last_sol_q)

    def _maybe_log_blend(self, current_q: np.ndarray, sol_q: np.ndarray, cmd_q: np.ndarray) -> None:
        if not self._run_blend_active:
            return
        if self._blend_log_interval <= 0.0:
            return
        now = time.monotonic()
        last = self._blend_last_log_t
        if last is not None and (now - last) < float(self._blend_log_interval):
            return
        self._blend_last_log_t = now

        def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
            try:
                return float(np.max(np.abs(a - b)))
            except Exception:
                return float("nan")

        elapsed = (
            now - self._run_blend_start_t
            if self._run_blend_start_t is not None
            else float("nan")
        )
        logger.info(
            "[G1IK] Blend active (t=%.2fs/%.2fs) "
            "max|sol-current|=%.4f rad, max|cmd-current|=%.4f rad, max|sol-cmd|=%.4f rad",
            float(elapsed),
            float(self._run_blend_duration),
            _max_abs(sol_q, current_q),
            _max_abs(cmd_q, current_q),
            _max_abs(sol_q, cmd_q),
        )


    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        st = self.state

        vr_data = self.television_shm.read_data()
        obs_data = self.obs_shm.read_data()
        obs_q = self._read_observation_q(obs_data)

        raw_q = self._compose_current_q(obs_data)
        current_q = self._sanitize_current_q(raw_q)  # ✅ 이후 current_q는 항상 유효 ndarray
        if obs_q is not None:
            self._publish_ee_poses(current_q)

        if st == ModeState.HOME:
            self._was_running = False
            # HOME에서는 RUN 블렌드를 리셋
            self._reset_run_blend()

            # HOME 동안 VR base를 갱신(유효 데이터만)
            self._update_vr_base_mats(vr_data)

            if tr.reason == "home_set":
                self._clear_home_reference()

            if not self._home_set and not self._maybe_set_home(vr_data, current_q):
                logger.debug("[G1IK] Waiting for home reference.")
                return

            # ✅ HOME: 항상 안전 seed 유지(로봇 미연결이어도 last/default로 유지됨)
            self._publish_ik_targets(self._robot_left_init, self._robot_right_init, self._robot_head_init)
            self._publish_targets(current_q)
            self._reset_ik_motion_state(current_q)
            self._last_sol_q = current_q.copy()
            return

        if st != ModeState.RUN:
            self._was_running = False
            if obs_q is None:
                if not self._auto_home_wait_logged:
                    logger.info("[G1IK] waiting for body observation before auto set_home")
                    self._auto_home_wait_logged = True
            else:
                self._maybe_auto_set_home_before_run(vr_data, current_q)
            # RUN 이외 상태에서는 블렌드 유지하지 않음
            self._reset_run_blend()
            return

        home_just_set = False
        if not self._home_set:
            if not self._maybe_set_home(vr_data, current_q):
                logger.debug("[G1IK] Waiting for home reference.")
                return
            home_just_set = True

        # RUN 진입 직후에는 masterarm 경로와 동일하게 1회 hold 후 진행
        if home_just_set:
            self._publish_ik_targets(self._robot_left_init, self._robot_right_init, self._robot_head_init)
            self._publish_targets(current_q)
            self._reset_ik_motion_state(current_q)
            self._last_sol_q = current_q.copy()
            return

        run_just_started = not self._was_running
        self._was_running = True

        # left_target = self._build_target(
        #     vr_data["left_wrist_mat"], self._base_left, self._robot_left_init
        # )
        # right_target = self._build_target(
        #     vr_data["right_wrist_mat"], self._base_right, self._robot_right_init
        # )
        # head_target = self._build_target(
        #     vr_data["head_mat"], self._base_head, self._robot_head_init
        # )

        left_target = self._build_target_unity(
            vr_data["left_wrist_mat"], self._vr_left_base_mat, self._robot_left_init
        )
        right_target = self._build_target_unity(
            vr_data["right_wrist_mat"], self._vr_right_base_mat, self._robot_right_init
        )
        head_target = self._build_target_unity(
            vr_data["head_mat"], self._vr_head_base_mat, self._robot_head_init
        )
        self._publish_ik_targets(left_target, right_target, head_target)
        # Continue from the last command so a slowly changing HMD target is not
        # restarted from quantized/lagging encoders on every IK tick.  Keep the
        # command inside a bounded envelope around observation for safety.
        ik_seed_q = self._resolve_ik_seed_q(current_q)

        try:
            sol_q, _ = self.ik_solver.solve_ik(
                left_target, right_target, head_target,
                current_lr_arm_motor_q=ik_seed_q,
                head_rotation_weight=HMD_HEAD_ROTATION_WEIGHT,
                use_hand_targets=self._profile.use_hand_targets,
                arm_lock_q=current_q if self._profile.freeze_arms else None,
                arm_lock_weight=VR_MASTERARM_ARM_LOCK_WEIGHT if self._profile.freeze_arms else 0.0,
            )
            self._maybe_log_ik_solve_info()
        except Exception:
            logger.exception("[G1IK] IK solve failed.")
            emergency = self.shared_event.get("emergency")
            if hasattr(emergency, "set"):
                try:
                    emergency.set()
                except Exception:
                    pass
            return

        sol_q = np.asarray(sol_q, dtype=np.float64).reshape(-1)
        if sol_q.shape != ik_seed_q.shape or not np.all(np.isfinite(sol_q)):
            logger.warning("[G1IK] Invalid IK solution. Keeping previous command.")
            sol_q = ik_seed_q.copy()
        # Always blend the first valid Unity IK solution from the measured
        # robot state.  Previously blending only started for a >0.6 rad jump,
        # so ordinary Start takeovers bypassed interpolation entirely.
        if run_just_started:
            self._start_run_blend(current_q)
            logger.info(
                "[G1IK] RUN entry blend started from measured state (duration=%.2fs)",
                float(self._run_blend_duration),
            )
        # RUN 중 급격한 변화를 감지하면 그 시점부터 블렌딩 시작
        self._maybe_trigger_sudden_blend(sol_q)
        cmd_q = self._apply_run_blend(sol_q)
        self._maybe_log_blend(current_q, sol_q, cmd_q)
        previous_q = np.asarray(self._last_sol_q, dtype=np.float64).copy()
        self._last_sol_q = np.asarray(cmd_q, dtype=np.float64)
        self._publish_targets(self._last_sol_q)
        commit = getattr(self.ik_solver, "commit_published_command", None)
        if callable(commit):
            try:
                commit(self._last_sol_q, previous_q)
            except Exception:
                logger.exception(
                    "[G1IK] Failed to commit published command; resetting IK motion state."
                )
                self._reset_ik_motion_state(self._last_sol_q)



    def on_stop(self) -> None:
        if self._shared_memory:
            for key, mgr in self._shared_memory.items():
                try:
                    mgr.worker_close()
                except Exception:
                    logger.exception(f"[{self.ctx.name}] failed to close shared memory {key}")

        logger.info(f"[{self.ctx.name}] stop")
