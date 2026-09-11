from __future__ import annotations

from dataclasses import dataclass, replace
import logging_mp
import time


from ..core.events import EventSnapshot
from ..core.math.transforms import apply_common_frame_relative_pose
from ..core.state_machine import ModeState, TransitionResult
from ..core.worker_base import SingleRateWorker, WorkerContext

logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)

import numpy as np
from ..robot_control.kinematics.joints import LEG_INDICES, WAIST_INDICES, ARM_INDICES, NECK_INDICES
from ..robot_control.kinematics.ik.prox_ik_pelvis_env import IGRIS_C_UpperIK, IKConfig
from ..robot_control.kinematics.ik.waist_task_transition import (
    WaistTaskTransition,
)


VR_MASTERARM_ARM_LOCK_WEIGHT = 1000.0
HMD_HEAD_TRANSLATION_WEIGHT = 1.0
HMD_HEAD_ROTATION_WEIGHT = 7.0
CONTROLLER_CHEST_TRANSLATION_WEIGHT = 2.0
CONTROLLER_CHEST_ROTATION_WEIGHT = 2.0
IK_COMMAND_LEAD_LIMIT_WAIST = 0.20
IK_COMMAND_LEAD_LIMIT_ARM = 0.15
IK_COMMAND_LEAD_LIMIT_NECK = 0.35
WAIST_IK_TO_CONTROLLER_ORDER = np.asarray((2, 1, 0), dtype=np.intp)
WAIST_CONTROLLER_TO_IK_ORDER = WAIST_IK_TO_CONTROLLER_ORDER
IK_RATE_LIMIT_REFERENCE_HZ = 50.0


def scale_ik_config_rate_limits(
    cfg: IKConfig,
    *,
    worker_hz: float,
    reference_hz: float = IK_RATE_LIMIT_REFERENCE_HZ,
) -> IKConfig:
    """Preserve physical motion limits when the IK tick rate changes.

    The QP state names use ``dq``/``ddq``/``dddq``, but the values are
    discrete position differences per solver tick.  A first, second, or
    third position difference therefore scales with ``dt``, ``dt**2``, or
    ``dt**3`` respectively.
    """
    worker_hz = float(worker_hz)
    reference_hz = float(reference_hz)
    if not np.isfinite(worker_hz) or worker_hz <= 0.0:
        raise ValueError(f"worker_hz must be finite and positive, got {worker_hz!r}")
    if not np.isfinite(reference_hz) or reference_hz <= 0.0:
        raise ValueError(
            f"reference_hz must be finite and positive, got {reference_hz!r}"
        )

    dt_scale = reference_hz / worker_hz
    source = cfg.rate_limit
    scaled_rate_limit = replace(
        source,
        dq_max=float(source.dq_max) * dt_scale,
        ddq_max=float(source.ddq_max) * dt_scale**2,
        dddq_max=float(source.dddq_max) * dt_scale**3,
    )
    waist_transition = cfg.waist_transition
    if waist_transition.base_max_delta_per_step is not None:
        waist_transition = replace(
            waist_transition,
            base_max_delta_per_step=(
                float(waist_transition.base_max_delta_per_step) * dt_scale
            ),
        )
    return replace(
        cfg,
        rate_limit=scaled_rate_limit,
        waist_transition=waist_transition,
    )


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
    "unity_hybrid": IKModeProfile(
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

    def __init__(self, ctx: WorkerContext, hz: float = 100.0) -> None:
        super().__init__(ctx, hz=hz)
        self._counter = 0
        self._shared_memory = ctx.shared_memory
        self._owns_shared_memory = False

        self.mode = self.ctx.run_config.mode
        self.teleop_device = self.ctx.run_config.teleop_device
        self._profile = self._resolve_profile(self.mode, self.teleop_device)
        source_ik_cfg = IKConfig.from_sources(profile=self.teleop_device)
        effective_ik_cfg = scale_ik_config_rate_limits(
            source_ik_cfg,
            worker_hz=self.hz,
        )
        self.ik_solver = IGRIS_C_UpperIK(cfg=effective_ik_cfg)
        self._waist_transition = WaistTaskTransition(
            self.ik_solver.cfg.waist_transition,
            waist_dof=len(WAIST_INDICES),
        )


        self.television_shm = self._shared_memory.get("television_shm")
        self.obs_shm = self._shared_memory.get("obs_shm")
        self.act_shm = self._shared_memory.get("act_shm")
        self.ee_shm = self._shared_memory.get("ee_shm")
        self.ik_target_shm = self._shared_memory.get("ik_target_shm")

        # 홈 위치 관련 변수
        self._home_set = False

        self._robot_left_init = None
        self._robot_right_init = None
        self._robot_head_init = None
        self._robot_chest_init = None
        self._fixed_leg_q: np.ndarray | None = None
        self._auto_home_wait_logged = False

        self._last_sol_q = self._default_home_q()
        self._previous_published_q: np.ndarray | None = None
        # HOME -> RUN 블렌딩(완만한 수렴) 파라미터
        self._run_blend_active = False
        self._run_blend_start_t: float | None = None
        self._run_blend_duration = 1.0  # seconds
        self._run_blend_q0: np.ndarray | None = None
        # RUN 중 급격한 변화 감지 시 블렌딩 시작 기준 (rad)
        self._run_blend_trigger_max_delta: float | None = 0.6
        # 블렌딩 디버그 로그
        self._blend_log_interval: float = 1.0  # seconds
        self._blend_last_log_t: float | None = None
        self._ik_info_log_interval: float = 1.0
        self._ik_info_last_log_t: float | None = None
        self._chest_log_last_t: float | None = None

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
        rate_limit = self.ik_solver.cfg.rate_limit
        logger.info(
            "[%s] IK rate limits scaled from %.1f Hz reference: "
            "dq=%.6g ddq=%.6g dddq=%.6g per tick",
            self.ctx.name,
            IK_RATE_LIMIT_REFERENCE_HZ,
            rate_limit.dq_max,
            rate_limit.ddq_max,
            rate_limit.dddq_max,
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

    @staticmethod
    def _read_observation_leg(obs: dict) -> np.ndarray | None:
        try:
            obs_seq = float(np.asarray(obs.get("obs_seq", 0.0)).reshape(()).item())
            leg = np.asarray(obs["obs_leg"], dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if obs_seq <= 0.0 or leg.size != len(LEG_INDICES) or not np.all(np.isfinite(leg)):
            return None
        return leg.copy()

    def _latch_fixed_leg_target(self, obs: dict, *, force: bool = False) -> None:
        if self._fixed_leg_q is not None and not force:
            return
        leg = self._read_observation_leg(obs)
        if leg is not None:
            self._fixed_leg_q = leg
            logger.info("[G1IK] latched fixed teleop leg target from observation")


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

        return self._limit_command_lead(last_q, current_q)

    @staticmethod
    def _command_lead_limits(q_size: int) -> np.ndarray:
        expected_size = len(WAIST_INDICES) + len(ARM_INDICES) + len(NECK_INDICES)
        if q_size != expected_size:
            return np.zeros(q_size, dtype=np.float64)
        return np.concatenate(
            (
                np.full(len(WAIST_INDICES), IK_COMMAND_LEAD_LIMIT_WAIST),
                np.full(len(ARM_INDICES), IK_COMMAND_LEAD_LIMIT_ARM),
                np.full(len(NECK_INDICES), IK_COMMAND_LEAD_LIMIT_NECK),
            )
        )

    @classmethod
    def _limit_command_lead(
        cls,
        command_q: np.ndarray,
        current_q: np.ndarray,
        *,
        preserve_waist: bool = False,
    ) -> np.ndarray:
        command = np.asarray(command_q, dtype=np.float64).reshape(-1)
        current = np.asarray(current_q, dtype=np.float64).reshape(-1)
        if command.shape != current.shape:
            return current.copy()
        limits = cls._command_lead_limits(command.size)
        limited = current + np.clip(command - current, -limits, limits)
        if preserve_waist and limited.size >= len(WAIST_INDICES):
            limited[: len(WAIST_INDICES)] = command[: len(WAIST_INDICES)]
        return limited

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
            "err_t=%.4f->%.4f err_r=%.4f->%.4f col=%s margin=%s "
            "waist_bound=%s acc_override=%s",
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
            info.get("waist_bound_active"),
            info.get("waist_acceleration_override"),
        )
        if "chest_trans_error_before" in info:
            logger.info(
                "[G1IK] target residual "
                "head=(%.4fm/%.1fdeg -> %.4fm/%.1fdeg) "
                "chest=(%.4fm/%.1fdeg -> %.4fm/%.1fdeg)",
                float(info.get("head_trans_error_before", 0.0) or 0.0),
                np.rad2deg(float(info.get("head_rot_error_before", 0.0) or 0.0)),
                float(info.get("head_trans_error_after", 0.0) or 0.0),
                np.rad2deg(float(info.get("head_rot_error_after", 0.0) or 0.0)),
                float(info.get("chest_trans_error_before", 0.0) or 0.0),
                np.rad2deg(float(info.get("chest_rot_error_before", 0.0) or 0.0)),
                float(info.get("chest_trans_error_after", 0.0) or 0.0),
                np.rad2deg(float(info.get("chest_rot_error_after", 0.0) or 0.0)),
            )

    def _maybe_set_home(
        self,
        current_q: np.ndarray | None = None,
        *,
        force: bool = False,
    ) -> bool:
        # VR HOME is owned by UnityRosridgeWorker; this worker only owns robot anchors.
        if self._home_set and not force:
            return True

        if current_q is None:
            current_q = self._last_sol_q if self._last_sol_q is not None else self._default_home_q()

        current_q = np.asarray(current_q, dtype=np.float64).reshape(-1)

        try:
            l_pose, r_pose, h_pose = self.ik_solver.get_ee_poses(current_q)
            chest_pose = self.ik_solver.get_chest_pose(current_q)
        except Exception:
            logger.exception("[G1IK] Failed to compute initial EE pose.")
            return False

        self._robot_left_init  = np.asarray(l_pose.homogeneous, dtype=np.float64)
        self._robot_right_init = np.asarray(r_pose.homogeneous, dtype=np.float64)
        self._robot_head_init  = np.asarray(h_pose.homogeneous, dtype=np.float64)
        self._robot_chest_init = np.asarray(chest_pose.homogeneous, dtype=np.float64)

        first_capture = not self._home_set
        self._home_set = True
        if first_capture:
            logger.info("[G1IK] Robot EE anchor set (HOME).")
        return True

    def _clear_home_reference(self) -> None:
        self._home_set = False
        self._robot_left_init = None
        self._robot_right_init = None
        self._robot_head_init = None
        self._robot_chest_init = None

    def _reset_ik_motion_state(self, q: np.ndarray) -> None:
        resetter = getattr(self.ik_solver, "reset_motion_state", None)
        if not callable(resetter):
            return
        try:
            resetter(np.asarray(q, dtype=np.float64).reshape(-1))
        except Exception:
            logger.debug("[G1IK] Failed to reset IK motion state.", exc_info=True)

    def _record_published_command(
        self,
        q: np.ndarray,
        *,
        reset_motion_state: bool = False,
    ) -> None:
        published_q = np.asarray(q, dtype=np.float64).reshape(-1).copy()
        if not np.all(np.isfinite(published_q)):
            raise ValueError("published IK command must be finite")

        previous_q = self._previous_published_q
        if (
            reset_motion_state
            or previous_q is None
            or np.asarray(previous_q).shape != published_q.shape
        ):
            self._reset_ik_motion_state(published_q)
            self._waist_transition.reset(
                published_q[: len(WAIST_INDICES)],
                activation=0.0,
                timestamp=time.monotonic(),
            )
        else:
            previous_q = np.asarray(previous_q, dtype=np.float64).reshape(-1)
            committer = getattr(
                self.ik_solver,
                "commit_published_command",
                None,
            )
            try:
                if callable(committer):
                    committer(published_q, previous_q)
                self._waist_transition.commit_published_command(
                    published_q[: len(WAIST_INDICES)]
                )
            except Exception:
                logger.exception(
                    "[G1IK] Failed to commit published command state; "
                    "resetting solver state to the published command."
                )
                self._reset_ik_motion_state(published_q)
                self._waist_transition.reset(
                    published_q[: len(WAIST_INDICES)],
                    activation=self._waist_transition.previous_activation,
                    timestamp=time.monotonic(),
                )

        self._previous_published_q = published_q
        self._last_sol_q = published_q.copy()

    def _ensure_published_command_reference(
        self,
        fallback_q: np.ndarray,
    ) -> np.ndarray:
        fallback = np.asarray(fallback_q, dtype=np.float64).reshape(-1)
        previous = self._previous_published_q
        if previous is not None:
            previous_arr = np.asarray(previous, dtype=np.float64).reshape(-1)
            if (
                previous_arr.shape == fallback.shape
                and np.all(np.isfinite(previous_arr))
            ):
                transition_waist = (
                    self._waist_transition.previous_published_waist
                )
                if transition_waist is None:
                    self._waist_transition.reset(
                        previous_arr[: len(WAIST_INDICES)],
                        activation=0.0,
                        timestamp=time.monotonic(),
                    )
                elif not np.allclose(
                    transition_waist,
                    previous_arr[: len(WAIST_INDICES)],
                    atol=1e-10,
                    rtol=0.0,
                ):
                    self._waist_transition.commit_published_command(
                        previous_arr[: len(WAIST_INDICES)]
                    )
                return previous_arr.copy()

        logger.warning(
            "[G1IK] Published-command state was missing; "
            "initializing it from the current hold command."
        )
        self._previous_published_q = fallback.copy()
        self._last_sol_q = fallback.copy()
        self._reset_ik_motion_state(fallback)
        self._waist_transition.reset(
            fallback[: len(WAIST_INDICES)],
            activation=0.0,
            timestamp=time.monotonic(),
        )
        return fallback.copy()


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


    def _refresh_robot_home_before_run(self, current_q: np.ndarray) -> None:
        first_capture = not self._home_set
        if self._maybe_set_home(current_q, force=True):
            self._auto_home_wait_logged = False
            if first_capture:
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
    def _build_target_unity(vr_relative: np.ndarray, robot_init: np.ndarray) -> np.ndarray:
        """
        Apply the Unity bridge's HOME-relative pose to the robot HOME anchor.
        """
        relative_mat = IGRISIKWorker._as_valid_mat4(vr_relative)
        if relative_mat is None:
            robot_init_mat = IGRISIKWorker._as_valid_mat4(robot_init)
            if robot_init_mat is not None:
                return robot_init_mat.copy()
            return np.eye(4, dtype=np.float64)

        robot_init_mat = IGRISIKWorker._as_valid_mat4(robot_init)
        if robot_init_mat is None:
            return np.eye(4, dtype=np.float64)

        return apply_common_frame_relative_pose(relative_mat, robot_init_mat)

    @staticmethod
    def _build_chest_target_unity(
        chest_relative: np.ndarray,
        robot_init: np.ndarray,
    ) -> np.ndarray:
        relative_mat = IGRISIKWorker._as_valid_mat4(chest_relative)
        robot_init_mat = IGRISIKWorker._as_valid_mat4(robot_init)
        if relative_mat is None or robot_init_mat is None:
            return np.eye(4, dtype=np.float64)

        return apply_common_frame_relative_pose(relative_mat, robot_init_mat)

    def _solve_active_targets(
        self,
        left_target: np.ndarray,
        right_target: np.ndarray,
        head_target: np.ndarray,
        *,
        chest_target: np.ndarray | None,
        waist_activation: float,
        previous_published_q: np.ndarray,
        waist_delta_limit: float,
        waist_hold_weight: float,
        ik_seed_q: np.ndarray,
        current_q: np.ndarray,
    ):
        (
            head_translation_weight,
            head_rotation_weight,
            chest_translation_weight,
            chest_rotation_weight,
        ) = self._upper_body_task_weights(
            waist_activation,
            transition=self._waist_transition,
        )
        if chest_target is None:
            chest_translation_weight = 0.0
            chest_rotation_weight = 0.0

        return self.ik_solver.solve_ik(
            left_target,
            right_target,
            head_target,
            head_translation_weight=head_translation_weight,
            head_rotation_weight=head_rotation_weight,
            chest_pose=chest_target,
            use_chest_target=chest_target is not None,
            chest_translation_weight=chest_translation_weight,
            chest_rotation_weight=chest_rotation_weight,
            current_lr_arm_motor_q=ik_seed_q,
            waist_lock_q=previous_published_q,
            waist_lock_weight=waist_hold_weight,
            waist_motion_reference_q=previous_published_q,
            waist_delta_limit=waist_delta_limit,
            commit_state=False,
            use_hand_targets=self._profile.use_hand_targets,
            arm_lock_q=current_q if self._profile.freeze_arms else None,
            arm_lock_weight=(
                VR_MASTERARM_ARM_LOCK_WEIGHT if self._profile.freeze_arms else 0.0
            ),
        )

    @staticmethod
    def _upper_body_task_weights(
        waist_activation: float,
        *,
        transition: WaistTaskTransition | None = None,
    ) -> tuple[float, float, float, float]:
        activation = float(np.clip(waist_activation, 0.0, 1.0))
        head_translation_weight = HMD_HEAD_TRANSLATION_WEIGHT
        if transition is not None:
            head_translation_weight = transition.head_translation_weight(
                activation,
                base_weight=HMD_HEAD_TRANSLATION_WEIGHT,
            )

        return (
            head_translation_weight,
            HMD_HEAD_ROTATION_WEIGHT,
            CONTROLLER_CHEST_TRANSLATION_WEIGHT * activation,
            CONTROLLER_CHEST_ROTATION_WEIGHT * activation,
        )

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
        torso_target: np.ndarray | None = None,
        torso_alpha: float = 0.0,
        chest_target: np.ndarray | None = None,
        chest_alpha: float = 0.0,
        chest_target_valid: bool | None = None,
    ) -> None:
        if self.ik_target_shm is None:
            return
        left = self._as_valid_mat4(left_target)
        right = self._as_valid_mat4(right_target)
        head = self._as_valid_mat4(head_target)
        torso = self._as_valid_mat4(torso_target)
        chest = self._as_valid_mat4(chest_target)
        torso_valid = torso is not None and float(torso_alpha) > 0.0
        chest_valid = chest is not None and (
            bool(chest_target_valid)
            if chest_target_valid is not None
            else float(chest_alpha) > 0.0
        )
        if left is None or right is None or head is None:
            try:
                self.ik_target_shm.write_data(
                    target_valid=0.0,
                    target_seq=time.monotonic(),
                    torso_target_valid=0.0,
                    torso_alpha=0.0,
                    chest_target_valid=0.0,
                    chest_alpha=0.0,
                )
            except Exception:
                logger.debug("[G1IK] Failed to clear IK target shm.", exc_info=True)
            return
        try:
            self.ik_target_shm.write_data(
                target_valid=1.0,
                target_seq=time.monotonic(),
                torso_target_valid=1.0 if torso_valid else 0.0,
                torso_alpha=float(np.clip(torso_alpha, 0.0, 1.0)),
                chest_target_valid=1.0 if chest_valid else 0.0,
                chest_alpha=float(np.clip(chest_alpha, 0.0, 1.0)),
                left_wrist_mat=left,
                right_wrist_mat=right,
                head_mat=head,
                torso_mat=torso if torso is not None else np.eye(4, dtype=np.float64),
                chest_mat=chest if chest is not None else np.eye(4, dtype=np.float64),
            )
        except Exception:
            logger.debug("[G1IK] Failed to write IK targets to shm.", exc_info=True)


    def _publish_targets(self, q: np.ndarray | None) -> bool:
        if q is None or self.act_shm is None:
            return False

        waist_len = len(WAIST_INDICES)
        arm_len = len(ARM_INDICES)
        head_len = len(NECK_INDICES)

        expected_len = waist_len + arm_len + head_len
        if q.shape[0] < expected_len:
            logger.warning(
                "[G1IK] IK result length %d shorter than expected %d", q.shape[0], expected_len
            )
            return False

        # IK/URDF waist order is (Pitch, Roll, Yaw); controller order is (Yaw, Roll, Pitch).
        waist_q = np.asarray(q[:waist_len], dtype=np.float64)
        if waist_len >= 3:
            waist_q = waist_q[WAIST_IK_TO_CONTROLLER_ORDER]

        neck_q = np.asarray(
            q[waist_len + arm_len : waist_len + arm_len + head_len],
            dtype=np.float64
        )

        kwargs = dict(
            act_waist=waist_q,
            act_neck=neck_q,
        )

        if self._fixed_leg_q is not None:
            kwargs["act_leg"] = self._fixed_leg_q.copy()

        if self._profile.publish_arm:
            kwargs["act_arm"] = np.asarray(
                q[waist_len : waist_len + arm_len],
                dtype=np.float64
            )


        self.act_shm.write_data(**kwargs)
        return True

    def _reset_run_blend(self) -> None:
        self._run_blend_active = False
        self._run_blend_start_t = None
        self._run_blend_q0 = None
        self._blend_last_log_t = None

    def _start_run_blend(self, q0: np.ndarray) -> None:
        self._run_blend_active = True
        self._run_blend_start_t = time.monotonic()
        self._run_blend_q0 = np.asarray(q0, dtype=np.float64).copy()

    def _apply_run_blend(
        self,
        sol_q: np.ndarray,
        *,
        preserve_waist: bool = False,
    ) -> np.ndarray:
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
        blended = (1.0 - alpha) * self._run_blend_q0 + alpha * sol_q
        if preserve_waist and blended.size >= len(WAIST_INDICES):
            blended[: len(WAIST_INDICES)] = sol_q[: len(WAIST_INDICES)]
        return blended

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

        if self.state == ModeState.HOME:
            self._latch_fixed_leg_target(
                obs_data,
                force=tr.reason == "home_set",
            )
        else:
            self._latch_fixed_leg_target(obs_data)

        raw_q = self._compose_current_q(obs_data)
        current_q = self._sanitize_current_q(raw_q)  # ✅ 이후 current_q는 항상 유효 ndarray
        if obs_q is not None:
            self._publish_ee_poses(obs_q)

        if st == ModeState.HOME:
            # HOME에서는 RUN 블렌드를 리셋
            self._reset_run_blend()

            if tr.reason == "home_set":
                self._clear_home_reference()

            robot_anchor_q = obs_q if obs_q is not None else current_q
            if not self._maybe_set_home(
                robot_anchor_q,
                force=obs_q is not None,
            ):
                logger.debug("[G1IK] Waiting for home reference.")
                return

            # ✅ HOME: 항상 안전 seed 유지(로봇 미연결이어도 last/default로 유지됨)
            self._publish_ik_targets(self._robot_left_init, self._robot_right_init, self._robot_head_init)
            if self._publish_targets(current_q):
                self._record_published_command(
                    current_q,
                    reset_motion_state=True,
                )
            return

        if st != ModeState.RUN:
            if obs_q is None:
                if not self._auto_home_wait_logged:
                    logger.info("[G1IK] waiting for body observation before auto set_home")
                    self._auto_home_wait_logged = True
            elif st == ModeState.WAIT_START:
                self._refresh_robot_home_before_run(obs_q)
            elif not self._home_set:
                self._refresh_robot_home_before_run(obs_q)
            # RUN 이외 상태에서는 블렌드 유지하지 않음
            self._reset_run_blend()
            return

        home_just_set = False
        if not self._home_set:
            robot_anchor_q = obs_q if obs_q is not None else current_q
            if not self._maybe_set_home(robot_anchor_q):
                logger.debug("[G1IK] Waiting for home reference.")
                return
            home_just_set = True

        # RUN 진입 직후에는 masterarm 경로와 동일하게 1회 hold 후 진행
        if home_just_set:
            self._publish_ik_targets(self._robot_left_init, self._robot_right_init, self._robot_head_init)
            if self._publish_targets(current_q):
                self._record_published_command(
                    current_q,
                    reset_motion_state=True,
                )
            return

        if tr.reason in {"start_set", "home_cleared_start_set"}:
            run_hold_q = self._ensure_published_command_reference(current_q)
            self._waist_transition.reset(
                run_hold_q[: len(WAIST_INDICES)],
                activation=0.0,
                timestamp=time.monotonic(),
            )
            self._reset_ik_motion_state(run_hold_q)

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
            vr_data["left_wrist_mat"], self._robot_left_init
        )
        right_target = self._build_target_unity(
            vr_data["right_wrist_mat"], self._robot_right_init
        )
        head_target = self._build_target_unity(
            vr_data["head_mat"], self._robot_head_init
        )
        control_now = time.monotonic()
        try:
            chest_confidence = float(
                np.asarray(
                    vr_data.get("chest_alpha", 0.0)
                ).reshape(()).item()
            )
            chest_source_valid = bool(
                float(
                    np.asarray(
                        vr_data.get("chest_source_valid", 0.0)
                    ).reshape(()).item()
                )
                > 0.5
            )
        except Exception:
            chest_confidence = float("nan")
            chest_source_valid = False
        chest_input = self._as_valid_mat4(vr_data.get("chest_mat"))
        chest_target_valid = bool(
            chest_source_valid
            and chest_input is not None
            and self._robot_chest_init is not None
        )
        if chest_target_valid:
            chest_target = self._build_chest_target_unity(
                chest_input,
                self._robot_chest_init,
            )
        else:
            chest_target = None

        previous_published_q = self._ensure_published_command_reference(
            current_q
        )
        transition_result = self._waist_transition.update_activation(
            chest_confidence,
            target_valid=chest_target_valid,
            now=control_now,
        )
        waist_activation = transition_result.filtered_activation
        waist_delta_limit = self._waist_transition.activated_delta_limit(
            waist_activation,
            fallback_limit=self.ik_solver.cfg.rate_limit.dq_max,
        )
        waist_hold_weight = self._waist_transition.waist_hold_weight(
            waist_activation
        )

        self._publish_ik_targets(
            left_target,
            right_target,
            head_target,
            chest_target=chest_target,
            chest_alpha=transition_result.confidence,
            chest_target_valid=chest_target_valid,
        )
        # Continue the command trajectory instead of restarting every QP from
        # the measured pose. The tracking envelope in _resolve_ik_seed_q keeps
        # the command bounded when an actuator lags.
        ik_seed_q = self._resolve_ik_seed_q(current_q)
        # Waist transition state is command-based, not encoder-based. Keep the
        # QP waist seed aligned with the last successful publish.
        ik_seed_q[: len(WAIST_INDICES)] = previous_published_q[
            : len(WAIST_INDICES)
        ]

        try:
            sol_q, _ = self._solve_active_targets(
                left_target,
                right_target,
                head_target,
                chest_target=chest_target,
                waist_activation=waist_activation,
                previous_published_q=previous_published_q,
                waist_delta_limit=waist_delta_limit,
                waist_hold_weight=waist_hold_weight,
                ik_seed_q=ik_seed_q,
                current_q=current_q,
            )
            self._maybe_log_ik_solve_info()
            solve_info_getter = getattr(
                self.ik_solver,
                "get_last_solve_info",
                None,
            )
            solve_info = (
                solve_info_getter()
                if callable(solve_info_getter)
                else {}
            )
            if solve_info and solve_info.get("success") is False:
                logger.error(
                    "[G1IK] IK solve failed closed: %s",
                    solve_info.get("error", solve_info.get("status")),
                )
                sol_q = previous_published_q.copy()

            log_now = time.monotonic()
            if (
                self._chest_log_last_t is None
                or log_now - self._chest_log_last_t >= 1.0
            ):
                self._chest_log_last_t = log_now
                active_targets = ["head"]
                if self._profile.use_hand_targets:
                    active_targets.append("wrists")
                if chest_target is not None:
                    active_targets.append("chest")
                head_trans_w, head_rot_w, chest_trans_w, chest_rot_w = (
                    self._upper_body_task_weights(
                        waist_activation,
                        transition=self._waist_transition,
                    )
                )
                logger.info(
                    "[G1IK] waist transition targets=%s chest_conf=%.3f valid=%s "
                    "forced_off=%s raw_act=%.3f act=%.3f "
                    "weights=head(%.3f,%.1f) chest(%.3f,%.3f) "
                    "waist_limit=%.4f hold_w=%.3f "
                    "c_left=%.3f c_right=%.3f "
                    "seed_lead=(waist=%.3f arm=%.3f neck=%.3f)",
                    "+".join(active_targets),
                    transition_result.confidence,
                    chest_target_valid,
                    transition_result.forced_disable,
                    transition_result.raw_activation,
                    waist_activation,
                    head_trans_w,
                    head_rot_w,
                    chest_trans_w,
                    chest_rot_w,
                    waist_delta_limit,
                    waist_hold_weight,
                    float(vr_data.get("left_controller_confidence", 0.0)),
                    float(vr_data.get("right_controller_confidence", 0.0)),
                    float(np.max(np.abs(ik_seed_q[:3] - current_q[:3]))),
                    float(np.max(np.abs(ik_seed_q[3:17] - current_q[3:17]))),
                    float(np.max(np.abs(ik_seed_q[17:19] - current_q[17:19]))),
                )
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
            sol_q = previous_published_q.copy()
        sol_q = self._limit_command_lead(
            sol_q,
            current_q,
            preserve_waist=True,
        )
        # RUN 중 급격한 변화를 감지하면 그 시점부터 블렌딩 시작
        self._maybe_trigger_sudden_blend(sol_q)
        cmd_q = self._apply_run_blend(
            sol_q,
            preserve_waist=True,
        )
        cmd_q = self._limit_command_lead(
            cmd_q,
            current_q,
            preserve_waist=True,
        )
        self._maybe_log_blend(current_q, sol_q, cmd_q)
        cmd_q = np.asarray(cmd_q, dtype=np.float64).copy()

        expected_waist_limit = waist_delta_limit + 1e-9
        waist_delta = np.abs(
            cmd_q[: len(WAIST_INDICES)]
            - previous_published_q[: len(WAIST_INDICES)]
        )
        if np.any(waist_delta > expected_waist_limit):
            logger.error(
                "[G1IK] Final waist command escaped QP transition window; "
                "holding previous published waist. delta=%s limit=%.6f",
                waist_delta,
                waist_delta_limit,
            )
            cmd_q[: len(WAIST_INDICES)] = previous_published_q[
                : len(WAIST_INDICES)
            ]

        if self._publish_targets(cmd_q):
            self._record_published_command(cmd_q)



    def on_stop(self) -> None:
        if self._shared_memory:
            for key, mgr in self._shared_memory.items():
                try:
                    mgr.worker_close()
                except Exception:
                    logger.exception(f"[{self.ctx.name}] failed to close shared memory {key}")

        logger.info(f"[{self.ctx.name}] stop")
