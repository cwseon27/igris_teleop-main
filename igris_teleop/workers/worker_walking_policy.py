from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

import logging_mp

from ..core.events import EventSnapshot
from ..core.state_machine import TransitionResult
from ..core.worker_base import SingleRateWorker, WorkerContext
from ..policies.walking import (
    WALKING_ACTION_DIM,
    WALKING_LEG_DEFAULT_Q,
    WALKING_MAX_SINGLE_OBS_DIM,
    WALKING_NEUTRAL_FULL_Q,
    WALKING_NEUTRAL_HAND_Q,
    WALKING_OBS_STALE_TIMEOUT,
    WALKING_OBS_SCALES,
    WALKING_START_BLEND_TIME,
    WALKING_START_BASE_PITCH_TOL,
    WALKING_START_BASE_ROLL_TOL,
    WALKING_START_DQ_TOL,
    WALKING_START_GYRO_TOL,
    WALKING_START_POSE_CHECK_INDICES,
    WALKING_START_POSE_TOL,
    WALKING_START_REASON_LOG_INTERVAL,
    WALKING_START_SETTLE_TIME,
    WALKING_START_TIMEOUT,
    clamp_walking_command,
    get_walking_policy_profile,
    resolve_walking_policy_path,
    resolve_walking_policy_profile,
    walking_command_is_zero,
    walking_cmd_limits,
    walking_policy_profile_code,
)


logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)

DEFAULT_WALKING_IMU_TO_BASE_QUAT_WXYZ = np.array(
    [0.7071068, 0.0, 0.7071068, 0.0],
    dtype=np.float64,
)
WORLD_GRAVITY_VECTOR = np.array([0.0, 0.0, -1.0], dtype=np.float64)


@dataclass(frozen=True)
class WalkingObservationInputs:
    obs_seq: float
    q: np.ndarray
    dq: np.ndarray
    quat: np.ndarray
    gyro_base: np.ndarray
    rpy_base: np.ndarray
    projected_gravity: np.ndarray


class WalkingPolicyWorker(SingleRateWorker):
    def __init__(self, ctx: WorkerContext, hz: float | None = None) -> None:
        self.profile_name = resolve_walking_policy_profile(getattr(ctx.run_config, "walking_policy_profile", None))
        self.profile = get_walking_policy_profile(self.profile_name)
        super().__init__(ctx, hz=self.profile.loop_hz if hz is None else hz)
        self._shared_memory = ctx.shared_memory
        self._owns_shared_memory = False

        self.obs_shm = self._shared_memory.get("obs_shm")
        self.act_shm = self._shared_memory.get("act_shm")
        self.walking_cmd_shm = self._shared_memory.get("walking_cmd_shm")
        self.walking_debug_shm = self._shared_memory.get("walking_debug_shm")

        self.policy_path = resolve_walking_policy_path(
            getattr(ctx.run_config, "walking_policy_path", None),
            profile=self.profile_name,
        )
        self.command_limits = walking_cmd_limits(self.profile_name)

        self._torch = None
        self._policy = None
        self._ort = None
        self._ort_session = None
        self._ort_input_name = None
        self._ort_output_name = None
        self._was_running = False
        self._policy_tick = 0
        self._obs_history: deque[np.ndarray] = deque(maxlen=max(1, self.profile.frame_stack))
        self._previous_action = np.zeros((WALKING_ACTION_DIM,), dtype=np.float64)
        self._policy_enabled_prev = False
        raw_startup_blend_enabled = getattr(ctx.run_config, "walking_startup_blend_enabled", None)
        self._startup_blend_enabled = True if raw_startup_blend_enabled is None else bool(raw_startup_blend_enabled)
        self._imu_to_base_rot = self._resolve_imu_to_base_rotation()
        self._debug_seq = 0.0
        self._reset_runtime_buffers()

    def on_start(self) -> None:
        self._load_policy()
        self._write_policy_enabled(False)
        logger.info(
            "[%s] start walking_profile=%s backend=%s hz=%.1f startup_blend=%s walking_policy=%s",
            self.ctx.name,
            self.profile_name,
            self.profile.backend,
            self.hz,
            self._startup_blend_enabled,
            self.policy_path,
        )
        self._write_neutral_targets()

    def _load_policy(self) -> None:
        if self.profile.backend == "torchscript":
            self._load_torchscript_policy()
            return
        if self.profile.backend == "onnx":
            self._load_onnx_policy()
            return
        raise RuntimeError(f"Unsupported walking backend: {self.profile.backend}")

    def _load_torchscript_policy(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("WalkingPolicyWorker requires torch to load TorchScript policy.") from exc

        if not self.policy_path.is_file():
            raise FileNotFoundError(f"Walking policy not found: {self.policy_path}")

        self._torch = torch
        self._policy = torch.jit.load(str(self.policy_path), map_location="cpu")
        self._policy.eval()

        dummy = torch.zeros((1, self.profile.frame_stack * self.profile.single_obs_dim), dtype=torch.float32)
        with torch.inference_mode():
            out = self._policy(dummy)
        out_vec = self._normalize_action_output(out)
        if out_vec.size != WALKING_ACTION_DIM:
            raise ValueError(
                f"Walking policy action dim mismatch: expected {WALKING_ACTION_DIM}, got {out_vec.size}"
            )

    def _load_onnx_policy(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "WalkingPolicyWorker v2 requires onnxruntime. Install it in the project environment first."
            ) from exc

        if not self.policy_path.is_file():
            raise FileNotFoundError(f"Walking policy not found: {self.policy_path}")

        session = ort.InferenceSession(str(self.policy_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected 1 ONNX input, got {len(inputs)}")
        if not outputs:
            raise ValueError("Expected at least one ONNX output")

        input_shape = inputs[0].shape
        output_shape = outputs[0].shape
        input_last_dim = input_shape[-1] if input_shape else None
        output_last_dim = output_shape[-1] if output_shape else None
        if isinstance(input_last_dim, int) and input_last_dim != self.profile.single_obs_dim:
            raise ValueError(
                f"Walking ONNX input dim mismatch: expected {self.profile.single_obs_dim}, got {input_last_dim}"
            )
        if isinstance(output_last_dim, int) and output_last_dim != WALKING_ACTION_DIM:
            raise ValueError(
                f"Walking ONNX output dim mismatch: expected {WALKING_ACTION_DIM}, got {output_last_dim}"
            )

        dummy = np.zeros((1, self.profile.single_obs_dim), dtype=np.float32)
        out = session.run([outputs[0].name], {inputs[0].name: dummy})[0]
        out_vec = self._normalize_action_output(out)
        if out_vec.size != WALKING_ACTION_DIM:
            raise ValueError(
                f"Walking ONNX action dim mismatch: expected {WALKING_ACTION_DIM}, got {out_vec.size}"
            )

        self._ort = ort
        self._ort_session = session
        self._ort_input_name = inputs[0].name
        self._ort_output_name = outputs[0].name

    @staticmethod
    def _normalize_action_output(out: object) -> np.ndarray:
        if isinstance(out, (tuple, list)):
            out = out[0]
        if hasattr(out, "detach"):
            out = out.detach().cpu().numpy()
        return np.asarray(out, dtype=np.float64).reshape(-1)

    def _reset_runtime_buffers(self) -> None:
        self._policy_tick = 0
        self._previous_action = np.zeros((WALKING_ACTION_DIM,), dtype=np.float64)
        self._obs_history = deque(
            [np.zeros((1, self.profile.single_obs_dim), dtype=np.float32) for _ in range(max(1, self.profile.frame_stack))],
            maxlen=max(1, self.profile.frame_stack),
        )
        self._startup_phase = "idle"
        self._startup_started_at: float | None = None
        self._startup_settle_since: float | None = None
        self._startup_blend_started_at: float | None = None
        self._startup_last_wait_log_at: float | None = None
        self._last_step_monotonic: float | None = None
        self._last_obs_seq: int | None = None
        self._last_obs_seq_change_at: float | None = None
        self._obs_seq_tracking_active = False
        self._v2_phase_start_tick = 0
        self._v2_cmd_was_zero = True

    def _rpy_deg_to_rotation_matrix(self, rpy_deg: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64).reshape(3))
        return self._rpy_to_rotation_matrix(np.array([roll, pitch, yaw], dtype=np.float64))

    @staticmethod
    def _rpy_to_rotation_matrix(rpy_rad: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = np.asarray(rpy_rad, dtype=np.float64).reshape(3)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float64,
        )

    def _default_imu_to_base_rotation(self) -> np.ndarray:
        rot = self._quat_wxyz_to_rotation_matrix(DEFAULT_WALKING_IMU_TO_BASE_QUAT_WXYZ)
        if rot is None:
            return np.eye(3, dtype=np.float64)
        return rot

    def _resolve_imu_to_base_rotation(self) -> np.ndarray:
        raw_value = getattr(self.ctx.run_config, "walking_imu_to_base_rpy_deg", None)
        if raw_value is None:
            raw_value = os.getenv("IGRIS_WALKING_IMU_TO_BASE_RPY_DEG")

        if raw_value is None or str(raw_value).strip() == "":
            default_rot = self._default_imu_to_base_rotation()
            logger.info(
                "[%s] walking IMU->base rotation: XML default quat_wxyz=%s (override with IGRIS_WALKING_IMU_TO_BASE_RPY_DEG=roll,pitch,yaw)",
                self.ctx.name,
                DEFAULT_WALKING_IMU_TO_BASE_QUAT_WXYZ.tolist(),
            )
            return default_rot

        try:
            tokens = str(raw_value).replace(",", " ").replace(";", " ").split()
            values = np.asarray([float(token) for token in tokens], dtype=np.float64)
        except Exception:
            values = np.empty((0,), dtype=np.float64)

        if values.size != 3 or not np.all(np.isfinite(values)):
            default_rot = self._default_imu_to_base_rotation()
            logger.warning(
                "[%s] invalid IGRIS_WALKING_IMU_TO_BASE_RPY_DEG=%r, using XML default quat_wxyz=%s",
                self.ctx.name,
                raw_value,
                DEFAULT_WALKING_IMU_TO_BASE_QUAT_WXYZ.tolist(),
            )
            return default_rot

        rot = self._rpy_deg_to_rotation_matrix(values)
        logger.info("[%s] walking IMU->base rotation rpy_deg=%s", self.ctx.name, values.tolist())
        return rot

    def _quat_wxyz_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray | None:
        q = np.asarray(quat, dtype=np.float64).reshape(-1)
        if q.size < 4:
            return None

        norm = np.linalg.norm(q[:4])
        if not np.isfinite(norm) or norm <= 1e-9:
            return None

        w, x, y, z = q[:4] / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _rotation_matrix_to_rpy(self, rot: np.ndarray) -> np.ndarray:
        r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
        t2 = float(np.clip(-r[2, 0], -1.0, 1.0))
        return np.array(
            [
                math.atan2(r[2, 1], r[2, 2]),
                math.asin(t2),
                math.atan2(r[1, 0], r[0, 0]),
            ],
            dtype=np.float64,
        )

    def _base_rotation_from_imu_quat(self, quat_wxyz: np.ndarray) -> np.ndarray | None:
        imu_rot = self._quat_wxyz_to_rotation_matrix(quat_wxyz)
        if imu_rot is None:
            return None
        return imu_rot @ self._imu_to_base_rot.T

    def _normalize_imu_inputs(
        self,
        quat_wxyz: np.ndarray,
        gyro_imu: np.ndarray,
        fallback_rpy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        gyro_base = self._imu_to_base_rot @ np.asarray(gyro_imu, dtype=np.float64).reshape(3)
        base_rot = self._base_rotation_from_imu_quat(quat_wxyz)
        if base_rot is None:
            return gyro_base, np.asarray(fallback_rpy, dtype=np.float64).reshape(3)

        # Match the MuJoCo asset contract: base_link_imu_site is mounted with a
        # fixed rotation relative to base_link, so convert IMU-frame state back
        # into the base frame before building the policy observation.
        return gyro_base, self._rotation_matrix_to_rpy(base_rot)

    def _projected_gravity(self, quat_wxyz: np.ndarray, fallback_rpy: np.ndarray) -> np.ndarray:
        base_rot = self._base_rotation_from_imu_quat(quat_wxyz)
        if base_rot is None:
            base_rot = self._rpy_to_rotation_matrix(np.asarray(fallback_rpy, dtype=np.float64).reshape(3))
        return np.asarray(base_rot.T @ WORLD_GRAVITY_VECTOR, dtype=np.float64).reshape(3)

    def _write_policy_enabled(self, enabled: bool) -> None:
        if self.walking_cmd_shm is None:
            return
        try:
            self.walking_cmd_shm.write_data(
                profile_code=np.float64(walking_policy_profile_code(self.profile_name)),
                policy_enabled=np.float64(1.0 if enabled else 0.0),
            )
        except Exception:
            logger.warning("[walking_policy] failed to write policy_enabled=%s", enabled, exc_info=True)

    def _command_is_zero(self, cmd: dict[str, float]) -> bool:
        return walking_command_is_zero(cmd)

    @staticmethod
    def _wrap_to_pi(phase: float) -> float:
        return float((phase + math.pi) % (2.0 * math.pi) - math.pi)

    def _seed_obs_history(self, single_obs: np.ndarray) -> None:
        seeded = np.asarray(single_obs, dtype=np.float32).reshape(1, self.profile.single_obs_dim)
        self._obs_history = deque(
            [seeded.copy() for _ in range(max(1, self.profile.frame_stack))],
            maxlen=max(1, self.profile.frame_stack),
        )

    def _startup_metrics(self, q: np.ndarray, dq: np.ndarray, gyro: np.ndarray) -> tuple[float, float, float]:
        pose_delta = np.abs(np.asarray(q, dtype=np.float64).reshape(-1) - WALKING_LEG_DEFAULT_Q)
        pose_check_indices = np.asarray(WALKING_START_POSE_CHECK_INDICES, dtype=np.int64).reshape(-1)
        if pose_check_indices.size > 0:
            pose_error = float(np.max(pose_delta[pose_check_indices]))
        else:
            pose_error = float(np.max(pose_delta))
        dq_max = float(np.max(np.abs(dq)))
        gyro_max = float(np.max(np.abs(gyro)))
        return pose_error, dq_max, gyro_max

    @staticmethod
    def _startup_base_tilt_metrics(rpy_base: np.ndarray) -> tuple[float, float]:
        rpy = np.asarray(rpy_base, dtype=np.float64).reshape(-1)
        if rpy.size < 2:
            return float("inf"), float("inf")
        return float(abs(rpy[0])), float(abs(rpy[1]))

    def _startup_reasons(self, q: np.ndarray, dq: np.ndarray, gyro: np.ndarray, rpy_base: np.ndarray) -> list[str]:
        reasons: list[str] = []
        pose_error, dq_max, gyro_max = self._startup_metrics(q, dq, gyro)
        base_roll_abs, base_pitch_abs = self._startup_base_tilt_metrics(rpy_base)
        if pose_error > WALKING_START_POSE_TOL:
            reasons.append(f"stance 미도달 pose_err={pose_error:.3f}rad")
        if dq_max > WALKING_START_DQ_TOL:
            reasons.append(f"다리 속도 큼 dq={dq_max:.3f}rad/s")
        if gyro_max > WALKING_START_GYRO_TOL:
            reasons.append(f"IMU gyro 큼 gyro={gyro_max:.3f}rad/s")
        if base_roll_abs > WALKING_START_BASE_ROLL_TOL:
            reasons.append(
                f"base roll 큼 roll={base_roll_abs:.3f}rad tol={WALKING_START_BASE_ROLL_TOL:.3f}"
            )
        if base_pitch_abs > WALKING_START_BASE_PITCH_TOL:
            reasons.append(
                f"base pitch 큼 pitch={base_pitch_abs:.3f}rad tol={WALKING_START_BASE_PITCH_TOL:.3f}"
            )
        return reasons

    @staticmethod
    def _startup_phase_code(phase: str) -> float:
        phase_map = {
            "idle": 0.0,
            "waiting": 1.0,
            "blending": 2.0,
            "running": 3.0,
        }
        return phase_map.get(phase, -1.0)

    def _startup_in_progress(self) -> bool:
        return self._startup_phase in {"waiting", "blending"}

    def _update_obs_seq_tracking(self, obs_seq: float, now: float) -> float | None:
        try:
            seq_value = int(np.rint(float(obs_seq)))
        except Exception:
            return None
        if not np.isfinite(obs_seq):
            return None
        if self._last_obs_seq is None:
            self._last_obs_seq = seq_value
            self._last_obs_seq_change_at = now
            return None
        if seq_value != self._last_obs_seq:
            self._obs_seq_tracking_active = True
            self._last_obs_seq = seq_value
            self._last_obs_seq_change_at = now
            return None
        if not self._obs_seq_tracking_active or self._last_obs_seq_change_at is None:
            return None
        stale_age = now - self._last_obs_seq_change_at
        if stale_age >= WALKING_OBS_STALE_TIMEOUT:
            return stale_age
        return None

    def _measure_loop_dt(self, now: float) -> float:
        if self._last_step_monotonic is None:
            loop_dt = float("nan")
        else:
            loop_dt = float(now - self._last_step_monotonic)
        self._last_step_monotonic = now
        return loop_dt

    def _publish_debug_snapshot(
        self,
        *,
        now: float,
        cmd: dict[str, float] | None = None,
        single_obs: np.ndarray | None = None,
        action_raw: np.ndarray | None = None,
        action_applied: np.ndarray | None = None,
        gyro_base: np.ndarray | None = None,
        rpy_base: np.ndarray | None = None,
        target_leg_q: np.ndarray | None = None,
        pose_err_max: float = float("nan"),
        dq_max: float = float("nan"),
        gyro_max: float = float("nan"),
        blend_alpha: float = 0.0,
    ) -> None:
        if self.walking_debug_shm is None:
            return

        loop_dt = self._measure_loop_dt(now)
        self._debug_seq += 1.0

        cmd = cmd or {name: 0.0 for name in self.command_limits}
        obs_vec = np.full((WALKING_MAX_SINGLE_OBS_DIM,), np.nan, dtype=np.float64)
        if single_obs is not None:
            single_obs_vec = np.asarray(single_obs, dtype=np.float64).reshape(-1)
            obs_vec[: min(single_obs_vec.size, WALKING_MAX_SINGLE_OBS_DIM)] = single_obs_vec[:WALKING_MAX_SINGLE_OBS_DIM]

        raw_action = np.zeros((WALKING_ACTION_DIM,), dtype=np.float64)
        if action_raw is not None:
            raw_action = np.asarray(action_raw, dtype=np.float64).reshape(-1)[:WALKING_ACTION_DIM]

        applied_action = np.zeros((WALKING_ACTION_DIM,), dtype=np.float64)
        if action_applied is not None:
            applied_action = np.asarray(action_applied, dtype=np.float64).reshape(-1)[:WALKING_ACTION_DIM]

        gyro_vec = np.full((3,), np.nan, dtype=np.float64)
        if gyro_base is not None:
            gyro_vec = np.asarray(gyro_base, dtype=np.float64).reshape(-1)[:3]

        rpy_vec = np.full((3,), np.nan, dtype=np.float64)
        if rpy_base is not None:
            rpy_vec = np.asarray(rpy_base, dtype=np.float64).reshape(-1)[:3]

        leg_target = WALKING_LEG_DEFAULT_Q.copy()
        if target_leg_q is not None:
            leg_target = np.asarray(target_leg_q, dtype=np.float64).reshape(-1)[:WALKING_ACTION_DIM]

        try:
            self.walking_debug_shm.write_data(
                seq=np.float64(self._debug_seq),
                policy_tick=np.float64(self._policy_tick),
                startup_phase_code=np.float64(self._startup_phase_code(self._startup_phase)),
                blend_alpha=np.float64(blend_alpha),
                pose_err_max=np.float64(pose_err_max),
                dq_max=np.float64(dq_max),
                gyro_max=np.float64(gyro_max),
                loop_dt=np.float64(loop_dt),
                policy_cmd_vx=np.float64(cmd.get("vx", 0.0)),
                policy_cmd_vy=np.float64(cmd.get("vy", 0.0)),
                policy_cmd_dyaw=np.float64(cmd.get("dyaw", 0.0)),
                policy_action_raw=raw_action,
                policy_action_applied=applied_action,
                previous_action=np.asarray(self._previous_action, dtype=np.float64).reshape(-1),
                policy_single_obs=obs_vec,
                policy_gyro_base=gyro_vec,
                policy_rpy_base=rpy_vec,
                policy_target_leg_q=leg_target,
            )
        except Exception:
            logger.debug("[walking_policy] failed to publish walking debug snapshot", exc_info=True)

    def _log_startup_wait(self, now: float, message: str) -> None:
        if (
            self._startup_last_wait_log_at is None
            or (now - self._startup_last_wait_log_at) >= WALKING_START_REASON_LOG_INTERVAL
        ):
            logger.info("[%s] walking start waiting: %s", self.ctx.name, message)
            self._startup_last_wait_log_at = now

    def _begin_startup(self, now: float) -> None:
        self._startup_phase = "waiting"
        self._startup_started_at = now
        self._startup_settle_since = None
        self._startup_blend_started_at = None
        self._startup_last_wait_log_at = None
        logger.info("[%s] walking start armed, waiting for safe startup conditions", self.ctx.name)

    def _disable_policy_with_reason(self, reason: str) -> None:
        logger.warning("[%s] walking start rejected: %s", self.ctx.name, reason)
        self._write_policy_enabled(False)
        self._reset_runtime_buffers()
        self._policy_enabled_prev = False

    def _read_walking_command(self) -> dict[str, float]:
        cmd = {name: 0.0 for name in self.command_limits}
        if self.walking_cmd_shm is None:
            return cmd
        try:
            data = self.walking_cmd_shm.read_data()
        except Exception:
            logger.debug("[walking_policy] failed to read walking_cmd_shm", exc_info=True)
            return cmd

        for field in cmd:
            try:
                cmd[field] = clamp_walking_command(
                    field,
                    float(np.asarray(data.get(field, 0.0)).reshape(()).item()),
                    profile=self.profile_name,
                )
            except Exception:
                cmd[field] = 0.0
        return cmd

    def _policy_enabled(self) -> bool:
        if self.walking_cmd_shm is None:
            return False
        try:
            data = self.walking_cmd_shm.read_data()
            return bool(float(np.asarray(data.get("policy_enabled", 0.0)).reshape(()).item()) > 0.5)
        except Exception:
            logger.debug("[walking_policy] failed to read policy_enabled", exc_info=True)
            return False

    def _read_observation_inputs(self) -> WalkingObservationInputs | None:
        if self.obs_shm is None:
            return None
        try:
            obs = self.obs_shm.read_data()
            obs_seq = float(np.asarray(obs.get("obs_seq", np.nan), dtype=np.float64).reshape(()).item())
            q = np.asarray(obs.get("obs_leg"), dtype=np.float64).reshape(-1)
            dq = np.asarray(obs.get("obs_leg_dq"), dtype=np.float64).reshape(-1)
            quat = np.asarray(obs.get("obs_imu_quat"), dtype=np.float64).reshape(-1)
            gyro = np.asarray(obs.get("obs_imu_gyro"), dtype=np.float64).reshape(-1)
            rpy = np.asarray(obs.get("obs_imu_rpy"), dtype=np.float64).reshape(-1)
        except Exception:
            logger.debug("[walking_policy] failed to read obs_shm", exc_info=True)
            return None

        if q.size != 12 or dq.size != 12 or quat.size < 4 or gyro.size != 3 or rpy.size != 3:
            return None

        gyro_base, rpy_base = self._normalize_imu_inputs(quat, gyro, rpy)
        projected_gravity = self._projected_gravity(quat, rpy_base)
        return WalkingObservationInputs(
            obs_seq=obs_seq,
            q=q,
            dq=dq,
            quat=np.asarray(quat, dtype=np.float64).reshape(-1),
            gyro_base=np.asarray(gyro_base, dtype=np.float64).reshape(3),
            rpy_base=np.asarray(rpy_base, dtype=np.float64).reshape(3),
            projected_gravity=np.asarray(projected_gravity, dtype=np.float64).reshape(3),
        )

    def _v1_phase(self) -> float:
        return 2.0 * math.pi * (self._policy_tick / self.profile.loop_hz) / self.profile.cycle_time

    def _v2_gait_phase(self, cmd: dict[str, float]) -> tuple[float, float]:
        if self._command_is_zero(cmd):
            self._v2_cmd_was_zero = True
            return math.pi, math.pi
        if self._v2_cmd_was_zero:
            self._v2_phase_start_tick = self._policy_tick
            self._v2_cmd_was_zero = False
        phase = 2.0 * math.pi * (
            (self._policy_tick - self._v2_phase_start_tick) / self.profile.loop_hz
        ) / self.profile.cycle_time
        left_phase = self._wrap_to_pi(phase)
        right_phase = self._wrap_to_pi(phase + math.pi)
        return left_phase, right_phase

    def _build_v1_single_observation(
        self,
        inputs: WalkingObservationInputs,
        cmd: dict[str, float],
    ) -> np.ndarray:
        obs = np.zeros((1, self.profile.single_obs_dim), dtype=np.float32)
        phase = self._v1_phase()

        obs[0, 0] = math.sin(phase)
        obs[0, 1] = math.cos(phase)
        obs[0, 2] = cmd["vx"] * WALKING_OBS_SCALES["lin_vel"]
        obs[0, 3] = cmd["vy"] * WALKING_OBS_SCALES["lin_vel"]
        obs[0, 4] = cmd["dyaw"] * WALKING_OBS_SCALES["ang_vel"]
        obs[0, 5:17] = (inputs.q - WALKING_LEG_DEFAULT_Q) * WALKING_OBS_SCALES["dof_pos"]
        obs[0, 17:29] = inputs.dq * WALKING_OBS_SCALES["dof_vel"]
        obs[0, 29:41] = self._previous_action
        obs[0, 41:44] = inputs.gyro_base
        obs[0, 44:47] = inputs.rpy_base
        return np.clip(obs, -self.profile.obs_clip, self.profile.obs_clip)

    def _build_v2_single_observation(
        self,
        inputs: WalkingObservationInputs,
        cmd: dict[str, float],
    ) -> np.ndarray:
        obs = np.zeros((1, self.profile.single_obs_dim), dtype=np.float32)
        left_phase, right_phase = self._v2_gait_phase(cmd)

        obs[0, 0:12] = self._previous_action
        obs[0, 12:15] = inputs.gyro_base * 0.5
        obs[0, 15] = cmd["dyaw"]
        obs[0, 16:18] = np.array([cmd["vx"], cmd["vy"]], dtype=np.float32)
        obs[0, 18:20] = np.array([math.cos(left_phase), math.cos(right_phase)], dtype=np.float32)
        obs[0, 20:32] = inputs.q - WALKING_LEG_DEFAULT_Q
        obs[0, 32:44] = inputs.dq * 0.08
        obs[0, 44:47] = inputs.projected_gravity
        obs[0, 47:49] = np.array([math.sin(left_phase), math.sin(right_phase)], dtype=np.float32)
        return np.clip(obs, -self.profile.obs_clip, self.profile.obs_clip)

    def _build_single_observation(
        self,
        inputs: WalkingObservationInputs,
        cmd: dict[str, float],
    ) -> np.ndarray:
        if self.profile.backend == "torchscript":
            return self._build_v1_single_observation(inputs, cmd)
        return self._build_v2_single_observation(inputs, cmd)

    def _infer_action(self, single_obs: np.ndarray) -> np.ndarray:
        if self.profile.backend == "torchscript":
            assert self._policy is not None
            assert self._torch is not None

            self._obs_history.append(single_obs.astype(np.float32, copy=False))
            stacked = np.concatenate(list(self._obs_history), axis=1)
            policy_input = self._torch.from_numpy(stacked.astype(np.float32, copy=False))
            with self._torch.inference_mode():
                out = self._policy(policy_input)
            action = self._normalize_action_output(out)
        else:
            assert self._ort_session is not None
            assert self._ort_input_name is not None
            assert self._ort_output_name is not None
            policy_input = np.asarray(single_obs, dtype=np.float32).reshape(1, self.profile.single_obs_dim)
            out = self._ort_session.run([self._ort_output_name], {self._ort_input_name: policy_input})[0]
            action = self._normalize_action_output(out)

        if action.size != WALKING_ACTION_DIM:
            raise ValueError(f"Walking policy produced invalid action size: {action.size}")
        return np.clip(action, -self.profile.action_clip, self.profile.action_clip)

    def _write_targets(self, leg_target_q: np.ndarray) -> None:
        if self.act_shm is None:
            return

        self.act_shm.write_data(
            act_leg=np.asarray(leg_target_q, dtype=np.float64).reshape(-1),
            act_waist=WALKING_NEUTRAL_FULL_Q[:3],
            act_arm=WALKING_NEUTRAL_FULL_Q[15:29],
            act_neck=WALKING_NEUTRAL_FULL_Q[29:31],
            act_hand=WALKING_NEUTRAL_HAND_Q,
        )

    def _write_neutral_targets(self) -> None:
        self._write_targets(WALKING_LEG_DEFAULT_Q)

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        if self.profile.backend == "torchscript" and self._policy is None:
            return
        if self.profile.backend == "onnx" and self._ort_session is None:
            return

        policy_enabled = self._policy_enabled()
        if self.state.name != "RUN" or not policy_enabled:
            if self._was_running or self._startup_phase != "idle":
                self._reset_runtime_buffers()
            self._policy_enabled_prev = policy_enabled
            self._was_running = False
            self._write_neutral_targets()
            self._publish_debug_snapshot(
                now=time.monotonic(),
                cmd=self._read_walking_command(),
                target_leg_q=WALKING_LEG_DEFAULT_Q,
            )
            return

        now = time.monotonic()
        cmd = self._read_walking_command()
        if policy_enabled and not self._policy_enabled_prev:
            self._reset_runtime_buffers()
            if not self._command_is_zero(cmd):
                cmd_text = ", ".join(f"{name}={value:.3f}" for name, value in cmd.items())
                self._disable_policy_with_reason(f"command가 0이 아님 ({cmd_text})")
                self._write_neutral_targets()
                self._publish_debug_snapshot(
                    now=now,
                    cmd=cmd,
                    target_leg_q=WALKING_LEG_DEFAULT_Q,
                )
                return
            self._begin_startup(now)
        self._policy_enabled_prev = True

        inputs = self._read_observation_inputs()
        if inputs is None:
            if self._startup_in_progress() and self._startup_started_at is not None:
                if (now - self._startup_started_at) >= WALKING_START_TIMEOUT:
                    self._disable_policy_with_reason("startup timeout (관측 입력 없음)")
            else:
                self._log_startup_wait(now, "관측 입력 대기 중")
            self._write_neutral_targets()
            self._publish_debug_snapshot(
                now=now,
                cmd=cmd,
                target_leg_q=WALKING_LEG_DEFAULT_Q,
            )
            return

        pose_err_max, dq_max, gyro_max = self._startup_metrics(inputs.q, inputs.dq, inputs.gyro_base)
        obs_stale_age = self._update_obs_seq_tracking(inputs.obs_seq, now)
        if obs_stale_age is not None:
            obs_seq_value = int(np.rint(float(inputs.obs_seq)))
            self._disable_policy_with_reason(
                f"observation stale for {obs_stale_age:.2f}s (obs_seq={obs_seq_value})"
            )
            self._write_neutral_targets()
            self._publish_debug_snapshot(
                now=now,
                cmd=cmd,
                gyro_base=inputs.gyro_base,
                rpy_base=inputs.rpy_base,
                target_leg_q=WALKING_LEG_DEFAULT_Q,
                pose_err_max=pose_err_max,
                dq_max=dq_max,
                gyro_max=gyro_max,
            )
            return

        if self._startup_in_progress() and not self._command_is_zero(cmd):
            cmd_text = ", ".join(f"{name}={value:.3f}" for name, value in cmd.items())
            self._disable_policy_with_reason(f"startup 중 command가 0이 아님 ({cmd_text})")
            self._write_neutral_targets()
            self._publish_debug_snapshot(
                now=now,
                cmd=cmd,
                gyro_base=inputs.gyro_base,
                rpy_base=inputs.rpy_base,
                target_leg_q=WALKING_LEG_DEFAULT_Q,
                pose_err_max=pose_err_max,
                dq_max=dq_max,
                gyro_max=gyro_max,
            )
            return

        if self._startup_phase == "waiting":
            reasons = self._startup_reasons(inputs.q, inputs.dq, inputs.gyro_base, inputs.rpy_base)
            if reasons:
                self._startup_settle_since = None
                if self._startup_started_at is not None and (now - self._startup_started_at) >= WALKING_START_TIMEOUT:
                    self._disable_policy_with_reason(f"startup timeout ({'; '.join(reasons)})")
                    self._write_neutral_targets()
                    self._publish_debug_snapshot(
                        now=now,
                        cmd=cmd,
                        gyro_base=inputs.gyro_base,
                        rpy_base=inputs.rpy_base,
                        target_leg_q=WALKING_LEG_DEFAULT_Q,
                        pose_err_max=pose_err_max,
                        dq_max=dq_max,
                        gyro_max=gyro_max,
                    )
                    return
                self._log_startup_wait(now, "; ".join(reasons))
                self._write_neutral_targets()
                self._publish_debug_snapshot(
                    now=now,
                    cmd=cmd,
                    gyro_base=inputs.gyro_base,
                    rpy_base=inputs.rpy_base,
                    target_leg_q=WALKING_LEG_DEFAULT_Q,
                    pose_err_max=pose_err_max,
                    dq_max=dq_max,
                    gyro_max=gyro_max,
                )
                return

            if self._startup_settle_since is None:
                self._startup_settle_since = now
                self._log_startup_wait(now, "startup 조건 만족, 안정화 확인 중")
                self._write_neutral_targets()
                self._publish_debug_snapshot(
                    now=now,
                    cmd=cmd,
                    gyro_base=inputs.gyro_base,
                    rpy_base=inputs.rpy_base,
                    target_leg_q=WALKING_LEG_DEFAULT_Q,
                    pose_err_max=pose_err_max,
                    dq_max=dq_max,
                    gyro_max=gyro_max,
                )
                return

            settle_elapsed = now - self._startup_settle_since
            if settle_elapsed < WALKING_START_SETTLE_TIME:
                self._log_startup_wait(
                    now,
                    f"startup 조건 만족, 안정화 유지 {settle_elapsed:.2f}/{WALKING_START_SETTLE_TIME:.2f}s",
                )
                self._write_neutral_targets()
                self._publish_debug_snapshot(
                    now=now,
                    cmd=cmd,
                    gyro_base=inputs.gyro_base,
                    rpy_base=inputs.rpy_base,
                    target_leg_q=WALKING_LEG_DEFAULT_Q,
                    pose_err_max=pose_err_max,
                    dq_max=dq_max,
                    gyro_max=gyro_max,
                )
                return

            single_obs = self._build_single_observation(inputs, cmd)
            self._seed_obs_history(single_obs)
            self._startup_last_wait_log_at = None
            if self._startup_blend_enabled:
                self._startup_phase = "blending"
                self._startup_blend_started_at = now
                logger.info("[%s] walking startup settled, blending policy action in", self.ctx.name)
            else:
                self._startup_phase = "running"
                self._startup_started_at = None
                self._startup_settle_since = None
                self._startup_blend_started_at = None
                logger.info("[%s] walking startup settled, startup blend disabled", self.ctx.name)

        single_obs = self._build_single_observation(inputs, cmd)
        action_raw = self._infer_action(single_obs)
        action_applied = action_raw.copy()
        blend_alpha = 1.0
        if self._startup_phase == "blending":
            blend_started_at = self._startup_blend_started_at or now
            blend_alpha = min(1.0, max(0.0, (now - blend_started_at) / WALKING_START_BLEND_TIME))
            action_applied = action_applied * blend_alpha
            if blend_alpha >= 1.0:
                self._startup_phase = "running"
                self._startup_started_at = None
                self._startup_settle_since = None
                self._startup_blend_started_at = None
                self._startup_last_wait_log_at = None
                logger.info("[%s] walking startup complete, policy fully active", self.ctx.name)
        leg_target_q = action_applied * self.profile.action_scale + WALKING_LEG_DEFAULT_Q

        self._write_targets(leg_target_q)
        self._publish_debug_snapshot(
            now=now,
            cmd=cmd,
            single_obs=single_obs,
            action_raw=action_raw,
            action_applied=action_applied,
            gyro_base=inputs.gyro_base,
            rpy_base=inputs.rpy_base,
            target_leg_q=leg_target_q,
            pose_err_max=pose_err_max,
            dq_max=dq_max,
            gyro_max=gyro_max,
            blend_alpha=blend_alpha,
        )
        self._previous_action = action_raw
        self._policy_tick += 1
        self._was_running = True

    def on_stop(self) -> None:
        self._write_neutral_targets()
        if self._shared_memory:
            for key, mgr in self._shared_memory.items():
                try:
                    mgr.worker_close()
                except Exception:
                    logger.exception("[%s] failed to close shared memory %s", self.ctx.name, key)
        logger.info("[%s] stop", self.ctx.name)
