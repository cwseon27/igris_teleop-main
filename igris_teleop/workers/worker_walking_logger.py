from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import logging_mp

from ..core.events import EventSnapshot
from ..core.project_paths import LOGS_ROOT
from ..core.state_machine import TransitionResult
from ..core.worker_base import SingleRateWorker, WorkerContext
from ..robot_control.kinematics.joints import ARM_INDICES, LEG_INDICES, NECK_INDICES, WAIST_INDICES
from ..sharedmemory.shm_schema import MODE_LAYOUT
from ..policies.walking import (
    WALKING_LEG_DEFAULT_Q,
    WALKING_MAX_SINGLE_OBS_DIM,
    WALKING_NEUTRAL_FULL_Q,
    WALKING_OBS_STALE_TIMEOUT,
    WALKING_START_BASE_PITCH_TOL,
    WALKING_START_BASE_ROLL_TOL,
    WALKING_START_POSE_CHECK_INDICES,
    default_walking_joint_profile_path,
    get_walking_policy_profile,
    resolve_walking_policy_profile,
)


logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)

MODE_FIELD_NAMES = tuple(name for (name, _, _) in MODE_LAYOUT)
SIMULATOR_MODE_KEY = "simulator"
LEG_JOINT_LABELS = tuple(idx.name for idx in LEG_INDICES)
WAIST_JOINT_LABELS = tuple(idx.name for idx in WAIST_INDICES)
ARM_JOINT_LABELS = tuple(idx.name for idx in ARM_INDICES)
NECK_JOINT_LABELS = tuple(idx.name for idx in NECK_INDICES)
WALKING_ARM_NEUTRAL_Q = np.asarray(WALKING_NEUTRAL_FULL_Q[list(ARM_INDICES)], dtype=np.float64)
WALKING_NECK_NEUTRAL_Q = np.asarray(WALKING_NEUTRAL_FULL_Q[list(NECK_INDICES)], dtype=np.float64)


class WalkingLoggerWorker(SingleRateWorker):
    def __init__(self, ctx: WorkerContext, hz: float | None = None) -> None:
        self.profile_name = resolve_walking_policy_profile(getattr(ctx.run_config, "walking_policy_profile", None))
        self.profile = get_walking_policy_profile(self.profile_name)
        super().__init__(ctx, hz=self.profile.loop_hz if hz is None else hz)
        self._shared_memory = ctx.shared_memory or {}
        self.obs_shm = self._shared_memory.get("obs_shm")
        self.act_shm = self._shared_memory.get("act_shm")
        self.tau_shm = self._shared_memory.get("tau_shm")
        self.mode_shm = self._shared_memory.get("mode_shm")
        self.walking_cmd_shm = self._shared_memory.get("walking_cmd_shm")
        self.walking_debug_shm = self._shared_memory.get("walking_debug_shm")

        self.log_dir = (LOGS_ROOT / "walking_logs").resolve()
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.runtime_environment = self._resolve_runtime_environment()
        self.log_stem = f"walking_debug_{self.runtime_environment}_{self.session_id}"
        self.csv_path = self.log_dir / f"{self.log_stem}.csv"
        self.meta_path = self.log_dir / f"{self.log_stem}.json"
        self._fp = None
        self._writer = None
        self._fieldnames = self._build_fieldnames()

    def _build_fieldnames(self) -> list[str]:
        fields = [
            "wall_time",
            "mono_time",
            "worker_state",
            "transition_reason",
        ]
        fields.extend(f"mode_{name}" for name in MODE_FIELD_NAMES)
        fields.extend(("cmd_profile_code", "cmd_vx", "cmd_vy", "cmd_dyaw", "cmd_policy_enabled"))
        fields.append("obs_seq")
        fields.extend(self._vector_fieldnames("obs_waist", len(WAIST_JOINT_LABELS), WAIST_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("obs_leg", len(LEG_JOINT_LABELS), LEG_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("obs_leg_dq", len(LEG_JOINT_LABELS), LEG_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("obs_arm", 14))
        fields.extend(self._vector_fieldnames("obs_neck", 2))
        fields.extend(self._vector_fieldnames("obs_imu_quat", 4, ("w", "x", "y", "z")))
        fields.extend(self._vector_fieldnames("obs_imu_gyro", 3, ("x", "y", "z")))
        fields.extend(self._vector_fieldnames("obs_imu_rpy", 3, ("roll", "pitch", "yaw")))
        fields.extend(self._vector_fieldnames("act_waist", len(WAIST_JOINT_LABELS), WAIST_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("act_leg", len(LEG_JOINT_LABELS), LEG_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("act_arm", 14))
        fields.extend(self._vector_fieldnames("act_neck", 2))
        fields.extend(self._vector_fieldnames("tau_est_waist", len(WAIST_JOINT_LABELS), WAIST_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("tau_est_leg", len(LEG_JOINT_LABELS), LEG_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("tau_est_arm", 14))
        fields.extend(self._vector_fieldnames("tau_est_neck", 2))
        fields.extend(
            (
                "dbg_seq",
                "dbg_policy_tick",
                "dbg_startup_phase_code",
                "dbg_blend_alpha",
                "dbg_pose_err_max",
                "dbg_dq_max",
                "dbg_gyro_max",
                "dbg_loop_dt",
                "dbg_policy_cmd_vx",
                "dbg_policy_cmd_vy",
                "dbg_policy_cmd_dyaw",
            )
        )
        fields.extend(self._vector_fieldnames("dbg_policy_action_raw", len(LEG_JOINT_LABELS), LEG_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("dbg_policy_action_applied", len(LEG_JOINT_LABELS), LEG_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("dbg_previous_action", len(LEG_JOINT_LABELS), LEG_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("dbg_policy_single_obs", WALKING_MAX_SINGLE_OBS_DIM))
        fields.extend(self._vector_fieldnames("dbg_policy_gyro_base", 3, ("x", "y", "z")))
        fields.extend(self._vector_fieldnames("dbg_policy_rpy_base", 3, ("roll", "pitch", "yaw")))
        fields.extend(self._vector_fieldnames("dbg_policy_target_leg_q", len(LEG_JOINT_LABELS), LEG_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("dbg_active_kp_leg", len(LEG_JOINT_LABELS), LEG_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("dbg_active_kd_leg", len(LEG_JOINT_LABELS), LEG_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("dbg_active_kp_waist", len(WAIST_JOINT_LABELS), WAIST_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("dbg_active_kd_waist", len(WAIST_JOINT_LABELS), WAIST_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("dbg_active_kp_arm", len(ARM_JOINT_LABELS), ARM_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("dbg_active_kd_arm", len(ARM_JOINT_LABELS), ARM_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("dbg_active_kp_neck", len(NECK_JOINT_LABELS), NECK_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("dbg_active_kd_neck", len(NECK_JOINT_LABELS), NECK_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("derived_leg_target_error", len(LEG_JOINT_LABELS), LEG_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("derived_leg_default_error", len(LEG_JOINT_LABELS), LEG_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("derived_waist_target_error", len(WAIST_JOINT_LABELS), WAIST_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("derived_arm_target_error", len(ARM_JOINT_LABELS), ARM_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("derived_arm_neutral_error", len(ARM_JOINT_LABELS), ARM_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("derived_neck_target_error", len(NECK_JOINT_LABELS), NECK_JOINT_LABELS))
        fields.extend(self._vector_fieldnames("derived_neck_neutral_error", len(NECK_JOINT_LABELS), NECK_JOINT_LABELS))
        return fields

    @staticmethod
    def _vector_fieldnames(prefix: str, size: int, labels: tuple[str, ...] | None = None) -> list[str]:
        if labels is None:
            labels = tuple(str(idx) for idx in range(size))
        return [f"{prefix}_{label}" for label in labels[:size]]

    @staticmethod
    def _safe_read(shm_mgr) -> dict:
        if shm_mgr is None:
            return {}
        try:
            return shm_mgr.read_data()
        except Exception:
            return {}

    @staticmethod
    def _read_scalar(data: dict, key: str, default: float = float("nan")) -> float:
        try:
            return float(np.asarray(data.get(key, default), dtype=np.float64).reshape(()).item())
        except Exception:
            return float(default)

    @staticmethod
    def _read_vector(data: dict, key: str, size: int) -> np.ndarray:
        out = np.full((size,), np.nan, dtype=np.float64)
        try:
            arr = np.asarray(data.get(key), dtype=np.float64).reshape(-1)
        except Exception:
            return out
        n = min(size, arr.size)
        if n > 0:
            out[:n] = arr[:n]
        return out

    @staticmethod
    def _fill_vector(row: dict[str, object], prefix: str, values: np.ndarray, labels: tuple[str, ...] | None = None) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if labels is None:
            labels = tuple(str(idx) for idx in range(flat.size))
        for idx, value in enumerate(flat):
            label = labels[idx] if idx < len(labels) else str(idx)
            row[f"{prefix}_{label}"] = float(value)

    @staticmethod
    def _read_bool(data: dict, key: str) -> bool | None:
        if key not in data:
            return None
        try:
            return bool(np.asarray(data.get(key), dtype=np.bool_).reshape(()).item())
        except Exception:
            return None

    @staticmethod
    def _normalize_runtime_environment(raw_value: object) -> str:
        value = str(raw_value or "").strip().lower()
        if value in {"sim", "simulator", "simulation", "mujoco"}:
            return "sim"
        return "real"

    def _resolve_runtime_environment(self) -> str:
        mode_data = self._safe_read(self.mode_shm)
        simulator_running = self._read_bool(mode_data, SIMULATOR_MODE_KEY)
        if simulator_running is not None:
            return "sim" if simulator_running else "real"
        return self._normalize_runtime_environment(getattr(self.ctx.run_config, "runtime_environment", None))

    def _cleanup_previous_environment_logs(self) -> None:
        keep_paths = {self.csv_path, self.meta_path}
        pattern = f"walking_debug_{self.runtime_environment}_*"
        for path in sorted(self.log_dir.glob(pattern)):
            if path in keep_paths or path.suffix not in {".csv", ".json"}:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except Exception:
                logger.warning("[%s] failed to remove old walking log %s", self.ctx.name, path, exc_info=True)

    def _write_metadata(self) -> None:
        payload = {
            "session_id": self.session_id,
            "runtime_environment": self.runtime_environment,
            "csv_path": str(self.csv_path),
            "worker_name": self.ctx.name,
            "walking_policy_profile": self.profile_name,
            "walking_policy_backend": self.profile.backend,
            "walking_policy_path": getattr(self.ctx.run_config, "walking_policy_path", None),
            "walking_joint_profile_path": str(default_walking_joint_profile_path(self.profile_name)),
            "log_hz": float(self.hz),
            "mode_field_names": list(MODE_FIELD_NAMES),
            "leg_joint_labels": list(LEG_JOINT_LABELS),
            "waist_joint_labels": list(WAIST_JOINT_LABELS),
            "arm_joint_labels": list(ARM_JOINT_LABELS),
            "neck_joint_labels": list(NECK_JOINT_LABELS),
            "walking_leg_default_q": WALKING_LEG_DEFAULT_Q.tolist(),
            "walking_neutral_full_q": WALKING_NEUTRAL_FULL_Q.tolist(),
            "walking_neutral_arm_q": WALKING_ARM_NEUTRAL_Q.tolist(),
            "walking_neutral_neck_q": WALKING_NECK_NEUTRAL_Q.tolist(),
            "walking_action_scale": float(self.profile.action_scale),
            "walking_frame_stack": int(self.profile.frame_stack),
            "walking_single_obs_dim": int(self.profile.single_obs_dim),
            "walking_cycle_time": float(self.profile.cycle_time),
            "obs_stale_timeout": float(WALKING_OBS_STALE_TIMEOUT),
            "startup_base_roll_tol": float(WALKING_START_BASE_ROLL_TOL),
            "startup_base_pitch_tol": float(WALKING_START_BASE_PITCH_TOL),
            "startup_pose_check_indices": np.asarray(WALKING_START_POSE_CHECK_INDICES, dtype=np.int64).tolist(),
            "startup_phase_code": {
                "idle": 0,
                "waiting": 1,
                "blending": 2,
                "running": 3,
            },
        }
        self.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def on_start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_previous_environment_logs()
        self._fp = self.csv_path.open("w", newline="", encoding="utf-8", buffering=1)
        self._writer = csv.DictWriter(self._fp, fieldnames=self._fieldnames)
        self._writer.writeheader()
        self._write_metadata()
        logger.info(
            "[%s] walking debug logger started (%s): %s",
            self.ctx.name,
            self.runtime_environment,
            self.csv_path,
        )

    def _build_row(self, ev: EventSnapshot, tr: TransitionResult) -> dict[str, object]:
        mode_data = self._safe_read(self.mode_shm)
        cmd_data = self._safe_read(self.walking_cmd_shm)
        obs_data = self._safe_read(self.obs_shm)
        act_data = self._safe_read(self.act_shm)
        tau_data = self._safe_read(self.tau_shm)
        dbg_data = self._safe_read(self.walking_debug_shm)

        obs_waist = self._read_vector(obs_data, "obs_waist", len(WAIST_JOINT_LABELS))
        obs_leg = self._read_vector(obs_data, "obs_leg", len(LEG_JOINT_LABELS))
        obs_leg_dq = self._read_vector(obs_data, "obs_leg_dq", len(LEG_JOINT_LABELS))
        obs_arm = self._read_vector(obs_data, "obs_arm", 14)
        obs_neck = self._read_vector(obs_data, "obs_neck", 2)
        obs_imu_quat = self._read_vector(obs_data, "obs_imu_quat", 4)
        obs_imu_gyro = self._read_vector(obs_data, "obs_imu_gyro", 3)
        obs_imu_rpy = self._read_vector(obs_data, "obs_imu_rpy", 3)

        act_waist = self._read_vector(act_data, "act_waist", len(WAIST_JOINT_LABELS))
        act_leg = self._read_vector(act_data, "act_leg", len(LEG_JOINT_LABELS))
        act_arm = self._read_vector(act_data, "act_arm", 14)
        act_neck = self._read_vector(act_data, "act_neck", 2)

        tau_waist = self._read_vector(tau_data, "tau_est_waist", len(WAIST_JOINT_LABELS))
        tau_leg = self._read_vector(tau_data, "tau_est_leg", len(LEG_JOINT_LABELS))
        tau_arm = self._read_vector(tau_data, "tau_est_arm", 14)
        tau_neck = self._read_vector(tau_data, "tau_est_neck", 2)

        dbg_policy_action_raw = self._read_vector(dbg_data, "policy_action_raw", len(LEG_JOINT_LABELS))
        dbg_policy_action_applied = self._read_vector(dbg_data, "policy_action_applied", len(LEG_JOINT_LABELS))
        dbg_previous_action = self._read_vector(dbg_data, "previous_action", len(LEG_JOINT_LABELS))
        dbg_single_obs = self._read_vector(dbg_data, "policy_single_obs", WALKING_MAX_SINGLE_OBS_DIM)
        dbg_gyro_base = self._read_vector(dbg_data, "policy_gyro_base", 3)
        dbg_rpy_base = self._read_vector(dbg_data, "policy_rpy_base", 3)
        dbg_target_leg_q = self._read_vector(dbg_data, "policy_target_leg_q", len(LEG_JOINT_LABELS))
        dbg_active_kp_leg = self._read_vector(dbg_data, "active_kp_leg", len(LEG_JOINT_LABELS))
        dbg_active_kd_leg = self._read_vector(dbg_data, "active_kd_leg", len(LEG_JOINT_LABELS))
        dbg_active_kp_waist = self._read_vector(dbg_data, "active_kp_waist", len(WAIST_JOINT_LABELS))
        dbg_active_kd_waist = self._read_vector(dbg_data, "active_kd_waist", len(WAIST_JOINT_LABELS))
        dbg_active_kp_arm = self._read_vector(dbg_data, "active_kp_arm", len(ARM_JOINT_LABELS))
        dbg_active_kd_arm = self._read_vector(dbg_data, "active_kd_arm", len(ARM_JOINT_LABELS))
        dbg_active_kp_neck = self._read_vector(dbg_data, "active_kp_neck", len(NECK_JOINT_LABELS))
        dbg_active_kd_neck = self._read_vector(dbg_data, "active_kd_neck", len(NECK_JOINT_LABELS))

        leg_target_error = act_leg - obs_leg
        leg_default_error = obs_leg - WALKING_LEG_DEFAULT_Q
        waist_target_error = act_waist - obs_waist
        arm_target_error = act_arm - obs_arm
        arm_neutral_error = obs_arm - WALKING_ARM_NEUTRAL_Q
        neck_target_error = act_neck - obs_neck
        neck_neutral_error = obs_neck - WALKING_NECK_NEUTRAL_Q

        row: dict[str, object] = {
            "wall_time": time.time(),
            "mono_time": time.monotonic(),
            "worker_state": self.state.name,
            "transition_reason": tr.reason,
            "cmd_profile_code": self._read_scalar(cmd_data, "profile_code"),
            "cmd_vx": self._read_scalar(cmd_data, "vx"),
            "cmd_vy": self._read_scalar(cmd_data, "vy"),
            "cmd_dyaw": self._read_scalar(cmd_data, "dyaw"),
            "cmd_policy_enabled": self._read_scalar(cmd_data, "policy_enabled"),
            "obs_seq": self._read_scalar(obs_data, "obs_seq"),
            "dbg_seq": self._read_scalar(dbg_data, "seq"),
            "dbg_policy_tick": self._read_scalar(dbg_data, "policy_tick"),
            "dbg_startup_phase_code": self._read_scalar(dbg_data, "startup_phase_code"),
            "dbg_blend_alpha": self._read_scalar(dbg_data, "blend_alpha"),
            "dbg_pose_err_max": self._read_scalar(dbg_data, "pose_err_max"),
            "dbg_dq_max": self._read_scalar(dbg_data, "dq_max"),
            "dbg_gyro_max": self._read_scalar(dbg_data, "gyro_max"),
            "dbg_loop_dt": self._read_scalar(dbg_data, "loop_dt"),
            "dbg_policy_cmd_vx": self._read_scalar(dbg_data, "policy_cmd_vx"),
            "dbg_policy_cmd_vy": self._read_scalar(dbg_data, "policy_cmd_vy"),
            "dbg_policy_cmd_dyaw": self._read_scalar(dbg_data, "policy_cmd_dyaw"),
        }

        for name in MODE_FIELD_NAMES:
            try:
                row[f"mode_{name}"] = int(bool(np.asarray(mode_data.get(name, False)).reshape(()).item()))
            except Exception:
                row[f"mode_{name}"] = 0

        self._fill_vector(row, "obs_waist", obs_waist, WAIST_JOINT_LABELS)
        self._fill_vector(row, "obs_leg", obs_leg, LEG_JOINT_LABELS)
        self._fill_vector(row, "obs_leg_dq", obs_leg_dq, LEG_JOINT_LABELS)
        self._fill_vector(row, "obs_arm", obs_arm)
        self._fill_vector(row, "obs_neck", obs_neck)
        self._fill_vector(row, "obs_imu_quat", obs_imu_quat, ("w", "x", "y", "z"))
        self._fill_vector(row, "obs_imu_gyro", obs_imu_gyro, ("x", "y", "z"))
        self._fill_vector(row, "obs_imu_rpy", obs_imu_rpy, ("roll", "pitch", "yaw"))
        self._fill_vector(row, "act_waist", act_waist, WAIST_JOINT_LABELS)
        self._fill_vector(row, "act_leg", act_leg, LEG_JOINT_LABELS)
        self._fill_vector(row, "act_arm", act_arm)
        self._fill_vector(row, "act_neck", act_neck)
        self._fill_vector(row, "tau_est_waist", tau_waist, WAIST_JOINT_LABELS)
        self._fill_vector(row, "tau_est_leg", tau_leg, LEG_JOINT_LABELS)
        self._fill_vector(row, "tau_est_arm", tau_arm)
        self._fill_vector(row, "tau_est_neck", tau_neck)
        self._fill_vector(row, "dbg_policy_action_raw", dbg_policy_action_raw, LEG_JOINT_LABELS)
        self._fill_vector(row, "dbg_policy_action_applied", dbg_policy_action_applied, LEG_JOINT_LABELS)
        self._fill_vector(row, "dbg_previous_action", dbg_previous_action, LEG_JOINT_LABELS)
        self._fill_vector(row, "dbg_policy_single_obs", dbg_single_obs)
        self._fill_vector(row, "dbg_policy_gyro_base", dbg_gyro_base, ("x", "y", "z"))
        self._fill_vector(row, "dbg_policy_rpy_base", dbg_rpy_base, ("roll", "pitch", "yaw"))
        self._fill_vector(row, "dbg_policy_target_leg_q", dbg_target_leg_q, LEG_JOINT_LABELS)
        self._fill_vector(row, "dbg_active_kp_leg", dbg_active_kp_leg, LEG_JOINT_LABELS)
        self._fill_vector(row, "dbg_active_kd_leg", dbg_active_kd_leg, LEG_JOINT_LABELS)
        self._fill_vector(row, "dbg_active_kp_waist", dbg_active_kp_waist, WAIST_JOINT_LABELS)
        self._fill_vector(row, "dbg_active_kd_waist", dbg_active_kd_waist, WAIST_JOINT_LABELS)
        self._fill_vector(row, "dbg_active_kp_arm", dbg_active_kp_arm, ARM_JOINT_LABELS)
        self._fill_vector(row, "dbg_active_kd_arm", dbg_active_kd_arm, ARM_JOINT_LABELS)
        self._fill_vector(row, "dbg_active_kp_neck", dbg_active_kp_neck, NECK_JOINT_LABELS)
        self._fill_vector(row, "dbg_active_kd_neck", dbg_active_kd_neck, NECK_JOINT_LABELS)
        self._fill_vector(row, "derived_leg_target_error", leg_target_error, LEG_JOINT_LABELS)
        self._fill_vector(row, "derived_leg_default_error", leg_default_error, LEG_JOINT_LABELS)
        self._fill_vector(row, "derived_waist_target_error", waist_target_error, WAIST_JOINT_LABELS)
        self._fill_vector(row, "derived_arm_target_error", arm_target_error, ARM_JOINT_LABELS)
        self._fill_vector(row, "derived_arm_neutral_error", arm_neutral_error, ARM_JOINT_LABELS)
        self._fill_vector(row, "derived_neck_target_error", neck_target_error, NECK_JOINT_LABELS)
        self._fill_vector(row, "derived_neck_neutral_error", neck_neutral_error, NECK_JOINT_LABELS)
        return row

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        if self._writer is None:
            return
        self._writer.writerow(self._build_row(ev, tr))

    def on_stop(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            except Exception:
                pass
        if self._shared_memory:
            for key, mgr in self._shared_memory.items():
                try:
                    mgr.worker_close()
                except Exception:
                    logger.exception("[%s] failed to close shared memory %s", self.ctx.name, key)
        logger.info(
            "[%s] walking debug logger stopped (%s): %s",
            self.ctx.name,
            self.runtime_environment,
            self.csv_path,
        )
