from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.events import EventSnapshot
from ..core.project_paths import (
    CHECKPOINTS_ROOT,
    DATASETS_ROOT,
    INFERENCE_LOGS_ROOT,
    allocate_inference_log_session_id,
    inference_log_dir_name,
    release_inference_log_session_id,
    resolve_under_root,
)
from ..core.state_machine import ModeState, TransitionResult
from ..core.worker_base import DualRateWorker, WorkerContext
from ..core.math.trajectories import cubic_spline, finite_difference_derivative, quintic_spline_multi

try:
    from ..robot_control.base_contorl import ARM_INDICES, NECK_INDICES, WAIST_INDICES
except Exception:
    from ..sharedmemory.shm_schema import (
        ARM_INDICES as ARM_DIM,
        NECK_INDICES as NECK_DIM,
        WAIST_INDICES as WAIST_DIM,
    )

    ARM_INDICES = list(range(int(ARM_DIM)))
    NECK_INDICES = list(range(int(NECK_DIM)))
    WAIST_INDICES = list(range(int(WAIST_DIM)))

try:
    from action_lipo import ActionLiPo
except Exception:
    ActionLiPo = None

import logging_mp

logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)


HAND_DIM = 12
BASE_ACTION_DIM = HAND_DIM + len(ARM_INDICES) + len(NECK_INDICES)
FULL_WAIST_DIM = len(WAIST_INDICES)
FULL_ACTION_DIM = BASE_ACTION_DIM + FULL_WAIST_DIM

DEFAULT_INFERENCE_DATASET_FOLDER = "0304_실증과제_dataset_train/0304_padding"
DEFAULT_INFERENCE_PRETRAINED_REL = "ACT_0304_padding/checkpoints/200000/pretrained_model"
DEFAULT_INFERENCE_POLICY = "act"
DEFAULT_INFERENCE_ACT_N_ACTION_STEP = 100
DEFAULT_INFERENCE_DIFFUSION_N_ACTION_STEP = 8
DEFAULT_INFERENCE_PI_N_ACTION_STEP = 50

DEFAULT_RUN_ENTRY_INTERP_SEC = 2.0
DEFAULT_SAFETY_LOG_INTERVAL_SEC = 1.0
DEFAULT_LIPO_BLENDING_HORIZON = 10
DEFAULT_LIPO_DELAY_STEPS = 5

scaled_safety_margin = 0.8
DEFAULT_HAND_SAFE_VEL_LIMIT = 5.0 * scaled_safety_margin
DEFAULT_ARM_SAFE_VEL_LIMIT = 100.0 * scaled_safety_margin
DEFAULT_NECK_SAFE_VEL_LIMIT = 100.0 * scaled_safety_margin
DEFAULT_WAIST_SAFE_VEL_LIMIT = 100.0 * scaled_safety_margin
PI_POLICY_NAMES = frozenset({"pi0", "pi0.5", "pi05"})
OPENPI_OUTPUT_ACTION_DIM = BASE_ACTION_DIM


@dataclass
class ChunkBundle:
    start_step: int
    raw_chunk: np.ndarray
    ref_chunk: np.ndarray
    optimized_chunk: np.ndarray
    received_at: float


def _read_pretrained_feature_dims(pretrained_dir: Path) -> tuple[int | None, int | None, int | None]:
    config_path = pretrained_dir / "config.json"
    if not config_path.exists():
        return _read_openpi_feature_dims(pretrained_dir)

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return _read_openpi_feature_dims(pretrained_dir)

    state_shape = config.get("input_features", {}).get("observation.state", {}).get("shape")
    torque_shape = config.get("input_features", {}).get("observation.torque", {}).get("shape")
    action_shape = config.get("output_features", {}).get("action", {}).get("shape")
    state_dim = int(state_shape[0]) if state_shape else None
    torque_dim = int(torque_shape[0]) if torque_shape else None
    action_dim = int(action_shape[0]) if action_shape else None
    if state_dim is None and torque_dim is None and action_dim is None:
        return _read_openpi_feature_dims(pretrained_dir)
    return state_dim, torque_dim, action_dim


def _load_openpi_metadata(pretrained_dir: Path) -> dict[str, Any] | None:
    metadata_path = pretrained_dir / "metadata.pt"
    if not metadata_path.exists():
        return None

    try:
        import torch
    except Exception:
        return None

    try:
        payload = torch.load(metadata_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _find_openpi_norm_stats(pretrained_dir: Path) -> Path | None:
    for path in sorted(pretrained_dir.glob("assets/**/norm_stats.json")):
        try:
            if path.stat().st_size > 0:
                return path
        except Exception:
            continue
    return None


def _read_openpi_action_dim(pretrained_dir: Path) -> int | None:
    norm_stats_path = _find_openpi_norm_stats(pretrained_dir)
    if norm_stats_path is not None:
        try:
            payload = json.loads(norm_stats_path.read_text(encoding="utf-8"))
            stats = payload.get("norm_stats", payload)
            action_stats = stats.get("actions") or stats.get("action")
            means = action_stats.get("mean") if isinstance(action_stats, dict) else None
            if means:
                return int(len(means))
        except Exception:
            logger.exception("failed to read OpenPI norm_stats from %s", norm_stats_path)
    return None


def _read_openpi_action_horizon(pretrained_dir: Path) -> int | None:
    metadata = _load_openpi_metadata(pretrained_dir)
    if not metadata:
        return None

    cfg = metadata.get("config")
    if not isinstance(cfg, dict):
        return None
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        return None
    raw = model_cfg.get("action_horizon")
    try:
        return int(raw)
    except Exception:
        return None


def _read_openpi_feature_dims(pretrained_dir: Path) -> tuple[int | None, int | None, int | None]:
    action_dim = _read_openpi_action_dim(pretrained_dir)
    return None, None, action_dim


def _normalize_inference_policy_name(policy: str | None) -> str:
    txt = str(policy or "").strip().lower()
    if txt == "pi05":
        return "pi0.5"
    return txt


def _is_openpi_checkpoint_dir(path: Path) -> bool:
    return any((path / marker).exists() for marker in ("metadata.pt", "model.safetensors", "params"))


def _resolve_openpi_checkpoint_dir(path: Path) -> Path:
    for ancestor in (path, *path.parents):
        if _is_openpi_checkpoint_dir(ancestor):
            return ancestor
        if ancestor.parent == ancestor:
            break
    if not path.exists() or not path.is_dir():
        return path

    candidates: list[Path] = []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        if _is_openpi_checkpoint_dir(child):
            candidates.append(child)
            continue
        for grandchild in child.iterdir():
            if grandchild.is_dir() and _is_openpi_checkpoint_dir(grandchild):
                candidates.append(grandchild)

    if not candidates:
        return path

    def _sort_key(candidate: Path) -> tuple[int, float]:
        try:
            numeric = int(candidate.name)
        except ValueError:
            numeric = -1
        try:
            mtime = float(candidate.stat().st_mtime)
        except Exception:
            mtime = 0.0
        return numeric, mtime

    candidates.sort(key=_sort_key, reverse=True)
    return candidates[0]


def _resolve_n_action_step(ctx: WorkerContext, policy: str) -> int:
    raw = getattr(ctx.run_config, "inference_n_action_step", None)
    if raw is not None:
        return max(1, int(raw))

    normalized_policy = _normalize_inference_policy_name(policy)
    if normalized_policy in PI_POLICY_NAMES:
        pretrained_path = _resolve_pretrained_path(ctx)
        openpi_horizon = _read_openpi_action_horizon(pretrained_path)
        if openpi_horizon is not None:
            return max(1, openpi_horizon)
        return DEFAULT_INFERENCE_PI_N_ACTION_STEP

    if policy in {"diffusion", "diffusion_policy", "diffusion policy"}:
        return DEFAULT_INFERENCE_DIFFUSION_N_ACTION_STEP
    return DEFAULT_INFERENCE_ACT_N_ACTION_STEP


def _resolve_pretrained_path(ctx: WorkerContext) -> Path:
    raw = (
        getattr(ctx.run_config, "inference_pretrained_rel", None)
        or DEFAULT_INFERENCE_PRETRAINED_REL
    )
    resolved = resolve_under_root(CHECKPOINTS_ROOT, raw)
    policy = _normalize_inference_policy_name(getattr(ctx.run_config, "inference_policy", DEFAULT_INFERENCE_POLICY))
    if policy in PI_POLICY_NAMES:
        return _resolve_openpi_checkpoint_dir(resolved)
    return resolved


class InferenceOptimizeWorker(DualRateWorker):
    """Post-process inference_result_shm into smooth, safe commands for act_shm."""

    def __init__(self, ctx: WorkerContext, slow_hz: float = 30.0, fast_hz: float = 100.0) -> None:
        super().__init__(ctx, slow_hz=slow_hz, fast_hz=fast_hz)

        self._shared_memory = ctx.shared_memory or {}
        self.obs_shm = self._shared_memory.get("obs_shm")
        self.act_shm = self._shared_memory.get("act_shm")
        self.inference_result_shm = self._shared_memory.get("inference_result_shm")
        if self.act_shm is None:
            raise RuntimeError("act_shm is required for inference optimizer")
        if self.inference_result_shm is None:
            raise RuntimeError("inference_result_shm is required for inference optimizer")

        self.inference_policy = _normalize_inference_policy_name(
            getattr(self.ctx.run_config, "inference_policy", DEFAULT_INFERENCE_POLICY) or DEFAULT_INFERENCE_POLICY
        )
        self.dataset_path = resolve_under_root(
            DATASETS_ROOT,
            getattr(self.ctx.run_config, "inference_dataset_folder", None) or DEFAULT_INFERENCE_DATASET_FOLDER,
        )
        self.pretrained_path = _resolve_pretrained_path(ctx)
        _, _, pretrained_action_dim = _read_pretrained_feature_dims(self.pretrained_path)
        default_action_dim = OPENPI_OUTPUT_ACTION_DIM if self.inference_policy in PI_POLICY_NAMES else FULL_ACTION_DIM
        self.action_dim = int(pretrained_action_dim or default_action_dim)
        if self.action_dim not in (BASE_ACTION_DIM, FULL_ACTION_DIM):
            raise RuntimeError(
                "Unsupported action dim for optimize worker: "
                f"{self.action_dim} (supported: {BASE_ACTION_DIM}, {FULL_ACTION_DIM})"
            )

        self.hand_len = HAND_DIM
        self.arm_len = len(ARM_INDICES)
        self.neck_len = len(NECK_INDICES)
        self.waist_len = FULL_WAIST_DIM if self.action_dim == FULL_ACTION_DIM else 0
        self.use_waist = self.waist_len > 0
        self.zero_waist_action = np.zeros((FULL_WAIST_DIM,), dtype=np.float32)

        self.enable_optimizer = True
        self.enable_spline = True
        self.spline = "quintic"
        self.n_action_step = _resolve_n_action_step(ctx, self.inference_policy)
        self.blending_horizon = min(DEFAULT_LIPO_BLENDING_HORIZON, max(1, self.n_action_step - 1))
        self.lipo_delay_steps = min(DEFAULT_LIPO_DELAY_STEPS, max(0, self.blending_horizon - 1))

        self.control_dt = 1.0 / self.fast_hz
        self.inference_dt = 1.0 / self.slow_hz
        self.run_entry_duration = max(
            self.control_dt,
            float(
                getattr(
                    self.ctx.run_config,
                    "inference_run_entry_duration_sec",
                    DEFAULT_RUN_ENTRY_INTERP_SEC,
                )
            ),
        )
        self.safety_log_interval = max(
            0.0,
            float(
                getattr(
                    self.ctx.run_config,
                    "inference_safety_log_interval_sec",
                    DEFAULT_SAFETY_LOG_INTERVAL_SEC,
                )
            ),
        )

        self.group_velocity_limits = {
            "hand": max(
                1e-6,
                float(getattr(self.ctx.run_config, "inference_safe_hand_vel_limit", DEFAULT_HAND_SAFE_VEL_LIMIT)),
            ),
            "arm": max(
                1e-6,
                float(getattr(self.ctx.run_config, "inference_safe_arm_vel_limit", DEFAULT_ARM_SAFE_VEL_LIMIT)),
            ),
            "neck": max(
                1e-6,
                float(getattr(self.ctx.run_config, "inference_safe_neck_vel_limit", DEFAULT_NECK_SAFE_VEL_LIMIT)),
            ),
            "waist": max(
                1e-6,
                float(getattr(self.ctx.run_config, "inference_safe_waist_vel_limit", DEFAULT_WAIST_SAFE_VEL_LIMIT)),
            ),
        }
        self.velocity_limit = self._build_joint_velocity_limit_vector()
        self.fast_delta_limit = (self.velocity_limit * np.float32(self.control_dt)).astype(np.float32, copy=False)
        self.chunk_delta_limit = (self.velocity_limit * np.float32(self.inference_dt)).astype(np.float32, copy=False)

        self._chunk_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._chunk_bundles: dict[int, ChunkBundle] = {}
        self._chunk_history_limit = 64
        self.current_bundle: ChunkBundle | None = None
        self.next_bundle: ChunkBundle | None = None
        self.action_memory: deque[np.ndarray] = deque(maxlen=2)
        self._last_seen_result_seq = self._read_inference_seq()
        self._lipo = None
        self._lipo_chunk_size = 0
        self._lipo_blend_horizon = 0
        self._lipo_delay = 0
        self._lipo_warned_unavailable = False

        self._run_mode_active = False
        self.step_counter = 0
        self.segment_elapsed = 0.0
        self.pre_pos: np.ndarray | None = None
        self._active_chunk_start_step: int | None = None
        self._last_logged_sample_token: tuple[int, int] | None = None
        self._last_logged_chunk_apply_step: int | None = None
        self._last_publish_safety_log_at = 0.0
        self._last_chunk_safety_log_at = 0.0

        self.run_entry_interp_active = False
        self.run_entry_elapsed = 0.0
        self.run_entry_anchor_pos: np.ndarray | None = None

        self.log_root_dir = INFERENCE_LOGS_ROOT
        self.log_session_dir: Path | None = None
        self.log_started_at = time.monotonic()
        self.log_session_id = time.strftime("%Y%m%d_%H%M%S")
        self._last_log_export: dict[str, object] | None = None
        self.raw_action_log: list[tuple[float, np.ndarray]] = []
        self.ref_action_log: list[tuple[float, np.ndarray]] = []
        self.optimized_action_log: list[tuple[float, np.ndarray]] = []
        self.publish_action_log: list[tuple[float, np.ndarray]] = []
        self.chunk_apply_log: list[tuple[float, int]] = []

    def on_start(self) -> None:
        logger.info(
            "[%s] start (slow=%.1fHz fast=%.1fHz policy=%s dataset=%s checkpoint=%s "
            "action_dim=%s n_action_step=%s blend=%s delay=%s)",
            self.ctx.name,
            self.slow_hz,
            self.fast_hz,
            self.inference_policy,
            self.dataset_path,
            self.pretrained_path,
            self.action_dim,
            self.n_action_step,
            self.blending_horizon,
            self.lipo_delay_steps,
        )

    def do_slow(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        if self.state != ModeState.RUN:
            return

        try:
            self._poll_inference_result()
        except Exception:
            logger.exception("[%s] failed to process inference_result_shm", self.ctx.name)

    def do_fast(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        if self.state != ModeState.RUN:
            if self._run_mode_active:
                self._run_mode_active = False
                self._finalize_log_session(phase="run stop")
                self.reset_runtime_buffers(start_new_session=False)
            return

        if not self._run_mode_active:
            self._run_mode_active = True
            self.reset_runtime_buffers(start_new_session=True)
            logger.info("[%s] entered RUN, waiting for inference_result_shm", self.ctx.name)

        self.segment_elapsed += self.control_dt
        while self.segment_elapsed >= self.inference_dt:
            self.segment_elapsed -= self.inference_dt
            self.step_counter += 1

        bundle, sample_idx, target_pos, chunk_applied = self._compute_active_target()
        if chunk_applied and bundle is not None:
            self._append_chunk_apply_log(
                step_index=bundle.start_step,
                log_time=max(0.0, self._log_time() - self.segment_elapsed),
            )
        if bundle is not None:
            self._append_chunk_sample_logs(
                bundle,
                sample_idx=sample_idx,
                log_time=max(0.0, self._log_time() - self.segment_elapsed),
            )

        if bundle is not None and not self.run_entry_interp_active and self.run_entry_anchor_pos is None:
            self._arm_run_entry_interpolation()

        if self.run_entry_interp_active:
            target_pos = self._blend_run_entry_target(target_pos)

        if target_pos is None:
            target_pos = self._read_current_act_position()

        target_pos = self._apply_safety(target_pos)
        self._publish_pos(target_pos)
        self._prune_chunk_history()

    def reset_runtime_buffers(self, *, start_new_session: bool) -> None:
        self.step_counter = 0
        self.segment_elapsed = 0.0
        self.pre_pos = None
        self._active_chunk_start_step = None
        self._last_logged_sample_token = None
        self._last_logged_chunk_apply_step = None
        self._last_publish_safety_log_at = 0.0
        self._last_chunk_safety_log_at = 0.0
        self.run_entry_interp_active = False
        self.run_entry_elapsed = 0.0
        self.run_entry_anchor_pos = None
        self.action_memory.clear()
        with self._chunk_lock:
            self._chunk_bundles.clear()
            self.current_bundle = None
            self.next_bundle = None
        self._last_seen_result_seq = self._read_inference_seq()

        if start_new_session:
            self.log_started_at = time.monotonic()
            self.log_session_id = allocate_inference_log_session_id(self.inference_policy)
        self.log_session_dir = None
        self._last_log_export = None
        with self._log_lock:
            self.raw_action_log.clear()
            self.ref_action_log.clear()
            self.optimized_action_log.clear()
            self.publish_action_log.clear()
            self.chunk_apply_log.clear()

    def _read_inference_seq(self) -> int:
        try:
            data = self.inference_result_shm.read_data()
        except Exception:
            return 0
        return int(round(float(data.get("seq", 0.0))))

    def _poll_inference_result(self) -> None:
        result = self.inference_result_shm.read_data()
        seq = int(round(float(result["seq"])))
        valid = bool(int(round(float(result["valid"]))))
        if valid and seq < self._last_seen_result_seq:
            logger.info(
                "[%s] detected inference_result_shm seq reset (%s -> %s)",
                self.ctx.name,
                self._last_seen_result_seq,
                seq,
            )
            self._last_seen_result_seq = 0

        if not valid or seq <= self._last_seen_result_seq:
            return

        request_step = int(round(float(result["request_step"])))
        rows = int(round(float(result["n_action_steps"])))
        cols = int(round(float(result["action_dim"])))
        if rows <= 0 or cols <= 0:
            self._last_seen_result_seq = seq
            return
        if cols != self.action_dim:
            raise RuntimeError(
                f"inference_result_shm action dim mismatch: expected {self.action_dim}, got {cols}"
            )

        payload = np.asarray(result["action_chunk"], dtype=np.float32)
        raw_chunk = payload[:rows, :cols].copy()
        if raw_chunk.ndim != 2 or raw_chunk.shape[1] != self.action_dim:
            raise RuntimeError(f"Invalid chunk shape from inference_result_shm: {raw_chunk.shape}")

        self.n_action_step = int(raw_chunk.shape[0])
        self.blending_horizon = min(DEFAULT_LIPO_BLENDING_HORIZON, max(1, self.n_action_step - 1))
        self.lipo_delay_steps = min(DEFAULT_LIPO_DELAY_STEPS, max(0, self.blending_horizon - 1))

        if not self._should_accept_bundle_start(request_step):
            self._last_seen_result_seq = seq
            return

        prev_raw = self.action_memory[-1] if self.action_memory else None
        self.action_memory.append(raw_chunk.copy())

        ref_chunk = raw_chunk.copy()
        optimized_chunk = raw_chunk.copy()
        if self.enable_optimizer and prev_raw is not None and raw_chunk.shape[0] > 1:
            optimized_chunk, ref_chunk = self._compute_lipo(raw_chunk, prev_raw)

        optimized_chunk = self._apply_chunk_step_safety(optimized_chunk)

        bundle = ChunkBundle(
            start_step=request_step,
            raw_chunk=raw_chunk,
            ref_chunk=ref_chunk,
            optimized_chunk=optimized_chunk,
            received_at=time.monotonic(),
        )
        self._stage_bundle_for_playback(bundle)

        self._last_seen_result_seq = seq

    def _compute_lipo(self, raw_chunk: np.ndarray, prev_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if ActionLiPo is None:
            if not self._lipo_warned_unavailable:
                logger.warning("[%s] action_lipo is unavailable, using raw chunks only", self.ctx.name)
                self._lipo_warned_unavailable = True
            return raw_chunk.copy(), raw_chunk.copy()

        self._ensure_lipo(chunk_size=raw_chunk.shape[0])
        if self._lipo is None:
            return raw_chunk.copy(), raw_chunk.copy()

        solved, ref = self._lipo.solve(
            raw_chunk,
            prev_raw,
            len_past_actions=self._lipo_blend_horizon,
        )
        if solved is None:
            logger.warning("[%s] LiPo solve failed, falling back to raw chunk", self.ctx.name)
            return raw_chunk.copy(), raw_chunk.copy()

        solved_arr = np.asarray(solved, dtype=np.float32)
        ref_arr = np.asarray(ref if ref is not None else raw_chunk, dtype=np.float32)
        if solved_arr.shape != raw_chunk.shape:
            raise RuntimeError(f"LiPo solved chunk shape mismatch: {solved_arr.shape} != {raw_chunk.shape}")
        if ref_arr.shape != raw_chunk.shape:
            ref_arr = raw_chunk.copy()
        return solved_arr.copy(), ref_arr.copy()

    def _ensure_lipo(self, *, chunk_size: int) -> None:
        blend_horizon = min(DEFAULT_LIPO_BLENDING_HORIZON, max(1, chunk_size - 1))
        delay_steps = min(DEFAULT_LIPO_DELAY_STEPS, max(0, blend_horizon - 1))
        if (
            self._lipo is not None
            and self._lipo_chunk_size == chunk_size
            and self._lipo_blend_horizon == blend_horizon
            and self._lipo_delay == delay_steps
        ):
            return

        if ActionLiPo is None:
            self._lipo = None
            return

        self._lipo = ActionLiPo(
            chunk_size=chunk_size,
            blending_horizon=blend_horizon,
            action_dim=self.action_dim,
            len_time_delay=delay_steps,
            dt=self.inference_dt,
        )
        self._lipo_chunk_size = chunk_size
        self._lipo_blend_horizon = blend_horizon
        self._lipo_delay = delay_steps

    @staticmethod
    def _bundle_blending_horizon(bundle: ChunkBundle) -> int:
        chunk_len = int(bundle.optimized_chunk.shape[0])
        return min(DEFAULT_LIPO_BLENDING_HORIZON, max(1, chunk_len - 1))

    @classmethod
    def _bundle_request_stride(cls, bundle: ChunkBundle) -> int:
        chunk_len = int(bundle.optimized_chunk.shape[0])
        return max(1, chunk_len - cls._bundle_blending_horizon(bundle))

    def _minimum_next_start_step_locked(self) -> int | None:
        if self.current_bundle is None:
            return None
        return int(self.current_bundle.start_step + self._bundle_request_stride(self.current_bundle))

    def _should_accept_bundle_start(self, request_step: int) -> bool:
        with self._chunk_lock:
            if self.current_bundle is None:
                return True
            min_next_start = self._minimum_next_start_step_locked()
            if min_next_start is None or request_step < min_next_start:
                return False
            if self.next_bundle is None:
                return True
            return request_step < self.next_bundle.start_step

    def _stage_bundle_for_playback(self, bundle: ChunkBundle) -> None:
        with self._chunk_lock:
            self._chunk_bundles[bundle.start_step] = bundle
            while len(self._chunk_bundles) > self._chunk_history_limit:
                oldest = min(self._chunk_bundles)
                self._chunk_bundles.pop(oldest, None)

            if self.current_bundle is None:
                self.current_bundle = bundle
                self.next_bundle = None
                return

            min_next_start = self._minimum_next_start_step_locked()
            if min_next_start is None or bundle.start_step < min_next_start:
                return

            if self.next_bundle is None or bundle.start_step < self.next_bundle.start_step:
                self.next_bundle = bundle

    def _advance_active_bundle(self) -> tuple[ChunkBundle | None, bool]:
        with self._chunk_lock:
            if self.current_bundle is None:
                return None, False

            chunk_applied = False
            if self.next_bundle is not None and self.step_counter >= self.next_bundle.start_step:
                self.current_bundle = self.next_bundle
                self.next_bundle = None
                chunk_applied = True

            return self.current_bundle, chunk_applied

    def _compute_active_target(self) -> tuple[ChunkBundle | None, int, np.ndarray | None, bool]:
        bundle, chunk_applied = self._advance_active_bundle()
        if bundle is None:
            return None, 0, None, False

        chunk = bundle.optimized_chunk
        if chunk.ndim != 2 or chunk.shape[0] == 0:
            return None, 0, None, chunk_applied

        if self.step_counter < bundle.start_step:
            return bundle, 0, chunk[0].copy(), chunk_applied

        local_idx = max(0, self.step_counter - bundle.start_step)
        if local_idx >= chunk.shape[0] - 1:
            sample_idx = chunk.shape[0] - 1
            pos = chunk[sample_idx].copy()
        else:
            sample_idx = local_idx
            pos = self.compute_spline_interpolation(chunk, local_idx, self.segment_elapsed)
        return bundle, sample_idx, pos, chunk_applied

    def _prune_chunk_history(self) -> None:
        min_keep_step = max(-1, self.step_counter - max(self.n_action_step * 2, 32))
        with self._chunk_lock:
            stale = [start for start in self._chunk_bundles if start < min_keep_step]
            for start in stale:
                self._chunk_bundles.pop(start, None)

    def _chunk_transition_anchor(self) -> np.ndarray | None:
        bundle = self.current_bundle
        if bundle is None:
            return None

        chunk = np.asarray(bundle.optimized_chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != self.action_dim or chunk.shape[0] == 0:
            return None

        anchor_idx = int(
            np.clip(
                chunk.shape[0] - self._bundle_blending_horizon(bundle),
                0,
                chunk.shape[0] - 1,
            )
        )
        return chunk[anchor_idx].copy()

    def _arm_run_entry_interpolation(self) -> None:
        self.run_entry_interp_active = True
        self.run_entry_elapsed = 0.0
        self.run_entry_anchor_pos = self._read_current_act_position()
        logger.info(
            "[%s] armed run-entry interpolation (duration=%.2fs)",
            self.ctx.name,
            self.run_entry_duration,
        )

    def _blend_run_entry_target(self, target_pos: np.ndarray | None) -> np.ndarray:
        anchor = self.run_entry_anchor_pos
        if anchor is None:
            self.run_entry_interp_active = False
            return target_pos if target_pos is not None else self._read_current_act_position()

        self.run_entry_elapsed = min(self.run_entry_elapsed + self.control_dt, self.run_entry_duration)
        ratio = float(np.clip(self.run_entry_elapsed / self.run_entry_duration, 0.0, 1.0))
        alpha = ratio * ratio * (3.0 - 2.0 * ratio)
        target = anchor if target_pos is None else np.asarray(target_pos, dtype=np.float32).reshape(-1)
        if target.size != self.action_dim:
            target = anchor

        pos = (1.0 - alpha) * anchor + alpha * target
        if self.run_entry_elapsed >= self.run_entry_duration:
            self.run_entry_interp_active = False
            self.run_entry_anchor_pos = target.copy()
            logger.info("[%s] completed run-entry interpolation", self.ctx.name)
        return pos.astype(np.float32, copy=False)

    def compute_spline_interpolation(self, optimized_chunk: np.ndarray, local_idx: int, dt: float) -> np.ndarray:
        if optimized_chunk.ndim != 2:
            raise ValueError(f"optimized_chunk must be 2D, got {optimized_chunk.shape}")
        if optimized_chunk.shape[0] < 2:
            return optimized_chunk[-1].copy()

        idx0 = int(np.clip(local_idx, 0, optimized_chunk.shape[0] - 2))
        idx1 = idx0 + 1
        t = float(np.clip(dt, 0.0, self.inference_dt))

        vel_chunk = finite_difference_derivative(optimized_chunk, 1, self.inference_dt, 0)
        acc_chunk = finite_difference_derivative(vel_chunk, 1, self.inference_dt, 0)

        x0_pos = optimized_chunk[idx0]
        x0_vel = vel_chunk[idx0]
        x0_acc = acc_chunk[idx0]
        x1_pos = optimized_chunk[idx1]
        x1_vel = vel_chunk[idx1]
        x1_acc = acc_chunk[idx1]

        if self.spline == "cubic":
            return cubic_spline(t, 0.0, self.inference_dt, x0_pos, x1_pos, x0_vel, x1_vel)
        if self.spline == "quintic":
            return quintic_spline_multi(
                t,
                0.0,
                self.inference_dt,
                x0_pos,
                x0_vel,
                x0_acc,
                x1_pos,
                x1_vel,
                x1_acc,
            )[0]
        raise ValueError(f"Unsupported spline type: {self.spline}")

    def _compose_obs_position(self, obs_data: dict) -> np.ndarray:
        parts = [
            np.asarray(obs_data["obs_hand"], dtype=np.float32).reshape(-1),
            np.asarray(obs_data["obs_arm"], dtype=np.float32).reshape(-1),
            np.asarray(obs_data["obs_neck"], dtype=np.float32).reshape(-1),
        ]
        if self.use_waist:
            parts.append(np.asarray(obs_data["obs_waist"], dtype=np.float32).reshape(-1)[: self.waist_len])
        pos = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        if pos.size != self.action_dim:
            raise RuntimeError(f"obs position dim mismatch: {pos.size} != {self.action_dim}")
        return pos

    def _compose_act_position(self, act_data: dict) -> np.ndarray:
        parts = [
            np.asarray(act_data["act_hand"], dtype=np.float32).reshape(-1),
            np.asarray(act_data["act_arm"], dtype=np.float32).reshape(-1),
            np.asarray(act_data["act_neck"], dtype=np.float32).reshape(-1),
        ]
        if self.use_waist:
            parts.append(np.asarray(act_data["act_waist"], dtype=np.float32).reshape(-1)[: self.waist_len])
        pos = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        if pos.size != self.action_dim:
            raise RuntimeError(f"act position dim mismatch: {pos.size} != {self.action_dim}")
        return pos

    def _read_current_obs_position(self) -> np.ndarray:
        if self.obs_shm is None:
            return np.zeros((self.action_dim,), dtype=np.float32)
        try:
            return self._compose_obs_position(self.obs_shm.read_data())
        except Exception:
            logger.exception("[%s] failed to read obs_shm, falling back to zeros", self.ctx.name)
            return np.zeros((self.action_dim,), dtype=np.float32)

    def _read_current_act_position(self) -> np.ndarray:
        try:
            return self._compose_act_position(self.act_shm.read_data())
        except Exception:
            logger.exception("[%s] failed to read act_shm, falling back to obs_shm", self.ctx.name)
            return self._read_current_obs_position()

    def _clamp_action_step(
        self,
        target: np.ndarray,
        anchor: np.ndarray,
        delta_limit: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        target_arr = np.asarray(target, dtype=np.float32).reshape(-1)
        anchor_arr = np.asarray(anchor, dtype=np.float32).reshape(-1)
        limit_arr = np.asarray(delta_limit, dtype=np.float32).reshape(-1)

        raw_delta = target_arr - anchor_arr
        clipped_delta = np.clip(raw_delta, -limit_arr, limit_arr)
        clamped_pos = (anchor_arr + clipped_delta).astype(np.float32, copy=False)
        changed_mask = np.abs(raw_delta) > (limit_arr + 1e-8)
        max_raw_delta = float(np.max(np.abs(raw_delta))) if raw_delta.size else 0.0
        max_excess_delta = float(np.max(np.maximum(np.abs(raw_delta) - limit_arr, 0.0))) if raw_delta.size else 0.0
        return clamped_pos, changed_mask, max_raw_delta, max_excess_delta

    def _log_safety_clamp(
        self,
        *,
        scope: str,
        num_clamped: int,
        max_raw_delta: float,
        max_excess_delta: float,
        max_limit_delta: float,
    ) -> None:
        if num_clamped <= 0:
            return

        now = time.monotonic()
        if scope == "publish":
            last_logged_at = self._last_publish_safety_log_at
        else:
            last_logged_at = self._last_chunk_safety_log_at

        if self.safety_log_interval > 0.0 and (now - last_logged_at) < self.safety_log_interval:
            return

        if scope == "publish":
            self._last_publish_safety_log_at = now
        else:
            self._last_chunk_safety_log_at = now

        logger.warning(
            "[%s] %s safety clamp applied (joints=%s raw_max_delta=%.4f rad "
            "max_excess=%.4f rad limit_max=%.4f rad)",
            self.ctx.name,
            scope,
            num_clamped,
            max_raw_delta,
            max_excess_delta,
            max_limit_delta,
        )

    def _apply_safety(self, pos: np.ndarray) -> np.ndarray:
        safe_pos = np.asarray(pos, dtype=np.float32).reshape(-1)
        if self.pre_pos is None:
            self.pre_pos = safe_pos.copy()
            return safe_pos

        clamped_pos, changed_mask, max_raw_delta, max_excess_delta = self._clamp_action_step(
            safe_pos,
            self.pre_pos,
            self.fast_delta_limit,
        )
        if np.any(changed_mask):
            self._log_safety_clamp(
                scope="publish",
                num_clamped=int(np.count_nonzero(changed_mask)),
                max_raw_delta=max_raw_delta,
                max_excess_delta=max_excess_delta,
                max_limit_delta=float(np.max(self.fast_delta_limit)),
            )

        self.pre_pos = clamped_pos.copy()
        return clamped_pos

    def _apply_chunk_step_safety(self, chunk: np.ndarray) -> np.ndarray:
        safe_chunk = np.asarray(chunk, dtype=np.float32).copy()
        if safe_chunk.ndim != 2 or safe_chunk.shape[1] != self.action_dim:
            raise ValueError(f"chunk dim mismatch: {safe_chunk.shape}")

        anchor = self._chunk_transition_anchor()
        start_idx = 0
        if anchor is None:
            if safe_chunk.shape[0] == 0:
                return safe_chunk
            anchor = safe_chunk[0].copy()
            start_idx = 1
        total_clamped = 0
        max_raw_delta = 0.0
        max_excess_delta = 0.0

        for sample_idx in range(start_idx, safe_chunk.shape[0]):
            clamped_pos, changed_mask, raw_max, excess_max = self._clamp_action_step(
                safe_chunk[sample_idx],
                anchor,
                self.chunk_delta_limit,
            )
            safe_chunk[sample_idx] = clamped_pos
            anchor = clamped_pos
            total_clamped += int(np.count_nonzero(changed_mask))
            max_raw_delta = max(max_raw_delta, raw_max)
            max_excess_delta = max(max_excess_delta, excess_max)

        if total_clamped > 0:
            self._log_safety_clamp(
                scope="chunk",
                num_clamped=total_clamped,
                max_raw_delta=max_raw_delta,
                max_excess_delta=max_excess_delta,
                max_limit_delta=float(np.max(self.chunk_delta_limit)),
            )
        return safe_chunk

    def _split_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != self.action_dim:
            raise ValueError(f"action dim mismatch: {arr.size} != {self.action_dim}")

        i = 0
        act_hand = arr[i : i + self.hand_len].copy()
        i += self.hand_len
        act_arm = arr[i : i + self.arm_len].copy()
        i += self.arm_len
        act_neck = arr[i : i + self.neck_len].copy()
        i += self.neck_len
        act_waist = self.zero_waist_action.copy()
        if self.use_waist:
            act_waist[: self.waist_len] = arr[i : i + self.waist_len]
        return act_hand, act_arm, act_neck, act_waist

    def _publish_pos(self, pos: np.ndarray) -> None:
        act_hand, act_arm, act_neck, act_waist = self._split_action(pos)
        self.act_shm.write_data(
            act_arm=act_arm,
            act_neck=act_neck,
            act_hand=act_hand,
            act_waist=act_waist,
        )
        self._append_publish_action_log(pos)

    def _build_joint_velocity_limit_vector(self) -> np.ndarray:
        parts = [
            np.full((self.hand_len,), self.group_velocity_limits["hand"], dtype=np.float32),
            np.full((self.arm_len,), self.group_velocity_limits["arm"], dtype=np.float32),
            np.full((self.neck_len,), self.group_velocity_limits["neck"], dtype=np.float32),
        ]
        if self.use_waist:
            parts.append(np.full((self.waist_len,), self.group_velocity_limits["waist"], dtype=np.float32))
        limit = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        if limit.size != self.action_dim:
            raise RuntimeError(f"velocity limit dim mismatch: {limit.size} != {self.action_dim}")
        return limit

    def _log_time(self) -> float:
        return time.monotonic() - self.log_started_at

    def _append_publish_action_log(self, action: np.ndarray) -> None:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != self.action_dim:
            return
        with self._log_lock:
            self.publish_action_log.append((self._log_time(), arr.copy()))

    def _append_chunk_apply_log(self, *, step_index: int, log_time: float) -> None:
        token = int(step_index)
        if token == self._last_logged_chunk_apply_step:
            return
        self._last_logged_chunk_apply_step = token
        self._active_chunk_start_step = token
        with self._log_lock:
            self.chunk_apply_log.append((float(log_time), token))

    def _append_chunk_sample_logs(self, bundle: ChunkBundle, *, sample_idx: int, log_time: float) -> None:
        sample_idx = int(np.clip(sample_idx, 0, bundle.optimized_chunk.shape[0] - 1))
        token = (bundle.start_step, sample_idx)
        if token == self._last_logged_sample_token:
            return
        self._last_logged_sample_token = token

        with self._log_lock:
            self.raw_action_log.append((float(log_time), bundle.raw_chunk[sample_idx].copy()))
            self.ref_action_log.append((float(log_time), bundle.ref_chunk[sample_idx].copy()))
            self.optimized_action_log.append((float(log_time), bundle.optimized_chunk[sample_idx].copy()))

    def _snapshot_logs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self._log_lock:
            raw_log = [(t, arr.copy()) for t, arr in self.raw_action_log]
            ref_log = [(t, arr.copy()) for t, arr in self.ref_action_log]
            optimized_log = [(t, arr.copy()) for t, arr in self.optimized_action_log]
            publish_log = [(t, arr.copy()) for t, arr in self.publish_action_log]
            chunk_apply_log = [(t, step) for t, step in self.chunk_apply_log]

        def _pack(log: list[tuple[float, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
            if not log:
                return (
                    np.empty((0,), dtype=np.float32),
                    np.empty((0, self.action_dim), dtype=np.float32),
                )
            return (
                np.asarray([t for t, _ in log], dtype=np.float32),
                np.stack([arr for _, arr in log], axis=0).astype(np.float32, copy=False),
            )

        raw_t, raw_v = _pack(raw_log)
        ref_t, ref_v = _pack(ref_log)
        optimized_t, optimized_v = _pack(optimized_log)
        publish_t, publish_v = _pack(publish_log)
        if chunk_apply_log:
            chunk_apply_t = np.asarray([t for t, _ in chunk_apply_log], dtype=np.float32)
            chunk_apply_step = np.asarray([step for _, step in chunk_apply_log], dtype=np.int32)
        else:
            chunk_apply_t = np.empty((0,), dtype=np.float32)
            chunk_apply_step = np.empty((0,), dtype=np.int32)

        return (
            raw_t,
            raw_v,
            ref_t,
            ref_v,
            optimized_t,
            optimized_v,
            publish_t,
            publish_v,
            chunk_apply_t,
            chunk_apply_step,
        )

    def _action_group_specs(self) -> list[tuple[str, int, int]]:
        specs = [
            ("hand", 0, self.hand_len),
            ("arm", self.hand_len, self.hand_len + self.arm_len),
            ("neck", self.hand_len + self.arm_len, self.hand_len + self.arm_len + self.neck_len),
        ]
        if self.use_waist:
            specs.append(("waist", self.hand_len + self.arm_len + self.neck_len, self.action_dim))
        return specs

    def _current_log_session_dir(self) -> Path:
        return self.log_root_dir / inference_log_dir_name(self.inference_policy, self.log_session_id)

    def _has_pending_logs(self) -> bool:
        with self._log_lock:
            return any(
                (
                    self.raw_action_log,
                    self.ref_action_log,
                    self.optimized_action_log,
                    self.publish_action_log,
                    self.chunk_apply_log,
                )
            )

    def _log_stop_summary(self, export_meta: dict[str, object], *, phase: str) -> None:
        session_dir = export_meta.get("session_dir")
        saved_files = export_meta.get("saved_files")
        files_str = ",".join(saved_files) if isinstance(saved_files, list) and saved_files else "none"
        logger.info(
            "[%s] %s (session=%s elapsed=%.2fs export=%s dir=%s files=%s samples=%s)",
            self.ctx.name,
            phase,
            self.log_session_id,
            self._log_time(),
            export_meta.get("status", "unknown"),
            str(session_dir) if session_dir is not None else "n/a",
            files_str,
            export_meta.get("sample_summary", "unavailable"),
        )

    def _save_action_logs(self) -> dict[str, object]:
        (
            raw_t,
            raw_v,
            ref_t,
            ref_v,
            optimized_t,
            optimized_v,
            publish_t,
            publish_v,
            chunk_apply_t,
            chunk_apply_step,
        ) = self._snapshot_logs()
        sample_summary = (
            f"raw={raw_t.size}, ref={ref_t.size}, optimized={optimized_t.size}, "
            f"publish={publish_t.size}, chunk_marks={chunk_apply_t.size}"
        )
        if raw_t.size == 0 and publish_t.size == 0:
            self.log_session_dir = None
            self._last_log_export = {
                "session_id": self.log_session_id,
                "status": "no samples; log export skipped",
                "session_dir": None,
                "saved_files": [],
                "sample_summary": sample_summary,
            }
            return self._last_log_export

        self.log_session_dir = self._current_log_session_dir()
        self.log_session_dir.mkdir(parents=True, exist_ok=True)
        npz_path = self.log_session_dir / "actions.npz"
        plot_path = self.log_session_dir / "actions.png"

        np.savez_compressed(
            npz_path,
            raw_time=raw_t,
            raw_action=raw_v,
            ref_time=ref_t,
            ref_action=ref_v,
            optimized_time=optimized_t,
            optimized_action=optimized_v,
            chunk_time=optimized_t,
            chunk_action=optimized_v,
            publish_time=publish_t,
            publish_action=publish_v,
            chunk_apply_time=chunk_apply_t,
            chunk_apply_step=chunk_apply_step,
            action_dim=np.asarray([self.action_dim], dtype=np.int32),
            n_action_step=np.asarray([self.n_action_step], dtype=np.int32),
            blending_horizon=np.asarray([self.blending_horizon], dtype=np.int32),
            inference_delay_steps=np.asarray([self.lipo_delay_steps], dtype=np.int32),
            inference_dt=np.asarray([self.inference_dt], dtype=np.float32),
        )

        saved_files = ["actions.npz"]
        export_status = "npz saved, plot skipped"
        fig = None
        plt = None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            group_specs = self._action_group_specs()
            fig, axes = plt.subplots(len(group_specs), 1, sharex=True, figsize=(15, 3.3 * len(group_specs)))
            axes = np.atleast_1d(axes)

            for ax, (name, start, end) in zip(axes, group_specs):
                colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(end - start, 1)))
                label_once = {"raw": False, "ref": False, "opt": False, "publish": False, "apply": False}
                for color, dim in zip(colors, range(start, end)):
                    if raw_t.size > 0:
                        ax.plot(
                            raw_t,
                            raw_v[:, dim],
                            color=color,
                            linestyle="--",
                            linewidth=0.8,
                            alpha=0.45,
                            label="raw policy chunk" if not label_once["raw"] else None,
                        )
                        label_once["raw"] = True
                    if ref_t.size > 0:
                        ax.plot(
                            ref_t,
                            ref_v[:, dim],
                            color=color,
                            linestyle=":",
                            linewidth=0.9,
                            alpha=0.55,
                            label="LiPo reference" if not label_once["ref"] else None,
                        )
                        label_once["ref"] = True
                    if optimized_t.size > 0:
                        ax.plot(
                            optimized_t,
                            optimized_v[:, dim],
                            color=color,
                            linewidth=1.0,
                            alpha=0.9,
                            label="LiPo optimized" if not label_once["opt"] else None,
                        )
                        label_once["opt"] = True
                    if publish_t.size > 0:
                        ax.plot(
                            publish_t,
                            publish_v[:, dim],
                            color=color,
                            linewidth=0.8,
                            alpha=0.25,
                            label="published action" if not label_once["publish"] else None,
                        )
                        label_once["publish"] = True

                for t in chunk_apply_t:
                    ax.axvline(
                        float(t),
                        color="black",
                        linewidth=0.8,
                        alpha=0.12,
                        label="chunk applied" if not label_once["apply"] else None,
                    )
                    label_once["apply"] = True

                ax.set_title(f"{name} [{start}:{end}]")
                ax.set_ylabel("rad")
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
                ax.legend(loc="upper right", fontsize=8, ncol=2)

            axes[-1].set_xlabel("time [s]")
            fig.suptitle(
                "LiPo-inspired inference smoothing\n"
                f"policy={self.inference_policy} | dataset={self.dataset_path.name} | checkpoint={self.pretrained_path.parent.parent.name}",
                fontsize=11,
            )
            fig.tight_layout()
            fig.savefig(plot_path, dpi=180)
            saved_files.append("actions.png")
            export_status = "npz+plot saved"
        except Exception as exc:
            logger.warning("[%s] failed to render optimize plot: %s", self.ctx.name, exc)
        finally:
            if fig is not None and plt is not None:
                plt.close(fig)

        self._last_log_export = {
            "session_id": self.log_session_id,
            "status": export_status,
            "session_dir": self.log_session_dir,
            "saved_files": saved_files,
            "sample_summary": sample_summary,
        }
        return self._last_log_export

    def _finalize_log_session(self, *, phase: str) -> dict[str, object]:
        export_meta = self._last_log_export
        if export_meta is not None and export_meta.get("session_id") == self.log_session_id:
            self._log_stop_summary(export_meta, phase=phase)
            return export_meta

        export_meta = self._save_action_logs()
        self._log_stop_summary(export_meta, phase=phase)
        return export_meta

    def on_stop(self) -> None:
        log_export: dict[str, object] | None = None
        try:
            if self._run_mode_active:
                self._run_mode_active = False
                log_export = self._finalize_log_session(phase="worker stop")
            elif self._has_pending_logs():
                log_export = self._finalize_log_session(phase="worker stop")
        except Exception:
            logger.exception("[%s] failed to save optimize logs", self.ctx.name)
            self._last_log_export = {
                "session_id": self.log_session_id,
                "status": "log export failed",
                "session_dir": self.log_session_dir or self._current_log_session_dir(),
                "saved_files": [],
                "sample_summary": "unavailable",
            }
            log_export = self._last_log_export
        finally:
            release_inference_log_session_id(self.inference_policy, self.log_session_id)

        for key, mgr in self._shared_memory.items():
            try:
                mgr.worker_close()
            except Exception:
                logger.exception("[%s] failed to close shared memory %s", self.ctx.name, key)

        if log_export is None:
            logger.info("[%s] worker stop", self.ctx.name)
