from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act_gradcam import ACTPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import populate_queues
from lerobot.configs.types import FeatureType
from lerobot.utils.constants import OBS_IMAGES

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
from ..sharedmemory.shm_schema import (
    CAMERA,
    INFERENCE_RESULT_MAX_ACTION_DIM,
    INFERENCE_RESULT_MAX_STEPS,
)

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

import logging_mp

logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)


HAND_DIM = 12
BASE_STATE_DIM = HAND_DIM + len(ARM_INDICES) + len(NECK_INDICES)
FULL_WAIST_DIM = len(WAIST_INDICES)
FULL_STATE_DIM = BASE_STATE_DIM + FULL_WAIST_DIM

DEFAULT_INFERENCE_DATASET_FOLDER = "0304_실증과제_dataset_train/0304_padding"
DEFAULT_INFERENCE_PRETRAINED_REL = "ACT_0304_padding/checkpoints/200000/pretrained_model"
DEFAULT_INFERENCE_POLICY = "act"
DEFAULT_INFERENCE_ACT_CHUNK_SIZE = 100
DEFAULT_INFERENCE_ACT_N_ACTION_STEP = 100
DEFAULT_INFERENCE_DIFFUSION_HORIZON = 16
DEFAULT_INFERENCE_DIFFUSION_N_ACTION_STEP = 8
DEFAULT_INFERENCE_USE_DATASET_STATE = False
DEFAULT_INFERENCE_USE_DATASET_TORQUE = False
DEFAULT_INFERENCE_USE_DATASET_VIDEO = False
DEFAULT_INFERENCE_DATASET_EPISODE = 0
DEFAULT_INFERENCE_DATASET_STRIDE = 1
DEFAULT_INFERENCE_DATASET_LOOP = True


def _read_pretrained_feature_dims(pretrained_dir: Path) -> tuple[int | None, int | None, int | None]:
    config_path = pretrained_dir / "config.json"
    if not config_path.exists():
        return None, None, None

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None, None

    state_shape = config.get("input_features", {}).get("observation.state", {}).get("shape")
    torque_shape = config.get("input_features", {}).get("observation.torque", {}).get("shape")
    action_shape = config.get("output_features", {}).get("action", {}).get("shape")
    state_dim = int(state_shape[0]) if state_shape else None
    torque_dim = int(torque_shape[0]) if torque_shape else None
    action_dim = int(action_shape[0]) if action_shape else None
    return state_dim, torque_dim, action_dim


def _read_pretrained_policy_type(pretrained_dir: Path) -> str | None:
    config_path = pretrained_dir / "config.json"
    if not config_path.exists():
        return None

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    raw = config.get("policy_type") or config.get("type") or config.get("_target_")
    if raw is None:
        return None
    txt = str(raw).strip().lower()
    return txt or None


def _normalize_policy_family(name: str | None) -> str | None:
    if not name:
        return None
    policy = str(name).strip().lower()
    if policy in {"act"}:
        return "act"
    if policy in {"diffusion", "diffusion_policy", "diffusion policy"}:
        return "diffusion"
    return None


class InferenceLeRobotWorker(DualRateWorker):
    """Inference-only worker.

    The fast loop only schedules requests. The slow loop runs policy inference and
    writes the resulting action chunk to `inference_result_shm`.
    """

    def __init__(self, ctx: WorkerContext, slow_hz: float = 30.0, fast_hz: float = 100.0) -> None:
        super().__init__(ctx, slow_hz=slow_hz, fast_hz=fast_hz)

        self._shared_memory = ctx.shared_memory or {}
        self.obs_shm = self._shared_memory.get("obs_shm")
        self.tau_shm = self._shared_memory.get("tau_shm")
        self.camera_shm = self._shared_memory.get("camera_shm")
        self.inference_result_shm = self._shared_memory.get("inference_result_shm")
        if self.inference_result_shm is None:
            raise RuntimeError("inference_result_shm is required for inference_lerobot")

        self.dataset_repo_id = "IGRIS_C"
        dataset_folder_name = (
            getattr(self.ctx.run_config, "inference_dataset_folder", None)
            or DEFAULT_INFERENCE_DATASET_FOLDER
        )
        self.dataset_path = resolve_under_root(DATASETS_ROOT, dataset_folder_name)

        pretrained_rel = (
            getattr(self.ctx.run_config, "inference_pretrained_rel", None)
            or DEFAULT_INFERENCE_PRETRAINED_REL
        )
        self.pretrained_path = resolve_under_root(CHECKPOINTS_ROOT, pretrained_rel)
        if not self.pretrained_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {self.pretrained_path}")

        pretrained_policy_type = _read_pretrained_policy_type(self.pretrained_path)
        pretrained_state_dim, pretrained_torque_dim, pretrained_action_dim = _read_pretrained_feature_dims(
            self.pretrained_path
        )

        dataset_metadata = LeRobotDatasetMetadata(self.dataset_repo_id, root=str(self.dataset_path))
        features_spec = deepcopy(dataset_metadata.features)
        if "observation.state" in features_spec and pretrained_state_dim is not None:
            features_spec["observation.state"]["shape"] = [pretrained_state_dim]
        if "action" in features_spec and pretrained_action_dim is not None:
            features_spec["action"]["shape"] = [pretrained_action_dim]
        if pretrained_torque_dim is None:
            features_spec.pop("observation.torque", None)
        elif "observation.torque" in features_spec:
            features_spec["observation.torque"]["shape"] = [pretrained_torque_dim]

        features = dataset_to_policy_features(features_spec)
        self.output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
        self.input_features = {k: ft for k, ft in features.items() if k not in self.output_features}
        if "action" not in self.output_features:
            raise RuntimeError("Policy output feature 'action' is required")

        self.action_dim = int(self.output_features["action"].shape[0])
        self.state_dim = int(self.input_features["observation.state"].shape[0]) if "observation.state" in self.input_features else 0
        self.torque_dim = (
            int(self.input_features["observation.torque"].shape[0])
            if "observation.torque" in self.input_features
            else 0
        )
        if self.state_dim not in (0, BASE_STATE_DIM, FULL_STATE_DIM):
            raise RuntimeError(
                "Unsupported observation.state dim for obs_shm composition: "
                f"{self.state_dim} (supported: 0, {BASE_STATE_DIM}, {FULL_STATE_DIM})"
            )
        self.use_waist_state = self.state_dim == FULL_STATE_DIM
        if self.action_dim > INFERENCE_RESULT_MAX_ACTION_DIM:
            raise RuntimeError(
                "action_dim exceeds inference_result_shm capacity: "
                f"{self.action_dim} > {INFERENCE_RESULT_MAX_ACTION_DIM}"
            )

        self.inference_policy = str(
            getattr(self.ctx.run_config, "inference_policy", DEFAULT_INFERENCE_POLICY) or DEFAULT_INFERENCE_POLICY
        ).strip().lower()
        selected_policy_family = _normalize_policy_family(self.inference_policy)
        pretrained_policy_family = _normalize_policy_family(pretrained_policy_type)
        if (
            selected_policy_family is not None
            and pretrained_policy_family is not None
            and selected_policy_family != pretrained_policy_family
        ):
            raise ValueError(
                "Selected inference policy does not match checkpoint policy type: "
                f"selected={self.inference_policy}, checkpoint={pretrained_policy_type}, "
                f"path={self.pretrained_path}"
            )
        self.chunk_size = max(
            1,
            int(
                getattr(self.ctx.run_config, "inference_chunk_size", None)
                or DEFAULT_INFERENCE_ACT_CHUNK_SIZE
            ),
        )
        self.horizon = max(
            8,
            int(
                getattr(self.ctx.run_config, "inference_horizon", None)
                or DEFAULT_INFERENCE_DIFFUSION_HORIZON
            ),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.inference_policy == "act":
            self.n_action_step = max(
                1,
                int(
                    getattr(self.ctx.run_config, "inference_n_action_step", None)
                    or DEFAULT_INFERENCE_ACT_N_ACTION_STEP
                ),
            )
            if self.n_action_step > self.chunk_size:
                raise ValueError("ACT n_action_step must be <= chunk_size")
            cfg = ACTConfig(
                input_features=self.input_features,
                output_features=self.output_features,
                chunk_size=self.chunk_size,
                n_action_steps=self.n_action_step,
            )
            self.policy = ACTPolicy.from_pretrained(
                pretrained_name_or_path=str(self.pretrained_path),
                config=cfg,
            )
        elif self.inference_policy in {"diffusion", "diffusion_policy", "diffusion policy"}:
            self.n_action_step = max(
                1,
                int(
                    getattr(self.ctx.run_config, "inference_n_action_step", None)
                    or DEFAULT_INFERENCE_DIFFUSION_N_ACTION_STEP
                ),
            )
            if self.horizon % 8 != 0:
                raise ValueError("Diffusion horizon must be a multiple of 8")
            if self.n_action_step > self.horizon:
                raise ValueError("Diffusion n_action_step must be <= horizon")
            cfg = DiffusionConfig(
                input_features=self.input_features,
                output_features=self.output_features,
                horizon=self.horizon,
                n_action_steps=self.n_action_step,
            )
            self.policy = DiffusionPolicy.from_pretrained(
                pretrained_name_or_path=str(self.pretrained_path),
                config=cfg,
            )
        else:
            raise ValueError(f"Unknown inference policy: {self.inference_policy}")

        self.policy_config = cfg
        self.policy.to(self.device)
        self.policy.reset()
        self.policy.eval()

        device_str = str(self.device)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.pretrained_path),
            preprocessor_overrides={
                "device_processor": {"device": device_str},
                "normalizer_processor": {"device": device_str},
            },
            postprocessor_overrides={
                "unnormalizer_processor": {"device": device_str},
                "device_processor": {"device": "cpu"},
            },
        )

        self.use_dataset_state = bool(
            getattr(self.ctx.run_config, "inference_use_dataset_state", DEFAULT_INFERENCE_USE_DATASET_STATE)
        )
        self.use_dataset_tau = bool(
            getattr(self.ctx.run_config, "inference_use_dataset_tau", DEFAULT_INFERENCE_USE_DATASET_TORQUE)
        )
        self.use_dataset_camera = bool(
            getattr(self.ctx.run_config, "inference_use_dataset_camera", DEFAULT_INFERENCE_USE_DATASET_VIDEO)
        )
        self.use_dataset_observation = any(
            (self.use_dataset_state, self.use_dataset_tau, self.use_dataset_camera)
        )

        self.dataset_episode = int(
            getattr(self.ctx.run_config, "inference_dataset_episode", DEFAULT_INFERENCE_DATASET_EPISODE)
        )
        self.dataset_stride = max(
            1,
            int(getattr(self.ctx.run_config, "inference_dataset_stride", DEFAULT_INFERENCE_DATASET_STRIDE)),
        )
        self.dataset_loop = bool(
            getattr(self.ctx.run_config, "inference_dataset_loop", DEFAULT_INFERENCE_DATASET_LOOP)
        )
        self.dataset_fps = float(self.slow_hz)
        self.dataset: LeRobotDataset | None = None
        self.dataset_episode_indices: list[int] = []
        self.dataset_frame_ptr = 0
        self.dataset_done = False
        self.dataset_idle_after_completion = False
        self.dataset_stream_armed = False
        self.dataset_stream_started_at: float | None = None
        self.dataset_frame_cache: dict | None = None
        self.dataset_frame_cache_ptr: int | None = None
        self._dataset_lock = threading.Lock()

        self._camera_schema_map = {name: shape for name, shape, _dtype in CAMERA}
        self._camera_feature_map = {
            "observation.image.stereo_left": "stereo_left",
            "observation.image.stereo_right": "stereo_right",
            "observation.image.realsense_head": "realsense_head",
            "observation.image.realsense_wrist_left": "realsense_wrist_left",
            "observation.image.realsense_wrist_right": "realsense_wrist_right",
        }

        self.flag_inference = False
        self._run_mode_active = False
        self._pending_policy_reset = threading.Event()
        self.segment_elapsed = 0.0
        self.step_counter = 0
        self._pending_request_step: int | None = None
        self._result_seq = 0

        self.control_dt = 1.0 / self.fast_hz
        self.inference_dt = 1.0 / self.slow_hz
        self.log_root_dir = INFERENCE_LOGS_ROOT
        self.log_session_dir: Path | None = None
        self.log_session_id = time.strftime("%Y%m%d_%H%M%S")
        self.log_started_at = time.monotonic()
        self._last_log_export: dict[str, object] | None = None

        if self.n_action_step > INFERENCE_RESULT_MAX_STEPS:
            raise RuntimeError(
                "n_action_step exceeds inference_result_shm capacity: "
                f"{self.n_action_step} > {INFERENCE_RESULT_MAX_STEPS}"
            )

        if self.use_dataset_observation:
            self._init_dataset_observation_source()

        self._clear_inference_result_shm()

        logger.info(
            "[%s] initialized policy=%s dataset=%s checkpoint=%s state_dim=%s action_dim=%s torque_dim=%s",
            self.ctx.name,
            self.inference_policy,
            self.dataset_path,
            self.pretrained_path,
            self.state_dim,
            self.action_dim,
            self.torque_dim,
        )
        logger.info(
            "[%s] input sources state=%s video=%s torque=%s",
            self.ctx.name,
            "dataset" if self.use_dataset_state else "shm",
            "dataset" if self.use_dataset_camera else "shm",
            "dataset" if self.use_dataset_tau else "shm",
        )

    def on_start(self) -> None:
        logger.info(
            "[%s] start (dual-rate slow=%.1fHz fast=%.1fHz, policy=%s)",
            self.ctx.name,
            self.slow_hz,
            self.fast_hz,
            self.inference_policy,
        )

    def do_slow(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        if self.state != ModeState.RUN:
            self.flag_inference = False
            self._pending_policy_reset.clear()
            return

        if self.dataset_idle_after_completion:
            return

        if self._pending_policy_reset.is_set():
            self.policy.reset()
            self._pending_policy_reset.clear()
            logger.info("[%s] reset policy state on RUN entry", self.ctx.name)

        if not self.flag_inference:
            return

        try:
            request_step = self.run_inference_once()
            logger.info("[%s] completed inference request for step %s", self.ctx.name, request_step)
        except Exception:
            self.flag_inference = False
            logger.exception("[%s] do_slow inference failed", self.ctx.name)

    def do_fast(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        if self.state != ModeState.RUN:
            if self._run_mode_active:
                self._run_mode_active = False
                self._finalize_log_session(phase="run stop")
                self.reset_runtime_buffers(arm_dataset_stream=False, start_new_session=False)
            return

        if self.dataset_idle_after_completion:
            return

        if not self._run_mode_active:
            self._run_mode_active = True
            self.reset_runtime_buffers(
                arm_dataset_stream=self.use_dataset_observation,
                start_new_session=True,
            )
            self._pending_policy_reset.set()
            self.flag_inference = True
            self._pending_request_step = 0
            logger.info("[%s] entered RUN, requesting first inference", self.ctx.name)
            return

        self.segment_elapsed += self.control_dt
        while self.segment_elapsed >= self.inference_dt:
            self.segment_elapsed -= self.inference_dt
            self.step_counter += 1
            if not self.flag_inference and not self.dataset_idle_after_completion:
                self.flag_inference = True
                self._pending_request_step = self.step_counter

    def run_inference_once(self) -> int:
        if self.use_dataset_observation:
            frame = self._get_dataset_frame()
            obs_data = None if self.use_dataset_state else self._read_required_shm(self.obs_shm, "obs_shm")
            tau_data = None if self.use_dataset_tau else self._read_optional_shm(self.tau_shm)
            camera_data = None if self.use_dataset_camera else self._read_optional_shm(self.camera_shm)
            batch = self._build_batch_from_dataset_frame(frame, obs_data, tau_data, camera_data)
        else:
            obs_data = self._read_required_shm(self.obs_shm, "obs_shm")
            tau_data = self._read_optional_shm(self.tau_shm)
            camera_data = self._read_optional_shm(self.camera_shm)
            batch = self._build_batch(obs_data, tau_data, camera_data)

        request_step = self.step_counter if self._pending_request_step is None else self._pending_request_step
        gradcam_inputs = None
        if self.inference_policy == "act" and getattr(self.policy.config, "image_features", None):
            gradcam_inputs = {
                key: batch[key]
                for key in self.policy.config.image_features
                if key in batch
            }

        batch = self.preprocessor(batch)

        gradcam_payload = None
        if isinstance(self.policy, DiffusionPolicy):
            with torch.inference_mode():
                queue_batch = dict(batch)
                if self.policy.config.image_features:
                    queue_batch[OBS_IMAGES] = torch.stack(
                        [queue_batch[key] for key in self.policy.config.image_features],
                        dim=-4,
                    )
                self.policy._queues = populate_queues(self.policy._queues, queue_batch)
                act = self.policy.predict_action_chunk(queue_batch)
        elif gradcam_inputs:
            act, gradcam_payload = self.policy.predict_action_chunk_with_gradcam(
                batch,
                camera_inputs=gradcam_inputs,
            )
        else:
            with torch.inference_mode():
                act = self.policy.predict_action_chunk(batch)

        act = self.postprocessor(act)
        raw_chunk = np.asarray(act.detach().cpu().numpy()[0], dtype=np.float32)
        if raw_chunk.ndim != 2:
            raw_chunk = raw_chunk.reshape(raw_chunk.shape[0], -1).astype(np.float32, copy=False)
        raw_chunk = raw_chunk[: self.n_action_step]
        if raw_chunk.shape != (self.n_action_step, self.action_dim):
            raise ValueError(
                f"Invalid action chunk shape: expected {(self.n_action_step, self.action_dim)}, got {raw_chunk.shape}"
            )

        saved_gradcam_files = self._save_gradcam_visualizations(gradcam_payload, request_step=request_step)
        if saved_gradcam_files:
            logger.info(
                "[%s] saved %d Grad-CAM overlay(s) for step %s in %s",
                self.ctx.name,
                len(saved_gradcam_files),
                request_step,
                self.log_session_dir,
            )

        self._write_inference_result(raw_chunk, request_step=request_step)
        self.flag_inference = False
        self._pending_request_step = None

        if (
            self.use_dataset_observation
            and not self.dataset_loop
            and self._is_final_dataset_frame_ptr(self.dataset_frame_ptr)
        ):
            self.dataset_done = True
            self.dataset_idle_after_completion = True
            logger.info("[%s] dataset observation completed; idling inference worker", self.ctx.name)

        return request_step

    def reset_runtime_buffers(self, *, arm_dataset_stream: bool, start_new_session: bool) -> None:
        self.flag_inference = False
        self._pending_policy_reset.clear()
        self.segment_elapsed = 0.0
        self.step_counter = 0
        self._pending_request_step = None
        self._result_seq = 0
        if start_new_session:
            self.log_started_at = time.monotonic()
            self.log_session_id = allocate_inference_log_session_id(self.inference_policy)
        self.log_session_dir = None
        self._last_log_export = None
        self.dataset_done = False
        self.dataset_idle_after_completion = False
        if self.use_dataset_observation:
            self._reset_dataset_sequence(arm_stream=arm_dataset_stream)
        self._clear_inference_result_shm()

    def _log_time(self) -> float:
        return time.monotonic() - self.log_started_at

    def _current_log_session_dir(self) -> Path:
        return self.log_root_dir / inference_log_dir_name(self.inference_policy, self.log_session_id)

    def _ensure_log_session_dir(self) -> Path:
        if self.log_session_dir is None:
            self.log_session_dir = self._current_log_session_dir()
            self.log_session_dir.mkdir(parents=True, exist_ok=True)
        return self.log_session_dir

    @staticmethod
    def _sanitize_gradcam_name(name: str) -> str:
        return name.replace("/", "_").replace(".", "_")

    def _gradcam_root_dir(self) -> Path:
        return self._ensure_log_session_dir() / "grad_cam"

    def _has_pending_gradcam_logs(self) -> bool:
        if self.log_session_dir is None:
            return False
        images_dir = self.log_session_dir / "grad_cam" / "images"
        return images_dir.exists() and any(images_dir.glob("*.png"))

    def _save_gradcam_visualizations(self, payload: dict | None, *, request_step: int) -> list[str]:
        if not payload:
            return []

        images = payload.get("images")
        if not isinstance(images, dict) or not images:
            return []

        gradcam_dir = self._gradcam_root_dir()
        meta_dir = gradcam_dir / "meta"
        images_dir = gradcam_dir / "images"
        meta_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        written_files: list[str] = []
        meta = {
            "request_step": int(request_step),
            "log_time_sec": float(self._log_time()),
            "target_action_step": int(payload.get("target_action_step", 0)),
            "target_action_index": payload.get("target_action_index"),
            "target_score": float(payload.get("target_score", 0.0)),
            "image_files": {},
        }

        for feature_key, image_payload in images.items():
            rgb_image = np.asarray(image_payload.get("image"), dtype=np.uint8)
            heatmap = np.asarray(image_payload.get("heatmap"), dtype=np.float32)
            if rgb_image.ndim != 3 or heatmap.ndim != 2:
                continue

            heatmap_u8 = np.clip(heatmap * 255.0, 0.0, 255.0).astype(np.uint8, copy=False)
            colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(rgb_image, 0.55, colored, 0.45, 0.0)

            filename = (
                f"step_{int(request_step):06d}_{self._sanitize_gradcam_name(str(feature_key))}_gradcam.png"
            )
            out_path = images_dir / filename
            if cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
                rel_path = f"images/{filename}"
                written_files.append(rel_path)
                meta["image_files"][str(feature_key)] = rel_path

        meta_path = meta_dir / f"step_{int(request_step):06d}_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if written_files:
            written_files.append(f"meta/{meta_path.name}")
        return written_files

    def _save_gradcam_videos(self) -> list[str]:
        gradcam_dir = self._gradcam_root_dir()
        images_dir = gradcam_dir / "images"
        if not images_dir.exists():
            return []

        grouped_frames: dict[str, list[tuple[int, Path]]] = {}
        for image_path in sorted(images_dir.glob("step_*_gradcam.png")):
            stem = image_path.stem
            if not stem.startswith("step_"):
                continue
            remainder = stem[len("step_") :]
            if "_" not in remainder:
                continue
            step_txt, observation_name = remainder.split("_", 1)
            if observation_name.endswith("_gradcam"):
                observation_name = observation_name[: -len("_gradcam")]
            try:
                step_idx = int(step_txt)
            except ValueError:
                continue
            grouped_frames.setdefault(observation_name, []).append((step_idx, image_path))

        written_files: list[str] = []
        fps = float(max(1.0, round(1.0 / max(self.inference_dt, 1e-6))))
        for observation_name, frames in grouped_frames.items():
            frames.sort(key=lambda item: item[0])
            first_frame = cv2.imread(str(frames[0][1]), cv2.IMREAD_COLOR)
            if first_frame is None:
                continue

            height, width = first_frame.shape[:2]
            video_path = gradcam_dir / f"{observation_name}.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                logger.warning("[%s] failed to open Grad-CAM video writer for %s", self.ctx.name, video_path)
                continue

            try:
                for idx, (step_idx, frame_path) in enumerate(frames):
                    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                    if idx + 1 < len(frames):
                        next_step = frames[idx + 1][0]
                        repeat = max(1, int(next_step - step_idx))
                    else:
                        repeat = 1

                    for _ in range(repeat):
                        writer.write(frame)
            finally:
                writer.release()

            written_files.append(video_path.name)

        return written_files

    def _log_stop_summary(self, export_meta: dict[str, object], *, phase: str) -> None:
        session_dir = export_meta.get("session_dir")
        saved_files = export_meta.get("saved_files")
        files_str = ",".join(saved_files) if isinstance(saved_files, list) and saved_files else "none"
        logger.info(
            "[%s] %s (session=%s elapsed=%.2fs export=%s dir=%s files=%s)",
            self.ctx.name,
            phase,
            self.log_session_id,
            self._log_time(),
            export_meta.get("status", "unknown"),
            str(session_dir) if session_dir is not None else "n/a",
            files_str,
        )

    def _finalize_log_session(self, *, phase: str) -> dict[str, object]:
        export_meta = self._last_log_export
        if export_meta is not None and export_meta.get("session_id") == self.log_session_id:
            self._log_stop_summary(export_meta, phase=phase)
            return export_meta

        if not self._has_pending_gradcam_logs():
            self._last_log_export = {
                "session_id": self.log_session_id,
                "status": "no Grad-CAM samples; log export skipped",
                "session_dir": self.log_session_dir,
                "saved_files": [],
            }
            self._log_stop_summary(self._last_log_export, phase=phase)
            release_inference_log_session_id(self.inference_policy, self.log_session_id)
            return self._last_log_export

        saved_videos = self._save_gradcam_videos()
        saved_files = [f"grad_cam/{name}" for name in saved_videos]
        self._last_log_export = {
            "session_id": self.log_session_id,
            "status": "Grad-CAM images+videos saved" if saved_videos else "Grad-CAM images saved",
            "session_dir": self.log_session_dir,
            "saved_files": saved_files,
        }
        self._log_stop_summary(self._last_log_export, phase=phase)
        release_inference_log_session_id(self.inference_policy, self.log_session_id)
        return self._last_log_export

    def _write_inference_result(self, chunk: np.ndarray, *, request_step: int) -> None:
        rows, cols = chunk.shape
        if rows > INFERENCE_RESULT_MAX_STEPS:
            raise RuntimeError(f"chunk rows exceed shm capacity: {rows} > {INFERENCE_RESULT_MAX_STEPS}")
        if cols > INFERENCE_RESULT_MAX_ACTION_DIM:
            raise RuntimeError(
                f"chunk action dim exceeds shm capacity: {cols} > {INFERENCE_RESULT_MAX_ACTION_DIM}"
            )

        payload = np.zeros(
            (INFERENCE_RESULT_MAX_STEPS, INFERENCE_RESULT_MAX_ACTION_DIM),
            dtype=np.float32,
        )
        payload[:rows, :cols] = chunk.astype(np.float32, copy=False)

        self._result_seq += 1
        self.inference_result_shm.write_data(
            seq=np.float32(self._result_seq),
            valid=np.float32(1.0),
            request_step=np.float32(request_step),
            timestamp=np.float32(time.time()),
            n_action_steps=np.float32(rows),
            action_dim=np.float32(cols),
            action_chunk=payload,
        )

    def _clear_inference_result_shm(self) -> None:
        zeros = np.zeros(
            (INFERENCE_RESULT_MAX_STEPS, INFERENCE_RESULT_MAX_ACTION_DIM),
            dtype=np.float32,
        )
        self.inference_result_shm.write_data(
            seq=np.float32(0.0),
            valid=np.float32(0.0),
            request_step=np.float32(0.0),
            timestamp=np.float32(0.0),
            n_action_steps=np.float32(0.0),
            action_dim=np.float32(0.0),
            action_chunk=zeros,
        )

    @staticmethod
    def _read_optional_shm(shm):
        if shm is None:
            return None
        return shm.read_data()

    @staticmethod
    def _read_required_shm(shm, name: str) -> dict:
        if shm is None:
            raise RuntimeError(f"{name} is required for sensor-based inference input")
        return shm.read_data()

    def _compose_obs_position(self, obs_data: dict) -> np.ndarray:
        parts = [
            np.asarray(obs_data["obs_hand"], dtype=np.float32).reshape(-1),
            np.asarray(obs_data["obs_arm"], dtype=np.float32).reshape(-1),
            np.asarray(obs_data["obs_neck"], dtype=np.float32).reshape(-1),
        ]
        if self.use_waist_state:
            waist = np.asarray(obs_data["obs_waist"], dtype=np.float32).reshape(-1)
            if waist.size < FULL_WAIST_DIM:
                raise RuntimeError(f"obs_waist too short: {waist.size} < {FULL_WAIST_DIM}")
            parts.append(waist[:FULL_WAIST_DIM])

        obs_position = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        if obs_position.size < self.state_dim:
            raise RuntimeError(f"obs dim mismatch: {obs_position.size} < {self.state_dim}")
        return obs_position[: self.state_dim]

    def _compose_obs_torque(self, tau_data: dict) -> np.ndarray | None:
        if self.torque_dim <= 0:
            return None

        parts = [np.asarray(tau_data["tau_est_arm"], dtype=np.float32).reshape(-1)]
        current_dim = parts[0].size
        if self.torque_dim > current_dim:
            parts.append(np.asarray(tau_data["tau_est_waist"], dtype=np.float32).reshape(-1))
            current_dim += parts[-1].size

        if current_dim < self.torque_dim:
            raise RuntimeError(f"torque dim mismatch: {current_dim} < {self.torque_dim}")

        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)[: self.torque_dim]

    @staticmethod
    def _img_to_chw_float01(img) -> torch.Tensor:
        if torch.is_tensor(img):
            t = img
            if t.ndim != 3:
                raise ValueError(f"Unexpected image tensor shape: {tuple(t.shape)}")
            if t.shape[0] not in (1, 3, 4) and t.shape[-1] in (1, 3, 4):
                t = t.permute(2, 0, 1).contiguous()
            if t.shape[0] == 4:
                t = t[:3]
            t = t.float()
            if t.max().item() > 1.5:
                t = t / 255.0
            return t

        arr = np.asarray(img)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected image ndarray shape: {arr.shape}")
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]

        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
        if t.max().item() > 1.5:
            t = t / 255.0
        return t

    def _append_dataset_camera_features_to_batch(self, batch: dict, frame: dict, *, strict: bool) -> None:
        for feature_key, feat in self.policy.config.input_features.items():
            if not str(feat.type).endswith("VISUAL"):
                continue

            if feature_key not in frame:
                if strict:
                    available_keys = ", ".join(sorted(frame.keys()))
                    raise KeyError(
                        f"Dataset frame missing {feature_key}. Available keys: {available_keys}"
                    )
                continue

            img_t = self._img_to_chw_float01(frame[feature_key])
            _c, h, w = feat.shape
            if tuple(img_t.shape[-2:]) != (h, w):
                img_t = torchvision.transforms.functional.resize(
                    img_t,
                    size=[h, w],
                    antialias=True,
                )
            batch[feature_key] = img_t.unsqueeze(0).to(self.device, non_blocking=True)

    def _append_camera_features_to_batch(self, batch: dict, camera_data: dict | None, *, strict: bool) -> None:
        for feature_key, feat in self.policy.config.input_features.items():
            if not str(feat.type).endswith("VISUAL"):
                continue

            raw_name = self._camera_feature_map.get(feature_key)
            if raw_name is None:
                if strict:
                    raise KeyError(f"Unsupported camera feature key: {feature_key}")
                continue

            if camera_data is None or raw_name not in camera_data:
                if strict:
                    raise KeyError(
                        f"camera_shm missing {raw_name} for required feature {feature_key}"
                    )
                continue

            img = camera_data[raw_name]
            _c, h, w = feat.shape
            img = cv2.resize(img, (w, h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
            batch[feature_key] = img_t.unsqueeze(0).to(self.device, non_blocking=True)

    def _build_batch(self, obs_data: dict, tau_data: dict | None, camera_data: dict | None) -> dict:
        batch: dict[str, torch.Tensor] = {}
        self._append_camera_features_to_batch(batch, camera_data, strict=True)

        if "observation.state" in self.policy.config.input_features:
            obs_position = np.asarray(self._compose_obs_position(obs_data), dtype=np.float32)
            batch["observation.state"] = torch.from_numpy(obs_position).unsqueeze(0).to(
                self.device,
                non_blocking=True,
            )

        if "observation.torque" in self.policy.config.input_features:
            if tau_data is None:
                raise RuntimeError("tau_shm input is required for observation.torque")
            obs_torque = self._compose_obs_torque(tau_data)
            if obs_torque is not None:
                batch["observation.torque"] = torch.from_numpy(obs_torque).unsqueeze(0).to(
                    self.device,
                    non_blocking=True,
                )

        return batch

    def _build_batch_from_dataset_frame(
        self,
        frame: dict,
        obs_data: dict | None,
        tau_data: dict | None,
        camera_data: dict | None,
    ) -> dict:
        batch: dict[str, torch.Tensor] = {}

        if self.use_dataset_camera:
            self._append_dataset_camera_features_to_batch(batch, frame, strict=True)
        else:
            self._append_camera_features_to_batch(batch, camera_data, strict=True)

        if "observation.state" in self.policy.config.input_features:
            if self.use_dataset_state:
                obs_position = self._compose_dataset_obs_position(frame)
            else:
                if obs_data is None:
                    raise RuntimeError("obs_shm input is required when inference_use_dataset_state is False")
                obs_position = self._compose_obs_position(obs_data)
            batch["observation.state"] = torch.from_numpy(obs_position).unsqueeze(0).to(
                self.device,
                non_blocking=True,
            )

        if "observation.torque" in self.policy.config.input_features:
            if self.use_dataset_tau:
                obs_torque = self._compose_dataset_obs_torque(frame)
            else:
                if tau_data is None:
                    raise RuntimeError("tau_shm input is required when inference_use_dataset_tau is False")
                obs_torque = self._compose_obs_torque(tau_data)
            if obs_torque is not None:
                batch["observation.torque"] = torch.from_numpy(obs_torque).unsqueeze(0).to(
                    self.device,
                    non_blocking=True,
                )

        return batch

    def _init_dataset_observation_source(self) -> None:
        dataset_fps_override = getattr(self.ctx.run_config, "inference_dataset_fps", None)
        if dataset_fps_override is not None:
            self.dataset_fps = float(dataset_fps_override)
        else:
            info_path = self.dataset_path / "meta" / "info.json"
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                    self.dataset_fps = float(info.get("fps", self.dataset_fps))
                except Exception:
                    logger.exception("[%s] failed to read dataset fps from %s", self.ctx.name, info_path)

        self.dataset = LeRobotDataset(
            self.dataset_repo_id,
            root=str(self.dataset_path),
            episodes=[self.dataset_episode],
        )
        self.dataset_episode_indices = [
            row_pos
            for row_pos, episode_idx in enumerate(self.dataset.hf_dataset["episode_index"])
            if int(episode_idx) == self.dataset_episode
        ]
        if not self.dataset_episode_indices:
            available_eps = sorted({int(e) for e in self.dataset.hf_dataset["episode_index"]})
            if not available_eps:
                raise RuntimeError("Dataset has no episodes")
            self.dataset_episode = int(available_eps[0])
            self.dataset_episode_indices = [
                row_pos
                for row_pos, episode_idx in enumerate(self.dataset.hf_dataset["episode_index"])
                if int(episode_idx) == self.dataset_episode
            ]

        self._reset_dataset_sequence(arm_stream=False)
        self._get_dataset_frame()

    def _reset_dataset_sequence(self, *, arm_stream: bool) -> None:
        self.dataset_frame_ptr = 0
        self.dataset_done = False
        self.dataset_idle_after_completion = False
        self.dataset_stream_armed = arm_stream
        self.dataset_stream_started_at = time.monotonic() if arm_stream else None
        self.dataset_frame_cache = None
        self.dataset_frame_cache_ptr = None

    def _dataset_num_steps(self) -> int:
        if not self.dataset_episode_indices:
            return 0
        return max(1, (len(self.dataset_episode_indices) + self.dataset_stride - 1) // self.dataset_stride)

    def _is_final_dataset_frame_ptr(self, frame_ptr: int) -> bool:
        if not self.dataset_episode_indices:
            return False
        final_step_ptr = min(
            (self._dataset_num_steps() - 1) * self.dataset_stride,
            len(self.dataset_episode_indices) - 1,
        )
        return int(frame_ptr) >= int(final_step_ptr)

    def _advance_dataset_sequence(self, *, force_next: bool = False) -> int:
        if not self.dataset_episode_indices:
            return 0

        if force_next:
            self.dataset_frame_ptr = min(
                self.dataset_frame_ptr + max(1, self.dataset_stride),
                len(self.dataset_episode_indices) - 1,
            )
            return self.dataset_frame_ptr

        if not self.dataset_stream_armed or self.dataset_stream_started_at is None:
            self.dataset_frame_ptr = 0
            return self.dataset_frame_ptr

        num_steps = self._dataset_num_steps()
        elapsed = max(0.0, time.monotonic() - self.dataset_stream_started_at)
        step_idx = int(elapsed * self.dataset_fps / float(self.dataset_stride))
        if self.dataset_loop and num_steps > 0:
            step_idx %= num_steps
        else:
            step_idx = min(step_idx, num_steps - 1)

        self.dataset_frame_ptr = min(
            step_idx * self.dataset_stride,
            len(self.dataset_episode_indices) - 1,
        )
        return self.dataset_frame_ptr

    def _get_dataset_frame(self) -> dict:
        if self.dataset is None or not self.dataset_episode_indices:
            raise RuntimeError("dataset observation source is not initialized")

        max_attempts = len(self.dataset_episode_indices)
        attempts = 0
        while attempts < max_attempts:
            with self._dataset_lock:
                self._advance_dataset_sequence()
                current_ptr = int(self.dataset_frame_ptr)

                if self.dataset_frame_cache is not None and self.dataset_frame_cache_ptr == current_ptr:
                    frame = self.dataset_frame_cache
                else:
                    row_idx = int(self.dataset_episode_indices[current_ptr])
                    try:
                        frame = self.dataset[row_idx]
                    except IndexError as exc:
                        if "Invalid frame index" not in str(exc):
                            raise
                        logger.warning(
                            "[%s] skip dataset frame row_idx=%s due to video decode mismatch: %s",
                            self.ctx.name,
                            row_idx,
                            exc,
                        )
                        self._advance_dataset_sequence(force_next=True)
                        attempts += 1
                        continue

                    self.dataset_frame_cache = frame
                    self.dataset_frame_cache_ptr = current_ptr

            return frame

        raise RuntimeError("Failed to load a valid dataset frame for inference")

    def _compose_dataset_obs_position(self, frame: dict) -> np.ndarray:
        if "observation.state" not in frame:
            raise KeyError("Dataset frame missing observation.state")

        obs_position = np.asarray(frame["observation.state"], dtype=np.float32).reshape(-1)
        if obs_position.size < self.state_dim:
            raise RuntimeError(
                f"Dataset observation.state too short: {obs_position.size} < {self.state_dim}"
            )
        return obs_position[: self.state_dim].astype(np.float32, copy=False)

    def _compose_dataset_obs_torque(self, frame: dict) -> np.ndarray | None:
        if self.torque_dim <= 0:
            return None
        if "observation.torque" not in frame:
            raise KeyError("Dataset frame missing observation.torque")

        obs_torque = np.asarray(frame["observation.torque"], dtype=np.float32).reshape(-1)
        if obs_torque.size < self.torque_dim:
            raise RuntimeError(
                f"Dataset observation.torque too short: {obs_torque.size} < {self.torque_dim}"
            )
        return obs_torque[: self.torque_dim].astype(np.float32, copy=False)

    def on_stop(self) -> None:
        try:
            if self._run_mode_active:
                self._run_mode_active = False
                self._finalize_log_session(phase="worker stop")
            elif self._has_pending_gradcam_logs():
                self._finalize_log_session(phase="worker stop")
        except Exception:
            logger.exception("[%s] failed to finalize Grad-CAM logs", self.ctx.name)
        finally:
            release_inference_log_session_id(self.inference_policy, self.log_session_id)

        try:
            self._clear_inference_result_shm()
        except Exception:
            logger.exception("[%s] failed to clear inference_result_shm", self.ctx.name)

        for key, mgr in self._shared_memory.items():
            try:
                mgr.worker_close()
            except Exception:
                logger.exception("[%s] failed to close shared memory %s", self.ctx.name, key)

        logger.info("[%s] stop", self.ctx.name)
