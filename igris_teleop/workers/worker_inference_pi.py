from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.events import EventSnapshot
from ..core.project_paths import CHECKPOINTS_ROOT, DATASETS_ROOT, resolve_under_root
from ..core.state_machine import ModeState, TransitionResult
from ..core.worker_base import DualRateWorker, WorkerContext
from ..sharedmemory.shm_schema import (
    CAMERA,
    INFERENCE_RESULT_MAX_ACTION_DIM,
    INFERENCE_RESULT_MAX_STEPS,
)

try:
    from ..robot_control.base_contorl import ARM_INDICES, NECK_INDICES
except Exception:
    from ..sharedmemory.shm_schema import ARM_INDICES as ARM_DIM
    from ..sharedmemory.shm_schema import NECK_INDICES as NECK_DIM

    ARM_INDICES = list(range(int(ARM_DIM)))
    NECK_INDICES = list(range(int(NECK_DIM)))

import logging_mp

logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)


HAND_DIM = 12
STATE_ACTION_DIM = HAND_DIM + len(ARM_INDICES) + len(NECK_INDICES)

DEFAULT_INFERENCE_DATASET_FOLDER = "0304_실증과제_dataset_train/0304_padding"
DEFAULT_INFERENCE_PRETRAINED_REL = "ACT_0304_padding/checkpoints/200000/pretrained_model"
DEFAULT_INFERENCE_POLICY = "pi0"
DEFAULT_INFERENCE_USE_DATASET_STATE = False
DEFAULT_INFERENCE_USE_DATASET_TORQUE = False
DEFAULT_INFERENCE_USE_DATASET_VIDEO = False
DEFAULT_INFERENCE_DATASET_EPISODE = 0
DEFAULT_INFERENCE_DATASET_STRIDE = 1
DEFAULT_INFERENCE_DATASET_LOOP = True
DEFAULT_PI_ACTION_HORIZON = 50
DEFAULT_PI_ACTION_DIM = STATE_ACTION_DIM
PI_POLICY_NAMES = frozenset({"pi0", "pi0.5", "pi05"})

OPENPI_CONFIG_FALLBACKS = {
    "pi0": ("pi0_full_igrisc_4task", "pi0_lora_igrisc_4task"),
    "pi0.5": ("pi05_full_igrisc_4task", "pi05_lora_igrisc_4task"),
}

OPENPI_DATASET_CONFIGS = {
    "pi0": {
        "0304_lifting": {
            "low_mem": "igris_lifting_low_mem_finetune",
            "full": "igris_lifting_full_finetune",
        },
        "0304_padding": {
            "low_mem": "igris_padding_low_mem_finetune",
            "full": "igris_padding_full_finetune",
        },
        "0304_tape": {
            "low_mem": "igris_tape_low_mem_finetune",
            "full": "igris_tape_full_finetune",
        },
        "0304_throwing": {
            "low_mem": "igris_throwing_low_mem_finetune",
            "full": "igris_throwing_full_finetune",
        },
        "igrisc_4task": {
            "low_mem": "pi0_lora_igrisc_4task",
            "full": "pi0_full_igrisc_4task",
        },
    },
    "pi0.5": {
        "0304_lifting": {
            "low_mem": "pi05_igris_lifting_low_mem_finetune",
            "full": "pi05_igris_lifting_low_mem_finetune",
        },
        "0304_padding": {
            "low_mem": "pi05_igris_padding_low_mem_finetune",
            "full": "pi05_igris_padding_low_mem_finetune",
        },
        "0304_tape": {
            "low_mem": "pi05_igris_tape_low_mem_finetune",
            "full": "pi05_igris_tape_low_mem_finetune",
        },
        "0304_throwing": {
            "low_mem": "pi05_igris_throwing_low_mem_finetune",
            "full": "pi05_igris_throwing_low_mem_finetune",
        },
        "igrisc_4task": {
            "low_mem": "pi05_lora_igrisc_4task",
            "full": "pi05_full_igrisc_4task",
        },
    },
}

OPENPI_CONFIG_NAMES_BY_POLICY = {
    policy: frozenset(
        {
            name
            for dataset_configs in policy_configs.values()
            for name in dataset_configs.values()
        }
    )
    for policy, policy_configs in OPENPI_DATASET_CONFIGS.items()
}

CAMERA_FEATURE_KEYS = {
    "observation/image/realsense_head": ("realsense_head", "observation.image.realsense_head"),
    "observation/image/realsense_wrist_left": ("realsense_wrist_left", "observation.image.realsense_wrist_left"),
    "observation/image/realsense_wrist_right": ("realsense_wrist_right", "observation.image.realsense_wrist_right"),
}


def _normalize_inference_policy(policy: str | None) -> str:
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


def _resolve_pretrained_path(ctx: WorkerContext) -> Path:
    raw = (
        getattr(ctx.run_config, "inference_pretrained_rel", None)
        or DEFAULT_INFERENCE_PRETRAINED_REL
    )
    resolved = resolve_under_root(CHECKPOINTS_ROOT, raw)
    return _resolve_openpi_checkpoint_dir(resolved)


def _load_checkpoint_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    metadata_path = checkpoint_dir / "metadata.pt"
    if not metadata_path.exists():
        return {}

    try:
        import torch
    except Exception:
        logger.warning("PyTorch is unavailable, so metadata.pt cannot be loaded from %s", metadata_path)
        return {}

    try:
        payload = torch.load(metadata_path, map_location="cpu", weights_only=False)
    except Exception:
        logger.exception("failed to load metadata.pt from %s", metadata_path)
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_repo_token(raw: str | Path | None) -> str | None:
    txt = str(raw or "").strip().replace("\\", "/")
    if not txt:
        return None
    parts = [part for part in txt.split("/") if part]
    if not parts:
        return None
    return parts[-1].lower()


def _discover_checkpoint_asset_repo_id(checkpoint_dir: Path) -> str | None:
    assets_dir = checkpoint_dir / "assets"
    if not assets_dir.is_dir():
        return None

    for owner_dir in sorted(assets_dir.iterdir()):
        if not owner_dir.is_dir():
            continue
        for repo_dir in sorted(owner_dir.iterdir()):
            if repo_dir.is_dir():
                return f"{owner_dir.name}/{repo_dir.name}"
    return None


def _checkpoint_variant(checkpoint_dir: Path) -> str:
    lowered = str(checkpoint_dir).lower()
    if "low_mem" in lowered or "lora" in lowered:
        return "low_mem"
    return "full"


def _find_norm_stats_path(checkpoint_dir: Path) -> Path | None:
    for path in sorted(checkpoint_dir.glob("assets/**/norm_stats.json")):
        try:
            if path.stat().st_size > 0:
                return path
        except Exception:
            continue
    return None


def _read_norm_stats_dims_from_path(norm_stats_path: Path) -> tuple[int | None, int | None]:
    try:
        payload = json.loads(norm_stats_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("failed to read norm_stats.json from %s", norm_stats_path)
        return None, None

    stats = payload.get("norm_stats", payload)
    state_stats = stats.get("state")
    action_stats = stats.get("actions") or stats.get("action")
    state_mean = state_stats.get("mean") if isinstance(state_stats, dict) else None
    action_mean = action_stats.get("mean") if isinstance(action_stats, dict) else None
    state_dim = int(len(state_mean)) if state_mean else None
    action_dim = int(len(action_mean)) if action_mean else None
    return state_dim, action_dim


def _read_norm_stats_dims(checkpoint_dir: Path) -> tuple[int | None, int | None]:
    norm_stats_path = _find_norm_stats_path(checkpoint_dir)
    if norm_stats_path is None:
        return None, None

    return _read_norm_stats_dims_from_path(norm_stats_path)


def _load_norm_stats_from_dataset(dataset_path: Path):
    stats_path = dataset_path / "meta" / "stats.json"
    if not stats_path.exists():
        return None

    try:
        from openpi.shared.normalize import NormStats
    except ImportError:
        return None

    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("failed to read dataset stats.json from %s", stats_path)
        return None

    def _to_norm_stats(raw_key: str):
        raw_stats = payload.get(raw_key)
        if not isinstance(raw_stats, dict):
            return None

        mean = raw_stats.get("mean")
        std = raw_stats.get("std")
        if mean is None or std is None:
            return None

        q01 = raw_stats.get("q01")
        q99 = raw_stats.get("q99")
        return NormStats(
            mean=np.asarray(mean, dtype=np.float32),
            std=np.asarray(std, dtype=np.float32),
            q01=None if q01 is None else np.asarray(q01, dtype=np.float32),
            q99=None if q99 is None else np.asarray(q99, dtype=np.float32),
        )

    state_stats = _to_norm_stats("observation.state")
    action_stats = _to_norm_stats("action")
    if state_stats is None or action_stats is None:
        return None

    return {
        "state": state_stats,
        "actions": action_stats,
    }


def _metadata_model_config(metadata: dict[str, Any]) -> dict[str, Any]:
    cfg = metadata.get("config")
    if not isinstance(cfg, dict):
        return {}
    model_cfg = cfg.get("model")
    return model_cfg if isinstance(model_cfg, dict) else {}


def _resolve_config_name(
    policy: str,
    checkpoint_dir: Path,
    metadata: dict[str, Any],
    dataset_path: Path | None = None,
) -> str:
    cfg = metadata.get("config")
    if isinstance(cfg, dict):
        raw_name = cfg.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            return raw_name.strip()

    known_policy_configs = OPENPI_CONFIG_NAMES_BY_POLICY.get(policy, frozenset())
    for candidate in (checkpoint_dir, *checkpoint_dir.parents):
        name = candidate.name.strip()
        if name in known_policy_configs:
            return name
        if candidate.parent == candidate:
            break

    variant = _checkpoint_variant(checkpoint_dir)
    dataset_candidates: list[str] = []
    for raw in (
        _discover_checkpoint_asset_repo_id(checkpoint_dir),
        None if dataset_path is None else dataset_path.name,
        dataset_path,
    ):
        token = _normalize_repo_token(raw)
        if token and token not in dataset_candidates:
            dataset_candidates.append(token)

    policy_configs = OPENPI_DATASET_CONFIGS.get(policy, {})
    for dataset_token in dataset_candidates:
        config_names = policy_configs.get(dataset_token)
        if not config_names:
            continue
        if variant in config_names:
            return config_names[variant]
        if "low_mem" in config_names:
            return config_names["low_mem"]
        if "full" in config_names:
            return config_names["full"]

    full_name, lora_name = OPENPI_CONFIG_FALLBACKS.get(
        policy,
        OPENPI_CONFIG_FALLBACKS["pi0"],
    )
    if variant == "low_mem":
        return lora_name
    return full_name


class LocalInferenceDataset:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        self.tasks = self._load_tasks(root)
        self.rows = self._load_rows(root)

    @staticmethod
    def _load_tasks(root: Path) -> dict[int, str]:
        tasks_path = root / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            tasks: dict[int, str] = {}
            with tasks_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    tasks[int(record["task_index"])] = str(record.get("task", ""))
            return tasks

        parquet_path = root / "meta" / "tasks.parquet"
        if not parquet_path.exists():
            return {}

        try:
            import pandas as pd
        except ImportError:
            return {}

        task_frame = pd.read_parquet(parquet_path)
        tasks: dict[int, str] = {}
        for record in task_frame.to_dict(orient="records"):
            task_index = int(record.get("task_index", -1))
            tasks[task_index] = str(record.get("task") or "")
        return tasks

    @staticmethod
    def _parse_chunk_file_indices(path: Path) -> tuple[int, int]:
        chunk_name = path.parent.name
        file_name = path.name
        try:
            chunk_index = int(chunk_name.split("-")[-1])
        except Exception:
            chunk_index = 0
        digits = "".join(ch for ch in file_name if ch.isdigit())
        try:
            file_index = int(digits)
        except Exception:
            file_index = 0
        return chunk_index, file_index

    def _load_rows(self, root: Path) -> list[dict[str, Any]]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for dataset-backed OpenPI inference") from exc

        parquet_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
        if not parquet_files:
            parquet_files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

        rows: list[dict[str, Any]] = []
        for parquet_path in parquet_files:
            frame_table = pd.read_parquet(parquet_path)
            chunk_index, file_index = self._parse_chunk_file_indices(parquet_path)
            for record in frame_table.to_dict(orient="records"):
                row = {key: self._normalize_value(value) for key, value in record.items()}
                row["_chunk_index"] = chunk_index
                row["_file_index"] = file_index
                rows.append(row)
        return rows

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def collect_episode_indices(self, episode_index: int) -> list[int]:
        indices = [
            idx
            for idx, row in enumerate(self.rows)
            if int(row.get("episode_index", -1)) == int(episode_index)
        ]
        if indices:
            return indices

        available = sorted({int(row.get("episode_index", -1)) for row in self.rows if "episode_index" in row})
        if not available:
            raise RuntimeError("Dataset has no episode_index column")

        fallback_episode = available[0]
        return [
            idx
            for idx, row in enumerate(self.rows)
            if int(row.get("episode_index", -1)) == fallback_episode
        ]

    def get_frame(self, row_index: int, *, include_images: bool) -> dict[str, Any]:
        frame = dict(self.rows[row_index])
        task_index = int(frame.get("task_index", -1))
        frame["task"] = self.tasks.get(task_index, "")
        if include_images:
            for slash_key, (_camera_key, video_key) in CAMERA_FEATURE_KEYS.items():
                frame[slash_key] = self._load_video_frame(
                    video_key,
                    chunk_index=int(frame.get("_chunk_index", 0)),
                    file_index=int(frame.get("_file_index", 0)),
                    episode_index=int(frame.get("episode_index", 0)),
                    frame_index=int(frame.get("frame_index", 0)),
                )
        return frame

    def _load_video_frame(
        self,
        video_key: str,
        *,
        chunk_index: int,
        file_index: int,
        episode_index: int,
        frame_index: int,
    ) -> np.ndarray:
        try:
            import av
        except ImportError as exc:
            raise ImportError("PyAV is required for dataset video-backed OpenPI inference") from exc

        chunks_size = int(self.info.get("chunks_size", 1) or 1)
        template = str(self.info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"))
        rel_path = template.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
            episode_index=episode_index,
            episode_chunk=episode_index // max(chunks_size, 1),
        )
        video_path = self.root / rel_path
        with av.open(str(video_path)) as container:
            for current_index, frame in enumerate(container.decode(video=0)):
                if current_index == frame_index:
                    return frame.to_ndarray(format="rgb24")
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")


class InferencePIWorker(DualRateWorker):
    """OpenPI-based inference-only worker.

    The worker only writes action chunks to `inference_result_shm`. LiPo blending and
    publication to `act_shm` happen in `worker_optimize.py`.
    """

    def __init__(self, ctx: WorkerContext, slow_hz: float = 30.0, fast_hz: float = 100.0) -> None:
        super().__init__(ctx, slow_hz=slow_hz, fast_hz=fast_hz)

        self._shared_memory = ctx.shared_memory or {}
        self.obs_shm = self._shared_memory.get("obs_shm")
        self.camera_shm = self._shared_memory.get("camera_shm")
        self.inference_result_shm = self._shared_memory.get("inference_result_shm")
        if self.inference_result_shm is None:
            raise RuntimeError("inference_result_shm is required for inference_pi")

        self.inference_policy = _normalize_inference_policy(
            getattr(self.ctx.run_config, "inference_policy", DEFAULT_INFERENCE_POLICY) or DEFAULT_INFERENCE_POLICY
        )
        if self.inference_policy not in PI_POLICY_NAMES:
            raise ValueError(f"Unsupported OpenPI policy selection: {self.inference_policy}")

        self.dataset_path = resolve_under_root(
            DATASETS_ROOT,
            getattr(self.ctx.run_config, "inference_dataset_folder", None) or DEFAULT_INFERENCE_DATASET_FOLDER,
        )
        self.pretrained_path = _resolve_pretrained_path(ctx)
        if not self.pretrained_path.exists():
            raise FileNotFoundError(f"OpenPI checkpoint path not found: {self.pretrained_path}")
        if not _is_openpi_checkpoint_dir(self.pretrained_path):
            raise RuntimeError(
                "OpenPI inference expects a checkpoint step directory containing metadata.pt, "
                "model.safetensors, or params. "
                f"Selected path: {self.pretrained_path}"
            )

        self.metadata = _load_checkpoint_metadata(self.pretrained_path)
        self.config_name = _resolve_config_name(
            self.inference_policy,
            self.pretrained_path,
            self.metadata,
            self.dataset_path,
        )
        model_cfg = _metadata_model_config(self.metadata)
        self.model_action_horizon = int(model_cfg.get("action_horizon", DEFAULT_PI_ACTION_HORIZON) or DEFAULT_PI_ACTION_HORIZON)
        state_dim, action_dim = _read_norm_stats_dims(self.pretrained_path)
        self.state_dim = int(state_dim or DEFAULT_PI_ACTION_DIM)
        self.action_dim = int(action_dim or DEFAULT_PI_ACTION_DIM)
        if self.state_dim > STATE_ACTION_DIM:
            raise RuntimeError(
                f"Unsupported OpenPI state dim {self.state_dim}; sensor composition supports up to {STATE_ACTION_DIM}"
            )
        if self.action_dim > INFERENCE_RESULT_MAX_ACTION_DIM:
            raise RuntimeError(
                "action_dim exceeds inference_result_shm capacity: "
                f"{self.action_dim} > {INFERENCE_RESULT_MAX_ACTION_DIM}"
            )

        requested_n_action_step = getattr(self.ctx.run_config, "inference_n_action_step", None)
        if requested_n_action_step is None:
            self.n_action_step = self.model_action_horizon
        else:
            self.n_action_step = min(self.model_action_horizon, max(1, int(requested_n_action_step)))
        if self.n_action_step > INFERENCE_RESULT_MAX_STEPS:
            raise RuntimeError(
                "n_action_step exceeds inference_result_shm capacity: "
                f"{self.n_action_step} > {INFERENCE_RESULT_MAX_STEPS}"
            )

        self.inference_instruction = str(getattr(self.ctx.run_config, "inference_instruction", "") or "").strip()
        self.use_dataset_state = bool(
            getattr(self.ctx.run_config, "inference_use_dataset_state", DEFAULT_INFERENCE_USE_DATASET_STATE)
        )
        self.use_dataset_tau = bool(
            getattr(self.ctx.run_config, "inference_use_dataset_tau", DEFAULT_INFERENCE_USE_DATASET_TORQUE)
        )
        self.use_dataset_camera = bool(
            getattr(self.ctx.run_config, "inference_use_dataset_camera", DEFAULT_INFERENCE_USE_DATASET_VIDEO)
        )
        self.use_dataset_observation = any((self.use_dataset_state, self.use_dataset_camera))
        if self.use_dataset_tau:
            logger.info("[%s] OpenPI backend ignores torque inputs; dataset torque toggle is ignored", self.ctx.name)

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
        self.dataset: LocalInferenceDataset | None = None
        self.dataset_episode_indices: list[int] = []
        self.dataset_frame_ptr = 0
        self.dataset_done = False
        self.dataset_idle_after_completion = False
        self.dataset_stream_armed = False
        self.dataset_stream_started_at: float | None = None
        self.dataset_frame_cache: dict[str, Any] | None = None
        self.dataset_frame_cache_ptr: int | None = None
        self._dataset_lock = threading.Lock()

        self.flag_inference = False
        self._run_mode_active = False
        self.segment_elapsed = 0.0
        self.step_counter = 0
        self._pending_request_step: int | None = None
        self._result_seq = 0

        self.control_dt = 1.0 / self.fast_hz
        self.inference_dt = 1.0 / self.slow_hz

        self.policy = self._load_policy()

        if self.use_dataset_observation:
            self._init_dataset_observation_source()

        self._clear_inference_result_shm()

        logger.info(
            "[%s] initialized OpenPI policy=%s config=%s checkpoint=%s state_dim=%s action_dim=%s horizon=%s",
            self.ctx.name,
            self.inference_policy,
            self.config_name,
            self.pretrained_path,
            self.state_dim,
            self.action_dim,
            self.model_action_horizon,
        )
        logger.info(
            "[%s] input sources state=%s video=%s prompt=%s",
            self.ctx.name,
            "dataset" if self.use_dataset_state else "shm",
            "dataset" if self.use_dataset_camera else "shm",
            self.inference_instruction if self.inference_instruction else "<empty>",
        )

    def _load_policy(self):
        try:
            from openpi.policies import policy_config
            from openpi.training import config as training_config
        except ImportError as exc:
            raise ImportError(
                "OpenPI packages are not available in this interpreter. "
                "Launch inference_pi with the OpenPI virtualenv."
            ) from exc

        try:
            train_config = training_config.get_config(self.config_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resolve OpenPI training config '{self.config_name}' for checkpoint {self.pretrained_path}"
            ) from exc

        metadata_cfg = self.metadata.get("config")
        if isinstance(metadata_cfg, dict):
            model_overrides = metadata_cfg.get("model")
            if isinstance(model_overrides, dict):
                model_fields = {field.name for field in dataclasses.fields(train_config.model)}
                overrides = {key: value for key, value in model_overrides.items() if key in model_fields}
                if overrides:
                    train_config = dataclasses.replace(
                        train_config,
                        model=dataclasses.replace(train_config.model, **overrides),
                    )

        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        checkpoint_norm_stats_path: Path | None = None
        if data_config.asset_id:
            checkpoint_norm_stats_path = self.pretrained_path / "assets" / data_config.asset_id / "norm_stats.json"

        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            pytorch_device = "cpu"

        norm_stats = None
        if checkpoint_norm_stats_path is not None and checkpoint_norm_stats_path.exists():
            try:
                if checkpoint_norm_stats_path.stat().st_size > 0:
                    _, checkpoint_action_dim = _read_norm_stats_dims_from_path(checkpoint_norm_stats_path)
                    model_action_dim = int(getattr(train_config.model, "action_dim", self.action_dim) or self.action_dim)
                    if checkpoint_action_dim is not None and checkpoint_action_dim > model_action_dim:
                        norm_stats = _load_norm_stats_from_dataset(self.dataset_path)
                        if norm_stats is not None:
                            logger.warning(
                                "[%s] checkpoint norm stats action dim %s exceeds model action dim %s at %s; "
                                "falling back to dataset stats from %s",
                                self.ctx.name,
                                checkpoint_action_dim,
                                model_action_dim,
                                checkpoint_norm_stats_path,
                                self.dataset_path,
                            )
                    else:
                        logger.info(
                            "[%s] using checkpoint norm stats from %s",
                            self.ctx.name,
                            checkpoint_norm_stats_path,
                        )
                else:
                    norm_stats = _load_norm_stats_from_dataset(self.dataset_path)
                    if norm_stats is not None:
                        logger.warning(
                            "[%s] checkpoint norm stats are empty at %s; falling back to dataset stats from %s",
                            self.ctx.name,
                            checkpoint_norm_stats_path,
                            self.dataset_path,
                        )
            except Exception:
                norm_stats = _load_norm_stats_from_dataset(self.dataset_path)
        else:
            norm_stats = _load_norm_stats_from_dataset(self.dataset_path)
            if norm_stats is not None:
                logger.warning(
                    "[%s] checkpoint norm stats missing for asset_id=%s; falling back to dataset stats from %s",
                    self.ctx.name,
                    data_config.asset_id,
                    self.dataset_path,
                )

        try:
            return policy_config.create_trained_policy(
                train_config,
                self.pretrained_path,
                pytorch_device=pytorch_device,
                norm_stats=norm_stats,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load OpenPI policy from "
                f"{self.pretrained_path}. Check that the checkpoint files are valid and match "
                f"the selected config '{self.config_name}'. Original error: {exc}"
            ) from exc

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
            return

        if self.dataset_idle_after_completion or not self.flag_inference:
            return

        try:
            request_step = self.run_inference_once()
            logger.info("[%s] completed OpenPI inference request for step %s", self.ctx.name, request_step)
        except Exception:
            self.flag_inference = False
            logger.exception("[%s] OpenPI inference failed", self.ctx.name)

    def do_fast(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        if self.state != ModeState.RUN:
            if self._run_mode_active:
                self._run_mode_active = False
                self.reset_runtime_buffers(arm_dataset_stream=False)
            return

        if self.dataset_idle_after_completion:
            return

        if not self._run_mode_active:
            self._run_mode_active = True
            self.reset_runtime_buffers(arm_dataset_stream=self.use_dataset_observation)
            self.flag_inference = True
            self._pending_request_step = 0
            logger.info("[%s] entered RUN, requesting first OpenPI inference", self.ctx.name)
            return

        self.segment_elapsed += self.control_dt
        while self.segment_elapsed >= self.inference_dt:
            self.segment_elapsed -= self.inference_dt
            self.step_counter += 1
            if not self.flag_inference and not self.dataset_idle_after_completion:
                self.flag_inference = True
                self._pending_request_step = self.step_counter

    def run_inference_once(self) -> int:
        frame = None
        if self.use_dataset_observation:
            frame = self._get_dataset_frame()
            obs_data = None if self.use_dataset_state else self._read_required_shm(self.obs_shm, "obs_shm")
            camera_data = None if self.use_dataset_camera else self._read_required_shm(self.camera_shm, "camera_shm")
            observation = self._build_observation_from_dataset_frame(frame, obs_data, camera_data)
        else:
            obs_data = self._read_required_shm(self.obs_shm, "obs_shm")
            camera_data = self._read_required_shm(self.camera_shm, "camera_shm")
            observation = self._build_observation(obs_data, camera_data, prompt=None)

        request_step = self.step_counter if self._pending_request_step is None else self._pending_request_step
        result = self.policy.infer(observation)
        raw_chunk = np.asarray(result.get("actions"), dtype=np.float32)
        if raw_chunk.ndim != 2:
            raw_chunk = raw_chunk.reshape(raw_chunk.shape[0], -1).astype(np.float32, copy=False)
        if raw_chunk.shape[1] < self.action_dim:
            raise RuntimeError(
                f"OpenPI output action dim {raw_chunk.shape[1]} is smaller than expected {self.action_dim}"
            )
        raw_chunk = raw_chunk[: self.n_action_step, : self.action_dim].astype(np.float32, copy=False)
        if raw_chunk.size == 0:
            raise RuntimeError("OpenPI policy returned an empty action chunk")

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
            logger.info("[%s] dataset observation completed; idling OpenPI inference worker", self.ctx.name)

        return request_step

    def reset_runtime_buffers(self, *, arm_dataset_stream: bool) -> None:
        self.flag_inference = False
        self.segment_elapsed = 0.0
        self.step_counter = 0
        self._pending_request_step = None
        self._result_seq = 0
        self.dataset_done = False
        self.dataset_idle_after_completion = False
        if self.use_dataset_observation:
            self._reset_dataset_sequence(arm_stream=arm_dataset_stream)
        self._clear_inference_result_shm()

    @staticmethod
    def _read_required_shm(shm, name: str) -> dict:
        if shm is None:
            raise RuntimeError(f"{name} is required for OpenPI inference input")
        return shm.read_data()

    def _compose_obs_position(self, obs_data: dict) -> np.ndarray:
        obs_position = np.concatenate(
            [
                np.asarray(obs_data["obs_hand"], dtype=np.float32).reshape(-1),
                np.asarray(obs_data["obs_arm"], dtype=np.float32).reshape(-1),
                np.asarray(obs_data["obs_neck"], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        if obs_position.size < self.state_dim:
            raise RuntimeError(f"obs dim mismatch: {obs_position.size} < {self.state_dim}")
        return obs_position[: self.state_dim]

    def _compose_dataset_obs_position(self, frame: dict[str, Any]) -> np.ndarray:
        if "observation.state" not in frame:
            raise KeyError("Dataset frame missing observation.state")

        obs_position = np.asarray(frame["observation.state"], dtype=np.float32).reshape(-1)
        if obs_position.size < self.state_dim:
            raise RuntimeError(
                f"Dataset observation.state too short: {obs_position.size} < {self.state_dim}"
            )
        return obs_position[: self.state_dim].astype(np.float32, copy=False)

    @staticmethod
    def _normalize_camera_image(img: Any) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected camera image shape: {arr.shape}")
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr.astype(np.uint8, copy=False)

    def _camera_images_from_shm(self, camera_data: dict) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for slash_key, (camera_key, _video_key) in CAMERA_FEATURE_KEYS.items():
            if camera_key not in camera_data:
                raise KeyError(f"camera_shm missing {camera_key} for OpenPI inference")
            image = self._normalize_camera_image(camera_data[camera_key])
            images[slash_key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return images

    def _camera_images_from_dataset(self, frame: dict[str, Any]) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for slash_key in CAMERA_FEATURE_KEYS:
            if slash_key not in frame:
                raise KeyError(f"Dataset frame missing {slash_key}")
            images[slash_key] = self._normalize_camera_image(frame[slash_key])
        return images

    def _resolve_prompt(self, frame: dict[str, Any] | None) -> str:
        if self.inference_instruction:
            return self.inference_instruction
        if frame is not None:
            task = str(frame.get("task", "") or "").strip()
            if task:
                return task
        return ""

    def _build_observation(
        self,
        obs_data: dict,
        camera_data: dict,
        *,
        prompt: str | None,
    ) -> dict[str, Any]:
        observation: dict[str, Any] = {
            "observation/state": self._compose_obs_position(obs_data),
            **self._camera_images_from_shm(camera_data),
        }
        observation["prompt"] = prompt if prompt is not None else self._resolve_prompt(None)
        return observation

    def _build_observation_from_dataset_frame(
        self,
        frame: dict[str, Any],
        obs_data: dict | None,
        camera_data: dict | None,
    ) -> dict[str, Any]:
        if self.use_dataset_state:
            state = self._compose_dataset_obs_position(frame)
        else:
            if obs_data is None:
                raise RuntimeError("obs_shm input is required when inference_use_dataset_state is False")
            state = self._compose_obs_position(obs_data)

        if self.use_dataset_camera:
            images = self._camera_images_from_dataset(frame)
        else:
            if camera_data is None:
                raise RuntimeError("camera_shm input is required when inference_use_dataset_camera is False")
            images = self._camera_images_from_shm(camera_data)

        return {
            "observation/state": state,
            **images,
            "prompt": self._resolve_prompt(frame),
        }

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

        self.dataset = LocalInferenceDataset(self.dataset_path)
        self.dataset_episode_indices = self.dataset.collect_episode_indices(self.dataset_episode)
        if not self.dataset_episode_indices:
            raise RuntimeError(f"Episode {self.dataset_episode} was not found in dataset {self.dataset_path}")

        first_row = self.dataset.rows[self.dataset_episode_indices[0]]
        first_episode = int(first_row.get("episode_index", self.dataset_episode))
        if first_episode != self.dataset_episode:
            logger.info(
                "[%s] dataset episode %s not found; falling back to episode %s",
                self.ctx.name,
                self.dataset_episode,
                first_episode,
            )
            self.dataset_episode = first_episode

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

    def _get_dataset_frame(self) -> dict[str, Any]:
        if self.dataset is None or not self.dataset_episode_indices:
            raise RuntimeError("dataset observation source is not initialized")

        max_attempts = len(self.dataset_episode_indices)
        attempts = 0
        while attempts < max_attempts:
            with self._dataset_lock:
                self._advance_dataset_sequence()
                current_ptr = int(self.dataset_frame_ptr)

                if self.dataset_frame_cache is not None and self.dataset_frame_cache_ptr == current_ptr:
                    return self.dataset_frame_cache

                row_idx = int(self.dataset_episode_indices[current_ptr])
                try:
                    frame = self.dataset.get_frame(row_idx, include_images=self.use_dataset_camera)
                except Exception:
                    self._advance_dataset_sequence(force_next=True)
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    continue

                self.dataset_frame_cache = frame
                self.dataset_frame_cache_ptr = current_ptr
                return frame

        raise RuntimeError("Failed to load a valid dataset frame for OpenPI inference")

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

    def on_stop(self) -> None:
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
