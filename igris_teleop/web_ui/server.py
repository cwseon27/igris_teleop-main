from __future__ import annotations

import json
import logging
import mimetypes
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from igris_teleop.core.events import LEVEL_EVENTS, EventBus
from igris_teleop.core.project_paths import (
    CHECKPOINTS_ROOT,
    COLLECT_DATA_CONFIG_PATH,
    DATASETS_ROOT,
    REPO_ROOT,
)
from igris_teleop.core.worker_base import (
    CAMERA_MODE_CHOICES,
    DEFAULT_CAMERA_MODE,
    allowed_teleop_hand_sources,
    resolve_teleop_hand_source,
)
from igris_teleop.head_start_guard import format_head_guard_message, head_guard_is_blocking
from igris_teleop.sharedmemory.shm_schema import RECORD_TASK_NAME_LEN
from igris_teleop.sim.experiment_scenes import (
    EXPERIMENT_TASK_BY_ID,
    experiment_tasks_payload,
)
from igris_teleop.sim.stereo_camera import (
    STEREO_CAMERA_BASELINE_M,
    STEREO_CAMERA_MAX_BASELINE_M,
    STEREO_CAMERA_MIN_BASELINE_M,
    validate_stereo_baseline,
)
from igris_teleop.policies.walking import (
    WALKING_PROFILE_CHOICES,
    default_walking_policy_path,
    resolve_walking_policy_profile,
    walking_command_is_zero,
    walking_policy_profile_code,
)
from igris_teleop.training.analysis.plot_inference_npz import (
    ARM_JOINT_LABELS,
    ARM_LEN,
    HAND_LEN,
    NECK_JOINT_LABELS,
    NECK_LEN,
    WAIST_JOINT_LABELS,
    WAIST_LEN,
)
from igris_teleop.training.datasets.segment_labeling import (
    LABEL_FEATURE_KEY,
    UNASSIGNED_CLASS_ID,
    default_label_classes,
    ensure_classes_cover_ids,
    episode_has_unassigned_segments,
    get_labeling_backend_status,
    label_name_for_id,
    load_annotation_sidecar,
    normalize_boundary_segments,
    rewrite_dataset_segment_labels_in_place,
    segments_from_label_ids,
    segments_to_label_ids,
)
from igris_teleop.workers.command_process import (
    CommandProcessSpec,
    ManagedCommandProcess,
    ROS_JAZZY_SETUP_BASH,
    build_ros_shell_argv,
)

try:  # OpenCV is already part of the runtime dependency set.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional fallback
    cv2 = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional fallback
    Image = None

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).with_name("static")

MODE_CHOICES = ("teleop", "walking", "inference", "replay")
UNITY_TELEOP_DEVICES = frozenset({"unity", "unity_hybrid"})
VR_POSE_TELEOP_DEVICES = frozenset({"unity", "unity_hybrid", "vr_masterarm"})
TELEOP_DEVICE_CHOICES = ("unity", "unity_hybrid", "vr_masterarm", "masterarm")
DEFAULT_INFERENCE_POLICY = "act"
RELIABILITY_MODEL_VARIANTS = ("rnn", "histgb", "always_1")
DEFAULT_RELIABILITY_MODEL_VARIANT = "rnn"
RELIABILITY_ROS_WS_ROOT = Path(
    os.getenv("IGRIS_RELIABILITY_ROS_WS_ROOT") or str(REPO_ROOT / "ros_ws")
).expanduser().resolve()
RELIABILITY_ROS_WS_SETUP_BASH = Path(
    os.getenv("IGRIS_RELIABILITY_ROS_WS_SETUP_BASH")
    or str(RELIABILITY_ROS_WS_ROOT / "install" / "setup.bash")
).expanduser().resolve()
RELIABILITY_POLICY_ROOT = Path(
    os.getenv("IGRIS_RELIABILITY_POLICY_ROOT")
    or str(REPO_ROOT / "policy_archive" / "reliability")
).expanduser().resolve()
RELIABILITY_PYTHON = Path(
    os.getenv("IGRIS_RELIABILITY_PYTHON") or str(REPO_ROOT / ".venv-ml" / "bin" / "python")
).expanduser().absolute()
MEDIAPIPE_PYTHON = Path(
    os.getenv("IGRIS_MEDIAPIPE_PYTHON")
    or str(REPO_ROOT / ".venv-mediapipe" / "bin" / "python")
).expanduser().absolute()
HYBRID_PREVIEW_ROOT = Path(
    os.getenv("IGRIS_HYBRID_PREVIEW_ROOT") or "/dev/shm/igris_hybrid_teleop_preview"
).expanduser().resolve()
HYBRID_CONFIG_PATH = Path(
    os.getenv("IGRIS_HYBRID_CONFIG_PATH")
    or str(REPO_ROOT / "igris_artifacts" / "config" / "hybrid_teleop.json")
).expanduser().resolve()
HYBRID_PREVIEW_SIDES = frozenset({"left", "right"})
HYBRID_PREVIEW_STAGES = frozenset({"raw", "trapezoid", "mediapipe"})
CAMERA_NAMES = (
    "realsense_wrist_left",
    "stereo_left",
    "realsense_wrist_right",
    "realsense_head",
    "stereo_right",
)
DATASET_VIEWER_MAX_DATASETS = 500
DATASET_VIEWER_SCAN_DEPTH = 5
DATASET_VIEWER_DEFAULT_MAX_POINTS = 900
DATASET_VIEWER_MAX_POINTS = 2000
DATASET_VIEWER_FRAME_CACHE_LIMIT = 24
OBSERVATION_KEY_CANDIDATES = (
    "observation.state",
    "observation_state",
    "state",
    "observation",
)
EE_POSE_KEY_CANDIDATES = (
    "observation.ee_pose",
    "observation_ee_pose",
    "ee_pose",
)
ACTION_KEY_CANDIDATES = (
    "action",
    "actions",
)
FILE_BROWSER_KINDS = frozenset(
    {
        "inference_dataset",
        "inference_checkpoint",
        "replay_dataset",
        "walking_policy",
        "dataset_viewer_root",
    }
)

MANUAL_WORKER_GROUPS: dict[str, tuple[str, ...]] = {
    "simulator": ("simulator",),
    "bridge_control": ("control", "hand"),
    "leader_ros": ("leader_ros_tcp_endpoint", "leader_ros_node"),
    "collect_data": ("collect_data",),
}
MANUAL_WORKER_GROUP_LABELS: dict[str, str] = {
    "simulator": "simulator",
    "bridge_control": "control",
    "leader_ros": "leader_ros",
    "collect_data": "collect_data",
}
MANUAL_ONE_SHOT_GROUPS: frozenset[str] = frozenset({"simulator", "bridge_control"})


def _manual_worker_names_for_group(
    group: str,
    mode: str | None,
    teleop_device: str | None,
) -> tuple[str, ...]:
    if group != "leader_ros" or mode != "teleop":
        return MANUAL_WORKER_GROUPS[group]
    if teleop_device in UNITY_TELEOP_DEVICES:
        return ("leader_ros_tcp_endpoint",)
    if teleop_device == "masterarm":
        return ("leader_ros_node",)
    return MANUAL_WORKER_GROUPS[group]


def _default_viser_url() -> str:
    explicit_url = str(os.getenv("IGRIS_VISER_URL") or "").strip()
    if explicit_url:
        return explicit_url
    scheme = str(os.getenv("IGRIS_VISER_SCHEME") or "http").strip() or "http"
    host = str(os.getenv("IGRIS_VISER_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    port = str(os.getenv("IGRIS_VISER_PORT") or "8080").strip() or "8080"
    return f"{scheme}://{host}:{port}/"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "<none>", "null"}:
        return None
    return text


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    return value


def _scalar_bool(value: Any) -> bool:
    try:
        return bool(np.asarray(value).reshape(()).item())
    except Exception:
        return bool(value)


def _scalar_int(value: Any) -> int:
    try:
        return int(np.asarray(value).reshape(()).item())
    except Exception:
        return 0


def _scalar_float(value: Any) -> float:
    try:
        return float(np.asarray(value).reshape(()).item())
    except Exception:
        return 0.0


def _json_float(value: Any, digits: int = 3) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return round(out, digits)


def _float_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if shape is not None:
        try:
            arr = arr.reshape(shape)
        except Exception:
            return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr.copy()


def _flat_values(value: Any, *, limit: int, digits: int = 3) -> list[float | None]:
    arr = _float_array(value)
    if arr is None:
        return []
    flat = arr.reshape(-1)[:limit]
    return [_json_float(item, digits) for item in flat]


def _vector_summary(value: Any, *, limit: int, digits: int = 3) -> dict[str, Any]:
    arr = _float_array(value)
    if arr is None:
        return {"valid": False, "size": 0, "values": [], "norm": None, "max_abs": None}
    flat = arr.reshape(-1)
    return {
        "valid": True,
        "size": int(flat.size),
        "values": [_json_float(item, digits) for item in flat[:limit]],
        "norm": _json_float(np.linalg.norm(flat), digits),
        "max_abs": _json_float(np.max(np.abs(flat)) if flat.size else 0.0, digits),
    }


def _pose_summary(value: Any) -> dict[str, Any]:
    mat = _float_array(value, (4, 4))
    if mat is None:
        return {"valid": False, "xyz": [], "rpy_deg": []}
    r = mat[:3, :3]
    roll = np.arctan2(r[2, 1], r[2, 2])
    pitch = np.arctan2(-r[2, 0], np.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return {
        "valid": True,
        "xyz": [_json_float(item, 4) for item in mat[:3, 3]],
        "rpy_deg": [_json_float(item, 1) for item in np.degrees([roll, pitch, yaw])],
    }


def _hand_points_summary(value: Any) -> dict[str, Any]:
    arr = _float_array(value, (5, 3))
    if arr is None:
        return {"valid": False, "centroid": [], "first": [], "spread": None}
    centroid = np.mean(arr, axis=0)
    spread = np.max(np.linalg.norm(arr - centroid, axis=1))
    return {
        "valid": True,
        "centroid": [_json_float(item, 4) for item in centroid],
        "first": [_json_float(item, 4) for item in arr[0]],
        "spread": _json_float(spread, 4),
    }


def _array_fingerprint(value: Any, digits: int = 5) -> tuple[Any, ...] | None:
    arr = _float_array(value)
    if arr is None:
        return None
    rounded = np.round(arr.reshape(-1), digits)
    return (tuple(arr.shape), tuple(float(item) for item in rounded))


def _decode_uint8_text(raw: Any) -> str:
    if raw is None:
        return ""
    try:
        arr = np.asarray(raw, dtype=np.uint8).reshape(-1)
    except Exception:
        return ""
    return arr.tobytes().split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()


def _encode_uint8_text(text: str | None, length: int) -> np.ndarray:
    arr = np.zeros((length,), dtype=np.uint8)
    clean = str(text or "").strip()
    if not clean:
        return arr
    encoded = clean.encode("utf-8")[: length - 1]
    if encoded:
        arr[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return arr


def _load_collect_defaults() -> tuple[str, list[str]]:
    try:
        import yaml  # type: ignore

        with COLLECT_DATA_CONFIG_PATH.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except Exception:
        return "IGRIS_C", ["Pick and Place"]

    dataset_cfg = payload.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return "IGRIS_C", ["Pick and Place"]

    repo_id = str(dataset_cfg.get("repo_id", "") or "").strip() or "IGRIS_C"
    tasks: list[str] = []
    raw_tasks = dataset_cfg.get("tasks")
    if isinstance(raw_tasks, (list, tuple)):
        for raw in raw_tasks:
            task = str(raw or "").strip()
            if task and task not in tasks:
                tasks.append(task)
    fallback_task = str(dataset_cfg.get("task_name", "") or "").strip()
    if fallback_task and fallback_task not in tasks:
        tasks.insert(0, fallback_task)
    return repo_id, tasks or ["Pick and Place"]


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _ensure_browser_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("failed to create browser root: %s", resolved, exc_info=True)
    return resolved


def _browser_root_containing(path: Path, preferred_root: Path) -> Path:
    resolved_path = path.expanduser().resolve()
    resolved_root = preferred_root.expanduser().resolve()
    if _path_is_relative_to(resolved_path, resolved_root):
        return resolved_root
    return resolved_path.parent.resolve()


def _display_path(path: Path, *, base_root: Path, absolute: bool = False) -> str:
    path = path.resolve()
    if absolute:
        return str(path)
    try:
        return str(path.relative_to(base_root))
    except ValueError:
        return str(path)


def _dataset_viewer_external_python() -> Path | None:
    if _env_bool("IGRIS_DATASET_VIEWER_DISABLE_EXTERNAL", False):
        return None

    raw = str(os.getenv("IGRIS_DATASET_VIEWER_PYTHON") or "").strip()
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    if _env_bool("IGRIS_DATASET_VIEWER_USE_ML", True):
        candidates.append(REPO_ROOT / ".venv-ml" / "bin" / "python")

    try:
        current = Path(sys.executable).resolve()
    except Exception:
        current = None  # type: ignore[assignment]

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        if current is not None and resolved == current:
            continue
        return resolved
    return None


def _require_lerobot_dataset() -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

        return LeRobotDataset
    except Exception:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

            return LeRobotDataset
        except Exception as exc:
            raise RuntimeError(
                "LeRobot dataset dependencies are unavailable. "
                "Install `lerobot`, `pyarrow`, and video/image runtime dependencies "
                "in the Python environment used to launch the web UI."
            ) from exc


def _find_lerobot_dataset_dirs(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        return []

    out: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and len(out) < DATASET_VIEWER_MAX_DATASETS:
        path, depth = stack.pop()
        if (path / "meta" / "info.json").is_file():
            out.append(path)
            continue
        if depth >= DATASET_VIEWER_SCAN_DEPTH:
            continue
        try:
            children = sorted(
                [child for child in path.iterdir() if child.is_dir()],
                key=lambda child: child.name.lower(),
                reverse=True,
            )
        except OSError:
            continue
        for child in children:
            if child.name.startswith(".") or child.name == "__pycache__":
                continue
            stack.append((child, depth + 1))
    return sorted(out, key=lambda path: str(path).lower())


def _load_dataset_info_json(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid info.json payload: {info_path}")
    return payload


def _feature_dict(info: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    raw = info.get("features", {})
    return raw if isinstance(raw, dict) else {}


def _image_keys_from_info(info: Mapping[str, Any] | dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key, spec in _feature_dict(info).items():
        if not isinstance(key, str) or not key.startswith("observation.image."):
            continue
        if isinstance(spec, dict):
            dtype = str(spec.get("dtype", "")).strip().lower()
            if dtype not in {"video", "image"}:
                continue
        keys.append(key)
    return sorted(keys)


def _ee_pose_key_from_info(info: Mapping[str, Any] | dict[str, Any]) -> str:
    features = _feature_dict(info)
    for key in EE_POSE_KEY_CANDIDATES:
        if key in features:
            return key
    return ""


def _image_stream_name(image_key: str) -> str:
    text = str(image_key or "").strip()
    if not text:
        return ""
    return text.rsplit(".", 1)[-1]


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:
        return default


def _extract_dataset_vector(row: Any, candidates: tuple[str, ...]) -> np.ndarray:
    for key in candidates:
        value = _row_value(row, key, None)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size > 0:
            return arr
    return np.zeros((0,), dtype=np.float32)


def _extract_dataset_label_id(row: Any) -> int:
    value = _row_value(row, LABEL_FEATURE_KEY, None)
    if value is None:
        return UNASSIGNED_CLASS_ID
    try:
        arr = np.asarray(value, dtype=np.int64).reshape(-1)
    except Exception:
        return UNASSIGNED_CLASS_ID
    if arr.size <= 0:
        return UNASSIGNED_CLASS_ID
    return int(arr[0])


def _dataset_frame_to_uint8_rgb(frame: Any) -> np.ndarray:
    arr = frame
    if hasattr(arr, "detach") and hasattr(arr, "cpu"):
        arr = arr.detach().cpu().numpy()
    elif hasattr(arr, "numpy") and not isinstance(arr, np.ndarray):
        try:
            arr = arr.numpy()
        except Exception:
            pass
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected image shape: {tuple(arr.shape)}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Unexpected image channels: {tuple(arr.shape)}")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    arr = np.asarray(arr, dtype=np.float32)
    maxv = float(np.nanmax(arr)) if arr.size else 1.0
    if maxv <= 1.5:
        arr = arr * 255.0
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _dataset_segment_specs() -> tuple[tuple[str, int, list[str]], ...]:
    return (
        ("hand", HAND_LEN, [f"hand_{idx}" for idx in range(HAND_LEN)]),
        ("arm", ARM_LEN, list(ARM_JOINT_LABELS)),
        ("neck", NECK_LEN, list(NECK_JOINT_LABELS)),
        ("waist", WAIST_LEN, list(WAIST_JOINT_LABELS)),
    )


def _slice_dataset_feature(values: np.ndarray, start: int, stop: int) -> np.ndarray:
    if values.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32)
    dim = int(values.shape[1])
    lo = max(0, min(start, dim))
    hi = max(lo, min(stop, dim))
    return np.asarray(values[:, lo:hi], dtype=np.float32)


def _extract_dataset_ee_tracks(ee_pose_series: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(ee_pose_series, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 48:
        return {}
    try:
        mats = arr[:, :48].reshape(arr.shape[0], 3, 4, 4)
    except Exception:
        return {}
    return {
        "head": np.asarray(mats[:, 0, :3, 3], dtype=np.float32),
        "left": np.asarray(mats[:, 1, :3, 3], dtype=np.float32),
        "right": np.asarray(mats[:, 2, :3, 3], dtype=np.float32),
    }


def _dataset_sample_indices(length: int, max_points: int) -> np.ndarray:
    length = max(0, int(length))
    max_points = max(16, min(DATASET_VIEWER_MAX_POINTS, int(max_points)))
    if length <= max_points:
        return np.arange(length, dtype=np.int64)
    idxs = np.linspace(0, length - 1, num=max_points, dtype=np.int64)
    return np.unique(idxs)


def _clean_json_float(value: Any, digits: int = 5) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return round(out, digits)


def _json_float_vector(values: Any, digits: int = 5) -> list[float | None]:
    arr = np.asarray(values).reshape(-1)
    return [_clean_json_float(item, digits) for item in arr]


def _json_float_matrix(values: np.ndarray, indices: np.ndarray, digits: int = 5) -> list[list[float | None]]:
    arr = np.asarray(values)
    if arr.ndim != 2 or arr.shape[0] <= 0:
        return []
    safe_indices = np.asarray(indices, dtype=np.int64)
    safe_indices = safe_indices[(safe_indices >= 0) & (safe_indices < arr.shape[0])]
    return [_json_float_vector(arr[int(idx)], digits) for idx in safe_indices]


def _dataset_viewer_segments_payload(
    dataset_dir: Path,
    info: Mapping[str, Any],
    episode_idx: int,
    frame_numbers: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    feature_available = LABEL_FEATURE_KEY in _feature_dict(info)
    try:
        sidecar = load_annotation_sidecar(dataset_dir)
    except Exception:
        sidecar = {
            "feature_key": LABEL_FEATURE_KEY,
            "classes": default_label_classes(),
            "episodes": {},
        }

    raw_episodes = sidecar.get("episodes", {})
    sidecar_segments = raw_episodes.get(int(episode_idx)) if isinstance(raw_episodes, Mapping) else None
    if sidecar_segments:
        segments = normalize_boundary_segments(sidecar_segments, frame_numbers)
    elif feature_available or np.any(np.asarray(label_ids, dtype=np.int64) != UNASSIGNED_CLASS_ID):
        segments = segments_from_label_ids(frame_numbers, label_ids)
    else:
        segments = normalize_boundary_segments([], frame_numbers)

    classes = ensure_classes_cover_ids(
        sidecar.get("classes", default_label_classes()) if isinstance(sidecar, Mapping) else default_label_classes(),
        [int(segment["class_id"]) for segment in segments],
    )
    normalized_label_ids = segments_to_label_ids(segments, frame_numbers)
    return (
        [{"id": int(item["id"]), "name": str(item["name"])} for item in classes],
        [
            {
                "start_frame": int(segment["start_frame"]),
                "end_frame": int(segment["end_frame"]),
                "class_id": int(segment["class_id"]),
                "class_name": label_name_for_id(classes, int(segment["class_id"])),
            }
            for segment in segments
        ],
        normalized_label_ids,
    )


def _decode_dataset_image_value(value: Any, dataset_root: Path) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        raw_bytes = value.get("bytes")
        if raw_bytes:
            return _decode_dataset_image_value(raw_bytes, dataset_root)
        raw_path = value.get("path")
        if raw_path:
            return _decode_dataset_image_value(raw_path, dataset_root)
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = np.frombuffer(bytes(value), dtype=np.uint8)
        if cv2 is not None:
            bgr = cv2.imdecode(payload, cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if Image is not None:
            import io

            with Image.open(io.BytesIO(bytes(value))) as image:
                return np.asarray(image.convert("RGB"))
        return None
    if isinstance(value, str):
        image_path = Path(value).expanduser()
        if not image_path.is_absolute():
            image_path = dataset_root / image_path
        if not image_path.is_file():
            return None
        if cv2 is not None:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if Image is not None:
            with Image.open(image_path) as image:
                return np.asarray(image.convert("RGB"))
        return None
    return _dataset_frame_to_uint8_rgb(value)


def _validate_reliability_model_variant(value: str | None, *, field_name: str) -> str:
    variant = _normalize_optional(value) or DEFAULT_RELIABILITY_MODEL_VARIANT
    if variant not in RELIABILITY_MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported {field_name}={variant!r}; "
            f"choose one of {', '.join(RELIABILITY_MODEL_VARIANTS)}"
        )
    return variant


@dataclass(frozen=True)
class HybridTeleopConfig:
    left_device: int | str = 1
    right_device: int | str = 0
    trapezoid_preprocess: bool = True
    trapezoid_bottom_width: int = 320
    swap_lr: bool = True
    mirror: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "left_device": self.left_device,
            "right_device": self.right_device,
            "trapezoid_preprocess": self.trapezoid_preprocess,
            "trapezoid_bottom_width": self.trapezoid_bottom_width,
            "swap_lr": self.swap_lr,
            "mirror": self.mirror,
        }


def _coerce_hybrid_bool(value: Any, fallback: bool) -> bool:
    if value is None or value == "":
        return bool(fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


V4L_BY_PATH = Path("/dev/v4l/by-path")


def _stable_camera_devices() -> dict[str, str]:
    """Map capture nodes to udev port paths; by-id can collide on identical cameras."""
    devices: dict[str, str] = {}
    for path in sorted(V4L_BY_PATH.glob("*-video-index0")):
        if path.exists():
            devices.setdefault(str(path.resolve()), str(path))
    return devices


def _coerce_camera_device(value: Any, *, name: str) -> int | str:
    text = str(value).strip()
    if text.isdigit():
        index = int(text)
        if not 0 <= index <= 99:
            raise ValueError(f"{name} must be between 0 and 99 or a /dev camera path")
        return _stable_camera_devices().get(f"/dev/video{index}", index)
    path = Path(text)
    if not path.is_absolute() or not path.is_relative_to("/dev") or ".." in path.parts:
        raise ValueError(f"{name} must be a camera number or an absolute /dev camera path")
    # Keep port symlinks even while unplugged. Resolving them here would persist
    # the very /dev/videoN number that changes after a USB reconnect.
    return _stable_camera_devices().get(text, text)


def _camera_device_identity(device: int | str) -> str:
    path = Path(f"/dev/video{device}" if isinstance(device, int) else device)
    return str(path.resolve())


def _hybrid_config_from_mapping(
    payload: Mapping[str, Any] | None,
    *,
    base: HybridTeleopConfig | None = None,
) -> HybridTeleopConfig:
    source = payload or {}
    defaults = base or HybridTeleopConfig()

    def read_int(name: str, fallback: int) -> int:
        value = source.get(name, fallback)
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc

    left_device = _coerce_camera_device(source.get("left_device", defaults.left_device), name="left_device")
    right_device = _coerce_camera_device(source.get("right_device", defaults.right_device), name="right_device")
    bottom_width = read_int("trapezoid_bottom_width", defaults.trapezoid_bottom_width)
    if _camera_device_identity(left_device) == _camera_device_identity(right_device):
        raise ValueError("Left and right hand cameras must use different devices")
    if bottom_width < 1 or bottom_width > 640:
        raise ValueError("trapezoid_bottom_width must be between 1 and 640 pixels")

    return HybridTeleopConfig(
        left_device=left_device,
        right_device=right_device,
        trapezoid_preprocess=_coerce_hybrid_bool(
            source.get("trapezoid_preprocess"), defaults.trapezoid_preprocess
        ),
        trapezoid_bottom_width=bottom_width,
        swap_lr=_coerce_hybrid_bool(source.get("swap_lr"), defaults.swap_lr),
        mirror=_coerce_hybrid_bool(source.get("mirror"), defaults.mirror),
    )


def _load_hybrid_config(path: Path = HYBRID_CONFIG_PATH) -> HybridTeleopConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            return _hybrid_config_from_mapping(payload)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("Failed to load hybrid teleop config %s: %s", path, exc)
    return HybridTeleopConfig()


def _save_hybrid_config(config: HybridTeleopConfig, path: Path = HYBRID_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(config.as_dict(), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _launch_bool(value: bool) -> str:
    return "true" if value else "false"


def _hybrid_runtime_environment() -> dict[str, str]:
    return {
        "IGRIS_PROJECT_ROOT": str(REPO_ROOT),
        "IGRIS_RELIABILITY_POLICY_ROOT": str(RELIABILITY_POLICY_ROOT),
        "IGRIS_RELIABILITY_PYTHON": str(RELIABILITY_PYTHON),
        "IGRIS_MEDIAPIPE_PYTHON": str(MEDIAPIPE_PYTHON),
    }


def _validate_reliability_policy_assets(*, hand_variant: str, controller_variant: str) -> None:
    required: set[Path] = set()
    for kind, variant in (("hand", hand_variant), ("controller", controller_variant)):
        if variant == "always_1":
            continue
        required.add(RELIABILITY_POLICY_ROOT / kind / f"{kind}_{variant}.joblib")
        if variant == "rnn":
            required.add(RELIABILITY_POLICY_ROOT / kind / f"{kind}_histgb.joblib")

    missing = sorted(path for path in required if not path.is_file())
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Reliability policy assets not found: {missing_text}")


def _clear_hybrid_preview_files(preview_root: Path = HYBRID_PREVIEW_ROOT) -> None:
    preview_root.mkdir(parents=True, exist_ok=True)
    for side in HYBRID_PREVIEW_SIDES:
        for stage in HYBRID_PREVIEW_STAGES:
            try:
                (preview_root / f"{side}_{stage}.jpg").unlink()
            except FileNotFoundError:
                pass
        try:
            (preview_root / f"{side}_status.json").unlink()
        except FileNotFoundError:
            pass


def _build_reliability_process_spec(
    *,
    hand_model_variant: str,
    controller_model_variant: str,
    hybrid_config: HybridTeleopConfig | None = None,
    preview_output_dir: Path = HYBRID_PREVIEW_ROOT,
) -> CommandProcessSpec:
    if not RELIABILITY_ROS_WS_ROOT.is_dir():
        raise FileNotFoundError(f"Reliability ROS2 workspace not found: {RELIABILITY_ROS_WS_ROOT}")
    if not ROS_JAZZY_SETUP_BASH.is_file():
        raise FileNotFoundError(f"ROS Jazzy setup.bash not found: {ROS_JAZZY_SETUP_BASH}")
    if not RELIABILITY_ROS_WS_SETUP_BASH.is_file():
        raise FileNotFoundError(f"Reliability workspace setup.bash not found: {RELIABILITY_ROS_WS_SETUP_BASH}")
    _validate_reliability_policy_assets(
        hand_variant=hand_model_variant,
        controller_variant=controller_model_variant,
    )

    hybrid = hybrid_config or HybridTeleopConfig()
    command_tokens = (
        "ros2",
        "launch",
        "igris_reliability_runtime",
        "reliability_teleop.launch.py",
        f"hand_model_variant:={hand_model_variant}",
        f"controller_model_variant:={controller_model_variant}",
        "left_controller_pose_topic:=/left_controller/poses",
        "right_controller_pose_topic:=/right_controller/poses",
        f"left_camera_device:={hybrid.left_device}",
        f"right_camera_device:={hybrid.right_device}",
        f"left_camera_mirror:={_launch_bool(hybrid.mirror)}",
        f"right_camera_mirror:={_launch_bool(hybrid.mirror)}",
        f"left_camera_swap_lr:={_launch_bool(hybrid.swap_lr)}",
        f"right_camera_swap_lr:={_launch_bool(hybrid.swap_lr)}",
        f"trapezoid_preprocess:={_launch_bool(hybrid.trapezoid_preprocess)}",
        f"trapezoid_bottom_width:={hybrid.trapezoid_bottom_width}",
        f"mediapipe_preview_output_dir:={preview_output_dir}",
    )
    return CommandProcessSpec(
        name="reliability_runtime",
        argv=build_ros_shell_argv(
            command_tokens,
            ros_setup_bash=ROS_JAZZY_SETUP_BASH,
            workspace_setup_bash=RELIABILITY_ROS_WS_SETUP_BASH,
        ),
        cwd=RELIABILITY_ROS_WS_ROOT,
        startup_grace_s=1.0,
        env=_hybrid_runtime_environment(),
    )


def _build_hybrid_test_process_spec(
    config: HybridTeleopConfig,
    *,
    preview_output_dir: Path = HYBRID_PREVIEW_ROOT,
) -> CommandProcessSpec:
    if not RELIABILITY_ROS_WS_ROOT.is_dir():
        raise FileNotFoundError(f"Reliability ROS2 workspace not found: {RELIABILITY_ROS_WS_ROOT}")
    if not RELIABILITY_ROS_WS_SETUP_BASH.is_file():
        raise FileNotFoundError(f"Reliability workspace setup.bash not found: {RELIABILITY_ROS_WS_SETUP_BASH}")
    command_tokens = (
        "ros2",
        "launch",
        "mediapipe_hand_pose_bridge",
        "mediapipe_dual_camera_hand_pose_bridge.launch.py",
        f"left_device:={config.left_device}",
        f"right_device:={config.right_device}",
        "model_complexity:=1",
        "max_hands:=1",
        f"left_mirror:={_launch_bool(config.mirror)}",
        f"right_mirror:={_launch_bool(config.mirror)}",
        f"left_swap_lr:={_launch_bool(config.swap_lr)}",
        f"right_swap_lr:={_launch_bool(config.swap_lr)}",
        f"trapezoid_preprocess:={_launch_bool(config.trapezoid_preprocess)}",
        f"trapezoid_bottom_width:={config.trapezoid_bottom_width}",
        f"preview_output_dir:={preview_output_dir}",
        "show_image:=false",
        "show_assignment_debug:=true",
        "draw_landmarks:=true",
        "publish_rate_hz:=30.0",
    )
    return CommandProcessSpec(
        name="hybrid_mediapipe_test",
        argv=build_ros_shell_argv(
            command_tokens,
            ros_setup_bash=ROS_JAZZY_SETUP_BASH,
            workspace_setup_bash=RELIABILITY_ROS_WS_SETUP_BASH,
        ),
        cwd=RELIABILITY_ROS_WS_ROOT,
        startup_grace_s=1.0,
        env=_hybrid_runtime_environment(),
    )


class ReliabilityRuntimeManager:
    def __init__(self, *, log_queue: Any = None) -> None:
        self._log_queue = log_queue
        self._process: ManagedCommandProcess | None = None
        self._hand_model_variant = DEFAULT_RELIABILITY_MODEL_VARIANT
        self._controller_model_variant = DEFAULT_RELIABILITY_MODEL_VARIANT
        self._started_at: float | None = None
        self._last_exit_code: int | None = None
        self._last_error = ""

    def status(self) -> dict[str, Any]:
        rc = self._poll()
        running = self._process is not None and rc is None
        uptime = time.time() - self._started_at if running and self._started_at is not None else None
        return {
            "running": running,
            "status": "ALIVE" if running else "STOPPED",
            "pid": self._process.pid if running and self._process is not None else None,
            "hand_model_variant": self._hand_model_variant,
            "controller_model_variant": self._controller_model_variant,
            "workspace": str(RELIABILITY_ROS_WS_ROOT),
            "policy_root": str(RELIABILITY_POLICY_ROOT),
            "reliability_python": str(RELIABILITY_PYTHON),
            "mediapipe_python": str(MEDIAPIPE_PYTHON),
            "started_at": self._started_at if running else None,
            "uptime_sec": round(float(uptime), 1) if uptime is not None else None,
            "last_exit_code": self._last_exit_code,
            "last_error": self._last_error,
        }

    def start(
        self,
        *,
        hand_model_variant: str,
        controller_model_variant: str,
        hybrid_config: HybridTeleopConfig,
        preview_output_dir: Path = HYBRID_PREVIEW_ROOT,
    ) -> dict[str, Any]:
        rc = self._poll()
        if self._process is not None and rc is None:
            return self.status()

        spec = _build_reliability_process_spec(
            hand_model_variant=hand_model_variant,
            controller_model_variant=controller_model_variant,
            hybrid_config=hybrid_config,
            preview_output_dir=preview_output_dir,
        )
        proc = ManagedCommandProcess(spec, log_queue=self._log_queue)
        try:
            proc.start()
        except Exception as exc:
            self._process = None
            self._last_exit_code = None
            self._last_error = str(exc)
            raise

        self._process = proc
        self._hand_model_variant = hand_model_variant
        self._controller_model_variant = controller_model_variant
        self._started_at = time.time()
        self._last_exit_code = None
        self._last_error = ""
        return self.status()

    def stop(self) -> dict[str, Any]:
        if self._process is not None:
            self._process.terminate()
            self._last_exit_code = self._process.poll()
            self._process = None
            self._started_at = None
        return self.status()

    def _poll(self) -> int | None:
        if self._process is None:
            return self._last_exit_code
        rc = self._process.poll()
        if rc is not None:
            self._last_exit_code = rc
            self._started_at = None
            try:
                self._process.wait(timeout=0.0)
            except Exception:
                pass
        return rc


class HybridMediaPipeTestManager:
    def __init__(self, *, log_queue: Any = None) -> None:
        self._log_queue = log_queue
        self._process: ManagedCommandProcess | None = None
        self._config = HybridTeleopConfig()
        self._started_at: float | None = None
        self._last_exit_code: int | None = None
        self._last_error = ""

    def status(self) -> dict[str, Any]:
        rc = self._poll()
        running = self._process is not None and rc is None
        uptime = time.time() - self._started_at if running and self._started_at is not None else None
        return {
            "running": running,
            "status": "ALIVE" if running else "STOPPED",
            "pid": self._process.pid if running and self._process is not None else None,
            "config": self._config.as_dict(),
            "uptime_sec": round(float(uptime), 1) if uptime is not None else None,
            "last_exit_code": self._last_exit_code,
            "last_error": self._last_error,
        }

    def start(self, config: HybridTeleopConfig) -> dict[str, Any]:
        self.stop()
        _clear_hybrid_preview_files()
        spec = _build_hybrid_test_process_spec(config)
        proc = ManagedCommandProcess(spec, log_queue=self._log_queue)
        try:
            proc.start()
        except Exception as exc:
            self._process = None
            self._last_exit_code = None
            self._last_error = str(exc)
            raise
        self._process = proc
        self._config = config
        self._started_at = time.time()
        self._last_exit_code = None
        self._last_error = ""
        return self.status()

    def stop(self) -> dict[str, Any]:
        if self._process is not None:
            self._process.terminate()
            self._last_exit_code = self._process.poll()
            self._process = None
            self._started_at = None
        return self.status()

    def _poll(self) -> int | None:
        if self._process is None:
            return self._last_exit_code
        rc = self._process.poll()
        if rc is not None:
            self._last_exit_code = rc
            self._started_at = None
            if not self._last_error:
                self._last_error = (
                    f"MediaPipe test exited (code {rc}); check camera devices and runtime logs"
                )
            try:
                self._process.wait(timeout=0.0)
            except Exception:
                pass
        return rc


class _LocalLeRobotParquetDataset:
    def __init__(self, dataset_dir: Path, info: Mapping[str, Any], episode: int) -> None:
        self.root = dataset_dir.expanduser().resolve()
        self.info = dict(info)
        self.episode = int(episode)
        self.image_keys = _image_keys_from_info(self.info)
        self.video_keys = {
            key
            for key, spec in _feature_dict(self.info).items()
            if isinstance(spec, Mapping) and str(spec.get("dtype", "")).strip().lower() == "video"
        }
        self._video_files_by_key: dict[str, list[Path]] = {}
        self.episode_metadata = self._load_episode_metadata()
        self.rows = self._load_rows()
        # Match the part of HuggingFace Dataset used by _load_dataset_viewer_episode_arrays.
        self.hf_dataset = [dict(row) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = dict(self.rows[int(idx)])
        for image_key in self.image_keys:
            if image_key in row and image_key not in self.video_keys:
                decoded = _decode_dataset_image_value(row.get(image_key), self.root)
                if decoded is not None:
                    row[image_key] = decoded
                continue
            row[image_key] = self._load_video_frame(row, image_key)
        return row

    def close(self) -> None:
        return None

    def _load_rows(self) -> list[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Dataset viewer needs either `lerobot` or `pyarrow` to read local episodes. "
                "Run `./igris_teleop/setup_runtime_venv.sh` and launch the web UI with `.venv/bin/python`, "
                "or install `lerobot` in the current Python environment."
            ) from exc

        parquet_files = self._episode_parquet_files()
        rows: list[dict[str, Any]] = []
        for parquet_path in parquet_files:
            table = pq.read_table(parquet_path)
            for row in table.to_pylist():
                if not isinstance(row, dict):
                    continue
                try:
                    episode_idx = int(np.asarray(row.get("episode_index", self.episode)).reshape(()).item())
                except Exception:
                    episode_idx = self.episode
                if episode_idx == self.episode:
                    rows.append(dict(row))
        rows.sort(key=lambda row: int(np.asarray(row.get("frame_index", 0)).reshape(()).item()))
        return rows

    def _episode_parquet_files(self) -> list[Path]:
        data_dir = self.root / "data"
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset data directory: {data_dir}")
        exact_names = {
            f"episode_{self.episode:06d}.parquet",
            f"episode_{self.episode}.parquet",
        }
        exact = sorted(path for path in data_dir.rglob("*.parquet") if path.name in exact_names)
        if exact:
            return exact
        all_files = sorted(data_dir.rglob("*.parquet"))
        if not all_files:
            raise FileNotFoundError(f"No parquet files found under {data_dir}")
        return all_files

    def _load_episode_metadata(self) -> dict[int, dict[str, Any]]:
        meta_dir = self.root / "meta" / "episodes"
        if not meta_dir.is_dir():
            return {}
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception:
            return {}

        columns = {
            "episode_index",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
        }
        for image_key in self.image_keys:
            prefix = f"videos/{image_key}"
            columns.update(
                {
                    f"{prefix}/chunk_index",
                    f"{prefix}/file_index",
                    f"{prefix}/from_timestamp",
                    f"{prefix}/to_timestamp",
                }
            )

        metadata: dict[int, dict[str, Any]] = {}
        for parquet_path in sorted(meta_dir.rglob("*.parquet")):
            try:
                schema_names = set(pq.read_schema(parquet_path).names)
                selected_columns = [column for column in columns if column in schema_names]
                if "episode_index" not in selected_columns:
                    continue
                table = pq.read_table(parquet_path, columns=selected_columns)
            except Exception as exc:
                logger.debug("failed to read dataset episode metadata %s: %s", parquet_path, exc)
                continue
            for row in table.to_pylist():
                if not isinstance(row, dict):
                    continue
                episode_idx = self._optional_int(row.get("episode_index"))
                if episode_idx is not None:
                    metadata[episode_idx] = row
        return metadata

    def _load_video_frame(self, row: Mapping[str, Any], image_key: str) -> np.ndarray:
        video_path = self._resolve_video_path(row, image_key)
        fps = float(self.info.get("fps", 30.0) or 30.0)
        frame_index, timestamp = self._video_frame_position(row, image_key, fps=fps)
        codec = ""
        feature_spec = _feature_dict(self.info).get(image_key, {})
        if isinstance(feature_spec, Mapping):
            feature_info = feature_spec.get("info", {})
            if isinstance(feature_info, Mapping):
                codec = str(feature_info.get("video.codec", "") or "").strip().lower()
        if codec == "av1":
            return self._load_video_frame_ffmpeg(video_path, frame_index=frame_index, fps=fps, timestamp=timestamp)
        if cv2 is None:
            return self._load_video_frame_ffmpeg(video_path, frame_index=frame_index, fps=fps, timestamp=timestamp)

        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                return self._load_video_frame_ffmpeg(video_path, frame_index=frame_index, fps=fps, timestamp=timestamp)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                return self._load_video_frame_ffmpeg(video_path, frame_index=frame_index, fps=fps, timestamp=timestamp)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    @staticmethod
    def _load_video_frame_ffmpeg(
        video_path: Path,
        *,
        frame_index: int,
        fps: float,
        timestamp: float | None = None,
    ) -> np.ndarray:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                f"Failed to read frame {frame_index} from {video_path}; "
                "install ffmpeg or use a Python environment with AV1-capable video decoding."
            )
        video_timestamp = max(
            0.0,
            float(timestamp) if timestamp is not None else float(frame_index) / max(float(fps), 1.0),
        )
        commands = [
            [
                ffmpeg,
                "-hide_banner",
                "-v",
                "error",
                "-ss",
                f"{video_timestamp:.6f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "pipe:1",
            ],
            [
                ffmpeg,
                "-hide_banner",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"select=eq(n\\,{max(0, int(frame_index))})",
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "pipe:1",
            ],
        ]
        stderr_text = ""
        for command in commands:
            proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stderr_text = proc.stderr.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0 or not proc.stdout:
                continue
            arr = np.frombuffer(proc.stdout, dtype=np.uint8)
            if cv2 is not None:
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if Image is not None:
                import io

                with Image.open(io.BytesIO(proc.stdout)) as image:
                    return np.asarray(image.convert("RGB"))
        detail = f": {stderr_text}" if stderr_text else ""
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}{detail}")

    def _video_frame_position(
        self,
        row: Mapping[str, Any],
        image_key: str,
        *,
        fps: float,
    ) -> tuple[int, float | None]:
        local_frame_index = self._optional_int(row.get("frame_index")) or 0
        local_timestamp = self._optional_float(row.get("timestamp"))
        episode_idx = self._row_episode_index(row)
        episode_meta = self.episode_metadata.get(episode_idx, {})
        video_from_timestamp = self._optional_float(
            episode_meta.get(f"videos/{image_key}/from_timestamp") if episode_meta else None
        )
        if local_timestamp is not None and video_from_timestamp is not None:
            video_timestamp = max(0.0, video_from_timestamp + local_timestamp)
            return max(0, int(round(video_timestamp * max(float(fps), 1.0)))), video_timestamp

        global_index = self._optional_int(row.get("index"))
        if global_index is not None and len(self._video_files_for_key(image_key)) == 1:
            return max(0, global_index), max(0.0, float(global_index) / max(float(fps), 1.0))

        timestamp = local_timestamp
        if timestamp is None:
            timestamp = float(local_frame_index) / max(float(fps), 1.0)
        return max(0, local_frame_index), max(0.0, timestamp)

    def _resolve_video_path(self, row: Mapping[str, Any], image_key: str) -> Path:
        chunks_size = int(self.info.get("chunks_size", 1) or 1)
        episode_idx = self._row_episode_index(row)
        episode_meta = self.episode_metadata.get(episode_idx, {})
        video_chunk_index = self._optional_int(
            episode_meta.get(f"videos/{image_key}/chunk_index") if episode_meta else None
        )
        video_file_index = self._optional_int(
            episode_meta.get(f"videos/{image_key}/file_index") if episode_meta else None
        )
        data_chunk_index = self._optional_int(episode_meta.get("data/chunk_index") if episode_meta else None)
        data_file_index = self._optional_int(episode_meta.get("data/file_index") if episode_meta else None)
        row_chunk_index = self._optional_int(row.get("_chunk_index"))
        row_file_index = self._optional_int(row.get("_file_index"))
        chunk_index = next(
            value
            for value in (
                row_chunk_index,
                video_chunk_index,
                data_chunk_index,
                episode_idx // max(chunks_size, 1),
            )
            if value is not None
        )
        file_index = next(
            value
            for value in (
                row_file_index,
                video_file_index,
                data_file_index,
                episode_idx % max(chunks_size, 1),
            )
            if value is not None
        )
        template = str(
            self.info.get(
                "video_path",
                "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            )
        )

        format_payload = {
            "video_key": image_key,
            "chunk_index": chunk_index,
            "file_index": file_index,
            "episode_index": episode_idx,
            "episode_chunk": episode_idx // max(chunks_size, 1),
        }
        candidates: list[Path] = []
        try:
            candidates.append(self.root / template.format(**format_payload))
        except Exception:
            pass
        candidates.extend(
            [
                self.root / "videos" / image_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4",
                self.root / "videos" / image_key / f"chunk-{chunk_index:03d}" / f"episode_{episode_idx:06d}.mp4",
                self.root / "videos" / image_key / f"episode_{episode_idx:06d}.mp4",
                self.root / "videos" / image_key / "chunk-000" / f"episode_{episode_idx:06d}.mp4",
            ]
        )
        video_files = self._video_files_for_key(image_key)
        if len(video_files) == 1:
            candidates.append(video_files[0])
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_file():
                return resolved
        raise FileNotFoundError(f"No video file found for {image_key} episode {episode_idx}")

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            arr = np.asarray(value)
            if arr.size == 0:
                return None
            return int(arr.reshape(-1)[0].item())
        except Exception:
            try:
                return int(value)
            except Exception:
                return None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            arr = np.asarray(value)
            if arr.size == 0:
                return None
            return float(arr.reshape(-1)[0].item())
        except Exception:
            try:
                return float(value)
            except Exception:
                return None

    def _row_episode_index(self, row: Mapping[str, Any]) -> int:
        return self._optional_int(row.get("episode_index")) or self.episode

    def _video_files_for_key(self, image_key: str) -> list[Path]:
        if image_key not in self._video_files_by_key:
            video_dir = self.root / "videos" / image_key
            if video_dir.is_dir():
                self._video_files_by_key[image_key] = sorted(
                    path.resolve() for path in video_dir.rglob("*.mp4") if path.is_file()
                )
            else:
                self._video_files_by_key[image_key] = []
        return self._video_files_by_key[image_key]


@dataclass
class WebUIState:
    bus: EventBus
    supervisor: Any
    shared_memory: Mapping[str, Any]
    log_queue: Any
    initial_mode: str | None
    initial_teleop_device: str | None
    initial_teleop_hand_source: str | None
    initial_walking_policy_profile: str | None
    initial_walking_policy_path: str | None
    initial_camera_mode: str = DEFAULT_CAMERA_MODE
    lock: threading.RLock = field(default_factory=threading.RLock)
    selected_mode: str | None = None
    selected_teleop_device: str | None = None
    selected_teleop_hand_source: str | None = None
    selected_inference_policy: str | None = DEFAULT_INFERENCE_POLICY
    selection_locked: bool = False
    applied_mode: str | None = None
    applied_teleop_device: str | None = None
    applied_teleop_hand_source: str | None = None
    applied_inference_policy: str | None = None
    applied_camera_mode: str = DEFAULT_CAMERA_MODE
    collect_dataset_repo_id: str = "IGRIS_C"
    collect_tasks: list[str] = field(default_factory=lambda: ["Pick and Place"])
    selected_collect_task: str | None = "Pick and Place"
    manual_one_shot_started: set[str] = field(default_factory=set)
    recent_logs: deque[str] = field(default_factory=lambda: deque(maxlen=500))
    input_monitor_fingerprints: dict[str, tuple[Any, ...] | None] = field(default_factory=dict)
    status_message: str = ""
    viser_url: str = field(default_factory=_default_viser_url)
    dataset_viewer_cache_key: tuple[str, int] | None = None
    dataset_viewer_cache_dataset: Any = None
    dataset_viewer_cache_rows: list[int] = field(default_factory=list)
    dataset_viewer_frame_cache: OrderedDict[tuple[str, int], bytes] = field(default_factory=OrderedDict)
    reliability_runtime: ReliabilityRuntimeManager = field(init=False)
    hybrid_test_runtime: HybridMediaPipeTestManager = field(init=False)
    hybrid_config: HybridTeleopConfig = field(init=False)

    def __post_init__(self) -> None:
        repo_id, tasks = _load_collect_defaults()
        self.collect_dataset_repo_id = repo_id
        self.collect_tasks = tasks
        self.selected_collect_task = tasks[0] if tasks else None
        self.selected_mode = self.initial_mode
        self.selected_teleop_device = self.initial_teleop_device
        self.selected_teleop_hand_source = self.initial_teleop_hand_source
        self.applied_camera_mode = self.initial_camera_mode
        self.reliability_runtime = ReliabilityRuntimeManager(log_queue=self.log_queue)
        self.hybrid_test_runtime = HybridMediaPipeTestManager(log_queue=self.log_queue)
        self.hybrid_config = _load_hybrid_config()
        sim_config_shm = self._shm("sim_config_shm")
        if sim_config_shm is not None:
            try:
                sim_config = sim_config_shm.read_data()
                raw_baseline = float(
                    np.asarray(sim_config.get("stereo_baseline_m")).reshape(-1)[0]
                )
                validate_stereo_baseline(raw_baseline)
            except (TypeError, ValueError, IndexError):
                sim_config_shm.write_data(stereo_baseline_m=STEREO_CAMERA_BASELINE_M)

    def _shm(self, name: str) -> Any:
        return self.shared_memory.get(name)

    def _read_sim_stereo_baseline_m(self) -> float:
        data = self._safe_read_shm("sim_config_shm") or {}
        try:
            value = float(np.asarray(data.get("stereo_baseline_m")).reshape(-1)[0])
            return validate_stereo_baseline(value)
        except (TypeError, ValueError, IndexError):
            return STEREO_CAMERA_BASELINE_M

    def set_sim_stereo(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not bool(self._worker_alive().get("simulator", False)):
                raise ValueError("Start the simulator before changing sim eye distance")
            sim_config_shm = self._shm("sim_config_shm")
            if sim_config_shm is None:
                raise ValueError("Simulation camera configuration is unavailable")
            raw_baseline = payload.get("baseline_m")
            if raw_baseline is None:
                raise ValueError("baseline_m is required")
            baseline_m = validate_stereo_baseline(float(raw_baseline))
            sim_config_shm.write_data(stereo_baseline_m=baseline_m)
            self.status_message = (
                f"Sim eye distance: {baseline_m * 1000.0:.0f} mm (translation only)"
            )
            return self.status()

    def _experiment_status(self, alive: Mapping[str, bool]) -> dict[str, Any]:
        enabled = bool(alive.get("simulator", False))
        data = self._safe_read_shm("sim_config_shm") or {}

        def _scene_int(name: str) -> int:
            try:
                return int(round(float(np.asarray(data.get(name, 0.0)).reshape(()))))
            except (TypeError, ValueError):
                return 0

        command_seq = _scene_int("scene_command_seq")
        applied_seq = _scene_int("scene_applied_seq")
        requested_task_id = _scene_int("scene_task_id")
        active_task_id = _scene_int("scene_active_task_id")
        status_code = _scene_int("scene_status_code")
        pending = enabled and command_seq != applied_seq

        if not enabled:
            state = "SIM_OFF"
        elif status_code < 0 and not pending:
            state = "ERROR"
        elif pending:
            state = "PENDING"
        elif active_task_id in EXPERIMENT_TASK_BY_ID:
            state = "ACTIVE"
        else:
            state = "EMPTY"

        active_task = EXPERIMENT_TASK_BY_ID.get(active_task_id)
        return {
            "enabled": enabled,
            "state": state,
            "pending": pending,
            "command_seq": command_seq,
            "applied_seq": applied_seq,
            "requested_task_id": requested_task_id,
            "active_task_id": active_task_id,
            "active_task_name": active_task.name if active_task is not None else None,
            "tasks": experiment_tasks_payload(),
        }

    def set_experiment_scene(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not bool(self._worker_alive().get("simulator", False)):
                raise ValueError("Start the simulator before changing the experiment scene")
            sim_config_shm = self._shm("sim_config_shm")
            if sim_config_shm is None:
                raise ValueError("Simulation scene configuration is unavailable")

            action = str(payload.get("action") or "").strip().lower()
            if action == "all_reset":
                task_id = 0
            elif action in {"spawn", "task_reset"}:
                try:
                    task_id = int(payload.get("task_id"))
                except (TypeError, ValueError) as exc:
                    raise ValueError("task_id is required") from exc
                if task_id not in EXPERIMENT_TASK_BY_ID:
                    raise ValueError(f"Unknown experiment task id: {task_id}")
            else:
                raise ValueError(f"Unknown experiment scene action: {action}")

            current = self._safe_read_shm("sim_config_shm") or {}

            def _current_seq(name: str) -> int:
                try:
                    return int(round(float(np.asarray(current.get(name, 0.0)).reshape(()))))
                except (TypeError, ValueError):
                    return 0

            command_seq = max(
                _current_seq("scene_command_seq"),
                _current_seq("scene_applied_seq"),
            ) + 1
            sim_config_shm.write_data(
                scene_command_seq=float(command_seq),
                scene_task_id=float(task_id),
                scene_status_code=1.0,
            )

            if task_id == 0:
                self.status_message = "Experiment scene cleared"
            else:
                task = EXPERIMENT_TASK_BY_ID[task_id]
                verb = "reset requested" if action == "task_reset" else "spawn requested"
                self.status_message = f"{task.name}: {verb}"
            return self.status()

    def _drain_logs(self) -> None:
        if self.log_queue is None:
            return
        for _ in range(250):
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            self.recent_logs.append(str(line))

    def _worker_alive(self) -> dict[str, bool]:
        return dict(self.supervisor.alive_map())

    def _manual_running(self) -> set[str]:
        return set(self.supervisor.current_manual_workers())

    def _mode_running(self) -> set[str]:
        return set(self.supervisor.current_mode_workers())

    def _manual_group_status(self, alive: Mapping[str, bool], running: set[str]) -> list[dict[str, Any]]:
        available = set(self.supervisor.list_manual_workers())
        groups: list[dict[str, Any]] = []
        for group in MANUAL_WORKER_GROUPS:
            worker_names = _manual_worker_names_for_group(
                group,
                self.selected_mode,
                self.selected_teleop_device,
            )
            enabled = all(worker_name in available for worker_name in worker_names)
            alive_flags = [bool(alive.get(worker_name, False)) for worker_name in worker_names]
            if alive_flags and all(alive_flags):
                status = "ALIVE"
            elif any(alive_flags):
                status = "PARTIAL"
            else:
                status = "STOPPED"
            group_running = all(worker_name in running for worker_name in worker_names)
            if group in MANUAL_ONE_SHOT_GROUPS and (group_running or any(alive_flags)):
                self.manual_one_shot_started.add(group)
            can_toggle = enabled and not (
                group in MANUAL_ONE_SHOT_GROUPS and group in self.manual_one_shot_started
            )
            groups.append(
                {
                    "name": group,
                    "label": MANUAL_WORKER_GROUP_LABELS.get(group, group),
                    "workers": list(worker_names),
                    "enabled": enabled,
                    "running": group_running,
                    "status": status,
                    "can_toggle": can_toggle,
                }
            )
        return groups

    def _mode_applied_for_level_events(self) -> bool:
        return self.selection_locked and self.applied_mode is not None

    def _start_level_controls_enabled(self) -> bool:
        return self._mode_applied_for_level_events() and self.bus.is_level_set("ready")

    def _enforce_level_event_gate(self) -> None:
        if not self._mode_applied_for_level_events():
            self.bus.clear_level("ready")
            self.bus.clear_level("start")
            self.bus.clear_level("home")
            self.bus.clear_level("hand_init")
            return
        if not self.bus.is_level_set("ready"):
            self.bus.clear_level("start")
            self.bus.clear_level("home")
            self.bus.clear_level("hand_init")

    def _mode_worker_status(
        self,
        mode: str | None,
        teleop_device: str | None,
        inference_policy: str | None,
        alive: Mapping[str, bool],
        running: set[str],
    ) -> dict[str, Any]:
        if mode is None:
            return {"hint": "mode를 먼저 선택하세요.", "workers": []}
        if mode == "teleop" and teleop_device is None:
            return {"hint": "teleop device를 선택하세요.", "workers": []}
        candidates = self.supervisor.list_selectable_mode_workers(
            mode,
            teleop_device,
            inference_policy=inference_policy,
        )
        if not candidates:
            return {"hint": "선택 가능한 mode worker가 없습니다.", "workers": []}
        return {
            "hint": "",
            "workers": [
                {
                    "name": worker,
                    "alive": bool(alive.get(worker, False)),
                    "running": worker in running,
                    "status": "ALIVE" if alive.get(worker, False) else "STOPPED",
                }
                for worker in candidates
            ],
        }

    def selectable_mode_workers(
        self,
        mode: str | None,
        teleop_device: str | None,
        inference_policy: str | None = None,
    ) -> list[str]:
        with self.lock:
            return self.supervisor.list_selectable_mode_workers(mode, teleop_device, inference_policy)

    def file_browser(
        self,
        *,
        kind: str,
        raw_path: str | None = None,
        profile: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            browser = self._file_browser_config(kind, profile)
            browse_root: Path = browser["browse_root"]
            start_root: Path = browser["start_root"]
            value_root: Path = browser["value_root"]
            select_type: str = browser["select_type"]
            absolute_value: bool = browser["absolute_value"]
            extensions: tuple[str, ...] = browser["extensions"]

            cwd = self._resolve_browser_cwd(
                browse_root=browse_root,
                start_root=start_root,
                value_root=value_root,
                raw_path=raw_path,
            )
            entries: list[dict[str, Any]] = []
            try:
                children = sorted(
                    cwd.iterdir(),
                    key=lambda path: (not path.is_dir(), path.name.lower()),
                )
            except OSError as exc:
                raise ValueError(f"Cannot read directory: {cwd}") from exc

            for child in children:
                try:
                    resolved = child.resolve()
                except OSError:
                    continue
                if not _path_is_relative_to(resolved, browse_root):
                    continue
                is_dir = resolved.is_dir()
                is_file = resolved.is_file()
                if not is_dir and not is_file:
                    continue
                if is_file and select_type != "file":
                    continue
                selectable = (select_type == "directory" and is_dir) or (
                    select_type == "file"
                    and is_file
                    and (not extensions or resolved.suffix.lower() in extensions)
                )
                entries.append(
                    {
                        "name": resolved.name,
                        "path": str(resolved),
                        "display": _display_path(resolved, base_root=value_root, absolute=absolute_value),
                        "type": "directory" if is_dir else "file",
                        "selectable": selectable,
                    }
                )

            parent = cwd.parent.resolve()
            parent_value = None
            if cwd != browse_root and _path_is_relative_to(parent, browse_root):
                parent_value = str(parent)
            return {
                "kind": kind,
                "title": browser["title"],
                "select_type": select_type,
                "base_root": str(browse_root),
                "base_display": str(browse_root),
                "preferred_root": str(value_root),
                "cwd": str(cwd),
                "cwd_display": _display_path(cwd, base_root=value_root),
                "parent": parent_value,
                "current_selectable": select_type == "directory",
                "current_value": _display_path(cwd, base_root=value_root, absolute=absolute_value),
                "entries": entries,
            }

    def _file_browser_config(self, kind: str, profile: str | None) -> dict[str, Any]:
        if kind not in FILE_BROWSER_KINDS:
            raise ValueError(f"Unsupported file browser kind={kind!r}")
        if kind in {"inference_dataset", "replay_dataset"}:
            dataset_root = _ensure_browser_root(DATASETS_ROOT)
            browse_root = _browser_root_containing(dataset_root, Path.home())
            return {
                "title": "Select dataset folder",
                "browse_root": browse_root,
                "start_root": dataset_root,
                "value_root": dataset_root,
                "select_type": "directory",
                "absolute_value": False,
                "extensions": (),
            }
        if kind == "dataset_viewer_root":
            dataset_root = _ensure_browser_root(DATASETS_ROOT)
            browse_root = _browser_root_containing(dataset_root, Path.home())
            return {
                "title": "Select dataset root or dataset folder",
                "browse_root": browse_root,
                "start_root": dataset_root,
                "value_root": dataset_root,
                "select_type": "directory",
                "absolute_value": True,
                "extensions": (),
            }
        if kind == "inference_checkpoint":
            checkpoint_root = _ensure_browser_root(CHECKPOINTS_ROOT)
            browse_root = _browser_root_containing(checkpoint_root, Path.home())
            return {
                "title": "Select checkpoint/pretrained_model folder",
                "browse_root": browse_root,
                "start_root": checkpoint_root,
                "value_root": checkpoint_root,
                "select_type": "directory",
                "absolute_value": False,
                "extensions": (),
            }

        resolved_profile = resolve_walking_policy_profile(profile)
        default_path = default_walking_policy_path(resolved_profile)
        value_root = _ensure_browser_root(default_path.parent)
        browse_root = _browser_root_containing(value_root, Path.home())
        return {
            "title": "Select walking model",
            "browse_root": browse_root,
            "start_root": value_root,
            "value_root": value_root,
            "select_type": "file",
            "absolute_value": True,
            "extensions": (".pt", ".onnx"),
        }

    @staticmethod
    def _resolve_browser_cwd(
        *,
        browse_root: Path,
        start_root: Path,
        value_root: Path,
        raw_path: str | None,
    ) -> Path:
        browse_root = browse_root.resolve()
        start_root = start_root.resolve()
        value_root = value_root.resolve()

        def _fallback() -> Path:
            for candidate in (start_root, value_root, Path.home(), browse_root):
                try:
                    resolved = candidate.resolve()
                except OSError:
                    continue
                if resolved.is_dir() and _path_is_relative_to(resolved, browse_root):
                    return resolved
            return browse_root

        text = str(raw_path or "").strip()
        if not text:
            return _fallback()

        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = value_root / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            try:
                resolved = candidate.parent.resolve()
            except OSError:
                return _fallback()

        if resolved.is_file():
            resolved = resolved.parent.resolve()
        if not resolved.exists():
            try:
                resolved = resolved.parent.resolve()
            except OSError:
                return _fallback()
        if not resolved.is_dir() or not _path_is_relative_to(resolved, browse_root):
            return _fallback()
        return resolved

    def _dataset_viewer_root(self, raw_root: str | None = None) -> Path:
        text = str(raw_root or "").strip()
        candidate = Path(text).expanduser() if text else DATASETS_ROOT
        if not candidate.is_absolute():
            candidate = DATASETS_ROOT / candidate
        return candidate.resolve()

    def _resolve_dataset_viewer_dataset_dir(self, raw_dataset: str, root: Path) -> Path:
        text = str(raw_dataset or "").strip()
        if not text:
            raise ValueError("dataset is required")
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise ValueError(f"Cannot resolve dataset path: {candidate}") from exc
        if not (resolved / "meta" / "info.json").is_file():
            raise ValueError(f"Not a LeRobot dataset folder: {resolved}")
        return resolved

    def dataset_viewer_datasets(self, *, root: str | None = None) -> dict[str, Any]:
        with self.lock:
            dataset_root = self._dataset_viewer_root(root)
            datasets: list[dict[str, Any]] = []
            for dataset_dir in _find_lerobot_dataset_dirs(dataset_root):
                try:
                    info = _load_dataset_info_json(dataset_dir)
                except Exception as exc:
                    datasets.append(
                        {
                            "name": dataset_dir.name,
                            "path": str(dataset_dir),
                            "display": _display_path(dataset_dir, base_root=dataset_root),
                            "error": str(exc),
                        }
                    )
                    continue
                display = _display_path(dataset_dir, base_root=dataset_root)
                if display == ".":
                    display = dataset_dir.name
                datasets.append(
                    {
                        "name": dataset_dir.name,
                        "path": str(dataset_dir),
                        "display": display,
                        "total_episodes": int(info.get("total_episodes", 0) or 0),
                        "total_frames": int(info.get("total_frames", 0) or 0),
                        "fps": float(info.get("fps", 30.0) or 30.0),
                        "image_keys": _image_keys_from_info(info),
                    }
                )
            return {
                "ok": True,
                "root": str(dataset_root),
                "datasets": datasets,
                "truncated": len(datasets) >= DATASET_VIEWER_MAX_DATASETS,
            }

    def _close_dataset_viewer_cache(self) -> None:
        dataset = self.dataset_viewer_cache_dataset
        self.dataset_viewer_cache_key = None
        self.dataset_viewer_cache_dataset = None
        self.dataset_viewer_cache_rows = []
        self.dataset_viewer_frame_cache.clear()
        if dataset is None:
            return
        close_fn = getattr(dataset, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    def _open_dataset_viewer_episode(self, dataset_dir: Path, episode: int) -> tuple[Any, list[int]]:
        key = (str(dataset_dir.resolve()), int(episode))
        if self.dataset_viewer_cache_key == key and self.dataset_viewer_cache_dataset is not None:
            return self.dataset_viewer_cache_dataset, list(self.dataset_viewer_cache_rows)

        self._close_dataset_viewer_cache()
        dataset = self._open_lerobot_dataset_or_local_fallback(dataset_dir, int(episode))

        row_positions = self._dataset_viewer_episode_rows(dataset, int(episode))
        if not row_positions:
            close_fn = getattr(dataset, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            raise RuntimeError(f"Episode {episode} has no frames.")

        self.dataset_viewer_cache_key = key
        self.dataset_viewer_cache_dataset = dataset
        self.dataset_viewer_cache_rows = list(row_positions)
        self.dataset_viewer_frame_cache.clear()
        return dataset, list(row_positions)

    def _open_lerobot_dataset_or_local_fallback(self, dataset_dir: Path, episode: int) -> Any:
        try:
            dataset_cls = _require_lerobot_dataset()
        except RuntimeError as lerobot_exc:
            logger.info("lerobot unavailable; using local parquet dataset reader: %s", lerobot_exc)
            return self._open_local_parquet_dataset(dataset_dir, episode)

        repo_id = dataset_dir.name
        errors: list[str] = []
        constructors = (
            lambda: dataset_cls(repo_id=repo_id, root=dataset_dir, episodes=[int(episode)]),
            lambda: dataset_cls(repo_id, root=dataset_dir, episodes=[int(episode)]),
            lambda: dataset_cls(str(dataset_dir), episodes=[int(episode)]),
            lambda: dataset_cls(dataset_dir, episodes=[int(episode)]),
        )

        for ctor in constructors:
            try:
                return ctor()
            except TypeError as exc:
                errors.append(str(exc))
                continue
            except Exception as exc:
                text = str(exc)
                if "https://huggingface.co/api/datasets" in text:
                    errors.append("constructor attempted remote HF lookup")
                    continue
                raise

        try:
            return self._open_local_parquet_dataset(dataset_dir, episode)
        except Exception as fallback_exc:
            detail = errors[-1] if errors else "unknown constructor error"
            raise RuntimeError(
                f"Failed to open local dataset at {dataset_dir}: {detail}; "
                f"fallback reader also failed: {fallback_exc}"
            ) from fallback_exc

    def _open_local_parquet_dataset(self, dataset_dir: Path, episode: int) -> Any:
        info = _load_dataset_info_json(dataset_dir)
        return _LocalLeRobotParquetDataset(dataset_dir, info, episode)

    @staticmethod
    def _dataset_viewer_episode_rows(dataset: Any, episode: int) -> list[int]:
        rows: list[int] = []
        hf_dataset = getattr(dataset, "hf_dataset", None)
        if hf_dataset is not None:
            try:
                rows = [
                    pos
                    for pos, episode_idx in enumerate(hf_dataset["episode_index"])
                    if int(episode_idx) == int(episode)
                ]
            except Exception:
                rows = []
        if rows:
            return rows
        try:
            return list(range(len(dataset)))
        except Exception:
            return []

    def _load_dataset_viewer_episode_arrays(
        self,
        dataset: Any,
        row_positions: list[int],
        ee_feature_key: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_values: list[np.ndarray] = []
        act_values: list[np.ndarray] = []
        ee_values: list[np.ndarray] = []
        label_values: list[int] = []
        frame_numbers: list[int] = []
        timestamps: list[float] = []

        hf_dataset = getattr(dataset, "hf_dataset", None)
        ee_candidates = tuple(key for key in (ee_feature_key, *EE_POSE_KEY_CANDIDATES) if key)
        for row_pos in row_positions:
            row = None
            if hf_dataset is not None:
                try:
                    row = hf_dataset[int(row_pos)]
                except Exception:
                    row = None
            if row is None:
                row = dataset[int(row_pos)]

            obs_values.append(_extract_dataset_vector(row, OBSERVATION_KEY_CANDIDATES))
            act_values.append(_extract_dataset_vector(row, ACTION_KEY_CANDIDATES))
            ee_values.append(_extract_dataset_vector(row, ee_candidates))
            label_values.append(_extract_dataset_label_id(row))
            frame_numbers.append(int(np.asarray(_row_value(row, "frame_index", row_pos)).reshape(()).item()))
            timestamps.append(float(np.asarray(_row_value(row, "timestamp", row_pos / 30.0)).reshape(()).item()))

        max_obs_dim = max((arr.size for arr in obs_values), default=0)
        max_act_dim = max((arr.size for arr in act_values), default=0)
        max_ee_dim = max((arr.size for arr in ee_values), default=0)
        obs_series = np.full((len(obs_values), max_obs_dim), np.nan, dtype=np.float32)
        act_series = np.full((len(act_values), max_act_dim), np.nan, dtype=np.float32)
        ee_series = np.full((len(ee_values), max_ee_dim), np.nan, dtype=np.float32)
        for idx, arr in enumerate(obs_values):
            if arr.size:
                obs_series[idx, : arr.size] = arr
        for idx, arr in enumerate(act_values):
            if arr.size:
                act_series[idx, : arr.size] = arr
        for idx, arr in enumerate(ee_values):
            if arr.size:
                ee_series[idx, : arr.size] = arr
        return (
            obs_series,
            act_series,
            ee_series,
            np.asarray(label_values, dtype=np.int64),
            np.asarray(frame_numbers, dtype=np.int64),
            np.asarray(timestamps, dtype=np.float32),
        )

    @staticmethod
    def _dataset_viewer_plot_payloads(
        obs_series: np.ndarray,
        act_series: np.ndarray,
        frame_count: int,
        sample_indices: np.ndarray,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        cursor = 0
        max_dim = max(
            int(obs_series.shape[1]) if obs_series.ndim == 2 else 0,
            int(act_series.shape[1]) if act_series.ndim == 2 else 0,
        )
        for name, length, labels in _dataset_segment_specs():
            obs_seg = _slice_dataset_feature(obs_series, cursor, cursor + length)
            act_seg = _slice_dataset_feature(act_series, cursor, cursor + length)
            seg_dim = max(
                int(obs_seg.shape[1]) if obs_seg.ndim == 2 else 0,
                int(act_seg.shape[1]) if act_seg.ndim == 2 else 0,
            )
            if cursor >= max_dim:
                seg_dim = 0
                obs_seg = np.zeros((frame_count, 0), dtype=np.float32)
                act_seg = np.zeros((frame_count, 0), dtype=np.float32)
            payloads.append(
                {
                    "name": name,
                    "title": name.title(),
                    "labels": list(labels[:seg_dim]),
                    "obs": _json_float_matrix(obs_seg, sample_indices),
                    "act": _json_float_matrix(act_seg, sample_indices),
                }
            )
            cursor += length
        return payloads

    def dataset_viewer_episode(
        self,
        *,
        root: str | None = None,
        dataset: str,
        episode: int,
        image_key: str | None = None,
        max_points: int = DATASET_VIEWER_DEFAULT_MAX_POINTS,
    ) -> dict[str, Any]:
        with self.lock:
            dataset_root = self._dataset_viewer_root(root)
            dataset_dir = self._resolve_dataset_viewer_dataset_dir(dataset, dataset_root)
            external_payload = self._dataset_viewer_episode_external(
                dataset_root=dataset_root,
                dataset_dir=dataset_dir,
                episode=int(episode),
                image_key=image_key,
                max_points=max_points,
            )
            if external_payload is not None:
                return external_payload

            info = _load_dataset_info_json(dataset_dir)
            total_episodes = int(info.get("total_episodes", 0) or 0)
            episode_idx = int(episode)
            if episode_idx < 0 or (total_episodes > 0 and episode_idx >= total_episodes):
                raise ValueError(f"episode out of range: {episode_idx}")

            image_keys = _image_keys_from_info(info)
            selected_image_key = str(image_key or "").strip()
            if selected_image_key not in image_keys:
                selected_image_key = image_keys[0] if image_keys else ""

            dataset_obj, row_positions = self._open_dataset_viewer_episode(dataset_dir, episode_idx)
            ee_feature_key = _ee_pose_key_from_info(info)
            obs_series, act_series, ee_series, raw_label_ids, frame_numbers, timestamps = (
                self._load_dataset_viewer_episode_arrays(dataset_obj, row_positions, ee_feature_key)
            )
            classes, segments, label_ids = _dataset_viewer_segments_payload(
                dataset_dir,
                info,
                episode_idx,
                frame_numbers,
                raw_label_ids,
            )
            sample_indices = _dataset_sample_indices(len(row_positions), max_points)
            plots = self._dataset_viewer_plot_payloads(obs_series, act_series, len(row_positions), sample_indices)
            ee_tracks = {
                name: _json_float_matrix(points, sample_indices, digits=4)
                for name, points in _extract_dataset_ee_tracks(ee_series).items()
            }
            display = _display_path(dataset_dir, base_root=dataset_root)
            if display == ".":
                display = dataset_dir.name
            image_streams = [
                {"key": key, "name": _image_stream_name(key) or key}
                for key in image_keys
            ]
            label_names = [label_name_for_id(classes, int(item)) for item in label_ids]
            return {
                "ok": True,
                "root": str(dataset_root),
                "dataset": {
                    "name": dataset_dir.name,
                    "path": str(dataset_dir),
                    "display": display,
                    "total_episodes": total_episodes,
                    "total_frames": int(info.get("total_frames", 0) or 0),
                    "fps": float(info.get("fps", 30.0) or 30.0),
                    "chunks_size": int(info.get("chunks_size", 0) or 0),
                    "image_streams": image_streams,
                    "features": sorted(_feature_dict(info).keys()),
                },
                "episode": episode_idx,
                "frame_count": len(row_positions),
                "sample_indices": [int(idx) for idx in sample_indices],
                "frame_numbers": [int(frame_numbers[int(idx)]) for idx in sample_indices],
                "timestamps": [_clean_json_float(timestamps[int(idx)], 4) for idx in sample_indices],
                "frame_numbers_full": [int(item) for item in frame_numbers],
                "timestamps_full": [_clean_json_float(item, 4) for item in timestamps],
                "label_ids_full": [int(item) for item in label_ids],
                "label_names_full": label_names,
                "image_key": selected_image_key,
                "plots": plots,
                "ee_tracks": ee_tracks,
                "classes": classes,
                "segments": segments,
                "dimensions": {
                    "observation": int(obs_series.shape[1]) if obs_series.ndim == 2 else 0,
                    "action": int(act_series.shape[1]) if act_series.ndim == 2 else 0,
                    "ee_pose": int(ee_series.shape[1]) if ee_series.ndim == 2 else 0,
                },
            }

    def _dataset_viewer_episode_external(
        self,
        *,
        dataset_root: Path,
        dataset_dir: Path,
        episode: int,
        image_key: str | None,
        max_points: int,
    ) -> dict[str, Any] | None:
        python_bin = _dataset_viewer_external_python()
        if python_bin is None:
            return None

        env = dict(os.environ)
        env["IGRIS_DATASET_VIEWER_DISABLE_EXTERNAL"] = "1"
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(REPO_ROOT)
        )
        cmd = [
            str(python_bin),
            "-m",
            "igris_teleop.web_ui.dataset_viewer_helper",
            "episode",
            "--root",
            str(dataset_root),
            "--dataset",
            str(dataset_dir),
            "--episode",
            str(int(episode)),
            "--max-points",
            str(int(max_points)),
        ]
        if image_key:
            cmd.extend(["--image-key", str(image_key)])
        timeout_s = _optional_float(os.getenv("IGRIS_DATASET_VIEWER_HELPER_TIMEOUT")) or 120.0
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=timeout_s,
            )
        except Exception as exc:
            logger.warning("dataset viewer external helper failed to start: %s", exc)
            return None
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            detail = stderr or proc.stdout.strip()
            logger.warning(
                "dataset viewer external helper failed python=%s returncode=%s detail=%s",
                python_bin,
                proc.returncode,
                detail,
            )
            return None
        try:
            payload = json.loads(proc.stdout)
        except Exception as exc:
            logger.warning("dataset viewer external helper returned invalid JSON: %s stderr=%s", exc, stderr)
            return None
        if not isinstance(payload, dict) or not payload.get("ok", False):
            logger.warning("dataset viewer external helper returned error: %s stderr=%s", payload, stderr)
            return None
        payload["reader"] = {
            "kind": "external_python",
            "python": str(python_bin),
        }
        return payload

    def dataset_viewer_save_labels(
        self,
        *,
        root: str | None = None,
        dataset: str,
        episode: int,
        classes: Any,
        segments: Any,
    ) -> dict[str, Any]:
        with self.lock:
            dataset_root = self._dataset_viewer_root(root)
            dataset_dir = self._resolve_dataset_viewer_dataset_dir(dataset, dataset_root)
            episode_idx = int(episode)
            external_payload = self._dataset_viewer_save_labels_external(
                dataset_dir=dataset_dir,
                episode=episode_idx,
                classes=classes,
                segments=segments,
            )
            if external_payload is not None:
                self._close_dataset_viewer_cache()
                return external_payload

            return self._dataset_viewer_save_labels_local(
                dataset_dir=dataset_dir,
                episode=episode_idx,
                classes=classes,
                segments=segments,
            )

    def _dataset_viewer_save_labels_external(
        self,
        *,
        dataset_dir: Path,
        episode: int,
        classes: Any,
        segments: Any,
    ) -> dict[str, Any] | None:
        python_bin = _dataset_viewer_external_python()
        if python_bin is None:
            return None

        env = dict(os.environ)
        env["IGRIS_DATASET_VIEWER_DISABLE_EXTERNAL"] = "1"
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(REPO_ROOT)
        )
        cmd = [
            str(python_bin),
            "-m",
            "igris_teleop.web_ui.dataset_viewer_helper",
            "save-labels",
            "--dataset",
            str(dataset_dir),
            "--episode",
            str(int(episode)),
        ]
        stdin_payload = json.dumps(
            {
                "classes": classes,
                "segments": segments,
            },
            ensure_ascii=False,
            default=_json_scalar,
        )
        timeout_s = _optional_float(os.getenv("IGRIS_DATASET_VIEWER_HELPER_TIMEOUT")) or 120.0
        try:
            proc = subprocess.run(
                cmd,
                input=stdin_payload,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=timeout_s,
            )
        except Exception as exc:
            logger.warning("dataset viewer label helper failed to start: %s", exc)
            return None
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            detail = stderr or proc.stdout.strip()
            raise RuntimeError(f"dataset label helper failed: {detail}")
        try:
            payload = json.loads(proc.stdout)
        except Exception as exc:
            raise RuntimeError(f"dataset label helper returned invalid JSON: {exc}; {stderr}") from exc
        if not isinstance(payload, dict) or not payload.get("ok", False):
            raise RuntimeError(f"dataset label helper returned error: {payload}; {stderr}")
        payload["reader"] = {
            "kind": "external_python",
            "python": str(python_bin),
        }
        return payload

    def _dataset_viewer_save_labels_local(
        self,
        *,
        dataset_dir: Path,
        episode: int,
        classes: Any,
        segments: Any,
    ) -> dict[str, Any]:
        ready, message = get_labeling_backend_status()
        if not ready:
            raise RuntimeError(message)
        if not isinstance(segments, list):
            raise ValueError("segments must be a list")

        info = _load_dataset_info_json(dataset_dir)
        total_episodes = int(info.get("total_episodes", 0) or 0)
        episode_idx = int(episode)
        if episode_idx < 0 or (total_episodes > 0 and episode_idx >= total_episodes):
            raise ValueError(f"episode out of range: {episode_idx}")

        dataset_obj, row_positions = self._open_dataset_viewer_episode(dataset_dir, episode_idx)
        ee_feature_key = _ee_pose_key_from_info(info)
        try:
            _, _, _, _, frame_numbers, _ = self._load_dataset_viewer_episode_arrays(
                dataset_obj,
                row_positions,
                ee_feature_key,
            )
        finally:
            self._close_dataset_viewer_cache()

        normalized_segments = normalize_boundary_segments(segments, frame_numbers)
        if episode_has_unassigned_segments(normalized_segments):
            raise ValueError("Assign labels to every segment before saving.")

        try:
            sidecar = load_annotation_sidecar(dataset_dir)
        except Exception:
            sidecar = {
                "feature_key": LABEL_FEATURE_KEY,
                "classes": default_label_classes(),
                "episodes": {},
            }
        raw_episodes = sidecar.get("episodes", {}) if isinstance(sidecar, Mapping) else {}
        merged_episodes: dict[int, list[dict[str, int]]] = {}
        if isinstance(raw_episodes, Mapping):
            for raw_episode, raw_segments in raw_episodes.items():
                try:
                    merged_episodes[int(raw_episode)] = [dict(segment) for segment in raw_segments]
                except Exception:
                    continue
        merged_episodes[episode_idx] = [dict(segment) for segment in normalized_segments]

        merged_classes = ensure_classes_cover_ids(
            classes if isinstance(classes, list) else sidecar.get("classes", default_label_classes()),
            [int(segment["class_id"]) for episode_segments in merged_episodes.values() for segment in episode_segments],
        )
        normalized_label_ids = segments_to_label_ids(normalized_segments, frame_numbers)

        result = rewrite_dataset_segment_labels_in_place(
            dataset_dir,
            classes=merged_classes,
            episode_segments=merged_episodes,
        )
        self._close_dataset_viewer_cache()
        return {
            "ok": True,
            "dataset": str(dataset_dir),
            "episode": episode_idx,
            "classes": [{"id": int(item["id"]), "name": str(item["name"])} for item in merged_classes],
            "segments": [
                {
                    "start_frame": int(segment["start_frame"]),
                    "end_frame": int(segment["end_frame"]),
                    "class_id": int(segment["class_id"]),
                    "class_name": label_name_for_id(merged_classes, int(segment["class_id"])),
                }
                for segment in normalized_segments
            ],
            "label_ids_full": [int(item) for item in normalized_label_ids],
            "label_names_full": [label_name_for_id(merged_classes, int(item)) for item in normalized_label_ids],
            "result": result,
        }

    def dataset_viewer_frame_jpeg(
        self,
        *,
        root: str | None = None,
        dataset: str,
        episode: int,
        frame: int,
        image_key: str | None = None,
        max_width: int = 960,
    ) -> bytes | None:
        with self.lock:
            dataset_root = self._dataset_viewer_root(root)
            dataset_dir = self._resolve_dataset_viewer_dataset_dir(dataset, dataset_root)
            info = _load_dataset_info_json(dataset_dir)
            image_keys = _image_keys_from_info(info)
            selected_image_key = str(image_key or "").strip()
            if selected_image_key not in image_keys:
                selected_image_key = image_keys[0] if image_keys else ""
            if not selected_image_key:
                raise ValueError("Selected dataset has no image stream")

            dataset_obj, row_positions = self._open_dataset_viewer_episode(dataset_dir, int(episode))
            frame_idx = max(0, min(len(row_positions) - 1, int(frame)))
            cache_key = (selected_image_key, frame_idx)
            cached = self.dataset_viewer_frame_cache.get(cache_key)
            if cached is not None:
                self.dataset_viewer_frame_cache.move_to_end(cache_key)
                return cached

            row = dataset_obj[int(row_positions[frame_idx])]
            raw_frame = _row_value(row, selected_image_key, None)
            if raw_frame is None:
                raise KeyError(f"Image stream is missing from frame: {selected_image_key}")
            rgb = _dataset_frame_to_uint8_rgb(raw_frame)
            rgb = _resize_rgb_to_max_width(rgb, max_width=max_width)
            encoded = _encode_jpeg(rgb)
            if encoded is None:
                return None
            self.dataset_viewer_frame_cache[cache_key] = encoded
            self.dataset_viewer_frame_cache.move_to_end(cache_key)
            while len(self.dataset_viewer_frame_cache) > DATASET_VIEWER_FRAME_CACHE_LIMIT:
                self.dataset_viewer_frame_cache.popitem(last=False)
            return encoded

    def status(self) -> dict[str, Any]:
        with self.lock:
            self._enforce_level_event_gate()
            self._drain_logs()
            alive = self._worker_alive()
            manual_running = self._manual_running()
            mode_running = self._mode_running()
            mode_worker_status = self._mode_worker_status(
                self.selected_mode,
                self.selected_teleop_device,
                self.selected_inference_policy if self.selected_mode == "inference" else None,
                alive,
                mode_running,
            )
            manual_groups = self._manual_group_status(alive, manual_running)
            levels = {name: self.bus.is_level_set(name) for name in LEVEL_EVENTS}
            mode_applied = self._mode_applied_for_level_events()
            start_enabled = self._start_level_controls_enabled()
            collect_alive = bool(alive.get("collect_data", False))
            hand_init_enabled = (
                mode_applied
                and bool(levels.get("ready", False))
                and not bool(levels.get("start", False))
                and not bool(levels.get("home", False))
                and bool(alive.get("hand", False))
            )
            self._write_mode_shm(alive)
            status = {
                "levels": levels,
                "camera_mode": self.applied_camera_mode,
                "sim_stereo": {
                    "enabled": bool(alive.get("simulator", False)),
                    "baseline_m": self._read_sim_stereo_baseline_m(),
                    "min_baseline_m": STEREO_CAMERA_MIN_BASELINE_M,
                    "max_baseline_m": STEREO_CAMERA_MAX_BASELINE_M,
                    "calibration_map_applied": False,
                    "extrinsic_mode": "translation_only",
                },
                "experiments": self._experiment_status(alive),
                "selected": {
                    "mode": self.selected_mode,
                    "teleop_device": self.selected_teleop_device,
                    "teleop_hand_source": self.selected_teleop_hand_source,
                    "inference_policy": (
                        self.selected_inference_policy if self.selected_mode == "inference" else None
                    ),
                },
                "applied": {
                    "mode": self.applied_mode,
                    "teleop_device": self.applied_teleop_device,
                    "teleop_hand_source": self.applied_teleop_hand_source,
                    "inference_policy": (
                        self.applied_inference_policy if self.applied_mode == "inference" else None
                    ),
                },
                "workers": {
                    "alive": alive,
                    "manual_running": sorted(manual_running),
                    "mode_running": sorted(mode_running),
                    "manual_groups": manual_groups,
                    "mode_workers": mode_worker_status,
                },
                "record": self._record_status(),
                "dataset": self._dataset_status(),
                "walking": self._walking_status(),
                "input_monitor": self._input_monitor_status(alive, manual_groups),
                "runtime_diagnostics": self._runtime_diagnostics_status(alive),
                "reliability_runtime": self.reliability_runtime.status(),
                "hybrid_teleop": self._hybrid_teleop_status(),
                "ui": {
                    "selection_locked": self.selection_locked,
                    "mode_applied": mode_applied,
                    "ready_enabled": mode_applied,
                    "start_enabled": start_enabled,
                    "home_enabled": start_enabled,
                    "hand_init_enabled": hand_init_enabled,
                    "record_start_enabled": collect_alive and self._shm("record_shm") is not None,
                    "record_done_enabled": collect_alive and self._shm("record_shm") is not None,
                    "record_reset_enabled": collect_alive and self._shm("record_shm") is not None,
                    "walking_start_enabled": self.applied_mode == "walking" and bool(levels.get("start", False)),
                },
                "options": {
                    "modes": list(MODE_CHOICES),
                    "teleop_devices": list(TELEOP_DEVICE_CHOICES),
                    "teleop_hand_sources_by_device": {
                        device: list(allowed_teleop_hand_sources(device))
                        for device in TELEOP_DEVICE_CHOICES
                    },
                    "camera_modes": list(CAMERA_MODE_CHOICES),
                    "camera_names": list(CAMERA_NAMES),
                    "collect_dataset_repo_id": self.collect_dataset_repo_id,
                    "collect_tasks": list(self.collect_tasks),
                    "initial_walking_policy_profile": self.initial_walking_policy_profile,
                    "initial_walking_policy_path": self.initial_walking_policy_path,
                    "walking_profiles": list(WALKING_PROFILE_CHOICES),
                    "walking_policy_defaults": {
                        profile: str(default_walking_policy_path(profile))
                        for profile in WALKING_PROFILE_CHOICES
                    },
                    "reliability_model_variants": list(RELIABILITY_MODEL_VARIANTS),
                    "experiment_tasks": experiment_tasks_payload(),
                    "hybrid_camera_devices": self._hybrid_camera_device_choices(),
                    "viser_url": self.viser_url,
                },
                "message": self.status_message,
                "logs": list(self.recent_logs)[-200:],
            }
            return status

    def telemetry(self) -> dict[str, Any]:
        with self.lock:
            obs = self._safe_read_shm("obs_shm")
            act = self._safe_read_shm("act_shm")
            if not obs or not act:
                return {"ok": False, "reason": "obs/action shared memory unavailable"}
            payload: dict[str, Any] = {"ok": True, "timestamp": time.time()}
            for key in ("arm", "leg", "waist", "neck", "hand"):
                obs_key = f"obs_{key}"
                act_key = f"act_{key}"
                if obs_key in obs:
                    payload[obs_key] = np.asarray(obs[obs_key], dtype=float).reshape(-1).tolist()
                if act_key in act:
                    payload[act_key] = np.asarray(act[act_key], dtype=float).reshape(-1).tolist()
            if "obs_imu_quat" in obs:
                payload["obs_imu_quat"] = np.asarray(obs["obs_imu_quat"], dtype=float).reshape(-1).tolist()
            if "obs_imu_rpy" in obs:
                payload["obs_imu_rpy"] = np.asarray(obs["obs_imu_rpy"], dtype=float).reshape(-1).tolist()
            return payload

    def apply_mode(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            prev_applied_mode = self.applied_mode
            mode = _normalize_optional(payload.get("mode"))
            if mode is not None and mode not in MODE_CHOICES:
                raise ValueError(f"Unsupported mode={mode!r}")
            teleop_device = _normalize_optional(payload.get("teleop_device"))
            if mode != "teleop":
                teleop_device = None
            elif teleop_device not in TELEOP_DEVICE_CHOICES:
                raise ValueError("teleop_device is required for teleop mode")

            teleop_hand_source = _normalize_optional(payload.get("teleop_hand_source"))
            if mode == "teleop":
                teleop_hand_source = resolve_teleop_hand_source(teleop_device, teleop_hand_source)
            else:
                teleop_hand_source = None

            camera_mode = _normalize_optional(payload.get("camera_mode")) or self.applied_camera_mode
            if camera_mode not in CAMERA_MODE_CHOICES:
                raise ValueError(f"Unsupported camera_mode={camera_mode!r}")
            camera_changed = camera_mode != self.applied_camera_mode
            if camera_mode != self.applied_camera_mode:
                self.applied_camera_mode = self.supervisor.apply_camera_settings(camera_mode)

            inference_policy = _normalize_optional(payload.get("inference_policy"))
            if mode == "inference" and inference_policy is None:
                inference_policy = DEFAULT_INFERENCE_POLICY
            if inference_policy in {"act"}:
                chunk_size = _optional_int(payload.get("inference_chunk_size"))
                n_action_step = _optional_int(payload.get("inference_n_action_step"))
                if chunk_size is not None and n_action_step is not None and n_action_step > chunk_size:
                    raise ValueError("ACT n_action_step must be <= chunk size")
            if inference_policy in {"diffusion", "diffusion_policy"}:
                horizon = _optional_int(payload.get("inference_horizon"))
                n_action_step = _optional_int(payload.get("inference_n_action_step"))
                if horizon is not None and horizon % 8 != 0:
                    raise ValueError("Diffusion horizon must be a multiple of 8")
                if horizon is not None and n_action_step is not None and n_action_step > horizon:
                    raise ValueError("Diffusion n_action_step must be <= horizon")

            can_apply_mode = mode is not None and (mode != "teleop" or teleop_device is not None)
            if not can_apply_mode:
                if camera_changed:
                    self.status_message = f"Applied camera mode: {self.applied_camera_mode}"
                    return self.status()
                raise ValueError("Select mode/device before applying workers")

            selected_workers_raw = payload.get("selected_mode_workers")
            if isinstance(selected_workers_raw, list):
                selected_workers = {str(item) for item in selected_workers_raw if str(item).strip()}
            else:
                selected_workers = set(
                    self.supervisor.list_selectable_mode_workers(mode, teleop_device, inference_policy)
                )

            running = self.supervisor.apply_mode_workers(
                mode=mode,
                teleop_device=teleop_device,
                teleop_hand_source=teleop_hand_source,
                selected_mode_workers=selected_workers,
                walking_policy_profile=_normalize_optional(payload.get("walking_policy_profile")),
                walking_policy_path=_normalize_optional(payload.get("walking_policy_path")),
                walking_startup_blend_enabled=_optional_bool(payload.get("walking_startup_blend_enabled")),
                inference_dataset_folder=_normalize_optional(payload.get("inference_dataset_folder")),
                inference_pretrained_rel=_normalize_optional(payload.get("inference_pretrained_rel")),
                inference_policy=inference_policy,
                inference_chunk_size=_optional_int(payload.get("inference_chunk_size")),
                inference_horizon=_optional_int(payload.get("inference_horizon")),
                inference_n_action_step=_optional_int(payload.get("inference_n_action_step")),
                inference_use_dataset_state=_optional_bool(payload.get("inference_use_dataset_state")),
                inference_use_dataset_tau=_optional_bool(payload.get("inference_use_dataset_tau")),
                inference_use_dataset_camera=_optional_bool(payload.get("inference_use_dataset_camera")),
                inference_instruction=_normalize_optional(payload.get("inference_instruction")),
                replay_dataset_folder=_normalize_optional(payload.get("replay_dataset_folder")),
            )

            self.selected_mode = mode
            self.selected_teleop_device = teleop_device
            self.selected_teleop_hand_source = teleop_hand_source
            self.selected_inference_policy = inference_policy if mode == "inference" else None
            self.selection_locked = True
            self.applied_mode = mode
            self.applied_teleop_device = teleop_device if mode == "teleop" else None
            self.applied_teleop_hand_source = teleop_hand_source if mode == "teleop" else None
            self.applied_inference_policy = inference_policy if mode == "inference" else None
            self.status_message = f"Applied mode workers: {', '.join(sorted(running)) if running else 'none'}"
            if mode == "walking" or prev_applied_mode == "walking":
                self._reset_walking_policy_command(
                    _normalize_optional(payload.get("walking_policy_profile"))
                    or self.initial_walking_policy_profile
                )
            self._write_mode_shm()
            return self.status()

    def toggle_manual_group(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            group = str(payload.get("group") or "").strip()
            if group not in MANUAL_WORKER_GROUPS:
                raise ValueError(f"Unknown manual worker group={group!r}")
            action = str(payload.get("action") or "toggle").strip().lower()
            if action not in {"toggle", "start", "stop"}:
                raise ValueError(f"Unsupported action={action!r}")

            mode = _normalize_optional(payload.get("mode", self.selected_mode))
            teleop_device = _normalize_optional(payload.get("teleop_device", self.selected_teleop_device))
            teleop_hand_source = _normalize_optional(
                payload.get("teleop_hand_source", self.selected_teleop_hand_source)
            )
            if mode != "teleop":
                teleop_device = None
                teleop_hand_source = None
            elif teleop_device is not None:
                teleop_hand_source = resolve_teleop_hand_source(teleop_device, teleop_hand_source)

            worker_names = _manual_worker_names_for_group(group, mode, teleop_device)
            all_worker_names = MANUAL_WORKER_GROUPS[group]
            running = self._manual_running()
            is_running = all(worker_name in running for worker_name in worker_names)
            should_start = action == "start" or (action == "toggle" and not is_running)

            if should_start:
                for worker_name in reversed(all_worker_names):
                    if worker_name not in worker_names and worker_name in running:
                        self.supervisor.stop_manual_worker(worker_name)
                        running.discard(worker_name)
                started_now: list[str] = []
                try:
                    for worker_name in worker_names:
                        if worker_name in running:
                            continue
                        collect_dataset_repo_id = None
                        if worker_name == "collect_data":
                            collect_dataset_repo_id = (
                                _normalize_optional(payload.get("collect_dataset_repo_id"))
                                or self.collect_dataset_repo_id
                            )
                        self.supervisor.start_manual_worker(
                            worker_name,
                            mode=mode,
                            teleop_device=teleop_device,
                            teleop_hand_source=teleop_hand_source,
                            collect_dataset_repo_id=collect_dataset_repo_id,
                        )
                        started_now.append(worker_name)
                        running.add(worker_name)
                except Exception:
                    for worker_name in reversed(started_now):
                        try:
                            self.supervisor.stop_manual_worker(worker_name)
                        except Exception:
                            logger.exception(
                                "failed to roll back manual worker %s", worker_name
                            )
                        running.discard(worker_name)
                    raise
                self.status_message = f"Started {MANUAL_WORKER_GROUP_LABELS.get(group, group)}"
            else:
                for worker_name in reversed(all_worker_names):
                    if worker_name in running:
                        self.supervisor.stop_manual_worker(worker_name)
                self.status_message = f"Stopped {MANUAL_WORKER_GROUP_LABELS.get(group, group)}"

            self.selected_mode = mode
            self.selected_teleop_device = teleop_device
            self.selected_teleop_hand_source = teleop_hand_source
            self._write_mode_shm()
            return self.status()

    def select_mode(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.selection_locked:
                raise ValueError("Mode selection is locked after Apply workers")
            mode = _normalize_optional(payload.get("mode"))
            if mode is not None and mode not in MODE_CHOICES:
                raise ValueError(f"Unsupported mode={mode!r}")
            teleop_device = _normalize_optional(payload.get("teleop_device"))
            if mode != "teleop":
                teleop_device = None
            elif teleop_device is not None and teleop_device not in TELEOP_DEVICE_CHOICES:
                raise ValueError(f"Unsupported teleop_device={teleop_device!r}")
            teleop_hand_source = _normalize_optional(payload.get("teleop_hand_source"))
            if mode == "teleop" and teleop_device is not None:
                teleop_hand_source = resolve_teleop_hand_source(teleop_device, teleop_hand_source)
            else:
                teleop_hand_source = None
            inference_policy = _normalize_optional(payload.get("inference_policy"))
            if mode == "inference" and inference_policy is None:
                inference_policy = DEFAULT_INFERENCE_POLICY
            self.selected_mode = mode
            self.selected_teleop_device = teleop_device
            self.selected_teleop_hand_source = teleop_hand_source
            self.selected_inference_policy = inference_policy if mode == "inference" else None
            self.status_message = "Mode selection changed. Click Apply workers."
            return self.status()

    def set_level(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            name = str(payload.get("name") or "").strip()
            if name not in LEVEL_EVENTS:
                raise ValueError(f"Unknown level event={name!r}")
            enabled = bool(payload.get("value"))

            if name == "camera":
                camera_mode = _normalize_optional(payload.get("camera_mode")) or self.applied_camera_mode
                if enabled:
                    if camera_mode not in CAMERA_MODE_CHOICES:
                        raise ValueError(f"Unsupported camera_mode={camera_mode!r}")
                    self.applied_camera_mode = self.supervisor.apply_camera_settings(camera_mode)
                    self.bus.set_level("camera")
                    self.status_message = f"Camera started ({self.applied_camera_mode})"
                else:
                    self.bus.clear_level("camera")
                    self.status_message = "Camera stopped"
                self._write_mode_shm()
                return self.status()

            if name == "ready" and enabled and not self._mode_applied_for_level_events():
                raise ValueError("Apply mode workers before ready")
            if name == "hand_init":
                if enabled:
                    if not self._mode_applied_for_level_events() or not self.bus.is_level_set("ready"):
                        raise ValueError("Apply mode workers and set ready before hand init")
                    if self.bus.is_level_set("start") or self.bus.is_level_set("home"):
                        raise ValueError("Clear start/home before hand init")
                    alive = self._worker_alive()
                    if not bool(alive.get("hand", False)):
                        raise ValueError("Start hand worker before hand init")
                    self.bus.set_level("hand_init")
                    self.status_message = "hand init requested"
                else:
                    self.bus.clear_level("hand_init")
                    self.status_message = "hand init cleared"
                self._write_mode_shm()
                return self.status()
            if name in {"start", "home"} and enabled:
                if not self._mode_applied_for_level_events() or not self.bus.is_level_set("ready"):
                    raise ValueError("Apply mode workers and set ready first")
            if name == "start" and enabled:
                blocked = self._start_block_message()
                if blocked:
                    raise ValueError(blocked)

            if enabled:
                self.bus.set_level(name)
                if name == "start" and "home" in LEVEL_EVENTS:
                    self.bus.clear_level("home")
                if name == "home":
                    self.bus.clear_level("start")
            else:
                self.bus.clear_level(name)
                if name == "ready":
                    self.bus.clear_level("start")
                    self.bus.clear_level("home")
            self.status_message = f"{name} {'set' if enabled else 'cleared'}"
            self._write_mode_shm()
            return self.status()

    def trigger_record(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            action = str(payload.get("action") or "").strip()
            if action not in {"start", "done", "reset"}:
                raise ValueError(f"Unsupported record action={action!r}")

            alive = self._worker_alive()
            if not alive.get("collect_data", False):
                raise ValueError("collect_data worker is not running")
            record_shm = self._shm("record_shm")
            if record_shm is None:
                raise ValueError("record shared memory unavailable")

            if action == "start":
                task_name = _normalize_optional(payload.get("task_name")) or self.selected_collect_task
                if not task_name:
                    raise ValueError("task_name is required for record start")
                self.selected_collect_task = task_name
                if task_name not in self.collect_tasks:
                    self.collect_tasks.append(task_name)
                self._write_record_task(task_name)
                updates = {"record_start": True, "record_done": False, "record_reset": False}
            elif action == "done":
                updates = {"record_start": False, "record_done": True}
            else:
                updates = {"record_start": False, "record_done": False, "record_reset": True}

            record_shm.write_data(**updates)
            self.status_message = f"record {action} triggered"
            return self.status()

    def write_walking_command(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            shm = self._shm("walking_cmd_shm")
            if shm is None:
                raise ValueError("walking command shared memory unavailable")
            profile = _normalize_optional(payload.get("profile")) or self.initial_walking_policy_profile
            updates = {
                "vx": _optional_float(payload.get("vx")) or 0.0,
                "vy": _optional_float(payload.get("vy")) or 0.0,
                "dyaw": _optional_float(payload.get("dyaw")) or 0.0,
            }
            policy_enabled = _optional_bool(payload.get("policy_enabled"))
            if policy_enabled is not None:
                if policy_enabled:
                    if self.applied_mode != "walking" or not self.bus.is_level_set("start"):
                        raise ValueError("Enable walking mode and start first")
                    if not walking_command_is_zero(updates):
                        raise ValueError("Set walking command to zero before Walking Start")
                updates["policy_enabled"] = 1.0 if policy_enabled else 0.0
            if profile:
                updates["profile_code"] = float(walking_policy_profile_code(profile))
            shm.write_data(**updates)
            self.status_message = "walking command updated"
            return self.status()

    def set_reliability_runtime(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            action = str(payload.get("action") or "toggle").strip().lower()
            if action not in {"toggle", "start", "stop"}:
                raise ValueError(f"Unsupported reliability action={action!r}")

            current = self.reliability_runtime.status()
            should_start = action == "start" or (action == "toggle" and not bool(current.get("running")))
            if should_start:
                self.hybrid_config = _hybrid_config_from_mapping(payload, base=self.hybrid_config)
                _save_hybrid_config(self.hybrid_config)
                hand_model_variant = _validate_reliability_model_variant(
                    payload.get("hand_model_variant"),
                    field_name="hand_model_variant",
                )
                controller_model_variant = _validate_reliability_model_variant(
                    payload.get("controller_model_variant"),
                    field_name="controller_model_variant",
                )
                self.hybrid_test_runtime.stop()
                _clear_hybrid_preview_files()
                self.reliability_runtime.start(
                    hand_model_variant=hand_model_variant,
                    controller_model_variant=controller_model_variant,
                    hybrid_config=self.hybrid_config,
                )
                self.status_message = (
                    "Started reliability runtime "
                    f"(hand={hand_model_variant}, controller={controller_model_variant})"
                )
            else:
                self.reliability_runtime.stop()
                self.status_message = "Stopped reliability runtime"
            return self.status()

    def set_hybrid_teleop(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            action = str(payload.get("action") or "save").strip().lower()
            if action not in {"save", "start_test", "stop_test", "toggle_test"}:
                raise ValueError(f"Unsupported hybrid teleop action={action!r}")

            self.hybrid_config = _hybrid_config_from_mapping(payload, base=self.hybrid_config)
            _save_hybrid_config(self.hybrid_config)

            test_status = self.hybrid_test_runtime.status()
            should_start_test = action == "start_test" or (
                action == "toggle_test" and not bool(test_status.get("running"))
            )
            should_stop_test = action == "stop_test" or (
                action == "toggle_test" and bool(test_status.get("running"))
            )

            if should_start_test:
                if bool(self.reliability_runtime.status().get("running")):
                    raise ValueError("Stop reliability before starting MediaPipe test")
                self.hybrid_test_runtime.start(self.hybrid_config)
                self.status_message = "Started hybrid MediaPipe camera test"
            elif should_stop_test:
                self.hybrid_test_runtime.stop()
                self.status_message = "Stopped hybrid MediaPipe camera test"
            elif action == "save":
                if bool(test_status.get("running")):
                    self.hybrid_test_runtime.start(self.hybrid_config)
                    self.status_message = "Applied hybrid settings and restarted MediaPipe test"
                else:
                    self.status_message = "Saved hybrid teleop settings"
            return self.status()

    def _hybrid_camera_device_choices(self) -> list[int | str]:
        stable = _stable_camera_devices()
        detected: list[int | str] = list(stable.values())
        for path in Path("/dev").glob("video*"):
            suffix = path.name[len("video") :]
            if suffix.isdigit() and str(path) not in stable:
                # Index 1 is normally a UVC metadata node, not an image stream.
                index_path = Path("/sys/class/video4linux") / path.name / "index"
                try:
                    if index_path.read_text().strip() != "0":
                        continue
                except OSError:
                    pass
                detected.append(int(suffix))
        # Preserve saved selections while disconnected so reopening/saving the
        # UI cannot silently assign a different camera to that hand.
        for device in (self.hybrid_config.left_device, self.hybrid_config.right_device):
            if device not in detected:
                detected.append(device)
        return detected

    @staticmethod
    def _hybrid_preview_side_status(side: str) -> dict[str, Any]:
        path = HYBRID_PREVIEW_ROOT / f"{side}_status.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"side": side, "camera_ok": False, "fresh": False, "detections": []}
        if not isinstance(payload, dict):
            return {"side": side, "camera_ok": False, "fresh": False, "detections": []}
        try:
            age_sec = max(0.0, time.time() - float(payload.get("updated_at", 0.0)))
        except (TypeError, ValueError):
            age_sec = float("inf")
        payload["age_sec"] = round(age_sec, 2) if np.isfinite(age_sec) else None
        payload["fresh"] = age_sec <= 2.0
        return payload

    def _hybrid_teleop_status(self) -> dict[str, Any]:
        test = self.hybrid_test_runtime.status()
        reliability = self.reliability_runtime.status()
        source = "reliability" if reliability.get("running") else "test" if test.get("running") else "stopped"
        return {
            "config": self.hybrid_config.as_dict(),
            "test": test,
            "source": source,
            "preview_root": str(HYBRID_PREVIEW_ROOT),
            "left": self._hybrid_preview_side_status("left"),
            "right": self._hybrid_preview_side_status("right"),
        }

    def hybrid_preview_jpeg(self, side: str, stage: str) -> bytes | None:
        clean_side = str(side).strip().lower()
        clean_stage = str(stage).strip().lower()
        if clean_side not in HYBRID_PREVIEW_SIDES or clean_stage not in HYBRID_PREVIEW_STAGES:
            raise ValueError("Unknown hybrid preview stream")
        path = HYBRID_PREVIEW_ROOT / f"{clean_side}_{clean_stage}.jpg"
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def camera_jpeg(self, camera_name: str, *, max_width: int = 960) -> bytes | None:
        if camera_name not in CAMERA_NAMES:
            raise ValueError(f"Unknown camera={camera_name!r}")
        camera_shm = self._shm("camera_shm")
        if camera_shm is None:
            return None
        try:
            data = camera_shm.read_data()
        except Exception:
            return None
        frame = data.get(camera_name)
        if frame is None:
            return None
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        arr = np.ascontiguousarray(arr)
        if max_width > 0 and arr.shape[1] > max_width:
            scale = float(max_width) / float(arr.shape[1])
            target_size = (max_width, max(1, int(round(arr.shape[0] * scale))))
            if cv2 is not None:
                arr = cv2.resize(arr, target_size, interpolation=cv2.INTER_AREA)
            elif Image is not None:
                image = Image.fromarray(arr, mode="RGB")
                arr = np.asarray(image.resize(target_size))
        return _encode_jpeg(arr)

    def _safe_read_shm(self, name: str) -> dict[str, Any] | None:
        shm = self._shm(name)
        if shm is None:
            return None
        try:
            return shm.read_data()
        except Exception:
            return None

    def _record_status(self) -> dict[str, Any]:
        record = self._safe_read_shm("record_shm") or {}
        task = self._safe_read_shm("record_task_shm") or {}
        active = [
            key
            for key in ("record_start", "record_done", "record_reset")
            if key in record and _scalar_bool(record[key])
        ]
        task_name = _decode_uint8_text(task.get("task_name"))
        return {
            "active": active,
            "state": ", ".join(active) if active else "idle",
            "task_name": task_name or self.selected_collect_task or "",
            "task_valid": _scalar_bool(task.get("task_valid", False)),
        }

    def _dataset_status(self) -> dict[str, Any]:
        info = self._safe_read_shm("dataset_info_shm") or {}
        stats = self._safe_read_shm("dataset_stats_shm") or {}
        folder = _decode_uint8_text(info.get("dataset_folder"))
        return {
            "folder": folder,
            "current_episode_frames": _scalar_int(stats.get("current_episode_frames", 0)),
            "total_saved_frames": _scalar_int(stats.get("total_saved_frames", 0)),
            "num_episodes": _scalar_int(stats.get("num_episodes", 0)),
        }

    def _walking_status(self) -> dict[str, Any]:
        data = self._safe_read_shm("walking_cmd_shm") or {}
        return {
            "profile_code": _scalar_float(data.get("profile_code", 0.0)),
            "vx": _scalar_float(data.get("vx", 0.0)),
            "vy": _scalar_float(data.get("vy", 0.0)),
            "dyaw": _scalar_float(data.get("dyaw", 0.0)),
            "policy_enabled": bool(_scalar_float(data.get("policy_enabled", 0.0))),
        }

    def _runtime_diagnostics_status(self, alive: Mapping[str, bool]) -> dict[str, Any]:
        source = getattr(self.supervisor, "runtime_diagnostics", None)
        if source is None:
            return {"timestamp": time.time(), "rows": []}
        try:
            raw = dict(source)
        except Exception:
            return {"timestamp": time.time(), "rows": []}

        now = time.time()
        stale_after_s = 3.0
        rows: list[dict[str, Any]] = []
        for key, value in raw.items():
            if not isinstance(value, Mapping):
                continue
            worker = str(value.get("worker") or str(key).split(":", 1)[0])
            loop = str(value.get("loop") or "main")
            updated_at = _json_float(value.get("updated_at"), 6)
            age_s = max(0.0, now - float(updated_at)) if updated_at is not None else None
            stopped = bool(value.get("stopped", False))
            alive_known = worker in alive
            alive_flag = bool(alive.get(worker, False)) if alive_known else not stopped and (
                age_s is None or age_s <= stale_after_s
            )
            if stopped and not alive_flag and age_s is not None and age_s > 30.0:
                continue

            if stopped and not alive_flag:
                status = "STOPPED"
            elif age_s is not None and age_s > stale_after_s:
                status = "STALE"
            elif alive_flag:
                status = "OK"
            else:
                status = "WAIT"

            rows.append(
                {
                    "key": str(value.get("key") or key),
                    "worker": worker,
                    "loop": loop,
                    "status": status,
                    "alive": alive_flag,
                    "target_hz": _json_float(value.get("target_hz")),
                    "actual_hz": _json_float(value.get("actual_hz")),
                    "period_ms": _json_float(value.get("period_ms")),
                    "jitter_ms": _json_float(value.get("jitter_ms")),
                    "latency_ms": _json_float(value.get("latency_ms")),
                    "max_jitter_ms": _json_float(value.get("max_jitter_ms")),
                    "max_latency_ms": _json_float(value.get("max_latency_ms")),
                    "samples": _scalar_int(value.get("samples", 0)),
                    "age_s": _json_float(age_s, 1),
                }
            )

        loop_order = {"main": 0, "slow": 1, "fast": 2}
        status_order = {"OK": 0, "STALE": 1, "WAIT": 2, "STOPPED": 3}
        rows.sort(
            key=lambda row: (
                status_order.get(str(row.get("status")), 9),
                str(row.get("worker", "")),
                loop_order.get(str(row.get("loop")), 9),
            )
        )
        return {"timestamp": now, "rows": rows}

    def _changed_since_last(self, key: str, fingerprint: tuple[Any, ...] | None) -> bool:
        missing = object()
        previous = self.input_monitor_fingerprints.get(key, missing)  # type: ignore[arg-type]
        self.input_monitor_fingerprints[key] = fingerprint
        return previous is not missing and previous != fingerprint

    @staticmethod
    def _group_status(groups: list[dict[str, Any]], name: str) -> dict[str, Any]:
        for group in groups:
            if group.get("name") == name:
                return group
        return {"name": name, "label": name, "workers": [], "status": "STOPPED", "running": False}

    def _input_monitor_status(
        self,
        alive: Mapping[str, bool],
        manual_groups: list[dict[str, Any]],
    ) -> dict[str, Any]:
        teleop_device = self.applied_teleop_device or self.selected_teleop_device
        hand_source = self.applied_teleop_hand_source or self.selected_teleop_hand_source
        leader_group = self._group_status(manual_groups, "leader_ros")

        act = self._safe_read_shm("act_shm") or {}
        arm_arr = _float_array(act.get("act_arm"), (14,))
        hand_arr = _float_array(act.get("act_hand"), (12,))
        arm_summary = _vector_summary(arm_arr, limit=14)
        hand_summary = _vector_summary(hand_arr, limit=12)
        arm_norm = float(arm_summary["norm"] or 0.0)
        hand_norm = float(hand_summary["norm"] or 0.0)
        leader_arm_fp = _array_fingerprint(arm_arr)
        leader_hand_fp = _array_fingerprint(hand_arr)

        television = self._safe_read_shm("television_shm") or {}
        vr_pose_keys = ("head_mat", "torso_mat", "chest_mat", "left_wrist_mat", "right_wrist_mat")
        vr_hand_keys = ("left_hand", "right_hand")
        vr_fp = tuple(
            (key, _array_fingerprint(television.get(key)))
            for key in (*vr_pose_keys, *vr_hand_keys)
        )
        vr_poses = {
            "head": _pose_summary(television.get("head_mat")),
            "torso": _pose_summary(television.get("torso_mat")),
            "chest": _pose_summary(television.get("chest_mat")),
            "left_wrist": _pose_summary(television.get("left_wrist_mat")),
            "right_wrist": _pose_summary(television.get("right_wrist_mat")),
        }
        vr_hands = {
            "left": _hand_points_summary(television.get("left_hand")),
            "right": _hand_points_summary(television.get("right_hand")),
        }
        valid_pose_count = sum(1 for item in vr_poses.values() if item.get("valid"))

        leader_bridge_alive = bool(alive.get("master_arm_ros_bridge", False))
        unity_bridge_alive = bool(alive.get("unity_bridge", False))
        return {
            "timestamp": time.time(),
            "mode": {
                "selected": self.selected_mode,
                "applied": self.applied_mode,
                "teleop_device": teleop_device,
                "hand_source": hand_source,
            },
            "leader_arm": {
                "enabled": teleop_device in {"masterarm", "vr_masterarm"},
                "bridge_alive": leader_bridge_alive,
                "connection": "ALIVE" if leader_bridge_alive else "STOPPED",
                "leader_ros_status": leader_group.get("status", "STOPPED"),
                "leader_ros_workers": [
                    {"name": name, "alive": bool(alive.get(str(name), False))}
                    for name in leader_group.get("workers", [])
                ],
                "arm_signal": "OK" if arm_summary["valid"] and arm_norm > 1e-6 else "WAIT",
                "hand_signal": "OK" if hand_summary["valid"] and hand_norm > 1e-6 else "WAIT",
                "arm_changed": self._changed_since_last("leader_arm", leader_arm_fp),
                "hand_changed": self._changed_since_last("leader_hand", leader_hand_fp),
                "act_arm": arm_summary,
                "act_hand": hand_summary,
                "hand_source": hand_source,
            },
            "vr": {
                "enabled": teleop_device in VR_POSE_TELEOP_DEVICES,
                "bridge_alive": unity_bridge_alive,
                "connection": "ALIVE" if unity_bridge_alive else "STOPPED",
                "ros_tcp_alive": bool(alive.get("leader_ros_tcp_endpoint", False)),
                "signal": "OK" if valid_pose_count else "WAIT",
                "changed": self._changed_since_last("vr", vr_fp),
                "valid_pose_count": valid_pose_count,
                "torso_alpha": _json_float(_scalar_float(television.get("torso_alpha"))),
                "torso_source_valid": _json_float(_scalar_float(television.get("torso_source_valid"))),
                "chest_alpha": _json_float(_scalar_float(television.get("chest_alpha"))),
                "chest_source_valid": _json_float(_scalar_float(television.get("chest_source_valid"))),
                "left_controller_confidence": _json_float(_scalar_float(television.get("left_controller_confidence"))),
                "right_controller_confidence": _json_float(_scalar_float(television.get("right_controller_confidence"))),
                "poses": vr_poses,
                "hands": vr_hands,
            },
        }

    def _write_record_task(self, task_name: str | None) -> None:
        shm = self._shm("record_task_shm")
        if shm is None:
            return
        task_arr = _encode_uint8_text(task_name, RECORD_TASK_NAME_LEN)
        shm.write_data(task_valid=np.uint8(1 if str(task_name or "").strip() else 0), task_name=task_arr)

    def _reset_walking_policy_command(self, profile: str | None) -> None:
        shm = self._shm("walking_cmd_shm")
        if shm is None:
            return
        updates: dict[str, Any] = {
            "vx": np.float64(0.0),
            "vy": np.float64(0.0),
            "dyaw": np.float64(0.0),
            "policy_enabled": np.float64(0.0),
        }
        try:
            if profile:
                updates["profile_code"] = np.float64(walking_policy_profile_code(profile))
            shm.write_data(**updates)
        except Exception:
            logger.debug("walking command reset failed", exc_info=True)

    def _write_mode_shm(self, alive: Mapping[str, bool] | None = None) -> None:
        shm = self._shm("mode_shm")
        if shm is None:
            return
        if alive is None:
            try:
                alive = self._worker_alive()
            except Exception:
                alive = {}
        levels = {name: self.bus.is_level_set(name) for name in LEVEL_EVENTS}
        updates = {
            "start": bool(levels.get("start", False)),
            "ready": bool(levels.get("ready", False)),
            "run": bool(levels.get("start", False)),
            "home": bool(levels.get("home", False)),
            "teleop": self.applied_mode == "teleop",
            "walking": self.applied_mode == "walking",
            "done": False,
            "reset": False,
            "replay": self.applied_mode == "replay",
            "deploy": self.applied_mode == "inference",
            "simulator": bool(alive.get("simulator", False)),
        }
        try:
            shm.write_data(**updates)
        except Exception:
            logger.debug("mode shared memory write failed", exc_info=True)

    def _start_block_message(self) -> str | None:
        if self.applied_mode != "teleop" or self.applied_teleop_device not in VR_POSE_TELEOP_DEVICES:
            return None
        shm = self._shm("teleop_guard_shm")
        if shm is None:
            return "Start blocked: VR head start guard status unavailable"
        try:
            status = shm.read_data()
        except Exception:
            return "Start blocked: VR head start guard status unavailable"
        if head_guard_is_blocking(status):
            return f"Start blocked: {format_head_guard_message(status)}"
        guard_active = bool(np.asarray(status.get("guard_active", 0.0)).reshape(()).item())
        if not guard_active:
            return "Start blocked: VR head start guard is not active yet"
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _resize_rgb_to_max_width(rgb: np.ndarray, *, max_width: int) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.uint8)
    if max_width <= 0 or arr.ndim != 3 or arr.shape[1] <= max_width:
        return np.ascontiguousarray(arr)
    scale = float(max_width) / float(arr.shape[1])
    target_size = (max_width, max(1, int(round(arr.shape[0] * scale))))
    if cv2 is not None:
        return cv2.resize(np.ascontiguousarray(arr), target_size, interpolation=cv2.INTER_AREA)
    if Image is not None:
        image = Image.fromarray(arr, mode="RGB")
        return np.asarray(image.resize(target_size))
    return np.ascontiguousarray(arr)


def _encode_jpeg(rgb: np.ndarray) -> bytes | None:
    if cv2 is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        return encoded.tobytes()
    if Image is not None:
        import io

        out = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(out, format="JPEG", quality=80)
        return out.getvalue()
    return None


class _WebHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], state: WebUIState) -> None:
        super().__init__(server_address, _RequestHandler)
        self.state = state
        self.timeout = 0.2


class _RequestHandler(BaseHTTPRequestHandler):
    server: _WebHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("web ui: " + format, *args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._serve_static("index.html")
            return
        if path.startswith("/static/"):
            self._serve_static(path[len("/static/") :])
            return
        if path == "/api/status":
            self._write_json(self.server.state.status())
            return
        if path == "/api/telemetry":
            self._write_json(self.server.state.telemetry())
            return
        if path == "/api/mode-workers":
            query = parse_qs(parsed.query)
            mode = _normalize_optional(_first_query(query, "mode"))
            teleop_device = _normalize_optional(_first_query(query, "teleop_device"))
            inference_policy = _normalize_optional(_first_query(query, "inference_policy"))
            workers = self.server.state.selectable_mode_workers(mode, teleop_device, inference_policy)
            self._write_json({"workers": workers})
            return
        if path == "/api/file-browser":
            query = parse_qs(parsed.query)
            kind = str(_first_query(query, "kind") or "").strip()
            raw_path = _first_query(query, "path")
            profile = _normalize_optional(_first_query(query, "profile"))
            try:
                payload = self.server.state.file_browser(kind=kind, raw_path=raw_path, profile=profile)
            except ValueError as exc:
                self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._write_json(payload)
            return
        if path == "/api/dataset-viewer/datasets":
            query = parse_qs(parsed.query)
            try:
                payload = self.server.state.dataset_viewer_datasets(root=_first_query(query, "root"))
            except ValueError as exc:
                self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception as exc:
                logger.exception("dataset viewer list failed")
                self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            self._write_json(payload)
            return
        if path == "/api/dataset-viewer/episode":
            query = parse_qs(parsed.query)
            try:
                payload = self.server.state.dataset_viewer_episode(
                    root=_first_query(query, "root"),
                    dataset=str(_first_query(query, "dataset") or ""),
                    episode=_optional_int(_first_query(query, "episode")) or 0,
                    image_key=_first_query(query, "image_key"),
                    max_points=_optional_int(_first_query(query, "max_points"))
                    or DATASET_VIEWER_DEFAULT_MAX_POINTS,
                )
            except ValueError as exc:
                self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception as exc:
                logger.exception("dataset viewer episode failed")
                self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            self._write_json(payload)
            return
        if path == "/api/dataset-viewer/frame.jpg":
            self._serve_dataset_viewer_frame(parse_qs(parsed.query))
            return
        if path.startswith("/camera/") and path.endswith(".jpg"):
            camera_name = unquote(path[len("/camera/") : -len(".jpg")])
            query = parse_qs(parsed.query)
            max_width = _optional_int(_first_query(query, "max_width")) or 960
            self._serve_camera_jpeg(camera_name, max_width=max_width)
            return
        if path.startswith("/camera/") and path.endswith(".mjpg"):
            camera_name = unquote(path[len("/camera/") : -len(".mjpg")])
            query = parse_qs(parsed.query)
            max_width = _optional_int(_first_query(query, "max_width")) or 960
            self._serve_camera_mjpeg(camera_name, max_width=max_width)
            return
        if path.startswith("/hybrid-preview/") and path.endswith(".jpg"):
            stream_name = unquote(path[len("/hybrid-preview/") : -len(".jpg")])
            parts = stream_name.split("/", 1)
            if len(parts) != 2:
                self._write_error(HTTPStatus.BAD_REQUEST, "bad hybrid preview path")
                return
            self._serve_hybrid_preview_jpeg(parts[0], parts[1])
            return
        self._write_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            payload = self._read_json()
            if path == "/api/apply-mode":
                self._write_json(self.server.state.apply_mode(payload))
                return
            if path == "/api/select-mode":
                self._write_json(self.server.state.select_mode(payload))
                return
            if path == "/api/manual":
                self._write_json(self.server.state.toggle_manual_group(payload))
                return
            if path == "/api/level":
                self._write_json(self.server.state.set_level(payload))
                return
            if path == "/api/record":
                self._write_json(self.server.state.trigger_record(payload))
                return
            if path == "/api/dataset-viewer/labels":
                self._write_json(
                    self.server.state.dataset_viewer_save_labels(
                        root=payload.get("root"),
                        dataset=str(payload.get("dataset") or ""),
                        episode=_optional_int(payload.get("episode")) or 0,
                        classes=payload.get("classes", []),
                        segments=payload.get("segments", []),
                    )
                )
                return
            if path == "/api/walking-command":
                self._write_json(self.server.state.write_walking_command(payload))
                return
            if path == "/api/reliability-runtime":
                self._write_json(self.server.state.set_reliability_runtime(payload))
                return
            if path == "/api/hybrid-teleop":
                self._write_json(self.server.state.set_hybrid_teleop(payload))
                return
            if path == "/api/sim-stereo":
                self._write_json(self.server.state.set_sim_stereo(payload))
                return
            if path == "/api/experiment-scene":
                self._write_json(self.server.state.set_experiment_scene(payload))
                return
            if path == "/api/shutdown":
                self.server.state.bus.set_level("shutdown")
                self.server.state.status_message = "shutdown requested"
                self._write_json(self.server.state.status())
                return
            self._write_error(HTTPStatus.NOT_FOUND, "not found")
        except ValueError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            logger.exception("web ui request failed")
            self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _serve_static(self, relative_path: str) -> None:
        clean = Path(unquote(relative_path))
        if clean.is_absolute() or any(part == ".." for part in clean.parts):
            self._write_error(HTTPStatus.BAD_REQUEST, "bad path")
            return
        path = (STATIC_DIR / clean).resolve()
        try:
            path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._write_error(HTTPStatus.BAD_REQUEST, "bad path")
            return
        if not path.is_file():
            self._write_error(HTTPStatus.NOT_FOUND, "not found")
            return
        content = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _serve_camera_jpeg(self, camera_name: str, *, max_width: int) -> None:
        frame = self.server.state.camera_jpeg(camera_name, max_width=max_width)
        if frame is None:
            self._write_error(HTTPStatus.SERVICE_UNAVAILABLE, "camera frame unavailable or JPEG encoder missing")
            return
        self._write_image_response(frame)

    def _serve_hybrid_preview_jpeg(self, side: str, stage: str) -> None:
        try:
            frame = self.server.state.hybrid_preview_jpeg(side, stage)
        except ValueError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if frame is None:
            self._write_error(HTTPStatus.SERVICE_UNAVAILABLE, "hybrid preview frame unavailable")
            return
        self._write_image_response(frame)

    def _serve_dataset_viewer_frame(self, query: Mapping[str, list[str]]) -> None:
        try:
            frame = self.server.state.dataset_viewer_frame_jpeg(
                root=_first_query(query, "root"),
                dataset=str(_first_query(query, "dataset") or ""),
                episode=_optional_int(_first_query(query, "episode")) or 0,
                frame=_optional_int(_first_query(query, "frame")) or 0,
                image_key=_first_query(query, "image_key"),
                max_width=_optional_int(_first_query(query, "max_width")) or 960,
            )
        except ValueError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except Exception as exc:
            logger.exception("dataset viewer frame failed")
            self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        if frame is None:
            self._write_error(HTTPStatus.SERVICE_UNAVAILABLE, "dataset frame unavailable or JPEG encoder missing")
            return
        self._write_image_response(frame)

    def _write_image_response(self, frame: bytes) -> None:
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(frame)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("image response client disconnected")
            return

    def _serve_camera_mjpeg(self, camera_name: str, *, max_width: int) -> None:
        boundary = "igrisframe"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        while not self.server.state.bus.is_level_set("shutdown"):
            try:
                frame = self.server.state.camera_jpeg(camera_name, max_width=max_width)
                if frame is not None:
                    self.wfile.write(f"--{boundary}\r\n".encode("ascii"))
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                time.sleep(0.12)
            except (BrokenPipeError, ConnectionResetError):
                break

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON payload must be an object")
        return data

    def _write_json(self, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=_json_scalar).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _write_error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"ok": False, "error": message}, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def _first_query(query: Mapping[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]


def run_web_ui(
    bus: EventBus,
    supervisor: Any,
    shared_memory: Mapping[str, Any],
    *,
    log_queue: Any,
    initial_mode: str | None,
    initial_teleop_device: str | None,
    initial_teleop_hand_source: str | None,
    initial_walking_policy_profile: str | None,
    initial_walking_policy_path: str | None,
    initial_camera_mode: str,
    host: str,
    port: int,
    open_browser: bool = False,
) -> None:
    state = WebUIState(
        bus=bus,
        supervisor=supervisor,
        shared_memory=shared_memory,
        log_queue=log_queue,
        initial_mode=initial_mode,
        initial_teleop_device=initial_teleop_device,
        initial_teleop_hand_source=initial_teleop_hand_source,
        initial_walking_policy_profile=initial_walking_policy_profile,
        initial_walking_policy_path=initial_walking_policy_path,
        initial_camera_mode=initial_camera_mode,
    )
    viser_bridge = None
    if not os.getenv("IGRIS_VISER_URL") and _env_bool("IGRIS_VISER_AUTOSTART", True):
        try:
            from .viser_bridge import WebUIViserBridge

            viser_host = str(os.getenv("IGRIS_VISER_BIND_HOST") or os.getenv("IGRIS_VISER_HOST") or "127.0.0.1")
            viser_port = _optional_int(os.getenv("IGRIS_VISER_PORT")) or 8080
            display_host = "127.0.0.1" if viser_host in {"", "0.0.0.0"} else viser_host
            viser_bridge = WebUIViserBridge(
                shared_memory,
                host=viser_host,
                port=viser_port,
                display_host=display_host,
            )
            if viser_bridge.start():
                state.viser_url = viser_bridge.url
            else:
                viser_bridge = None
        except Exception:
            logger.exception("[web-ui] failed to initialize Viser bridge")

    httpd = _WebHTTPServer((host, int(port)), state)
    actual_host, actual_port = httpd.server_address[:2]
    display_host = "127.0.0.1" if actual_host in {"", "0.0.0.0"} else actual_host
    url = f"http://{display_host}:{actual_port}/"
    logger.info("[web-ui] listening on %s", url)
    print(f"[web-ui] listening on {url}", flush=True)

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            logger.debug("failed to open browser", exc_info=True)

    try:
        while not bus.is_level_set("shutdown"):
            httpd.handle_request()
            try:
                if supervisor.has_unexpected_base_failure():
                    logger.error("[web-ui] base worker failure detected -> request shutdown")
                    bus.set_level("shutdown")
            except Exception:
                logger.debug("web ui watchdog failed", exc_info=True)
        logger.info("[web-ui] shutdown observed")
    finally:
        try:
            state._close_dataset_viewer_cache()
        except Exception:
            logger.debug("dataset viewer cache close failed", exc_info=True)
        try:
            state.reliability_runtime.stop()
        except Exception:
            logger.debug("reliability runtime stop failed", exc_info=True)
        try:
            state.hybrid_test_runtime.stop()
        except Exception:
            logger.debug("hybrid MediaPipe test stop failed", exc_info=True)
        if viser_bridge is not None:
            viser_bridge.stop()
        httpd.server_close()
