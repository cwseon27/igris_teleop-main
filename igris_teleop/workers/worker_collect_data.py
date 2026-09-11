from __future__ import annotations

import json
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import numpy as np
import yaml
from datetime import datetime
from zoneinfo import ZoneInfo

from ..core.events import EventSnapshot
from ..core.state_machine import ModeState, TransitionResult
from ..core.worker_base import SingleRateWorker, WorkerContext
from ..core.record_state_machine import RecordEventSnapshot, RecordState, step_record
from ..core.project_paths import (
    ARTIFACTS_ROOT,
    COLLECT_DATA_CONFIG_PATH,
    COLLECT_DATA_LOGS_ROOT,
    DATASETS_ROOT,
    resolve_under_root,
)
from ..sharedmemory.shm_schema import DATASET_FOLDER_NAME_LEN, RECORD_TASK_NAME_LEN


import logging_mp
logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)



try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 0.4.x
except Exception:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # 0.3.x fallback

# ─────────────────────────────────────────────────────────────
# Path defaults (file location based)
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG_PATH = COLLECT_DATA_CONFIG_PATH
DEFAULT_DATASET_PATH = DATASETS_ROOT
MISSING_CAMERA_WARN_INTERVAL_SEC = 5.0

# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────
def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def np_dtype(dtype: str):
    table = {
        "float32": np.float32,
        "float64": np.float64,
        "uint8": np.uint8,
        "int32": np.int32,
        "int64": np.int64,
        "bool": np.bool_,
    }
    if dtype not in table:
        raise KeyError(f"Unsupported np_dtype: {dtype}. Choose one of {list(table.keys())}")
    return table[dtype]

def get_episode_buffer_size(dataset) -> int:
    if dataset is None:
        return 0

    # LeRobot 0.4.x stores the mutable episode buffer on `dataset.writer`.
    writer = getattr(dataset, "writer", None)
    buf = getattr(writer, "episode_buffer", None)
    if isinstance(buf, dict):
        return int(buf.get("size", 0) or 0)

    # Older variants may expose the buffer directly on the dataset object.
    buf = getattr(dataset, "episode_buffer", None)
    if isinstance(buf, dict):
        return int(buf.get("size", 0) or 0)
    return 0

# ─────────────────────────────────────────────────────────────
# Feature Spec
# ─────────────────────────────────────────────────────────────
@dataclass
class FeatureSpec:
    # LeRobot feature key, e.g. "observation.image.realsense", "observation.state", "action"
    key: str
    use: bool

    # lerobot meta dtype category: "video", "image", "float", "int", "bool"
    dtype: str

    # numpy cast dtype: "uint8", "float32", "float64", ...
    np_dtype: str

    # for image/video: [H, W, C] / for vector: [N] (optional)
    shape: Optional[list[int]]

    # dimension names (optional)
    names: Optional[list[str]]

    # shared memory spec:
    # - image/video: {"name": "camera_shm", "field": "realsense"}
    # - vector:      {"name": "obs_shm", "fields": ["obs_leg","obs_arm",...]}
    # - scalar:      {"name": "...", "field": "..."}
    shm: dict


def build_lerobot_features(specs: list[FeatureSpec], use_videos: bool) -> dict[str, dict]:
    """
    LeRobotDataset.create(..., features=...)에 들어갈 meta.features dict 생성.
    - 이미지/비디오는 dtype을 "video" 또는 "image"로
    - vector는 dtype을 "float"/"int"/"bool" 등 LeRobot 범주로
    """
    features: dict[str, dict] = {}

    for s in specs:
        if not s.use:
            continue

        lerobot_dtype = s.dtype

        # 이미지/비디오 키는 use_videos에 따라 video/image로 강제 정규화
        if lerobot_dtype in ("video", "image"):
            lerobot_dtype = "video" if use_videos else "image"
            if not s.shape:
                raise ValueError(f"[features] '{s.key}' is image/video but shape is missing.")
            if len(s.shape) != 3:
                raise ValueError(f"[features] '{s.key}' image/video shape must be [H,W,C], got: {s.shape}")

            features[s.key] = {
                "dtype": lerobot_dtype,
                "shape": s.shape,
                "names": s.names,
            }
            
        else:
            features[s.key] = {
                "dtype": s.np_dtype,
                "shape": (int(s.shape[0]),), # Only vector
                "names": s.names[0],
            }

    return features



class CollectDataWorker(SingleRateWorker):
    """Single-rate 워커 예제: 상태에 따라 카운터를 업데이트."""

    def __init__(self, ctx: WorkerContext, hz: float = 10.0) -> None:
        super().__init__(ctx, hz=hz)
        self._counter = 0
        self._shared_memory = ctx.shared_memory
        self._owns_shared_memory = False
        
        self._record_state = RecordState.WAIT

        self.record_shm = self._shared_memory.get("record_shm")
        self.record_task_shm = self._shared_memory.get("record_task_shm")
        self.dataset_info_shm = self._shared_memory.get("dataset_info_shm")
        self.dataset_stats_shm = self._shared_memory.get("dataset_stats_shm")

        self.camera_shm = self._shared_memory.get("camera_shm")
        self.obs_shm = self._shared_memory.get("obs_shm")
        self.act_shm = self._shared_memory.get("act_shm")
        self.tau_shm = self._shared_memory.get("tau_shm")
        self.ee_shm = self._shared_memory.get("ee_shm")

        self.dataset = None
        self._recording = False
        self._total_saved_frames = 0
        self._last_missing_camera_warn_at: dict[str, float] = {}
        self._consecutive_frame_skips = 0

        self.config_path = str(DEFAULT_CONFIG_PATH)

        # parse config with guard
        self.cfg = load_yaml(self.config_path) or {}
        ds_cfg = self.cfg.get("dataset", {})

        runtime_repo_id = getattr(self.ctx.run_config, "collect_dataset_repo_id", None)
        self.repo_id = runtime_repo_id or ds_cfg.get("repo_id", "IGRIS_C")
        
        # NOTE: config의 상대경로는 CWD가 아니라 `igris_artifacts/` 기준으로 해석한다.
        # 예: root: "datasets" -> <repo>/igris_artifacts/datasets
        base_root = resolve_under_root(ARTIFACTS_ROOT, ds_cfg.get("root"), default=DEFAULT_DATASET_PATH)
        safe_repo = self.repo_id.replace("/", "__")
        ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
        self._session_id = ts
        dataset_root = base_root / f"{safe_repo}_{ts}"
        self.root = str(dataset_root)
        self._publish_dataset_folder(self.root)

        self.fps = int(ds_cfg.get("fps", 30))
        self.use_videos = bool(ds_cfg.get("use_videos", True))
        self.private = bool(ds_cfg.get("private", True))
        self.task_name = str(ds_cfg.get("task_name", "Pick and Place") or "Pick and Place")
        self.task_names = self._normalize_task_names(ds_cfg.get("tasks"), self.task_name)
        self._episode_task_name = self.task_name

        # 종료 시 남은 버퍼 저장 여부 (기본: 저장 안 함, 요구에 맞게 바꿔도 됨)
        self.flush_on_close = bool(ds_cfg.get("flush_on_close", False))

        self.num_image_writer_processes = int(ds_cfg.get("num_image_writer_processes", 0))
        self.num_image_writer_threads = int(ds_cfg.get("num_image_writer_threads", 8))
        self.perf_log_enabled = bool(ds_cfg.get("perf_log_enabled", True))
        self.perf_log_every_n_frames = max(1, int(ds_cfg.get("perf_log_every_n_frames", 10)))
        self.perf_log_slow_frame_ms = float(ds_cfg.get("perf_log_slow_frame_ms", 40.0))
        perf_log_root = resolve_under_root(
            ARTIFACTS_ROOT,
            ds_cfg.get("perf_log_dir"),
            default=COLLECT_DATA_LOGS_ROOT,
        )
        self.perf_log_path = perf_log_root / f"collect_data_perf_{safe_repo}_{self._session_id}.jsonl"
        self._perf_log_fp = None

        # load feature specs (robust)
        self.specs: list[FeatureSpec] = []
        for it in (self.cfg.get("features", []) or []):
            try:
                self.specs.append(
                    FeatureSpec(
                        key=it["key"],
                        use=bool(it.get("use", False)),
                        dtype=str(it.get("dtype", "float")).lower(),
                        np_dtype=str(it.get("np_dtype", "float32")).lower(),
                        shape=it.get("shape", None),
                        names=it.get("names", None),
                        shm=it.get("shm", {}),
                    )
                )
            except Exception as exc:
                logger.warning("[CollectDataWorker] bad feature entry %s (skip): %s", it, exc)

        # build LeRobot features
        try:
            lerobot_features = build_lerobot_features(self.specs, use_videos=self.use_videos)
        except Exception as exc:
            # 이 단계에서 죽으면 프로세스가 exitcode!=0로 main 전체 종료 유발 가능
            logger.exception("[CollectDataWorker] build_lerobot_features failed: %s", exc)
            raise

        # create dataset (guard)
        try:
            self.dataset = LeRobotDataset.create(
                self.repo_id,
                self.fps,
                root=self.root,
                features=lerobot_features,
                use_videos=self.use_videos,
                image_writer_processes=self.num_image_writer_processes,
                image_writer_threads=self.num_image_writer_threads,
            )
        except Exception as exc:
            logger.exception("[CollectDataWorker] LeRobotDataset.create failed: %s", exc)
            raise
        else:
            self._write_dataset_stats(
                current_frames=0,
                total_saved=0,
                num_episodes=getattr(self.dataset, "num_episodes", 0) or 0,
            )
            self._open_perf_log()

        logger.info(
            "[CollectDataWorker] init done repo_id=%s fps=%s use_videos=%s root=%s config=%s",
            self.repo_id, self.fps, self.use_videos, self.root, self.config_path
        )

    def on_start(self) -> None:
        logger.info(f"[{self.ctx.name}] start READY. Press [1]=record_start, [2]=episode_done, [3]=episode_reset")

    def _open_perf_log(self) -> None:
        if not self.perf_log_enabled:
            return
        try:
            self.perf_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._perf_log_fp = open(self.perf_log_path, "a", encoding="utf-8", buffering=1)
            self._log_perf_event(
                "init",
                pid=os.getpid(),
                repo_id=self.repo_id,
                dataset_root=self.root,
                fps=self.fps,
                use_videos=self.use_videos,
                num_image_writer_processes=self.num_image_writer_processes,
                num_image_writer_threads=self.num_image_writer_threads,
                perf_log_every_n_frames=self.perf_log_every_n_frames,
                perf_log_slow_frame_ms=self.perf_log_slow_frame_ms,
            )
            logger.info("[CollectDataWorker] perf log -> %s", self.perf_log_path)
        except Exception:
            self._perf_log_fp = None
            logger.exception("[CollectDataWorker] failed to open perf log: %s", self.perf_log_path)

    def _close_perf_log(self) -> None:
        fp = self._perf_log_fp
        self._perf_log_fp = None
        if fp is None:
            return
        try:
            fp.close()
        except Exception:
            logger.exception("[CollectDataWorker] failed to close perf log")

    def _log_perf_event(self, event: str, **payload: Any) -> None:
        fp = self._perf_log_fp
        if fp is None:
            return
        record = {
            "event": event,
            "time": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="milliseconds"),
        }
        record.update(payload)
        try:
            fp.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
        except Exception:
            logger.exception("[CollectDataWorker] failed to write perf log event=%s", event)

    def _resource_usage_snapshot(self) -> dict[str, float | int]:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
        except Exception:
            return {}
        return {
            "proc_user_cpu_s": round(float(usage.ru_utime), 6),
            "proc_sys_cpu_s": round(float(usage.ru_stime), 6),
            "proc_maxrss_kb": int(usage.ru_maxrss),
        }

    def _image_writer_queue_size(self) -> int | None:
        writer = getattr(self.dataset, "writer", None)
        image_writer = getattr(writer, "image_writer", None)
        queue_obj = getattr(image_writer, "queue", None)
        if queue_obj is None:
            return None
        try:
            return int(queue_obj.qsize())
        except Exception:
            return None

    def _perf_common_snapshot(self, *, buffer_frames: int | None = None) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "mode_state": self.state.name,
            "record_state": self._record_state.name,
            "recording": bool(self._recording),
            "total_saved_frames": int(self._total_saved_frames),
            "image_writer_queue_size": self._image_writer_queue_size(),
        }
        if buffer_frames is not None:
            snapshot["buffer_frames"] = int(buffer_frames)
        snapshot.update(self._resource_usage_snapshot())
        return snapshot

    def _should_log_frame_perf(self, current_len: int, step_ms: float) -> bool:
        return (
            current_len <= 3
            or current_len % self.perf_log_every_n_frames == 0
            or step_ms >= self.perf_log_slow_frame_ms
        )

    @staticmethod
    def _normalize_task_names(raw_tasks: Any, fallback: str) -> list[str]:
        tasks: list[str] = []
        if isinstance(raw_tasks, (list, tuple)):
            for raw in raw_tasks:
                text = str(raw or "").strip()
                if text and text not in tasks:
                    tasks.append(text)

        fallback_text = str(fallback or "").strip()
        if fallback_text and fallback_text not in tasks:
            tasks.insert(0, fallback_text)
        return tasks or ["Pick and Place"]

    @staticmethod
    def _decode_task_name(raw: Any) -> str:
        if raw is None:
            return ""
        try:
            arr = np.asarray(raw, dtype=np.uint8).reshape(-1)
        except Exception:
            return ""
        data = arr.tobytes().split(b"\x00", 1)[0]
        return data.decode("utf-8", errors="ignore").strip()

    def _read_selected_task_name(self) -> str:
        if self.record_task_shm is None:
            return self.task_name
        try:
            data = self.record_task_shm.read_data()
        except Exception:
            logger.exception("[CollectDataWorker] failed to read record task shared memory")
            return self.task_name

        try:
            valid = bool(int(np.asarray(data.get("task_valid", 0), dtype=np.uint8).reshape(()).item()))
        except Exception:
            valid = False
        task_name = self._decode_task_name(data.get("task_name"))
        if valid and task_name:
            return task_name[:RECORD_TASK_NAME_LEN]
        return self.task_name

    def _read_shm(self, shm_alias: str) -> dict[str, Any]:
        if shm_alias == "camera_shm":
            return self.camera_shm.read_data()
        if shm_alias == "obs_shm":
            return self.obs_shm.read_data()
        if shm_alias == "act_shm":
            return self.act_shm.read_data()
        if shm_alias == "tau_shm":
            return self.tau_shm.read_data()
        if shm_alias == "ee_shm":
            return self.ee_shm.read_data()
        raise ValueError(f"Unknown shm alias: {shm_alias}")

    def _read_record_snapshot(self) -> RecordEventSnapshot:
        if self.record_shm is None:
            return RecordEventSnapshot()
        try:
            data = self.record_shm.read_data()
        except Exception:
            logger.exception("[CollectDataWorker] failed to read record shared memory")
            return RecordEventSnapshot()
        snapshot = RecordEventSnapshot(
            record_start=self._as_bool(data.get("record_start")),
            record_done=self._as_bool(data.get("record_done")),
            record_reset=self._as_bool(data.get("record_reset")),
        )
        self._clear_record_flags()
        return snapshot

    def _clear_record_flags(self) -> None:
        if self.record_shm is None:
            return
        try:
            self.record_shm.write_data(record_start=False, record_done=False, record_reset=False)
        except Exception:
            logger.exception("[CollectDataWorker] failed to clear record shared memory flags")

    def _sync_mode_with_record_state(self, prev_state: RecordState, new_state: RecordState) -> None:
        """Drive global mode from record triggers."""
        if prev_state == new_state:
            return
        if new_state == RecordState.RECORD_START:
            self.ctx.bus.set_level("start")
            self.ctx.bus.clear_level("home")
            logger.info("[CollectDataWorker] record_start -> set RUN (start on, home off)")
        elif new_state == RecordState.RECORD_DONE:
            self.ctx.bus.set_level("home")
            self.ctx.bus.clear_level("start")
            logger.info("[CollectDataWorker] record_done -> set HOME (home on, start off)")

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if value is None:
            return False
        if hasattr(value, "reshape"):
            try:
                flat = value.reshape(-1)
            except Exception:
                pass
            else:
                if flat.size > 0:
                    return bool(flat[0])
        try:
            return bool(value)
        except Exception:
            return False

    @staticmethod
    def _camera_frame_ready(value: Any) -> bool:
        if value is None:
            return False
        try:
            arr = np.asarray(value)
        except Exception:
            return False
        return bool(arr.size > 0 and np.any(arr))

    def _warn_missing_camera_frame(self, *, feature_key: str, field: str) -> None:
        now = time.monotonic()
        last_logged = self._last_missing_camera_warn_at.get(feature_key, 0.0)
        if (now - last_logged) < MISSING_CAMERA_WARN_INTERVAL_SEC:
            return
        self._last_missing_camera_warn_at[feature_key] = now
        logger.warning(
            "[CollectDataWorker] waiting for camera frame key=%s field=%s; skipping current frame",
            feature_key,
            field,
        )


    def build_frame(self) -> Optional[dict[str, Any]]:
        """
        config에 정의된 feature들을 SHM에서 읽어 frame dict 생성.
        - image/video: shm.field (string)
        - vector: shm.fields (list[str])
        """
        frame: dict[str, Any] = {}
        shm_cache: dict[str, dict[str, Any]] = {}

        for s in self.specs:
            if not s.use:
                continue

            shm_spec = s.shm or {}
            shm_name = shm_spec.get("name", None)
            if shm_name is None:
                logger.warning("[CollectDataWorker] feature '%s' missing shm.name", s.key)
                continue

            # SHM read caching
            if shm_name not in shm_cache:
                try:
                    shm_cache[shm_name] = self._read_shm(shm_name)
                except Exception as exc:
                    logger.warning("[CollectDataWorker] read_shm('%s') failed: %s", shm_name, exc)
                    return None
            data = shm_cache[shm_name]

            # image/video
            if s.dtype in ("image", "video"):
                field = shm_spec.get("field", None)
                if not field or not isinstance(field, str):
                    logger.warning("[CollectDataWorker] feature '%s' needs shm.field(str) for image/video", s.key)
                    continue

                img = data.get(field) if isinstance(data, dict) else data[field]
                if shm_name == "camera_shm" and not self._camera_frame_ready(img):
                    self._warn_missing_camera_frame(feature_key=s.key, field=field)
                    return None
                if img is None:
                    return None

                # 최소복사: dtype만 맞추고 contiguous면 그대로 사용
                try:
                    arr = np.asarray(img, dtype=np_dtype(s.np_dtype))
                except Exception as exc:
                    logger.warning("[CollectDataWorker] image cast failed key=%s: %s", s.key, exc)
                    return None

                if arr.ndim == 3 and arr.shape[-1] >= 3:
                    arr = np.ascontiguousarray(arr[..., :3])
                
                # shape 검증(옵션)
                if s.shape:
                    if list(arr.shape) != list(s.shape):
                        logger.warning(
                            "[CollectDataWorker] image shape mismatch key=%s expect=%s got=%s",
                            s.key, s.shape, list(arr.shape)
                        )
                        return None

                frame[s.key] = arr
                continue

            # scalar (optional support)
            if s.dtype in ("float", "int", "bool") and isinstance(shm_spec.get("field", None), str):
                field = shm_spec.get("field")
                v = data.get(field) if isinstance(data, dict) else data[field]
                if v is None:
                    return None
                try:
                    arr = np.asarray(v, dtype=np_dtype(s.np_dtype)).reshape(())

                    frame[s.key] = arr
                except Exception as exc:
                    logger.warning("[CollectDataWorker] scalar cast failed key=%s: %s", s.key, exc)
                    return None
                continue

            # vector: shm.fields
            fields = shm_spec.get("fields", None)
            if not isinstance(fields, (list, tuple)) or len(fields) == 0:
                logger.warning("[CollectDataWorker] feature '%s' needs shm.fields(list[str]) for vector", s.key)
                continue

            parts = []
            for f in fields:
                v = data.get(f) if isinstance(data, dict) else data[f]
                if v is None:
                    return None
                parts.append(np.asarray(v).reshape(-1))

            try:
                vec = np.concatenate(parts, axis=0).astype(np_dtype(s.np_dtype), copy=False)
            except Exception as exc:
                logger.warning("[CollectDataWorker] vector build/cast failed key=%s: %s", s.key, exc)
                return None

            # shape 검증(옵션)
            if s.shape and len(s.shape) == 1:
                if vec.shape[0] != int(s.shape[0]):
                    logger.warning(
                        "[CollectDataWorker] vector len mismatch key=%s expect=%s got=%s",
                        s.key, s.shape[0], vec.shape[0]
                    )
                    return None

            frame[s.key] = vec

        frame["task"] = self._episode_task_name

        return frame

    def _publish_dataset_folder(self, path: str) -> None:
        shm = self.dataset_info_shm
        if shm is None:
            return
        try:
            encoded = path.encode("utf-8")[:DATASET_FOLDER_NAME_LEN]
            arr = np.zeros((DATASET_FOLDER_NAME_LEN,), dtype=np.uint8)
            if encoded:
                arr[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
            shm.write_data(dataset_folder=arr)
        except Exception:
            logger.exception("[CollectDataWorker] failed to write dataset info shared memory")

    def _write_dataset_stats(
        self,
        *,
        current_frames: int | None = None,
        total_saved: int | None = None,
        num_episodes: int | None = None,
    ) -> None:
        shm = self.dataset_stats_shm
        if shm is None:
            return
        updates: dict[str, int | np.int64] = {}
        if current_frames is not None:
            updates["current_episode_frames"] = np.int64(current_frames)
        if total_saved is not None:
            updates["total_saved_frames"] = np.int64(total_saved)
        if num_episodes is not None:
            updates["num_episodes"] = np.int64(num_episodes)
        if not updates:
            return
        try:
            shm.write_data(**updates)
        except Exception:
            logger.exception("[CollectDataWorker] failed to write dataset stats shared memory")

    def _handle_saved_episode(self, frame_count: int) -> None:
        self._total_saved_frames += max(frame_count, 0)
        num_eps = getattr(self.dataset, "num_episodes", 0) if self.dataset is not None else 0
        self._write_dataset_stats(
            current_frames=0,
            total_saved=self._total_saved_frames,
            num_episodes=num_eps or 0,
        )

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        step_started = time.perf_counter()
        snapshot = self._read_record_snapshot()
        prev_record_state = self._record_state
        transition = step_record(self._record_state, snapshot)
        self._record_state = transition.state
        self._sync_mode_with_record_state(prev_record_state, self._record_state)

        st = self.state
        
        if st == ModeState.RUN:
            buf_len = get_episode_buffer_size(self.dataset)

            if self._record_state == RecordState.RECORD_RESET:
                if not self._recording:
                    # record 중이 아닌데 reset 누름 -> 경고 로그만
                    logger.warning(
                        "[CollectDataWorker] episode_reset received while NOT recording -> ignored (buffer_len=%s)",
                        buf_len,
                    )
                    return

                # record 중이면 reset 수행
                try:
                    self.dataset.clear_episode_buffer()
                    logger.info("[CollectDataWorker] episode_reset -> cleared episode buffer (prev_len=%s)", buf_len)
                except Exception:
                    logger.exception("[CollectDataWorker] clear_episode_buffer failed")
                finally:
                    self._recording = False
                    self._write_dataset_stats(current_frames=0)
                    self._consecutive_frame_skips = 0
                    self._log_perf_event("episode_reset", **self._perf_common_snapshot(buffer_frames=buf_len))
                return
        
            if self._record_state == RecordState.RECORD_START:
                if not self._recording:
                    self._episode_task_name = self._read_selected_task_name()
                    self._recording = True
                    self._consecutive_frame_skips = 0
                    logger.info("[CollectDataWorker] Start Recording task=%r", self._episode_task_name)
                    self._log_perf_event(
                        "record_start",
                        task=self._episode_task_name,
                        **self._perf_common_snapshot(buffer_frames=buf_len),
                    )
                    
            elif self._record_state == RecordState.RECORD_DONE:
                if not self._recording:
                    logger.warning(
                        "[CollectDataWorker] episode_done received while NOT recording -> ignored (buffer_len=%s)",
                        buf_len,
                    )
                else:
                    if buf_len <= 0:
                        logger.warning(
                            "[CollectDataWorker] episode_done but buffer empty -> ignored (recording will stop)"
                        )
                        self._write_dataset_stats(current_frames=0)
                    else:
                        try:
                            logger.info("[CollectDataWorker] episode_done -> save_episode (len=%s)", buf_len)
                            logger.info("[CollectDataWorker] total_episode: (%s)", self.dataset.num_episodes)
                            save_started = time.perf_counter()
                            self.dataset.save_episode()
                            save_ms = (time.perf_counter() - save_started) * 1000.0
                            self._handle_saved_episode(buf_len)
                            self._log_perf_event(
                                "save_episode",
                                task=self._episode_task_name,
                                episode_frames=int(buf_len),
                                save_episode_ms=round(save_ms, 3),
                                num_episodes=int(getattr(self.dataset, "num_episodes", 0) or 0),
                                **self._perf_common_snapshot(buffer_frames=0),
                            )
                        except Exception:
                            logger.exception("[CollectDataWorker] save_episode failed")
                            self._write_dataset_stats(current_frames=0)
                            self._log_perf_event(
                                "save_episode_failed",
                                task=self._episode_task_name,
                                episode_frames=int(buf_len),
                                **self._perf_common_snapshot(buffer_frames=buf_len),
                            )
                    self._recording = False
                    self._consecutive_frame_skips = 0
                self._record_state = RecordState.WAIT
                return

            # 4) recording이 아니면 프레임 수집 안 함
            if not self._recording:
                return

            # 5) 프레임 1개 수집
            build_started = time.perf_counter()
            frame = self.build_frame()
            build_ms = (time.perf_counter() - build_started) * 1000.0
            if frame is None:
                self._consecutive_frame_skips += 1
                step_ms = (time.perf_counter() - step_started) * 1000.0
                if (
                    self._consecutive_frame_skips <= 3
                    or self._consecutive_frame_skips % 50 == 0
                    or step_ms >= self.perf_log_slow_frame_ms
                ):
                    self._log_perf_event(
                        "frame_skipped",
                        task=self._episode_task_name,
                        build_frame_ms=round(build_ms, 3),
                        step_ms=round(step_ms, 3),
                        consecutive_skips=int(self._consecutive_frame_skips),
                        **self._perf_common_snapshot(buffer_frames=buf_len),
                    )
                return

            try:
                add_started = time.perf_counter()
                self.dataset.add_frame(frame) # lerobot 0.4.x
                add_ms = (time.perf_counter() - add_started) * 1000.0
                current_len = get_episode_buffer_size(self.dataset)
                self._write_dataset_stats(current_frames=current_len)
                self._consecutive_frame_skips = 0
                step_ms = (time.perf_counter() - step_started) * 1000.0
                if self._should_log_frame_perf(current_len, step_ms):
                    self._log_perf_event(
                        "frame_perf",
                        task=self._episode_task_name,
                        frame_index=int(current_len - 1),
                        build_frame_ms=round(build_ms, 3),
                        add_frame_ms=round(add_ms, 3),
                        step_ms=round(step_ms, 3),
                        slow_frame=bool(step_ms >= self.perf_log_slow_frame_ms),
                        **self._perf_common_snapshot(buffer_frames=current_len),
                    )
            except Exception:
                logger.exception("[CollectDataWorker] add_frame failed (check feature keys/dtypes/shapes)")
                # 정책: recording 중단(폭주 방지)
                self._recording = False
                self._write_dataset_stats(current_frames=0)
                self._log_perf_event(
                    "add_frame_failed",
                    task=self._episode_task_name,
                    build_frame_ms=round(build_ms, 3),
                    **self._perf_common_snapshot(buffer_frames=buf_len),
                )


    def on_stop(self) -> None:
        try:
            n = get_episode_buffer_size(self.dataset) if self.dataset is not None else 0
            if self.flush_on_close and n and n > 0:
                logger.info("[CollectDataWorker] flush_on_close -> save_episode (len=%s)", n)
                self.dataset.save_episode()
            else:
                # 안전하게 버퍼 폐기
                try:
                    self.dataset.clear_episode_buffer()
                except Exception:
                    pass
                
            finalize = getattr(self.dataset, "finalize", None)
            if callable(finalize):
                finalize()
            else:
                writer = getattr(self.dataset, "writer", None)
                stop_image_writer = getattr(writer, "stop_image_writer", None)
                if callable(stop_image_writer):
                    stop_image_writer()
                else:
                    dataset_stop_image_writer = getattr(self.dataset, "stop_image_writer", None)
                    if callable(dataset_stop_image_writer):
                        dataset_stop_image_writer()
            
        except Exception:
            logger.exception("[CollectDataWorker] close dataset flush failed")
        finally:
            self._log_perf_event("stop", **self._perf_common_snapshot(buffer_frames=0))
            self._close_perf_log()

        
        if self._shared_memory:
            for key, mgr in self._shared_memory.items():
                try:
                    mgr.worker_close()
                except Exception:
                    logger.exception(f"[{self.ctx.name}] failed to close shared memory {key}")

        logger.info(f"[{self.ctx.name}] stop")
