from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ..core.project_paths import DATASETS_ROOT, resolve_under_root
from ..core.state_machine import ModeState
from ..core.worker_base import DualRateWorker, WorkerContext
import logging_mp

logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)

HAND_INDICES = 12
DEFAULT_DATASET_PATH = DATASETS_ROOT
DEFAULT_REPLAY_DATASET_FOLDER = "0304_실증과제_dataset_test/0304_lifting"
DEFAULT_REPLAY_EPISODE = 21


class DatasetReplayWorker(DualRateWorker):
    """Replay dataset actions directly into act_shm."""

    def __init__(self, ctx: WorkerContext, slow_hz: float = 30.0, fast_hz: float = 100.0) -> None:
        super().__init__(ctx, slow_hz=slow_hz, fast_hz=fast_hz)

        self.loop = False
        self.stride = 1
        self.dataset_repo_id = "IGRIS_C"

        self._shared_memory = ctx.shared_memory or {}
        self.act_shm = self._shared_memory.get("act_shm")
        if self.act_shm is None:
            raise RuntimeError("act_shm is required for replay")

        self.hand_len = HAND_INDICES
        from ..robot_control.kinematics.joints import ARM_INDICES, NECK_INDICES, WAIST_INDICES

        self.arm_len = len(ARM_INDICES)
        self.neck_len = len(NECK_INDICES)
        self.waist_len = len(WAIST_INDICES)
        self.base_action_dim = self.hand_len + self.arm_len + self.neck_len
        self.full_action_dim = self.base_action_dim + self.waist_len

        dataset_folder_name = (
            getattr(self.ctx.run_config, "replay_dataset_folder", None)
            or DEFAULT_REPLAY_DATASET_FOLDER
        )
        self.dataset_root = str(resolve_under_root(DEFAULT_DATASET_PATH, dataset_folder_name))
        self.episode = self._resolve_episode(DEFAULT_REPLAY_EPISODE)

        self.dataset = LeRobotDataset(
            self.dataset_repo_id,
            root=self.dataset_root,
            episodes=[self.episode],
        )
        self.episode_indices = [
            row_pos
            for row_pos, episode_idx in enumerate(self.dataset.hf_dataset["episode_index"])
            if int(episode_idx) == self.episode
        ]
        if not self.episode_indices:
            raise RuntimeError(f"Empty episode: {self.episode}")
        if min(self.episode_indices) < 0 or max(self.episode_indices) >= len(self.dataset):
            raise RuntimeError(
                "Invalid episode row indices: expected positional indices in "
                f"[0, {len(self.dataset) - 1}], got min={min(self.episode_indices)}, "
                f"max={max(self.episode_indices)}"
            )

        self._ptr = 0
        self._done = False
        self.pred_log: list[tuple[float, np.ndarray]] = []

        logger.info(
            f"[DatasetReplayWorker] dataset_root={self.dataset_root}, episode={self.episode}, "
            f"steps={len(self.episode_indices)}, stride={self.stride}, loop={self.loop}, "
            f"action_dims=({self.base_action_dim}|{self.full_action_dim})"
        )

    def _resolve_episode(self, requested_episode: int) -> int:
        info_path = Path(self.dataset_root) / "meta" / "info.json"
        total_episodes = 0
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                total_episodes = int(info.get("total_episodes", 0))
            except Exception:
                logger.exception("[DatasetReplayWorker] failed to read dataset info from %s", info_path)

        if total_episodes <= 0:
            return requested_episode

        episode = int(np.clip(requested_episode, 0, total_episodes - 1))
        if episode != requested_episode:
            logger.warning(
                "[DatasetReplayWorker] requested episode %s is out of range. "
                "Using episode %s (total_episodes=%s).",
                requested_episode,
                episode,
                total_episodes,
            )
        return episode

    def _load_action(self, row_idx: int) -> np.ndarray:
        row = self.dataset.hf_dataset[int(row_idx)]
        if "action" not in row:
            raise KeyError("Dataset row missing action")

        action = np.asarray(row["action"], dtype=np.float32).reshape(-1)
        if action.size not in (self.base_action_dim, self.full_action_dim):
            raise RuntimeError(
                f"Dataset action dim mismatch: expected {self.base_action_dim} or "
                f"{self.full_action_dim}, got {action.size}"
            )
        return action

    def _split_action(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size not in (self.base_action_dim, self.full_action_dim):
            raise ValueError(
                f"action dim mismatch: expected {self.base_action_dim} or "
                f"{self.full_action_dim}, got {arr.size}"
            )

        i = 0
        act_hand = arr[i : i + self.hand_len].copy()
        i += self.hand_len
        act_arm = arr[i : i + self.arm_len].copy()
        i += self.arm_len
        act_neck = arr[i : i + self.neck_len].copy()
        i += self.neck_len

        act_waist = np.zeros((self.waist_len,), dtype=np.float32)
        if arr.size == self.full_action_dim:
            act_waist = arr[i : i + self.waist_len].copy()

        return act_hand, act_arm, act_neck, act_waist

    def _advance_ptr(self) -> None:
        self._ptr += self.stride
        if self._ptr >= len(self.episode_indices):
            if self.loop:
                self._ptr = 0
                logger.info("[DatasetReplayWorker] episode loop restart")
            else:
                self._done = True
                logger.info("[DatasetReplayWorker] episode finished (loop=False)")

    def do_slow(self, ev, tr) -> None:
        if self._done:
            return

        if self.state != ModeState.RUN:
            self.act_shm.write_data(
                act_arm=[
                    -0.17098,
                    0.38123,
                    0.33176,
                    -1.19044,
                    0.25468,
                    -0.42747,
                    -0.09709,
                    -0.17098,
                    -0.38123,
                    -0.33176,
                    -1.19044,
                    -0.25468,
                    0.42747,
                    -0.09709,
                ],
                act_neck=[0.0, 0.0],
                act_hand=[0.0] * self.hand_len,
                act_waist=[0.0] * self.waist_len,
            )
            return

        row_idx = int(self.episode_indices[self._ptr])
        action = self._load_action(row_idx)
        act_hand, act_arm, act_neck, act_waist = self._split_action(action)
        self.act_shm.write_data(
            act_arm=act_arm,
            act_neck=act_neck,
            act_hand=act_hand,
            act_waist=act_waist,
        )
        self.pred_log.append((time.time(), action.copy()))
        self._advance_ptr()

    def do_fast(self, ev, tr) -> None:
        return

    def on_stop(self) -> None:
        if self._shared_memory:
            for key, mgr in self._shared_memory.items():
                try:
                    mgr.worker_close()
                except Exception:
                    logger.exception(f"[{self.ctx.name}] failed to close shared memory {key}")

        logger.info(f"[{self.ctx.name}] stop")
