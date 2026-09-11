from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

from .shm_schema import (
    CAMERA,
    DATASET_INFO,
    DATASET_STATS,
    INFERENCE_RESULT,
    IK_TARGET,
    MODE_LAYOUT,
    POSE_REQUEST,
    RECORD_MODE_LAYOUT,
    RECORD_TASK,
    ROBOT_ACTION,
    ROBOT_EE,
    ROBOT_OBS,
    ROBOT_TAU,
    SIM_CONFIG,
    TELEVISION,
    TELEOP_GUARD,
    WALKING_COMMAND,
    WALKING_DEBUG,
)


@dataclass(frozen=True)
class SharedMemorySpec:
    shm_key: str
    schema: Any
    lock_key: str


SHARED_MEMORY_SPECS: tuple[SharedMemorySpec, ...] = (
    SharedMemorySpec("camera_shm", CAMERA, "camera_lock"),
    SharedMemorySpec("sim_config_shm", SIM_CONFIG, "sim_config_lock"),
    SharedMemorySpec("television_shm", TELEVISION, "television_lock"),
    SharedMemorySpec("mode_shm", MODE_LAYOUT, "mode_lock"),
    SharedMemorySpec("record_shm", RECORD_MODE_LAYOUT, "record_lock"),
    SharedMemorySpec("record_task_shm", RECORD_TASK, "record_task_lock"),
    SharedMemorySpec("pose_request_shm", POSE_REQUEST, "pose_request_lock"),
    SharedMemorySpec("dataset_info_shm", DATASET_INFO, "dataset_info_lock"),
    SharedMemorySpec("dataset_stats_shm", DATASET_STATS, "dataset_stats_lock"),
    SharedMemorySpec("ee_shm", ROBOT_EE, "ee_lock"),
    SharedMemorySpec("ik_target_shm", IK_TARGET, "ik_target_lock"),
    SharedMemorySpec("teleop_guard_shm", TELEOP_GUARD, "teleop_guard_lock"),
    SharedMemorySpec("act_shm", ROBOT_ACTION, "act_lock"),
    SharedMemorySpec("obs_shm", ROBOT_OBS, "obs_lock"),
    SharedMemorySpec("tau_shm", ROBOT_TAU, "tau_lock"),
    SharedMemorySpec("inference_result_shm", INFERENCE_RESULT, "inference_result_lock"),
    SharedMemorySpec("walking_cmd_shm", WALKING_COMMAND, "walking_cmd_lock"),
    SharedMemorySpec("walking_debug_shm", WALKING_DEBUG, "walking_debug_lock"),
)

SHARED_MEMORY_KEYS: tuple[str, ...] = tuple(spec.shm_key for spec in SHARED_MEMORY_SPECS)


def default_shm_names() -> dict[str, str]:
    return {spec.shm_key: spec.shm_key for spec in SHARED_MEMORY_SPECS}


def create_shared_locks() -> dict[str, Any]:
    return {spec.lock_key: mp.Lock() for spec in SHARED_MEMORY_SPECS}
