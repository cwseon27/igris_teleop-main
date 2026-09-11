from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from ..core.project_paths import REPO_ROOT
from ..core.worker_base import WorkerContext
from .command_process import (
    CommandProcessSpec,
    build_leader_node_process_spec,
    build_ros_tcp_endpoint_process_spec,
)


WorkerBuilder = Callable[[WorkerContext], object]
DEFAULT_IGRIS_IK_HZ = 100.0
IGRIS_IK_HZ_ENV = "IGRIS_IK_HZ"


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    builder: WorkerBuilder | None = None
    daemon: bool = True
    external: bool = False
    python: str | None = None
    command_spec: CommandProcessSpec | None = None


def _uses_unity_baseline(ctx: WorkerContext) -> bool:
    return (
        ctx.run_config.mode == "teleop"
        and ctx.run_config.teleop_device == "unity"
    )


def build_single(ctx: WorkerContext) -> object:
    from ._single_example import SingleExampleWorker

    return SingleExampleWorker(ctx, hz=10.0)


def build_dual(ctx: WorkerContext) -> object:
    from ._dual_example import DualExampleWorker

    return DualExampleWorker(ctx, slow_hz=10.0, fast_hz=10.0)


def build_keyboard(ctx: WorkerContext) -> object:
    from .keyboard_worker import KeyboardWorker

    return KeyboardWorker(ctx, hz=60.0)


def build_unity(ctx: WorkerContext) -> object:
    if _uses_unity_baseline(ctx):
        from .worker_unity_bridge_baseline import UnityRosridgeWorker

        return UnityRosridgeWorker(ctx, hz=60.0)

    from .worker_unity_bridge import UnityRosridgeWorker

    return UnityRosridgeWorker(ctx, hz=60.0)


def resolve_igris_ik_hz(
    environ: Mapping[str, str] | None = None,
) -> float:
    values = os.environ if environ is None else environ
    raw_value = values.get(IGRIS_IK_HZ_ENV, str(DEFAULT_IGRIS_IK_HZ))
    try:
        hz = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{IGRIS_IK_HZ_ENV} must be a finite positive number, got {raw_value!r}"
        ) from exc
    if not math.isfinite(hz) or hz <= 0.0:
        raise ValueError(
            f"{IGRIS_IK_HZ_ENV} must be a finite positive number, got {raw_value!r}"
        )
    return hz


def build_igris_ik(ctx: WorkerContext) -> object:
    if _uses_unity_baseline(ctx):
        from .worker_igris_ik_baseline import IGRISIKWorker

        return IGRISIKWorker(ctx, hz=resolve_igris_ik_hz())

    from .worker_igris_ik import IGRISIKWorker

    return IGRISIKWorker(ctx, hz=resolve_igris_ik_hz())


def build_control(ctx: WorkerContext) -> object:
    from .worker_control import ControlWorker

    return ControlWorker(ctx, slow_hz=100.0, fast_hz=100.0)


def build_camera(ctx: WorkerContext) -> object:
    from .worker_camera import StereoCameraWorker

    return StereoCameraWorker(ctx, hz=30.0)


def build_camera_realsense(ctx: WorkerContext) -> object:
    from .worker_camera import RealSenseCameraWorker

    return RealSenseCameraWorker(ctx, hz=30.0)


def build_camera_topic(ctx: WorkerContext) -> object:
    from .worker_camera_topic import CameraTopicWorker

    return CameraTopicWorker(ctx, hz=30.0)


def build_hand(ctx: WorkerContext) -> object:
    # The legacy Unity bridge/IK path is intentionally retained for its pose
    # convention, but the legacy hand worker talks to the old in-process DDS
    # transport and never starts the robot-compatible native ROS hand bridge.
    # All teleop devices must use the maintained hand worker so Hand Initial
    # and continuous commands share the same verified transport.
    from .worker_hand import HandWorker

    return HandWorker(ctx, hz=50.0)


def build_collect_data(ctx: WorkerContext) -> object:
    from .worker_collect_data import CollectDataWorker

    return CollectDataWorker(ctx, hz=30.0)


def build_inference_lerobot(ctx: WorkerContext) -> object:
    from .worker_inference_lerobot import InferenceLeRobotWorker

    return InferenceLeRobotWorker(ctx, slow_hz=30.0, fast_hz=100.0)


def build_inference_optimize(ctx: WorkerContext) -> object:
    from .worker_optimize import InferenceOptimizeWorker

    return InferenceOptimizeWorker(ctx, slow_hz=30.0, fast_hz=100.0)


def build_inference_pi(ctx: WorkerContext) -> object:
    from .worker_inference_pi import InferencePIWorker

    return InferencePIWorker(ctx, slow_hz=30.0, fast_hz=100.0)


def build_replay(ctx: WorkerContext) -> object:
    from .worker_replay import DatasetReplayWorker

    return DatasetReplayWorker(ctx, slow_hz=30.0, fast_hz=100.0)


def build_walking_policy(ctx: WorkerContext) -> object:
    from .worker_walking_policy import WalkingPolicyWorker

    return WalkingPolicyWorker(ctx)


def build_walking_logger(ctx: WorkerContext) -> object:
    from .worker_walking_logger import WalkingLoggerWorker

    return WalkingLoggerWorker(ctx)


def build_master_arm_ros_bridge(ctx: WorkerContext) -> object:
    from .worker_master_arm_bridge import MasterarmRosridgeWorker

    return MasterarmRosridgeWorker(ctx)


def build_external_state_logger(ctx: WorkerContext) -> object:
    from ._external_state_logger import ExternalStateLoggerWorker

    return ExternalStateLoggerWorker(ctx, hz=5.0)


def build_simulator(ctx: WorkerContext) -> object:
    from .worker_simulator import MujocoSimulationWorker

    return MujocoSimulationWorker(ctx)


def _default_simulator_python() -> str:
    raw = os.environ.get("IGRIS_SIM_PYTHON")
    if raw:
        return raw
    return str((REPO_ROOT / ".venv-sim" / "bin" / "python").absolute())


def _default_ik_python() -> str:
    raw = os.environ.get("IGRIS_IK_PYTHON") or os.environ.get("IGRIS_GEOM_PYTHON")
    if raw:
        return raw
    return str((REPO_ROOT / ".venv-ik" / "bin" / "python").absolute())


def _default_ml_python() -> str:
    raw = os.environ.get("IGRIS_ML_PYTHON")
    if raw:
        return raw
    return str((REPO_ROOT / ".venv-ml" / "bin" / "python").absolute())


def _default_openpi_python() -> str:
    raw = os.environ.get("IGRIS_OPENPI_PYTHON")
    if raw:
        return raw
    return "/home/son/git_clone_project/openpi/.venv/bin/python"


WORKER_SPECS: dict[str, WorkerSpec] = {
    "single": WorkerSpec(name="single", builder=build_single),
    "dual": WorkerSpec(name="dual", builder=build_dual),
    "camera": WorkerSpec(name="camera", builder=build_camera),
    "camera_realsense": WorkerSpec(name="camera_realsense", builder=build_camera_realsense),
    "camera_topic": WorkerSpec(name="camera_topic", builder=build_camera_topic),
    "keyboard": WorkerSpec(name="keyboard", builder=build_keyboard),
    "control": WorkerSpec(name="control", builder=build_control),
    "igris_ik": WorkerSpec(
        name="igris_ik",
        builder=build_igris_ik,
        external=True,
        python=_default_ik_python(),
    ),
    "hand": WorkerSpec(
        name="hand",
        builder=build_hand,
        external=True,
        python=_default_ml_python(),
    ),
    "collect_data": WorkerSpec(
        name="collect_data",
        builder=build_collect_data,
        daemon=False,
        external=True,
        python=_default_ml_python(),
    ),
    "inference_lerobot": WorkerSpec(
        name="inference_lerobot",
        builder=build_inference_lerobot,
        external=True,
        python=_default_ml_python(),
    ),
    "inference_optimize": WorkerSpec(
        name="inference_optimize",
        builder=build_inference_optimize,
        external=True,
        python=_default_ml_python(),
    ),
    "inference_pi": WorkerSpec(
        name="inference_pi",
        builder=build_inference_pi,
        external=True,
        python=_default_openpi_python(),
    ),
    "replay": WorkerSpec(
        name="replay",
        builder=build_replay,
        external=True,
        python=_default_ml_python(),
    ),
    "walking_policy": WorkerSpec(name="walking_policy", builder=build_walking_policy),
    "walking_logger": WorkerSpec(name="walking_logger", builder=build_walking_logger),
    "unity_bridge": WorkerSpec(name="unity_bridge", builder=build_unity),
    "master_arm_ros_bridge": WorkerSpec(name="master_arm_ros_bridge", builder=build_master_arm_ros_bridge),
    "leader_ros_tcp_endpoint": WorkerSpec(
        name="leader_ros_tcp_endpoint",
        command_spec=build_ros_tcp_endpoint_process_spec(),
    ),
    "leader_ros_node": WorkerSpec(
        name="leader_ros_node",
        command_spec=build_leader_node_process_spec(),
    ),
    "simulator": WorkerSpec(
        name="simulator",
        builder=build_simulator,
        external=True,
        python=_default_simulator_python(),
    ),
    "external_state_logger": WorkerSpec(
        name="external_state_logger",
        builder=build_external_state_logger,
        external=True,
        python=None,
    ),
}


def list_worker_names() -> list[str]:
    return sorted(WORKER_SPECS.keys())


def get_worker_spec(name: str) -> WorkerSpec:
    return WORKER_SPECS[name]


def build_worker(name: str, ctx: WorkerContext) -> object:
    spec = get_worker_spec(name)
    if spec.builder is None:
        raise ValueError(f"Worker {name} does not use a Python builder")
    return spec.builder(ctx)


DEBUG_WORKERS: frozenset[str] = frozenset({"external_state_logger"})
ALWAYS_ON_WORKERS: frozenset[str] = frozenset({"keyboard", "camera_topic"})
MANUAL_UI_WORKER_ORDER: tuple[str, ...] = (
    "simulator",
    "control",
    "hand",
    "leader_ros_tcp_endpoint",
    "leader_ros_node",
    "collect_data",
)
MANUAL_UI_WORKERS: frozenset[str] = frozenset(MANUAL_UI_WORKER_ORDER)
CRITICAL_ONCE_STARTED_WORKERS: frozenset[str] = frozenset({"keyboard", "camera_topic", "control"})
# Backward-compatible alias for existing imports.
BASE_WORKERS: frozenset[str] = CRITICAL_ONCE_STARTED_WORKERS

INFERENCE_LEROBOT_POLICIES: frozenset[str] = frozenset({"act", "diffusion", "diffusion_policy"})
INFERENCE_PI_POLICIES: frozenset[str] = frozenset({"pi0", "pi0.5", "pi05"})


MODE_SELECTABLE_WORKERS: dict[str, frozenset[str]] = {
    "teleop": frozenset(),
    "walking": frozenset({"walking_policy", "walking_logger"}),
    "inference": frozenset({"inference_lerobot", "inference_optimize", "inference_pi"}),
    "replay": frozenset({"replay"}),
}

TELEOP_DEVICE_SELECTABLE_WORKERS: dict[str, frozenset[str]] = {
    "unity": frozenset({"unity_bridge", "igris_ik"}),
    "unity_hybrid": frozenset({"unity_bridge", "igris_ik"}),
    "vr_masterarm": frozenset({"unity_bridge", "master_arm_ros_bridge", "igris_ik"}),
    "masterarm": frozenset({"master_arm_ros_bridge"}),
}

_MODE_WORKER_GROUPS = tuple(MODE_SELECTABLE_WORKERS.values()) + tuple(TELEOP_DEVICE_SELECTABLE_WORKERS.values())
MODE_WORKER_POOL: frozenset[str] = frozenset(
    worker_name
    for worker_group in _MODE_WORKER_GROUPS
    for worker_name in worker_group
)


def _normalize_inference_policy(inference_policy: Optional[str]) -> str:
    policy = str(inference_policy or "").strip().lower()
    if policy == "pi05":
        return "pi0.5"
    return policy


def get_inference_worker_name(inference_policy: Optional[str]) -> str:
    policy = _normalize_inference_policy(inference_policy)
    if policy in INFERENCE_PI_POLICIES:
        return "inference_pi"
    if not policy or policy in INFERENCE_LEROBOT_POLICIES:
        return "inference_lerobot"
    return "inference_lerobot"


def get_mode_worker_candidates(
    mode: Optional[str],
    teleop_device: Optional[str],
    inference_policy: Optional[str] = None,
) -> set[str]:
    if mode is None:
        return set()
    if mode not in MODE_SELECTABLE_WORKERS:
        raise ValueError(f"Unknown mode={mode}")

    if mode == "inference":
        primary_worker = get_inference_worker_name(inference_policy)
        workers = {primary_worker, "inference_optimize"}
    else:
        workers = set(MODE_SELECTABLE_WORKERS[mode])
    if mode == "teleop" and teleop_device in TELEOP_DEVICE_SELECTABLE_WORKERS:
        workers.update(TELEOP_DEVICE_SELECTABLE_WORKERS[teleop_device])

    unknown = workers.difference(WORKER_SPECS)
    if unknown:
        raise ValueError(f"Mode workers not in WORKER_SPECS: {sorted(unknown)}")
    return workers


def get_mode_workers_for_apply(
    mode: Optional[str],
    teleop_device: Optional[str],
    inference_policy: Optional[str] = None,
) -> set[str]:
    if mode is None:
        raise ValueError("mode is not selected")

    workers = get_mode_worker_candidates(mode, teleop_device, inference_policy)
    if mode == "teleop":
        if teleop_device is None:
            raise ValueError("teleop mode requires a teleop_device")
        if teleop_device not in TELEOP_DEVICE_SELECTABLE_WORKERS:
            raise ValueError(f"Unknown teleop_device={teleop_device}")
    return workers
