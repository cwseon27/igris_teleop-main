from __future__ import annotations

import os
import threading
import time

try:
    import rclpy
except ImportError:
    rclpy = None

from ..core.events import EventSnapshot
from ..core.state_machine import TransitionResult
from ..core.worker_base import (
    CAMERA_MODE_PUB,
    CAMERA_MODE_SUB,
    DEFAULT_CAMERA_DOMAIN_ID,
    SingleRateWorker,
    WorkerContext,
)

from ..teleop_devices.cameras.ros_interface import CameraROSInterface

import logging_mp
import numpy as np

from ..sim.stereo_camera import STEREO_CAMERA_BASELINE_M, validate_stereo_baseline

logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)


class CameraTopicWorker(SingleRateWorker):
    """Bridge physical camera topics, or publish MuJoCo stereo SHM frames."""

    def __init__(self, ctx: WorkerContext, hz: float = 30.0) -> None:
        super().__init__(ctx, hz=hz)

        global rclpy
        if rclpy is None:
            raise RuntimeError("rclpy is not available. Please source ROS2 and install rclpy.")

        self._shared_memory = ctx.shared_memory
        self._owns_shared_memory = False

        self.camera_shm = self._shared_memory.get("camera_shm")
        self.sim_config_shm = self._shared_memory.get("sim_config_shm")
        self._camera_mode = str(ctx.run_config.camera_mode or CAMERA_MODE_SUB)
        self._camera_domain_id = int(ctx.run_config.camera_domain_id or DEFAULT_CAMERA_DOMAIN_ID)
        self._simulation_source = str(ctx.run_config.runtime_environment or "").lower() in {
            "sim",
            "simulator",
            "simulation",
            "mujoco",
        }
        self.iface: CameraROSInterface | None = None
        self.bridge = None
        self.sim_publisher = None
        self._spin_thread: threading.Thread | None = None
        self._init_retry_interval_s = 2.0
        self._last_init_attempt_ts = 0.0
        self._last_init_error_log_ts = 0.0
        self._wait_log_interval_sec = 5.0
        self._last_wait_log_at = 0.0

    def _spin_ros(self) -> None:
        if self.iface is None:
            return
        try:
            rclpy.spin(self.iface)
        except Exception:
            logger.exception("[CameraTopic] spin failed.")

    def _ensure_rclpy_initialized(self) -> None:
        try:
            rclpy.init(args=None)
        except RuntimeError:
            pass

    def _create_camera_sub_interface(self) -> None:
        logger.info("[CameraTopic] creating CameraROSInterface...")
        if self.iface is not None:
            return

        self._ensure_rclpy_initialized()

        try:
            self.iface = CameraROSInterface()
        except Exception:
            logger.exception("[CameraTopic] failed to create CameraROSInterface")
            self.iface = None
            return

        self._spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self._spin_thread.start()

    def _create_camera_pub_bridge(self) -> None:
        if self.bridge is not None:
            return

        self._ensure_rclpy_initialized()

        backend = str(os.getenv("IGRIS_CAMERA_PUB_BACKEND", "robot_ros_bridge")).strip().lower()
        if backend in {"python", "python_sdk", "legacy", "cyclonedds"}:
            from ..teleop_devices.cameras.cyclonedds_bridge import CycloneDDSCameraBridge

            self.bridge = CycloneDDSCameraBridge(domain_id=self._camera_domain_id)
            return

        from ..teleop_devices.cameras.robot_dds_ros_bridge import RobotDDSROSCameraBridge

        self.bridge = RobotDDSROSCameraBridge()

    def _create_sim_publisher(self) -> None:
        if self.sim_publisher is not None:
            return

        self._ensure_rclpy_initialized()
        from ..teleop_devices.cameras.sim_stereo_publisher import SimStereoCameraPublisher

        self.sim_publisher = SimStereoCameraPublisher()

    def _backend_ready(self) -> bool:
        if self._simulation_source:
            return self.sim_publisher is not None
        if self._camera_mode == CAMERA_MODE_PUB:
            return self.bridge is not None
        return self.iface is not None

    def _sim_stereo_baseline_m(self) -> float:
        if self.sim_config_shm is None:
            return STEREO_CAMERA_BASELINE_M
        try:
            raw_baseline = self.sim_config_shm.read_data()["stereo_baseline_m"]
            value = float(np.asarray(raw_baseline).reshape(-1)[0])
            return validate_stereo_baseline(value)
        except (KeyError, TypeError, ValueError, IndexError):
            return STEREO_CAMERA_BASELINE_M

    def _teardown_backend(self) -> None:
        had_backend = (
            self.iface is not None
            or self.bridge is not None
            or self.sim_publisher is not None
            or self._spin_thread is not None
        )

        if self.iface is not None:
            try:
                self.iface.destroy_node()
            except Exception:
                logger.exception("[CameraTopic] failed to destroy ROS node.")
            self.iface = None

        if self.bridge is not None:
            try:
                self.bridge.stop()
            except Exception:
                logger.exception("[CameraTopic] failed to stop CycloneDDS bridge.")
            try:
                self.bridge.destroy_node()
            except Exception:
                logger.exception("[CameraTopic] failed to destroy CycloneDDS bridge node.")
            self.bridge = None

        if self.sim_publisher is not None:
            try:
                self.sim_publisher.destroy_node()
            except Exception:
                logger.exception("[CameraTopic] failed to destroy simulation camera publisher.")
            self.sim_publisher = None

        if getattr(self, "_spin_thread", None):
            self._spin_thread.join(timeout=1.0)
            self._spin_thread = None

        if had_backend:
            try:
                if hasattr(rclpy, "try_shutdown"):
                    rclpy.try_shutdown()
                elif rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                logger.exception("[CameraTopic] failed to shutdown rclpy while tearing down backend.")

    def _log_init_error(self, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_init_error_log_ts < 5.0:
            return
        self._last_init_error_log_ts = now
        logger.warning(
            "[CameraTopic] failed to initialize camera backend mode=%s domain_id=%d: %s",
            self._camera_mode,
            self._camera_domain_id,
            exc,
        )

    def _log_waiting_for_topics(self) -> None:
        iface = self.iface
        if iface is None:
            return

        now = time.monotonic()
        if (now - self._last_wait_log_at) < self._wait_log_interval_sec:
            return

        missing = iface.get_missing_frame_topics() if hasattr(iface, "get_missing_frame_topics") else {}
        if not missing:
            return

        details: list[str] = []
        for key, topic in missing.items():
            publishers = iface.count_publishers_for_key(key) if hasattr(iface, "count_publishers_for_key") else None
            if publishers is None:
                details.append(f"{key}({topic})")
            else:
                details.append(f"{key}({topic}, publishers={publishers})")

        logger.warning("[CameraTopic] waiting for camera topics/frames: %s", ", ".join(details))
        self._last_wait_log_at = now

    def _ensure_backend_ready(self, ev: EventSnapshot) -> bool:
        if self._backend_ready():
            return True
        if not ev.level.get("camera", False):
            return False

        now = time.monotonic()
        if now - self._last_init_attempt_ts < self._init_retry_interval_s:
            return False
        self._last_init_attempt_ts = now

        try:
            if self._simulation_source:
                self._create_sim_publisher()
            elif self._camera_mode == CAMERA_MODE_PUB:
                self._create_camera_pub_bridge()
            else:
                self._create_camera_sub_interface()
        except Exception as exc:
            self.bridge = None
            self.iface = None
            self.sim_publisher = None
            self._log_init_error(exc)
            return False
        return self._backend_ready()

    def on_start(self) -> None:
        logger.info(
            "[%s] start (single-rate %.1f Hz, camera_mode=%s, camera_domain_id=%d, source=%s)",
            self.ctx.name,
            self.hz,
            self._camera_mode,
            self._camera_domain_id,
            "mujoco" if self._simulation_source else "physical",
        )

    def step_once(self, ev: EventSnapshot, tr: TransitionResult) -> None:
        del tr

        if not ev.level.get("camera", False):
            if self._backend_ready():
                self._teardown_backend()
            return

        if not self._ensure_backend_ready(ev):
            return

        if self._simulation_source:
            if self.sim_publisher is None or self.camera_shm is None:
                return
            try:
                self.sim_publisher.publish_frames(
                    self.camera_shm.read_data(),
                    baseline_m=self._sim_stereo_baseline_m(),
                )
            except Exception:
                logger.error("[CameraTopic] Failed to publish simulated stereo frames.", exc_info=True)
            return
        if self._camera_mode == CAMERA_MODE_PUB:
            if self.bridge is None:
                return
            payload = self.bridge.poll()
        else:
            if self.iface is None:
                return
            frames = self.iface.get_frames()
            payload = {k: v for k, v in frames.items() if v is not None}
        if not payload:
            if self._camera_mode == CAMERA_MODE_SUB:
                self._log_waiting_for_topics()
            return

        try:
            self.camera_shm.write_data(**payload)
        except Exception:
            logger.error("[CameraTopic] Failed to write camera_shm.", exc_info=True)

    def on_stop(self) -> None:
        if self._shared_memory:
            for key, mgr in self._shared_memory.items():
                try:
                    mgr.worker_close()
                except Exception:
                    logger.exception(f"[{self.ctx.name}] failed to close shared memory {key}")

        self._teardown_backend()

        try:
            if hasattr(rclpy, "try_shutdown"):
                rclpy.try_shutdown()
            elif rclpy.ok():
                rclpy.shutdown()
        except Exception:
            logger.exception("[CameraTopic] failed to shutdown rclpy.")

        logger.info(f"[{self.ctx.name}] stop")
