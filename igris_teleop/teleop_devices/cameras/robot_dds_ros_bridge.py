from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import rclpy

from ...core.project_paths import REPO_ROOT
from .ros_interface import CameraROSInterface

import logging_mp

logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)


DEFAULT_ROBOT_CAMERA_NAMESPACE = "igris_c_IG05"
DEFAULT_ROBOT_CAMERA_DOMAIN_ID = 0
DEFAULT_CYCLONEDDS_URI = REPO_ROOT / "local_state" / "cyclonedds_igris_lan.xml"
DEFAULT_ROBOT_BRIDGE_EXECUTABLE = (
    REPO_ROOT
    / "ros_ws"
    / "install"
    / "igris_c_sensor"
    / "lib"
    / "igris_c_sensor"
    / "igris_c_sensor_robot_bridge_node"
)


class RobotDDSROSCameraBridge:
    """Run the robot-compatible DDS camera bridge and read its ROS2 image topics.

    The Python `igris_c_sdk` bundled with this workspace is older than the robot
    camera publisher currently running on IGRIS-C.  The C++ bridge installed by
    `igris_c_sensor` uses the robot-side DDS message definition and a BestEffort
    reader, then republishes the frames on the ROS topics already consumed by
    `CameraROSInterface`.
    """

    def __init__(
        self,
        *,
        domain_id: int = DEFAULT_ROBOT_CAMERA_DOMAIN_ID,
        dds_namespace: str = DEFAULT_ROBOT_CAMERA_NAMESPACE,
        startup_timeout_s: float = 8.0,
    ) -> None:
        self._domain_id = int(os.getenv("IGRIS_ROBOT_CAMERA_DDS_DOMAIN_ID", str(domain_id)))
        self._dds_namespace = str(os.getenv("IGRIS_ROBOT_CAMERA_DDS_NAMESPACE", dds_namespace)).strip()
        self._startup_timeout_s = max(0.0, float(startup_timeout_s))
        self._process: Optional[subprocess.Popen[str]] = None
        self._log_file: Optional[Path] = None
        self._log_handle: Optional[object] = None
        self._iface: Optional[CameraROSInterface] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._stopped = False

        try:
            self._start_process()
            self._create_ros_interface()
        except Exception:
            self.stop()
            raise

    def _default_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if "CYCLONEDDS_URI" not in env and DEFAULT_CYCLONEDDS_URI.exists():
            env["CYCLONEDDS_URI"] = f"file://{DEFAULT_CYCLONEDDS_URI}"
        env.setdefault("ROS_DOMAIN_ID", "0")

        ros_lib_dir = "/opt/ros/jazzy/lib"
        existing_ld_library_path = env.get("LD_LIBRARY_PATH", "")
        ld_parts = [item for item in existing_ld_library_path.split(os.pathsep) if item]
        if ros_lib_dir not in ld_parts and Path(ros_lib_dir).is_dir():
            env["LD_LIBRARY_PATH"] = os.pathsep.join([ros_lib_dir, *ld_parts])
        return env

    def _bridge_command(self) -> list[str]:
        ros_args = [
            "--ros-args",
            "-p",
            f"domain_id:={self._domain_id}",
            "-p",
            f"dds_namespace:={self._dds_namespace}",
        ]

        executable = DEFAULT_ROBOT_BRIDGE_EXECUTABLE
        if executable.is_file() and os.access(executable, os.X_OK):
            return [str(executable), *ros_args]

        ros2 = shutil.which("ros2")
        if not ros2:
            raise RuntimeError(
                "igris_c_sensor_robot_bridge_node is not installed and ros2 executable is not available; "
                "build ros_ws and source ROS2 first."
            )

        return [
            ros2,
            "run",
            "igris_c_sensor",
            "igris_c_sensor_robot_bridge_node",
            *ros_args,
        ]

    def _cyclonedds_config_file(self, env: dict[str, str]) -> Optional[Path]:
        raw_uri = str(env.get("CYCLONEDDS_URI", "")).strip()
        if not raw_uri:
            return None

        parsed = urlparse(raw_uri)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        if parsed.scheme:
            return None
        return Path(raw_uri)

    def _configured_cyclonedds_interface(self, env: dict[str, str]) -> Optional[str]:
        config_file = self._cyclonedds_config_file(env)
        if config_file is None or not config_file.exists():
            return None

        try:
            root = ET.parse(config_file).getroot()
        except Exception:
            logger.warning(
                "[RobotDDSROSCameraBridge] failed to parse CYCLONEDDS_URI=%s",
                config_file,
                exc_info=True,
            )
            return None

        for element in root.findall(".//NetworkInterface"):
            name = str(element.attrib.get("name", "")).strip()
            autodetermine = str(element.attrib.get("autodetermine", "false")).strip().lower()
            if name and autodetermine not in {"1", "true", "yes", "on"}:
                return name
        return None

    def _interface_has_ipv4(self, interface_name: str) -> bool:
        try:
            result = subprocess.run(
                ["ip", "-o", "-4", "addr", "show", "dev", interface_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1.0,
                check=False,
            )
        except Exception:
            logger.debug(
                "[RobotDDSROSCameraBridge] failed to inspect interface %s",
                interface_name,
                exc_info=True,
            )
            return False
        return result.returncode == 0 and bool(result.stdout.strip())

    def _wait_for_configured_cyclonedds_interface(self, env: dict[str, str]) -> None:
        interface_name = self._configured_cyclonedds_interface(env)
        if not interface_name:
            return

        deadline = time.monotonic() + max(1.0, self._startup_timeout_s)
        logged_wait = False
        while time.monotonic() < deadline:
            if self._interface_has_ipv4(interface_name):
                if logged_wait:
                    logger.info(
                        "[RobotDDSROSCameraBridge] CycloneDDS interface %s is ready",
                        interface_name,
                    )
                return
            if not logged_wait:
                logger.warning(
                    "[RobotDDSROSCameraBridge] waiting for CycloneDDS interface %s to have an IPv4 address",
                    interface_name,
                )
                logged_wait = True
            time.sleep(0.25)

        raise RuntimeError(
            "CycloneDDS interface is not ready: "
            f"{interface_name}. Check LAN link/IP before starting real robot camera bridge."
        )

    def _start_process(self) -> None:
        cmd = self._bridge_command()
        env = self._default_env()

        logger.info(
            "[RobotDDSROSCameraBridge] starting robot DDS camera bridge domain_id=%d namespace=%s",
            self._domain_id,
            self._dds_namespace,
        )
        log_dir = REPO_ROOT / "igris_artifacts" / "logs" / "camera_bridge"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = log_dir / f"robot_bridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self._log_handle = self._log_file.open("a", encoding="utf-8")

        self._wait_for_configured_cyclonedds_interface(env)

        max_attempts = 3
        monitor_s = min(max(2.5, self._startup_timeout_s / 3.0), self._startup_timeout_s)
        last_returncode: Optional[int] = None
        for attempt in range(1, max_attempts + 1):
            self._process = subprocess.Popen(
                cmd,
                env=env,
                text=True,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
            )

            deadline = time.monotonic() + monitor_s
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    last_returncode = self._process.returncode
                    logger.warning(
                        "[RobotDDSROSCameraBridge] robot DDS camera bridge exited during startup "
                        "with code %s; attempt %d/%d; log=%s",
                        last_returncode,
                        attempt,
                        max_attempts,
                        self._log_file,
                    )
                    self._process = None
                    if attempt < max_attempts:
                        time.sleep(0.5)
                    break
                time.sleep(0.1)
            else:
                return

        raise RuntimeError(
            "igris_c_sensor_robot_bridge_node exited during startup "
            f"with code {last_returncode}; log={self._log_file}"
        )

    def _spin_ros(self) -> None:
        if self._iface is None:
            return
        try:
            rclpy.spin(self._iface)
        except Exception:
            if not self._stopped:
                logger.exception("[RobotDDSROSCameraBridge] ROS spin failed")

    def _create_ros_interface(self) -> None:
        if not rclpy.ok():
            try:
                rclpy.init(args=None)
            except RuntimeError:
                pass
        self._iface = CameraROSInterface()
        self._spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self._spin_thread.start()

    def poll(self) -> dict[str, object]:
        if self._process is not None and self._process.poll() is not None:
            raise RuntimeError(
                "igris_c_sensor_robot_bridge_node stopped "
                f"with code {self._process.returncode}"
            )
        if self._iface is None:
            return {}
        frames = self._iface.get_frames()
        return {key: value for key, value in frames.items() if value is not None}

    def destroy_node(self) -> None:
        if self._iface is not None:
            try:
                self._iface.destroy_node()
            except Exception:
                logger.debug("[RobotDDSROSCameraBridge] failed to destroy CameraROSInterface", exc_info=True)
            self._iface = None

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        self.destroy_node()

        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=3.0)
            self._process = None

        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                logger.debug("[RobotDDSROSCameraBridge] failed to close process log", exc_info=True)
            self._log_handle = None

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
            self._spin_thread = None
