from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from igris_teleop.core.project_paths import PACKAGE_ROOT

logger = logging.getLogger(__name__)

WAIST_CONTROLLER_TO_URDF_ORDER = np.asarray((2, 1, 0), dtype=np.intp)
VISER_EXPECTED_Q_SIZE = 31
IK_TARGET_FRAME_SPECS = {
    "left_wrist_mat": ("/targets/left_wrist", (40, 140, 255)),
    "right_wrist_mat": ("/targets/right_wrist", (255, 90, 90)),
    "head_mat": ("/targets/head", (80, 220, 120)),
    "torso_mat": ("/targets/torso", (255, 190, 60)),
    "chest_mat": ("/targets/chest", (50, 220, 220)),
}


class WebUIViserBridge:
    """Small bridge that hosts Viser and mirrors SHM robot joint state."""

    def __init__(
        self,
        shared_memory: Mapping[str, Any],
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        display_host: str | None = None,
        hz: float = 20.0,
    ) -> None:
        self.shared_memory = shared_memory
        self.host = host or "127.0.0.1"
        self.port = int(port)
        self.display_host = display_host or ("127.0.0.1" if self.host in {"", "0.0.0.0"} else self.host)
        self.hz = max(1.0, float(hz))
        self.url = f"http://{self.display_host}:{self.port}/"

        self._server: Any | None = None
        self._urdf: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_q: np.ndarray | None = None
        self._target_frames: dict[str, Any] = {}

    def start(self) -> bool:
        try:
            import viser  # type: ignore
            from viser.extras import ViserUrdf  # type: ignore
        except Exception:
            logger.exception("[viser] viser is not available; robot visualizer disabled")
            return False

        urdf_path = self._urdf_path()
        if not urdf_path.is_file():
            logger.error("[viser] URDF not found: %s", urdf_path)
            return False

        try:
            self._server = viser.ViserServer(host=self.host, port=self.port, label="IGRIS Teleop", verbose=False)
            self.url = f"http://{self.display_host}:{self.port}/"
            self._server.initial_camera.position = (1.4, -2.2, 1.1)
            self._server.initial_camera.look_at = (0.0, 0.0, 0.25)
            self._server.initial_camera.up = (0.0, 0.0, 1.0)
            self._server.scene.add_grid("/grid", width=2.0, height=2.0, plane="xy", position=(0.0, 0.0, -0.55))
            self._urdf = ViserUrdf(
                self._server,
                urdf_path,
                root_node_name="/igris",
                load_meshes=True,
                load_collision_meshes=False,
            )
            joint_count = len(self._urdf.get_actuated_joint_names())
            self._last_q = np.zeros(joint_count, dtype=np.float64)
            self._urdf.update_cfg(self._last_q)
            self._init_target_frames()
        except Exception:
            logger.exception("[viser] failed to start Viser server")
            self.stop()
            return False

        self._thread = threading.Thread(target=self._run, name="web-ui-viser-bridge", daemon=True)
        self._thread.start()
        logger.info("[viser] listening on %s", self.url)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                logger.debug("[viser] server stop failed", exc_info=True)
        self._server = None
        self._urdf = None
        self._target_frames = {}

    @staticmethod
    def _urdf_path() -> Path:
        return (PACKAGE_ROOT / "robot_control" / "asset" / "urdf" / "igris_c_v2_pelvis.urdf").resolve()

    def _run(self) -> None:
        period = 1.0 / self.hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                q = self._read_robot_q()
                if q is not None and self._urdf is not None:
                    self._last_q = q
                    self._urdf.update_cfg(q)
                self._update_target_frame_handles(self._read_target_frames())
            except Exception:
                logger.debug("[viser] robot state update failed", exc_info=True)
            elapsed = time.monotonic() - t0
            self._stop.wait(max(0.0, period - elapsed))

    def _init_target_frames(self) -> None:
        if self._server is None:
            return
        for field, (name, color) in IK_TARGET_FRAME_SPECS.items():
            self._target_frames[field] = self._server.scene.add_frame(
                name,
                axes_length=0.14,
                axes_radius=0.006,
                origin_radius=0.018,
                origin_color=color,
                visible=False,
            )

    @staticmethod
    def _as_valid_mat4(mat: np.ndarray | None) -> np.ndarray | None:
        if mat is None:
            return None
        arr = np.asarray(mat, dtype=np.float64)
        if arr.shape != (4, 4):
            return None
        if not np.all(np.isfinite(arr)):
            return None
        if not np.any(arr):
            return None
        return arr

    @staticmethod
    def _rotation_matrix_to_wxyz(rot_mat: np.ndarray) -> np.ndarray:
        rot = np.asarray(rot_mat, dtype=np.float64).reshape(3, 3)
        u, _, vt = np.linalg.svd(rot)
        rot = u @ vt
        if np.linalg.det(rot) < 0.0:
            u[:, -1] *= -1.0
            rot = u @ vt

        trace = float(np.trace(rot))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            quat = np.array(
                [
                    0.25 * s,
                    (rot[2, 1] - rot[1, 2]) / s,
                    (rot[0, 2] - rot[2, 0]) / s,
                    (rot[1, 0] - rot[0, 1]) / s,
                ],
                dtype=np.float64,
            )
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            quat = np.array(
                [
                    (rot[2, 1] - rot[1, 2]) / s,
                    0.25 * s,
                    (rot[0, 1] + rot[1, 0]) / s,
                    (rot[0, 2] + rot[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif rot[1, 1] > rot[2, 2]:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quat = np.array(
                [
                    (rot[0, 2] - rot[2, 0]) / s,
                    (rot[0, 1] + rot[1, 0]) / s,
                    0.25 * s,
                    (rot[1, 2] + rot[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quat = np.array(
                [
                    (rot[1, 0] - rot[0, 1]) / s,
                    (rot[0, 2] + rot[2, 0]) / s,
                    (rot[1, 2] + rot[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
        norm = float(np.linalg.norm(quat))
        if norm <= 0.0 or not np.isfinite(norm):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        quat = quat / norm
        if quat[0] < 0.0:
            quat = -quat
        return quat

    def _read_target_frames(self) -> dict[str, np.ndarray]:
        shm = self.shared_memory.get("ik_target_shm")
        if shm is None:
            return {}
        try:
            data = shm.read_data()
        except Exception:
            return {}
        try:
            if float(np.asarray(data.get("target_valid", 0.0)).reshape(())) < 0.5:
                return {}
        except Exception:
            return {}

        frames: dict[str, np.ndarray] = {}
        torso_target_valid = False
        chest_target_valid = False
        try:
            torso_target_valid = float(np.asarray(data.get("torso_target_valid", 0.0)).reshape(())) >= 0.5
        except Exception:
            torso_target_valid = False
        try:
            chest_target_valid = float(np.asarray(data.get("chest_target_valid", 0.0)).reshape(())) >= 0.5
        except Exception:
            chest_target_valid = False

        for field in IK_TARGET_FRAME_SPECS:
            if field == "torso_mat" and not torso_target_valid:
                continue
            if field == "chest_mat" and not chest_target_valid:
                continue
            mat = self._as_valid_mat4(data.get(field))
            if mat is not None:
                frames[field] = mat.copy()
        return frames

    def _update_target_frame_handles(self, target_frames: Mapping[str, np.ndarray]) -> None:
        if not self._target_frames:
            return
        for field, handle in self._target_frames.items():
            mat = target_frames.get(field)
            if mat is None:
                handle.visible = False
                continue
            handle.position = mat[:3, 3]
            handle.wxyz = self._rotation_matrix_to_wxyz(mat[:3, :3])
            handle.visible = True

    def _read_robot_q(self) -> np.ndarray | None:
        act_q = self._read_q_from_shm("act_shm", "act")
        obs_q = self._read_q_from_shm("obs_shm", "obs")
        if act_q is not None and np.linalg.norm(act_q) > 1e-8:
            return act_q
        if obs_q is not None:
            return obs_q
        return act_q if act_q is not None else self._last_q

    def _read_q_from_shm(self, shm_name: str, prefix: str) -> np.ndarray | None:
        shm = self.shared_memory.get(shm_name)
        if shm is None:
            return None
        try:
            data = shm.read_data()
        except Exception:
            return None

        fields = (
            f"{prefix}_waist",
            f"{prefix}_leg",
            f"{prefix}_arm",
            f"{prefix}_neck",
        )
        parts: list[np.ndarray] = []
        for field in fields:
            raw = data.get(field)
            if raw is None:
                return None
            arr = np.asarray(raw, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(arr)):
                return None
            if field == f"{prefix}_waist":
                if arr.size != WAIST_CONTROLLER_TO_URDF_ORDER.size:
                    return None
                # SHM/controller waist order is yaw, roll, pitch; URDF/Viser starts pitch, roll, yaw.
                arr = arr[WAIST_CONTROLLER_TO_URDF_ORDER]
            parts.append(arr)
        q = np.concatenate(parts)
        if q.size != VISER_EXPECTED_Q_SIZE:
            return None
        return q
