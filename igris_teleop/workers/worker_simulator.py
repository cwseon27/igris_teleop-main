from __future__ import annotations

import os
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import logging_mp
import numpy as np
import yaml

from ..core.project_paths import (
    JOINT_SETTING_PATH,
    REPO_ROOT,
    ROBOT_CONTROL_LOGS_ROOT,
    SIM_ARM_GAIN_PATH,
    SIM_NECK_GAIN_PATH,
    SIM_WAIST_GAIN_PATH,
    WALKING_JOINT_SETTING_PATH,
)
from ..core.rate import Rate
from ..core.worker_base import SIM_ROBOT_DDS_DOMAIN_ID, WorkerContext
from ..robot_control.controller.core import load_joint_profile
from ..robot_control.kinematics.joints import ARM_INDICES, LEG_INDICES, NECK_INDICES, NUM_MOTORS, WAIST_INDICES
from ..robot_control.kinematics.pr2ab import (
    default_pr2ab_pairs,
    load_pr2ab_transforms_from_yaml,
    map_ms_linear_to_pjs,
    map_ms_to_pjs,
    map_pjs_linear_to_ms,
    map_pjs_to_ms,
)
from ..policies.walking import WALKING_NEUTRAL_HAND_Q
from ..sim.experiment_scenes import (
    EXPERIMENT_TASK_IDS,
    EXPERIMENT_VISUAL_ONLY_SUFFIX,
    experiment_task_id_from_name,
)
from ..sim.hand_physics import (
    HAND_NEAR_OBJECT_CLOSING_TORQUE_LIMIT,
    HAND_NEAR_OBJECT_OPENING_BRAKE_LIMIT,
    HAND_SERVO_KD,
    HAND_SERVO_KP,
)
from ..sim.stereo_camera import (
    STEREO_CAMERA_BASELINE_M,
    STEREO_CAMERA_HEIGHT,
    STEREO_CAMERA_HZ,
    STEREO_CAMERA_WIDTH,
    STEREO_LEFT_CAMERA_NAME,
    STEREO_RIGHT_CAMERA_NAME,
    validate_stereo_baseline,
)

try:
    import igris_c_sdk as igc_sdk
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("igris_c_sdk is required for the simulator worker") from exc

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("mujoco is required for the simulator worker") from exc


logger = logging_mp.get_logger(__name__, level=logging_mp.INFO)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}

SIM_DOMAIN_ID = SIM_ROBOT_DDS_DOMAIN_ID
SIM_CONTROL_HZ = 300.0
SIM_PHYSICS_HZ = 3000.0
SIM_RENDER_HZ = 60.0
SIM_CONTROL_DT = 1.0 / SIM_CONTROL_HZ
SIM_TIMESTEP = 1.0 / SIM_PHYSICS_HZ
SIM_RENDER_DT = 1.0 / SIM_RENDER_HZ
SIM_CAMERA_DT = 1.0 / STEREO_CAMERA_HZ
SIM_SUBSTEPS_PER_CONTROL = int(round(SIM_CONTROL_DT / SIM_TIMESTEP))
SIM_SUSPENDED_Z_OFFSET = 0.25
WALKING_MODE_NAME = "walking"

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_BMS_REQUEST = "rt/service/bms_init/request"
TOPIC_BMS_RESPONSE = "rt/service/bms_init/response"
TOPIC_TORQUE_REQUEST = "rt/service/torque/request"
TOPIC_TORQUE_RESPONSE = "rt/service/torque/response"
TOPIC_CONTROL_MODE_REQUEST = "rt/service/control_mode/request"
TOPIC_CONTROL_MODE_RESPONSE = "rt/service/control_mode/response"

FLOATING_BASE_JOINT = "floating_base"
PELVIS_SITE_NAME = "pelvis_link_site"
LEFT_SOLE_SITE_NAME = "left_sole_site"
RIGHT_SOLE_SITE_NAME = "right_sole_site"
FLOOR_GEOM_NAME = "floor"
LEFT_FOOT_COLLISION_GEOM_NAME = "Link_Ankle_Roll_Left_collision"
RIGHT_FOOT_COLLISION_GEOM_NAME = "Link_Ankle_Roll_Right_collision"
IMU_GYRO_SENSOR_NAME = "imu-pelvis-angular-velocity"
IMU_ACCEL_SENSOR_NAME = "imu-pelvis-linear-acceleration"
IMU_QUAT_SENSOR_NAME = "imu-pelvis-quat"
TOPIC_HANDCMD = "rt/handcmd"
TOPIC_HANDSTATE = "rt/handstate"
DEFAULT_XML_PATH = (
    REPO_ROOT / "igris_teleop" / "sim" / "robot" / "mujoco" / "igris_c_v2_with_hand.xml"
).resolve()
DEFAULT_PR2AB_LOG_PATH = (ROBOT_CONTROL_LOGS_ROOT / "pr2ab_calibration.yaml").resolve()
HAND_TARGET_LENGTH = 12
HAND_MOTOR_IDS: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26)
HAND_INIT_TRIGGER_ID = 99
HAND_INIT_SWEEP_TARGET = 0.2
HAND_INIT_SWEEP_DURATION_S = 0.6
WALKING_STARTUP_PHASE_BLENDING = 2.0
WALKING_STARTUP_PHASE_RUNNING = 3.0
WALKING_SOLE_TARGET_HEIGHT = 0.0095
WALKING_WAIST_LOCK_KP = np.asarray((1200.0, 800.0, 1200.0), dtype=np.float32)
WALKING_WAIST_LOCK_KD = np.asarray((60.0, 40.0, 60.0), dtype=np.float32)
WAIST_INDEX_ARRAY = np.asarray(tuple(int(idx) for idx in WAIST_INDICES), dtype=np.int32)
ARM_INDEX_ARRAY = np.asarray(tuple(int(idx) for idx in ARM_INDICES), dtype=np.int32)
NECK_INDEX_ARRAY = np.asarray(tuple(int(idx) for idx in NECK_INDICES), dtype=np.int32)
UPPER_BODY_GRAVITY_COMPENSATION_INDEX_ARRAY = np.concatenate(
    (ARM_INDEX_ARRAY, NECK_INDEX_ARRAY),
).astype(np.int32, copy=False)
HAND_COLLISION_CATEGORY = 2
EXPERIMENT_SUPPORT_COLLISION_CONTYPE = 4
EXPERIMENT_SUPPORT_COLLISION_CONAFFINITY = 7
EXPERIMENT_GRASP_COLLISION_CONTYPE = 8
EXPERIMENT_GRASP_COLLISION_CONAFFINITY = 14
EXPERIMENT_MARKER_ALPHA = 0.42

PJS_JOINT_NAMES: tuple[str, ...] = (
    "Joint_Waist_Yaw",
    "Joint_Waist_Roll",
    "Joint_Waist_Pitch",
    "Joint_Hip_Pitch_Left",
    "Joint_Hip_Roll_Left",
    "Joint_Hip_Yaw_Left",
    "Joint_Knee_Pitch_Left",
    "Joint_Ankle_Pitch_Left",
    "Joint_Ankle_Roll_Left",
    "Joint_Hip_Pitch_Right",
    "Joint_Hip_Roll_Right",
    "Joint_Hip_Yaw_Right",
    "Joint_Knee_Pitch_Right",
    "Joint_Ankle_Pitch_Right",
    "Joint_Ankle_Roll_Right",
    "Joint_Shoulder_Pitch_Left",
    "Joint_Shoulder_Roll_Left",
    "Joint_Shoulder_Yaw_Left",
    "Joint_Elbow_Pitch_Left",
    "Joint_Wrist_Yaw_Left",
    "Joint_Wrist_Roll_Left",
    "Joint_Wrist_Pitch_Left",
    "Joint_Shoulder_Pitch_Right",
    "Joint_Shoulder_Roll_Right",
    "Joint_Shoulder_Yaw_Right",
    "Joint_Elbow_Pitch_Right",
    "Joint_Wrist_Yaw_Right",
    "Joint_Wrist_Roll_Right",
    "Joint_Wrist_Pitch_Right",
    "Joint_Neck_Yaw",
    "Joint_Neck_Pitch",
)

HAND_JOINT_NAMES_DDS_ORDER: tuple[str, ...] = (
    "Joint_Thumb_Middle_Right",
    "Joint_Index_Middle_Right",
    "Joint_Middle_Middle_Right",
    "Joint_Ring_Middle_Right",
    "Joint_Little_Middle_Right",
    "Joint_Thumb_Proximal_Right",
    "Joint_Thumb_Middle_Left",
    "Joint_Index_Middle_Left",
    "Joint_Middle_Middle_Left",
    "Joint_Ring_Middle_Left",
    "Joint_Little_Middle_Left",
    "Joint_Thumb_Proximal_Left",
)
HAND_REVERSED_NORMALIZED_JOINTS: frozenset[str] = frozenset({"Joint_Thumb_Proximal_Right"})
HAND_MOTOR_ID_TO_INDEX: dict[int, int] = {motor_id: idx for idx, motor_id in enumerate(HAND_MOTOR_IDS)}


@dataclass(frozen=True)
class JointBinding:
    name: str
    joint_id: int
    actuator_id: int
    qpos_adr: int
    dof_adr: int
    ctrl_limit: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ExperimentFreeJointBinding:
    name: str
    task_id: int
    qpos_adr: int
    dof_adr: int
    body_id: int
    initial_qpos: np.ndarray


def _quat_wxyz_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat_wxyz]

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def _mat_to_quat_wxyz(rot_mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(rot_mat, dtype=np.float64).reshape(9)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


class MujocoSimulationWorker:
    def __init__(self, ctx: WorkerContext) -> None:
        self.ctx = ctx
        self._shared_memory = ctx.shared_memory or {}
        self._stop_event = ctx.stop_event
        self.walking_cmd_shm = self._shared_memory.get("walking_cmd_shm")
        self.walking_debug_shm = self._shared_memory.get("walking_debug_shm")
        self.act_shm = self._shared_memory.get("act_shm")
        self.obs_shm = self._shared_memory.get("obs_shm")
        self.mode_shm = self._shared_memory.get("mode_shm")
        self.camera_shm = self._shared_memory.get("camera_shm")
        self.sim_config_shm = self._shared_memory.get("sim_config_shm")

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.viewer: mujoco.viewer.Handle | None = None
        self._camera_thread: threading.Thread | None = None
        self._camera_thread_stop = threading.Event()
        self._camera_state_lock = threading.Lock()
        self._camera_qpos_snapshot: np.ndarray | None = None

        self._joint_bindings: tuple[JointBinding, ...] = ()
        self._joint_qpos_adrs = np.zeros(NUM_MOTORS, dtype=np.int32)
        self._joint_dof_adrs = np.zeros(NUM_MOTORS, dtype=np.int32)
        self._actuator_ids = np.zeros(NUM_MOTORS, dtype=np.int32)
        self._ctrl_limits = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._joint_lower = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._joint_upper = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._hand_bindings: tuple[JointBinding, ...] = ()
        self._hand_qpos_adrs = np.zeros(HAND_TARGET_LENGTH, dtype=np.int32)
        self._hand_dof_adrs = np.zeros(HAND_TARGET_LENGTH, dtype=np.int32)
        self._hand_actuator_ids = np.zeros(HAND_TARGET_LENGTH, dtype=np.int32)
        self._hand_ctrl_limits = np.zeros(HAND_TARGET_LENGTH, dtype=np.float32)
        self._hand_open_q = np.zeros(HAND_TARGET_LENGTH, dtype=np.float32)
        self._hand_closed_q = np.zeros(HAND_TARGET_LENGTH, dtype=np.float32)
        self._hand_closing_direction = np.ones(HAND_TARGET_LENGTH, dtype=np.float32)
        self._hand_collision_geom_to_actuator_index: dict[int, int] = {}
        self._experiment_free_joints: tuple[ExperimentFreeJointBinding, ...] = ()
        self._experiment_last_command_seq: int | None = None
        self._active_experiment_task_id = 0

        self._free_qpos_slice = slice(0, 7)
        self._free_qvel_slice = slice(0, 6)
        self._pelvis_site_id = -1
        self._sole_site_ids = np.full(2, -1, dtype=np.int32)
        self._floor_geom_id = -1
        self._foot_collision_geom_ids = np.full(2, -1, dtype=np.int32)
        self._grounded_pelvis_pos = np.zeros(3, dtype=np.float64)
        self._grounded_pelvis_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._suspended_pelvis_pos = np.zeros(3, dtype=np.float64)
        self._suspended_pelvis_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self._gyro_slice = slice(0, 3)
        self._accel_slice = slice(0, 3)
        self._quat_slice = slice(0, 4)

        self._command_lock = threading.Lock()
        self._mode = igc_sdk.KinematicMode.PJS
        self._target_q = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._target_dq = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._target_tau = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._kp = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._kd = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._sim_waist_kp: np.ndarray | None = None
        self._sim_waist_kd: np.ndarray | None = None
        self._sim_arm_kp: np.ndarray | None = None
        self._sim_arm_kd: np.ndarray | None = None
        self._sim_neck_kp: np.ndarray | None = None
        self._sim_neck_kd: np.ndarray | None = None
        self._hand_command_lock = threading.Lock()
        self._hand_target_normalized = np.zeros(HAND_TARGET_LENGTH, dtype=np.float32)
        self._hand_init_started_at: float | None = None

        self._service_lock = threading.Lock()
        self._bms_init_type = igc_sdk.BmsInitType.BMS_AND_MOTOR_INIT
        self._torque_enabled = True
        self._control_mode = igc_sdk.ControlMode.CONTROL_MODE_LOW_LEVEL
        self._walking_policy_enabled_prev = False
        self._walking_restore_joint_q = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._walking_restore_free_qpos = np.zeros(7, dtype=np.float64)
        self._walking_restore_free_qvel = np.zeros(6, dtype=np.float64)
        self._walking_restore_grounded_pelvis_pos = np.zeros(3, dtype=np.float64)
        self._walking_restore_grounded_pelvis_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._walking_locked_waist_q = np.zeros(WAIST_INDEX_ARRAY.size, dtype=np.float32)
        self._walking_restore_hand_normalized = np.asarray(WALKING_NEUTRAL_HAND_Q, dtype=np.float32).reshape(-1)
        self._direct_teleop_last_log_t: float | None = None
        self._teleop_fixed_leg_q: np.ndarray | None = None
        # The simulator used to read act_shm directly in teleop mode. That
        # bypasses ControlWorker's ready/start/shutdown interpolation and makes
        # MuJoCo jump straight to the leader/IK target. Keep this only as an
        # explicit fallback for legacy debugging.
        self._direct_teleop_action_enabled = _env_flag(
            "IGRIS_SIM_DIRECT_TELEOP_ACTION",
            False,
        )
        if self._direct_teleop_action_enabled:
            logger.warning(
                "[%s] IGRIS_SIM_DIRECT_TELEOP_ACTION is enabled: simulator will bypass "
                "ControlWorker entry/shutdown interpolation and read act_shm directly.",
                self.ctx.name,
            )

        self._tick = 0
        self._pr2ab_pairs = load_pr2ab_transforms_from_yaml(
            DEFAULT_PR2AB_LOG_PATH,
            base_pairs=default_pr2ab_pairs(),
            logger=logger,
        )

        self.channel_factory = igc_sdk.ChannelFactory.Instance()
        self.lowstate_pub: igc_sdk.LowStatePublisher | None = None
        self.lowcmd_sub: igc_sdk.LowCmdSubscriber | None = None
        self.handstate_pub: igc_sdk.HandStatePublisher | None = None
        self.handcmd_sub: igc_sdk.HandCmdSubscriber | None = None
        self.bms_resp_pub: igc_sdk.ServiceResponsePublisher | None = None
        self.torque_resp_pub: igc_sdk.ServiceResponsePublisher | None = None
        self.control_mode_resp_pub: igc_sdk.ServiceResponsePublisher | None = None
        self.bms_req_sub: igc_sdk.BmsInitCmdSubscriber | None = None
        self.torque_req_sub: igc_sdk.TorqueCmdSubscriber | None = None
        self.control_mode_req_sub: igc_sdk.ControlModeCmdSubscriber | None = None

    def should_stop(self) -> bool:
        if self._stop_event is None:
            return False
        try:
            return bool(self._stop_event.is_set())
        except Exception:
            return False

    @staticmethod
    def _sim_camera_always_on() -> bool:
        value = os.environ.get("IGRIS_SIM_CAMERA_ALWAYS_ON", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _sim_camera_requested(self) -> bool:
        if self._sim_camera_always_on():
            return True
        try:
            return bool(self.ctx.bus.is_level_set("camera"))
        except Exception:
            return False

    def _capture_camera_state(self) -> None:
        if self.data is None or self.camera_shm is None:
            return
        snapshot = np.asarray(self.data.qpos, dtype=np.float64).copy()
        with self._camera_state_lock:
            self._camera_qpos_snapshot = snapshot

    def _sim_stereo_baseline_m(self) -> float:
        if self.sim_config_shm is None:
            return STEREO_CAMERA_BASELINE_M
        try:
            raw_baseline = self.sim_config_shm.read_data()["stereo_baseline_m"]
            value = float(np.asarray(raw_baseline).reshape(-1)[0])
            return validate_stereo_baseline(value)
        except (KeyError, TypeError, ValueError, IndexError):
            return STEREO_CAMERA_BASELINE_M

    def _camera_render_loop(self) -> None:
        if self.model is None or self.camera_shm is None:
            return

        # The render model is private to this thread so changing camera extrinsics
        # cannot race the physics model. Only the cameras' lateral positions move.
        camera_model = mujoco.MjModel.from_xml_path(str(self._xml_path()))
        camera_data = mujoco.MjData(camera_model)
        if camera_model.nq != self.model.nq:
            raise RuntimeError("Stereo render model qpos layout differs from the physics model")
        left_camera_id = mujoco.mj_name2id(
            camera_model, mujoco.mjtObj.mjOBJ_CAMERA, STEREO_LEFT_CAMERA_NAME
        )
        right_camera_id = mujoco.mj_name2id(
            camera_model, mujoco.mjtObj.mjOBJ_CAMERA, STEREO_RIGHT_CAMERA_NAME
        )
        if left_camera_id < 0 or right_camera_id < 0:
            raise RuntimeError("Stereo cameras are missing from the render model")
        camera_center_y = 0.5 * float(
            camera_model.cam_pos[left_camera_id, 1] + camera_model.cam_pos[right_camera_id, 1]
        )
        applied_baseline_m: float | None = None
        applied_experiment_task_id: int | None = None
        renderer: mujoco.Renderer | None = None
        next_frame_at = time.perf_counter()
        last_error_log_at = 0.0
        try:
            while not self.should_stop() and not self._camera_thread_stop.is_set():
                if not self._sim_camera_requested():
                    self._camera_thread_stop.wait(0.2)
                    next_frame_at = time.perf_counter()
                    continue

                try:
                    if renderer is None:
                        renderer = mujoco.Renderer(
                            camera_model,
                            height=STEREO_CAMERA_HEIGHT,
                            width=STEREO_CAMERA_WIDTH,
                        )
                        logger.info(
                            "[%s] simulated stereo renderer started (%dx%d @ %.1f Hz)",
                            self.ctx.name,
                            STEREO_CAMERA_WIDTH,
                            STEREO_CAMERA_HEIGHT,
                            STEREO_CAMERA_HZ,
                        )

                    with self._camera_state_lock:
                        qpos = None if self._camera_qpos_snapshot is None else self._camera_qpos_snapshot.copy()
                    if qpos is None:
                        self._camera_thread_stop.wait(0.05)
                        continue

                    baseline_m = self._sim_stereo_baseline_m()
                    if baseline_m != applied_baseline_m:
                        camera_model.cam_pos[left_camera_id, 1] = camera_center_y + 0.5 * baseline_m
                        camera_model.cam_pos[right_camera_id, 1] = camera_center_y - 0.5 * baseline_m
                        applied_baseline_m = baseline_m
                        logger.info(
                            "[%s] sim stereo eye distance %.1f mm (translation only)",
                            self.ctx.name,
                            baseline_m * 1000.0,
                        )

                    experiment_task_id = int(self._active_experiment_task_id)
                    if experiment_task_id != applied_experiment_task_id:
                        self._set_experiment_geom_visibility(camera_model, experiment_task_id)
                        applied_experiment_task_id = experiment_task_id

                    camera_data.qpos[:] = qpos
                    mujoco.mj_forward(camera_model, camera_data)
                    renderer.update_scene(camera_data, camera=STEREO_LEFT_CAMERA_NAME)
                    left = np.ascontiguousarray(renderer.render()).copy()
                    renderer.update_scene(camera_data, camera=STEREO_RIGHT_CAMERA_NAME)
                    right = np.ascontiguousarray(renderer.render()).copy()
                    self.camera_shm.write_data(stereo_left=left, stereo_right=right)
                except Exception as exc:
                    now = time.perf_counter()
                    if now - last_error_log_at >= 5.0:
                        last_error_log_at = now
                        logger.warning("[%s] simulated stereo render failed: %s", self.ctx.name, exc)
                    if renderer is not None:
                        try:
                            renderer.close()
                        except Exception:
                            pass
                        renderer = None
                    self._camera_thread_stop.wait(1.0)

                next_frame_at += SIM_CAMERA_DT
                delay = next_frame_at - time.perf_counter()
                if delay > 0.0:
                    self._camera_thread_stop.wait(delay)
                elif delay < -SIM_CAMERA_DT:
                    next_frame_at = time.perf_counter()
        finally:
            if renderer is not None:
                try:
                    renderer.close()
                except Exception:
                    logger.debug("[%s] failed to close stereo renderer", self.ctx.name, exc_info=True)

    def _start_camera_renderer(self) -> None:
        if self.camera_shm is None or self._camera_thread is not None:
            return
        self._capture_camera_state()
        self._camera_thread_stop.clear()
        self._camera_thread = threading.Thread(
            target=self._camera_render_loop,
            name="mujoco-stereo-renderer",
            daemon=True,
        )
        self._camera_thread.start()

    def _stop_camera_renderer(self) -> None:
        self._camera_thread_stop.set()
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=3.0)
            if self._camera_thread.is_alive():
                logger.warning("[%s] simulated stereo renderer did not stop within timeout", self.ctx.name)
            self._camera_thread = None

    def _xml_path(self) -> Path:
        env = os.environ.get("IGRIS_SIM_XML_PATH")
        if env:
            return Path(env).expanduser().resolve()
        return DEFAULT_XML_PATH

    def _resolve_bindings(
        self,
        model: mujoco.MjModel,
        names: tuple[str, ...],
    ) -> tuple[
        tuple[JointBinding, ...],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        bindings: list[JointBinding] = []
        qpos_adrs = np.zeros(len(names), dtype=np.int32)
        dof_adrs = np.zeros(len(names), dtype=np.int32)
        actuator_ids = np.zeros(len(names), dtype=np.int32)
        ctrl_limits = np.zeros(len(names), dtype=np.float32)
        lowers = np.zeros(len(names), dtype=np.float32)
        uppers = np.zeros(len(names), dtype=np.float32)
        for idx, name in enumerate(names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"MuJoCo XML is missing required joint/actuator: {name}")
            qpos_adr = int(model.jnt_qposadr[joint_id])
            dof_adr = int(model.jnt_dofadr[joint_id])
            ctrl_limit = float(
                max(
                    abs(float(model.actuator_ctrlrange[actuator_id, 0])),
                    abs(float(model.actuator_ctrlrange[actuator_id, 1])),
                )
            )
            lower = float(model.jnt_range[joint_id, 0])
            upper = float(model.jnt_range[joint_id, 1])
            bindings.append(
                JointBinding(
                    name=name,
                    joint_id=joint_id,
                    actuator_id=actuator_id,
                    qpos_adr=qpos_adr,
                    dof_adr=dof_adr,
                    ctrl_limit=ctrl_limit,
                    lower=lower,
                    upper=upper,
                )
            )
            qpos_adrs[idx] = qpos_adr
            dof_adrs[idx] = dof_adr
            actuator_ids[idx] = actuator_id
            ctrl_limits[idx] = ctrl_limit
            lowers[idx] = lower
            uppers[idx] = upper
        return tuple(bindings), qpos_adrs, dof_adrs, actuator_ids, ctrl_limits, lowers, uppers

    def _resolve_joint_bindings(self, model: mujoco.MjModel) -> tuple[JointBinding, ...]:
        (
            bindings,
            self._joint_qpos_adrs,
            self._joint_dof_adrs,
            self._actuator_ids,
            self._ctrl_limits,
            self._joint_lower,
            self._joint_upper,
        ) = self._resolve_bindings(model, PJS_JOINT_NAMES)
        return bindings

    def _resolve_hand_bindings(self, model: mujoco.MjModel) -> tuple[JointBinding, ...]:
        (
            bindings,
            self._hand_qpos_adrs,
            self._hand_dof_adrs,
            self._hand_actuator_ids,
            self._hand_ctrl_limits,
            lower,
            upper,
        ) = self._resolve_bindings(model, HAND_JOINT_NAMES_DDS_ORDER)
        self._hand_open_q = lower.copy()
        self._hand_closed_q = upper.copy()
        for idx, binding in enumerate(bindings):
            if binding.name in HAND_REVERSED_NORMALIZED_JOINTS:
                self._hand_open_q[idx] = upper[idx]
                self._hand_closed_q[idx] = lower[idx]
        self._hand_closing_direction = np.sign(
            self._hand_closed_q - self._hand_open_q
        ).astype(np.float32)
        return bindings

    def _resolve_sensor_slice(self, name: str, expected_dim: int) -> slice:
        assert self.model is not None
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise ValueError(f"MuJoCo XML is missing required sensor: {name}")
        adr = int(self.model.sensor_adr[sensor_id])
        dim = int(self.model.sensor_dim[sensor_id])
        if dim != expected_dim:
            raise ValueError(f"Sensor {name} dim {dim} != expected {expected_dim}")
        return slice(adr, adr + dim)

    @staticmethod
    def _body_has_free_joint(model: mujoco.MjModel, body_id: int) -> bool:
        current_body_id = int(body_id)
        while current_body_id > 0:
            joint_adr = int(model.body_jntadr[current_body_id])
            joint_num = int(model.body_jntnum[current_body_id])
            for joint_id in range(joint_adr, joint_adr + joint_num):
                if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                    return True
            current_body_id = int(model.body_parentid[current_body_id])
        return False

    @staticmethod
    def _set_experiment_geom_visibility(
        model: mujoco.MjModel,
        active_task_id: int,
    ) -> None:
        for geom_id in range(int(model.ngeom)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            task_id = experiment_task_id_from_name(name)
            if task_id is None:
                continue
            enabled = task_id == int(active_task_id)
            visual_only = str(name).endswith(EXPERIMENT_VISUAL_ONLY_SUFFIX)
            model.geom_rgba[geom_id, 3] = (
                EXPERIMENT_MARKER_ALPHA if enabled and visual_only else float(enabled)
            )
            if enabled and not visual_only:
                if MujocoSimulationWorker._body_has_free_joint(
                    model,
                    int(model.geom_bodyid[geom_id]),
                ):
                    model.geom_contype[geom_id] = EXPERIMENT_GRASP_COLLISION_CONTYPE
                    model.geom_conaffinity[geom_id] = (
                        EXPERIMENT_GRASP_COLLISION_CONAFFINITY
                    )
                else:
                    model.geom_contype[geom_id] = EXPERIMENT_SUPPORT_COLLISION_CONTYPE
                    model.geom_conaffinity[geom_id] = (
                        EXPERIMENT_SUPPORT_COLLISION_CONAFFINITY
                    )
            else:
                model.geom_contype[geom_id] = 0
                model.geom_conaffinity[geom_id] = 0

    @staticmethod
    def _resolve_hand_collision_geom_to_actuator_index(
        model: mujoco.MjModel,
    ) -> dict[int, int]:
        actuator_index_by_joint_name = {
            name: index for index, name in enumerate(HAND_JOINT_NAMES_DDS_ORDER)
        }
        geom_to_actuator_index: dict[int, int] = {}
        for geom_id in range(int(model.ngeom)):
            if not int(model.geom_contype[geom_id]) & HAND_COLLISION_CATEGORY:
                continue
            body_id = int(model.geom_bodyid[geom_id])
            while body_id > 0:
                joint_adr = int(model.body_jntadr[body_id])
                joint_num = int(model.body_jntnum[body_id])
                actuator_index = None
                for joint_id in range(joint_adr, joint_adr + joint_num):
                    joint_name = mujoco.mj_id2name(
                        model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        joint_id,
                    )
                    actuator_index = actuator_index_by_joint_name.get(str(joint_name))
                    if actuator_index is not None:
                        break
                if actuator_index is not None:
                    geom_to_actuator_index[geom_id] = actuator_index
                    break
                body_id = int(model.body_parentid[body_id])
        return geom_to_actuator_index

    @staticmethod
    def _resolve_experiment_free_joints(
        model: mujoco.MjModel,
    ) -> tuple[ExperimentFreeJointBinding, ...]:
        bindings: list[ExperimentFreeJointBinding] = []
        for joint_id in range(int(model.njnt)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            task_id = experiment_task_id_from_name(name)
            if task_id is None:
                continue
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError(f"Experiment object joint must be free: {name}")
            qpos_adr = int(model.jnt_qposadr[joint_id])
            dof_adr = int(model.jnt_dofadr[joint_id])
            bindings.append(
                ExperimentFreeJointBinding(
                    name=str(name),
                    task_id=task_id,
                    qpos_adr=qpos_adr,
                    dof_adr=dof_adr,
                    body_id=int(model.jnt_bodyid[joint_id]),
                    initial_qpos=np.asarray(
                        model.qpos0[qpos_adr : qpos_adr + 7],
                        dtype=np.float64,
                    ).copy(),
                )
            )
        return tuple(bindings)

    def _initialize_experiment_scene(self) -> None:
        assert self.model is not None
        assert self.data is not None

        geom_task_ids: set[int] = set()
        for geom_id in range(int(self.model.ngeom)):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            task_id = experiment_task_id_from_name(name)
            if task_id is not None:
                geom_task_ids.add(task_id)
        missing = sorted(EXPERIMENT_TASK_IDS.difference(geom_task_ids))
        if missing:
            raise ValueError(f"MuJoCo XML is missing experiment task geometry: {missing}")

        self._experiment_free_joints = self._resolve_experiment_free_joints(self.model)
        self._set_experiment_geom_visibility(self.model, 0)
        for binding in self._experiment_free_joints:
            self.data.qpos[binding.qpos_adr : binding.qpos_adr + 7] = binding.initial_qpos
            self.data.qvel[binding.dof_adr : binding.dof_adr + 6] = 0.0
        self._active_experiment_task_id = 0
        self._experiment_last_command_seq = None
        mujoco.mj_forward(self.model, self.data)

    def _apply_experiment_scene(self, task_id: int, command_seq: int) -> None:
        assert self.model is not None
        assert self.data is not None
        if task_id != 0 and task_id not in EXPERIMENT_TASK_IDS:
            raise ValueError(f"Unknown experiment task id: {task_id}")

        self._set_experiment_geom_visibility(self.model, task_id)
        for binding in self._experiment_free_joints:
            self.data.qpos[binding.qpos_adr : binding.qpos_adr + 7] = binding.initial_qpos
            self.data.qvel[binding.dof_adr : binding.dof_adr + 6] = 0.0
            self.data.qacc[binding.dof_adr : binding.dof_adr + 6] = 0.0
            self.data.qacc_warmstart[binding.dof_adr : binding.dof_adr + 6] = 0.0
            self.data.qfrc_applied[binding.dof_adr : binding.dof_adr + 6] = 0.0
            self.data.xfrc_applied[binding.body_id, :] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._active_experiment_task_id = int(task_id)
        if self.sim_config_shm is not None:
            self.sim_config_shm.write_data(
                scene_applied_seq=float(command_seq),
                scene_active_task_id=float(task_id),
                scene_status_code=2.0,
            )
        logger.info(
            "[%s] experiment scene applied: task_id=%d command_seq=%d",
            self.ctx.name,
            task_id,
            command_seq,
        )

    def _sync_experiment_scene_command(self, *, force: bool = False) -> None:
        if self.sim_config_shm is None:
            if force:
                self._apply_experiment_scene(0, 0)
            return
        try:
            config = self.sim_config_shm.read_data()
            command_seq = int(round(float(np.asarray(config["scene_command_seq"]).reshape(()))))
            task_id = int(round(float(np.asarray(config["scene_task_id"]).reshape(()))))
        except (KeyError, TypeError, ValueError):
            command_seq = 0
            task_id = 0

        if not force and command_seq == self._experiment_last_command_seq:
            return
        self._experiment_last_command_seq = command_seq
        try:
            self._apply_experiment_scene(task_id, command_seq)
        except Exception:
            logger.exception(
                "[%s] failed to apply experiment scene task_id=%d command_seq=%d",
                self.ctx.name,
                task_id,
                command_seq,
            )
            self.sim_config_shm.write_data(
                scene_applied_seq=float(command_seq),
                scene_active_task_id=float(self._active_experiment_task_id),
                scene_status_code=-1.0,
            )

    def _load_hold_gains(self) -> tuple[np.ndarray, np.ndarray]:
        profile_path = WALKING_JOINT_SETTING_PATH if self.ctx.run_config.mode == "walking" else JOINT_SETTING_PATH
        kp, kd, _default_q, _waypoint_1, _waypoint_2 = load_joint_profile(profile_path)
        return np.asarray(kp, dtype=np.float32), np.asarray(kd, dtype=np.float32)

    def _load_sim_waist_gains(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._load_sim_group_gains(
            SIM_WAIST_GAIN_PATH,
            expected_size=WAIST_INDEX_ARRAY.size,
            group_name="waist",
        )

    def _load_sim_arm_gains(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._load_sim_group_gains(
            SIM_ARM_GAIN_PATH,
            expected_size=ARM_INDEX_ARRAY.size,
            group_name="arm",
        )

    def _load_sim_neck_gains(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._load_sim_group_gains(
            SIM_NECK_GAIN_PATH,
            expected_size=NECK_INDEX_ARRAY.size,
            group_name="neck",
        )

    def _load_sim_group_gains(
        self,
        path: Path,
        *,
        expected_size: int,
        group_name: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not path.is_file():
            return None, None

        with path.open("r", encoding="utf-8") as file_obj:
            cfg = yaml.safe_load(file_obj) or {}

        mode_key = WALKING_MODE_NAME if self.ctx.run_config.mode == WALKING_MODE_NAME else "default"
        section = cfg.get(mode_key) or cfg.get("default")
        if section is None:
            return None, None

        kp = np.asarray(section["kp"], dtype=np.float32).reshape(-1)
        kd = np.asarray(section["kd"], dtype=np.float32).reshape(-1)
        if kp.size != expected_size:
            raise ValueError(
                f"sim {group_name} kp length {kp.size} != {expected_size} in {path}"
            )
        if kd.size != expected_size:
            raise ValueError(
                f"sim {group_name} kd length {kd.size} != {expected_size} in {path}"
            )
        return kp, kd

    def _apply_sim_waist_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        if self._sim_waist_kp is None or self._sim_waist_kd is None:
            return
        kp[WAIST_INDEX_ARRAY] = self._sim_waist_kp
        kd[WAIST_INDEX_ARRAY] = self._sim_waist_kd

    def _apply_sim_arm_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        if self._sim_arm_kp is None or self._sim_arm_kd is None:
            return
        kp[ARM_INDEX_ARRAY] = self._sim_arm_kp
        kd[ARM_INDEX_ARRAY] = self._sim_arm_kd

    def _apply_sim_neck_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        if self._sim_neck_kp is None or self._sim_neck_kd is None:
            return
        kp[NECK_INDEX_ARRAY] = self._sim_neck_kp
        kd[NECK_INDEX_ARRAY] = self._sim_neck_kd

    def _current_pelvis_pose(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.data is not None
        if self._pelvis_site_id < 0:
            raise RuntimeError("pelvis site has not been initialized")
        pos = np.asarray(self.data.site_xpos[self._pelvis_site_id], dtype=np.float64).copy()
        quat = _mat_to_quat_wxyz(np.asarray(self.data.site_xmat[self._pelvis_site_id], dtype=np.float64).reshape(3, 3))
        return pos, quat

    def _initialize_sim(self) -> None:
        xml_path = self._xml_path()
        if not xml_path.is_file():
            raise FileNotFoundError(f"Simulator XML not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = SIM_TIMESTEP
        self.data = mujoco.MjData(self.model)

        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self._initialize_experiment_scene()

        self._joint_bindings = self._resolve_joint_bindings(self.model)
        self._hand_bindings = self._resolve_hand_bindings(self.model)
        self._hand_collision_geom_to_actuator_index = (
            self._resolve_hand_collision_geom_to_actuator_index(self.model)
        )
        for camera_name in (STEREO_LEFT_CAMERA_NAME, STEREO_RIGHT_CAMERA_NAME):
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id < 0:
                raise ValueError(f"MuJoCo XML is missing required stereo camera: {camera_name}")
        self._gyro_slice = self._resolve_sensor_slice(IMU_GYRO_SENSOR_NAME, 3)
        self._accel_slice = self._resolve_sensor_slice(IMU_ACCEL_SENSOR_NAME, 3)
        self._quat_slice = self._resolve_sensor_slice(IMU_QUAT_SENSOR_NAME, 4)
        self._pelvis_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, PELVIS_SITE_NAME)
        if self._pelvis_site_id < 0:
            raise ValueError(f"MuJoCo XML is missing required site: {PELVIS_SITE_NAME}")
        self._sole_site_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, LEFT_SOLE_SITE_NAME),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_SOLE_SITE_NAME),
            ],
            dtype=np.int32,
        )
        if np.any(self._sole_site_ids < 0):
            raise ValueError(
                f"MuJoCo XML is missing required walking sole site(s): {LEFT_SOLE_SITE_NAME}, {RIGHT_SOLE_SITE_NAME}"
            )
        self._floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
        self._foot_collision_geom_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_FOOT_COLLISION_GEOM_NAME),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FOOT_COLLISION_GEOM_NAME),
            ],
            dtype=np.int32,
        )
        if self._floor_geom_id < 0 or np.any(self._foot_collision_geom_ids < 0):
            raise ValueError(
                "MuJoCo XML is missing required walking floor/foot collision geoms: "
                f"{FLOOR_GEOM_NAME}, {LEFT_FOOT_COLLISION_GEOM_NAME}, {RIGHT_FOOT_COLLISION_GEOM_NAME}"
            )

        floating_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, FLOATING_BASE_JOINT)
        if floating_joint_id < 0:
            raise ValueError(f"MuJoCo XML is missing required free joint: {FLOATING_BASE_JOINT}")
        self._free_qpos_slice = slice(
            int(self.model.jnt_qposadr[floating_joint_id]),
            int(self.model.jnt_qposadr[floating_joint_id]) + 7,
        )
        self._free_qvel_slice = slice(
            int(self.model.jnt_dofadr[floating_joint_id]),
            int(self.model.jnt_dofadr[floating_joint_id]) + 6,
        )

        grounded_pelvis_pos, grounded_pelvis_quat = self._current_pelvis_pose()
        self._grounded_pelvis_pos = grounded_pelvis_pos
        self._grounded_pelvis_quat = grounded_pelvis_quat
        self._update_grounded_pelvis_target()
        self._suspended_pelvis_pos = self._grounded_pelvis_pos.copy()
        self._suspended_pelvis_pos[2] += SIM_SUSPENDED_Z_OFFSET
        self._suspended_pelvis_quat = self._grounded_pelvis_quat.copy()

        if self._should_lock_base():
            self._lock_free_base()
            mujoco.mj_forward(self.model, self.data)

        current_q = self._read_joint_q()
        hold_kp, hold_kd = self._load_hold_gains()
        self._sim_waist_kp, self._sim_waist_kd = self._load_sim_waist_gains()
        self._sim_arm_kp, self._sim_arm_kd = self._load_sim_arm_gains()
        self._sim_neck_kp, self._sim_neck_kd = self._load_sim_neck_gains()
        self._apply_sim_waist_gains(hold_kp, hold_kd)
        self._apply_sim_arm_gains(hold_kp, hold_kd)
        self._apply_sim_neck_gains(hold_kp, hold_kd)
        with self._command_lock:
            self._target_q = current_q.copy()
            self._target_dq.fill(0.0)
            self._target_tau.fill(0.0)
            self._kp = hold_kp
            self._kd = hold_kd
        self._walking_restore_joint_q = current_q.copy()
        self._walking_restore_free_qpos = np.asarray(self.data.qpos[self._free_qpos_slice], dtype=np.float64).copy()
        self._walking_restore_free_qvel = np.asarray(self.data.qvel[self._free_qvel_slice], dtype=np.float64).copy()
        self._walking_restore_grounded_pelvis_pos = self._grounded_pelvis_pos.copy()
        self._walking_restore_grounded_pelvis_quat = self._grounded_pelvis_quat.copy()
        self._walking_locked_waist_q = current_q[WAIST_INDEX_ARRAY].copy()
        with self._hand_command_lock:
            self._hand_target_normalized = self._hand_q_to_normalized(self._read_hand_q())
            self._walking_restore_hand_normalized = self._hand_target_normalized.copy()
            self._hand_init_started_at = None
        self._walking_policy_enabled_prev = self._walking_policy_enabled()
        self._sync_experiment_scene_command(force=True)

        logger.info(
            "[%s] loaded XML=%s control_hz=%.1f physics_hz=%.1f substeps=%d render_hz=%.1f mode=%s base_locked=%s hand_joints=%d stereo_cameras=%d sim_waist_override=%s sim_arm_override=%s sim_neck_override=%s",
            self.ctx.name,
            xml_path,
            SIM_CONTROL_HZ,
            SIM_PHYSICS_HZ,
            SIM_SUBSTEPS_PER_CONTROL,
            SIM_RENDER_HZ,
            self.ctx.run_config.mode,
            self._should_lock_base(),
            len(self._hand_bindings),
            self.model.ncam,
            self._sim_waist_kp is not None,
            self._sim_arm_kp is not None,
            self._sim_neck_kp is not None,
        )

    def _initialize_dds(self) -> None:
        self.channel_factory.Init(int(SIM_DOMAIN_ID))

        self.lowstate_pub = igc_sdk.LowStatePublisher(TOPIC_LOWSTATE)
        if not self.lowstate_pub.init():
            raise RuntimeError(f"Failed to init LowStatePublisher({TOPIC_LOWSTATE})")

        self.lowcmd_sub = igc_sdk.LowCmdSubscriber(TOPIC_LOWCMD)
        if not self.lowcmd_sub.init(self._on_lowcmd):
            raise RuntimeError(f"Failed to init LowCmdSubscriber({TOPIC_LOWCMD})")

        self.handstate_pub = igc_sdk.HandStatePublisher(TOPIC_HANDSTATE)
        if not self.handstate_pub.init():
            raise RuntimeError(f"Failed to init HandStatePublisher({TOPIC_HANDSTATE})")

        self.handcmd_sub = igc_sdk.HandCmdSubscriber(TOPIC_HANDCMD)
        if not self.handcmd_sub.init(self._on_handcmd):
            raise RuntimeError(f"Failed to init HandCmdSubscriber({TOPIC_HANDCMD})")

        self.bms_resp_pub = igc_sdk.ServiceResponsePublisher(TOPIC_BMS_RESPONSE)
        self.torque_resp_pub = igc_sdk.ServiceResponsePublisher(TOPIC_TORQUE_RESPONSE)
        self.control_mode_resp_pub = igc_sdk.ServiceResponsePublisher(TOPIC_CONTROL_MODE_RESPONSE)
        for topic, pub in (
            (TOPIC_BMS_RESPONSE, self.bms_resp_pub),
            (TOPIC_TORQUE_RESPONSE, self.torque_resp_pub),
            (TOPIC_CONTROL_MODE_RESPONSE, self.control_mode_resp_pub),
        ):
            if pub is None or not pub.init():
                raise RuntimeError(f"Failed to init ServiceResponsePublisher({topic})")

        self.bms_req_sub = igc_sdk.BmsInitCmdSubscriber(TOPIC_BMS_REQUEST)
        self.torque_req_sub = igc_sdk.TorqueCmdSubscriber(TOPIC_TORQUE_REQUEST)
        self.control_mode_req_sub = igc_sdk.ControlModeCmdSubscriber(TOPIC_CONTROL_MODE_REQUEST)
        if not self.bms_req_sub.init(self._on_bms_request):
            raise RuntimeError(f"Failed to init BmsInitCmdSubscriber({TOPIC_BMS_REQUEST})")
        if not self.torque_req_sub.init(self._on_torque_request):
            raise RuntimeError(f"Failed to init TorqueCmdSubscriber({TOPIC_TORQUE_REQUEST})")
        if not self.control_mode_req_sub.init(self._on_control_mode_request):
            raise RuntimeError(
                f"Failed to init ControlModeCmdSubscriber({TOPIC_CONTROL_MODE_REQUEST})"
            )

    def _on_lowcmd(self, msg: igc_sdk.LowCmd) -> None:
        motors = msg.motors()
        with self._command_lock:
            self._mode = msg.kinematic_mode()
            for idx in range(NUM_MOTORS):
                motor = motors[idx]
                self._target_q[idx] = float(motor.q())
                self._target_dq[idx] = float(motor.dq())
                self._target_tau[idx] = float(motor.tau())
                self._kp[idx] = float(motor.kp())
                self._kd[idx] = float(motor.kd())
            self._apply_sim_waist_gains(self._kp, self._kd)
            self._apply_sim_arm_gains(self._kp, self._kd)
            self._apply_sim_neck_gains(self._kp, self._kd)

    def _on_handcmd(self, msg: igc_sdk.HandCmd) -> None:
        motor_cmd = list(msg.motor_cmd())
        if len(motor_cmd) == 1 and int(motor_cmd[0].id()) == HAND_INIT_TRIGGER_ID:
            with self._hand_command_lock:
                self._hand_init_started_at = time.perf_counter()
            logger.info("[%s] hand init trigger received", self.ctx.name)
            return

        with self._hand_command_lock:
            self._hand_init_started_at = None
            for cmd in motor_cmd:
                motor_id = int(cmd.id())
                idx = HAND_MOTOR_ID_TO_INDEX.get(motor_id)
                if idx is None:
                    logger.debug("[%s] ignoring unknown hand motor id=%s", self.ctx.name, motor_id)
                    continue
                self._hand_target_normalized[idx] = float(_clamp01(cmd.q()))

    def _publish_service_response(
        self,
        publisher: igc_sdk.ServiceResponsePublisher | None,
        request_id: str,
        message: str,
    ) -> None:
        if publisher is None:
            return
        response = igc_sdk.ServiceResponse()
        response.request_id(str(request_id))
        response.success(True)
        response.message(str(message))
        response.error_code(0)
        publisher.write(response)

    def _on_bms_request(self, cmd: igc_sdk.BmsInitCmd) -> None:
        with self._service_lock:
            self._bms_init_type = cmd.init()
        self._publish_service_response(self.bms_resp_pub, cmd.request_id(), "Motor Initialization succeeded")

    def _on_torque_request(self, cmd: igc_sdk.TorqueCmd) -> None:
        with self._service_lock:
            self._torque_enabled = cmd.torque() != igc_sdk.TorqueType.TORQUE_OFF
        label = "Torque Control succeeded" if self._torque_enabled else "Torque Off succeeded"
        self._publish_service_response(self.torque_resp_pub, cmd.request_id(), label)

    def _on_control_mode_request(self, cmd: igc_sdk.ControlModeCmd) -> None:
        with self._service_lock:
            self._control_mode = cmd.mode()
        mode_name = "LOW_LEVEL" if self._control_mode == igc_sdk.ControlMode.CONTROL_MODE_LOW_LEVEL else "HIGH_LEVEL"
        self._publish_service_response(
            self.control_mode_resp_pub,
            cmd.request_id(),
            f"Switched to {mode_name} control mode",
        )

    def _walking_policy_enabled(self) -> bool:
        if self.walking_cmd_shm is None:
            return False
        try:
            data = self.walking_cmd_shm.read_data()
            return bool(float(np.asarray(data.get("policy_enabled", 0.0)).reshape(()).item()) > 0.5)
        except Exception:
            logger.debug("[walking_sim] failed to read walking_cmd_shm", exc_info=True)
            return False

    def _walking_startup_status(self) -> tuple[float, float]:
        if self.walking_debug_shm is None:
            return 0.0, 0.0
        try:
            data = self.walking_debug_shm.read_data()
            phase_code = float(np.asarray(data.get("startup_phase_code", 0.0)).reshape(()).item())
            blend_alpha = float(np.asarray(data.get("blend_alpha", 0.0)).reshape(()).item())
            return phase_code, blend_alpha
        except Exception:
            logger.debug("[walking_sim] failed to read walking_debug_shm", exc_info=True)
            return 0.0, 0.0

    def _walking_base_unlock_ready(self) -> bool:
        if not self._walking_policy_enabled():
            return False
        phase_code, blend_alpha = self._walking_startup_status()
        if phase_code >= WALKING_STARTUP_PHASE_RUNNING:
            return True
        if phase_code >= WALKING_STARTUP_PHASE_BLENDING and blend_alpha > 0.0:
            return True
        return False

    def _should_lock_base(self) -> bool:
        mode = self.ctx.run_config.mode
        if mode == WALKING_MODE_NAME:
            return not self._walking_base_unlock_ready()
        return mode in (None, "teleop", "inference", "replay")

    def _read_joint_q(self) -> np.ndarray:
        assert self.data is not None
        return np.asarray(self.data.qpos[self._joint_qpos_adrs], dtype=np.float32).copy()

    def _read_joint_dq(self) -> np.ndarray:
        assert self.data is not None
        return np.asarray(self.data.qvel[self._joint_dof_adrs], dtype=np.float32).copy()

    def _read_joint_tau(self) -> np.ndarray:
        assert self.data is not None
        actuator_force = np.asarray(self.data.actuator_force, dtype=np.float32)
        return actuator_force[self._actuator_ids].copy()

    def _read_hand_q(self) -> np.ndarray:
        assert self.data is not None
        return np.asarray(self.data.qpos[self._hand_qpos_adrs], dtype=np.float32).copy()

    def _read_hand_dq(self) -> np.ndarray:
        assert self.data is not None
        return np.asarray(self.data.qvel[self._hand_dof_adrs], dtype=np.float32).copy()

    def _read_hand_tau(self) -> np.ndarray:
        assert self.data is not None
        actuator_force = np.asarray(self.data.actuator_force, dtype=np.float32)
        return actuator_force[self._hand_actuator_ids].copy()

    def _hand_q_to_normalized(self, q: np.ndarray) -> np.ndarray:
        q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
        denom = self._hand_closed_q - self._hand_open_q
        normalized = np.zeros(HAND_TARGET_LENGTH, dtype=np.float32)
        safe = np.abs(denom) > 1e-6
        normalized[safe] = (q_arr[safe] - self._hand_open_q[safe]) / denom[safe]
        return np.clip(normalized, 0.0, 1.0)

    def _hand_normalized_to_q(self, normalized: np.ndarray) -> np.ndarray:
        normalized_arr = np.clip(np.asarray(normalized, dtype=np.float32).reshape(-1), 0.0, 1.0)
        return self._hand_open_q + normalized_arr * (self._hand_closed_q - self._hand_open_q)

    def _read_imu(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.data is not None
        sensordata = np.asarray(self.data.sensordata, dtype=np.float32)
        quat = sensordata[self._quat_slice].copy()
        gyro = sensordata[self._gyro_slice].copy()
        accel = sensordata[self._accel_slice].copy()
        rpy = _quat_wxyz_to_rpy(quat)
        return quat, gyro, accel, rpy

    def _lock_free_base(self) -> None:
        assert self.data is not None
        if self.ctx.run_config.mode in (WALKING_MODE_NAME, "teleop"):
            target_pelvis_pos = self._grounded_pelvis_pos
            target_pelvis_quat = self._grounded_pelvis_quat
        else:
            target_pelvis_pos = self._suspended_pelvis_pos
            target_pelvis_quat = self._suspended_pelvis_quat

        base_qpos = np.asarray(self.data.qpos[self._free_qpos_slice], dtype=np.float64).copy()
        base_pos = base_qpos[:3]
        base_quat = base_qpos[3:7]
        base_rot = _quat_wxyz_to_mat(base_quat)

        pelvis_pos, pelvis_quat = self._current_pelvis_pose()
        pelvis_rot = _quat_wxyz_to_mat(pelvis_quat)

        base_to_pelvis_rot = base_rot.T @ pelvis_rot
        base_to_pelvis_pos = base_rot.T @ (pelvis_pos - base_pos)

        target_pelvis_rot = _quat_wxyz_to_mat(target_pelvis_quat)
        target_base_rot = target_pelvis_rot @ base_to_pelvis_rot.T
        target_base_pos = target_pelvis_pos - target_base_rot @ base_to_pelvis_pos

        self.data.qpos[self._free_qpos_slice.start : self._free_qpos_slice.start + 3] = target_base_pos
        self.data.qpos[self._free_qpos_slice.start + 3 : self._free_qpos_slice.stop] = _mat_to_quat_wxyz(target_base_rot)
        self.data.qvel[self._free_qvel_slice] = 0.0

    def _current_min_sole_height(self) -> float:
        assert self.data is not None
        return float(np.min(np.asarray(self.data.site_xpos[self._sole_site_ids, 2], dtype=np.float64)))

    def _current_min_foot_contact_dist(self) -> float | None:
        assert self.data is not None
        min_dist: float | None = None
        floor_geom_id = int(self._floor_geom_id)
        foot_geom_ids = {int(idx) for idx in np.asarray(self._foot_collision_geom_ids, dtype=np.int32).reshape(-1)}
        for contact_idx in range(int(self.data.ncon)):
            contact = self.data.contact[contact_idx]
            geom_pair = {int(contact.geom1), int(contact.geom2)}
            if floor_geom_id not in geom_pair:
                continue
            if not geom_pair.intersection(foot_geom_ids):
                continue
            dist = float(contact.dist)
            if min_dist is None or dist < min_dist:
                min_dist = dist
        return min_dist

    def _update_grounded_pelvis_target(self) -> None:
        pelvis_pos, pelvis_quat = self._current_pelvis_pose()
        min_sole_height = self._current_min_sole_height()
        min_contact_dist = self._current_min_foot_contact_dist()
        self._grounded_pelvis_pos[:2] = pelvis_pos[:2]
        if min_contact_dist is not None and min_contact_dist < 0.0:
            target_z = float(pelvis_pos[2] - min_contact_dist)
        else:
            target_z = float(pelvis_pos[2] - (min_sole_height - WALKING_SOLE_TARGET_HEIGHT))
        self._grounded_pelvis_pos[2] = target_z
        self._grounded_pelvis_quat = pelvis_quat.copy()

    def _update_walking_ground_target(self) -> None:
        if self.ctx.run_config.mode != WALKING_MODE_NAME:
            return
        self._update_grounded_pelvis_target()

    def _capture_walking_restore_state(self) -> None:
        self._walking_restore_joint_q = self._read_joint_q()
        assert self.data is not None
        self._walking_restore_free_qpos = np.asarray(self.data.qpos[self._free_qpos_slice], dtype=np.float64).copy()
        self._walking_restore_free_qvel = np.asarray(self.data.qvel[self._free_qvel_slice], dtype=np.float64).copy()
        self._walking_restore_grounded_pelvis_pos = self._grounded_pelvis_pos.copy()
        self._walking_restore_grounded_pelvis_quat = self._grounded_pelvis_quat.copy()
        self._walking_locked_waist_q = self._walking_restore_joint_q[WAIST_INDEX_ARRAY].copy()
        self._walking_restore_hand_normalized = self._hand_q_to_normalized(self._read_hand_q())

    def _restore_walking_pose(self) -> None:
        assert self.data is not None
        assert self.model is not None
        self.data.qpos[self._free_qpos_slice] = self._walking_restore_free_qpos.astype(np.float64, copy=False)
        self.data.qvel[self._free_qvel_slice] = self._walking_restore_free_qvel.astype(np.float64, copy=False)
        self.data.qpos[self._joint_qpos_adrs] = self._walking_restore_joint_q.astype(np.float64, copy=False)
        self.data.qvel[self._joint_dof_adrs] = 0.0
        self.data.qpos[self._hand_qpos_adrs] = self._hand_normalized_to_q(self._walking_restore_hand_normalized).astype(
            np.float64,
            copy=False,
        )
        self.data.qvel[self._hand_dof_adrs] = 0.0
        self._grounded_pelvis_pos = self._walking_restore_grounded_pelvis_pos.copy()
        self._grounded_pelvis_quat = self._walking_restore_grounded_pelvis_quat.copy()
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._lock_free_base()
        mujoco.mj_forward(self.model, self.data)

        current_q = self._read_joint_q()
        current_hand_normalized = self._hand_q_to_normalized(self._read_hand_q())
        with self._command_lock:
            self._target_q = current_q.copy()
            self._target_dq.fill(0.0)
            self._target_tau.fill(0.0)
        with self._hand_command_lock:
            self._hand_target_normalized = current_hand_normalized.copy()
            self._hand_init_started_at = None

    def _sync_walking_policy_transition(self) -> bool:
        if self.ctx.run_config.mode != WALKING_MODE_NAME:
            return False

        policy_enabled = self._walking_policy_enabled()
        if policy_enabled and not self._walking_policy_enabled_prev:
            self._capture_walking_restore_state()
        elif not policy_enabled and self._walking_policy_enabled_prev:
            self._restore_walking_pose()
            self._walking_policy_enabled_prev = policy_enabled
            return True
        self._walking_policy_enabled_prev = policy_enabled
        return False

    def _is_teleop_run(self) -> bool:
        if self.ctx.run_config.mode != "teleop":
            return False
        if self.mode_shm is None:
            return True
        try:
            data = self.mode_shm.read_data()
            return bool(data.get("teleop", False)) and bool(data.get("run", False))
        except Exception:
            return True

    def _read_action_target_from_shm(self) -> tuple[np.ndarray, np.ndarray | None] | None:
        if self.act_shm is None:
            return None
        try:
            act = self.act_shm.read_data()
            target_q = self._read_joint_q()
            if self._teleop_fixed_leg_q is None:
                self._teleop_fixed_leg_q = np.asarray(
                    target_q[list(LEG_INDICES)], dtype=np.float32
                ).copy()
                logger.info("[%s] latched fixed teleop leg target", self.ctx.name)
            target_q[list(WAIST_INDICES)] = np.asarray(act["act_waist"], dtype=np.float32).reshape(-1)
            target_q[list(LEG_INDICES)] = self._teleop_fixed_leg_q
            target_q[list(ARM_INDICES)] = np.asarray(act["act_arm"], dtype=np.float32).reshape(-1)
            target_q[list(NECK_INDICES)] = np.asarray(act["act_neck"], dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(target_q)):
                return None
            hand = np.asarray(act.get("act_hand"), dtype=np.float32).reshape(-1)
            if hand.size != HAND_TARGET_LENGTH or not np.all(np.isfinite(hand)):
                hand_target = None
            else:
                hand_target = np.clip(hand, 0.0, 1.0)
            return target_q, hand_target
        except Exception:
            return None

    def _apply_direct_teleop_action_target(self) -> None:
        if not self._direct_teleop_action_enabled:
            return
        if not self._is_teleop_run():
            self._teleop_fixed_leg_q = None
            return
        target = self._read_action_target_from_shm()
        if target is None:
            return
        target_q, hand_target = target
        with self._command_lock:
            self._mode = igc_sdk.KinematicMode.PJS
            self._target_q = np.clip(target_q, self._joint_lower, self._joint_upper)
            self._target_dq.fill(0.0)
            self._target_tau.fill(0.0)
        if hand_target is not None:
            with self._hand_command_lock:
                self._hand_target_normalized = hand_target.copy()
                self._hand_init_started_at = None
        now = time.perf_counter()
        if self._direct_teleop_last_log_t is None or now - self._direct_teleop_last_log_t >= 2.0:
            self._direct_teleop_last_log_t = now
            logger.info("[%s] direct teleop target applied from act_shm", self.ctx.name)

    def _decode_targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self._command_lock:
            mode = self._mode
            target_q = self._target_q.copy()
            target_dq = self._target_dq.copy()
            target_tau = self._target_tau.copy()
            kp = self._kp.copy()
            kd = self._kd.copy()

        if mode == igc_sdk.KinematicMode.MS:
            target_q, _unconfigured = map_ms_to_pjs(target_q, self._pr2ab_pairs)
            target_dq = map_ms_linear_to_pjs(target_dq, self._pr2ab_pairs)

        target_q = np.clip(target_q, self._joint_lower, self._joint_upper)
        if self.ctx.run_config.mode == WALKING_MODE_NAME:
            target_q[WAIST_INDEX_ARRAY] = self._walking_locked_waist_q
            target_dq[WAIST_INDEX_ARRAY] = 0.0
            target_tau[WAIST_INDEX_ARRAY] = 0.0
            kp[WAIST_INDEX_ARRAY] = WALKING_WAIST_LOCK_KP
            kd[WAIST_INDEX_ARRAY] = WALKING_WAIST_LOCK_KD
        return target_q, target_dq, target_tau, kp, kd

    def _decode_hand_targets(self) -> np.ndarray:
        with self._hand_command_lock:
            normalized = self._hand_target_normalized.copy()
            init_started_at = self._hand_init_started_at

        if init_started_at is not None:
            elapsed = time.perf_counter() - float(init_started_at)
            half = HAND_INIT_SWEEP_DURATION_S * 0.5
            if elapsed < half:
                value = HAND_INIT_SWEEP_TARGET * float(elapsed / max(half, 1e-6))
                normalized = np.full(HAND_TARGET_LENGTH, value, dtype=np.float32)
            elif elapsed < HAND_INIT_SWEEP_DURATION_S:
                alpha = float((elapsed - half) / max(half, 1e-6))
                value = HAND_INIT_SWEEP_TARGET * (1.0 - alpha)
                normalized = np.full(HAND_TARGET_LENGTH, value, dtype=np.float32)
            else:
                normalized = np.zeros(HAND_TARGET_LENGTH, dtype=np.float32)
                with self._hand_command_lock:
                    if self._hand_init_started_at == init_started_at:
                        self._hand_init_started_at = None
                        self._hand_target_normalized.fill(0.0)

        return self._hand_normalized_to_q(normalized)

    def _compute_ctrl(self) -> np.ndarray:
        assert self.data is not None

        with self._service_lock:
            torque_enabled = bool(self._torque_enabled)
            low_level = self._control_mode == igc_sdk.ControlMode.CONTROL_MODE_LOW_LEVEL

        if not torque_enabled or not low_level:
            return np.zeros(NUM_MOTORS, dtype=np.float32)

        target_q, target_dq, target_tau, kp, kd = self._decode_targets()
        current_q = self._read_joint_q()
        current_dq = self._read_joint_dq()
        ctrl = target_tau + kp * (target_q - current_q) + kd * (target_dq - current_dq)
        ctrl = self._apply_upper_body_gravity_compensation(ctrl)
        return np.clip(ctrl, -self._ctrl_limits, self._ctrl_limits)

    def _apply_upper_body_gravity_compensation(self, ctrl: np.ndarray) -> np.ndarray:
        compensated = np.asarray(ctrl, dtype=np.float32).copy()
        if self.data is None or self._joint_dof_adrs.size != NUM_MOTORS:
            return compensated

        upper_body_dof_adrs = self._joint_dof_adrs[UPPER_BODY_GRAVITY_COMPENSATION_INDEX_ARRAY]
        compensated[UPPER_BODY_GRAVITY_COMPENSATION_INDEX_ARRAY] += np.asarray(
            self.data.qfrc_bias[upper_body_dof_adrs],
            dtype=np.float32,
        )
        return compensated

    def _apply_neck_gravity_compensation(self, ctrl: np.ndarray) -> np.ndarray:
        compensated = np.asarray(ctrl, dtype=np.float32).copy()
        if self.data is None or self._joint_dof_adrs.size != NUM_MOTORS:
            return compensated
        neck_dof_adrs = self._joint_dof_adrs[NECK_INDEX_ARRAY]
        compensated[NECK_INDEX_ARRAY] += np.asarray(
            self.data.qfrc_bias[neck_dof_adrs],
            dtype=np.float32,
        )
        return compensated

    def _hand_actuators_near_grasp_object(self) -> np.ndarray:
        near = np.zeros(HAND_TARGET_LENGTH, dtype=bool)
        if self.model is None or self.data is None:
            return near

        for contact_id in range(int(self.data.ncon)):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if int(self.model.geom_contype[geom1]) & EXPERIMENT_GRASP_COLLISION_CONTYPE:
                hand_geom_id = geom2
            elif int(self.model.geom_contype[geom2]) & EXPERIMENT_GRASP_COLLISION_CONTYPE:
                hand_geom_id = geom1
            else:
                continue

            actuator_index = self._hand_collision_geom_to_actuator_index.get(
                hand_geom_id
            )
            if actuator_index is not None:
                near[actuator_index] = True
        return near

    def _limit_near_object_hand_closing_torque(
        self,
        ctrl: np.ndarray,
        near_actuators: np.ndarray,
        position_error: np.ndarray,
    ) -> np.ndarray:
        limited = np.asarray(ctrl, dtype=np.float32).copy()
        near = np.asarray(near_actuators, dtype=bool).reshape(HAND_TARGET_LENGTH)
        closing_error = (
            np.asarray(position_error, dtype=np.float32).reshape(HAND_TARGET_LENGTH)
            * self._hand_closing_direction
        )
        closing_contacts = near & (closing_error > 0.0)
        closing_torque = limited * self._hand_closing_direction
        closing_torque[closing_contacts] = np.clip(
            closing_torque[closing_contacts],
            -HAND_NEAR_OBJECT_OPENING_BRAKE_LIMIT,
            HAND_NEAR_OBJECT_CLOSING_TORQUE_LIMIT,
        )
        limited = closing_torque * self._hand_closing_direction
        return limited

    def _compute_hand_ctrl(self) -> np.ndarray:
        target_q = self._decode_hand_targets()
        current_q = self._read_hand_q()
        current_dq = self._read_hand_dq()
        position_error = target_q - current_q
        ctrl = HAND_SERVO_KP * position_error - HAND_SERVO_KD * current_dq
        ctrl = self._limit_near_object_hand_closing_torque(
            ctrl,
            self._hand_actuators_near_grasp_object(),
            position_error,
        )
        return np.clip(ctrl, -self._hand_ctrl_limits, self._hand_ctrl_limits)

    def _step_physics(self) -> None:
        assert self.data is not None
        assert self.model is not None

        if self._sync_walking_policy_transition():
            return

        if self._should_lock_base():
            if self.ctx.run_config.mode == WALKING_MODE_NAME:
                mujoco.mj_forward(self.model, self.data)
                self._update_walking_ground_target()
            self._lock_free_base()
            mujoco.mj_forward(self.model, self.data)

        self._apply_direct_teleop_action_target()
        ctrl = self._compute_ctrl()
        hand_ctrl = self._compute_hand_ctrl()
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self._actuator_ids] = ctrl
        self.data.ctrl[self._hand_actuator_ids] = hand_ctrl
        mujoco.mj_step(self.model, self.data)

        if self._should_lock_base():
            if self.ctx.run_config.mode == WALKING_MODE_NAME:
                mujoco.mj_forward(self.model, self.data)
                self._update_walking_ground_target()
            self._lock_free_base()
            mujoco.mj_forward(self.model, self.data)

    def _publish_lowstate(self) -> None:
        if self.lowstate_pub is None:
            return

        q_joint = self._read_joint_q()
        dq_joint = self._read_joint_dq()
        tau_joint = self._read_joint_tau()
        quat, gyro, accel, rpy = self._read_imu()
        q_motor, _unconfigured = map_pjs_to_ms(q_joint, self._pr2ab_pairs)
        dq_motor = map_pjs_linear_to_ms(dq_joint, self._pr2ab_pairs)

        msg = igc_sdk.LowState()
        joint_state = msg.joint_state()
        motor_state = msg.motor_state()
        imu_state = msg.imu_state()
        for idx in range(NUM_MOTORS):
            joint_state[idx].q(float(q_joint[idx]))
            joint_state[idx].dq(float(dq_joint[idx]))
            joint_state[idx].tau_est(float(tau_joint[idx]))
            joint_state[idx].status_bits(0)

            motor_state[idx].q(float(q_motor[idx]))
            motor_state[idx].dq(float(dq_motor[idx]))
            motor_state[idx].tau_est(float(tau_joint[idx]))
            motor_state[idx].status_bits(0)
            motor_state[idx].temperature(25)

        imu_state.quaternion(quat.tolist())
        imu_state.gyroscope(gyro.tolist())
        imu_state.accelerometer(accel.tolist())
        imu_state.rpy(rpy.tolist())
        msg.tick(int(self._tick))
        self.lowstate_pub.write(msg)
        self._tick += 1

    def _publish_handstate(self) -> None:
        q_hand = self._hand_q_to_normalized(self._read_hand_q())
        obs_shm = getattr(self, "obs_shm", None)
        if obs_shm is not None and self._tick % 3 == 0:
            try:
                obs_shm.write_data(obs_hand=np.asarray(q_hand, dtype=np.float64))
            except Exception:
                logger.debug("[%s] failed to write sim obs_hand", self.ctx.name, exc_info=True)

        if self.handstate_pub is None:
            return

        dq_raw = self._read_hand_dq()
        dq_hand = np.zeros(HAND_TARGET_LENGTH, dtype=np.float32)
        dq_scale = self._hand_closed_q - self._hand_open_q
        safe = np.abs(dq_scale) > 1e-6
        dq_hand[safe] = dq_raw[safe] / dq_scale[safe]
        tau_hand = self._read_hand_tau()
        motor_state: list[igc_sdk.MotorState] = []
        for idx in range(HAND_TARGET_LENGTH):
            state = igc_sdk.MotorState()
            state.q(float(q_hand[idx]))
            state.dq(float(dq_hand[idx]))
            state.tau_est(float(tau_hand[idx]))
            state.status_bits(0)
            state.temperature(25)
            motor_state.append(state)

        msg = igc_sdk.HandState()
        msg.motor_state(motor_state)
        self.handstate_pub.write(msg)

    def _cleanup(self) -> None:
        self._stop_camera_renderer()

        for sub in (
            self.lowcmd_sub,
            self.handcmd_sub,
            self.bms_req_sub,
            self.torque_req_sub,
            self.control_mode_req_sub,
        ):
            if sub is None:
                continue
            try:
                sub.stop()
            except Exception:
                logger.debug("[%s] failed to stop subscriber", self.ctx.name, exc_info=True)

        for pub in (
            self.lowstate_pub,
            self.handstate_pub,
            self.bms_resp_pub,
            self.torque_resp_pub,
            self.control_mode_resp_pub,
        ):
            if pub is None:
                continue
            try:
                pub.stop()
            except Exception:
                logger.debug("[%s] failed to stop publisher", self.ctx.name, exc_info=True)

        try:
            self.channel_factory.Release()
        except Exception:
            logger.debug("[%s] failed to release ChannelFactory", self.ctx.name, exc_info=True)

        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                logger.debug("[%s] failed to close MuJoCo viewer", self.ctx.name, exc_info=True)

        for key, mgr in self._shared_memory.items():
            try:
                mgr.worker_close()
            except Exception:
                logger.debug("[%s] failed to close shared memory %s", self.ctx.name, key, exc_info=True)

    def run(self) -> None:
        self._initialize_sim()
        self._initialize_dds()
        assert self.model is not None
        assert self.data is not None

        control_rate = Rate(SIM_CONTROL_HZ)
        next_render_at = time.perf_counter()

        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self.viewer = viewer
                self._start_camera_renderer()
                while not self.should_stop():
                    if not viewer.is_running():
                        logger.info("[%s] viewer closed; stopping simulator worker", self.ctx.name)
                        break

                    with viewer.lock():
                        self._sync_experiment_scene_command()
                        for _ in range(SIM_SUBSTEPS_PER_CONTROL):
                            self._step_physics()
                        self._publish_lowstate()
                        self._publish_handstate()
                        self._capture_camera_state()

                    now = time.perf_counter()
                    if now >= next_render_at:
                        viewer.sync()
                        next_render_at += SIM_RENDER_DT
                        if now > next_render_at + SIM_RENDER_DT:
                            next_render_at = now + SIM_RENDER_DT
                    control_rate.sleep()
        finally:
            self._cleanup()
            logger.info("[%s] simulator worker stopped", self.ctx.name)
