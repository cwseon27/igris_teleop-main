from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, List, Literal, Tuple

import numpy as np
import pinocchio as pin
import yaml

from igris_teleop.core.project_paths import JOINT_SETTING_PATH

WAIST_CONTROLLER_TO_MODEL_ORDER = np.asarray((2, 1, 0), dtype=np.intp)

try:
    from proxsuite import proxqp
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    proxqp = None

def describe_ik_dependency_context() -> dict[str, str]:
    return {
        "python": sys.executable,
        "pinocchio": getattr(pin, "__file__", ""),
        "pinocchio_version": str(getattr(pin, "__version__", "")),
        "proxsuite": getattr(importlib.import_module("proxsuite"), "__file__", "")
        if proxqp is not None
        else "",
        "numpy": getattr(np, "__file__", ""),
        "numpy_version": str(getattr(np, "__version__", "")),
    }


@dataclass(frozen=True)
class IKWeights:
    w_trans: float = 50.0
    w_rot: float = 0.5
    w_reg: float = 0.02
    # Damping must not overwhelm the HMD rotational task.  Command continuity
    # is enforced by the rate/acceleration bounds and the published-command
    # state, rather than by a large zero-motion penalty.
    w_smooth: float = 0.25
    w_jerk: float = 0.0
    w_col: float = 5.0
    w_damping: float = 1e-6


@dataclass(frozen=True)
class CollisionParams:
    enabled: bool = True
    margin: float = 0.02
    activation_distance: float = 0.05
    min_distance_eps: float = 1e-6
    slack_weight: float = 1e5
    slack_max: float = 1.0


@dataclass(frozen=True)
class RateLimitParams:
    dq_max: float = 0.05
    ddq_max: float = 0.03
    dddq_max: float = 0.0


@dataclass(frozen=True)
class SolverParams:
    max_iter: int = 50
    eps_abs: float = 1e-5
    eps_rel: float = 1e-5
    verbose: bool = False
    update_preconditioner: bool = True
    linearization_iters: int = 1


@dataclass(frozen=True)
class IKConfig:
    weights: IKWeights = IKWeights()
    collision: CollisionParams = CollisionParams()
    rate_limit: RateLimitParams = RateLimitParams()
    solver: SolverParams = SolverParams()

    @classmethod
    def from_sources(
        cls,
        *,
        profile: str | None = None,
        config_path: str | Path | None = None,
    ) -> "IKConfig":
        cfg = cls()
        path = config_path or os.environ.get("IGRIS_IK_CONFIG")
        if path:
            cfg = _apply_config_file(cfg, Path(path), profile)
        cfg = _apply_env_overrides(cfg)
        if profile:
            cfg = _apply_env_overrides(cfg, profile)
        return cfg


def _coerce_like(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def _apply_section(cfg: IKConfig, section_name: str, values: Any) -> IKConfig:
    if not isinstance(values, dict):
        return cfg
    current = getattr(cfg, section_name)
    valid = {field.name: field for field in fields(current)}
    updates: dict[str, Any] = {}
    for key, value in values.items():
        if key in valid:
            updates[key] = _coerce_like(value, getattr(current, key))
    if not updates:
        return cfg
    return replace(cfg, **{section_name: replace(current, **updates)})


def _apply_config_dict(cfg: IKConfig, values: Any) -> IKConfig:
    if not isinstance(values, dict):
        return cfg
    for section_name in ("weights", "collision", "rate_limit", "solver"):
        cfg = _apply_section(cfg, section_name, values.get(section_name))
    return cfg


def _apply_config_file(cfg: IKConfig, path: Path, profile: str | None) -> IKConfig:
    if not path.is_file():
        raise FileNotFoundError(f"IK config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _apply_config_dict(cfg, raw.get("default", raw))
    if profile:
        profiles = raw.get("profiles", {})
        if isinstance(profiles, dict):
            cfg = _apply_config_dict(cfg, profiles.get(profile, {}))
    return cfg


def _apply_env_overrides(cfg: IKConfig, profile: str | None = None) -> IKConfig:
    profile_part = ""
    if profile:
        profile_part = "".join(ch if ch.isalnum() else "_" for ch in profile.upper()) + "_"
    for section_name in ("weights", "collision", "rate_limit", "solver"):
        current = getattr(cfg, section_name)
        updates: dict[str, Any] = {}
        for field in fields(current):
            env_name = f"IGRIS_IK_{profile_part}{field.name.upper()}"
            if env_name in os.environ:
                updates[field.name] = _coerce_like(os.environ[env_name], getattr(current, field.name))
        if updates:
            cfg = replace(cfg, **{section_name: replace(current, **updates)})
    return cfg


AttachKind = Literal["frame", "joint"]


class IGRIS_C_UpperIKProxSuite:
    """Pinocchio + ProxSuite differential IK.

    This class keeps the public contract of the legacy CasADi/IPOPT IK
    solver, but solves a per-tick QP over dq instead of a nonlinear program
    over q.
    """

    def __init__(self, cfg: IKConfig | None = None):
        if proxqp is None:
            raise ImportError(
                "proxsuite is required for IGRIS_C_UpperIKProxSuite. "
                "Install it in the IK environment, e.g. `.venv-ik/bin/python -m pip install proxsuite`."
            )

        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.cfg = cfg or IKConfig()

        self._load_robot()
        self._build_reduced_robot()
        self._add_end_effector_frames()
        self._build_collision_model()
        self._init_qp()
        self._init_filters_and_state()
        self._last_solve_info: dict[str, Any] = {}

        print("finish IK initialize (Pinocchio + ProxSuite)")

    @staticmethod
    def resolve_attach_id(model: pin.Model, name: str) -> Tuple[AttachKind, int]:
        if model.existFrame(name):
            return ("frame", model.getFrameId(name))
        jid = model.getJointId(name)
        if jid != 0:
            return ("joint", jid)
        raise ValueError(f"Neither frame nor joint found: {name}")

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _project_rotation(R: np.ndarray) -> np.ndarray:
        U, _, Vt = np.linalg.svd(R)
        R_proj = U @ Vt
        if np.linalg.det(R_proj) < 0.0:
            U[:, -1] *= -1.0
            R_proj = U @ Vt
        return R_proj

    @classmethod
    def _sanitize_pose_matrix(cls, pose: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        try:
            arr = np.asarray(pose, dtype=np.float64).reshape(4, 4).copy()
        except Exception:
            return fallback.copy()
        if not np.all(np.isfinite(arr)):
            return fallback.copy()
        arr[:3, :3] = cls._project_rotation(arr[:3, :3])
        arr[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return arr

    @staticmethod
    def _se3_from_homogeneous(T: np.ndarray) -> pin.SE3:
        return pin.SE3(T[:3, :3].copy(), T[:3, 3].copy())

    def _unique_joint_ids(self, joint_names: List[str]) -> List[int]:
        ids: List[int] = []
        seen = set()
        for name in joint_names:
            jid = self.robot.model.getJointId(name)
            if jid == 0:
                print(f"[IGRIS_C_UpperIKProxSuite] joint '{name}' not found in model. Skipping.")
                continue
            if jid in seen:
                print(f"[IGRIS_C_UpperIKProxSuite] joint '{name}' (id {jid}) duplicated. Skipping.")
                continue
            seen.add(jid)
            ids.append(jid)
        return ids

    def _clip_q_to_limits(self, q: np.ndarray) -> np.ndarray:
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        lower = np.asarray(self.reduced_robot.model.lowerPositionLimit, dtype=np.float64).reshape(-1)
        upper = np.asarray(self.reduced_robot.model.upperPositionLimit, dtype=np.float64).reshape(-1)
        if q_arr.shape != lower.shape:
            raise ValueError(f"q shape {q_arr.shape} does not match limits {lower.shape}")
        return np.clip(q_arr, lower, upper)

    def _load_default_init_data(self) -> np.ndarray:
        with JOINT_SETTING_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        default_q = np.asarray(cfg.get("default_dof_pos", []), dtype=np.float64).reshape(-1)
        if default_q.size < 31:
            raise ValueError(f"default_dof_pos length {default_q.size} < 31 in {JOINT_SETTING_PATH}")

        waist = default_q[:3][WAIST_CONTROLLER_TO_MODEL_ORDER]
        arm = default_q[15:29]
        neck = default_q[29:31]
        init_data = np.concatenate((waist, arm, neck))

        if init_data.size != self.reduced_robot.model.nq:
            raise ValueError(
                f"default init_data length {init_data.size} != reduced nq {self.reduced_robot.model.nq}"
            )
        return init_data

    def _resolve_urdf_paths(self) -> Tuple[Path, Path, Path]:
        base_dir = Path(__file__).resolve().parents[2]
        asset_dir = base_dir / "asset"
        urdf_dir = asset_dir / "urdf"
        urdf_path = urdf_dir / "igris_c_v2_pelvis.urdf"
        return asset_dir, urdf_dir, urdf_path

    def _load_robot(self) -> None:
        asset_dir, urdf_dir, urdf_path = self._resolve_urdf_paths()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), [str(urdf_dir), str(asset_dir)])

    def _build_reduced_robot(self) -> None:
        joints_to_lock_names = [
            "3_Joint_Hip_Pitch_Left",
            "4_Joint_Hip_Roll_Left",
            "5_Joint_Hip_Yaw_Left",
            "6_Joint_Knee_Pitch_Left",
            "7_Joint_Ankle_Pitch_Left",
            "8_Joint_Ankle_Roll_Left",
            "9_Joint_Hip_Pitch_Right",
            "10_Joint_Hip_Roll_Right",
            "11_Joint_Hip_Yaw_Right",
            "12_Joint_Knee_Pitch_Right",
            "13_Joint_Ankle_Pitch_Right",
            "14_Joint_Ankle_Roll_Right",
        ]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self._unique_joint_ids(joints_to_lock_names),
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

    def _add_end_effector_frames(self) -> None:
        model = self.reduced_robot.model

        l_wrist_parent = model.getJointId("21_Joint_Wrist_Pitch_Left")
        r_wrist_parent = model.getJointId("28_Joint_Wrist_Pitch_Right")
        h_wrist_parent = model.getJointId("30_Joint_Neck_Pitch")

        model.addFrame(
            pin.Frame(
                "L_ee",
                l_wrist_parent,
                pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.05])),
                pin.FrameType.OP_FRAME,
            )
        )
        model.addFrame(
            pin.Frame(
                "R_ee",
                r_wrist_parent,
                pin.SE3(np.eye(3), np.array([0.0, 0.0, -0.05])),
                pin.FrameType.OP_FRAME,
            )
        )
        model.addFrame(
            pin.Frame(
                "H_ee",
                h_wrist_parent,
                pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.15])),
                pin.FrameType.OP_FRAME,
            )
        )

        torso_rot = pin.AngleAxis(-np.pi / 2.0, np.array([0.0, 0.0, 1.0])).matrix()
        torso_offset = pin.SE3(torso_rot, np.array([0.05, 0.0, 0.3]))
        self.torso_frame_name = "T_ee"
        if not model.existFrame(self.torso_frame_name):
            if model.existFrame("body_link"):
                body_frame_id = model.getFrameId("body_link")
                body_frame = model.frames[body_frame_id]
                model.addFrame(
                    pin.Frame(
                        self.torso_frame_name,
                        body_frame.parentJoint,
                        body_frame_id,
                        body_frame.placement * torso_offset,
                        pin.FrameType.OP_FRAME,
                    )
                )
            else:
                t_parent = model.getJointId("2_Joint_Waist_Yaw")
                if t_parent == 0:
                    raise ValueError("Cannot resolve torso frame: body_link and 2_Joint_Waist_Yaw not found")
                model.addFrame(
                    pin.Frame(self.torso_frame_name, t_parent, torso_offset, pin.FrameType.OP_FRAME)
                )

        self.reduced_robot.data = model.createData()

        self.L_hand_id = model.getFrameId("L_ee", pin.FrameType.OP_FRAME)
        self.R_hand_id = model.getFrameId("R_ee", pin.FrameType.OP_FRAME)
        self.Head_id = model.getFrameId("H_ee", pin.FrameType.OP_FRAME)
        self.Torso_id = model.getFrameId(self.torso_frame_name, pin.FrameType.OP_FRAME)

    def _build_collision_model(self) -> None:
        self.collision_spheres = [
            ("2_Joint_Waist_Yaw", np.array([0.0, 0.00, 0.10]), 0.13),
            ("2_Joint_Waist_Yaw", np.array([0.010, 0.00, 0.25]), 0.10),
            ("18_Joint_Elbow_Pitch_Left", np.array([0.00, 0.00, -0.00]), 0.06),
            ("21_Joint_Wrist_Pitch_Left", np.array([0.00, 0.00, -0.05]), 0.05),
            ("25_Joint_Elbow_Pitch_Right", np.array([0.00, 0.00, -0.00]), 0.06),
            ("28_Joint_Wrist_Pitch_Right", np.array([0.00, 0.00, -0.05]), 0.05),
        ]
        self.sphere_attach: list[tuple[AttachKind, int]] = []
        self.sphere_local: list[np.ndarray] = []
        self.sphere_r: list[float] = []
        for name, plocal, radius in self.collision_spheres:
            self.sphere_attach.append(self.resolve_attach_id(self.reduced_robot.model, name))
            self.sphere_local.append(np.asarray(plocal, dtype=np.float64).reshape(3))
            self.sphere_r.append(float(radius))

        self.collision_pairs = [
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 5),
            (2, 4),
            (3, 5),
        ]

    def _init_qp(self) -> None:
        self._waist_dof = 3
        self._neck_dof = 2
        nq = self.reduced_robot.model.nq
        nv = self.reduced_robot.model.nv
        if nq != nv:
            raise ValueError(f"Expected nq == nv for reduced upper body, got nq={nq}, nv={nv}")
        if nq <= self._waist_dof + self._neck_dof:
            raise ValueError(f"Unexpected reduced nq={nq}; expected upper-body layout")

        self._waist_slice = slice(0, self._waist_dof)
        self._arm_slice = slice(self._waist_dof, nq - self._neck_dof)
        self._neck_slice = slice(nq - self._neck_dof, nq)
        self._identity = np.eye(nv, dtype=np.float64)
        self._constraint_inf = 1e20
        self._n_collision_constraints = len(self.collision_pairs)
        self._n_slack = self._n_collision_constraints
        self._slack_offset = nv
        self._n_vars = nv + self._n_slack
        self._collision_constraint_offset = self._n_vars
        self._qp = proxqp.dense.QP(
            self._n_vars,
            0,
            self._n_vars + self._n_collision_constraints,
        )
        self._qp.settings.max_iter = int(self.cfg.solver.max_iter)
        self._qp.settings.eps_abs = float(self.cfg.solver.eps_abs)
        self._qp.settings.eps_rel = float(self.cfg.solver.eps_rel)
        self._qp.settings.verbose = bool(self.cfg.solver.verbose)
        self._qp.settings.initial_guess = proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        self._qp_initialized = False

    def _init_filters_and_state(self) -> None:
        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self._last_dq = np.zeros(self.reduced_robot.model.nv)
        self._prev_dq = np.zeros(self.reduced_robot.model.nv)
        self._default_init_data = None
        try:
            self._default_init_data = self._load_default_init_data()
            self.init_data = self._default_init_data.copy()
        except Exception as exc:
            print(f"[IGRIS_C_UpperIKProxSuite] default_dof_pos load failed: {exc}")

    def _add_least_squares_cost(
        self,
        H: np.ndarray,
        g: np.ndarray,
        J: np.ndarray,
        target: np.ndarray,
        weight: float,
    ) -> None:
        weight = float(weight)
        if weight <= 0.0:
            return
        J_arr = np.asarray(J, dtype=np.float64)
        if J_arr.shape[1] == self.reduced_robot.model.nv and self._n_vars > self.reduced_robot.model.nv:
            padded = np.zeros((J_arr.shape[0], self._n_vars), dtype=np.float64)
            padded[:, : self.reduced_robot.model.nv] = J_arr
            J_arr = padded
        target_arr = np.asarray(target, dtype=np.float64).reshape(J_arr.shape[0])
        H += weight * (J_arr.T @ J_arr)
        g -= weight * (J_arr.T @ target_arr)

    def _add_frame_task(
        self,
        H: np.ndarray,
        g: np.ndarray,
        q: np.ndarray,
        frame_id: int,
        target_pose: np.ndarray,
        trans_weight: float,
        rot_weight: float,
    ) -> None:
        if trans_weight <= 0.0 and rot_weight <= 0.0:
            return

        model = self.reduced_robot.model
        data = self.reduced_robot.data
        current = data.oMf[frame_id].copy()
        target = self._se3_from_homogeneous(target_pose)
        current_to_target = current.actInv(target)
        err = pin.log(current_to_target).vector
        frame_jacobian = pin.computeFrameJacobian(
            model,
            data,
            q,
            frame_id,
            pin.ReferenceFrame.LOCAL,
        )
        # Linearize log(SE3) itself.  The raw frame Jacobian is only accurate
        # close to zero error and caused pitch/yaw to stall or move in bursts
        # for ordinary HMD rotations.
        J = pin.Jlog6(current_to_target.inverse()) @ frame_jacobian

        self._add_least_squares_cost(H, g, J[:3, :], err[:3], trans_weight)
        self._add_least_squares_cost(H, g, J[3:6, :], err[3:6], rot_weight)

    def _sphere_positions_and_jacobians(self, q: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        model = self.reduced_robot.model
        data = self.reduced_robot.data
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q)

        positions: list[np.ndarray] = []
        jacobians: list[np.ndarray] = []
        for (kind, idx), plocal in zip(self.sphere_attach, self.sphere_local):
            if kind == "frame":
                oM = data.oMf[idx]
                J6 = pin.getFrameJacobian(model, data, idx, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            else:
                oM = data.oMi[idx]
                J6 = pin.getJointJacobian(model, data, idx, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

            r_world = oM.rotation @ plocal
            positions.append(oM.translation + r_world)
            jacobians.append(J6[:3, :] - self._skew(r_world) @ J6[3:6, :])
        return positions, jacobians

    def _add_collision_cost(self, H: np.ndarray, g: np.ndarray, q: np.ndarray) -> None:
        col = self.cfg.collision
        if not col.enabled or self.cfg.weights.w_col <= 0.0:
            return

        positions, jacobians = self._sphere_positions_and_jacobians(q)
        for i, j in self.collision_pairs:
            diff = positions[i] - positions[j]
            dist = float(np.linalg.norm(diff))
            if dist < col.min_distance_eps:
                continue

            safe_dist = self.sphere_r[i] + self.sphere_r[j] + col.margin
            if dist >= safe_dist + col.activation_distance:
                continue

            desired_distance_increase = max(0.0, safe_dist - dist)
            if desired_distance_increase <= 0.0:
                continue

            normal = diff / dist
            grad_dist = normal @ (jacobians[i] - jacobians[j])
            self._add_least_squares_cost(
                H,
                g,
                grad_dist.reshape(1, -1),
                np.array([desired_distance_increase], dtype=np.float64),
                self.cfg.weights.w_col,
            )

    def _collision_margins(self, q: np.ndarray) -> list[float]:
        if not self.collision_pairs:
            return []
        positions, _ = self._sphere_positions_and_jacobians(q)
        margins: list[float] = []
        for i, j in self.collision_pairs:
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            safe_dist = self.sphere_r[i] + self.sphere_r[j] + self.cfg.collision.margin
            margins.append(dist - safe_dist)
        return margins

    def _build_qp_matrices(
        self,
        q_current: np.ndarray,
        q_nom: np.ndarray,
        left_wrist: np.ndarray,
        right_wrist: np.ndarray,
        head_pose: np.ndarray,
        torso_pose: np.ndarray | None,
        use_left_hand_target: bool,
        use_right_hand_target: bool,
        use_head_target: bool,
        head_translation_weight: float,
        head_rotation_weight: float,
        use_torso_target: bool,
        use_torso_translation_target: bool,
        torso_translation_weight: float,
        torso_rotation_weight: float,
        arm_lock_q: np.ndarray,
        arm_lock_weight: float,
        waist_lock_q: np.ndarray,
        waist_lock_weight: float,
        step_lower: np.ndarray,
        step_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        model = self.reduced_robot.model
        data = self.reduced_robot.data
        nv = model.nv
        w = self.cfg.weights

        pin.forwardKinematics(model, data, q_current)
        pin.updateFramePlacements(model, data)

        H = np.zeros((self._n_vars, self._n_vars), dtype=np.float64)
        H[:nv, :nv] = w.w_damping * np.eye(nv, dtype=np.float64)
        if self._n_slack > 0:
            slack_slice = slice(self._slack_offset, self._slack_offset + self._n_slack)
            H[slack_slice, slack_slice] = self.cfg.collision.slack_weight * np.eye(self._n_slack)
        g = np.zeros(self._n_vars, dtype=np.float64)

        if use_left_hand_target:
            self._add_frame_task(H, g, q_current, self.L_hand_id, left_wrist, w.w_trans, w.w_rot)
        if use_right_hand_target:
            self._add_frame_task(H, g, q_current, self.R_hand_id, right_wrist, w.w_trans, w.w_rot)
        if use_head_target:
            self._add_frame_task(
                H,
                g,
                q_current,
                self.Head_id,
                head_pose,
                w.w_trans * max(0.0, float(head_translation_weight)),
                w.w_rot * max(0.0, float(head_rotation_weight)),
            )

        if use_torso_target and torso_pose is not None:
            self._add_frame_task(
                H,
                g,
                q_current,
                self.Torso_id,
                torso_pose,
                w.w_trans * max(0.0, float(torso_translation_weight)) if use_torso_translation_target else 0.0,
                w.w_rot * max(0.0, float(torso_rotation_weight)),
            )

        self._add_least_squares_cost(H, g, self._identity, q_nom - q_current, w.w_reg)
        # Smoothness should damp motion, not keep "momentum" from the previous command.
        self._add_least_squares_cost(H, g, self._identity, np.zeros(nv), w.w_smooth)
        if w.w_jerk > 0.0:
            jerk_target = 2.0 * self._last_dq - self._prev_dq
            self._add_least_squares_cost(H, g, self._identity, jerk_target, w.w_jerk)

        if arm_lock_weight > 0.0:
            J_arm = self._identity[self._arm_slice, :]
            self._add_least_squares_cost(
                H,
                g,
                J_arm,
                arm_lock_q[self._arm_slice] - q_current[self._arm_slice],
                arm_lock_weight,
            )

        if waist_lock_weight > 0.0:
            J_waist = self._identity[self._waist_slice, :]
            self._add_least_squares_cost(
                H,
                g,
                J_waist,
                waist_lock_q[self._waist_slice] - q_current[self._waist_slice],
                waist_lock_weight,
            )

        self._add_collision_cost(H, g, q_current)

        H = 0.5 * (H + H.T)
        n_in = self._n_vars + self._n_collision_constraints
        C = np.zeros((n_in, self._n_vars), dtype=np.float64)
        l = np.full(n_in, -self._constraint_inf, dtype=np.float64)
        u = np.full(n_in, self._constraint_inf, dtype=np.float64)
        C[: self._n_vars, :] = np.eye(self._n_vars, dtype=np.float64)
        l[:nv] = step_lower
        u[:nv] = step_upper
        if self._n_slack > 0:
            slack_slice = slice(self._slack_offset, self._slack_offset + self._n_slack)
            l[slack_slice] = 0.0
            u[slack_slice] = max(0.0, float(self.cfg.collision.slack_max))

        build_info: dict[str, Any] = {
            "active_collision_constraints": 0,
            "min_collision_margin_before": None,
        }
        self._add_collision_constraints(C, l, u, q_current, build_info)
        return H, g, C, l, u, build_info

    def _add_collision_constraints(
        self,
        C: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
        q: np.ndarray,
        build_info: dict[str, Any],
    ) -> None:
        col = self.cfg.collision
        if not col.enabled or self._n_collision_constraints == 0:
            return

        positions, jacobians = self._sphere_positions_and_jacobians(q)
        min_margin = self._constraint_inf
        active_count = 0
        for row_offset, (i, j) in enumerate(self.collision_pairs):
            row = self._collision_constraint_offset + row_offset
            diff = positions[i] - positions[j]
            dist = float(np.linalg.norm(diff))
            safe_dist = self.sphere_r[i] + self.sphere_r[j] + col.margin
            margin = dist - safe_dist
            min_margin = min(min_margin, margin)
            if dist < col.min_distance_eps:
                continue
            if margin >= col.activation_distance:
                continue

            normal = diff / dist
            C[row, : self.reduced_robot.model.nv] = normal @ (jacobians[i] - jacobians[j])
            if self._n_slack > 0:
                C[row, self._slack_offset + row_offset] = 1.0
            # If already in collision, enforce "do not worsen" and let the
            # soft collision cost push outward. Requiring full recovery in one
            # bounded dq step can make the whole teleop QP infeasible.
            l[row] = -margin if margin >= 0.0 else 0.0
            u[row] = self._constraint_inf
            active_count += 1

        build_info["active_collision_constraints"] = active_count
        build_info["min_collision_margin_before"] = None if min_margin == self._constraint_inf else min_margin

    def _solve_qp(
        self,
        H: np.ndarray,
        g: np.ndarray,
        C: np.ndarray,
        l: np.ndarray,
        u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._qp_initialized:
            self._qp.init(H, g, None, None, C, l, u, compute_preconditioner=True)
            self._qp_initialized = True
        else:
            self._qp.update(
                H=H,
                g=g,
                C=C,
                l=l,
                u=u,
                update_preconditioner=bool(self.cfg.solver.update_preconditioner),
            )

        self._qp.solve()
        status = str(self._qp.results.info.status)
        x = np.asarray(self._qp.results.x, dtype=np.float64).reshape(-1)
        if "SOLVED" not in status or not np.all(np.isfinite(x)):
            raise RuntimeError(f"ProxQP failed with status={status}")
        dq = x[: self.reduced_robot.model.nv]
        slack = x[self._slack_offset : self._slack_offset + self._n_slack]
        return dq, slack

    def _target_residuals(
        self,
        q: np.ndarray,
        targets: list[tuple[str, int, np.ndarray, bool]],
    ) -> dict[str, float]:
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
        out: dict[str, float] = {}
        trans_values: list[float] = []
        rot_values: list[float] = []
        for name, frame_id, target_pose, enabled in targets:
            if not enabled:
                continue
            current = self.reduced_robot.data.oMf[frame_id].copy()
            target = self._se3_from_homogeneous(target_pose)
            err = pin.log(current.actInv(target)).vector
            trans = float(np.linalg.norm(err[:3]))
            rot = float(np.linalg.norm(err[3:6]))
            out[f"{name}_trans_error"] = trans
            out[f"{name}_rot_error"] = rot
            trans_values.append(trans)
            rot_values.append(rot)
        out["max_trans_error"] = max(trans_values, default=0.0)
        out["max_rot_error"] = max(rot_values, default=0.0)
        return out

    def get_last_solve_info(self) -> dict[str, Any]:
        return dict(self._last_solve_info)

    def reset_motion_state(self, q: np.ndarray | None = None) -> None:
        self._last_dq = np.zeros(self.reduced_robot.model.nv)
        self._prev_dq = np.zeros(self.reduced_robot.model.nv)
        if q is not None:
            self.init_data = self._clip_q_to_limits(np.asarray(q, dtype=np.float64).reshape(-1))

    def commit_published_command(
        self,
        command_q: np.ndarray,
        previous_command_q: np.ndarray,
    ) -> None:
        """Keep rate-limit history aligned with the command actually published."""
        command = np.asarray(command_q, dtype=np.float64).reshape(-1)
        previous = np.asarray(previous_command_q, dtype=np.float64).reshape(-1)
        expected = (self.reduced_robot.model.nq,)
        if command.shape != expected or previous.shape != expected:
            raise ValueError(
                "published command shape mismatch: "
                f"command={command.shape}, previous={previous.shape}, expected={expected}"
            )
        if not np.all(np.isfinite(command)) or not np.all(np.isfinite(previous)):
            raise ValueError("published commands must be finite")

        clipped_command = self._clip_q_to_limits(command)
        clipped_previous = self._clip_q_to_limits(previous)
        if not np.allclose(command, clipped_command, atol=1e-10, rtol=0.0):
            raise ValueError("published command is outside joint limits")
        if not np.allclose(previous, clipped_previous, atol=1e-10, rtol=0.0):
            raise ValueError("previous published command is outside joint limits")

        previous_delta = self._last_dq.copy()
        self.init_data = clipped_command.copy()
        self._prev_dq = previous_delta
        self._last_dq = clipped_command - clipped_previous

    def _current_pose_matrix(self, q: np.ndarray, frame_id: int) -> np.ndarray:
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
        return np.asarray(self.reduced_robot.data.oMf[frame_id].homogeneous, dtype=np.float64)

    def _motion_step_bounds(self, q_reference: np.ndarray, q_current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model = self.reduced_robot.model
        nv = model.nv
        lower_q = np.asarray(model.lowerPositionLimit, dtype=np.float64)
        upper_q = np.asarray(model.upperPositionLimit, dtype=np.float64)
        total_lower = np.maximum(lower_q - q_reference, -np.full(nv, self.cfg.rate_limit.dq_max))
        total_upper = np.minimum(upper_q - q_reference, np.full(nv, self.cfg.rate_limit.dq_max))

        ddq_max = float(self.cfg.rate_limit.ddq_max)
        if ddq_max > 0.0 and np.isfinite(ddq_max):
            acc_lower = self._last_dq - ddq_max
            acc_upper = self._last_dq + ddq_max
            next_lower = np.maximum(total_lower, acc_lower)
            next_upper = np.minimum(total_upper, acc_upper)
            valid = next_lower <= next_upper
            total_lower[valid] = next_lower[valid]
            total_upper[valid] = next_upper[valid]

        dddq_max = float(self.cfg.rate_limit.dddq_max)
        if dddq_max > 0.0 and np.isfinite(dddq_max):
            jerk_center = 2.0 * self._last_dq - self._prev_dq
            jerk_lower = jerk_center - dddq_max
            jerk_upper = jerk_center + dddq_max
            next_lower = np.maximum(total_lower, jerk_lower)
            next_upper = np.minimum(total_upper, jerk_upper)
            valid = next_lower <= next_upper
            total_lower[valid] = next_lower[valid]
            total_upper[valid] = next_upper[valid]

        delta_so_far = np.asarray(q_current - q_reference, dtype=np.float64).reshape(nv)
        step_lower = total_lower - delta_so_far
        step_upper = total_upper - delta_so_far
        invalid = step_lower > step_upper
        if np.any(invalid):
            step_lower[invalid] = 0.0
            step_upper[invalid] = 0.0
        return step_lower, step_upper

    def solve_ik(
        self,
        left_wrist,
        right_wrist,
        head_pose,
        torso_pose=None,
        use_torso_target: bool = False,
        use_torso_translation_target: bool = True,
        torso_target_weight: float = 1.0,
        torso_translation_weight: float | None = None,
        torso_rotation_weight: float | None = None,
        current_lr_arm_motor_q=None,
        current_lr_arm_motor_dq=None,
        box_pose=None,
        use_hand_targets: bool | None = None,
        use_left_hand_target: bool = True,
        use_right_hand_target: bool = True,
        use_head_target: bool = True,
        head_translation_weight: float = 1.0,
        head_rotation_weight: float = 1.0,
        arm_lock_q=None,
        arm_lock_weight: float = 0.0,
        waist_lock_q=None,
        waist_lock_weight: float = 0.0,
    ):
        del current_lr_arm_motor_dq, box_pose

        q_init = (
            np.asarray(current_lr_arm_motor_q, dtype=np.float64).copy()
            if current_lr_arm_motor_q is not None
            else (self._default_init_data.copy() if self._default_init_data is not None else self.init_data.copy())
        )
        q_init = self._clip_q_to_limits(q_init)
        if self.init_data.shape == q_init.shape:
            seed_jump = float(np.max(np.abs(q_init - self.init_data)))
            if seed_jump > max(0.03, 0.5 * float(self.cfg.rate_limit.dq_max)):
                self.reset_motion_state(q_init)
        q_nom = self._clip_q_to_limits(self._default_init_data.copy() if self._default_init_data is not None else q_init)

        current_left_pose = self._current_pose_matrix(q_init, self.L_hand_id)
        current_right_pose = self._current_pose_matrix(q_init, self.R_hand_id)
        current_head_pose = self._current_pose_matrix(q_init, self.Head_id)
        current_torso_pose = self._current_pose_matrix(q_init, self.Torso_id)

        left_wrist = self._sanitize_pose_matrix(left_wrist, current_left_pose)
        right_wrist = self._sanitize_pose_matrix(right_wrist, current_right_pose)
        head_pose = self._sanitize_pose_matrix(head_pose, current_head_pose)

        if torso_pose is None:
            torso_target_pose = current_torso_pose
        else:
            torso_target_pose = self._sanitize_pose_matrix(torso_pose, current_torso_pose)

        if use_hand_targets is not None:
            use_left_hand_target = bool(use_hand_targets)
            use_right_hand_target = bool(use_hand_targets)

        if arm_lock_q is None:
            arm_lock_ref = q_init.copy()
        else:
            arm_lock_ref = np.asarray(arm_lock_q, dtype=np.float64).reshape(-1)
            if arm_lock_ref.shape != q_init.shape or not np.all(np.isfinite(arm_lock_ref)):
                arm_lock_ref = q_init.copy()
        arm_lock_ref = self._clip_q_to_limits(arm_lock_ref)

        if waist_lock_q is None:
            waist_lock_ref = q_init.copy()
        else:
            waist_lock_ref = np.asarray(waist_lock_q, dtype=np.float64).reshape(-1)
            if waist_lock_ref.shape != q_init.shape or not np.all(np.isfinite(waist_lock_ref)):
                waist_lock_ref = q_init.copy()
        waist_lock_ref = self._clip_q_to_limits(waist_lock_ref)

        torso_trans_w = (
            max(0.0, float(torso_target_weight))
            if torso_translation_weight is None
            else max(0.0, float(torso_translation_weight))
        )
        torso_rot_w = (
            max(0.0, float(torso_target_weight))
            if torso_rotation_weight is None
            else max(0.0, float(torso_rotation_weight))
        )

        targets_for_telemetry = [
            ("left", self.L_hand_id, left_wrist, bool(use_left_hand_target)),
            ("right", self.R_hand_id, right_wrist, bool(use_right_hand_target)),
            ("head", self.Head_id, head_pose, bool(use_head_target)),
            ("torso", self.Torso_id, torso_target_pose, bool(use_torso_target)),
        ]
        residuals_before = self._target_residuals(q_init, targets_for_telemetry)

        fallback_q = q_init.copy()
        start_time = time.perf_counter()
        try:
            q_work = q_init.copy()
            total_qp_iters = 0
            max_step_dq = 0.0
            max_collision_slack = 0.0
            build_info: dict[str, Any] = {
                "active_collision_constraints": 0,
                "min_collision_margin_before": None,
            }
            sqp_iters = max(1, int(self.cfg.solver.linearization_iters))
            for _ in range(sqp_iters):
                step_lower, step_upper = self._motion_step_bounds(q_init, q_work)
                H, g, C, l, u, build_info = self._build_qp_matrices(
                    q_current=q_work,
                    q_nom=q_nom,
                    left_wrist=left_wrist,
                    right_wrist=right_wrist,
                    head_pose=head_pose,
                    torso_pose=torso_target_pose,
                    use_left_hand_target=bool(use_left_hand_target),
                    use_right_hand_target=bool(use_right_hand_target),
                    use_head_target=bool(use_head_target),
                    head_translation_weight=head_translation_weight,
                    head_rotation_weight=head_rotation_weight,
                    use_torso_target=bool(use_torso_target),
                    use_torso_translation_target=bool(use_torso_translation_target),
                    torso_translation_weight=torso_trans_w,
                    torso_rotation_weight=torso_rot_w,
                    arm_lock_q=arm_lock_ref,
                    arm_lock_weight=max(0.0, float(arm_lock_weight)),
                    waist_lock_q=waist_lock_ref,
                    waist_lock_weight=max(0.0, float(waist_lock_weight)),
                    step_lower=step_lower,
                    step_upper=step_upper,
                )
                dq_step, slack = self._solve_qp(H, g, C, l, u)
                total_qp_iters += 1
                max_step_dq = max(max_step_dq, float(np.max(np.abs(dq_step))) if dq_step.size else 0.0)
                max_collision_slack = max(
                    max_collision_slack,
                    float(np.max(np.abs(slack))) if slack.size else 0.0,
                )
                q_next = pin.integrate(self.reduced_robot.model, q_work, dq_step)
                q_next = self._clip_q_to_limits(q_next)
                q_work = q_next
                if np.max(np.abs(dq_step)) < 1e-7:
                    break

            final_lower, final_upper = self._motion_step_bounds(q_init, q_init)
            final_delta = np.clip(np.asarray(q_work - q_init, dtype=np.float64), final_lower, final_upper)
            sol_q = self._clip_q_to_limits(q_init + final_delta)
            self.init_data = sol_q.copy()
            returned_delta = np.asarray(sol_q - q_init, dtype=np.float64).reshape(-1)
            self._prev_dq = self._last_dq.copy()
            self._last_dq = returned_delta.copy()
            residuals_after = self._target_residuals(sol_q, targets_for_telemetry)
            margins_after = self._collision_margins(sol_q) if self.cfg.collision.enabled else []

            info = self._qp.results.info
            self._last_solve_info = {
                "success": True,
                "status": str(info.status),
                "iter": int(info.iter),
                "objective": float(info.objValue),
                "primal_residual": float(info.pri_res),
                "dual_residual": float(info.dua_res),
                "solve_time_s": float(getattr(info, "solve_time", 0.0)),
                "setup_time_s": float(getattr(info, "setup_time", 0.0)),
                "wall_time_s": float(time.perf_counter() - start_time),
                "sqp_iters": int(total_qp_iters),
                "max_dq": max_step_dq,
                "max_returned_delta": float(np.max(np.abs(returned_delta))) if returned_delta.size else 0.0,
                "max_waist_delta": float(np.max(np.abs(returned_delta[self._waist_slice]))),
                "max_arm_delta": float(np.max(np.abs(returned_delta[self._arm_slice]))),
                "max_neck_delta": float(np.max(np.abs(returned_delta[self._neck_slice]))),
                "max_collision_slack": max_collision_slack,
                "active_collision_constraints": int(build_info.get("active_collision_constraints", 0)),
                "min_collision_margin_before": build_info.get("min_collision_margin_before"),
                "min_collision_margin_after": min(margins_after) if margins_after else None,
                "max_trans_error_before": float(residuals_before.get("max_trans_error", 0.0)),
                "max_rot_error_before": float(residuals_before.get("max_rot_error", 0.0)),
                "max_trans_error_after": float(residuals_after.get("max_trans_error", 0.0)),
                "max_rot_error_after": float(residuals_after.get("max_rot_error", 0.0)),
            }

            nv = self.reduced_robot.model.nv
            tau = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                np.zeros(nv),
                np.zeros(nv),
            )
            return sol_q, tau

        except Exception as exc:
            print(f"[IGRIS_C_UpperIKProxSuite] IK failed: {exc}")
            self._last_solve_info = {
                "success": False,
                "status": "exception",
                "error": str(exc),
                "wall_time_s": float(time.perf_counter() - start_time),
                "max_trans_error_before": float(residuals_before.get("max_trans_error", 0.0)),
                "max_rot_error_before": float(residuals_before.get("max_rot_error", 0.0)),
            }
            return fallback_q, np.zeros(self.reduced_robot.model.nv)

    def get_ee_poses(self, q):
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q_arr)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
        return (
            self.reduced_robot.data.oMf[self.L_hand_id],
            self.reduced_robot.data.oMf[self.R_hand_id],
            self.reduced_robot.data.oMf[self.Head_id],
        )

    def get_torso_pose(self, q):
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q_arr)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
        return self.reduced_robot.data.oMf[self.Torso_id]

    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = np.asarray(human_left_pose, dtype=np.float64).copy()
        robot_right_pose = np.asarray(human_right_pose, dtype=np.float64).copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def close(self) -> None:
        return


# Drop-in alias for experiments that import IGRIS_C_UpperIK from this module.
IGRIS_C_UpperIK = IGRIS_C_UpperIKProxSuite


if __name__ == "__main__":
    arm_ik = IGRIS_C_UpperIKProxSuite()
    q0 = arm_ik._default_init_data.copy() if arm_ik._default_init_data is not None else arm_ik.init_data.copy()
    l0, r0, h0 = arm_ik.get_ee_poses(q0)
    L_target = l0.homogeneous.copy()
    R_target = r0.homogeneous.copy()
    H_target = h0.homogeneous.copy()
    L_target[:3, 3] += np.array([0.0, 0.03, 0.0])
    R_target[:3, 3] += np.array([0.0, -0.03, 0.0])
    q, tau = arm_ik.solve_ik(L_target, R_target, H_target, current_lr_arm_motor_q=q0)
    print("q:", q)
    print("tau:", tau)
