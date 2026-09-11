from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from igris_teleop.sim.hand_physics import (
    HAND_NEAR_OBJECT_CLOSING_TORQUE_LIMIT,
    HAND_NEAR_OBJECT_OPENING_BRAKE_LIMIT,
    HAND_SERVO_KD,
    HAND_SERVO_KP,
)


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "robot"
    / "mujoco"
    / "igris_c_v2_with_hand.xml"
)

HAND_JOINT_NAMES = (
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
HAND_COLLISION_CATEGORY = 2
RIGHT_REVERSED_JOINT = "Joint_Thumb_Proximal_Right"


@dataclass(frozen=True)
class PhysicsSettings:
    integrator: str = "Euler"
    solver: str = "Newton"
    cone: str = "elliptic"
    iterations: int = 500
    tolerance: float = 1e-7
    noslip_iterations: int = 100
    noslip_tolerance: float = 1e-7


@dataclass(frozen=True)
class GraspStabilityResult:
    max_object_speed_m_s: float
    max_object_displacement_m: float
    final_object_displacement_m: float
    final_vertical_displacement_m: float
    max_normal_force_n: float
    max_penetration_m: float
    contact_samples: int
    final_closed_fingers: int


_INTEGRATORS = {
    "Euler": mujoco.mjtIntegrator.mjINT_EULER,
    "RK4": mujoco.mjtIntegrator.mjINT_RK4,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
_SOLVERS = {
    "PGS": mujoco.mjtSolver.mjSOL_PGS,
    "CG": mujoco.mjtSolver.mjSOL_CG,
    "Newton": mujoco.mjtSolver.mjSOL_NEWTON,
}
_CONES = {
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
}


def _object_id(model: mujoco.MjModel, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo model is missing {name!r}")
    return int(object_id)


def _apply_physics_settings(
    model: mujoco.MjModel,
    settings: PhysicsSettings,
) -> None:
    try:
        model.opt.integrator = _INTEGRATORS[settings.integrator]
        model.opt.solver = _SOLVERS[settings.solver]
        model.opt.cone = _CONES[settings.cone]
    except KeyError as exc:
        raise ValueError(f"unsupported MuJoCo physics setting: {exc.args[0]}") from exc
    model.opt.iterations = int(settings.iterations)
    model.opt.tolerance = float(settings.tolerance)
    model.opt.noslip_iterations = int(settings.noslip_iterations)
    model.opt.noslip_tolerance = float(settings.noslip_tolerance)


def _resolve_hand_collision_mapping(
    model: mujoco.MjModel,
) -> dict[int, int]:
    actuator_index_by_joint = {
        name: index for index, name in enumerate(HAND_JOINT_NAMES)
    }
    mapping: dict[int, int] = {}
    for geom_id in range(int(model.ngeom)):
        if not int(model.geom_contype[geom_id]) & HAND_COLLISION_CATEGORY:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        while body_id > 0:
            joint_start = int(model.body_jntadr[body_id])
            joint_count = int(model.body_jntnum[body_id])
            for joint_id in range(joint_start, joint_start + joint_count):
                joint_name = mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_id,
                )
                actuator_index = actuator_index_by_joint.get(str(joint_name))
                if actuator_index is not None:
                    mapping[geom_id] = actuator_index
                    body_id = 0
                    break
            else:
                body_id = int(model.body_parentid[body_id])
    return mapping


def _hand_state_indices(
    model: mujoco.MjModel,
) -> tuple[set[int], set[int]]:
    qpos_addresses: set[int] = set()
    dof_addresses: set[int] = set()
    tokens = ("Thumb_", "Index_", "Middle_", "Ring_", "Little_")
    for joint_id in range(int(model.njnt)):
        name = (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
            or ""
        )
        if any(token in name for token in tokens):
            qpos_addresses.add(int(model.jnt_qposadr[joint_id]))
            dof_addresses.add(int(model.jnt_dofadr[joint_id]))
    return qpos_addresses, dof_addresses


def run_left_cube_grasp(
    *,
    model_path: str | Path = MODEL_PATH,
    physics: PhysicsSettings = PhysicsSettings(),
    actuator_torque_limit_nm: float | None = None,
    contact_closing_torque_limit_nm: float = (
        HAND_NEAR_OBJECT_CLOSING_TORQUE_LIMIT
    ),
    contact_opening_brake_limit_nm: float = (
        HAND_NEAR_OBJECT_OPENING_BRAKE_LIMIT
    ),
    hand_servo_kp: float = HAND_SERVO_KP,
    hand_servo_kd: float = HAND_SERVO_KD,
    cube_local_position_m: tuple[float, float, float] = (
        0.008,
        0.038,
        0.105,
    ),
) -> GraspStabilityResult:
    model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
    _apply_physics_settings(model, physics)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    model.opt.gravity[:] = 0.0

    cube_geom_id = _object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "exp_task1_cube_2_geom",
    )
    for geom_id in range(int(model.ngeom)):
        is_hand = bool(
            int(model.geom_contype[geom_id]) & HAND_COLLISION_CATEGORY
        )
        if not is_hand and geom_id != cube_geom_id:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0

    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    actuator_ids: list[int] = []
    open_positions: list[float] = []
    closed_positions: list[float] = []
    for name in HAND_JOINT_NAMES:
        joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = _object_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        actuator_ids.append(actuator_id)
        lower, upper = model.jnt_range[joint_id]
        if name == RIGHT_REVERSED_JOINT:
            lower, upper = upper, lower
        open_positions.append(float(lower))
        closed_positions.append(float(upper))

    qpos_idx = np.asarray(qpos_addresses, dtype=np.int32)
    dof_idx = np.asarray(dof_addresses, dtype=np.int32)
    actuator_idx = np.asarray(actuator_ids, dtype=np.int32)
    open_q = np.asarray(open_positions, dtype=np.float64)
    closed_q = np.asarray(closed_positions, dtype=np.float64)
    closing_direction = np.sign(closed_q - open_q)
    actuator_limits = np.max(
        np.abs(model.actuator_ctrlrange[actuator_idx]),
        axis=1,
    )
    if actuator_torque_limit_nm is not None:
        requested_limit = float(actuator_torque_limit_nm)
        if not np.isfinite(requested_limit) or requested_limit <= 0.0:
            raise ValueError("actuator torque limit must be finite and positive")
        actuator_limits[:] = requested_limit

    hand_qpos_addresses, hand_dof_addresses = _hand_state_indices(model)
    geom_to_actuator = _resolve_hand_collision_mapping(model)

    data.qpos[qpos_idx] = open_q
    mujoco.mj_forward(model, data)
    hand_body_id = _object_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "Left_Hand",
    )
    hand_rotation = np.asarray(
        data.xmat[hand_body_id],
        dtype=np.float64,
    ).reshape(3, 3).copy()
    hand_position = np.asarray(
        data.xpos[hand_body_id],
        dtype=np.float64,
    ).copy()
    cube_initial_position = (
        hand_position
        + hand_rotation
        @ np.asarray(cube_local_position_m, dtype=np.float64)
    )

    cube_joint_id = _object_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "exp_task1_cube_2_joint",
    )
    cube_qpos_address = int(model.jnt_qposadr[cube_joint_id])
    cube_dof_address = int(model.jnt_dofadr[cube_joint_id])
    data.qpos[cube_qpos_address : cube_qpos_address + 3] = (
        cube_initial_position
    )
    data.qpos[cube_qpos_address + 3 : cube_qpos_address + 7] = (
        data.xquat[hand_body_id]
    )
    mujoco.mj_forward(model, data)

    initial_qpos = data.qpos.copy()
    fixed_qpos = np.asarray(
        [
            index
            for index in range(int(model.nq))
            if index not in hand_qpos_addresses
            and not cube_qpos_address <= index < cube_qpos_address + 7
        ],
        dtype=np.int32,
    )
    fixed_dofs = np.asarray(
        [
            index
            for index in range(int(model.nv))
            if index not in hand_dof_addresses
            and not cube_dof_address <= index < cube_dof_address + 6
        ],
        dtype=np.int32,
    )

    close_step = int(round(0.1 / float(model.opt.timestep)))
    gravity_step = int(round(0.7 / float(model.opt.timestep)))
    total_steps = int(round(0.9 / float(model.opt.timestep)))
    first_contact_seen = False
    max_normal_force = 0.0
    max_penetration = 0.0
    max_object_speed = 0.0
    max_object_displacement = 0.0
    contact_samples = 0

    for step in range(total_steps):
        if step == gravity_step:
            model.opt.gravity[:] = (0.0, 0.0, -9.81)

        data.qpos[fixed_qpos] = initial_qpos[fixed_qpos]
        data.qvel[fixed_dofs] = 0.0
        mujoco.mj_forward(model, data)

        target_q = open_q if step < close_step else closed_q
        position_error = target_q - data.qpos[qpos_idx]
        ctrl = (
            float(hand_servo_kp) * position_error
            - float(hand_servo_kd) * data.qvel[dof_idx]
        )
        near = np.zeros(len(HAND_JOINT_NAMES), dtype=bool)
        for contact_id in range(int(data.ncon)):
            contact = data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if cube_geom_id not in (geom1, geom2):
                continue
            hand_geom_id = geom2 if geom1 == cube_geom_id else geom1
            actuator_index = geom_to_actuator.get(hand_geom_id)
            if actuator_index is not None:
                near[actuator_index] = True
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_id, force)
            max_normal_force = max(
                max_normal_force,
                abs(float(force[0])),
            )
            max_penetration = max(
                max_penetration,
                -float(contact.dist),
            )
            contact_samples += 1
            first_contact_seen = True

        closing_contacts = near & (
            position_error * closing_direction > 0.0
        )
        signed_torque = ctrl * closing_direction
        signed_torque[closing_contacts] = np.clip(
            signed_torque[closing_contacts],
            -float(contact_opening_brake_limit_nm),
            float(contact_closing_torque_limit_nm),
        )
        ctrl = signed_torque * closing_direction

        data.ctrl[:] = 0.0
        data.ctrl[actuator_idx] = np.clip(
            ctrl,
            -actuator_limits,
            actuator_limits,
        )
        mujoco.mj_step(model, data)

        if first_contact_seen:
            object_speed = float(
                np.linalg.norm(
                    data.qvel[cube_dof_address : cube_dof_address + 3]
                )
            )
            displacement = float(
                np.linalg.norm(
                    hand_rotation.T
                    @ (
                        data.qpos[
                            cube_qpos_address : cube_qpos_address + 3
                        ]
                        - cube_initial_position
                    )
                )
            )
            max_object_speed = max(max_object_speed, object_speed)
            max_object_displacement = max(
                max_object_displacement,
                displacement,
            )

    final_displacement = (
        hand_rotation.T
        @ (
            data.qpos[cube_qpos_address : cube_qpos_address + 3]
            - cube_initial_position
        )
    )
    final_normalized = (
        data.qpos[qpos_idx] - open_q
    ) / (closed_q - open_q)
    return GraspStabilityResult(
        max_object_speed_m_s=max_object_speed,
        max_object_displacement_m=max_object_displacement,
        final_object_displacement_m=float(
            np.linalg.norm(final_displacement)
        ),
        final_vertical_displacement_m=float(final_displacement[2]),
        max_normal_force_n=max_normal_force,
        max_penetration_m=max_penetration,
        contact_samples=contact_samples,
        final_closed_fingers=int(
            np.count_nonzero(final_normalized[6:] > 0.25)
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark MuJoCo hand/cube grasp stability.",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--actuator-torque-limit-nm", type=float)
    parser.add_argument(
        "--contact-closing-torque-limit-nm",
        type=float,
        default=HAND_NEAR_OBJECT_CLOSING_TORQUE_LIMIT,
    )
    parser.add_argument(
        "--integrator",
        choices=tuple(_INTEGRATORS),
        default="Euler",
    )
    parser.add_argument(
        "--solver",
        choices=tuple(_SOLVERS),
        default="Newton",
    )
    parser.add_argument(
        "--cone",
        choices=tuple(_CONES),
        default="elliptic",
    )
    parser.add_argument("--noslip-iterations", type=int, default=100)
    parser.add_argument(
        "--cube-local-position-m",
        type=float,
        nargs=3,
        default=(0.008, 0.038, 0.105),
        metavar=("X", "Y", "Z"),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_left_cube_grasp(
        model_path=args.model,
        physics=PhysicsSettings(
            integrator=args.integrator,
            solver=args.solver,
            cone=args.cone,
            noslip_iterations=args.noslip_iterations,
        ),
        actuator_torque_limit_nm=args.actuator_torque_limit_nm,
        contact_closing_torque_limit_nm=(
            args.contact_closing_torque_limit_nm
        ),
        cube_local_position_m=tuple(args.cube_local_position_m),
    )
    print(
        "[grasp-benchmark] "
        f"speed={result.max_object_speed_m_s:.4f} m/s "
        f"max_disp={1000.0 * result.max_object_displacement_m:.1f} mm "
        f"final_disp={1000.0 * result.final_object_displacement_m:.1f} mm "
        f"final_z={1000.0 * result.final_vertical_displacement_m:+.1f} mm "
        f"force={result.max_normal_force_n:.1f} N "
        f"penetration={1000.0 * result.max_penetration_m:.2f} mm "
        f"contacts={result.contact_samples} "
        f"closed_fingers={result.final_closed_fingers}"
    )


if __name__ == "__main__":
    main()
