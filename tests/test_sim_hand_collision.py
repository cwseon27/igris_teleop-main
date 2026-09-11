from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from igris_teleop.workers.worker_simulator import (  # noqa: E402
    EXPERIMENT_GRASP_COLLISION_CONAFFINITY,
    EXPERIMENT_GRASP_COLLISION_CONTYPE,
    EXPERIMENT_SUPPORT_COLLISION_CONAFFINITY,
    EXPERIMENT_SUPPORT_COLLISION_CONTYPE,
    HAND_JOINT_NAMES_DDS_ORDER,
    HAND_NEAR_OBJECT_CLOSING_TORQUE_LIMIT,
    HAND_NEAR_OBJECT_OPENING_BRAKE_LIMIT,
    HAND_SERVO_KD,
    HAND_SERVO_KP,
    HAND_TARGET_LENGTH,
    MujocoSimulationWorker,
)


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "igris_teleop"
    / "sim"
    / "robot"
    / "mujoco"
    / "igris_c_v2_with_hand.xml"
)


def test_near_object_limiter_only_reduces_closing_torque() -> None:
    worker = MujocoSimulationWorker.__new__(MujocoSimulationWorker)
    worker._hand_closing_direction = np.ones(HAND_TARGET_LENGTH, dtype=np.float32)
    worker._hand_closing_direction[5] = -1.0
    requested_torque = HAND_NEAR_OBJECT_CLOSING_TORQUE_LIMIT * 2.0
    ctrl = worker._hand_closing_direction * requested_torque
    ctrl[1] = -worker._hand_closing_direction[1] * requested_torque
    near = np.zeros(HAND_TARGET_LENGTH, dtype=bool)
    near[[0, 1, 5, 7]] = True
    position_error = worker._hand_closing_direction.copy()
    position_error[7] *= -1.0

    limited = worker._limit_near_object_hand_closing_torque(
        ctrl,
        near,
        position_error,
    )

    expected = ctrl.copy()
    expected[[0, 5]] = worker._hand_closing_direction[[0, 5]] * (
        HAND_NEAR_OBJECT_CLOSING_TORQUE_LIMIT
    )
    expected[1] = -worker._hand_closing_direction[1] * (
        HAND_NEAR_OBJECT_OPENING_BRAKE_LIMIT
    )
    np.testing.assert_allclose(limited, expected)


def test_near_object_detection_is_scoped_to_contacting_fingers() -> None:
    worker = MujocoSimulationWorker.__new__(MujocoSimulationWorker)
    worker.model = SimpleNamespace(
        geom_contype=np.asarray([2, 2, 8, 4, 2], dtype=np.int32)
    )
    worker.data = SimpleNamespace(
        ncon=3,
        contact=[
            SimpleNamespace(geom1=0, geom2=2),
            SimpleNamespace(geom1=1, geom2=3),
            SimpleNamespace(geom1=4, geom2=2),
        ],
    )
    worker._hand_collision_geom_to_actuator_index = {0: 0, 1: 7}

    near = worker._hand_actuators_near_grasp_object()

    np.testing.assert_array_equal(np.flatnonzero(near), [0])


def test_hand_collision_mapping_ignores_palm_and_maps_distal_to_driver() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    mapping = (
        MujocoSimulationWorker._resolve_hand_collision_geom_to_actuator_index(model)
    )

    palm_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "Left_Palm_collision",
    )
    index_distal_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "Link_Index_Distal_Left_collision",
    )
    thumb_proximal_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "Link_Thumb_Proximal_Left_collision",
    )
    assert len(mapping) == 22
    assert palm_id not in mapping
    assert mapping[index_distal_id] == 7
    assert mapping[thumb_proximal_id] == 11


def _mesh_vertices_in_body_frame(
    model: mujoco.MjModel,
    mesh_name: str,
) -> np.ndarray:
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
    vertex_adr = int(model.mesh_vertadr[mesh_id])
    vertex_num = int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(
        model.mesh_vert[vertex_adr : vertex_adr + vertex_num],
        dtype=np.float64,
    )
    rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, model.mesh_quat[mesh_id])
    return (
        vertices @ rotation.reshape(3, 3).T
        + np.asarray(model.mesh_pos[mesh_id], dtype=np.float64)
    )


@pytest.mark.parametrize(
    ("side", "mesh_name", "grip_direction"),
    (
        ("Left", "Palm_Left", 1.0),
        ("Right", "Palm_Right", -1.0),
    ),
)
def test_upper_palm_collision_is_inset_from_visual_mesh(
    side: str,
    mesh_name: str,
    grip_direction: float,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    lower_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"{side}_Palm_collision",
    )
    grip_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"{side}_Palm_grip_collision",
    )
    assert lower_id >= 0
    assert grip_id >= 0

    vertices = _mesh_vertices_in_body_frame(model, mesh_name)
    upper_vertices = vertices[vertices[:, 2] >= 0.075]
    visual_surface = float(np.max(grip_direction * upper_vertices[:, 1]))
    collision_surface = float(
        grip_direction * model.geom_pos[grip_id, 1]
        + model.geom_size[grip_id, 1]
    )

    assert collision_surface <= visual_surface + 1e-6
    assert visual_surface - collision_surface <= 0.005
    assert float(model.geom_margin[grip_id]) == 0.0


def test_hand_actuators_and_contact_servo_allow_strong_grasp() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    for name in HAND_JOINT_NAMES_DDS_ORDER:
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        np.testing.assert_allclose(
            model.actuator_ctrlrange[actuator_id],
            (-15.0, 15.0),
        )

    assert HAND_NEAR_OBJECT_CLOSING_TORQUE_LIMIT == 2.8
    assert HAND_NEAR_OBJECT_OPENING_BRAKE_LIMIT == 1.0
    assert HAND_SERVO_KP == 30.0
    assert HAND_SERVO_KD == 1.2


@pytest.mark.parametrize(
    ("geom_name", "expected_size"),
    (
        (
            "Link_Thumb_Proximal_Left_collision",
            (0.0072, 0.0056, 0.01384702),
        ),
        (
            "Link_Thumb_Middle_Left_collision",
            (0.0063, 0.0049, 0.02000625),
        ),
        (
            "Link_Thumb_Distal_Left_collision",
            (0.0054, 0.0042, 0.0175),
        ),
        (
            "Link_Index_Middle_Left_collision",
            (0.0063, 0.0049, 0.02000625),
        ),
        (
            "Link_Index_Distal_Left_collision",
            (0.0054, 0.0042, 0.0175),
        ),
        (
            "Link_Thumb_Proximal_Right_collision",
            (0.0072, 0.0056, 0.01384061),
        ),
        (
            "Link_Thumb_Middle_Right_collision",
            (0.0063, 0.0049, 0.02000625),
        ),
        (
            "Link_Thumb_Distal_Right_collision",
            (0.0054, 0.0042, 0.0175),
        ),
        (
            "Link_Index_Middle_Right_collision",
            (0.0063, 0.0049, 0.02000625),
        ),
        (
            "Link_Index_Distal_Right_collision",
            (0.0054, 0.0042, 0.0175),
        ),
    ),
)
def test_finger_collision_boxes_are_inset(
    geom_name: str,
    expected_size: tuple[float, float, float],
) -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_name,
    )

    assert geom_id >= 0
    assert int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX)
    np.testing.assert_allclose(
        model.geom_size[geom_id],
        expected_size,
        atol=1e-8,
    )
    assert float(model.geom_friction[geom_id, 0]) == pytest.approx(1.0)


def test_cube_grasp_is_damped_and_holds_against_gravity() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    model.opt.gravity[:] = 0.0

    cube_geom_name = "exp_task1_cube_2_geom"
    cube_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        cube_geom_name,
    )
    assert cube_geom_id >= 0

    # Isolate one hand and one cube so this test measures grasp contact only.
    for geom_id in range(int(model.ngeom)):
        is_hand = bool(int(model.geom_contype[geom_id]) & 2)
        if not is_hand and geom_id != cube_geom_id:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0

    qpos_adrs: list[int] = []
    dof_adrs: list[int] = []
    actuator_ids: list[int] = []
    open_q: list[float] = []
    closed_q: list[float] = []
    for name in HAND_JOINT_NAMES_DDS_ORDER:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
        dof_adrs.append(int(model.jnt_dofadr[joint_id]))
        actuator_ids.append(actuator_id)
        lower, upper = model.jnt_range[joint_id]
        if name == "Joint_Thumb_Proximal_Right":
            lower, upper = upper, lower
        open_q.append(float(lower))
        closed_q.append(float(upper))

    qpos_idx = np.asarray(qpos_adrs, dtype=np.int32)
    dof_idx = np.asarray(dof_adrs, dtype=np.int32)
    actuator_idx = np.asarray(actuator_ids, dtype=np.int32)
    open_arr = np.asarray(open_q, dtype=np.float64)
    closed_arr = np.asarray(closed_q, dtype=np.float64)
    closing_direction = np.sign(closed_arr - open_arr)
    ctrl_limits = np.max(
        np.abs(model.actuator_ctrlrange[actuator_idx]),
        axis=1,
    )

    hand_qpos_adrs: set[int] = set()
    hand_dof_adrs: set[int] = set()
    for joint_id in range(int(model.njnt)):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )
        if name and any(
            token in name
            for token in ("Thumb_", "Index_", "Middle_", "Ring_", "Little_")
        ):
            hand_qpos_adrs.add(int(model.jnt_qposadr[joint_id]))
            hand_dof_adrs.add(int(model.jnt_dofadr[joint_id]))

    data.qpos[qpos_idx] = open_arr
    mujoco.mj_forward(model, data)
    hand_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "Left_Hand",
    )
    hand_rotation = np.asarray(
        data.xmat[hand_body_id],
        dtype=np.float64,
    ).reshape(3, 3).copy()
    hand_position = np.asarray(data.xpos[hand_body_id], dtype=np.float64).copy()
    cube_initial_position = (
        hand_position
        + hand_rotation @ np.asarray((0.008, 0.038, 0.105), dtype=np.float64)
    )

    cube_joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "exp_task1_cube_2_joint",
    )
    cube_qpos_adr = int(model.jnt_qposadr[cube_joint_id])
    cube_dof_adr = int(model.jnt_dofadr[cube_joint_id])
    data.qpos[cube_qpos_adr : cube_qpos_adr + 3] = cube_initial_position
    data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7] = data.xquat[hand_body_id]
    mujoco.mj_forward(model, data)

    initial_qpos = data.qpos.copy()
    fixed_qpos = np.asarray(
        [
            index
            for index in range(int(model.nq))
            if index not in hand_qpos_adrs
            and not cube_qpos_adr <= index < cube_qpos_adr + 7
        ],
        dtype=np.int32,
    )
    fixed_dofs = np.asarray(
        [
            index
            for index in range(int(model.nv))
            if index not in hand_dof_adrs
            and not cube_dof_adr <= index < cube_dof_adr + 6
        ],
        dtype=np.int32,
    )

    worker = MujocoSimulationWorker.__new__(MujocoSimulationWorker)
    worker.model = model
    worker.data = data
    worker._hand_closing_direction = closing_direction.astype(np.float32)
    worker._hand_collision_geom_to_actuator_index = (
        MujocoSimulationWorker._resolve_hand_collision_geom_to_actuator_index(
            model
        )
    )

    max_normal_force = 0.0
    max_penetration = 0.0
    max_object_speed = 0.0
    contact_steps = 0
    for step in range(2700):
        if step == 2100:
            model.opt.gravity[:] = (0.0, 0.0, -9.81)

        data.qpos[fixed_qpos] = initial_qpos[fixed_qpos]
        data.qvel[fixed_dofs] = 0.0
        mujoco.mj_forward(model, data)

        target_q = open_arr if step < 300 else closed_arr
        position_error = target_q - data.qpos[qpos_idx]
        ctrl = (
            HAND_SERVO_KP * position_error
            - HAND_SERVO_KD * data.qvel[dof_idx]
        )
        near = worker._hand_actuators_near_grasp_object()
        if np.any(near):
            contact_steps += 1
        ctrl = worker._limit_near_object_hand_closing_torque(
            ctrl,
            near,
            position_error,
        )

        for contact_id in range(int(data.ncon)):
            contact = data.contact[contact_id]
            if cube_geom_id not in (int(contact.geom1), int(contact.geom2)):
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_id, force)
            max_normal_force = max(max_normal_force, abs(float(force[0])))
            max_penetration = max(max_penetration, -float(contact.dist))

        data.ctrl[:] = 0.0
        data.ctrl[actuator_idx] = np.clip(ctrl, -ctrl_limits, ctrl_limits)
        mujoco.mj_step(model, data)
        max_object_speed = max(
            max_object_speed,
            float(np.linalg.norm(data.qvel[cube_dof_adr : cube_dof_adr + 3])),
        )

    cube_displacement_hand = hand_rotation.T @ (
        data.qpos[cube_qpos_adr : cube_qpos_adr + 3]
        - cube_initial_position
    )
    final_normalized = (
        data.qpos[qpos_idx] - open_arr
    ) / (closed_arr - open_arr)

    assert contact_steps > 1000
    assert max_object_speed < 0.5
    # MuJoCo 3.3 and 3.8 differ for peak impulse and penetration.
    assert max_normal_force < 300.0
    assert max_penetration < 0.006
    assert float(np.linalg.norm(cube_displacement_hand)) < 0.015
    assert abs(float(cube_displacement_hand[2])) < 0.010
    assert int(np.count_nonzero(final_normalized[6:] > 0.25)) >= 4


def test_experiment_visibility_uses_static_and_free_object_masks() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    MujocoSimulationWorker._set_experiment_geom_visibility(model, 2)

    table_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "exp_task2_table_top",
    )
    cube_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "exp_task2_cube_1_geom",
    )
    assert int(model.geom_contype[table_id]) == EXPERIMENT_SUPPORT_COLLISION_CONTYPE
    assert (
        int(model.geom_conaffinity[table_id])
        == EXPERIMENT_SUPPORT_COLLISION_CONAFFINITY
    )
    assert int(model.geom_contype[cube_id]) == EXPERIMENT_GRASP_COLLISION_CONTYPE
    assert (
        int(model.geom_conaffinity[cube_id])
        == EXPERIMENT_GRASP_COLLISION_CONAFFINITY
    )
