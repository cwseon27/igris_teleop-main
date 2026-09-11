from __future__ import annotations

from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from igris_teleop.sim.experiment_scenes import (
    EXPERIMENT_TASK_IDS,
    experiment_task_id_from_name,
    experiment_tasks_payload,
)


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "igris_teleop"
    / "sim"
    / "robot"
    / "mujoco"
    / "igris_c_v2_with_hand.xml"
)
GRASP_OBJECT_GEOM_NAMES = {
    "exp_task1_cube_2_geom",
    "exp_task1_cube_3_geom",
    "exp_task2_cube_1_geom",
    "exp_task2_cube_2_geom",
    "exp_task3_large_box_geom",
    "exp_task4_peg_geom",
}


def _model_elements() -> tuple[ET.Element, dict[str, ET.Element]]:
    root = ET.parse(MODEL_PATH).getroot()
    elements = {
        str(element.get("name")): element
        for element in root.findall(".//*[@name]")
    }
    return root, elements


def _values(element: ET.Element, attribute: str) -> np.ndarray:
    return np.fromstring(str(element.get(attribute)), sep=" ")


def test_experiment_catalog_contains_four_tasks() -> None:
    payload = experiment_tasks_payload()

    assert EXPERIMENT_TASK_IDS == {1, 2, 3, 4}
    assert [task["task_id"] for task in payload] == [1, 2, 3, 4]
    assert payload[1]["key"] == "cube_stacking"
    assert payload[1]["goal"] == "Pick up one cube and stack it on the other cube"
    assert payload[0]["objects"].startswith("Two 50")
    assert "mass 300 g" in payload[2]["objects"]
    assert experiment_task_id_from_name("exp_task3_large_box_geom") == 3
    assert experiment_task_id_from_name("table_top") is None


def test_experiment_model_contains_requested_workspace_dimensions() -> None:
    _root, elements = _model_elements()

    np.testing.assert_allclose(
        _values(elements["exp_task1_table_top"], "size"),
        [0.30, 0.80, 0.025],
    )
    np.testing.assert_allclose(
        _values(elements["exp_task3_left_table_top"], "size"),
        [0.30, 0.30, 0.025],
    )
    np.testing.assert_allclose(
        _values(elements["exp_task3_large_box_geom"], "size"),
        [0.15, 0.20, 0.15],
    )
    assert float(elements["exp_task3_large_box_geom"].get("mass", "0")) == 0.30
    assert float(_values(elements["exp_task3_large_box_geom"], "friction")[0]) >= 2.0
    assert elements["exp_task3_large_box_geom"].get("priority") == "2"

    grasp_default = next(
        element
        for element in _root.findall("./default/default")
        if element.get("class") == "grasp_object"
    ).find("geom")
    assert grasp_default is not None
    assert grasp_default.get("priority") == "2"
    assert grasp_default.get("condim") == "4"
    assert float(_values(grasp_default, "friction")[0]) == 1.0
    np.testing.assert_allclose(_values(grasp_default, "solref"), [0.007, 1.5])


def test_experiment_model_has_resettable_objects_for_every_task() -> None:
    root, elements = _model_elements()
    free_joints = [
        joint
        for joint in root.findall(".//freejoint")
        if experiment_task_id_from_name(joint.get("name")) is not None
    ]
    counts = Counter(
        experiment_task_id_from_name(joint.get("name"))
        for joint in free_joints
    )

    assert counts == {1: 2, 2: 2, 3: 1, 4: 1}
    for name in (
        "exp_task1_cube_2_geom",
        "exp_task1_cube_3_geom",
    ):
        np.testing.assert_allclose(_values(elements[name], "size"), [0.025, 0.025, 0.025])
    for name in (
        "exp_task2_cube_1_geom",
        "exp_task2_cube_2_geom",
    ):
        assert elements[name].get("type") == "box"
        np.testing.assert_allclose(_values(elements[name], "size"), [0.025, 0.025, 0.025])


def test_experiment_workspaces_are_close_and_objects_start_above_table() -> None:
    _root, elements = _model_elements()
    table_top_z = 0.8

    for task_id in (1, 2, 4):
        table = elements[f"exp_task{task_id}_table_top"]
        table_pos = _values(table, "pos")
        table_size = _values(table, "size")
        assert np.isclose(table_pos[0] - table_size[0], 0.17)

    tray_pos = _values(elements["exp_task1_tray_base"], "pos")
    assert abs(float(tray_pos[1])) <= 0.30
    task1_x_positions = []
    for index in (2, 3):
        body_pos = _values(elements[f"exp_task1_cube_{index}"], "pos")
        task1_x_positions.append(float(body_pos[0]))
        assert abs(float(body_pos[1])) <= 0.30
        assert body_pos[2] - 0.025 >= table_top_z + 0.009
    np.testing.assert_allclose(task1_x_positions, [0.42, 0.53])
    assert "exp_task1_cube_1" not in elements

    task2_positions = np.asarray(
        [
            _values(elements["exp_task2_cube_1"], "pos"),
            _values(elements["exp_task2_cube_2"], "pos"),
        ]
    )
    np.testing.assert_allclose(task2_positions[:, 0], [0.42, 0.42])
    np.testing.assert_allclose(task2_positions[:, 1], [0.08, -0.08])
    assert np.all(task2_positions[:, 2] - 0.025 >= table_top_z + 0.009)
    assert "exp_task2_tray_base" not in elements

    left_table_pos = _values(elements["exp_task3_left_table"], "pos")
    right_table_pos = _values(elements["exp_task3_right_table"], "pos")
    np.testing.assert_allclose(left_table_pos[:2], [0.40, 0.40])
    np.testing.assert_allclose(right_table_pos[:2], [0.40, -0.40])
    large_box_pos = _values(elements["exp_task3_large_box"], "pos")
    assert large_box_pos[2] - 0.15 >= table_top_z + 0.009

    peg_pos = _values(elements["exp_task4_peg"], "pos")
    hole_pos = _values(elements["exp_task4_hole_marker"], "pos")
    assert abs(float(peg_pos[1])) <= 0.11
    assert abs(float(hole_pos[1])) <= 0.11
    assert np.isclose(peg_pos[0], 0.42)
    assert np.isclose(hole_pos[0], 0.42)
    assert np.isclose(hole_pos[1], -0.02)
    np.testing.assert_allclose(
        [
            _values(elements["exp_task4_socket_side_left"], "pos")[0],
            _values(elements["exp_task4_socket_side_right"], "pos")[0],
            np.mean(
                [
                    _values(elements["exp_task4_socket_side_front"], "pos")[0],
                    _values(elements["exp_task4_socket_side_back"], "pos")[0],
                ]
            ),
        ],
        [0.42, 0.42, 0.42],
    )
    assert peg_pos[2] - 0.06 >= table_top_z + 0.009


def test_experiment_geometry_starts_hidden_with_collision_candidates_compiled() -> None:
    root, _elements = _model_elements()
    experiment_geoms = [
        geom
        for geom in root.findall(".//geom")
        if experiment_task_id_from_name(geom.get("name")) is not None
    ]

    assert experiment_geoms
    assert all(float(_values(geom, "rgba")[3]) == 0.0 for geom in experiment_geoms)
    for geom in experiment_geoms:
        name = str(geom.get("name"))
        if name.endswith("_marker"):
            assert geom.get("contype") == "0"
            assert geom.get("conaffinity") == "0"
        elif name in GRASP_OBJECT_GEOM_NAMES:
            assert geom.get("class") == "grasp_object"
            assert geom.get("contype") == "8"
            assert geom.get("conaffinity") == "14"
        else:
            assert geom.get("contype") == "4"
            assert geom.get("conaffinity") == "7"


def test_hand_collision_primitives_match_visible_contact_surface() -> None:
    root, _elements = _model_elements()
    hand_geoms = [
        geom
        for geom in root.findall(".//geom")
        if geom.get("contype") == "2"
    ]

    assert len(hand_geoms) == 26
    assert Counter(geom.get("type") for geom in hand_geoms) == {"box": 26}
    for geom in hand_geoms:
        assert geom.get("conaffinity") == "12"
        assert float(geom.get("margin", "0")) == 0.0
        assert float(geom.get("gap", "0")) == 0.0
