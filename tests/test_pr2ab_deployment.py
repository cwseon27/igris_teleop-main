"""Deployment checks for the real controller's calibration, without robot I/O.

Keep these tests runnable in a clean source export: the operational calibration
must be in the repository, not supplied by ignored logs or a ROS installation.
"""

import hashlib
from pathlib import Path
from unittest.mock import Mock

import igris_c_sdk as igc_sdk
import numpy as np
import pytest
import yaml

from igris_teleop.robot_control.controller.igris_controller import BaseController
from igris_teleop.robot_control.kinematics.joints import NUM_MOTORS
from igris_teleop.robot_control.kinematics.pr2ab import default_pr2ab_pairs
from igris_teleop.workers.worker_control import DEFAULT_PR2AB_LOG_PATH


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_RELATIVE_PATH = Path("igris_artifacts/logs/robot_control/pr2ab_calibration.yaml")
PAIR_NAMES = ("waist_rp", "l_ankle_pr", "r_ankle_pr", "l_wrist_rp", "r_wrist_rp")


def _offline_ms_controller(path: Path) -> BaseController:
    # Never call the constructor: it initializes hardware-facing resources.
    controller = BaseController.__new__(BaseController)
    controller._kinematic_mode = igc_sdk.KinematicMode.MS
    controller._pr2ab_config_path = path
    controller._pr2ab_pairs = default_pr2ab_pairs()
    controller.load_pr2ab_transforms_from_yaml(path)
    controller.wait_for_state = Mock(return_value=True)
    controller.get_motor_q = Mock()
    return controller


@pytest.fixture
def calibration_payload():
    with DEFAULT_PR2AB_LOG_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_real_worker_calibration_is_bundled_at_its_actual_runtime_path():
    assert DEFAULT_PR2AB_LOG_PATH == REPO_ROOT / CALIBRATION_RELATIVE_PATH
    assert DEFAULT_PR2AB_LOG_PATH.is_file(), (
        "The real MS controller requires its operational calibration in the "
        "source checkout, including ab_limits; ignored local logs are not sufficient."
    )


@pytest.mark.parametrize("name", PAIR_NAMES)
def test_bundled_pairs_are_complete_and_loaded_by_actual_controller(name, calibration_payload):
    items = calibration_payload["pairs"]
    assert set(items) == set(PAIR_NAMES)
    raw = items[name]
    controller = _offline_ms_controller(DEFAULT_PR2AB_LOG_PATH)
    pair = controller.get_pr2ab_pair(name)
    assert pair.configured()

    for field, shape in (("M", (2, 2)), ("pr_center", (2,)), ("ab_center", (2,)), ("ab_limits", (2, 2))):
        expected = np.asarray(raw[field], dtype=np.float64)
        loaded = np.asarray(getattr(pair, field), dtype=np.float64)
        assert expected.shape == shape
        assert loaded.shape == shape
        assert np.isfinite(expected).all()
        np.testing.assert_array_equal(loaded, expected)

    assert np.linalg.matrix_rank(pair.M) == 2
    np.testing.assert_allclose(pair.M @ np.linalg.inv(pair.M), np.eye(2), atol=1e-12)
    limits = np.asarray(pair.ab_limits)
    assert np.all(limits[:, 0] < limits[:, 1])
    controller.wait_for_state.assert_not_called()
    controller.get_motor_q.assert_not_called()


def test_operational_calibration_is_covered_by_runtime_asset_manifest():
    manifest = (REPO_ROOT / "docs/runtime_assets.sha256").read_text(encoding="utf-8")
    entries = {}
    for line in manifest.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        digest, relative_path = line.split(maxsplit=1)
        entries[relative_path.lstrip("*")] = digest

    expected_digest = entries[CALIBRATION_RELATIVE_PATH.as_posix()]
    assert hashlib.sha256(DEFAULT_PR2AB_LOG_PATH.read_bytes()).hexdigest() == expected_digest


def test_deployed_calibration_passes_ms_start_guard_with_stubbed_in_limit_state():
    controller = _offline_ms_controller(DEFAULT_PR2AB_LOG_PATH)
    synthetic_ms = np.zeros(NUM_MOTORS, dtype=np.float64)
    for pair in controller._pr2ab_pairs.values():
        # Midpoints are only synthetic test feedback, never robot commands.
        synthetic_ms[[pair.ab_i1, pair.ab_i2]] = np.asarray(pair.ab_limits).mean(axis=1)
    controller.get_motor_q.return_value = synthetic_ms

    assert controller.get_unconfigured_pr2ab_pairs() == []
    assert controller.get_pr2ab_pairs_missing_limits() == []
    controller.ensure_ms_start_guard(state_timeout=0.125)

    controller.wait_for_state.assert_called_once_with(timeout=0.125)
    controller.get_motor_q.assert_called_once_with()


def test_ms_guard_still_blocks_missing_calibration_before_waiting_for_state(tmp_path):
    controller = _offline_ms_controller(tmp_path / "missing_calibration.yaml")

    with pytest.raises(RuntimeError, match="PR2AB calibration file not found"):
        controller.ensure_ms_start_guard(state_timeout=0.125)

    controller.wait_for_state.assert_not_called()
    controller.get_motor_q.assert_not_called()


def test_ms_guard_still_blocks_incomplete_calibration_before_waiting_for_state(tmp_path, calibration_payload):
    for item in calibration_payload["pairs"].values():
        item.pop("ab_limits")
    incomplete_path = tmp_path / "calibration_without_ab_limits.yaml"
    incomplete_path.write_text(yaml.safe_dump(calibration_payload), encoding="utf-8")
    controller = _offline_ms_controller(incomplete_path)
    assert controller.get_unconfigured_pr2ab_pairs() == []
    assert controller.get_pr2ab_pairs_missing_limits() == sorted(PAIR_NAMES)

    with pytest.raises(RuntimeError, match="PR2AB pairs are missing ab_limits"):
        controller.ensure_ms_start_guard(state_timeout=0.125)

    controller.wait_for_state.assert_not_called()
    controller.get_motor_q.assert_not_called()
