from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
from geometry_msgs.msg import Pose, PoseArray
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions

from openxr_hand_to_igris_viewer.hand_fusion import ReliabilityAwareHandCommandFusion
from openxr_hand_to_igris_viewer.hybrid_openxr_mediapipe_pose_bridge import (
    FINGER_NAMES,
    HybridOpenXRMediaPipePoseBridge,
)


def _pose(close: float, *, axis: str) -> PoseArray:
    msg = PoseArray()
    msg.poses = [Pose() for _ in range(6)]
    msg.header.frame_id = "openxr" if axis == "x" else "mediapipe"
    for pose in msg.poses[1:]:
        setattr(pose.position, axis, 1.0 - close)
    return msg


def _bridge(*, openxr_only: bool) -> HybridOpenXRMediaPipePoseBridge:
    # Exercise the actual node's timer callback with in-memory publishers. No
    # ROS context, DDS participant, robot service or motor command is created.
    node = object.__new__(HybridOpenXRMediaPipePoseBridge)
    node.openxr_only = openxr_only
    node.min_pose_count = 6
    node.pose_stale_sec = 0.2
    node.tracked_stale_sec = 0.5
    node.confidence_stale_sec = 0.5
    node.default_confidence = 0.0
    node.confidence_min = 0.0
    node.confidence_max = 1.0
    node.confidence = None
    node.confidence_time = None
    node.openxr_pose = _pose(0.2, axis="x")
    node.mediapipe_pose = _pose(0.9, axis="y")
    node.openxr_pose_time = node.mediapipe_pose_time = 1.0
    node.openxr_tracked = node.mediapipe_tracked = True
    node.openxr_tracked_time = node.mediapipe_tracked_time = 1.0
    calibration = {
        f"{finger}_{bound}": value
        for finger in FINGER_NAMES
        for bound, value in (("min", 0.0), ("max", 1.0))
    }
    node.openxr_calib = node.mediapipe_calib = node.output_calib = calibration
    node.command_fusion = ReliabilityAwareHandCommandFusion(
        openxr_only=openxr_only,
        close_rate_per_sec=1000.0,
        open_rate_per_sec=1000.0,
    )
    node.now_sec = lambda: 1.0
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: PoseArray().header.stamp)
    )
    node.get_logger = lambda: SimpleNamespace(info=lambda _: None)
    node.last_source = None
    node.last_output_pose = None
    node.debug_log = False
    node.published = {"pose": [], "normalized": [], "tracked": []}
    for name, messages in node.published.items():
        setattr(node, f"{name}_pub", SimpleNamespace(publish=messages.append))
    return node


def test_always_one_bridge_uses_vr_for_commands_and_pose_without_confidence_messages() -> None:
    node = _bridge(openxr_only=True)
    node.publish_hybrid()
    assert node.published["normalized"][-1].data == pytest.approx([0.2] * 5)
    assert node.published["tracked"][-1].data is True
    output = node.published["pose"][-1]
    assert output.header.frame_id == "openxr"
    for pose in output.poses[1:]:
        assert pose.position.x == pytest.approx(0.8)
        assert pose.position.y == 0.0

    node.openxr_tracked = False
    node.publish_hybrid()
    assert node.published["normalized"][-1].data == pytest.approx([0.2] * 5)
    assert node.published["tracked"][-1].data is False
    assert node.published["pose"][-1].header.frame_id == "openxr"
    assert node.last_source == "hold_openxr_missing"


def test_always_one_bridge_rejects_stale_or_nonfinite_vr_with_valid_mediapipe() -> None:
    node = _bridge(openxr_only=True)
    node.publish_hybrid()
    node.openxr_pose_time = 0.0
    node.publish_hybrid()
    assert node.published["normalized"][-1].data == pytest.approx([0.2] * 5)
    assert node.published["tracked"][-1].data is False

    node.openxr_pose_time = 1.0
    node.openxr_pose.poses[1].position.x = float("nan")
    node.publish_hybrid()
    assert node.published["normalized"][-1].data == pytest.approx([0.2] * 5)
    assert node.published["tracked"][-1].data is False
    # Invalid tracking must not leak NaN directions into the compatibility
    # PoseArray output while the normalized command correctly holds.
    for pose in node.published["pose"][-1].poses[1:]:
        assert pose.position.x == pytest.approx(0.8)
        assert pose.position.y == 0.0


def test_normal_hybrid_bridge_keeps_calibrated_camera_fallback() -> None:
    node = _bridge(openxr_only=False)
    node.openxr_tracked = False
    node.publish_hybrid()
    assert node.published["normalized"][-1].data == pytest.approx([0.9] * 5)
    assert node.published["tracked"][-1].data is True
    assert node.published["pose"][-1].header.frame_id == "mediapipe"


def test_openxr_only_startup_without_vr_does_not_publish_camera_pose_or_activate_command() -> None:
    node = _bridge(openxr_only=True)
    node.openxr_pose = None
    node.openxr_pose_time = None
    node.openxr_tracked = False
    node.publish_hybrid()
    assert list(node.published["normalized"][-1].data) == [0.0] * 5
    assert node.published["tracked"][-1].data is False
    assert node.published["pose"] == []
    assert node.last_source == "hold_openxr_missing"


@pytest.mark.parametrize("variant,expected", [("always_1", "true"), ("rnn", "false"), ("histgb", "false")])
def test_reliability_launch_forwards_strict_mode_to_both_hand_nodes(
    variant: str, expected: str, monkeypatch
) -> None:
    src = Path(__file__).resolve().parents[2]

    def load_launch(path: Path):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    runtime = load_launch(src / "igris_reliability_runtime/launch/reliability_teleop.launch.py")
    bridge = load_launch(src / "openxr_hand_to_igris_viewer/launch/hybrid_hand_pose_bridge.launch.py")
    context = LaunchContext()
    context.launch_configurations["hand_model_variant"] = variant
    definitions = []
    real_node = runtime.Node
    def record_node(**kwargs):
        definitions.append(kwargs)
        return real_node(**kwargs)
    monkeypatch.setattr(runtime, "Node", record_node)
    actions = runtime.generate_launch_description().entities
    fusion = next(item for item in definitions if item["executable"] == "hand_retarget_fusion")
    resolved_bool = fusion["parameters"][0]["openxr_only"].evaluate(context)
    assert resolved_bool is (expected == "true")
    # The distance-based legacy bridge must not run alongside motor retargeting.
    assert len([a for a in actions if isinstance(a, IncludeLaunchDescription)]) == 1
    resolved = "true" if resolved_bool else "false"

    # Evaluate both included node definitions without starting any processes.
    context.launch_configurations["openxr_only"] = resolved
    context.launch_configurations["hand_side"] = "both"
    for action in bridge.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    monkeypatch.setattr(bridge, "Node", lambda **kwargs: kwargs)
    definitions = bridge.launch_setup(context)
    assert len(definitions) == 2
    for definition in definitions:
        assert definition["parameters"][0]["openxr_only"] is (expected == "true")
