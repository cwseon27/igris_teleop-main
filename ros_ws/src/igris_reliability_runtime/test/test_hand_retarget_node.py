"""Node callback tests without starting ROS participants or robot processes."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("rclpy")
from geometry_msgs.msg import Pose, PoseArray
from igris_reliability_runtime.hand_retarget_node import ReliabilityHandRetargetNode


def node_fixture():
    node = object.__new__(ReliabilityHandRetargetNode)
    node.options = {"pose_stale_sec": .2, "tracked_stale_sec": .5, "confidence_stale_sec": .5}
    node.poses = {}
    node.tracking = {}
    node.confidence = {}
    node.last_source = {}
    node.last_warning = 0.
    node.get_logger = lambda: SimpleNamespace(info=lambda _: None, warning=lambda _: None)
    return node


@pytest.mark.parametrize("source,count", [("openxr", 6), ("mediapipe", 21)])
def test_correct_pose_contract_and_stale_tracking_gate(source, count):
    node = node_fixture()
    key = ("left", source)
    msg = PoseArray(poses=[Pose() for _ in range(count)])
    with patch("igris_reliability_runtime.hand_retarget_node.time.monotonic", return_value=10.):
        node.on_pose(key, msg)
    assert node.observation(key, 10.01) is None  # no tracking=true yet
    node.tracking[key] = (True, 10.)
    points, stamp = node.observation(key, 10.01)
    assert points.shape == ((5, 3) if source == "openxr" else (21, 3))
    assert stamp == 10.
    assert node.observation(key, 10.21) is None
    assert node.observation(key, 9.9) is None
    node.on_pose(key, PoseArray(poses=[Pose()]))
    assert key not in node.poses


@pytest.mark.parametrize("confidence,stamp,expected", [(np.nan, 10., 0.), (np.inf, 10., 0.), (.8, 9., 0.), (.8, 10., .8)])
def test_nonfinite_or_expired_confidence_never_becomes_full_trust(confidence, stamp, expected):
    node = node_fixture()
    calls = []
    result = SimpleNamespace(command=(.1, .2, .3, .4, .5, .9), tracked=True, source="hybrid")
    def update(**kwargs):
        calls.append(kwargs)
        return result
    node.engines = {"left": SimpleNamespace(update=update, last_error={})}
    commands, tracking, order = [], [], []
    node.publishers_by_side = {"left": (
        SimpleNamespace(publish=lambda msg: (commands.append(msg), order.append("command"))),
        SimpleNamespace(publish=lambda msg: (tracking.append(msg), order.append("tracked"))),
    )}
    node.confidence["left"] = (confidence, stamp)
    with patch("igris_reliability_runtime.hand_retarget_node.time.monotonic", return_value=10.01):
        node.publish_commands()
    assert calls[0]["confidence"] == expected
    assert list(commands[0].data) == pytest.approx(result.command)
    assert tracking[0].data
    assert order == ["tracked", "command"]
