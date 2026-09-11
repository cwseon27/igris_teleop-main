from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip('rclpy')
pytest.importorskip('mediapipe')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mediapipe_hand_pose_bridge.mediapipe_hand_pose_publisher import MediaPipeHandPosePublisher  # noqa: E402


@pytest.mark.parametrize('side, expected', [('Left', [('Left', False)]), ('Right', [('Right', False)]), ('Any', [('Right', False), ('Left', False)])])
def test_missing_camera_publishes_only_assigned_sides_untracked(side, expected):
    flags, previews = [], []
    node = SimpleNamespace(
        cap=SimpleNamespace(read=lambda: (False, None)),
        target_hand=side,
        publish_inactive_tracked=False,
        preview_last_write_t=float('-inf'),
        preview_write_period=0.1,
        _write_preview_status=lambda **status: previews.append(status),
        publish_tracked=lambda label, tracked: flags.append((label, tracked)),
        video='',
    )
    node.publish_untracked_for_active_labels = MethodType(
        MediaPipeHandPosePublisher.publish_untracked_for_active_labels, node,
    )
    MediaPipeHandPosePublisher.step(node)
    assert flags == expected
    assert previews == [{'camera_ok': False, 'detections': [], 'pose_by_label': {}}]


def test_loop_video_eof_also_invalidates_previous_tracking():
    flags, seeks = [], []
    node = SimpleNamespace(
        cap=SimpleNamespace(read=lambda: (False, None), set=lambda *args: seeks.append(args)),
        target_hand='Right',
        publish_inactive_tracked=False,
        preview_last_write_t=float('-inf'),
        preview_write_period=0.1,
        _write_preview_status=lambda **status: None,
        publish_tracked=lambda label, tracked: flags.append((label, tracked)),
        video='recording.mp4',
        loop_video=True,
    )
    node.publish_untracked_for_active_labels = MethodType(
        MediaPipeHandPosePublisher.publish_untracked_for_active_labels, node,
    )
    MediaPipeHandPosePublisher.step(node)
    assert flags == [('Right', False)]
    assert len(seeks) == 1
