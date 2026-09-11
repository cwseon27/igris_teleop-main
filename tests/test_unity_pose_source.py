from __future__ import annotations

from igris_teleop.teleop_devices.unity.pose_source import should_accept_pose_source
from igris_teleop.teleop_devices.unity.hand_source import hybrid_hand_pose_is_usable
from igris_teleop.core.worker_base import (
    allowed_teleop_hand_sources,
    resolve_teleop_hand_source,
    teleop_uses_hybrid_torso,
)


def test_hybrid_torso_is_enabled_for_both_hybrid_devices() -> None:
    assert teleop_uses_hybrid_torso("unity_hybrid")
    assert teleop_uses_hybrid_torso("vr_masterarm")
    assert not teleop_uses_hybrid_torso("unity")


def test_vr_masterarm_defaults_to_same_vr_hand_path_as_unity_hybrid() -> None:
    assert allowed_teleop_hand_sources("vr_masterarm") == ("vr", "masterarm")
    assert resolve_teleop_hand_source("vr_masterarm", None) == "vr"
    assert resolve_teleop_hand_source("vr_masterarm", "masterarm") == "masterarm"


def test_fresh_hybrid_hand_does_not_require_openxr_wrist_tracking() -> None:
    assert hybrid_hand_pose_is_usable(
        pose_received=True,
        pose_timestamp=9.9,
        tracked=True,
        tracked_timestamp=9.9,
        now=10.0,
        stale_after_sec=0.25,
    )


def test_hybrid_hand_rejects_untracked_or_stale_data() -> None:
    assert not hybrid_hand_pose_is_usable(
        pose_received=True,
        pose_timestamp=9.9,
        tracked=False,
        tracked_timestamp=9.9,
        now=10.0,
        stale_after_sec=0.25,
    )
    assert not hybrid_hand_pose_is_usable(
        pose_received=True,
        pose_timestamp=9.0,
        tracked=True,
        tracked_timestamp=9.9,
        now=10.0,
        stale_after_sec=0.25,
    )


def test_pose_only_hybrid_publisher_remains_compatible() -> None:
    assert hybrid_hand_pose_is_usable(
        pose_received=True,
        pose_timestamp=9.9,
        tracked=None,
        tracked_timestamp=None,
        now=10.0,
        stale_after_sec=0.25,
    )


def test_preferred_controller_pose_topic_wins_over_fresh_legacy_duplicate() -> None:
    assert not should_accept_pose_source(
        preferred_topic="/left_controller/poses",
        incoming_topic="/left_controller/pose",
        topic_last_seen={"/left_controller/poses": 10.0},
        now=10.05,
        stale_after_sec=0.2,
    )


def test_legacy_controller_pose_topic_is_used_when_preferred_is_stale() -> None:
    assert should_accept_pose_source(
        preferred_topic="/left_controller/poses",
        incoming_topic="/left_controller/pose",
        topic_last_seen={"/left_controller/poses": 10.0},
        now=10.21,
        stale_after_sec=0.2,
    )


def test_preferred_controller_pose_topic_is_always_accepted() -> None:
    assert should_accept_pose_source(
        preferred_topic="/right_controller/poses",
        incoming_topic="/right_controller/poses",
        topic_last_seen={},
        now=1.0,
        stale_after_sec=0.2,
    )
