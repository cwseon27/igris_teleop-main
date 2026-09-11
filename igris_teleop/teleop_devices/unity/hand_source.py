from __future__ import annotations


def hybrid_hand_pose_is_usable(
    *,
    pose_received: bool,
    pose_timestamp: float | None,
    tracked: bool | None,
    tracked_timestamp: float | None,
    now: float,
    stale_after_sec: float,
) -> bool:
    """Return whether a hybrid hand pose is fresh and currently tracked."""
    if not pose_received or pose_timestamp is None:
        return False

    stale_after_sec = max(0.0, float(stale_after_sec))
    if float(now) - float(pose_timestamp) > stale_after_sec:
        return False

    # Pose-only publishers predate the tracked topic, so retain compatibility
    # until a tracked stream has actually been observed.
    if tracked_timestamp is None:
        return True
    if float(now) - float(tracked_timestamp) > stale_after_sec:
        return False
    return bool(tracked)
