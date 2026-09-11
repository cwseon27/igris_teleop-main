from __future__ import annotations

from collections.abc import Mapping


def should_accept_pose_source(
    *,
    preferred_topic: str,
    incoming_topic: str,
    topic_last_seen: Mapping[str, float],
    now: float,
    stale_after_sec: float,
) -> bool:
    """Prefer the configured pose topic while allowing a stale-source fallback."""
    if incoming_topic == preferred_topic:
        return True

    preferred_seen = topic_last_seen.get(preferred_topic)
    if preferred_seen is None:
        return True
    return float(now) - float(preferred_seen) > max(0.0, float(stale_after_sec))
