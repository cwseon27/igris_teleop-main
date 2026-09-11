"""Tests for leader timing statistics."""

from igris_leader_control.timing import LeaderTimingWindow
from igris_leader_control.timing import percentile
from igris_leader_control.timing import summarize_ns

import pytest


def test_percentile_interpolates_and_rejects_invalid_percent():
    """Percentiles interpolate deterministically and validate input."""
    assert percentile([], 50.0) is None
    assert percentile([0, 10, 20, 30], 50.0) == pytest.approx(15.0)
    assert percentile([0, 10, 20, 30], 95.0) == pytest.approx(28.5)

    with pytest.raises(ValueError):
        percentile([1], 100.1)


def test_summarize_ns_reports_milliseconds():
    """Nanosecond samples are exposed in human-readable milliseconds."""
    summary = summarize_ns([1_000_000, 2_000_000, 3_000_000])

    assert summary == {
        'count': 3,
        'p50_ms': 2.0,
        'p95_ms': pytest.approx(2.9),
        'p99_ms': pytest.approx(2.98),
        'max_ms': 3.0,
    }


def test_timing_window_tracks_drops_and_resets():
    """A bounded window tracks publication outcomes and can be reset."""
    window = LeaderTimingWindow(max_samples=2)
    window.observe_ns('loop', 1_000_000)
    window.observe_ns('loop', 2_000_000)
    window.observe_ns('loop', 3_000_000)
    window.observe_ns('txrx', 500_000)
    window.observe_ns('callback', 800_000)
    window.record_attempt(published=True)
    window.record_attempt(published=False)

    snapshot = window.snapshot(reset=True)

    assert snapshot['attempts'] == 2
    assert snapshot['published'] == 1
    assert snapshot['dropped'] == 1
    assert snapshot['metrics']['loop']['count'] == 2
    assert snapshot['metrics']['loop']['p50_ms'] == pytest.approx(2.5)
    assert window.snapshot()['attempts'] == 0


def test_timing_window_rejects_unknown_or_negative_observation():
    """Invalid metric names and durations are rejected."""
    window = LeaderTimingWindow()

    with pytest.raises(KeyError):
        window.observe_ns('unknown', 1)
    with pytest.raises(ValueError):
        window.observe_ns('loop', -1)
