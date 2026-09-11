from __future__ import annotations

import pytest

from openxr_hand_to_igris_viewer.hand_fusion import ReliabilityAwareHandCommandFusion


def _fusion(**overrides) -> ReliabilityAwareHandCommandFusion:
    values = {
        "previous_command_weight": 0.0,
        "close_rate_per_sec": 1000.0,
        "open_rate_per_sec": 1000.0,
        "nominal_rate_hz": 30.0,
    }
    values.update(overrides)
    return ReliabilityAwareHandCommandFusion(**values)


def test_both_sources_blend_in_normalized_close_space() -> None:
    result = _fusion().update(
        openxr_close=[1.0] * 5,
        mediapipe_close=[0.0] * 5,
        confidence=0.25,
        openxr_ready=True,
        mediapipe_ready=True,
        now=1.0,
    )
    assert result.source == "hybrid"
    assert result.tracked is True
    assert result.command == pytest.approx([0.25] * 5)


def test_gamma_can_make_mediapipe_dominate_earlier() -> None:
    result = _fusion(confidence_gamma=2.0).update(
        openxr_close=[1.0] * 5,
        mediapipe_close=[0.0] * 5,
        confidence=0.5,
        openxr_ready=True,
        mediapipe_ready=True,
        now=1.0,
    )
    assert result.theta == pytest.approx(0.25)
    assert result.command == pytest.approx([0.25] * 5)


def test_unreliable_openxr_only_and_missing_sources_hold_previous() -> None:
    fusion = _fusion(openxr_only_confidence_threshold=0.6)
    seeded = fusion.update(
        openxr_close=[0.4] * 5,
        mediapipe_close=None,
        confidence=1.0,
        openxr_ready=True,
        mediapipe_ready=False,
        now=1.0,
    )
    unreliable = fusion.update(
        openxr_close=[1.0] * 5,
        mediapipe_close=None,
        confidence=0.2,
        openxr_ready=True,
        mediapipe_ready=False,
        now=1.1,
    )
    missing = fusion.update(
        openxr_close=None,
        mediapipe_close=None,
        confidence=1.0,
        openxr_ready=False,
        mediapipe_ready=False,
        now=1.2,
    )

    assert seeded.command == pytest.approx([0.4] * 5)
    assert unreliable.command == seeded.command
    assert unreliable.source == "hold_openxr_unreliable"
    assert unreliable.tracked is False
    assert missing.command == seeded.command
    assert missing.source == "hold_no_source"
    assert missing.tracked is False


def test_media_pipe_only_is_valid_fallback() -> None:
    result = _fusion().update(
        openxr_close=None,
        mediapipe_close=[0.1, 0.2, 0.3, 0.4, 0.5],
        confidence=0.0,
        openxr_ready=False,
        mediapipe_ready=True,
        now=1.0,
    )
    assert result.source == "mediapipe_only"
    assert result.tracked is True
    assert result.command == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])


@pytest.mark.parametrize("confidence", [0.0, 0.2, 1.0, float("nan")])
def test_openxr_only_ignores_mediapipe_and_confidence_stream(confidence: float) -> None:
    result = _fusion(openxr_only=True).update(
        openxr_close=[0.1, 0.2, 0.3, 0.4, 0.5],
        mediapipe_close=[1.0] * 5,
        confidence=confidence,
        openxr_ready=True,
        mediapipe_ready=True,
        now=1.0,
    )
    assert result.command == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result.source == "openxr_only"
    assert result.tracked is True
    assert result.confidence == 1.0
    assert result.theta == 1.0


@pytest.mark.parametrize("invalid_openxr", [None, [float("nan")] * 5, [0.7] * 4])
def test_openxr_only_holds_when_vr_missing_or_invalid_despite_valid_camera(invalid_openxr) -> None:
    fusion = _fusion(openxr_only=True)
    seeded = fusion.update(
        openxr_close=[0.3] * 5,
        mediapipe_close=[1.0] * 5,
        confidence=1.0,
        openxr_ready=True,
        mediapipe_ready=True,
        now=1.0,
    )
    missing = fusion.update(
        openxr_close=invalid_openxr,
        mediapipe_close=[1.0] * 5,
        confidence=0.0,
        openxr_ready=True,
        mediapipe_ready=True,
        now=2.0,
    )
    assert missing.command == seeded.command
    assert missing.source == "hold_openxr_missing"
    assert missing.tracked is False
    assert fusion.last_mediapipe is None


def test_openxr_only_tracking_loss_and_recovery_preserve_rate_limit() -> None:
    fusion = _fusion(openxr_only=True, close_rate_per_sec=1.0, nominal_rate_hz=10.0)
    seeded = fusion.update(
        openxr_close=[0.1] * 5,
        mediapipe_close=[0.9] * 5,
        confidence=1.0,
        openxr_ready=True,
        mediapipe_ready=True,
        now=1.0,
    )
    lost = fusion.update(
        openxr_close=[1.0] * 5,
        mediapipe_close=[0.9] * 5,
        confidence=1.0,
        openxr_ready=False,
        mediapipe_ready=True,
        now=1.1,
    )
    recovered = fusion.update(
        openxr_close=[1.0] * 5,
        mediapipe_close=[0.9] * 5,
        confidence=0.0,
        openxr_ready=True,
        mediapipe_ready=True,
        now=1.2,
    )
    assert lost.command == seeded.command
    assert lost.tracked is False
    assert recovered.command == pytest.approx([0.2] * 5)
    assert recovered.tracked is True


def test_high_inferred_confidence_still_allows_normal_hybrid_dropout_fallback() -> None:
    result = _fusion().update(
        openxr_close=None,
        mediapipe_close=[0.4] * 5,
        confidence=1.0,
        openxr_ready=False,
        mediapipe_ready=True,
        now=1.0,
    )
    assert result.command == pytest.approx([0.4] * 5)
    assert result.source == "mediapipe_only"
    assert result.tracked is True


def test_short_mediapipe_dropout_reuses_last_valid_command() -> None:
    fusion = _fusion(source_dropout_grace_sec=0.15)
    seeded = fusion.update(
        openxr_close=[1.0] * 5,
        mediapipe_close=[0.0] * 5,
        confidence=0.0,
        openxr_ready=True,
        mediapipe_ready=True,
        now=1.0,
    )
    dropout = fusion.update(
        openxr_close=[1.0] * 5,
        mediapipe_close=None,
        confidence=0.0,
        openxr_ready=True,
        mediapipe_ready=False,
        now=1.1,
    )
    expired = fusion.update(
        openxr_close=[1.0] * 5,
        mediapipe_close=None,
        confidence=0.0,
        openxr_ready=True,
        mediapipe_ready=False,
        now=1.2,
    )

    assert seeded.command == pytest.approx([0.0] * 5)
    assert dropout.source == "hybrid"
    assert dropout.tracked is True
    assert dropout.command == seeded.command
    assert expired.source == "hold_openxr_unreliable"
    assert expired.tracked is False


def test_default_rate_limits_reach_full_motion_without_slow_lag() -> None:
    fusion = ReliabilityAwareHandCommandFusion(nominal_rate_hz=30.0)
    first = fusion.update(
        openxr_close=[1.0] * 5,
        mediapipe_close=None,
        confidence=1.0,
        openxr_ready=True,
        mediapipe_ready=False,
        now=1.0,
    )
    second = fusion.update(
        openxr_close=[1.0] * 5,
        mediapipe_close=None,
        confidence=1.0,
        openxr_ready=True,
        mediapipe_ready=False,
        now=1.0 + 1.0 / 30.0,
    )

    assert first.command == pytest.approx([0.5] * 5)
    assert second.command == pytest.approx([1.0] * 5)


def test_close_and_open_changes_are_rate_limited() -> None:
    fusion = _fusion(close_rate_per_sec=1.0, open_rate_per_sec=2.0, nominal_rate_hz=10.0)
    closing = fusion.update(
        openxr_close=[1.0] * 5,
        mediapipe_close=None,
        confidence=1.0,
        openxr_ready=True,
        mediapipe_ready=False,
        now=1.0,
    )
    opening = fusion.update(
        openxr_close=[0.0] * 5,
        mediapipe_close=None,
        confidence=1.0,
        openxr_ready=True,
        mediapipe_ready=False,
        now=1.1,
    )
    assert closing.command == pytest.approx([0.1] * 5)
    assert opening.command == pytest.approx([0.0] * 5)
