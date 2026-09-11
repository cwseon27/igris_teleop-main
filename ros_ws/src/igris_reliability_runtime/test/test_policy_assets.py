from __future__ import annotations

import numpy as np
import pytest

from igris_reliability_runtime.model_runtime import ALWAYS_ONE_VARIANT, load_policy_bundle


@pytest.mark.parametrize("kind,feature_dim", [("hand", 166), ("controller", 32)])
@pytest.mark.parametrize("variant", ["rnn", "histgb"])
def test_vendored_policy_loads_and_predicts(kind: str, feature_dim: int, variant: str) -> None:
    bundle, path, loaded_variant = load_policy_bundle(kind, variant, fallback_variant="")
    window = np.zeros(
        (1, int(bundle["window_size"]), feature_dim),
        dtype=np.float32,
    )

    left = np.asarray(bundle["left_model"].predict_confidence(window)).reshape(-1)
    right = np.asarray(bundle["right_model"].predict_confidence(window)).reshape(-1)

    assert path.is_file()
    assert loaded_variant == variant
    assert left.shape == (1,)
    assert right.shape == (1,)
    assert 0.0 <= float(left[0]) <= 1.0
    assert 0.0 <= float(right[0]) <= 1.0


@pytest.mark.parametrize("kind", ["hand", "controller"])
def test_always_one_policy_returns_one_for_every_batch_item(kind: str) -> None:
    bundle, path, loaded_variant = load_policy_bundle(kind, ALWAYS_ONE_VARIANT)
    window = np.zeros(
        (3, int(bundle["window_size"]), int(bundle["feature_dim"])),
        dtype=np.float32,
    )

    left = np.asarray(bundle["left_model"].predict_confidence(window)).reshape(-1)
    right = np.asarray(bundle["right_model"].predict_confidence(window)).reshape(-1)

    assert str(path) == "<built-in-always_1>"
    assert loaded_variant == ALWAYS_ONE_VARIANT
    np.testing.assert_array_equal(left, np.ones(3, dtype=np.float32))
    np.testing.assert_array_equal(right, np.ones(3, dtype=np.float32))
