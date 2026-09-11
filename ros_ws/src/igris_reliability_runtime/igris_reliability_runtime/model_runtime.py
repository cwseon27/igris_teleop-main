from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np


PACKAGE_NAME = "igris_reliability_runtime"
SUPPORTED_KINDS = ("hand", "controller")
TRAINED_VARIANTS = ("rnn", "histgb")
ALWAYS_ONE_VARIANT = "always_1"
SUPPORTED_VARIANTS = TRAINED_VARIANTS + (ALWAYS_ONE_VARIANT,)
POLICY_ROOT_ENV = "IGRIS_RELIABILITY_POLICY_ROOT"


class ConstantConfidenceModel:
    def __init__(self, value: float = 1.0):
        self.value = float(value)

    def predict_confidence(self, batch) -> np.ndarray:
        batch_array = np.asarray(batch)
        batch_size = int(batch_array.shape[0]) if batch_array.ndim > 0 else 1
        return np.full((batch_size,), self.value, dtype=np.float32)


def package_share_directory() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory(PACKAGE_NAME))
    except Exception:
        return Path(__file__).resolve().parents[1]


def policy_root_directory() -> Path:
    explicit_root = str(os.getenv(POLICY_ROOT_ENV) or "").strip()
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        candidate = parent / "policy_archive" / "reliability"
        if candidate.is_dir():
            return candidate.resolve()

    return package_share_directory() / "policies"


def policy_path(kind: str, variant: str, override: str = "") -> Path:
    if override:
        return Path(override).expanduser().resolve()
    kind = str(kind).strip().lower()
    variant = str(variant).strip().lower()
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"unsupported policy kind {kind!r}; expected one of {SUPPORTED_KINDS}")
    if variant not in TRAINED_VARIANTS:
        raise ValueError(
            f"policy variant {variant!r} has no model asset; expected one of {TRAINED_VARIANTS}"
        )
    return policy_root_directory() / kind / f"{kind}_{variant}.joblib"


def _validate_bundle(bundle: Any, path: Path) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        raise TypeError(f"policy bundle at {path} is not a dictionary")
    required = ("window_size", "feature_dim", "left_model", "right_model")
    missing = [key for key in required if key not in bundle]
    if missing:
        raise KeyError(f"policy bundle at {path} is missing {missing}")
    for side in ("left_model", "right_model"):
        if not callable(getattr(bundle[side], "predict_confidence", None)):
            raise TypeError(f"{side} in {path} does not provide predict_confidence()")
    return bundle


def _constant_policy_bundle(kind: str) -> dict[str, Any]:
    if kind == "hand":
        metadata = {
            "feature_mode": "hmd_relative_motion_tracking_v2",
            "feature_dim": 166,
            "hmd_pose_dim": 7,
            "left_hand_pose_dim": 42,
            "right_hand_pose_dim": 42,
        }
    else:
        metadata = {
            "feature_mode": "controller_motion_tracking_v1",
            "feature_dim": 32,
            "left_controller_pose_dim": 7,
            "right_controller_pose_dim": 7,
        }
    return {
        "window_size": 1,
        "pose_stale_sec": 0.2,
        "left_model": ConstantConfidenceModel(1.0),
        "right_model": ConstantConfidenceModel(1.0),
        **metadata,
    }


def load_policy_bundle(
    kind: str,
    variant: str = "rnn",
    override: str = "",
    fallback_variant: str = "histgb",
) -> tuple[dict[str, Any], Path, str]:
    kind = str(kind).strip().lower()
    variant = str(variant).strip().lower()
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"unsupported policy kind {kind!r}; expected one of {SUPPORTED_KINDS}")
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"unsupported policy variant {variant!r}; expected one of {SUPPORTED_VARIANTS}"
        )
    if variant == ALWAYS_ONE_VARIANT:
        path = Path("<built-in-always_1>")
        return _validate_bundle(_constant_policy_bundle(kind), path), path, variant

    primary = policy_path(kind, variant, override)
    try:
        return _validate_bundle(joblib.load(primary), primary), primary, variant
    except Exception:
        if override or not fallback_variant or fallback_variant == variant:
            raise
        fallback = policy_path(kind, fallback_variant)
        return _validate_bundle(joblib.load(fallback), fallback), fallback, fallback_variant


def main_verify(args=None) -> None:
    parser = argparse.ArgumentParser(description="Load and validate vendored IGRIS policies.")
    parser.add_argument("--kind", choices=SUPPORTED_KINDS, default="hand")
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS, default="rnn")
    parsed = parser.parse_args(args=args)
    bundle, path, loaded_variant = load_policy_bundle(parsed.kind, parsed.variant)
    print(
        f"loaded kind={parsed.kind} variant={loaded_variant} path={path} "
        f"window={bundle['window_size']} feature_dim={bundle['feature_dim']}"
    )


if __name__ == "__main__":
    main_verify()
