from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from igris_teleop.workers import registry


def _context(teleop_device: str) -> SimpleNamespace:
    return SimpleNamespace(
        run_config=SimpleNamespace(mode="teleop", teleop_device=teleop_device)
    )


@pytest.mark.parametrize(
    ("builder", "class_name", "baseline_module", "current_module"),
    (
        (
            registry.build_unity,
            "UnityRosridgeWorker",
            "igris_teleop.workers.worker_unity_bridge_baseline",
            "igris_teleop.workers.worker_unity_bridge",
        ),
        (
            registry.build_igris_ik,
            "IGRISIKWorker",
            "igris_teleop.workers.worker_igris_ik_baseline",
            "igris_teleop.workers.worker_igris_ik",
        ),
        (
            registry.build_hand,
            "HandWorker",
            "igris_teleop.workers.worker_hand_baseline",
            "igris_teleop.workers.worker_hand",
        ),
    ),
)
@pytest.mark.parametrize(
    ("teleop_device", "expected_path"),
    (
        ("unity", "baseline"),
        ("unity_hybrid", "current"),
        ("vr_masterarm", "current"),
    ),
)
def test_unity_baseline_worker_routing(
    monkeypatch: pytest.MonkeyPatch,
    builder,
    class_name: str,
    baseline_module: str,
    current_module: str,
    teleop_device: str,
    expected_path: str,
) -> None:
    monkeypatch.delenv("IGRIS_IK_HZ", raising=False)

    def make_worker(path: str):
        return lambda ctx, hz: (path, ctx.run_config.teleop_device, hz)

    monkeypatch.setitem(
        sys.modules,
        baseline_module,
        types.SimpleNamespace(**{class_name: make_worker("baseline")}),
    )
    monkeypatch.setitem(
        sys.modules,
        current_module,
        types.SimpleNamespace(**{class_name: make_worker("current")}),
    )

    worker = builder(_context(teleop_device))

    routed_path = "current" if builder is registry.build_hand else expected_path
    assert worker[0] == routed_path
    assert worker[1] == teleop_device
    if builder is registry.build_igris_ik:
        expected_hz = 100.0
    elif builder is registry.build_unity:
        expected_hz = 60.0
    else:
        expected_hz = 50.0
    assert worker[2] == expected_hz


@pytest.mark.parametrize(
    ("teleop_device", "module_name"),
    (
        ("unity", "igris_teleop.workers.worker_igris_ik_baseline"),
        ("vr_masterarm", "igris_teleop.workers.worker_igris_ik"),
    ),
)
def test_ik_rate_can_be_overridden_by_environment(
    monkeypatch: pytest.MonkeyPatch,
    teleop_device: str,
    module_name: str,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        module_name,
        types.SimpleNamespace(
            IGRISIKWorker=lambda ctx, hz: (ctx.run_config.teleop_device, hz)
        ),
    )
    monkeypatch.setenv("IGRIS_IK_HZ", "125")

    worker = registry.build_igris_ik(_context(teleop_device))

    assert worker == (teleop_device, 125.0)


@pytest.mark.parametrize("raw_value", ("0", "-1", "nan", "inf", "fast"))
def test_invalid_current_ik_rate_is_rejected(raw_value: str) -> None:
    with pytest.raises(ValueError, match="IGRIS_IK_HZ"):
        registry.resolve_igris_ik_hz({"IGRIS_IK_HZ": raw_value})


def test_unity_baseline_requires_teleop_mode() -> None:
    ctx = SimpleNamespace(
        run_config=SimpleNamespace(mode="inference", teleop_device="unity")
    )

    assert not registry._uses_unity_baseline(ctx)


def test_unity_hmd_ik_seed_continues_command_inside_observation_envelope() -> None:
    pytest.importorskip("pinocchio")
    from igris_teleop.workers.worker_igris_ik_baseline import IGRISIKWorker

    worker = IGRISIKWorker.__new__(IGRISIKWorker)
    current = IGRISIKWorker._default_home_q()
    worker._last_sol_q = current + 1.0

    seed = worker._resolve_ik_seed_q(current)

    np.testing.assert_allclose(seed[:3] - current[:3], 0.20)
    np.testing.assert_allclose(seed[3:17] - current[3:17], 0.15)
    np.testing.assert_allclose(seed[17:19] - current[17:19], 0.35)
