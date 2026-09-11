from __future__ import annotations

from igris_teleop.core.worker_base import LoopDiagnostics, RunConfig, WorkerContext


def test_loop_diagnostics_publishes_timing_snapshot() -> None:
    sink: dict[str, dict] = {}
    ctx = WorkerContext(
        name="unit_worker",
        bus=None,  # type: ignore[arg-type]
        run_config=RunConfig(mode=None, teleop_device=None),
        runtime_diagnostics=sink,
    )
    diagnostics = LoopDiagnostics(ctx=ctx, loop="main", target_hz=10.0, publish_interval_s=0.2)

    diagnostics.observe(start_s=10.0, end_s=10.001)
    diagnostics.observe(start_s=10.1, end_s=10.102)
    diagnostics.stop()

    payload = sink["unit_worker"]
    assert payload["worker"] == "unit_worker"
    assert payload["loop"] == "main"
    assert payload["target_hz"] == 10.0
    assert payload["actual_hz"] == 10.0
    assert payload["period_ms"] == 100.0
    assert payload["latency_ms"] is not None
    assert payload["stopped"] is True
