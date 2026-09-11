from __future__ import annotations

from igris_teleop.core import rate as rate_module
from igris_teleop.core.rate import Rate


def test_rate_skips_missed_deadlines_after_overrun(monkeypatch) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr(rate_module.time, "perf_counter", lambda: clock["now"])
    monkeypatch.setattr(rate_module.time, "sleep", lambda delay: None)

    rate = Rate(100.0)
    clock["now"] = 0.105
    rate.sleep()

    assert rate.next_t > clock["now"]
    assert rate.next_t <= clock["now"] + rate.dt
