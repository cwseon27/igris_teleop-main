from __future__ import annotations

import pytest

qos_module = pytest.importorskip("rclpy.qos")
publisher_module = pytest.importorskip(
    "igris_teleop.teleop_devices.cameras.sim_stereo_publisher"
)


def test_sim_stereo_qos_matches_unity_reliable_subscriber() -> None:
    qos = publisher_module.build_sim_stereo_qos()

    assert qos.reliability == qos_module.ReliabilityPolicy.RELIABLE
    assert qos.history == qos_module.HistoryPolicy.KEEP_LAST
    assert qos.durability == qos_module.DurabilityPolicy.VOLATILE
    assert qos.depth == 1
