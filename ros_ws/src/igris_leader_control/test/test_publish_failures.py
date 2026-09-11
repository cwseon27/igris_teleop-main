"""Failure-path tests for the leader serial callback."""

import importlib.util
import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock

if importlib.util.find_spec('dynamixel_sdk') is None:
    sdk_stub = ModuleType('dynamixel_sdk')
    sdk_stub.COMM_SUCCESS = 0
    sys.modules['dynamixel_sdk'] = sdk_stub

from igris_leader_control import leader_node
from igris_leader_control.timing import LeaderTimingWindow


def _fake_node(reader):
    """Build only the attributes needed before a failed read returns."""
    return SimpleNamespace(
        _last_callback_started_ns=None,
        _timing=LeaderTimingWindow(),
        group_sync_read=reader,
        ids=[11, 12],
        ADDR_PRESENT_POSITION=132,
        LEN_PRESENT_POSITION=4,
        _drop_sample=Mock(),
    )


def test_transaction_failure_does_not_reach_publish_path():
    """A failed transaction returns before constructing a JointState."""
    reader = Mock()
    reader.txRxPacket.return_value = leader_node.COMM_SUCCESS - 1
    node = _fake_node(reader)

    leader_node.IgrisLeaderNode.publish_data(node)

    node._drop_sample.assert_called_once()
    reader.isAvailable.assert_not_called()
    reader.getData.assert_not_called()


def test_missing_motor_data_does_not_reach_publish_path():
    """Drop a partial sync-read instead of mixing fresh and stale data."""
    reader = Mock()
    reader.txRxPacket.return_value = leader_node.COMM_SUCCESS
    reader.isAvailable.side_effect = [True, False]
    node = _fake_node(reader)

    leader_node.IgrisLeaderNode.publish_data(node)

    node._drop_sample.assert_called_once()
    reader.getData.assert_not_called()
