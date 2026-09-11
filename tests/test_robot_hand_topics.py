from __future__ import annotations

import sys
import types

import pytest


_fake_retargeting = types.ModuleType("igris_teleop.hand_control.hand_retargeting")
_fake_retargeting.HandRetargeting = object
sys.modules.setdefault("igris_teleop.hand_control.hand_retargeting", _fake_retargeting)

from igris_teleop.hand_control import robot_hand


class _FakeChannelFactory:
    def IsInitialized(self) -> bool:
        return False

    def Init(self, domain_id: int) -> None:
        self.domain_id = int(domain_id)

    def Release(self) -> None:
        pass


class _FakeChannelFactoryType:
    _instance = _FakeChannelFactory()

    @classmethod
    def Instance(cls) -> _FakeChannelFactory:
        return cls._instance


class _FakeSubscriber:
    def __init__(self, topic: str) -> None:
        self.topic = topic

    def init(self, callback) -> bool:
        self.callback = callback
        return True

    def stop(self) -> None:
        pass


class _FakePublisher:
    def __init__(self, topic: str) -> None:
        self.topic = topic

    def init(self) -> bool:
        return True

    def stop(self) -> None:
        pass


class _FakeSdk:
    ChannelFactory = _FakeChannelFactoryType
    HandStateSubscriber = _FakeSubscriber
    HandCmdPublisher = _FakePublisher


def test_real_hand_dds_defaults_include_rt_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(robot_hand, "igc_sdk", _FakeSdk)
    monkeypatch.delenv("IGRIS_HAND_STATE_TOPIC", raising=False)
    monkeypatch.delenv("IGRIS_HAND_CMD_TOPIC", raising=False)
    monkeypatch.delenv("IGRIS_HAND_DDS_NAMESPACE", raising=False)
    monkeypatch.delenv("IGRIS_ROBOT_DDS_NAMESPACE", raising=False)

    interface = robot_hand.IgrisHandDDSInterface(domain_id=0)
    try:
        assert interface._state_sub.topic == "igris_c_IG05/rt/handstate"
        assert interface._cmd_pub.topic == "igris_c_IG05/rt/handcmd"
    finally:
        interface.stop()


def test_real_hand_dds_topic_override_remains_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(robot_hand, "igc_sdk", _FakeSdk)
    monkeypatch.setenv("IGRIS_HAND_STATE_TOPIC", "/custom/handstate")
    monkeypatch.setenv("IGRIS_HAND_CMD_TOPIC", "/custom/handcmd")
    monkeypatch.setenv("IGRIS_HAND_DDS_NAMESPACE", "test_robot")

    interface = robot_hand.IgrisHandDDSInterface(domain_id=0)
    try:
        assert interface._state_sub.topic == "test_robot/custom/handstate"
        assert interface._cmd_pub.topic == "test_robot/custom/handcmd"
    finally:
        interface.stop()


def test_hand_dds_namespace_can_be_disabled_for_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(robot_hand, "igc_sdk", _FakeSdk)
    monkeypatch.delenv("IGRIS_HAND_DDS_NAMESPACE", raising=False)
    monkeypatch.delenv("IGRIS_ROBOT_DDS_NAMESPACE", raising=False)

    interface = robot_hand.IgrisHandDDSInterface(domain_id=99, dds_namespace="")
    try:
        assert interface._state_sub.topic == "rt/handstate"
        assert interface._cmd_pub.topic == "rt/handcmd"
    finally:
        interface.stop()


def test_already_scoped_hand_topic_is_not_prefixed_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(robot_hand, "igc_sdk", _FakeSdk)
    monkeypatch.setenv("IGRIS_HAND_CMD_TOPIC", "/igris_c_IG05/rt/handcmd")
    monkeypatch.delenv("IGRIS_HAND_DDS_NAMESPACE", raising=False)
    monkeypatch.delenv("IGRIS_ROBOT_DDS_NAMESPACE", raising=False)

    interface = robot_hand.IgrisHandDDSInterface(domain_id=0)
    try:
        assert interface._cmd_pub.topic == "igris_c_IG05/rt/handcmd"
    finally:
        interface.stop()
