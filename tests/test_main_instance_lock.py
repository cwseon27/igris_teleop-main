from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

from igris_teleop.main import (
    IgrisInstanceAlreadyRunningError,
    _IgrisInstanceLock,
    _assert_web_port_available,
    _run_worker,
)


def test_instance_lock_blocks_a_second_runtime_and_reports_owner(tmp_path) -> None:
    path = tmp_path / "igris.lock"
    first = _IgrisInstanceLock.acquire(web_host="127.0.0.1", web_port=8123, path=path)
    try:
        with pytest.raises(IgrisInstanceAlreadyRunningError, match=r"pid=.*8123"):
            _IgrisInstanceLock.acquire(web_host="127.0.0.1", web_port=8124, path=path)
    finally:
        first.close()

    replacement = _IgrisInstanceLock.acquire(web_host="127.0.0.1", web_port=8124, path=path)
    replacement.close()


def test_web_port_preflight_rejects_an_existing_listener() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = int(listener.getsockname()[1])

        with pytest.raises(RuntimeError, match=str(port)):
            _assert_web_port_available("127.0.0.1", port)


def test_web_port_preflight_accepts_an_available_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    _assert_web_port_available("127.0.0.1", port)


def test_spawned_worker_does_not_wait_for_ui_log_queue_feeder(monkeypatch) -> None:
    class FakeQueue:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel_join_thread(self) -> None:
            self.cancelled = True

        def put_nowait(self, _message: str) -> None:
            pass

    class Worker:
        def __init__(self, _ctx) -> None:
            pass

        def run(self) -> None:
            pass

    queue = FakeQueue()
    ctx = SimpleNamespace(
        name="test_worker",
        log_queue=queue,
        shared_memory={},
        shm_name=None,
        shared_lock=None,
    )
    monkeypatch.setattr("igris_teleop.main.signal.signal", lambda *_args: None)

    _run_worker(Worker, ctx)

    assert queue.cancelled is True
