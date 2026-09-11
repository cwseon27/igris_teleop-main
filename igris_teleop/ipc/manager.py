from __future__ import annotations

import os
import secrets
import socket
from multiprocessing import Event, Lock
from multiprocessing.managers import AcquirerProxy, BaseManager, DictProxy, EventProxy
from typing import Any, Dict, Tuple

_LEVEL_EVENTS: Dict[str, Event] = {}
_LOCKS: Dict[str, Lock] = {}
_RUNTIME_DIAGNOSTICS: Dict[str, dict[str, Any]] = {}


def _get_event(name: str) -> Event:
    event = _LEVEL_EVENTS.get(name)
    if event is None:
        event = Event()
        _LEVEL_EVENTS[name] = event
    return event


def _get_lock(name: str) -> Lock:
    lock = _LOCKS.get(name)
    if lock is None:
        lock = Lock()
        _LOCKS[name] = lock
    return lock


def _get_runtime_diagnostics() -> Dict[str, dict[str, Any]]:
    return _RUNTIME_DIAGNOSTICS


class IPCManager(BaseManager):
    pass


IPCManager.register("get_event", callable=_get_event, proxytype=EventProxy)
IPCManager.register("get_lock", callable=_get_lock, proxytype=AcquirerProxy)
IPCManager.register("get_runtime_diagnostics", callable=_get_runtime_diagnostics, proxytype=DictProxy)


def worker_stop_event_name(name: str) -> str:
    return f"worker_stop:{name}"


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _default_authkey() -> bytes:
    env = os.environ.get("IPC_AUTHKEY_HEX")
    if env:
        return decode_authkey(env)
    return secrets.token_bytes(16)


def encode_authkey(authkey: bytes) -> str:
    return authkey.hex()


def decode_authkey(text: str) -> bytes:
    return bytes.fromhex(text)


def start_ipc_server(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    authkey: bytes | None = None,
) -> Tuple[IPCManager, str, int, bytes]:
    if port is None:
        port = _pick_free_port(host)
    if authkey is None:
        authkey = _default_authkey()
    manager = IPCManager(address=(host, port), authkey=authkey)
    manager.start()
    started_host, started_port = manager.address
    return manager, str(started_host), int(started_port), authkey


def connect_ipc_server(host: str, port: int, authkey: bytes) -> IPCManager:
    manager = IPCManager(address=(host, port), authkey=authkey)
    manager.connect()
    return manager
