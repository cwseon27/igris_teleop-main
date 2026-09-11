from __future__ import annotations

import argparse
import json
from typing import Any

import logging_mp

from igris_teleop.core.events import EventBus, create_shared_events
from igris_teleop.core.worker_base import RunConfig, WorkerContext
from igris_teleop.ipc.manager import connect_ipc_server, decode_authkey, worker_stop_event_name
from igris_teleop.sharedmemory.shm_init_clean import _init_shared_memory
from .registry import build_worker, list_worker_names


logger = logging_mp.get_logger(__name__)
logging_mp.basic_config(level=logging_mp.INFO)


def _decode_json_argument(payload: str, expected_type: type) -> Any:
    value = json.loads(payload)
    if not isinstance(value, expected_type):
        raise ValueError(f"Expected JSON {expected_type.__name__}, got {type(value).__name__}")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a worker in an external process.")
    parser.add_argument("--worker", required=True, choices=list_worker_names())
    parser.add_argument("--ipc-host", required=True)
    parser.add_argument("--ipc-port", type=int, required=True)
    parser.add_argument("--ipc-auth", required=True, help="Auth key hex string.")
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--shm-map", required=True, help="JSON object of shm_key->shm_name")
    parser.add_argument("--lock-names", required=True, help="JSON array of lock names")
    args = parser.parse_args(argv)

    authkey = decode_authkey(args.ipc_auth)
    manager = connect_ipc_server(args.ipc_host, args.ipc_port, authkey)

    level = create_shared_events(manager)
    bus = EventBus(level)

    lock_names = _decode_json_argument(args.lock_names, list)
    shm_name = _decode_json_argument(args.shm_map, dict)
    run_config = RunConfig.from_json(args.run_config)

    shared_lock = {str(name): manager.get_lock(str(name)) for name in lock_names}
    shared_memory = _init_shared_memory(shm_name, shared_lock)
    stop_event = manager.get_event(worker_stop_event_name(args.worker))
    runtime_diagnostics = manager.get_runtime_diagnostics()

    ctx = WorkerContext(
        name=args.worker,
        bus=bus,
        run_config=run_config,
        stop_event=stop_event,
        runtime_diagnostics=runtime_diagnostics,
        shared_lock=shared_lock,
        shm_name=shm_name,
        shared_memory=shared_memory,
    )
    logger.info("[external:%s] start", args.worker)
    worker = build_worker(args.worker, ctx)
    worker.run()
    logger.info("[external:%s] stop", args.worker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
