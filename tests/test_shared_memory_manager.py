from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys
import uuid
from multiprocessing import shared_memory

import numpy as np

from igris_teleop.sharedmemory.shmManager import SharedMemoryManager


def test_shared_memory_manager_recreates_stale_smaller_segment() -> None:
    name = f"igris_test_stale_{uuid.uuid4().hex}"
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;"
                "from multiprocessing import resource_tracker, shared_memory;"
                "shm = shared_memory.SharedMemory(name=sys.argv[1], create=True, size=4);"
                "resource_tracker.unregister(shm._name, 'shared_memory');"
                "shm.close()"
            ),
            name,
        ],
        check=True,
    )
    schema = [("value", (8,), np.float32)]
    mgr = None

    try:
        mgr = SharedMemoryManager(schema, mp.Lock(), name)

        assert mgr.shm.size == 8 * np.dtype(np.float32).itemsize
        mgr.write_data(value=np.arange(8, dtype=np.float32))
        np.testing.assert_array_equal(mgr.read_data()["value"], np.arange(8, dtype=np.float32))
    finally:
        if mgr is not None:
            try:
                mgr.main_unlink()
            except FileNotFoundError:
                pass
        try:
            leftover = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            pass
        else:
            leftover.close()
            leftover.unlink()
