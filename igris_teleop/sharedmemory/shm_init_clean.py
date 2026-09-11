from multiprocessing import shared_memory

from .shmManager import SharedMemoryManager
from .resources import SHARED_MEMORY_KEYS, SHARED_MEMORY_SPECS

import logging_mp
logger = logging_mp.get_logger(__name__)

def _cleanup_shared_memory(shm_name: dict) -> None:
    """Best-effort unlink of SHMs we own to avoid resource_tracker warnings."""
    for key in SHARED_MEMORY_KEYS:
        name = shm_name.get(key)
        if not name:
            continue
        try:
            shm = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            continue
        except Exception:
            logger.exception("[MAIN] Failed to access SHM %s", key)
            continue
        try:
            shm.unlink()
            logger.info("[MAIN] unlinked SHM %s", key)
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("[MAIN] Failed to unlink SHM %s", key)
        finally:
            try:
                shm.close()
            except Exception:
                pass


def _init_shared_memory(shm_name: dict, shared_lock: dict) -> dict[str, SharedMemoryManager]:
    """Create SHMs once in the main process so children only attach."""
    created: dict[str, SharedMemoryManager] = {}
    for spec in SHARED_MEMORY_SPECS:
        name = shm_name.get(spec.shm_key)
        lock = shared_lock.get(spec.lock_key)
        if not name or lock is None:
            continue
        try:
            created[spec.shm_key] = SharedMemoryManager(spec.schema, lock, name)
        except Exception:
            logger.exception("[MAIN] Failed to init shared memory %s", spec.shm_key)
    return created
