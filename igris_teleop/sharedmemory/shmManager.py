# shm_manager.py
import numpy as np
from multiprocessing import shared_memory
from multiprocessing import resource_tracker
import multiprocessing as mp
import inspect

class SharedMemoryManager:
    def __init__(self,
                 schema,#: list[tuple[str, tuple[int,...], type]],
                 lock: mp.Lock,
                 name: str):
        """
        schema: [(field_name, shape, dtype), ...]
        """
        self.lock   = lock
        self.name   = name
        self._parse_schema(schema)
        self._create_or_connect()
        self._recreate_if_size_mismatch()
        # 1D 버퍼 뷰
        self._buf = np.ndarray((self.length,), dtype=self.dtype, buffer=self.shm.buf)

        if self._owner:
            with self.lock:
                self._buf[:] = 0

    def _parse_schema(self, schema):
        offset = 0
        # 첫 번째 항목의 dtype이 전체 공통 dtype이라 가정
        self.dtype = schema[0][2]
        self._fields = {}
        for fname, shape, dt in schema:
            cnt = int(np.prod(shape))
            self._fields[fname] = {"shape":shape, "offset":offset, "count":cnt}
            offset += cnt
        self.length   = offset
        self.shm_size = offset * np.dtype(self.dtype).itemsize

    def _create_or_connect(self):
        has_track = "track" in inspect.signature(shared_memory.SharedMemory).parameters
        self._has_track = has_track

        def _unregister_attached_shm(shm):
            # Python < 3.13 has no track=False for attach-only opens, so non-owner
            # worker processes must unregister attached SHM from resource_tracker.
            if has_track:
                return
            raw_name = getattr(shm, "_name", None) or getattr(shm, "name", None)
            if not raw_name:
                return
            try:
                resource_tracker.unregister(raw_name, "shared_memory")
            except Exception:
                pass

        def _open_existing():
            if has_track:
                try:
                    return shared_memory.SharedMemory(name=self.name, track=False)
                except TypeError:
                    pass
            return shared_memory.SharedMemory(name=self.name)

        try:
            # 기존에 있으면 연결
            self.shm = _open_existing()
            self._owner = False
            _unregister_attached_shm(self.shm)
            
        except FileNotFoundError:
            # 없으면 생성 시도
            try:
                self.shm = shared_memory.SharedMemory(
                    name=self.name, create=True, size=self.shm_size)
                self._owner = True
            except FileExistsError:
                # 누군가 먼저 만들었으면 다시 연결
                self.shm = _open_existing()
                self._owner = False
                _unregister_attached_shm(self.shm)

    def _recreate_if_size_mismatch(self):
        current_size = int(getattr(self.shm, "size", 0) or 0)
        if current_size == self.shm_size:
            return

        try:
            if not getattr(self, "_has_track", False):
                raw_name = getattr(self.shm, "_name", None) or getattr(self.shm, "name", None)
                if raw_name:
                    try:
                        resource_tracker.register(raw_name, "shared_memory")
                    except Exception:
                        pass
            self.shm.unlink()
        except FileNotFoundError:
            pass
        finally:
            self.shm.close()

        self.shm = shared_memory.SharedMemory(
            name=self.name,
            create=True,
            size=self.shm_size,
        )
        self._owner = True


    def write_data(self, **kwargs):
        with self.lock:
            # 1) 기존 공유메모리 전체를 복사해서 스냅샷 확보
            flat = self._buf.copy()
            # 2) 전달된 필드만 덮어쓰기
            for k, v in kwargs.items():
                meta = self._fields[k]
                arr  = np.asarray(v, dtype=self.dtype).reshape(-1)
                s, e = meta["offset"], meta["offset"] + meta["count"]
                flat[s:e] = arr
            # 3) 수정된 전체 버퍼를 한 번에 반영
            self._buf[:] = flat

    def read_data(self):
        with self.lock:
            data = self._buf.copy()
        out = {}
        for k,meta in self._fields.items():
            s,e = meta["offset"], meta["offset"]+meta["count"]
            out[k] = data[s:e].reshape(meta["shape"])
        return out

    def main_unlink(self):
        if not getattr(self, "_has_track", False):
            raw_name = getattr(self.shm, "_name", None) or getattr(self.shm, "name", None)
            if raw_name:
                try:
                    resource_tracker.register(raw_name, "shared_memory")
                except Exception:
                    pass
        self.shm.close()
        self.shm.unlink()

    def worker_close(self):
        self.shm.close()
