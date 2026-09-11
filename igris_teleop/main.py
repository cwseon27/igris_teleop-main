from __future__ import annotations

import argparse
import fcntl
import logging
import json
import multiprocessing as mp
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO
from logging.handlers import QueueHandler

def _ensure_mp_start_method() -> None:
    """
    종료 안정성을 위해 기본은 spawn 사용.
    필요 시 IGRIS_MP_START_METHOD 환경변수로 override 가능.
    """
    requested = (os.getenv("IGRIS_MP_START_METHOD") or "spawn").strip().lower()
    available = set(mp.get_all_start_methods())
    if requested not in available:
        for cand in ("spawn", "forkserver", "fork"):
            if cand in available:
                requested = cand
                break
        else:
            return
    try:
        mp.set_start_method(requested)
    except (RuntimeError, ValueError):
        # 이미 설정되었거나 지원되지 않는 경우 그대로 둔다.
        pass


_ensure_mp_start_method()


import logging_mp

_EXTRA_LOG_HANDLERS: list[logging.Handler] = []


class _QueueTeeStream:
    def __init__(self, stream, log_queue: Any, source: str) -> None:
        self._stream = stream
        self._log_queue = log_queue
        self._source = source
        self._buffer = ""

    def __getattr__(self, name: str):
        return getattr(self._stream, name)

    def write(self, data: str) -> int:
        if not data:
            return 0
        written = self._stream.write(data)
        self._buffer += data
        while True:
            newline_idx = self._buffer.find("\n")
            if newline_idx < 0:
                break
            line = self._buffer[:newline_idx].rstrip("\r")
            self._buffer = self._buffer[newline_idx + 1 :]
            self._enqueue_line(line)
        return written

    def flush(self) -> None:
        self._stream.flush()
        if self._buffer:
            self._enqueue_line(self._buffer.rstrip("\r"))
            self._buffer = ""

    def _enqueue_line(self, line: str) -> None:
        if self._log_queue is None:
            return
        try:
            self._log_queue.put_nowait(f"[{self._source}] {line}")
        except Exception:
            pass


class _UILogHandler(logging.Handler):
    def __init__(self, log_queue: Any) -> None:
        super().__init__()
        self._log_queue = log_queue
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if self._log_queue is None:
            return
        try:
            self._log_queue.put_nowait(self.format(record))
        except Exception:
            pass


def _attach_extra_handlers(lg: logging.Logger) -> None:
    for handler in _EXTRA_LOG_HANDLERS:
        if handler not in lg.handlers:
            lg.addHandler(handler)


def _register_extra_log_handler(handler: logging.Handler) -> None:
    if handler in _EXTRA_LOG_HANDLERS:
        return
    _EXTRA_LOG_HANDLERS.append(handler)
    _attach_extra_handlers(logging.getLogger())
    for existing in logging.root.manager.loggerDict.values():
        if isinstance(existing, logging.Logger):
            _attach_extra_handlers(existing)


def _unregister_extra_log_handler(handler: logging.Handler) -> None:
    if handler in _EXTRA_LOG_HANDLERS:
        _EXTRA_LOG_HANDLERS.remove(handler)
    logging.getLogger().removeHandler(handler)
    for existing in logging.root.manager.loggerDict.values():
        if isinstance(existing, logging.Logger) and handler in existing.handlers:
            existing.removeHandler(handler)


def _install_stdio_tee(log_queue: Any, source_prefix: str):
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _QueueTeeStream(orig_stdout, log_queue, f"{source_prefix}/stdout")
    sys.stderr = _QueueTeeStream(orig_stderr, log_queue, f"{source_prefix}/stderr")
    return orig_stdout, orig_stderr


def _restore_stdio(orig_stdout, orig_stderr) -> None:
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr


def _patch_logging_mp_for_spawn() -> None:
    """
    logging_mp는 spawn child에서 _log_queue가 없는 상태로 QueueHandler(None)을
    붙여 logging 에러를 유발할 수 있어 child에서는 stream fallback으로 우회한다.
    """
    if getattr(logging_mp, "_igris_spawn_patch", False):
        return

    original_get_logger = logging_mp.get_logger

    def _safe_get_logger(name=None, level=None):
        queue_obj = getattr(logging_mp, "_log_queue", None)
        if queue_obj is None and mp.current_process().name != "MainProcess":
            lg = logging.getLogger(name)
            # 기존에 잘못 붙은 QueueHandler(None) 제거
            bad_handlers = [
                h for h in lg.handlers
                if isinstance(h, QueueHandler) and getattr(h, "queue", None) is None
            ]
            for h in bad_handlers:
                try:
                    lg.removeHandler(h)
                except Exception:
                    pass

            if level is not None:
                lg.setLevel(level)
            lg.propagate = False

            if not any(getattr(h, "_igris_fallback", False) for h in lg.handlers):
                h = logging.StreamHandler()
                h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
                setattr(h, "_igris_fallback", True)
                lg.addHandler(h)
            _attach_extra_handlers(lg)
            return lg

        lg = original_get_logger(name=name, level=level)
        _attach_extra_handlers(lg)
        return lg

    logging_mp.get_logger = _safe_get_logger
    setattr(logging_mp, "_igris_spawn_patch", True)


_patch_logging_mp_for_spawn()

logger = logging_mp.get_logger(__name__)
logging_mp.basic_config(level=logging_mp.INFO)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CYCLONEDDS_URI = _PROJECT_ROOT / "local_state" / "cyclonedds_igris_lan.xml"


def _ensure_robot_dds_env_defaults() -> None:
    """Default real-robot DDS traffic to the configured IGRIS LAN profile."""
    if os.environ.get("CYCLONEDDS_URI"):
        return
    if not _DEFAULT_CYCLONEDDS_URI.exists():
        return
    os.environ["CYCLONEDDS_URI"] = f"file://{_DEFAULT_CYCLONEDDS_URI}"


def _prepend_path_value(existing: str | None, path: str) -> str:
    parts = [item for item in (existing or "").split(os.pathsep) if item]
    parts = [item for item in parts if item != path]
    return os.pathsep.join([path, *parts]) if parts else path


def _filter_external_env_path(value: str | None) -> str:
    if not value:
        return ""

    blocked_tokens = (
        "/opt/ros/",
        "/opt/openrobots/",
        "/colcon_ws/install/",
        "/ros_ws/install/",
    )
    kept: list[str] = []
    for item in value.split(os.pathsep):
        if not item:
            continue
        expanded = os.path.expandvars(os.path.expanduser(item))
        if any(token in expanded for token in blocked_tokens):
            continue
        kept.append(item)
    return os.pathsep.join(kept)


def _runtime_site_packages_for(executable: str) -> Path:
    prefix = Path(executable).resolve().parent.parent
    return (
        prefix
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def _is_ros_igris_c_sdk_python_path(path_value: str) -> bool:
    expanded = os.path.expandvars(os.path.expanduser(str(path_value)))
    normalized = expanded.replace("\\", "/")
    return (
        "/ros_ws/install/igris_c_sdk/" in normalized
        or "/ros_ws/build/igris_c_sdk/" in normalized
    )


def _filter_ros_igris_c_sdk_pythonpath(value: str | None) -> str:
    if not value:
        return ""
    kept: list[str] = []
    for item in value.split(os.pathsep):
        if not item:
            continue
        if _is_ros_igris_c_sdk_python_path(item):
            continue
        kept.append(item)
    return os.pathsep.join(kept)


@contextmanager
def _sim_control_raw_sdk_import_context():
    """Make spawned sim-control children import the raw IGRIS SDK.

    When the GUI is launched from a sourced ROS workspace, the ROS interface
    package ``ros_ws/install/igris_c_sdk`` appears before the runtime venv on
    ``PYTHONPATH``.  That package contains ROS messages/srvs, but not the raw
    SDK ``ChannelFactory`` used by MuJoCo domain-99 control.  The multiprocessing
    spawn child inherits the parent's import path, so sanitize it only while the
    sim control process is spawned.
    """
    old_pythonpath = os.environ.get("PYTHONPATH")
    old_sys_path = list(sys.path)
    raw_site = _runtime_site_packages_for(sys.executable)
    try:
        filtered_env_path = _filter_ros_igris_c_sdk_pythonpath(old_pythonpath)
        if raw_site.is_dir():
            filtered_env_path = _prepend_path_value(filtered_env_path, str(raw_site))
        if filtered_env_path:
            os.environ["PYTHONPATH"] = filtered_env_path
        else:
            os.environ.pop("PYTHONPATH", None)

        new_sys_path: list[str] = []
        if raw_site.is_dir():
            new_sys_path.append(str(raw_site))
        for item in old_sys_path:
            if not item:
                new_sys_path.append(item)
                continue
            if _is_ros_igris_c_sdk_python_path(item):
                continue
            if item not in new_sys_path:
                new_sys_path.append(item)
        sys.path[:] = new_sys_path
        yield
    finally:
        if old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pythonpath
        sys.path[:] = old_sys_path


def _should_spawn_control_with_raw_sdk(name: str, run_config: RunConfig) -> bool:
    return name == "control" and _is_sim_runtime(run_config)


def _is_sim_runtime(run_config: RunConfig | None) -> bool:
    runtime = str(getattr(run_config, "runtime_environment", "") or "").strip().lower()
    return runtime in {"sim", "simulation", "simulator", "mujoco"}


def _build_external_worker_env(
    python_executable: str,
    *,
    worker_name: str | None = None,
    run_config: RunConfig | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    ros_prefix = Path(env.get("IGRIS_ROS_PREFIX") or "/opt/ros/jazzy")

    for key in (
        "AMENT_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "CMAKE_PREFIX_PATH",
        "ROS_DISTRO",
        "ROS_VERSION",
        "ROS_PYTHON_VERSION",
        "ROS_LOCALHOST_ONLY",
        "RMW_IMPLEMENTATION",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "VIRTUAL_ENV",
        "PYTHONHOME",
    ):
        env.pop(key, None)

    filtered_pythonpath = _filter_external_env_path(env.get("PYTHONPATH"))
    if filtered_pythonpath:
        env["PYTHONPATH"] = filtered_pythonpath
    else:
        env.pop("PYTHONPATH", None)

    filtered_ld_library_path = _filter_external_env_path(env.get("LD_LIBRARY_PATH"))

    prefix = Path(python_executable).resolve().parent.parent
    if (prefix / "conda-meta").is_dir():
        env["CONDA_PREFIX"] = str(prefix)
        env["CONDA_DEFAULT_ENV"] = prefix.name
    elif (prefix / "pyvenv.cfg").is_file():
        env["VIRTUAL_ENV"] = str(prefix)

    lib_dir = prefix / "lib"
    if lib_dir.is_dir():
        filtered_ld_library_path = _prepend_path_value(filtered_ld_library_path, str(lib_dir))

    bin_dir = prefix / "bin"
    if bin_dir.is_dir():
        env["PATH"] = _prepend_path_value(env.get("PATH"), str(bin_dir))

    if filtered_ld_library_path:
        env["LD_LIBRARY_PATH"] = filtered_ld_library_path
    else:
        env.pop("LD_LIBRARY_PATH", None)

    if worker_name == "hand" and not _is_sim_runtime(run_config):
        # The real-hand worker must keep DexRetargeting in .venv-ml, but its
        # ROS front-end also imports rclpy/std_msgs/std_srvs to talk to the
        # schema-compatible native DDS bridge.  Generic external-worker
        # isolation deliberately strips all ROS paths, which otherwise makes
        # Hand Initial (a separate ros2 CLI subprocess) work while continuous
        # teleoperation can never create its command publisher.
        ros_python = (
            ros_prefix
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        ros_lib = ros_prefix / "lib"
        if ros_python.is_dir():
            env["PYTHONPATH"] = _prepend_path_value(env.get("PYTHONPATH"), str(ros_python))
        if ros_lib.is_dir():
            env["LD_LIBRARY_PATH"] = _prepend_path_value(
                env.get("LD_LIBRARY_PATH"), str(ros_lib)
            )

    if worker_name == "simulator" or _is_sim_runtime(run_config):
        # The desktop process defaults CYCLONEDDS_URI to the real-robot LAN
        # profile.  Simulation workers must not inherit that profile: when the
        # LAN adapter is unplugged CycloneDDS fails before MuJoCo can publish
        # domain-99 lowstate, and the UI can then start control in a confusing
        # half-real/half-sim state.
        env.pop("CYCLONEDDS_URI", None)

    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


from igris_teleop.core.events import EventBus, create_shared_events
from igris_teleop.core.worker_base import (
    CAMERA_MODE_CHOICES,
    DEFAULT_CAMERA_DOMAIN_ID,
    DEFAULT_CAMERA_MODE,
    RunConfig,
    WorkerContext,
    resolve_teleop_hand_source,
)
from igris_teleop.ipc.manager import encode_authkey, start_ipc_server, worker_stop_event_name
from igris_teleop.sharedmemory.shmManager import SharedMemoryManager
from igris_teleop.sharedmemory.shm_init_clean import _cleanup_shared_memory, _init_shared_memory
from igris_teleop.sharedmemory.resources import SHARED_MEMORY_SPECS, default_shm_names
from igris_teleop.workers.command_process import (
    ManagedCommandProcess,
    build_leader_node_process_spec,
)
from igris_teleop.workers.registry import (
    ALWAYS_ON_WORKERS,
    CRITICAL_ONCE_STARTED_WORKERS,
    DEBUG_WORKERS,
    MANUAL_UI_WORKER_ORDER,
    MANUAL_UI_WORKERS,
    MODE_WORKER_POOL,
    WORKER_SPECS,
    get_mode_worker_candidates,
    get_mode_workers_for_apply,
)
from igris_teleop.policies.walking import (
    DEFAULT_WALKING_POLICY_PROFILE,
    WALKING_PROFILE_CHOICES,
    default_walking_policy_path,
    resolve_walking_policy_profile,
)

from igris_teleop.web_ui.tunnels import _stop_ngrok


STARTUP_BLOCKED_EXIT_CODE = 73


class IgrisInstanceAlreadyRunningError(RuntimeError):
    pass


class _IgrisInstanceLock:
    def __init__(self, handle: TextIO, path: Path) -> None:
        self._handle = handle
        self.path = path

    @classmethod
    def acquire(
        cls,
        *,
        web_host: str,
        web_port: int,
        path: Path | None = None,
    ) -> "_IgrisInstanceLock":
        lock_path = path or Path(
            os.getenv("IGRIS_INSTANCE_LOCK_FILE")
            or f"/tmp/igris_teleop_{os.getuid()}.lock"
        )
        lock_path = lock_path.expanduser().resolve()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            raw_owner = handle.read().strip()
            handle.close()
            owner = ""
            try:
                metadata = json.loads(raw_owner)
                owner_host = str(metadata.get("web_host") or "127.0.0.1")
                if owner_host in {"", "0.0.0.0"}:
                    owner_host = "127.0.0.1"
                owner = (
                    f" (pid={metadata.get('pid', '?')}, "
                    f"web=http://{owner_host}:{metadata.get('web_port', '?')}/)"
                )
            except Exception:
                if raw_owner:
                    owner = f" ({raw_owner})"
            raise IgrisInstanceAlreadyRunningError(
                "Another IGRIS teleop instance is already running"
                f"{owner}. Stop the existing GUI before launching again."
            ) from exc

        metadata = {
            "pid": os.getpid(),
            "web_host": str(web_host),
            "web_port": int(web_port),
            "started_at": time.time(),
        }
        handle.seek(0)
        handle.truncate()
        json.dump(metadata, handle, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        return cls(handle, lock_path)

    def close(self) -> None:
        if self._handle.closed:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()


def _assert_web_port_available(host: str, port: int) -> None:
    if int(port) == 0:
        return
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((str(host), int(port)))
        except OSError as exc:
            display_host = "127.0.0.1" if host in {"", "0.0.0.0"} else host
            raise RuntimeError(
                f"Web port is already in use: http://{display_host}:{int(port)}/"
            ) from exc


def _proc_is_alive(proc: object) -> bool:
    if hasattr(proc, "is_alive"):
        return bool(proc.is_alive())
    if hasattr(proc, "poll"):
        return proc.poll() is None
    return False


def _proc_join(proc: object, timeout: float) -> None:
    if hasattr(proc, "join"):
        proc.join(timeout=timeout)
        return
    if hasattr(proc, "wait"):
        try:
            proc.wait(timeout=timeout)
        except Exception:
            pass


def _proc_terminate(proc: object) -> None:
    if hasattr(proc, "terminate"):
        try:
            proc.terminate()
        except Exception:
            pass


def _proc_kill(proc: object) -> None:
    if hasattr(proc, "kill"):
        try:
            proc.kill()
            return
        except Exception:
            pass
    _proc_terminate(proc)


def _proc_close(proc: object) -> None:
    if hasattr(proc, "close"):
        try:
            proc.close()
        except Exception:
            pass


def _pump_subprocess_stream(stream: TextIO, log_queue: Any, source: str) -> None:
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            text = line.rstrip("\r\n")
            if not text:
                continue
            if log_queue is not None:
                try:
                    log_queue.put_nowait(f"[{source}] {text}")
                except Exception:
                    pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _join_log_threads(threads: tuple[threading.Thread, ...], timeout: float = 0.2) -> None:
    for thread in threads:
        try:
            thread.join(timeout=timeout)
        except Exception:
            pass


def _resolve_external_python(python_path: str | None) -> str:
    if not python_path:
        return sys.executable
    expanded = Path(os.path.expandvars(os.path.expanduser(python_path)))
    if not expanded.is_absolute():
        expanded = (_PROJECT_ROOT / expanded).resolve()
    return str(expanded)


def _start_external_worker(
    *,
    name: str,
    python_path: str | None,
    ipc_host: str,
    ipc_port: int,
    ipc_auth: str,
    lock_names: list[str],
    shm_name: dict[str, str],
    run_config: RunConfig,
    log_queue: Any,
) -> tuple[subprocess.Popen[str], tuple[threading.Thread, ...]]:
    resolved_python = _resolve_external_python(python_path)
    cmd = [
        resolved_python,
        "-m",
        "igris_teleop.workers._run_external",
        "--worker",
        name,
        "--ipc-host",
        ipc_host,
        "--ipc-port",
        str(ipc_port),
        "--ipc-auth",
        ipc_auth,
        "--run-config",
        run_config.to_json(),
        "--shm-map",
        json.dumps(shm_name, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        "--lock-names",
        json.dumps(lock_names, ensure_ascii=True, separators=(",", ":")),
    ]
    env = _build_external_worker_env(resolved_python, worker_name=name, run_config=run_config)
    logger.info("[MAIN] starting external worker %s: %s", name, " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(_PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    threads: list[threading.Thread] = []
    for stream_name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        if stream is None:
            continue
        thread = threading.Thread(
            target=_pump_subprocess_stream,
            args=(stream, log_queue, f"{name}/{stream_name}"),
            name=f"{name}-{stream_name}-pump",
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    return proc, tuple(threads)


def _run_worker(worker_ctor: Callable[[WorkerContext], object], ctx: WorkerContext) -> None:
    """프로세스 엔트리포인트."""
    # Ctrl+C is delivered to the whole terminal process group.  Only the main
    # supervisor should translate it into ordered worker stop events; otherwise
    # every child raises KeyboardInterrupt in the middle of its safety motion.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if ctx.log_queue is not None:
        # The web UI is the consumer of this display-only queue.  During
        # shutdown the UI can disappear before a worker has emitted its final
        # lines, leaving the multiprocessing Queue feeder blocked on a full
        # pipe.  A spawned child normally waits for that feeder at interpreter
        # exit, even though the robot worker itself (including Torque OFF) has
        # already completed.  Never let optional UI log delivery keep a worker
        # process alive.
        try:
            ctx.log_queue.cancel_join_thread()
        except (AttributeError, OSError, ValueError):
            pass
    if ctx.name == "control":
        # The control process has a 300 Hz desired-command thread and two 100 Hz
        # worker loops. CPython's 5 ms default GIL slice exceeds one command
        # period; a shorter local slice prevents a telemetry loop from holding
        # the interpreter across multiple desired-command deadlines.
        try:
            switch_interval_s = float(os.getenv("IGRIS_CONTROL_GIL_SWITCH_S", "0.001"))
            sys.setswitchinterval(min(0.005, max(0.0005, switch_interval_s)))
        except (TypeError, ValueError):
            pass
    orig_streams = None
    if ctx.log_queue is not None:
        orig_streams = _install_stdio_tee(ctx.log_queue, ctx.name)
    try:
        # 자식 프로세스에서는 main에서 만든 SHM 핸들을 상속하지 않고 attach-only로 연다.
        if ctx.shared_memory is None and ctx.shm_name is not None and ctx.shared_lock is not None:
            try:
                ctx.shared_memory = _init_shared_memory(ctx.shm_name, ctx.shared_lock)
            except Exception:
                logger.exception("[worker:%s] failed to attach shared memories", ctx.name)
                ctx.shared_memory = {}
        w = worker_ctor(ctx)
        w.run()  # type: ignore[attr-defined]
    finally:
        if orig_streams is not None:
            _restore_stdio(*orig_streams)

class WorkerSupervisor:
    """런타임 워커 start/stop/reconcile 담당."""

    def __init__(
        self,
        *,
        bus: EventBus,
        ipc_manager: Any,
        ipc_host: str,
        ipc_port: int,
        ipc_authkey: bytes,
        shared_lock: dict[str, Any],
        shm_name: dict[str, str],
        shared_memory: dict[str, SharedMemoryManager],
        log_queue: Any,
        base_workers: set[str],
        fixed_workers: set[str],
        manual_workers: set[str],
        manual_worker_order: tuple[str, ...],
        camera_mode: str = DEFAULT_CAMERA_MODE,
        camera_domain_id: int = DEFAULT_CAMERA_DOMAIN_ID,
    ) -> None:
        self.bus = bus
        self.ipc_manager = ipc_manager
        self.ipc_host = ipc_host
        self.ipc_port = ipc_port
        self.ipc_auth = encode_authkey(ipc_authkey)
        self.shared_lock = shared_lock
        self.shm_name = shm_name
        self.shared_memory = shared_memory
        self.log_queue = log_queue
        self.runtime_diagnostics = ipc_manager.get_runtime_diagnostics()

        self.base_workers = set(base_workers)
        self.fixed_workers = set(fixed_workers)
        self.manual_workers = set(manual_workers)
        self.manual_worker_order = tuple(manual_worker_order)
        self.camera_mode = str(camera_mode)
        self.camera_domain_id = int(camera_domain_id)

        self.mode_worker_pool: set[str] = set(MODE_WORKER_POOL)

        self.procs: Dict[str, object] = {}
        self.proc_threads: dict[str, tuple[threading.Thread, ...]] = {}
        self.stop_events: dict[str, Any] = {}
        self.worker_configs: dict[str, RunConfig] = {}
        self.intended_running: set[str] = set()

        unknown_fixed = self.fixed_workers.difference(WORKER_SPECS.keys())
        if unknown_fixed:
            raise ValueError(f"Fixed workers not in WORKER_SPECS: {sorted(unknown_fixed)}")
        unknown_manual = self.manual_workers.difference(WORKER_SPECS.keys())
        if unknown_manual:
            raise ValueError(f"Manual workers not in WORKER_SPECS: {sorted(unknown_manual)}")
        unknown_base = self.base_workers.difference(WORKER_SPECS.keys())
        if unknown_base:
            raise ValueError(f"Base workers not in WORKER_SPECS: {sorted(unknown_base)}")
        if self.camera_mode not in CAMERA_MODE_CHOICES:
            raise ValueError(f"Unsupported camera_mode={self.camera_mode!r}")

    def _fixed_run_config(self) -> RunConfig:
        return RunConfig(
            mode=None,
            teleop_device=None,
            runtime_environment=self._current_runtime_environment(),
            camera_mode=self.camera_mode,
            camera_domain_id=self.camera_domain_id,
        )

    def _ctx(self, name: str, run_config: RunConfig, stop_event: Any) -> WorkerContext:
        return WorkerContext(
            name=name,
            bus=self.bus,
            log_queue=self.log_queue,
            shared_lock=self.shared_lock,
            shm_name=self.shm_name,
            shared_memory=None,
            run_config=run_config,
            stop_event=stop_event,
            runtime_diagnostics=self.runtime_diagnostics,
        )

    def _current_runtime_environment(self, *, assume_simulator_running: Optional[bool] = None) -> str:
        self._reap_dead_processes()
        simulator_running = "simulator" in self.intended_running
        if assume_simulator_running is not None:
            simulator_running = bool(assume_simulator_running)
        return "sim" if simulator_running else "real"

    def _with_runtime_environment(
        self,
        run_config: RunConfig,
        *,
        assume_simulator_running: Optional[bool] = None,
    ) -> RunConfig:
        return replace(
            run_config,
            runtime_environment=self._current_runtime_environment(
                assume_simulator_running=assume_simulator_running,
            ),
        )

    def _refresh_walking_logger_runtime_environment(
        self,
        *,
        assume_simulator_running: Optional[bool] = None,
    ) -> None:
        logger_name = "walking_logger"
        if logger_name not in self.intended_running:
            return

        current_cfg = self.worker_configs.get(logger_name)
        if current_cfg is None or current_cfg.mode != "walking":
            return

        updated_cfg = self._with_runtime_environment(
            current_cfg,
            assume_simulator_running=assume_simulator_running,
        )
        if updated_cfg == current_cfg:
            return
        self._start_worker(logger_name, updated_cfg)

    def _refresh_bridge_control_runtime_environment(
        self,
        *,
        assume_simulator_running: Optional[bool] = None,
    ) -> None:
        for name in ("control", "hand"):
            if name not in self.intended_running or name not in self.manual_workers:
                continue
            current_cfg = self.worker_configs.get(name)
            if current_cfg is None:
                continue
            updated_cfg = self._with_runtime_environment(
                current_cfg,
                assume_simulator_running=assume_simulator_running,
            )
            if updated_cfg == current_cfg:
                continue
            self._start_worker(name, updated_cfg)

    def _refresh_camera_runtime_environment(
        self,
        *,
        assume_simulator_running: Optional[bool] = None,
    ) -> None:
        name = "camera_topic"
        if name not in self.intended_running:
            return
        current_cfg = self.worker_configs.get(name)
        if current_cfg is None:
            return
        updated_cfg = self._with_runtime_environment(
            current_cfg,
            assume_simulator_running=assume_simulator_running,
        )
        if updated_cfg == current_cfg:
            return
        self._start_worker(name, updated_cfg)

    def _reap_dead_processes(self) -> None:
        for name, proc in list(self.procs.items()):
            if _proc_is_alive(proc):
                continue
            try:
                _proc_join(proc, timeout=0.0)
            except Exception:
                pass
            self.procs.pop(name, None)
            _join_log_threads(self.proc_threads.pop(name, ()), timeout=0.0)
            self.stop_events.pop(name, None)
            self.worker_configs.pop(name, None)
            if name in self.manual_workers and name not in self.base_workers:
                self.intended_running.discard(name)

    def _start_worker(self, name: str, run_config: RunConfig) -> None:
        self._reap_dead_processes()
        if name not in WORKER_SPECS:
            raise ValueError(f"Unknown worker={name}")

        proc = self.procs.get(name)
        if proc is not None and _proc_is_alive(proc):
            if self.worker_configs.get(name) == run_config:
                self.intended_running.add(name)
                return
            self._stop_worker(name, timeout_s=3.0)

        spec = WORKER_SPECS[name]
        if spec.command_spec is not None:
            stop_event = None
            command_spec = spec.command_spec
            if name == "leader_ros_node":
                command_spec = build_leader_node_process_spec(
                    hand_enabled=run_config.teleop_hand_source == "masterarm",
                )
            proc = ManagedCommandProcess(command_spec, log_queue=self.log_queue)
            logger.info("[MAIN] starting command worker %s: %s", name, " ".join(command_spec.argv))
            proc.start()
            proc_threads = ()
        elif spec.external:
            stop_event = self.ipc_manager.get_event(worker_stop_event_name(name))
            try:
                stop_event.clear()
            except Exception:
                pass
            proc, proc_threads = _start_external_worker(
                name=name,
                python_path=spec.python,
                ipc_host=self.ipc_host,
                ipc_port=self.ipc_port,
                ipc_auth=self.ipc_auth,
                lock_names=list(self.shared_lock.keys()),
                shm_name=self.shm_name,
                run_config=run_config,
                log_queue=self.log_queue,
            )
        else:
            stop_event = mp.Event()
            worker_ctor = spec.builder
            if worker_ctor is None:
                raise ValueError(f"Worker {name} does not use a Python builder")
            proc = mp.Process(
                target=_run_worker,
                args=(worker_ctor, self._ctx(name, run_config, stop_event)),
                daemon=spec.daemon,
            )
            if _should_spawn_control_with_raw_sdk(name, run_config):
                with _sim_control_raw_sdk_import_context():
                    proc.start()
            else:
                proc.start()
            proc_threads = ()

        self.procs[name] = proc
        self.proc_threads[name] = proc_threads
        self.stop_events[name] = stop_event
        self.worker_configs[name] = run_config
        self.intended_running.add(name)

    def _stop_worker(self, name: str, timeout_s: float) -> None:
        proc = self.procs.get(name)
        stop_event = self.stop_events.get(name)

        if stop_event is not None:
            try:
                stop_event.set()
            except Exception:
                pass

        if proc is not None and _proc_is_alive(proc):
            try:
                _proc_join(proc, timeout=timeout_s)
            except Exception:
                pass

        if proc is not None and _proc_is_alive(proc):
            try:
                _proc_terminate(proc)
            except Exception:
                pass
            try:
                _proc_join(proc, timeout=1.0)
            except Exception:
                pass

        if proc is not None and _proc_is_alive(proc):
            logger.warning("[MAIN] worker %s ignored terminate; sending kill", name)
            try:
                _proc_kill(proc)
            except Exception:
                pass
            try:
                _proc_join(proc, timeout=1.0)
            except Exception:
                pass

        if proc is not None and not _proc_is_alive(proc):
            _proc_close(proc)

        self.procs.pop(name, None)
        _join_log_threads(self.proc_threads.pop(name, ()), timeout=0.5)
        self.stop_events.pop(name, None)
        self.worker_configs.pop(name, None)
        self.intended_running.discard(name)

    def start_fixed_workers(self) -> None:
        fixed_cfg = self._fixed_run_config()
        for name in sorted(self.fixed_workers):
            self._start_worker(name, fixed_cfg)
        self.intended_running.update(self.fixed_workers)

    def ensure_fixed_workers_running(self) -> None:
        fixed_cfg = self._fixed_run_config()
        for name in sorted(self.fixed_workers):
            if name in self.procs and _proc_is_alive(self.procs[name]):
                continue
            self._start_worker(name, fixed_cfg)
        self.intended_running.update(self.fixed_workers)

    def apply_camera_settings(self, camera_mode: str) -> str:
        resolved_camera_mode = str(camera_mode)
        if resolved_camera_mode not in CAMERA_MODE_CHOICES:
            raise ValueError(f"Unsupported camera_mode={resolved_camera_mode!r}")

        self.camera_mode = resolved_camera_mode
        if "camera_topic" in self.fixed_workers:
            self._start_worker("camera_topic", self._fixed_run_config())
            self.intended_running.add("camera_topic")
        return self.camera_mode

    def list_selectable_mode_workers(
        self,
        mode: Optional[str],
        teleop_device: Optional[str],
        inference_policy: Optional[str] = None,
    ) -> list[str]:
        try:
            workers = get_mode_worker_candidates(mode, teleop_device, inference_policy)
        except ValueError:
            workers = set()
        return sorted(workers)

    def current_mode_workers(self) -> set[str]:
        self._reap_dead_processes()
        return {name for name in self.intended_running if name in self.mode_worker_pool}

    def list_manual_workers(self) -> list[str]:
        return [name for name in self.manual_worker_order if name in self.manual_workers]

    def current_manual_workers(self) -> set[str]:
        self._reap_dead_processes()
        return {name for name in self.intended_running if name in self.manual_workers}

    def start_manual_worker(
        self,
        name: str,
        *,
        mode: Optional[str] = None,
        teleop_device: Optional[str] = None,
        teleop_hand_source: Optional[str] = None,
        collect_dataset_repo_id: Optional[str] = None,
    ) -> None:
        if name not in self.manual_workers:
            raise ValueError(f"Worker is not UI-manual: {name}")
        self.ensure_fixed_workers_running()
        resolved_teleop_hand_source = None
        if mode == "teleop":
            resolved_teleop_hand_source = resolve_teleop_hand_source(teleop_device, teleop_hand_source)
        run_config = RunConfig(
            mode=mode,
            teleop_device=teleop_device if mode == "teleop" else None,
            teleop_hand_source=resolved_teleop_hand_source,
            collect_dataset_repo_id=collect_dataset_repo_id if name == "collect_data" else None,
        )
        run_config = self._with_runtime_environment(
            run_config,
            assume_simulator_running=True if name == "simulator" else None,
        )
        self._start_worker(name, run_config)
        if name == "simulator":
            self._refresh_walking_logger_runtime_environment(assume_simulator_running=True)
            self._refresh_bridge_control_runtime_environment(assume_simulator_running=True)
            self._refresh_camera_runtime_environment(assume_simulator_running=True)

    def stop_manual_worker(self, name: str) -> None:
        if name not in self.manual_workers:
            raise ValueError(f"Worker is not UI-manual: {name}")
        stop_timeout_s = 75.0 if name == "control" else 3.0
        self._stop_worker(name, timeout_s=stop_timeout_s)
        if name == "simulator":
            self._refresh_walking_logger_runtime_environment(assume_simulator_running=False)
            self._refresh_bridge_control_runtime_environment(assume_simulator_running=False)
            self._refresh_camera_runtime_environment(assume_simulator_running=False)

    def apply_mode_workers(
        self,
        *,
        mode: Optional[str],
        teleop_device: Optional[str],
        teleop_hand_source: Optional[str],
        selected_mode_workers: set[str],
        walking_policy_profile: Optional[str] = None,
        walking_policy_path: Optional[str] = None,
        walking_startup_blend_enabled: Optional[bool] = None,
        inference_dataset_folder: Optional[str] = None,
        inference_pretrained_rel: Optional[str] = None,
        inference_policy: Optional[str] = None,
        inference_chunk_size: Optional[int] = None,
        inference_horizon: Optional[int] = None,
        inference_n_action_step: Optional[int] = None,
        inference_use_dataset_state: Optional[bool] = None,
        inference_use_dataset_tau: Optional[bool] = None,
        inference_use_dataset_camera: Optional[bool] = None,
        inference_instruction: Optional[str] = None,
        replay_dataset_folder: Optional[str] = None,
    ) -> set[str]:
        allowed = get_mode_workers_for_apply(mode, teleop_device, inference_policy)
        unknown_selected = set(selected_mode_workers).difference(allowed)
        if unknown_selected:
            raise ValueError(f"Selected workers are not allowed for current mode/device: {sorted(unknown_selected)}")

        resolved_teleop_hand_source = None
        if mode == "teleop":
            resolved_teleop_hand_source = resolve_teleop_hand_source(teleop_device, teleop_hand_source)

        mode_cfg = RunConfig(
            mode=mode,
            teleop_device=teleop_device if mode == "teleop" else None,
            teleop_hand_source=resolved_teleop_hand_source,
            walking_policy_profile=walking_policy_profile if mode == "walking" else None,
            walking_policy_path=walking_policy_path if mode == "walking" else None,
            walking_startup_blend_enabled=walking_startup_blend_enabled if mode == "walking" else None,
            inference_dataset_folder=inference_dataset_folder if mode == "inference" else None,
            inference_pretrained_rel=inference_pretrained_rel if mode == "inference" else None,
            inference_policy=inference_policy if mode == "inference" else None,
            inference_chunk_size=inference_chunk_size if mode == "inference" else None,
            inference_horizon=inference_horizon if mode == "inference" else None,
            inference_n_action_step=inference_n_action_step if mode == "inference" else None,
            inference_use_dataset_state=inference_use_dataset_state if mode == "inference" else None,
            inference_use_dataset_tau=inference_use_dataset_tau if mode == "inference" else None,
            inference_use_dataset_camera=inference_use_dataset_camera if mode == "inference" else None,
            inference_instruction=inference_instruction if mode == "inference" else None,
            replay_dataset_folder=replay_dataset_folder if mode == "replay" else None,
        )
        mode_cfg = self._with_runtime_environment(mode_cfg)

        self._reap_dead_processes()

        current_mode_workers = {name for name in self.intended_running if name in self.mode_worker_pool}

        to_stop = {name for name in current_mode_workers if name not in selected_mode_workers}
        for name in selected_mode_workers:
            proc = self.procs.get(name)
            if proc is not None and _proc_is_alive(proc) and self.worker_configs.get(name) != mode_cfg:
                to_stop.add(name)

        for name in sorted(to_stop):
            self._stop_worker(name, timeout_s=3.0)

        self.ensure_fixed_workers_running()

        for name in sorted(selected_mode_workers):
            self._start_worker(name, mode_cfg)

        for name in self.mode_worker_pool:
            if name not in selected_mode_workers:
                self.intended_running.discard(name)
        self.intended_running.update(selected_mode_workers)
        self.intended_running.update(self.fixed_workers)

        bridge_manual_cfg = RunConfig(
            mode=mode,
            teleop_device=teleop_device if mode == "teleop" else None,
            teleop_hand_source=resolved_teleop_hand_source,
        )
        bridge_manual_cfg = self._with_runtime_environment(bridge_manual_cfg)
        for name in ("simulator", "control", "hand"):
            if name in self.intended_running and name in self.manual_workers:
                self._start_worker(name, bridge_manual_cfg)

        return set(selected_mode_workers)

    def alive_map(self) -> dict[str, bool]:
        self._reap_dead_processes()
        return {name: (name in self.procs and _proc_is_alive(self.procs[name])) for name in WORKER_SPECS}

    def has_unexpected_base_failure(self) -> bool:
        self._reap_dead_processes()
        for name in self.base_workers:
            if name in self.intended_running and name not in self.procs:
                return True
        return False

    def shutdown_all(self, timeout_s: float = 90.0) -> None:
        deadline = time.time() + float(timeout_s)

        def _ordered_shutdown_names() -> tuple[list[str], list[str]]:
            running_names = [name for name, proc in self.procs.items() if _proc_is_alive(proc)]
            late_stop_workers = {"simulator"}
            early_priority = ("control", "hand")

            late = [name for name in running_names if name in late_stop_workers]
            early_selected: list[str] = []
            for name in early_priority:
                if name in running_names and name not in late and name not in early_selected:
                    early_selected.append(name)
            for name in running_names:
                if name in late or name in early_selected:
                    continue
                early_selected.append(name)
            return early_selected, late

        def _signal_and_join(names: list[str]) -> None:
            for name in names:
                stop_event = self.stop_events.get(name)
                proc = self.procs.get(name)
                if stop_event is None:
                    if proc is not None and _proc_is_alive(proc):
                        logger.info("[MAIN] terminating command worker %s", name)
                        _proc_terminate(proc)
                    continue
                try:
                    stop_event.set()
                except Exception:
                    pass

            for name in names:
                proc = self.procs.get(name)
                if proc is None or not _proc_is_alive(proc):
                    continue
                remain = max(0.0, deadline - time.time())
                try:
                    _proc_join(proc, timeout=remain)
                except Exception:
                    pass

        early_shutdown_names, late_shutdown_names = _ordered_shutdown_names()
        _signal_and_join(early_shutdown_names)
        _signal_and_join(late_shutdown_names)

        lingering = [name for name, proc in self.procs.items() if _proc_is_alive(proc)]
        if lingering:
            logger.warning("[MAIN] workers still alive after graceful shutdown wait: %s", lingering)

        for name, proc in list(self.procs.items()):
            if not _proc_is_alive(proc):
                continue
            try:
                _proc_terminate(proc)
            except Exception:
                pass
            try:
                _proc_join(proc, timeout=1.0)
            except Exception:
                pass

        lingering_after_terminate = [name for name, proc in self.procs.items() if _proc_is_alive(proc)]
        if lingering_after_terminate:
            logger.warning("[MAIN] workers still alive after terminate; sending kill: %s", lingering_after_terminate)

        for name in lingering_after_terminate:
            proc = self.procs.get(name)
            if proc is None or not _proc_is_alive(proc):
                continue
            try:
                _proc_kill(proc)
            except Exception:
                pass
            try:
                _proc_join(proc, timeout=1.0)
            except Exception:
                pass

        lingering_after_kill = [name for name, proc in self.procs.items() if _proc_is_alive(proc)]
        if lingering_after_kill:
            logger.error("[MAIN] workers still alive after kill: %s", lingering_after_kill)

        for proc in list(self.procs.values()):
            if not _proc_is_alive(proc):
                _proc_close(proc)

        for threads in self.proc_threads.values():
            _join_log_threads(threads, timeout=0.5)
        self.procs.clear()
        self.proc_threads.clear()
        self.stop_events.clear()
        self.worker_configs.clear()
        self.intended_running.clear()


def _join_workers(
    supervisor: WorkerSupervisor,
    *,
    timeout_s: float = 90.0,
    shm_name: dict[str, str],
    main_shms: dict[str, SharedMemoryManager],
    ngrok_proc: mp.Process | None = None,
) -> None:
    logger.info("[MAIN] supervisor shutdown start")
    supervisor.shutdown_all(timeout_s=timeout_s)
    logger.info("[MAIN] supervisor shutdown done")

    for key, mgr in main_shms.items():
        try:
            mgr.main_unlink()
            logger.info("[MAIN] unlinked SHM %s", key)
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("[MAIN] Failed to unlink SHM %s", key)

    logger.info("[MAIN] best-effort shared memory cleanup start")
    _cleanup_shared_memory(shm_name)
    logger.info("[MAIN] best-effort shared memory cleanup done")
    logger.info("[MAIN] ngrok stop start")
    _stop_ngrok(ngrok_proc)
    logger.info("[MAIN] ngrok stop done")

def _stop_logging_listener_safely(timeout_s: float = 2.0) -> None:
    """
    logging_mp.stop_listener_process()가 환경에 따라 블로킹될 수 있어
    daemon thread + timeout으로 안전 종료한다.
    """
    done = threading.Event()

    def _worker() -> None:
        try:
            logging_mp.stop_listener_process()
        except Exception:
            logger.exception("[MAIN] stop_listener_process failed")
        finally:
            done.set()

    t = threading.Thread(target=_worker, name="stop-logging-listener", daemon=True)
    t.start()
    t.join(timeout=max(0.1, float(timeout_s)))
    if not done.is_set():
        logger.warning("[MAIN] stop_listener_process timeout (continuing shutdown)")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["teleop", "walking", "inference", "replay"], default=None)
    p.add_argument("--teleop-device", choices=["unity", "unity_hybrid", "vr_masterarm", "masterarm"], default=None)
    p.add_argument("--teleop-hand-source", choices=["vr", "masterarm"], default=None)
    p.add_argument("--camera-mode", choices=CAMERA_MODE_CHOICES, default=DEFAULT_CAMERA_MODE)
    p.add_argument("--camera-domain-id", type=int, default=DEFAULT_CAMERA_DOMAIN_ID)
    p.add_argument("--walking-policy-profile", choices=WALKING_PROFILE_CHOICES, default=DEFAULT_WALKING_POLICY_PROFILE)
    p.add_argument(
        "--walking-policy-path",
        default=None,
        help="walking model path; absolute or relative to the selected profile root under igris_artifacts/walking",
    )
    p.add_argument("--web-host", default=os.getenv("IGRIS_WEB_HOST", "127.0.0.1"))
    p.add_argument("--web-port", type=int, default=int(os.getenv("IGRIS_WEB_PORT", "8000")))
    p.add_argument(
        "--web-open-browser",
        action="store_true",
        default=(os.getenv("IGRIS_WEB_OPEN_BROWSER", "0").strip().lower() in {"1", "true", "yes", "on"}),
    )
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    mp.freeze_support()
    _ensure_robot_dds_env_defaults()
    args = parse_args(argv)
    instance_lock: _IgrisInstanceLock | None = None
    try:
        instance_lock = _IgrisInstanceLock.acquire(
            web_host=args.web_host,
            web_port=args.web_port,
        )
        _assert_web_port_available(args.web_host, args.web_port)
    except (IgrisInstanceAlreadyRunningError, RuntimeError) as exc:
        if instance_lock is not None:
            instance_lock.close()
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(STARTUP_BLOCKED_EXIT_CODE) from exc
    log_queue = mp.Queue(-1)

    initial_mode: Optional[str] = args.mode
    initial_teleop_device: Optional[str] = args.teleop_device
    initial_teleop_hand_source: Optional[str] = args.teleop_hand_source
    initial_camera_mode: str = str(args.camera_mode)
    initial_camera_domain_id: int = int(args.camera_domain_id)
    initial_walking_policy_profile: Optional[str] = resolve_walking_policy_profile(args.walking_policy_profile)
    initial_walking_policy_path: Optional[str] = args.walking_policy_path

    if initial_mode != "teleop" and initial_teleop_device is not None:
        logger.warning("[main] --teleop-device is ignored unless --mode teleop")
        initial_teleop_device = None

    if initial_mode != "teleop" and initial_teleop_hand_source is not None:
        logger.warning("[main] --teleop-hand-source is ignored unless --mode teleop")
        initial_teleop_hand_source = None

    if initial_mode != "walking" and initial_walking_policy_profile != DEFAULT_WALKING_POLICY_PROFILE:
        logger.warning("[main] --walking-policy-profile is ignored unless --mode walking")
        initial_walking_policy_profile = DEFAULT_WALKING_POLICY_PROFILE

    if initial_mode != "walking" and initial_walking_policy_path:
        logger.warning("[main] --walking-policy-path is ignored unless --mode walking")
        initial_walking_policy_path = None

    if initial_mode == "teleop" and initial_teleop_device is None:
        logger.warning("[main] teleop selected but teleop-device is not set (Apply workers disabled until selected)")
        if initial_teleop_hand_source is not None:
            logger.warning("[main] --teleop-hand-source is ignored until --teleop-device is selected")
            initial_teleop_hand_source = None

    if initial_mode == "teleop" and initial_teleop_device is not None:
        try:
            initial_teleop_hand_source = resolve_teleop_hand_source(
                initial_teleop_device,
                initial_teleop_hand_source,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    if initial_mode == "walking" and not initial_walking_policy_path:
        initial_walking_policy_path = str(default_walking_policy_path(initial_walking_policy_profile))

    logger.info(
        "[main] start initial_mode=%s initial_teleop_device=%s initial_teleop_hand_source=%s "
        "initial_walking_policy_profile=%s initial_walking_policy_path=%s "
        "initial_camera_mode=%s initial_camera_domain_id=%d",
        initial_mode,
        initial_teleop_device,
        initial_teleop_hand_source,
        initial_walking_policy_profile,
        initial_walking_policy_path,
        initial_camera_mode,
        initial_camera_domain_id,
    )
    try:
        logger.info("[main] mp_start_method=%s", mp.get_start_method())
    except Exception:
        pass

    ipc_manager, ipc_host, ipc_port, ipc_authkey = start_ipc_server()
    logger.info("[main] ipc manager listening on %s:%s", ipc_host, ipc_port)

    level = create_shared_events(ipc_manager)
    bus = EventBus(level)

    shared_lock = {spec.lock_key: ipc_manager.get_lock(spec.lock_key) for spec in SHARED_MEMORY_SPECS}
    shm_name = default_shm_names()
    main_shms = _init_shared_memory(shm_name, shared_lock)

    fixed_workers = set(ALWAYS_ON_WORKERS)
    if args.debug:
        fixed_workers.update(DEBUG_WORKERS)

    supervisor = WorkerSupervisor(
        bus=bus,
        ipc_manager=ipc_manager,
        ipc_host=ipc_host,
        ipc_port=ipc_port,
        ipc_authkey=ipc_authkey,
        shared_lock=shared_lock,
        shm_name=shm_name,
        shared_memory=main_shms,
        log_queue=log_queue,
        base_workers=set(CRITICAL_ONCE_STARTED_WORKERS),
        fixed_workers=fixed_workers,
        manual_workers=set(MANUAL_UI_WORKERS),
        manual_worker_order=MANUAL_UI_WORKER_ORDER,
        camera_mode=initial_camera_mode,
        camera_domain_id=initial_camera_domain_id,
    )
    supervisor.start_fixed_workers()

    ngrok_proc = None

    try:
        from igris_teleop.web_ui.server import run_web_ui

        run_web_ui(
            bus,
            supervisor,
            shared_memory=main_shms,
            log_queue=log_queue,
            initial_mode=initial_mode,
            initial_teleop_device=initial_teleop_device,
            initial_teleop_hand_source=initial_teleop_hand_source,
            initial_walking_policy_profile=initial_walking_policy_profile,
            initial_walking_policy_path=initial_walking_policy_path,
            initial_camera_mode=initial_camera_mode,
            host=args.web_host,
            port=args.web_port,
            open_browser=args.web_open_browser,
        )
    finally:
        try:
            bus.set_level("shutdown")
        except Exception:
            pass

        logger.info("[main] shutdown begin")
        _join_workers(
            supervisor,
            # Real-robot shutdown includes four bounded pose moves, arrival
            # checks, and the control-mode/Torque-OFF service calls.  Do not
            # SIGTERM the control worker while that safety sequence is active.
            timeout_s=90.0,
            shm_name=shm_name,
            main_shms=main_shms,
            ngrok_proc=ngrok_proc,
        )
        logger.info("[main] workers/shm cleanup done")

        logger.info("[main] exit")
        try:
            ipc_manager.shutdown()
        except Exception:
            logger.exception("[main] failed to shutdown ipc manager")
        # run_web_ui(), the only consumer of this display queue, has already
        # returned.  Waiting for the Queue feeder here can therefore deadlock
        # forever if the pipe filled while workers were finishing.  Persistent
        # logs use the normal logging path; pending GUI-only lines may safely be
        # discarded at this point.
        try:
            log_queue.cancel_join_thread()
        except (AttributeError, OSError, ValueError):
            pass
        try:
            log_queue.close()
        except Exception:
            pass
        _stop_logging_listener_safely(timeout_s=2.0)
        instance_lock.close()


if __name__ == "__main__":
    main()
