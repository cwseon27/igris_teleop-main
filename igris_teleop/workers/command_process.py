from __future__ import annotations

import os
import grp
import pwd
import shlex
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core.project_paths import REPO_ROOT


ROS_JAZZY_SETUP_BASH = Path("/opt/ros/jazzy/setup.bash")
ROS_WS_ROOT = (REPO_ROOT / "ros_ws").resolve()
ROS_WS_SETUP_BASH = (ROS_WS_ROOT / "install" / "setup.bash").resolve()
ROS_TCP_ENDPOINT_PORT = 10000


def _ros_workspace_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    explicit_root = str(os.getenv("IGRIS_ROS_WS_ROOT") or "").strip()
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser().resolve())
    candidates.extend(
        (
            ROS_WS_ROOT,
            (REPO_ROOT.parent / "ros_ws").resolve(),
            (REPO_ROOT.parent / "ros2_ws").resolve(),
        )
    )

    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def resolve_ros_workspace_root(package_name: str) -> Path:
    candidates = _ros_workspace_candidates()
    for root in candidates:
        if (root / "install" / "setup.bash").is_file() and (root / "install" / package_name).exists():
            return root
    for root in candidates:
        if (root / "install" / "setup.bash").is_file():
            return root
    return candidates[0]


def _tcp_port_is_bound(host: str, port: int) -> bool:
    probe_host = "0.0.0.0" if host in {"", "localhost"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((probe_host, int(port)))
        except OSError:
            return True
    return False


def _quote_command(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def ros_tcp_endpoint_command_tokens() -> tuple[str, ...]:
    return ("ros2", "run", "ros_tcp_endpoint", "default_server_endpoint")


def leader_node_command_tokens(*, hand_enabled: bool = True) -> tuple[str, ...]:
    return (
        "ros2",
        "run",
        "igris_leader_control",
        "leader_node",
        "--ros-args",
        "-p",
        f"hand_enabled:={'true' if hand_enabled else 'false'}",
    )


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _current_user_is_listed_in_group(group_name: str) -> bool:
    try:
        group = grp.getgrnam(group_name)
        user = pwd.getpwuid(os.getuid())
    except Exception:
        return False
    return (
        group.gr_gid in os.getgroups()
        or user.pw_gid == group.gr_gid
        or user.pw_name in set(group.gr_mem)
    )


def _serial_group_name_if_available() -> str | None:
    if _truthy_env("IGRIS_LEADER_DISABLE_SG_DIALOUT", default=False):
        return None
    group_name = (os.getenv("IGRIS_LEADER_SERIAL_GROUP") or "dialout").strip()
    if not group_name or shutil.which("sg") is None:
        return None
    if not _current_user_is_listed_in_group(group_name):
        return None
    return group_name


def _build_serial_group_ros_shell_argv(
    command_tokens: Sequence[str],
    *,
    ros_setup_bash: Path = ROS_JAZZY_SETUP_BASH,
    workspace_setup_bash: Path = ROS_WS_SETUP_BASH,
) -> tuple[str, ...]:
    """Run leader serial node with dialout while preserving ROS setup.

    A common field failure mode is: the user has been added to ``dialout`` in
    /etc/group, but the already-running desktop/UI session still lacks group 20.
    In that state /dev/ttyUSB0 is correctly ``root:dialout`` yet the leader
    node exits with PermissionError.  The ROS setup must be sourced *inside*
    the ``sg`` shell; otherwise sg may drop LD_LIBRARY_PATH and rclpy fails to
    load librcl_action.so.
    """
    script = build_ros_shell_script(
        command_tokens,
        ros_setup_bash=ros_setup_bash,
        workspace_setup_bash=workspace_setup_bash,
    )
    group_name = _serial_group_name_if_available()
    if not group_name:
        return ("bash", "-lc", script)
    return ("sg", group_name, "-c", f"bash -lc {shlex.quote(script)}")


def build_ros_shell_script(
    command_tokens: Sequence[str],
    *,
    ros_setup_bash: Path = ROS_JAZZY_SETUP_BASH,
    workspace_setup_bash: Path = ROS_WS_SETUP_BASH,
) -> str:
    return "\n".join(
        (
            "set -eo pipefail",
            "set +u",
            f"source {shlex.quote(str(ros_setup_bash))}",
            f"source {shlex.quote(str(workspace_setup_bash))}",
            "set -u",
            f"exec {_quote_command(command_tokens)}",
        )
    )


def build_ros_shell_argv(
    command_tokens: Sequence[str],
    *,
    ros_setup_bash: Path = ROS_JAZZY_SETUP_BASH,
    workspace_setup_bash: Path = ROS_WS_SETUP_BASH,
) -> tuple[str, ...]:
    return (
        "bash",
        "-lc",
        build_ros_shell_script(
            command_tokens,
            ros_setup_bash=ros_setup_bash,
            workspace_setup_bash=workspace_setup_bash,
        ),
    )


@dataclass(frozen=True)
class CommandProcessSpec:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    startup_grace_s: float = 0.5
    env: Mapping[str, str] | None = None
    reuse_tcp_listener: tuple[str, int] | None = None


def build_ros_tcp_endpoint_process_spec() -> CommandProcessSpec:
    workspace_root = resolve_ros_workspace_root("ros_tcp_endpoint")
    return CommandProcessSpec(
        name="leader_ros_tcp_endpoint",
        argv=build_ros_shell_argv(
            ros_tcp_endpoint_command_tokens(),
            workspace_setup_bash=workspace_root / "install" / "setup.bash",
        ),
        cwd=workspace_root,
        startup_grace_s=0.5,
        reuse_tcp_listener=("0.0.0.0", ROS_TCP_ENDPOINT_PORT),
    )


def build_leader_node_process_spec(*, hand_enabled: bool = True) -> CommandProcessSpec:
    workspace_root = resolve_ros_workspace_root("igris_leader_control")
    runtime_site_packages = (
        REPO_ROOT
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    pythonpath_parts = [str(runtime_site_packages)]
    inherited_pythonpath = str(os.getenv("PYTHONPATH") or "").strip()
    if inherited_pythonpath:
        pythonpath_parts.append(inherited_pythonpath)
    return CommandProcessSpec(
        name="leader_ros_node",
        argv=_build_serial_group_ros_shell_argv(
            leader_node_command_tokens(hand_enabled=hand_enabled),
            workspace_setup_bash=workspace_root / "install" / "setup.bash",
        ),
        cwd=workspace_root,
        startup_grace_s=1.5,
        # The UI log panel consumes stdout line by line. Keep the 50 Hz
        # terminal dashboard disabled for the managed leader node so only
        # connection state changes are logged.
        env={
            "IGRIS_LEADER_DASHBOARD": "0",
            # ros2 console scripts use /usr/bin/python3. Expose the launcher's
            # runtime venv so the pure-Python Dynamixel SDK is available too.
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        },
    )


class ManagedCommandProcess:
    def __init__(self, spec: CommandProcessSpec, *, log_queue: Any = None) -> None:
        self._spec = spec
        self._log_queue = log_queue
        self._proc: subprocess.Popen[str] | None = None
        self._threads: tuple[threading.Thread, ...] = ()
        self._reused_tcp_listener = False
        self._released = False
        self._recent_output: deque[str] = deque(maxlen=12)
        self._output_lock = threading.Lock()

    @property
    def pid(self) -> int | None:
        if self._proc is None:
            return None
        return self._proc.pid

    def poll(self) -> int | None:
        if self._reused_tcp_listener:
            if self._released:
                return 0
            listener = self._spec.reuse_tcp_listener
            if listener is not None and _tcp_port_is_bound(*listener):
                return None
            self._write_log(
                f"[{self._spec.name}/stderr] reused TCP listener is no longer available; "
                "starting a managed replacement"
            )
            # A new GUI session can briefly see and reuse the previous session's
            # endpoint while that previous supervisor is still shutting down.
            # When the old listener subsequently disappears, take ownership of
            # the now-free port instead of leaving Unity topics without a
            # publisher until the operator toggles the worker manually.
            self._reused_tcp_listener = False
            try:
                self.start()
            except Exception as exc:
                self._released = True
                self._write_log(
                    f"[{self._spec.name}/stderr] failed to replace reused TCP listener: {exc}"
                )
                return 1
            return self._proc.poll() if self._proc is not None else 1
        if self._proc is None:
            return None
        return self._proc.poll()

    def wait(self, timeout: float | None = None) -> int | None:
        if self._reused_tcp_listener:
            return self.poll()
        if self._proc is None:
            return None
        try:
            rc = self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        self._join_threads(timeout=0.5)
        return rc

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        self._released = False
        listener = self._spec.reuse_tcp_listener
        if listener is not None and _tcp_port_is_bound(*listener):
            self._reused_tcp_listener = True
            self._proc = None
            host, port = listener
            self._write_log(
                f"[{self._spec.name}/stdout] reusing existing TCP endpoint on {host}:{port}"
            )
            return
        self._reused_tcp_listener = False

        env = os.environ.copy()
        if self._spec.env:
            env.update({str(k): str(v) for k, v in self._spec.env.items()})
        env.setdefault("PYTHONUNBUFFERED", "1")

        proc = subprocess.Popen(
            list(self._spec.argv),
            cwd=str(self._spec.cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )

        threads: list[threading.Thread] = []
        for stream_name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
            if stream is None:
                continue
            thread = threading.Thread(
                target=self._pump_stream,
                args=(stream, f"{self._spec.name}/{stream_name}"),
                name=f"{self._spec.name}-{stream_name}-pump",
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        self._proc = proc
        self._threads = tuple(threads)

        deadline = time.monotonic() + max(0.0, float(self._spec.startup_grace_s))
        while time.monotonic() < deadline:
            rc = proc.poll()
            if rc is not None:
                self._join_threads(timeout=0.2)
                message = f"{self._spec.name} exited during startup with code {rc}"
                with self._output_lock:
                    detail = "\n".join(self._recent_output)
                if detail:
                    message = f"{message}:\n{detail}"
                raise RuntimeError(message)
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def terminate(self) -> None:
        if self._reused_tcp_listener:
            self._released = True
            return
        proc = self._proc
        if proc is None:
            return

        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                proc.terminate()

            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
                    proc.kill()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass

        self._join_threads(timeout=0.5)

    def _pump_stream(self, stream: Any, source: str) -> None:
        try:
            for raw_line in iter(stream.readline, ""):
                if not raw_line:
                    break
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                self._write_log(f"[{source}] {line}")
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _write_log(self, message: str) -> None:
        with self._output_lock:
            self._recent_output.append(message)
        if self._log_queue is None:
            return
        try:
            self._log_queue.put_nowait(message)
        except Exception:
            pass

    def _join_threads(self, timeout: float) -> None:
        for thread in self._threads:
            try:
                thread.join(timeout=timeout)
            except Exception:
                pass
