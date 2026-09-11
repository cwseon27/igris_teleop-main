from __future__ import annotations

import queue
import socket
import sys
import textwrap
import time
from pathlib import Path

import pytest

from igris_teleop.core.project_paths import REPO_ROOT
from igris_teleop.workers.command_process import (
    CommandProcessSpec,
    ManagedCommandProcess,
    _ros_workspace_candidates,
    build_leader_node_process_spec,
    build_ros_shell_script,
    leader_node_command_tokens,
    resolve_ros_workspace_root,
    ros_tcp_endpoint_command_tokens,
)


def test_ros_workspace_candidates_include_sibling_ros_ws() -> None:
    assert (REPO_ROOT.parent / "ros_ws").resolve() in _ros_workspace_candidates()


def test_build_ros_shell_script_sources_ros_then_workspace() -> None:
    script = build_ros_shell_script(
        ros_tcp_endpoint_command_tokens(),
        ros_setup_bash=Path("/opt/ros/jazzy/setup.bash"),
        workspace_setup_bash=Path("/tmp/ws/install/setup.bash"),
    )

    lines = [line.strip() for line in script.splitlines() if line.strip()]
    assert lines == [
        "set -eo pipefail",
        "set +u",
        "source /opt/ros/jazzy/setup.bash",
        "source /tmp/ws/install/setup.bash",
        "set -u",
        "exec ros2 run ros_tcp_endpoint default_server_endpoint",
    ]


def test_leader_node_command_enables_hand_by_default() -> None:
    assert leader_node_command_tokens() == (
        "ros2",
        "run",
        "igris_leader_control",
        "leader_node",
        "--ros-args",
        "-p",
        "hand_enabled:=true",
    )


def test_leader_node_command_can_disable_hand() -> None:
    assert leader_node_command_tokens(hand_enabled=False)[-1] == "hand_enabled:=false"


def test_leader_node_process_disables_dashboard_for_ui_logs() -> None:
    spec = build_leader_node_process_spec()

    assert spec.env is not None
    assert spec.env["IGRIS_LEADER_DASHBOARD"] == "0"
    assert ".venv" in spec.env["PYTHONPATH"]


def test_managed_command_process_raises_when_command_exits_during_startup() -> None:
    proc = ManagedCommandProcess(
        CommandProcessSpec(
            name="startup_fail",
            argv=(sys.executable, "-u", "-c", "raise SystemExit(7)"),
            cwd=REPO_ROOT,
            startup_grace_s=0.2,
        ),
        log_queue=queue.Queue(),
    )

    with pytest.raises(RuntimeError, match="startup_fail exited during startup with code 7"):
        proc.start()


def test_managed_command_process_startup_error_includes_child_output() -> None:
    proc = ManagedCommandProcess(
        CommandProcessSpec(
            name="startup_detail",
            argv=(
                sys.executable,
                "-u",
                "-c",
                "import sys; print('missing serial permission', file=sys.stderr); raise SystemExit(8)",
            ),
            cwd=REPO_ROOT,
            startup_grace_s=0.2,
        )
    )

    with pytest.raises(RuntimeError, match="missing serial permission"):
        proc.start()


def test_resolve_ros_workspace_root_falls_back_to_sibling_install(monkeypatch, tmp_path: Path) -> None:
    local_root = tmp_path / "project" / "ros_ws"
    sibling_root = tmp_path / "project" / "ros2_ws"
    (sibling_root / "install" / "ros_tcp_endpoint").mkdir(parents=True)
    (sibling_root / "install" / "setup.bash").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "igris_teleop.workers.command_process._ros_workspace_candidates",
        lambda: (local_root, sibling_root),
    )

    assert resolve_ros_workspace_root("ros_tcp_endpoint") == sibling_root


def test_managed_command_process_reuses_existing_tcp_listener() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()
    messages: queue.Queue[str] = queue.Queue()
    proc = ManagedCommandProcess(
        CommandProcessSpec(
            name="reused_listener",
            argv=(sys.executable, "-u", "-c", "raise SystemExit(7)"),
            cwd=REPO_ROOT,
            startup_grace_s=0.2,
            reuse_tcp_listener=(host, port),
        ),
        log_queue=messages,
    )

    try:
        proc.start()
        assert proc.poll() is None
        assert "reusing existing TCP endpoint" in messages.get_nowait()
        proc.terminate()
        assert proc.poll() == 0
        assert listener.fileno() >= 0
    finally:
        listener.close()


def test_managed_command_process_replaces_disappeared_reused_listener() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()
    replacement_script = textwrap.dedent(
        """
        import socket
        import sys
        import time

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind((sys.argv[1], int(sys.argv[2])))
        listener.listen(1)
        print("replacement-ready", flush=True)
        while True:
            time.sleep(0.1)
        """
    ).strip()
    messages: queue.Queue[str] = queue.Queue()
    proc = ManagedCommandProcess(
        CommandProcessSpec(
            name="recovering_listener",
            argv=(sys.executable, "-u", "-c", replacement_script, host, str(port)),
            cwd=REPO_ROOT,
            startup_grace_s=0.2,
            reuse_tcp_listener=(host, port),
        ),
        log_queue=messages,
    )

    try:
        proc.start()
        assert proc.pid is None
        listener.close()

        assert proc.poll() is None
        assert proc.pid is not None
        assert proc.poll() is None
        emitted = []
        while not messages.empty():
            emitted.append(messages.get_nowait())
        assert any("starting a managed replacement" in message for message in emitted)
    finally:
        proc.terminate()
        if listener.fileno() >= 0:
            listener.close()


def test_managed_command_process_poll_detects_unexpected_exit() -> None:
    proc = ManagedCommandProcess(
        CommandProcessSpec(
            name="unexpected_exit",
            argv=(
                sys.executable,
                "-u",
                "-c",
                "import time; print('ready', flush=True); time.sleep(0.15); raise SystemExit(9)",
            ),
            cwd=REPO_ROOT,
            startup_grace_s=0.0,
        ),
        log_queue=queue.Queue(),
    )

    proc.start()
    deadline = time.monotonic() + 2.0
    rc = None
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            break
        time.sleep(0.05)

    assert rc == 9
    assert proc.wait(timeout=0.1) == 9


def test_managed_command_process_terminate_kills_process_group(tmp_path: Path) -> None:
    child_term_file = tmp_path / "child_terminated.txt"
    child_script = textwrap.dedent(
        """
        import pathlib
        import signal
        import sys
        import time

        marker = pathlib.Path(sys.argv[1])

        def _handle_term(_sig, _frame):
            marker.write_text("terminated", encoding="utf-8")
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _handle_term)
        signal.signal(signal.SIGINT, _handle_term)
        print("child-ready", flush=True)
        while True:
            time.sleep(0.1)
        """
    ).strip()
    parent_script = textwrap.dedent(
        """
        import subprocess
        import sys
        import time

        marker = sys.argv[1]
        child_script = sys.argv[2]
        subprocess.Popen([sys.executable, "-u", "-c", child_script, marker])
        print("parent-ready", flush=True)
        while True:
            time.sleep(0.1)
        """
    ).strip()

    proc = ManagedCommandProcess(
        CommandProcessSpec(
            name="process_group",
            argv=(
                sys.executable,
                "-u",
                "-c",
                parent_script,
                str(child_term_file),
                child_script,
            ),
            cwd=REPO_ROOT,
            startup_grace_s=0.2,
        ),
        log_queue=queue.Queue(),
    )

    proc.start()
    proc.terminate()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not child_term_file.exists():
        time.sleep(0.05)

    assert child_term_file.read_text(encoding="utf-8") == "terminated"
