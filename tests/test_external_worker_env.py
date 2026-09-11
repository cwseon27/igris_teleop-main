from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.modules.setdefault(
    "logging_mp",
    types.SimpleNamespace(
        INFO=logging.INFO,
        _log_queue=None,
        basic_config=lambda level=None: logging.basicConfig(level=level),
        get_logger=lambda name=None, level=None: logging.getLogger(name),
    ),
)

from igris_teleop import main
from igris_teleop.workers import registry


def test_external_worker_env_isolates_python_user_site(monkeypatch, tmp_path: Path) -> None:
    conda_prefix = tmp_path / ".conda-ik"
    conda_bin = conda_prefix / "bin"
    conda_lib = conda_prefix / "lib"
    conda_bin.mkdir(parents=True)
    conda_lib.mkdir()
    (conda_prefix / "conda-meta").mkdir()

    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(("/opt/ros/jazzy/lib/python3.12/site-packages", "/tmp/keep")))
    monkeypatch.setenv("LD_LIBRARY_PATH", os.pathsep.join(("/opt/openrobots/lib", "/tmp/libkeep")))
    monkeypatch.setenv("ROS_DISTRO", "jazzy")
    monkeypatch.setenv("AMENT_PREFIX_PATH", "/opt/ros/jazzy")
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/parent-venv")
    monkeypatch.setenv("PYTHONHOME", "/tmp/parent-pythonhome")

    env = main._build_external_worker_env(str(conda_bin / "python"))

    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONPATH"] == "/tmp/keep"
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [str(conda_lib), "/tmp/libkeep"]
    assert env["PATH"].split(os.pathsep)[0] == str(conda_bin)
    assert env["CONDA_PREFIX"] == str(conda_prefix)
    assert "VIRTUAL_ENV" not in env
    assert "PYTHONHOME" not in env
    assert "ROS_DISTRO" not in env
    assert "AMENT_PREFIX_PATH" not in env


def test_real_hand_external_env_restores_only_required_ros_runtime(monkeypatch, tmp_path: Path) -> None:
    ml_prefix = tmp_path / ".venv-ml"
    (ml_prefix / "bin").mkdir(parents=True)
    (ml_prefix / "lib").mkdir()
    (ml_prefix / "pyvenv.cfg").write_text("", encoding="utf-8")

    ros_prefix = tmp_path / "opt" / "ros" / "jazzy"
    ros_python = (
        ros_prefix
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    ros_python.mkdir(parents=True)
    monkeypatch.setenv("IGRIS_ROS_PREFIX", str(ros_prefix))
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(ros_prefix))
    monkeypatch.setenv("PYTHONPATH", "/tmp/ml-extra")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/ml-lib")

    env = main._build_external_worker_env(
        str(ml_prefix / "bin" / "python"),
        worker_name="hand",
        run_config=SimpleNamespace(runtime_environment="real"),
    )

    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(ros_python)
    assert env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(ros_prefix / "lib")
    assert "AMENT_PREFIX_PATH" not in env


def test_default_ik_python_uses_ik_venv(monkeypatch, tmp_path: Path) -> None:
    ik_python = tmp_path / ".venv-ik" / "bin" / "python"

    monkeypatch.delenv("IGRIS_IK_PYTHON", raising=False)
    monkeypatch.delenv("IGRIS_GEOM_PYTHON", raising=False)
    monkeypatch.setattr(registry, "REPO_ROOT", tmp_path)

    assert registry._default_ik_python() == str(ik_python.absolute())


def test_default_ik_python_env_override_wins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IGRIS_IK_PYTHON", "/tmp/custom-ik-python")
    monkeypatch.setattr(registry, "REPO_ROOT", tmp_path)

    assert registry._default_ik_python() == "/tmp/custom-ik-python"


def test_default_ml_python_uses_ml_venv(monkeypatch, tmp_path: Path) -> None:
    ml_python = tmp_path / ".venv-ml" / "bin" / "python"

    monkeypatch.delenv("IGRIS_ML_PYTHON", raising=False)
    monkeypatch.setattr(registry, "REPO_ROOT", tmp_path)

    assert registry._default_ml_python() == str(ml_python.absolute())


def test_default_ml_python_env_override_wins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IGRIS_ML_PYTHON", "/tmp/custom-ml-python")
    monkeypatch.setattr(registry, "REPO_ROOT", tmp_path)

    assert registry._default_ml_python() == "/tmp/custom-ml-python"


def test_ml_workers_are_external() -> None:
    for name in ("hand", "collect_data", "inference_lerobot", "inference_optimize", "replay"):
        spec = registry.WORKER_SPECS[name]
        assert spec.external is True
        assert spec.python == registry._default_ml_python()
