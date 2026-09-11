from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import igris_teleop.web_ui.server as web_ui_server
from igris_teleop.web_ui.server import WebUIState, _decode_uint8_text


class FakeBus:
    def __init__(self) -> None:
        self.levels = {name: False for name in web_ui_server.LEVEL_EVENTS}

    def set_level(self, name: str) -> None:
        self.levels[name] = True

    def clear_level(self, name: str) -> None:
        self.levels[name] = False

    def is_level_set(self, name: str) -> bool:
        return self.levels[name]


class FakeSupervisor:
    def __init__(self) -> None:
        self.mode_running: set[str] = set()
        self.manual_running: set[str] = set()
        self.applied_payload = None
        self.camera_mode = "camera_pub"
        self.collect_alive = False
        self.fail_on_start: str | None = None

    def alive_map(self) -> dict[str, bool]:
        return {
            "collect_data": self.collect_alive,
            "control": "control" in self.manual_running,
            "hand": "hand" in self.manual_running,
            "leader_ros_node": "leader_ros_node" in self.manual_running,
            "leader_ros_tcp_endpoint": "leader_ros_tcp_endpoint" in self.manual_running,
            "simulator": "simulator" in self.manual_running,
        }

    def current_manual_workers(self) -> set[str]:
        return set(self.manual_running)

    def current_mode_workers(self) -> set[str]:
        return set(self.mode_running)

    def list_manual_workers(self) -> list[str]:
        return [
            "simulator",
            "control",
            "hand",
            "leader_ros_tcp_endpoint",
            "leader_ros_node",
            "collect_data",
        ]

    def list_selectable_mode_workers(self, mode, teleop_device, inference_policy=None) -> list[str]:
        if mode == "teleop" and teleop_device in {"unity", "unity_hybrid"}:
            return ["igris_ik", "unity_bridge"]
        if mode == "teleop" and teleop_device == "vr_masterarm":
            return ["igris_ik", "master_arm_ros_bridge", "unity_bridge"]
        if mode == "teleop" and teleop_device == "masterarm":
            return ["master_arm_ros_bridge"]
        if mode == "walking":
            return ["walking_logger", "walking_policy"]
        if mode == "inference" and inference_policy in {"pi0", "pi0.5"}:
            return ["inference_optimize", "inference_pi"]
        if mode == "inference":
            return ["inference_lerobot", "inference_optimize"]
        return []

    def apply_camera_settings(self, camera_mode: str) -> str:
        self.camera_mode = camera_mode
        return camera_mode

    def apply_mode_workers(self, **kwargs):
        self.applied_payload = kwargs
        self.mode_running = set(kwargs["selected_mode_workers"])
        return set(self.mode_running)

    def start_manual_worker(self, name: str, **kwargs) -> None:
        if name == self.fail_on_start:
            raise RuntimeError(f"failed to start {name}")
        self.manual_running.add(name)

    def stop_manual_worker(self, name: str) -> None:
        self.manual_running.discard(name)


class FakeShm:
    def __init__(self, data=None) -> None:
        self.data = dict(data or {})
        self.writes: list[dict] = []

    def read_data(self):
        return dict(self.data)

    def write_data(self, **kwargs) -> None:
        self.writes.append(kwargs)
        self.data.update(kwargs)


def make_state(supervisor: FakeSupervisor | None = None, shared_memory=None) -> WebUIState:
    return WebUIState(
        bus=FakeBus(),
        supervisor=supervisor or FakeSupervisor(),
        shared_memory=shared_memory or {},
        log_queue=None,
        initial_mode=None,
        initial_teleop_device=None,
        initial_teleop_hand_source=None,
        initial_walking_policy_profile="v2_fast_sac",
        initial_walking_policy_path=None,
        initial_camera_mode="camera_pub",
    )


def test_reliability_launch_explicitly_uses_plural_controller_topics() -> None:
    hybrid = web_ui_server.HybridTeleopConfig(
        left_device=4,
        right_device=6,
        trapezoid_preprocess=True,
        trapezoid_bottom_width=288,
        swap_lr=True,
        mirror=False,
    )
    spec = web_ui_server._build_reliability_process_spec(
        hand_model_variant="rnn",
        controller_model_variant="rnn",
        hybrid_config=hybrid,
    )
    command = " ".join(spec.argv)

    assert "left_controller_pose_topic:=/left_controller/poses" in command
    assert "right_controller_pose_topic:=/right_controller/poses" in command
    assert "left_camera_device:=4" in command
    assert "right_camera_device:=6" in command
    assert "left_camera_mirror:=false" in command
    assert "left_camera_swap_lr:=true" in command
    assert "trapezoid_preprocess:=true" in command
    assert "trapezoid_bottom_width:=288" in command


def test_reliability_runtime_and_policy_assets_are_repository_local() -> None:
    project_root = Path(__file__).resolve().parents[1]

    assert web_ui_server.RELIABILITY_ROS_WS_ROOT == (project_root / "ros_ws").resolve()
    assert web_ui_server.RELIABILITY_POLICY_ROOT == (
        project_root / "policy_archive" / "reliability"
    ).resolve()

    for package_name in (
        "igris_reliability_runtime",
        "mediapipe_hand_pose_bridge",
        "openxr_hand_to_igris_viewer",
    ):
        assert (project_root / "ros_ws" / "src" / package_name / "package.xml").is_file()

    for kind in ("hand", "controller"):
        for variant in ("rnn", "histgb"):
            assert (
                project_root
                / "policy_archive"
                / "reliability"
                / kind
                / f"{kind}_{variant}.joblib"
            ).is_file()

    spec = web_ui_server._build_reliability_process_spec(
        hand_model_variant="rnn",
        controller_model_variant="rnn",
    )
    assert spec.cwd == (project_root / "ros_ws").resolve()
    assert spec.env is not None
    assert spec.env["IGRIS_PROJECT_ROOT"] == str(project_root)
    assert spec.env["IGRIS_RELIABILITY_POLICY_ROOT"] == str(
        (project_root / "policy_archive" / "reliability").resolve()
    )
    assert spec.env["IGRIS_RELIABILITY_PYTHON"] == str(
        project_root / ".venv-ml" / "bin" / "python"
    )
    assert spec.env["IGRIS_MEDIAPIPE_PYTHON"] == str(
        project_root / ".venv-mediapipe" / "bin" / "python"
    )


def test_reliability_always_one_variants_are_exposed_and_forwarded() -> None:
    assert "always_1" in web_ui_server.RELIABILITY_MODEL_VARIANTS
    assert "always_1" in make_state().status()["options"]["reliability_model_variants"]
    assert (
        web_ui_server._validate_reliability_model_variant(
            "always_1",
            field_name="hand_model_variant",
        )
        == "always_1"
    )

    spec = web_ui_server._build_reliability_process_spec(
        hand_model_variant="always_1",
        controller_model_variant="histgb",
    )
    command = " ".join(spec.argv)

    assert "hand_model_variant:=always_1" in command
    assert "controller_model_variant:=histgb" in command


def test_reliability_model_selection_is_not_overwritten_while_stopped() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_js = (
        project_root / "igris_teleop" / "web_ui" / "static" / "app.js"
    ).read_text(encoding="utf-8")

    assert "runtime?.hand_model_variant && !running" not in app_js
    assert "runtime?.controller_model_variant && !running" not in app_js
    assert "runtime?.hand_model_variant && running" in app_js
    assert "runtime?.controller_model_variant && running" in app_js


def test_hybrid_teleop_defaults_match_established_camera_pipeline() -> None:
    config = web_ui_server.HybridTeleopConfig()

    assert config.left_device == 1
    assert config.right_device == 0
    assert config.trapezoid_preprocess is True
    assert config.trapezoid_bottom_width == 320
    assert config.swap_lr is True
    assert config.mirror is False


def test_hybrid_mediapipe_test_uses_selected_settings() -> None:
    config = web_ui_server.HybridTeleopConfig(
        left_device=2,
        right_device=5,
        trapezoid_preprocess=False,
        trapezoid_bottom_width=400,
        swap_lr=False,
        mirror=True,
    )

    spec = web_ui_server._build_hybrid_test_process_spec(config)
    command = " ".join(spec.argv)

    assert "left_device:=2" in command
    assert "right_device:=5" in command
    assert "left_mirror:=true" in command
    assert "left_swap_lr:=false" in command
    assert "trapezoid_preprocess:=false" in command
    assert "trapezoid_bottom_width:=400" in command
    assert "preview_output_dir:=" in command


def test_hybrid_config_rejects_duplicate_devices_and_invalid_bottom_width() -> None:
    with pytest.raises(ValueError, match="different device"):
        web_ui_server._hybrid_config_from_mapping({"left_device": 1, "right_device": 1})

    with pytest.raises(ValueError, match="between 1 and 640"):
        web_ui_server._hybrid_config_from_mapping({"trapezoid_bottom_width": 641})


def test_camera_selection_persists_port_path_across_renumbering(monkeypatch, tmp_path) -> None:
    left = "/dev/v4l/by-path/usb-left-video-index0"
    right = "/dev/v4l/by-path/usb-right-video-index0"
    monkeypatch.setattr(web_ui_server, "_stable_camera_devices", lambda: {
        "/dev/video5": left, "/dev/video2": right,
    })
    config = web_ui_server._hybrid_config_from_mapping({"left_device": "5", "right_device": 2})
    path = tmp_path / "hybrid.json"
    web_ui_server._save_hybrid_config(config, path)
    monkeypatch.setattr(web_ui_server, "_stable_camera_devices", lambda: {
        "/dev/video7": left, "/dev/video3": right,
    })
    reloaded = web_ui_server._load_hybrid_config(path)
    assert reloaded.left_device == left
    assert reloaded.right_device == right
    command = " ".join(web_ui_server._build_reliability_process_spec(
        hand_model_variant="always_1", controller_model_variant="always_1", hybrid_config=reloaded,
    ).argv)
    assert f"left_camera_device:={left}" in command
    assert f"right_camera_device:={right}" in command
    test_command = " ".join(web_ui_server._build_hybrid_test_process_spec(reloaded).argv)
    assert f"left_device:={left}" in test_command
    assert f"right_device:={right}" in test_command


def test_camera_config_retains_unplugged_port_and_rejects_alias_duplicates(monkeypatch) -> None:
    left = "/dev/v4l/by-path/usb-left-video-index0"
    right = "/dev/v4l/by-path/usb-right-video-index0"
    monkeypatch.setattr(web_ui_server, "_stable_camera_devices", lambda: {})
    config = web_ui_server._hybrid_config_from_mapping({"left_device": left, "right_device": right})
    state = make_state()
    state.hybrid_config = config
    assert {left, right} <= set(state._hybrid_camera_device_choices())
    monkeypatch.setattr(web_ui_server, "_stable_camera_devices", lambda: {"/dev/video2": right})
    with pytest.raises(ValueError, match="different devices"):
        web_ui_server._hybrid_config_from_mapping({"left_device": 2, "right_device": right})


def test_camera_discovery_deduplicates_udev_links_and_excludes_metadata(monkeypatch, tmp_path) -> None:
    node = tmp_path / "video2"
    node.touch()
    links = tmp_path / "by-path"
    links.mkdir()
    (links / "pci-usb-video-index0").symlink_to(node)
    (links / "pci-usbv2-video-index0").symlink_to(node)
    (links / "pci-usb-video-index1").symlink_to(tmp_path / "video3")
    monkeypatch.setattr(web_ui_server, "V4L_BY_PATH", links)
    assert web_ui_server._stable_camera_devices() == {str(node): str(links / "pci-usb-video-index0")}


def test_hybrid_mediapipe_test_reports_unexpected_launcher_exit() -> None:
    class ExitedProcess:
        pid = 123

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait(timeout: float) -> int:
            assert timeout == 0.0
            return 0

    manager = web_ui_server.HybridMediaPipeTestManager()
    manager._process = ExitedProcess()
    manager._started_at = time.time()

    status = manager.status()

    assert status["running"] is False
    assert status["last_exit_code"] == 0
    assert "check camera devices" in status["last_error"]


def test_apply_mode_defaults_to_selectable_workers_and_resolves_hand_source() -> None:
    supervisor = FakeSupervisor()
    state = make_state(supervisor)

    status = state.apply_mode({"mode": "teleop", "teleop_device": "unity"})

    assert supervisor.applied_payload["selected_mode_workers"] == {"igris_ik", "unity_bridge"}
    assert supervisor.applied_payload["teleop_hand_source"] == "vr"
    assert state.applied_mode == "teleop"
    assert status["applied"]["teleop_hand_source"] == "vr"
    assert status["ui"]["selection_locked"] is True


def test_status_includes_runtime_diagnostics_rows() -> None:
    supervisor = FakeSupervisor()
    supervisor.manual_running.add("control")
    supervisor.runtime_diagnostics = {
        "control:fast": {
            "key": "control:fast",
            "worker": "control",
            "loop": "fast",
            "target_hz": 100.0,
            "actual_hz": 99.8,
            "period_ms": 10.02,
            "jitter_ms": 0.12,
            "latency_ms": 1.4,
            "max_jitter_ms": 0.5,
            "max_latency_ms": 2.0,
            "samples": 120,
            "updated_at": time.time(),
            "stopped": False,
        }
    }
    state = make_state(supervisor)

    status = state.status()

    row = status["runtime_diagnostics"]["rows"][0]
    assert row["key"] == "control:fast"
    assert row["status"] == "OK"
    assert row["actual_hz"] == 99.8
    assert row["latency_ms"] == 1.4


def test_sim_stereo_eye_distance_is_sim_only_and_written_to_shm() -> None:
    supervisor = FakeSupervisor()
    sim_config_shm = FakeShm({"stereo_baseline_m": 0.120})
    state = make_state(supervisor, {"sim_config_shm": sim_config_shm})

    with pytest.raises(ValueError, match="Start the simulator"):
        state.set_sim_stereo({"baseline_m": 0.064})

    supervisor.manual_running.add("simulator")
    status = state.set_sim_stereo({"baseline_m": 0.064})

    assert sim_config_shm.data["stereo_baseline_m"] == 0.064
    assert status["sim_stereo"] == {
        "enabled": True,
        "baseline_m": 0.064,
        "min_baseline_m": 0.04,
        "max_baseline_m": 0.2,
        "calibration_map_applied": False,
        "extrinsic_mode": "translation_only",
    }


def test_sim_stereo_invalid_shared_value_is_initialized_to_default() -> None:
    sim_config_shm = FakeShm({"stereo_baseline_m": 0.0})

    state = make_state(shared_memory={"sim_config_shm": sim_config_shm})

    assert sim_config_shm.data["stereo_baseline_m"] == 0.120
    assert state.status()["sim_stereo"]["baseline_m"] == 0.120


def test_experiment_scene_requires_simulator_and_sends_ordered_commands() -> None:
    supervisor = FakeSupervisor()
    sim_config_shm = FakeShm(
        {
            "stereo_baseline_m": 0.120,
            "scene_command_seq": 0.0,
            "scene_task_id": 0.0,
            "scene_applied_seq": 0.0,
            "scene_active_task_id": 0.0,
            "scene_status_code": 0.0,
        }
    )
    state = make_state(supervisor, {"sim_config_shm": sim_config_shm})

    with pytest.raises(ValueError, match="Start the simulator"):
        state.set_experiment_scene({"action": "spawn", "task_id": 1})

    supervisor.manual_running.add("simulator")
    status = state.set_experiment_scene({"action": "spawn", "task_id": 3})

    assert sim_config_shm.data["scene_command_seq"] == 1.0
    assert sim_config_shm.data["scene_task_id"] == 3.0
    assert sim_config_shm.data["scene_status_code"] == 1.0
    assert status["experiments"]["state"] == "PENDING"
    assert len(status["experiments"]["tasks"]) == 4

    sim_config_shm.data.update(
        scene_applied_seq=1.0,
        scene_active_task_id=3.0,
        scene_status_code=2.0,
    )
    status = state.set_experiment_scene({"action": "task_reset", "task_id": 3})
    assert sim_config_shm.data["scene_command_seq"] == 2.0
    assert sim_config_shm.data["scene_task_id"] == 3.0
    assert status["experiments"]["pending"] is True

    sim_config_shm.data["scene_applied_seq"] = 2.0
    status = state.set_experiment_scene({"action": "all_reset"})
    assert sim_config_shm.data["scene_command_seq"] == 3.0
    assert sim_config_shm.data["scene_task_id"] == 0.0
    assert status["experiments"]["requested_task_id"] == 0


def test_experiment_scene_rejects_unknown_task() -> None:
    supervisor = FakeSupervisor()
    supervisor.manual_running.add("simulator")
    state = make_state(
        supervisor,
        {
            "sim_config_shm": FakeShm(
                {
                    "stereo_baseline_m": 0.120,
                    "scene_command_seq": 0.0,
                    "scene_applied_seq": 0.0,
                }
            )
        },
    )

    with pytest.raises(ValueError, match="Unknown experiment task"):
        state.set_experiment_scene({"action": "spawn", "task_id": 99})


def test_leader_ros_unity_starts_endpoint_only() -> None:
    supervisor = FakeSupervisor()
    state = make_state(supervisor)

    status = state.toggle_manual_group(
        {"group": "leader_ros", "action": "start", "mode": "teleop", "teleop_device": "unity"}
    )

    assert supervisor.manual_running == {"leader_ros_tcp_endpoint"}
    leader_group = next(group for group in status["workers"]["manual_groups"] if group["name"] == "leader_ros")
    assert leader_group["workers"] == ["leader_ros_tcp_endpoint"]
    assert leader_group["running"] is True


def test_unity_hybrid_defaults_to_vr_hand_and_unity_workers() -> None:
    supervisor = FakeSupervisor()
    state = make_state(supervisor)

    status = state.apply_mode({"mode": "teleop", "teleop_device": "unity_hybrid"})

    assert supervisor.applied_payload["selected_mode_workers"] == {"igris_ik", "unity_bridge"}
    assert supervisor.applied_payload["teleop_hand_source"] == "vr"
    assert status["applied"]["teleop_device"] == "unity_hybrid"
    assert status["applied"]["teleop_hand_source"] == "vr"


def test_vr_masterarm_defaults_to_vr_hand_and_both_input_bridges() -> None:
    supervisor = FakeSupervisor()
    state = make_state(supervisor)

    status = state.apply_mode({"mode": "teleop", "teleop_device": "vr_masterarm"})

    assert supervisor.applied_payload["selected_mode_workers"] == {
        "igris_ik",
        "master_arm_ros_bridge",
        "unity_bridge",
    }
    assert supervisor.applied_payload["teleop_hand_source"] == "vr"
    assert status["applied"]["teleop_device"] == "vr_masterarm"
    assert status["applied"]["teleop_hand_source"] == "vr"


def test_leader_ros_unity_hybrid_starts_endpoint_only() -> None:
    supervisor = FakeSupervisor()
    state = make_state(supervisor)

    status = state.toggle_manual_group(
        {"group": "leader_ros", "action": "start", "mode": "teleop", "teleop_device": "unity_hybrid"}
    )

    assert supervisor.manual_running == {"leader_ros_tcp_endpoint"}
    leader_group = next(group for group in status["workers"]["manual_groups"] if group["name"] == "leader_ros")
    assert leader_group["workers"] == ["leader_ros_tcp_endpoint"]
    assert leader_group["running"] is True


def test_leader_ros_switch_to_masterarm_stops_endpoint_and_starts_node() -> None:
    supervisor = FakeSupervisor()
    supervisor.manual_running = {"leader_ros_tcp_endpoint"}
    state = make_state(supervisor)

    state.toggle_manual_group(
        {"group": "leader_ros", "action": "start", "mode": "teleop", "teleop_device": "masterarm"}
    )

    assert supervisor.manual_running == {"leader_ros_node"}


def test_leader_ros_start_rolls_back_endpoint_when_node_fails() -> None:
    supervisor = FakeSupervisor()
    supervisor.fail_on_start = "leader_ros_node"
    state = make_state(supervisor)

    with pytest.raises(RuntimeError, match="failed to start leader_ros_node"):
        state.toggle_manual_group(
            {
                "group": "leader_ros",
                "action": "start",
                "mode": "teleop",
                "teleop_device": "vr_masterarm",
                "teleop_hand_source": "vr",
            }
        )

    assert supervisor.manual_running == set()


def test_start_level_requires_applied_mode_and_ready() -> None:
    state = make_state()

    with pytest.raises(ValueError, match="Apply mode workers"):
        state.set_level({"name": "start", "value": True})


def test_ready_is_gated_until_mode_is_applied() -> None:
    state = make_state()

    status = state.select_mode({"mode": "teleop", "teleop_device": "unity"})

    assert status["selected"]["mode"] == "teleop"
    assert status["ui"]["ready_enabled"] is False
    with pytest.raises(ValueError, match="Apply mode workers"):
        state.set_level({"name": "ready", "value": True})

    state.bus.set_level("ready")
    status = state.status()
    assert status["levels"]["ready"] is False

    status = state.apply_mode({"mode": "teleop", "teleop_device": "unity"})
    assert status["ui"]["ready_enabled"] is True
    status = state.set_level({"name": "ready", "value": True})
    assert status["levels"]["ready"] is True
    assert status["ui"]["start_enabled"] is True


def test_hand_init_level_requires_ready_and_hand_worker() -> None:
    supervisor = FakeSupervisor()
    state = make_state(supervisor)

    state.apply_mode({"mode": "teleop", "teleop_device": "unity"})

    with pytest.raises(ValueError, match="ready"):
        state.set_level({"name": "hand_init", "value": True})

    state.set_level({"name": "ready", "value": True})
    with pytest.raises(ValueError, match="hand worker"):
        state.set_level({"name": "hand_init", "value": True})

    supervisor.manual_running.add("hand")
    status = state.set_level({"name": "hand_init", "value": True})

    assert status["levels"]["hand_init"] is True
    assert status["ui"]["hand_init_enabled"] is True

    state.set_level({"name": "hand_init", "value": False})
    state.bus.set_level("start")
    with pytest.raises(ValueError, match="start/home"):
        state.set_level({"name": "hand_init", "value": True})


def test_inference_policy_selection_controls_mode_workers() -> None:
    supervisor = FakeSupervisor()
    state = make_state(supervisor)

    status = state.select_mode({"mode": "inference", "inference_policy": "pi0"})

    worker_names = {worker["name"] for worker in status["workers"]["mode_workers"]["workers"]}
    assert worker_names == {"inference_optimize", "inference_pi"}

    status = state.apply_mode({"mode": "inference", "inference_policy": "pi0"})

    assert supervisor.applied_payload["selected_mode_workers"] == {"inference_optimize", "inference_pi"}
    assert supervisor.applied_payload["inference_policy"] == "pi0"
    assert status["applied"]["inference_policy"] == "pi0"


def test_walking_start_requires_applied_mode_start_level_and_zero_command() -> None:
    walking_cmd_shm = FakeShm({"vx": 0.2, "vy": 0.0, "dyaw": 0.0, "policy_enabled": 1.0})
    state = make_state(shared_memory={"walking_cmd_shm": walking_cmd_shm})

    state.apply_mode({"mode": "walking", "walking_policy_profile": "v2_fast_sac"})
    assert walking_cmd_shm.data["vx"] == 0.0
    assert walking_cmd_shm.data["policy_enabled"] == 0.0

    with pytest.raises(ValueError, match="start first"):
        state.write_walking_command({"vx": 0.0, "vy": 0.0, "dyaw": 0.0, "policy_enabled": True})

    state.set_level({"name": "ready", "value": True})
    state.set_level({"name": "start", "value": True})

    with pytest.raises(ValueError, match="zero"):
        state.write_walking_command({"vx": 0.1, "vy": 0.0, "dyaw": 0.0, "policy_enabled": True})

    state.write_walking_command({"vx": 0.0, "vy": 0.0, "dyaw": 0.0, "policy_enabled": True})
    assert walking_cmd_shm.data["policy_enabled"] == 1.0


def test_file_browser_returns_relative_dataset_selection(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "datasets"
    target = dataset_root / "run_a" / "episode_0"
    target.mkdir(parents=True)
    monkeypatch.setattr(web_ui_server, "DATASETS_ROOT", dataset_root)
    state = make_state()

    payload = state.file_browser(kind="inference_dataset", raw_path="run_a")

    assert payload["cwd_display"] == "run_a"
    entries = {entry["name"]: entry for entry in payload["entries"]}
    assert entries["episode_0"]["selectable"] is True
    assert entries["episode_0"]["display"] == "run_a/episode_0"
    assert payload["parent"] is not None


def test_file_browser_can_navigate_above_default_dataset_root(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "datasets"
    external_root = tmp_path / "external"
    dataset_root.mkdir()
    external_root.mkdir()
    monkeypatch.setattr(web_ui_server, "DATASETS_ROOT", dataset_root)
    state = make_state()

    payload = state.file_browser(kind="inference_dataset", raw_path=str(external_root))

    assert payload["base_root"] == str(tmp_path.resolve())
    assert payload["cwd"] == str(external_root.resolve())
    assert payload["current_value"] == str(external_root.resolve())


def test_file_browser_returns_absolute_walking_model_selection(tmp_path, monkeypatch) -> None:
    model_root = tmp_path / "walking"
    model = model_root / "policy_1.pt"
    onnx_model = model_root / "model_0039000.onnx"
    model_root.mkdir()
    model.write_bytes(b"")
    onnx_model.write_bytes(b"")
    monkeypatch.setattr(web_ui_server, "default_walking_policy_path", lambda profile=None: model)
    state = make_state()

    payload = state.file_browser(kind="walking_policy", profile="v1")

    entries = {entry["name"]: entry for entry in payload["entries"]}
    assert entries["policy_1.pt"]["selectable"] is True
    assert entries["model_0039000.onnx"]["selectable"] is True
    assert entries["policy_1.pt"]["display"] == str(model.resolve())


def test_file_browser_dataset_viewer_root_returns_absolute_directory(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "datasets"
    target = dataset_root / "run_a"
    target.mkdir(parents=True)
    monkeypatch.setattr(web_ui_server, "DATASETS_ROOT", dataset_root)
    state = make_state()

    payload = state.file_browser(kind="dataset_viewer_root")

    entries = {entry["name"]: entry for entry in payload["entries"]}
    assert entries["run_a"]["selectable"] is True
    assert entries["run_a"]["display"] == str(target.resolve())
    assert payload["current_value"] == str(dataset_root.resolve())
    assert payload["parent"] == str(tmp_path.resolve())


def _write_dataset_info(dataset_dir, *, total_episodes=1, total_frames=2) -> None:
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(
        web_ui_server.json.dumps(
            {
                "fps": 30,
                "total_episodes": total_episodes,
                "total_frames": total_frames,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [31]},
                    "action": {"dtype": "float32", "shape": [31]},
                    "observation.image.front": {"dtype": "image", "shape": [4, 4, 3]},
                },
            }
        ),
        encoding="utf-8",
    )


def test_dataset_viewer_lists_nested_lerobot_datasets(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_dir = dataset_root / "group_a" / "run_001"
    _write_dataset_info(dataset_dir)
    monkeypatch.setattr(web_ui_server, "DATASETS_ROOT", dataset_root)
    state = make_state()

    payload = state.dataset_viewer_datasets()

    assert payload["root"] == str(dataset_root.resolve())
    assert len(payload["datasets"]) == 1
    assert payload["datasets"][0]["display"] == "group_a/run_001"
    assert payload["datasets"][0]["image_keys"] == ["observation.image.front"]


def test_dataset_viewer_episode_uses_lerobot_dataset(monkeypatch, tmp_path) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_dir = dataset_root / "run_001"
    _write_dataset_info(dataset_dir)
    monkeypatch.setattr(web_ui_server, "DATASETS_ROOT", dataset_root)

    class FakeDataset:
        def __init__(self, *args, **kwargs) -> None:
            self.hf_dataset = {"episode_index": [0, 0]}
            self.rows = [
                {
                    "episode_index": 0,
                    "frame_index": 0,
                    "timestamp": 0.0,
                    "observation.state": np.arange(31, dtype=np.float32),
                    "action": np.arange(31, dtype=np.float32) * 0.1,
                    "observation.image.front": np.zeros((4, 4, 3), dtype=np.uint8),
                },
                {
                    "episode_index": 0,
                    "frame_index": 1,
                    "timestamp": 1.0 / 30.0,
                    "observation.state": np.arange(31, dtype=np.float32) + 1,
                    "action": np.arange(31, dtype=np.float32) * 0.1 + 1,
                    "observation.image.front": np.ones((4, 4, 3), dtype=np.uint8) * 255,
                },
            ]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[int(idx)]

    monkeypatch.setattr(web_ui_server, "_require_lerobot_dataset", lambda: FakeDataset)
    state = make_state()

    payload = state.dataset_viewer_episode(dataset="run_001", episode=0)

    assert payload["frame_count"] == 2
    assert payload["frame_numbers_full"] == [0, 1]
    assert payload["image_key"] == "observation.image.front"
    assert {plot["name"] for plot in payload["plots"]} == {"hand", "arm", "neck", "waist"}
    assert payload["dimensions"]["observation"] == 31
    assert payload["segments"][0]["class_name"] == "unassigned"


def test_dataset_viewer_falls_back_when_lerobot_is_missing(monkeypatch, tmp_path) -> None:
    dataset_dir = tmp_path / "run_001"
    dataset_dir.mkdir()
    state = make_state()

    class FakeDirectDataset:
        hf_dataset = [{"episode_index": 0}, {"episode_index": 0}]

        def __len__(self) -> int:
            return 2

        def __getitem__(self, idx):
            return self.hf_dataset[int(idx)]

    def missing_lerobot():
        raise RuntimeError("missing lerobot")

    monkeypatch.setattr(web_ui_server, "_require_lerobot_dataset", missing_lerobot)
    monkeypatch.setattr(
        WebUIState,
        "_open_local_parquet_dataset",
        lambda self, dataset_dir_arg, episode: FakeDirectDataset(),
    )

    dataset, rows = state._open_dataset_viewer_episode(dataset_dir, 0)

    assert isinstance(dataset, FakeDirectDataset)
    assert rows == [0, 1]


def test_local_dataset_uses_episode_video_metadata_for_frame_position(tmp_path) -> None:
    dataset_dir = tmp_path / "run_001"
    video_path = dataset_dir / "videos" / "observation.image.front" / "chunk-000" / "file-000.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"")
    dataset = object.__new__(web_ui_server._LocalLeRobotParquetDataset)
    dataset.root = dataset_dir.resolve()
    dataset.info = {
        "fps": 30.0,
        "chunks_size": 1000,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    dataset.episode = 2
    dataset.image_keys = ["observation.image.front"]
    dataset._video_files_by_key = {}
    dataset.episode_metadata = {
        2: {
            "videos/observation.image.front/chunk_index": 0,
            "videos/observation.image.front/file_index": 0,
            "videos/observation.image.front/from_timestamp": 10.0,
        }
    }
    row = {
        "episode_index": 2,
        "frame_index": 3,
        "timestamp": 0.1,
        "index": 303,
    }

    resolved = dataset._resolve_video_path(row, "observation.image.front")
    frame_index, timestamp = dataset._video_frame_position(row, "observation.image.front", fps=30.0)

    assert resolved == video_path.resolve()
    assert frame_index == 303
    assert timestamp == pytest.approx(10.1)


def test_dataset_viewer_save_labels_merges_sidecar_and_rewrites(monkeypatch, tmp_path) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_dir = dataset_root / "run_001"
    _write_dataset_info(dataset_dir, total_episodes=2, total_frames=6)
    sidecar_path = dataset_dir / "meta" / "annotation_segment_labels.json"
    sidecar_path.write_text(
        web_ui_server.json.dumps(
            {
                "feature_key": web_ui_server.LABEL_FEATURE_KEY,
                "classes": [{"id": 0, "name": "success"}, {"id": 1, "name": "failure"}],
                "episodes": {
                    "1": [{"start_frame": 0, "end_frame": 2, "class_id": 1}],
                },
            }
        ),
        encoding="utf-8",
    )
    state = make_state()

    class FakeDataset:
        def close(self) -> None:
            pass

    def fake_arrays(self, dataset, row_positions, ee_feature_key):
        return (
            np.zeros((3, 0), dtype=np.float32),
            np.zeros((3, 0), dtype=np.float32),
            np.zeros((3, 0), dtype=np.float32),
            np.full((3,), web_ui_server.UNASSIGNED_CLASS_ID, dtype=np.int64),
            np.asarray([0, 1, 2], dtype=np.int64),
            np.asarray([0.0, 1.0 / 30.0, 2.0 / 30.0], dtype=np.float32),
        )

    captured = {}

    def fake_rewrite(dataset_dir_arg, *, classes, episode_segments):
        captured["dataset_dir"] = dataset_dir_arg
        captured["classes"] = classes
        captured["episode_segments"] = episode_segments
        return {"backup_dir": str(tmp_path / "backup"), "rewritten_files": 4}

    monkeypatch.setattr(web_ui_server, "_dataset_viewer_external_python", lambda: None)
    monkeypatch.setattr(web_ui_server, "get_labeling_backend_status", lambda: (True, "ok"))
    monkeypatch.setattr(
        WebUIState,
        "_open_dataset_viewer_episode",
        lambda self, dataset_dir_arg, episode: (FakeDataset(), [0, 1, 2]),
    )
    monkeypatch.setattr(WebUIState, "_load_dataset_viewer_episode_arrays", fake_arrays)
    monkeypatch.setattr(web_ui_server, "rewrite_dataset_segment_labels_in_place", fake_rewrite)

    payload = state.dataset_viewer_save_labels(
        root=str(dataset_root),
        dataset="run_001",
        episode=0,
        classes=[{"id": 0, "name": "success"}, {"id": 2, "name": "recovery"}],
        segments=[
            {"start_frame": 0, "end_frame": 1, "class_id": 0},
            {"start_frame": 2, "end_frame": 2, "class_id": 2},
        ],
    )

    assert payload["ok"] is True
    assert captured["dataset_dir"] == dataset_dir.resolve()
    assert 1 in captured["episode_segments"]
    assert captured["episode_segments"][0][-1]["class_id"] == 2
    assert {item["id"] for item in captured["classes"]} == {0, 1, 2}
    assert payload["label_names_full"] == ["success", "success", "recovery"]


def test_dataset_viewer_episode_can_use_external_python(monkeypatch, tmp_path) -> None:
    dataset_root = tmp_path / "datasets"
    dataset_dir = dataset_root / "run_001"
    _write_dataset_info(dataset_dir)
    state = make_state()
    fake_python = tmp_path / ".venv-ml" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    class FakeCompleted:
        returncode = 0
        stderr = ""
        stdout = web_ui_server.json.dumps({"ok": True, "episode": 0, "frame_count": 2})

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return FakeCompleted()

    monkeypatch.setattr(web_ui_server, "_dataset_viewer_external_python", lambda: fake_python)
    monkeypatch.setattr(web_ui_server.subprocess, "run", fake_run)

    payload = state.dataset_viewer_episode(root=str(dataset_root), dataset=str(dataset_dir), episode=0)

    assert payload["reader"]["kind"] == "external_python"
    assert payload["reader"]["python"] == str(fake_python)
    assert captured["cmd"][:3] == [str(fake_python), "-m", "igris_teleop.web_ui.dataset_viewer_helper"]
    assert captured["env"]["IGRIS_DATASET_VIEWER_DISABLE_EXTERNAL"] == "1"


def test_record_start_writes_task_and_record_flags() -> None:
    supervisor = FakeSupervisor()
    supervisor.collect_alive = True
    record_shm = FakeShm({"record_start": False, "record_done": False, "record_reset": False})
    task_shm = FakeShm({"task_valid": np.uint8(0), "task_name": np.zeros((512,), dtype=np.uint8)})
    state = make_state(
        supervisor,
        {
            "record_shm": record_shm,
            "record_task_shm": task_shm,
        },
    )

    state.trigger_record({"action": "start", "task_name": "Pick and Place"})

    assert record_shm.data["record_start"] is True
    assert record_shm.data["record_done"] is False
    assert _decode_uint8_text(task_shm.data["task_name"]) == "Pick and Place"
    assert int(task_shm.data["task_valid"]) == 1
