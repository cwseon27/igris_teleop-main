from __future__ import annotations

import json
import os
import time
from pathlib import Path


CORE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = CORE_ROOT.parent
REPO_ROOT = PACKAGE_ROOT.parent

ARTIFACTS_ROOT = (REPO_ROOT / "igris_artifacts").resolve()
DATASETS_ROOT = (ARTIFACTS_ROOT / "datasets").resolve()
CHECKPOINTS_ROOT = (ARTIFACTS_ROOT / "checkpoints").resolve()
WALKING_POLICIES_ROOT = (ARTIFACTS_ROOT / "walking").resolve()
LOGS_ROOT = (ARTIFACTS_ROOT / "logs").resolve()

CONFIG_ROOT = (PACKAGE_ROOT / "config").resolve()
DATA_CONFIG_ROOT = (CONFIG_ROOT / "data").resolve()
ROBOT_CONTROL_CONFIG_ROOT = (CONFIG_ROOT / "robot_control").resolve()

COLLECT_DATA_CONFIG_PATH = (DATA_CONFIG_ROOT / "collect_data.yaml").resolve()
JOINT_SETTING_PATH = (ROBOT_CONTROL_CONFIG_ROOT / "joint_setting_v2.yaml").resolve()
WALKING_JOINT_SETTING_PATH = (ROBOT_CONTROL_CONFIG_ROOT / "joint_setting_walking.yaml").resolve()
WALKING_V2_JOINT_SETTING_PATH = (ROBOT_CONTROL_CONFIG_ROOT / "joint_setting_walking_v2.yaml").resolve()
SIM_WAIST_GAIN_PATH = (ROBOT_CONTROL_CONFIG_ROOT / "joint_setting_sim_waist.yaml").resolve()
SIM_NECK_GAIN_PATH = (ROBOT_CONTROL_CONFIG_ROOT / "joint_setting_sim_neck.yaml").resolve()
SIM_ARM_GAIN_PATH = (ROBOT_CONTROL_CONFIG_ROOT / "joint_setting_sim_arm.yaml").resolve()
PR2AB_CALIBRATION_CONFIG_PATH = (ROBOT_CONTROL_CONFIG_ROOT / "pr2ab_calibration.yaml").resolve()
INIT_SETTING_PATH = (ROBOT_CONTROL_CONFIG_ROOT / "init_setting.yaml").resolve()

ROBOT_CONTROL_OUTPUTS_ROOT = (PACKAGE_ROOT / "robot_control" / "outputs").resolve()
ROBOT_CONTROL_LOGS_ROOT = (LOGS_ROOT / "robot_control").resolve()
COLLECT_DATA_LOGS_ROOT = (LOGS_ROOT / "collect_data").resolve()
INFERENCE_LOGS_ROOT = (LOGS_ROOT / "inference_logs").resolve()
TUNING_RUNS_ROOT = (LOGS_ROOT / "tuning_runs").resolve()
ACTUATORNET_DATASETS_ROOT = (LOGS_ROOT / "actuatornet_datasets").resolve()


def _normalize_inference_log_policy(policy: str | None) -> str:
    txt = str(policy or "unknown").strip().lower().replace(" ", "_")
    return txt or "unknown"


def inference_log_dir_name(policy: str | None, session_id: str) -> str:
    safe_policy = _normalize_inference_log_policy(policy)
    return f"infernce_{safe_policy}_{session_id}"


def _inference_session_marker_path(policy: str | None) -> Path:
    safe_policy = _normalize_inference_log_policy(policy)
    return INFERENCE_LOGS_ROOT / f".active_{safe_policy}_session.json"


def allocate_inference_log_session_id(policy: str | None, *, stale_after_sec: float = 30.0) -> str:
    INFERENCE_LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    marker_path = _inference_session_marker_path(policy)
    now = float(time.time())
    safe_policy = _normalize_inference_log_policy(policy)

    def _read_existing() -> tuple[str | None, float | None]:
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            return None, None
        session_id = str(payload.get("session_id", "") or "").strip()
        created_at = payload.get("created_at")
        try:
            created_at_f = float(created_at)
        except Exception:
            created_at_f = None
        if not session_id:
            return None, created_at_f
        return session_id, created_at_f

    session_id, created_at = _read_existing()
    if session_id and created_at is not None and (now - created_at) <= float(stale_after_sec):
        return session_id

    new_session_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    payload = {
        "policy": safe_policy,
        "session_id": new_session_id,
        "created_at": now,
    }

    try:
        fd = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        session_id, created_at = _read_existing()
        if session_id and created_at is not None and (now - created_at) <= float(stale_after_sec):
            return session_id
        marker_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return new_session_id
    else:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=True, indent=2)
        return new_session_id


def release_inference_log_session_id(policy: str | None, session_id: str | None) -> None:
    if not session_id:
        return
    marker_path = _inference_session_marker_path(policy)
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except Exception:
        try:
            marker_path.unlink()
        except FileNotFoundError:
            pass
        return

    if str(payload.get("session_id", "") or "").strip() != str(session_id).strip():
        return
    try:
        marker_path.unlink()
    except FileNotFoundError:
        return


def resolve_under_root(base_root: Path, raw_path: str | Path | None, *, default: Path | None = None) -> Path:
    if raw_path is None or str(raw_path).strip() == "":
        if default is None:
            raise ValueError("default must be provided when raw_path is empty")
        return default.resolve()

    candidate = Path(str(raw_path).strip()).expanduser()
    if not candidate.is_absolute():
        candidate = base_root / candidate
    return candidate.resolve()


def latest_pretrained_model(checkpoints_root: Path) -> Path:
    run_dirs = sorted(
        [path for path in checkpoints_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directory found under {checkpoints_root}")

    for run_dir in run_dirs:
        checkpoint_root = run_dir / "checkpoints"
        if not checkpoint_root.is_dir():
            continue

        last_model = checkpoint_root / "last" / "pretrained_model"
        if last_model.is_dir():
            return last_model

        numeric_dirs: list[tuple[int, float, Path]] = []
        other_dirs: list[tuple[float, Path]] = []
        for path in checkpoint_root.iterdir():
            if not path.is_dir():
                continue

            model_dir = path / "pretrained_model"
            if not model_dir.is_dir():
                continue

            try:
                numeric_dirs.append((int(path.name), path.stat().st_mtime, model_dir))
            except ValueError:
                other_dirs.append((path.stat().st_mtime, model_dir))

        if numeric_dirs:
            numeric_dirs.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return numeric_dirs[0][2]

        if other_dirs:
            other_dirs.sort(key=lambda item: item[0], reverse=True)
            return other_dirs[0][1]

    raise FileNotFoundError(f"No pretrained_model directory found under {checkpoints_root}")
