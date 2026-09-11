from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import yaml

from igris_teleop.core.project_paths import PR2AB_CALIBRATION_CONFIG_PATH, ROBOT_CONTROL_OUTPUTS_ROOT
from .joints import JointIndex, MotorIndex


LoggerLike = object


@dataclass
class PR2ABPair:
    name: str
    pr_i1: int
    pr_i2: int
    ab_i1: int
    ab_i2: int
    M: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float64))
    pr_center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    ab_center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    ab_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    enabled: bool = False

    def configured(self) -> bool:
        return bool(self.enabled)


def _robot_control_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def normalize_ab_limits(raw_limits):
    if raw_limits is None:
        return None
    if len(raw_limits) != 2:
        raise ValueError("ab_limits must have length 2")
    first = raw_limits[0]
    second = raw_limits[1]
    return ((float(first[0]), float(first[1])), (float(second[0]), float(second[1])))


def default_pr2ab_pairs() -> Dict[str, PR2ABPair]:
    return {
        "waist_rp": PR2ABPair(
            "waist_rp",
            int(JointIndex.WAIST_ROLL),
            int(JointIndex.WAIST_PITCH),
            int(MotorIndex.WAIST_L),
            int(MotorIndex.WAIST_R),
        ),
        "l_ankle_pr": PR2ABPair(
            "l_ankle_pr",
            int(JointIndex.L_ANKLE_PITCH),
            int(JointIndex.L_ANKLE_ROLL),
            int(MotorIndex.ANKLE_OUT_L),
            int(MotorIndex.ANKLE_IN_L),
        ),
        "r_ankle_pr": PR2ABPair(
            "r_ankle_pr",
            int(JointIndex.R_ANKLE_PITCH),
            int(JointIndex.R_ANKLE_ROLL),
            int(MotorIndex.ANKLE_OUT_R),
            int(MotorIndex.ANKLE_IN_R),
        ),
        "l_wrist_rp": PR2ABPair(
            "l_wrist_rp",
            int(JointIndex.L_WRIST_ROLL),
            int(JointIndex.L_WRIST_PITCH),
            int(MotorIndex.WRIST_FRONT_L),
            int(MotorIndex.WRIST_BACK_L),
        ),
        "r_wrist_rp": PR2ABPair(
            "r_wrist_rp",
            int(JointIndex.R_WRIST_ROLL),
            int(JointIndex.R_WRIST_PITCH),
            int(MotorIndex.WRIST_FRONT_R),
            int(MotorIndex.WRIST_BACK_R),
        ),
    }


def resolve_pr2ab_config_path(raw_path: Optional[str]) -> Path:
    config_path = PR2AB_CALIBRATION_CONFIG_PATH
    output_path = ROBOT_CONTROL_OUTPUTS_ROOT / "pr2ab_calibration.yaml"
    if raw_path is None:
        if config_path.is_file():
            return config_path
        if output_path.is_file():
            return output_path
        return config_path

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    candidates = (
        _robot_control_dir() / raw_path,
        _repo_root() / raw_path,
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def set_pr2ab_transform(
    pairs: Dict[str, PR2ABPair],
    *,
    name: str,
    M: np.ndarray,
    pr_center: Optional[Sequence[float]] = None,
    ab_center: Optional[Sequence[float]] = None,
    ab_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> None:
    if name not in pairs:
        raise KeyError(f"Unknown PR2AB pair name '{name}'. Available: {list(pairs.keys())}")

    pair = pairs[name]
    pair.M = np.asarray(M, dtype=np.float64).reshape(2, 2)
    if pr_center is not None:
        pair.pr_center = np.asarray(pr_center, dtype=np.float64).reshape(2)
    if ab_center is not None:
        pair.ab_center = np.asarray(ab_center, dtype=np.float64).reshape(2)
    pair.ab_limits = normalize_ab_limits(ab_limits)
    pair.enabled = True


def load_pr2ab_transforms_from_yaml(
    cfg_path: Path,
    *,
    base_pairs: Optional[Dict[str, PR2ABPair]] = None,
    logger: Optional[LoggerLike] = None,
) -> Dict[str, PR2ABPair]:
    pairs = base_pairs or default_pr2ab_pairs()
    if not cfg_path.is_file():
        if logger is not None:
            logger.info("[PR2AB] config not found: %s", cfg_path)
        return pairs

    try:
        with cfg_path.open("r", encoding="utf-8") as file_obj:
            payload = yaml.safe_load(file_obj) or {}
    except Exception as exc:
        if logger is not None:
            logger.warning("[PR2AB] failed to read config '%s': %s", cfg_path, exc)
        return pairs

    items = payload.get("pairs", payload)
    if not isinstance(items, dict):
        if logger is not None:
            logger.warning("[PR2AB] invalid config format in '%s'", cfg_path)
        return pairs

    for name, item in items.items():
        if name not in pairs:
            if logger is not None:
                logger.warning("[PR2AB] unknown pair '%s' in '%s'", name, cfg_path)
            continue
        if not isinstance(item, dict):
            if logger is not None:
                logger.warning("[PR2AB] invalid entry for '%s': expected mapping", name)
            continue
        try:
            set_pr2ab_transform(
                pairs,
                name=name,
                M=np.asarray(item["M"], dtype=np.float64).reshape(2, 2),
                pr_center=item.get("pr_center"),
                ab_center=item.get("ab_center"),
                ab_limits=item.get("ab_limits"),
            )
        except Exception as exc:
            if logger is not None:
                logger.warning("[PR2AB] failed to load pair '%s': %s", name, exc)
    return pairs


def save_pr2ab_transforms_to_yaml(cfg_path: Path, pairs: Dict[str, PR2ABPair]) -> None:
    payload = {"pairs": {}}
    for name, pair in pairs.items():
        item = {
            "M": np.asarray(pair.M, dtype=float).tolist(),
            "pr_center": np.asarray(pair.pr_center, dtype=float).tolist(),
            "ab_center": np.asarray(pair.ab_center, dtype=float).tolist(),
            "enabled": bool(pair.enabled),
        }
        if pair.ab_limits is not None:
            item["ab_limits"] = [
                [float(pair.ab_limits[0][0]), float(pair.ab_limits[0][1])],
                [float(pair.ab_limits[1][0]), float(pair.ab_limits[1][1])],
            ]
        payload["pairs"][name] = item

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=True, allow_unicode=False)


def apply_pr2ab_pair(q_pjs: np.ndarray, pair: PR2ABPair) -> np.ndarray:
    pr = np.array([q_pjs[pair.pr_i1], q_pjs[pair.pr_i2]], dtype=np.float64)
    ab = pair.ab_center + pair.M @ (pr - pair.pr_center)
    return ab.astype(np.float32)


def _solve_pair_inverse(pair: PR2ABPair, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(2)
    try:
        return np.linalg.solve(pair.M, values)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(pair.M) @ values


def apply_pr2ab_linear_pair(vec_pjs: np.ndarray, pair: PR2ABPair) -> np.ndarray:
    pr = np.array([vec_pjs[pair.pr_i1], vec_pjs[pair.pr_i2]], dtype=np.float64)
    ab = pair.M @ pr
    return ab.astype(np.float32)


def apply_ab2pr_pair(q_ms: np.ndarray, pair: PR2ABPair) -> np.ndarray:
    ab = np.array([q_ms[pair.ab_i1], q_ms[pair.ab_i2]], dtype=np.float64)
    pr = pair.pr_center + _solve_pair_inverse(pair, ab - pair.ab_center)
    return pr.astype(np.float32)


def apply_ab2pr_linear_pair(vec_ms: np.ndarray, pair: PR2ABPair) -> np.ndarray:
    ab = np.array([vec_ms[pair.ab_i1], vec_ms[pair.ab_i2]], dtype=np.float64)
    pr = _solve_pair_inverse(pair, ab)
    return pr.astype(np.float32)


def map_pjs_to_ms(q_pjs: np.ndarray, pairs: Dict[str, PR2ABPair]) -> tuple[np.ndarray, list[str]]:
    q_ms = np.array(q_pjs, dtype=np.float32, copy=True)
    unconfigured: list[str] = []
    for name, pair in pairs.items():
        if not pair.configured():
            unconfigured.append(name)
            continue
        ab = apply_pr2ab_pair(q_pjs, pair)
        q_ms[pair.ab_i1] = ab[0]
        q_ms[pair.ab_i2] = ab[1]
    return q_ms, unconfigured


def map_pjs_linear_to_ms(vec_pjs: np.ndarray, pairs: Dict[str, PR2ABPair]) -> np.ndarray:
    vec_ms = np.array(vec_pjs, dtype=np.float32, copy=True)
    for pair in pairs.values():
        if not pair.configured():
            continue
        ab = apply_pr2ab_linear_pair(vec_pjs, pair)
        vec_ms[pair.ab_i1] = ab[0]
        vec_ms[pair.ab_i2] = ab[1]
    return vec_ms


def map_ms_to_pjs(q_ms: np.ndarray, pairs: Dict[str, PR2ABPair]) -> tuple[np.ndarray, list[str]]:
    q_pjs = np.array(q_ms, dtype=np.float32, copy=True)
    unconfigured: list[str] = []
    for name, pair in pairs.items():
        if not pair.configured():
            unconfigured.append(name)
            continue
        pr = apply_ab2pr_pair(q_ms, pair)
        q_pjs[pair.pr_i1] = pr[0]
        q_pjs[pair.pr_i2] = pr[1]
    return q_pjs, unconfigured


def map_ms_linear_to_pjs(vec_ms: np.ndarray, pairs: Dict[str, PR2ABPair]) -> np.ndarray:
    vec_pjs = np.array(vec_ms, dtype=np.float32, copy=True)
    for pair in pairs.values():
        if not pair.configured():
            continue
        pr = apply_ab2pr_linear_pair(vec_ms, pair)
        vec_pjs[pair.pr_i1] = pr[0]
        vec_pjs[pair.pr_i2] = pr[1]
    return vec_pjs


def clip_ms_by_pairs(q_ms: np.ndarray, pairs: Dict[str, PR2ABPair]) -> np.ndarray:
    q_ms = np.array(q_ms, dtype=np.float32, copy=True)
    for pair in pairs.values():
        if not pair.configured() or pair.ab_limits is None:
            continue
        lower = np.array([pair.ab_limits[0][0], pair.ab_limits[1][0]], dtype=np.float32)
        upper = np.array([pair.ab_limits[0][1], pair.ab_limits[1][1]], dtype=np.float32)
        current = np.array([q_ms[pair.ab_i1], q_ms[pair.ab_i2]], dtype=np.float32)
        clipped = np.clip(current, lower, upper)
        q_ms[pair.ab_i1] = clipped[0]
        q_ms[pair.ab_i2] = clipped[1]
    return q_ms
