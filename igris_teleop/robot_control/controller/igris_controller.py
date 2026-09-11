from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import logging_mp

from .core import ControllerCore
from .pr2ab import (
    PR2ABPair,
    clip_ms_by_pairs,
    default_pr2ab_pairs,
    load_pr2ab_transforms_from_yaml,
    map_pjs_to_ms,
    normalize_ab_limits,
    resolve_pr2ab_config_path,
    save_pr2ab_transforms_to_yaml,
    set_pr2ab_transform,
)
from .kinematics import JointIndex


MS_START_LIMIT_TOLERANCE = 0.05

logger_mp = logging_mp.get_logger(__name__, level=logging_mp.INFO)

NECK_PITCH_MS_SIGN = -1.0


class BaseController(ControllerCore):
    """Assembled IGRIS controller with optional PR->AB mapping in MS mode."""

    def __init__(
        self,
        domain_id: int = 0,
        control_hz: float = 300.0,
        kinematic_mode=None,
        use_motor_state: bool = False,
        auto_service_init: bool = True,
        bms_init_type=None,
        torque_type=None,
        control_mode=None,
        service_timeout_ms: int = 30000,
        service_init_delay: float = 0.5,
        pr2ab_config_path: Optional[str] = None,
        joint_profile_path: Optional[str] = None,
        start_control_loop: bool = True,
    ):
        import igris_c_sdk as igc_sdk

        if kinematic_mode is None:
            kinematic_mode = igc_sdk.KinematicMode.PJS
        if bms_init_type is None:
            bms_init_type = igc_sdk.BmsInitType.BMS_AND_MOTOR_INIT
        if torque_type is None:
            torque_type = igc_sdk.TorqueType.TORQUE_ON
        if control_mode is None:
            control_mode = igc_sdk.ControlMode.CONTROL_MODE_LOW_LEVEL

        self._pr2ab_pairs = default_pr2ab_pairs()
        self._warned_unconfigured_ms = False
        self._pr2ab_config_path = resolve_pr2ab_config_path(pr2ab_config_path)

        super().__init__(
            domain_id=domain_id,
            control_hz=control_hz,
            kinematic_mode=kinematic_mode,
            use_motor_state=use_motor_state,
            auto_service_init=auto_service_init,
            bms_init_type=bms_init_type,
            torque_type=torque_type,
            control_mode=control_mode,
            service_timeout_ms=service_timeout_ms,
            service_init_delay=service_init_delay,
            joint_profile_path=joint_profile_path,
            start_thread=False,
        )

        self.load_pr2ab_transforms_from_yaml(self._pr2ab_config_path)
        if start_control_loop:
            self._start_control_loop()

    def load_pr2ab_transforms_from_yaml(self, cfg_path: Path) -> None:
        self._pr2ab_pairs = load_pr2ab_transforms_from_yaml(
            cfg_path,
            base_pairs=self._pr2ab_pairs,
            logger=logger_mp,
        )

    def save_pr2ab_transforms_to_yaml(self, cfg_path: Optional[Path] = None) -> Path:
        target_path = cfg_path or self._pr2ab_config_path
        save_pr2ab_transforms_to_yaml(target_path, self._pr2ab_pairs)
        return target_path

    def set_pr2ab_transform(
        self,
        name: str,
        M: np.ndarray,
        pr_center: Optional[Sequence[float]] = None,
        ab_center: Optional[Sequence[float]] = None,
        ab_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> None:
        set_pr2ab_transform(
            self._pr2ab_pairs,
            name=name,
            M=M,
            pr_center=pr_center,
            ab_center=ab_center,
            ab_limits=normalize_ab_limits(ab_limits),
        )
        self._warned_unconfigured_ms = False

    def get_pr2ab_pair(self, name: str) -> PR2ABPair:
        if name not in self._pr2ab_pairs:
            raise KeyError(f"Unknown PR2AB pair name '{name}'. Available: {list(self._pr2ab_pairs.keys())}")
        return self._pr2ab_pairs[name]

    def start_control_loop(self) -> None:
        self._start_control_loop()

    def get_unconfigured_pr2ab_pairs(self) -> list[str]:
        return sorted(name for name, pair in self._pr2ab_pairs.items() if not pair.configured())

    def get_pr2ab_pairs_missing_limits(self) -> list[str]:
        return sorted(
            name
            for name, pair in self._pr2ab_pairs.items()
            if pair.configured() and pair.ab_limits is None
        )

    def get_ms_limit_violations(self) -> list[dict[str, object]]:
        q_ms = self.get_motor_q()
        if q_ms is None:
            raise RuntimeError("current motor state is unavailable")

        q_ms = np.asarray(q_ms, dtype=np.float64).reshape(-1)
        violations: list[dict[str, object]] = []
        for name, pair in self._pr2ab_pairs.items():
            if not pair.configured() or pair.ab_limits is None:
                continue

            lower = np.array([pair.ab_limits[0][0], pair.ab_limits[1][0]], dtype=np.float64)
            upper = np.array([pair.ab_limits[0][1], pair.ab_limits[1][1]], dtype=np.float64)
            current = np.array([q_ms[pair.ab_i1], q_ms[pair.ab_i2]], dtype=np.float64)
            below = np.maximum(lower - current, 0.0)
            above = np.maximum(current - upper, 0.0)
            overrun = np.maximum(below, above)
            if np.any(overrun > MS_START_LIMIT_TOLERANCE):
                violations.append(
                    {
                        "name": name,
                        "current": current,
                        "lower": lower,
                        "upper": upper,
                        "overrun": overrun,
                    }
                )
        return violations

    def ensure_ms_start_guard(self, *, state_timeout: float | None = 5.0) -> None:
        import igris_c_sdk as igc_sdk

        if self._kinematic_mode != igc_sdk.KinematicMode.MS:
            return

        if not self._pr2ab_config_path.is_file():
            raise RuntimeError(
                "MS start blocked: PR2AB calibration file not found at "
                f"{self._pr2ab_config_path}"
            )

        unconfigured = self.get_unconfigured_pr2ab_pairs()
        if unconfigured:
            raise RuntimeError(
                "MS start blocked: PR2AB pairs are not configured: "
                f"{unconfigured}"
            )

        missing_limits = self.get_pr2ab_pairs_missing_limits()
        if missing_limits:
            raise RuntimeError(
                "MS start blocked: PR2AB pairs are missing ab_limits: "
                f"{missing_limits}"
            )

        if not self.wait_for_state(timeout=state_timeout):
            raise RuntimeError(
                f"MS start blocked: no LowState received within {state_timeout}s"
            )

        violations = self.get_ms_limit_violations()
        if violations:
            details: list[str] = []
            for item in violations:
                current = np.asarray(item["current"], dtype=np.float64)
                lower = np.asarray(item["lower"], dtype=np.float64)
                upper = np.asarray(item["upper"], dtype=np.float64)
                overrun = np.asarray(item["overrun"], dtype=np.float64)
                details.append(
                    f"{item['name']}: current={current.tolist()} "
                    f"limits={[lower.tolist(), upper.tolist()]} "
                    f"overrun={overrun.tolist()}"
                )
            raise RuntimeError(
                "MS start blocked: current motor state exceeds PR2AB limits: "
                + "; ".join(details)
            )

    def _map_command_q(self, q_pjs: np.ndarray) -> np.ndarray:
        q_ms, unconfigured = map_pjs_to_ms(q_pjs, self._pr2ab_pairs)
        if unconfigured and not self._warned_unconfigured_ms:
            logger_mp.warning(
                "[BaseController] MS mode but some PR2AB transforms are not configured (skipped): %s",
                unconfigured,
            )
            self._warned_unconfigured_ms = True
        # Visualizer, IK, shared memory, and LowState joint_state all use the
        # robot PJS/URDF convention.  Real-robot LowCmd in MS mode is interpreted
        # in motor convention.  Most scalar joints share the same positive
        # direction, but the physical neck pitch motor is mounted with the
        # opposite sign.  Keep the PJS target unchanged for the visualizer and
        # flip only the outgoing MS command.
        neck_pitch_idx = int(JointIndex.NECK_PITCH)
        q_ms[neck_pitch_idx] = float(NECK_PITCH_MS_SIGN) * float(q_pjs[neck_pitch_idx])
        return q_ms

    def _clip_ms_command_q(self, q_ms: np.ndarray) -> np.ndarray:
        return clip_ms_by_pairs(q_ms, self._pr2ab_pairs)

    def set_kinematic_mode(self, mode):
        super().set_kinematic_mode(mode)
        self._warned_unconfigured_ms = False


IgrisController = BaseController
