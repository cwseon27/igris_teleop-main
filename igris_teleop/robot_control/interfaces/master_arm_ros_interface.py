import json
import os
import threading
import time
from pathlib import Path
from typing import List, Dict
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

TARGET_LENGTH = 14
HAND_LENGTH = 4

JOINT_NAMES = [
    "Shoulder_Pitch_L",
    "Shoulder_Roll_L",
    "Shoulder_Yaw_L",
    "Elbow_Pitch_L",
    "Wrist_Yaw_L",
    "Wrist_Roll_L",
    "Wrist_Pitch_L",
    "Shoulder_Pitch_R",
    "Shoulder_Roll_R",
    "Shoulder_Yaw_R",
    "Elbow_Pitch_R",
    "Wrist_Yaw_R",
    "Wrist_Roll_R",
    "Wrist_Pitch_R",
]
ARM_SRC_NAMES = [
    "15_Joint_Shoulder_Pitch_Left",
    "16_Joint_Shoulder_Roll_Left",
    "17_Joint_Shoulder_Yaw_Left",
    "18_Joint_Elbow_Pitch_Left",
    "19_Joint_Wrist_Yaw_Left",
    "20_Joint_Wrist_Roll_Left",
    "21_Joint_Wrist_Pitch_Left",
    "22_Joint_Shoulder_Pitch_Right",
    "23_Joint_Shoulder_Roll_Right",
    "24_Joint_Shoulder_Yaw_Right",
    "25_Joint_Elbow_Pitch_Right",
    "26_Joint_Wrist_Yaw_Right",
    "27_Joint_Wrist_Roll_Right",
    "28_Joint_Wrist_Pitch_Right",
]

HAND_JOINT_NAMES = [
    "Finger_T_R",
    "Finger_F_R",
    "Finger_T_L",
    "Finger_F_L",
]
HAND_RAW_OPEN_CLOSE = {
    "Finger_T_R": (-0.03, 1.6),
    "Finger_F_R": (0.03, -1.47),
    "Finger_T_L": (0.05, -1.57),
    "Finger_F_L": (-0.01, 1.50),
}

HAND_CALIB_PATH = Path(os.path.expanduser("~/.ros/igris_masterarm_hand_calib.json"))
HAND_CALIB_MIN_RANGE = 0.01
HAND_CALIB_SAVE_INTERVAL_S = 1.0
HAND_CLOSE_INVERT_BY_INDEX = [
    False,  # Finger_T_R: 오른손 엄지
    False,  # Finger_F_R: 오른손 나머지 손가락
    True,   # Finger_T_L: 왼손 엄지
    True,   # Finger_F_L: 왼손 나머지 손가락
]

ACT_HAND_LENGTH = 12

ACT_HAND_JOINT_NAMES = [
    "right_thumb_flx",
    "right_index_flx",
    "right_middle_flx",
    "right_ring_flx",
    "right_little_flx",
    "right_thumb_abd",
    "left_thumb_flx",
    "left_index_flx",
    "left_middle_flx",
    "left_ring_flx",
    "left_little_flx",
    "left_thumb_abd",
]

ARM_ZERO_TOL = 1e-6


def _normalize_open0_close1(value: float, open_value: float, close_value: float) -> float:
    if np.isclose(open_value, close_value):
        return 0.0
    if open_value < close_value:
        normalized = (float(value) - float(open_value)) / (float(close_value) - float(open_value))
    else:
        normalized = 1.0 - (
            (float(value) - float(close_value)) / (float(open_value) - float(close_value))
        )
    return float(np.clip(normalized, 0.0, 1.0))


class MasterArmROSInterface(Node):
    """IGRIS-C Master Arm ROS2 토픽 인터페이스 (JointState 기반)."""

    def __init__(self) -> None:
        super().__init__("igris_c_hand_bridge")

        # arm + hand가 같은 JointState로 오는 경우가 많아, 하나의 sub로 함께 처리
        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            # Leader motion is a latest-value stream. Retaining old samples
            # makes the simulated arm trail behind a fast hand motion.
            1,
        )

        self._present_position_lock = threading.Lock()
        self._hand_position_lock = threading.Lock()
        # `_has_position` intentionally remains a one-shot "first sample" event.
        # This separate condition is notified for every fresh arm JointState so
        # consumers can wait without clearing (and racing on) `_has_position`.
        self._arm_update_condition = threading.Condition()

        self._present_position: List[float] = [0.0] * TARGET_LENGTH
        self._hand_position: List[float] = [0.0] * HAND_LENGTH  # <-- 수정: HAND_LENGTH

        self._has_position = threading.Event()
        self._has_hand_position = threading.Event()  # <-- 추가
        self._arm_seq = 0
        self._hand_seq = 0
        self._last_arm_update_monotonic: float | None = None
        self._last_hand_update_monotonic: float | None = None
        self._hand_calib = self._load_hand_calibration()
        self._hand_calib_dirty = False
        self._last_hand_calib_save_monotonic = 0.0

    # ----------------------------- hand calibration ------------------------
    def _load_hand_calibration(self) -> Dict[str, Dict[str, float | None]]:
        calib: Dict[str, Dict[str, float | None]] = {
            jn: {"min": None, "max": None} for jn in HAND_JOINT_NAMES
        }
        try:
            with HAND_CALIB_PATH.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            return calib
        except Exception as exc:
            self.get_logger().warning(f"Failed to load hand calibration {HAND_CALIB_PATH}: {exc}")
            return calib

        if not isinstance(raw, dict):
            return calib

        for jn in HAND_JOINT_NAMES:
            item = raw.get(jn)
            if not isinstance(item, dict):
                continue
            min_v = self._finite_or_none(item.get("min"))
            max_v = self._finite_or_none(item.get("max"))
            calib[jn] = {"min": min_v, "max": max_v}
        return calib

    @staticmethod
    def _finite_or_none(value) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(out):
            return None
        return out

    def _update_hand_calibration_locked(self, joint_name: str, value: float) -> None:
        if not np.isfinite(value):
            return

        item = self._hand_calib.setdefault(joint_name, {"min": None, "max": None})
        min_v = item.get("min")
        max_v = item.get("max")
        changed = False

        if min_v is None or value < float(min_v):
            item["min"] = float(value)
            changed = True
        if max_v is None or value > float(max_v):
            item["max"] = float(value)
            changed = True

        if changed:
            self._hand_calib_dirty = True

    def _maybe_save_hand_calibration(self, *, force: bool = False) -> None:
        with self._hand_position_lock:
            if not self._hand_calib_dirty and not force:
                return
            now = time.monotonic()
            if not force and now - self._last_hand_calib_save_monotonic < HAND_CALIB_SAVE_INTERVAL_S:
                return
            data = {
                jn: {
                    "min": self._hand_calib.get(jn, {}).get("min"),
                    "max": self._hand_calib.get(jn, {}).get("max"),
                }
                for jn in HAND_JOINT_NAMES
            }
            self._hand_calib_dirty = False
            self._last_hand_calib_save_monotonic = now

        try:
            HAND_CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = HAND_CALIB_PATH.with_suffix(HAND_CALIB_PATH.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.write("\n")
            tmp_path.replace(HAND_CALIB_PATH)
        except Exception as exc:
            with self._hand_position_lock:
                self._hand_calib_dirty = True
            self.get_logger().warning(f"Failed to save hand calibration {HAND_CALIB_PATH}: {exc}")

    @staticmethod
    def _fallback_normalized_hand_value(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _normalized_hand_value_locked(self, joint_name: str, value: float) -> float:
        item = self._hand_calib.get(joint_name, {})
        min_v = item.get("min")
        max_v = item.get("max")
        if min_v is None or max_v is None:
            return self._fallback_normalized_hand_value(value)

        span = float(max_v) - float(min_v)
        if span < HAND_CALIB_MIN_RANGE:
            return self._fallback_normalized_hand_value(value)

        return float(np.clip((float(value) - float(min_v)) / span, 0.0, 1.0))

    def _close_amount_from_hand_value_locked(self, idx: int, value: float) -> float:
        normalized = self._normalized_hand_value_locked(HAND_JOINT_NAMES[idx], value)
        close_amount = 1.0 - normalized
        if HAND_CLOSE_INVERT_BY_INDEX[idx]:
            close_amount = 1.0 - close_amount
        return float(np.clip(close_amount, 0.0, 1.0))

    # ----------------------------- sub callbacks ---------------------------
    def _on_joint_state(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return

        # name -> position 매핑 (길이가 다르거나 일부만 와도 안전)
        name_to_pos: Dict[str, float] = {}
        n = min(len(msg.name), len(msg.position))
        for i in range(n):
            name_to_pos[msg.name[i]] = float(msg.position[i])

        # arm 업데이트
        arm_updated_any = False
        with self._present_position_lock:
            updated = list(self._present_position)
            for idx, src_name in enumerate(ARM_SRC_NAMES):
                if src_name in name_to_pos:
                    updated[idx] = name_to_pos[src_name]
                    arm_updated_any = True
            self._present_position = updated
            if arm_updated_any:
                self._arm_seq += 1
                self._last_arm_update_monotonic = time.monotonic()
                self._has_position.set()

        # hand 업데이트
        hand_updated_any = False
        with self._hand_position_lock:
            updated_hand = list(self._hand_position)
            for idx, jn in enumerate(HAND_JOINT_NAMES):
                if jn in name_to_pos:
                    value = name_to_pos[jn]
                    updated_hand[idx] = value
                    self._update_hand_calibration_locked(jn, value)
                    hand_updated_any = True
            self._hand_position = updated_hand
            if hand_updated_any:
                self._hand_seq += 1
                self._last_hand_update_monotonic = time.monotonic()

        if hand_updated_any:
            self._has_hand_position.set()

        if arm_updated_any:
            # Arm and any hand fields carried by the same JointState are fully
            # committed before the low-latency bridge waiter is released.
            with self._arm_update_condition:
                self._arm_update_condition.notify_all()

        if hand_updated_any:
            self._maybe_save_hand_calibration()

    # ----------------------------- getters --------------------------------
    def wait_for_first_position(self, timeout: float = 5.0) -> bool:
        """첫 arm JointState(arm joint 중 1개라도) 수신까지 대기."""
        return self._has_position.wait(timeout=timeout)

    def get_present_position(self) -> List[float]:
        """arm 14개 조인트 포지션을 JOINT_NAMES 순서로 반환."""
        with self._present_position_lock:
            return list(self._present_position)

    @staticmethod
    def _build_arm_status(
        present: np.ndarray,
        *,
        seq: int,
        last_update: float | None,
        has_position: bool,
    ) -> Dict[str, float]:
        if last_update is None:
            age_s = float("inf")
        else:
            age_s = max(0.0, float(time.monotonic() - last_update))
        return {
            "has_position": 1.0 if has_position else 0.0,
            "seq": float(seq),
            "age_s": age_s,
            "is_all_zero": 1.0 if np.linalg.norm(present) < ARM_ZERO_TOL else 0.0,
        }

    def get_arm_snapshot(self) -> tuple[List[float], Dict[str, float]]:
        """Return position and freshness metadata from one locked snapshot."""
        with self._present_position_lock:
            present_list = list(self._present_position)
            seq = int(self._arm_seq)
            last_update = self._last_arm_update_monotonic
            has_position = self._has_position.is_set()
        present = np.asarray(present_list, dtype=np.float64)
        status = self._build_arm_status(
            present,
            seq=seq,
            last_update=last_update,
            has_position=has_position,
        )
        return present_list, status

    def get_arm_status(self) -> Dict[str, float]:
        _, status = self.get_arm_snapshot()
        return status

    def _current_arm_seq(self) -> int:
        with self._present_position_lock:
            return int(self._arm_seq)

    def wait_for_arm_update(self, after_seq: int, timeout: float | None = None) -> int:
        """Wait for a newer arm sample and return the latest sequence.

        The sequence is checked again while holding the condition, so a sample
        that arrives just before the wait cannot be lost. A bounded timeout is
        supplied by the bridge to retain its periodic stale-data watchdog.
        """
        expected = int(after_seq)
        wait_timeout = None if timeout is None else max(0.0, float(timeout))
        with self._arm_update_condition:
            self._arm_update_condition.wait_for(
                lambda: self._current_arm_seq() != expected,
                timeout=wait_timeout,
            )
        return self._current_arm_seq()

    # ----------------------------- hand getters ----------------------------
    def wait_for_first_hand_position(self, timeout: float = 5.0) -> bool:
        """첫 hand JointState(hand joint 중 1개라도) 수신까지 대기."""
        return self._has_hand_position.wait(timeout=timeout)

    def get_hand_position(self) -> List[float]:
        """hand 4개 조인트 raw position을 HAND_JOINT_NAMES 순서로 반환."""
        with self._hand_position_lock:
            return list(self._hand_position)

    def get_hand_status(self) -> Dict[str, float]:
        with self._hand_position_lock:
            hand = np.asarray(self._hand_position, dtype=np.float64)
            seq = float(self._hand_seq)
            last_update = self._last_hand_update_monotonic
        if last_update is None:
            age_s = float("inf")
        else:
            age_s = max(0.0, float(time.monotonic() - last_update))
        return {
            "has_position": 1.0 if self._has_hand_position.is_set() else 0.0,
            "seq": seq,
            "age_s": age_s,
            "is_all_zero": 1.0 if np.linalg.norm(hand) < ARM_ZERO_TOL else 0.0,
        }



    def get_act_hand_12(self) -> List[float]:
        """
        ROS hand raw value(4) -> act_hand close amount(12, open=0 close=1) 확장 매핑.
        순서: right_thumb_flx right_index_flx right_middle_flx right_ring_flx right_little_flx
              right_thumb_abd left_thumb_flx left_index_flx left_middle_flx left_ring_flx left_little_flx left_thumb_abd
        """
        # hand 4채널은 HAND_JOINT_NAMES 순서라고 가정:
        # [Finger_T_R, Finger_F_R, Finger_T_L, Finger_F_L]
        with self._hand_position_lock:
            raw_values = list(self._hand_position)

        finger_t_r, finger_f_r, finger_t_l, finger_f_l = [
            _normalize_open0_close1(value, *HAND_RAW_OPEN_CLOSE[joint_name])
            for joint_name, value in zip(HAND_JOINT_NAMES, raw_values)
        ]

        act = [0.0] * ACT_HAND_LENGTH

        # Right
        act[0] = finger_t_r               # right_thumb_flx
        act[1] = finger_f_r               # right_index_flx
        act[2] = finger_f_r               # right_middle_flx
        act[3] = finger_f_r               # right_ring_flx
        act[4] = finger_f_r               # right_little_flx
        act[5] = finger_t_r               # right_thumb_abd (엄지 2개 동일)

        # Left
        act[6] = finger_t_l               # left_thumb_flx
        act[7] = finger_f_l               # left_index_flx
        act[8] = finger_f_l               # left_middle_flx
        act[9] = finger_f_l               # left_ring_flx
        act[10] = finger_f_l              # left_little_flx
        act[11] = finger_t_l              # left_thumb_abd (엄지 2개 동일)

        return act

    def destroy_node(self) -> bool:
        self._maybe_save_hand_calibration(force=True)
        return super().destroy_node()

    def get_all_position(self) -> List[float]:
        """
        arm(14) + hand(4)를 이어붙여 반환.
        반환 순서: JOINT_NAMES 다음 HAND_JOINT_NAMES
        """
        with self._present_position_lock, self._hand_position_lock:
            return list(self._present_position) + list(self._hand_position)

    def get_position_dict(self) -> Dict[str, float]:
        """현재 보관 중인 값을 joint name -> position dict로 반환."""
        out: Dict[str, float] = {}
        with self._present_position_lock:
            for jn, v in zip(JOINT_NAMES, self._present_position):
                out[jn] = v
        with self._hand_position_lock:
            for jn, v in zip(HAND_JOINT_NAMES, self._hand_position):
                out[jn] = v
        return out
