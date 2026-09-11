import logging_mp
import numpy as np
import yaml
from pathlib import Path

from .command_range import compress_normalized_hand_command, normalize_retargeted_hand_joints
from .dex_retargeting.retargeting_config import RetargetingConfig

logger_mp = logging_mp.get_logger(__name__)

base_dir = Path(__file__).resolve().parent
HAND_REFERENCE_SHAPE = (5, 3)
HAND_MOTOR_COUNT = 6
LEFT_INVALID_REFERENCE = np.asarray((0.15, 0.8, -0.3), dtype=np.float64)


# URDF 기본 탐색 디렉터리를 hand_control/hand_urdf로 설정
class HandRetargeting:
    def __init__(self, *, hand_side=None):
        if hand_side not in (None, "left", "right"):
            raise ValueError("hand_side must be left, right or None")
        hand_urdf_dir = base_dir / "hand_urdf"
        RetargetingConfig.set_default_urdf_dir(hand_urdf_dir)

        config_file_path = hand_urdf_dir / "igris_hand.yml"

        try:
            with config_file_path.open('r') as f:
                self.cfg = yaml.safe_load(f)

            # 좌/우 어느 한쪽만 있어도 동작하도록 유연화
            self.left_retargeting = None
            self.right_retargeting = None
            self.left_retargeting_joint_names = None
            self.right_retargeting_joint_names = None

            if 'left' in self.cfg and hand_side in (None, "left"):
                left_retargeting_config = RetargetingConfig.from_dict(self.cfg['left'])
                self.left_retargeting = left_retargeting_config.build()
                self.left_retargeting_joint_names = self.left_retargeting.joint_names

            if 'right' in self.cfg and hand_side in (None, "right"):
                right_retargeting_config = RetargetingConfig.from_dict(self.cfg['right'])
                self.right_retargeting = right_retargeting_config.build()
                self.right_retargeting_joint_names = self.right_retargeting.joint_names

            if self.left_retargeting is None and self.right_retargeting is None:
                raise ValueError("Configuration file must contain at least one of 'left' or 'right' keys.")



            self.left_inspire_api_joint_names  = [
                '1_Joint_Thumb_Middle', '3_Joint_Index_Middle', '5_Joint_Middle_Middle',
                '7_Joint_Ring_Middle', '9_Joint_Little_Middle', '0_Joint_Thumb_Proximal'
            ]
            self.right_inspire_api_joint_names = [
                '1_Joint_Thumb_Middle', '3_Joint_Index_Middle', '5_Joint_Middle_Middle',
                '7_Joint_Ring_Middle', '9_Joint_Little_Middle', '0_Joint_Thumb_Proximal'
            ]
            self.left_dex_retargeting_to_hardware = (
                [self.left_retargeting_joint_names.index(name) for name in self.left_inspire_api_joint_names]
                if self.left_retargeting else None
            )
            self.right_dex_retargeting_to_hardware = (
                [self.right_retargeting_joint_names.index(name) for name in self.right_inspire_api_joint_names]
                if self.right_retargeting else None
            )


        except FileNotFoundError:
            logger_mp.error(f"Configuration file not found: {config_file_path}")
            raise
        except yaml.YAMLError as e:
            logger_mp.error(f"YAML error while reading {config_file_path}: {e}")
            raise
        except Exception as e:
            logger_mp.error(f"An error occurred: {e}")
            raise

    @staticmethod
    def _reference_valid(reference, *, left_hand: bool) -> bool:
        points = np.asarray(reference, dtype=np.float64)
        if points.shape != HAND_REFERENCE_SHAPE or not np.all(np.isfinite(points)):
            return False
        if np.allclose(points, 0.0):
            return False
        if left_hand and np.allclose(points[4], LEFT_INVALID_REFERENCE):
            return False
        return True

    def _retarget_side(self, reference, *, left_hand: bool, strict: bool = False) -> np.ndarray:
        if not self._reference_valid(reference, left_hand=left_hand):
            if strict:
                raise ValueError("invalid retargeting fingertip reference")
            return np.zeros(HAND_MOTOR_COUNT, dtype=np.float64)

        retargeting = self.left_retargeting if left_hand else self.right_retargeting
        mapping = (
            self.left_dex_retargeting_to_hardware
            if left_hand
            else self.right_dex_retargeting_to_hardware
        )
        if retargeting is None or mapping is None:
            if strict:
                raise ValueError("requested hand retargeter is not initialized")
            return np.zeros(HAND_MOTOR_COUNT, dtype=np.float64)

        reference_points = np.asarray(reference, dtype=np.float64)
        retargeted = np.asarray(retargeting.retarget(reference_points), dtype=np.float64).reshape(-1)
        hardware_joints = retargeted[np.asarray(mapping, dtype=np.int64)]
        if strict and not np.all(np.isfinite(hardware_joints)):
            raise ValueError("retargeting produced nonfinite joints")
        return normalize_retargeted_hand_joints(
            hardware_joints,
            right_hand=not left_hand,
        )

    def retarget_single(self, reference, *, left_hand: bool) -> np.ndarray:
        """Strict single-source path; same solver, mapping and gain as dual VR.

        Callers keep independent instances for each sensor so the optimizer warm
        start and low-pass history never alternate between VR and camera inputs.
        Invalid observations are raised for the caller to hold, not sent as open.
        """
        return compress_normalized_hand_command(
            self._retarget_side(reference, left_hand=left_hand, strict=True),
            dtype=np.float64,
        )

    def retarget_normalized(self, left_reference, right_reference) -> np.ndarray:
        """Return [right 6, left 6] normalized finger-bend commands."""
        left_bend = self._retarget_side(left_reference, left_hand=True)
        right_bend = self._retarget_side(right_reference, left_hand=False)
        return compress_normalized_hand_command(
            np.concatenate((right_bend, left_bend)),
            dtype=np.float64,
        )
