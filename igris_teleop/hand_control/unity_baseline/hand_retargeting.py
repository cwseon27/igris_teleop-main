from .dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import yaml
from enum import Enum
import os
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

base_dir = Path(__file__).resolve().parent
parent_dir = base_dir.parent

# URDF 기본 탐색 디렉터리를 hand_control/hand_urdf로 설정
class HandRetargeting:
    def __init__(self):
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

            if 'left' in self.cfg:
                left_retargeting_config = RetargetingConfig.from_dict(self.cfg['left'])
                self.left_retargeting = left_retargeting_config.build()
                self.left_retargeting_joint_names = self.left_retargeting.joint_names

            if 'right' in self.cfg:
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
