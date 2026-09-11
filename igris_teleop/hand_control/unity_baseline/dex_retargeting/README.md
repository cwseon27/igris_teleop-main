# hand_control/dex_retargeting

손 retargeting을 위해 레포 내부에 포함한 dex-retargeting 계열 코드입니다.

## 주요 파일

- `retargeting_config.py`: retargeting 설정 loader입니다.
- `seq_retarget.py`: sequence retargeting 실행 로직입니다.
- `optimizer.py`, `optimizer_utils.py`: retargeting optimization 구현입니다.
- `kinematics_adaptor.py`, `robot_wrapper.py`: hand kinematics adapter입니다.
- `yourdfpy.py`: URDF parsing helper입니다.

외부 패키지 import에 의존하지 않고 현재 레포의 hand 모델에 맞춰 수정할 수 있도록 vendored 상태로 유지합니다.
