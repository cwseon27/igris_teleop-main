# hand_control

손 pose 입력을 IGRIS hand command로 변환하는 retargeting 계층입니다.

## 구성

- `hand_retargeting.py`: hand tip/keypoint 입력을 hand joint command로 변환하는 핵심 로직입니다.
- `robot_hand.py`: IGRIS hand 모델과 command helper입니다.
- `command_range.py`: hand command range와 clamp 관련 정의입니다.
- `dex_retargeting/`: dex-retargeting에서 가져와 레포 내부에 포함한 retargeting 구현입니다.
- `hand_urdf/`: 좌/우 hand URDF, mesh, retargeting config입니다.

`worker_hand.py`는 이 폴더의 retargeting 로직을 사용하며, 기본적으로 `.venv-ml`에서 external worker로 실행됩니다.
