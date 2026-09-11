# robot_control/controller

상위 로봇 제어 로직을 둡니다. hardware interface보다 위에서 joint command, kinematics helper, 안전 종료 절차를 구성합니다.

- `igris_controller.py`: IGRIS-C controller 핵심 구현입니다.
- `core.py`: controller 공통 타입/유틸입니다.
- `kinematics.py`, `pr2ab.py`: compatibility wrapper 성격의 운동학/parallel joint helper입니다.
- `shutdown_helper.py`: 종료 시 motor/controller 상태 정리 helper입니다.
