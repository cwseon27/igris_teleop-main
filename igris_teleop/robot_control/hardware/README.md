# robot_control/hardware

실기 하드웨어와 직접 통신하는 저수준 제어 interface입니다.

- `motor_control.py`: motor command/state 입출력 wrapper입니다.
- `base_control.py`: base control 관련 helper입니다.

이 폴더의 변경은 실제 로봇 명령에 직접 연결될 수 있으므로 simulator나 dry-run 경로에서 먼저 확인합니다.
