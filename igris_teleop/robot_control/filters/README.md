# robot_control/filters

제어 신호나 관측값에 적용하는 필터를 둡니다.

- `torque_kalman.py`: torque 관측/추정에 쓰는 Kalman filter입니다.

필터 파라미터 변경은 실기 제어 응답에 직접 영향을 줄 수 있으므로 simulator와 로그 비교 후 반영합니다.
