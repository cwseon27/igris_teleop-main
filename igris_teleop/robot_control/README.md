# robot_control

IGRIS-C 로봇 제어, 운동학, 하드웨어 인터페이스를 모아둔 폴더입니다. 실기 제어 worker와 IK worker가 이 폴더의 모듈을 사용합니다.

## 구성

- `controller/`: 상위 controller, PR2AB, kinematics helper, shutdown helper입니다.
- `hardware/`: `igris_c_sdk` 기반 motor/base control 인터페이스입니다.
- `interfaces/`: master arm ROS interface 등 외부 인터페이스 adapter입니다.
- `fk/`: 상체 forward kinematics helper입니다.
- `ik/`: 현재 worker가 import하는 IK compatibility 경로입니다.
- `kinematics/`: Pinocchio/ProxSuite 기반 IK 구현과 joint 정의입니다.
- `filters/`: torque Kalman filter 같은 제어 필터입니다.
- `diagnostics/`: 진단/로그 관련 모듈입니다.
- `scripts/`: gain tuning, calibration, rollout 같은 운영 스크립트입니다.
- `asset/`: robot model/asset 관련 문서와 파일입니다.

IK 전용 환경은 repo root의 `.venv-ik`를 사용합니다. 설치는 `./install_igris_teleop.sh` 또는 `./igris_teleop/robot_control/ik/setup_ik_env.sh --reset`으로 수행합니다.
