# robot_control/kinematics/ik

canonical inverse kinematics 구현과 IK 전용 환경 설치 스크립트를 둡니다.

- `prox_ik_pelvis_env.py`: Pinocchio/ProxSuite 기반 상체 IK 환경입니다.
- `setup_ik_env.sh`: `.venv-ik`를 만들고 IK smoke test를 실행합니다.

설치:

```bash
./igris_teleop/robot_control/kinematics/ik/setup_ik_env.sh --reset
```

런타임 worker는 기본적으로 repo root의 `.venv-ik/bin/python`으로 이 코드를 실행합니다.
