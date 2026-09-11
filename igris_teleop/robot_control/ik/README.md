# robot_control/ik

IK compatibility 경로입니다. 현재 실제 ProxSuite/Pinocchio 기반 구현은 [robot_control/kinematics/ik/](../kinematics/ik/README.md)에 있고, 이 폴더는 기존 import와 설치 스크립트 경로를 유지합니다.

- `prox_ik_pelvis_env.py`: kinematics IK 구현으로 연결되는 wrapper입니다.
- `setup_ik_env.sh`: IK 전용 `.venv-ik` 설치 wrapper입니다.

새 IK 코드는 가능하면 `robot_control/kinematics/ik` 아래에 추가합니다.
