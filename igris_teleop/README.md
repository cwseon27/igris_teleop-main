# igris_teleop

메인 Python 패키지입니다. `python -m igris_teleop.main` 실행 시 이 패키지 안의 worker registry, shared memory, Web UI, worker process들이 함께 초기화됩니다.

## 주요 진입점

- `main.py`: 전체 런타임 entrypoint입니다. worker process 생성, Web UI 실행, logging bridge, shared memory 초기화를 담당합니다.
- `setup_runtime_venv.sh`: 기본 런타임 `.venv`를 구성합니다.
- `setup_ml_venv.sh`: LeRobot/PyTorch 기반 `.venv-ml`을 구성합니다.
- `setup_runtime_env.sh`, `setup_runtime_conda.sh`: 이전 conda/venv wrapper 호환용 설치 스크립트입니다.
- `head_start_guard.py`: head/start 관련 guard 로직입니다.

## 하위 폴더

- [core/](./core/README.md): worker 공통 기반, event/state machine, rate 유틸
- [sharedmemory/](./sharedmemory/README.md): 프로세스 간 공유 메모리 구조
- [workers/](./workers/README.md): 실제 기능 worker 구현과 registry
- [web_ui/](./web_ui/README.md): 브라우저 UI 서버와 정적 asset
- [robot_control/](./robot_control/README.md): 로봇 제어, IK/FK, hardware interface
- [hand_control/](./hand_control/README.md): hand retargeting과 URDF/mesh
- [teleop_devices/](./teleop_devices/README.md): Unity, ROS, DDS, camera 입력 장치
- [sim/](./sim/README.md): MuJoCo simulator
- [training/](./training/README.md): dataset, train, inference utility
- [policies/](./policies/README.md): walking policy profile
- [config/](./config/README.md): YAML 설정 파일
- [ipc/](./ipc/README.md): 프로세스 매니저 보조 모듈
