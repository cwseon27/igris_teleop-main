# IGRIS Teleop 설치 가이드

이 문서는 Python 환경 설치 스크립트 사용법을 정리합니다. 새 PC로 source/policy를
이식하고 내부 Reliability/MediaPipe ROS package까지 설치하는 전체 절차는
[install_info.md](./install_info.md)를 참고합니다.

## 한 번에 설치

기본 설치는 `uv`를 확인하고, 없으면 설치한 뒤, 런타임/시뮬레이터/IK/ML venv와 Desktop launcher를 순서대로 구성합니다.

```bash
cd /path/to/igris_teleop-main
./install_igris_teleop.sh
```

기존 venv를 지우고 다시 만들 때:

```bash
./install_igris_teleop.sh --reset
```

이 스크립트만으로 MediaPipe와 ROS 패키지 설치까지 끝나지는 않습니다.
새 PC에서는 반드시 [전체 설치·이식 절차](install_info.md)를 끝까지 수행하세요.

ML 환경을 의도적으로 설치하지 않을 경우 아래 옵션을 사용할 수 있지만,
기본 `run_igris_teleop.sh`는 runtime/sim/IK/ML 네 환경을 모두 확인하므로
이 상태에서는 기본 launcher를 사용할 수 없습니다:

```bash
./install_igris_teleop.sh --skip-ml
```

Desktop launcher만 다시 설치하려면:

```bash
./install_igris_teleop.sh --desktop-only
```

## 설치되는 항목

- `uv`: 없으면 `https://astral.sh/uv/install.sh`로 설치합니다.
- `.venv`: 기본 런타임, Web UI, 카메라, 로봇 제어용 환경입니다.
- `.venv-sim`: MuJoCo simulator 전용 환경입니다.
- `.venv-ik`: Pinocchio/ProxSuite 기반 IK 전용 환경입니다.
- `.venv-ml`: LeRobot, PyTorch, inference/training 전용 환경입니다.
- `.venv-mediapipe`와 내부 ROS 패키지는 이 스크립트가 설치하지 않습니다.
  [install_info.md](install_info.md)의 별도 절차를 수행합니다.
- Desktop launcher: `~/.local/share/applications/igris-teleop.desktop`와 `~/Desktop/igris-teleop.desktop`를 만듭니다.

## 주요 옵션

```bash
./install_igris_teleop.sh --help
```

- `--reset`: 선택된 venv를 삭제 후 재설치합니다. 삭제 대상에 `pyvenv.cfg`가 없으면 중단합니다.
- `--refresh-locks`: `requirements/*.lock.txt`를 다시 생성한 뒤 설치합니다.
- `--skip-venvs`: venv 설치를 모두 건너뜁니다.
- `--skip-runtime`, `--skip-sim`, `--skip-ik`, `--skip-ml`: 특정 venv만 건너뜁니다.
- `--skip-desktop`: Desktop launcher 설치를 건너뜁니다.
- `--python PATH`: venv 생성에 사용할 Python을 지정합니다. 기본값은 `python3.12`입니다.
- `--torch-backend cpu`: ML 설치를 CPU PyTorch 기준으로 맞춥니다. 전달된 lock이
  `cu130`이면 `--refresh-locks`도 함께 사용합니다.

## 개별 설치

문제 범위를 좁혀야 할 때는 개별 스크립트를 직접 실행할 수 있습니다.

```bash
./igris_teleop/setup_runtime_venv.sh --reset
./igris_teleop/sim/setup_sim_venv.sh
./igris_teleop/robot_control/ik/setup_ik_env.sh --reset
./igris_teleop/setup_ml_venv.sh
./install_igris_teleop.sh --desktop-only
```

lock 파일만 다시 만들 때:

```bash
./requirements/compile_uv_locks.sh
```

## 설치 확인

각 venv 스크립트는 설치 마지막에 import smoke test와 `pip check`를 실행합니다. 전체 설치가 끝나면 다음으로 실행할 수 있습니다.

```bash
./run_igris_teleop.sh
```

Desktop 아이콘으로 실행할 때 옵션은 `igris_teleop_desktop_options.env`에서 수정합니다. 아이콘 우클릭 메뉴의 `Edit Launch Options`로도 같은 파일을 열 수 있습니다.
