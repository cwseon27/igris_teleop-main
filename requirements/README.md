# requirements

목적별 Python dependency 입력 파일과 `uv` lock 파일을 관리하는 폴더입니다.

## 파일 구조

- `runtime.in`, `runtime.lock.txt`: 기본 런타임 `.venv`
- `sim.in`, `sim.lock.txt`: MuJoCo simulator `.venv-sim`
- `ik.in`, `ik.lock.txt`: Pinocchio/ProxSuite IK `.venv-ik`
- `ml.in`, `ml.lock.txt`: LeRobot/PyTorch `.venv-ml`
- `mediapipe.in`: hybrid hand camera용 `.venv-mediapipe`
- `runtime.txt`: 기존 pip workflow 호환용 direct dependency 목록입니다.
- `compile_uv_locks.sh`: 모든 lock 파일을 다시 생성합니다.

lock 재생성:

```bash
./requirements/compile_uv_locks.sh
```

기본 PyTorch backend는 `cu130`입니다. CPU lock/install이 필요하면 `IGRIS_TORCH_BACKEND=cpu`를 사용합니다.

Reliability RNN/HistGB가 사용하는 `scikit-learn`과 `joblib`은 `ml.in`/`ml.lock.txt`에
포함됩니다. MediaPipe는 OpenCV/protobuf 의존성 충돌을 피하기 위해 별도 환경으로
설치합니다.
