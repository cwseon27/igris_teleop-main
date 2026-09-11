# workers

실제 teleop 기능을 실행하는 worker 구현과 registry입니다. `igris_teleop.main`은 이 폴더의 registry를 기준으로 worker를 생성하거나 별도 Python venv에서 external worker로 실행합니다.

## 핵심 파일

- `registry.py`: worker 이름, 생성 함수, external 실행 여부, 사용할 Python 경로를 정의합니다.
- `_run_external.py`: `.venv-sim`, `.venv-ik`, `.venv-ml` 같은 별도 venv에서 worker를 실행하는 wrapper입니다.
- `command_process.py`: ROS launch/node처럼 Python worker가 아닌 외부 command process spec을 만듭니다.
- `keyboard_worker.py`: keyboard event를 `ready/start/home/shutdown` 이벤트로 변환합니다.

## 주요 worker

- `worker_unity_bridge.py`: Unity/ROS pose 입력을 teleop shared memory로 전달합니다.
- `worker_master_arm_bridge.py`: master arm ROS 입력을 teleop 입력으로 전달합니다.
- `worker_igris_ik.py`: VR/masterarm pose와 robot obs를 받아 상체 IK action을 생성합니다.
- `worker_hand.py`: hand tip pose를 IGRIS hand command로 retargeting합니다.
- `worker_control.py`: `igris_c_sdk` 기반 실기/저수준 제어 worker입니다.
- `worker_simulator.py`: MuJoCo simulator worker입니다.
- `worker_camera.py`, `worker_camera_topic.py`: camera frame을 shared memory로 전달합니다.
- `worker_collect_data.py`: LeRobot episode/frame 저장 worker입니다.
- `worker_inference_lerobot.py`, `worker_inference_pi.py`, `worker_optimize.py`: policy inference worker입니다.
- `worker_replay.py`: 저장된 dataset replay worker입니다.
- `worker_walking_policy.py`, `worker_walking_logger.py`: walking policy와 walking log worker입니다.

새 worker를 추가할 때는 구현 파일을 만든 뒤 `registry.py`에 `WorkerSpec`을 등록해야 Web UI에서 실행할 수 있습니다.
