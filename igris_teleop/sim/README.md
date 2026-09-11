# sim

MuJoCo 기반 IGRIS-C 시뮬레이터와 전용 설치 스크립트를 둔 폴더입니다.

## 구성

- `setup_sim_venv.sh`: simulator 전용 `.venv-sim` 생성 스크립트입니다.
- `robot/`: MuJoCo XML, mesh, robot asset입니다.
- `stereo_camera.py`: head stereo camera 해상도, baseline, ROS topic 공통 설정입니다.
- `validate_sim_model.py`: hand actuator/mimic과 stereo mount 검증 스크립트입니다.
- `grasp_stability_benchmark.py`: 손으로 cube를 닫아 잡을 때 속도, 이탈량,
  접촉력과 침투량을 재현하는 MuJoCo 검증 스크립트입니다.

기본 simulator XML은 `robot/mujoco/igris_c_v2_with_hand.xml`입니다. 다른 XML을 사용할 때는 `IGRIS_SIM_XML_PATH`를 지정합니다.

```bash
IGRIS_SIM_XML_PATH=/path/to/robot.xml ./run_igris_teleop.sh
```

runtime에서는 `worker_simulator.py`가 `.venv-sim`의 Python으로 external worker 실행됩니다.

## Hand와 head stereo camera

기본 XML은 `ros2_ws/src/igris_c_description_public`의 IGRIS-C hand 형상을 기준으로
양손에 12개 actuator와 10개 distal mimic joint를 포함합니다. 텔레옵 중에는
`act_shm.act_hand`의 12개 `0..1` 명령을 simulator가 직접 읽어 손가락 joint target으로
변환합니다. 손바닥과 각 phalanx에는 mesh 대신 단순 box collision geometry가 있으며,
자기 로봇 링크와의 충돌은 contact category로 분리되어 기본 자세를 방해하지 않습니다.
손 actuator limit은 15 Nm, 물체 접촉 중 closing torque limit은 2.8 Nm입니다.
접촉 solver는 안정성 비교 결과가 가장 좋았던 Euler/Newton/elliptic 조합을 사용합니다.

`Link_Neck_Pitch`에는 ZED 2i와 같은 120 mm baseline의 투명 stereo camera가 부착되어
있습니다. Camera 버튼이 켜지면 MuJoCo가 1280x720 RGB 좌우 영상을 30 Hz로
`camera_shm.stereo_left/right`에 기록합니다. 런타임 camera worker는 시뮬레이터 실행
상태를 자동 감지해 다음 ROS2 topic도 발행합니다.

- `/left/image_rect/compressed`
- `/right/image_rect/compressed`
- `/left/camera_info`
- `/right/camera_info`

simulator worker가 `ALIVE`이면 Web UI의 Workers 영역에 `sim eye distance`가
표시됩니다. 기본값은 120 mm이며 40~200 mm 범위에서 조절할 수 있습니다. 이 설정은
좌우 가상 카메라를 중심으로 대칭 병진시키고 `/right/camera_info`의 `Tx`를 함께
갱신합니다. 시뮬레이션 영상에는 물리 카메라 calibration/rectification map을 적용하지
않으며 카메라 회전, 렌즈 왜곡 및 다른 환경에는 영향을 주지 않습니다.

렌더링을 Camera 버튼과 무관하게 항상 켜려면 `IGRIS_SIM_CAMERA_ALWAYS_ON=1`을
설정합니다. 모델 자체 검증은 다음 명령으로 실행합니다.

```bash
.venv-sim/bin/python -m igris_teleop.sim.validate_sim_model
.venv-sim/bin/python -m igris_teleop.sim.grasp_stability_benchmark
```
