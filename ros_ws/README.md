# ros_ws

IGRIS teleop와 같이 사용하는 ROS2 workspace입니다. leader arm, stereo camera, Unity bridge, visualization, ROS TCP endpoint 관련 ROS package를 이곳에서 build/run합니다.

## 구성

- `src/DynamixelSDK/`: Dynamixel SDK ROS package입니다.
- `src/ROS-TCP-Endpoint/`: Unity ROS TCP endpoint입니다.
- `src/igris_c_description_public/`: IGRIS-C description/mesh/URDF package입니다.
- `src/igris_c_ros_bridge/`: IGRIS-C hand/control bridge 관련 package입니다.
- `src/igris_leader_control/`: leader arm 제어 package입니다.
- `src/igris_teleop_visualize/`: leader arm/teleop visualization launch package입니다.
- `src/stereo_sbs_cam_pub/`: stereo camera publish package입니다.
- `src/igris_reliability_runtime/`: hand/controller confidence policy inference와 통합 launch입니다.
- `src/mediapipe_hand_pose_bridge/`: 좌우 camera MediaPipe hand landmark publisher입니다.
- `src/openxr_hand_to_igris_viewer/`: OpenXR/MediaPipe hand fusion bridge입니다.
- `build/`, `install/`, `log/`: `colcon build` 산출물이며 git 추적 대상이 아닙니다.

## Build

```bash
cd ~/igris_teleop_v4/ros_ws

source ../.venv/bin/activate
source /opt/ros/jazzy/setup.bash

colcon build
source install/setup.bash
```

Reliability/MediaPipe 패키지만 빌드할 때:

```bash
colcon build --symlink-install --packages-select \
  mediapipe_hand_pose_bridge \
  openxr_hand_to_igris_viewer \
  igris_reliability_runtime
```

Policy 파일은 workspace 밖으로 나가지 않고 repository root의
`policy_archive/reliability`에서 읽습니다. Web UI는 필요한 경로와 Python executable을
managed ROS process에 자동 전달합니다.

## 주요 실행

```bash
ros2 launch stereo_sbs_cam_pub run.launch.py
ros2 launch igris_teleop_visualize visualize.launch.py
ros2 run ros_tcp_endpoint default_server_endpoint
```

## Hand on/off

`ros2 launch igris_teleop_visualize visualize.launch.py`에서 손 사용 여부는 launch argument가 아니라 `leader_node`가 읽는 환경변수 `IGRIS_LEADER_HAND`로 결정됩니다.

- 기본값: `0` 또는 unset
- `0`: 손 모터는 읽지 않고 `Finger_*` joint는 `0.0` 더미 값으로 publish
- `1`: 손 모터 ID `18, 19, 28, 29`를 함께 읽어서 publish

```bash
IGRIS_LEADER_HAND=0 ros2 launch igris_teleop_visualize visualize.launch.py
IGRIS_LEADER_HAND=1 ros2 launch igris_teleop_visualize visualize.launch.py
```

실행 로그에 `Leader hand mode: ENABLED ...` 또는 `DISABLED ...`가 출력됩니다.

## Leader serial 권한

`/dev/ttyUSB0`가 `root:dialout`의 `0660` 장치라면 현재 사용자가 읽고 쓸 수 있어야
leader node가 시작됩니다. 현재 로그인 세션에 즉시 권한을 주려면 다음을 실행합니다.

```bash
sudo setfacl -m u:$USER:rw /dev/ttyUSB0
```

재부팅이나 USB 재연결 후에도 유지하려면 사용자를 `dialout`에 추가한 뒤 로그아웃하고
다시 로그인합니다.

```bash
sudo usermod -aG dialout $USER
```

## 운영 메모

- ROS2 bridge를 사용하는 Python 실행은 먼저 `.venv`와 ROS2 setup을 모두 source해야 합니다.
- Web UI의 ROS 관련 worker는 이 workspace의 build/install 상태에 영향을 받습니다.
- Unity 연결은 `ros_tcp_endpoint` 실행 상태를 확인해야 합니다.
