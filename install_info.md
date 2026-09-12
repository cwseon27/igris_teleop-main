# 다른 PC로 이식하기: 설치·복원·실행·검증

갱신: 2026-09-12. 기준 환경은 Ubuntu 24.04 x86_64, Python 3.12, ROS 2 Jazzy입니다.
이 문서는 현재 PC의 기능을 새 PC에서 재현하기 위한 전체 절차입니다.
설치 스크립트 설명은 [INSTALL.md](./INSTALL.md), 모드별 사용법은
[README.md](./README.md)를 참고하세요.

소스만 복사하는 것과 현재 운용 상태를 보존하는 것은 다릅니다. **소스·정책·보정 맵·
SDK는 보존하고, Python 환경과 ROS 빌드 결과는 새 PC에서 다시 만듭니다.**
카메라 경로, LAN 주소, 외부 ROS workspace, 실제 장착 보정은 새 PC에서 확인합니다.
이 절차는 PC에만 적용하며 로봇 내부 코드·서비스·네트워크를 변경하지 않습니다.

## 1. 반드시 보존할 파일과 별도 준비물

| 항목 | 프로젝트 내 위치 / 준비 사항 |
| --- | --- |
| 실행 코드·설치 스크립트·의존성 lock | `igris_teleop/`, `ros_ws/src/`, `requirements/`, 루트 실행 스크립트 |
| Hand / controller(chest) RNN·HistGB 4개 | `policy_archive/reliability/`, `policy_archive/SHA256SUMS` |
| 현재 4:3 stereo 보정 맵 | `ros_ws/src/stereo_sbs_cam_pub/config/stereo_rectify_maps_tuned.npz` |
| 카메라 코드의 보정 맵 사본과 16:9 백업 | `igris_teleop/teleop_devices/cameras/` 및 위 `config/`의 `*.npz` |
| 손 retargeting URDF·mesh·설정 | `igris_teleop/hand_control/hand_urdf/`, `igris_teleop/hand_control/unity_baseline/` |
| 몸체 URDF·mesh·IK·관절 보정 | `igris_teleop/robot_control/asset/`, `igris_teleop/config/`, `igris_teleop/sim/robot/` |
| **실물 MS 시작용 PR2AB 보정본** | **`igris_artifacts/logs/robot_control/pr2ab_calibration.yaml`** — 5개 pair의 변환과 `ab_limits` 포함, Git 추적 필수 |
| Python/C++ 공개 SDK 배포물 | `ros_ws/src/igris_c_ros_bridge/thirdparty/igris_c_sdk_public/`의 `dist/`, `include/`, `lib/`, `thirdparty/` |
| 현재 로봇 hand 메시지와 맞는 SDK snapshot | `local_state/robot_hand_sdk/include/igris_sdk/igris_c_msgs.hpp`, `local_state/robot_hand_sdk/lib/libigris_sdk.a` |
| Walking policy | `igris_artifacts/walking/policy_1.pt`, `igris_artifacts/walking/v2/model_0030000.onnx` |
| ML vendor patch / Web UI 정적 자산 | `vendor_patches/`, `igris_teleop/web_ui/static/` |
| 개인별 설정 — 별도 백업 | `igris_artifacts/config/hybrid_teleop.json`, `local_state/cyclonedds_igris_lan.xml`, 개인 실행 옵션·환경변수 |
| 외부 구성 요소 | 새 PC의 기존 ROS-TCP-Endpoint, Unity/OpenXR 클라이언트, 연결할 VR/USB 장치 |

`local_state/robot_hand_sdk`는 일반 캐시가 아니라 **현재 실물 손 bridge 빌드에 필요한
추적 대상 자산**입니다. 기존 공개 Python SDK wheel과 이 snapshot은 용도가 다릅니다.
둘 중 하나로 통합하거나 임의 SDK 버전으로 바꾸면 메시지 호환성 문제가 재발할 수
있습니다. 배포 라이선스와 provenance도 해당 디렉터리의 README와 함께 보존합니다.

IGRIS SDK에는 x86_64 / CPython 3.12 바이너리가 포함되어 있습니다. ARM, Windows,
다른 Python 버전에 그대로 옮기는 것은 지원되지 않습니다. ACT/Diffusion/PI 등의
개인 checkpoint, dataset, 외부 Unity 프로젝트/APK, 외부 ROS-TCP-Endpoint 설치물은
이 저장소만으로 복원되지 않습니다. 해당 기능을 사용했다면 별도로 확보하세요.
SSH 키·비밀번호·토큰은 Git에 넣지 않습니다.

PR2AB 운용본은 경로에 `logs`가 있어도 재생성 가능한 로그가 아닙니다. 현재 제어
worker는 위 경로를 명시적으로 읽습니다. 동명의
`igris_teleop/config/robot_control/pr2ab_calibration.yaml`에는 `ab_limits`가 없어
단순 복사로 대체할 수 없습니다. 이식 시 운용본을 보존하며, 다른 로봇/기구 구성에는
보정의 호환성을 별도로 확인해야 합니다. 안전 검사 제거, 임의 범위 확대, PJS 모드로
강제 전환하는 방식으로 파일 누락을 우회하지 마세요.

## 2. 원본 PC 백업과 새 PC 체크아웃

프레임워크를 정상 종료한 후 같은 Git commit을 새 PC에 내려받습니다.

```bash
git clone https://github.com/cwseon27/igris_teleop-main.git
cd igris_teleop-main
git rev-parse HEAD
export IGRIS_ROOT="$(pwd -P)"
```

저장소가 Private이면 접근 권한이 있는 GitHub 계정으로 인증해야 합니다. Git에는 commit한
파일만 포함되므로 원본 PC에서 `git status --short`도 확인하세요. `.venv*`, ROS
`build/install/log`, 캐시, 일반 실행 로그, 녹화물은 Git으로 복원하지 않습니다.
예외적으로 실물 PR2AB 운용 보정본은 Git에 포함합니다. 향후 Git LFS를
도입한다면 `git lfs pull`까지 실행하여 pointer가 아닌 실제 자산을 확보합니다.

현재 카메라/개인 LAN 설정은 별도 비공개 백업으로 옮깁니다. 원본 PC의 프로젝트
루트에서 다음 예시를 실행하면 존재하는 설정만 보관합니다.

```bash
IGRIS_SETTINGS_BACKUP="$HOME/igris_settings_$(date +%Y%m%d_%H%M%S).tar.gz"
IGRIS_SETTINGS_FILES=()
for item in igris_artifacts/config/hybrid_teleop.json \
  local_state/cyclonedds_igris_lan.xml \
  local_state/controller_to_chest_calibration.env \
  local_state/controller_to_chest_calibration.json \
  igris_artifacts/logs/robot_control/pr2ab_calibration.yaml \
  igris_teleop_desktop_options.env; do
  if [ -f "$item" ]; then IGRIS_SETTINGS_FILES+=("$item"); fi
done
if ((${#IGRIS_SETTINGS_FILES[@]})); then
  tar -czf "$IGRIS_SETTINGS_BACKUP" "${IGRIS_SETTINGS_FILES[@]}"
  sha256sum "$IGRIS_SETTINGS_BACKUP"
fi
```

추가로 바꾼 환경변수, GUI 밖의 calibration 파일, 사용자 checkpoint도 목록과 함께
백업합니다. 새 PC에 압축을 풀 때는 내용을 먼저 확인하고 기존 설정을 덮어쓰기 전
별도 보관하세요. 원본 LAN 주소/USB 경로는 복원 후 그대로 적용하지 말고 6절대로
확인합니다. `.bashrc` 전체나 원본 PC의 `/etc` 설정은 덮어쓰지 않습니다.

정책·SDK·보정 자산의 무결성을 확인합니다.

```bash
cd "$IGRIS_ROOT/policy_archive"
sha256sum -c SHA256SUMS
cd "$IGRIS_ROOT"
sha256sum -c docs/runtime_assets.sha256
test -f local_state/robot_hand_sdk/include/igris_sdk/igris_c_msgs.hpp
test -f local_state/robot_hand_sdk/lib/libigris_sdk.a
(cd local_state/robot_hand_sdk && sha256sum -c SHA256SUMS)
test -f ros_ws/src/stereo_sbs_cam_pub/config/stereo_rectify_maps_tuned.npz
test -f igris_artifacts/logs/robot_control/pr2ab_calibration.yaml
```

### 기존 clone의 PR2AB 누락 오류 복구

최초 배포 commit `aefe8dc`에는 실물에서 읽는 PR2AB 운용본이 빠져 있었습니다.
이후 수정은 원래 PC의 정상 운용본(값과 경로 동일)을 배포에 포함합니다. 실패 원인이
controller에 저장되므로 파일만 내려받고 실행 중인 프로세스를 그대로 두면 안 됩니다.

프레임워크를 정상 종료한 후, **갱신할 프로젝트 루트**에서 실행합니다. 다른 도구가
이미 같은 경로에 파일을 만들었다면 먼저 백업하고, 그 PC에 맞춰 별도로 보정한
파일이면 배포본으로 덮기 전에 비교하세요.

```bash
IGRIS_PR2AB="igris_artifacts/logs/robot_control/pr2ab_calibration.yaml"
if [ -f "$IGRIS_PR2AB" ]; then
  IGRIS_CALIB_BACKUP="$(mktemp -d /tmp/igris-pr2ab-backup.XXXXXX)"
  cp -p "$IGRIS_PR2AB" "$IGRIS_CALIB_BACKUP/pr2ab_calibration.yaml"
  printf '기존 보정 백업: %s\n' "$IGRIS_CALIB_BACKUP"
fi
git pull --ff-only
sha256sum -c docs/runtime_assets.sha256
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_pr2ab_deployment.py
```

이 테스트는 로봇에 접속하지 않고 배포 파일·5개 pair·모터 범위와 시작 안전 검사를
확인합니다. 검사 후 기존 launcher로 프레임워크를 완전히 재시작하세요. 이번 보정
배포 수정 자체는 venv 재설치나 ROS 재빌드가 필요하지 않습니다.

## 3. 새 PC 시스템 준비

ROS 2 Jazzy 자체는 먼저 설치되어 있어야 합니다. 기존 `ros_tcp_endpoint`는 **재이식하거나
재빌드하지 않고 사용**합니다. 새 PC에 없다면 compatible endpoint를 별도 설치해야
하며 아래 내부 ROS build가 이를 대신하지는 않습니다.

```bash
uname -m
python3.12 --version
test -f /opt/ros/jazzy/setup.bash

sudo apt update
sudo apt install -y \
  build-essential cmake curl git acl v4l-utils \
  libgl1 libglib2.0-0 libportaudio2 libssl-dev libopencv-dev \
  python3.12 python3.12-venv python3-pip \
  python3-colcon-common-extensions python3-rosdep python3-pytest \
  ros-jazzy-dynamixel-sdk
```

`rosdep`을 처음 쓰는 PC만 `sudo rosdep init`을 실행하고, 이후 `rosdep update`를
실행합니다. ROS package 의존성은 5절에서 선택한 source를 기준으로 설치합니다.

## 4. Python 환경은 모두 새로 설치

표준 launcher는 다음 **다섯 환경 중 앞의 네 개**를 시작 전에 검사합니다.
Reliability + MediaPipe를 포함한 전체 기능에는 다섯 개 모두 필요합니다.

| 환경 | 역할 | 설치 방법 |
| --- | --- | --- |
| `.venv` | GUI, orchestration, ROS/로봇 runtime | 메인 installer |
| `.venv-sim` | MuJoCo | 메인 installer |
| `.venv-ik` | Pinocchio / ProxSuite 몸체 IK | 메인 installer |
| `.venv-ml` | 손 retargeting, Reliability, PyTorch/LeRobot | 메인 installer |
| `.venv-mediapipe` | MediaPipe / 전용 OpenCV / ROS camera 입력 | 별도 MediaPipe installer |

원본 환경을 복사하거나 이름만 변경하지 않습니다. conda/다른 venv가 활성화되어
있다면 새 터미널에서 시작하세요. 새 체크아웃에서는 `--reset`이 필요 없습니다.
`--reset`은 이미 존재하는 선택된 환경을 삭제 후 재생성하므로 실행 중에는 사용하지 않습니다.

```bash
cd "$IGRIS_ROOT"
# 제공된 cu130 lock과 호환되는 NVIDIA 환경:
./install_igris_teleop.sh --torch-backend cu130

# 이어서 MediaPipe 전용 환경 설치:
bash ros_ws/src/mediapipe_hand_pose_bridge/scripts/setup_mediapipe_venv.sh
```

CUDA 구성과 맞지 않거나 CPU 환경이면 위 첫 명령 대신 다음을 사용합니다.

```bash
./install_igris_teleop.sh --refresh-locks --torch-backend cpu
```

`--refresh-locks`는 모든 `requirements/*.lock.txt`를 다시 생성합니다. 원래 lock을
보존한 체크아웃에서 실행하고 변경 diff를 확인하세요. GPU driver/CUDA 호환성과
CPU에서의 실제 처리 속도는 새 PC에서 별도로 확인해야 합니다.

메인 installer는 **MediaPipe 환경, ROS package 빌드, LAN/USB 권한을 설정하지 않습니다.**
`--skip-ml`은 전체 teleop 설치의 지름길이 아닙니다. 표준 launcher가 `.venv-ml`을
요구하고, 현재 공통 hand retarget backend는 `histgb`/`always_1`에서도 Torch,
Pinocchio, NLopt를 사용합니다. MediaPipe 환경은 NumPy 1.26.4 / OpenCV contrib
4.11.0.86 / MediaPipe 0.10.21 / JAX 0.7.1로 분리되어 있으므로 `.venv-ml`에 합치지 마세요.

## 5. 내부 ROS workspace 빌드

새 PC의 `/opt/ros/jazzy`와 기존 endpoint workspace를 source합니다. 외부 endpoint의
실제 경로를 지정하고, 내부 package는 새 프로젝트 경로에서 빌드합니다.

```bash
source /opt/ros/jazzy/setup.bash
export IGRIS_ROS_WS_ROOT="/absolute/path/to/existing_endpoint_workspace"
source "$IGRIS_ROS_WS_ROOT/install/setup.bash"
ros2 pkg prefix ros_tcp_endpoint

cd "$IGRIS_ROOT/ros_ws"
rosdep install --from-paths \
  src/igris_c_sdk \
  src/igris_lowcmd_relay \
  src/igris_c_ros_bridge/igris_c_hand \
  src/igris_c_ros_bridge/igris_c_sensor \
  src/stereo_sbs_cam_pub \
  src/igris_leader_control \
  src/mediapipe_hand_pose_bridge \
  src/openxr_hand_to_igris_viewer \
  src/igris_reliability_runtime \
  --ignore-src -r -y

colcon build --symlink-install --packages-select \
  igris_c_sdk igris_lowcmd_relay igris_c_hand igris_c_sensor stereo_sbs_cam_pub \
  igris_leader_control mediapipe_hand_pose_bridge \
  openxr_hand_to_igris_viewer igris_reliability_runtime

source install/setup.bash
for package in igris_c_sdk igris_lowcmd_relay igris_c_hand igris_c_sensor stereo_sbs_cam_pub \
  igris_leader_control mediapipe_hand_pose_bridge \
  openxr_hand_to_igris_viewer igris_reliability_runtime; do
  ros2 pkg prefix "$package"
done
ros2 pkg prefix ros_tcp_endpoint
```

앞의 9개는 새 `<repo>/ros_ws/install/...`, endpoint는 기존 설치를 가리켜야 합니다.
이 명령에는 `ros_tcp_endpoint`가 포함되지 않습니다. 복사된 오래된 `build/install/log`
때문에 이전 절대 경로가 나타난다면 그 결과를 재사용하지 말고 깨끗한 체크아웃에서
다시 빌드하세요.

- Reliability 핵심 3개: `igris_reliability_runtime`, `mediapipe_hand_pose_bridge`,
  `openxr_hand_to_igris_viewer`.
- 실물 body 제어: ROS interface `igris_c_sdk`와 300 Hz `igris_lowcmd_relay`.
- 실물 손: robot-matched snapshot을 링크하는 **PC 측** `igris_c_hand`.
- 실물 이미지: 현재 로봇 메시지용 `igris_c_sensor_robot_bridge_node`를 설치하는
  `igris_c_sensor`. Python 환경이나 Reliability 3개만 설치해서는 이미지가 복원되지 않습니다.
- `stereo_sbs_cam_pub`: 로컬 SBS publisher와 보정 자산. `igris_c_sensor`가 실행 의존성으로
  선언하므로 깨끗한 설치에서는 함께 빌드합니다.
- masterarm: `igris_leader_control`. 별도 시각화 launch를 쓴다면
  `igris_teleop_visualize` 의존성과 package도 추가 빌드합니다.

`igris_c_hand`가 snapshot 누락으로 실패하면 위 필수 파일을 원본에서 복구하세요.
로봇 내부에 접속하거나 다른 SDK로 교체해서 우회할 단계가 아닙니다.

## 6. PC마다 다시 확인할 네트워크·장치 설정

### LAN과 DDS

현재 로봇 구성의 참고값은 robot LAN `192.168.11.2`, Wi-Fi `192.168.4.1`,
robot namespace `igris_c_IG05`입니다. 다른 로봇에도 같은 값이라고 가정하지 마세요.
새 PC의 실제 LAN 인터페이스와 미사용 IPv4를 확인하고, 인터넷용 Wi-Fi의 default
route는 유지합니다. 로봇 전용 직접 LAN에는 인터넷 gateway를 중복 지정하지 않습니다.

```bash
ip -brief address
ip route
ip route get 192.168.11.2
```

`local_state/cyclonedds_igris_lan.xml`을 복원했다면 `NetworkInterface address`는
**로봇 주소가 아니라 새 PC의 LAN 주소**로 바꾸세요. 다음은 PC가
`192.168.11.18/24`를 실제로 사용하며 주소 충돌이 없을 때의 예시입니다.
동일한 [배포용 예시 파일](./docs/examples/cyclonedds_igris_lan.xml)도 포함되어 있습니다.
원본 설정이 없다면 이를 `local_state/cyclonedds_igris_lan.xml`로 복사하고 실제 PC 주소로
편집한 뒤 사용하세요. 예시 파일 자체는 launcher가 자동 적용하지 않습니다.

```xml
<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface address="192.168.11.18" autodetermine="false" />
      </Interfaces>
      <DontRoute>true</DontRoute>
    </General>
  </Domain>
</CycloneDDS>
```

이 LAN 전용 profile의 `DontRoute`는 로봇이 동시에 광고하는 Wi-Fi locator를
잘못 선택하는 것을 방지하기 위한 설정입니다. 해당 파일이 있으면 launcher가
기본 적용합니다. 별도 SDK GUI/터미널도 같은 profile로 진단하려면:

```bash
export CYCLONEDDS_URI="file://$IGRIS_ROOT/local_state/cyclonedds_igris_lan.xml"
```

DDS 설정은 프로세스 시작 시 읽으므로 변경 후 관련 프로세스를 모두 재시작합니다.
`hand_init` 성공이나 `ros2 topic list`만으로 연속 명령·이미지 전달까지 정상이라고
판단하지 마세요. 도메인, namespace, QoS 및 실제 sample 수신을 확인합니다.
실물 raw DDS body/hand/camera는 현재 domain 0, 시뮬레이터 로봇 DDS는 domain 99를
사용합니다. 일반 camera-mode 옵션의 domain 기본값 1과 혼동하지 마세요.

### 손 카메라의 고정 경로

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-path/
ls -l /dev/v4l/by-id/
```

GUI `Hybrid Teleop Info`에서 좌우 영상을 확인하고 `/dev/v4l/by-path/...-video-index0`
또는 유일한 serial의 `by-id`로 저장합니다. 동일 serial 카메라는 `by-id`가 충돌할
수 있으므로 USB 포트 기반 `by-path`를 씁니다. `/dev/video2` 같은 숫자만 고정하지
마세요. 새 PC/USB 포트/허브 배치가 달라지면 기존 `by-path`도 다시 선택해야 합니다.
기존 udev 고정 symlink를 쓰는 경우에도 새 PC의 serial/USB 속성부터 확인합니다.

MediaPipe 기본값은 카메라당 손 1개, `model_complexity=1`, `mirror=false`,
`swap_lr=true`, trapezoid 활성화입니다. 복원한
`igris_artifacts/config/hybrid_teleop.json`에는 개인 설정이 저장됩니다.
GUI `Start MediaPipe test`에서 raw / 전처리 / landmark를 확인하세요.
Reliability를 시작하면 GUI가 카메라 테스트를 종료하므로 두 프로세스로 같은
카메라를 동시에 열지 않습니다.

### USB 권한·장착 보정

leader serial 접근이 필요하면 `sudo usermod -aG dialout "$USER"`, 카메라의
장치 그룹 권한이 필요하면 `sudo usermod -aG video "$USER"` 후 로그아웃/로그인합니다.
현재 세션에만 serial ACL을 줄 때는 실제 장치를 확인한 후
`sudo setfacl -m "u:$USER:rw" /dev/ttyUSB0`을 사용합니다. 재연결 후 ACL이 사라질 수 있습니다.

leader 관절 offset, PR2AB 보정, controller-to-chest transform은 **같은 장착 상태일 때만**
재사용합니다. 컨트롤러 장착이 바뀌면 ROS 입력이 들어오는 상태에서
`.venv/bin/python controller_calibration.py`를 실행하여 보정합니다.
새 PC로 옮긴다는 이유만으로 기존 정상 OpenXR 축변환·관절 부호를 수정하지 마세요.

## 7. 로봇을 움직이지 않는 설치 검증

프로젝트 루트에서 ROS와 내부 overlay를 source합니다.

```bash
cd "$IGRIS_ROOT"
source /opt/ros/jazzy/setup.bash
source ros_ws/install/setup.bash
export IGRIS_PROJECT_ROOT="$IGRIS_ROOT"
export IGRIS_RELIABILITY_PYTHON="$IGRIS_ROOT/.venv-ml/bin/python"
export IGRIS_MEDIAPIPE_PYTHON="$IGRIS_ROOT/.venv-mediapipe/bin/python"

ros2 run igris_reliability_runtime verify_policy_load --kind hand --variant rnn
ros2 run igris_reliability_runtime verify_policy_load --kind hand --variant histgb
ros2 run igris_reliability_runtime verify_policy_load --kind controller --variant rnn
ros2 run igris_reliability_runtime verify_policy_load --kind controller --variant histgb

.venv-mediapipe/bin/python -c 'import mediapipe, cv2, rclpy; print(mediapipe.__version__, cv2.__version__)'
.venv-ml/bin/python -c 'import torch, pinocchio, nlopt; from igris_teleop.hand_control.hand_retargeting import HandRetargeting; HandRetargeting(); print("hand retarget ready")'
.venv-sim/bin/python -m igris_teleop.sim.validate_sim_model
```

policy `path=`가 새 프로젝트 내부를 가리키는지 확인합니다. 두 라이브러리 모두
`igris_c_sdk`라는 이름을 쓰므로, raw SDK를 확인할 때 ROS interface package가 먼저
import되지 않도록 별도 명령으로 검사합니다.

```bash
env -u PYTHONPATH .venv/bin/python -c 'import igris_c_sdk; assert hasattr(igris_c_sdk, "ChannelFactory"); print(igris_c_sdk.__file__)'
```

회귀 검사는 환경을 나눠 실행합니다. 루트 테스트에서 선택적 의존성 때문에 skip된
손·IK 테스트를 전체 통과라고 간주하지 말고 다음 전용 환경 검사도 실행하세요.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_pr2ab_deployment.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-mediapipe/bin/python -m pytest -q \
  ros_ws/src/mediapipe_hand_pose_bridge/test \
  ros_ws/src/openxr_hand_to_igris_viewer/test
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-ml/bin/python -m pytest -q \
  ros_ws/src/igris_reliability_runtime/test \
  tests/test_hand_retargeting.py tests/test_hand_optimizer_gradient.py \
  tests/test_hybrid_motor_command.py tests/test_hand_hybrid_command_hold.py \
  tests/test_hand_worker_sim.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-ik/bin/python -m pytest -q \
  tests/test_prox_ik_pelvis_env.py tests/test_waist_task_transition.py
```

GUI가 쓰는 launch/script도 확인합니다.

```bash
ros2 pkg executables igris_reliability_runtime
ros2 launch igris_reliability_runtime reliability_teleop.launch.py --show-args
```

GUI `Start Reliability`는 `<repo>/ros_ws/install/setup.bash`,
`<repo>/policy_archive/reliability`, `.venv-ml`, `.venv-mediapipe`를 기본 사용합니다.
이전 PC의 `IGRIS_RELIABILITY_*`, `IGRIS_MEDIAPIPE_PYTHON`, `IGRIS_ML_PYTHON`,
`IGRIS_IK_PYTHON`, `IGRIS_SIM_PYTHON` 절대 경로가 shell에 남아 있으면 제거하거나
새 경로로 갱신하세요.

## 8. 실행 및 시뮬레이션 우선 검증

기존 ROS-TCP-Endpoint workspace를 source한 터미널에서 시작합니다. Unity 클라이언트의
접속 IP도 새 PC의 실제 주소로 설정하고 TCP port 10000 연결을 확인합니다.
이미 같은 endpoint가 실행 중이면 중복으로 실행하지 않습니다.

```bash
cd "$IGRIS_ROOT"
source /opt/ros/jazzy/setup.bash
source "$IGRIS_ROS_WS_ROOT/install/setup.bash"
source ros_ws/install/setup.bash

# VR 상체/손 + controller chest hybrid:
./run_igris_teleop.sh --mode teleop --teleop-device unity_hybrid --teleop-hand-source vr

# 또는 VR 상체/손 + masterarm 팔:
# ./run_igris_teleop.sh --mode teleop --teleop-device vr_masterarm --teleop-hand-source vr
```

Web UI 기본 주소는 `http://127.0.0.1:8000/`입니다. `unity`는 Reliability/MediaPipe를
사용하지 않는 VR-only baseline입니다. 공통 VR/MediaPipe 손 retargeting을 검증하려면
`unity_hybrid` 또는 `vr_masterarm`, hand source `vr`를 선택합니다.

처음에는 실물 로봇 전원을 끄거나 통신을 분리하고 GUI에서 simulator를 먼저 켭니다.
simulator가 `ALIVE`이고 제어 대상이 sim/domain 99인지 확인한 뒤 `control` 그룹
(몸체와 손 worker)을 시작합니다. Ready → Start → Home → Shutdown을 시뮬레이션에서 확인하세요.
실물 제어를 켜놓은 상태에서 simulator를 추가로 켜는 방식으로 시험하지 않습니다.

GUI에서 카메라 확인 후 `Start Reliability`를 누르고 다음을 확인합니다.

1. hand/controller의 `rnn`, `histgb`가 정상 시작하고 confidence topic이 갱신됩니다.
2. VR 손 입력 `/left_hand/poses`, `/right_hand/poses`와 `is_tracked`, MediaPipe의
   `/left_mediapipe_hand/all_poses`, `/right_mediapipe_hand/all_poses`가 실제 sample을 냅니다.
3. `/left_hybrid_hand/motor_normalized`, `/right_hybrid_hand/motor_normalized`에는
   각각 6개 값, 각 `/motor_tracked`에는 유효 상태가 옵니다. VR와 MediaPipe 모두
   동일 `HandRetargeting`/URDF를 거친 최종 모터 명령을 혼합하는 경로입니다.
4. hand `always_1`에서는 VR만 사용하고 MediaPipe로 전환하지 않습니다. 입력 소실 시
   이미 활성화된 손은 마지막 명령을 유지합니다.
5. 카메라 worker와 sim stereo를 켰을 때 GUI와 VR 영상도 확인합니다.

현재 코드에는 6모터 명령용 공유메모리 필드가 추가되어 있습니다. 버전 교체 후에는
**프레임워크와 Reliability를 모두 종료한 후 재실행**해야 합니다. old/new worker를
섞거나 일부분만 재시작하지 마세요. Reliability를 끄고 raw VR로 돌아갈 때도
마지막 명령 hold latch를 해제하려면 전체 재시작이 필요합니다.

시뮬레이션 검증 후 실물은 별도 세션으로 시험합니다. 비상정지·호이스트·주변 확보,
관절 방향/범위/게인·최신 state·제어 지연을 확인한 뒤 사용자가 Ready/Hand Initial을
실행하세요. 설치 성공이 실제 로봇 운용 안전성이나 새 PC 성능을 보장하지는 않습니다.

실물 stereo 경로는 `/igris_c_IG05/sensor/eyes_stereo/compressed`의 정방향 SBS를
좌우 분리 → 각 4:3 보정 map 적용 → 기존 `/left/image_rect/compressed`,
`/right/image_rect/compressed` 출력입니다. 별도 180도 회전은 하지 않습니다.
현재 4:3 map과 입력·출력 크기가 일치하면 크기 변환이 필요하지 않습니다. 코드에는
규격 불일치 시 resize 경로가 있으므로 다른 카메라/map을 쓸 때는 크기도 확인하세요.
topic 이름만 보이고 영상 sample이 없다면 camera bridge 설치,
도메인/namespace/LAN profile과 원본 수신부터 확인하세요.

## 9. 이식 완료 체크리스트

- [ ] 같은 commit의 source·lock·정책·URDF/mesh·보정 map 확보 및 checksum 확인
- [ ] 공개 SDK 배포물 + robot-matched hand SDK snapshot 확보
- [ ] 실제 MS 경로의 PR2AB 운용본 및 5개 pair의 `ab_limits` 검사 (일반 로그와 함께 제외하지 않기)
- [ ] 필요한 개인 설정·checkpoint·Unity 클라이언트 별도 백업/복원
- [ ] Python 환경 5개를 새 PC에서 생성
- [ ] Reliability 핵심 3개와 PC용 body/hand/image/leader ROS package 빌드
- [ ] 외부 ROS-TCP-Endpoint 재사용 및 새 PC IP로 Unity 연결
- [ ] LAN profile의 **PC 주소**, USB 고정 경로, 권한·장착 보정 재확인
- [ ] hand/controller RNN·HistGB 4개 load, MediaPipe/OpenCV/rclpy import 확인
- [ ] 전용 환경별 회귀 검사 및 MuJoCo model 검증
- [ ] 전체 재시작 후 GUI Start Reliability와 실제 topic sample 확인
- [ ] 실물과 분리된 시뮬레이션에서 Ready/Start/Home/Shutdown 및 손/영상 확인

추가 동작 설명은
[Reliability runtime 문서](./ros_ws/src/igris_reliability_runtime/README.md)를 참고하세요.

## 10. 문서 갱신 시 검증 범위 (2026-09-12)

- Git 배포 대상만 별도 임시 경로에 꺼내어 핵심 바이너리 자산 18개와 hand SDK
  header/library checksum을 확인했습니다.
- 기존 ROS `build/install/log`를 복사하지 않고 5절의 내부 9개 패키지를 새 경로에서
  빌드했습니다. 기존 ROS-TCP-Endpoint는 수정하거나 빌드하지 않았습니다.
- 새 소스 경로의 GUI Reliability 실행 명세를 검사했습니다. 이 검사는 설치된 Python
  의존성 환경을 명시적으로 재사용했으며 실제 GUI/로봇 제어 프로세스는 시작하지 않았습니다.
- 현재 PC에서 정책 4개 로드, MediaPipe/OpenCV/rclpy import, 루트 테스트 241개 통과
  (8개 skip), MediaPipe/viewer 테스트 47개 및 runtime/손 관련 테스트 39개 통과를
  확인했습니다. 테스트 집합 일부가 겹치므로 합산하지 않습니다.

이는 소스·자산 보존과 새 경로 빌드 확인이며, 별도 PC의 GPU/USB/네트워크와 실제
로봇 동작까지 검증했다는 의미는 아닙니다. 이식 후 위 체크리스트를 다시 수행하세요.

### PR2AB 배포 누락 정정 (2026-09-12)

위 최초 배포 검증은 실물 MS 시작에 필요한 로그 경로의 PR2AB 운용본 누락을 잡지
못했습니다. 정상 운용본을 값·경로 변경 없이 Git에 추가하고 자산 manifest를 19개로
확장했습니다. 새 소스 체크아웃에서 보정 파일과 5개 pair의 `ab_limits`를 검사하는
비구동 테스트 10개가 통과했고, 루트 회귀 테스트는 251개 통과·8개 skip입니다.
실제 로봇의 state/토크/운동 검증은 수행하지 않았으며 제어 수식과 안전 검사는
변경하지 않았습니다.
