# igris_reliability_runtime

IGRIS teleop에서 OpenXR hand와 chest-mounted controller의 추적 신뢰도를 30 Hz로
추론하는 ROS2 패키지입니다. 학습 workspace에 런타임 의존하지 않도록 joblib wrapper인
`train_lib`를 이 패키지에, RNN/HistGB policy를 repository의
`policy_archive/reliability`에 보관합니다.

## Topics

Hand inference:

- input: `/hmd/pose`, `/left_hand/poses`, `/right_hand/poses`
- input: `/left_hand/is_tracked`, `/right_hand/is_tracked`
- output: `/left_hand/openxr_confidence`, `/right_hand/openxr_confidence`

Controller inference:

- input defaults: `/left_controller/poses`, `/right_controller/poses`
- input: `/left_controller/is_tracked`, `/right_controller/is_tracked`
- output: `/left_controller/tracking_confidence`, `/right_controller/tracking_confidence`

통합 launch도 Unity publisher의 `/left_controller/poses`와
`/right_controller/poses`를 controller inference node에 전달합니다.

Hand fusion output:

- `/left_hybrid_hand/motor_normalized`, `/right_hybrid_hand/motor_normalized`
  (`Float32MultiArray`, 6개: thumb bend, index, middle, ring, little, thumb spread;
  normalized 0=open, 1=closed, 기존 close gain 적용 완료)
- `/left_hybrid_hand/motor_tracked`, `/right_hybrid_hand/motor_tracked` (`Bool`)

기본 launch는 `hand_retarget_fusion`을 실행합니다. 입력은 OpenXR의
`/{side}_hand/poses` (wrist+tips 6개) 및 MediaPipe의
`/{side}_mediapipe_hand/all_poses` (21개)와 각각의 `is_tracked`입니다.
원시 pose topic은 변경하지 않습니다. 기존 distance-based bridge의
`poses`/`is_tracked`/`finger_normalized` 출력은 기본 launch에서 더 이상
발행하지 않으며 legacy bridge를 동시에 실행하지 마세요.

## Build and run

```bash
cd /path/to/igris_teleop-main/ros_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select \
  mediapipe_hand_pose_bridge \
  openxr_hand_to_igris_viewer \
  igris_reliability_runtime
source install/setup.bash

export IGRIS_PROJECT_ROOT="$(cd .. && pwd)"
export IGRIS_RELIABILITY_POLICY_ROOT="$IGRIS_PROJECT_ROOT/policy_archive/reliability"
export IGRIS_RELIABILITY_PYTHON="$IGRIS_PROJECT_ROOT/.venv-ml/bin/python"
export IGRIS_MEDIAPIPE_PYTHON="$IGRIS_PROJECT_ROOT/.venv-mediapipe/bin/python"

ros2 run igris_reliability_runtime verify_policy_load --kind hand --variant rnn
ros2 run igris_reliability_runtime verify_policy_load --kind controller --variant rnn

ros2 launch igris_reliability_runtime reliability_teleop.launch.py
```

MediaPipe camera를 별도로 실행하거나 camera가 없는 점검 환경에서는:

```bash
ros2 launch igris_reliability_runtime reliability_teleop.launch.py start_mediapipe:=false
```

confidence 모델에는 `hand_model_variant:=histgb`와 `controller_model_variant:=histgb`도
지정할 수 있습니다. RNN load가 실패하면 node는 vendored HistGB policy로 자동
fallback합니다. 다만 공통 hand retargeter는 모델 variant와 관계없이 PyTorch,
Pinocchio, NLopt가 필요하므로 `.venv-ml`을 사용해야 합니다.

모델과 tracking gate를 우회해 confidence 출력만 확인할 때는 hand와 controller에
각각 `always_1`을 선택할 수 있습니다. 이 variant는 입력 pose나 tracked 상태와
무관하게 좌우 confidence를 30 Hz로 항상 `1.0` publish합니다. 추가로 hand의
`always_1`은 VR-only retargeting으로 고정하며 MediaPipe/confidence를 구독하지
않습니다. VR 입력이 유효한지에 대한 검사는 우회하지 않습니다.

```bash
ros2 launch igris_reliability_runtime reliability_teleop.launch.py \
  hand_model_variant:=always_1 \
  controller_model_variant:=always_1
```

GUI는 기본적으로 repository의 `.venv-ml/bin/python`과
`policy_archive/reliability`를 사용합니다. 직접 launch할 때는 위 환경변수를 지정합니다.
MediaPipe는 repository의 `.venv-mediapipe`를 사용합니다.

## Safety behavior

- model warmup, missing topic, stale pose에는 confidence 0을 publish합니다.
- hand fusion은 두 source 각각을 같은 DexRetargeting으로 변환한 후
  normalized 6-motor 공간에서 blend합니다. 소스별 optimizer/filter 상태는 독립적입니다.
- OpenXR만 있고 confidence가 threshold보다 낮으면 이전 command를 hold합니다.
- OpenXR가 없고 MediaPipe가 있으면 MediaPipe를 사용합니다.
- 둘 다 없으면 이전 command를 hold하고 motor_tracked=false를 publish합니다.
- 최초 유효 관측 전의 zero placeholder로 제어를 활성화하지 않습니다.
- 활성화 후 통신이 중단되면 수신부도 마지막 6개 motor command를 유지하며,
  legacy retargeter로 갑자기 전환하지 않습니다. 의도적인 Reliability-OFF/raw-VR
  전환에는 전체 프레임워크 재시작이 필요합니다.
- motor command와 torso task activation은 각각 rate limit됩니다.

## Teleop integration

`igris_teleop-main`의 Unity bridge는 raw OpenXR hand pose에서 wrist/arm 자세를 계속
사용하고, hybrid hand topic에서는 6개 motor command만 사용합니다. 따라서
MediaPipe wrist 원점이나 camera 좌표가 arm IK로 유입되지 않습니다.

VR fingertip은 기존 `hand2igris.T @ grd_yup2grd_zup` 변환을 그대로 사용합니다.
MediaPipe는 wrist와 index/middle/ring/little MCP로 직교 손바닥 기준축을 만들고
각 손 URDF의 neutral MCP 기준축으로 회전 정렬합니다. 길이를 바꾸지 않으며
카메라의 강체 회전/병진에 불변입니다. 사다리꼴 영상 전처리나 사람별 손 크기
차이까지 보정하는 것은 아닙니다. 기존 JSON 거리 calibration은 이 새 경로에서
참조하지 않습니다. 손 URDF/solver 설정과 gain은 두 소스에 공통으로 적용됩니다.

공유메모리 6-motor 필드가 추가되었으므로 업데이트 후 기존 framework와 Reliability를
모두 종료하고 재실행해야 합니다. 기존 로봇 SDK 토픽·서비스·hand init은 변경하지 않습니다.

컨트롤러 상체 입력은 다음 순서로 처리됩니다.

1. 각 controller pose에 고정 `controller_to_chest` transform을 합성합니다.
2. 좌/우 confidence로 위치 평균과 quaternion SLERP를 계산합니다.
3. 좌/우 후보 불일치가 크면 consistency gate로 activation을 낮춥니다.
4. OR confidence와 gate를 합친 torso alpha를 비대칭 rate limit합니다.
5. alpha를 기존 단일 IK/QP의 torso task weight로 전달합니다. 별도 관절 벡터를
   계산하거나 두 IK 결과를 관절 공간에서 blend하지 않습니다.

기본 controller-to-chest 값은 장착 위치를 위한 시작값일 뿐입니다. teleop worker를
시작하기 전에 각 컨트롤러 좌표계에서 chest 좌표계로 가는 transform을
`x,y,z,qx,qy,qz,qw` 순서로 설정합니다.

```bash
export IGRIS_LEFT_CONTROLLER_TO_CHEST="0.064,0,0,0,0,0,1"
export IGRIS_RIGHT_CONTROLLER_TO_CHEST="-0.064,0,0,0,0,0,1"
```

운용 중 1 Hz 로그의 `alpha`, `gate`, 좌/우 confidence, position/rotation disagreement를
확인합니다. controller pose 또는 confidence가 stale이면 해당 confidence는 0으로
clamp되고 torso task는 마지막 target을 유지한 채 빠르게 비활성화됩니다.
