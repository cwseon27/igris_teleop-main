# openxr_hand_to_igris_viewer

VR/OpenXR hand PoseArray topic을 받아서:

1. 각 fingertip position의 norm을 계산
2. calibration JSON의 min/max로 0~1 정규화
3. `/right_hand/finger_normalized` 또는 `/left_hand/finger_normalized` publish
4. IGRIS hand URDF joint 이름에 맞춰 `/joint_states` publish
5. `robot_state_publisher` + RViz로 hand 움직임 확인

## 입력 topic

기본 오른손:

- `/right_hand/poses` (`geometry_msgs/PoseArray`)
- `/right_hand/is_tracked` (`std_msgs/Bool`)
- `/right_mediapipe_hand/poses` (`geometry_msgs/PoseArray`, fallback)

기본 왼손:

- `/left_hand/poses`
- `/left_hand/is_tracked`
- `/left_mediapipe_hand/poses` (fallback)

PoseArray 순서:

- poses[0] = wrist
- poses[1] = thumb tip
- poses[2] = index tip
- poses[3] = middle tip
- poses[4] = ring tip
- poses[5] = little tip

입력 선택:

- `/right_hand/is_tracked == true`: `/right_hand/poses` 사용
- `/right_hand/is_tracked == false`: `/right_mediapipe_hand/poses` 사용

## Hybrid OpenXR + MediaPipe bridge

`hybrid_openxr_mediapipe_pose_bridge`는 OpenXR pose, OpenXR confidence, MediaPipe pose를 받아서 중간 PoseArray를 publish합니다.

기본 오른손 입출력:

- 입력 OpenXR pose: `/right_hand/poses`
- 입력 OpenXR tracked: `/right_hand/is_tracked`
- 입력 OpenXR confidence: `/right_hand/openxr_confidence`
- 입력 MediaPipe pose: `/right_mediapipe_hand/poses`
- 입력 MediaPipe tracked: `/right_mediapipe_hand/is_tracked`
- 출력 hybrid pose: `/right_hybrid_hand/poses`
- 출력 hybrid tracked: `/right_hybrid_hand/is_tracked`
- 출력 normalized command: `/right_hybrid_hand/finger_normalized`

가중치는 fingertip pose가 아니라 calibration된 finger close amount에 사용합니다.

```text
u_openxr = normalize_distance(openxr_pose, openxr_calibration)
u_mediapipe = normalize_distance(mediapipe_pose, mediapipe_calibration)
theta = clamp(openxr_confidence, 0, 1) ** gamma
u_raw = theta * u_openxr + (1 - theta) * u_mediapipe
u_cmd = smooth_and_rate_limit(u_raw, u_previous)
```

실제 노드는 pose/confidence freshness와 source availability를 같이 봅니다. OpenXR가 stale이거나
untracked이면 MediaPipe 쪽으로 넘어갑니다. MediaPipe가 없고 OpenXR confidence가 threshold보다
낮으면 이전 command를 hold합니다. 둘 다 없을 때도 이전 command를 유지하며 tracked=false를
publish합니다. Hybrid PoseArray는 기존 viewer와 teleop retargeter 호환을 위해 rate-limited
`u_cmd`를 output calibration 거리로 다시 바꾼 debug/compatibility 표현입니다.

OpenXR와 MediaPipe의 fingertip 거리 scale이 다를 수 있으므로 bridge는 calibration remap도 지원합니다. 기본값은 다음 파일입니다.

- OpenXR/output calibration: `ros_ws/src/igris_reliability_runtime/config/hand/openxr_hand_calibration.json`
- MediaPipe calibration: `ros_ws/src/igris_reliability_runtime/config/hand/mediapipe_hand_calibration.json`

MediaPipe pose는 MediaPipe calibration으로 finger close amount를 구한 뒤 OpenXR/output calibration 거리 공간으로 변환되어 blending됩니다.

## 실행

```bash
cd /path/to/igris_teleop-main/ros_ws
colcon build --packages-select openxr_hand_to_igris_viewer
source install/setup.bash

ros2 launch openxr_hand_to_igris_viewer view_igris_hand.launch.py hand_side:=right robot_hand_side:=right
```

하이브리드 bridge만 실행:

```bash
ros2 launch openxr_hand_to_igris_viewer hybrid_hand_pose_bridge.launch.py hand_side:=right
```

MediaPipe calibration JSON 경로를 직접 지정:

```bash
ros2 launch openxr_hand_to_igris_viewer hybrid_hand_pose_bridge.launch.py \
  hand_side:=right \
  mediapipe_calibration_json:=/path/to/mediapipe_hand_calibration.json
```

오른손 하이브리드 결과를 RViz/IGRIS viewer로 확인:

```bash
ros2 launch openxr_hand_to_igris_viewer view_hybrid_igris_hand.launch.py hand_side:=right robot_hand_side:=right
```

양손 hybrid pose topic만 만들 때:

```bash
ros2 launch openxr_hand_to_igris_viewer hybrid_hand_pose_bridge.launch.py hand_side:=both
```

기본 launch는 패키지 내부의 오른손 전용 URDF를 사용합니다.
URDF mesh도 `openxr_hand_to_igris_viewer/meshes/igris_c_end_effector/hand`에서 로드합니다.

다른 URDF를 쓰려면:

```bash
ros2 launch openxr_hand_to_igris_viewer view_igris_hand.launch.py urdf_path:=/path/to/custom.urdf
```

## 확인

```bash
ros2 topic echo /right_hand/finger_normalized
ros2 topic echo /joint_states
```

## 정규화 수식

각 손가락에 대해:

```text
distance = norm(fingertip_position)
s = (max_distance - distance) / (max_distance - min_distance)
s = clip(s, 0, 1)
```

의미:

- s = 0: 손가락 열림
- s = 1: 손가락 닫힘

## IGRIS hand joint mapping

Right hand:

- thumb: Right_0_Joint_Thumb_Proximal, Right_1_Joint_Thumb_Middle, Right_2_Joint_Thumb_Distal
- index: Right_3_Joint_Index_Middle, Right_4_Joint_Index_Distal
- middle: Right_5_Joint_Middle_Middle, Right_6_Joint_Middle_Distal
- ring: Right_7_Joint_Ring_Middle, Right_8_Joint_Ring_Distal
- little: Right_9_Joint_Little_Middle, Right_10_Joint_Little_Distal

현재 패키지에 포함된 기본 URDF는 오른손만 포함합니다.
