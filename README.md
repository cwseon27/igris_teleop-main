# IGRIS Teleop

IGRIS-C 로봇의 텔레오퍼레이션, MuJoCo 시뮬레이션, 데이터 수집 및 policy 실행을 위한 멀티프로세스 프레임워크입니다. Web UI에서 입력 장치, 제어 worker, 카메라, Reliability runtime을 관리합니다.

```text
Unity / OpenXR / leader arm
        ↓
상체 IK · 팔 입력 · 손 retargeting
        ↓
실물 로봇 또는 MuJoCo → 데이터 수집 / replay / inference
```

- [설치·다른 PC로 이식](./install_info.md)
- [실행 모드](#실행-모드)
- [GUI 운영 순서](#gui-운영-순서)
- [Reliability와 hybrid 손 제어](#reliability와-hybrid-손-제어)
- [카메라·보정](#카메라보정)
- [검증·문제 확인](#검증문제-확인)

## 설치와 빠른 시작

기준 환경은 **Ubuntu 24.04 x86_64 / Python 3.12 / ROS 2 Jazzy**입니다. SDK 바이너리, ROS 환경 및 PyTorch backend 제약이 있으므로 다른 OS·아키텍처에서 그대로 동작한다고 가정하지 마세요.

처음 설치하거나 다른 PC로 옮길 때는 [install_info.md](./install_info.md)를 먼저 따르세요. 이 문서는 소스·모델·보정 파일의 보존, 새 venv 생성, 내부 ROS 패키지 빌드, 네트워크·장치 설정 및 설치 검증을 포함합니다. 기존 `ros_tcp_endpoint`가 있다면 해당 설치를 사용하고 다시 이식하거나 빌드하지 않습니다.

설치가 완료된 repository 루트에서:

```bash
# Web UI에서 mode/device를 선택
./run_igris_teleop.sh

# hybrid: VR 손 + HMD/controller 상체
./run_igris_teleop.sh \
  --mode teleop \
  --teleop-device unity_hybrid \
  --teleop-hand-source vr
```

Web UI 기본 주소는 <http://127.0.0.1:8000/>입니다. 실행 옵션은 CLI 또는 [igris_teleop_desktop_options.env](./igris_teleop_desktop_options.env)에서 지정합니다. `--mode teleop`만 지정하면 GUI에서 device를 선택해야 `Apply workers`가 활성화됩니다.

`install_igris_teleop.sh`는 기본 Python 환경과 Desktop launcher 설치용입니다. MediaPipe 전용 환경 및 내부 ROS 빌드는 별도 절차이며, 상세 옵션은 [INSTALL.md](./INSTALL.md)에 있습니다. 기본 launcher는 `.venv`, `.venv-sim`, `.venv-ik`, `.venv-ml`을 모두 확인하므로 `--skip-ml` 설치만으로 전체 실행 준비가 끝나지는 않습니다.

## 실행 모드

| `--teleop-device` | 입력 및 동작 | hand source |
| --- | --- | --- |
| `unity` | 원본 VR-only baseline. HMD 기반 상체 IK 및 VR 손 retargeting | `vr` |
| `unity_hybrid` | HMD + chest-mounted controller 상체 IK, VR 팔/손 입력 | `vr` |
| `vr_masterarm` | leader arm으로 양팔 제어, HMD/controller로 상체 제어 | `vr` 기본, `masterarm` 선택 가능 |
| `masterarm` | leader arm 입력 | `masterarm` |

`unity`에는 Reliability confidence, MediaPipe fallback, controller chest fusion이 적용되지 않습니다. 해당 기능은 `unity_hybrid` 또는 `vr_masterarm`에서 사용하세요.

```bash
# 원본 VR-only baseline
./run_igris_teleop.sh --mode teleop --teleop-device unity

# leader 팔 + VR 손 + HMD/controller 상체
./run_igris_teleop.sh \
  --mode teleop \
  --teleop-device vr_masterarm \
  --teleop-hand-source vr

# leader 팔·손
./run_igris_teleop.sh --mode teleop --teleop-device masterarm

# 학습된 policy / 데이터 replay / walking policy
./run_igris_teleop.sh --mode inference
./run_igris_teleop.sh --mode replay
./run_igris_teleop.sh --mode walking
```

Inference/replay에는 사용할 checkpoint/dataset을 별도로 지정해야 합니다. 관련 설명은 [training](./igris_teleop/training/README.md), [walking profiles](./igris_teleop/policies/walking/README.md)를 참고하세요.

## GUI 운영 순서

새 PC나 설정 변경 후에는 실물 로봇 연결을 해제한 상태에서 시뮬레이션을 먼저 확인하세요. Web UI 실행과 실제 모터 제어 시작은 별도 단계입니다.

1. `mode`, `teleop device`, `hand source`를 선택합니다.
2. 시뮬레이션이면 **`simulator`를 먼저 시작**하고 정상 실행 상태를 확인합니다. 이후 `control`을 시작합니다. GUI의 `control` 그룹에는 몸체와 손 worker가 포함됩니다.
3. 입력에 맞춰 `leader_ros`와 `Apply workers`를 실행하고 VR/leader 입력이 갱신되는지 확인합니다. `leader_ros` 그룹은 device에 따라 기존 ROS TCP endpoint, leader serial node 또는 둘 다 실행합니다. 같은 node를 별도 터미널에서 중복 실행하지 마세요.
4. Hybrid Reliability를 사용할 경우 카메라 설정을 확인하고 `Start reliability`를 실행합니다. 영상이 필요하면 camera mode를 선택하고 Camera `Start`도 실행합니다.
5. `ready`로 준비 자세 이동을 완료한 뒤, 손 사용 시 `hand initial` 성공을 확인하고 `start`를 실행합니다. Entry interpolation이 완료되어야 정상 teleoperation으로 넘어갑니다.
6. `home`은 기본 자세 복귀에 사용합니다. 종료는 `Shutdown (set level)`로 요청하고, 실물에서는 자세 복귀와 `Torque OFF` 결과까지 확인합니다.

실물 제어 시에는 simulator를 실행하지 않고, 대상 namespace·LAN 경로·신선한 상태 수신을 먼저 확인해야 합니다. **시뮬레이터 ON/OFF를 실물 로봇에 대한 물리적 안전 차단 장치로 간주하지 마세요.** E-stop, 호이스트, 주변 공간을 확보하고 한 번에 하나의 제어 클라이언트만 사용하세요. 손 초기화도 실제 움직임을 일으킬 수 있습니다.

공유메모리를 사용하므로 프레임워크는 PC당 한 인스턴스만 실행합니다. 중복 실행 오류가 나면 기존 GUI를 사용하거나 정상 종료 후 재실행하세요. 업데이트 후, 특히 공유메모리 schema가 바뀐 경우에는 **프레임워크와 Reliability를 모두 종료한 뒤 재실행**해야 합니다. 일부 worker만 재시작하면 안 됩니다.

## Reliability와 hybrid 손 제어

대상은 `unity_hybrid` 또는 `vr_masterarm`, hand source=`vr`입니다. GUI의 `Start reliability`는 repository 내부 `ros_ws`의 통합 launch를 실행합니다. 기본적으로 hand/controller confidence 추론, 두 카메라 MediaPipe, 손 retargeting/fusion이 함께 시작됩니다.

```text
VR wrist/tips ── 기존 VR 좌표 변환 ───────┐
                                        ├─ 같은 HandRetargeting + 손 URDF
MediaPipe 21 landmarks ── 손바닥 축 정렬 ─┘          ↓
                                       소스별 6개 모터 명령 → confidence 혼합
```

두 소스는 같은 DexRetargeting 코드와 설정을 사용하되 optimizer/filter 상태는 독립적입니다. 엄지 굽힘·벌림을 독립적으로 유지하고, 기존 손목–손끝 거리 calibration 기반 5개 닫힘 값은 기본 Reliability 경로에서 사용하지 않습니다. MediaPipe의 wrist 위치는 팔 IK 입력을 덮어쓰지 않습니다.

- Hand/controller 모델은 RNN, HistGB, `always 1`을 각각 선택할 수 있습니다. 변경 후 Reliability를 다시 시작해야 적용됩니다.
- Hand `always 1`은 **VR-only**입니다. MediaPipe로 전환하지 않으며, VR 추적 중단 시 마지막 유효 명령을 유지합니다. 유효한 VR pose 요구 조건은 그대로입니다.
- 활성화된 hybrid 손은 추적·통신이 끊겨도 마지막 6개 명령을 유지합니다. 이는 torque off가 아닙니다. Reliability를 끄고 raw VR로 돌아가려면 전체 프레임워크를 재시작해야 합니다.
- 출력은 `/left_hybrid_hand/motor_normalized`, `/right_hybrid_hand/motor_normalized`와 각 `motor_tracked`입니다. 모터 명령은 gain 적용이 끝난 6개 값이며 기존 로봇 SDK hand topic/service는 별개로 유지됩니다.
- 사람별 손 크기, 카메라 영상 전처리 및 기구적 한계까지 자동 보정되지는 않습니다. 오프라인 회귀 테스트 통과가 실물 추종 품질이나 안전을 보장하지 않습니다.

터미널 직접 실행, topic 명세 및 tuning은 [Reliability runtime README](./ros_ws/src/igris_reliability_runtime/README.md)를 참고하세요. GUI와 수동 launch를 동시에 실행하지 마세요. 카메라 없이 확인하거나 MediaPipe를 별도 실행하는 경우에는 `start_mediapipe:=false`를 사용합니다.

### 상체 controller 보정

Hybrid controller 입력은 `/left_controller/poses`, `/right_controller/poses`이며 singular `/.../pose`도 호환됩니다. 장착 위치에 맞는 controller-to-chest 변환을 teleop 시작 전에 적용해야 합니다. 순서는 `x,y,z,qx,qy,qz,qw`입니다.

```bash
# 장착 위치에 맞게 교체할 시작값
export IGRIS_LEFT_CONTROLLER_TO_CHEST="0.064,0,0,0,0,0,1"
export IGRIS_RIGHT_CONTROLLER_TO_CHEST="-0.064,0,0,0,0,0,1"

# controller pose가 발행 중인 상태에서 보정값 측정
source /opt/ros/jazzy/setup.bash
source ros_ws/install/setup.bash
.venv/bin/python controller_calibration.py
```

보정 도구는 준비 5초 후 10초간 샘플링하고 `export` 값을 출력합니다. 결과는 `local_state/controller_to_chest_calibration.env`와 `.json`에도 저장됩니다. 다른 PC로 옮길 때 보존해야 하지만, 장착 위치가 달라지면 다시 측정하세요.

<details>
<summary>상체 confidence 전환과 고급 IK 설정</summary>

Hybrid chest는 기존 `T_ee`를 덮어쓰지 않는 별도 `C_ee` task이며 Visualizer의 `/targets/chest`로 표시됩니다. Controller pose가 유효하면 confidence 0에서도 진단용 target은 보일 수 있지만 chest task weight는 0입니다.

기본 `0.2/0.8` threshold와 smoothstep, rise/fall rate limit으로 waist activation을 조절합니다. Invalid pose/source 또는 confidence 0에서는 허리를 마지막 publish 명령에 고정하고 HMD orientation 추종은 유지합니다. 회복 시 waist QP delta 범위를 점진적으로 넓힙니다. Hybrid HMD position/orientation multiplier는 `1.0/7.0`, chest는 각각 `2.0*activation`입니다. `unity` baseline의 HMD orientation multiplier는 별도 `6.0`입니다.

`IGRIS_IK_CONFIG` YAML의 `waist_transition` 또는 아래 환경변수로 설정합니다. Low-confidence HMD translation scaling은 기본적으로 꺼져 있습니다.

```bash
export IGRIS_IK_WAIST_TRANSITION_LOW_THRESHOLD=0.2
export IGRIS_IK_WAIST_TRANSITION_HIGH_THRESHOLD=0.8
export IGRIS_IK_WAIST_TRANSITION_ACTIVATION_RISE_RATE=2.0
export IGRIS_IK_WAIST_TRANSITION_ACTIVATION_FALL_RATE=4.0
export IGRIS_IK_WAIST_TRANSITION_SCALE_HEAD_TRANSLATION_WITH_ACTIVATION=false
export IGRIS_IK_WAIST_TRANSITION_LOW_CONF_HEAD_TRANSLATION_WEIGHT=0.0
```

Profile별 접두사는 `IGRIS_IK_UNITY_HYBRID_WAIST_TRANSITION_...`, `IGRIS_IK_VR_MASTERARM_WAIST_TRANSITION_...`입니다. IK worker의 `waist transition`, `target residual` 로그를 함께 확인하세요.

</details>

## 카메라·보정

### MediaPipe hand 카메라

`Hybrid Teleop Info`에서 좌우 카메라와 전처리를 설정합니다. `/dev/videoN`은 재연결 시 바뀔 수 있으므로 `/dev/v4l/by-path/...-video-index0` 고정 경로를 사용하세요. 동일 serial의 카메라 두 대는 `by-id`가 충돌할 수 있습니다. USB 포트/허브 또는 PC가 바뀌면 경로와 좌우 대응을 다시 선택해야 합니다.

기본 전처리는 카메라별 손 1개, MediaPipe full model(`model_complexity=1`), `mirror=false`, `swap_lr=true`, trapezoid 활성화(bottom width `320 px`)입니다. `Start MediaPipe test`에서 RAW·전처리·landmark를 확인하세요. Reliability 시작 시 테스트는 종료되어 카메라 중복 점유를 피합니다.

설정은 `igris_artifacts/config/hybrid_teleop.json`에 저장됩니다. 프레임 읽기 실패 시 tracked=false를 발행하고 같은 장치 경로로 재연결을 시도합니다. 반복 실패 시 케이블·허브·USB 전원도 확인하세요.

### 로봇 stereo 영상

실물 eye 입력은 정방향 SBS 영상이며, PC bridge가 좌우 분할 후 각 calibration map을 적용해 `/left/image_rect/compressed`, `/right/image_rect/compressed`로 전달합니다. Eyes 영상에 별도 180° 회전을 적용하지 않습니다. 현재 기본 출력은 눈당 `640×480`(4:3)입니다. 입력 크기와 map 규격이 일치하는지도 확인하세요.

현재 기본 map은 `ros_ws/src/stereo_sbs_cam_pub/config/stereo_rectify_maps_tuned.npz`입니다. 카메라 교체 시 해당 장치의 보정값을 사용해야 합니다. 로봇 영상과 로컬 MediaPipe 카메라는 서로 다른 입력 경로입니다.

MuJoCo에서는 Camera 시작 시 head stereo가 GUI와 동일 좌우 ROS topic에 연결됩니다. `sim eye distance`는 시뮬레이션 카메라 간격만 조절하며 실물 calibration을 대체하지 않습니다.

## 검증·문제 확인

설치 검증의 전체 명령은 [install_info.md](./install_info.md)에 있습니다. 네 가지 RNN/HistGB policy 로드, MediaPipe/OpenCV/rclpy import, 내부 ROS 패키지, GUI launch 경로와 환경별 테스트를 확인하세요.

```bash
# 기본 회귀 테스트: 모터 명령을 보내는 실물 실행 명령이 아님
.venv/bin/python -m pytest -q tests

# 모델 파일 무결성
(cd policy_archive && sha256sum -c SHA256SUMS)
```

선택 의존성이 없는 환경에서는 일부 테스트가 skip됩니다. Retargeting/IK/ML/MediaPipe 테스트는 해당 전용 환경에서도 실행해야 합니다. [tests 안내](./tests/README.md)를 참고하세요.

로그는 `igris_artifacts/logs/`와 ROS의 `~/.ros/log/`에서 확인합니다. Topic 이름이 보이는 것만으로 수신 성공을 판단하지 말고 실제 timestamp·메시지 갱신·tracking과 프로세스 오류를 확인하세요. 특히 hand init 성공은 연속 HandCmd 전달 성공과 다릅니다. DDS interface/locator, namespace, domain은 [설치 가이드](./install_info.md)의 네트워크 점검을 따르세요.

## 프로젝트 구조

| 경로 | 역할 |
| --- | --- |
| [igris_teleop/](./igris_teleop/README.md) | 런타임, worker, Web UI, 공유메모리 |
| [robot_control/](./igris_teleop/robot_control/README.md) | 몸체 제어, IK/FK, gain/pose와 robot asset |
| [hand_control/](./igris_teleop/hand_control/README.md) | 손 retargeting, URDF, 모터 명령 변환 |
| [ros_ws/](./ros_ws/README.md) | 내부 ROS 패키지와 bridge/leader/카메라 코드 |
| [policy_archive/](./policy_archive/README.md) | Reliability 모델과 checksum |
| [requirements/](./requirements/README.md) | 환경별 입력 의존성과 lock 파일 |
| [third_party/](./third_party/README.md), [vendor_patches/](./vendor_patches/README.md) | 외부 코드·SDK 및 적용 patch |
| [tests/](./tests/README.md) | 회귀 테스트 |
| [igris_artifacts/](./igris_artifacts/README.md) | 사용자 설정, 모델, dataset, 실행 로그 |
| [local_state/](./local_state/README.md) | SDK snapshot, PC별 DDS/장치/보정 및 generated state |

Python 환경은 `.venv`(기본), `.venv-sim`(MuJoCo), `.venv-ik`(IK), `.venv-ml`(학습·추론·손 retargeting), `.venv-mediapipe`(MediaPipe)로 분리됩니다. Worker별 선택은 [registry.py](./igris_teleop/workers/registry.py)와 ROS wrapper에서 관리합니다. 새 PC에서는 이 환경과 ROS `build/install/log`를 복사하지 말고 새로 만드세요.

Git에 포함된 소스만 clone하면 무시된 로컬 설정·보정 결과·사용자 dataset까지 자동 이식되는 것은 아닙니다. 손실 없는 이식을 위해 [install_info.md](./install_info.md)의 별도 보존 항목도 함께 이전하고, 계정 정보·인증키·실행 로그를 repository에 올리지 마세요.
