# IGRIS 12-DOF FastSAC deploy handoff

## 목적

이 문서는 `exp:igris-12dof-fast-sac`로 학습한 IGRIS 하체 12자유도 보행 정책을 MuJoCo sim-to-sim 또는 Booster lowcmd 기반 실기체 경로로 deploy 할 때 필요한 핵심 계약값을 한 문서로 모아둔 정리다.

예시로 받은 `igris_sim2real_handoff.md`와 달리, 이 프로젝트의 현재 코드는 다음 특징을 가진다.

- observation history를 쌓지 않고 single-frame `49`차원 observation을 사용한다.
- deploy loop는 `50 Hz`다.
- 자세 입력은 Euler angle이 아니라 `projected_gravity`와 base quaternion 기반으로 맞춘다.
- FastSAC deploy는 `.pt`가 아니라 `.onnx`를 직접 사용한다.

## 기준 코드

아래 파일들을 source of truth로 봤다.

- `src/holosoma/holosoma/config_values/loco/igris/experiment.py`
- `src/holosoma/holosoma/config_values/loco/igris/observation.py`
- `src/holosoma/holosoma/config_values/loco/igris/command.py`
- `src/holosoma/holosoma/config_values/loco/igris/randomization.py`
- `src/holosoma/holosoma/config_values/robot.py`
- `src/holosoma_inference/holosoma_inference/config/config_values/inference.py`
- `src/holosoma_inference/holosoma_inference/config/config_values/observation.py`
- `src/holosoma_inference/holosoma_inference/config/config_values/robot.py`
- `src/holosoma_inference/holosoma_inference/config/config_values/task.py`
- `src/holosoma_inference/holosoma_inference/policies/base.py`
- `src/holosoma_inference/holosoma_inference/policies/locomotion.py`

추가 확인:

- 로컬 ONNX `logs/hv-igris-manager/20260404_084423-igris_12dof_fast_sac_manager-locomotion/model_0018000.onnx`
  - 입력 shape `(1, 49)`
  - 출력 shape `(1, 12)`
  - metadata에 `dof_names`, `kp`, `kd`, `action_scale`, `command_ranges`, `experiment_config`, `robot_urdf` 포함

## 빠른 실행 절차

### 1. MuJoCo 실행

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:igris-12dof
```

### 2. Policy 실행

중요:

- 현재 repo에는 `source_interface_setup.sh`가 없다.
- deploy 환경 활성화는 `source scripts/source_inference_setup.sh`를 써야 한다.

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:igris-12dof-loco \
    --task.model-path <path-to-igris-fast-sac.onnx> \
    --task.no-use-joystick \
    --task.interface lo
```

FastSAC ONNX는 일반적으로 아래처럼 run 디렉토리 바로 아래에 생긴다.

```text
logs/hv-igris-manager/<run_dir>/model_XXXXXXX.onnx
```

예시:

- `logs/hv-igris-manager/20260404_084423-igris_12dof_fast_sac_manager-locomotion/model_0018000.onnx`
- `logs/hv-igris-manager/20260405_103532-igris_12dof_fast_sac_manager-locomotion/model_0028000.onnx`

### 3. MuJoCo / policy 조작

- MuJoCo 창에서 `8`: 호이스트를 내려서 발을 땅에 닿게 함
- MuJoCo 창에서 `9`: 호이스트 해제
- policy 터미널에서 `]`: policy start
- policy 터미널에서 `=`: walk mode 진입
- policy 터미널에서 `o`: policy stop
- policy 터미널에서 `i`: default pose로 init
- policy 터미널에서 `w a s d`: 선속도 제어
- policy 터미널에서 `q e`: yaw 속도 제어
- policy 터미널에서 `z`: velocity zero

## 반드시 전달해야 하는 핵심 계약

- Training preset: `exp:igris-12dof-fast-sac`
- Inference preset: `inference:igris-12dof-loco`
- Policy observation: `1 x 49 = 49`
- Policy action: `12`
- Policy control period: `0.02 s`
- Policy control rate: `50 Hz`
- Training sim timestep: `0.005 s`
- Training simulator frequency: `200 Hz`
- Training control decimation: `4`
- Gait period: `1.0 s`
- Desired base height: `0.95 m`
- Command range:
  - `vx`: `[-0.8, 0.8]`
  - `vy`: `[-0.4, 0.4]`
  - `yaw`: `[-0.8, 0.8]`
- Training command sampler에는 `heading: [-3.14, 3.14]`도 있지만, 현재 deploy actor observation은 heading을 직접 입력으로 쓰지 않는다.

- Deploy-side action decoding:

```python
target_q = action * 0.25 + default_q
```

- Deploy runtime command sent by the current Booster path:

```python
cmd_q = target_q
cmd_dq = 0.0
cmd_tau = 0.0
cmd_kp = kp
cmd_kd = kd
```

- 만약 Booster low-level position controller 대신 별도 임베디드 제어기가 `q_target`를 받아 PD torque로 바꿔야 한다면, 동등한 형태는 아래다.

```python
tau = (target_q - q) * kp + (0.0 - dq) * kd
tau = clip(tau, -tau_limit, tau_limit)
```

- Actor observation은 single-frame이라 외부에서 긴 history buffer를 유지할 필요는 없다.
- 대신 deploy 런타임은 반드시 아래 내부 상태를 유지해야 한다.
  - `previous_action`
  - `phase(left, right)`
  - `stand/walk state`
  - `current commanded vx, vy, yaw`

중요:

- 이 locomotion preset의 deploy-side action decode는 끝까지 scalar `0.25`를 쓴다.
- FastSAC ONNX output을 deploy 쪽에서 다시 `tanh`나 per-joint boundary로 재해석하면 안 된다.
- inference robot config 안의 `default_per_joint_action_scale`는 현재 locomotion `BasePolicy` 경로에서는 사용되지 않는다.
- 따라서 실기체 middleware에서 per-joint effort/p-gain 비율을 추가로 곱하면 중복 스케일링이 된다.

## 실기체 / 미들웨어에서 반드시 들어와야 하는 입력

- 다리 12개 joint position `q`
- 다리 12개 joint velocity `dq`
- base orientation quaternion
  - 현재 Booster state processor는 IMU `rpy`를 quaternion으로 바꿔서 넣는다.
- base angular velocity gyro
- command `vx`, `vy`, `yaw`
- 직전 policy action
- gait phase `(left, right)`
- stand / walk mode 상태

주의:

- 현재 IGRIS locomotion actor는 `base_pos`와 `base_lin_vel`를 observation으로 사용하지 않는다.
- 현재 Booster IGRIS 경로에서는 `base_pos`와 `base_lin_vel`가 zero-filled여도 동작하도록 짜여 있다.
- `projected_gravity`는 별도 Euler 입력이 아니라 base quaternion으로부터 계산된다.
- 실기체가 풀바디라 하더라도 현재 preset은 하체 12자유도만 제어한다.
- 허리/팔/목 유지 제어는 이 policy 밖의 상위 제어기에서 따로 잡아줘야 한다.

## 현재 Booster low-state packet layout

현재 `BoosterStateProcessor`가 만드는 low-state packet은 아래 형태다.

| Slice | Meaning | Size | IGRIS locomotion policy 사용 여부 |
| --- | --- | ---: | --- |
| `0:3` | `base_pos` | 3 | 사용 안 함 |
| `3:7` | `base_quat` | 4 | 사용 |
| `7:19` | `joint_pos` | 12 | 사용 |
| `19:22` | `base_lin_vel` | 3 | 사용 안 함 |
| `22:25` | `base_ang_vel` | 3 | 사용 |
| `25:37` | `joint_vel` | 12 | 사용 |
| `37:55` | `tau_est` | 18 | 사용 안 함 |
| `55:73` | `ddq` | 18 | 사용 안 함 |

즉, 현재 locomotion policy는 실질적으로 앞의 `37`개 상태만 읽는다.

참고:

- 코드상 특정 interface가 `37 + 3 = 40`차원 상태를 주면, 뒤 `3`개를 precomputed `projected_gravity`로 사용한다.
- 현재 IGRIS Booster 구현은 그 `3`개를 붙이지 않으므로 quaternion에서 직접 `projected_gravity`를 계산한다.

## Observation layout

single observation `49`차원 구성은 아래와 같다.

중요:

- 현재 training manager와 inference runtime 모두 observation term을 이름 기준 alphabetical order로 concatenate한다.
- 따라서 아래 순서가 실제 ONNX 입력 순서다.

| Index | Meaning | Size |
| --- | --- | ---: |
| `0:12` | previous action | 12 |
| `12:15` | base angular velocity | 3 |
| `15:16` | command yaw velocity | 1 |
| `16:18` | command `vx`, `vy` | 2 |
| `18:20` | gait phase `cos(left)`, `cos(right)` | 2 |
| `20:32` | `(q - default_q)` | 12 |
| `32:44` | `dq` | 12 |
| `44:47` | projected gravity | 3 |
| `47:49` | gait phase `sin(left)`, `sin(right)` | 2 |

실제 관측 생성식은 아래 흐름을 맞추면 된다.

```python
obs[0:12] = previous_action
obs[12:15] = omega * 0.5
obs[15:16] = cmd_yaw
obs[16:18] = [cmd_vx, cmd_vy]
obs[18:20] = [cos(phase_left), cos(phase_right)]
obs[20:32] = q - default_q
obs[32:44] = dq * 0.08
obs[44:47] = projected_gravity
obs[47:49] = [sin(phase_left), sin(phase_right)]
```

phase 관련 주의:

- 초기 phase는 `left = 0`, `right = pi`다.
- 매 tick마다 `2 * pi / (50 * 1.0)`만큼 phase를 전진시킨다.
- stand 상태에서 velocity command가 거의 `0`이면 양발 phase를 둘 다 `pi`로 맞춘다.

## Joint order and control contract

아래 순서를 deploy와 정확히 맞춰야 한다.

| Policy joint | Default q [rad] | Kp | Kd | Torque limit | Vel limit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Joint_Hip_Pitch_Left` | -0.05 | 200.0 | 5.0 | 150.0 | 100.0 |
| `Joint_Hip_Roll_Left` | 0.0 | 250.0 | 5.0 | 120.0 | 100.0 |
| `Joint_Hip_Yaw_Left` | 0.0 | 150.0 | 2.0 | 60.0 | 100.0 |
| `Joint_Knee_Pitch_Left` | 0.36 | 300.0 | 5.0 | 150.0 | 100.0 |
| `Joint_Ankle_Pitch_Left` | -0.25 | 50.0 | 2.5 | 90.0 | 100.0 |
| `Joint_Ankle_Roll_Left` | 0.0 | 50.0 | 2.5 | 90.0 | 100.0 |
| `Joint_Hip_Pitch_Right` | -0.05 | 200.0 | 5.0 | 150.0 | 100.0 |
| `Joint_Hip_Roll_Right` | 0.0 | 250.0 | 5.0 | 120.0 | 100.0 |
| `Joint_Hip_Yaw_Right` | 0.0 | 150.0 | 2.0 | 60.0 | 100.0 |
| `Joint_Knee_Pitch_Right` | 0.36 | 300.0 | 5.0 | 150.0 | 100.0 |
| `Joint_Ankle_Pitch_Right` | -0.25 | 50.0 | 2.5 | 90.0 | 100.0 |
| `Joint_Ankle_Roll_Right` | 0.0 | 50.0 | 2.5 | 90.0 | 100.0 |

기계적 joint position limit은 robot config에 이미 정의되어 있고, FastSAC actor는 그 limit를 기준으로 학습됐다.

하지만 deploy 코드 자체는 `q_target`를 별도로 joint limit에 clamp하지 않는다.

따라서:

- 현재 Holosoma 그대로 쓰면 ONNX output과 `0.25` decode를 그대로 따라간다.
- 별도 middleware / 임베디드로 재구현할 때는 safety layer에서 joint position saturation을 추가하는 편이 안전하다.

## 학습 시 robustness 관련 값

실기체 팀과 공유할 만한 값들:

- IMU-style actor observation noise:
  - `base_ang_vel` gaussian std: `0.12`
  - `projected_gravity` gaussian std: `(0.036, 0.036, 0.036)`
- friction randomization: `[0.6, 1.1]`
- link mass randomization: `[0.98, 1.05]`
- PD gain randomization:
  - `kp`: `[0.98, 1.02]`
  - `kd`: `[0.98, 1.02]`
- gait period randomization width: `0.15`
- command resampling time: `10.0 s`
- stand probability during training commands: `0.25`
- reset joint position scale range: `[0.9, 1.1]`

현재 preset에서 꺼져 있는 항목:

- push randomization
- action delay randomization
- torque RFI
- dof position bias randomization
- base mass randomization
- base COM randomization

## 알려진 차이점 / 주의사항

- 예시로 받은 구버전 `igris_walk_ppo` 문서와 달리, 이 정책은 `15 x 47` stacked observation을 쓰지 않는다.
- base Euler angle을 직접 넣지 않고 `projected_gravity`를 쓴다.
- deploy rate는 `100 Hz`가 아니라 `50 Hz`다.
- IGRIS FastSAC ONNX는 `logs/.../model_XXXXXXX.onnx` 형태로 checkpoint와 같은 폴더에 저장된다.

추론 기반 주의:

- 현재 repo에는 IGRIS 전용 real-robot 네트워크 설정 문서가 커밋되어 있지 않다.
- 코드상으로는 `booster` DDS low-state / low-cmd transport를 쓰도록 연결되어 있지만, 실제 NIC 이름, 고정 IP, gateway 같은 값은 하드웨어팀과 별도 합의가 필요하다.

실무적으로는 아래 세 줄만 틀리지 않게 맞추면 deploy 문제의 대부분을 피할 수 있다.

- ONNX 입력은 `49`차원 single-frame actor observation
- action decode는 `target_q = action * 0.25 + default_q`
- joint order는 문서의 12개 순서를 그대로 사용
