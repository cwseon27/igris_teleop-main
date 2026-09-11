# robot_control/scripts

`robot_control/scripts` 아래의 실행 스크립트와 보조 모듈 정리.

## 실행 위치

아래 둘 중 하나로 실행하면 된다.

리포지토리 루트에서 실행:

```bash
python igris_teleop/robot_control/scripts/<script>.py ...
```

`igris_teleop/` 디렉터리 안에서 실행:

```bash
python robot_control/scripts/<script>.py ...
```

## 공통 전제

- `igris_c_sdk` 런타임이 import 가능해야 한다.
- DDS/로봇 통신이 정상이어야 한다.
- 스크립트는 내부에서 `IgrisController`를 생성하고 low-level 제어를 수행한다.
- 기본 출력물은 `igris_teleop/outputs/` 또는 `igris_teleop/robot_control/outputs/` 아래에 저장된다.
- 실제 로봇에서 실행하므로 진폭, 주파수, duration은 보수적으로 시작하는 것이 좋다.

## 좌표계 기준

- 기본 기준은 `joint/PJS`이다.
- `waist` 그룹은 IMU 자세 `rpy`가 아니라 `waist yaw / roll / pitch joint`를 뜻한다.
- `record_actuatornet.py`의 입력 command는 기본적으로 `joint/PJS` 기준이다.
- `record_actuatornet.py`의 출력 log에는 `joint`와 `motor` 관측이 둘 다 들어간다.
- `tune_gains.py`의 `kp/kd` sweep은 `joint/PJS` 기준 gain이다.
- `tune_gains.py`의 tracking metric도 `cmd_q_target_pjs` 대 `obs_q_joint`를 비교하므로 `joint/PJS` 기준이다.
- 예외적으로 `calibrate_pr2ab.py`는 `PR(joint)`와 `AB(motor)`의 관계를 직접 다룬다.

## 파일별 설명

### `calibrate_pr2ab.py`

2자유도 병렬 메커니즘의 `PR(joint)` <-> `AB(motor)` 관계를 보정하거나 검증하는 스크립트.

기준:

- 입력/명령 생성은 `PR(joint)` 기준
- 검증 시에는 `AB(motor)` 응답도 함께 사용
- 최종 결과는 `PR -> AB` 변환행렬

대상 pair:

- `waist_rp`
- `l_ankle_pr`
- `r_ankle_pr`
- `l_wrist_rp`
- `r_wrist_rp`

주요 기능:

- `--task calib`: 변환행렬 추정
- `--task verify`: 저장된 변환행렬 검증
- `--pair all`: 모든 pair 순차 실행
- 보정 결과를 YAML로 저장
- 검증 결과와 plot 저장

대표 실행 예시:

```bash
python igris_teleop/robot_control/scripts/calibrate_pr2ab.py --pair all
python igris_teleop/robot_control/scripts/calibrate_pr2ab.py --pair waist_rp --task calib
python igris_teleop/robot_control/scripts/calibrate_pr2ab.py --pair waist_rp --task verify --mode ms
```

주요 출력:

- `robot_control/outputs/pr2ab_calibration.yaml`
- `robot_control/outputs/pr2ab_verify.yaml`
- `robot_control/outputs/*_transform_result.png`

### `record_actuatornet.py`

지정한 관절 그룹에 `hold / sine / chirp` 입력을 넣고 command/observation 로그를 저장하는 데이터 수집 스크립트.

기준:

- 입력 command: `joint/PJS`
- 저장되는 observation: `joint + motor`
- `waist`는 `yaw / roll / pitch joint` 기준

주요 기능:

- `--joint-group`: `waist`, `leg`, `arm`, `neck`, `all`
- `--profile`: `hold`, `sine`, `chirp`
- `--record-hz`: 로깅 주파수
- `--move-to-default`: 시작 전에 `default_pos`로 이동
- `.npz + .json` 형태로 에피소드 저장

대표 실행 예시:

```bash
python igris_teleop/robot_control/scripts/record_actuatornet.py \
  --joint-group waist \
  --profile chirp \
  --duration 12 \
  --record-hz 100 \
  --move-to-default
```

추가 예시:

```bash
python igris_teleop/robot_control/scripts/record_actuatornet.py \
  --joint-group arm \
  --profile sine \
  --duration 10 \
  --amplitude 0.05 \
  --frequency 0.3 \
  --move-to-default
```

출력 위치:

- 기본: `igris_teleop/outputs/actuatornet_datasets/episode_<profile>_<group>_<timestamp>.npz`
- 메타데이터: 같은 이름의 `.json`

주의:

- 현재 `joint-group waist`는 허리 3축 전체에 같은 파형을 넣는다.
- 축별 식별이 목적이면 그룹을 더 세분화하거나 스크립트를 축 단위로 확장하는 것이 더 적절하다.

### `tune_gains.py`

선택한 관절 그룹에 대해 `kp/kd` 배율 조합을 sweep 하면서 tracking metric을 비교하는 스크립트.

기준:

- gain sweep 대상: `joint/PJS` gain
- 입력 profile: `joint/PJS`
- 평가 metric: `joint/PJS` tracking 기준
- sine/chirp 중심점은 현재 실제 joint가 아니라 현재 command target(`q_target_pjs`) 기준
- observation 로그에는 `joint + motor`가 모두 포함됨
- 기본 비용함수는 `tracking`이며 `q/dq` 추종 오차를 우선한다

주요 기능:

- `--search-mode grid`: 지정한 `kp/kd scale` 조합 전체 sweep
- `--search-mode adaptive`: 기본 gain(`1.0, 1.0`)에서 시작해 `kp/kd`를 올릴지 내릴지 로그 기반으로 판단하며 국소 탐색
- `--cost-mode tracking`: 기본값, `q/dq` tracking 오차 우선
- `--cost-mode balanced`: tracking과 torque를 절충
- `--cost-mode effort`: 예전 방식과 비슷하게 torque penalty 비중이 큼
- `--motion-preset`: 여러 동작 조건을 한 번에 묶어서 평가하는 preset
- `--motion-spec`: 여러 동작 조건을 직접 반복 지정
- `--motion-aggregate`: 여러 동작 cost를 `mean`, `max`, `meanmax`로 집계
- `--profile`: `sine` 또는 `chirp`
- `--kp-scale`, `--kd-scale`: 쉼표 구분 배율 리스트
- 기본적으로 시작 전 `s` 입력을 기다리고, `--auto-start`로 생략 가능
- 실행 중 `Ctrl+C`를 누르면 현재 run까지 저장하고 빠르게 abort
- 각 조합마다 rollout 수행
- `mae_q`, `rmse_q`, `mae_dq`, `mae_tau`, `cost` 계산
- 모든 trial metric은 `summary.json`에 남기고, 실제 `.npz/.json/.png`는 subset별 최종 best만 저장
- 최종 best tracking plot `.png`는 기본적으로 함께 저장하며 `--no-plot`으로 비활성화 가능
- `--move-to-default` 사용 시 각 run 전에 `default_pos`로 복귀
- `--reset-duration`, `--settle-sec`로 run 간 초기화 시간 조절
- `--joint-mode`로 `together`, `single`, `pairs`, `single+pairs`, `single+pairs+all` 선택 가능
- `--joint-mode auto`는 `waist`일 때 기본적으로 `single+pairs`를 사용
- 세부 group으로 `l_arm`, `r_arm`, `l_wrist`, `r_wrist`, `l_shoulder`, `r_shoulder`, `l_elbow`, `r_elbow`도 지원
- `--apply-best`를 주면 튜닝 종료 직후 `joint_setting.yaml`에 단축 최적 gain을 바로 반영 가능
- `--apply-joint-profile`로 직접 덮어쓸 joint profile 경로를 지정 가능

대표 실행 예시:

```bash
python igris_teleop/robot_control/scripts/tune_gains.py \
  --search-mode grid \
  --cost-mode tracking \
  --joint-group waist \
  --profile sine \
  --duration 8 \
  --kp-scale 0.5,0.8,1.0,1.2,1.5 \
  --kd-scale 0.5,0.8,1.0,1.2 \
  --move-to-default
```

적응형 탐색 예시:

```bash
python igris_teleop/robot_control/scripts/tune_gains.py \
  --search-mode adaptive \
  --cost-mode tracking \
  --joint-group waist \
  --profile sine \
  --duration 6 \
  --amplitude 0.03 \
  --move-to-default \
  --adaptive-iters 5 \
  --adaptive-kp-delta 0.2 \
  --adaptive-kd-delta 0.2 \
  --kp-min-scale 0.5 \
  --kp-max-scale 1.8 \
  --kd-min-scale 0.5 \
  --kd-max-scale 1.8 \
  --auto-start
```

오른팔 손목만 따로 튜닝하려면:

```bash
python igris_teleop/robot_control/scripts/tune_gains.py \
  --search-mode adaptive \
  --cost-mode tracking \
  --joint-group r_wrist \
  --joint-mode single \
  --profile sine \
  --duration 5 \
  --amplitude 0.01 \
  --frequency 0.2 \
  --move-to-default \
  --auto-start
```

waist 3축을 예전처럼 한 번에 모두 묶어 튜닝하려면:

```bash
python igris_teleop/robot_control/scripts/tune_gains.py \
  --search-mode adaptive \
  --cost-mode tracking \
  --joint-group waist \
  --joint-mode together \
  --profile sine \
  --duration 8 \
  --move-to-default \
  --auto-start
```

튜닝이 끝나면 최적값을 바로 `joint_setting.yaml`에 반영하려면:

```bash
python igris_teleop/robot_control/scripts/tune_gains.py \
  --search-mode adaptive \
  --cost-mode tracking \
  --joint-group waist \
  --joint-mode single \
  --profile sine \
  --duration 5 \
  --amplitude 0.02 \
  --frequency 0.2 \
  --move-to-default \
  --apply-best
```

`l_elbow`처럼 한 가지 동작에 과적합되지 않게 여러 동작을 묶어 튜닝하려면:

```bash
python igris_teleop/robot_control/scripts/tune_gains.py \
  --search-mode adaptive \
  --cost-mode tracking \
  --joint-group l_elbow \
  --joint-mode single \
  --move-to-default \
  --motion-preset elbow \
  --motion-aggregate meanmax \
  --apply-best
```

직접 여러 동작을 지정하려면:

```bash
python igris_teleop/robot_control/scripts/tune_gains.py \
  --search-mode adaptive \
  --cost-mode tracking \
  --joint-group l_elbow \
  --joint-mode single \
  --move-to-default \
  --motion-spec sine:0.02:0.2:5 \
  --motion-spec sine:0.05:0.2:5 \
  --motion-spec sine:0.05:0.5:5 \
  --motion-spec chirp:0.03:0.1:0.8:5 \
  --motion-aggregate meanmax \
  --apply-best
```

출력 위치:

- 기본: `igris_teleop/outputs/tuning_runs/<timestamp>/`
- subset별 최종 best log: `best_<subset>.npz`
- subset별 최종 best plot: `best_<subset>.png`
- 각 best metadata: `best_<subset>.json`
- 요약: `summary.json`
- multi-motion일 때는 best 후보의 각 motion log가 `best_<subset>__<motion>.npz` 형태로 저장됨

주의:

- 현재 기본 `joint-mode auto`에서는 `joint-group waist`를 `yaw`, `roll`, `pitch` 단축과 2축 조합으로 나눠서 튜닝한다.
- 허리 3축 전체를 동시에 흔들고 싶으면 `--joint-mode together`를 명시하면 된다.
- 큰 그룹에서 `pairs` 계열 모드는 run 수가 급격히 늘어난다.
- `search-mode adaptive`는 기본 gain을 기준으로 `kp`와 `kd`를 올릴지 내릴지 실제 tracking cost로 판단한다.
- 기본 `cost-mode tracking`은 torque를 덜 쓰는 것보다 잘 따라가는 것을 우선한다.
- `adaptive`에서 탐색 범위는 `--kp-min-scale`, `--kp-max-scale`, `--kd-min-scale`, `--kd-max-scale`로 제한하는 편이 안전하다.
- `--apply-best`는 단축 결과만 반영하므로, 직접 적용 목적이면 `--joint-mode single`로 실행하는 편이 맞다.
- 튜닝이 중간에 `Ctrl+C`로 중단되면 `--apply-best`는 건너뛴다.
- 같은 단일 동작에서 `--apply-best`를 반복하면 gain이 계속 커질 수 있으므로, 팔/팔꿈치/손목은 `--motion-preset` 또는 여러 `--motion-spec`을 묶는 편이 안전하다.

### `apply_tuned_gains.py`

`tune_gains.py`가 만든 `summary.json`에서 단축(single-axis) 최적 결과만 골라 `joint_setting.yaml`에 반영 가능한 새 YAML을 만드는 스크립트.

기준:

- 자동 적용 대상은 `best_by_target` 안의 단축 결과만 사용한다.
- `pairs`나 `together` 결과는 자동 반영하지 않는다.
- 최신 `summary.json`이면 run 당시의 baseline gain 메타데이터를 사용해 절대 `kp/kd`를 그대로 복원한다.
- 오래된 `summary.json`처럼 메타데이터가 없으면 기본적으로 적용을 거부하고, `--allow-scale-fallback`을 줘야 현재 YAML에 scale을 다시 곱한다.

권장 절차:

1. `tune_gains.py`를 `--joint-mode single`로 실행한다.
2. 생성된 `summary.json`으로 `apply_tuned_gains.py`를 실행한다.
3. 먼저 `joint_setting_autotuned.yaml`을 검토한다.
4. 검토 후 필요하면 원본 `joint_setting.yaml`에 반영한다.

대표 실행 예시:

```bash
python igris_teleop/robot_control/scripts/apply_tuned_gains.py \
  igris_teleop/outputs/tuning_runs/20260315_232159/summary.json
```

출력 파일을 직접 지정하려면:

```bash
python igris_teleop/robot_control/scripts/apply_tuned_gains.py \
  igris_teleop/outputs/tuning_runs/20260315_232159/summary.json \
  --output igris_teleop/robot_control/config/joint_setting_l_shoulder_autotuned.yaml
```

입력 파일을 바로 덮어쓰려면:

```bash
python igris_teleop/robot_control/scripts/apply_tuned_gains.py \
  igris_teleop/outputs/tuning_runs/20260315_232159/summary.json \
  --overwrite
```

오래된 summary를 현재 YAML 기준 scale 재적용으로 강제로 처리하려면:

```bash
python igris_teleop/robot_control/scripts/apply_tuned_gains.py \
  igris_teleop/outputs/tuning_runs/<timestamp>/summary.json \
  --allow-scale-fallback
```

출력 위치:

- 기본: `robot_control/config/joint_setting_autotuned.yaml`
- `--output` 지정 시 해당 경로
- `--overwrite` 지정 시 입력 `joint_setting.yaml` 자체

### `common_rollout.py`

직접 실행하는 스크립트가 아니라 rollout 공용 유틸 모듈.

포함 기능:

- `make_hold_profile`
- `make_sine_profile`
- `make_chirp_profile`
- `run_rollout`
- `compute_tracking_metrics`
- `save_episode_npz`

다른 스크립트에서 공통으로 사용한다.

### `__init__.py`

패키지 인식용 파일. 직접 실행하지 않는다.

## 빠른 시작

보정:

```bash
python igris_teleop/robot_control/scripts/calibrate_pr2ab.py --pair waist_rp --task calib
```

데이터 수집:

```bash
python igris_teleop/robot_control/scripts/record_actuatornet.py \
  --joint-group waist \
  --profile chirp \
  --duration 12 \
  --record-hz 100 \
  --move-to-default
```

gain sweep:

```bash
python igris_teleop/robot_control/scripts/tune_gains.py \
  --joint-group waist \
  --profile sine \
  --duration 8 \
  --kp-scale 0.5,0.8,1.0,1.2,1.5 \
  --kd-scale 0.5,0.8,1.0,1.2 \
  --move-to-default
```
