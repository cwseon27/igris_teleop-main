# config

런타임에서 읽는 YAML 설정을 모아둔 폴더입니다. 코드 기본값보다 운영 파라미터 성격이 강한 값은 이곳에 둡니다.

## 구성

- `data/collect_data.yaml`: 데이터 수집 worker의 dataset/repo/frame 설정입니다.
- `robot_control/joint_setting*.yaml`: 기본 제어 gain과 joint limit/stance 설정입니다.
- `robot_control/joint_setting_walking.yaml`: walking v1 profile용 설정입니다.
- `robot_control/joint_setting_walking_v2.yaml`: walking `v2_fast_sac` profile용 설정입니다.
- `robot_control/init_setting.yaml`: 초기 pose/초기화 관련 설정입니다.
- `robot_control/pr2ab_calibration.yaml`: PR2AB 보정값입니다.

설정 파일을 바꾼 뒤에는 관련 worker를 재시작해야 반영됩니다.
