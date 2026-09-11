# config/robot_control

로봇 제어와 walking에 쓰는 joint/gain/stance 설정 파일을 둡니다.

- `joint_setting.yaml`: 기본 제어 설정입니다.
- `joint_setting_v2.yaml`, `joint_setting_sim_waist.yaml`: 변형 제어 설정입니다.
- `joint_setting_walking.yaml`: walking v1 설정입니다.
- `joint_setting_walking_v2.yaml`: walking `v2_fast_sac` 설정입니다.
- `init_setting.yaml`: 초기 pose 관련 설정입니다.
- `pr2ab_calibration.yaml`: parallel joint 보정값입니다.

실기 제어에 직접 영향을 주므로 변경 후에는 simulator 또는 저속/무부하 조건에서 먼저 확인합니다.
