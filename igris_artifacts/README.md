# igris_artifacts

실행 중 생성되거나 외부에서 가져온 산출물을 두는 폴더입니다. 대부분은 Git 제외
대상이지만 README, 일부 기본 walking model, 아래 필수 PR2AB 보정본은 추적합니다.

## 하위 폴더

- `datasets/`: 수집된 LeRobot dataset 위치입니다.
- `checkpoints/`: 학습 checkpoint 위치입니다.
- `logs/`: Desktop launcher, inference, robot control, walking log 위치입니다.
- `walking/`: walking policy model과 handoff 문서입니다.

## 실물 시작에 필요한 보정 파일

`logs/robot_control/pr2ab_calibration.yaml`은 로그가 아니라 실제 제어 worker가
MS 모드에서 읽는 필수 PR2AB 보정 파일입니다. 허리·양쪽 발목·양쪽 손목의 변환과
모터 범위(`ab_limits`)가 들어 있으며 Git에 포함합니다. 같은 디렉터리의 다른 로그는
계속 제외합니다. 기존 운용본의 값과 실행 경로는 바꾸지 않았습니다.

`igris_teleop/config/robot_control/pr2ab_calibration.yaml`은 동명의 이전 보정본이며
`ab_limits`가 없어 이 파일의 대체재가 아닙니다. 새 PC에서 파일이 없다는 이유로
이전 파일을 복사하거나 안전 검사를 해제하지 마세요.

보정값은 로봇과 기구 구성에 종속됩니다. 다른 로봇/변경된 기구에 무조건 적용할 수
없습니다. 현재 배포본의 무결성과 비구동 검사 방법은 [install_info.md](../install_info.md)를
참고하세요.

큰 dataset/checkpoint/log는 repo에 commit하지 않습니다. 필요한 경우 외부 스토리지나 별도 artifact 관리 경로를 사용합니다.
