# igris_artifacts

실행 중 생성되거나 외부에서 가져온 산출물을 두는 폴더입니다. 대부분의 내용은 git 추적 대상이 아니며, README와 일부 기본 walking model만 추적합니다.

## 하위 폴더

- `datasets/`: 수집된 LeRobot dataset 위치입니다.
- `checkpoints/`: 학습 checkpoint 위치입니다.
- `logs/`: Desktop launcher, inference, robot control, walking log 위치입니다.
- `walking/`: walking policy model과 handoff 문서입니다.

큰 dataset/checkpoint/log는 repo에 commit하지 않습니다. 필요한 경우 외부 스토리지나 별도 artifact 관리 경로를 사용합니다.
