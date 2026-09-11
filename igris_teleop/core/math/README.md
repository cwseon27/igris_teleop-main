# core/math

pose, transform, trajectory 계산을 위한 공통 수학 유틸입니다.

- `transforms.py`: 좌표계/pose 변환 helper입니다.
- `trajectories.py`: 보간과 trajectory helper입니다.

특정 worker 내부에 중복 수학 코드를 넣기보다 이 폴더에 공통 함수를 둡니다.
