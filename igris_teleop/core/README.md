# core

멀티프로세스 런타임에서 공통으로 사용하는 작은 기반 모듈입니다. 특정 worker 기능보다 lifecycle, event, rate, 상태 전이 같은 공통 규칙을 둡니다.

## 주요 파일

- `worker_base.py`: worker가 공유하는 context/base class 정의입니다.
- `events.py`: `ready`, `start`, `home`, `shutdown`, record trigger 같은 event bus 정의입니다.
- `state_machine.py`: teleop 전체 상태 전이입니다.
- `record_state_machine.py`: 데이터 수집 record 상태 전이입니다.
- `rate.py`: loop 주기 제어 유틸입니다.
- `project_paths.py`: repo root와 공통 경로 계산입니다.
- `math/`: pose/trajectory 변환 유틸입니다.

새 worker를 추가할 때는 이 폴더의 event/state 규칙을 먼저 맞춘 뒤 [workers/](../workers/README.md)에 구현을 추가합니다.
