# sharedmemory

worker 사이에서 camera frame, teleop pose, robot observation/action, dataset 상태를 공유하는 shared memory 계층입니다.

## 주요 파일

- `shm_schema.py`: 공유 메모리 block 이름, shape, dtype 같은 schema 정의입니다.
- `shmManager.py`: shared memory 생성/attach/read/write helper입니다.
- `shm_init_clean.py`: 실행 전 shared memory 초기화와 정리 유틸입니다.
- `resources.py`: shared memory resource 관리 보조 함수입니다.

schema를 변경하면 생산자 worker와 소비자 worker를 같이 확인해야 합니다. frame shape나 dtype 변경은 Web UI, camera worker, dataset 저장까지 영향을 줍니다.
