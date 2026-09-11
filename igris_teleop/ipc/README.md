# ipc

프로세스 실행과 종료를 보조하는 IPC 관련 모듈입니다. 현재는 런타임 worker orchestration을 단순화하기 위한 manager 계층을 둡니다.

## 주요 파일

- `manager.py`: 프로세스 관리 보조 로직입니다.

대용량 실시간 데이터 공유는 이 폴더가 아니라 [sharedmemory/](../sharedmemory/README.md)의 shared memory schema를 사용합니다.
