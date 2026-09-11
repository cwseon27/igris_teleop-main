# config/data

데이터 수집 worker가 읽는 dataset 관련 설정을 둡니다.

- `collect_data.yaml`: repo id, episode 저장, frame 구성 등 collect worker 설정입니다.

dataset 저장 위치나 metadata 구성을 바꿀 때는 [workers/worker_collect_data.py](../../workers/worker_collect_data.py)와 함께 확인합니다.
