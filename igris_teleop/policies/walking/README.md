# policies/walking

walking policy profile 정의를 둡니다.

- `walking_profile.py`: profile 이름, backend, 기본 model path, observation/action 규칙을 정의합니다.

새 walking policy를 추가할 때는 여기에 profile을 추가하고 [workers/worker_walking_policy.py](../../workers/worker_walking_policy.py)의 처리 경로를 함께 확인합니다.
