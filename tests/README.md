# tests

pytest 기반 회귀 테스트입니다. runtime process 전체를 띄우기보다, command process, external worker env, shared memory, Web UI helper, IK smoke 성격의 단위 테스트를 둡니다.

## 실행

```bash
source .venv/bin/activate
pytest tests
```

IK 관련 테스트는 `.venv-ik` 또는 lock된 IK dependency가 필요할 수 있습니다. 설치는 [INSTALL.md](../INSTALL.md)를 기준으로 맞춥니다.
