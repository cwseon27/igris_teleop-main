# policies

학습된 policy를 런타임에서 선택하고 해석하기 위한 profile 코드를 둔 폴더입니다.

## 구성

- `walking/walking_profile.py`: walking profile별 backend, model path, observation/action 해석 규칙을 정의합니다.

현재 walking worker는 `v1` TorchScript와 `v2_fast_sac` ONNX profile을 구분해서 실행합니다. 기본 model 산출물은 [igris_artifacts/](../../igris_artifacts/README.md)에 둡니다.
