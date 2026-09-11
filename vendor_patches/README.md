# vendor_patches

외부 패키지에 적용해야 하는 local patch 파일을 보관합니다. 패키지 설치 후 site-packages 내부에 복사되는 파일은 이곳에서 원본을 관리합니다.

## 현재 구성

- `lerobot_modeling_act_gradcam.py`: `lerobot.policies.act.modeling_act_gradcam`으로 복사되는 Grad-CAM 지원 ACT policy 구현입니다.

복사는 `igris_teleop/setup_ml_venv.sh`에서 수행합니다.
