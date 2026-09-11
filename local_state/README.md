# local_state

개발 PC별 설정과 generated/build state를 두는 폴더입니다. 대부분은 Git 제외 대상이지만,
실물 손 브리지가 사용하는 `robot_hand_sdk` 스냅샷은 이식에 필수이므로 추적합니다.

## 현재 용도

- `ros_ws_generated/`: ROS2 관련 generated build/install/log state를 둘 수 있는 위치입니다.
- `cyclonedds_igris_lan.xml`: 현재 PC의 LAN 주소를 사용하는 DDS 설정입니다. 별도 백업하되
  새 PC의 인터페이스/주소에 맞춰 재설정해야 합니다. Git에는 포함하지 않습니다.
- [`robot_hand_sdk/`](robot_hand_sdk/README.md): `igris_c_hand` 빌드에 필요한 메시지 헤더와
  정적 라이브러리입니다. 일반 build cache가 아니므로 삭제하거나 빼고 이식하면 안 됩니다.

자세한 백업·재설치 절차는 [install_info.md](../install_info.md)를 참고하세요.
