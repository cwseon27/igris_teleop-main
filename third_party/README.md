# third_party

외부 SDK의 레거시 탐색 위치입니다. 현재 배포본의 이 폴더 아래 SDK 디렉터리는
비어 있으며 실제 배포물은 아래 경로에 보관합니다.

## 현재 구성

- [`ros_ws/src/igris_c_ros_bridge/thirdparty/igris_c_sdk_public`](../ros_ws/src/igris_c_ros_bridge/thirdparty/igris_c_sdk_public/README.md):
  public SDK wheel, headers, static libraries, 예제, 라이선스입니다.
- [`local_state/robot_hand_sdk`](../local_state/robot_hand_sdk/README.md):
  실물 손 브리지에 필요한 별도의 로봇 호환 메시지/라이브러리 스냅샷입니다.

설치 스크립트는 레거시 경로가 없으면 실제 배포 경로의 `dist/igris_c_sdk-*.whl`을
탐색합니다. 각 설치 환경의 검사와 전체 빌드 절차는 [install_info.md](../install_info.md)를
따르세요. 소스와 함께 라이브러리·wheel·라이선스 파일도 보존해야 합니다.

루트 `.gitmodules`는 원본 프로젝트의 외부 의존성 URL 기록입니다. 현재 배포본에는
해당 gitlink가 없으므로 `git submodule update`만으로 설치가 완성되지 않습니다.
특히 이미 설치된 ROS-TCP-Endpoint는 새 PC의 외부 workspace를 지정해 재사용합니다.
