# docs

코드와 직접 연결되지는 않지만 운영 중 필요한 설정 메모를 둔 폴더입니다.

## 현재 문서

- `camera_dds_setting.txt`: camera DDS 설정과 확인 절차 메모입니다.
- `runtime_assets.sha256`: 배포에 필요한 모델·보정 맵·SDK 바이너리 18개의 SHA-256입니다.
  저장소 루트에서 `sha256sum -c docs/runtime_assets.sha256`로 누락·손상을 검사합니다.
- `examples/cyclonedds_igris_lan.xml`: 새 PC용 LAN 설정 예시입니다. 실제 PC의 주소로
  편집한 뒤 사용해야 하며 자동 적용되지 않습니다.

다른 PC로 옮기는 전체 설치 절차는 [install_info.md](../install_info.md)가 기준입니다.
[INSTALL.md](../INSTALL.md)는 Python 설치 스크립트의 옵션을 설명합니다.
