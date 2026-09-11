# teleop_devices

Unity, ROS, DDS, camera 등 외부 teleop 입력 장치와 bridge 코드를 둔 폴더입니다.

## 구성

- `unity/`: Unity ROS bridge, Unity pose constants, camera shared memory bridge입니다.
- `cameras/`: ROS camera interface, CycloneDDS camera bridge, DDS probe, threaded camera helper입니다.
- `cameras/tests/`: camera 장치 연결과 frame 확인용 수동 테스트 스크립트입니다.

camera DDS sender 상태 확인 예시:

```bash
python -m igris_teleop.teleop_devices.cameras.dds_probe --domain-id 10
```
