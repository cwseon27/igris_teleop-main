# teleop_devices/cameras

카메라 입력과 camera transport bridge를 둡니다.

- `ros_interface.py`: ROS camera topic interface입니다.
- `cyclonedds_bridge.py`: CycloneDDS camera bridge입니다.
- `dds_probe.py`: robot camera DDS sender 확인 도구입니다.
- `threaded_camera.py`: threaded camera capture helper입니다.
- `stereo_rectify_maps_tuned.npz`: stereo 보정 map입니다.
- `tests/`: 장치별 수동 camera 테스트입니다.

DDS probe:

```bash
python -m igris_teleop.teleop_devices.cameras.dds_probe --domain-id 10
```
