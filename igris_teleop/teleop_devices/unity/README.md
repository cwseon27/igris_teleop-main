# teleop_devices/unity

Unity에서 들어오는 pose/camera 데이터를 Python runtime과 연결하는 bridge 코드입니다.

- `unity_ros_interface.py`: Unity/ROS pose interface입니다.
- `camera_shm_bridge.py`: camera shared memory bridge입니다.
- `constants.py`: Unity pose/topic 관련 상수입니다.

Unity TCP endpoint 실행은 [ros_ws/README.md](../../../ros_ws/README.md)의 ROS TCP endpoint 항목을 참고합니다.
