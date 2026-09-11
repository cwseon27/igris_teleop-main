# robot_control/interfaces

로봇 제어 계층이 외부 시스템과 통신할 때 쓰는 adapter를 둡니다.

- `master_arm_ros_interface.py`: master arm ROS interface helper입니다.

worker에서 직접 외부 API를 깊게 다루기보다 이 폴더의 interface를 통해 분리합니다.
