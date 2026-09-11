# hand_control/hand_urdf

IGRIS hand retargeting에 사용하는 URDF, mesh, YAML 설정입니다.

- `igris_hand.yml`: retargeting 설정입니다.
- `left_hand_igris_c.urdf`, `right_hand_igris_c.urdf`: 좌/우 hand URDF입니다.
- `meshes/`: URDF에서 참조하는 hand mesh입니다.

URDF 경로나 joint 이름을 바꾸면 [hand_retargeting.py](../hand_retargeting.py)와 worker hand 경로를 함께 확인합니다.
