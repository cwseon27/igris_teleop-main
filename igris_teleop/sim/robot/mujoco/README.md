# sim/robot/mujoco

MuJoCo simulator에서 로드하는 XML model을 둡니다.

- `igris_c_v2_with_hand.xml`: 12 actuator hand, distal mimic joint, head stereo camera를 포함한 기본 simulator XML입니다.
- `igris_c_v2.xml`: body 중심 XML입니다.
- `igris_c_v2_parallel.xml`: parallel joint 실험/검증용 XML입니다.

런타임에서 다른 XML을 쓰려면 `IGRIS_SIM_XML_PATH`를 지정합니다.
