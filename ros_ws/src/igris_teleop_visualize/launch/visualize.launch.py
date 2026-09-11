import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    description_pkg_path = get_package_share_directory('igris_c_description_public')
    urdf_file_name = 'igris_c_v2.urdf' 
    urdf_path = os.path.join(description_pkg_path, 'urdf', urdf_file_name)

    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    # [Node 1] Leader Arm Node (이제 얘가 대장입니다)
    # 직접 /joint_states로 쏘기 때문에 remappings 삭제
    leader_control_node = Node(
        package='igris_leader_control',
        executable='leader_node',
        name='leader_node',
        output='screen',
        parameters=[{'hand_enabled': True}],
    )

    # [Node 2] Robot State Publisher
    # URDF에 있는 이름은 TF 계산하고, 모르는 이름(Finger_T_R 등)은 무시합니다(에러 안 남).
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}],
        # arguments=[urdf_path]
    )

    # [Node 3] RViz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    return LaunchDescription([
        leader_control_node,
        robot_state_publisher_node,  # joint_state_publisher 제거됨
        rviz_node
    ])
