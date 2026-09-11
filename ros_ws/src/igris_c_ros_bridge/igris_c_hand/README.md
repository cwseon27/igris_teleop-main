# igris_c_hand

`igris_c_hand` is a ROS2 bridge package that exposes the IGRIS-C hand DDS interface through ROS2 topics and a service. The bridge uses the wire schema of the robot's installed hand node (including the `HandCmd`/`HandState` header), which differs from the older public SDK checkout on the robot.

## Purpose

- Subscribe to DDS `rt/handstate` and publish ROS2 `std_msgs/msg/Float32MultiArray`
- Convert ROS2 hand commands into DDS `rt/handcmd` messages and publish them periodically
- Expose the hand initialization trigger as a ROS2 service

## Parameters

- `domain_id` (`int`, default: `0`): DDS domain ID
- `dds_namespace` (`string`, default: `igris_c_IG05`)
- `dds_command_topic` (`string`, default: `igris_c_IG05/rt/handcmd`)
- `dds_state_topic` (`string`, default: `igris_c_IG05/rt/handstate`)
- `ros_command_topic` (`string`, default: `/igris_teleop/hand/command`)
- `ros_state_topic` (`string`, default: `/igris_teleop/hand/state`)
- `ros_init_service` (`string`, default: `/igris_teleop/hand/init`)
- `publish_rate_hz` (`double`, default: `100.0`)

## ROS2 Interface

- Topic `/igris_teleop/hand/command` (`std_msgs/msg/Float32MultiArray`)
- Topic `/igris_teleop/hand/state` (`std_msgs/msg/Float32MultiArray`)
- Service `/igris_teleop/hand/init` (`std_srvs/srv/Trigger`)

The command array must contain 12 float values in this motor order:

- `11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26`

Each value is expected to be in the `0.0` to `1.0` range. Out-of-range values are clamped by the node.

The state array publishes `HandState.motor_state()[i].q()` in the received order.

## Hand Initialization

Hand initialization is not a separate DDS service in the SDK example. It is triggered by publishing a `HandCmd` message with the special motor ID `99`.

In ROS2, use:

```bash
ros2 service call /igris_c/hand/init std_srvs/srv/Trigger
```

After initialization is requested, periodic command publishing remains paused until a new ROS hand command is received.

## Run

```bash
ros2 run igris_c_hand igris_c_hand_bridge_node
```

Example command publish:

```bash
ros2 topic pub /igris_c/hand/command std_msgs/msg/Float32MultiArray \
  "{data: [0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0]}"
```
