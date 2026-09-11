# IGRIS-C ROS2 Bridge

[![Build and Test - Jazzy](https://github.com/mjlee111/igris_c_ros_bridge/actions/workflows/jazzy.yml/badge.svg)](https://github.com/mjlee111/igris_c_ros_bridge/actions/workflows/jazzy.yml)

ROS2 bridge packages for IGRIS-C.

## Overview

This repository provides ROS2 packages that subscribe to IGRIS-C data over CycloneDDS and republish it into ROS2-native topics.

```text
IGRIS-C DDS -> Bridge Node -> ROS2 Topics
```

## Supported Packages

| Package | Type | Description |
| --- | --- | --- |
| `igris_c_sensor` | Sensor bridge | Republishes IGRIS-C compressed sensor image topics to ROS2 `sensor_msgs/msg/CompressedImage`. |
| `igris_c_hand` | Hand bridge | Bridges IGRIS-C hand state and command topics to ROS2 `std_msgs/msg/Float32MultiArray`, with a ROS init service. |