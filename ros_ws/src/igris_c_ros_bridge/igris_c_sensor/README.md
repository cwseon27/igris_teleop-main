# igris_c_sensor

`igris_c_sensor` is a ROS2 bridge package that republishes IGRIS-C DDS sensor streams as ROS2 `sensor_msgs/msg/CompressedImage` topics.

## Purpose

- Subscribe to IGRIS-C DDS compressed image topics
- Publish the same streams to ROS2 using `sensor_msgs/msg/CompressedImage`
- Use the current ROS time if the DDS header does not contain a timestamp

## Parameters

- `domain_id` (`int`, default: `10`): DDS domain ID
- `igris_topics` (`string[]`): list of DDS topics to bridge
- `frame_id` (`string`, default: `igris_c`): fallback frame ID when the DDS header frame ID is empty
- `resize_scale` (`double`, default: `0.5`): output scale for ordinary republished color images. Depth and stereo SBS streams preserve their native resolution
- `combined_resize_scale` (`double`, default: `0.5`): scale applied to `left_hand`, `d435_color`, and `right_hand` before composing the combined color strip
- `combined_color_topic` (`string`, default: `/rs_comp/combined/color/image/compressed`): compressed topic for the combined `left_hand + d435_color + right_hand` image
- `combined_jpeg_quality` (`int`, default: `85`): JPEG quality used for the combined color output
- `hand_rotate_180` (`bool`, default: `true`): rotate the `left_hand` and `right_hand` image topics by 180 degrees before republishing and before composing the combined color strip
- `stereo_enabled` (`bool`, default: `true`): enable split and rectification for the ROS2 stereo SBS topic
- `stereo_ros_source_topic` (`string`, default: `/igris_c_IG05/sensor/eyes_stereo/compressed`): ROS2 compressed SBS input topic
- `stereo_map_path` (`string`, default: `stereo_rectify_maps_tuned.yml.gz`): rectification map file resolved from `stereo_sbs_cam_pub/share/config` or the source tree
- `stereo_swap_lr` (`bool`, default: `false`): swap the split left/right images before rectification when the incoming SBS halves are reversed
- `stereo_left_topic` (`string`, default: `/left/image_rect/compressed`): rectified left compressed image topic
- `stereo_right_topic` (`string`, default: `/right/image_rect/compressed`): rectified right compressed image topic
- `stereo_left_info_topic` (`string`, default: `/igris_c/sensor/left/camera_info`): rectified left camera info topic
- `stereo_right_info_topic` (`string`, default: `/igris_c/sensor/right/camera_info`): rectified right camera info topic
- `stereo_left_frame_id` (`string`, default: `left_camera`): frame id for the rectified left stream
- `stereo_right_frame_id` (`string`, default: `right_camera`): frame id for the rectified right stream
- `stereo_jpeg_quality` (`int`, default: `85`): JPEG quality used for the rectified stereo outputs

Default `igris_topics`:

- `igris_c/sensor/d435_color`
- `igris_c/sensor/d435_depth`
- `igris_c/sensor/eyes_stereo`
- `igris_c/sensor/left_hand`
- `igris_c/sensor/right_hand`

## ROS2 Interface

Each DDS topic is republished to a ROS2 topic with a `/compressed` suffix.

- Example: DDS `igris_c/sensor/d435_color` -> ROS2 `igris_c/sensor/d435_color/compressed`

When `stereo_enabled` is true, the ROS2 topic configured by `stereo_ros_source_topic` is decoded as SBS stereo, split into left/right, rectified with the calibration maps from `stereo_sbs_cam_pub`, and republished to:

- `/left/image_rect/compressed`
- `/right/image_rect/compressed`
- `/igris_c/sensor/left/camera_info`
- `/igris_c/sensor/right/camera_info`

The bridge also publishes a horizontally concatenated combined color image:

- `/rs_comp/combined/color/image/compressed`

The `igris_c/sensor/left_hand/compressed` and `igris_c/sensor/right_hand/compressed` outputs are rotated by 180 degrees by default. The same rotation is applied before those images are inserted into the combined color strip.

Stereo processing order:

1. Decode the compressed SBS frame from `/igris_c_IG05/sensor/eyes_stereo/compressed` (`1280x480`)
2. Split into left/right (`640x480`)
3. Optionally swap the split left/right images if `stereo_swap_lr` is enabled
4. Preserve the upright source orientation without rotation
5. Verify that each eye exactly matches the calibration map size (`640x480`); reject mismatched frames instead of stretching them
6. Apply stereo rectification and publish the native `640x480` result without a post-rectification resize

Message type:

- `sensor_msgs/msg/CompressedImage`

When `resize_scale` is less than `1.0`, the bridge decodes ordinary incoming color frames, resizes them, and encodes them again before publishing. JPEG and PNG streams are supported for resizing. The stereo SBS input is always republished at native resolution so its split halves match the calibration maps.

## Run

```bash
ros2 run igris_c_sensor igris_c_sensor_bridge_node
```

When the workspace is sourced through `install/setup.bash`, `ROS_DOMAIN_ID` defaults to `97` unless it is already set in the shell.

Example with a custom topic list:

```bash
ros2 run igris_c_sensor igris_c_sensor_bridge_node --ros-args \
  -p igris_topics:="[igris_c/sensor/d435_color,igris_c/sensor/d435_depth]"
```

Example with half-resolution output:

```bash
ros2 run igris_c_sensor igris_c_sensor_bridge_node --ros-args \
  -p resize_scale:=0.5
```

Example with stereo split + rectify outputs:

```bash
ros2 run igris_c_sensor igris_c_sensor_bridge_node --ros-args \
  -p resize_scale:=0.5 \
  -p stereo_enabled:=true
```
