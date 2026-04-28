# ros_sensors_package

`ros_sensors_package` is a ROS 2 `ament_python` package created from the sensor code in this environment so the Jetson board can publish both lidar and RGBD data as standard ROS 2 topics over DDS.

## Package function

The package has two main jobs:

1. `livox_points_publisher`
   - Receives Livox MID-360 lidar data through the Livox SDK2 UDP pipeline.
   - Aggregates packets into frames.
   - Publishes the result as `sensor_msgs/PointCloud2` on `/livox/points`.

2. `rgbd_usb_publisher`
   - Reads a USB RGBD camera connected to the Jetson.
   - Publishes color and depth images as ROS 2 topics.
   - Publishes matching `CameraInfo` messages.

There is also an older optional bridge:

- `rgbd_zmq_publisher`
  - Reads RGBD data from a ZeroMQ stream instead of from a local USB camera.
  - This is not the main Jetson deployment path.

## Network model

From the files in this environment:

- The Livox MID-360 communicates with the host using the Livox SDK2 network path over UDP.
- The active config is in [mid360_config.json](/home/ag/ef_ws/g1/dev/old_scripts/navigation/slam/clean_livox_env/mid360_config.json).
- The host IP in that config was updated for the Jetson board: `192.168.123.164`.
- The RGBD camera is expected to be connected directly to the Jetson by USB.

Livox ports in the config:

- lidar side:
  - `56100` command
  - `56200` push
  - `56300` point
  - `56400` imu
  - `56500` log
- host side:
  - `56101` command
  - `56201` push
  - `56301` point
  - `56401` imu
  - `56501` log
- multicast:
  - `224.1.1.5`

## ROS 2 topics

The package publishes:

- `/livox/points` as `sensor_msgs/PointCloud2`
- `/rgbd/color/image_raw` as `sensor_msgs/Image`
- `/rgbd/depth/image_raw` as `sensor_msgs/Image`
- `/rgbd/color/camera_info` as `sensor_msgs/CameraInfo`
- `/rgbd/depth/camera_info` as `sensor_msgs/CameraInfo`

## Build

After placing the package in a ROS 2 workspace:

```bash
colcon build --packages-select ros_sensors_package
```

Then source the workspace:

```bash
source /path/to/your_ws/install/setup.bash
```

## Run

Recommended launch on the Jetson:

```bash
ros2 launch ros_sensors_package jetson_sensors.launch.py
```

You can also run the nodes separately:

```bash
ros2 run ros_sensors_package livox_points_publisher
ros2 run ros_sensors_package rgbd_usb_publisher
```

Optional older RGBD ZMQ bridge:

```bash
ros2 run ros_sensors_package rgbd_zmq_publisher
```

## Check published topics

List the topics:

```bash
ros2 topic list
```

Check publish rate:

```bash
ros2 topic hz /livox/points
ros2 topic hz /rgbd/color/image_raw
ros2 topic hz /rgbd/depth/image_raw
```

Inspect message contents:

```bash
ros2 topic echo /livox/points
ros2 topic echo /rgbd/color/camera_info
ros2 topic echo /rgbd/depth/camera_info
```

## Visualize in RViz2

Start RViz:

```bash
rviz2
```

Add these displays:

- `PointCloud2` with topic `/livox/points`
- `Image` with topic `/rgbd/color/image_raw`
- `Image` with topic `/rgbd/depth/image_raw`

## Notes

- ROS 2 topics are transported using DDS through the ROS 2 middleware.
- `rgbd_usb_publisher` requires `pyrealsense2` on the Jetson.
- If `/livox/points` is empty, verify the lidar is actually sending to `192.168.123.164`.
- If the RGBD node does not start, verify the USB camera is connected and detected by the Jetson.
