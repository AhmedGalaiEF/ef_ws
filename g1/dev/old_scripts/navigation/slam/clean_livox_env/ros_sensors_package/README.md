# ros_sensors_package

ROS 2 DDS bridge package for the sensor paths present in this workspace.

## What the existing files show

- `mid360_config.json` configures the Livox MID-360 network:
  - lidar-side ports: `56100` command, `56200` push, `56300` point, `56400` imu, `56500` log
  - host-side ports: `56101` command, `56201` push, `56301` point, `56401` imu, `56501` log
  - multicast group: `224.1.1.5`
- host IP is now set for the Jetson board: `192.168.123.164`
- `livox2_python.py` uses Livox SDK2 callbacks to receive raw UDP packets and decode them into XYZ points.
- `sdk_client.py` and `sdk_sensors.py` show that the robot boards also expose deskewed lidar and IMU on CycloneDDS-backed Unitree topics such as `rt/utlidar/cloud_deskewed` and `rt/utlidar/imu_livox_mid360`.
- `rgbd_client.py` shows an older RGBD transport over ZeroMQ `SUB` on `tcp://10.34.0.83:5555` as multipart `rgb_jpeg`, `depth_png`, `depth_scale`.
- `rgbd_usb_publisher.py` is the Jetson-oriented path: it captures directly from a USB RGBD camera using `pyrealsense2` and republishes onto ROS 2 topics.

## Topics published by this package

- `/livox/points` as `sensor_msgs/PointCloud2`
- `/rgbd/color/image_raw` as `sensor_msgs/Image`
- `/rgbd/depth/image_raw` as `sensor_msgs/Image`
- `/rgbd/color/camera_info` as `sensor_msgs/CameraInfo`
- `/rgbd/depth/camera_info` as `sensor_msgs/CameraInfo`

## Notes

- This package publishes onto ROS 2 topics, which means transport is DDS through the active ROS 2 RMW implementation.
- On the Jetson deployment, the lidar should send UDP traffic to `192.168.123.164`, and the RGBD camera is expected to be locally attached over USB.
- The depth scale is preserved in the depth `CameraInfo.d` array because no dedicated custom message was added here.
- Camera intrinsics are parameters; if you know `fx`, `fy`, `cx`, and `cy`, pass them to the RGBD node.
- `rgbd_usb_publisher` requires `pyrealsense2` on the Jetson.

## Build

Place `ros_sensors_package` in a ROS 2 workspace `src/` directory, then:

```bash
colcon build --packages-select ros_sensors_package
source install/setup.bash
```

## Run

```bash
ros2 launch ros_sensors_package sensors_bridge.launch.py
ros2 launch ros_sensors_package jetson_sensors.launch.py
```

Or individually:

```bash
ros2 run ros_sensors_package livox_points_publisher
ros2 run ros_sensors_package rgbd_usb_publisher
ros2 run ros_sensors_package rgbd_zmq_publisher
```
