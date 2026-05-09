# ROS 2 Topic Inventory

Generated: 2026-05-09T21:09:53+02:00

Environment:
- `source /opt/ros/foxy/setup.bash`
- `source /home/unitree/cyclonedds_ws/install/setup.bash`
- `source /home/unitree/EF/ef_ws/install/setup.bash`
- `ROS_LOCALHOST_ONLY=0`
- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`
- `CYCLONEDDS_URI=/home/unitree/cyclonedds_ws/cyclonedds.xml`
- Robot subnet interface: `eth0` at `192.168.123.164/24`

Probe method:
- `ros2 topic list -t` returned 131 ROS topics.
- A 20 second all-topic subscriber captured samples from 27 topics.
- Raw CycloneDDS built-in discovery was also run on `eth0` for comparison with the ROS graph.

Status meanings:
- `LIVE`: at least one message was received during the 20 second probe.
- `IDLE`: topic is present in the ROS graph, but no message arrived during the probe. For request/cmd topics this is normal unless a client is actively publishing.
- `TYPE ERROR`: topic is present, but this local overlay could not import the message type support needed to subscribe.
- `DDS INTERNAL`: discovered at DDS level, not a normal ROS application topic.

## Highlighted Topics

### Lowstate: Joint Poses And Velocities

Primary topics:
- `/lowstate` and `/lf/lowstate` are `LIVE`.
- Type: `unitree_hg/msg/LowState`.
- Structure: `version:uint32[2]`, `mode_pr:uint8`, `mode_machine:uint8`, `tick:uint32`, `imu_state:IMUState`, `motor_state:MotorState[35]`, `wireless_remote:uint8[40]`, `reserve:uint32[4]`, `crc:uint32`.
- Joint data is in `motor_state[0..34]`.
- Each `MotorState` has `mode`, `q`, `dq`, `ddq`, `tau_est`, `temperature[2]`, `vol`, `sensor[2]`, `motorstate`, `reserve[4]`.
- `q` is joint position, `dq` is joint velocity, `ddq` is joint acceleration, and `tau_est` is estimated torque.
- Sample: `/lf/lowstate` tick `5931734`, first motor `q=-0.329729`, `dq=0.022907`, `tau_est=1.582031`.

Related low-level command topics:
- `/lowcmd` is `LIVE` and contains `unitree_hg/msg/LowCmd` with `motor_cmd:MotorCmd[35]`.
- `/arm_sdk`, `/armsdk`, `/user_lowcmd`, and `/loco_sdk` were listed but did not publish a sample during the probe.

### General State

Primary topics:
- `/api/robot_state/request` and `/api/robot_state/response` are `LIVE`.
- Type: `unitree_api/msg/Request` and `unitree_api/msg/Response`.
- Response data is a JSON string listing service names and statuses. A sample included services such as `ai_sport`, `audio_player_service`, `auto_test_arm`, `auto_test_low`, `bashrunner`, `basic_service`, `battery_guard`, `dex3_service_l`, and `dex3_service_r`.
- `/api/sport/response` is `LIVE` and returned `{"data":501}` for API id `7001`.
- `/arm/action/state` is `LIVE` and returned JSON such as `{"holding": false, "id": 0, "name": ""}`.
- `/public_network_status` is `LIVE` and returned `{"network_status": "NetworkStatus.ON_WIFI_CONNECTED"}`.
- `/rtc/state` is `LIVE` and returned `{"connection_state":"not_connected"}`.

### Odometry

Primary topics:
- `/dog_odom` is `LIVE`.
- `/odommodestate` and `/lf/odommodestate` are `LIVE`.
- `/odom`, `/unitree/slam_mapping/odom`, and `/unitree/slam_relocation/odom` were listed but did not publish samples during the 20 second probe.

Structures:
- `/dog_odom`: `nav_msgs/msg/Odometry` with `header`, `child_frame_id`, `pose.pose.position/orientation`, and `twist.twist.linear/angular`.
- `/odommodestate`: `unitree_go/msg/SportModeState` with `position[3]`, `velocity[3]`, `yaw_speed`, `body_height`, `range_obstacle[4]`, `foot_force[4]`, and foot pose/speed arrays.
- Sample `/dog_odom`: frame `odom`, child `robot_center`, position about `x=-0.0416`, `y=0.0992`, `z=0.7088`.

### SLAM: Start, Stop, Relocation, Add Pose, Navigate, Obstacle Avoidance, Maps, Pointclouds

Control topics:
- `/api/slam_operate/request` is the command/request channel. It was listed but no request sample was observed.
- `/api/slam_operate/response` is the response channel. It was listed but no response sample was observed.
- Structure: `unitree_api/msg/Request` has `header`, `parameter:string`, `binary:uint8[]`; `unitree_api/msg/Response` has `header`, `data:string`, `binary:int8[]`.
- Workspace code maps SLAM operations through `slam_operate`, including start mapping, end/stop mapping, close SLAM, save/load map paths, and pose navigation. Those are API payloads on the request/response topics, not separate ROS topics.

State topics:
- `/slam_info` is `LIVE`. Type: `std_msgs/msg/String`; payload is JSON.
- Sample `/slam_info` contained `currentPose`, `startPose`, `targetPose`, `is_arrived`, `obsInfo`, and `progress`.
- `/slam_key_info` was listed but did not publish a sample.
- `/unitree_slam/waypoints` was listed but did not publish a sample.

Mapping and relocation topics:
- `/unitree/slam_mapping/points`, `/unitree/slam_relocation/points`, `/unitree/slam_relocation/web_points`, `/unitree/slam_relocation/global_map`: listed, no sample.
- `/unitree/slam_mapping/odom`, `/unitree/slam_relocation/odom`: listed, no sample.
- `/global_map`: listed as `nav_msgs/msg/OccupancyGrid`, no sample.
- `/gridmap` and `/planner_map`: listed as `grid_map_msgs/msg/GridMap`, but local type support is missing.

Obstacle avoidance and range:
- `/utlidar/range_info`: listed as `geometry_msgs/msg/PointStamped`, no sample.
- `/collision_clouds`, `/pre_collision_clouds`, `/safe_clouds`, `/pre_safe_clouds`, `/warning_clouds`, `/no_warning_clouds`, `/grid_clouds`: listed as `sensor_msgs/msg/PointCloud2`, no sample.
- `/ele_clouds` is `LIVE`, but the sample was an empty point cloud (`width=0`).

Pointcloud:
- `/utlidar/cloud_livox_mid360` is `LIVE`; sample point cloud had `frame_id=livox_frame`, `height=1`, `width=20256`, fields `x`, `y`, `z`, `intensity`, and byte-packed point data.
- `/utlidar/cloud_deskewed` was listed but did not publish a sample.

Saved map access/load:
- No separate saved-map list topic was found in `ros2 topic list`.
- Map save/load appears to be requested through `/api/slam_operate/request` with a map path parameter, and responses arrive on `/api/slam_operate/response`.
- `/global_map`, `/unitree/slam_relocation/global_map`, `/gridmap`, `/planner_map`, and `/utlidar/map_state` are map/state data topics, not file-access endpoints.

### IMU

Primary topics:
- `/dog_imu_raw` is `LIVE`.
- `/utlidar/imu_livox_mid360` is `LIVE`.
- Type: `sensor_msgs/msg/Imu`.
- Structure: `header`, `orientation`, `orientation_covariance[9]`, `angular_velocity`, `angular_velocity_covariance[9]`, `linear_acceleration`, `linear_acceleration_covariance[9]`.
- `/dog_imu_raw` sample frame: `dog_imu_link`.
- `/utlidar/imu_livox_mid360` sample frame: `livox_frame`.

### Secondary IMU

Primary topics:
- `/secondary_imu` is `LIVE`.
- `/lf/secondary_imu` was listed but did not publish a sample through the all-topic subscriber.
- Type: `unitree_hg/msg/IMUState`.
- Structure: `quaternion:float[4]`, `gyroscope:float[3]`, `accelerometer:float[3]`, `rpy:float[3]`, `temperature:int16`.
- Sample `/secondary_imu` temperature was `79`.

### RGBD / Video

Observed video topic:
- `/frontvideostream` is `LIVE`.
- Type: `unitree_go/msg/Go2FrontVideoData`.
- Structure: `time_frame:uint64`, `video720p:uint8[]`, `video360p:uint8[]`, `video180p:uint8[]`.
- Sample had non-empty `video720p`.

Not observed as ROS RGBD topics:
- No standard RGBD topics such as `/camera/color/image_raw`, `/camera/depth/image_rect_raw`, `/camera/color/camera_info`, or `/camera/depth/camera_info` appeared in `ros2 topic list`.
- `/api/videohub/request`, `/api/videohub/response`, and `/videohub/inner` exist but were idle during the probe.

### Stand Height

State:
- `body_height` exists in `unitree_go/msg/SportModeState` on `/odommodestate` and `/lf/odommodestate`.
- Both odometry-mode state topics were `LIVE`.
- Sample `/odommodestate` had `body_height=0.0`.

Command:
- No standalone `/stand_height` topic was found.
- Stand height commands appear to be API/sport commands rather than a dedicated topic. Workspace code calls `SetStandHeight(...)`, which is sent via the Unitree sport API path, not a named ROS topic.

### DDS Topics From `192.168.123.161` Not Listed By `ros2 topic list`

No extra published application topics from the `eth0` DDS discovery pass were found beyond the ROS 2 topic list, after normalizing `rt/<name>` to `/<name>`.

The raw DDS scan did show a few names that explain why DDS and `ros2 topic list` can differ:
- `ros_discovery_info`: DDS/RMW discovery metadata, not a normal application topic shown by `ros2 topic list`.
- `pixel_topic_` and `hight_map`: discovered as DDS subscriptions, not publications. They are not "returning data" because no publisher was discovered during the scan.
- DDS names are prefixed as `rt/...`; ROS 2 presents those as `/...`.
- Topics whose local type support is missing or inconsistent can still appear in `ros2 topic list` because discovery advertises the type name, but local subscription/import fails.

## Data Structure Reference

| Type | Structure summary |
|---|---|
| `std_msgs/msg/String` | `data:string` |
| `sensor_msgs/msg/Imu` | `header`, `orientation`, `orientation_covariance[9]`, `angular_velocity`, `angular_velocity_covariance[9]`, `linear_acceleration`, `linear_acceleration_covariance[9]` |
| `nav_msgs/msg/Odometry` | `header`, `child_frame_id`, `pose:PoseWithCovariance`, `twist:TwistWithCovariance` |
| `sensor_msgs/msg/PointCloud2` | `header`, `height`, `width`, `fields[]`, `is_bigendian`, `point_step`, `row_step`, `data:uint8[]`, `is_dense` |
| `nav_msgs/msg/OccupancyGrid` | `header`, `info:MapMetaData`, `data:int8[]` |
| `geometry_msgs/msg/PointStamped` | `header`, `point:{x,y,z}` |
| `unitree_api/msg/Request` | `header:RequestHeader`, `parameter:string`, `binary:uint8[]` |
| `unitree_api/msg/Response` | `header:ResponseHeader`, `data:string`, `binary:int8[]` |
| `unitree_hg/msg/LowState` | `version[2]`, `mode_pr`, `mode_machine`, `tick`, `imu_state`, `motor_state[35]`, `wireless_remote[40]`, `reserve[4]`, `crc` |
| `unitree_hg/msg/LowCmd` | `mode_pr`, `mode_machine`, `motor_cmd[35]`, `reserve[4]`, `crc` |
| `unitree_hg/msg/MotorState` | `mode`, `q`, `dq`, `ddq`, `tau_est`, `temperature[2]`, `vol`, `sensor[2]`, `motorstate`, `reserve[4]` |
| `unitree_hg/msg/MotorCmd` | `mode`, `q`, `dq`, `tau`, `kp`, `kd`, `reserve` |
| `unitree_hg/msg/IMUState` | `quaternion[4]`, `gyroscope[3]`, `accelerometer[3]`, `rpy[3]`, `temperature` |
| `unitree_go/msg/SportModeState` | `stamp`, `error_code`, `imu_state`, `mode`, `progress`, `gait_type`, `foot_raise_height`, `position[3]`, `body_height`, `velocity[3]`, `yaw_speed`, `range_obstacle[4]`, `foot_force[4]`, `foot_position_body[12]`, `foot_speed_body[12]` |
| `unitree_go/msg/Go2FrontVideoData` | `time_frame`, `video720p:uint8[]`, `video360p:uint8[]`, `video180p:uint8[]` |
| `unitree_go/msg/HeightMap` | `stamp`, `frame_id`, `resolution`, `width`, `height`, `origin[2]`, `data:float32[]` |
| `unitree_hg/msg/BmsState` | `version_high`, `version_low`, `fn`, `cell_vol[40]`, `bmsvoltage[3]`, `current`, `soc`, `soh`, `temperature[12]`, `cycle`, `manufacturer_date`, `bmsstate[5]`, `reserve[3]` |
| `unitree_hg/msg/MainBoardState` | `fan_state[6]`, `temperature[6]`, `value[6]`, `state[6]` |
| `unitree_hg/msg/HandState` | `motor_state[]`, `press_sensor_state[]`, `imu_state`, `power_v`, `power_a`, `system_v`, `device_v`, `error[2]`, `reserve[2]` |
| `unitree_hg/msg/HandCmd` | `motor_cmd[]`, `reserve[4]` |
| `unitree_go/msg/WirelessController` | `lx`, `ly`, `rx`, `ry`, `keys` |
| `unitree_go/msg/Error` | `source`, `state` |
| `unitree_go/msg/AudioData` | `time_frame`, `data:uint8[]` |

## Complete Topic Inventory

| Topic | Type | Status | Result / structure |
|---|---|---|---|
| `/SymState` | `unitree_go/msg/SymState` | TYPE ERROR | Listed, but local import failed: `unitree_go.msg` has no `SymState`. |
| `/api/action_store/request` | `unitree_api/msg/Request` | IDLE | Request envelope: `header`, `parameter`, `binary`. No sample. |
| `/api/action_store/response` | `unitree_api/msg/Response` | IDLE | Response envelope: `header`, `data`, `binary`. No sample. |
| `/api/arm/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/arm/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/audiohub/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/audiohub/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/bashrunner/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/bashrunner/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/basic_clearoip/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/basic_clearoip/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/basic_clearoip_lease/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/basic_clearoip_lease/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/basic_demarcate/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/basic_demarcate/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/basic_demarcate_lease/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/basic_demarcate_lease/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/basic_softlimit/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/basic_softlimit/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/basic_softlimit_lease/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/basic_softlimit_lease/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/basic_taumax/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/basic_taumax/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/basic_taumax_lease/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/basic_taumax_lease/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/config/request` | `unitree_api/msg/Request` | LIVE | Sample parameter JSON for `led_json`; envelope `header`, `parameter`, `binary`. |
| `/api/config/response` | `unitree_api/msg/Response` | LIVE | Sample status code `0`; envelope `header`, `data`, `binary`. |
| `/api/dex3_msg_controller/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/dex3_msg_controller/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/gesture/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/gpt/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/gpt/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/motion_switcher/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/motion_switcher/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/rm_con/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/robot_state/request` | `unitree_api/msg/Request` | LIVE | Repeated state requests; envelope `header`, `parameter`, `binary`. |
| `/api/robot_state/response` | `unitree_api/msg/Response` | LIVE | JSON service-status list in `data`; sample status code `0`. |
| `/api/robot_type_service/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/robot_type_service/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/slam_operate/request` | `unitree_api/msg/Request` | IDLE | SLAM command envelope. No active command during probe. |
| `/api/slam_operate/response` | `unitree_api/msg/Response` | IDLE | SLAM response envelope. No response during probe. |
| `/api/sport/request` | `unitree_api/msg/Request` | LIVE | Sport API request envelope. |
| `/api/sport/response` | `unitree_api/msg/Response` | LIVE | Sample `data` was `{"data":501}`. |
| `/api/videohub/request` | `unitree_api/msg/Request` | IDLE | Video API request envelope. No sample. |
| `/api/videohub/response` | `unitree_api/msg/Response` | IDLE | Video API response envelope. No sample. |
| `/api/voice/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/voice/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/api/vui/request` | `unitree_api/msg/Request` | IDLE | Request envelope. No sample. |
| `/api/vui/response` | `unitree_api/msg/Response` | IDLE | Response envelope. No sample. |
| `/arm/action/state` | `std_msgs/msg/String` | LIVE | JSON string; sample `{"holding": false, "id": 0, "name": ""}`. |
| `/arm_sdk` | `unitree_hg/msg/LowCmd` | IDLE | Low command structure. No sample. |
| `/armsdk` | `unitree_hg/msg/LowCmd` | IDLE | Low command structure. No sample. |
| `/audio_msg` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/audio_msg/filter` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/audiosender` | `unitree_go/msg/AudioData` | IDLE | `time_frame`, `data:uint8[]`. No sample. |
| `/collision_clouds` | `sensor_msgs/msg/PointCloud2` | IDLE | Point cloud structure. No sample. |
| `/config_change_status` | `unitree_go/msg/ConfigChangeStatus` | TYPE ERROR | Listed, but local import failed: `unitree_go.msg` has no `ConfigChangeStatus`. |
| `/dex3/left/cmd` | `unitree_hg/msg/HandCmd` | IDLE | `motor_cmd[]`, `reserve[4]`. No sample. |
| `/dex3/left/state` | `unitree_hg/msg/HandState` | IDLE | Hand state structure. No sample. |
| `/dex3/right/cmd` | `unitree_hg/msg/HandCmd` | IDLE | `motor_cmd[]`, `reserve[4]`. No sample. |
| `/dex3/right/state` | `unitree_hg/msg/HandState` | IDLE | Hand state structure. No sample. |
| `/dog_imu_raw` | `sensor_msgs/msg/Imu` | LIVE | IMU sample from `dog_imu_link`. |
| `/dog_odom` | `nav_msgs/msg/Odometry` | LIVE | Odometry sample from `odom` to `robot_center`. |
| `/ele_clouds` | `sensor_msgs/msg/PointCloud2` | LIVE | Point cloud sample with `width=0`, `frame_id=livox_frame`. |
| `/event/action_store` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/frontvideostream` | `unitree_go/msg/Go2FrontVideoData` | LIVE | Non-empty `video720p` byte array. |
| `/gesture/result` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/global_map` | `nav_msgs/msg/OccupancyGrid` | IDLE | Occupancy grid. No sample. |
| `/gpt_cmd` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/gpt_state` | `std_msgs/msg/String` | LIVE | JSON state string; sample includes `state:false`, `llm_name:"no"`. |
| `/gptflowfeedback` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/grid_clouds` | `sensor_msgs/msg/PointCloud2` | IDLE | Point cloud structure. No sample. |
| `/gridmap` | `grid_map_msgs/msg/GridMap` | TYPE ERROR | Listed, but `grid_map_msgs` is not installed in this overlay. |
| `/lf/agvalarmstate` | `unitree_go/msg/Error` | IDLE | `source`, `state`. No sample. |
| `/lf/agvbmsstate` | `unitree_hg/msg/AgvBmsState` | TYPE ERROR | Listed, but local import failed: `unitree_hg.msg` has no `AgvBmsState`. |
| `/lf/battery_alarm` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/lf/bmsstate` | `unitree_hg/msg/BmsState` | LIVE | Battery fields including `cell_vol[40]`, `current`, `soc`, `temperature[12]`. |
| `/lf/dex3/left/state` | `unitree_hg/msg/HandState` | IDLE | Hand state structure. No sample. |
| `/lf/dex3/right/state` | `unitree_hg/msg/HandState` | IDLE | Hand state structure. No sample. |
| `/lf/emergency_stop` | `unitree_go/msg/Error` | IDLE | `source`, `state`. No sample. |
| `/lf/lowstate` | `unitree_hg/msg/LowState` | LIVE | Lowstate with `motor_state[35]`; joint `q`, `dq`, `ddq`, `tau_est`. |
| `/lf/mainboardstate` | `unitree_hg/msg/MainBoardState` | LIVE | `fan_state[6]`, `temperature[6]`, `value[6]`, `state[6]`. |
| `/lf/odommodestate` | `unitree_go/msg/SportModeState` | LIVE | Sport/odom state with pose, velocity, obstacle range, body height. |
| `/lf/secondary_imu` | `unitree_hg/msg/IMUState` | IDLE | IMUState structure. No sample. |
| `/lf/sportmodestate` | `unitree_go/msg/SportModeState, unitree_hg/msg/SportModeState` | IDLE | Ambiguous advertised types; subscribed to `unitree_go` type, no sample. |
| `/loco_sdk` | `unitree_hg/msg/LowState` | IDLE | Lowstate structure. No sample. |
| `/log_system_inbound` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/log_system_outbound` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/lowcmd` | `unitree_hg/msg/LowCmd` | LIVE | Low command sample with `motor_cmd[35]`. |
| `/lowstate` | `unitree_hg/msg/LowState` | LIVE | Lowstate with `motor_state[35]`; joint `q`, `dq`, `ddq`, `tau_est`. |
| `/lowstate_doubleimu` | `unitree_hg_doubleimu/msg/doubleIMUState` | TYPE ERROR | Listed, but `unitree_hg_doubleimu` is not installed in this overlay. |
| `/multiplestate` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/no_warning_clouds` | `sensor_msgs/msg/PointCloud2` | IDLE | Point cloud structure. No sample. |
| `/odom` | `nav_msgs/msg/Odometry` | IDLE | Odometry structure. No sample. |
| `/odommodestate` | `unitree_go/msg/SportModeState` | LIVE | Sport/odom state; sample had `position`, `velocity`, `body_height=0.0`, `range_obstacle[4]`. |
| `/parameter_events` | `rcl_interfaces/msg/ParameterEvent` | IDLE | ROS parameter event structure. No sample. |
| `/planner_map` | `grid_map_msgs/msg/GridMap` | TYPE ERROR | Listed, but `grid_map_msgs` is not installed in this overlay. |
| `/pre_collision_clouds` | `sensor_msgs/msg/PointCloud2` | IDLE | Point cloud structure. No sample. |
| `/pre_safe_clouds` | `sensor_msgs/msg/PointCloud2` | IDLE | Point cloud structure. No sample. |
| `/public_network_status` | `std_msgs/msg/String` | LIVE | JSON network status string. |
| `/rosout` | `rcl_interfaces/msg/Log` | LIVE | ROS log messages from the probe node. |
| `/rtc/state` | `std_msgs/msg/String` | LIVE | JSON connection state string. |
| `/rtc_status` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/safe_clouds` | `sensor_msgs/msg/PointCloud2` | IDLE | Point cloud structure. No sample. |
| `/secondary_imu` | `unitree_hg/msg/IMUState` | LIVE | `quaternion[4]`, `gyroscope[3]`, `accelerometer[3]`, `rpy[3]`, `temperature`. |
| `/selftest` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/servicestate` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/servicestateactivate` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/slam_info` | `std_msgs/msg/String` | LIVE | JSON SLAM state string with pose/progress/obstacle fields. |
| `/slam_key_info` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/sportmodestate` | `unitree_hg/msg/SportModeState` | TYPE ERROR | Listed, but local import failed: `unitree_hg.msg` has no `SportModeState`. |
| `/unitree/slam_mapping/odom` | `nav_msgs/msg/Odometry` | IDLE | SLAM mapping odometry. No sample. |
| `/unitree/slam_mapping/points` | `sensor_msgs/msg/PointCloud2` | IDLE | SLAM mapping point cloud. No sample. |
| `/unitree/slam_relocation/global_map` | `sensor_msgs/msg/PointCloud2` | IDLE | Relocation global map point cloud. No sample. |
| `/unitree/slam_relocation/odom` | `nav_msgs/msg/Odometry` | IDLE | Relocation odometry. No sample. |
| `/unitree/slam_relocation/points` | `sensor_msgs/msg/PointCloud2` | IDLE | Relocation point cloud. No sample. |
| `/unitree/slam_relocation/web_points` | `sensor_msgs/msg/PointCloud2` | IDLE | Relocation web point cloud. No sample. |
| `/unitree_slam/waypoints` | `std_msgs/msg/String` | IDLE | Waypoint string payload. No sample. |
| `/user_lowcmd` | `unitree_hg/msg/LowCmd` | IDLE | Low command structure. No sample. |
| `/utlidar/cloud_deskewed` | `sensor_msgs/msg/PointCloud2` | IDLE | Deskewed lidar point cloud. No sample. |
| `/utlidar/cloud_livox_mid360` | `sensor_msgs/msg/PointCloud2` | LIVE | Lidar point cloud sample from `livox_frame`, width about `20256`. |
| `/utlidar/imu_livox_mid360` | `sensor_msgs/msg/Imu` | LIVE | Lidar IMU sample from `livox_frame`. |
| `/utlidar/map_state` | `unitree_go/msg/HeightMap` | IDLE | Height map structure: `resolution`, `width`, `height`, `origin[2]`, `data[]`. No sample. |
| `/utlidar/range_info` | `geometry_msgs/msg/PointStamped` | IDLE | Obstacle/range point. No sample. |
| `/videohub/inner` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/warning_clouds` | `sensor_msgs/msg/PointCloud2` | IDLE | Point cloud structure. No sample. |
| `/webrtcreq` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/webrtcres` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/wirelesscontroller` | `unitree_go/msg/WirelessController` | IDLE | `lx`, `ly`, `rx`, `ry`, `keys`. No sample. |
| `/xfk_webrtcreq` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
| `/xfk_webrtcres` | `std_msgs/msg/String` | IDLE | String payload. No sample. |
