# ROS2 Topic Inventory

Generated: 2026-05-02T14:28:38

Probe method: `ros2 topic list -t`, then `ros2 topic echo --no-arr <topic>` with a 3.0 second timeout per non-command topic.

Status meanings:
- `LIVE`: at least one message sample was observed during the probe.
- `COMMAND/REQUEST`: command or request topic; usually quiet until something publishes a command.
- `IDLE RESPONSE`: response topic; usually quiet until a matching request is active.
- `IDLE SYSTEM`: normal ROS system topic that may be quiet.
- `NO SAMPLE`: topic exists, but no message arrived during this short probe.

Summary: 130 topics total, 9 live, 59 command/idle/system, 62 no sample.

## Live Topics

| Topic | Type | Sample |
|---|---|---|
| `/dog_imu_raw` | `sensor_msgs/msg/Imu` | header: stamp: sec: 1777724672 nanosec: 504264761 frame_id: dog_imu_link orientation: x: -0.0071146320551633835 y: -0.003938368521630764 z: -0.00903232116252184 w: 0.99992620944... |
| `/dog_odom` | `nav_msgs/msg/Odometry` | header: stamp: sec: 1777724675 nanosec: 433380106 frame_id: odom child_frame_id: robot_center pose: pose: position: x: 0.04996326565742493 y: 0.004511621780693531 z: 0.683653235... |
| `/lf/bmsstate` | `unitree_hg/msg/BmsState` | version_high: 1 version_low: 6 fn: 5 cell_vol: '<array type: uint16[40]>' bmsvoltage: '<array type: uint32[3]>' current: -2116 soc: 31 soh: 99 temperature: '<array type: int16[1... |
| `/lf/lowstate` | `unitree_hg/msg/LowState` | version: '<array type: uint32[2]>' mode_pr: 0 mode_machine: 5 tick: 5742548 imu_state: quaternion: '<array type: float[4]>' gyroscope: '<array type: float[3]>' accelerometer: '<... |
| `/lf/odommodestate` | `unitree_go/msg/SportModeState` | stamp: sec: 0 nanosec: 0 error_code: 0 imu_state: quaternion: '<array type: float[4]>' gyroscope: '<array type: float[3]>' accelerometer: '<array type: float[3]>' rpy: '<array t... |
| `/lowstate` | `unitree_hg/msg/LowState` | version: '<array type: uint32[2]>' mode_pr: 0 mode_machine: 5 tick: 5767555 imu_state: quaternion: '<array type: float[4]>' gyroscope: '<array type: float[3]>' accelerometer: '<... |
| `/odommodestate` | `unitree_go/msg/SportModeState` | stamp: sec: 0 nanosec: 0 error_code: 0 imu_state: quaternion: '<array type: float[4]>' gyroscope: '<array type: float[3]>' accelerometer: '<array type: float[3]>' rpy: '<array t... |
| `/secondary_imu` | `unitree_hg/msg/IMUState` | quaternion: '<array type: float[4]>' gyroscope: '<array type: float[3]>' accelerometer: '<array type: float[3]>' rpy: '<array type: float[3]>' temperature: 79 --- quaternion: '<... |
| `/utlidar/imu_livox_mid360` | `sensor_msgs/msg/Imu` | header: stamp: sec: 1777724823 nanosec: 588977152 frame_id: livox_frame orientation: x: 0.0 y: 0.0 z: 0.0 w: 0.0 orientation_covariance: '<array type: double[9]>' angular_veloci... |

## Command, Request, Response, Or Idle System Topics

| Topic | Type | Status |
|---|---|---|
| `/api/action_store/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/action_store/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/arm/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/arm/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/audiohub/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/audiohub/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/bashrunner/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/bashrunner/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/basic_clearoip/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/basic_clearoip/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/basic_clearoip_lease/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/basic_clearoip_lease/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/basic_demarcate/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/basic_demarcate/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/basic_demarcate_lease/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/basic_demarcate_lease/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/basic_softlimit/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/basic_softlimit/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/basic_softlimit_lease/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/basic_softlimit_lease/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/basic_taumax/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/basic_taumax/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/basic_taumax_lease/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/basic_taumax_lease/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/config/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/config/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/dex3_msg_controller/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/dex3_msg_controller/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/gesture/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/gpt/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/gpt/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/motion_switcher/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/motion_switcher/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/rm_con/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/robot_state/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/robot_state/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/robot_type_service/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/robot_type_service/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/slam_operate/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/slam_operate/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/sport/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/sport/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/videohub/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/videohub/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/videohub_chest/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/videohub_chest/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/voice/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/voice/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/api/vui/request` | `unitree_api/msg/Request` | COMMAND/REQUEST |
| `/api/vui/response` | `unitree_api/msg/Response` | IDLE RESPONSE |
| `/arm_sdk` | `unitree_hg/msg/LowCmd` | COMMAND/REQUEST |
| `/armsdk` | `unitree_hg/msg/LowCmd` | COMMAND/REQUEST |
| `/dex3/left/cmd` | `unitree_hg/msg/HandCmd` | COMMAND/REQUEST |
| `/dex3/right/cmd` | `unitree_hg/msg/HandCmd` | COMMAND/REQUEST |
| `/gpt_cmd` | `std_msgs/msg/String` | COMMAND/REQUEST |
| `/lowcmd` | `unitree_hg/msg/LowCmd` | COMMAND/REQUEST |
| `/parameter_events` | `rcl_interfaces/msg/ParameterEvent` | IDLE SYSTEM |
| `/rosout` | `rcl_interfaces/msg/Log` | IDLE SYSTEM |
| `/user_lowcmd` | `unitree_hg/msg/LowCmd` | COMMAND/REQUEST |

## No Sample During Probe

| Topic | Type |
|---|---|
| `/SymState` | `unitree_go/msg/SymState` |
| `/arm/action/state` | `std_msgs/msg/String` |
| `/audio_msg` | `std_msgs/msg/String` |
| `/audio_msg/filter` | `std_msgs/msg/String` |
| `/audiosender` | `unitree_go/msg/AudioData` |
| `/collision_clouds` | `sensor_msgs/msg/PointCloud2` |
| `/config_change_status` | `unitree_go/msg/ConfigChangeStatus` |
| `/dex3/left/state` | `unitree_hg/msg/HandState` |
| `/dex3/right/state` | `unitree_hg/msg/HandState` |
| `/ele_clouds` | `sensor_msgs/msg/PointCloud2` |
| `/event/action_store` | `std_msgs/msg/String` |
| `/frontvideostream` | `unitree_go/msg/Go2FrontVideoData` |
| `/gesture/result` | `std_msgs/msg/String` |
| `/global_map` | `nav_msgs/msg/OccupancyGrid` |
| `/gpt_state` | `std_msgs/msg/String` |
| `/gptflowfeedback` | `std_msgs/msg/String` |
| `/grid_clouds` | `sensor_msgs/msg/PointCloud2` |
| `/gridmap` | `grid_map_msgs/msg/GridMap` |
| `/lf/agvalarmstate` | `unitree_go/msg/Error` |
| `/lf/agvbmsstate` | `unitree_hg/msg/AgvBmsState` |
| `/lf/battery_alarm` | `std_msgs/msg/String` |
| `/lf/dex3/left/state` | `unitree_hg/msg/HandState` |
| `/lf/dex3/right/state` | `unitree_hg/msg/HandState` |
| `/lf/emergency_stop` | `unitree_go/msg/Error` |
| `/lf/mainboardstate` | `unitree_hg/msg/MainBoardState` |
| `/lf/secondary_imu` | `unitree_hg/msg/IMUState` |
| `/lf/sportmodestate` | `unitree_go/msg/SportModeState, unitree_hg/msg/SportModeState` |
| `/loco_sdk` | `unitree_hg/msg/LowState` |
| `/log_system_inbound` | `std_msgs/msg/String` |
| `/log_system_outbound` | `std_msgs/msg/String` |
| `/lowstate_doubleimu` | `unitree_hg_doubleimu/msg/doubleIMUState` |
| `/multiplestate` | `std_msgs/msg/String` |
| `/no_warning_clouds` | `sensor_msgs/msg/PointCloud2` |
| `/planner_map` | `grid_map_msgs/msg/GridMap` |
| `/pre_collision_clouds` | `sensor_msgs/msg/PointCloud2` |
| `/pre_safe_clouds` | `sensor_msgs/msg/PointCloud2` |
| `/public_network_status` | `std_msgs/msg/String` |
| `/rtc/state` | `std_msgs/msg/String` |
| `/rtc_status` | `std_msgs/msg/String` |
| `/safe_clouds` | `sensor_msgs/msg/PointCloud2` |
| `/selftest` | `std_msgs/msg/String` |
| `/servicestate` | `std_msgs/msg/String` |
| `/servicestateactivate` | `std_msgs/msg/String` |
| `/slam_info` | `std_msgs/msg/String` |
| `/slam_key_info` | `std_msgs/msg/String` |
| `/sportmodestate` | `unitree_hg/msg/SportModeState` |
| `/unitree/slam_mapping/odom` | `nav_msgs/msg/Odometry` |
| `/unitree/slam_mapping/points` | `sensor_msgs/msg/PointCloud2` |
| `/unitree/slam_relocation/global_map` | `sensor_msgs/msg/PointCloud2` |
| `/unitree/slam_relocation/odom` | `nav_msgs/msg/Odometry` |
| `/unitree/slam_relocation/points` | `sensor_msgs/msg/PointCloud2` |
| `/unitree/slam_relocation/web_points` | `sensor_msgs/msg/PointCloud2` |
| `/unitree_slam/waypoints` | `std_msgs/msg/String` |
| `/utlidar/cloud_livox_mid360` | `sensor_msgs/msg/PointCloud2` |
| `/utlidar/range_info` | `geometry_msgs/msg/PointStamped` |
| `/videohub/inner` | `std_msgs/msg/String` |
| `/warning_clouds` | `sensor_msgs/msg/PointCloud2` |
| `/webrtcreq` | `std_msgs/msg/String` |
| `/webrtcres` | `std_msgs/msg/String` |
| `/wirelesscontroller` | `unitree_go/msg/WirelessController` |
| `/xfk_webrtcreq` | `std_msgs/msg/String` |
| `/xfk_webrtcres` | `std_msgs/msg/String` |
