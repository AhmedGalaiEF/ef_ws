# ROS Debug Notes

Generated: 2026-05-09

Working directory used in the session:

```bash
unitree@ubuntu:~/EF/ef_ws/g1$
```

## Local File Context

`service_toggle.py` was checked and edited:

```bash
wc service_toggle.py -l
# 235 service_toggle.py

nano service_toggle.py
esp
# prompted for sudo password
```

Notes:
- `service_toggle.py` exists under `~/EF/ef_ws/g1`.
- It was 235 lines at the time of the session.
- `esp` was run after editing and required sudo.

## ROS Environment

The session sourced ROS Foxy directly:

```bash
source /opt/ros/foxy/setup.bash
```

Important implication:
- This was not the full overlay used by some other probes, for example it did not explicitly source `/home/unitree/cyclonedds_ws/install/setup.bash` or `/home/unitree/EF/ef_ws/install/setup.bash`.
- Even with only Foxy sourced, `unitree_api`, `unitree_go`, and `unitree_hg` were visible in `ros2 pkg list`.

## SLAM Topics

Command:

```bash
ros2 topic list | grep slam
```

Result:

```text
/api/slam_operate/request
/api/slam_operate/response
/slam_info
/slam_key_info
/unitree/slam_mapping/odom
/unitree/slam_mapping/points
/unitree/slam_relocation/global_map
/unitree/slam_relocation/odom
/unitree/slam_relocation/points
/unitree/slam_relocation/web_points
/unitree_slam/waypoints
```

Same result was observed with hidden topics included:

```bash
ros2 topic list --include-hidden-topics | grep slam
```

Conclusion:
- There are no additional hidden SLAM topics beyond the visible list.
- SLAM control appears topic/API based through `/api/slam_operate/request` and `/api/slam_operate/response`, not through ROS services.

## Map Topics

Command:

```bash
ros2 topic list | grep map
```

Result:

```text
/global_map
/gridmap
/planner_map
/unitree/slam_mapping/odom
/unitree/slam_mapping/points
/unitree/slam_relocation/global_map
/utlidar/map_state
```

Same result was observed with hidden topics included:

```bash
ros2 topic list --include-hidden-topics | grep map
```

Conclusion:
- No extra hidden map topics were found.
- Saved map load/save is not exposed as a ROS service in this session. It is expected to be part of the `slam_operate` request/response API payloads.

## Range And Obstacle Topics

Command:

```bash
ros2 topic list | grep range
```

Result:

```text
/utlidar/range_info
```

Command:

```bash
ros2 topic list | grep obst
```

Result:

```text
# no output
```

Conclusion:
- The only topic matching `range` was `/utlidar/range_info`.
- No topic name contains `obst`.
- Obstacle-related data may still be represented by pointcloud topics such as `/collision_clouds`, `/safe_clouds`, `/warning_clouds`, `/pre_collision_clouds`, and `/pre_safe_clouds`, but they do not contain `obst` in the topic name.

## State Topics

Command:

```bash
ros2 topic list | grep state
```

Result:

```text
/api/robot_state/request
/api/robot_state/response
/arm/action/state
/dex3/left/state
/dex3/right/state
/gpt_state
/lf/agvalarmstate
/lf/agvbmsstate
/lf/bmsstate
/lf/dex3/left/state
/lf/dex3/right/state
/lf/lowstate
/lf/mainboardstate
/lf/odommodestate
/lf/sportmodestate
/lowstate
/lowstate_doubleimu
/multiplestate
/odommodestate
/rtc/state
/servicestate
/servicestateactivate
/sportmodestate
/utlidar/map_state
```

Same result was observed with hidden topics included:

```bash
ros2 topic list --include-hidden-topics | grep state
```

Conclusion:
- No additional hidden state topics were found.
- Important state channels are `/lowstate`, `/lf/lowstate`, `/odommodestate`, `/lf/odommodestate`, `/api/robot_state/request`, and `/api/robot_state/response`.

## Services And Nodes

Commands:

```bash
ros2 service list
ros2 node list
```

Observed result:
- Both commands returned no visible entries in this session.

Conclusion:
- The robot interfaces visible from this shell are mainly DDS/ROS topic endpoints.
- Standard ROS 2 services and nodes were not discoverable from this environment at the time.
- This supports the observation that Unitree APIs such as `slam_operate`, `sport`, `robot_state`, and `config` are exposed through request/response topics rather than standard ROS services.

## Package Inventory

Command:

```bash
ros2 pkg list
```

Observed package categories:
- ROS Foxy core packages: `rcl`, `rclcpp`, `rclpy`, `ros2cli`, `ros2topic`, `ros2service`, `ros2node`, `ros2pkg`, `rosidl_*`, `rmw_*`.
- Message packages: `std_msgs`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `map_msgs`, `trajectory_msgs`, `visualization_msgs`, `diagnostic_msgs`.
- Navigation packages: `navigation2`, `nav2_amcl`, `nav2_controller`, `nav2_costmap_2d`, `nav2_map_server`, `nav2_planner`, `nav2_bt_navigator`, `nav2_waypoint_follower`, `nav2_util`.
- Visualization/tools: `rviz2`, `rqt_*`, `tf2_*`, `robot_state_publisher`.
- Unitree packages: `unitree_api`, `unitree_go`, `unitree_hg`.

Notable absence:
- `grid_map_msgs` was not listed, even though `/gridmap` and `/planner_map` advertise `grid_map_msgs/msg/GridMap`.
- `unitree_hg_doubleimu` was not listed, even though `/lowstate_doubleimu` advertises `unitree_hg_doubleimu/msg/doubleIMUState`.

This explains local type-support errors when attempting to subscribe to those topics from this shell.

## Executable Inventory

Command:

```bash
ros2 pkg executables
```

Observed result:
- Standard ROS demo/tool executables were listed, including `demo_nodes_cpp`, `demo_nodes_py`, `image_tools`, `joy`, `nav2_*`, `robot_localization`, `robot_state_publisher`, `rviz2`, `teleop_twist_*`, `tf2_ros`, `tf2_tools`, and `topic_monitor`.
- No Unitree executables were shown for `unitree_api`, `unitree_go`, or `unitree_hg`.

Conclusion:
- The Unitree packages visible here provide message definitions, but not command-line ROS executables.
- Operational control is expected to come from scripts in the workspace and DDS/API request topics, not `ros2 run unitree_* ...`.

## Interface Command Note

Command:

```bash
ros2 interface show
```

Result:

```text
usage: ros2 interface show [-h] type
ros2 interface show: error: the following arguments are required: type
```

Correct usage:

```bash
ros2 interface show unitree_hg/msg/LowState
ros2 interface show unitree_go/msg/SportModeState
ros2 interface show unitree_api/msg/Request
ros2 interface show unitree_api/msg/Response
```

## Topic List Help

Command:

```bash
ros2 topic list --help
```

Useful options:
- `-t`, `--show-types`: show topic types.
- `-c`, `--count-topics`: only show count.
- `--include-hidden-topics`: include hidden topics.
- `--spin-time SPIN_TIME`: wait for discovery when not using an existing daemon.
- `--no-daemon`: avoid using the ROS daemon.

## Full Topic List With Types

Command:

```bash
ros2 topic list -t
```

Result:

```text
/SymState [unitree_go/msg/SymState]
/api/action_store/request [unitree_api/msg/Request]
/api/action_store/response [unitree_api/msg/Response]
/api/arm/request [unitree_api/msg/Request]
/api/arm/response [unitree_api/msg/Response]
/api/audiohub/request [unitree_api/msg/Request]
/api/audiohub/response [unitree_api/msg/Response]
/api/bashrunner/request [unitree_api/msg/Request]
/api/bashrunner/response [unitree_api/msg/Response]
/api/basic_clearoip/request [unitree_api/msg/Request]
/api/basic_clearoip/response [unitree_api/msg/Response]
/api/basic_clearoip_lease/request [unitree_api/msg/Request]
/api/basic_clearoip_lease/response [unitree_api/msg/Response]
/api/basic_demarcate/request [unitree_api/msg/Request]
/api/basic_demarcate/response [unitree_api/msg/Response]
/api/basic_demarcate_lease/request [unitree_api/msg/Request]
/api/basic_demarcate_lease/response [unitree_api/msg/Response]
/api/basic_softlimit/request [unitree_api/msg/Request]
/api/basic_softlimit/response [unitree_api/msg/Response]
/api/basic_softlimit_lease/request [unitree_api/msg/Request]
/api/basic_softlimit_lease/response [unitree_api/msg/Response]
/api/basic_taumax/request [unitree_api/msg/Request]
/api/basic_taumax/response [unitree_api/msg/Response]
/api/basic_taumax_lease/request [unitree_api/msg/Request]
/api/basic_taumax_lease/response [unitree_api/msg/Response]
/api/config/request [unitree_api/msg/Request]
/api/config/response [unitree_api/msg/Response]
/api/dex3_msg_controller/request [unitree_api/msg/Request]
/api/dex3_msg_controller/response [unitree_api/msg/Response]
/api/gesture/request [unitree_api/msg/Request]
/api/gpt/request [unitree_api/msg/Request]
/api/gpt/response [unitree_api/msg/Response]
/api/motion_switcher/request [unitree_api/msg/Request]
/api/motion_switcher/response [unitree_api/msg/Response]
/api/rm_con/request [unitree_api/msg/Request]
/api/robot_state/request [unitree_api/msg/Request]
/api/robot_state/response [unitree_api/msg/Response]
/api/robot_type_service/request [unitree_api/msg/Request]
/api/robot_type_service/response [unitree_api/msg/Response]
/api/slam_operate/request [unitree_api/msg/Request]
/api/slam_operate/response [unitree_api/msg/Response]
/api/sport/request [unitree_api/msg/Request]
/api/sport/response [unitree_api/msg/Response]
/api/videohub/request [unitree_api/msg/Request]
/api/videohub/response [unitree_api/msg/Response]
/api/voice/request [unitree_api/msg/Request]
/api/voice/response [unitree_api/msg/Response]
/api/vui/request [unitree_api/msg/Request]
/api/vui/response [unitree_api/msg/Response]
/arm/action/state [std_msgs/msg/String]
/arm_sdk [unitree_hg/msg/LowCmd]
/armsdk [unitree_hg/msg/LowCmd]
/audio_msg [std_msgs/msg/String]
/audio_msg/filter [std_msgs/msg/String]
/audiosender [unitree_go/msg/AudioData]
/collision_clouds [sensor_msgs/msg/PointCloud2]
/config_change_status [unitree_go/msg/ConfigChangeStatus]
/dex3/left/cmd [unitree_hg/msg/HandCmd]
/dex3/left/state [unitree_hg/msg/HandState]
/dex3/right/cmd [unitree_hg/msg/HandCmd]
/dex3/right/state [unitree_hg/msg/HandState]
/dog_imu_raw [sensor_msgs/msg/Imu]
/dog_odom [nav_msgs/msg/Odometry]
/ele_clouds [sensor_msgs/msg/PointCloud2]
/event/action_store [std_msgs/msg/String]
/frontvideostream [unitree_go/msg/Go2FrontVideoData]
/gesture/result [std_msgs/msg/String]
/global_map [nav_msgs/msg/OccupancyGrid]
/gpt_cmd [std_msgs/msg/String]
/gpt_state [std_msgs/msg/String]
/gptflowfeedback [std_msgs/msg/String]
/grid_clouds [sensor_msgs/msg/PointCloud2]
/gridmap [grid_map_msgs/msg/GridMap]
/lf/agvalarmstate [unitree_go/msg/Error]
/lf/agvbmsstate [unitree_hg/msg/AgvBmsState]
/lf/battery_alarm [std_msgs/msg/String]
/lf/bmsstate [unitree_hg/msg/BmsState]
/lf/dex3/left/state [unitree_hg/msg/HandState]
/lf/dex3/right/state [unitree_hg/msg/HandState]
/lf/emergency_stop [unitree_go/msg/Error]
/lf/lowstate [unitree_hg/msg/LowState]
/lf/mainboardstate [unitree_hg/msg/MainBoardState]
/lf/odommodestate [unitree_go/msg/SportModeState]
/lf/secondary_imu [unitree_hg/msg/IMUState]
/lf/sportmodestate [unitree_go/msg/SportModeState, unitree_hg/msg/SportModeState]
/loco_sdk [unitree_hg/msg/LowState]
/log_system_inbound [std_msgs/msg/String]
/log_system_outbound [std_msgs/msg/String]
/lowcmd [unitree_hg/msg/LowCmd]
/lowstate [unitree_hg/msg/LowState]
/lowstate_doubleimu [unitree_hg_doubleimu/msg/doubleIMUState]
/multiplestate [std_msgs/msg/String]
/no_warning_clouds [sensor_msgs/msg/PointCloud2]
/odom [nav_msgs/msg/Odometry]
/odommodestate [unitree_go/msg/SportModeState]
/parameter_events [rcl_interfaces/msg/ParameterEvent]
/planner_map [grid_map_msgs/msg/GridMap]
/pre_collision_clouds [sensor_msgs/msg/PointCloud2]
/pre_safe_clouds [sensor_msgs/msg/PointCloud2]
/public_network_status [std_msgs/msg/String]
/rosout [rcl_interfaces/msg/Log]
/rtc/state [std_msgs/msg/String]
/rtc_status [std_msgs/msg/String]
/safe_clouds [sensor_msgs/msg/PointCloud2]
/secondary_imu [unitree_hg/msg/IMUState]
/selftest [std_msgs/msg/String]
/servicestate [std_msgs/msg/String]
/servicestateactivate [std_msgs/msg/String]
/slam_info [std_msgs/msg/String]
/slam_key_info [std_msgs/msg/String]
/sportmodestate [unitree_hg/msg/SportModeState]
/unitree/slam_mapping/odom [nav_msgs/msg/Odometry]
/unitree/slam_mapping/points [sensor_msgs/msg/PointCloud2]
/unitree/slam_relocation/global_map [sensor_msgs/msg/PointCloud2]
/unitree/slam_relocation/odom [nav_msgs/msg/Odometry]
/unitree/slam_relocation/points [sensor_msgs/msg/PointCloud2]
/unitree/slam_relocation/web_points [sensor_msgs/msg/PointCloud2]
/unitree_slam/waypoints [std_msgs/msg/String]
/user_lowcmd [unitree_hg/msg/LowCmd]
/utlidar/cloud_deskewed [sensor_msgs/msg/PointCloud2]
/utlidar/cloud_livox_mid360 [sensor_msgs/msg/PointCloud2]
/utlidar/imu_livox_mid360 [sensor_msgs/msg/Imu]
/utlidar/map_state [unitree_go/msg/HeightMap]
/utlidar/range_info [geometry_msgs/msg/PointStamped]
/videohub/inner [std_msgs/msg/String]
/warning_clouds [sensor_msgs/msg/PointCloud2]
/webrtcreq [std_msgs/msg/String]
/webrtcres [std_msgs/msg/String]
/wirelesscontroller [unitree_go/msg/WirelessController]
/xfk_webrtcreq [std_msgs/msg/String]
/xfk_webrtcres [std_msgs/msg/String]
```

## Hidden Topic Check

Command:

```bash
ros2 topic list --include-hidden-topics
```

Observed result:
- The hidden-topic list matched the normal visible topic list.
- No additional hidden topic names were discovered.

Conclusion:
- For this robot/session, hidden ROS topics do not add additional SLAM, state, map, range, or obstacle topics.

