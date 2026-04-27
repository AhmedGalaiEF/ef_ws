# Jetson Board Inspection Summary

## Environment

- Shell: `/bin/bash`
- ROS distro: `foxy`
- ROS version: `2`
- RMW implementation: `rmw_cyclonedds_cpp`
- `ros2` binary: `/opt/ros/foxy/bin/ros2`
- `ROS_DOMAIN_ID` was unset
- `AMENT_PREFIX_PATH` included `/home/unitree/cyclonedds_ws/install/rmw_cyclonedds_cpp` and `/opt/ros/foxy`

## System State

- Hostname: `ubuntu`
- OS: Ubuntu 20.04.6 LTS (`focal`)
- Kernel: `5.10.104-tegra`
- Architecture: `aarch64`
- Root filesystem: `/dev/nvme0n1p1`, about 1.9T total, ~21G used
- Memory: 15 GiB RAM, about 2.2 GiB used during capture
- Swap: 7.5 GiB, unused during capture
- Uptime at capture: about 6 minutes

## Running Services and Processes

Key running services included:

- `ssh.service`
- `docker.service`
- `containerd.service`
- `master_service.service`
- `unitree-upgrade.service`
- `nvargus-daemon.service`
- `NetworkManager.service`
- `gdm.service`

Relevant Unitree processes seen:

- `/unitree/module/master_service/master_service`
- `/unitree/ota/pipe/ota_pipe_service`
- `/unitree/module/video_hub_pc4/videohub_pc4 /dev/video4`
- `/unitree/module/video_hub_pc4/videohub_pc4_chest /dev/video10`

Notably, `ps -ef | grep -i ros` showed no explicit ROS user processes at that moment.

## Resource / Thermal Snapshot

`tegrastats` showed:

- RAM usage around 2.38 / 15.4 GB
- Swap usage 0
- CPU load light to moderate
- GPU activity 0%
- Temperatures roughly 39 C to 42 C

`nvidia-smi` was not available, which is normal on Jetson.

## Network / Ports

Observed open listeners included:

- `22` (`ssh`)
- `80`
- `4000` TCP and UDP
- DDS-related UDP ports `7400` and `7401`
- `7001` on localhost / IPv6 localhost
- `111` (`rpcbind`)

This suggests board services and DDS middleware were active even though no ROS 2 nodes were listed.

## ROS 2 Package State

Installed ROS 2 packages included:

- Core Foxy packages
- Navigation 2 (`nav2_*`)
- `robot_localization`
- `cv_bridge`
- `image_transport`
- `joy`
- `rmw_cyclonedds_cpp`

The package list looked like a standard Foxy desktop/navigation-oriented install plus CycloneDDS support, not a clearly custom G1 ROS 2 application workspace.

## Workspaces and Source Trees

Detected ROS-related workspaces / trees:

- `/home/unitree/cyclonedds_ws`
- `/home/unitree/unitree/Odometer_service`

`cyclonedds_ws` contained:

- `src/cyclonedds-0.10.2`
- `src/ros2/rmw_cyclonedds/rmw_cyclonedds_cpp`
- matching `build/`, `install/`, and `log/`

`Odometer_service` looked like a ROS 1 / catkin stack, not ROS 2:

- many `package.xml` files under catkin build/profile directories
- SVO / VIO related packages such as `svo`, `svo_ros`, `svo_msgs`, `vikit_*`, `rpg_common`

No `*.launch.py` or `*.launch.xml` files were found in the searched home directory.

## ROS Graph Snapshot

### Nodes

- `ros2 node list` returned nothing

### Services

- `ros2 service list` returned nothing

### Actions

- `ros2 action list` returned nothing

### Topics

`ros2 topic list` returned many active topics, including:

- low-level robot state and command topics:
  - `/lowcmd`
  - `/lowstate`
  - `/user_lowcmd`
  - `/multiplestate`
  - `/sportmodestate`
  - `/odommodestate`
- API request/response topics:
  - `/api/sport/request`, `/api/sport/response`
  - `/api/arm/request`, `/api/arm/response`
  - `/api/robot_state/request`, `/api/robot_state/response`
  - `/api/motion_switcher/request`, `/api/motion_switcher/response`
  - `/api/slam_operate/request`, `/api/slam_operate/response`
- SLAM / mapping topics:
  - `/unitree/slam_mapping/odom`
  - `/unitree/slam_mapping/points`
  - `/unitree/slam_relocation/odom`
  - `/unitree/slam_relocation/points`
  - `/unitree_slam/waypoints`
  - `/slam_info`
  - `/slam_key_info`
  - `/global_map`
  - `/planner_map`
  - `/gridmap`
- LiDAR / IMU topics:
  - `/utlidar/cloud_livox_mid360`
  - `/utlidar/imu_livox_mid360`
  - `/dog_imu_raw`
  - `/secondary_imu`
- camera / media / WebRTC topics:
  - `/frontvideostream`
  - `/videohub/inner`
  - `/webrtcreq`
  - `/webrtcres`
- arm / dexterous hand topics:
  - `/dex3/left/cmd`
  - `/dex3/right/cmd`
  - `/dex3/left/state`
  - `/dex3/right/state`

## Key Interpretation

- The board was correctly configured for ROS 2 Foxy with CycloneDDS.
- Unitree platform services were running.
- The system was publishing a substantial ROS 2 topic graph.
- Despite active topics, ROS 2 node/service/action discovery returned empty. That usually points to one of:
  - vendor components publishing DDS topics without normal ROS 2 graph visibility
  - discovery / namespace / middleware quirks
  - the shell not being in the same discovery context as the publishers
- The only clearly identified source workspace was `cyclonedds_ws`; no obvious ROS 2 application workspace or launch files for G1 were found in the searched paths.
- There is also an `Odometer_service` tree, but it appears to be ROS 1 catkin-based visual odometry code rather than the main ROS 2 runtime stack.

## Most Important Takeaways

- ROS 2 Foxy is installed and sourced.
- CycloneDDS is the active middleware.
- Unitree services, video processes, and DDS traffic are active.
- Many robot topics are present, especially for motion, SLAM, LiDAR, audio, video, and arm control.
- ROS 2 graph introspection is incomplete because nodes, services, and actions appeared empty.
