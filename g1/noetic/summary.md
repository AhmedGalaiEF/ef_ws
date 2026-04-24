# Jetson Board Noetic Inspection Summary

## Environment

- Shell: `/bin/bash`
- ROS distro: `noetic`
- ROS version: `1`
- `roscore`: `/opt/ros/noetic/bin/roscore`
- `roslaunch`: `/opt/ros/noetic/bin/roslaunch`
- `ROS_MASTER_URI`: `http://localhost:11311`
- `ROS_IP` was unset
- `ROS_HOSTNAME` was unset
- `ROS_PACKAGE_PATH`: `/opt/ros/noetic/share`
- `CMAKE_PREFIX_PATH`: `/opt/ros/noetic`
- `PYTHONPATH`: `/opt/ros/noetic/lib/python3/dist-packages`

## System State

- Hostname: `ubuntu`
- OS: Ubuntu 20.04.6 LTS (`focal`)
- Kernel: `5.10.104-tegra`
- Architecture: `aarch64`
- Root filesystem: `/dev/nvme0n1p1`, about 1.9T total, ~21G used
- Memory: 15 GiB RAM, about 2.2 GiB used during capture
- Swap: 7.5 GiB, unused during capture
- Uptime at capture: about 3 minutes

## Running Services and Processes

Key running services included:

- `ssh.service`
- `docker.service`
- `containerd.service`
- `master_service.service`
- `nvargus-daemon.service`
- `NetworkManager.service`
- `gdm.service`

Relevant Unitree processes seen:

- `/unitree/module/master_service/master_service`
- `/unitree/ota/pipe/ota_pipe_service`
- `/unitree/module/video_hub_pc4/videohub_pc4 /dev/video4`
- `/unitree/module/video_hub_pc4/videohub_pc4_chest /dev/video10`

Notably:

- `ps -ef | grep -i ros` showed no active ROS 1 processes
- no `roscore` or master process was visible

## Resource / Thermal Snapshot

`tegrastats` showed:

- RAM usage around 2.36 / 15.4 GB
- Swap usage 0
- CPU load light
- GPU activity 0%
- Temperatures roughly 38 C to 41 C

## Network / Ports

Observed open listeners included:

- `22` (`ssh`)
- `80`
- `4000` TCP and UDP
- DDS-related UDP ports `7400` and `7401`
- `7001` localhost
- `111` (`rpcbind`)

There was no clear listener on ROS master port `11311`.

## ROS 1 Installation State

Installed packages were a broad Noetic desktop-style set, including:

- camera and image packages
- point cloud and PCL packages
- `tf`, `tf2`, RViz, and `rqt_*`
- `realsense2_camera`
- controllers and hardware interface packages

Notably absent:

- no `unitree` ROS package was found in `rospack list`
- `rospack list | grep -i unitree` returned nothing

This looks like a standard Noetic install, not an active Unitree ROS 1 application workspace.

## Workspaces and Source Trees

Detected relevant directories included:

- `/home/unitree/unitree_sdk2-main`
- `/home/unitree/unitree/Odometer_service`
- `/home/unitree/cyclonedds_ws`

### `unitree_sdk2-main`

- SDK source tree with examples and libraries
- not a ROS 1 catkin workspace

### `cyclonedds_ws`

- ROS 2 / CycloneDDS workspace
- unrelated to active Noetic runtime

### `Odometer_service`

This is the main ROS 1 style tree found on the board:

- catkin-based structure
- many `package.xml` files
- SVO / VIO related packages such as:
  - `svo`
  - `svo_ros`
  - `svo_msgs`
  - `vikit_*`
  - `rpg_common`
  - `minkindr_*`

Launch files found there included:

- `live_nodelet.launch`
- `euroc_vio_stereo.launch`
- `euroc_global_map_mono.launch`
- `rs_camera.launch`
- `euroc_vio_mono.launch`
- frontend launch files for mono and stereo IMU workflows

No `.bag` files were found.

## ROS Master / Graph State

This was the dominant finding of the session:

- `rosnode list` failed with `ERROR: Unable to communicate with master!`
- `rostopic list` failed with the same error
- `rosservice list` failed with the same error
- `rosparam list` failed with the same error
- `rosparam get /robot_description` failed
- `rostopic echo /tf` and `/tf_static` failed
- `rosrun tf view_frames` failed because it could not register with the master

Interpretation:

- the Noetic environment was sourced correctly
- but no ROS 1 master was running at `http://localhost:11311`
- therefore there was no active ROS 1 graph to inspect during the capture

## Additional Notes

- The command intended to extract the host from `ROS_MASTER_URI` failed because the `sed` expression was broken across lines.
- A recursive grep for `source /opt/ros/noetic` was interrupted and also hit normal permission-denied paths under `/etc`.
- Because the master was unavailable, all topic, node, service, TF, and parameter discovery was blocked.

## Key Interpretation

- ROS 1 Noetic is installed and sourced.
- The board was not running a ROS 1 master during the inspection.
- No active ROS 1 nodes or topics were visible because graph discovery depends on a live master.
- There is a ROS 1 catkin project on disk in `~/unitree/unitree/Odometer_service`, focused on visual odometry / SVO.
- The rest of the board state still pointed more strongly to Unitree platform services and ROS 2 / DDS runtime components than to an active ROS 1 stack.

## Most Important Takeaways

- Noetic is present, but inactive at runtime in this session.
- `ROS_MASTER_URI` pointed to `localhost:11311`, but nothing was listening there.
- There were no Unitree ROS 1 packages registered in `rospack`.
- The only clear ROS 1 application tree found was `Odometer_service`, which appears to be an odometry / vision stack rather than the main robot runtime.
