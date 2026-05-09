# Unitree SLAM `keyDemo` Notes

Generated: 2026-05-09

Source reviewed:
- `/home/unitree/slam_example/src/keyDemo.cpp`
- `/home/unitree/slam_example/README.md`
- `/home/unitree/EF/ef_ws/g1/slam_toggle.py`
- `/home/unitree/EF/topics.md`
- `/home/unitree/EF/ros_debug.md`

## What `keyDemo` Does

`keyDemo` is a Unitree SDK2 DDS client for the `slam_operate` API. It is not a ROS node and it does not use standard ROS services.

Run form from the example README:

```bash
cd /home/unitree/slam_example/build
./keyDemo eth0
```

The interface argument must be on the Unitree `192.168.123.x` subnet. In the current setup that interface is `eth0`.

## API IDs Used

`keyDemo` registers these `slam_operate` API IDs:

| API ID | Meaning in code |
|---:|---|
| `1801` | Start mapping |
| `1802` | End mapping and save map |
| `1804` | Start relocation / initialize pose from saved map |
| `1102` | Pose navigation |
| `1201` | Pause navigation |
| `1202` | Resume navigation |
| `1901` | Stop SLAM node |

The corresponding ROS/DDS request-response topics from `ros2 topic list` are:

```text
/api/slam_operate/request
/api/slam_operate/response
```

## Keyboard Controls

| Key | Action |
|---|---|
| `q` | Start mapping |
| `w` | End mapping and save map |
| `a` | Start relocation from the saved map |
| `s` | Add current SLAM pose to the in-memory task list |
| `d` | Execute the in-memory task list |
| `f` | Clear the in-memory task list |
| `z` | Pause navigation |
| `x` | Resume navigation |
| other | Stop task thread and stop SLAM |

## Map Save And Reload

The example uses one hard-coded map path:

```text
/home/unitree/test.pcd
```

Pressing `w` calls API `1802` with:

```json
{
  "data": {
    "address": "/home/unitree/test.pcd"
  }
}
```

That saves the map to disk.

Pressing `a` calls API `1804` with the current/default pose plus:

```json
{
  "data": {
    "address": "/home/unitree/test.pcd"
  }
}
```

That loads the same saved map and starts relocation against it.

Answer: yes, the same map can be accessed after `keyDemo` is relaunched, as long as the file still exists and `a` is used to relocate against `/home/unitree/test.pcd`.

Caveats:
- The saved map file persists.
- The `poseList` does not persist. It is only an in-memory `std::vector<poseDate>` in the running process.
- On relaunch, add navigation poses again with `s` after successful relocation.
- The destructor calls stop SLAM on process exit, so relaunch starts a new client/session.
- To use another map name, change the hard-coded path or use the Python wrapper described below.

## Navigation Flow

The demo does not compute paths itself.

Navigation flow:

1. `/slam_info` is received on DDS topic `rt/slam_info`.
2. The current pose is parsed from `data.currentPose`.
3. Pressing `s` stores that current pose in `poseList`.
4. Pressing `d` loops through `poseList`.
5. For each target pose, `keyDemo` calls API `1102` (`pose_nav`) with:

```json
{
  "data": {
    "targetPose": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "q_x": 0.0,
      "q_y": 0.0,
      "q_z": 0.0,
      "q_w": 1.0
    },
    "mode": 1
  }
}
```

6. The demo waits for `/slam_key_info` on DDS topic `rt/slam_key_info`.
7. When `/slam_key_info` reports `type == "task_result"`, it reads `data.is_arrived`.

If `is_arrived` is false, the demo prints a failure message. It does not inspect the local obstacle map or replan itself.

## How Obstacle Avoidance Is Applied

Obstacle avoidance is handled inside Unitree's internal SLAM/navigation stack, not in `keyDemo`.

`keyDemo` does not subscribe to:
- `/utlidar/range_info`
- `/collision_clouds`
- `/safe_clouds`
- `/warning_clouds`
- `/pre_collision_clouds`
- `/pre_safe_clouds`
- `/no_warning_clouds`
- `/grid_clouds`
- `/utlidar/cloud_livox_mid360`
- `/utlidar/cloud_deskewed`

It only sends high-level navigation goals through `slam_operate` and listens for status/result strings.

Practical interpretation:
- The internal planner decides whether a path is valid.
- Obstacle avoidance/rejection is surfaced to the demo indirectly through the `pose_nav` return code and `/slam_key_info`.
- A navigation failure may mean the planner rejected the goal, the robot could not reach it, localization was not valid, or an obstacle/map condition blocked the path.

## Did The Obstacle-Avoidance Topics Publish Data?

From the 2026-05-09 all-topic probe documented in `/home/unitree/EF/topics.md`:

| Topic | Type | Probe status | Data observed |
|---|---|---|---|
| `/utlidar/cloud_livox_mid360` | `sensor_msgs/msg/PointCloud2` | `LIVE` | Yes. Meaningful raw Livox cloud, sample width about `20256`, frame `livox_frame`. |
| `/ele_clouds` | `sensor_msgs/msg/PointCloud2` | `LIVE` | Technically yes, but sample was empty: `width=0`, frame `livox_frame`. |
| `/utlidar/range_info` | `geometry_msgs/msg/PointStamped` | `IDLE` | No sample during the 20 second probe. |
| `/utlidar/cloud_deskewed` | `sensor_msgs/msg/PointCloud2` | `IDLE` | No sample during the 20 second probe. |
| `/collision_clouds` | `sensor_msgs/msg/PointCloud2` | `IDLE` | No sample during the 20 second probe. |
| `/pre_collision_clouds` | `sensor_msgs/msg/PointCloud2` | `IDLE` | No sample during the 20 second probe. |
| `/safe_clouds` | `sensor_msgs/msg/PointCloud2` | `IDLE` | No sample during the 20 second probe. |
| `/pre_safe_clouds` | `sensor_msgs/msg/PointCloud2` | `IDLE` | No sample during the 20 second probe. |
| `/warning_clouds` | `sensor_msgs/msg/PointCloud2` | `IDLE` | No sample during the 20 second probe. |
| `/no_warning_clouds` | `sensor_msgs/msg/PointCloud2` | `IDLE` | No sample during the 20 second probe. |
| `/grid_clouds` | `sensor_msgs/msg/PointCloud2` | `IDLE` | No sample during the 20 second probe. |

Answer: only `/utlidar/cloud_livox_mid360` clearly published useful obstacle-relevant data in that probe. `/ele_clouds` published an empty cloud. The named obstacle/range products were advertised but did not publish samples during the 20 second observation window.

That does not prove those topics never publish. They may only publish when a specific SLAM mode is active, when obstacles are detected, when a planner/debug option is enabled, or when another internal component subscribes. It does mean `keyDemo` is not consuming them directly.

## Python Wrapper Alternative

The workspace has a more flexible wrapper:

```bash
cd /home/unitree/EF/ef_ws/g1
./slam_toggle.py save /home/unitree/map.pcd --iface eth0 --domain-id 0
./slam_toggle.py load /home/unitree/map.pcd --iface eth0 --domain-id 0
./slam_toggle.py nav --x 1.0 --y 0.0 --yaw 0.0 --iface eth0 --domain-id 0
```

Unlike `keyDemo`, this wrapper allows passing the map path from the command line.

