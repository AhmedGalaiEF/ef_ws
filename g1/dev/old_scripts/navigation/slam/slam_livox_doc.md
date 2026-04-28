# SLAM Architecture: Livox-SDK approach vs. Robot-internal SLAM

Two completely separate SLAM systems are available for this robot. They share no code, no topics, and no data with each other.

---

## 1. Livox-SDK SLAM (used by `slam_dual_window.py`)

### What it does

The entire pipeline runs on the **local computer** (the machine running `slam_dual_window.py`). The robot carries the sensor and receives motion commands; it computes nothing.

```
MID-360 LiDAR (physically on robot)
    │  raw UDP packets  →  192.168.123.222 (your host)
    ▼
livox2_python.py          (ctypes wrapper around liblivox_lidar_sdk_shared.so)
    │  aggregated numpy xyz array, one per ~0.35 s frame
    ▼
KISS-ICP                  (kiss_icp Python package, CPU-only)
    │  accumulated world-frame point cloud + 4×4 pose matrix
    ▼
_build_map_snapshot()     (background thread in slam_dual_window.py)
    │  480×480 occupancy grid + BGR canvas
    ▼
A* path planner           (slam_dual_window.py)
    ▼
slam_motion_worker.py     (subprocess, JSON over stdin/stdout pipe)
    │  loco_move() calls  →  DDS/CycloneDDS
    ▼
Unitree G-1 motors
```

### Key files

| File | Role |
|------|------|
| `livox2_python.py` | ctypes wrapper for Livox-SDK2. Receives raw UDP packets from the MID-360, buffers them, assembles full frames, calls `handle_points(xyz)` |
| `live_slam.py` | Subclasses `Livox2`. `handle_points` applies mount-flip/tilt correction then feeds each frame to KISS-ICP. Extracts the accumulated map and current 4×4 pose |
| `slam_dual_window.py` | Qt GUI. Consumes map + pose from `live_slam.py`, builds occupancy grid, runs A* path planning, sends move commands to the motion worker |
| `slam_motion_worker.py` | Subprocess that holds the Unitree SDK connection. Accepts JSON commands (`move`, `stop`, `boot`, …) over stdin, executes them with per-move cancellation |

### What "no ROS" means here

- No ROS2 nodes, topics, or message types are involved at any stage.
- The Livox MID-360 streams raw Cartesian point packets over **UDP** directly to the host (`192.168.123.222`). The Livox-SDK2 native shared library (`liblivox_lidar_sdk_shared.so`) handles the protocol; Python interacts with it via ctypes.
- KISS-ICP is a pure Python/C++ pip package — it has no ROS dependency.
- The occupancy grid, path planning, and map rendering are all plain numpy/OpenCV operations in Python.
- Motion commands go through the Unitree SDK2 (`unitree_sdk2py`) using DDS (CycloneDDS), which is also not ROS.

### What the robot contributes

Only the physical sensor (Livox MID-360) and its actuators. It does not run any SLAM software for this pipeline.

---

## 2. Robot-internal SLAM (Unitree built-in)

### What it does

The Unitree G-1 runs its own proprietary SLAM stack **internally on the robot's onboard computer**. It uses the robot's own sensor suite and publishes results over DDS topics on the same network. You interact with it by sending RPC calls and subscribing to DDS topics from the local computer via `unitree_sdk2py`.

### Key topics and service

| Identifier | Type | Description |
|------------|------|-------------|
| `slam_operate` service | RPC (unitree_sdk2py `Client`) | Start/stop mapping, navigate to pose, pause/resume |
| `rt/slam_info` | `std_msgs/String` (JSON payload) | Current robot pose (`currentPose.x/y/q_*`) published by the robot |
| `rt/slam_key_info` | `std_msgs/String` (JSON payload) | Same content, alternate topic |
| `rt/unitree/slam_mapping/odom` | `nav_msgs/Odometry` | Odometry from the internal SLAM process |
| `rt/odom` | `nav_msgs/Odometry` | General odometry |

### Key API calls (via `sdk_slam.py` → `sdk_client.py`)

```python
robot.start_slam("indoor")             # API 1801 – start the robot's mapping process
robot.stop_slam(save_path)             # API 1802 – end mapping and optionally save map
robot.init_pose(x, y, z, qx, qy, qz, qw, address)  # API 1804 – set initial pose on a saved map
robot._run_pose_nav(x, y, yaw)         # API 1102 – send a navigation goal to the robot's planner
robot.pause_nav()                       # API 1201
robot.resume_nav()                      # API 1202
robot.close_slam()                      # API 1901 – shut down SLAM without saving

robot.get_slam_pose()                   # read latest pose from rt/slam_info or rt/slam_key_info
robot.get_slam_info()                   # raw JSON string from rt/slam_info
```

### Architecture

```
Robot onboard computer
├── Internal SLAM process  (closed-source, runs on robot)
│       │  publishes pose + map via DDS
│       ▼
│   rt/slam_info  (DDS topic)
│   rt/slam_key_info
│   rt/unitree/slam_mapping/odom
│
└── slam_operate RPC service  (DDS service)

Local computer
├── sdk_slam.py  →  SlamInfoSubscriber, SlamOdomSubscriber, SlamOperateClient
└── sdk_client.py  →  start_slam(), stop_slam(), get_slam_pose(), _run_pose_nav()
```

The robot's own path planner handles obstacle avoidance and navigation internally once a goal pose is sent via `pose_nav`. The local computer only sends goals and reads pose feedback — it does not see raw point clouds or build maps.

---

## Side-by-side comparison

| Aspect | Livox-SDK (slam_dual_window) | Robot-internal SLAM |
|--------|------------------------------|---------------------|
| **Where SLAM runs** | Local computer | Robot onboard computer |
| **Sensor access** | Direct UDP from MID-360 to host | Robot's own internal sensor pipeline |
| **Map visible locally?** | Yes — full accumulated point cloud + occupancy grid rendered in the GUI | No — only a pose (x, y, yaw) comes back; the map stays on the robot |
| **Path planning** | Local A* on a numpy occupancy grid | Internal robot planner (closed-source) |
| **Navigation commands** | `loco_move()` velocity commands at ~100ms intervals | Single `pose_nav` goal call; robot drives itself |
| **Dynamic obstacles** | Visible in real-time (raw LiDAR frames shown in red on map) | Handled internally, not directly observable |
| **ROS required?** | No | No (uses DDS / unitree_sdk2py, not ROS2) |
| **Saved map format** | `.npz` (numpy compressed, local file) | File path on the **robot's** filesystem (passed to `end_mapping`) |
| **Freeze/resume mapping** | `SLAM_SESSION.set_mapping(False/True)` — stops feeding frames to KISS-ICP | N/A — robot controls its own mapping state |
| **Requires Livox-SDK2 installed locally?** | Yes (`liblivox_lidar_sdk_shared.so`) | No |

---

## Why two systems?

The Livox-SDK approach gives full local visibility into the map and complete control over path planning, making it easy to add custom logic (occupancy grids, A*, dynamic obstacle overlays, UI interaction). The robot-internal SLAM is simpler to invoke (one RPC call) but is a black box — you send a goal and the robot decides how to get there, without exposing intermediate map data to the local computer.
