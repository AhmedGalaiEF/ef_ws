# Current Diff

This documents the current difference between `minimal/sdk_wrapper_v3.py:G1` and `sdk_client.py:Robot`.

## Summary

`G1` is still a much smaller and more ad hoc wrapper than `Robot`.

- `Robot` is a broad end-user SDK with sensor lifecycle management, richer motion and navigation helpers, chat and vision integration, full Dex3 helper abstractions, teaching and replay, and more robust SLAM and IK tooling.
- `G1` is a thin wrapper that covers the methods from `minimal/sdk_wrapper.py` plus a few extra convenience helpers, but several of those methods are implemented in a simpler or approximate way.

## What G1 Currently Covers

`G1` currently implements these main areas:

- FSM and locomotion:
  `zero_torque_mode`, `damp_mode`, `prepare_mode`, `walk_mode`, `run_mode`, `loco_move`, `loco_stop`
- Services and dev mode:
  `get_service`, `toggle_service`, `toggle_dev_mode`
- State and sensing:
  `get_state`, `get_gait`, `toggle_gait`, `get_lowstate`, `get_odom`, `get_imus`, `get_battery`, `get_rgbd`, `get_point_cloud`, `get_slam_info`, `get_mic_input`
- Audio and lighting:
  `say`, `set_headlight`
- Arms and hands:
  `release_arms`, `engage_arms`, `move_ll_joint`, `open_dex3_hand`, `close_dex3_hand`, `get_dex3_hand_sensors`, `open_inspire_hand`, `close_inspire_hand`, `get_inspire_hand_sensors`
- SLAM and navigation:
  `start_mapping`, `stop_mapping`, `relocate`, `add_map_pose`, `navigate`
- Gestures:
  `clap`, `face_wave`, `high_wave`, `shake_hand`, `hug`, `left_kiss`
- Extra helpers:
  `ik_move_ee`, `extend_arm`

## Methods Robot Has That G1 Does Not

`Robot` exposes a much larger surface that `G1` does not currently match.

- Sensor lifecycle and readiness:
  `start_sensors`, `wait_for_sport_state`, `wait_for_low_state`, `get_sensor_timestamps`, `sensors_stale`
- Additional body state helpers:
  `get_body_height`, `get_position`, `get_velocity`, `get_yaw`, `is_moving`, `get_robot_state`
- Richer lowstate and joint access:
  `get_low_state_snapshot`, `get_joint_positions`, `get_joint_velocities`, `get_joint_torques`, `get_joint_position`, `get_joint_states`
- Richer hand API:
  `hand_open`, `hand_close`, `release_fingers`, `stop_release_fingers`, `unrelease_fingers`, `zero_torque_fingers`, `hand_pose`, `hand_move_finger`, `get_hand_state_snapshot`, `get_tactile_pressures`
- Richer camera and vision API:
  `get_camera_image_jpeg`, `get_detection_image_jpeg`, `get_rgb_jpeg`, `get_camera_frame_bgr`, `get_camera_frame_rgb`, `detect`
- Chat and Ollama helpers:
  `chat`, `stop_chat`, plus the internal Ollama model and server helpers
- Additional locomotion and control helpers:
  `move_for`, `stop_moving`, `stop`, USB controller helpers, several `fsm_*` aliases
- Richer SLAM and path helpers:
  `set_path_point`, `get_path_points`, `clear_path_points`, `navigate_path`, `get_slam_key`, `get_slam_pose`, `get_slam_pose_status`, `get_slam_odom_pose`, `debug_api`
- Boot and recovery helpers:
  `hanged_boot`, `hanging_boot`
- Teaching and replay:
  `teach`, `repeat`
- Extra arm and motion helpers:
  `move_upper_body_joint`, `extend_arm_forward`, `retract_arm_forward`

## Shared Methods With Behavioral Differences

Some methods exist in both wrappers, but the current implementation differs noticeably.

### Dev Mode

- `Robot` uses motion-switcher style dev-mode helpers.
- `G1.toggle_dev_mode()` now always uses service toggling through `ai_sport`, because that matched the observed robot behavior better.

### Gait

- `Robot` mainly reads gait from state and relies on the SDK methods directly.
- `G1` currently uses `_gait_override` because the observed DDS state was not reliably reflecting normal vs continuous gait on the tested setup.

### Sensors

- `Robot` has explicit subscriber startup and health tracking.
- `G1` creates a smaller set of subscribers directly in `__init__` and reads latest values opportunistically.

### Point Cloud

- `Robot` has broader lidar and cloud handling across multiple topics, timestamps, and helpers.
- `G1.get_point_cloud()` is a thinner latest-message decoder over a few candidate SLAM cloud topics.

### RGBD

- `Robot` has dedicated RGB, JPEG, frame, and RGBD helpers.
- `G1.get_rgbd()` is a direct ZeroMQ payload fetcher aimed at the `5555` stream path.

### Microphone Input

- `Robot.get_mic()` supports richer collection modes, including CLI and ROS 2 paths.
- `G1.get_mic_input()` is a smaller DDS-based collector over `rt/audio_msg`, now with optional `duration_s`.

### SLAM Navigation

- `Robot.navigate_path()` includes additional validation, tracing, status reasoning, and path management.
- `G1.navigate()` currently queues points and directly calls the SLAM `pose_nav` RPC.

### Hand State

- `Robot` has a fuller Dex3 hand abstraction and richer state helpers.
- `G1.get_dex3_hand_sensors()` reads the same DDS hand state family, but only produces a compact snapshot.

### Inspire Hands

- `Robot` does not currently represent Inspire hands in the same minimal inline way.
- `G1` uses a direct Modbus write path similar to `inspire_sdk.py`, but still does not provide a true Inspire sensor-read implementation.

## Important Quality Gaps In G1

The biggest remaining gaps are:

- `ik_move_ee` is still approximate:
  it is not a true DLS IK solve and does not match the full kinematic behavior of the dedicated arm control tools.
- `extend_arm` depends on that same approximate EE-to-joint mapping.
- `get_gait` and `toggle_gait` still rely partly on cached state because the live gait topic did not behave reliably.
- `navigate` is functional but less defensive and less observable than `Robot.navigate_path`.
- `teach` and `repeat` are still not implemented.

## G1-Only Additions

There are also a few helpers currently present on `G1` that are not exposed on `Robot` under the same names or in the same shape.

- `toggle_dev_mode` as a direct service toggle wrapper
- `ik_move_ee`
- `extend_arm`
- the exact `minimal/sdk_wrapper.py`-style method naming and return shapes

## Practical Conclusion

At the moment:

- `Robot` is the more complete and better-engineered general SDK wrapper.
- `G1` is closer to a compatibility wrapper for the original minimal notebook-style API.
- `G1` now covers a useful subset of the workflows, but it still does not replace `Robot` feature-for-feature or behavior-for-behavior.
