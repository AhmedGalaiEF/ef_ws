G1 Sport Mode ROS Driver
A ROS2 package that enables control of the Unitree G1 humanoid robot through standard ROS cmd_vel commands, replacing the need for a remote controller.

Overview
This package provides a bridge between ROS2 navigation commands and the Unitree G1's sport mode API. It subscribes to geometry_msgs/Twist messages on the /cmd_vel topic and translates them into Unitree API requests for robot locomotion control.

Features
Standard ROS Interface: Uses the widely-adopted cmd_vel topic for robot control
Sport Mode Integration: Directly interfaces with Unitree G1's sport mode API
Real-time Control: Low-latency translation of movement commands
Full Mobility: Supports forward/backward, strafing, and rotation movements
Navigation Compatible: Works with ROS2 navigation stacks and teleop tools
Node Details
g1_sport_mode_driver
Subscribed Topics: - /cmd_vel (geometry_msgs/Twist) - Velocity commands for robot movement

Published Topics: - /sport_mode/request (unitree_api/Request) - Unitree API requests for sport mode control

Command Mapping: - linear.x → vx (forward/backward velocity) - linear.y → vy (left/right strafe velocity)
- angular.z → vyaw (yaw rotation velocity)

g1_sport_ros.py
A Python ROS2 node that directly subscribes to cmd_vel and controls the G1 robot in sport mode.

⚠️ IMPORTANT: This node must run in a separate ROS domain (e.g., domain 1) to avoid DDS conflicts with the Unitree SDK.

Subscribed Topics: - /cmd_vel (geometry_msgs/Twist) - Maps to linear and angular velocities for the robot

Features: - Interface Parameter: Accepts interface as a parameter for the network interface (default eth0) - Throttled Logging: Logs every 100 messages for optimal performance - Safety: Ensures robot stops on exit or in case of errors - Domain Separation: Uses separate ROS domain to prevent DDS conflicts

Usage:

# IMPORTANT: Unset CYCLONEDDS_HOME (set by G1 manufacturer) before running
unset CYCLONEDDS_HOME

# Run the node in ROS domain 1 (REQUIRED)
ROS_DOMAIN_ID=1 ros2 run g1_sport_mode_ros g1_sport_ros.py --ros-args -p interface:=eth0
Sending commands (from same domain):

# In another terminal, use the same domain
ROS_DOMAIN_ID=1 ros2 run teleop_twist_keyboard teleop_twist_keyboard
This node is designed to provide real-time control of the G1 robot, translating geometry_msgs/Twist commands into movement using the Unitree SDK.

Dependencies
rclcpp - ROS2 C++ client library
rclpy - ROS2 Python client library
geometry_msgs - Standard ROS geometry messages
unitree_api - Unitree robot API messages
unitree_sdk2_python - Unitree SDK2 Python bindings
Building
# From your ROS2 workspace root
colcon build --packages-select g1_sport_mode_ros

# Source the workspace
source install/setup.bash
Usage
1. Start the driver node
ros2 run g1_sport_mode_ros g1_sport_mode_driver
2. Control the robot
Option A: Keyboard teleop

ros2 run teleop_twist_keyboard teleop_twist_keyboard
Option B: Custom velocity commands

ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
Option C: Navigation stack Use with any ROS2 navigation stack that publishes to /cmd_vel

API Details
The node uses Unitree API ID 1004 for sport mode control with JSON-formatted parameters:

{
    "vx": 0.5,    // Forward velocity (m/s)
    "vy": 0.0,    // Strafe velocity (m/s)
    "vyaw": 0.0   // Yaw velocity (rad/s)
}
Domain Configuration
Critical Requirement: The g1_sport_ros.py node must run in a separate ROS domain to avoid DDS conflicts.

Unitree SDK: Uses DDS domain 0 to communicate with the G1 robot
ROS2 Node: Must use domain 1 (or any domain != 0) to avoid conflicts
All ROS2 tools: Must use the same domain as the node to communicate
Example workflow:

# Terminal 1: Start the G1 control node (unset CYCLONEDDS_HOME first!)
unset CYCLONEDDS_HOME
ROS_DOMAIN_ID=1 ros2 run g1_sport_mode_ros g1_sport_ros.py --ros-args -p interface:=eth0

# Terminal 2: Control with keyboard
ROS_DOMAIN_ID=1 ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Terminal 3: Check topics
ROS_DOMAIN_ID=1 ros2 topic list
Safety Notes
Ensure the robot is in a safe environment before testing
Start with small velocity values for initial testing
The robot should be in sport mode for this driver to work effectively
Always have a way to emergency stop the robot
Troubleshooting
DDS_RETCODE_BAD_PARAMETER during CyclonDDS topic initialization: - Cause: Conflicting CyclonDDS installations due to CYCLONEDDS_HOME environment variable set by G1 manufacturer - Solution: Unset the CYCLONEDDS_HOME variable before running the node - Command: unset CYCLONEDDS_HOME && ROS_DOMAIN_ID=1 ros2 run g1_sport_mode_ros g1_sport_ros.py --ros-args -p interface:=eth0

ChannelFactory create domain error: - Solution: Run the node with ROS_DOMAIN_ID=1 (or any domain != 0) - Cause: DDS domain conflict between ROS2 and Unitree SDK - Command: ROS_DOMAIN_ID=1 ros2 run g1_sport_mode_ros g1_sport_ros.py

No response from robot: - Verify the robot is in sport mode - Check that unitree_api messages are being published correctly - Ensure network interface parameter is correct (default: eth0)

Build errors: - Ensure all dependencies are installed - Verify unitree_api package is built and sourced - Run rosdep install --from-paths src --ignore-src -r -y

Connection issues: - Check network connectivity to the robot - Verify Unitree API is accessible

Integration Examples
This driver enables the G1 to work seamlessly with:

ROS2 Navigation Stack: For autonomous navigation
MoveIt2: For coordinated motion planning
Behavior Trees: For complex behavior orchestration
SLAM: For simultaneous localization and mapping while moving
Custom Controllers: Any application that publishes Twist messages
License
TODO: Add appropriate license information

Contributing
Contributions are welcome! Please submit pull requests or issues through the appropriate channels.
