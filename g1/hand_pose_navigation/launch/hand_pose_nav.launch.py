"""
ROS 2 launch file for the hand_pose_navigation pipeline.

Launch:
    ros2 launch hand_pose_navigation hand_pose_nav.launch.py

Optional overrides:
    ros2 launch hand_pose_navigation hand_pose_nav.launch.py arm:=left detection_method:=color

All parameters are forwarded to hand_pose_nav_node.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # ── Declare all tunable launch arguments ──────────────────────────
    args = [
        DeclareLaunchArgument("arm", default_value="right",
                              description="Which arm to control: left | right"),
        DeclareLaunchArgument("detection_method", default_value="aruco",
                              description="Detection method: aruco | color | center"),
        DeclareLaunchArgument("aruco_id", default_value="0",
                              description="ArUco marker ID to track"),
        DeclareLaunchArgument("marker_size_m", default_value="0.05",
                              description="Physical ArUco marker side length (metres)"),
        DeclareLaunchArgument("standoff_m", default_value="0.08",
                              description="Pre-grasp standoff distance (metres)"),
        DeclareLaunchArgument("rate_hz", default_value="10.0",
                              description="Tracking loop rate (Hz)"),
        DeclareLaunchArgument("timeout_s", default_value="30.0",
                              description="Max tracking duration; 0 = unlimited"),
        DeclareLaunchArgument("ik_solver", default_value="dls",
                              description="IK backend: dls | scipy | pin"),
        DeclareLaunchArgument("iface", default_value="eth0",
                              description="Network interface for robot SDK DDS"),
        DeclareLaunchArgument("domain_id", default_value="0",
                              description="DDS domain ID"),
    ]

    # ── Main orchestrator node ────────────────────────────────────────
    nav_node = Node(
        package="hand_pose_navigation",
        executable="hand_pose_nav_node",
        name="hand_pose_nav",
        output="screen",
        parameters=[{
            "arm": LaunchConfiguration("arm"),
            "detection_method": LaunchConfiguration("detection_method"),
            "aruco_id": LaunchConfiguration("aruco_id"),
            "marker_size_m": LaunchConfiguration("marker_size_m"),
            "standoff_m": LaunchConfiguration("standoff_m"),
            "rate_hz": LaunchConfiguration("rate_hz"),
            "timeout_s": LaunchConfiguration("timeout_s"),
            "ik_solver": LaunchConfiguration("ik_solver"),
            "iface": LaunchConfiguration("iface"),
            "domain_id": LaunchConfiguration("domain_id"),
        }],
    )

    return LaunchDescription(args + [nav_node])
