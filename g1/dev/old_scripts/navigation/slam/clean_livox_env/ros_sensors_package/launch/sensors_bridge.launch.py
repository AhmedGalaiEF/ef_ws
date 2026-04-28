from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="ros_sensors_package",
                executable="livox_points_publisher",
                name="livox_points_publisher",
                output="screen",
            ),
            Node(
                package="ros_sensors_package",
                executable="rgbd_usb_publisher",
                name="rgbd_usb_publisher",
                output="screen",
            ),
        ]
    )
