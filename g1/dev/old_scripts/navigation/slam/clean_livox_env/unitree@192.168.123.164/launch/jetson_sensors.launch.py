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
                parameters=[
                    {
                        "host_ip": "192.168.123.164",
                    }
                ],
            ),
            Node(
                package="ros_sensors_package",
                executable="rgbd_usb_publisher",
                name="rgbd_usb_publisher",
                output="screen",
                parameters=[
                    {
                        "width": 1280,
                        "height": 720,
                        "fps": 30,
                        "fx": 623.53829072479584,
                        "fy": 623.53829072479584,
                        "cx": 639.5,
                        "cy": 359.5,
                    }
                ],
            ),
        ]
    )
