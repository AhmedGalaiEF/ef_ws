from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="g1_approval_ros",
                executable="command_gateway",
                name="command_gateway",
                output="screen",
            ),
            Node(
                package="g1_approval_ros",
                executable="approval_console",
                name="approval_console",
                output="screen",
            ),
        ]
    )
