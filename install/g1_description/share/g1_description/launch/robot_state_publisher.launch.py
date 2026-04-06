from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_type = LaunchConfiguration('robot_type')
    network_interface = LaunchConfiguration('network_interface')
    simulation = LaunchConfiguration('simulation')
    use_sim_time = LaunchConfiguration('use_sim_time')
    publish_frequency = LaunchConfiguration('publish_frequency')

    urdf_path = (
        Path(get_package_share_directory("g1_description"))
        / "urdf"
        / "g1_29dof_with_hand_rev_1_0_pkg.urdf"
    )

    robot_description = {
        "robot_description": urdf_path.read_text()
    }

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {
            'publish_frequency': publish_frequency,
            'use_sim_time': use_sim_time
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_type', default_value='g1'),
        DeclareLaunchArgument('network_interface', default_value='eth0'),
        DeclareLaunchArgument('simulation', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('publish_frequency', default_value='100.0'),
        node_robot_state_publisher,
    ])
