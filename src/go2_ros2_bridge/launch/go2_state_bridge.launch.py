from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    interface = LaunchConfiguration("interface")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("use_rviz")

    robot_description = ParameterValue(
        Command(
            [
                "cat",
                " ",
                PathJoinSubstitution(
                    [FindPackageShare("unitree_description"), "model", "go2", "go2.urdf"]
                ),
            ]
        ),
        value_type=str,
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description, "use_sim_time": use_sim_time}
        ],
    )

    bridge = Node(
        package="go2_ros2_bridge",
        executable="go2_state_bridge",
        output="screen",
        parameters=[{"interface": interface, "publish_rate_hz": 50.0}],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=[
            "-d",
            PathJoinSubstitution(
                [FindPackageShare("unitree_description"), "rviz", "go2.rviz"]
            ),
        ],
        condition=IfCondition(use_rviz),
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("interface", default_value=""),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            robot_state_publisher,
            bridge,
            rviz,
        ]
    )
