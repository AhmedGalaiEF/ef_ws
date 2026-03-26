from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    robot_type = LaunchConfiguration('robot_type')
    network_interface = LaunchConfiguration('network_interface')
    use_sim_time = LaunchConfiguration('use_sim_time')
    publish_frequency = LaunchConfiguration('publish_frequency')
    use_gui = LaunchConfiguration('use_gui')
    rviz_config = LaunchConfiguration('rviz_config')

    urdf_name = 'g1'

    robot_description_command = Command([
        FindExecutable(name='xacro'),
        " ",
        "/home/melodse/ef_ws/src/unitree_description/urdf/g1/g1.xacro",
        " ", "robot_type:=", robot_type,
        " ", "simulation:=", "true",
        " ", "network_interface:=", network_interface
    ])

    robot_description = {
        "robot_description": ParameterValue(
            robot_description_command,
            value_type=str
        )
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

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
        condition=UnlessCondition(use_gui),
    )

    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
        condition=IfCondition(use_gui),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_type', default_value='g1'),
        DeclareLaunchArgument('network_interface', default_value='eth0'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('publish_frequency', default_value='100.0'),
        DeclareLaunchArgument('use_gui', default_value='false'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution([
                FindPackageShare('g1_description'),
                'rviz',
                'g1.rviz'
            ])
        ),
        node_robot_state_publisher,
        joint_state_publisher,
        joint_state_publisher_gui,
        rviz,
    ])
