from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from pathlib import Path
import os


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_gui = LaunchConfiguration('use_gui')
    rviz_config = LaunchConfiguration('rviz_config')
    use_state_publisher = LaunchConfiguration('use_state_publisher')
    use_joint_state_publisher = LaunchConfiguration('use_joint_state_publisher')

    repo_root = Path(__file__).resolve().parent.parent
    urdf_path = repo_root / 'model' / 'go2' / 'go2.urdf'
    urdf_text = urdf_path.read_text()
    urdf_text = urdf_text.replace(
        'package://unitree_description/',
        f'file://{repo_root.as_posix()}/',
    )

    robot_description = {
        'robot_description': ParameterValue(urdf_text, value_type=str)
    }

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
        condition=IfCondition(use_state_publisher),
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
        condition=IfCondition(PythonExpression([
            "'", use_state_publisher, "' == 'true' and '", use_joint_state_publisher,
            "' == 'true' and '", use_gui, "' == 'false'"
        ])),
    )

    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
        condition=IfCondition(PythonExpression([
            "'", use_state_publisher, "' == 'true' and '", use_joint_state_publisher,
            "' == 'true' and '", use_gui, "' == 'true'"
        ])),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
    )

    cmd_vel_pub = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '-r', '1',
            '/cmd_vel', 'geometry_msgs/msg/Twist',
            '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}',
        ],
        name='cmd_vel_pub',
        output='screen',
        additional_env={'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp'},
        condition=IfCondition(use_joint_state_publisher),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('use_gui', default_value='false'),
        DeclareLaunchArgument('use_state_publisher', default_value='true'),
        DeclareLaunchArgument('use_joint_state_publisher', default_value='true'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution([
                os.fspath(repo_root),
                'rviz',
                'go2.rviz',
            ]),
        ),
        node_robot_state_publisher,
        joint_state_publisher,
        joint_state_publisher_gui,
        rviz,
        cmd_vel_pub,
    ])
