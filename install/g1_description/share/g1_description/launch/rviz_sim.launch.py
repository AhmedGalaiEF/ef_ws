from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_type = LaunchConfiguration('robot_type')
    network_interface = LaunchConfiguration('network_interface')
    use_sim_time = LaunchConfiguration('use_sim_time')
    publish_frequency = LaunchConfiguration('publish_frequency')
    use_gui = LaunchConfiguration('use_gui')
    publish_joint_states = LaunchConfiguration('publish_joint_states')
    use_zero_joint_state_publisher = LaunchConfiguration('use_zero_joint_state_publisher')
    rviz_config = LaunchConfiguration('rviz_config')
    use_demo_motion = LaunchConfiguration('use_demo_motion')
    demo_mode = LaunchConfiguration('demo_mode')

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

    joint_state_publishers = GroupAction(
        condition=IfCondition(PythonExpression(["'", use_demo_motion, "' == 'false' and '", publish_joint_states, "' == 'true'"])),
        actions=[
            Node(
                package='joint_state_publisher',
                executable='joint_state_publisher',
                output='screen',
                parameters=[robot_description, {'use_sim_time': use_sim_time}],
                condition=UnlessCondition(use_gui),
            ),
            Node(
                package='joint_state_publisher_gui',
                executable='joint_state_publisher_gui',
                output='screen',
                parameters=[robot_description, {'use_sim_time': use_sim_time}],
                condition=IfCondition(use_gui),
            ),
        ],
    )

    zero_joint_state_publisher = ExecuteProcess(
        cmd=[
            FindExecutable(name='python3'),
            PathJoinSubstitution([
                FindPackageShare('g1_description'),
                'scripts',
                'zero_joint_state_publisher.py',
            ]),
        ],
        output='screen',
        condition=IfCondition(PythonExpression(["'", use_demo_motion, "' == 'false' and '", use_zero_joint_state_publisher, "' == 'true'"])),
    )

    demo_joint_motion = Node(
        package='g1_description',
        executable='demo_joint_motion',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'mode': demo_mode}],
        condition=IfCondition(use_demo_motion),
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
        DeclareLaunchArgument('publish_joint_states', default_value='false'),
        DeclareLaunchArgument('use_zero_joint_state_publisher', default_value='true'),
        DeclareLaunchArgument('use_demo_motion', default_value='false'),
        DeclareLaunchArgument('demo_mode', default_value='pose'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution([
                FindPackageShare('g1_description'),
                'rviz',
                'g1.rviz'
            ])
        ),
        node_robot_state_publisher,
        joint_state_publishers,
        zero_joint_state_publisher,
        demo_joint_motion,
        rviz,
    ])
