from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_gui = LaunchConfiguration('use_gui')
    publish_joint_states = LaunchConfiguration('publish_joint_states')
    use_zero_joint_state_publisher = LaunchConfiguration('use_zero_joint_state_publisher')
    rviz_config = LaunchConfiguration('rviz_config')

    robot_description_command = Command([
        FindExecutable(name='cat'),
        " ",
        PathJoinSubstitution([
            FindPackageShare('unitree_description'),
            'model',
            'go2',
            'go2.urdf',
        ]),
    ])

    robot_description = {
        'robot_description': ParameterValue(
            robot_description_command,
            value_type=str,
        )
    }

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
        condition=IfCondition(publish_joint_states),
    )

    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
        condition=IfCondition(PythonExpression(["'", publish_joint_states, "' == 'true' and '", use_gui, "' == 'true'"])),
    )

    zero_joint_state_publisher = ExecuteProcess(
        cmd=[
            FindExecutable(name='python3'),
            PathJoinSubstitution([
                FindPackageShare('unitree_description'),
                'scripts',
                'zero_joint_state_publisher.py',
            ]),
        ],
        output='screen',
        condition=IfCondition(use_zero_joint_state_publisher),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('use_gui', default_value='false'),
        DeclareLaunchArgument('publish_joint_states', default_value='false'),
        DeclareLaunchArgument('use_zero_joint_state_publisher', default_value='true'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution([
                FindPackageShare('unitree_description'),
                'rviz',
                'go2.rviz',
            ]),
        ),
        node_robot_state_publisher,
        joint_state_publisher,
        joint_state_publisher_gui,
        zero_joint_state_publisher,
        rviz,
    ])
