from launch import LaunchDescription
from launch_ros.actions import Node
from pathlib import Path

def generate_launch_description():

    repo_root = Path(__file__).resolve().parent.parent
    urdf_file = repo_root / 'model' / 'go2' / 'go2.urdf'
    urdf_text = urdf_file.read_text()
    urdf_text = urdf_text.replace(
        'package://unitree_description/',
        f'file://{repo_root.as_posix()}/',
    )

    return LaunchDescription([

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': urdf_text,
            }]
        ),

        # Joint State Publisher GUI (for manual control)
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui'
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', str(repo_root / 'rviz' / 'go2.rviz')],
        ),
    ])
