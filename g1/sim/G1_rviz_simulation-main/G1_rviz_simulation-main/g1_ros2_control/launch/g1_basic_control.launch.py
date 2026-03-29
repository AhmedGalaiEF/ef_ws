import os
import yaml
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch import LaunchDescription
from launch.actions import OpaqueFunction, SetLaunchConfiguration, DeclareLaunchArgument
from launch.substitutions import Command, PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_temp_config(config_path, package_name, kv_pairs):
    try:
        pkg_dir = get_package_share_directory(package_name)
    except PackageNotFoundError:
        # Allow launching directly from source tree without installing package.
        pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_path = os.path.join(pkg_dir, config_path)
    dst_path = os.path.join('/tmp', package_name, 'temp_controllers.yaml')
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    with open(src_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    for dotted_key, raw_val in kv_pairs:
        parts = dotted_key.split('.')
        if parts[1] != 'ros__parameters':
            parts.insert(1, 'ros__parameters')

        val = yaml.safe_load(raw_val)
        cur = cfg
        for k in parts[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[parts[-1]] = val

    with open(dst_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
        print(f"[launch] Wrote temporary controllers.yaml -> {dst_path}")

    return dst_path


def control_spawner(names):
    return Node(
        package='controller_manager',
        executable='spawner',
        arguments=[*names, '--param-file', LaunchConfiguration('controllers_yaml')],
        output='screen'
    )


def setup_controllers(context):
    robot_type_value = LaunchConfiguration('robot_type').perform(context)
    package_name = 'g1_ros2_control'
    try:
        pkg_dir = get_package_share_directory(package_name)
    except PackageNotFoundError:
        pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Prefer robot-specific config if present, otherwise use the repo default.
    candidate_rel_paths = [
        f'config/{robot_type_value}/controllers.yaml',
        'config/controllers.yaml',
    ]
    controllers_config_path = None
    for rel_path in candidate_rel_paths:
        if os.path.exists(os.path.join(pkg_dir, rel_path)):
            controllers_config_path = rel_path
            break
    if controllers_config_path is None:
        raise FileNotFoundError(
            f"No controllers config found in package '{package_name}'. "
            f"Tried: {candidate_rel_paths}"
        )

    temp_path = generate_temp_config(
        controllers_config_path,
        package_name,
        []  # no kv overrides
    )

    set_controllers_yaml = SetLaunchConfiguration('controllers_yaml', temp_path)
    # spawner = control_spawner(['state_estimator', 'standby_controller'])
    spawner = control_spawner(['state_estimator'])
    return [set_controllers_yaml, spawner]


def generate_launch_description():
    robot_type = LaunchConfiguration('robot_type')
    network_interface = LaunchConfiguration('network_interface')

    urdf_name = 'g1'  # or dynamically from robot_type if needed

    robot_description_command = Command([
        "xacro",
        " ",
        PathJoinSubstitution([
            FindPackageShare("unitree_description"),
            "urdf",
            urdf_name,
            "robot.xacro"
        ]),
        " ", "robot_type:=", robot_type,
        " ", "simulation:=false",
        " ", "network_interface:=", network_interface
    ])

    robot_description = {"robot_description": robot_description_command}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'publish_frequency': 500.0}],
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, LaunchConfiguration('controllers_yaml')],
        output="both",
        respawn=True,
    )

    controllers_opaque_func = OpaqueFunction(function=setup_controllers)

    return LaunchDescription([
        DeclareLaunchArgument('robot_type', default_value='g1'),
        DeclareLaunchArgument('network_interface', default_value='eth0'),
        controllers_opaque_func,
        control_node,
        node_robot_state_publisher,
    ])
