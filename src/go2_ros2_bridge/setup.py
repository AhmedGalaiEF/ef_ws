from setuptools import find_packages, setup

package_name = "go2_ros2_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/go2_state_bridge.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="melodse",
    maintainer_email="melodse@local",
    description="Minimal ROS2 bridge for Unitree Go2 joint states",
    license="MIT",
    entry_points={
        "console_scripts": [
            "go2_state_bridge = go2_ros2_bridge.go2_state_bridge:main",
        ],
    },
)
