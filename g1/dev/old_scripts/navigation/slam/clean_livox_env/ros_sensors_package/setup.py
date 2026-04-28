from glob import glob
import os

from setuptools import find_packages, setup


package_name = "ros_sensors_package"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name), ["README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ag",
    maintainer_email="ag@example.com",
    description="ROS 2 DDS bridge for Livox lidar and RGBD streams used in this workspace.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "livox_points_publisher = ros_sensors_package.livox_points_publisher:main",
            "rgbd_zmq_publisher = ros_sensors_package.rgbd_zmq_publisher:main",
            "rgbd_usb_publisher = ros_sensors_package.rgbd_usb_publisher:main",
        ],
    },
)
