from setuptools import find_packages, setup

package_name = "g1_approval_ros"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/g1_approval_demo.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="User",
    maintainer_email="user@example.com",
    description="Sample ROS 2 Python package for mediated robot command approval and audit logging.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "command_gateway = g1_approval_ros.command_gateway:main",
            "approval_console = g1_approval_ros.approval_console:main",
            "request_demo = g1_approval_ros.request_demo:main",
        ],
    },
)
