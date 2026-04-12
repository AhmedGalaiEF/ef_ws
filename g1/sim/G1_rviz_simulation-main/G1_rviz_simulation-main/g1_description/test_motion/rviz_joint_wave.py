#!/usr/bin/env python3
import math
import xml.etree.ElementTree as ET

# importing ros2 package utilities to find the URDF file
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory


# This node publishes a JointState message to /joint_states with a few joints
def load_joint_names():
    urdf_path = (
        get_package_share_directory("g1_description")
        + "/urdf/g1_29dof_with_hand_rev_1_0_pkg.urdf"
    )
    root = ET.parse(urdf_path).getroot()
    names = []
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        names.append(joint.get("name"))
    return names

# This node publishes a JointState message to /joint_states with a few joints
class JointWave(Node):
    def __init__(self):
        super().__init__("rviz_joint_wave")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.joint_names = load_joint_names()
        self.start = self.get_clock().now()
        self.timer = self.create_timer(0.02, self.tick)  # 50 Hz update

        # Choose a few joints to animate; others stay at zero.
        self.animated = [
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_elbow_joint",
            "right_elbow_joint",
        ]

        self.get_logger().info(
            f"Publishing {len(self.joint_names)} joints to /joint_states"
        )

# This function is called every 20ms by the timer. It computes the current time, creates a JointState message, and fills in the positions of the animated joints using sine waves.
    def tick(self):
        t = (self.get_clock().now() - self.start).nanoseconds / 1e9
        # Joint angles are in radians.
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = [0.0] * len(self.joint_names)

        for name in self.animated:
            if name in self.joint_names:
                i = self.joint_names.index(name)
                phase = 0.0
                amp = 0.6
                if "knee" in name:
                    phase = math.pi / 2.0
                    amp = 0.8
                msg.position[i] = amp * math.sin(1.5 * t + phase)

        self.pub.publish(msg)

# def main() initializes the ROS2 node, creates an instance of the JointWave class, and starts
def main():
    rclpy.init()
    node = JointWave()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

# calls the main function when the script is executed
if __name__ == "__main__":
    main()
