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

# This node publishes a JointState message to /joint_states with a leg kick motion.
class LegKick(Node):
    def __init__(self):
        super().__init__("rviz_joint_leg_kick")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.joint_names = load_joint_names()
        self.start = self.get_clock().now()
        self.timer = self.create_timer(0.02, self.tick)
        self.get_logger().info("Publishing leg kick motion to /joint_states")

# set_joint is a helper function that sets the position of a joint in the JointState message if the joint name exists in the list of joint names.
    def set_joint(self, msg, name, value):
        if name in self.joint_names:
            i = self.joint_names.index(name)
            msg.position[i] = value

# This function is called every 20ms by the timer. It computes the current time, creates a JointState message, and fills in the positions of the animated joints using sine waves.
    def tick(self):
        t = (self.get_clock().now() - self.start).nanoseconds / 1e9
        lift = max(0.0, math.sin(1.4 * t))

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = [0.0] * len(self.joint_names)

        # Left leg kick forward
        self.set_joint(msg, "left_hip_pitch_joint", -0.15 + 0.35 * lift)
        self.set_joint(msg, "left_knee_joint", 0.2 + 0.45 * lift)
        self.set_joint(msg, "left_ankle_pitch_joint", -0.15 - 0.25 * lift)

        # Right leg supports
        self.set_joint(msg, "right_hip_pitch_joint", -0.1)
        self.set_joint(msg, "right_knee_joint", 0.2)
        self.set_joint(msg, "right_ankle_pitch_joint", -0.1)

        self.pub.publish(msg)

# def main() initializes the ROS2 node, creates an instance of the LegKick class, and starts spinning the node to process callbacks until interrupted.
def main():
    rclpy.init()
    node = LegKick()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

# calls the main function when the script is executed
if __name__ == "__main__":
    main()
