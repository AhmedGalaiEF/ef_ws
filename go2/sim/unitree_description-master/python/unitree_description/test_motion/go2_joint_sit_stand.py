#!/usr/bin/env python3
import math
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory


def load_joint_names():
    urdf_path = (
        get_package_share_directory("unitree_description")
        + "/model/go2/go2.urdf"
    )
    root = ET.parse(urdf_path).getroot()
    names = []
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        names.append(joint.get("name"))
    return names


class SitStand(Node):
    def __init__(self):
        super().__init__("go2_joint_sit_stand")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.joint_names = load_joint_names()
        self.start = self.get_clock().now()
        self.timer = self.create_timer(0.02, self.tick)
        self.get_logger().info("Publishing sit/stand motion to /joint_states")

    def set_joint(self, msg, name, value):
        if name in self.joint_names:
            i = self.joint_names.index(name)
            msg.position[i] = value

    def tick(self):
        t = (self.get_clock().now() - self.start).nanoseconds / 1e9
        phase = 0.5 * (1.0 + math.sin(0.8 * t))
        hip = 0.0
        thigh = 0.6 + 0.8 * phase
        calf = -1.1 - 1.0 * phase

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = [0.0] * len(self.joint_names)

        for leg in ("FL", "FR", "RL", "RR"):
            self.set_joint(msg, f"{leg}_hip_joint", hip)
            self.set_joint(msg, f"{leg}_thigh_joint", thigh)
            self.set_joint(msg, f"{leg}_calf_joint", calf)

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SitStand()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
