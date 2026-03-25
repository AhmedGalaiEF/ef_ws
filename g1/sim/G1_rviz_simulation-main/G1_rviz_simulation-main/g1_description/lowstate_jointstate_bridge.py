#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from unitree_hg.msg import LowState 
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class G1JointStateBridge(Node):
    def __init__(self):
        super().__init__('g1_joint_state_bridge')

        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        self.sub = self.create_subscription(LowState, 'lf/lowstate', self.callback, 10)
        self.pub = self.create_publisher(JointState, '/joint_states', qos)

        self.joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint"
        ]

    def callback(self, msg: LowState):
        # PRINT FULL LOWSTATE FOR DEBUGGING
        # print("=== /lf/lowstate DEBUG ===")
        # print(f"Num motors: {len(msg.motor_state)}")
        # print("Motor states:")
        # for i, motor in enumerate(msg.motor_state):
        #     print(f"  Motor {i}: q={motor.q:.3f}, dq={getattr(motor, 'dq', 'N/A')}, tau={getattr(motor, 'tau', 'N/A')}")
        # print(f"Joint names count: {len(self.joint_names)}")
        # print("========================")

        expected = len(self.joint_names)
        actual = len(msg.motor_state)
        if actual < expected:
            self.get_logger().warn(
                f"lowstate motor_state too short: {actual} < {expected}. "
                "Filling missing joints with zeros."
            )
        positions = [m.q for m in msg.motor_state[:expected]]
        if len(positions) < expected:
            positions.extend([0.0] * (expected - len(positions)))

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names
        js.position = positions

        self.pub.publish(js)

def main(args=None):
    rclpy.init(args=args)
    node = G1JointStateBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
