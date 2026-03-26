#!/usr/bin/env python3

from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

try:
    from unitree_hg.msg import LowState
except ImportError as exc:
    raise ImportError(
        "Failed to import 'unitree_hg.msg.LowState'. "
        "Source your Unitree ROS2 workspace first, e.g.:\n"
        "  source /home/ag/ros2_humble/install/setup.bash\n"
        "  source /home/ag/academy/academy_content/docs/repos/unitree_ros2/cyclonedds_ws/install/setup.bash"
    ) from exc


class G1JointStateBridge(Node):
    def __init__(self):
        super().__init__('g1_joint_state_bridge')

        default_joint_names = [
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

        self.declare_parameter("input_topic", "lf/lowstate")
        self.declare_parameter("output_topic", "/joint_states")
        self.declare_parameter("qos_depth", 10)
        self.declare_parameter("qos_reliability", "best_effort")
        self.declare_parameter("warn_period_sec", 5.0)
        self.declare_parameter("publish_velocity", True)
        self.declare_parameter("publish_effort", False)
        self.declare_parameter("joint_names", default_joint_names)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        qos_depth = int(self.get_parameter("qos_depth").value)
        reliability_str = str(self.get_parameter("qos_reliability").value).strip().lower()
        self.warn_period_sec = float(self.get_parameter("warn_period_sec").value)
        self.publish_velocity = bool(self.get_parameter("publish_velocity").value)
        self.publish_effort = bool(self.get_parameter("publish_effort").value)
        self.joint_names: List[str] = list(self.get_parameter("joint_names").value)

        if reliability_str == "reliable":
            reliability = QoSReliabilityPolicy.RELIABLE
        else:
            reliability = QoSReliabilityPolicy.BEST_EFFORT

        qos = QoSProfile(depth=qos_depth, reliability=reliability)

        self.sub = self.create_subscription(LowState, input_topic, self.callback, qos)
        self.pub = self.create_publisher(JointState, output_topic, qos)

        self._last_short_warn_time = 0.0
        self._last_long_warn_time = 0.0

        self.get_logger().info(
            f"Bridge ready: {input_topic} -> {output_topic}, "
            f"joints={len(self.joint_names)}, qos_depth={qos_depth}, reliability={reliability_str}"
        )

    def _warn_throttled(self, key: str, text: str):
        now = self.get_clock().now().nanoseconds / 1e9
        last = self._last_short_warn_time if key == "short" else self._last_long_warn_time
        if now - last >= self.warn_period_sec:
            self.get_logger().warn(text)
            if key == "short":
                self._last_short_warn_time = now
            else:
                self._last_long_warn_time = now

    def callback(self, msg: LowState):
        motor_states = list(getattr(msg, "motor_state", []))
        expected = len(self.joint_names)
        actual = len(motor_states)

        if actual < expected:
            self._warn_throttled(
                "short",
                f"lowstate motor_state too short: {actual} < {expected}. "
                "Filling missing joints with zeros."
            )
        elif actual > expected:
            self._warn_throttled(
                "long",
                f"lowstate motor_state longer than expected: {actual} > {expected}. "
                "Ignoring extra joints."
            )

        used_states = motor_states[:expected]
        positions = [float(getattr(m, "q", 0.0)) for m in used_states]
        velocities = [float(getattr(m, "dq", 0.0)) for m in used_states]
        efforts = [float(getattr(m, "tau_est", getattr(m, "tau", 0.0))) for m in used_states]

        missing = expected - len(used_states)
        if missing > 0:
            positions.extend([0.0] * missing)
            velocities.extend([0.0] * missing)
            efforts.extend([0.0] * missing)

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names
        js.position = positions
        if self.publish_velocity:
            js.velocity = velocities
        if self.publish_effort:
            js.effort = efforts

        self.pub.publish(js)

def main(args=None):
    rclpy.init(args=args)
    node = G1JointStateBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
