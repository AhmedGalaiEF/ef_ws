import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class DemoJointMotionNode(Node):
    def __init__(self) -> None:
        super().__init__('g1_demo_joint_motion')
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('amplitude', 0.6)
        self.declare_parameter('frequency_hz', 0.25)
        self.declare_parameter('mode', 'pose')

        rate_hz = float(self.get_parameter('rate_hz').value)
        self._amplitude = float(self.get_parameter('amplitude').value)
        self._frequency_hz = float(self.get_parameter('frequency_hz').value)
        self._mode = str(self.get_parameter('mode').value).strip().lower()

        self._publisher = self.create_publisher(JointState, 'joint_states', 10)
        self._start_time = self.get_clock().now()

        self._configure_joints()

        self._timer = self.create_timer(1.0 / rate_hz, self._on_timer)

    def _configure_joints(self) -> None:
        if self._mode == 'walk':
            # Walk-in-place: legs opposite phase, arms swing.
            self._joints = [
                'left_hip_pitch_joint',
                'left_knee_joint',
                'left_ankle_pitch_joint',
                'right_hip_pitch_joint',
                'right_knee_joint',
                'right_ankle_pitch_joint',
                'left_shoulder_pitch_joint',
                'right_shoulder_pitch_joint',
                'waist_yaw_joint',
            ]
            self._phases = {
                'left_hip_pitch_joint': 0.0,
                'left_knee_joint': math.pi / 2.0,
                'left_ankle_pitch_joint': math.pi,
                'right_hip_pitch_joint': math.pi,
                'right_knee_joint': 3.0 * math.pi / 2.0,
                'right_ankle_pitch_joint': 0.0,
                'left_shoulder_pitch_joint': math.pi,
                'right_shoulder_pitch_joint': 0.0,
                'waist_yaw_joint': math.pi / 4.0,
            }
            return

        # Default pose sweep includes hands.
        self._joints = [
            'left_hip_pitch_joint',
            'left_knee_joint',
            'left_ankle_pitch_joint',
            'right_hip_pitch_joint',
            'right_knee_joint',
            'right_ankle_pitch_joint',
            'left_shoulder_pitch_joint',
            'left_elbow_joint',
            'right_shoulder_pitch_joint',
            'right_elbow_joint',
            'waist_yaw_joint',
            'waist_pitch_joint',
            'left_hand_thumb_0_joint',
            'left_hand_thumb_1_joint',
            'left_hand_thumb_2_joint',
            'left_hand_index_0_joint',
            'left_hand_index_1_joint',
            'left_hand_middle_0_joint',
            'left_hand_middle_1_joint',
            'right_hand_thumb_0_joint',
            'right_hand_thumb_1_joint',
            'right_hand_thumb_2_joint',
            'right_hand_index_0_joint',
            'right_hand_index_1_joint',
            'right_hand_middle_0_joint',
            'right_hand_middle_1_joint',
        ]

        # Per-joint phase offsets to make motion feel more natural.
        self._phases = {
            'left_hip_pitch_joint': 0.0,
            'left_knee_joint': math.pi / 3.0,
            'left_ankle_pitch_joint': math.pi / 2.0,
            'right_hip_pitch_joint': math.pi,
            'right_knee_joint': 4.0 * math.pi / 3.0,
            'right_ankle_pitch_joint': 3.0 * math.pi / 2.0,
            'left_shoulder_pitch_joint': math.pi / 4.0,
            'left_elbow_joint': math.pi / 2.0,
            'right_shoulder_pitch_joint': 5.0 * math.pi / 4.0,
            'right_elbow_joint': 3.0 * math.pi / 2.0,
            'waist_yaw_joint': 0.0,
            'waist_pitch_joint': math.pi / 2.0,
            'left_hand_thumb_0_joint': 0.0,
            'left_hand_thumb_1_joint': math.pi / 4.0,
            'left_hand_thumb_2_joint': math.pi / 2.0,
            'left_hand_index_0_joint': math.pi / 3.0,
            'left_hand_index_1_joint': 2.0 * math.pi / 3.0,
            'left_hand_middle_0_joint': math.pi / 6.0,
            'left_hand_middle_1_joint': 5.0 * math.pi / 6.0,
            'right_hand_thumb_0_joint': math.pi,
            'right_hand_thumb_1_joint': 5.0 * math.pi / 4.0,
            'right_hand_thumb_2_joint': 3.0 * math.pi / 2.0,
            'right_hand_index_0_joint': 4.0 * math.pi / 3.0,
            'right_hand_index_1_joint': 5.0 * math.pi / 3.0,
            'right_hand_middle_0_joint': 7.0 * math.pi / 6.0,
            'right_hand_middle_1_joint': 11.0 * math.pi / 6.0,
        }
    def _on_timer(self) -> None:
        now = self.get_clock().now()
        t = (now - self._start_time).nanoseconds * 1e-9
        base = 2.0 * math.pi * self._frequency_hz * t

        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = list(self._joints)
        msg.position = [
            self._amplitude * math.sin(base + self._phases[name])
            for name in msg.name
        ]
        self._publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = DemoJointMotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
