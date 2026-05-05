"""
Step 1 — Calibrate RGB-D camera
================================
Publishes the static transform  camera_link -> camera_color_optical_frame
so that depth and colour measurements share a common TF ancestry.

For the Intel RealSense D435/D435i the colour optical frame sits at the
colour sensor's physical location with a -90° rotation around X (OpenCV
convention: Z forward, X right, Y down).

Usage (standalone):
    ros2 run hand_pose_navigation camera_tf_publisher

Or construct and spin inside another node / launch file.
"""
from __future__ import annotations

import math
from typing import Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros


# D435 factory extrinsic: colour optical frame relative to camera_link.
# Adjust xyz_m and rpy_rad if your calibration differs.
_DEFAULT_XYZ_M: Tuple[float, float, float] = (0.0, 0.015, 0.0)
_DEFAULT_RPY_RAD: Tuple[float, float, float] = (-math.pi / 2, 0.0, -math.pi / 2)


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """ZYX Euler -> quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class CameraTFPublisher(Node):
    """
    Step 1: Publish static TF  camera_link -> camera_color_optical_frame.

    Parameters (ROS params):
        parent_frame  (str)   default "camera_link"
        child_frame   (str)   default "camera_color_optical_frame"
        tx, ty, tz    (float) translation in metres
        roll, pitch, yaw (float) rotation in radians (ZYX Euler)
    """

    def __init__(self) -> None:
        super().__init__("camera_tf_publisher")

        # Declare overrideable parameters
        self.declare_parameter("parent_frame", "camera_link")
        self.declare_parameter("child_frame", "camera_color_optical_frame")
        self.declare_parameter("tx", _DEFAULT_XYZ_M[0])
        self.declare_parameter("ty", _DEFAULT_XYZ_M[1])
        self.declare_parameter("tz", _DEFAULT_XYZ_M[2])
        self.declare_parameter("roll", _DEFAULT_RPY_RAD[0])
        self.declare_parameter("pitch", _DEFAULT_RPY_RAD[1])
        self.declare_parameter("yaw", _DEFAULT_RPY_RAD[2])

        self._broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._publish_static_tf()
        self.get_logger().info(
            f"[Step 1] Camera TF published: "
            f"{self.get_parameter('parent_frame').value} -> "
            f"{self.get_parameter('child_frame').value}"
        )

    # ------------------------------------------------------------------
    def _publish_static_tf(self) -> None:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter("parent_frame").value
        t.child_frame_id = self.get_parameter("child_frame").value

        t.transform.translation.x = self.get_parameter("tx").value
        t.transform.translation.y = self.get_parameter("ty").value
        t.transform.translation.z = self.get_parameter("tz").value

        qx, qy, qz, qw = _rpy_to_quat(
            self.get_parameter("roll").value,
            self.get_parameter("pitch").value,
            self.get_parameter("yaw").value,
        )
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self._broadcaster.sendTransform(t)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
