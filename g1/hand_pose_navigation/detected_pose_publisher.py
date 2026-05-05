"""
Step 3 — Publish detected pose as a TF frame
=============================================
Receives DetectionResult objects and broadcasts them as a dynamic TF frame
named ``object_visible_pose`` (child of ``camera_color_optical_frame``).

This makes the target reachable from anywhere in the TF tree via a single
lookupTransform call, decoupling detection from planning.
"""
from __future__ import annotations

import threading
from typing import Optional

import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
import numpy as np

from .target_detector import DetectionResult


class DetectedPosePublisher(Node):
    """
    Step 3: Broadcast object_visible_pose as a dynamic TF frame.

    Call update(result) whenever a new DetectionResult is available;
    the broadcaster is thread-safe.

    Parameters (ROS params):
        camera_frame     (str)   default "camera_color_optical_frame"
        object_frame     (str)   default "object_visible_pose"
        publish_rate_hz  (float) default 30.0 — timer re-publishes last pose
    """

    def __init__(self) -> None:
        super().__init__("detected_pose_publisher")

        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("object_frame", "object_visible_pose")
        self.declare_parameter("publish_rate_hz", 30.0)

        self._broadcaster = tf2_ros.TransformBroadcaster(self)
        self._lock = threading.Lock()
        self._last_result: Optional[DetectionResult] = None

        rate = self.get_parameter("publish_rate_hz").value
        self._timer = self.create_timer(1.0 / rate, self._timer_cb)

        self.get_logger().info(
            f"[Step 3] DetectedPosePublisher ready — broadcasting "
            f"{self.get_parameter('object_frame').value}"
        )

    # ------------------------------------------------------------------
    def update(self, result: DetectionResult) -> None:
        """Thread-safe update with a fresh detection."""
        with self._lock:
            self._last_result = result
        self._broadcast(result)

    # ------------------------------------------------------------------
    def _timer_cb(self) -> None:
        with self._lock:
            result = self._last_result
        if result is not None:
            self._broadcast(result)

    # ------------------------------------------------------------------
    def _broadcast(self, result: DetectionResult) -> None:
        T = result.T_camera_object
        R = T[:3, :3]
        t = T[:3, 3]
        quat = _rotation_matrix_to_quat(R)

        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = self.get_parameter("camera_frame").value
        ts.child_frame_id = self.get_parameter("object_frame").value

        ts.transform.translation.x = float(t[0])
        ts.transform.translation.y = float(t[1])
        ts.transform.translation.z = float(t[2])
        ts.transform.rotation.x = float(quat[0])
        ts.transform.rotation.y = float(quat[1])
        ts.transform.rotation.z = float(quat[2])
        ts.transform.rotation.w = float(quat[3])

        self._broadcaster.sendTransform(ts)


# ---------------------------------------------------------------------------
def _rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion [x, y, z, w]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)
