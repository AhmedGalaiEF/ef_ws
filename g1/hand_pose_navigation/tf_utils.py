"""
Step 5 — Transform target to robot base frame
===============================================
Wraps tf2_ros.Buffer to provide typed lookups and numpy-matrix helpers.

Key operation:
    T_base_target = tf_buffer.lookupTransform("base_link", "object_visible_pose")

All transforms are returned as 4×4 numpy arrays for easy composition.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped


def stamped_to_matrix(ts: TransformStamped) -> np.ndarray:
    """Convert a TransformStamped to a 4×4 homogeneous matrix."""
    t = ts.transform.translation
    q = ts.transform.rotation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_to_R(q.x, q.y, q.z, q.w)
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def matrix_to_stamped(
    T: np.ndarray,
    parent_frame: str,
    child_frame: str,
    stamp,
) -> TransformStamped:
    """Convert a 4×4 matrix to a TransformStamped."""
    ts = TransformStamped()
    ts.header.stamp = stamp
    ts.header.frame_id = parent_frame
    ts.child_frame_id = child_frame

    ts.transform.translation.x = float(T[0, 3])
    ts.transform.translation.y = float(T[1, 3])
    ts.transform.translation.z = float(T[2, 3])

    qx, qy, qz, qw = _R_to_quat(T[:3, :3])
    ts.transform.rotation.x = qx
    ts.transform.rotation.y = qy
    ts.transform.rotation.z = qz
    ts.transform.rotation.w = qw
    return ts


def _quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _R_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return (
            (R[2, 1] - R[1, 2]) * s,
            (R[0, 2] - R[2, 0]) * s,
            (R[1, 0] - R[0, 1]) * s,
            0.25 / s,
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


# ---------------------------------------------------------------------------

class TFUtils:
    """
    Step 5: Thin wrapper around tf2_ros for matrix-valued lookups.

    Args:
        node: an active rclpy Node (provides clock + executor)
        timeout_s: default lookup timeout in seconds
    """

    def __init__(self, node: Node, timeout_s: float = 0.1) -> None:
        self._node = node
        self._timeout_s = timeout_s
        self._buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._buffer, node)

    # ------------------------------------------------------------------
    def lookup(
        self,
        target_frame: str,
        source_frame: str,
        timeout_s: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Return T such that  p_target = T @ p_source  (4×4 numpy matrix).
        Returns None if the transform is unavailable.
        """
        t = timeout_s or self._timeout_s
        try:
            ts = self._buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                Duration(seconds=t),
            )
            return stamped_to_matrix(ts)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self._node.get_logger().debug(
                f"[Step 5] TF lookup {target_frame}<-{source_frame} failed: {exc}"
            )
            return None

    # ------------------------------------------------------------------
    def base_to_target(
        self,
        base_frame: str = "base_link",
        object_frame: str = "object_visible_pose",
        timeout_s: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Shortcut for the primary use-case:
            T_base_target = lookupTransform(base_link, object_visible_pose)
        """
        return self.lookup(base_frame, object_frame, timeout_s)

    # ------------------------------------------------------------------
    def camera_in_base(
        self,
        base_frame: str = "base_link",
        camera_frame: str = "camera_color_optical_frame",
        timeout_s: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """T_base_camera — needed to project depth points into robot frame."""
        return self.lookup(base_frame, camera_frame, timeout_s)

    # ------------------------------------------------------------------
    @property
    def buffer(self) -> tf2_ros.Buffer:
        return self._buffer
