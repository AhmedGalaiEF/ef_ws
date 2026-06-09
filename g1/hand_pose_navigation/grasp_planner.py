"""
Step 6 — Define desired hand pose with grasp offset
=====================================================
Given T_base_target (target object pose in robot base frame), compute
T_base_hand_desired by applying a configurable pre-grasp approach offset.

The offset is expressed in the object frame:
    z  — along the approach direction (positive = away from object)
    x  — lateral offset (finger alignment)
    y  — vertical offset

Usage:
    planner = GraspPlanner(arm="right", standoff_m=0.10)
    T_desired = planner.compute(T_base_target)
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class GraspPlanner:
    """
    Step 6: Compute T_base_hand_desired from T_base_target.

    The wrist should approach the object from a direction specified by
    ``approach_axis`` (in object frame), at ``standoff_m`` distance.
    An optional ``lateral_m`` / ``vertical_m`` offset adjusts finger
    alignment.

    Args:
        arm:           "left" | "right"
        standoff_m:    distance to keep between wrist and object (metres)
        lateral_m:     lateral correction in object frame
        vertical_m:    vertical correction in object frame
        approach_axis: which axis of the object frame to approach along
                       (+Z is conventional for flat-table grasps)
    """

    def __init__(
        self,
        arm: str = "right",
        standoff_m: float = 0.08,
        lateral_m: float = 0.0,
        vertical_m: float = 0.0,
        approach_axis: str = "+z",
    ) -> None:
        self.arm = arm
        self.standoff_m = standoff_m
        self.lateral_m = lateral_m
        self.vertical_m = vertical_m
        self._approach_vec = _parse_axis(approach_axis)

    # ------------------------------------------------------------------
    def compute(self, T_base_target: np.ndarray) -> np.ndarray:
        """
        Returns T_base_hand_desired (4×4).

        The hand z-axis is aligned with -approach_axis (wrist pointing
        toward the object), and the position is shifted back by standoff_m.
        """
        # Build the grasp transform in object frame
        T_object_grasp = _build_grasp_offset(
            approach_vec=self._approach_vec,
            standoff_m=self.standoff_m,
            lateral_m=self.lateral_m,
            vertical_m=self.vertical_m,
            arm=self.arm,
        )
        return T_base_target @ T_object_grasp

    # ------------------------------------------------------------------
    def compute_pregrasp_sequence(
        self,
        T_base_target: np.ndarray,
        num_waypoints: int = 5,
    ) -> list:
        """
        Return a list of intermediate poses leading to the grasp pose.
        Useful for generating a Cartesian approach trajectory.
        """
        T_grasp = self.compute(T_base_target)
        T_far = self.compute_with_standoff(T_base_target, standoff_m=self.standoff_m * 3)
        waypoints = []
        for i in range(num_waypoints + 1):
            alpha = i / num_waypoints
            T_wp = _interpolate_SE3(T_far, T_grasp, alpha)
            waypoints.append(T_wp)
        return waypoints

    def compute_with_standoff(
        self, T_base_target: np.ndarray, standoff_m: float
    ) -> np.ndarray:
        old = self.standoff_m
        self.standoff_m = standoff_m
        result = self.compute(T_base_target)
        self.standoff_m = old
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_axis(spec: str) -> np.ndarray:
    sign = -1.0 if spec.startswith("-") else 1.0
    ax = spec.lstrip("+-").lower()
    vecs = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    return np.array(vecs[ax], dtype=np.float64) * sign


def _build_grasp_offset(
    approach_vec: np.ndarray,
    standoff_m: float,
    lateral_m: float,
    vertical_m: float,
    arm: str,
) -> np.ndarray:
    """
    Build T_object_grasp:
      - Hand approaches along approach_vec
      - Wrist z-axis points opposite to approach direction
      - Small lateral correction for left/right asymmetry
    """
    # Wrist z points toward object (opposite approach)
    z_wrist = -approach_vec / np.linalg.norm(approach_vec)

    # Choose a sensible wrist y (world up, projected perpendicular to z)
    world_up = np.array([0, 0, 1], dtype=np.float64)
    if abs(np.dot(z_wrist, world_up)) > 0.9:
        world_up = np.array([1, 0, 0], dtype=np.float64)
    x_wrist = np.cross(world_up, z_wrist)
    x_wrist /= np.linalg.norm(x_wrist)
    y_wrist = np.cross(z_wrist, x_wrist)

    R = np.column_stack([x_wrist, y_wrist, z_wrist])

    # Position: back off along approach direction
    t = approach_vec * standoff_m
    t += x_wrist * lateral_m
    t += y_wrist * vertical_m

    # Mirror lateral for left arm
    if arm == "left":
        t[1] = -t[1]

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _interpolate_SE3(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """Linear position + SLERP rotation interpolation."""
    t = (1 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]

    q0 = _R_to_quat(T0[:3, :3])
    q1 = _R_to_quat(T1[:3, :3])
    q = _slerp(q0, q1, alpha)
    R = _quat_to_R(*q)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _R_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        return 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        return (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s
    else:
        s = 2.0 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        return (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s


def _quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _slerp(q0, q1, t):
    q0 = np.array(q0)
    q1 = np.array(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1, 1)
    theta = np.arccos(dot)
    if abs(theta) < 1e-6:
        return tuple((1 - t) * q0 + t * q1)
    return tuple((np.sin((1 - t) * theta) / np.sin(theta)) * q0 +
                 (np.sin(t * theta) / np.sin(theta)) * q1)
