"""
Step 8b — Swept-path obstacle checking
=========================================
ReachabilityChecker (Step 8) only validates the final joint configuration.
That misses collisions that happen partway through a move even though both
endpoints look safe — e.g. a straight-line joint interpolation that dips
through the table on the way to a pose that is itself above the table.

This module checks a set of points sampled along the interpolated path
between the current and desired arm configuration against a small set of
geometric obstacle proxies (deliberately simple, not a full scene):

    table — horizontal plane + rough footprint, from table_estimator.py
    self  — the *other* arm's current position, approximated as a capsule
            from that arm's shoulder to its wrist
    torso — same cylinder proxy ReachabilityChecker already uses
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .reachability_checker import CheckResult
from .table_estimator import TablePlane

if TYPE_CHECKING:
    from .arm_fk import ArmFK


@dataclass
class Obstacles:
    table: Optional[TablePlane] = None
    opposite_shoulder_base: Optional[np.ndarray] = None
    opposite_wrist_base: Optional[np.ndarray] = None
    torso_radius_m: float = 0.18
    torso_z: Tuple[float, float] = (-0.1, 0.55)
    self_collision_radius_m: float = 0.09
    table_margin_m: float = 0.03
    table_xy_margin_m: float = 0.05


def _point_segment_distance(
    point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    radius: float,
) -> float:
    """Signed distance from point to the surface of a capsule (line segment + radius)."""
    d = seg_end - seg_start
    d_len = np.linalg.norm(d)
    if d_len < 1e-9:
        return float(np.linalg.norm(point - seg_start)) - radius
    d_unit = d / d_len
    proj = np.clip(np.dot(point - seg_start, d_unit), 0.0, d_len)
    closest = seg_start + proj * d_unit
    return float(np.linalg.norm(point - closest)) - radius


def check_swept_path(
    fk: "ArmFK",
    q_arm_start: np.ndarray,
    q_arm_goal: np.ndarray,
    obstacles: Obstacles,
    n_samples: int = 8,
) -> CheckResult:
    """Sample n_samples+1 points along the straight-line joint interpolation
    from q_arm_start to q_arm_goal and check the elbow + wrist/hand position
    at each against the obstacle proxies in `obstacles`.
    """
    reasons: List[str] = []
    q_start = np.asarray(q_arm_start, dtype=np.float64)
    q_goal = np.asarray(q_arm_goal, dtype=np.float64)
    torso_start = np.array([0.0, 0.0, obstacles.torso_z[0]])
    torso_end = np.array([0.0, 0.0, obstacles.torso_z[1]])
    have_opposite_arm = (
        obstacles.opposite_shoulder_base is not None
        and obstacles.opposite_wrist_base is not None
    )

    for i in range(n_samples + 1):
        alpha = i / n_samples if n_samples > 0 else 1.0
        q = (1.0 - alpha) * q_start + alpha * q_goal
        elbow = fk.compute_arm_partial(q, 4)[:3, 3]
        hand = fk.compute_arm(q)[:3, 3]

        for label, point in (("elbow", elbow), ("hand", hand)):
            if obstacles.table is not None:
                t = obstacles.table
                within_x = (
                    (t.x_range_m[0] - obstacles.table_xy_margin_m)
                    <= point[0]
                    <= (t.x_range_m[1] + obstacles.table_xy_margin_m)
                )
                within_y = (
                    (t.y_range_m[0] - obstacles.table_xy_margin_m)
                    <= point[1]
                    <= (t.y_range_m[1] + obstacles.table_xy_margin_m)
                )
                if within_x and within_y and point[2] < t.z_base_m + obstacles.table_margin_m:
                    reasons.append(
                        f"[alpha={alpha:.2f}] {label} would clip table "
                        f"(z={point[2]:.3f}m < table {t.z_base_m:.3f}m + "
                        f"{obstacles.table_margin_m:.3f}m margin)"
                    )

            if label == "hand":
                torso_dist = _point_segment_distance(
                    point, torso_start, torso_end, obstacles.torso_radius_m
                )
                if torso_dist < 0.02:
                    reasons.append(
                        f"[alpha={alpha:.2f}] {label} too close to torso "
                        f"(clearance={torso_dist:.3f}m)"
                    )

            if have_opposite_arm:
                self_dist = _point_segment_distance(
                    point,
                    obstacles.opposite_shoulder_base,
                    obstacles.opposite_wrist_base,
                    obstacles.self_collision_radius_m,
                )
                if self_dist < 0.02:
                    reasons.append(
                        f"[alpha={alpha:.2f}] {label} too close to opposite arm (clearance={self_dist:.3f}m)"
                    )

    if len(reasons) > 6:
        reasons = reasons[:5] + [f"... {len(reasons) - 5} more swept-path violations"]

    return CheckResult(safe=len(reasons) == 0, reasons=reasons)
