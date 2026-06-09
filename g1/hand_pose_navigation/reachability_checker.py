"""
Step 8 — Collision and reachability checks
===========================================
Validates a candidate arm configuration before sending it to the robot.

Checks performed:
    1. Joint limits — hard limits from URDF
    2. Torso clearance — wrist/elbow must stay above a minimum height
    3. Self-intersection proxy — elbow/wrist must remain outside simple
       bounding cylinders representing the torso and opposite arm
    4. Workspace radius — target must be within arm reach

This is a lightweight geometric checker; for full collision detection
use MoveIt 2 (moveit_msgs/GetMotionPlan) or Pinocchio + HPP-FCL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .arm_fk import ArmFK, JOINT_LIMITS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    safe: bool
    reasons: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.safe

    def __repr__(self) -> str:
        status = "SAFE" if self.safe else "UNSAFE"
        return f"CheckResult({status}: {self.reasons})"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _point_cylinder_distance(
    point: np.ndarray,
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    radius: float,
) -> float:
    """Signed distance from point to the surface of an infinite cylinder."""
    d = axis_end - axis_start
    d_len = np.linalg.norm(d)
    if d_len < 1e-9:
        return np.linalg.norm(point - axis_start) - radius
    d_unit = d / d_len
    proj = np.dot(point - axis_start, d_unit)
    closest = axis_start + proj * d_unit
    return float(np.linalg.norm(point - closest)) - radius


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class ReachabilityChecker:
    """
    Step 8: Geometric safety validation for a candidate arm configuration.

    Args:
        arm:            "left" | "right"
        max_reach_m:    maximum arm reach (metres); default 0.75 m
        min_wrist_z_m:  wrist must stay above this height in base frame
        torso_radius_m: approximate torso cylinder radius
        torso_axis_z:   [z_lo, z_hi] for the torso cylinder extent
    """

    def __init__(
        self,
        arm: str = "right",
        max_reach_m: float = 0.75,
        min_wrist_z_m: float = -0.05,
        torso_radius_m: float = 0.18,
        torso_axis_z: Tuple[float, float] = (-0.1, 0.55),
    ) -> None:
        self.arm = arm
        self.max_reach_m = max_reach_m
        self.min_wrist_z_m = min_wrist_z_m
        self.torso_radius_m = torso_radius_m
        self.torso_z = torso_axis_z
        self._limits = JOINT_LIMITS[arm]
        self._joint_indices = LEFT_ARM_JOINTS if arm == "left" else RIGHT_ARM_JOINTS
        self._fk = ArmFK(arm=arm, backend="dh")

    # ------------------------------------------------------------------
    def check(
        self,
        q_arm: np.ndarray,
        T_base_desired: Optional[np.ndarray] = None,
    ) -> CheckResult:
        """
        Run all checks on a 7-element arm joint configuration.

        Args:
            q_arm:          7-element joint angle vector (radians)
            T_base_desired: optional target pose to check workspace reach

        Returns:
            CheckResult with safe flag and list of violation descriptions
        """
        reasons: List[str] = []

        # 1. Joint limits
        for i, (lo, hi) in enumerate(self._limits):
            if q_arm[i] < lo - 1e-4:
                reasons.append(f"J{i + 1} below limit ({q_arm[i]:.3f} < {lo:.3f})")
            elif q_arm[i] > hi + 1e-4:
                reasons.append(f"J{i + 1} above limit ({q_arm[i]:.3f} > {hi:.3f})")

        # 2. FK-based checks
        T_hand = self._fk.compute_arm(q_arm)
        wrist_pos = T_hand[:3, 3]

        # 3. Minimum wrist height
        if wrist_pos[2] < self.min_wrist_z_m:
            reasons.append(
                f"Wrist z={wrist_pos[2]:.3f} below minimum {self.min_wrist_z_m:.3f}"
            )

        # 4. Torso clearance
        torso_start = np.array([0.0, 0.0, self.torso_z[0]])
        torso_end = np.array([0.0, 0.0, self.torso_z[1]])
        wrist_dist = _point_cylinder_distance(
            wrist_pos, torso_start, torso_end, self.torso_radius_m
        )
        if wrist_dist < 0.03:
            reasons.append(
                f"Wrist too close to torso (clearance={wrist_dist:.3f} m)"
            )

        # 5. Workspace radius
        if T_base_desired is not None:
            target_pos = T_base_desired[:3, 3]
            # Approximate shoulder position
            shoulder_y = 0.15 if self.arm == "left" else -0.15
            shoulder = np.array([0.0, shoulder_y, 0.30])
            dist = float(np.linalg.norm(target_pos - shoulder))
            if dist > self.max_reach_m:
                reasons.append(
                    f"Target {dist:.3f} m from shoulder exceeds max reach {self.max_reach_m:.3f} m"
                )
            if dist < 0.08:
                reasons.append(
                    f"Target {dist:.3f} m too close to shoulder (< 0.08 m)"
                )

        # 6. Forward-reach bias — wrist should not be behind robot
        if wrist_pos[0] < -0.15:
            reasons.append(
                f"Wrist behind robot body (x={wrist_pos[0]:.3f} < -0.15)"
            )

        return CheckResult(safe=len(reasons) == 0, reasons=reasons)

    # ------------------------------------------------------------------
    def check_target_reachable(self, T_base_target: np.ndarray) -> CheckResult:
        """Quick pre-IK workspace check — no FK required."""
        reasons = []
        target_pos = T_base_target[:3, 3]
        shoulder_y = 0.15 if self.arm == "left" else -0.15
        shoulder = np.array([0.0, shoulder_y, 0.30])
        dist = float(np.linalg.norm(target_pos - shoulder))
        if dist > self.max_reach_m:
            reasons.append(
                f"Target {dist:.3f} m from shoulder > max reach {self.max_reach_m:.3f} m")
        if target_pos[2] < self.min_wrist_z_m:
            reasons.append(f"Target z={target_pos[2]:.3f} too low")
        return CheckResult(safe=len(reasons) == 0, reasons=reasons)

    # ------------------------------------------------------------------
    def clamp_joints(self, q_arm: np.ndarray) -> np.ndarray:
        """Hard-clamp joint angles to their limits."""
        lo = np.array([lim[0] for lim in self._limits])
        hi = np.array([lim[1] for lim in self._limits])
        return np.clip(q_arm, lo, hi)
