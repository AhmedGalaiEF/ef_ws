"""
Step 9 — Send low-level arm command
=====================================
Wraps the G1 Robot SDK arm publisher (_ArmSdkPublisher) to send
joint-space commands with trajectory interpolation and safety gating.

The arm SDK expects the 30-DOF joint array published to ``rt/arm_sdk``
with PD gains.  We build that from the 7-DOF arm solution produced by
the IK solver.

Usage:
    executor = ArmExecutor(robot, arm="right")
    executor.execute(q_arm_desired, duration_s=2.0)
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# SDK imports from parent modules directory
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))
    from sdk_client import Robot, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
except ImportError:
    Robot = None  # type: ignore
    LEFT_ARM_JOINTS = list(range(15, 22))
    RIGHT_ARM_JOINTS = list(range(22, 29))

from .arm_fk import ArmFK
from .reachability_checker import ReachabilityChecker


# ---------------------------------------------------------------------------
# Default PD gains (from sdk_client._ArmSdkPublisher defaults)
# ---------------------------------------------------------------------------
_DEFAULT_KP: Dict[int, float] = {}   # use arm SDK defaults
_DEFAULT_KD: Dict[int, float] = {}

_KP_ARM = 60.0   # position gain for arm joints
_KD_ARM = 2.0    # damping gain for arm joints


class ArmExecutor:
    """
    Step 9: Execute an arm joint target using the Robot SDK.

    Args:
        robot:       Robot instance from sdk_client
        arm:         "left" | "right"
        kp:          proportional gain for all arm joints
        kd:          derivative gain for all arm joints
        rate_hz:     command rate during trajectory interpolation
        safety_gate: if True, refuse to send commands that fail reachability check
    """

    def __init__(
        self,
        robot,
        arm: str = "right",
        kp: float = _KP_ARM,
        kd: float = _KD_ARM,
        rate_hz: float = 50.0,
        safety_gate: bool = True,
    ) -> None:
        self.robot = robot
        self.arm = arm
        self.kp = kp
        self.kd = kd
        self.rate_hz = rate_hz
        self.safety_gate = safety_gate
        self._joint_indices = LEFT_ARM_JOINTS if arm == "left" else RIGHT_ARM_JOINTS
        self._fk = ArmFK(arm=arm, backend="dh")
        self._checker = ReachabilityChecker(arm=arm)

    # ------------------------------------------------------------------
    def execute(
        self,
        q_arm_desired: np.ndarray,
        duration_s: float = 2.0,
        q_arm_start: Optional[np.ndarray] = None,
        T_base_desired: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Interpolate from current arm pose to q_arm_desired and send commands.

        Args:
            q_arm_desired: 7-element target joint angles (radians)
            duration_s:    total move duration
            q_arm_start:   override start configuration (default: read from robot)
            T_base_desired: optional target pose for safety check context

        Returns:
            dict with "success", "duration_s", "steps", "final_q"
        """
        # Safety check
        if self.safety_gate:
            result = self._checker.check(q_arm_desired, T_base_desired)
            if not result.safe:
                return {
                    "success": False,
                    "reason": "safety_gate",
                    "violations": result.reasons,
                    "duration_s": 0.0,
                    "steps": 0,
                }

        # Get start configuration
        if q_arm_start is None:
            q_arm_start = self._read_current_arm_q()

        steps = max(1, int(duration_s * self.rate_hz))
        dt = duration_s / steps
        final_q = q_arm_start.copy()

        for i in range(steps):
            alpha = _smooth_step((i + 1) / steps)
            q_cmd = (1 - alpha) * q_arm_start + alpha * q_arm_desired
            self._send_command(q_cmd)
            final_q = q_cmd
            time.sleep(dt)

        return {
            "success": True,
            "duration_s": duration_s,
            "steps": steps,
            "final_q": final_q,
        }

    # ------------------------------------------------------------------
    def execute_cartesian(
        self,
        waypoints: List[np.ndarray],
        duration_per_wp_s: float = 1.0,
    ) -> Dict:
        """
        Execute a sequence of 7-DOF joint waypoints (e.g., pre-grasp sequence).

        Each element of waypoints is a 7-element joint angle vector.
        """
        results = []
        for wp in waypoints:
            result = self.execute(wp, duration_s=duration_per_wp_s)
            results.append(result)
            if not result["success"]:
                return {"success": False, "waypoint_results": results}
        return {"success": True, "waypoint_results": results}

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Hold current position by re-sending current joint state."""
        q_cur = self._read_current_arm_q()
        self._send_command(q_cur)

    # ------------------------------------------------------------------
    def _read_current_arm_q(self) -> np.ndarray:
        """Read current arm joint angles from the robot."""
        try:
            js = self.robot.get_joint_states()
            joints = js.get("joints", {})
            q = np.zeros(30)
            for name, data in joints.items():
                idx = data.get("index", -1)
                if 0 <= idx < 30:
                    q[idx] = data.get("position", 0.0)
            return q[self._joint_indices]
        except Exception:
            return np.zeros(7)

    # ------------------------------------------------------------------
    def _send_command(self, q_arm: np.ndarray) -> None:
        """
        Build the 30-DOF joint targets and publish via rt/arm_sdk.

        Only the arm joints for this side are set; the other 22 joints
        remain at whatever the loco controller holds.
        """
        # Build per-joint kp/kd overrides for arm joints only
        kp_by_joint = {idx: self.kp for idx in self._joint_indices}
        kd_by_joint = {idx: self.kd for idx in self._joint_indices}

        # Build 30-element target array: NaN keeps loco in control of non-arm joints
        targets = [float("nan")] * 30
        for i, joint_idx in enumerate(self._joint_indices):
            targets[joint_idx] = float(q_arm[i])

        # Replace NaN with 0 for indices that must be specified; arm SDK only
        # acts on joints where the weight blending has authority.
        try:
            self.robot._arm_pub.publish_targets(
                joint_targets=targets,
                kp=self.kp,
                kd=self.kd,
                kp_by_joint=kp_by_joint,
                kd_by_joint=kd_by_joint,
            )
        except AttributeError:
            # Fallback: use move_upper_body_joint for each joint sequentially
            # (less coordinated but functional)
            for i, joint_idx in enumerate(self._joint_indices):
                try:
                    self.robot.move_upper_body_joint(
                        joint_index=joint_idx,
                        target=float(q_arm[i]),
                        max_speed_rad_s=1.0,
                        timeout=0.1,
                    )
                except Exception:
                    pass


# ---------------------------------------------------------------------------
def _smooth_step(t: float) -> float:
    """Smooth-step ease: 3t²-2t³ (zero velocity at endpoints)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)
