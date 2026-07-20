"""
Experimental VLA (vision-language-action) planning backend — scaffolding only
================================================================================
No trained policy is wired in. This module exists so the website can offer a
"VLA" planning mode and a real backend can be dropped in later without
touching recognition_app.py or the grab flow: implement VLAPolicy.plan() (or
subclass it) and swap it in for DEFAULT_VLA_POLICY below.

The interface intentionally takes the same obstacle model
(obstacle_checker.Obstacles) as the geometric Direct/IK path, so a real
policy can be checked with the same swept-path safety gate in
arm_executor.ArmExecutor.execute() rather than trusting the policy blindly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .obstacle_checker import Obstacles


@dataclass
class VLAPlanResult:
    success: bool
    reason: str
    # 7-DOF arm joint waypoints (radians), only set when success is True.
    waypoints: Optional[List[np.ndarray]] = None


class VLAPolicy:
    """Pluggable VLA policy interface.

    The default implementation is a stub: it never produces motion, it only
    reports why. Subclass and override plan() to wire in a real model or
    inference server.
    """

    available: bool = False
    unavailable_reason: str = (
        "no VLA model wired in — this is UI/interface scaffolding only; "
        "implement VLAPolicy.plan() to enable it"
    )

    def plan(
        self,
        rgb_bgr: np.ndarray,
        depth_m: np.ndarray,
        instruction: str,
        q_arm_current: np.ndarray,
        arm: str,
        obstacles: Obstacles,
    ) -> VLAPlanResult:
        """Return a joint-space trajectory for `arm` that accomplishes
        `instruction`, or a failed VLAPlanResult explaining why not.

        Args:
            rgb_bgr, depth_m: latest camera frame
            instruction:      natural-language description of the task
                               (e.g. the selected object's label)
            q_arm_current:    7-element current arm joint angles (radians)
            arm:               "left" | "right"
            obstacles:        same obstacle model used by the geometric
                               path (table plane, opposite-arm proxy) —
                               a real policy should condition on this or at
                               minimum have its output re-checked against it
        """
        return VLAPlanResult(success=False, reason=self.unavailable_reason)


# Swap this out (or mutate DEFAULT_VLA_POLICY.__class__) once a real backend exists.
DEFAULT_VLA_POLICY = VLAPolicy()
