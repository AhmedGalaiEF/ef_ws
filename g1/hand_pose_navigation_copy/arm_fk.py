"""
Step 4 — Forward kinematics: joint states -> T_base_hand
==========================================================
Computes the hand (end-effector) pose in base_link coordinates given a
vector of arm joint angles.

Two backends are supported, selected automatically at import time:
    pinocchio  — full URDF-based FK (preferred, install: pip install pin)
    numpy DH   — lightweight analytical approximation for the G1 29-DOF arm

The G1 arm DH approximation uses the physical link lengths from the
g1_29dof_with_hand_rev_1_0_pkg.urdf.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# G1 arm joint indices in the 30-joint low-state array
# --------------------------------------------------------------------------

LEFT_ARM_JOINTS = [15, 16, 17, 18, 19, 20, 21]   # shoulder p/r/y, elbow, wrist r/p/y
RIGHT_ARM_JOINTS = [22, 23, 24, 25, 26, 27, 28]

# Joint limits [rad] — from URDF (conservative subset)
JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "left": [
        (-3.0890, 2.6700),   # shoulder pitch
        (-1.5708, 2.2000),   # shoulder roll
        (-2.1817, 2.1817),   # shoulder yaw
        (-1.0472, 2.0944),   # elbow
        (-1.9722, 1.9722),   # wrist roll
        (-1.6580, 1.6580),   # wrist pitch
        (-1.6580, 1.6580),   # wrist yaw
    ],
    "right": [
        (-2.6700, 3.0890),
        (-2.2000, 1.5708),
        (-2.1817, 2.1817),
        (-1.0472, 2.0944),
        (-1.9722, 1.9722),
        (-1.6580, 1.6580),
        (-1.6580, 1.6580),
    ],
}

# G1 right-arm DH parameters [a, d, alpha, theta_offset] (metres / radians)
# Derived from URDF link origins; mirrored for left arm.
_DH_RIGHT = np.array([
    # a       d       alpha       theta_off
    [0.000,  0.000, -np.pi/2,   0.000],  # J1 shoulder pitch
    [0.000,  0.000,  np.pi/2,   0.000],  # J2 shoulder roll
    [0.000,  0.300, -np.pi/2,   0.000],  # J3 shoulder yaw  (upper arm len ~0.30m)
    [0.000,  0.000,  np.pi/2,   0.000],  # J4 elbow
    [0.000,  0.250, -np.pi/2,   0.000],  # J5 wrist roll    (forearm len ~0.25m)
    [0.000,  0.000,  np.pi/2,   0.000],  # J6 wrist pitch
    [0.000,  0.100,  0.000,     0.000],  # J7 wrist yaw     (hand offset ~0.10m)
], dtype=np.float64)

# Right shoulder origin in base_link [x, y, z] (metres)
_RIGHT_SHOULDER_IN_BASE = np.array([0.0, -0.15, 0.30], dtype=np.float64)
_LEFT_SHOULDER_IN_BASE  = np.array([0.0,  0.15, 0.30], dtype=np.float64)


# --------------------------------------------------------------------------
# DH helper
# --------------------------------------------------------------------------

def _dh_matrix(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d     ],
        [0,   0,        0,       1     ],
    ], dtype=np.float64)


def _fk_dh(q: np.ndarray, dh: np.ndarray, shoulder_in_base: np.ndarray) -> np.ndarray:
    """7-DOF DH forward kinematics; returns 4×4 T_base_ee."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = shoulder_in_base
    for i, (a, d, alpha, theta_off) in enumerate(dh):
        T = T @ _dh_matrix(a, d, alpha, q[i] + theta_off)
    return T


# --------------------------------------------------------------------------
# Pinocchio backend (optional)
# --------------------------------------------------------------------------

_pin_model = None
_pin_data = None
_pin_ee_frame_id: Dict[str, int] = {}

_URDF_PATH = os.path.join(
    "/home/unitree/EF/ef_ws/g1/install/g1_description/share/"
    "g1_description/urdf/g1_29dof_with_hand_rev_1_0_pkg.urdf"
)
_EE_FRAME_NAMES = {
    "right": "right_hand_palm_link",
    "left":  "left_hand_palm_link",
}


def _try_load_pinocchio() -> bool:
    global _pin_model, _pin_data, _pin_ee_frame_id
    try:
        import pinocchio as pin  # type: ignore
        if not os.path.exists(_URDF_PATH):
            return False
        _pin_model = pin.buildModelFromUrdf(_URDF_PATH)
        _pin_data = _pin_model.createData()
        for side, name in _EE_FRAME_NAMES.items():
            if _pin_model.existFrame(name):
                _pin_ee_frame_id[side] = _pin_model.getFrameId(name)
        return True
    except Exception:
        return False


_USE_PIN = _try_load_pinocchio()


# --------------------------------------------------------------------------
# Public class
# --------------------------------------------------------------------------

class ArmFK:
    """
    Step 4: Compute T_base_hand from joint state angles.

    Args:
        arm:     "left" | "right"
        backend: "auto" | "pinocchio" | "dh"
    """

    def __init__(self, arm: str = "right", backend: str = "auto") -> None:
        if arm not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        self.arm = arm
        self.joint_indices = LEFT_ARM_JOINTS if arm == "left" else RIGHT_ARM_JOINTS
        self._shoulder = (
            _LEFT_SHOULDER_IN_BASE if arm == "left" else _RIGHT_SHOULDER_IN_BASE
        )

        if backend == "auto":
            self._use_pin = _USE_PIN and arm in _pin_ee_frame_id
        elif backend == "pinocchio":
            self._use_pin = True
        else:
            self._use_pin = False

    # ------------------------------------------------------------------
    def compute(self, q_full: np.ndarray) -> np.ndarray:
        """
        Compute end-effector pose.

        Args:
            q_full: length-30 array of all joint angles (low-state order)

        Returns:
            T: 4×4 homogeneous transform T_base_hand
        """
        q_arm = q_full[self.joint_indices]
        if self._use_pin:
            return self._fk_pin(q_full)
        return _fk_dh(q_arm, _DH_RIGHT, self._shoulder)

    def compute_arm(self, q_arm: np.ndarray) -> np.ndarray:
        """Compute from a 7-element arm-only joint vector."""
        return _fk_dh(q_arm, _DH_RIGHT, self._shoulder)

    # ------------------------------------------------------------------
    def _fk_pin(self, q_full: np.ndarray) -> np.ndarray:
        import pinocchio as pin  # type: ignore
        q_pin = pin.neutral(_pin_model)
        # Map the 30-DOF lowstate to pinocchio's nq (may differ — zero unmapped)
        nq = min(len(q_full), _pin_model.nq)
        q_pin[:nq] = q_full[:nq]
        pin.forwardKinematics(_pin_model, _pin_data, q_pin)
        pin.updateFramePlacements(_pin_model, _pin_data)
        frame_id = _pin_ee_frame_id[self.arm]
        SE3 = _pin_data.oMf[frame_id]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = SE3.rotation
        T[:3, 3] = SE3.translation
        return T

    # ------------------------------------------------------------------
    @staticmethod
    def joint_limits(arm: str = "right") -> List[Tuple[float, float]]:
        return JOINT_LIMITS[arm]

    @staticmethod
    def from_robot_sdk(robot, arm: str = "right") -> Tuple["ArmFK", np.ndarray]:
        """Convenience: construct FK and return current q_full from Robot SDK."""
        fk = ArmFK(arm=arm)
        js = robot.get_joint_states()
        joints = js.get("joints", {})
        q_full = np.zeros(30, dtype=np.float64)
        for name, data in joints.items():
            if "index" in data:
                idx = data["index"]
                if 0 <= idx < 30:
                    q_full[idx] = data.get("position", 0.0)
        return fk, q_full
