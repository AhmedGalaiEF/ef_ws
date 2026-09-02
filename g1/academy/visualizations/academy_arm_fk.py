"""
Step 4 — Forward kinematics: joint states -> T_base_hand
==========================================================
Computes the hand (end-effector) pose in base_link coordinates given a
vector of arm joint angles.

Three backends, selected at construction time:
    urdf       — direct URDF chain (exact geometry, no extra deps) [default]
    pinocchio  — pinocchio URDF-based FK (also exact, requires pip install pin)
    dh         — legacy analytical DH approximation (kept for reference)

URDF data sourced from:
    g1_29dof_with_hand_rev_1_0_pkg.urdf  (G1 rviz simulation package)

Shoulder origins in base_link (at zero waist angles):
    right: [ 0.000, -0.100,  0.292]
    left:  [ 0.000, +0.100,  0.292]
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# G1 arm joint indices in the 30-joint low-state array
# --------------------------------------------------------------------------

LEFT_ARM_JOINTS = [15, 16, 17, 18, 19, 20, 21]  # shoulder p/r/y, elbow, wrist r/p/y
RIGHT_ARM_JOINTS = [22, 23, 24, 25, 26, 27, 28]

# Joint limits [rad] — from URDF
JOINT_LIMITS: Dict[str, List[Tuple[float, float]]] = {
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

# --------------------------------------------------------------------------
# URDF-exact kinematic chain
# --------------------------------------------------------------------------

# torso_link origin in base_link at zero waist angles
# (cumulative: waist_roll_joint xyz=[-0.003964, 0, 0.044], waist_pitch xyz=[0,0,0])
_TORSO_IN_BASE = np.array([-0.003964, 0.0, 0.044], dtype=np.float64)

# Shoulder origins in base_link (torso_in_base + shoulder_pitch xyz)
_RIGHT_SHOULDER_IN_BASE = np.array([0.0, -0.100, 0.292], dtype=np.float64)
_LEFT_SHOULDER_IN_BASE = np.array([0.0, 0.100, 0.292], dtype=np.float64)

# Public: shoulder origins, keyed by side — used by obstacle_checker.py to
# build the opposite-arm self-collision capsule (shoulder -> live wrist).
SHOULDER_IN_BASE: Dict[str, np.ndarray] = {
    "right": _RIGHT_SHOULDER_IN_BASE,
    "left": _LEFT_SHOULDER_IN_BASE,
}

# Kinematic chain: one entry per joint (shoulder_pitch … wrist_yaw)
# Format: (xyz_in_parent, rpy_of_link_frame, joint_axis)
# Sourced verbatim from g1_29dof_with_hand_rev_1_0_pkg.urdf
_URDF_CHAIN: Dict[str, List] = {
    "right": [
        ([0.003956, -0.10021, 0.24778], [-0.27931, 0.0, 0.0], [0, 1, 0]),  # shoulder_pitch
        ([0.0, -0.038, -0.013831], [0.27925, 0.0, 0.0], [1, 0, 0]),  # shoulder_roll
        ([0.0, -0.00624, -0.1032], [0.0, 0.0, 0.0], [0, 0, 1]),  # shoulder_yaw
        ([0.015783, 0.0, -0.080518], [0.0, 0.0, 0.0], [0, 1, 0]),  # elbow
        ([0.100, -0.001888, -0.010], [0.0, 0.0, 0.0], [1, 0, 0]),  # wrist_roll
        ([0.038, 0.0, 0.0], [0.0, 0.0, 0.0], [0, 1, 0]),  # wrist_pitch
        ([0.046, 0.0, 0.0], [0.0, 0.0, 0.0], [0, 0, 1]),  # wrist_yaw
    ],
    "left": [
        ([0.003956, 0.10022, 0.24778], [0.27931, 0.0, 0.0], [0, 1, 0]),  # shoulder_pitch
        ([0.0, 0.038, -0.013831], [-0.27925, 0.0, 0.0], [1, 0, 0]),  # shoulder_roll
        ([0.0, 0.00624, -0.1032], [0.0, 0.0, 0.0], [0, 0, 1]),  # shoulder_yaw
        ([0.015783, 0.0, -0.080518], [0.0, 0.0, 0.0], [0, 1, 0]),  # elbow
        ([0.100, 0.001888, -0.010], [0.0, 0.0, 0.0], [1, 0, 0]),  # wrist_roll
        ([0.038, 0.0, 0.0], [0.0, 0.0, 0.0], [0, 1, 0]),  # wrist_pitch
        ([0.046, 0.0, 0.0], [0.0, 0.0, 0.0], [0, 0, 1]),  # wrist_yaw
    ],
}


def _T_from_xyz_rpy(xyz, rpy) -> np.ndarray:
    """4×4 transform: translate by xyz then rotate by rpy (URDF: Rz@Ry@Rx)."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def _T_from_axis_q(axis, q: float) -> np.ndarray:
    """4×4 rotation about unit axis by angle q (Rodrigues formula)."""
    ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
    K = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + np.sin(q) * K + (1.0 - np.cos(q)) * (K @ K)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


# _URDF_CHAIN above ends at the wrist_yaw JOINT's own frame — it does not
# include the fixed wrist_yaw_link -> hand_palm_link joint, so without this
# offset the "hand" pose this module returns is actually the wrist. The URDF
# palm offset was 4.15cm; this tuned value brings the frame 2cm closer to the
# wrist while preserving the small lateral sign flip by side.
_HAND_PALM_OFFSET: Dict[str, Tuple[List[float], List[float]]] = {
    "right": ([0.0215, -0.003, 0.0], [0.0, 0.0, 0.0]),
    "left": ([0.0215, 0.003, 0.0], [0.0, 0.0, 0.0]),
}


def _fk_urdf(q_arm: np.ndarray, arm: str) -> np.ndarray:
    """URDF-exact 7-DOF FK from base_link origin; returns 4×4 T_base_hand
    (hand = hand_palm_link, i.e. wrist_yaw_link plus the fixed palm offset).
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = _TORSO_IN_BASE.copy()
    for i, (xyz, rpy, axis) in enumerate(_URDF_CHAIN[arm]):
        T = T @ _T_from_xyz_rpy(xyz, rpy) @ _T_from_axis_q(axis, float(q_arm[i]))
    palm_xyz, palm_rpy = _HAND_PALM_OFFSET[arm]
    return T @ _T_from_xyz_rpy(palm_xyz, palm_rpy)


def _fk_urdf_partial(q_arm: np.ndarray, arm: str, n_joints: int) -> np.ndarray:
    """URDF FK stopped after n_joints steps; returns 4×4 transform at that frame.

    n_joints=0 → torso frame
    n_joints=1 → after shoulder_pitch applied
    n_joints=4 → after elbow applied  (elbow sphere centre)
    n_joints=5 → after wrist_roll applied
    n_joints=7 → full FK (== _fk_urdf)
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = _TORSO_IN_BASE.copy()
    chain = _URDF_CHAIN[arm]
    for i in range(min(n_joints, len(chain))):
        xyz, rpy, axis = chain[i]
        T = T @ _T_from_xyz_rpy(xyz, rpy) @ _T_from_axis_q(axis, float(q_arm[i]))
    return T


# --------------------------------------------------------------------------
# Legacy DH approximation (kept for reference / backward-compat)
# --------------------------------------------------------------------------

_DH_RIGHT = np.array([
    # a       d       alpha       theta_off
    [0.000, 0.000, -np.pi / 2, 0.000],  # J1 shoulder pitch
    [0.000, 0.000, np.pi / 2, 0.000],  # J2 shoulder roll
    [0.000, 0.200, -np.pi / 2, 0.000],  # J3 shoulder yaw  (upper arm ~0.20m)
    [0.000, 0.000, np.pi / 2, 0.000],  # J4 elbow
    [0.000, 0.185, -np.pi / 2, 0.000],  # J5 wrist roll    (forearm ~0.185m)
    [0.000, 0.000, np.pi / 2, 0.000],  # J6 wrist pitch
    [0.000, 0.084, 0.000, 0.000],  # J7 wrist yaw     (hand ~0.084m)
], dtype=np.float64)


def _dh_matrix(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1],
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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "install/g1_description/share/g1_description/urdf/g1_29dof_with_hand_rev_1_0_pkg.urdf",
)
_EE_FRAME_NAMES = {
    "right": "right_hand_palm_link",
    "left": "left_hand_palm_link",
}


def _try_load_pinocchio() -> bool:
    """The academy copy intentionally has no optional Pinocchio dependency."""
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
        backend: "auto" | "urdf" | "pinocchio" | "dh"

    Backend priority when "auto":
        pinocchio (if available) > urdf > dh
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
            if _USE_PIN and arm in _pin_ee_frame_id:
                self._backend = "pinocchio"
            else:
                self._backend = "urdf"
        else:
            self._backend = backend

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
        if self._backend == "pinocchio":
            return self._fk_pin(q_full)
        if self._backend == "urdf":
            return _fk_urdf(q_arm, self.arm)
        return _fk_dh(q_arm, _DH_RIGHT, self._shoulder)

    def compute_arm(self, q_arm: np.ndarray) -> np.ndarray:
        """Compute from a 7-element arm-only joint vector."""
        if self._backend == "urdf":
            return _fk_urdf(q_arm, self.arm)
        if self._backend == "pinocchio":
            # pinocchio needs full q — fall back to urdf for arm-only input
            return _fk_urdf(q_arm, self.arm)
        return _fk_dh(q_arm, _DH_RIGHT, self._shoulder)

    def compute_arm_partial(self, q_arm: np.ndarray, n_joints: int) -> np.ndarray:
        """URDF-chain FK stopped after n_joints (e.g. 4 = elbow, 7 = hand).

        Backend-independent (always uses the exact URDF chain) — intended
        for cheap collision-proxy points (elbow/wrist), not for IK targets.
        """
        return _fk_urdf_partial(q_arm, self.arm, n_joints)

    # ------------------------------------------------------------------
    def _fk_pin(self, q_full: np.ndarray) -> np.ndarray:
        raise RuntimeError("Pinocchio is not included in the academy kinematics module")

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
