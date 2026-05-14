"""
arm_sdk.py — Arm SDK wrapper for the Unitree G1 (rt/arm_sdk)
=============================================================
Controls the upper body (waist + arms, joints 12-28) via rt/arm_sdk.
Does NOT require developer mode or MotionSwitcher release.

Quick start:
    from arm_sdk import ArmSdk
    sdk = ArmSdk(iface="eth0", domain_id=0)

    # Direct joint command (upper body only):
    sdk.move_joint({15: 0.3, 16: -0.1})

    # Cartesian EE increment (right arm, +2 cm along X):
    info = sdk.ik_move_EE([0.02, 0, 0, 0, 0, 0], arm="right")

    # Resync EE targets to current measured pose (call after physical disturbance):
    sdk.resync()
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
_MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.abspath(os.path.join(_MODULES_DIR, ".."))
for _p in (_ROOT_DIR, _MODULES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dds_env import ensure_cyclonedds_environment
ensure_cyclonedds_environment()

try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC
except ImportError as exc:
    raise SystemExit("unitree_sdk2py not installed.") from exc

from hand_pose_navigation_copy.arm_fk import (
    ArmFK, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, JOINT_LIMITS,
)
from hand_pose_navigation_copy.arm_ik import ArmIK

# ── Constants ─────────────────────────────────────────────────────────────────
WAIST_JOINTS      = [12, 13, 14]
UPPER_BODY_JOINTS = WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
ARM_SDK_WEIGHT_IDX = 29

_ARM_JOINTS: Dict[str, List[int]] = {
    "left":  LEFT_ARM_JOINTS,
    "right": RIGHT_ARM_JOINTS,
}

DEFAULT_ARM_KP   = 30.0
DEFAULT_ARM_KD   = 1.5
DEFAULT_WAIST_KP = 200.0
DEFAULT_WAIST_KD = 12.0

# IK joint weights: higher = more costly to move = prefers to stay still.
# Outer joints (wrist) are cheap; inner joints (shoulder) are expensive.
# shoulder_p/r/y, elbow, wrist_r/p/y
_DEFAULT_ARM_IK_WEIGHTS = np.array([4.0, 4.0, 4.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float64)


# ── Rotation helpers ──────────────────────────────────────────────────────────

def _Rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

def _Ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)

def _Rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def _apply_pose_increment(T: np.ndarray, inc: np.ndarray) -> np.ndarray:
    """Apply [dx, dy, dz, droll, dpitch, dyaw] to a 4×4 pose."""
    T_new = T.copy()
    T_new[:3, 3] += inc[:3]
    T_new[:3, :3] = _Rz(inc[5]) @ _Ry(inc[4]) @ _Rx(inc[3]) @ T_new[:3, :3]
    return T_new


def _mirror_inc(inc: np.ndarray) -> np.ndarray:
    """Flip y, roll, yaw — apply to right arm for bilateral symmetric motion.

    Follows the ik_pose_cli_v3 convention where left is the reference arm:
      y    → negated  (right arm moves outward when left moves outward)
      roll → negated  (wrists mirror each other)
      yaw  → negated  (wrists mirror each other)
    """
    m = inc.copy()
    m[1] = -inc[1]
    m[3] = -inc[3]
    m[5] = -inc[5]
    return m


def _clamp_q(q: np.ndarray, limits) -> np.ndarray:
    lo = np.array([l[0] for l in limits], dtype=np.float64)
    hi = np.array([l[1] for l in limits], dtype=np.float64)
    return np.clip(q, lo, hi)


def _pose_error(T_des: np.ndarray, T_cur: np.ndarray) -> np.ndarray:
    pos_err = T_des[:3, 3] - T_cur[:3, 3]
    R_err   = T_des[:3, :3] @ T_cur[:3, :3].T
    rot_err = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ]) * 0.5
    return np.concatenate([pos_err, rot_err])


def _solve_dls(
    fk_fn,
    T_des: np.ndarray,
    q_init: np.ndarray,
    limits,
    *,
    fixed: frozenset = frozenset(),
    weights: Optional[np.ndarray] = None,
    max_iter: int = 64,
    damping: float = 0.05,
    tol_pos: float = 0.005,
    tol_rot: float = 0.02,
) -> Tuple[Optional[np.ndarray], Dict]:
    """Weighted DLS IK with optional fixed joints.

    weights : per-joint cost vector — higher means the joint is penalised for
              moving, so the solver preferentially moves cheaper (outer) joints.
              Uses the weighted pseudoinverse:
                  dq = W⁻¹ Jᵀ (J W⁻¹ Jᵀ + λ²I)⁻¹ err
    fixed   : local joint indices held at q_init throughout the solve.
    """
    n     = len(q_init)
    q     = _clamp_q(q_init.copy(), limits)
    W_inv = np.ones(n, dtype=np.float64) if weights is None else 1.0 / np.asarray(weights, dtype=np.float64)
    eps   = 1e-5
    for it in range(max_iter):
        T_cur = fk_fn(q)
        err   = _pose_error(T_des, T_cur)
        ep    = float(np.linalg.norm(err[:3]))
        er    = float(np.linalg.norm(err[3:]))
        if ep < tol_pos and er < tol_rot:
            return q, {"success": True, "error_pos_m": ep, "error_rot_rad": er, "iterations": it}
        J  = np.zeros((6, n), dtype=np.float64)
        p0 = T_cur[:3, 3]
        R0 = T_cur[:3, :3]
        for col in range(n):
            if col in fixed:
                continue
            q1 = q.copy(); q1[col] += eps
            T1 = fk_fn(q1)
            J[:3, col] = (T1[:3, 3] - p0) / eps
            dR = T1[:3, :3] @ R0.T
            J[3:, col] = np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]]) / (2*eps)
        # Weighted step: dq = W⁻¹ Jᵀ (J W⁻¹ Jᵀ + λ²I)⁻¹ err
        A   = J * W_inv[np.newaxis, :]          # J @ diag(W_inv), shape (6, n)
        dq  = W_inv * (J.T @ np.linalg.solve(A @ J.T + damping**2 * np.eye(6), err))
        norm_dq = float(np.linalg.norm(dq))
        if norm_dq > 0.3:
            dq *= 0.3 / norm_dq
        q = _clamp_q(q + dq, limits)
        for i in fixed:
            q[i] = float(q_init[i])
    T_cur = fk_fn(q)
    err   = _pose_error(T_des, T_cur)
    return None, {
        "success": False,
        "error_pos_m":   float(np.linalg.norm(err[:3])),
        "error_rot_rad": float(np.linalg.norm(err[3:])),
        "iterations": max_iter,
    }


# ── Low-state subscriber (upper body only) ────────────────────────────────────

class _UpperBodyStateReader:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._q: Optional[Dict[int, float]] = None
        try:
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as _LS
        except ImportError:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as _LS
        sub = ChannelSubscriber("rt/lowstate", _LS)
        sub.Init(self._cb, 200)

    def _cb(self, msg) -> None:
        try:
            q = {j: float(msg.motor_state[j].q) for j in UPPER_BODY_JOINTS}
        except Exception:
            return
        with self._lock:
            self._q = q

    def snapshot(self) -> Optional[Dict[int, float]]:
        with self._lock:
            return dict(self._q) if self._q is not None else None

    def wait(self, timeout: float = 3.0) -> Dict[int, float]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snap = self.snapshot()
            if snap is not None:
                return snap
            time.sleep(0.02)
        raise TimeoutError("Timed out waiting for rt/lowstate.")


# ── rt/arm_sdk publisher ──────────────────────────────────────────────────────

class _ArmSdkPublisher:
    def __init__(self) -> None:
        self._crc = CRC()
        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._cmd.mode_pr      = 0
        self._cmd.mode_machine = 0
        self._cmd.motor_cmd[ARM_SDK_WEIGHT_IDX].q = 1.0

    def publish(
        self,
        targets: Dict[int, float],
        *,
        arm_kp: float,
        arm_kd: float,
        waist_kp: float,
        waist_kd: float,
    ) -> None:
        for j in UPPER_BODY_JOINTS:
            mc      = self._cmd.motor_cmd[j]
            mc.mode = 1
            mc.q    = float(targets[j])
            mc.dq   = 0.0
            mc.tau  = 0.0
            mc.kp   = float(waist_kp if j in WAIST_JOINTS else arm_kp)
            mc.kd   = float(waist_kd if j in WAIST_JOINTS else arm_kd)
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)


# ── Main SDK ──────────────────────────────────────────────────────────────────

class ArmSdk:
    """
    Upper-body controller via rt/arm_sdk.

    Parameters
    ----------
    iface : str
        Network interface (e.g. "eth0").
    domain_id : int
        DDS domain ID.
    arm_kp, arm_kd : float
        Servo gains for arm joints.
    waist_kp, waist_kd : float
        Servo gains for waist joints.
    arm_ik_max_iter : int
        DLS IK iteration limit.
    """

    def __init__(
        self,
        iface: str = "eth0",
        domain_id: int = 0,
        arm_kp: float   = DEFAULT_ARM_KP,
        arm_kd: float   = DEFAULT_ARM_KD,
        waist_kp: float = DEFAULT_WAIST_KP,
        waist_kd: float = DEFAULT_WAIST_KD,
        arm_ik_max_iter: int = 24,
    ) -> None:
        self.arm_kp   = float(arm_kp)
        self.arm_kd   = float(arm_kd)
        self.waist_kp = float(waist_kp)
        self.waist_kd = float(waist_kd)

        ChannelFactoryInitialize(int(domain_id), str(iface))

        self._state = _UpperBodyStateReader()
        self._pub   = _ArmSdkPublisher()

        self._fk: Dict[str, ArmFK] = {
            "left":  ArmFK("left",  "urdf"),
            "right": ArmFK("right", "urdf"),
        }
        self._ik: Dict[str, ArmIK] = {
            "left":  ArmIK("left",  "dls", max_iter=arm_ik_max_iter, tol_pos_m=0.005, tol_rot_rad=0.02),
            "right": ArmIK("right", "dls", max_iter=arm_ik_max_iter, tol_pos_m=0.005, tol_rot_rad=0.02),
        }

        # Last commanded joint targets — seeded on first use (needed by move_joint).
        self._desired: Optional[Dict[int, float]] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _seed(self, timeout: float = 3.0) -> None:
        """Seed desired joint targets from current lowstate if not yet done."""
        if self._desired is not None:
            return
        q = self._state.wait(timeout)
        self._desired = dict(q)

    def _publish(self) -> None:
        assert self._desired is not None
        self._pub.publish(
            self._desired,
            arm_kp=self.arm_kp,
            arm_kd=self.arm_kd,
            waist_kp=self.waist_kp,
            waist_kd=self.waist_kd,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def resync(self, timeout: float = 3.0) -> None:
        """Resync desired joint targets to the current measured state.

        Call this after a physical disturbance or after loading a pose, so that
        subsequent move_joint calls hold the correct positions for unset joints.
        """
        q = self._state.wait(timeout)
        self._desired = dict(q)

    def move_joint(
        self,
        targets: Dict[int, float],
        *,
        timeout: float = 3.0,
    ) -> None:
        """Send one rt/arm_sdk packet with explicit joint targets.

        Parameters
        ----------
        targets : dict {joint_index: q_rad}
            Upper-body joints to command (indices 12-28).
            Unspecified joints are held at the last desired position.
        """
        self._seed(timeout)
        assert self._desired is not None
        for idx, val in targets.items():
            self._desired[int(idx)] = float(val)
        # Resync EE targets for any arm whose joints changed.
        for arm, joints in _ARM_JOINTS.items():
            if any(j in targets for j in joints):
                q_arm = self._q_arm(arm)
                self._target_T[arm] = self._fk[arm].compute_arm(q_arm).copy()
        self._publish()

    def ik_move_EE(
        self,
        pose_increment: "np.ndarray | List[float]",
        arm: str = "right",
        *,
        mirror: bool = True,
        fixed_joints: "Optional[List[int]]" = None,
        joint_weights: "Optional[np.ndarray]" = None,
        max_dq: float = 0.2,
        timeout: float = 3.0,
    ) -> Dict:
        """Move one (or both) arm end-effector(s) by a Cartesian increment.

        Warm-started from the last commanded joint state so repeated small
        increments accumulate correctly even if lowstate lags the command.

        Parameters
        ----------
        pose_increment : array-like [dx, dy, dz, droll, dpitch, dyaw]
            Increment in base_link frame (metres / radians).
        arm : "left" | "right" | "both"
        mirror : bool
            When arm="both", flip y/roll/yaw for the right arm so the motion
            is bilaterally symmetric (left is the reference). Default True.
        fixed_joints : list of int, optional
            Local joint indices (0-6) to hold fixed during the IK solve.
            e.g. [4, 5, 6] locks all three wrist joints.
        joint_weights : array-like of shape (7,), optional
            Per-joint cost weights.  Higher weight → joint moves less.
            Default: [4,4,4,2,1,1,1] — shoulder/hip costly, wrist/ankle cheap.
            Pass np.ones(7) for uniform (classic DLS) behaviour.
        max_dq : float
            Per-joint safety clamp on the IK step (rad).

        Returns
        -------
        dict with keys: success, error_pos_m, error_rot_rad, iterations.
        For "both" arms, returns the result of the last arm solved.
        """
        inc = np.asarray(pose_increment, dtype=np.float64)
        if inc.shape != (6,):
            raise ValueError(f"pose_increment must have 6 elements, got {inc.shape}")

        arms  = ["left", "right"] if arm == "both" else [str(arm)]
        fixed = frozenset(int(i) for i in fixed_joints) if fixed_joints else frozenset()
        w     = _DEFAULT_ARM_IK_WEIGHTS if joint_weights is None else np.asarray(joint_weights, dtype=np.float64)

        self._seed(timeout)
        assert self._desired is not None
        q_state = self._state.wait(timeout)  # fresh measured state for IK init

        info: Dict = {"success": False, "error_pos_m": 0.0, "error_rot_rad": 0.0, "iterations": 0}

        for a in arms:
            arm_inc = _mirror_inc(inc) if (mirror and arm == "both" and a == "right") else inc
            joints  = _ARM_JOINTS[a]
            limits  = JOINT_LIMITS[a]
            q_arm   = np.array([q_state[j] for j in joints])
            T_cur   = self._fk[a].compute_arm(q_arm)

            scales = (1.0, 0.5, 0.25, 0.1)
            solved = False
            for scale in scales:
                T_scaled = _apply_pose_increment(T_cur, arm_inc * scale)
                q_sol, info = _solve_dls(
                    self._fk[a].compute_arm, T_scaled, q_arm, limits,
                    fixed=fixed, weights=w,
                    max_iter=self._ik[a].max_iter,
                    damping=self._ik[a].damping,
                    tol_pos=self._ik[a].tol_pos_m,
                    tol_rot=self._ik[a].tol_rot_rad,
                )
                if q_sol is None:
                    continue

                delta = np.clip(q_sol - q_arm, -max_dq, max_dq)
                q_new = np.clip(q_arm + delta,
                                np.array([l[0] for l in limits]),
                                np.array([l[1] for l in limits]))
                for i, j in enumerate(joints):
                    self._desired[j] = float(q_new[i])
                solved = True
                break

        self._publish()
        return info

    def get_ee_pose(self, arm: str = "right", timeout: float = 3.0) -> np.ndarray:
        """Return the current measured 4×4 EE pose in base_link."""
        q_state = self._state.wait(timeout)
        q_arm   = np.array([q_state[j] for j in _ARM_JOINTS[arm]])
        return self._fk[arm].compute_arm(q_arm)

    def get_joint_targets(self) -> Dict[int, float]:
        """Return the current desired upper-body joint targets."""
        self._seed()
        assert self._desired is not None
        return dict(self._desired)
