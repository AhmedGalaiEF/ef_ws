#!/usr/bin/env python3
"""
6D EE IK Pose Control TUI — Smart Edition.

Extends ik_pose_cli.py with three layers on top of the base DLS IK:

  1. Rest-pose regularisation
       The exact joint configuration at reengage time (r → e) is captured
       as the "rest pose".  A null-space term in the IK pulls every joint
       back toward this configuration, avoiding arbitrary elbow flips and
       keeping the arm in a natural looking posture.

  2. Sphere-model collision avoidance
       Each arm is modelled as five spheres (shoulder, upper-arm, elbow,
       forearm, hand).  A repulsive potential in the IK null space pushes
       spheres away from:
         • user-defined obstacle spheres (added at runtime)
         • the opposite arm's spheres (in "both" mode)
       A separate hard-overlap check (no margin) flags actual collisions.

  3. Trajectory planning
       On each commanded move the desired joint path is linearly interpolated
       into N_TRAJ_STEPS waypoints and each is checked for hard collisions.
       If any waypoint collides the path is automatically rerouted through
       the rest pose.  Waypoints are queued; tick() advances the queue so
       the ramp always moves toward the next safe configuration.

Key bindings (same as ik_pose_cli.py, plus)
────────────────────────────────────────────
  ↑ / ↓  or  k / j    select DOF
  ← / →  or  - / +    decrement / increment selected DOF
  < / >                halve / double EE step
  [ / ]                halve / double max_dq
  m                    cycle arm mode
  y                    sync to current FK pose
  r                    release arms
  e                    reengage + capture rest pose
  z                    zero-gain hold
  s                    set ramp speed (prompt)
  d                    set max_dq (prompt)
  o                    add sphere obstacle  (prompt: x y z r)
  O                    clear all user obstacles
  q / Esc              quit
"""
from __future__ import annotations

import argparse
import curses
import math
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODULES_DIR = os.path.join(ROOT_DIR, "modules")
for _p in (ROOT_DIR, MODULES_DIR):
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

from sdk_client import Robot

from hand_pose_navigation_copy.arm_fk import (
    ArmFK, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS,
    JOINT_LIMITS,
    _DH_RIGHT, _dh_matrix,
    _LEFT_SHOULDER_IN_BASE, _RIGHT_SHOULDER_IN_BASE,
)
from hand_pose_navigation_copy.arm_ik import _pose_error, _numerical_jacobian

# ── Joint indices ─────────────────────────────────────────────────────────────
WAIST_JOINTS      = [12, 13, 14]
UPPER_BODY_JOINTS = WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

ARM_SDK_WEIGHT_INDEX = 29
WAIST_HOLD_KP        = 480.0
WAIST_HOLD_KD        = 12.0
DEFAULT_ARM_KP       = 30.0
DEFAULT_ARM_KD       = 1.5

ARM_CONTROL_MODES = ("both", "left", "right")
ARM_JOINTS: Dict[str, List[int]] = {
    "left":  LEFT_ARM_JOINTS,
    "right": RIGHT_ARM_JOINTS,
}
_SHOULDER_ORIGIN: Dict[str, np.ndarray] = {
    "left":  _LEFT_SHOULDER_IN_BASE,
    "right": _RIGHT_SHOULDER_IN_BASE,
}
JOINT_LABELS = ("sh_p", "sh_r", "sh_y", "elbow", "wr_r", "wr_p", "wr_y")

# ── DOF table ─────────────────────────────────────────────────────────────────
DOF_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
DOF_UNITS = ("m",  "m", "m", "rad",  "rad",   "rad")
N_DOFS    = 6

# ── Colour pairs ──────────────────────────────────────────────────────────────
C_GREEN  = 1
C_YELLOW = 2
C_RED    = 3
C_CYAN   = 4
C_SEL    = 5
C_BOLD   = 6

# ── Collision model parameters ────────────────────────────────────────────────
# Five spheres per arm: shoulder, upper-arm mid, elbow, forearm mid, hand
_ARM_SPHERE_RADII = (0.08, 0.06, 0.07, 0.05, 0.06)   # metres

# Repulsive potential: activates when gap < COLL_D_SAFE
COLL_D_SAFE   = 0.06   # metres
COLL_K_REP    = 3.0    # repulsive gain
COLL_NS_FREQ  = 5      # recompute repulsive gradient every N IK iterations

# Trajectory planner
N_TRAJ_STEPS   = 15    # intermediate waypoints to collision-check
ARRIVE_EPSILON = 0.02  # rad — per-joint tolerance to advance waypoint queue


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

_ROT_BY_AXIS = (_Rx, _Ry, _Rz)


def _rpy_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll  = math.atan2( R[2,1],  R[2,2])
        pitch = math.atan2(-R[2,0],  sy)
        yaw   = math.atan2( R[1,0],  R[0,0])
    else:
        roll  = math.atan2(-R[1,2],  R[1,1])
        pitch = math.atan2(-R[2,0],  sy)
        yaw   = 0.0
    return roll, pitch, yaw


# ── Sphere collision model ────────────────────────────────────────────────────

def _arm_sphere_centers(q_arm: np.ndarray, arm: str) -> List[np.ndarray]:
    """Five sphere centres for the arm via partial DH forward kinematics."""
    shoulder = _SHOULDER_ORIGIN[arm]
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = shoulder.copy()
    pts: List[np.ndarray] = [shoulder.copy()]   # pts[0] = shoulder
    for i, (a, d, alpha, theta_off) in enumerate(_DH_RIGHT):
        T = T @ _dh_matrix(a, d, alpha, q_arm[i] + theta_off)
        pts.append(T[:3, 3].copy())
    # pts[0]=shoulder, pts[3]=elbow, pts[5]=wrist, pts[7]=hand
    return [
        pts[0],                         # shoulder
        (pts[0] + pts[3]) * 0.5,        # upper-arm mid
        pts[3],                         # elbow
        (pts[3] + pts[5]) * 0.5,        # forearm mid
        pts[7],                         # hand
    ]


def _gap(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> float:
    """Surface-to-surface distance (negative = overlap)."""
    return float(np.linalg.norm(c1 - c2)) - r1 - r2


def _arm_in_hard_collision(
    q_arm: np.ndarray,
    arm: str,
    user_obs: List[Tuple],
    other_q: Optional[np.ndarray],
) -> bool:
    """True if any arm sphere overlaps (gap < 0) with obstacles or the other arm."""
    centers = _arm_sphere_centers(q_arm, arm)

    for obs in user_obs:
        oc = np.array(obs[:3], dtype=np.float64)
        or_ = float(obs[3])
        for ac, ar in zip(centers, _ARM_SPHERE_RADII):
            if _gap(ac, ar, oc, or_) < 0.0:
                return True

    if other_q is not None:
        other = "right" if arm == "left" else "left"
        for oc, or_ in zip(_arm_sphere_centers(other_q, other), _ARM_SPHERE_RADII):
            for ac, ar in zip(centers, _ARM_SPHERE_RADII):
                if _gap(ac, ar, oc, or_) < 0.0:
                    return True

    return False


def _repulsive_potential(
    q_arm: np.ndarray,
    arm: str,
    user_obs: List[Tuple],
    other_q: Optional[np.ndarray],
) -> float:
    """Sum of repulsive potentials that activate within COLL_D_SAFE of any obstacle."""
    centers = _arm_sphere_centers(q_arm, arm)
    V = 0.0

    all_obs: List[Tuple[np.ndarray, float]] = [
        (np.array(o[:3], dtype=np.float64), float(o[3])) for o in user_obs
    ]
    if other_q is not None:
        other = "right" if arm == "left" else "left"
        all_obs += list(zip(_arm_sphere_centers(other_q, other), _ARM_SPHERE_RADII))

    for ac, ar in zip(centers, _ARM_SPHERE_RADII):
        for oc, or_ in all_obs:
            d = _gap(ac, ar, oc, or_)
            if d < COLL_D_SAFE:
                d = max(d, 1e-3)
                V += COLL_K_REP * (1.0 / d - 1.0 / COLL_D_SAFE) ** 2 * 0.5
    return V


def _repulsive_gradient(
    q_arm: np.ndarray,
    arm: str,
    user_obs: List[Tuple],
    other_q: Optional[np.ndarray],
    eps: float = 1e-4,
) -> np.ndarray:
    """Finite-difference gradient of the repulsive potential w.r.t. joint angles."""
    V0 = _repulsive_potential(q_arm, arm, user_obs, other_q)
    grad = np.zeros(7)
    for i in range(7):
        qp = q_arm.copy(); qp[i] += eps
        grad[i] = (_repulsive_potential(qp, arm, user_obs, other_q) - V0) / eps
    return grad


# ── Smart IK solver ───────────────────────────────────────────────────────────

class SmartArmIK:
    """
    DLS inverse kinematics with two null-space secondary objectives:

      1. Rest-pose regularisation  — pulls joints toward the rest pose
         captured at reengage time (weight reg_weight).
      2. Collision repulsion       — pushes arm spheres away from obstacles
         and the other arm (weight rep_weight).

    Both secondaries are projected into the IK null space so they cannot
    disturb end-effector tracking unless the EE is already at its target.
    """

    def __init__(
        self,
        arm: str,
        max_iter: int = 80,
        tol_pos_m: float = 0.005,
        tol_rot_rad: float = 0.02,
        damping: float = 0.05,
        reg_weight: float = 0.05,
        rep_weight: float = 0.8,
    ) -> None:
        self.arm        = arm
        self.max_iter   = max_iter
        self.tol_pos    = tol_pos_m
        self.tol_rot    = tol_rot_rad
        self.damping    = damping
        self.reg_weight = reg_weight
        self.rep_weight = rep_weight
        self._fk        = ArmFK(arm, "dh")
        lo = np.array([lim[0] for lim in JOINT_LIMITS[arm]])
        hi = np.array([lim[1] for lim in JOINT_LIMITS[arm]])
        self._lo = lo
        self._hi = hi
        self._last_rep_grad: Optional[np.ndarray] = None

    def solve(
        self,
        T_des: np.ndarray,
        q_init: np.ndarray,
        rest_q: Optional[np.ndarray] = None,
        user_obs: Optional[List[Tuple]] = None,
        other_q: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        obs = user_obs or []
        q = np.clip(q_init.copy(), self._lo, self._hi)
        lam = self.damping

        for it in range(self.max_iter):
            T_cur = self._fk.compute_arm(q)
            err   = _pose_error(T_des, T_cur)
            ep    = float(np.linalg.norm(err[:3]))
            er    = float(np.linalg.norm(err[3:]))

            if ep < self.tol_pos and er < self.tol_rot:
                return q, {"success": True,
                           "error_pos_m": ep, "error_rot_rad": er,
                           "iterations": it}

            # ── Primary EE motion (DLS) ─────────────────────────────────────
            J    = _numerical_jacobian(q, self._fk)          # 6×7
            JJT  = J @ J.T                                    # 6×6
            Jreg = np.linalg.solve(JJT + lam**2 * np.eye(6), np.eye(6))
            J_pv = J.T @ Jreg                                 # 7×6  (DLS pseudo-inv)

            dq_EE = J_pv @ err
            scale = min(0.1, 0.05 / (float(np.linalg.norm(dq_EE)) + 1e-8))
            dq    = scale * dq_EE

            # ── Null-space projector  N = I - J⁺J ──────────────────────────
            N = np.eye(7) - J_pv @ J                          # 7×7

            # ── Secondary 1: rest-pose regularisation ───────────────────────
            if rest_q is not None and self.reg_weight > 0.0:
                dq += N @ (self.reg_weight * (rest_q - q))

            # ── Secondary 2: collision repulsion (cached every COLL_NS_FREQ) ─
            if self.rep_weight > 0.0 and (obs or other_q is not None):
                if it % COLL_NS_FREQ == 0:
                    self._last_rep_grad = _repulsive_gradient(
                        q, self.arm, obs, other_q
                    )
                if self._last_rep_grad is not None:
                    dq -= N @ (self.rep_weight * self._last_rep_grad)

            # ── Safety cap on total step ────────────────────────────────────
            norm_dq = float(np.linalg.norm(dq))
            if norm_dq > 0.15:
                dq *= 0.15 / norm_dq

            q = np.clip(q + dq, self._lo, self._hi)

        T_cur = self._fk.compute_arm(q)
        err   = _pose_error(T_des, T_cur)
        return None, {
            "success": False,
            "error_pos_m":    float(np.linalg.norm(err[:3])),
            "error_rot_rad":  float(np.linalg.norm(err[3:])),
            "iterations":     self.max_iter,
        }


# ── Trajectory planner ────────────────────────────────────────────────────────

class TrajectoryPlanner:
    """
    Plans a collision-free joint-space path from q_start to q_end.

    Strategy
    --------
    1. Linearly interpolate N_TRAJ_STEPS intermediate configurations.
    2. Hard-collision-check each one.
    3. If all clear → return [q_end].
    4. If any collides → reroute through rest_q (or zero config as fallback).
    """

    @staticmethod
    def plan(
        q_start: np.ndarray,
        q_end:   np.ndarray,
        arm: str,
        rest_q:   Optional[np.ndarray],
        user_obs: List[Tuple],
        other_q:  Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        ts = np.linspace(0.0, 1.0, N_TRAJ_STEPS + 2)[1:-1]

        for t in ts:
            q_mid = q_start + t * (q_end - q_start)
            if _arm_in_hard_collision(q_mid, arm, user_obs, other_q):
                return TrajectoryPlanner._reroute(
                    q_start, q_end, arm, rest_q, user_obs, other_q
                )
        return [q_end]

    @staticmethod
    def _reroute(
        q_start: np.ndarray,
        q_end:   np.ndarray,
        arm: str,
        rest_q:   Optional[np.ndarray],
        user_obs: List[Tuple],
        other_q:  Optional[np.ndarray],
    ) -> List[np.ndarray]:
        # Pick intermediate: rest pose if collision-free, else zero config
        if rest_q is not None and not _arm_in_hard_collision(
            rest_q, arm, user_obs, other_q
        ):
            q_mid = rest_q.copy()
        else:
            q_mid = np.zeros(7)

        half = N_TRAJ_STEPS // 2
        ts   = np.linspace(0.0, 1.0, half + 2)[1:]

        seg1 = [q_start + t * (q_mid - q_start) for t in ts[:-1]] + [q_mid.copy()]
        seg2 = [q_mid   + t * (q_end  - q_mid)  for t in ts[:-1]] + [q_end.copy()]
        return seg1 + seg2


# ── Robot infrastructure (identical to ik_pose_cli.py) ───────────────────────

def _resolve_lowstate_type():
    for path in (
        "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "unitree_sdk2py.idl.unitree_go.msg.dds_",
    ):
        try:
            mod = __import__(path, fromlist=["LowState_"])
            if hasattr(mod, "LowState_"):
                return getattr(mod, "LowState_")
        except Exception:
            pass
    return None


class UpperBodyStateSubscriber:
    def __init__(self, joints: List[int]) -> None:
        self._joints = [int(j) for j in joints]
        self._lock   = threading.Lock()
        self._pos: Dict[int, float] = {}
        self._ts  = 0.0
        t = _resolve_lowstate_type()
        if t is None:
            raise RuntimeError("LowState_ not found.")
        sub = ChannelSubscriber("rt/lowstate", t)
        sub.Init(self._on_msg, 200)

    def _on_msg(self, msg: Any) -> None:
        try:
            pos = {j: float(msg.motor_state[j].q) for j in self._joints}
        except Exception:
            return
        with self._lock:
            self._pos = pos
            self._ts  = time.time()

    def snapshot(self) -> Optional[Tuple[Dict[int, float], float]]:
        with self._lock:
            if not self._pos:
                return None
            return dict(self._pos), float(self._ts)


class ArmSDKPublisher:
    def __init__(self) -> None:
        self._crc = CRC()
        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._cmd.mode_pr = 0
        self._cmd.mode_machine = 0
        self._cmd.motor_cmd[ARM_SDK_WEIGHT_INDEX].q = 1.0

    def publish(self, targets: Dict[int, float], *,
                arm_kp: float, arm_kd: float,
                waist_kp: float, waist_kd: float) -> None:
        for j in UPPER_BODY_JOINTS:
            c = self._cmd.motor_cmd[j]
            c.mode = 1;  c.q = float(targets[j]);  c.dq = 0.0;  c.tau = 0.0
            if j in WAIST_JOINTS:
                c.kp = float(waist_kp);  c.kd = float(waist_kd)
            else:
                c.kp = float(arm_kp);    c.kd = float(arm_kd)
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def publish_zero_gains(self, hold: Dict[int, float]) -> None:
        for j in UPPER_BODY_JOINTS:
            c = self._cmd.motor_cmd[j]
            c.mode = 1;  c.q = float(hold[j]);  c.dq = 0.0
            c.kp = 0.0;  c.kd = 0.0;  c.tau = 0.0
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)


# ── Main TUI ──────────────────────────────────────────────────────────────────

class IKPoseCLISmart:
    def __init__(self, args: argparse.Namespace) -> None:
        self.iface            = str(args.iface)
        self.domain_id        = int(args.domain_id)
        self.rate_hz          = max(1.0, float(args.rate_hz))
        self.max_speed        = max(0.01, float(args.speed_rad_s))
        self.arm_kp           = float(args.kp)
        self.arm_kd           = float(args.kd)
        self.waist_kp         = float(WAIST_HOLD_KP)
        self.waist_kd         = float(WAIST_HOLD_KD)
        self.arm_control_mode = str(args.arm_control)
        self.dof_idx          = 0
        self.pos_step         = 0.01
        self.rot_step         = 0.05
        self.max_dq           = float(args.max_dq)

        # ── Joint state mirrors ───────────────────────────────────────────
        self.latest_positions: Dict[int, float] = {j: 0.0 for j in UPPER_BODY_JOINTS}
        self.current_targets:  Dict[int, float] = {j: 0.0 for j in UPPER_BODY_JOINTS}
        self.desired_targets:  Dict[int, float] = {j: 0.0 for j in UPPER_BODY_JOINTS}
        self.seeded = False
        self.armed  = True
        self._running = True
        self.status   = "Waiting for rt/lowstate…"

        # ── EE target poses ───────────────────────────────────────────────
        self.target_T: Dict[str, np.ndarray] = {
            "left":  np.eye(4, dtype=np.float64),
            "right": np.eye(4, dtype=np.float64),
        }

        # ── Smart IK state ────────────────────────────────────────────────
        # rest_pose: captured from live joints on reengage
        self.rest_pose: Dict[str, Optional[np.ndarray]] = {
            "left": None, "right": None,
        }
        self.rest_pose_captured = False

        # user-defined obstacles: list of [x, y, z, r]
        self.user_obs: List[Tuple[float, float, float, float]] = []

        # per-arm waypoint queues for trajectory execution
        self._wq: Dict[str, List[np.ndarray]] = {"left": [], "right": []}

        # FK / SmartIK
        self._fk: Dict[str, ArmFK] = {
            "left":  ArmFK("left",  "dh"),
            "right": ArmFK("right", "dh"),
        }
        self._ik: Dict[str, SmartArmIK] = {
            "left":  SmartArmIK("left",  reg_weight=args.reg_weight, rep_weight=args.rep_weight),
            "right": SmartArmIK("right", reg_weight=args.reg_weight, rep_weight=args.rep_weight),
        }
        self.ik_info: Dict[str, Dict] = {
            "left":  {"success": None, "error_pos_m": 0.0, "error_rot_rad": 0.0, "iterations": 0},
            "right": {"success": None, "error_pos_m": 0.0, "error_rot_rad": 0.0, "iterations": 0},
        }
        self.coll_status: Dict[str, bool] = {"left": False, "right": False}

        # ── Robot objects ─────────────────────────────────────────────────
        ChannelFactoryInitialize(self.domain_id, self.iface)
        self.state_sub = UpperBodyStateSubscriber(UPPER_BODY_JOINTS)
        self.pub       = ArmSDKPublisher()
        self.robot     = Robot(iface=self.iface, domain_id=self.domain_id,
                                auto_start_sensors=True)
        self._last_tick = time.monotonic()

        self._seed_from_state()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _seed_from_state(self) -> None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            snap = self.state_sub.snapshot()
            if snap:
                pos, _ = snap
                self.latest_positions = dict(pos)
                self.current_targets  = dict(pos)
                self.desired_targets  = dict(pos)
                self.seeded = True
                self._sync_ee_from_joints()
                self.status = f"Connected on {self.iface}"
                return
            time.sleep(0.02)

    def _sync_ee_from_joints(self) -> None:
        for arm, joints in ARM_JOINTS.items():
            q = np.array([self.desired_targets[j] for j in joints])
            self.target_T[arm] = self._fk[arm].compute_arm(q).copy()

    def _capture_rest_pose(self) -> None:
        for arm, joints in ARM_JOINTS.items():
            self.rest_pose[arm] = np.array(
                [self.latest_positions.get(j, self.desired_targets[j]) for j in joints]
            )
        self.rest_pose_captured = True

    # ── Query helpers ─────────────────────────────────────────────────────────

    def _active_arms(self) -> List[str]:
        return ["left", "right"] if self.arm_control_mode == "both" else [self.arm_control_mode]

    def _display_arm(self) -> str:
        return "right" if self.arm_control_mode == "right" else "left"

    def _ee_step(self, idx: Optional[int] = None) -> float:
        i = self.dof_idx if idx is None else idx
        return self.pos_step if i < 3 else self.rot_step

    def _set_ee_step(self, value: float) -> None:
        if self.dof_idx < 3:
            self.pos_step = value
        else:
            self.rot_step = value

    def _fk_live(self, arm: str) -> np.ndarray:
        joints = ARM_JOINTS[arm]
        q = np.array([self.latest_positions.get(j, self.current_targets[j])
                      for j in joints])
        return self._fk[arm].compute_arm(q)

    def _other_q(self, arm: str) -> Optional[np.ndarray]:
        other = "right" if arm == "left" else "left"
        if self.arm_control_mode != "both":
            return None
        return np.array([self.desired_targets[j] for j in ARM_JOINTS[other]])

    # ── IK + trajectory planning ──────────────────────────────────────────────

    def _adjust_dof(self, delta: float) -> None:
        for arm in self._active_arms():
            T_prev = self.target_T[arm].copy()
            T_new  = T_prev.copy()
            if self.dof_idx < 3:
                T_new[self.dof_idx, 3] += delta
            else:
                T_new[:3, :3] = _ROT_BY_AXIS[self.dof_idx - 3](delta) @ T_new[:3, :3]
            self.target_T[arm] = T_new
            ok = self._solve_and_plan(arm, T_prev)
            if not ok:
                self.target_T[arm] = T_prev

    def _solve_and_plan(self, arm: str, T_prev: np.ndarray) -> bool:
        """Run SmartIK then plan a collision-free trajectory to the solution."""
        joints  = ARM_JOINTS[arm]
        q_init  = np.array([self.desired_targets[j] for j in joints])
        other_q = self._other_q(arm)

        q_sol, info = self._ik[arm].solve(
            self.target_T[arm],
            q_init,
            rest_q   = self.rest_pose[arm],
            user_obs = self.user_obs,
            other_q  = other_q,
        )
        self.ik_info[arm] = info

        if q_sol is None:
            return False

        # Clamp per-joint delta
        delta   = q_sol - q_init
        delta   = np.clip(delta, -self.max_dq, self.max_dq)
        q_apply = q_init + delta

        # Collision-check live current position
        q_current = np.array([self.current_targets[j] for j in joints])

        waypoints = TrajectoryPlanner.plan(
            q_start  = q_current,
            q_end    = q_apply,
            arm      = arm,
            rest_q   = self.rest_pose[arm],
            user_obs = self.user_obs,
            other_q  = other_q,
        )

        # Queue waypoints; first one becomes the new desired target
        self._wq[arm] = waypoints
        if waypoints:
            for i, j in enumerate(joints):
                self.desired_targets[j] = float(waypoints[0][i])

        return True

    # ── Waypoint queue advancement ────────────────────────────────────────────

    def _advance_waypoints(self) -> None:
        for arm in self._active_arms():
            wq = self._wq[arm]
            if len(wq) <= 1:
                continue
            joints = ARM_JOINTS[arm]
            # Advance once current_targets have arrived at the current waypoint
            arrived = all(
                abs(self.current_targets[j] - self.desired_targets[j]) < ARRIVE_EPSILON
                for j in joints
            )
            if arrived:
                wq.pop(0)
                for i, j in enumerate(joints):
                    self.desired_targets[j] = float(wq[0][i])

    # ── Collision status update ───────────────────────────────────────────────

    def _update_coll_status(self) -> None:
        for arm in ("left", "right"):
            joints = ARM_JOINTS[arm]
            q = np.array([self.current_targets[j] for j in joints])
            other_q = None
            if self.arm_control_mode == "both":
                other = "right" if arm == "left" else "left"
                other_q = np.array([self.current_targets[j] for j in ARM_JOINTS[other]])
            self.coll_status[arm] = _arm_in_hard_collision(
                q, arm, self.user_obs, other_q
            )

    # ── Control loop tick ─────────────────────────────────────────────────────

    def _ramp_step(self, dt: float) -> None:
        step = max(1e-9, self.max_speed * dt)
        for j in UPPER_BODY_JOINTS:
            cur = float(self.current_targets[j])
            des = float(self.desired_targets[j])
            d   = des - cur
            if abs(d) <= step:
                self.current_targets[j] = des
            else:
                self.current_targets[j] = cur + math.copysign(step, d)

    def tick(self) -> None:
        snap = self.state_sub.snapshot()
        if snap is not None:
            pos, _ = snap
            self.latest_positions = pos
            if not self.seeded:
                self.seeded          = True
                self.current_targets = dict(pos)
                self.desired_targets = dict(pos)
                self._sync_ee_from_joints()

        if not self.seeded or not self.armed:
            return

        now = time.monotonic()
        dt  = max(1.0 / self.rate_hz, now - self._last_tick)
        self._last_tick = now

        self._advance_waypoints()
        self._ramp_step(dt)
        self._update_coll_status()

        self.pub.publish(
            self.current_targets,
            arm_kp=self.arm_kp, arm_kd=self.arm_kd,
            waist_kp=self.waist_kp, waist_kd=self.waist_kd,
        )
        rest_txt = "rest:OK" if self.rest_pose_captured else "rest:—"
        self.status = (
            f"Publishing {self.rate_hz:.0f} Hz  "
            f"ramp {self.max_speed:.3f} r/s  "
            f"max_dq {self.max_dq:.3f} rad  "
            f"arm:{self.arm_control_mode}  {rest_txt}"
        )

    # ── Drawing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _addnstr(win, y, x, text, n, attr=0):
        try: win.addnstr(y, x, text, n, attr)
        except curses.error: pass

    @staticmethod
    def _addstr(win, y, x, text, attr=0):
        try: win.addstr(y, x, text, attr)
        except curses.error: pass

    def _cp(self, p): return curses.color_pair(p) if curses.has_colors() else 0

    def draw(self, win, h: int, w: int) -> None:
        if h < 20 or w < 76:
            self._addstr(win, 0, 0, f"Terminal too small ({w}×{h}). Need ≥76×20.")
            return
        try: self._draw_all(win, h, w)
        except curses.error: pass

    def _draw_all(self, win, h: int, w: int) -> None:  # noqa: C901
        row = 0

        # ── Title ─────────────────────────────────────────────────────────
        title = "6D EE IK Pose Control — Smart"
        ca = self._cp(C_GREEN if self.seeded else C_RED) | curses.A_BOLD
        aa = self._cp(C_GREEN if self.armed  else C_RED) | curses.A_BOLD
        self._addstr(win, row, 0, "─" * w, self._cp(C_CYAN))
        self._addstr(win, row, max(0,(w-len(title))//2), title,
                     self._cp(C_CYAN)|curses.A_BOLD)
        self._addstr(win, row, w-22, f"[{'CONNECTED' if self.seeded else 'WAITING'}]", ca)
        self._addstr(win, row, w-12, f"[{'ARMED' if self.armed else 'RELEASED'}]",     aa)
        row += 1

        # ── Parameter bar ─────────────────────────────────────────────────
        arm_txt   = f"  Arm: [{self.arm_control_mode.upper()}]  (m)"
        param_txt = (f"ramp {self.max_speed:.3f} r/s (s)  "
                     f"max_dq {self.max_dq:.3f} (d/[/])")
        self._addnstr(win, row, 0, arm_txt, w//2)
        self._addnstr(win, row, w-len(param_txt)-2, param_txt, w, self._cp(C_YELLOW))
        row += 1

        # ── Smart-mode info bar ───────────────────────────────────────────
        rest_txt = ("rest: captured" if self.rest_pose_captured else "rest: not yet — press r then e")
        obs_txt  = f"{len(self.user_obs)} obstacle(s) (o/O)"
        smart_txt = f"  {rest_txt}   {obs_txt}"
        self._addnstr(win, row, 0, smart_txt, w, self._cp(C_CYAN))
        row += 1

        # ── Divider ───────────────────────────────────────────────────────
        self._addstr(win, row, 0, "─" * w, self._cp(C_CYAN))
        row += 1

        # ── DOF table ─────────────────────────────────────────────────────
        disp    = self._display_arm()
        T_cur   = self._fk_live(disp) if self.seeded else np.eye(4)
        T_tgt   = self.target_T[disp]
        cur_rpy = _rpy_from_R(T_cur[:3,:3])
        tgt_rpy = _rpy_from_R(T_tgt[:3,:3])

        hdr = (f"  {'DOF':<9}{'Live FK':<19}{'Target':<19}"
               f"{'Step':<15}  ({disp} arm)")
        self._addnstr(win, row, 0, hdr, w, curses.A_BOLD)
        row += 1

        for i in range(N_DOFS):
            sel  = (i == self.dof_idx)
            mark = "▶" if sel else " "
            step = self._ee_step(i)
            unit = DOF_UNITS[i]
            cur_v = float(T_cur[i,3]) if i < 3 else cur_rpy[i-3]
            tgt_v = float(T_tgt[i,3]) if i < 3 else tgt_rpy[i-3]
            line  = (f"{mark} {DOF_NAMES[i]:<9}"
                     f"{cur_v:+.4f} {unit:<5}"
                     f"   {tgt_v:+.4f} {unit:<5}"
                     f"   {step:.4f} {unit}")
            self._addnstr(win, row, 0, line, w,
                          (self._cp(C_SEL)|curses.A_BOLD) if sel else 0)
            row += 1

        # ── Divider ───────────────────────────────────────────────────────
        self._addstr(win, row, 0, "─" * w, self._cp(C_CYAN))
        row += 1

        # ── IK + collision status ─────────────────────────────────────────
        for arm in ("left", "right"):
            if row >= h - 7:
                break
            if self.arm_control_mode not in ("both", arm):
                continue
            info  = self.ik_info[arm]
            ok    = info.get("success")
            coll  = self.coll_status[arm]
            wq_n  = len(self._wq[arm])

            if ok is None:
                ik_txt = "pending"
                ia = 0
            elif ok:
                ik_txt = (f"OK  pos={info['error_pos_m']:.4f}m  "
                          f"rot={info['error_rot_rad']:.4f}rad  "
                          f"{info['iterations']}it")
                ia = self._cp(C_GREEN)
            else:
                ik_txt = (f"FAIL (rolled back)  "
                          f"pos={info['error_pos_m']:.4f}m")
                ia = self._cp(C_RED)

            coll_txt = "  [COLLISION]" if coll else ""
            wq_txt   = f"  [{wq_n} waypts]" if wq_n > 1 else ""
            line = f"  IK {arm:<5}: {ik_txt}{coll_txt}{wq_txt}"
            self._addnstr(win, row, 0, line, w,
                          ia | (curses.A_BOLD if coll else 0))
            row += 1

        # ── Obstacle list ─────────────────────────────────────────────────
        if row < h - 7 and self.user_obs:
            self._addstr(win, row, 0, "─" * w, self._cp(C_CYAN))
            row += 1
        for i, obs in enumerate(self.user_obs):
            if row >= h - 7:
                break
            txt = f"  Obs {i}: ({obs[0]:+.2f}, {obs[1]:+.2f}, {obs[2]:+.2f})  r={obs[3]:.2f}m"
            self._addnstr(win, row, 0, txt, w, self._cp(C_YELLOW))
            row += 1

        # ── Joint readout ─────────────────────────────────────────────────
        if row < h - 7:
            self._addstr(win, row, 0, "─" * w, self._cp(C_CYAN))
            row += 1
        if row < h - 7:
            self._addnstr(win, row, 0,
                          "  Joint targets (rad):                    "
                          "  Live feedback (rad):", w, curses.A_BOLD)
            row += 1
        for arm, joints in ARM_JOINTS.items():
            if row >= h - 7:
                break
            if self.arm_control_mode not in ("both", arm):
                continue
            prefix = f"  {arm.upper():<5}: "
            lbl    = "  ".join(f"{n:<5}" for n in JOINT_LABELS)
            tgt_v  = "  ".join(f"{self.desired_targets[j]:+.3f}" for j in joints)
            liv_v  = "  ".join(
                f"{self.latest_positions.get(j, self.current_targets[j]):+.3f}"
                for j in joints)
            self._addnstr(win, row, 0, prefix + lbl, w, self._cp(C_CYAN))
            row += 1
            if row < h - 7:
                half = w // 2
                self._addnstr(win, row, 0,    prefix + tgt_v, half)
                self._addnstr(win, row, half, prefix + liv_v, w - half, self._cp(C_YELLOW))
                row += 1

        # ── Footer ────────────────────────────────────────────────────────
        self._addstr(win, h-6, 0, "─" * w, self._cp(C_CYAN))
        h1 = "  ↑/↓ j/k: DOF   ← →/- +: adjust   < >: EE step   [ ]: max_dq   m: arm"
        h2 = "  y: sync   r: release   e: reengage+capture   z: zero   s: speed   d: max_dq"
        h3 = "  o: add obstacle   O: clear obstacles   q: quit"
        self._addnstr(win, h-5, 0, h1, w, self._cp(C_YELLOW))
        self._addnstr(win, h-4, 0, h2, w, self._cp(C_YELLOW))
        self._addnstr(win, h-3, 0, h3, w, self._cp(C_YELLOW))
        self._addstr(win, h-2, 0, "─" * w, self._cp(C_CYAN))
        sa = self._cp(C_GREEN if self.armed and self.seeded else C_RED)
        self._addnstr(win, h-1, 0, f"  {self.status}", w, sa)

    # ── Inline prompt ─────────────────────────────────────────────────────────

    def _prompt(self, win, h: int, w: int, label: str) -> str:
        curses.curs_set(1)
        win.timeout(-1)
        buf: List[str] = []
        while True:
            win.move(h-1, 0)
            win.clrtoeol()
            self._addnstr(win, h-1, 0, f"{label}: {''.join(buf)}▌"[:w], w, curses.A_BOLD)
            win.refresh()
            key = win.getch()
            if key in (curses.KEY_ENTER, 10, 13):  break
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                if buf: buf.pop()
            elif key == 27: buf = []; break
            elif 32 <= key <= 126: buf.append(chr(key))
        curses.curs_set(0)
        win.timeout(20)
        return "".join(buf).strip()

    # ── Key handler ───────────────────────────────────────────────────────────

    def handle_key(self, key: int, win, h: int, w: int) -> None:  # noqa: C901
        if key in (ord("q"), 27):
            self._running = False
            return

        if key in (curses.KEY_UP, ord("k")):
            self.dof_idx = max(0, self.dof_idx - 1); return
        if key in (curses.KEY_DOWN, ord("j")):
            self.dof_idx = min(N_DOFS-1, self.dof_idx + 1); return

        if key in (curses.KEY_LEFT, ord("-")):
            if self.seeded and self.armed: self._adjust_dof(-self._ee_step())
            return
        if key in (curses.KEY_RIGHT, ord("+")):
            if self.seeded and self.armed: self._adjust_dof(+self._ee_step())
            return

        if key == ord("<"):
            self._set_ee_step(max(0.0001, self._ee_step() / 2.0)); return
        if key == ord(">"):
            hi = 1.0 if self.dof_idx < 3 else math.pi
            self._set_ee_step(min(hi, self._ee_step() * 2.0)); return

        if key == ord("["):
            self.max_dq = max(0.005, self.max_dq / 2.0)
            self.status = f"max_dq → {self.max_dq:.4f} rad"; return
        if key == ord("]"):
            self.max_dq = min(math.pi, self.max_dq * 2.0)
            self.status = f"max_dq → {self.max_dq:.4f} rad"; return

        if key == ord("m"):
            idx = ARM_CONTROL_MODES.index(self.arm_control_mode)
            self.arm_control_mode = ARM_CONTROL_MODES[(idx+1) % len(ARM_CONTROL_MODES)]
            self.status = f"Arm → {self.arm_control_mode}"; return

        if key == ord("y"):
            snap = self.state_sub.snapshot()
            if snap: self.latest_positions = dict(snap[0])
            self.current_targets = dict(self.latest_positions)
            self.desired_targets = dict(self.latest_positions)
            self._wq = {"left": [], "right": []}
            self._sync_ee_from_joints()
            self.status = "Resynced EE targets to current hand pose"; return

        if key == ord("r"):
            try:
                self.robot.wait_for_low_state(timeout=2.0)
                self.robot.release_arms()
                self.armed = False
                self._wq = {"left": [], "right": []}
                self.status = "Arms released — move freely, press e to reengage"
            except Exception as exc:
                self.status = f"Release failed: {exc}"
            return

        if key == ord("e"):
            try:
                self.robot.wait_for_low_state(timeout=2.0)
                self.robot.unrelease_arms()
                self.armed = True
                snap = self.state_sub.snapshot()
                if snap: self.latest_positions = dict(snap[0])
                self.current_targets = dict(self.latest_positions)
                self.desired_targets = dict(self.latest_positions)
                self._wq = {"left": [], "right": []}
                self._sync_ee_from_joints()
                # Capture the post-reengage pose as the rest pose
                self._capture_rest_pose()
                self.status = "Reengaged — rest pose captured, ready for commands"
            except Exception as exc:
                self.status = f"Reengage failed: {exc}"
            return

        if key == ord("z"):
            self.pub.publish_zero_gains(self.current_targets)
            self.status = "Zero-gain hold sent"; return

        if key == ord("s"):
            val = self._prompt(win, h, w, f"Ramp speed r/s [{self.max_speed:.4f}]")
            try:
                self.max_speed = max(0.01, float(val))
                self.status = f"Ramp speed → {self.max_speed:.4f} r/s"
            except (ValueError, TypeError):
                if val: self.status = f"Invalid: {val!r}"
            return

        if key == ord("d"):
            val = self._prompt(win, h, w, f"Max joint delta rad [{self.max_dq:.4f}]")
            try:
                self.max_dq = max(0.005, min(math.pi, float(val)))
                self.status = f"max_dq → {self.max_dq:.4f} rad"
            except (ValueError, TypeError):
                if val: self.status = f"Invalid: {val!r}"
            return

        if key == ord("o"):
            raw = self._prompt(win, h, w, "Obstacle  x y z r  (metres)")
            try:
                parts = [float(v) for v in raw.split()]
                if len(parts) != 4:
                    raise ValueError("need 4 values")
                self.user_obs.append(tuple(parts))  # type: ignore[arg-type]
                self.status = (f"Obstacle {len(self.user_obs)-1} added: "
                               f"({parts[0]:.2f},{parts[1]:.2f},{parts[2]:.2f}) r={parts[3]:.2f}m")
            except (ValueError, TypeError) as exc:
                if raw: self.status = f"Invalid obstacle: {exc}"
            return

        if key == ord("O"):
            n = len(self.user_obs)
            self.user_obs.clear()
            self.status = f"Cleared {n} obstacle(s)"; return

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        curses.wrapper(self._curses_main)

    def _curses_main(self, stdscr) -> None:
        if curses.has_colors():
            curses.start_color(); curses.use_default_colors()
            curses.init_pair(C_GREEN,  curses.COLOR_GREEN,  -1)
            curses.init_pair(C_YELLOW, curses.COLOR_YELLOW, -1)
            curses.init_pair(C_RED,    curses.COLOR_RED,    -1)
            curses.init_pair(C_CYAN,   curses.COLOR_CYAN,   -1)
            curses.init_pair(C_SEL,    curses.COLOR_BLACK,  curses.COLOR_WHITE)
            curses.init_pair(C_BOLD,   curses.COLOR_BLACK,  curses.COLOR_CYAN)

        curses.curs_set(0)
        stdscr.timeout(20)
        self._last_tick = time.monotonic()
        dt_target = 1.0 / self.rate_hz

        while self._running:
            h, w = stdscr.getmaxyx()
            try:
                stdscr.erase()
                self.draw(stdscr, h, w)
                stdscr.refresh()
            except curses.error:
                pass

            key = stdscr.getch()
            if key != -1:
                self.handle_key(key, stdscr, h, w)

            now = time.monotonic()
            if now - self._last_tick >= dt_target:
                self.tick()

        try:
            self.robot.release_arms()
        except Exception:
            pass


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="6D EE IK Pose Control — Smart Edition (rest-pose reg + collision avoidance + trajectory planning)"
    )
    p.add_argument("--iface",        default="eth0")
    p.add_argument("--domain-id",    type=int,   default=0)
    p.add_argument("--rate-hz",      type=float, default=50.0)
    p.add_argument("--speed-rad-s",  type=float, default=0.5,
                   help="Joint ramp speed rad/s")
    p.add_argument("--max-dq",       type=float, default=0.1,
                   help="Max joint change per IK key-press (rad)")
    p.add_argument("--reg-weight",   type=float, default=0.05,
                   help="Null-space rest-pose regularisation weight (0=off)")
    p.add_argument("--rep-weight",   type=float, default=0.8,
                   help="Null-space collision repulsion weight (0=off)")
    p.add_argument("--kp",           type=float, default=DEFAULT_ARM_KP)
    p.add_argument("--kd",           type=float, default=DEFAULT_ARM_KD)
    p.add_argument("--arm-control",  choices=ARM_CONTROL_MODES, default="right")
    return p.parse_args()


def main() -> None:
    IKPoseCLISmart(_parse_args()).run()


if __name__ == "__main__":
    main()
