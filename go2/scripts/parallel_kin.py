"""
parallel_kin.py

Models the Go2 body as the end-effector (platform) of a *reconfigurable*
parallel kinematic robot.

Reconfigurable PKM model
------------------------
A classical PKM (Stewart platform) has fixed base-attachment points.  The Go2
is different: it is a *reconfigurable* parallel mechanism whose topology
changes during locomotion.

At any instant there are two populations of legs:

  Stance legs  -- grounded; foot contacts are temporarily fixed world points.
                  Together they form the parallel kinematic chains that constrain
                  the body's 6-DOF pose, exactly like a classical PKM.

  Swing leg(s) -- airborne; they are serial manipulators mounted on the moving
                  body platform.  Their motion exerts a reaction wrench on the
                  body that is transmitted through the stance legs as changes
                  in ground reaction forces.

The decomposition during 3-stance / 1-swing gait:

    World ─ stance_leg_1 ─┐
    World ─ stance_leg_2 ─┤─ Body (floating platform) ─ swing_leg ─ foot
    World ─ stance_leg_3 ─┘

Topology change events
----------------------
  lift_leg(leg)           swap leg from stance → swing (foot leaves ground)
  land_leg(leg, pos)      swap leg from swing → stance (foot touches ground)

The stance set changes at these events; between events the mechanism topology
is fixed and classical PKM equations apply.

Coordinate conventions
----------------------
World  : fixed inertial frame, z up.
Body   : origin at the body reference point, orientation by ZYX Euler angles
         (roll about x, pitch about y, yaw about z) matching the Unitree IMU.
Hip-i  : body frame translated by HIP_ORIGIN[i] -- the input frame for the
         analytic serial IK (same as in the gait scripts).

API summary
-----------
  pk = ParallelKinBody()
  pk.init_contacts_from_pose(pos, rpy, q_all)   # bootstrap from known state

  # Topology management
  pk.lift_leg(leg)                              # stance → swing
  pk.land_leg(leg, world_pos)                  # swing  → stance
  pk.update_contact(leg, world_pos)             # update contact without changing stance set

  # Stance-legs-only parallel kinematics (default: pk.stance_legs)
  q_all        = pk.body_ik(pos, rpy)           # parallel IK
  pos, rpy     = pk.body_fk(q_all)              # parallel FK  (Gauss-Newton)
  A            = pk.support_jacobian(q_all, support_legs)   # (3n × 6)
  v, w         = pk.body_velocity(q_all, qdot_all, support_legs)
  J_full       = pk.full_jacobian(q_all, support_legs)      # (6 × 3n)

  # Serial chain (per-leg)
  J_leg        = leg_jacobian(leg, qh, qt, qc)  # 3×3 Jacobian in hip frame

  # Swing-leg serial-on-platform kinematics
  p_foot       = pk.swing_foot_world(leg, q_all)            # foot via serial FK
  F, τ         = pk.reaction_wrench(leg, q_all, F_foot)     # wrench on body
  grfs         = pk.grf_distribution(q_all, ext_wrench)     # GRF split (min-norm)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── geometry (verified against go2.urdf) ─────────────────────────────────────

HIP_ORIGIN: Dict[str, Tuple[float, float, float]] = {
    "FL": ( 0.1934,  0.0465, 0.0),
    "FR": ( 0.1934, -0.0465, 0.0),
    "RL": (-0.1934,  0.0465, 0.0),
    "RR": (-0.1934, -0.0465, 0.0),
}
HIP_LATERAL_OFFSET = 0.0955   # thigh-joint y-offset from hip-joint
THIGH_LENGTH        = 0.213
CALF_LENGTH         = 0.213

LEG_INDEX: Dict[str, Tuple[int, int, int]] = {
    "FR": (0, 1, 2),
    "FL": (3, 4, 5),
    "RR": (6, 7, 8),
    "RL": (9, 10, 11),
}
LEG_ORDER = ["FR", "FL", "RR", "RL"]
LEG_SIGNS = {
    "FL": {"left":  1.0, "front":  1.0},
    "FR": {"left": -1.0, "front":  1.0},
    "RL": {"left":  1.0, "front": -1.0},
    "RR": {"left": -1.0, "front": -1.0},
}


# ── pure-math helpers ─────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX rotation matrix: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,     cp*sr,              cp*cr           ],
    ])


def rot_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract (roll, pitch, yaw) from a ZYX rotation matrix."""
    pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2))
    if abs(math.cos(pitch)) < 1e-9:           # gimbal lock
        return 0.0, pitch, math.atan2(-R[1, 2], R[1, 1])
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw  = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix such that skew(a) @ b == a × b."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ],
    ])


def _drpy(roll: float, pitch: float, yaw: float):
    """
    Partial derivatives of R(roll, pitch, yaw) w.r.t. each angle.
    Returns (dR/droll, dR/dpitch, dR/dyaw) as three 3×3 arrays.
    Used by the FK Gauss-Newton solver.
    """
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)

    Rx = np.array([[1,  0,   0 ], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0,  sp ], [0,  1,   0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0 ], [sy, cy,  0], [0,   0,  1]])

    dRx = np.array([[0, 0, 0], [0, -sr, -cr], [0, cr, -sr]])
    dRy = np.array([[-sp, 0, cp], [0, 0, 0], [-cp, 0, -sp]])
    dRz = np.array([[-sy, -cy, 0], [cy, -sy, 0], [0, 0, 0]])

    return Rz @ Ry @ dRx, Rz @ dRy @ Rx, dRz @ Ry @ Rx


# ── serial-chain FK, IK, and Jacobian ────────────────────────────────────────

def leg_fk(leg: str, q_hip: float, q_thigh: float, q_calf: float) -> np.ndarray:
    """
    Forward kinematics for one leg.
    Returns foot position in the hip frame (i.e. relative to HIP_ORIGIN[leg]).
    """
    s   = LEG_SIGNS[leg]["left"]
    st, ct   = math.sin(q_thigh), math.cos(q_thigh)
    stc, ctc = math.sin(q_thigh + q_calf), math.cos(q_thigh + q_calf)
    sh, ch   = math.sin(q_hip),   math.cos(q_hip)

    px  = -THIGH_LENGTH * st  - CALF_LENGTH * stc
    pzr = -THIGH_LENGTH * ct  - CALF_LENGTH * ctc
    py  =  s * HIP_LATERAL_OFFSET * ch - pzr * sh
    pz  =  s * HIP_LATERAL_OFFSET * sh + pzr * ch
    return np.array([px, py, pz])


def leg_ik(leg: str, foot_pos_hip: np.ndarray) -> Tuple[float, float, float]:
    """
    Analytic IK for one leg.
    foot_pos_hip : foot position relative to HIP_ORIGIN[leg].
    Returns (q_hip, q_thigh, q_calf).
    """
    x, y, z = foot_pos_hip
    s = LEG_SIGNS[leg]["left"]
    radial_sq = y*y + z*z - HIP_LATERAL_OFFSET**2
    radial    = math.sqrt(max(radial_sq, 1e-9))
    q_hip     = math.atan2(y, -z) - math.atan2(s * HIP_LATERAL_OFFSET, radial)
    knee_cos  = _clamp(
        (x*x + radial_sq - THIGH_LENGTH**2 - CALF_LENGTH**2) / (2*THIGH_LENGTH*CALF_LENGTH),
        -1.0, 1.0,
    )
    q_calf   = -math.acos(knee_cos)
    q_thigh  = math.atan2(-x, radial) - math.atan2(
        CALF_LENGTH * math.sin(q_calf),
        THIGH_LENGTH + CALF_LENGTH * math.cos(q_calf),
    )
    return q_hip, q_thigh, q_calf


def leg_jacobian(leg: str, q_hip: float, q_thigh: float, q_calf: float) -> np.ndarray:
    """
    Analytic 3×3 Jacobian of the serial leg chain in the hip frame.

        dp_foot = J @ [dq_hip, dq_thigh, q_calf]ᵀ

    Columns correspond to hip, thigh, calf joints.
    Rows correspond to x (forward), y (lateral), z (vertical) axes.

    Derived by differentiating leg_fk analytically:

        px  = -L1*sin(qt) - L2*sin(qt+qc)
        pzr = -L1*cos(qt) - L2*cos(qt+qc)       (intermediate scalar)
        py  =  s*d*cos(qh) - pzr*sin(qh)
        pz  =  s*d*sin(qh) + pzr*cos(qh)
    """
    s   = LEG_SIGNS[leg]["left"]
    sh, ch   = math.sin(q_hip),   math.cos(q_hip)
    st, ct   = math.sin(q_thigh), math.cos(q_thigh)
    stc, ctc = math.sin(q_thigh + q_calf), math.cos(q_thigh + q_calf)

    pzr = -THIGH_LENGTH * ct - CALF_LENGTH * ctc

    # ∂pzr/∂qt  and  ∂pzr/∂qc  (reused for py and pz partials)
    dpzr_dt = THIGH_LENGTH * st + CALF_LENGTH * stc
    dpzr_dc = CALF_LENGTH  * stc

    # Row 0 : px
    dpx = np.array([
        0.0,
        -THIGH_LENGTH * ct - CALF_LENGTH * ctc,   # ∂px/∂qt
        -CALF_LENGTH * ctc,                        # ∂px/∂qc
    ])

    # Row 1 : py = s*d*cos(qh) - pzr*sin(qh)
    dpy = np.array([
        -s * HIP_LATERAL_OFFSET * sh - pzr * ch,  # ∂py/∂qh
        -dpzr_dt * sh,                             # ∂py/∂qt
        -dpzr_dc * sh,                             # ∂py/∂qc
    ])

    # Row 2 : pz = s*d*sin(qh) + pzr*cos(qh)
    dpz = np.array([
        s * HIP_LATERAL_OFFSET * ch - pzr * sh,   # ∂pz/∂qh
        dpzr_dt * ch,                              # ∂pz/∂qt
        dpzr_dc * ch,                              # ∂pz/∂qc
    ])

    return np.vstack([dpx, dpy, dpz])   # (3, 3)


# ── parallel kinematic body model ─────────────────────────────────────────────

class ParallelKinBody:
    """
    The Go2 body modelled as the moving platform of a 4-legged PKM.

    Each grounded foot is the base of a serial chain.  Unlike a classical PKM
    where these bases are static, here each base position is refreshed via
    update_contact() whenever the foot plants at a new location.
    """

    def __init__(self) -> None:
        self.body_pos = np.zeros(3)          # body origin in world frame
        self.body_rpy = np.zeros(3)          # (roll, pitch, yaw) rad

        # Non-static base frame origins -- one per leg, world frame.
        # These are the attachment points that would be fixed in a classical PKM.
        self.foot_contacts: Dict[str, np.ndarray] = {
            leg: np.zeros(3) for leg in LEG_ORDER
        }

        # Active PKM topology: legs currently in the stance (grounded) set.
        # Swing legs are those NOT in this list.
        self.stance_legs: List[str] = list(LEG_ORDER)

    # ── contact management ───────────────────────────────────────────────────

    def update_contact(self, leg: str, world_pos) -> None:
        """
        Record a new foot contact position in the world frame.
        Call this each time foot `leg` plants after swinging.
        """
        self.foot_contacts[leg] = np.asarray(world_pos, dtype=float)

    def init_contacts_from_pose(
        self, pos, rpy, q_all: List[float]
    ) -> None:
        """
        Bootstrap all four contact positions from a known body pose + joint
        angles.  Useful at startup when absolute foot locations are not yet
        tracked independently.
        """
        R = rpy_to_rot(*rpy)
        p = np.asarray(pos, dtype=float)
        for leg in LEG_ORDER:
            q_h, q_t, q_c = self._q_leg(leg, q_all)
            foot_body = np.array(HIP_ORIGIN[leg]) + leg_fk(leg, q_h, q_t, q_c)
            self.foot_contacts[leg] = p + R @ foot_body
        self.body_pos = p.copy()
        self.body_rpy = np.asarray(rpy, dtype=float)

    # ── parallel IK ──────────────────────────────────────────────────────────

    def body_ik(
        self,
        pos,
        rpy,
        support_legs: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Parallel IK: given desired body pose, compute joint angles.

        For each leg the grounded foot contact (world frame) is the fixed
        base.  The problem reduces to expressing that contact in the leg's
        hip frame and calling the analytic serial IK -- one decoupled solve
        per leg, which is the standard PKM IK structure.

        Parameters
        ----------
        pos          : (3,) desired body position in world frame
        rpy          : (3,) desired (roll, pitch, yaw) in rad
        support_legs : solve only these legs; others are left as NaN

        Returns
        -------
        q_all : 12 joint angles in motor-index order (FR0..RL2)
        """
        R = rpy_to_rot(*rpy)
        p = np.asarray(pos, dtype=float)
        q_all: List[float] = [float("nan")] * 12

        for leg in (support_legs if support_legs is not None else self.stance_legs):
            foot_world = self.foot_contacts[leg]
            # express contact in body frame, then subtract hip origin
            foot_hip   = R.T @ (foot_world - p) - np.array(HIP_ORIGIN[leg])
            q_h, q_t, q_c = leg_ik(leg, foot_hip)
            hi, ti, ci = LEG_INDEX[leg]
            q_all[hi] = q_h
            q_all[ti] = q_t
            q_all[ci] = q_c

        self.body_pos = p.copy()
        self.body_rpy = np.asarray(rpy, dtype=float)
        return q_all

    # ── parallel FK ──────────────────────────────────────────────────────────

    def body_fk(
        self,
        q_all: List[float],
        support_legs: Optional[List[str]] = None,
        tol: float = 1e-7,
        max_iter: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parallel FK: given joint angles, recover body pose.

        Solves the (overdetermined) constraint system
            p + R(rpy) @ foot_body_i(q_i)  =  foot_contacts[i]
        via Gauss-Newton on the 6-DOF state  x = [pos; roll; pitch; yaw].

        With n support legs we have 3n equations and 6 unknowns.
        Minimum 2 legs for a unique solution; 3+ legs are over-constrained
        and the least-squares solution averages the redundant information.

        Returns
        -------
        pos : (3,) body position in world frame
        rpy : (3,) roll, pitch, yaw in rad
        """
        legs = support_legs if support_legs is not None else self.stance_legs

        # foot positions in body frame are constant during this solve
        foot_body = {
            leg: np.array(HIP_ORIGIN[leg]) + leg_fk(leg, *self._q_leg(leg, q_all))
            for leg in legs
        }

        x = np.concatenate([self.body_pos, self.body_rpy])   # warm-start

        for _ in range(max_iter):
            r, J = self._fk_residual_jac(x, foot_body, legs)
            dx   = np.linalg.lstsq(J, -r, rcond=None)[0]
            x   += dx
            if np.linalg.norm(dx) < tol:
                break

        self.body_pos = x[:3].copy()
        self.body_rpy = x[3:].copy()
        return self.body_pos.copy(), self.body_rpy.copy()

    # ── velocity kinematics ───────────────────────────────────────────────────

    def support_jacobian(
        self, q_all: List[float], support_legs: List[str]
    ) -> np.ndarray:
        """
        Build the (3n × 6) constraint Jacobian A for the grounded legs.

        The contact constraint for leg i (foot fixed in world):
            v_body  +  ω_body × r_i^W  =  0
        where  r_i^W = R_B @ foot_body_i  is the foot vector in world frame
        (relative to body origin).  In matrix form:

            [I₃  | −skew(r_i^W)] @ [v_body; ω_body] = 0

        Stacking over all n support legs gives:  A @ ξ = 0  (ξ = body twist).

        Returns
        -------
        A : (3·n, 6) ndarray
        """
        R = rpy_to_rot(*self.body_rpy)
        rows = []
        for leg in support_legs:
            foot_b  = np.array(HIP_ORIGIN[leg]) + leg_fk(leg, *self._q_leg(leg, q_all))
            r_world = R @ foot_b
            rows.append(np.hstack([np.eye(3), -skew(r_world)]))
        return np.vstack(rows)

    def body_velocity(
        self,
        q_all: List[float],
        qdot_all: List[float],
        support_legs: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute body linear and angular velocity from support-leg joint velocities.

        Differentiating the contact constraint for leg i:
            v_body + ω_body × r_i^W + R_B @ J_leg_i @ qdot_i  =  0

        Written as:
            A_i @ ξ  =  b_i
            A_i = [I | −skew(r_i^W)]       (3×6)
            b_i = −R_B @ J_leg_i @ qdot_i  (3,)

        The body twist ξ = [v_body; ω_body] is solved in the least-squares sense
        over all support legs.

        Returns
        -------
        v_body : (3,) linear velocity in world frame
        w_body : (3,) angular velocity in world frame
        """
        R = rpy_to_rot(*self.body_rpy)
        A_rows, b_rows = [], []
        for leg in support_legs:
            q_leg    = self._q_leg(leg, q_all)
            qdot_leg = self._q_leg(leg, qdot_all)
            foot_b   = np.array(HIP_ORIGIN[leg]) + leg_fk(leg, *q_leg)
            r_world  = R @ foot_b
            J_leg    = leg_jacobian(leg, *q_leg)
            A_rows.append(np.hstack([np.eye(3), -skew(r_world)]))
            b_rows.append(-(R @ J_leg) @ np.array(qdot_leg))
        A  = np.vstack(A_rows)
        b  = np.concatenate(b_rows)
        xi = np.linalg.lstsq(A, b, rcond=None)[0]
        return xi[:3], xi[3:]

    def full_jacobian(
        self, q_all: List[float], support_legs: List[str]
    ) -> np.ndarray:
        """
        (6 × 3n) Jacobian mapping support-leg joint velocities to body twist.

        In a classical PKM this is called the "forward velocity Jacobian" of
        the platform.  Derived by combining the constraint Jacobian A with the
        serial-chain Jacobians B:

            A @ ξ = B @ qdot_support
            ξ = A⁺ @ B @ qdot_support  →  J_full = A⁺ @ B

        where  B  is block-diagonal with blocks  −R_B @ J_leg_i.

        Returns
        -------
        J_full : (6, 3·n) ndarray
        """
        R = rpy_to_rot(*self.body_rpy)
        A_rows = []
        B_blocks = []
        for leg in support_legs:
            q_leg   = self._q_leg(leg, q_all)
            foot_b  = np.array(HIP_ORIGIN[leg]) + leg_fk(leg, *q_leg)
            r_world = R @ foot_b
            A_rows.append(np.hstack([np.eye(3), -skew(r_world)]))
            B_blocks.append(-(R @ leg_jacobian(leg, *q_leg)))

        A = np.vstack(A_rows)                              # (3n, 6)
        n = len(support_legs)
        B = np.zeros((3 * n, 3 * n))
        for i, blk in enumerate(B_blocks):
            B[3*i:3*i+3, 3*i:3*i+3] = blk                # block-diagonal

        return np.linalg.pinv(A) @ B                       # (6, 3n)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _q_leg(
        self, leg: str, q_all: List[float]
    ) -> Tuple[float, float, float]:
        hi, ti, ci = LEG_INDEX[leg]
        return q_all[hi], q_all[ti], q_all[ci]

    def _fk_residual_jac(
        self,
        x: np.ndarray,
        foot_body: Dict[str, np.ndarray],
        legs: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Residual and Jacobian for the Gauss-Newton FK solver.

        Residual for leg i:
            r_i = pos + R(rpy) @ foot_body_i − foot_contacts_i    (3,)

        Jacobian  ∂r_i/∂x  where  x = [pos; roll; pitch; yaw]:
            [∂r_i/∂pos | ∂r_i/∂rpy]  =  [I₃ | dR_dr@fb  dR_dp@fb  dR_dy@fb]
        """
        pos  = x[:3]
        roll, pitch, yaw = x[3], x[4], x[5]
        R    = rpy_to_rot(roll, pitch, yaw)
        dR_dr, dR_dp, dR_dy = _drpy(roll, pitch, yaw)

        r_rows, J_rows = [], []
        for leg in legs:
            fb  = foot_body[leg]
            r_rows.append(pos + R @ fb - self.foot_contacts[leg])
            J_rows.append(np.hstack([
                np.eye(3),
                np.column_stack([dR_dr @ fb, dR_dp @ fb, dR_dy @ fb]),
            ]))

        return np.concatenate(r_rows), np.vstack(J_rows)

    # ── topology management (stance ↔ swing) ─────────────────────────────────

    @property
    def swing_legs(self) -> List[str]:
        """Legs currently airborne (not in stance set)."""
        return [l for l in LEG_ORDER if l not in self.stance_legs]

    def lift_leg(self, leg: str) -> None:
        """
        Move leg from stance to swing.

        The foot contact record is preserved so that the last touch-down
        position remains available (e.g. for trajectory planning), but the
        leg is removed from the PKM constraint set.
        """
        if leg in self.stance_legs:
            self.stance_legs.remove(leg)

    def land_leg(self, leg: str, world_pos=None) -> None:
        """
        Move leg from swing to stance.

        If world_pos is provided the contact record is updated to the new
        touch-down location before adding the leg to the stance set.
        This is the reconfiguration event that changes the PKM topology.
        """
        if world_pos is not None:
            self.update_contact(leg, world_pos)
        if leg not in self.stance_legs:
            self.stance_legs.append(leg)

    # ── swing-leg serial-on-platform kinematics ───────────────────────────────

    def swing_foot_world(self, leg: str, q_all: List[float]) -> np.ndarray:
        """
        World-frame foot position for a swing leg via serial FK from the
        moving body platform.

        Unlike stance feet (whose world position is the stored contact),
        the swing foot moves freely as a serial manipulator attached to
        the body.  The kinematic chain is:

            World ← body_pos, R_body ← HIP_ORIGIN[leg] ← leg_fk(q)
        """
        R  = rpy_to_rot(*self.body_rpy)
        ho = np.array(HIP_ORIGIN[leg])
        return self.body_pos + R @ (ho + leg_fk(leg, *self._q_leg(leg, q_all)))

    def reaction_wrench(
        self,
        swing_leg: str,
        q_all: List[float],
        foot_force_world: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrench (force, torque about body origin) on the body due to a force
        applied at the swing-leg foot.

        When the swing leg accelerates or its foot contacts something, the
        force is transmitted through the chain back to the body as a reaction
        wrench (Newton's 3rd law, geometric/static case):

            F_body = −F_foot
            τ_body = −(r_foot^W × F_foot)

        where r_foot^W = R_body @ (hip_i + foot_hip_i) is the foot position
        relative to body origin expressed in the world frame.

        For inertial effects during fast swing, augment with the
        full rigid-body dynamics of the leg links.

        Returns
        -------
        force  : (3,) reaction force on body, world frame
        torque : (3,) reaction torque on body about body origin, world frame
        """
        R  = rpy_to_rot(*self.body_rpy)
        ho = np.array(HIP_ORIGIN[swing_leg])
        r  = R @ (ho + leg_fk(swing_leg, *self._q_leg(swing_leg, q_all)))
        f  = np.asarray(foot_force_world, dtype=float)
        return -f, -np.cross(r, f)

    def grf_distribution(
        self,
        q_all: List[float],
        body_wrench: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Minimum-norm ground reaction force distribution among stance legs.

        Solves the contact-force balance:

            A^T @ f_contacts = body_wrench      (6 equations, 3n unknowns)

        where A is the (3n×6) support Jacobian and f_contacts is the stacked
        foot-force vector.  The pseudo-inverse gives the minimum L2-norm
        solution, corresponding to equal load sharing in the least-squares
        sense.

        Typical use: pass the reaction wrench of the swing leg as body_wrench
        to see how it distributes among the stance feet.

        Parameters
        ----------
        q_all       : 12 joint angles
        body_wrench : (6,) [fx, fy, fz, τx, τy, τz] in world frame

        Returns
        -------
        forces : dict  leg → (3,) GRF in world frame
        """
        A = self.support_jacobian(q_all, self.stance_legs)   # (3n, 6)
        f = np.linalg.lstsq(A.T, np.asarray(body_wrench, dtype=float),
                            rcond=None)[0]
        return {leg: f[3*i: 3*i+3] for i, leg in enumerate(self.stance_legs)}


# ── ZMP and support polygon utilities ────────────────────────────────────────

def _polygon_contains_2d(
    point: np.ndarray,
    pts: List[np.ndarray],
    margin: float = 0.0,
) -> bool:
    """
    Test if a 2D point lies inside a convex polygon with inward safety margin.

    pts    : polygon vertices (any order; only x/y used).
    margin : required inward clearance from each edge in metres (>0 = strict interior).
    """
    n = len(pts)
    if n < 3:
        return False
    cx = sum(float(p[0]) for p in pts) / n
    cy = sum(float(p[1]) for p in pts) / n
    ordered = sorted(pts, key=lambda p: math.atan2(float(p[1]) - cy, float(p[0]) - cx))
    px, py = float(point[0]), float(point[1])
    for i in range(n):
        ax, ay = float(ordered[i][0]),        float(ordered[i][1])
        bx, by = float(ordered[(i+1)%n][0]),  float(ordered[(i+1)%n][1])
        ex, ey = bx - ax, by - ay
        elen   = math.sqrt(ex*ex + ey*ey)
        if elen < 1e-9:
            continue
        # CCW polygon: interior ↔ cross product > 0 for every edge.
        # Signed distance (positive = interior side) = cross / elen.
        cross = ex * (py - ay) - ey * (px - ax)
        if cross / elen < margin:
            return False
    return True


def _tripod_incenter(pts3: List[np.ndarray]) -> np.ndarray:
    """
    Incenter of a triangle: the point equidistant from all three edges.

    Returns a 2D array (x, y).  Placing the CoM here before a foot lift
    maximises ZMP distance from every edge of the remaining tripod polygon.
    """
    a = np.asarray(pts3, dtype=float)[:, :2]   # (3, 2)
    sides = [
        np.linalg.norm(a[1] - a[2]),   # opposite vertex 0
        np.linalg.norm(a[0] - a[2]),   # opposite vertex 1
        np.linalg.norm(a[0] - a[1]),   # opposite vertex 2
    ]
    return (sides[0]*a[0] + sides[1]*a[1] + sides[2]*a[2]) / sum(sides)


# ── webapp ────────────────────────────────────────────────────────────────────

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_M_FRAME = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)


def _r2v(p):
    """Robot [x,y,z] → Three.js [x, z, -y]."""
    return [float(p[0]), float(p[2]), float(-p[1])]


def _mat_to_quat(R):
    """Rotation matrix → [x, y, z, w] float list."""
    t = float(R[0, 0] + R[1, 1] + R[2, 2])
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        return [float((R[2,1]-R[1,2])*s), float((R[0,2]-R[2,0])*s),
                float((R[1,0]-R[0,1])*s), float(0.25 / s)]
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + float(R[0,0]) - float(R[1,1]) - float(R[2,2]))
        return [float(0.25*s), float((R[0,1]+R[1,0])/s),
                float((R[0,2]+R[2,0])/s), float((R[2,1]-R[1,2])/s)]
    if R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + float(R[1,1]) - float(R[0,0]) - float(R[2,2]))
        return [float((R[0,1]+R[1,0])/s), float(0.25*s),
                float((R[1,2]+R[2,1])/s), float((R[0,2]-R[2,0])/s)]
    s = 2.0 * math.sqrt(1.0 + float(R[2,2]) - float(R[0,0]) - float(R[1,1]))
    return [float((R[0,2]+R[2,0])/s), float((R[1,2]+R[2,1])/s),
            float(0.25*s), float((R[1,0]-R[0,1])/s)]


class AppState:
    _STAND_Q  = [0.0, 0.67, -1.3] * 4
    _STEP_ORD = ["RL", "FL", "RR", "FR"]
    _LIFT_Q   = [0.0, 0.9, -1.8]   # raised swing leg: knee tucked, foot above ground

    def __init__(self):
        self._lock         = threading.Lock()
        self._crawl_height = 0.28    # body height used during crawl stepping (m)
        self._step_dr      = 0.12    # foot reach: placed dr ahead of hip projection
        self._step_dphi    = 0.0     # step direction offset from body heading (rad)
        self._zmp_margin   = 0.02    # required ZMP clearance from polygon edge (m)
        self._reset_locked()

    def _reset_locked(self):
        self._pk   = ParallelKinBody()
        self._q    = list(self._STAND_Q)
        self._sidx = 0
        self._pk.init_contacts_from_pose(
            np.array([0.0, 0.0, self._crawl_height]), np.zeros(3), self._q)

    def _knee_in_hip(self, leg, qh, qt):
        s      = LEG_SIGNS[leg]["left"]
        sh, ch = math.sin(qh), math.cos(qh)
        st, ct = math.sin(qt),  math.cos(qt)
        return np.array([
            -THIGH_LENGTH * st,
             s * HIP_LATERAL_OFFSET * ch + THIGH_LENGTH * ct * sh,
             s * HIP_LATERAL_OFFSET * sh - THIGH_LENGTH * ct * ch,
        ])

    def _prepare_body_for_lift(self, leg: str) -> None:
        """
        Shift body CoM to the incenter of the remaining 3-foot support
        triangle before lifting `leg`.  Guarantees the ZMP stays strictly
        inside the tripod polygon throughout and after the stance transition.
        """
        remaining = [l for l in self._pk.stance_legs if l != leg]
        if len(remaining) < 3:
            return
        pts    = [self._pk.foot_contacts[l] for l in remaining]
        inc_xy = _tripod_incenter(pts)
        target = np.array([inc_xy[0], inc_xy[1], self._crawl_height])
        rpy    = self._pk.body_rpy.copy()
        try:
            q = self._pk.body_ik(target, rpy, support_legs=remaining)
            for sl in remaining:
                for idx in LEG_INDEX[sl]:
                    if math.isfinite(q[idx]):
                        self._q[idx] = q[idx]
        except Exception:
            pass

    def _state(self):
        pos  = self._pk.body_pos
        rpy  = self._pk.body_rpy
        R    = rpy_to_rot(*rpy)
        R3   = _M_FRAME @ R @ _M_FRAME.T
        quat = _mat_to_quat(R3)

        stance = self._pk.stance_legs
        swing  = self._pk.swing_legs

        legs = {}
        for leg in LEG_ORDER:
            ho         = np.array(HIP_ORIGIN[leg])
            qh, qt, qc = self._pk._q_leg(leg, self._q)
            knee_h     = self._knee_in_hip(leg, qh, qt)
            foot_h     = leg_fk(leg, qh, qt, qc)
            in_stance  = leg in stance
            legs[leg]  = {
                "hip":     _r2v(pos + R @ ho),
                "knee":    _r2v(pos + R @ (ho + knee_h)),
                "foot":    _r2v(pos + R @ (ho + foot_h)),
                # stance → stored contact; swing → last touchdown (grayed)
                "contact": _r2v(self._pk.foot_contacts[leg]),
                "stance":  in_stance,
            }

        # Condition number of the *current* support structure (stance only)
        try:
            A    = self._pk.support_jacobian(self._q, stance) if stance else None
            cond = float(np.linalg.cond(A)) if A is not None else 9999.0
        except Exception:
            cond = 9999.0
        if not math.isfinite(cond):
            cond = 9999.0

        # ZMP: quasi-static approximation for slow crawling (CoM → ground)
        zmp_world = np.array([pos[0], pos[1], 0.0])
        stance_xy = [self._pk.foot_contacts[l][:2] for l in stance]
        zmp_ok = (
            _polygon_contains_2d(zmp_world[:2], stance_xy, margin=self._zmp_margin)
            if len(stance) >= 3 else bool(len(stance) >= 1)
        )

        next_leg   = self._STEP_ORD[self._sidx % 4]
        next_phase = "land" if next_leg in swing else "lift"

        return {
            "body_pos":        _r2v(pos),
            "body_quat":       quat,
            "body_rpy":        [round(float(v), 4) for v in rpy],
            "legs":            legs,
            "stance_legs":     stance,
            "swing_legs":      swing,
            "support_polygon": [_r2v(self._pk.foot_contacts[l]) for l in stance],
            "cond_num":        round(cond, 2),
            "joint_angles":    {l: [round(float(v), 4) for v in self._pk._q_leg(l, self._q)]
                                for l in LEG_ORDER},
            "step_next":       next_leg,
            "step_phase":      next_phase,
            "zmp":             _r2v(zmp_world),
            "zmp_ok":          bool(zmp_ok),
            "crawl_height":    round(float(self._crawl_height), 3),
            "step_dr":         round(float(self._step_dr), 3),
            "step_dphi":       round(float(self._step_dphi), 4),
        }

    def render_state(self):
        with self._lock:
            return self._state()

    def set_pose(self, pos, rpy):
        with self._lock:
            try:
                q = self._pk.body_ik(np.asarray(pos, float), np.asarray(rpy, float))
                for sl in self._pk.stance_legs:
                    for idx in LEG_INDEX[sl]:
                        if math.isfinite(q[idx]):
                            self._q[idx] = q[idx]
            except Exception:
                pass

    def set_crawl_height(self, height: float) -> None:
        """Update the body height used during automated crawl stepping."""
        with self._lock:
            self._crawl_height = float(np.clip(height, 0.18, 0.42))

    def set_step_params(self, dr: float, dphi: float) -> None:
        """
        Set foot placement parameters for the swing leg.

        dr   : radial reach — foot placed dr metres from the hip projection.
        dphi : direction offset from body heading in radians.
        """
        with self._lock:
            self._step_dr   = float(np.clip(dr,   0.02, 0.30))
            self._step_dphi = float(np.clip(dphi, -math.pi, math.pi))

    def step_foot(self, leg=None):
        """
        ZMP-stable two-phase crawl step (3 feet always on ground).

        LIFT phase
        ----------
        1. Shift body CoM to the incenter of the remaining 3-foot tripod
           → ZMP guaranteed inside the triangle before and after lift.
        2. Remove leg from PKM stance set; raise swing joints.
        3. Re-solve stance leg joints via parallel IK at new body pose.

        LAND phase
        ----------
        1. Compute new contact: hip_world_XY + dr·[cos(yaw+dphi), sin(yaw+dphi)].
        2. Add leg to PKM stance set.
        3. Parallel IK over all four legs.
        """
        with self._lock:
            if leg is None:
                leg = self._STEP_ORD[self._sidx % 4]

            if leg in self._pk.stance_legs:
                # ── LIFT ──────────────────────────────────────────────────────
                # 1. Shift CoM to incenter of remaining 3-foot polygon
                self._prepare_body_for_lift(leg)
                # 2. Remove from PKM topology
                self._pk.lift_leg(leg)
                # 3. Raise swing leg
                hi, ti, ci = LEG_INDEX[leg]
                self._q[hi] = self._LIFT_Q[0]
                self._q[ti] = self._LIFT_Q[1]
                self._q[ci] = self._LIFT_Q[2]
                # 4. Recompute remaining stance joints at shifted body pose
                try:
                    q_st = self._pk.body_ik(
                        self._pk.body_pos.copy(), self._pk.body_rpy.copy()
                    )
                    for sl in self._pk.stance_legs:
                        for idx in LEG_INDEX[sl]:
                            if math.isfinite(q_st[idx]):
                                self._q[idx] = q_st[idx]
                except Exception:
                    pass

            else:
                # ── LAND ──────────────────────────────────────────────────────
                rpy = self._pk.body_rpy.copy()
                yaw = rpy[2]
                R   = rpy_to_rot(*rpy)
                ho  = np.array(HIP_ORIGIN[leg])
                # Target contact: below hip + (dr, dphi) displacement
                hip_world   = self._pk.body_pos + R @ ho
                new_contact = np.array([
                    hip_world[0] + self._step_dr * math.cos(yaw + self._step_dphi),
                    hip_world[1] + self._step_dr * math.sin(yaw + self._step_dphi),
                    0.0,
                ])
                self._pk.land_leg(leg, new_contact)
                try:
                    q = self._pk.body_ik(self._pk.body_pos.copy(), rpy)
                    if all(math.isfinite(v) for v in q):
                        self._q = q
                except Exception:
                    pass
                if leg == self._STEP_ORD[self._sidx % 4]:
                    self._sidx += 1

    def reset(self):
        with self._lock:
            self._reset_locked()


# ── HTML page ─────────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Go2 Reconfigurable PKM</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#e0e0e0;font-family:monospace;
     display:flex;height:100vh;overflow:hidden}
#cw{flex:1;position:relative}
canvas{display:block}
#panel{width:310px;min-width:260px;background:#161b22;overflow-y:auto;
       padding:12px;display:flex;flex-direction:column;gap:10px;
       border-left:1px solid #30363d}
h2{color:#58a6ff;font-size:11px;letter-spacing:1px;
   border-bottom:1px solid #21262d;padding-bottom:3px;margin-top:2px}
.row{display:flex;align-items:center;gap:6px}
.row label{width:52px;font-size:10px;color:#8b949e}
.row input[type=range]{flex:1;accent-color:#58a6ff;height:14px}
.row .val{width:42px;font-size:10px;color:#79c0ff;text-align:right}
.brow{display:flex;gap:5px;flex-wrap:wrap}
button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;
       padding:3px 8px;font-size:10px;cursor:pointer;border-radius:3px;
       font-family:monospace}
button:hover{background:#58a6ff;color:#000;border-color:#58a6ff}
button.go{background:#1f4d24;border-color:#3fb950}
button.go:hover{background:#3fb950;color:#000}
.kinfo{font-size:10px;line-height:1.8;color:#8b949e}
#cond-val{font-size:11px;font-weight:bold}
.ok{color:#3fb950}.bad{color:#f85149}
.topo{font-size:10px;line-height:1.6}
.topo .stance{color:#3fb950}.topo .swing{color:#cc44ff}
table{width:100%;border-collapse:collapse;font-size:10px}
td{padding:1px 3px}
td:first-child{width:36px}
td:not(:first-child){color:#79c0ff;text-align:right}
.s-badge{color:#3fb950;font-weight:bold}.w-badge{color:#cc44ff;font-weight:bold}
#legend{font-size:9px;color:#6e7681;line-height:1.9}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
</style>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
</head>
<body>
<div id="cw"><canvas id="c"></canvas></div>
<div id="panel">
  <h2>BODY POSE</h2>
  <div class="row"><label>roll</label>
    <input type="range" id="s-roll"   min="-0.4" max="0.4"  step="0.005" value="0">
    <span class="val" id="v-roll">0.000</span></div>
  <div class="row"><label>pitch</label>
    <input type="range" id="s-pitch"  min="-0.4" max="0.4"  step="0.005" value="0">
    <span class="val" id="v-pitch">0.000</span></div>
  <div class="row"><label>yaw</label>
    <input type="range" id="s-yaw"    min="-1.0" max="1.0"  step="0.01"  value="0">
    <span class="val" id="v-yaw">0.000</span></div>
  <div class="row"><label>height</label>
    <input type="range" id="s-height" min="0.18" max="0.44" step="0.005" value="0.28">
    <span class="val" id="v-height">0.280</span></div>
  <div class="row"><label>x fwd</label>
    <input type="range" id="s-x"      min="-0.15" max="0.15" step="0.005" value="0">
    <span class="val" id="v-x">0.000</span></div>
  <div class="row"><label>y left</label>
    <input type="range" id="s-y"      min="-0.15" max="0.15" step="0.005" value="0">
    <span class="val" id="v-y">0.000</span></div>

  <h2>TOPOLOGY</h2>
  <div class="topo">
    stance (PKM chains):&nbsp;<span id="topo-stance" class="stance">--</span><br>
    swing&nbsp;(serial/EE):&nbsp;<span id="topo-swing"  class="swing">--</span>
  </div>

  <h2>STEP CONTROL</h2>
  <div class="brow">
    <button id="b-auto" class="go">Auto Step</button>
    <button id="b-RL" onclick="stepFoot('RL')">RL ↑</button>
    <button id="b-FL" onclick="stepFoot('FL')">FL ↑</button>
    <button id="b-RR" onclick="stepFoot('RR')">RR ↑</button>
    <button id="b-FR" onclick="stepFoot('FR')">FR ↑</button>
    <button onclick="doReset()">Reset</button>
  </div>
  <div class="kinfo">
    next:&nbsp;<span id="step-next" style="color:#79c0ff">--</span>
    &nbsp;<span id="step-phase" style="color:#8b949e"></span>
  </div>
  <div class="row"><label>crawl h</label>
    <input type="range" id="s-ch" min="0.20" max="0.38" step="0.005" value="0.28">
    <span class="val" id="v-ch">0.280</span></div>
  <div class="row"><label>step dr</label>
    <input type="range" id="s-dr" min="0.04" max="0.25" step="0.005" value="0.12">
    <span class="val" id="v-dr">0.120</span></div>
  <div class="row"><label>step φ°</label>
    <input type="range" id="s-dphi" min="-90" max="90" step="1" value="0">
    <span class="val" id="v-dphi">0°</span></div>

  <h2>KINEMATICS</h2>
  <div class="kinfo">
    <div>cond(A):&nbsp;<span id="cond-val" class="ok">--</span></div>
    <div>ZMP:&nbsp;<span id="zmp-ok" class="ok">--</span></div>
    <div style="font-size:9px;color:#6e7681;margin-top:2px">
      A = stance support Jacobian (3n&times;6)<br>
      ZMP = quasi-static CoM projection<br>
      margin = 2 cm from polygon edge
    </div>
  </div>

  <h2>JOINT ANGLES (rad)</h2>
  <table id="jt">
    <tr><td></td><td>hip</td><td>thigh</td><td>calf</td></tr>
  </table>

  <h2>LEGEND</h2>
  <div id="legend">
    <div><span class="dot" style="background:#f0c040"></span>hip joint</div>
    <div><span class="dot" style="background:#40d0e0"></span>knee joint</div>
    <div><span class="dot" style="background:#ff8040"></span>stance foot / contact</div>
    <div><span class="dot" style="background:#cc44ff"></span>swing foot (serial FK)</div>
    <div><span class="dot" style="background:#ff4444"></span>stance base (PKM anchor)</div>
    <div><span class="dot" style="background:#444444"></span>last contact (ghost)</div>
    <div><span class="dot" style="background:#ffd700;border-radius:0"></span>support polygon (stance only)</div>
  </div>
</div>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const wrap = document.getElementById('cw');
const W = () => wrap.clientWidth, H = () => wrap.clientHeight;

const renderer = new THREE.WebGLRenderer({canvas: document.getElementById('c'), antialias: true});
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(W(), H());

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
scene.fog = new THREE.Fog(0x0d1117, 4, 10);

const camera = new THREE.PerspectiveCamera(55, W()/H(), 0.01, 20);
camera.position.set(1.0, 0.7, 0.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.15, 0);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dl = new THREE.DirectionalLight(0xffffff, 0.9);
dl.position.set(2, 4, 2);
scene.add(dl);

const grid = new THREE.GridHelper(3, 30, 0x21262d, 0x161b22);
scene.add(grid);

// ZMP indicator: flat sphere on ground, green=inside, red=outside
const zmpMat = new THREE.MeshLambertMaterial({color:0x00ff88, transparent:true, opacity:0.90});
const zmpSphere = new THREE.Mesh(new THREE.SphereGeometry(0.022, 12, 8), zmpMat);
scene.add(zmpSphere);

// ── materials ──────────────────────────────────────────────────────────────
// shared
const matBody  = new THREE.MeshLambertMaterial({color:0x2d5fa0, transparent:true, opacity:0.80});
const matHip   = new THREE.MeshLambertMaterial({color:0xf0c040});
const matKnee  = new THREE.MeshLambertMaterial({color:0x40d0e0});
const lPoly    = new THREE.LineBasicMaterial({color:0xffd700});
const matFill  = new THREE.MeshBasicMaterial({color:0xffd700, transparent:true, opacity:0.08,
                                              side:THREE.DoubleSide, depthWrite:false});
// stance leg materials (PKM chain)
const matFoot_st    = new THREE.MeshLambertMaterial({color:0xff8040});
const matContact_st = new THREE.MeshLambertMaterial({color:0xff4444});
const lThigh_st     = new THREE.LineBasicMaterial({color:0x4a7acc});
const lCalf_st      = new THREE.LineBasicMaterial({color:0x3a9a55});
const lAnchor_st    = new THREE.LineBasicMaterial({color:0x663322});
// swing leg materials (serial manipulator on platform)
const matFoot_sw    = new THREE.MeshLambertMaterial({color:0xcc44ff});
const matContact_sw = new THREE.MeshLambertMaterial({color:0x3a3a3a, transparent:true, opacity:0.5});
const lThigh_sw     = new THREE.LineBasicMaterial({color:0x7733aa, transparent:true, opacity:0.7});
const lCalf_sw      = new THREE.LineBasicMaterial({color:0x993388, transparent:true, opacity:0.7});
const lAnchor_sw    = new THREE.LineBasicMaterial({color:0x333333, transparent:true, opacity:0.4});

function mkLine(mat) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
  return new THREE.Line(g, mat);
}
function setLine(ln, a, b) {
  const p = ln.geometry.attributes.position;
  p.setXYZ(0, a.x, a.y, a.z); p.setXYZ(1, b.x, b.y, b.z);
  p.needsUpdate = true;
  ln.geometry.computeBoundingSphere();
}
const v3 = a => new THREE.Vector3(a[0], a[1], a[2]);

// ── body box ───────────────────────────────────────────────────────────────
const bodyMesh = new THREE.Mesh(new THREE.BoxGeometry(0.47, 0.05, 0.14), matBody);
scene.add(bodyMesh);
const bodyWire = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(0.47, 0.05, 0.14)),
  new THREE.LineBasicMaterial({color:0x79c0ff, opacity:0.5, transparent:true}));
bodyMesh.add(bodyWire);

// ── per-leg objects ────────────────────────────────────────────────────────
const LEGS = ['FR','FL','RR','RL'];
const lo = {};
for (const leg of LEGS) {
  const hipS     = new THREE.Mesh(new THREE.SphereGeometry(0.018, 12, 8), matHip);
  const kneeS    = new THREE.Mesh(new THREE.SphereGeometry(0.013, 12, 8), matKnee);
  const footS    = new THREE.Mesh(new THREE.SphereGeometry(0.020, 12, 8), matFoot_st);
  const contactS = new THREE.Mesh(new THREE.SphereGeometry(0.024, 12, 8), matContact_st);
  const thighL   = mkLine(lThigh_st);
  const calfL    = mkLine(lCalf_st);
  const anchorL  = mkLine(lAnchor_st);
  scene.add(hipS, kneeS, footS, contactS, thighL, calfL, anchorL);
  lo[leg] = {hipS, kneeS, footS, contactS, thighL, calfL, anchorL};
}

// ── support polygon (variable size: only stance feet) ─────────────────────
const polyGeo  = new THREE.BufferGeometry();
const polyLoop = new THREE.LineLoop(polyGeo, lPoly);
scene.add(polyLoop);
const fillMesh = new THREE.Mesh(new THREE.BufferGeometry(), matFill);
scene.add(fillMesh);

function updatePolygon(pts) {
  // pts: array of THREE.Vector3, stance contact positions
  const n = pts.length;
  if (n < 2) { polyLoop.visible = false; fillMesh.visible = false; return; }
  polyLoop.visible = true;
  // sort CCW around centroid (looking down -Y in Three.js = looking down -z_robot)
  const cx = pts.reduce((s,p)=>s+p.x,0)/n;
  const cz = pts.reduce((s,p)=>s+p.z,0)/n;
  const sorted = [...pts].sort((a,b)=>
    Math.atan2(a.z-cz,a.x-cx) - Math.atan2(b.z-cz,b.x-cx));
  const buf = new Float32Array(n*3);
  sorted.forEach((p,i)=>{buf[i*3]=p.x;buf[i*3+1]=p.y;buf[i*3+2]=p.z;});
  polyGeo.setAttribute('position', new THREE.BufferAttribute(buf, 3));
  polyGeo.computeBoundingSphere();
  if (n >= 3) {
    fillMesh.visible = true;
    const fv = [];
    for (let i=1;i<n-1;i++) {
      fv.push(sorted[0].x,sorted[0].y,sorted[0].z,
              sorted[i].x,sorted[i].y,sorted[i].z,
              sorted[i+1].x,sorted[i+1].y,sorted[i+1].z);
    }
    fillMesh.geometry.setAttribute('position',
      new THREE.BufferAttribute(new Float32Array(fv), 3));
    fillMesh.geometry.computeBoundingSphere();
  } else {
    fillMesh.visible = false;
  }
}

// ── apply state ────────────────────────────────────────────────────────────
function applyState(s) {
  const bp = v3(s.body_pos), bq = s.body_quat;
  bodyMesh.position.copy(bp);
  bodyMesh.quaternion.set(bq[0], bq[1], bq[2], bq[3]);

  for (const leg of LEGS) {
    const d  = s.legs[leg], o = lo[leg];
    const st = d.stance;
    const hp = v3(d.hip), kp = v3(d.knee), fp = v3(d.foot), cp = v3(d.contact);
    o.hipS.position.copy(hp);
    o.kneeS.position.copy(kp);
    o.footS.position.copy(fp);
    o.contactS.position.copy(cp);
    setLine(o.thighL, hp, kp);
    setLine(o.calfL,  kp, fp);
    // anchor: stance → contact→foot chain base; swing → ghost last contact
    setLine(o.anchorL, cp, fp);
    // swap materials based on stance/swing role
    o.footS.material    = st ? matFoot_st    : matFoot_sw;
    o.contactS.material = st ? matContact_st : matContact_sw;
    o.thighL.material   = st ? lThigh_st     : lThigh_sw;
    o.calfL.material    = st ? lCalf_st      : lCalf_sw;
    o.anchorL.material  = st ? lAnchor_st    : lAnchor_sw;
    // update per-leg step button text
    const btn = document.getElementById('b-'+leg);
    if (btn) btn.textContent = leg + (st ? ' ↑' : ' ↓');
  }

  // support polygon from stance contacts only
  updatePolygon(s.support_polygon.map(v3));

  // ZMP sphere
  const zp = v3(s.zmp);
  zmpSphere.position.set(zp.x, 0.006, zp.z);
  zmpMat.color.setHex(s.zmp_ok ? 0x00ff88 : 0xff2222);
  const zmEl = document.getElementById('zmp-ok');
  zmEl.textContent = s.zmp_ok ? 'inside ✓' : 'OUTSIDE ✗';
  zmEl.className   = s.zmp_ok ? 'ok' : 'bad';

  // kinematics panel
  const cel = document.getElementById('cond-val');
  const cn  = s.cond_num;
  cel.textContent = cn >= 9999 ? '∞' : cn.toFixed(2);
  cel.className   = cn > 80 ? 'bad' : 'ok';

  // topology panel
  document.getElementById('topo-stance').textContent =
    s.stance_legs.length ? s.stance_legs.join(' ') : 'none';
  document.getElementById('topo-swing').textContent  =
    s.swing_legs.length  ? s.swing_legs.join(' ')  : 'none';

  // step info
  document.getElementById('step-next').textContent  = s.step_next;
  document.getElementById('step-phase').textContent = '[' + s.step_phase + ']';

  // sync step param display from server state
  const chEl = document.getElementById('s-ch');
  if (Math.abs(+chEl.value - s.crawl_height) > 0.001) {
    chEl.value = s.crawl_height;
    document.getElementById('v-ch').textContent = s.crawl_height.toFixed(3);
  }

  // joint angle table with S/W badge
  const tbl = document.getElementById('jt');
  tbl.innerHTML = '<tr><td></td><td>hip</td><td>thigh</td><td>calf</td></tr>';
  for (const leg of LEGS) {
    const [qh,qt,qc] = s.joint_angles[leg];
    const st   = s.legs[leg].stance;
    const badge = st ? `<span class="s-badge">S</span>` : `<span class="w-badge">W</span>`;
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${badge} ${leg}</td><td>${qh.toFixed(3)}</td><td>${qt.toFixed(3)}</td><td>${qc.toFixed(3)}</td>`;
    tbl.appendChild(tr);
  }
}

async function poll() {
  try { const r = await fetch('/state'); applyState(await r.json()); } catch(_) {}
  setTimeout(poll, 100);
}
poll();

function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
animate();

new ResizeObserver(() => {
  renderer.setSize(W(), H());
  camera.aspect = W()/H();
  camera.updateProjectionMatrix();
}).observe(wrap);

// ── sliders ────────────────────────────────────────────────────────────────
const SL = {
  'roll':  {sid:'s-roll',   vid:'v-roll'},
  'pitch': {sid:'s-pitch',  vid:'v-pitch'},
  'yaw':   {sid:'s-yaw',    vid:'v-yaw'},
  'height':{sid:'s-height', vid:'v-height'},
  'x':     {sid:'s-x',      vid:'v-x'},
  'y':     {sid:'s-y',      vid:'v-y'},
};
function readPose() {
  return {
    pos: [+document.getElementById('s-x').value,
          +document.getElementById('s-y').value,
          +document.getElementById('s-height').value],
    rpy: [+document.getElementById('s-roll').value,
          +document.getElementById('s-pitch').value,
          +document.getElementById('s-yaw').value],
  };
}
for (const [, cfg] of Object.entries(SL)) {
  document.getElementById(cfg.sid).addEventListener('input', function() {
    document.getElementById(cfg.vid).textContent = (+this.value).toFixed(3);
    fetch('/set_pose', {method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify(readPose())});
  });
}

// Sync height slider → crawl height so stepping matches manual pose
document.getElementById('s-height').addEventListener('input', function() {
  fetch('/set_crawl_height', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({height: +this.value})});
});

// Step parameter sliders
document.getElementById('s-ch').addEventListener('input', function() {
  document.getElementById('v-ch').textContent = (+this.value).toFixed(3);
  fetch('/set_crawl_height', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({height: +this.value})});
});
document.getElementById('s-dr').addEventListener('input', function() {
  document.getElementById('v-dr').textContent = (+this.value).toFixed(3);
  _sendStepParams();
});
document.getElementById('s-dphi').addEventListener('input', function() {
  document.getElementById('v-dphi').textContent = (+this.value).toFixed(0) + '°';
  _sendStepParams();
});
function _sendStepParams() {
  fetch('/set_step_params', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      dr:   +document.getElementById('s-dr').value,
      dphi: +document.getElementById('s-dphi').value * Math.PI / 180,
    })});
}

window.stepFoot = function(leg) {
  const body = leg ? JSON.stringify({leg}) : '{}';
  fetch('/step_foot', {method:'POST', headers:{'Content-Type':'application/json'}, body});
};

window.doReset = function() {
  fetch('/reset', {method:'POST'}).then(() => {
    for (const [, cfg] of Object.entries(SL))
      document.getElementById(cfg.sid).value = cfg.sid === 's-height' ? '0.34' : '0';
    document.getElementById('v-height').textContent = '0.340';
    ['roll','pitch','yaw','x','y'].forEach(k =>
      document.getElementById('v-'+k).textContent = '0.000');
  });
};

let autoTimer = null;
document.getElementById('b-auto').addEventListener('click', function() {
  if (autoTimer) {
    clearInterval(autoTimer); autoTimer = null;
    this.textContent = 'Auto Step'; this.className = 'go';
  } else {
    autoTimer = setInterval(() => window.stepFoot(), 750);
    this.textContent = 'Stop Auto'; this.className = '';
  }
});
</script>
</body>
</html>"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

_app = AppState()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        if self.path == '/':
            self._send(200, 'text/html', HTML_PAGE.encode())
        elif self.path == '/state':
            self._send(200, 'application/json',
                       json.dumps(_app.render_state()).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        n    = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(n)) if n else {}
        if self.path == '/set_pose':
            _app.set_pose(body.get('pos', [0, 0, 0.28]),
                          body.get('rpy', [0, 0, 0]))
        elif self.path == '/step_foot':
            _app.step_foot(body.get('leg'))
        elif self.path == '/set_crawl_height':
            _app.set_crawl_height(body.get('height', 0.28))
        elif self.path == '/set_step_params':
            _app.set_step_params(body.get('dr', 0.12), body.get('dphi', 0.0))
        elif self.path == '/reset':
            _app.reset()
        else:
            self.send_error(404)
            return
        self._send(200, 'application/json', b'{"ok":true}')

    def _send(self, code, ctype, data):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data)


if __name__ == '__main__':
    import argparse
    import sys
    ap = argparse.ArgumentParser(description='Go2 parallel-kin webapp')
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args()
    srv  = ThreadingHTTPServer(('', args.port), _Handler)
    print(f'Parallel-kin visualizer → http://localhost:{args.port}')
    print('Ctrl-C to stop.')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nshutdown')
        srv.shutdown()
        sys.exit(0)
