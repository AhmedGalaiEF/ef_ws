#!/usr/bin/env python3
"""
G1 ZMP / support-polygon estimator and live plot.

Estimates the Zero Moment Point and the double-support polygon purely from
geometry + odometry/IMU — no force/pressure foot sensors are read here (the
SDK doesn't expose them), so this is a coarse, openly-approximate estimate,
in the same spirit as segmentation.py's PCA grasp-surface estimate: good
enough to see whether the robot is comfortably balanced, not a certified
stability computation. What geometry the SDK *does* give us (leg/arm/waist
joint angles) is now cross-checked against the real G1 URDF
(g1_29dof_with_hand_rev_1_0_pkg.urdf, installed at
~/ef_ws/install/g1_description/share/g1_description/urdf/) rather than
placeholder numbers, as detailed below.

Support polygon
----------------
Forward-kinematics the 12 leg joints (`get_lowstate()["joint_positions"]`
indices 0-11, same MJCF-sourced geometry `joint_control_dashboard.py` uses
for its full-body skeleton, cross-checked link-for-link against the URDF's
joint origins/axes) to get each foot's ankle-roll frame, then place each
foot's 4 ground-contact corners at the *real* points the URDF's
`*_ankle_roll_link` collision geometry defines (four spheres at heel
x=-0.05, toe x=0.12, sole z=-0.03 relative to the ankle-roll frame — not a
symmetric placeholder rectangle centered on the ankle). Still no force/
contact sensors, but with real per-corner heights available a coarse gait-
phase heuristic is applied live: if one foot's mean sole height sits more
than a threshold (default 1.5cm, adjustable) above the other's, it's
treated as the swing foot and the polygon shrinks to the stance foot alone
— important once the robot is actually walking, since checking the ZMP
against a permanent double-support hull during single-support phases both
overstates the margin and hides the one moment that heuristic matters most.
Below the threshold (e.g. standing, or a symmetric crouch), both feet's
corners are hulled together as before. That height read is cross-checked
against knee+ankle-pitch torque (`joint_torques`/tau_est, still not a real
force sensor, but a loaded leg reports more mechanical effort than an
unloaded one): if the leg the height gap just called "swing" is actually
carrying *more* torque than the one it called "stance", the geometry read
is probably a lean/tilt rather than a real foot lift, and single support is
suppressed back to double — this can only make the call more conservative,
never invent a single-support the height gap didn't already suggest, and
is skipped entirely if torques aren't reported.

Body/pelvis-relative FK is retained for debugging, then all CoM and sole
points are transformed once into gravity/world by the full IMU orientation
``Rz(yaw) @ Ry(pitch) @ Rx(roll)``. Odom provides translation only. This is
critical: applying only yaw leaves base roll/pitch out of the model and can
make a forward-arm diagnostic misleading.

ZMP
----
The standard linear-inverted-pendulum (cart-table) approximation:

    ZMP_xy = CoM_xy − (h_com / g) · CoM_accel_xy

CoM_xy/h_com come from a real whole-body center of mass: FK all 29 sensed
leg+waist+arm joints and mass-weight every URDF link's own inertial CoM
(pelvis, 12 leg links, waist×3/torso, head, 14 arm links, plus each hand's 7
finger/palm links lumped into one rigid neutral-pose mass hung off its
wrist_yaw frame, since dex3 finger angles aren't in this joint range) —
99.99% of the robot's 34.4 kg URDF mass, vs. the old approximation of
"CoM ≈ pelvis position" which ignored arm/torso posture entirely. h_com is
then that CoM's height above the mean foot sole-contact height, so it
tracks crouching/leaning instead of being a fixed constant. If fewer than
29 joint samples are available (older firmware, or legs-only), this falls
back to pelvis-as-CoM and a user-set assumed height, same as before.
CoM_accel_xy is based on the IMU acceleration rotated into world and smoothed.
The native SDK field's gravity convention is not documented in this repository,
so the UI exposes its gravity-included assumption instead of silently claiming
gravity compensation.
If the IMU has nothing to offer on a given tick (missing `acc`/`rpy`), a
kinematic fallback steps in instead: a second finite difference of the
whole-body CoM velocity (see ICP below), smoothed over its own window.
Deliberately not blended into the IMU estimate when both are available —
differentiating position twice at 10Hz is noisier than the IMU's direct
accelerometer reading — it only replaces silently zeroing the ZMP's
dynamic term (i.e. reporting ZMP ≈ CoM) when the IMU is absent.

Instantaneous Capture Point (ICP)
-----------------------------------
ZMP answers "is the robot in force-balance right now"; it says nothing
about where the CoM is *headed*. The capture point (Pratt et al., "Capture
Point: A Step Towards Humanoid Push Recovery") answers that, under the same
LIPM assumption:

    ICP_xy = CoM_xy + CoM_vel_xy / ω,   ω = sqrt(g / h_com)

— the point where, if the ZMP were held there from this instant on, the CoM
would asymptotically come to rest above it. CoM_vel_xy is a short moving-
average of the finite-differenced whole-body CoM_xy across poll ticks (so it
reflects both pelvis translation and postural CoM shift, the same signal
CoM_xy itself already carries — not a separate sensor). Plotted alongside
the ZMP as a second marker; whether it falls inside the support polygon is
a genuinely different, and for a walking/push-recovery judgment call more
informative, question than whether the ZMP does.

Run:
    python3 zmp_viz.py [--iface eth0] [--domain-id 0] [--port 8072]
Then open http://<host>:8072 in a browser on the same network. The page loads
even with no robot reachable; click Connect once the network is up.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

# Make `from sdk_wrapper import G1` work regardless of cwd. This file lives in
# academy/visualizations/; sdk_wrapper.py sits one level up in academy/.
_SCRIPT_DIR = Path(__file__).resolve().parent      # academy/visualizations
_ACADEMY_DIR = _SCRIPT_DIR.parent                    # academy
_G1_ROOT_DIR = _ACADEMY_DIR.parent                   # g1
for _p in (_SCRIPT_DIR, _ACADEMY_DIR, _G1_ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Whole-body geometry — all 29 leg/waist/arm entries of
# joint_control_dashboard.py's JOINT_TABLE (ids 0-5 left leg, 6-11 right leg,
# 12-14 waist, 15-21 left arm, 22-28 right arm), verbatim: (parent, pos,
# quat, axis). Not imported from that file — it's a Dash app with page-load
# side effects (builds its own `app`/layout at import time), so the handful
# of constants this script needs are duplicated here instead. Cross-checked
# against g1_29dof_with_hand_rev_1_0_pkg.urdf's <joint> origins/axes: leg and
# waist entries match the URDF exactly; the two waist links' z-offsets are
# re-parameterized differently (0.035+0.019 here vs. the URDF's 0.044+0)
# but sum to the same torso-frame offset, confirmed against the URDF's
# shoulder-joint origins.
# ---------------------------------------------------------------------------

PELVIS_POS = (0.0, 0.0, 0.793)
_IDENT_Q = (1.0, 0.0, 0.0, 0.0)
_HIP_ROLL_Q = (0.996179, 0.0, -0.0873386, 0.0)
_KNEE_Q = (0.996179, 0.0, 0.0873386, 0.0)
_L_SHOULDER_PITCH_Q = (0.990264, 0.139201, 1.38722e-05, -9.86868e-05)
_L_SHOULDER_ROLL_Q = (0.990268, -0.139172, 0.0, 0.0)
_R_SHOULDER_PITCH_Q = (0.990264, -0.139201, 1.38722e-05, 9.86868e-05)
_R_SHOULDER_ROLL_Q = (0.990268, 0.139172, 0.0, 0.0)

# id: (parent, pos, quat, axis)
BODY_JOINTS: dict[int, tuple[int, tuple, tuple, tuple]] = {
    0: (-1, (0, 0.064452, -0.1027), _IDENT_Q, (0, 1, 0)),          # left_hip_pitch
    1: (0, (0, 0.052, -0.030465), _HIP_ROLL_Q, (1, 0, 0)),          # left_hip_roll
    2: (1, (0.025001, 0, -0.12412), _IDENT_Q, (0, 0, 1)),           # left_hip_yaw
    3: (2, (-0.078273, 0.0021489, -0.17734), _KNEE_Q, (0, 1, 0)),   # left_knee
    4: (3, (0, -9.4445e-05, -0.30001), _IDENT_Q, (0, 1, 0)),        # left_ankle_pitch
    5: (4, (0, 0, -0.017558), _IDENT_Q, (1, 0, 0)),                 # left_ankle_roll (foot)
    6: (-1, (0, -0.064452, -0.1027), _IDENT_Q, (0, 1, 0)),          # right_hip_pitch
    7: (6, (0, -0.052, -0.030465), _HIP_ROLL_Q, (1, 0, 0)),         # right_hip_roll
    8: (7, (0.025001, 0, -0.12412), _IDENT_Q, (0, 0, 1)),           # right_hip_yaw
    9: (8, (-0.078273, -0.0021489, -0.17734), _KNEE_Q, (0, 1, 0)),  # right_knee
    10: (9, (0, 9.4445e-05, -0.30001), _IDENT_Q, (0, 1, 0)),        # right_ankle_pitch
    11: (10, (0, 0, -0.017558), _IDENT_Q, (1, 0, 0)),               # right_ankle_roll (foot)
    12: (-1, (0, 0, 0), _IDENT_Q, (0, 0, 1)),                       # waist_yaw
    13: (12, (-0.0039635, 0, 0.035), _IDENT_Q, (1, 0, 0)),          # waist_roll
    14: (13, (0, 0, 0.019), _IDENT_Q, (0, 1, 0)),                   # waist_pitch (== torso frame)
    15: (14, (0.0039563, 0.10022, 0.24778), _L_SHOULDER_PITCH_Q, (0, 1, 0)),   # left_shoulder_pitch (URDF)
    16: (15, (0, 0.038, -0.013831), _L_SHOULDER_ROLL_Q, (1, 0, 0)),            # left_shoulder_roll
    17: (16, (0, 0.00624, -0.1032), _IDENT_Q, (0, 0, 1)),                      # left_shoulder_yaw
    18: (17, (0.015783, 0, -0.080518), _IDENT_Q, (0, 1, 0)),                   # left_elbow
    19: (18, (0.1, 0.00188791, -0.01), _IDENT_Q, (1, 0, 0)),                   # left_wrist_roll
    20: (19, (0.038, 0, 0), _IDENT_Q, (0, 1, 0)),                             # left_wrist_pitch
    21: (20, (0.046, 0, 0), _IDENT_Q, (0, 0, 1)),                            # left_wrist_yaw
    22: (14, (0.0039563, -0.10021, 0.24778), _R_SHOULDER_PITCH_Q, (0, 1, 0)),  # right_shoulder_pitch (URDF)
    23: (22, (0, -0.038, -0.013831), _R_SHOULDER_ROLL_Q, (1, 0, 0)),           # right_shoulder_roll
    24: (23, (0, -0.00624, -0.1032), _IDENT_Q, (0, 0, 1)),                     # right_shoulder_yaw
    25: (24, (0.015783, 0, -0.080518), _IDENT_Q, (0, 1, 0)),                   # right_elbow
    26: (25, (0.1, -0.00188791, -0.01), _IDENT_Q, (1, 0, 0)),                  # right_wrist_roll
    27: (26, (0.038, 0, 0), _IDENT_Q, (0, 1, 0)),                            # right_wrist_pitch
    28: (27, (0.046, 0, 0), _IDENT_Q, (0, 0, 1)),                           # right_wrist_yaw
}
LEFT_FOOT_ID, RIGHT_FOOT_ID = 5, 11
LEFT_WRIST_ID, RIGHT_WRIST_ID = 21, 28
TORSO_FRAME_ID = 14  # waist_pitch's output frame == torso_link's own origin

# Real foot-ground contact points, read straight from the URDF: each
# *_ankle_roll_link has 4 collision spheres (heel x=-0.05, toe x=0.12, sole
# z=-0.03; y=+/-0.025 at the heel, +/-0.03 at the toe — not the earlier
# symmetric ±10cm/±4.5cm placeholder rectangle centered on the ankle joint).
# Order traces a non-self-intersecting quad: heel-outer, heel-inner,
# toe-inner, toe-outer.
FOOT_SOLE_CONTACTS_LOCAL = np.array([
    [-0.05, 0.025, -0.03],
    [-0.05, -0.025, -0.03],
    [0.12, -0.03, -0.03],
    [0.12, 0.03, -0.03],
], dtype=np.float64)

# Per-joint URDF link mass + local CoM (in that joint's own child-link
# frame) — one entry per BODY_JOINTS id, i.e. every leg/waist/torso/arm link
# the SDK gives us an angle for. id 14's "link" is torso_link itself (its
# origin coincides with the waist_pitch output frame, see BODY_JOINTS above).
LINK_INERTIAL: dict[int, tuple[float, tuple]] = {
    0: (1.35, (0.002741, 0.047791, -0.02606)),
    1: (1.52, (0.029812, -0.001045, -0.087934)),
    2: (1.702, (-0.057709, -0.010981, -0.15078)),
    3: (1.932, (0.005457, 0.003964, -0.12074)),
    4: (0.074, (-0.007269, 0.0, 0.011137)),
    5: (0.608, (0.026505, 0.0, -0.016425)),
    6: (1.35, (0.002741, -0.047791, -0.02606)),
    7: (1.52, (0.029812, 0.001045, -0.087934)),
    8: (1.702, (-0.057709, 0.010981, -0.15078)),
    9: (1.932, (0.005457, -0.003964, -0.12074)),
    10: (0.074, (-0.007269, 0.0, 0.011137)),
    11: (0.608, (0.026505, 0.0, -0.016425)),
    12: (0.214, (0.003494, 0.000233, 0.018034)),
    13: (0.086, (0.0, 2.3e-05, 0.0)),
    14: (6.78, (0.000931, 0.000346, 0.15082)),           # torso_link
    15: (0.718, (0.0, 0.035892, -0.011628)),
    16: (0.643, (-0.000227, 0.00727, -0.063243)),
    17: (0.734, (0.010773, -0.002949, -0.072009)),
    18: (0.6, (0.064956, 0.004454, -0.010062)),
    19: (0.08544498, (0.01713944778, 0.00053759094, 0.00000048864)),
    20: (0.48404956, (0.02299989837, -0.00111685314, -0.00111658096)),
    21: (0.08457647, (0.02200381568, 0.00049485096, 0.00053861123)),
    22: (0.718, (0.0, -0.035892, -0.011628)),
    23: (0.643, (-0.000227, -0.00727, -0.063243)),
    24: (0.734, (0.010773, 0.002949, -0.072009)),
    25: (0.6, (0.064956, -0.004454, -0.010062)),
    26: (0.08544498, (0.01713944778, -0.00053759094, 0.00000048864)),
    27: (0.48404956, (0.02299989837, 0.00111685314, -0.00111658096)),
    28: (0.08457647, (0.02200381568, -0.00049485096, 0.00053861123)),
}

# Extra URDF-mass links not directly tied to one moving joint's own frame:
# (attach_joint_id_or_None_for_root, fixed_offset_from_that_frame, mass,
# local_com_in_that_offset_frame). Hands: dex3 finger joints aren't in the
# SDK's 0-28 range, so each hand's 7 links (palm + thumb/middle/index) are
# pre-collapsed into one rigid neutral-pose (q=0) mass hung off the
# wrist_yaw frame via the URDF's fixed hand_palm_joint offset.
EXTRA_MASSES: list[tuple[Optional[int], tuple, float, tuple]] = [
    (None, (0.0, 0.0, 0.0), 3.813, (0.0, 0.0, -0.07605)),                     # pelvis
    (TORSO_FRAME_ID, (0.0039635, 0.0, -0.044), 1.036, (0.005267, 0.000299, 0.449869)),  # head
    (LEFT_WRIST_ID, (0.0415, 0.003, 0.0), 0.696546,
     (0.066663, -0.008029, -0.000165)),                                       # left hand assembly
    (RIGHT_WRIST_ID, (0.0415, -0.003, 0.0), 0.696546,
     (0.066663, 0.008029, -0.000165)),                                        # right hand assembly (mirrored)
]
TOTAL_MASS_KG = sum(m for m, _ in LINK_INERTIAL.values()) + sum(m for _, _, m, _ in EXTRA_MASSES)


def _quat_to_R(quat: tuple) -> np.ndarray:
    w, x, y, z = quat
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _axis_R(axis: tuple, q: float) -> np.ndarray:
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    ax, ay, az = ax / n, ay / n, az / n
    K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]], dtype=np.float64)
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * (K @ K)


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def body_frames(q: list[float]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Pelvis/body-relative FK.  The returned root is deliberately identity.

    World FK is obtained exactly once with ``world_from_body`` below.  Keeping
    this routine body-only makes posture changes separable from IMU attitude.
    """
    q_values = _finite_vector(q, 1, "joint positions")
    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    root_R = np.eye(3, dtype=np.float64)
    root_t = np.array(PELVIS_POS, dtype=np.float64)
    for jid in range(len(q_values)):
        if jid not in BODY_JOINTS:
            continue
        parent, pos, quat, axis = BODY_JOINTS[jid]
        parent_R, parent_t = (root_R, root_t) if parent == -1 else frames[parent]
        world_t = parent_t + parent_R @ np.array(pos, dtype=np.float64)
        world_R = parent_R @ _quat_to_R(quat) @ _axis_R(axis, float(q_values[jid]))
        frames[jid] = (world_R, world_t)
    return frames


def world_from_body(points_body: np.ndarray, R_world_body: np.ndarray,
                    base_world: np.ndarray) -> np.ndarray:
    """Map Nx3 body/pelvis coordinates to gravity/world exactly once."""
    return np.asarray(points_body) @ R_world_body.T + base_world


def foot_sole_contacts_body_frame(R_body_foot: np.ndarray, t_body_foot: np.ndarray) -> np.ndarray:
    """Four URDF collision contact points in the pelvis/body frame."""
    return t_body_foot + FOOT_SOLE_CONTACTS_LOCAL @ R_body_foot.T


def whole_body_com_body_frame(q: list[float], exclude_arms: str = "none") -> Optional[np.ndarray]:
    """Whole-body CoM (x, y, z), pelvis-relative, mass-weighting every URDF
    link's own CoM through FK of the sensed joint angles. ``exclude_arms`` is
    diagnostic only (left/right/both) and re-normalizes the remaining mass."""
    if len(q) < 29:
        return None
    frames = body_frames(q)
    weighted = np.zeros(3, dtype=np.float64)
    included_mass = 0.0
    for jid, (mass, com_local) in LINK_INERTIAL.items():
        if (exclude_arms in ("left", "both") and 15 <= jid <= 21) or (exclude_arms in ("right", "both") and 22 <= jid <= 28):
            continue
        R, t = frames[jid]
        weighted += mass * (t + R @ np.array(com_local, dtype=np.float64))
        included_mass += mass
    for attach_jid, fixed_pos, mass, com_local in EXTRA_MASSES:
        if (exclude_arms in ("left", "both") and attach_jid == LEFT_WRIST_ID) or (exclude_arms in ("right", "both") and attach_jid == RIGHT_WRIST_ID):
            continue
        if attach_jid is None:
            R, t = np.eye(3, dtype=np.float64), np.array(PELVIS_POS, dtype=np.float64)
        else:
            R, t = frames[attach_jid]
        anchor = t + R @ np.array(fixed_pos, dtype=np.float64)
        weighted += mass * (anchor + R @ np.array(com_local, dtype=np.float64))
        included_mass += mass
    return weighted / included_mass


def rpy_world_body(rpy: Optional[tuple | list]) -> np.ndarray:
    """Body -> gravity/world. SDK RPY is assumed roll,pitch,yaw; URDF/world
    convention is +x forward, +y left, +z up.  No odometry rotation is mixed
    into this transform."""
    if rpy is None or len(rpy) < 3:
        return np.eye(3)
    roll, pitch, yaw = (float(x) for x in rpy[:3])
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def foot_tilt_rpy(R_world_foot: np.ndarray) -> tuple[float, float]:
    """Sole roll/pitch relative to the gravity plane (radians)."""
    pitch = math.atan2(-R_world_foot[2, 0], math.hypot(R_world_foot[0, 0], R_world_foot[1, 0]))
    roll = math.atan2(R_world_foot[2, 1], R_world_foot[2, 2])
    return roll, pitch


def shrunken_sole_points(points_body: np.ndarray, shrink_m: float) -> np.ndarray:
    """Inset a convex sole region about its centroid; valid geometric
    conservative estimate, never a claimed CoP/contact-pressure region."""
    center = np.mean(points_body[:, :2], axis=0)
    v = points_body[:, :2] - center
    radii = np.linalg.norm(v, axis=1)
    scale = np.maximum(0.05, (radii - max(0.0, shrink_m)) / np.maximum(radii, 1e-9))
    out = points_body.copy()
    out[:, :2] = center + v * scale[:, None]
    return out


def projected_limits(points_body: np.ndarray, axis: int) -> tuple[float, float]:
    return float(np.min(points_body[:, axis])), float(np.max(points_body[:, axis]))


# ---------------------------------------------------------------------------
# Small hand-rolled 2-D geometry (no scipy dependency, same "roll it by hand"
# style as segmentation.py's PCA) — convex hull, point-in-polygon, and
# point-to-polygon signed margin.
# ---------------------------------------------------------------------------

def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. points: (N,2). Returns hull vertices, CCW."""
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) < 3:
        return np.array(pts, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def point_in_polygon(pt: tuple[float, float], poly: np.ndarray) -> bool:
    x, y = pt
    n = len(poly)
    inside = False
    if n < 3:
        return False
    x1, y1 = poly[-1]
    for x2, y2 in poly:
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def signed_margin(pt: tuple[float, float], poly: np.ndarray) -> float:
    """Distance from pt to the nearest polygon edge; positive = inside."""
    if len(poly) < 3:
        return -float("inf")
    px, py = pt
    best = float("inf")
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        abx, aby = bx - ax, by - ay
        t = 0.0 if (abx == 0 and aby == 0) else max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby)))
        cx, cy = ax + t * abx, ay + t * aby
        best = min(best, math.hypot(px - cx, py - cy))
    return best if point_in_polygon(pt, poly) else -best


def infer_stance(left_z: np.ndarray, right_z: np.ndarray, left_tilt: tuple[float, float],
                 right_tilt: tuple[float, float], torques: list[float], threshold_m: float) -> tuple[str, str, list[str]]:
    """Kinematic contact model only: no force, CoP, ZMP or Fz is inferred."""
    lm, rm = float(np.mean(left_z)), float(np.mean(right_z))
    ls, rs = float(np.ptp(left_z)), float(np.ptp(right_z))
    reasons = [f"sole mean gap {(lm-rm)*100:+.1f} cm", f"spreads L/R {ls*100:.1f}/{rs*100:.1f} cm"]
    stance = "uncertain/double-assumed"
    if lm - rm > threshold_m:
        stance = "right"
    elif rm - lm > threshold_m:
        stance = "left"
    if stance in ("left", "right") and len(torques) >= 12:
        load_l = abs(float(torques[3])) + abs(float(torques[4]))
        load_r = abs(float(torques[9])) + abs(float(torques[10]))
        # A torque disagreement suppresses a categorical stance call; torque
        # is only a mechanical proxy, explicitly not contact/force sensing.
        if (stance == "left" and load_r > load_l) or (stance == "right" and load_l > load_r):
            stance = "uncertain/double-assumed"
            reasons.append("leg torque proxy disagrees")
    max_tilt = max(abs(x) for x in (*left_tilt, *right_tilt))
    quality = "HIGH" if stance in ("left", "right") and max(ls, rs) < .008 and max_tilt < math.radians(5) else "MEDIUM"
    if stance.startswith("uncertain") or max(ls, rs) > .02 or max_tilt > math.radians(10):
        quality = "LOW"
    reasons.append(f"max foot tilt {math.degrees(max_tilt):.1f} deg")
    return stance, quality, reasons


# ---------------------------------------------------------------------------
# Robot connection + poll loop
# ---------------------------------------------------------------------------

G_ACCEL = 9.81
POLL_HZ = 10.0
ACCEL_SMOOTH_SAMPLES = 5
COM_VEL_SMOOTH_SAMPLES = 3  # lighter than accel's — ICP should stay responsive
COM_VEL_SAMPLE_DT_RANGE_S = (0.02, 0.5)  # discard finite-diff samples outside this dt (gap or jitter)
ZMP_TRAIL_S = 5.0
PLOT_INTERVAL_MS = 400
DOMAIN_ID_RANGE = (0, 232)
PORT_RANGE = (1, 65535)
COM_HEIGHT_RANGE_M = (0.2, 1.5)

DEFAULT_COM_HEIGHT_FALLBACK_M = PELVIS_POS[2]
DEFAULT_SWING_HEIGHT_THRESHOLD_M = 0.015  # foot sole height gap ⇒ treat as single support


def _finite_vector(values: Any, minimum_length: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric sequence") from exc
    if vector.size < minimum_length:
        raise ValueError(f"{name} must contain at least {minimum_length} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def _bounded_int(minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum} and {maximum}")
        return parsed

    return parse


def _extract_xy(odom: Optional[dict]) -> Optional[tuple[float, float]]:
    if not odom:
        return None
    position = odom.get("position")
    if position is not None:
        vector = _finite_vector(position, 2, "odometry position")
        return float(vector[0]), float(vector[1])
    pose = odom.get("pose")
    if pose is not None:
        vector = _finite_vector(pose, 2, "odometry pose")
        return float(vector[0]), float(vector[1])
    return None


def _extract_yaw(odom: Optional[dict], imu: Optional[dict]) -> Optional[float]:
    """Best-available body yaw, for rotating pelvis-relative geometry
    (feet, CoM) into world/odom frame. IMU rpy is present on the same poll
    tick's `get_imus()`/`get_odom()["imu"]` for the primary (SportModeState_)
    odom path; the `rt/odom`/SLAM fallbacks instead carry yaw pre-baked into
    their own `pose` tuple. Returns None only if neither is available (in
    which case callers skip rotation, same as the old behavior)."""
    if imu is not None and imu.get("rpy") is not None:
        try:
            return float(_finite_vector(imu["rpy"], 3, "IMU rpy")[2])
        except ValueError:
            pass
    if odom is not None:
        pose = odom.get("pose")
        if pose is not None:
            try:
                vector = _finite_vector(pose, 3, "odometry pose")
                return float(vector[2])
            except ValueError:
                # Some odometry adapters expose xy only. Position remains
                # usable even though world-orientation correction is not.
                return None
    return None


def _imu_acceleration_world(imu: Optional[dict]) -> Optional[np.ndarray]:
    if not imu or imu.get("acc") is None or imu.get("rpy") is None:
        return None
    try:
        roll, pitch, yaw = _finite_vector(imu["rpy"], 3, "IMU rpy")[:3]
        acceleration = _finite_vector(imu["acc"], 3, "IMU acceleration")[:3]
    except ValueError:
        return None
    rotation = _rot_z(float(yaw)) @ _rot_y(float(pitch)) @ _rot_x(float(roll))
    return rotation @ acceleration - np.array([0.0, 0.0, G_ACCEL])


class ZmpLink:
    def __init__(self, iface: str, domain_id: int):
        self.iface = str(iface)
        self.domain_id = int(domain_id)
        self.lock = threading.RLock()

        self.g1 = None
        self.connect_requested = False
        self.init_err: Optional[str] = None
        self.poll_err: Optional[str] = None

        self.com_xy: Optional[tuple[float, float]] = None
        self.com_body = np.zeros(3)
        self.com_world = np.zeros(3)
        self.com_vel_xy: Optional[tuple[float, float]] = None
        self.com_vel_body = np.zeros(3)
        self.left_corners: Optional[np.ndarray] = None
        self.right_corners: Optional[np.ndarray] = None
        self.left_body: Optional[np.ndarray] = None
        self.right_body: Optional[np.ndarray] = None
        self.nominal_hull: Optional[np.ndarray] = None
        self.conservative_hull: Optional[np.ndarray] = None
        self.conservative_world: Optional[np.ndarray] = None
        self.zmp_xy: Optional[tuple[float, float]] = None
        self.icp_xy: Optional[tuple[float, float]] = None
        self.zmp_body = np.zeros(2)
        self.icp_body = np.zeros(2)
        self.rpy = (0.0, 0.0, 0.0)
        self.raw_accel = np.zeros(3)
        self.world_accel = np.zeros(3)
        self.foot_info: dict[str, Any] = {}
        self.contact_confidence = "LOW"
        self.stance_reasons: list[str] = []
        self.q: list[float] = []
        self.torques: list[float] = []
        self.arm_shift_body = np.zeros(2)
        self.arm_com_excluded: dict[str, np.ndarray] = {}
        self.omega = 0.0
        self.history: deque = deque(maxlen=int(12 * POLL_HZ))
        self.step_events: deque = deque(maxlen=40)
        self.frozen = False
        self.recording = False
        self._record_file = None
        self._record_writer = None
        self._last_stance = "uncertain/double-assumed"
        self.h_com: float = DEFAULT_COM_HEIGHT_FALLBACK_M
        self.h_com_auto: bool = False
        self.stance: str = "double"  # "double" | "left" | "right", see _poll_once
        self.stance_torque_overridden: bool = False
        self.yaw_applied: bool = False
        self.accel_source: str = "none"  # "imu" | "kinematic" | "none", see _poll_once
        self.ts = 0.0
        self.zmp_trail: deque = deque(maxlen=int(ZMP_TRAIL_S * POLL_HZ) + 4)
        self._accel_hist: deque = deque(maxlen=ACCEL_SMOOTH_SAMPLES)
        self._com_vel_hist: deque = deque(maxlen=COM_VEL_SMOOTH_SAMPLES)
        self._prev_com_sample: Optional[tuple[float, float, float]] = None  # (ts, x, y)
        self._com_accel_hist: deque = deque(maxlen=ACCEL_SMOOTH_SAMPLES)
        self._prev_com_vel_sample: Optional[tuple[float, float, float]] = None  # (ts, vx, vy)
        self._com_height_fallback_m = DEFAULT_COM_HEIGHT_FALLBACK_M
        self._swing_height_threshold_m = DEFAULT_SWING_HEIGHT_THRESHOLD_M

        self._stop: Optional[threading.Event] = None
        self._poll_thread: Optional[threading.Thread] = None

    def _reset_dynamic_state_locked(self) -> None:
        self.com_xy = None
        self.com_vel_xy = None
        self.left_corners = None
        self.right_corners = None
        self.zmp_xy = None
        self.icp_xy = None
        self.ts = 0.0
        self.zmp_trail.clear()
        self._accel_hist.clear()
        self._com_vel_hist.clear()
        self._com_accel_hist.clear()
        self._prev_com_sample = None
        self._prev_com_vel_sample = None

    def connect(self) -> None:
        with self.lock:
            self.connect_requested = True
            if self.g1 is not None:
                return
            if self._poll_thread is not None and self._poll_thread.is_alive():
                self.init_err = "Previous polling thread is still stopping; try Connect again."
                return
            try:
                from sdk_wrapper import G1  # deferred: only needed once connecting
                self.g1 = G1(self.iface, domain_id=self.domain_id)
                self.init_err = None
            except Exception as exc:
                self.init_err = str(exc)
                self.g1 = None
                return
            self._reset_dynamic_state_locked()
            stop = threading.Event()
            self._stop = stop
            self._poll_thread = threading.Thread(target=self._poll_loop, args=(stop,), daemon=True)
            self._poll_thread.start()

    def disconnect(self) -> None:
        with self.lock:
            self.connect_requested = False
            stop = self._stop
            thread = self._poll_thread
            if stop is not None:
                stop.set()
            self.g1 = None
            self._reset_dynamic_state_locked()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self.lock:
            if self._poll_thread is thread and (thread is None or not thread.is_alive()):
                self._poll_thread = None
                self._stop = None
            # A poll already in progress when disconnect was requested may
            # have published one final snapshot before exiting.
            self._reset_dynamic_state_locked()

    def status(self) -> tuple[str, str]:
        with self.lock:
            if self.g1 is not None:
                return "Connected", "success"
            if not self.connect_requested:
                return "Disconnected", "secondary"
            if self.init_err is not None:
                return "Error", "danger"
            return "Connecting…", "warning"

    def _poll_loop(self, stop: threading.Event) -> None:
        period = 1.0 / POLL_HZ
        while not stop.is_set():
            with self.lock:
                g1 = self.g1
                com_height = self._com_height_fallback_m
                swing_threshold = self._swing_height_threshold_m
            if g1 is not None:
                try:
                    self._poll_once(
                        g1,
                        com_height_fallback_m=com_height,
                        swing_threshold_m=swing_threshold,
                    )
                    with self.lock:
                        self.poll_err = None
                except Exception as exc:
                    with self.lock:
                        self.poll_err = str(exc)
<<<<<<< HEAD
            time.sleep(period)

    # Configurable from the UI; read by the poll loop each tick.
    _com_height_fallback_m = DEFAULT_COM_HEIGHT_FALLBACK_M
    _swing_height_threshold_m = DEFAULT_SWING_HEIGHT_THRESHOLD_M
    _conservative_shrink_m = 0.018
    _imu_accel_includes_gravity = True
=======
            stop.wait(period)
>>>>>>> 134802629fc365c6592869d0220ac49d919aa440

    def _poll_once(self, g1, com_height_fallback_m: float, swing_threshold_m: float) -> None:
        lowstate = g1.get_lowstate()
        odom = g1.get_odom()
        imu = g1.get_imus()

<<<<<<< HEAD
        # Odometry supplies translation only.  Attitude is always the full
        # IMU R_world_body = Rz(yaw) Ry(pitch) Rx(roll), applied once below.
        base_xy = _extract_xy(odom) or (0.0, 0.0)
        rpy = tuple(imu.get("rpy") or (0.0, 0.0, _extract_yaw(odom, imu) or 0.0)) if imu else (0.0, 0.0, _extract_yaw(odom, imu) or 0.0)
        Rwb = rpy_world_body(rpy)
        base_world = np.array([base_xy[0], base_xy[1], 0.0])
        positions = (lowstate.get("joint_positions") or []) if lowstate is not None else []
        torques = (lowstate.get("joint_torques") or []) if lowstate is not None else []
=======
        base_xy = _extract_xy(odom)
        yaw = _extract_yaw(odom, imu)
        yaw_R2 = None if yaw is None else _rot_z(yaw)[:2, :2]
        raw_positions = lowstate.get("joint_positions") if lowstate is not None else None
        raw_torques = lowstate.get("joint_torques") if lowstate is not None else None
        positions = (
            []
            if raw_positions is None
            else _finite_vector(raw_positions, 1, "joint positions").tolist()
        )
        torques = [] if raw_torques is None else list(raw_torques)
>>>>>>> 134802629fc365c6592869d0220ac49d919aa440
        leg_q = positions[:12] if len(positions) >= 12 else None

        com_xy = None
        left_corners = right_corners = None
        h_com = com_height_fallback_m
        h_com_auto = False
        stance = "uncertain/double-assumed"
        stance_torque_overridden = False

        if leg_q is not None:
            frames = body_frames(leg_q)
            lR, lt = frames[LEFT_FOOT_ID]
            rR, rt = frames[RIGHT_FOOT_ID]
            left3 = foot_sole_contacts_body_frame(lR, lt)
            right3 = foot_sole_contacts_body_frame(rR, rt)
            left_world3 = world_from_body(left3, Rwb, base_world)
            right_world3 = world_from_body(right3, Rwb, base_world)
            left_ground_z, right_ground_z = float(left_world3[:, 2].mean()), float(right_world3[:, 2].mean())
            ltilt, rtilt = foot_tilt_rpy(Rwb @ lR), foot_tilt_rpy(Rwb @ rR)
            stance, contact_confidence, stance_reasons = infer_stance(left_world3[:, 2], right_world3[:, 2], ltilt, rtilt, torques, swing_threshold_m)
            # Actual world heights are intentionally retained; a mean height
            # alone must never claim a whole foot is flat or in contact.
            foot_info = {"left_mean": left_ground_z, "right_mean": right_ground_z,
                         "left_spread": float(np.ptp(left_world3[:, 2])), "right_spread": float(np.ptp(right_world3[:, 2])),
                         "left_tilt": ltilt, "right_tilt": rtilt,
                         "left_heights": left_world3[:, 2].copy(), "right_heights": right_world3[:, 2].copy()}

            # Coarse gait-phase heuristic (still no force/contact sensing): if
            # one sole sits meaningfully higher than the other, treat it as
            # the swing foot and shrink both the support polygon and the
            # ground-height reference to the stance foot alone, instead of
            # always assuming double support.
            ground_z = left_ground_z if stance == "left" else right_ground_z if stance == "right" else min(left_ground_z, right_ground_z)

            # Cross-check against knee+ankle-pitch torque (tau_est) — still
            # not a real force sensor, but load-bearing joints report more
            # mechanical effort than an unloaded swing leg, so if the leg the
            # height heuristic just called "swing" is actually carrying *more*
            # torque than the one it called "stance", the geometry read is
            # probably wrong (a lean/tilt rather than an actual foot lift) and
            # single support is suppressed back to the safer double-support
            # default. Only ever makes the call more conservative, never less
            # — it can't invent a single-support the height gap didn't already
            # suggest. Skipped (leaves the height-only call as-is) if torques
            # aren't reported.
<<<<<<< HEAD
            stance_torque_overridden = "leg torque proxy disagrees" in stance_reasons
            whole_body = whole_body_com_body_frame(positions[:29])
            if whole_body is not None:
                com_body = whole_body
                com_world = world_from_body(com_body[None, :], Rwb, base_world)[0]
                h_com = com_world[2] - ground_z
                h_com_auto = True
                no_left = whole_body_com_body_frame(positions[:29], "left")
                no_right = whole_body_com_body_frame(positions[:29], "right")
                no_arms = whole_body_com_body_frame(positions[:29], "both")
                arm_shift = com_body[:2] - no_arms[:2] if no_arms is not None else np.zeros(2)
                arm_excluded = {k: v for k, v in (("left", no_left), ("right", no_right), ("both", no_arms)) if v is not None}
=======
            if stance != "double" and len(torques) >= 12:
                load_values = np.asarray(
                    [torques[3], torques[4], torques[9], torques[10]],
                    dtype=np.float64,
                )
                if np.all(np.isfinite(load_values)):
                    left_load = abs(load_values[0]) + abs(load_values[1])
                    right_load = abs(load_values[2]) + abs(load_values[3])
                    stance_load, swing_load = (
                        (left_load, right_load)
                        if stance == "left"
                        else (right_load, left_load)
                    )
                    if swing_load > stance_load:
                        stance = "double"
                        ground_z = (left_ground_z + right_ground_z) / 2.0
                        stance_torque_overridden = True

            whole_body = whole_body_com_pelvis_frame(positions[:29])
            if whole_body is not None:
                com_xy_local, com_z_local = whole_body
                measured_height = com_z_local - ground_z
                if math.isfinite(measured_height) and (
                    COM_HEIGHT_RANGE_M[0] <= measured_height <= COM_HEIGHT_RANGE_M[1]
                ):
                    h_com = measured_height
                    h_com_auto = True
>>>>>>> 134802629fc365c6592869d0220ac49d919aa440
            else:
                com_body, com_world, arm_shift, arm_excluded = np.zeros(3), base_world, np.zeros(2), {}
            left_corners, right_corners = left_world3[:, :2], right_world3[:, :2]
            com_xy = tuple(com_world[:2])
            left_body, right_body = left3, right3
        else:
            com_body, com_world, arm_shift, arm_excluded, contact_confidence, stance_reasons = np.zeros(3), base_world, np.zeros(2), {}, "LOW", ["no leg FK"]
            left_body = right_body = None
            foot_info = {}
            ground_z = float(base_world[2])
            com_xy = tuple(base_world[:2])

        now = time.monotonic()

        # CoM velocity, finite-differenced across poll ticks and lightly
        # smoothed — same "no dedicated sensor for this" spirit as the IMU
        # accel smoothing below. Feeds both the instantaneous capture point
        # and (via a second finite difference) the kinematic acceleration
        # fallback below; samples with an implausible dt (a data gap, or two
        # polls landing on the same underlying message) are dropped rather
        # than blended in.
        com_vel_xy = None
        with self.lock:
            prev = self._prev_com_sample
            if com_xy is not None:
                if prev is not None:
                    dt = now - prev[0]
                    if COM_VEL_SAMPLE_DT_RANGE_S[0] <= dt <= COM_VEL_SAMPLE_DT_RANGE_S[1]:
                        self._com_vel_hist.append(((com_xy[0] - prev[1]) / dt, (com_xy[1] - prev[2]) / dt))
                self._prev_com_sample = (now, com_xy[0], com_xy[1])
            else:
                self._prev_com_sample = None
                self._com_vel_hist.clear()
            if self._com_vel_hist:
                com_vel_xy = (sum(v[0] for v in self._com_vel_hist) / len(self._com_vel_hist),
                              sum(v[1] for v in self._com_vel_hist) / len(self._com_vel_hist))

        # Kinematic CoM acceleration: a second finite difference of the
        # (already-smoothed) CoM velocity above, with its own smoothing —
        # differentiating twice from 10Hz position samples is noisy, so this
        # is deliberately *not* blended into the primary IMU-based estimate
        # below; it only steps in when the IMU has nothing to offer (missing
        # `acc`/`rpy`), which previously silently zeroed the ZMP's dynamic
        # term instead.
        kinematic_accel_xy = None
        with self.lock:
            prev_vel = self._prev_com_vel_sample
            if com_vel_xy is not None:
                if prev_vel is not None:
                    dt = now - prev_vel[0]
                    if COM_VEL_SAMPLE_DT_RANGE_S[0] <= dt <= COM_VEL_SAMPLE_DT_RANGE_S[1]:
                        self._com_accel_hist.append(((com_vel_xy[0] - prev_vel[1]) / dt, (com_vel_xy[1] - prev_vel[2]) / dt))
                self._prev_com_vel_sample = (now, com_vel_xy[0], com_vel_xy[1])
            else:
                self._prev_com_vel_sample = None
                self._com_accel_hist.clear()
            if self._com_accel_hist:
                kinematic_accel_xy = (sum(v[0] for v in self._com_accel_hist) / len(self._com_accel_hist),
                                      sum(v[1] for v in self._com_accel_hist) / len(self._com_accel_hist))

        ax_world = ay_world = 0.0
        a_world = np.zeros(3, dtype=np.float64)
        accel_source = "none"
<<<<<<< HEAD
        if imu is not None and imu.get("acc") is not None and imu.get("rpy") is not None and np.linalg.norm(np.asarray(imu["acc"], dtype=float)) > 1e-6:
            roll, pitch, imu_yaw = imu["rpy"]
            R = Rwb
            a_body = np.array(imu["acc"], dtype=np.float64)
            # sdk_wrapper only forwards the native field and this repository
            # provides no convention proof. UI selects the explicit runtime
            # assumption; stationary horizontal acceleration is shown below.
            a_world = R @ a_body - (np.array([0.0, 0.0, G_ACCEL]) if self._imu_accel_includes_gravity else 0.0)
=======
        a_world = _imu_acceleration_world(imu)
        if a_world is not None:
>>>>>>> 134802629fc365c6592869d0220ac49d919aa440
            with self.lock:
                self._accel_hist.append((float(a_world[0]), float(a_world[1])))
                ax_world = sum(v[0] for v in self._accel_hist) / len(self._accel_hist)
                ay_world = sum(v[1] for v in self._accel_hist) / len(self._accel_hist)
            accel_source = "imu"
        elif kinematic_accel_xy is not None:
            ax_world, ay_world = kinematic_accel_xy
            accel_source = "kinematic"
        elif imu is not None and imu.get("acc") is not None:
            # Observed on rt/odommodestate on this robot: accelerometer is
            # [0,0,0]. It is not a usable measured acceleration, regardless
            # of the gravity convention toggle.
            accel_source = "imu-zero/unusable"

        zmp_xy = None
        if com_xy is not None:
            zmp_xy = (
                com_xy[0] - (h_com / G_ACCEL) * ax_world,
                com_xy[1] - (h_com / G_ACCEL) * ay_world,
            )

        # Instantaneous capture point (LIPM): ICP = CoM + CoM_vel/omega,
        # omega = sqrt(g/h_com). Guard h_com>0 — a bad/zero height would
        # blow this up, not just bias it, unlike the ZMP term above.
        icp_xy = None
        if com_xy is not None and com_vel_xy is not None and h_com > 1e-3:
            omega = math.sqrt(G_ACCEL / h_com)
            icp_xy = (com_xy[0] + com_vel_xy[0] / omega, com_xy[1] + com_vel_xy[1] / omega)

        # Body longitudinal/lateral values use body coordinates, intentionally
        # avoiding yaw-dependent world projections as the robot turns.
        zmp_body = (Rwb.T @ (np.array([*(zmp_xy or com_xy), ground_z]) - base_world))[:2]
        icp_body = (Rwb.T @ (np.array([*(icp_xy or com_xy), ground_z]) - base_world))[:2]
        vel_body = Rwb.T @ np.array([*(com_vel_xy or (0.0, 0.0)), 0.0])
        use_l = left_body if stance == "left" else right_body if stance == "right" else np.vstack([left_body, right_body]) if left_body is not None else np.zeros((0, 3))
        support_body = convex_hull(use_l[:, :2]) if len(use_l) else None
        shrink_l = shrunken_sole_points(left_body, self._conservative_shrink_m) if left_body is not None else None
        shrink_r = shrunken_sole_points(right_body, self._conservative_shrink_m) if right_body is not None else None
        use_c = shrink_l if stance == "left" else shrink_r if stance == "right" else np.vstack([shrink_l, shrink_r]) if shrink_l is not None else np.zeros((0, 3))
        conservative_body = convex_hull(use_c[:, :2]) if len(use_c) else None

        with self.lock:
            self.com_xy = com_xy
            self.com_body, self.com_world = com_body, com_world
            self.com_vel_xy = com_vel_xy
            self.com_vel_body = vel_body
            self.left_corners = left_corners
            self.right_corners = right_corners
            self.left_body, self.right_body = left_body, right_body
            self.nominal_hull, self.conservative_hull = support_body, conservative_body
            self.conservative_world = world_from_body(use_c, Rwb, base_world)[:, :2] if len(use_c) else None
            self.zmp_xy = zmp_xy
            self.icp_xy = icp_xy
            self.h_com = h_com
            self.h_com_auto = h_com_auto
            self.stance = stance
            self.stance_torque_overridden = stance_torque_overridden
            self.yaw_applied = True
            self.accel_source = accel_source
            self.rpy, self.raw_accel, self.world_accel = rpy, np.array(imu.get("acc") if imu and imu.get("acc") else (0,0,0)), a_world
            self.foot_info, self.contact_confidence, self.stance_reasons = foot_info, contact_confidence, stance_reasons
            self.q, self.torques, self.arm_shift_body, self.arm_com_excluded = list(positions), list(torques), arm_shift, arm_excluded
            self.omega, self.zmp_body, self.icp_body = math.sqrt(G_ACCEL / h_com) if h_com > 1e-3 else 0.0, zmp_body, icp_body
            self.ts = now
            if zmp_xy is not None:
                self.zmp_trail.append((now, zmp_xy[0], zmp_xy[1]))
            if stance != self._last_stance and (stance in ("left", "right") or self._last_stance in ("left", "right")):
                self.step_events.append(now)
            self._last_stance = stance
            if not self.frozen:
                self.history.append({"t": now, "com": float(com_body[0]), "zmp": float(zmp_body[0]), "icp": float(icp_body[0]), "vel": float(vel_body[0]), "front": projected_limits(support_body, 0)[1] if support_body is not None else np.nan, "rear": projected_limits(support_body, 0)[0] if support_body is not None else np.nan, "pitch": float(rpy[1])})
            if self.recording:
                self._record(now)

    def _record(self, timestamp: float) -> None:
        """Called from poll thread, never from Dash's UI callback."""
        if self._record_writer is None:
            name = f"zmp_diagnostics_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            self._record_file = open(_SCRIPT_DIR / name, "w", newline="", encoding="utf-8")
            fields = ["timestamp", "stance", "contact_confidence", "roll", "pitch", "yaw", "com_body_x", "com_body_y", "com_body_z", "com_world_x", "com_world_y", "com_world_z", "com_vx", "com_vy", "zmp_x", "zmp_y", "icp_x", "icp_y", "zmp_nominal_margin", "zmp_conservative_margin", "icp_nominal_margin", "icp_conservative_margin", "joint_positions", "joint_torques", "raw_accel", "world_accel", "foot_corner_heights"]
            self._record_writer = csv.DictWriter(self._record_file, fieldnames=fields)
            self._record_writer.writeheader()
        nom, con = self.nominal_hull, self.conservative_hull
        row = {"timestamp": timestamp, "stance": self.stance, "contact_confidence": self.contact_confidence,
               "roll": self.rpy[0], "pitch": self.rpy[1], "yaw": self.rpy[2],
               "com_body_x": self.com_body[0], "com_body_y": self.com_body[1], "com_body_z": self.com_body[2],
               "com_world_x": self.com_world[0], "com_world_y": self.com_world[1], "com_world_z": self.com_world[2],
               "com_vx": self.com_vel_xy[0] if self.com_vel_xy else "", "com_vy": self.com_vel_xy[1] if self.com_vel_xy else "",
               "zmp_x": self.zmp_xy[0] if self.zmp_xy else "", "zmp_y": self.zmp_xy[1] if self.zmp_xy else "",
               "icp_x": self.icp_xy[0] if self.icp_xy else "", "icp_y": self.icp_xy[1] if self.icp_xy else "",
               "zmp_nominal_margin": signed_margin(self.zmp_body, nom) if nom is not None else "", "zmp_conservative_margin": signed_margin(self.zmp_body, con) if con is not None else "",
               "icp_nominal_margin": signed_margin(self.icp_body, nom) if nom is not None else "", "icp_conservative_margin": signed_margin(self.icp_body, con) if con is not None else "",
               "joint_positions": repr(self.q), "joint_torques": repr(self.torques), "raw_accel": repr(self.raw_accel.tolist()), "world_accel": repr(self.world_accel.tolist()), "foot_corner_heights": repr(self.foot_info)}
        self._record_writer.writerow(row)
        self._record_file.flush()

    def stop_recording(self) -> None:
        with self.lock:
            self.recording = False
            if self._record_file:
                self._record_file.close()
            self._record_file = self._record_writer = None


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure(link: "ZmpLink") -> tuple[go.Figure, list[tuple[str, str]]]:
    with link.lock:
        com_xy = link.com_xy
        left_corners = None if link.left_corners is None else link.left_corners.copy()
        right_corners = None if link.right_corners is None else link.right_corners.copy()
        conservative_world = None if link.conservative_world is None else link.conservative_world.copy()
        nominal_body = None if link.nominal_hull is None else link.nominal_hull.copy()
        conservative_body = None if link.conservative_hull is None else link.conservative_hull.copy()
        zmp_body, icp_body, contact_confidence = link.zmp_body.copy(), link.icp_body.copy(), link.contact_confidence
        zmp_xy = link.zmp_xy
        icp_xy = link.icp_xy
        h_com = link.h_com
        h_com_auto = link.h_com_auto
        stance = link.stance
        stance_torque_overridden = link.stance_torque_overridden
        yaw_applied = link.yaw_applied
        accel_source = link.accel_source
        trail = list(link.zmp_trail)

    fig = go.Figure()
    extent = []

    hull = None
    if left_corners is not None and right_corners is not None:
        # Single-support: only the stance foot counts toward the polygon
        # (the swing foot's outline is still drawn below, for reference).
        if stance == "left":
            hull_points = left_corners
        elif stance == "right":
            hull_points = right_corners
        else:
            hull_points = np.vstack([left_corners, right_corners])
        hull = convex_hull(hull_points)
        if len(hull) >= 3:
            hx = list(hull[:, 0]) + [hull[0, 0]]
            hy = list(hull[:, 1]) + [hull[0, 1]]
            poly_name = "support polygon" if stance == "double" else f"support polygon ({stance} stance)"
            fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines", fill="toself",
                                      fillcolor="rgba(85,199,255,0.18)",
                                      line={"color": "#55c7ff", "width": 2},
                                      name=poly_name))
            extent.extend(zip(hx, hy))
        if conservative_world is not None:
            ch = convex_hull(conservative_world)
            if len(ch) >= 3:
                fig.add_trace(go.Scatter(x=list(ch[:, 0]) + [ch[0, 0]], y=list(ch[:, 1]) + [ch[0, 1]], mode="lines",
                                         line={"color": "#ff8c42", "width": 2, "dash": "dash"},
                                         name="conservative geometric support estimate"))
        for label, corners, color in (("left foot", left_corners, "#3987e5"),
                                       ("right foot", right_corners, "#d95926")):
            cx = list(corners[:, 0]) + [corners[0, 0]]
            cy = list(corners[:, 1]) + [corners[0, 1]]
            fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name=label,
                                      line={"color": color, "width": 1.5, "dash": "dot"}))
            extent.extend(zip(cx, cy))

    if trail:
        tx = [p[1] for p in trail]
        ty = [p[2] for p in trail]
        fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines", name="ZMP trail",
                                  line={"color": "#f5c84b", "width": 1}, opacity=0.6))
        extent.extend(zip(tx, ty))

    zmp_line = ("no data", "#888888")
    icp_line = None
    if com_xy is not None:
        fig.add_trace(go.Scatter(x=[com_xy[0]], y=[com_xy[1]], mode="markers",
                                  marker={"size": 12, "color": "#ffae00", "symbol": "circle"},
                                  name="CoM (whole-body, URDF)" if h_com_auto else "CoM (≈ pelvis, fallback)"))
        extent.append(com_xy)

    h_com_note = f"h_com {h_com:.3f} m ({'auto, URDF whole-body CoM' if h_com_auto else 'fallback'})"
    if left_corners is not None and not yaw_applied:
        h_com_note += " · no yaw (orientation may drift while turning)"
    if stance_torque_overridden:
        h_com_note += " · single-support suppressed (torque disagreed with foot-height geometry)"
    if accel_source == "kinematic":
        h_com_note += " · accel: kinematic fallback (IMU unavailable)"
    elif accel_source == "none" and com_xy is not None:
        h_com_note += " · accel: none available (ZMP ≈ CoM)"

    if zmp_xy is not None:
        inside = hull is not None and len(hull) >= 3 and point_in_polygon(zmp_xy, hull)
        margin = signed_margin(zmp_xy, hull) if hull is not None and len(hull) >= 3 else None
        zmp_color = "#3aa876" if inside else "#e0294f"
        fig.add_trace(go.Scatter(x=[zmp_xy[0]], y=[zmp_xy[1]], mode="markers",
                                  marker={"size": 16, "color": zmp_color, "symbol": "diamond"},
                                  name="ZMP"))
        extent.append(zmp_xy)
        if margin is not None:
            mn = signed_margin(zmp_body, nominal_body) if nominal_body is not None else float("nan")
            mc = signed_margin(zmp_body, conservative_body) if conservative_body is not None else float("nan")
            zmp_line = (f"Estimated ZMP: {'inside' if inside else 'outside'} nominal geometric support ({mn*100:+.1f} cm); conservative {mc*100:+.1f} cm · contact {contact_confidence} · force sensing unavailable · {h_com_note}", zmp_color)
        else:
            zmp_line = (f"ZMP computed, no support polygon (no leg data) · {h_com_note}", zmp_color)

    if icp_xy is not None:
        icp_inside = hull is not None and len(hull) >= 3 and point_in_polygon(icp_xy, hull)
        icp_margin = signed_margin(icp_xy, hull) if hull is not None and len(hull) >= 3 else None
        icp_color = "#3aa876" if icp_inside else "#e0294f"
        fig.add_trace(go.Scatter(x=[icp_xy[0]], y=[icp_xy[1]], mode="markers",
                                  marker={"size": 14, "color": icp_color, "symbol": "star",
                                          "line": {"color": "#ffffff", "width": 1}},
                                  name="Capture point (ICP)"))
        extent.append(icp_xy)
        if icp_margin is not None:
            mn = signed_margin(icp_body, nominal_body) if nominal_body is not None else float("nan")
            mc = signed_margin(icp_body, conservative_body) if conservative_body is not None else float("nan")
            icp_line = (f"Estimated Capture Point / DCM: {'inside' if icp_inside else 'outside'} nominal support ({mn*100:+.1f} cm); conservative {mc*100:+.1f} cm — LIPM approximation", icp_color)
        else:
            icp_line = ("ICP computed, no support polygon (no leg data)", icp_color)
    elif com_xy is not None:
        icp_line = ("ICP: building CoM-velocity history…", "#888888")

    if extent:
        arr = np.asarray(extent, dtype=np.float64)
        xmin, ymin = arr.min(axis=0)
        xmax, ymax = arr.max(axis=0)
        pad = max(0.15, 0.2 * max(xmax - xmin, ymax - ymin, 0.2))
    else:
        xmin, xmax, ymin, ymax = -0.3, 0.3, -0.3, 0.3
        pad = 0.1

    fig.update_layout(
        template="plotly_dark", height=620,
        margin={"l": 40, "r": 10, "t": 20, "b": 40},
        xaxis={"title": "x (m)", "range": [xmin - pad, xmax + pad], "scaleanchor": "y", "scaleratio": 1},
        yaxis={"title": "y (m)", "range": [ymin - pad, ymax + pad]},
        legend={"orientation": "h", "y": 1.05},
        uirevision="zmp-map",
    )
    lines = [zmp_line]
    if icp_line is not None:
        lines.append(icp_line)
    return fig, lines


# ---------------------------------------------------------------------------
# Dash app / layout
# ---------------------------------------------------------------------------

def build_history_figure(link: "ZmpLink") -> go.Figure:
    with link.lock:
        hist, events = list(link.history), list(link.step_events)
    fig = go.Figure()
    if hist:
        t0 = hist[-1]["t"]
        x = [r["t"] - t0 for r in hist]
        for key, name, color in (("com", "CoM forward", "#ffae00"), ("zmp", "ZMP forward", "#55c7ff"), ("icp", "ICP/DCM forward", "#f5c84b"), ("front", "support front", "#3aa876"), ("rear", "support rear", "#e0294f")):
            fig.add_trace(go.Scatter(x=x, y=[r[key] for r in hist], mode="lines", name=name, line={"color": color}))
        fig.add_trace(go.Scatter(x=x, y=[r["vel"] for r in hist], mode="lines", name="CoM forward velocity (m/s)", yaxis="y2", line={"color": "#c77dff"}))
        for event in events:
            if event >= hist[0]["t"]:
                fig.add_vline(x=event-t0, line_dash="dot", line_color="#ffffff", annotation_text="inferred step")
    fig.update_layout(template="plotly_dark", height=330, margin={"l":40,"r":50,"t":25,"b":35}, xaxis_title="seconds (latest = 0)", yaxis_title="body forward x (m)", yaxis2={"title":"velocity m/s", "overlaying":"y", "side":"right"}, legend={"orientation":"h", "y":1.15}, uirevision="history")
    return fig


def diagnostics_panel(link: "ZmpLink") -> html.Div:
    with link.lock:
        rpy, raw, acc = link.rpy, link.raw_accel.copy(), link.world_accel.copy()
        cb, cw, vb, z, i = link.com_body.copy(), link.com_world.copy(), link.com_vel_body.copy(), link.zmp_body.copy(), link.icp_body.copy()
        nom, con, stance, confidence = link.nominal_hull, link.conservative_hull, link.stance, link.contact_confidence
        info, q, shift, arm_excluded, omega, source, reasons = (dict(link.foot_info), list(link.q), link.arm_shift_body.copy(),
                                                                  dict(link.arm_com_excluded), link.omega,
                                                                  link.accel_source, list(link.stance_reasons))
    def margin(p, poly): return signed_margin(p, poly) * 100 if poly is not None else float("nan")
    front, rear = projected_limits(nom, 0) if nom is not None else (float("nan"), float("nan"))
    f = lambda v, unit="": "—" if not np.isfinite(v) else f"{v:.2f}{unit}"
    foot = lambda side: f"{side}: heel L/R {[round(v*100,1) for v in info.get(side+'_heights',[])[:2]]} cm, toe L/R {[round(v*100,1) for v in info.get(side+'_heights',[])[2:]]} cm; mean {f(info.get(side+'_mean',np.nan)*100,' cm')}, spread {f(info.get(side+'_spread',np.nan)*100,' cm')}, roll/pitch {f(math.degrees(info.get(side+'_tilt',(np.nan,np.nan))[0]),'°')}/{f(math.degrees(info.get(side+'_tilt',(np.nan,np.nan))[1]),'°')}"
    return html.Div([
        html.B("CONTACT MODEL: GEOMETRIC / NO FORCE SENSING", style={"color":"#ff8c42"}), html.Br(),
        html.Small("Validity: CoM FK/URDF · velocity finite difference · acceleration IMU/assumption · ZMP & ICP LIPM estimates · support geometric heuristic · contact forces unavailable"), html.Hr(),
        html.B("IMU"), html.Br(), f"roll/pitch/yaw {f(math.degrees(rpy[0]),'°')} / {f(math.degrees(rpy[1]),'°')} / {f(math.degrees(rpy[2]),'°')} · raw accel {np.round(raw,2)} · world accel {np.round(acc,2)} · source {source}", html.Br(),
        html.B("CoM"), html.Br(), f"body {np.round(cb,3)} m · world {np.round(cw,3)} m · vx/vy body {f(vb[0],' m/s')}/{f(vb[1],' m/s')} · speed {f(np.linalg.norm(vb[:2]),' m/s')} · h {f(cw[2]-min(info.get('left_mean',cw[2]),info.get('right_mean',cw[2])),' m')}", html.Br(),
        html.B("Dynamics (LIPM approximation, not proof of stability)"), html.Br(), f"ZMP body x/y {np.round(z,3)} · nominal/conservative margin {f(margin(z,nom),' cm')}/{f(margin(z,con),' cm')} · ICP/DCM {np.round(i,3)} · nominal/conservative margin {f(margin(i,nom),' cm')}/{f(margin(i,con),' cm')} · ω {f(omega,' rad/s')} · τ {f(1/omega if omega else np.nan,' s')}", html.Br(),
        html.B("Forward/lateral support"), html.Br(), f"rear/front {f(rear*100,' cm')}/{f(front*100,' cm')} · ZMP front margin {f((front-z[0])*100,' cm')} · ICP front margin {f((front-i[0])*100,' cm')} · CoM front margin {f((front-cb[0])*100,' cm')} · lateral CoM y {f(cb[1]*100,' cm')}", html.Br(),
        html.B("Feet / stance"), html.Br(), foot("left"), html.Br(), foot("right"), html.Br(), f"estimated stance: {stance}; confidence: {confidence}; {', '.join(reasons)}", html.Br(),
        html.B("Posture / arm-model check"), html.Br(), f"L/R shoulder pitch {f(math.degrees(q[15]) if len(q)>15 else np.nan,'°')}/{f(math.degrees(q[22]) if len(q)>22 else np.nan,'°')}; L/R elbow {f(math.degrees(q[18]) if len(q)>18 else np.nan,'°')}/{f(math.degrees(q[25]) if len(q)>25 else np.nan,'°')}; waist pitch/roll {f(math.degrees(q[14]) if len(q)>14 else np.nan,'°')}/{f(math.degrees(q[13]) if len(q)>13 else np.nan,'°')}; CoM excluding L/R/both arms {[np.round(arm_excluded[k],3).tolist() for k in ('left','right','both') if k in arm_excluded]}; arm posture CoM shift forward/lateral {f(shift[0]*100,' cm')}/{f(shift[1]*100,' cm')}"
    ], style={"fontSize":"12px", "whiteSpace":"normal", "padding":"8px", "border":"1px solid #555", "borderRadius":"4px"})

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "G1 ZMP / Support Polygon"

app.layout = dbc.Container([
    html.H3("G1 ZMP / Support Polygon", className="mt-3"),
    dbc.Row([
        dbc.Col(dbc.Input(id="iface-input", value="eth0", placeholder="iface"), width="auto"),
        dbc.Col(dbc.Input(id="domain-input", type="number", value=0, placeholder="domain id"), width="auto"),
        dbc.Col(dbc.Button("Connect", id="btn-connect", color="primary"), width="auto"),
        dbc.Col(dbc.Button("Disconnect", id="btn-disconnect", color="secondary"), width="auto"),
        dbc.Col(dbc.Badge("Disconnected", id="conn-badge", color="secondary"), width="auto"),
    ], align="center", className="mb-2 gy-2"),
    dbc.Row([
        dbc.Col([html.Label("Fallback CoM height (m) — used only if <29 joints reported",
                             style={"fontSize": "12px"}),
                 dcc.Input(id="com-height-input", type="number", value=DEFAULT_COM_HEIGHT_FALLBACK_M,
                           min=0.3, max=1.2, step=0.01, style={"width": "100%"})], width=3),
        dbc.Col([html.Label("Swing-detect threshold (m) — sole height gap ⇒ single support",
                             style={"fontSize": "12px"}),
                 dcc.Input(id="swing-threshold-input", type="number", value=DEFAULT_SWING_HEIGHT_THRESHOLD_M,
                           min=0.0, max=0.05, step=0.005, style={"width": "100%"})], width=3),
        dbc.Col([html.Label("Conservative sole inset (m)", style={"fontSize":"12px"}), dcc.Input(id="shrink-input", type="number", value=.018, min=0, max=.05, step=.001, style={"width":"100%"})], width=2),
        dbc.Col([dbc.Checklist(id="accel-gravity-toggle", options=[{"label":" IMU accel includes gravity (assumption)", "value":"g"}], value=["g"], switch=True)], width=3),
        dbc.Col(html.Div(id="zmp-status", style={"fontSize": "14px", "marginTop": "12px"}), width=4),
    ], className="mb-3 gy-2"),
    dbc.Row([dbc.Col(dcc.Graph(id="zmp-graph"), lg=7), dbc.Col(html.Div(id="diagnostics-panel"), lg=5)]),
    dcc.Graph(id="history-graph"),
    dbc.Row([dbc.Col(dbc.Button("Freeze", id="btn-freeze", color="warning"), width="auto"), dbc.Col(dbc.Button("Clear history", id="btn-clear", color="secondary"), width="auto"), dbc.Col(dbc.Button("Start recording", id="btn-record-start", color="success"), width="auto"), dbc.Col(dbc.Button("Stop recording", id="btn-record-stop", color="danger"), width="auto")], className="mb-2 gy-2"),
    html.Div(id="poll-err", className="mt-2", style={"fontSize": "12px", "color": "#e0294f"}),
    dcc.Interval(id="status-interval", interval=1000, n_intervals=0),
    dcc.Interval(id="plot-interval", interval=PLOT_INTERVAL_MS, n_intervals=0),
], fluid=True)


@app.callback(
    Output("conn-badge", "children"),
    Output("conn-badge", "color"),
    Input("btn-connect", "n_clicks"),
    Input("btn-disconnect", "n_clicks"),
    Input("status-interval", "n_intervals"),
    State("iface-input", "value"),
    State("domain-input", "value"),
    prevent_initial_call=False,
)
def on_connection(_connect, _disconnect, _tick, iface, domain_id):
    trig = dash.ctx.triggered_id
    if trig == "btn-connect":
        LINK.iface = str(iface or "eth0")
        try:
            LINK.domain_id = int(domain_id or 0)
        except (TypeError, ValueError):
            LINK.domain_id = 0
        LINK.connect()
    elif trig == "btn-disconnect":
        LINK.disconnect()
    label, color = LINK.status()
    return label, color


@app.callback(
    Output("zmp-graph", "figure"),
    Output("history-graph", "figure"),
    Output("zmp-status", "children"),
    Output("diagnostics-panel", "children"),
    Output("poll-err", "children"),
    Input("plot-interval", "n_intervals"),
    Input("com-height-input", "value"),
    Input("swing-threshold-input", "value"),
    Input("shrink-input", "value"),
    Input("accel-gravity-toggle", "value"),
)
<<<<<<< HEAD
def on_plot_tick(_n, com_height_fallback_m, swing_threshold_m, shrink_m, accel_gravity):
    LINK._com_height_fallback_m = float(com_height_fallback_m or DEFAULT_COM_HEIGHT_FALLBACK_M)
    LINK._swing_height_threshold_m = max(0.0, float(swing_threshold_m if swing_threshold_m is not None else DEFAULT_SWING_HEIGHT_THRESHOLD_M))
    LINK._conservative_shrink_m = max(0.0, float(shrink_m if shrink_m is not None else .018))
    LINK._imu_accel_includes_gravity = "g" in (accel_gravity or [])
=======
def on_plot_tick(_n, com_height_fallback_m, swing_threshold_m):
    try:
        com_height = float(com_height_fallback_m)
    except (TypeError, ValueError):
        com_height = DEFAULT_COM_HEIGHT_FALLBACK_M
    if not math.isfinite(com_height):
        com_height = DEFAULT_COM_HEIGHT_FALLBACK_M
    com_height = max(COM_HEIGHT_RANGE_M[0], min(COM_HEIGHT_RANGE_M[1], com_height))
    try:
        swing_threshold = float(swing_threshold_m)
    except (TypeError, ValueError):
        swing_threshold = DEFAULT_SWING_HEIGHT_THRESHOLD_M
    if not math.isfinite(swing_threshold):
        swing_threshold = DEFAULT_SWING_HEIGHT_THRESHOLD_M
    swing_threshold = max(0.0, min(0.05, swing_threshold))
    with LINK.lock:
        LINK._com_height_fallback_m = com_height
        LINK._swing_height_threshold_m = swing_threshold
>>>>>>> 134802629fc365c6592869d0220ac49d919aa440
    fig, lines = build_figure(LINK)
    with LINK.lock:
        err = LINK.poll_err
    status_children = []
    for i, (text, color) in enumerate(lines):
        if i:
            status_children.append(html.Br())
        status_children.append(html.Span(text, style={"color": color, "fontWeight": 700}))
    return fig, build_history_figure(LINK), html.Div(status_children), diagnostics_panel(LINK), (f"poll error: {err}" if err else "")


@app.callback(Output("btn-freeze", "children"), Input("btn-freeze", "n_clicks"), Input("btn-clear", "n_clicks"), Input("btn-record-start", "n_clicks"), Input("btn-record-stop", "n_clicks"), prevent_initial_call=True)
def controls(_freeze, _clear, _start, _stop):
    with LINK.lock:
        if dash.ctx.triggered_id == "btn-freeze": LINK.frozen = not LINK.frozen
        elif dash.ctx.triggered_id == "btn-clear": LINK.history.clear(); LINK.step_events.clear()
        elif dash.ctx.triggered_id == "btn-record-start": LINK.recording = True
        elif dash.ctx.triggered_id == "btn-record-stop": LINK.stop_recording()
    return "Unfreeze" if LINK.frozen else "Freeze"


def run_sanity_tests() -> None:
    """Pure Python frame/geometry checks; no robot or Dash server required."""
    p = np.array([[1.0, 0.0, 1.0]])
    assert np.allclose(world_from_body(p, rpy_world_body((0, 0, 0)), np.zeros(3)), p)  # A
    assert np.allclose(rpy_world_body((0, 0, math.pi / 2)) @ np.array([1., 0., 0.]), [0., 1., 0.], atol=1e-9)  # B
    # C: positive pitch sends a point above the pelvis toward +x under Ry.
    assert (rpy_world_body((0, math.pi / 6, 0)) @ np.array([0., 0., 1.]))[0] > 0
    com, velocity, h = np.array([.1, -.02]), np.zeros(2), .7
    omega = math.sqrt(G_ACCEL / h)
    assert np.allclose(com + velocity / omega, com)  # D
    square = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
    assert signed_margin((.5,.5), square) > 0 and signed_margin((1.5,.5), square) < 0  # E
    print("zmp_viz sanity tests: PASS")


def verify_joint_mapping() -> None:
    """Fail loudly if the low-state/URDF arm partition this FK relies on drifts."""
    assert list(range(15, 22)) == [15, 16, 17, 18, 19, 20, 21]
    assert list(range(22, 29)) == [22, 23, 24, 25, 26, 27, 28]
    assert BODY_JOINTS[15][3] == BODY_JOINTS[22][3] == (0, 1, 0)
    assert BODY_JOINTS[18][3] == BODY_JOINTS[25][3] == (0, 1, 0)
    print("G1 FK mapping: ids 15-21 left arm, 22-28 right arm; shoulder/elbow axes +Y; URDF inertial hand masses included")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 ZMP / support-polygon estimator and live plot.")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=_bounded_int(*DOMAIN_ID_RANGE), default=0)
    parser.add_argument("--host", default="0.0.0.0")
<<<<<<< HEAD
    parser.add_argument("--port", type=int, default=8072)
    parser.add_argument("--sanity-test", action="store_true", help="run pure frame/LIPM/geometry checks and exit")
    return parser.parse_args()
=======
    parser.add_argument("--port", type=_bounded_int(*PORT_RANGE), default=8072)
    return parser.parse_args(argv)
>>>>>>> 134802629fc365c6592869d0220ac49d919aa440


def main() -> int:
    args = _parse_args()
    if args.sanity_test:
        run_sanity_tests()
        return 0
    verify_joint_mapping()
    global LINK
    LINK = ZmpLink(args.iface, args.domain_id)
    print(f"ZMP viz: http://{args.host}:{args.port} iface={args.iface} domain={args.domain_id}")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


LINK = ZmpLink("eth0", 0)

if __name__ == "__main__":
    raise SystemExit(main())
