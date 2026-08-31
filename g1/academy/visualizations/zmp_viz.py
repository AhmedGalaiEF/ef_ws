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

Both the support polygon and the CoM offset are computed pelvis-relative,
then rotated by the robot's current body yaw (IMU `rpy`, or the odom pose's
own yaw on the `rt/odom`/SLAM fallback paths) before being placed at the
odom xy position. This matters once the robot is turning while walking —
without it, the plotted geometry stays correct in shape but drifts out of
true world orientation as heading diverges from 0, which a single static
snapshot never exercises. If no yaw source is available on a given tick,
this falls back to the old unrotated placement, flagged in the status line.

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
CoM_accel_xy is the IMU's measured linear acceleration
(`get_imus()["acc"]`, gravity-compensated and rotated from body into world
frame via the IMU roll/pitch/yaw), smoothed over a short moving-average
window since raw accelerometer noise would otherwise dominate the estimate.
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
    15: (14, (0.0039563, 0.10022, 0.23778), _L_SHOULDER_PITCH_Q, (0, 1, 0)),   # left_shoulder_pitch
    16: (15, (0, 0.038, -0.013831), _L_SHOULDER_ROLL_Q, (1, 0, 0)),            # left_shoulder_roll
    17: (16, (0, 0.00624, -0.1032), _IDENT_Q, (0, 0, 1)),                      # left_shoulder_yaw
    18: (17, (0.015783, 0, -0.080518), _IDENT_Q, (0, 1, 0)),                   # left_elbow
    19: (18, (0.1, 0.00188791, -0.01), _IDENT_Q, (1, 0, 0)),                   # left_wrist_roll
    20: (19, (0.038, 0, 0), _IDENT_Q, (0, 1, 0)),                             # left_wrist_pitch
    21: (20, (0.046, 0, 0), _IDENT_Q, (0, 0, 1)),                            # left_wrist_yaw
    22: (14, (0.0039563, -0.10021, 0.23778), _R_SHOULDER_PITCH_Q, (0, 1, 0)),  # right_shoulder_pitch
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
    """FK for however many of the 29 body joints are given — pass 12 for
    legs-only (support polygon), 29 for legs+waist+arms (whole-body CoM).
    Returns {joint_id: (world_R, world_t)}, pelvis-relative (pelvis itself
    sits at PELVIS_POS in whatever frame `q` is silent about — i.e. this is
    base/pelvis-relative geometry, not world/odom; the odom pose is applied
    separately to place it in the world). Every BODY_JOINTS parent id is
    numerically smaller than its child, so a single forward pass suffices.
    """
    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    root_R = np.eye(3, dtype=np.float64)
    root_t = np.array(PELVIS_POS, dtype=np.float64)
    for jid in range(len(q)):
        if jid not in BODY_JOINTS:
            continue
        parent, pos, quat, axis = BODY_JOINTS[jid]
        parent_R, parent_t = (root_R, root_t) if parent == -1 else frames[parent]
        world_t = parent_t + parent_R @ np.array(pos, dtype=np.float64)
        world_R = parent_R @ _quat_to_R(quat) @ _axis_R(axis, float(q[jid]))
        frames[jid] = (world_R, world_t)
    return frames


def foot_sole_contacts_pelvis_frame(world_R: np.ndarray, world_t: np.ndarray) -> np.ndarray:
    """4 real sole-contact points (pelvis frame, xyz), from an ankle-roll
    frame — see FOOT_SOLE_CONTACTS_LOCAL."""
    return world_t + FOOT_SOLE_CONTACTS_LOCAL @ world_R.T


def whole_body_com_pelvis_frame(q: list[float]) -> Optional[tuple[np.ndarray, float]]:
    """Whole-body CoM (x, y, z), pelvis-relative, mass-weighting every URDF
    link's own CoM through FK of the sensed joint angles. Needs all 29
    leg+waist+arm joints; returns None if fewer are available (older
    firmware, or a caller that only has leg_q)."""
    if len(q) < 29:
        return None
    frames = body_frames(q)
    weighted = np.zeros(3, dtype=np.float64)
    for jid, (mass, com_local) in LINK_INERTIAL.items():
        R, t = frames[jid]
        weighted += mass * (t + R @ np.array(com_local, dtype=np.float64))
    for attach_jid, fixed_pos, mass, com_local in EXTRA_MASSES:
        if attach_jid is None:
            R, t = np.eye(3, dtype=np.float64), np.array(PELVIS_POS, dtype=np.float64)
        else:
            R, t = frames[attach_jid]
        anchor = t + R @ np.array(fixed_pos, dtype=np.float64)
        weighted += mass * (anchor + R @ np.array(com_local, dtype=np.float64))
    com = weighted / TOTAL_MASS_KG
    return com[:2], float(com[2])


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

DEFAULT_COM_HEIGHT_FALLBACK_M = PELVIS_POS[2]
DEFAULT_SWING_HEIGHT_THRESHOLD_M = 0.015  # foot sole height gap ⇒ treat as single support


def _extract_xy(odom: Optional[dict]) -> Optional[tuple[float, float]]:
    if not odom:
        return None
    position = odom.get("position")
    if position is not None:
        return float(position[0]), float(position[1])
    pose = odom.get("pose")
    if pose is not None:
        return float(pose[0]), float(pose[1])
    return None


def _extract_yaw(odom: Optional[dict], imu: Optional[dict]) -> Optional[float]:
    """Best-available body yaw, for rotating pelvis-relative geometry
    (feet, CoM) into world/odom frame. IMU rpy is present on the same poll
    tick's `get_imus()`/`get_odom()["imu"]` for the primary (SportModeState_)
    odom path; the `rt/odom`/SLAM fallbacks instead carry yaw pre-baked into
    their own `pose` tuple. Returns None only if neither is available (in
    which case callers skip rotation, same as the old behavior)."""
    if imu is not None and imu.get("rpy") is not None:
        return float(imu["rpy"][2])
    if odom is not None:
        pose = odom.get("pose")
        if pose is not None and len(pose) >= 3:
            return float(pose[2])
    return None


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
        self.com_vel_xy: Optional[tuple[float, float]] = None
        self.left_corners: Optional[np.ndarray] = None
        self.right_corners: Optional[np.ndarray] = None
        self.zmp_xy: Optional[tuple[float, float]] = None
        self.icp_xy: Optional[tuple[float, float]] = None
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

        self._stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None

    def connect(self) -> None:
        with self.lock:
            self.connect_requested = True
            if self.g1 is not None:
                return
            try:
                from sdk_wrapper import G1  # deferred: only needed once connecting
                self.g1 = G1(self.iface, domain_id=self.domain_id)
                self.init_err = None
            except Exception as exc:
                self.init_err = str(exc)
                self.g1 = None
                return
            self._stop.clear()
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()

    def disconnect(self) -> None:
        with self.lock:
            self.connect_requested = False
            self._stop.set()
            self.g1 = None

    def status(self) -> tuple[str, str]:
        with self.lock:
            if self.g1 is not None:
                return "Connected", "success"
            if not self.connect_requested:
                return "Disconnected", "secondary"
            if self.init_err is not None:
                return "Error", "danger"
            return "Connecting…", "warning"

    def _poll_loop(self) -> None:
        period = 1.0 / POLL_HZ
        while not self._stop.is_set():
            g1 = self.g1
            if g1 is not None:
                try:
                    self._poll_once(g1, com_height_fallback_m=self._com_height_fallback_m,
                                     swing_threshold_m=self._swing_height_threshold_m)
                    with self.lock:
                        self.poll_err = None
                except Exception as exc:
                    with self.lock:
                        self.poll_err = str(exc)
            time.sleep(period)

    # Configurable from the UI; read by the poll loop each tick.
    _com_height_fallback_m = DEFAULT_COM_HEIGHT_FALLBACK_M
    _swing_height_threshold_m = DEFAULT_SWING_HEIGHT_THRESHOLD_M

    def _poll_once(self, g1, com_height_fallback_m: float, swing_threshold_m: float) -> None:
        lowstate = g1.get_lowstate()
        odom = g1.get_odom()
        imu = g1.get_imus()

        base_xy = _extract_xy(odom)
        yaw = _extract_yaw(odom, imu)
        yaw_R2 = None if yaw is None else _rot_z(yaw)[:2, :2]
        positions = (lowstate.get("joint_positions") or []) if lowstate is not None else []
        torques = (lowstate.get("joint_torques") or []) if lowstate is not None else []
        leg_q = positions[:12] if len(positions) >= 12 else None

        com_xy = None
        left_corners = right_corners = None
        h_com = com_height_fallback_m
        h_com_auto = False
        stance = "double"
        stance_torque_overridden = False

        if leg_q is not None:
            frames = body_frames(leg_q)
            lR, lt = frames[LEFT_FOOT_ID]
            rR, rt = frames[RIGHT_FOOT_ID]
            left3 = foot_sole_contacts_pelvis_frame(lR, lt)
            right3 = foot_sole_contacts_pelvis_frame(rR, rt)
            left_ground_z = float(left3[:, 2].mean())
            right_ground_z = float(right3[:, 2].mean())
            left_corners, right_corners = left3[:, :2], right3[:, :2]

            # Coarse gait-phase heuristic (still no force/contact sensing): if
            # one sole sits meaningfully higher than the other, treat it as
            # the swing foot and shrink both the support polygon and the
            # ground-height reference to the stance foot alone, instead of
            # always assuming double support.
            gap = left_ground_z - right_ground_z
            if gap > swing_threshold_m:
                stance, ground_z = "right", right_ground_z
            elif -gap > swing_threshold_m:
                stance, ground_z = "left", left_ground_z
            else:
                stance, ground_z = "double", (left_ground_z + right_ground_z) / 2.0

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
            if stance != "double" and len(torques) >= 12:
                left_load = abs(torques[3]) + abs(torques[4])    # left knee + ankle_pitch
                right_load = abs(torques[9]) + abs(torques[10])  # right knee + ankle_pitch
                stance_load, swing_load = (left_load, right_load) if stance == "left" else (right_load, left_load)
                if swing_load > stance_load:
                    stance, ground_z = "double", (left_ground_z + right_ground_z) / 2.0
                    stance_torque_overridden = True

            whole_body = whole_body_com_pelvis_frame(positions[:29])
            if whole_body is not None:
                com_xy_local, com_z_local = whole_body
                h_com = com_z_local - ground_z
                h_com_auto = True
            else:
                com_xy_local = np.zeros(2, dtype=np.float64)  # legs-only: pelvis-as-CoM fallback

            if base_xy is not None:
                # Feet/CoM were computed pelvis-relative; rotate by body yaw
                # (when available) then shift into the same world/odom frame
                # the base (pelvis) position is reported in, so the plotted
                # polygon/CoM stay correctly oriented as the robot turns
                # instead of only being right while facing yaw=0. Skipping
                # rotation when yaw is unavailable is a small extra
                # approximation on top of the ones already documented above.
                if yaw_R2 is not None:
                    left_corners = left_corners @ yaw_R2.T
                    right_corners = right_corners @ yaw_R2.T
                    com_xy_local = yaw_R2 @ com_xy_local
                shift = np.array(base_xy, dtype=np.float64)
                left_corners = left_corners + shift
                right_corners = right_corners + shift
                com_xy = (float(shift[0] + com_xy_local[0]), float(shift[1] + com_xy_local[1]))
        elif base_xy is not None:
            com_xy = base_xy  # no leg data at all: fall back to pelvis-as-CoM

        now = time.time()

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
        accel_source = "none"
        if imu is not None and imu.get("acc") is not None and imu.get("rpy") is not None:
            roll, pitch, imu_yaw = imu["rpy"]
            R = _rot_z(imu_yaw) @ _rot_y(pitch) @ _rot_x(roll)
            a_body = np.array(imu["acc"], dtype=np.float64)
            a_world = R @ a_body - np.array([0.0, 0.0, G_ACCEL])
            with self.lock:
                self._accel_hist.append((float(a_world[0]), float(a_world[1])))
                ax_world = sum(v[0] for v in self._accel_hist) / len(self._accel_hist)
                ay_world = sum(v[1] for v in self._accel_hist) / len(self._accel_hist)
            accel_source = "imu"
        elif kinematic_accel_xy is not None:
            ax_world, ay_world = kinematic_accel_xy
            accel_source = "kinematic"

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

        with self.lock:
            self.com_xy = com_xy
            self.com_vel_xy = com_vel_xy
            self.left_corners = left_corners
            self.right_corners = right_corners
            self.zmp_xy = zmp_xy
            self.icp_xy = icp_xy
            self.h_com = h_com
            self.h_com_auto = h_com_auto
            self.stance = stance
            self.stance_torque_overridden = stance_torque_overridden
            self.yaw_applied = yaw_R2 is not None
            self.accel_source = accel_source
            self.ts = now
            if zmp_xy is not None:
                self.zmp_trail.append((now, zmp_xy[0], zmp_xy[1]))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure(link: "ZmpLink") -> tuple[go.Figure, list[tuple[str, str]]]:
    with link.lock:
        com_xy = link.com_xy
        left_corners = None if link.left_corners is None else link.left_corners.copy()
        right_corners = None if link.right_corners is None else link.right_corners.copy()
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
            zmp_line = (f"ZMP {'INSIDE' if inside else 'OUTSIDE'} support polygon — margin {margin * 100:+.1f} cm · {h_com_note}", zmp_color)
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
            icp_line = (f"ICP (where CoM is headed) {'INSIDE' if icp_inside else 'OUTSIDE'} support polygon — margin {icp_margin * 100:+.1f} cm", icp_color)
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
        dbc.Col(html.Div(id="zmp-status", style={"fontSize": "15px", "marginTop": "24px"}), width=6),
    ], className="mb-3 gy-2"),
    dcc.Graph(id="zmp-graph"),
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
    Output("zmp-status", "children"),
    Output("poll-err", "children"),
    Input("plot-interval", "n_intervals"),
    Input("com-height-input", "value"),
    Input("swing-threshold-input", "value"),
)
def on_plot_tick(_n, com_height_fallback_m, swing_threshold_m):
    LINK._com_height_fallback_m = float(com_height_fallback_m or DEFAULT_COM_HEIGHT_FALLBACK_M)
    LINK._swing_height_threshold_m = max(0.0, float(swing_threshold_m if swing_threshold_m is not None else DEFAULT_SWING_HEIGHT_THRESHOLD_M))
    fig, lines = build_figure(LINK)
    with LINK.lock:
        err = LINK.poll_err
    status_children = []
    for i, (text, color) in enumerate(lines):
        if i:
            status_children.append(html.Br())
        status_children.append(html.Span(text, style={"color": color, "fontWeight": 700}))
    return fig, html.Div(status_children), (f"poll error: {err}" if err else "")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 ZMP / support-polygon estimator and live plot.")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8072)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    global LINK
    LINK = ZmpLink(args.iface, args.domain_id)
    print(f"ZMP viz: http://{args.host}:{args.port} iface={args.iface} domain={args.domain_id}")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


LINK = ZmpLink("eth0", 0)

if __name__ == "__main__":
    raise SystemExit(main())
