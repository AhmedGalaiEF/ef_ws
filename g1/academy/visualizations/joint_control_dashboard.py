#!/usr/bin/env python3
"""
G1 Joint Control Dashboard.

Plots the 29 body joints of the Unitree G1 (legs, waist, arms — Dex3 hands are
out of scope, same as sdk_wrapper.G1.move_ll_joint) as a live 3D skeleton
and gives every joint its own "Control" button. Clicking a joint opens a modal
with sliders for q, dq, kp, kd, tau and a safety ramp duration, bounded by
that joint's own limits, and applying it only ever touches that one joint's
entry in a persistent per-joint command table — everything else keeps
publishing whatever it last held.

Two control paths, mirroring sdk_wrapper.G1:
  * Normal mode  — rt/arm_sdk, waist + arm joints only (legs stay with
    ai_sport). First joint edit ramps the arm_sdk weight 0→1 (same smoothstep
    engage_arms() uses) while holding the rest of the upper body at its
    current sensed pose; "Release Arms" ramps the weight back to 0
    (release_arms()) and hands control back to the ai_sport service.
  * Dev mode — rt/lowcmd, all 29 joints, ai_sport stopped. Toggling it calls
    G1.toggle_dev_mode() (== toggle_service("ai_sport")); turning it on seeds
    every joint's hold target from its current sensed position with the same
    default gains sdk_wrapper._LowCmd.write() uses, so nothing moves at
    the moment ai_sport lets go.

Geometry (joint offsets/quats/axes/limits) is read from the G1 29-DOF MJCF
description in ../../sim (g1_29dof.xml), which matches the joint order used
throughout sdk_wrapper.py (LOWCMD_JOINTS = legs 0-11, waist 12-14,
arms 15-28).

A separate "IK EE Controller" button drives one arm's end-effector directly,
using the same ArmFK/ArmIK pipeline as ../../hand_pose_navigation/recognition_app_v3.py
(compute the current hand pose with ArmFK, solve a target pose with ArmIK's
damped-least-squares solver, apply the resulting 7-joint solution): a modal
with 6 sliders (x, y, z, roll, pitch, yaw) sets an absolute target pose for
the chosen hand in base_link frame; on Apply, the solved joint angles are
written into the same per-joint command table as the individual joint
controls, ramped together as one coordinated move — only that arm's 7 joints
change, everything else keeps holding.

Run on the robot's Jetson / dev box (needs unitree_sdk2py + dash + plotly +
dash-bootstrap-components):

    python3 joint_control_dashboard.py --iface eth0 --domain-id 0 --port 8060

The page loads even with no robot reachable; click Connect once the network
is up.
"""
from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html

# Make `from sdk_wrapper import G1` and `from hand_pose_navigation...` work
# regardless of cwd. This file lives in academy/visualizations/; sdk_wrapper.py
# sits one level up in academy/, and the hand_pose_navigation package one more
# level up in g1/.
_SCRIPT_DIR = Path(__file__).resolve().parent      # academy/visualizations
_ACADEMY_DIR = _SCRIPT_DIR.parent                    # academy
_G1_ROOT_DIR = _ACADEMY_DIR.parent                   # g1
for _p in (_SCRIPT_DIR, _ACADEMY_DIR, _G1_ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# Joint geometry / limits — sourced verbatim from g1_29dof.xml (sim/, see
# ../../go2/sim/unitree_robots/g1/g1_29dof.xml), in the same 0-28 order
# sdk_wrapper.LOWCMD_JOINTS uses.
# ---------------------------------------------------------------------------

PELVIS_POS = (0.0, 0.0, 0.793)
LEG_GROUPS = ("left_leg", "right_leg")
UPPER_BODY_JOINTS = tuple(range(12, 29))
WAIST_JOINTS = (12, 13, 14)
ALL_JOINTS = tuple(range(29))

GROUP_ORDER = ["left_leg", "right_leg", "waist", "left_arm", "right_arm"]
GROUP_LABELS = {
    "left_leg": "Left Leg", "right_leg": "Right Leg", "waist": "Waist",
    "left_arm": "Left Arm", "right_arm": "Right Arm",
}
# First five slots of the validated dark-mode categorical palette, fixed
# order (never re-cycled) — see the dataviz skill's references/palette.md.
GROUP_COLORS = {
    "left_leg": "#3987e5", "right_leg": "#d95926", "waist": "#199e70",
    "left_arm": "#c98500", "right_arm": "#d55181",
}
PELVIS_COLOR = "#c3c2b7"


@dataclass(frozen=True)
class JointSpec:
    id: int
    name: str
    group: str
    parent: int  # -1 == pelvis root
    pos: tuple
    quat: tuple  # (w, x, y, z), fixed link-frame rotation
    axis: tuple
    q_min: float
    q_max: float
    tau_max: float
    kp_lowcmd: float
    kd_lowcmd: float
    kp_armsdk: float | None  # None for leg joints (never used in arm_sdk mode)
    kd_armsdk: float | None
    kp_max: float
    kd_max: float

    @property
    def label(self) -> str:
        return f"{self.id}: {self.name.replace('_', ' ')}"


def _spec(jid, name, group, parent, pos, quat, axis, q_min, q_max, tau_max,
          kp_lowcmd, kd_lowcmd, kp_armsdk, kd_armsdk, kp_max, kd_max) -> JointSpec:
    return JointSpec(jid, name, group, parent, pos, quat, axis, q_min, q_max, tau_max,
                      kp_lowcmd, kd_lowcmd, kp_armsdk, kd_armsdk, kp_max, kd_max)


_IDENT_Q = (1.0, 0.0, 0.0, 0.0)
_HIP_ROLL_Q = (0.996179, 0.0, -0.0873386, 0.0)
_KNEE_Q = (0.996179, 0.0, 0.0873386, 0.0)

JOINT_TABLE: list[JointSpec] = [
    # -- left leg (dev mode only) ------------------------------------------
    _spec(0, "left_hip_pitch", "left_leg", -1, (0, 0.064452, -0.1027), _IDENT_Q, (0, 1, 0),
          -2.5307, 2.8798, 88, 60, 1, None, None, 220, 10),
    _spec(1, "left_hip_roll", "left_leg", 0, (0, 0.052, -0.030465), _HIP_ROLL_Q, (1, 0, 0),
          -0.5236, 2.9671, 88, 60, 1, None, None, 220, 10),
    _spec(2, "left_hip_yaw", "left_leg", 1, (0.025001, 0, -0.12412), _IDENT_Q, (0, 0, 1),
          -2.7576, 2.7576, 88, 60, 1, None, None, 220, 10),
    _spec(3, "left_knee", "left_leg", 2, (-0.078273, 0.0021489, -0.17734), _KNEE_Q, (0, 1, 0),
          -0.087267, 2.8798, 139, 100, 2, None, None, 300, 12),
    _spec(4, "left_ankle_pitch", "left_leg", 3, (0, -9.4445e-05, -0.30001), _IDENT_Q, (0, 1, 0),
          -0.87267, 0.5236, 50, 40, 1, None, None, 150, 10),
    _spec(5, "left_ankle_roll", "left_leg", 4, (0, 0, -0.017558), _IDENT_Q, (1, 0, 0),
          -0.2618, 0.2618, 50, 40, 1, None, None, 150, 10),
    # -- right leg (dev mode only) ------------------------------------------
    _spec(6, "right_hip_pitch", "right_leg", -1, (0, -0.064452, -0.1027), _IDENT_Q, (0, 1, 0),
          -2.5307, 2.8798, 88, 60, 1, None, None, 220, 10),
    _spec(7, "right_hip_roll", "right_leg", 6, (0, -0.052, -0.030465), _HIP_ROLL_Q, (1, 0, 0),
          -2.9671, 0.5236, 88, 60, 1, None, None, 220, 10),
    _spec(8, "right_hip_yaw", "right_leg", 7, (0.025001, 0, -0.12412), _IDENT_Q, (0, 0, 1),
          -2.7576, 2.7576, 88, 60, 1, None, None, 220, 10),
    _spec(9, "right_knee", "right_leg", 8, (-0.078273, -0.0021489, -0.17734), _KNEE_Q, (0, 1, 0),
          -0.087267, 2.8798, 139, 100, 2, None, None, 300, 12),
    _spec(10, "right_ankle_pitch", "right_leg", 9, (0, 9.4445e-05, -0.30001), _IDENT_Q, (0, 1, 0),
          -0.87267, 0.5236, 50, 40, 1, None, None, 150, 10),
    _spec(11, "right_ankle_roll", "right_leg", 10, (0, 0, -0.017558), _IDENT_Q, (1, 0, 0),
          -0.2618, 0.2618, 50, 40, 1, None, None, 150, 10),
    # -- waist ----------------------------------------------------------------
    _spec(12, "waist_yaw", "waist", -1, (0, 0, 0), _IDENT_Q, (0, 0, 1),
          -2.618, 2.618, 88, 60, 1, 480, 12, 700, 24),
    _spec(13, "waist_roll", "waist", 12, (-0.0039635, 0, 0.035), _IDENT_Q, (1, 0, 0),
          -0.52, 0.52, 50, 40, 1, 480, 12, 700, 24),
    _spec(14, "waist_pitch", "waist", 13, (0, 0, 0.019), _IDENT_Q, (0, 1, 0),
          -0.52, 0.52, 50, 40, 1, 480, 12, 700, 24),
    # -- left arm ---------------------------------------------------------
    _spec(15, "left_shoulder_pitch", "left_arm", 14, (0.0039563, 0.10022, 0.23778),
          (0.990264, 0.139201, 1.38722e-05, -9.86868e-05), (0, 1, 0),
          -3.0892, 2.6704, 25, 40, 1, 30, 1.5, 150, 8),
    _spec(16, "left_shoulder_roll", "left_arm", 15, (0, 0.038, -0.013831),
          (0.990268, -0.139172, 0, 0), (1, 0, 0),
          -1.5882, 2.2515, 25, 40, 1, 30, 1.5, 150, 8),
    _spec(17, "left_shoulder_yaw", "left_arm", 16, (0, 0.00624, -0.1032), _IDENT_Q, (0, 0, 1),
          -2.618, 2.618, 25, 40, 1, 30, 1.5, 150, 8),
    _spec(18, "left_elbow", "left_arm", 17, (0.015783, 0, -0.080518), _IDENT_Q, (0, 1, 0),
          -1.0472, 2.0944, 25, 40, 1, 30, 1.5, 150, 8),
    _spec(19, "left_wrist_roll", "left_arm", 18, (0.1, 0.00188791, -0.01), _IDENT_Q, (1, 0, 0),
          -1.97222, 1.97222, 25, 40, 1, 30, 1.5, 150, 8),
    _spec(20, "left_wrist_pitch", "left_arm", 19, (0.038, 0, 0), _IDENT_Q, (0, 1, 0),
          -1.61443, 1.61443, 5, 40, 1, 30, 1.5, 90, 6),
    _spec(21, "left_wrist_yaw", "left_arm", 20, (0.046, 0, 0), _IDENT_Q, (0, 0, 1),
          -1.61443, 1.61443, 5, 40, 1, 30, 1.5, 90, 6),
    # -- right arm ----------------------------------------------------------
    _spec(22, "right_shoulder_pitch", "right_arm", 14, (0.0039563, -0.10021, 0.23778),
          (0.990264, -0.139201, 1.38722e-05, 9.86868e-05), (0, 1, 0),
          -3.0892, 2.6704, 25, 40, 1, 30, 1.5, 150, 8),
    _spec(23, "right_shoulder_roll", "right_arm", 22, (0, -0.038, -0.013831),
          (0.990268, 0.139172, 0, 0), (1, 0, 0),
          -2.2515, 1.5882, 25, 40, 1, 30, 1.5, 150, 8),
    _spec(24, "right_shoulder_yaw", "right_arm", 23, (0, -0.00624, -0.1032), _IDENT_Q, (0, 0, 1),
          -2.618, 2.618, 25, 40, 1, 30, 1.5, 150, 8),
    _spec(25, "right_elbow", "right_arm", 24, (0.015783, 0, -0.080518), _IDENT_Q, (0, 1, 0),
          -1.0472, 2.0944, 25, 40, 1, 30, 1.5, 150, 8),
    _spec(26, "right_wrist_roll", "right_arm", 25, (0.1, -0.00188791, -0.01), _IDENT_Q, (1, 0, 0),
          -1.97222, 1.97222, 25, 40, 1, 30, 1.5, 150, 8),
    _spec(27, "right_wrist_pitch", "right_arm", 26, (0.038, 0, 0), _IDENT_Q, (0, 1, 0),
          -1.61443, 1.61443, 5, 40, 1, 30, 1.5, 90, 6),
    _spec(28, "right_wrist_yaw", "right_arm", 27, (0.046, 0, 0), _IDENT_Q, (0, 0, 1),
          -1.61443, 1.61443, 5, 40, 1, 30, 1.5, 90, 6),
]
JOINT_BY_ID: dict[int, JointSpec] = {s.id: s for s in JOINT_TABLE}
JOINTS_BY_GROUP: dict[str, list[JointSpec]] = {g: [] for g in GROUP_ORDER}
for _s in JOINT_TABLE:
    JOINTS_BY_GROUP[_s.group].append(_s)

# Conservative default — the MJCF carries no velocity-limit field, so this is
# a dashboard-side safety cap, not a vendor spec.
DQ_CAP = 3.0
RAMP_MIN_S, RAMP_MAX_S, RAMP_DEFAULT_S = 0.05, 3.0, 0.6
ENGAGE_DURATION_S = 1.0   # matches sdk_wrapper.G1.engage_arms()
RELEASE_DURATION_S = 3.0  # matches sdk_wrapper.G1.release_arms()
POLL_HZ = 20.0
PUBLISH_HZ = 50.0
SERVICE_POLL_EVERY_S = 3.0
PLOT_INTERVAL_MS = 700
STATUS_INTERVAL_MS = 1200


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _smoothstep(ratio: float) -> float:
    ratio = _clamp(ratio, 0.0, 1.0)
    return ratio * ratio * (3.0 - 2.0 * ratio)


# ---------------------------------------------------------------------------
# Forward kinematics — same composition style as hand_pose_navigation/arm_fk.py,
# generalized to the full 29-joint chain for the skeleton plot.
# ---------------------------------------------------------------------------

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


def compute_skeleton(q_values: list[float]) -> dict[Any, tuple]:
    """q_values: length-29 list of joint angles (rad). Returns {"pelvis": (x,y,z), id: (x,y,z), ...}."""
    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    root_R = np.eye(3, dtype=np.float64)
    root_t = np.array(PELVIS_POS, dtype=np.float64)
    positions: dict[Any, tuple] = {"pelvis": tuple(root_t)}
    for jid in range(29):
        spec = JOINT_BY_ID[jid]
        parent_R, parent_t = (root_R, root_t) if spec.parent == -1 else frames[spec.parent]
        world_t = parent_t + parent_R @ np.array(spec.pos, dtype=np.float64)
        world_R = parent_R @ _quat_to_R(spec.quat) @ _axis_R(spec.axis, float(q_values[jid]))
        frames[jid] = (world_R, world_t)
        positions[jid] = tuple(world_t)
    return positions


# ---------------------------------------------------------------------------
# IK end-effector controller — same ArmFK/ArmIK primitives as
# hand_pose_navigation/recognition_app_v3.py (e.g. _lift_end_effector_z_slowly,
# _run_extend_arm_ee_ik): FK the current arm pose, build a target 4x4 pose,
# solve with ArmIK's damped-least-squares solver, apply the 7-joint result.
# ---------------------------------------------------------------------------

LEFT_ARM_JOINT_IDS = tuple(s.id for s in JOINTS_BY_GROUP["left_arm"])
RIGHT_ARM_JOINT_IDS = tuple(s.id for s in JOINTS_BY_GROUP["right_arm"])
ARM_JOINT_IDS = {"left": LEFT_ARM_JOINT_IDS, "right": RIGHT_ARM_JOINT_IDS}

EE_POS_RANGE_M = 0.35     # slider span around the current x/y/z, each direction
EE_ROT_RANGE_RAD = math.pi  # roll/pitch/yaw sliders always span -pi..pi
EE_IK_RAMP_S = 0.8        # coordinated 7-joint move gets a touch more time than a single joint
# Same ArmIK defaults sdk_wrapper.G1._arm_ik_solver() uses.
EE_IK_MAX_ITER = 24
EE_IK_TOL_POS_M = 0.005
EE_IK_TOL_ROT_RAD = 0.02


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def _R_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of _rpy_to_R (ZYX Euler extraction)."""
    pitch = math.asin(_clamp(-R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])  # gimbal lock: fold roll+yaw into roll
        yaw = 0.0
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Persistent per-joint command state + background publish/poll loops.
# ---------------------------------------------------------------------------

@dataclass
class JointCmdState:
    q: float
    dq: float
    kp: float
    kd: float
    tau: float
    ramp_from_q: float
    ramp_to_q: float
    ramp_start_t: float
    ramp_duration: float


class RobotLink:
    """Owns the G1 connection, the live-state poller, and the command publisher.

    Every joint edit only ever mutates its own JointCmdState entry — the
    publish loop re-sends the full persistent table every tick, so untouched
    joints keep publishing exactly what they held before.
    """

    def __init__(self, iface: str, domain_id: int):
        self.iface = str(iface)
        self.domain_id = int(domain_id)
        self.lock = threading.RLock()

        self.g1 = None
        self.connect_requested = False
        self.init_err: str | None = None

        self.dev_mode = False       # locally tracked — see note in toggle_dev_mode()
        self.arm_engaged = False
        self.weight_from = 0.0
        self.weight_to = 0.0
        self.weight_start_t = 0.0
        self.weight_duration = 1e-3

        self.cmd: dict[int, JointCmdState] = {}
        self.sensed_q = [0.0] * 29
        self.sensed_dq = [0.0] * 29
        self.sensed_tau = [0.0] * 29
        self.sensed_ts = 0.0
        self.mode_machine = 0
        self.service_row: dict | None = None
        self.service_ts = 0.0

        self._stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._pub_thread: threading.Thread | None = None

    # -- connection -----------------------------------------------------
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
            self._pub_thread = threading.Thread(target=self._publish_loop, daemon=True)
            self._pub_thread.start()

    def disconnect(self) -> None:
        with self.lock:
            self.connect_requested = False
            self._stop.set()
            self.g1 = None
            self.dev_mode = False
            self.arm_engaged = False
            self.cmd.clear()

    def status(self) -> tuple[str, str]:
        with self.lock:
            if self.g1 is not None:
                return "Connected", "success"
            if not self.connect_requested:
                return "Disconnected", "secondary"
            if self.init_err is not None:
                return "Error", "danger"
            return "Connecting…", "warning"

    # -- background loops -------------------------------------------------
    def _poll_loop(self) -> None:
        period = 1.0 / POLL_HZ
        last_service_poll = 0.0
        while not self._stop.is_set():
            g1 = self.g1
            if g1 is not None:
                try:
                    msg = g1._lowstate_msg()
                    if msg is not None:
                        q = [float(msg.motor_state[i].q) for i in range(29)]
                        dq = [float(msg.motor_state[i].dq) for i in range(29)]
                        tau = [float(msg.motor_state[i].tau_est) for i in range(29)]
                        mode_machine = int(getattr(msg, "mode_machine", 0))
                        with self.lock:
                            self.sensed_q, self.sensed_dq, self.sensed_tau = q, dq, tau
                            self.mode_machine = mode_machine
                            self.sensed_ts = time.time()
                except Exception:
                    pass
                now = time.monotonic()
                if now - last_service_poll >= SERVICE_POLL_EVERY_S:
                    last_service_poll = now
                    try:
                        row = g1.get_service("ai_sport")
                        with self.lock:
                            self.service_row = row
                            self.service_ts = time.time()
                    except Exception:
                        pass
            time.sleep(period)

    def _publish_loop(self) -> None:
        period = 1.0 / PUBLISH_HZ
        while not self._stop.is_set():
            t0 = time.monotonic()
            with self.lock:
                g1 = self.g1
                if g1 is not None:
                    now = time.monotonic()
                    weight = self._interp_weight(now)
                    self.weight_now = weight
                    try:
                        if self.dev_mode:
                            self._publish_lowcmd(g1, now)
                        elif weight > 1e-4 or self.arm_engaged:
                            self._publish_armsdk(g1, now, weight)
                    except Exception:
                        pass
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _interp_joint(self, state: JointCmdState, now: float) -> float:
        if state.ramp_duration <= 0:
            state.q = state.ramp_to_q
            return state.q
        ratio = (now - state.ramp_start_t) / state.ramp_duration
        if ratio >= 1.0:
            state.q = state.ramp_to_q
        else:
            state.q = state.ramp_from_q + (state.ramp_to_q - state.ramp_from_q) * _smoothstep(ratio)
        return state.q

    def _interp_weight(self, now: float) -> float:
        if self.weight_duration <= 0:
            return self.weight_to
        ratio = (now - self.weight_start_t) / self.weight_duration
        return self.weight_to if ratio >= 1.0 else self.weight_from + (self.weight_to - self.weight_from) * _smoothstep(ratio)

    def _publish_lowcmd(self, g1, now: float) -> None:
        client = g1._lowcmd_client()
        msg = client.msg
        msg.mode_machine = int(self.mode_machine)
        for jid in range(29):
            state = self.cmd.get(jid)
            if state is None:
                continue
            q = self._interp_joint(state, now)
            mc = msg.motor_cmd[jid]
            mc.mode = 1
            mc.q = float(q)
            mc.dq = float(state.dq)
            mc.tau = float(state.tau)
            mc.kp = float(state.kp)
            mc.kd = float(state.kd)
        msg.crc = client.crc.Crc(msg)
        client.pub.Write(msg)

    def _publish_armsdk(self, g1, now: float, weight: float) -> None:
        client = g1._arm_sdk_client()
        msg = client.msg
        for jid in UPPER_BODY_JOINTS:
            state = self.cmd.get(jid)
            if state is None:
                continue
            q = self._interp_joint(state, now)
            mc = msg.motor_cmd[jid]
            mc.mode = 1
            mc.q = float(q)
            mc.dq = float(state.dq)
            mc.tau = float(state.tau)
            mc.kp = float(state.kp)
            mc.kd = float(state.kd)
        msg.motor_cmd[29].q = float(_clamp(weight, 0.0, 1.0))
        msg.crc = client.crc.Crc(msg)
        client.pub.Write(msg)

    # -- engage / release / dev-mode --------------------------------------
    def _start_weight_ramp(self, target: float, duration: float) -> None:
        now = time.monotonic()
        self.weight_from = self._interp_weight(now)
        self.weight_to = target
        self.weight_start_t = now
        self.weight_duration = max(1e-3, duration)

    def _engage_arms(self) -> None:
        now = time.monotonic()
        for jid in UPPER_BODY_JOINTS:
            spec = JOINT_BY_ID[jid]
            q0 = float(self.sensed_q[jid])
            self.cmd[jid] = JointCmdState(
                q=q0, dq=0.0, kp=spec.kp_armsdk, kd=spec.kd_armsdk, tau=0.0,
                ramp_from_q=q0, ramp_to_q=q0, ramp_start_t=now, ramp_duration=1e-3,
            )
        self._start_weight_ramp(1.0, ENGAGE_DURATION_S)
        self.arm_engaged = True

    def release_arms(self) -> tuple[bool, str]:
        with self.lock:
            if self.g1 is None:
                return False, "Not connected."
            if not self.arm_engaged and self._interp_weight(time.monotonic()) <= 1e-4:
                return False, "Arms are not engaged — nothing to release."
            self._start_weight_ramp(0.0, RELEASE_DURATION_S)
            self.arm_engaged = False
        return True, f"Releasing arm/waist control back to ai_sport over {RELEASE_DURATION_S:.0f}s…"

    def toggle_dev_mode(self) -> tuple[bool, str]:
        # NOTE: sdk_wrapper.G1.toggle_dev_mode() == toggle_service("ai_sport"),
        # which flips the service based on ServiceList()'s `status` field. This
        # dashboard tracks dev_mode locally (optimistic, from what we just
        # commanded) rather than re-deriving it from that status field, whose
        # exact 0/nonzero polarity isn't nailed down from static reading alone
        # — the raw service row is still surfaced in the status bar so you can
        # cross-check against the robot before trusting the badge.
        with self.lock:
            if self.g1 is None:
                return False, "Not connected."
            turning_on = not self.dev_mode
            try:
                self.g1.toggle_dev_mode()
            except Exception as exc:
                return False, f"toggle_dev_mode() failed: {exc}"
            self.dev_mode = turning_on
            now = time.monotonic()
            if turning_on:
                for jid in range(29):
                    spec = JOINT_BY_ID[jid]
                    q0 = float(self.sensed_q[jid])
                    self.cmd[jid] = JointCmdState(
                        q=q0, dq=0.0, kp=spec.kp_lowcmd, kd=spec.kd_lowcmd, tau=0.0,
                        ramp_from_q=q0, ramp_to_q=q0, ramp_start_t=now, ramp_duration=1e-3,
                    )
        if turning_on:
            return True, "Dev mode ENABLED — publishing rt/lowcmd for all 29 joints, ai_sport stopped."
        return True, "Dev mode disabled — ai_sport resumed; rt/lowcmd publishing stopped."

    # -- joint edits --------------------------------------------------------
    def set_joint_target(self, joint_id: int, q: float, dq: float, kp: float, kd: float,
                          tau: float, ramp_s: float) -> tuple[bool, str]:
        spec = JOINT_BY_ID.get(int(joint_id))
        if spec is None:
            return False, "Unknown joint."
        q = _clamp(q, spec.q_min, spec.q_max)
        kp = _clamp(kp, 0.0, spec.kp_max)
        kd = _clamp(kd, 0.0, spec.kd_max)
        tau = _clamp(tau, -spec.tau_max, spec.tau_max)
        dq = _clamp(dq, -DQ_CAP, DQ_CAP)
        ramp_s = _clamp(ramp_s, RAMP_MIN_S, RAMP_MAX_S)
        with self.lock:
            if self.g1 is None:
                return False, "Not connected — click Connect first."
            if spec.group in LEG_GROUPS and not self.dev_mode:
                return False, "Leg joints require Dev Mode."
            if not self.dev_mode and not self.arm_engaged:
                self._engage_arms()
            prior = self.cmd.get(spec.id)
            start_q = prior.q if prior is not None else float(self.sensed_q[spec.id])
            self.cmd[spec.id] = JointCmdState(
                q=start_q, dq=dq, kp=kp, kd=kd, tau=tau,
                ramp_from_q=start_q, ramp_to_q=q,
                ramp_start_t=time.monotonic(), ramp_duration=ramp_s,
            )
        return True, (f"{spec.name} (id {spec.id}) → {q:+.3f} rad over {ramp_s:.2f}s "
                       f"(kp={kp:.1f} kd={kd:.2f} tau={tau:+.2f} dq={dq:+.2f})")

    # -- IK end-effector controller -----------------------------------------
    def _arm_q_locked(self, side: str) -> np.ndarray:
        """Current 7-joint arm state — the held command target where the
        joint has one, otherwise the last sensed position. Caller must hold self.lock."""
        return np.array([
            self.cmd[j].q if j in self.cmd else float(self.sensed_q[j])
            for j in ARM_JOINT_IDS[side]
        ], dtype=np.float64)

    def ee_pose_snapshot(self, side: str) -> dict:
        """Current hand pose (x, y, z, roll, pitch, yaw) in base_link frame, via ArmFK."""
        from hand_pose_navigation.arm_fk import ArmFK
        with self.lock:
            q = self._arm_q_locked(side)
        T = ArmFK(arm=side, backend="urdf").compute_arm(q)
        roll, pitch, yaw = _R_to_rpy(T[:3, :3])
        return {"x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3]),
                "roll": roll, "pitch": pitch, "yaw": yaw}

    def set_arm_ee_target(self, side: str, x: float, y: float, z: float,
                           roll: float, pitch: float, yaw: float) -> tuple[bool, str, dict | None]:
        """Solve IK for a target hand pose and apply the 7-joint solution as one
        coordinated ramped move. Same FK-then-IK approach recognition_app_v3.py
        uses (e.g. _lift_end_effector_z_slowly): ArmFK for the seed pose, ArmIK
        (dls solver) for the target — only this arm's 7 joints are touched."""
        if side not in ("left", "right"):
            return False, f"Unknown arm {side!r}.", None
        with self.lock:
            if self.g1 is None:
                return False, "Not connected — click Connect first.", None
            q_init = self._arm_q_locked(side)
            dev_mode = self.dev_mode
            already_engaged = self.arm_engaged

        from hand_pose_navigation.arm_ik import ArmIK
        T_target = np.eye(4, dtype=np.float64)
        T_target[:3, :3] = _rpy_to_R(roll, pitch, yaw)
        T_target[:3, 3] = [x, y, z]
        ik = ArmIK(arm=side, solver="dls", max_iter=EE_IK_MAX_ITER,
                   tol_pos_m=EE_IK_TOL_POS_M, tol_rot_rad=EE_IK_TOL_ROT_RAD)
        q_sol, info = ik.solve(T_target, q_init=q_init)
        if q_sol is None:
            return False, (f"IK did not converge for the {side} arm "
                            f"(pos err {info.get('error_pos_m', 0.0) * 1000:.0f} mm, "
                            f"rot err {math.degrees(info.get('error_rot_rad', 0.0)):.1f}°). "
                            "Target likely unreachable — try a smaller move."), info

        with self.lock:
            if self.g1 is None:
                return False, "Not connected — click Connect first.", None
            if not dev_mode and not already_engaged:
                self._engage_arms()
            now = time.monotonic()
            for jid, q_target in zip(ARM_JOINT_IDS[side], q_sol):
                spec = JOINT_BY_ID[jid]
                prior = self.cmd.get(jid)
                start_q = prior.q if prior is not None else float(self.sensed_q[jid])
                if prior is not None:
                    kp, kd = prior.kp, prior.kd
                elif self.dev_mode:
                    kp, kd = spec.kp_lowcmd, spec.kd_lowcmd
                else:
                    kp, kd = spec.kp_armsdk, spec.kd_armsdk
                self.cmd[jid] = JointCmdState(
                    q=start_q, dq=0.0, kp=kp, kd=kd, tau=0.0,
                    ramp_from_q=start_q, ramp_to_q=_clamp(float(q_target), spec.q_min, spec.q_max),
                    ramp_start_t=now, ramp_duration=EE_IK_RAMP_S,
                )
        return True, (f"{side.title()} hand → xyz=({x:+.3f},{y:+.3f},{z:+.3f}) m "
                       f"rpy=({math.degrees(roll):+.0f},{math.degrees(pitch):+.0f},{math.degrees(yaw):+.0f})° "
                       f"over {EE_IK_RAMP_S:.1f}s — solved in {info['iterations']} iters, "
                       f"pos err {info['error_pos_m'] * 1000:.1f} mm, "
                       f"rot err {math.degrees(info['error_rot_rad']):.1f}°."), info

    # -- snapshots for the UI ---------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            cmd_q = {jid: (st.q if st is not None else None) for jid, st in self.cmd.items()}
            return {
                "sensed_q": list(self.sensed_q),
                "sensed_dq": list(self.sensed_dq),
                "sensed_tau": list(self.sensed_tau),
                "sensed_ts": self.sensed_ts,
                "dev_mode": self.dev_mode,
                "arm_engaged": self.arm_engaged,
                "arm_weight": getattr(self, "weight_now", 0.0),
                "service_row": self.service_row,
                "cmd_q": cmd_q,
                "connected": self.g1 is not None,
            }

    def joint_modal_defaults(self, joint_id: int) -> dict:
        spec = JOINT_BY_ID[int(joint_id)]
        with self.lock:
            state = self.cmd.get(spec.id)
            sensed_q = float(self.sensed_q[spec.id]) if self.sensed_q else 0.0
            if state is not None:
                q, dq, kp, kd, tau = state.ramp_to_q, state.dq, state.kp, state.kd, state.tau
            else:
                q = sensed_q
                dq, tau = 0.0, 0.0
                if self.dev_mode:
                    kp, kd = spec.kp_lowcmd, spec.kd_lowcmd
                elif spec.kp_armsdk is not None:
                    kp, kd = spec.kp_armsdk, spec.kd_armsdk
                else:
                    kp, kd = spec.kp_lowcmd, spec.kd_lowcmd
            locked = spec.group in LEG_GROUPS and not self.dev_mode
        return {"spec": spec, "sensed_q": sensed_q, "q": q, "dq": dq, "kp": kp, "kd": kd,
                "tau": tau, "ramp_s": RAMP_DEFAULT_S, "locked": locked}


ROBOT_LINK = RobotLink(iface="eth0", domain_id=0)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_skeleton_figure(q_values: list[float]) -> go.Figure:
    positions = compute_skeleton(q_values)
    fig = go.Figure()
    for group in GROUP_ORDER:
        specs = JOINTS_BY_GROUP[group]
        xs, ys, zs = [], [], []
        for spec in specs:
            parent_pos = positions["pelvis"] if spec.parent == -1 else positions[spec.parent]
            child_pos = positions[spec.id]
            xs += [parent_pos[0], child_pos[0], None]
            ys += [parent_pos[1], child_pos[1], None]
            zs += [parent_pos[2], child_pos[2], None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line={"color": GROUP_COLORS[group], "width": 6},
            hoverinfo="skip", showlegend=False,
        ))
        mx, my, mz, text, custom = [], [], [], [], []
        for spec in specs:
            p = positions[spec.id]
            mx.append(p[0]); my.append(p[1]); mz.append(p[2])
            text.append(f"{spec.name} (id {spec.id})<br>{math.degrees(q_values[spec.id]):.1f}°")
            custom.append(spec.id)
        fig.add_trace(go.Scatter3d(
            x=mx, y=my, z=mz, mode="markers",
            marker={"size": 6, "color": GROUP_COLORS[group]},
            text=text, hoverinfo="text", customdata=custom,
            name=GROUP_LABELS[group],
        ))
    px, py, pz = positions["pelvis"]
    fig.add_trace(go.Scatter3d(
        x=[px], y=[py], z=[pz], mode="markers",
        marker={"size": 8, "color": PELVIS_COLOR, "symbol": "diamond"},
        name="pelvis", hoverinfo="text", text=["pelvis"],
    ))
    fig.update_layout(
        template="plotly_dark",
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "x", "range": [-0.5, 0.5]},
            "yaxis": {"title": "y", "range": [-0.6, 0.6]},
            "zaxis": {"title": "z", "range": [-0.1, 1.9]},
            "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 0.9}},
        },
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend={"orientation": "h", "y": 0.02},
        height=680,
        uirevision="skeleton",  # keep camera position across live refreshes
    )
    return fig


def empty_skeleton_figure() -> go.Figure:
    return build_skeleton_figure([0.0] * 29)


# ---------------------------------------------------------------------------
# Dash app / layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "G1 Joint Control"


def _joint_row(spec: JointSpec) -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(html.Span(spec.label, className="small"), width=7),
            dbc.Col(html.Span("--", id={"type": "joint-live-q", "index": spec.id}, className="small text-muted"), width=3),
            dbc.Col(
                dbc.Button("⚙", id={"type": "joint-open-btn", "index": spec.id}, size="sm",
                           color="outline-light", className="py-0 px-2"),
                width=2,
            ),
        ],
        className="align-items-center mb-1 g-1",
    )


def _group_card(group: str) -> dbc.Card:
    specs = JOINTS_BY_GROUP[group]
    body = [dbc.CardHeader(html.Span([
        html.Span("● ", style={"color": GROUP_COLORS[group]}), GROUP_LABELS[group],
    ]))]
    body.append(dbc.CardBody([_joint_row(s) for s in specs], className="py-2"))
    return dbc.Card(body, className="mb-2")


JOINT_PANEL = html.Div(
    [
        # Only the leg cards live inside this wrapper — its opacity/pointer-events
        # get toggled by dev-mode-store, since legs are only controllable in dev mode.
        html.Div(
            [_group_card("left_leg"), _group_card("right_leg")],
            id="leg-controls-wrapper",
        ),
        _group_card("waist"),
        _group_card("left_arm"),
        _group_card("right_arm"),
    ],
    style={"maxHeight": "680px", "overflowY": "auto"},
)


def _slider(id_, lo, hi, value, step) -> dcc.Slider:
    return dcc.Slider(id=id_, min=lo, max=hi, value=value, step=step,
                       tooltip={"placement": "bottom", "always_visible": True}, marks=None)


CONTROL_MODAL = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle(id="modal-title", children="Joint")),
        dbc.ModalBody(
            [
                html.Div(id="modal-alert"),
                html.Div(id="modal-readout", className="small text-muted mb-3"),
                dbc.Label("q — target position (rad)"),
                _slider("slider-q", -1.0, 1.0, 0.0, 0.001),
                dbc.Label("dq — target velocity (rad/s)", className="mt-3"),
                _slider("slider-dq", -DQ_CAP, DQ_CAP, 0.0, 0.01),
                dbc.Label("kp — position gain", className="mt-3"),
                _slider("slider-kp", 0.0, 1.0, 0.0, 0.1),
                dbc.Label("kd — velocity gain", className="mt-3"),
                _slider("slider-kd", 0.0, 1.0, 0.0, 0.01),
                dbc.Label("tau — feed-forward torque (N·m)", className="mt-3"),
                _slider("slider-tau", -1.0, 1.0, 0.0, 0.01),
                dbc.Label("Safety ramp duration (s)", className="mt-3"),
                _slider("slider-ramp", RAMP_MIN_S, RAMP_MAX_S, RAMP_DEFAULT_S, 0.05),
            ]
        ),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="modal-cancel-btn", color="secondary"),
            dbc.Button("Apply", id="modal-apply-btn", color="primary"),
        ]),
    ],
    id="joint-modal", is_open=False, size="lg",
)

DEV_MODE_CONFIRM_MODAL = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Enable Dev Mode?")),
        dbc.ModalBody(
            "This stops the ai_sport service and switches to raw rt/lowcmd for all "
            "29 joints, including the legs. There is no balance controller running "
            "under rt/lowcmd — only do this with the robot supported (e.g. on a "
            "stand) or otherwise safe to go limp/stiff on command."
        ),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="devmode-cancel-btn", color="secondary"),
            dbc.Button("Enable Dev Mode", id="devmode-confirm-btn", color="danger"),
        ]),
    ],
    id="devmode-confirm-modal", is_open=False,
)

EE_IK_MODAL = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("IK End-Effector Controller")),
        dbc.ModalBody(
            [
                dbc.RadioItems(
                    id="ee-arm-select",
                    options=[{"label": "Left hand", "value": "left"}, {"label": "Right hand", "value": "right"}],
                    value="right", inline=True, className="mb-2",
                ),
                html.Div(id="ee-ik-alert"),
                html.Div(id="ee-ik-readout", className="small text-muted mb-3"),
                dbc.Label("x (m)"),
                _slider("ee-slider-x", -1.0, 1.0, 0.0, 0.005),
                dbc.Label("y (m)", className="mt-3"),
                _slider("ee-slider-y", -1.0, 1.0, 0.0, 0.005),
                dbc.Label("z (m)", className="mt-3"),
                _slider("ee-slider-z", -1.0, 1.0, 0.0, 0.005),
                dbc.Label("roll (rad)", className="mt-3"),
                _slider("ee-slider-roll", -EE_ROT_RANGE_RAD, EE_ROT_RANGE_RAD, 0.0, 0.01),
                dbc.Label("pitch (rad)", className="mt-3"),
                _slider("ee-slider-pitch", -EE_ROT_RANGE_RAD, EE_ROT_RANGE_RAD, 0.0, 0.01),
                dbc.Label("yaw (rad)", className="mt-3"),
                _slider("ee-slider-yaw", -EE_ROT_RANGE_RAD, EE_ROT_RANGE_RAD, 0.0, 0.01),
            ]
        ),
        dbc.ModalFooter([
            dbc.Button("Sync to Current", id="ee-ik-sync-btn", color="secondary"),
            dbc.Button("Close", id="ee-ik-cancel-btn", color="secondary"),
            dbc.Button("Apply", id="ee-ik-apply-btn", color="primary"),
        ]),
    ],
    id="ee-ik-modal", is_open=False, size="lg",
)

app.layout = dbc.Container(
    [
        html.H3("G1 Joint Control Dashboard", className="mt-3 mb-3"),
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col([
                            html.Span("Status: ", className="fw-bold me-1"),
                            dbc.Badge("Disconnected", id="conn-badge", color="secondary"),
                        ], md=3, className="d-flex align-items-center"),
                        dbc.Col(dbc.Input(id="iface-input", value="eth0", placeholder="iface"), md=2),
                        dbc.Col(dbc.Input(id="domain-input", value="0", type="number", placeholder="domain id"), md=2),
                        dbc.Col(dbc.Button("Connect", id="btn-connect", color="primary", className="w-100"), md=2),
                        dbc.Col(dbc.Button("Disconnect", id="btn-disconnect", color="secondary", className="w-100"), md=2),
                    ],
                    className="g-2",
                )
            ),
            className="mb-3",
        ),
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col([
                            dbc.Button("Dev Mode: OFF", id="btn-dev-mode", color="secondary", className="w-100"),
                        ], md=3),
                        dbc.Col(html.Div(id="dev-mode-caption", className="small text-muted"), md=5),
                        dbc.Col(
                            dbc.Button("Release Arms → ai_sport", id="btn-release-arms", color="warning",
                                       className="w-100", disabled=True),
                            md=4,
                        ),
                    ],
                    className="align-items-center g-2",
                )
            ),
            className="mb-3",
        ),
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button("🦾 IK End-Effector Controller", id="btn-ee-ik", color="info",
                                       className="w-100"),
                            md=3,
                        ),
                        dbc.Col(
                            html.Div(
                                "Drive one arm's hand pose (x, y, z, roll, pitch, yaw) via inverse "
                                "kinematics — solves and moves that arm's 7 joints together.",
                                className="small text-muted",
                            ),
                            md=9,
                        ),
                    ],
                    className="align-items-center g-2",
                )
            ),
            className="mb-3",
        ),
        html.Div(id="action-toast", className="mb-2"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="skeleton-graph", figure=empty_skeleton_figure()), md=7),
                dbc.Col(JOINT_PANEL, md=5),
            ]
        ),
        CONTROL_MODAL,
        DEV_MODE_CONFIRM_MODAL,
        EE_IK_MODAL,
        dcc.Store(id="selected-joint-store"),
        dcc.Store(id="dev-mode-store", data=False),
        dcc.Interval(id="plot-interval", interval=PLOT_INTERVAL_MS, n_intervals=0),
        dcc.Interval(id="status-interval", interval=STATUS_INTERVAL_MS, n_intervals=0),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

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
    trig = ctx.triggered_id
    if trig == "btn-connect":
        ROBOT_LINK.iface = str(iface or "eth0")
        try:
            ROBOT_LINK.domain_id = int(domain_id or 0)
        except (TypeError, ValueError):
            ROBOT_LINK.domain_id = 0
        ROBOT_LINK.connect()
    elif trig == "btn-disconnect":
        ROBOT_LINK.disconnect()
    label, color = ROBOT_LINK.status()
    return label, color


@app.callback(
    Output("skeleton-graph", "figure"),
    Output("dev-mode-store", "data"),
    Output("btn-dev-mode", "children"),
    Output("btn-dev-mode", "color"),
    Output("dev-mode-caption", "children"),
    Output("btn-release-arms", "disabled"),
    Output({"type": "joint-live-q", "index": ALL}, "children"),
    Input("plot-interval", "n_intervals"),
    State({"type": "joint-live-q", "index": ALL}, "id"),
)
def on_plot_tick(_n, live_ids):
    snap = ROBOT_LINK.snapshot()
    fig = build_skeleton_figure(snap["sensed_q"])
    dev_mode = bool(snap["dev_mode"])
    dev_label = "Dev Mode: ON (rt/lowcmd)" if dev_mode else "Dev Mode: OFF (rt/arm_sdk)"
    dev_color = "danger" if dev_mode else "secondary"
    row = snap.get("service_row")
    raw = f" | ai_sport raw status: {row.get('status')}" if row else ""
    if dev_mode:
        caption = "All 29 joints on rt/lowcmd — ai_sport stopped." + raw
    else:
        caption = f"Waist + arms on rt/arm_sdk (weight {snap['arm_weight']:.2f}); legs stay with ai_sport." + raw
    release_disabled = not (dev_mode is False and (snap["arm_engaged"] or snap["arm_weight"] > 1e-3))
    live_values = [f"{snap['sensed_q'][item['index']]:.3f}" for item in live_ids]
    return fig, dev_mode, dev_label, dev_color, caption, release_disabled, live_values


@app.callback(
    Output("leg-controls-wrapper", "style"),
    Input("dev-mode-store", "data"),
)
def on_dev_mode_style(dev_mode):
    return {} if dev_mode else {"opacity": "0.55", "pointerEvents": "none"}


@app.callback(
    Output("devmode-confirm-modal", "is_open"),
    Input("btn-dev-mode", "n_clicks"),
    Input("devmode-cancel-btn", "n_clicks"),
    Input("devmode-confirm-btn", "n_clicks"),
    State("dev-mode-store", "data"),
    prevent_initial_call=True,
)
def on_dev_mode_modal(_open, _cancel, _confirm, dev_mode):
    trig = ctx.triggered_id
    if trig == "btn-dev-mode":
        return not dev_mode  # confirm only when turning ON; turning off is one click
    return False


@app.callback(
    Output("action-toast", "children"),
    Input("devmode-confirm-btn", "n_clicks"),
    Input("btn-dev-mode", "n_clicks"),
    Input("btn-release-arms", "n_clicks"),
    State("dev-mode-store", "data"),
    prevent_initial_call=True,
)
def on_dev_and_release_actions(_confirm, _toggle, _release, dev_mode):
    trig = ctx.triggered_id
    if trig == "devmode-confirm-btn" or (trig == "btn-dev-mode" and dev_mode):
        # Turning ON goes through the confirm modal; turning OFF (dev_mode
        # already True) doesn't need confirmation.
        ok, message = ROBOT_LINK.toggle_dev_mode()
        color = "success" if ok else "danger"
        return dbc.Alert(message, color=color, dismissable=True, duration=6000)
    if trig == "btn-release-arms":
        ok, message = ROBOT_LINK.release_arms()
        color = "success" if ok else "warning"
        return dbc.Alert(message, color=color, dismissable=True, duration=6000)
    return dash.no_update


@app.callback(
    Output("joint-modal", "is_open"),
    Output("modal-title", "children"),
    Output("modal-readout", "children"),
    Output("modal-alert", "children"),
    Output("selected-joint-store", "data"),
    Output("slider-q", "min"), Output("slider-q", "max"), Output("slider-q", "value"),
    Output("slider-dq", "min"), Output("slider-dq", "max"), Output("slider-dq", "value"),
    Output("slider-kp", "min"), Output("slider-kp", "max"), Output("slider-kp", "value"),
    Output("slider-kd", "min"), Output("slider-kd", "max"), Output("slider-kd", "value"),
    Output("slider-tau", "min"), Output("slider-tau", "max"), Output("slider-tau", "value"),
    Output("slider-ramp", "value"),
    Input({"type": "joint-open-btn", "index": ALL}, "n_clicks"),
    Input("skeleton-graph", "clickData"),
    Input("modal-cancel-btn", "n_clicks"),
    Input("modal-apply-btn", "n_clicks"),
    State("selected-joint-store", "data"),
    State("slider-q", "value"), State("slider-dq", "value"),
    State("slider-kp", "value"), State("slider-kd", "value"),
    State("slider-tau", "value"), State("slider-ramp", "value"),
    prevent_initial_call=True,
)
def on_joint_modal(_opens, click_data, _cancel, _apply, selected_joint,
                    slider_q, slider_dq, slider_kp, slider_kd, slider_tau, slider_ramp):
    trig = ctx.triggered_id
    no = dash.no_update

    if trig == "modal-cancel-btn":
        return (False,) + (no,) * 18

    if trig == "modal-apply-btn":
        if selected_joint is not None:
            ROBOT_LINK.set_joint_target(int(selected_joint), slider_q, slider_dq, slider_kp,
                                         slider_kd, slider_tau, slider_ramp)
        return (False,) + (no,) * 18

    joint_id = None
    if isinstance(trig, dict) and trig.get("type") == "joint-open-btn":
        joint_id = trig.get("index")
    elif trig == "skeleton-graph" and click_data:
        points = click_data.get("points") or []
        if points and points[0].get("customdata") is not None:
            joint_id = points[0]["customdata"]

    if joint_id is None:
        return (no,) * 19

    defaults = ROBOT_LINK.joint_modal_defaults(joint_id)
    spec: JointSpec = defaults["spec"]
    title = f"{spec.label} — {GROUP_LABELS[spec.group]}"
    readout = (f"Sensed q: {defaults['sensed_q']:.3f} rad ({math.degrees(defaults['sensed_q']):.1f}°) | "
               f"limits [{spec.q_min:.3f}, {spec.q_max:.3f}] rad | tau limit ±{spec.tau_max:.0f} N·m")
    alert = (dbc.Alert("Leg joints are locked outside Dev Mode. Enable Dev Mode to control this joint.",
                        color="warning", className="mb-2")
             if defaults["locked"] else "")

    return (
        True, title, readout, alert, spec.id,
        spec.q_min, spec.q_max, defaults["q"],
        -DQ_CAP, DQ_CAP, defaults["dq"],
        0.0, spec.kp_max, defaults["kp"],
        0.0, spec.kd_max, defaults["kd"],
        -spec.tau_max, spec.tau_max, defaults["tau"],
        defaults["ramp_s"],
    )


@app.callback(
    Output("modal-apply-btn", "disabled"),
    Input("selected-joint-store", "data"),
    Input("dev-mode-store", "data"),
)
def on_apply_lock(selected_joint, dev_mode):
    if selected_joint is None:
        return True
    spec = JOINT_BY_ID.get(int(selected_joint))
    if spec is None:
        return True
    return spec.group in LEG_GROUPS and not dev_mode


def _ee_readout(side: str, pose: dict) -> str:
    return (f"{side.title()} hand (base_link frame) — "
            f"x={pose['x']:+.3f} y={pose['y']:+.3f} z={pose['z']:+.3f} m | "
            f"roll={math.degrees(pose['roll']):+.1f}° pitch={math.degrees(pose['pitch']):+.1f}° "
            f"yaw={math.degrees(pose['yaw']):+.1f}°")


@app.callback(
    Output("ee-ik-modal", "is_open"),
    Output("ee-ik-readout", "children"),
    Output("ee-ik-alert", "children"),
    Output("ee-slider-x", "min"), Output("ee-slider-x", "max"), Output("ee-slider-x", "value"),
    Output("ee-slider-y", "min"), Output("ee-slider-y", "max"), Output("ee-slider-y", "value"),
    Output("ee-slider-z", "min"), Output("ee-slider-z", "max"), Output("ee-slider-z", "value"),
    Output("ee-slider-roll", "value"),
    Output("ee-slider-pitch", "value"),
    Output("ee-slider-yaw", "value"),
    Input("btn-ee-ik", "n_clicks"),
    Input("ee-arm-select", "value"),
    Input("ee-ik-sync-btn", "n_clicks"),
    Input("ee-ik-apply-btn", "n_clicks"),
    Input("ee-ik-cancel-btn", "n_clicks"),
    State("ee-slider-x", "value"), State("ee-slider-y", "value"), State("ee-slider-z", "value"),
    State("ee-slider-roll", "value"), State("ee-slider-pitch", "value"), State("ee-slider-yaw", "value"),
    prevent_initial_call=True,
)
def on_ee_ik_modal(_open, side, _sync, _apply, _cancel, sx, sy, sz, sroll, spitch, syaw):
    trig = ctx.triggered_id
    no = dash.no_update
    side = side or "right"

    if trig == "ee-ik-cancel-btn":
        return (False,) + (no,) * 14

    if trig == "ee-ik-apply-btn":
        if ROBOT_LINK.g1 is None:
            alert = dbc.Alert("Not connected — click Connect first.", color="danger",
                               dismissable=True, className="mb-2")
            return (no, no, alert) + (no,) * 12
        ok, message, _info = ROBOT_LINK.set_arm_ee_target(side, sx, sy, sz, sroll, spitch, syaw)
        color = "success" if ok else "danger"
        alert = dbc.Alert(message, color=color, dismissable=True, className="mb-2")
        return (no, no, alert) + (no,) * 12  # leave sliders as the operator set them

    # Open / arm-switch / sync-to-current: (re)populate from the live pose.
    pose = ROBOT_LINK.ee_pose_snapshot(side)
    readout = _ee_readout(side, pose)
    return (
        True, readout, "",
        pose["x"] - EE_POS_RANGE_M, pose["x"] + EE_POS_RANGE_M, pose["x"],
        pose["y"] - EE_POS_RANGE_M, pose["y"] + EE_POS_RANGE_M, pose["y"],
        pose["z"] - EE_POS_RANGE_M, pose["z"] + EE_POS_RANGE_M, pose["z"],
        pose["roll"], pose["pitch"], pose["yaw"],
    )


# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="G1 joint control dashboard.")
    parser.add_argument("--iface", default="eth0", help="DDS network interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    ROBOT_LINK.iface = args.iface
    ROBOT_LINK.domain_id = args.domain_id

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
