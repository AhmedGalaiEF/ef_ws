#!/usr/bin/env python3
"""
G1 ZMP / support-polygon estimator and live plot.

Estimates the Zero Moment Point and the double-support polygon purely from
geometry + odometry/IMU — no force/pressure foot sensors are read here (the
SDK doesn't expose them), so this is a coarse, openly-approximate estimate,
in the same spirit as segmentation.py's PCA grasp-surface estimate: good
enough to see whether the robot is comfortably balanced, not a certified
stability computation.

Support polygon
----------------
Forward-kinematics only the 12 leg joints (`get_lowstate()["joint_positions"]`
indices 0-11, same MJCF-sourced geometry `joint_control_dashboard.py` uses for
its full-body skeleton) to get each foot's ankle-roll frame in world/base
coordinates, plant a small fixed rectangle at each foot (configurable
half-length/half-width — the SDK doesn't hand us the real foot outline, so
this is an adjustable placeholder, not a spec), and take the convex hull of
both feet's 4 corners each. Always drawn as double support: there is no foot
contact/force sensing available here to detect single-support phases.

ZMP
----
The standard linear-inverted-pendulum (cart-table) approximation:

    ZMP_xy = CoM_xy − (h_com / g) · CoM_accel_xy

CoM_xy is approximated by the pelvis/base position from `get_odom()` (no
per-link mass data is available to compute a true whole-body CoM), h_com is a
configurable assumed CoM height (default 0.793 m — the same fixed pelvis
height `joint_control_dashboard.py` uses), and CoM_accel_xy is the IMU's
measured linear acceleration (`get_imus()["acc"]`, gravity-compensated and
rotated from body into world frame via the IMU roll/pitch/yaw), smoothed over
a short moving-average window since raw accelerometer noise would otherwise
dominate the estimate.

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
# Leg geometry — the 12 leg entries of joint_control_dashboard.py's
# JOINT_TABLE (ids 0-5 left leg, 6-11 right leg), verbatim: (parent, pos,
# quat, axis). Not imported from that file — it's a Dash app with page-load
# side effects (builds its own `app`/layout at import time), so the handful
# of constants this script needs are duplicated here instead.
# ---------------------------------------------------------------------------

PELVIS_POS = (0.0, 0.0, 0.793)
_IDENT_Q = (1.0, 0.0, 0.0, 0.0)
_HIP_ROLL_Q = (0.996179, 0.0, -0.0873386, 0.0)
_KNEE_Q = (0.996179, 0.0, 0.0873386, 0.0)

# id: (parent, pos, quat, axis)
LEG_JOINTS: dict[int, tuple[int, tuple, tuple, tuple]] = {
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
}
LEFT_FOOT_ID, RIGHT_FOOT_ID = 5, 11


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


def foot_frames(leg_q: list[float]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """leg_q: 12 sensed leg joint angles (ids 0-11). Returns
    {joint_id: (world_R, world_t)} for every leg joint, pelvis-relative
    (pelvis itself sits at PELVIS_POS in whatever frame `leg_q` is silent
    about — i.e. this is base/pelvis-relative geometry, not world/odom; the
    odom pose is applied separately to place it in the world).
    """
    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    root_R = np.eye(3, dtype=np.float64)
    root_t = np.array(PELVIS_POS, dtype=np.float64)
    for jid in range(12):
        parent, pos, quat, axis = LEG_JOINTS[jid]
        parent_R, parent_t = (root_R, root_t) if parent == -1 else frames[parent]
        world_t = parent_t + parent_R @ np.array(pos, dtype=np.float64)
        world_R = parent_R @ _quat_to_R(quat) @ _axis_R(axis, float(leg_q[jid]))
        frames[jid] = (world_R, world_t)
    return frames


def foot_corners_base(world_R: np.ndarray, world_t: np.ndarray,
                       half_length_m: float, half_width_m: float) -> np.ndarray:
    """4 foot corners (base/pelvis frame, xy only), from an ankle-roll frame."""
    local = np.array([
        [half_length_m, half_width_m, 0.0],
        [half_length_m, -half_width_m, 0.0],
        [-half_length_m, -half_width_m, 0.0],
        [-half_length_m, half_width_m, 0.0],
    ], dtype=np.float64)
    return (world_t + local @ world_R.T)[:, :2]


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
ZMP_TRAIL_S = 5.0
PLOT_INTERVAL_MS = 400

DEFAULT_COM_HEIGHT_M = PELVIS_POS[2]
DEFAULT_FOOT_HALF_LENGTH_M = 0.10
DEFAULT_FOOT_HALF_WIDTH_M = 0.045


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
        self.left_corners: Optional[np.ndarray] = None
        self.right_corners: Optional[np.ndarray] = None
        self.zmp_xy: Optional[tuple[float, float]] = None
        self.ts = 0.0
        self.zmp_trail: deque = deque(maxlen=int(ZMP_TRAIL_S * POLL_HZ) + 4)
        self._accel_hist: deque = deque(maxlen=ACCEL_SMOOTH_SAMPLES)

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
                    self._poll_once(g1, com_height_m=self._com_height_m,
                                     half_length_m=self._foot_half_length_m,
                                     half_width_m=self._foot_half_width_m)
                    with self.lock:
                        self.poll_err = None
                except Exception as exc:
                    with self.lock:
                        self.poll_err = str(exc)
            time.sleep(period)

    # Configurable from the UI; read by the poll loop each tick.
    _com_height_m = DEFAULT_COM_HEIGHT_M
    _foot_half_length_m = DEFAULT_FOOT_HALF_LENGTH_M
    _foot_half_width_m = DEFAULT_FOOT_HALF_WIDTH_M

    def _poll_once(self, g1, com_height_m: float, half_length_m: float, half_width_m: float) -> None:
        lowstate = g1.get_lowstate()
        odom = g1.get_odom()
        imu = g1.get_imus()

        com_xy = _extract_xy(odom)
        leg_q = None
        if lowstate is not None:
            positions = lowstate.get("joint_positions") or []
            if len(positions) >= 12:
                leg_q = positions[:12]

        left_corners = right_corners = None
        if leg_q is not None:
            frames = foot_frames(leg_q)
            lR, lt = frames[LEFT_FOOT_ID]
            rR, rt = frames[RIGHT_FOOT_ID]
            left_corners = foot_corners_base(lR, lt, half_length_m, half_width_m)
            right_corners = foot_corners_base(rR, rt, half_length_m, half_width_m)
            if com_xy is not None:
                # Feet were computed pelvis-relative; shift into the same
                # world/odom frame the CoM (pelvis) position is reported in.
                # (Rotation by odom yaw is intentionally skipped — get_odom()
                # doesn't reliably expose yaw across every fallback source; the
                # support polygon is drawn pelvis-forward-aligned, a small
                # extra approximation on top of the ones already documented
                # above.)
                shift = np.array(com_xy, dtype=np.float64)
                left_corners = left_corners + shift
                right_corners = right_corners + shift

        ax_world = ay_world = 0.0
        if imu is not None and imu.get("acc") is not None and imu.get("rpy") is not None:
            roll, pitch, yaw = imu["rpy"]
            R = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
            a_body = np.array(imu["acc"], dtype=np.float64)
            a_world = R @ a_body - np.array([0.0, 0.0, G_ACCEL])
            with self.lock:
                self._accel_hist.append((float(a_world[0]), float(a_world[1])))
                ax_world = sum(v[0] for v in self._accel_hist) / len(self._accel_hist)
                ay_world = sum(v[1] for v in self._accel_hist) / len(self._accel_hist)

        zmp_xy = None
        if com_xy is not None:
            zmp_xy = (
                com_xy[0] - (com_height_m / G_ACCEL) * ax_world,
                com_xy[1] - (com_height_m / G_ACCEL) * ay_world,
            )

        now = time.time()
        with self.lock:
            self.com_xy = com_xy
            self.left_corners = left_corners
            self.right_corners = right_corners
            self.zmp_xy = zmp_xy
            self.ts = now
            if zmp_xy is not None:
                self.zmp_trail.append((now, zmp_xy[0], zmp_xy[1]))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure(link: "ZmpLink") -> tuple[go.Figure, str, str]:
    with link.lock:
        com_xy = link.com_xy
        left_corners = None if link.left_corners is None else link.left_corners.copy()
        right_corners = None if link.right_corners is None else link.right_corners.copy()
        zmp_xy = link.zmp_xy
        trail = list(link.zmp_trail)

    fig = go.Figure()
    extent = []

    hull = None
    if left_corners is not None and right_corners is not None:
        both = np.vstack([left_corners, right_corners])
        hull = convex_hull(both)
        if len(hull) >= 3:
            hx = list(hull[:, 0]) + [hull[0, 0]]
            hy = list(hull[:, 1]) + [hull[0, 1]]
            fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines", fill="toself",
                                      fillcolor="rgba(85,199,255,0.18)",
                                      line={"color": "#55c7ff", "width": 2},
                                      name="support polygon"))
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

    status_text = "no data"
    status_color = "#888888"
    if com_xy is not None:
        fig.add_trace(go.Scatter(x=[com_xy[0]], y=[com_xy[1]], mode="markers",
                                  marker={"size": 12, "color": "#ffae00", "symbol": "circle"},
                                  name="CoM (≈ pelvis)"))
        extent.append(com_xy)

    if zmp_xy is not None:
        inside = hull is not None and len(hull) >= 3 and point_in_polygon(zmp_xy, hull)
        margin = signed_margin(zmp_xy, hull) if hull is not None and len(hull) >= 3 else None
        zmp_color = "#3aa876" if inside else "#e0294f"
        fig.add_trace(go.Scatter(x=[zmp_xy[0]], y=[zmp_xy[1]], mode="markers",
                                  marker={"size": 16, "color": zmp_color, "symbol": "diamond"},
                                  name="ZMP"))
        extent.append(zmp_xy)
        if margin is not None:
            status_text = f"ZMP {'INSIDE' if inside else 'OUTSIDE'} support polygon — margin {margin * 100:+.1f} cm"
        else:
            status_text = "ZMP computed, no support polygon (no leg data)"
        status_color = zmp_color

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
    return fig, status_text, status_color


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
        dbc.Col([html.Label("Assumed CoM height (m)", style={"fontSize": "12px"}),
                 dcc.Input(id="com-height-input", type="number", value=DEFAULT_COM_HEIGHT_M,
                           min=0.3, max=1.2, step=0.01, style={"width": "100%"})], width=2),
        dbc.Col([html.Label("Foot half-length (m)", style={"fontSize": "12px"}),
                 dcc.Input(id="foot-hl-input", type="number", value=DEFAULT_FOOT_HALF_LENGTH_M,
                           min=0.02, max=0.20, step=0.005, style={"width": "100%"})], width=2),
        dbc.Col([html.Label("Foot half-width (m)", style={"fontSize": "12px"}),
                 dcc.Input(id="foot-hw-input", type="number", value=DEFAULT_FOOT_HALF_WIDTH_M,
                           min=0.01, max=0.10, step=0.005, style={"width": "100%"})], width=2),
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
    Input("foot-hl-input", "value"),
    Input("foot-hw-input", "value"),
)
def on_plot_tick(_n, com_height_m, foot_hl, foot_hw):
    LINK._com_height_m = float(com_height_m or DEFAULT_COM_HEIGHT_M)
    LINK._foot_half_length_m = float(foot_hl or DEFAULT_FOOT_HALF_LENGTH_M)
    LINK._foot_half_width_m = float(foot_hw or DEFAULT_FOOT_HALF_WIDTH_M)
    fig, status_text, status_color = build_figure(LINK)
    with LINK.lock:
        err = LINK.poll_err
    return fig, html.Span(status_text, style={"color": status_color, "fontWeight": 700}), (f"poll error: {err}" if err else "")


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
