#!/usr/bin/env python3
"""
G1 raw state monitor.

Plain tables of the robot's raw state — no plots, just the numbers behind
`sdk_wrapper.G1`'s `get_*()` dictionary readers (lowstate, odometry, battery/
BMS, services, Dex3 hands, FSM/mode, SLAM info), each with a freshness badge
so a stale table reads as stale rather than as confidently-wrong data — the
same "defensive reads, freshness window" pattern the course's Task 4
(robot-state observation) teaches for `get_battery()`'s BMS topics.

Panels:
  * Lowstate  — one row per of the 29 body joints (q/dq/tau_est) + IMU.
  * Odometry  — position/velocity/mode/gait, whichever odom source responded.
  * Battery / BMS — power rail + full BMS block (SoC/SoH/cycle/cell
    voltages/temperatures), from `get_battery()`'s dedicated-BMS-topic-first,
    lowstate-fallback logic.
  * Dex3 hands — per-joint q/dq/tau + a tactile-pressure summary per pad.
  * Services — `get_service()`'s full list with description/status/protected.
  * FSM / mode — `get_state()`'s id/mode/motion_mode/gait.
  * SLAM info — the raw `/slam_info` (or `/slam_key_info`) string, pretty-
    printed if it parses as JSON.

Run:
    python3 state_monitor.py [--iface eth0] [--domain-id 0] [--port 8073]
Then open http://<host>:8073 in a browser on the same network. The page loads
even with no robot reachable; click Connect once the network is up.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, dash_table

# Make `from sdk_wrapper import G1` work regardless of cwd. This file lives in
# academy/visualizations/; sdk_wrapper.py sits one level up in academy/.
_SCRIPT_DIR = Path(__file__).resolve().parent      # academy/visualizations
_ACADEMY_DIR = _SCRIPT_DIR.parent                    # academy
_G1_ROOT_DIR = _ACADEMY_DIR.parent                   # g1
for _p in (_SCRIPT_DIR, _ACADEMY_DIR, _G1_ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Presentation-only labels for the 29 LOWCMD_JOINTS body joints, same order
# joint_control_dashboard.py's JOINT_TABLE uses (legs 0-11, waist 12-14,
# arms 15-28) — just the names, not the geometry this script doesn't need.
JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

POLL_PERIOD_S = 2.0          # get_state() alone makes several RPC/DDS reads — no need to hammer it
TABLE_REFRESH_MS = 1000
FRESH_S, STALE_S = 3.0, 10.0  # matches get_battery()'s own 3s BMS freshness window


# ---------------------------------------------------------------------------
# Robot connection + poll loop
# ---------------------------------------------------------------------------

class StateLink:
    def __init__(self, iface: str, domain_id: int):
        self.iface = str(iface)
        self.domain_id = int(domain_id)
        self.lock = threading.RLock()

        self.g1 = None
        self.connect_requested = False
        self.init_err: Optional[str] = None
        self.poll_err: Optional[str] = None

        self.state: Optional[dict] = None      # get_state(): id/mode/gait/battery/lowstate/services/slam_info
        self.odom: Optional[dict] = None
        self.hands: Optional[dict] = None       # get_dex3_hand_sensors(hand="both")
        self.ts = 0.0

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
        while not self._stop.is_set():
            g1 = self.g1
            if g1 is not None:
                try:
                    state = g1.get_state()
                    odom = g1.get_odom()
                    hands = g1.get_dex3_hand_sensors(hand="both")
                    with self.lock:
                        self.state, self.odom, self.hands = state, odom, hands
                        self.ts = time.time()
                        self.poll_err = None
                except Exception as exc:
                    with self.lock:
                        self.poll_err = str(exc)
            time.sleep(POLL_PERIOD_S)


# ---------------------------------------------------------------------------
# Table builders — one function per panel, each returning (rows, columns).
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            return f"[{len(value)} values] " + ", ".join(_fmt(v) for v in value[:8]) + ", …"
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def _cols(*names: str) -> list[dict]:
    return [{"name": n, "id": n} for n in names]


def lowstate_rows(lowstate: Optional[dict]) -> list[dict]:
    if not lowstate:
        return []
    q = lowstate.get("joint_positions") or []
    dq = lowstate.get("joint_velocities") or []
    tau = lowstate.get("joint_torques") or []
    n = max(len(q), len(dq), len(tau))
    rows = []
    for i in range(n):
        rows.append({
            "id": i,
            "name": JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"joint_{i}",
            "q (rad)": _fmt(q[i]) if i < len(q) else "—",
            "dq (rad/s)": _fmt(dq[i]) if i < len(dq) else "—",
            "tau_est (N·m)": _fmt(tau[i]) if i < len(tau) else "—",
        })
    return rows


def imu_rows(imu: Optional[dict]) -> list[dict]:
    if not imu:
        return []
    return [
        {"field": "rpy (roll, pitch, yaw)", "value": _fmt(imu.get("rpy"))},
        {"field": "gyro (rad/s)", "value": _fmt(imu.get("gyro"))},
        {"field": "acc (m/s²)", "value": _fmt(imu.get("acc"))},
    ]


def kv_rows(d: Optional[dict], skip: tuple[str, ...] = ("raw",)) -> list[dict]:
    if not d:
        return []
    rows = []
    for key, value in d.items():
        if key in skip:
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                rows.append({"field": f"{key}.{sub_key}", "value": _fmt(sub_value)})
        else:
            rows.append({"field": key, "value": _fmt(value)})
    return rows


def services_rows(services: Any) -> list[dict]:
    if not services:
        return []
    rows = list(services) if isinstance(services, list) else [services]
    return [
        {
            "name": r.get("name"),
            "description": r.get("description"),
            "status": _fmt(r.get("status")),
            "protected": _fmt(r.get("protected")),
        }
        for r in rows
    ]


def hand_rows(hands: Optional[dict]) -> list[dict]:
    if not hands:
        return []
    rows = []
    for side in ("left", "right"):
        snap = hands.get(side)
        if not snap or "positions" not in snap:
            continue
        for joint_name, q in snap.get("positions", {}).items():
            rows.append({
                "hand": side,
                "joint": joint_name,
                "q (rad)": _fmt(q),
                "dq (rad/s)": _fmt(snap.get("velocities", {}).get(joint_name)),
                "tau_est (N·m)": _fmt(snap.get("torques", {}).get(joint_name)),
            })
    return rows


def tactile_summary_rows(hands: Optional[dict]) -> list[dict]:
    if not hands:
        return []
    rows = []
    for side in ("left", "right"):
        snap = hands.get(side)
        pads = (snap or {}).get("tactile_pressures") or []
        for pad_idx, taxels in enumerate(pads):
            valid = [float(v) for v in taxels if abs(float(v) - 30000.0) >= 0.5]
            rows.append({
                "hand": side,
                "pad": pad_idx,
                "taxel count": len(taxels),
                "max pressure (raw)": _fmt(max(valid)) if valid else "—",
                "valid taxels": f"{len(valid)}/{len(taxels)}",
            })
    return rows


def _age_badge(ts: float) -> dbc.Badge:
    if not ts:
        return dbc.Badge("no data", color="secondary")
    age = time.time() - ts
    if age < FRESH_S:
        return dbc.Badge(f"fresh ({age:.1f}s)", color="success")
    if age < STALE_S:
        return dbc.Badge(f"aging ({age:.1f}s)", color="warning")
    return dbc.Badge(f"stale ({age:.1f}s)", color="danger")


# ---------------------------------------------------------------------------
# Dash app / layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "G1 State Monitor"

_TABLE_STYLE = dict(
    style_header={"backgroundColor": "#1c1a1e", "color": "#e8ecf3", "fontSize": "11px",
                  "textTransform": "uppercase"},
    style_cell={"backgroundColor": "#14171d", "color": "#e8ecf3", "fontSize": "12.5px",
                "border": "1px solid #2b303b", "padding": "4px 8px"},
    style_table={"maxHeight": "360px", "overflowY": "auto"},
    sort_action="native",
    page_size=30,
)


def _panel(title: str, table_id: str, columns: list[dict]) -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.Div([
            html.H5(title, style={"display": "inline-block", "marginRight": "10px"}),
            html.Span(id=f"{table_id}-age"),
        ]),
        dash_table.DataTable(id=table_id, columns=columns, data=[], **_TABLE_STYLE),
    ]), className="mb-3")


app.layout = dbc.Container([
    html.H3("G1 State Monitor", className="mt-3"),
    dbc.Row([
        dbc.Col(dbc.Input(id="iface-input", value="eth0", placeholder="iface"), width="auto"),
        dbc.Col(dbc.Input(id="domain-input", type="number", value=0, placeholder="domain id"), width="auto"),
        dbc.Col(dbc.Button("Connect", id="btn-connect", color="primary"), width="auto"),
        dbc.Col(dbc.Button("Disconnect", id="btn-disconnect", color="secondary"), width="auto"),
        dbc.Col(dbc.Badge("Disconnected", id="conn-badge", color="secondary"), width="auto"),
    ], align="center", className="mb-3 gy-2"),
    html.Div(id="poll-err", className="mb-2", style={"fontSize": "12px", "color": "#e0294f"}),

    dbc.Row([
        dbc.Col(_panel("Lowstate — joints", "table-joints",
                       _cols("id", "name", "q (rad)", "dq (rad/s)", "tau_est (N·m)")), md=8),
        dbc.Col(_panel("Lowstate — IMU", "table-imu", _cols("field", "value")), md=4),
    ]),
    dbc.Row([
        dbc.Col(_panel("Odometry", "table-odom", _cols("field", "value")), md=6),
        dbc.Col(_panel("FSM / mode", "table-fsm", _cols("field", "value")), md=6),
    ]),
    dbc.Row([
        dbc.Col(_panel("Battery / BMS", "table-battery", _cols("field", "value")), md=6),
        dbc.Col(_panel("Services", "table-services",
                       _cols("name", "description", "status", "protected")), md=6),
    ]),
    dbc.Row([
        dbc.Col(_panel("Dex3 hands — joints", "table-hand-joints",
                       _cols("hand", "joint", "q (rad)", "dq (rad/s)", "tau_est (N·m)")), md=7),
        dbc.Col(_panel("Dex3 hands — tactile summary", "table-hand-tactile",
                       _cols("hand", "pad", "taxel count", "max pressure (raw)", "valid taxels")), md=5),
    ]),
    dbc.Card(dbc.CardBody([
        html.Div([html.H5("SLAM info", style={"display": "inline-block", "marginRight": "10px"}),
                  html.Span(id="slam-age")]),
        html.Pre(id="slam-text", style={"maxHeight": "200px", "overflow": "auto",
                                         "background": "#0f1218", "padding": "8px",
                                         "fontSize": "12px", "marginBottom": 0}),
    ]), className="mb-3"),

    dcc.Interval(id="status-interval", interval=1000, n_intervals=0),
    dcc.Interval(id="table-interval", interval=TABLE_REFRESH_MS, n_intervals=0),
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
    Output("table-joints", "data"),
    Output("table-imu", "data"),
    Output("table-odom", "data"),
    Output("table-fsm", "data"),
    Output("table-battery", "data"),
    Output("table-services", "data"),
    Output("table-hand-joints", "data"),
    Output("table-hand-tactile", "data"),
    Output("slam-text", "children"),
    Output("table-joints-age", "children"),
    Output("table-imu-age", "children"),
    Output("table-odom-age", "children"),
    Output("table-fsm-age", "children"),
    Output("table-battery-age", "children"),
    Output("table-services-age", "children"),
    Output("table-hand-joints-age", "children"),
    Output("table-hand-tactile-age", "children"),
    Output("slam-age", "children"),
    Output("poll-err", "children"),
    Input("table-interval", "n_intervals"),
)
def on_table_tick(_n):
    with LINK.lock:
        state = dict(LINK.state) if LINK.state else None
        odom = dict(LINK.odom) if LINK.odom else None
        hands = LINK.hands
        ts = LINK.ts
        err = LINK.poll_err

    lowstate = (state or {}).get("lowstate")
    fsm = {k: state.get(k) for k in ("id", "mode", "motion_mode", "motion_code", "gait")} if state else None
    battery = (state or {}).get("battery")
    services = (state or {}).get("services")
    slam_info = (state or {}).get("slam_info")
    slam_text = "—"
    if slam_info:
        try:
            slam_text = json.dumps(json.loads(slam_info), indent=2, sort_keys=True)
        except Exception:
            slam_text = str(slam_info)

    # All panels come from the same combined poll tick, so they share one age badge.
    age = _age_badge(ts)
    ages = [age] * 9

    return (
        lowstate_rows(lowstate),
        imu_rows((lowstate or {}).get("imu")),
        kv_rows(odom),
        kv_rows(fsm) if fsm else [],
        kv_rows(battery),
        services_rows(services),
        hand_rows(hands),
        tactile_summary_rows(hands),
        slam_text,
        *ages,
        f"poll error: {err}" if err else "",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 raw state monitor (lowstate/odom/battery/services/hands tables).")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8073)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    global LINK
    LINK = StateLink(args.iface, args.domain_id)
    print(f"State monitor: http://{args.host}:{args.port} iface={args.iface} domain={args.domain_id}")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


LINK = StateLink("eth0", 0)

if __name__ == "__main__":
    raise SystemExit(main())
