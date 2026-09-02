#!/usr/bin/env python3
"""
Dash web-app version of ik_pose_cli_v3.py

Run:  python3 ik_pose_dash.py [same CLI flags as ik_pose_cli_v3.py]
Then open  http://localhost:8050  in a browser on the same network.

Extra flags (not in CLI v3):
  --port PORT    HTTP port (default 8050)
  --host HOST    bind address (default 0.0.0.0)
  --debug        enable Dash debug mode
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACADEMY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SOLVED_MODULES_DIR = os.path.join(ACADEMY_DIR, "solved", "modules")
for _p in reversed((SCRIPT_DIR, SOLVED_MODULES_DIR)):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# ── Import robot logic ────────────────────────────────────────────────────────
from ik_pose_cli_v3 import (
    IKPoseCLI, _parse_args, ControllerLockError,
    ARM_CONTROL_MODES, HAND_CONTROL_MODES,
    DOF_NAMES, DOF_UNITS, N_DOFS,
    ARM_JOINTS, JOINT_LABELS, UPPER_BODY_JOINTS, WAIST_JOINTS,
    LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS,
    _rpy_from_R,
)

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, ALL
import dash_bootstrap_components as dbc

# ── Globals ───────────────────────────────────────────────────────────────────
_ctrl: Optional[IKPoseCLI] = None
_lock = threading.RLock()

# Background tick thread
_tick_thread: Optional[threading.Thread] = None
_tick_running = False

# Hand action thread (blocking open/close calls run here)
_hand_thread: Optional[threading.Thread] = None
_hand_status = "idle"          # read by refresh callback for display


# ── Tick loop ─────────────────────────────────────────────────────────────────

def _tick_loop() -> None:
    global _tick_running
    while _tick_running:
        with _lock:
            c = _ctrl
            if c is not None and not c._closed:
                try:
                    c.tick()
                except Exception as exc:
                    c.status = f"Tick error: {exc}"
        # Sleep outside the lock
        hz = _ctrl.rate_hz if _ctrl else 25.0
        time.sleep(1.0 / hz)


# ── Hand action helpers ────────────────────────────────────────────────────────

def _run_hand_action(fn, *args, **kwargs) -> None:
    """Run a blocking hand call in a daemon thread so Dash stays responsive."""
    global _hand_thread, _hand_status

    def _worker():
        global _hand_status
        try:
            _hand_status = "running…"
            fn(*args, **kwargs)
            _hand_status = "done"
        except Exception as exc:
            _hand_status = f"error: {exc}"

    _hand_thread = threading.Thread(target=_worker, daemon=True)
    _hand_thread.start()


def _dex3_action(hand: str, action: str, hold_s: float, ramp_s: Optional[float]) -> None:
    """Call open/close on the Dex3HandController(s) already owned by the IK controller."""
    global _hand_status
    with _lock:
        c = _ctrl
        if c is None:
            return
        # make sure controllers exist for the requested side(s)
        sides = ["left", "right"] if hand == "both" else [hand]
        orig_mode = c.hand_control_mode
        # Temporarily widen to ensure the controller is initialised
        if hand == "both":
            c.hand_control_mode = "both"
        elif hand not in c.hand_control_mode:
            c.hand_control_mode = hand
        c._init_hand_controllers()
        controllers = {s: c.hand_controllers.get(s) for s in sides}
        c.hand_control_mode = orig_mode   # restore

    for side, ctrl in controllers.items():
        if ctrl is None:
            _hand_status = f"No Dex3 controller for {side}"
            continue
        ramp = ramp_s if ramp_s and ramp_s > 0 else None
        if action == "open":
            ctrl.open(hold_s=hold_s, ramp_s=ramp)
        else:
            ctrl.close(hold_s=hold_s, ramp_s=ramp)


# ── Layout helpers ─────────────────────────────────────────────────────────────

def _card(title: str, children, **kwargs) -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(html.B(title)),
        dbc.CardBody(children),
    ], className="mb-2", **kwargs)


def _dof_table() -> html.Div:
    header = dbc.Row([
        dbc.Col(html.Small("DOF"),    width=1),
        dbc.Col(html.Small("Live FK"), width=2),
        dbc.Col(html.Small("Target"),  width=2),
        dbc.Col(html.Small("Step"),    width=2),
        dbc.Col(html.Small("± Adjust"), width=2),
        dbc.Col(html.Small("Step ÷/×"), width=3),
    ], className="fw-bold mb-1 px-1")

    rows = []
    for i in range(N_DOFS):
        rows.append(dbc.Row([
            dbc.Col(html.Code(DOF_NAMES[i]), width=1),
            dbc.Col(html.Span("—", id={"type": "live-dof",  "index": i},
                              className="font-monospace small"), width=2),
            dbc.Col(html.Span("—", id={"type": "tgt-dof",   "index": i},
                              className="font-monospace small text-info"), width=2),
            dbc.Col(html.Span("—", id={"type": "step-dof",  "index": i},
                              className="font-monospace small text-muted"), width=2),
            dbc.Col(dbc.ButtonGroup([
                dbc.Button("−", id={"type": "dec-dof", "index": i},
                           color="secondary", size="sm", className="px-2"),
                dbc.Button("+", id={"type": "inc-dof", "index": i},
                           color="secondary", size="sm", className="px-2"),
            ]), width=2),
            dbc.Col(dbc.ButtonGroup([
                dbc.Button("÷2", id={"type": "halve-step",  "index": i},
                           color="outline-secondary", size="sm"),
                dbc.Button("×2", id={"type": "double-step", "index": i},
                           color="outline-secondary", size="sm"),
            ]), width=3),
        ], className="mb-1 align-items-center px-1"))
    return html.Div([header] + rows)


def _dex3_card() -> dbc.Card:
    return _card("Dex3 Hand Control (sdk_client / sdk_hand)", [
        dbc.Row([
            dbc.Col([
                html.Small("Tick-loop mode:", className="d-block text-muted"),
                dbc.RadioItems(
                    id="hand-mode",
                    options=[{"label": m, "value": m} for m in HAND_CONTROL_MODES],
                    value="off",
                    inline=True,
                    className="small",
                ),
            ], width=12, className="mb-2"),
        ]),
        dbc.Row([
            dbc.Col([
                html.Small("Grip %:", className="me-1"),
                dbc.InputGroup([
                    dbc.Input(id="grip-input", type="number", value=0,
                              step=5, min=0, max=100, style={"width": "70px"}),
                    dbc.Button("Set", id="grip-set", size="sm", color="primary"),
                ], size="sm"),
            ], width="auto"),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Small("One-shot open/close:", className="d-block text-muted"),
                dbc.Row([
                    dbc.Col([
                        html.Small("Side:"),
                        dbc.RadioItems(
                            id="dex3-side",
                            options=[
                                {"label": "Right", "value": "right"},
                                {"label": "Left",  "value": "left"},
                                {"label": "Both",  "value": "both"},
                            ],
                            value="right",
                            inline=True,
                            className="small",
                        ),
                    ], width="auto"),
                    dbc.Col([
                        html.Small("hold s:"),
                        dbc.Input(id="dex3-hold-s", type="number",
                                  value=0.6, step=0.1, min=0,
                                  style={"width": "65px"}, size="sm"),
                    ], width="auto"),
                    dbc.Col([
                        html.Small("ramp s (0=default):"),
                        dbc.Input(id="dex3-ramp-s", type="number",
                                  value=0, step=0.1, min=0,
                                  style={"width": "65px"}, size="sm"),
                    ], width="auto"),
                ], className="mb-1 align-items-end"),
                dbc.ButtonGroup([
                    dbc.Button("✋ Open",  id="dex3-open",  color="success", size="sm"),
                    dbc.Button("✊ Close", id="dex3-close", color="warning", size="sm"),
                ]),
            ], width=12),
        ]),
    ])


def make_layout() -> html.Div:
    return dbc.Container([

        # ── Title ─────────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.H4("6D EE IK Pose Control  v3", className="mb-0 mt-2"),
                    width="auto"),
            dbc.Col(html.Span(id="conn-badge"),  width="auto"),
            dbc.Col(html.Span(id="armed-badge"), width="auto"),
        ], className="align-items-center mb-2"),

        # ── Arm + waist + stiffness controls ──────────────────────────────────
        _card("Arm Controls", dbc.Row([
            dbc.Col([
                html.Small("Arm mode:", className="d-block"),
                dbc.RadioItems(
                    id="arm-mode",
                    options=[{"label": m, "value": m} for m in ARM_CONTROL_MODES],
                    value="right",
                    inline=True,
                ),
            ], width="auto"),
            dbc.Col([
                dbc.Switch(id="waist-toggle",   label="Waist hold",    value=True),
                dbc.Switch(id="orient-stiff",   label="Orient stiff",  value=True),
            ], width="auto"),
            dbc.Col([
                html.Small("Ramp r/s:"),
                dbc.InputGroup([
                    dbc.Input(id="ramp-input", type="number", value=0.2,
                              step=0.01, min=0.01, style={"width": "75px"}),
                    dbc.Button("Set", id="ramp-set", size="sm", color="primary"),
                ], size="sm", className="mb-1"),
                html.Small("max_dq rad:"),
                dbc.InputGroup([
                    dbc.Input(id="maxdq-input", type="number", value=0.2,
                              step=0.005, min=0.005, style={"width": "75px"}),
                    dbc.Button("Set", id="maxdq-set", size="sm", color="primary"),
                ], size="sm", className="mb-1"),
                html.Small("Waist PR kp:"),
                dbc.InputGroup([
                    dbc.Input(id="waist-kp-input", type="number", value=200.0,
                              step=10, min=0, style={"width": "75px"}),
                    dbc.Button("Set", id="waist-kp-set", size="sm", color="primary"),
                ], size="sm"),
            ], width="auto"),
            dbc.Col([
                html.Small("Actions:", className="d-block"),
                dbc.ButtonGroup([
                    dbc.Button("⟳ Sync EE",     id="btn-sync",      color="info",    size="sm"),
                    dbc.Button("↓ Release",      id="btn-release",   color="warning", size="sm"),
                    dbc.Button("↑ Reengage",     id="btn-reengage",  color="success", size="sm"),
                    dbc.Button("⚡ Zero Gain",   id="btn-zero-gain", color="danger",  size="sm"),
                ], vertical=True),
            ], width="auto"),
        ], className="align-items-start g-3")),

        # ── DOF table ─────────────────────────────────────────────────────────
        _card("End-Effector DOF Control  (display arm shown in live column)",
              _dof_table()),

        # ── IK status + joint readout ──────────────────────────────────────────
        _card("IK Status", dbc.Row([
            dbc.Col(html.Div(id="ik-status-left"),  width=6),
            dbc.Col(html.Div(id="ik-status-right"), width=6),
        ])),

        _card("Joint Readout",
              html.Div(id="joint-readout", className="font-monospace small")),

        # ── Poses + Sequence ──────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(_card("Saved Poses", [
                dbc.RadioItems(id="pose-selector", options=[], value=None,
                               className="small mb-2"),
                dbc.InputGroup([
                    dbc.Input(id="pose-name-input",
                              placeholder="Pose name…", size="sm"),
                    dbc.Button("💾 Save",    id="pose-save",    size="sm", color="success"),
                ], className="mb-1"),
                dbc.ButtonGroup([
                    dbc.Button("📂 Load",   id="pose-load",    size="sm", color="primary"),
                    dbc.Button("🗑 Delete", id="pose-delete",  size="sm", color="danger"),
                    dbc.Button("➕ Add→Seq",id="pose-add-seq", size="sm", color="secondary"),
                ]),
                dbc.Switch(id="include-waist-new",
                           label="Include waist in new seq steps",
                           value=True, className="mt-2 small"),
            ]), width=6),

            dbc.Col(_card("Sequence", [
                dbc.RadioItems(id="seq-selector", options=[], value=None,
                               className="small mb-2"),
                dbc.Row([
                    dbc.Col(html.Small("Gap s:"), width="auto"),
                    dbc.Col(dbc.Input(id="seq-gap-input", type="number",
                                     value=2.0, step=0.5, min=0,
                                     style={"width": "65px"}, size="sm"),
                            width="auto"),
                    dbc.Col(dbc.Button("Set", id="seq-gap-set",
                                       size="sm", color="primary"), width="auto"),
                ], className="mb-1 align-items-center"),
                dbc.ButtonGroup([
                    dbc.Button("▲",      id="seq-up",     size="sm", color="secondary"),
                    dbc.Button("▼",      id="seq-down",   size="sm", color="secondary"),
                    dbc.Button("✕ Remove",id="seq-remove",size="sm", color="danger"),
                    dbc.Button("▶ Run",  id="seq-run",    size="sm", color="success"),
                    dbc.Button("■ Stop", id="seq-stop",   size="sm", color="warning"),
                ]),
            ]), width=6),
        ]),

        # ── Hand controls ─────────────────────────────────────────────────────
        _dex3_card(),

        # ── Status bar ────────────────────────────────────────────────────────
        html.Div(id="status-bar", className="p-2 rounded small font-monospace mb-3",
                 style={"background": "#0f1117", "color": "#00d26a",
                        "border": "1px solid #333"}),

        # Interval + dummy stores for action callbacks
        dcc.Interval(id="interval", interval=100, n_intervals=0),
        dcc.Store(id="_act"),       # general action ack
        dcc.Store(id="_hand-act"),  # hand action ack

    ], fluid=True, style={"maxWidth": "1180px"})


# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="IK Pose Control",
    suppress_callback_exceptions=True,
)
app.layout = make_layout()


# ── State snapshot ────────────────────────────────────────────────────────────

def _snap() -> dict:
    """Return a JSON-safe snapshot of controller state. Empty dict if offline."""
    with _lock:
        c = _ctrl
        if c is None or c._closed:
            return {}

        arm = c._display_arm()
        T_live = c._fk_live(arm) if c.seeded else np.eye(4)
        T_tgt  = c.target_T[arm]
        rpy_live = _rpy_from_R(T_live[:3, :3])
        rpy_tgt  = _rpy_from_R(T_tgt[:3, :3])

        dof_live = [float(T_live[i, 3]) for i in range(3)] + list(map(float, rpy_live))
        dof_tgt  = [float(T_tgt[i, 3])  for i in range(3)] + list(map(float, rpy_tgt))
        steps    = [c.pos_step if i < 3 else c.rot_step for i in range(N_DOFS)]

        joint_tgt  = {a: [c.desired_targets[j]  for j in jl]
                      for a, jl in ARM_JOINTS.items()}
        joint_live = {a: [c.latest_positions.get(j, c.current_targets[j]) for j in jl]
                      for a, jl in ARM_JOINTS.items()}

        running_step = c.sequence_step_index - 1 if c.sequence_running else -1

        return {
            "seeded": c.seeded,
            "armed":  c.armed,
            "status": c.status,
            "arm_control_mode":  c.arm_control_mode,
            "waist_enabled":     c.waist_enabled,
            "orient_stiff":      c.orient_stiff,
            "hand_control_mode": c.hand_control_mode,
            "hand_grip_percent": c.hand_grip_percent,
            "max_speed":   c.max_speed,
            "max_dq":      c.max_dq,
            "waist_pr_kp": c.waist_pr_kp,
            "sequence_gap_s":    c.sequence_gap_s,
            "sequence_running":  c.sequence_running,
            "dof_live": dof_live,
            "dof_tgt":  dof_tgt,
            "steps":    steps,
            "ik_info": {k: dict(v) for k, v in c.ik_info.items()},
            "joint_tgt":  joint_tgt,
            "joint_live": joint_live,
            "saved_poses": [
                {"name": p.get("name", f"pose_{i}"),
                 "saved_at": str(p.get("saved_at", ""))[:19]}
                for i, p in enumerate(c.saved_poses)
            ],
            "sequence_steps": [
                {
                    "pose_name": (
                        c.saved_poses[s["pose_index"]].get("name", "?")
                        if 0 <= s.get("pose_index", -1) < len(c.saved_poses)
                        else "<missing>"
                    ),
                    "include_waist": s.get("include_waist", True),
                    "pose_index": s.get("pose_index", -1),
                }
                for s in c.sequence_steps
            ],
            "running_step": running_step,
        }


# ── Refresh callback ──────────────────────────────────────────────────────────

@app.callback(
    Output("conn-badge",  "children"),
    Output("armed-badge", "children"),
    Output("status-bar",  "children"),
    Output("status-bar",  "style"),
    Output({"type": "live-dof",  "index": ALL}, "children"),
    Output({"type": "tgt-dof",   "index": ALL}, "children"),
    Output({"type": "step-dof",  "index": ALL}, "children"),
    Output("ik-status-left",  "children"),
    Output("ik-status-right", "children"),
    Output("joint-readout",   "children"),
    Output("pose-selector",   "options"),
    Output("seq-selector",    "options"),
    Output("hand-status-bar", "children"),
    Input("interval", "n_intervals"),
)
def refresh(_n):
    s = _snap()
    offline_style = {"background": "#1a0000", "color": "#888",
                     "padding": "8px", "borderRadius": "4px",
                     "border": "1px solid #333", "fontFamily": "monospace"}
    empty = ["—"] * N_DOFS
    if not s:
        return (
            dbc.Badge("OFFLINE", color="secondary"),
            dbc.Badge("—",       color="secondary"),
            "Controller not running", offline_style,
            empty, empty, empty,
            "", "", "",
            [], [],
            _hand_status,
        )

    conn_color  = "success" if s["seeded"] else "danger"
    armed_color = "success" if s["armed"]  else "warning"
    conn_badge  = dbc.Badge("CONNECTED" if s["seeded"] else "WAITING", color=conn_color)
    armed_badge = dbc.Badge("ARMED"     if s["armed"]  else "RELEASED", color=armed_color)

    ok_color = "#00d26a" if (s["armed"] and s["seeded"]) else "#ff6b6b"
    status_style = {
        "background": "#0f1117", "color": ok_color,
        "padding": "8px", "borderRadius": "4px",
        "border": "1px solid #333", "fontFamily": "monospace",
    }

    def _fv(v, u):
        return f"{v:+.4f} {u}"

    live_v = [_fv(s["dof_live"][i], DOF_UNITS[i]) for i in range(N_DOFS)]
    tgt_v  = [_fv(s["dof_tgt"][i],  DOF_UNITS[i]) for i in range(N_DOFS)]
    step_v = [_fv(s["steps"][i],    DOF_UNITS[i]) for i in range(N_DOFS)]

    def _ik_div(arm):
        info = s["ik_info"].get(arm, {})
        ok = info.get("success")
        mode = s.get("arm_control_mode", "both")
        if mode not in ("both", arm):
            return html.Div()
        if ok is None:
            return html.Div(f"IK {arm}: pending", className="text-muted small")
        extra = ""
        m = info.get("mode", "")
        if m == "pos_shoulder_elbow":
            extra = "  shoulder+elbow"
        elif m == "pos_axis_clamped":
            extra = f"  axis-clamped ({info.get('axis_error_m', 0):.4f}m)"
        if ok:
            txt = (f"IK {arm}: ✓ OK  "
                   f"pos={info['error_pos_m']:.4f}m  "
                   f"rot={info['error_rot_rad']:.4f}rad  "
                   f"{info['iterations']}it{extra}")
            return html.Div(txt, className="text-success small font-monospace")
        txt = (f"IK {arm}: ✗ FAIL  "
               f"pos={info['error_pos_m']:.4f}m  "
               f"rot={info['error_rot_rad']:.4f}rad")
        return html.Div(txt, className="text-danger small font-monospace")

    ik_left  = _ik_div("left")
    ik_right = _ik_div("right")

    # Joint readout
    arm_mode = s.get("arm_control_mode", "both")
    rows = []
    for arm2, jlist in ARM_JOINTS.items():
        if arm_mode not in ("both", arm2):
            continue
        lbl = "  ".join(f"{n:<7}" for n in JOINT_LABELS)
        tgt = "  ".join(f"{v:+.3f}" for v in s["joint_tgt"][arm2])
        liv = "  ".join(f"{v:+.3f}" for v in s["joint_live"][arm2])
        rows += [
            html.Div([html.Span(f"{arm2.upper()}: ", className="text-info"),
                      html.Span(lbl)]),
            html.Div([html.Span("target: ", className="text-muted"),
                      html.Span(tgt),
                      html.Span("   live: ", className="text-warning ms-3"),
                      html.Span(liv, className="text-warning")]),
        ]
    joint_div = html.Div(rows)

    pose_opts = [
        {"label": f"{i}: {p['name']}  {p['saved_at']}", "value": i}
        for i, p in enumerate(s["saved_poses"])
    ]
    seq_opts = []
    for i, step in enumerate(s["sequence_steps"]):
        w = "waist" if step["include_waist"] else "arms"
        indicator = " ▶ ACTIVE" if i == s["running_step"] else ""
        seq_opts.append({
            "label": f"{i+1}: {step['pose_name']} [{w}]{indicator}",
            "value": i,
        })

    return (
        conn_badge, armed_badge,
        s["status"], status_style,
        live_v, tgt_v, step_v,
        ik_left, ik_right,
        joint_div,
        pose_opts, seq_opts,
        _hand_status,
    )


# ── DOF adjust ────────────────────────────────────────────────────────────────

@app.callback(
    Output("_act", "data", allow_duplicate=True),
    Input({"type": "inc-dof",    "index": ALL}, "n_clicks"),
    Input({"type": "dec-dof",    "index": ALL}, "n_clicks"),
    Input({"type": "halve-step", "index": ALL}, "n_clicks"),
    Input({"type": "double-step","index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def dof_buttons(*_):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    try:
        id_dict  = json.loads(ctx.triggered[0]["prop_id"].rsplit(".", 1)[0])
        btn_type = id_dict["type"]
        idx      = int(id_dict["index"])
    except Exception:
        return no_update

    with _lock:
        c = _ctrl
        if c is None or not c.seeded or not c.armed:
            return no_update
        prev = c.dof_idx
        c.dof_idx = idx
        try:
            if btn_type == "inc-dof":
                c._adjust_dof(+c._ee_step())
            elif btn_type == "dec-dof":
                c._adjust_dof(-c._ee_step())
            elif btn_type == "halve-step":
                c._set_ee_step(max(0.0001, c._ee_step() / 2.0))
            elif btn_type == "double-step":
                hi = 1.0 if idx < 3 else math.pi
                c._set_ee_step(min(hi, c._ee_step() * 2.0))
        finally:
            c.dof_idx = prev
    return no_update


# ── Arm mode / waist / stiffness ──────────────────────────────────────────────

@app.callback(
    Output("_act", "data", allow_duplicate=True),
    Input("arm-mode",     "value"),
    Input("waist-toggle", "value"),
    Input("orient-stiff", "value"),
    prevent_initial_call=True,
)
def settings_changed(arm_mode, waist_val, stiff_val):
    with _lock:
        c = _ctrl
        if c is None:
            return no_update
        if arm_mode is not None:
            c.arm_control_mode = arm_mode
        prev_waist = c.waist_enabled
        c.waist_enabled = bool(waist_val)
        if c.waist_enabled and not prev_waist:
            for j in WAIST_JOINTS:
                c.desired_targets[j] = c.latest_positions.get(j, c.desired_targets[j])
                c.current_targets[j] = c.desired_targets[j]
            c._sync_ee_from_joints()
        c.orient_stiff = bool(stiff_val)
    return no_update


# ── Parameter set ─────────────────────────────────────────────────────────────

@app.callback(
    Output("_act", "data", allow_duplicate=True),
    Input("ramp-set",      "n_clicks"),
    Input("maxdq-set",     "n_clicks"),
    Input("waist-kp-set",  "n_clicks"),
    State("ramp-input",     "value"),
    State("maxdq-input",    "value"),
    State("waist-kp-input", "value"),
    prevent_initial_call=True,
)
def param_set(r, d, w, ramp, maxdq, wkp):
    trig = (callback_context.triggered or [{}])[0].get("prop_id", "").split(".")[0]
    with _lock:
        c = _ctrl
        if c is None:
            return no_update
        if trig == "ramp-set"     and ramp  is not None:
            c.max_speed   = max(0.01, float(ramp))
            c.status = f"Ramp speed → {c.max_speed:.4f} r/s"
        elif trig == "maxdq-set"  and maxdq is not None:
            c.max_dq      = max(0.005, min(math.pi, float(maxdq)))
            c.status = f"max_dq → {c.max_dq:.4f} rad"
        elif trig == "waist-kp-set" and wkp is not None:
            c.waist_pr_kp = max(0.0, float(wkp))
            c.status = f"Waist PR kp → {c.waist_pr_kp:.1f}"
    return no_update


# ── Action buttons ────────────────────────────────────────────────────────────

@app.callback(
    Output("_act", "data", allow_duplicate=True),
    Input("btn-sync",      "n_clicks"),
    Input("btn-release",   "n_clicks"),
    Input("btn-reengage",  "n_clicks"),
    Input("btn-zero-gain", "n_clicks"),
    prevent_initial_call=True,
)
def action_buttons(*_):
    trig = (callback_context.triggered or [{}])[0].get("prop_id", "").split(".")[0]
    with _lock:
        c = _ctrl
        if c is None:
            return no_update
        if trig == "btn-sync":
            c._sync_targets_to_live()
            c.status = "EE targets resynced to current hand pose"
        elif trig == "btn-release":
            def _do():
                with _lock:
                    try:
                        c._release_arms()
                        c.armed = False
                        c.status = "Arms released"
                    except Exception as exc:
                        c.status = f"Release failed: {exc}"
            threading.Thread(target=_do, daemon=True).start()
        elif trig == "btn-reengage":
            def _do():
                with _lock:
                    try:
                        c._unrelease_arms()
                        c.armed = True
                        c._sync_targets_to_live()
                        c.status = "Reengaged — synced to live pose"
                    except Exception as exc:
                        c.status = f"Reengage failed: {exc}"
            threading.Thread(target=_do, daemon=True).start()
        elif trig == "btn-zero-gain":
            c.pub.publish_zero_gains(c.current_targets)
            c.status = "Zero-gain hold sent"
    return no_update


# ── Dex3 hand mode + grip ─────────────────────────────────────────────────────

@app.callback(
    Output("_act", "data", allow_duplicate=True),
    Input("hand-mode", "value"),
    Input("grip-set",  "n_clicks"),
    State("grip-input","value"),
    prevent_initial_call=True,
)
def dex3_mode_grip(hand_mode, _g, grip):
    trig = (callback_context.triggered or [{}])[0].get("prop_id", "").split(".")[0]
    with _lock:
        c = _ctrl
        if c is None:
            return no_update
        if trig == "hand-mode" and hand_mode is not None:
            c.hand_control_mode = hand_mode
            c._init_hand_controllers()
            c.status = f"Dex3 mode → {hand_mode}"
        elif trig == "grip-set" and grip is not None:
            c.hand_grip_percent = max(0.0, min(100.0, float(grip)))
            c._ensure_hand_control_for_grip()
            c._publish_hand_targets_once()
            c.status = f"Dex3 grip → {c.hand_grip_percent:.0f}%"
    return no_update


# ── Dex3 open / close ─────────────────────────────────────────────────────────

@app.callback(
    Output("_hand-act", "data", allow_duplicate=True),
    Input("dex3-open",  "n_clicks"),
    Input("dex3-close", "n_clicks"),
    State("dex3-side",   "value"),
    State("dex3-hold-s", "value"),
    State("dex3-ramp-s", "value"),
    prevent_initial_call=True,
)
def dex3_open_close(_o, _c, side, hold_s, ramp_s):
    trig = (callback_context.triggered or [{}])[0].get("prop_id", "").split(".")[0]
    action = "open" if trig == "dex3-open" else "close"
    hs  = max(0.0, float(hold_s or 0.6))
    rs  = float(ramp_s or 0) or None
    _run_hand_action(_dex3_action, side or "right", action, hs, rs)
    return no_update


# ── Pose actions ──────────────────────────────────────────────────────────────

@app.callback(
    Output("_act", "data", allow_duplicate=True),
    Input("pose-save",    "n_clicks"),
    Input("pose-load",    "n_clicks"),
    Input("pose-delete",  "n_clicks"),
    Input("pose-add-seq", "n_clicks"),
    State("pose-name-input",  "value"),
    State("pose-selector",    "value"),
    State("include-waist-new","value"),
    prevent_initial_call=True,
)
def pose_actions(*_):
    trig = (callback_context.triggered or [{}])[0].get("prop_id", "").split(".")[0]
    states = callback_context.states
    name     = states.get("pose-name-input.value")
    selected = states.get("pose-selector.value")
    waist    = bool(states.get("include-waist-new.value", True))

    with _lock:
        c = _ctrl
        if c is None:
            return no_update
        if trig == "pose-save":
            if name:
                c.saved_poses.append(c._pose_payload(name))
                c._write_pose_file()
                c.status = f"Saved pose '{name}'"
        elif trig == "pose-load":
            idx = selected
            if idx is not None and 0 <= idx < len(c.saved_poses):
                try:
                    c._apply_joint_pose(c.saved_poses[idx], include_waist=True)
                    c.status = f"Loaded '{c.saved_poses[idx].get('name','?')}'"
                except Exception as exc:
                    c.status = f"Load failed: {exc}"
        elif trig == "pose-delete":
            idx = selected
            if idx is not None and 0 <= idx < len(c.saved_poses):
                pname = c.saved_poses[idx].get("name", "?")
                new_steps = []
                for step in c.sequence_steps:
                    pi = step.get("pose_index", -1)
                    if pi == idx:
                        continue
                    new_steps.append({
                        "pose_index": pi - (1 if pi > idx else 0),
                        "include_waist": step.get("include_waist", True),
                    })
                c.sequence_steps = new_steps
                del c.saved_poses[idx]
                c.sequence_running = False
                c._write_pose_file()
                c.status = f"Deleted pose '{pname}'"
        elif trig == "pose-add-seq":
            idx = selected
            if idx is not None and 0 <= idx < len(c.saved_poses):
                c.include_waist_new = waist
                c.sequence_steps.append({"pose_index": idx, "include_waist": waist})
                c._write_pose_file()
                pname = c.saved_poses[idx].get("name", "?")
                c.status = f"Added '{pname}' to sequence"
    return no_update


# ── Sequence actions ──────────────────────────────────────────────────────────

@app.callback(
    Output("_act", "data", allow_duplicate=True),
    Input("seq-up",      "n_clicks"),
    Input("seq-down",    "n_clicks"),
    Input("seq-remove",  "n_clicks"),
    Input("seq-run",     "n_clicks"),
    Input("seq-stop",    "n_clicks"),
    Input("seq-gap-set", "n_clicks"),
    State("seq-selector",  "value"),
    State("seq-gap-input", "value"),
    prevent_initial_call=True,
)
def seq_actions(*_):
    trig = (callback_context.triggered or [{}])[0].get("prop_id", "").split(".")[0]
    states = callback_context.states
    idx = states.get("seq-selector.value")
    gap = states.get("seq-gap-input.value")

    with _lock:
        c = _ctrl
        if c is None:
            return no_update
        if trig == "seq-up":
            if idx is not None and 1 <= idx < len(c.sequence_steps):
                c.sequence_steps[idx-1], c.sequence_steps[idx] = (
                    c.sequence_steps[idx], c.sequence_steps[idx-1])
                c._write_pose_file()
        elif trig == "seq-down":
            if idx is not None and 0 <= idx < len(c.sequence_steps) - 1:
                c.sequence_steps[idx+1], c.sequence_steps[idx] = (
                    c.sequence_steps[idx], c.sequence_steps[idx+1])
                c._write_pose_file()
        elif trig == "seq-remove":
            if idx is not None and 0 <= idx < len(c.sequence_steps):
                del c.sequence_steps[idx]
                c.sequence_running = False
                c._write_pose_file()
                c.status = "Removed sequence step"
        elif trig == "seq-run":
            if not c.armed:
                c.status = "Reengage arms before running a sequence"
            elif not c.sequence_steps:
                c.status = "Add poses to the sequence first"
            else:
                c.sequence_running = True
                c.sequence_step_index = 0
                c.sequence_next_time_s = 0.0
                c.status = "Sequence started"
        elif trig == "seq-stop":
            c.sequence_running = False
            c.sequence_step_index = 0
            c.sequence_next_time_s = 0.0
            c.status = "Sequence stopped"
        elif trig == "seq-gap-set" and gap is not None:
            c.sequence_gap_s = max(0.0, float(gap))
            c._write_pose_file()
            c.status = f"Sequence gap → {c.sequence_gap_s:.1f} s"
    return no_update


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global _ctrl, _tick_running, _tick_thread

    # Split our extra flags from the ik_pose_cli_v3 flags
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--port",  type=int,  default=8050)
    extra_parser.add_argument("--host",  default="0.0.0.0")
    extra_parser.add_argument("--debug", action="store_true")
    extra_args, remaining = extra_parser.parse_known_args()

    # Parse remaining with the original CLI parser
    orig_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining
    args = _parse_args()
    sys.argv = orig_argv

    print("Initialising IK controller…")
    try:
        _ctrl = IKPoseCLI(args)
        _ctrl._last_tick = time.monotonic()
    except ControllerLockError as exc:
        print(f"ik_pose_dash: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    _tick_running = True
    _tick_thread = threading.Thread(target=_tick_loop, daemon=True)
    _tick_thread.start()

    print(f"Open  http://localhost:{extra_args.port}  in your browser.")

    import atexit, signal as _signal

    shutdown_done = False

    def _shutdown() -> None:
        nonlocal shutdown_done
        global _tick_running
        if shutdown_done:
            return
        shutdown_done = True
        _tick_running = False
        if _ctrl and not _ctrl._closed:
            _ctrl.close()
        if _tick_thread and _tick_thread.is_alive():
            _tick_thread.join(timeout=1.0)

    def _signal_stop(signum, _frame) -> None:
        _shutdown()
        raise SystemExit(128 + int(signum))

    atexit.register(_shutdown)
    for _sig in ("SIGINT", "SIGTERM"):
        try:
            _signal.signal(getattr(_signal, _sig), _signal_stop)
        except Exception:
            pass

    try:
        app.run(
            host=extra_args.host,
            port=extra_args.port,
            debug=extra_args.debug,
            use_reloader=False,
        )
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
