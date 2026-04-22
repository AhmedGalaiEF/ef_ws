#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
import time
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from sdk_client import Robot


ROBOT_LOCK = threading.Lock()
ROBOT_INSTANCE: Robot | None = None
ROBOT_ERROR: str | None = None
ROBOT_IFACE = "enp1s0"
ROBOT_DOMAIN_ID = 0


def _connect_robot(iface: str, domain_id: int) -> tuple[Robot | None, str]:
    global ROBOT_INSTANCE, ROBOT_ERROR
    with ROBOT_LOCK:
        if ROBOT_INSTANCE is not None and ROBOT_IFACE == iface and ROBOT_DOMAIN_ID == domain_id:
            return ROBOT_INSTANCE, "already connected"
        try:
            ROBOT_INSTANCE = Robot(
                iface=iface,
                domain_id=domain_id,
                safety_boot=True,
                auto_start_sensors=True,
            )
            ROBOT_ERROR = None
            return ROBOT_INSTANCE, "connected"
        except Exception as exc:
            ROBOT_INSTANCE = None
            ROBOT_ERROR = str(exc)
            return None, ROBOT_ERROR


def get_robot() -> Robot | None:
    with ROBOT_LOCK:
        return ROBOT_INSTANCE


def robot_status() -> tuple[str, str]:
    with ROBOT_LOCK:
        if ROBOT_INSTANCE is not None:
            return "Connected", "success"
        if ROBOT_ERROR:
            return "Error", "danger"
    return "Disconnected", "secondary"


def pretty(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    [
        html.H2("G1 Robot Dashboard", className="mt-3"),
        dbc.Row(
            [
                dbc.Col(dbc.Input(id="iface-input", value=ROBOT_IFACE, placeholder="iface"), md=3),
                dbc.Col(dbc.Input(id="domain-input", type="number", value=ROBOT_DOMAIN_ID), md=2),
                dbc.Col(dbc.Button("Connect", id="btn-connect", color="primary", className="w-100"), md=2),
                dbc.Col(html.Div(id="connect-result", className="pt-2"), md=5),
            ],
            className="g-2",
        ),
        dbc.Badge(id="conn-badge", className="mt-3"),
        dbc.Alert(id="action-status", children="Ready", color="secondary", className="mt-3"),
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Locomotion",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(dbc.Button("Balanced Stand", id="btn-stand", color="primary", className="w-100"), md=3),
                                dbc.Col(dbc.Button("Stop", id="btn-stop", color="secondary", className="w-100"), md=3),
                            ],
                            className="g-2 mt-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H6("Linear Velocity", className="mt-4"),
                                        html.Div("Left joystick: forward/back and strafe", className="small text-muted mb-2"),
                                        dbc.Row(
                                            [
                                                dbc.Col(dbc.Input(id="joy-linear-x", type="number", value=0.0, step=0.05, placeholder="vx"), md=6),
                                                dbc.Col(dbc.Input(id="joy-linear-y", type="number", value=0.0, step=0.05, placeholder="vy"), md=6),
                                            ],
                                            className="g-2",
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        html.H6("Angular Velocity", className="mt-4"),
                                        html.Div("Right joystick: yaw rate", className="small text-muted mb-2"),
                                        dbc.Input(id="joy-angular-z", type="number", value=0.0, step=0.05, placeholder="vyaw"),
                                    ],
                                    md=6,
                                ),
                            ],
                            className="g-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dbc.Button("Send Joystick Command", id="btn-joy-send", color="success", className="w-100 mt-3"), md=6),
                                dbc.Col(dbc.Button("Center Joysticks", id="btn-joy-center", color="secondary", className="w-100 mt-3"), md=6),
                            ],
                            className="g-2",
                        ),
                    ],
                ),
                dbc.Tab(
                    label="Hands",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div("Dex3 Hand", className="small text-muted mb-2"),
                                        dbc.RadioItems(
                                            id="hand-side",
                                            options=[
                                                {"label": "Right", "value": "right"},
                                                {"label": "Left", "value": "left"},
                                            ],
                                            value="right",
                                            inline=True,
                                        ),
                                    ],
                                    md=12,
                                ),
                            ],
                            className="g-2 mt-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dbc.Button("Open Hand", id="btn-hand-open", color="info", className="w-100"), md=6),
                                dbc.Col(dbc.Button("Close Hand", id="btn-hand-close", color="info", className="w-100"), md=6),
                            ],
                            className="g-2 mt-2",
                        )
                    ],
                ),
                dbc.Tab(
                    label="Speech",
                    children=[
                        dbc.InputGroup(
                            [
                                dbc.Input(id="say-text", value="Hello from the G1 dashboard.", type="text"),
                                dbc.Button("Say", id="btn-say", color="success"),
                            ],
                            className="mt-3",
                        )
                    ],
                ),
                dbc.Tab(
                    label="SLAM",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(dbc.Input(id="slam-type", value="indoor", type="text"), md=4),
                                dbc.Col(dbc.Input(id="slam-save-path", placeholder="optional save path", type="text"), md=5),
                                dbc.Col(dbc.Button("Start SLAM", id="btn-slam-start", color="primary", className="w-100"), md=3),
                            ],
                            className="g-2 mt-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dbc.Button("Get SLAM Pose", id="btn-slam-pose", color="info", className="w-100"), md=6),
                                dbc.Col(dbc.Button("Stop SLAM", id="btn-slam-stop", color="secondary", className="w-100"), md=6),
                            ],
                            className="g-2 mt-2",
                        ),
                    ],
                ),
                dbc.Tab(
                    label="Sensors",
                    children=[
                        dcc.Interval(id="sensor-interval", interval=1000, n_intervals=0),
                        html.H6("Robot State", className="mt-3"),
                        html.Pre(id="sensor-state", style={"maxHeight": "420px", "overflowY": "auto"}),
                        html.H6("IMU", className="mt-3"),
                        html.Pre(id="imu-state", style={"maxHeight": "220px", "overflowY": "auto"}),
                    ],
                ),
            ],
            className="mt-3",
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("conn-badge", "children"),
    Output("conn-badge", "color"),
    Output("connect-result", "children"),
    Input("btn-connect", "n_clicks"),
    State("iface-input", "value"),
    State("domain-input", "value"),
    prevent_initial_call=True,
)
def on_connect(_n: int, iface: str | None, domain_id: int | None) -> tuple[str, str, str]:
    global ROBOT_IFACE, ROBOT_DOMAIN_ID
    ROBOT_IFACE = str(iface or ROBOT_IFACE)
    ROBOT_DOMAIN_ID = int(domain_id or 0)
    _robot, message = _connect_robot(ROBOT_IFACE, ROBOT_DOMAIN_ID)
    label, color = robot_status()
    return label, color, message


@app.callback(
    Output("action-status", "children"),
    Output("action-status", "color"),
    Input("btn-stand", "n_clicks"),
    Input("btn-stop", "n_clicks"),
    Input("btn-joy-send", "n_clicks"),
    Input("btn-joy-center", "n_clicks"),
    Input("btn-hand-open", "n_clicks"),
    Input("btn-hand-close", "n_clicks"),
    Input("btn-say", "n_clicks"),
    Input("btn-slam-start", "n_clicks"),
    Input("btn-slam-pose", "n_clicks"),
    Input("btn-slam-stop", "n_clicks"),
    State("joy-linear-x", "value"),
    State("joy-linear-y", "value"),
    State("joy-angular-z", "value"),
    State("hand-side", "value"),
    State("say-text", "value"),
    State("slam-type", "value"),
    State("slam-save-path", "value"),
    prevent_initial_call=True,
)
def on_action(
    _stand: int | None,
    _stop: int | None,
    _joy_send: int | None,
    _joy_center: int | None,
    _hand_open: int | None,
    _hand_close: int | None,
    _say: int | None,
    _slam_start: int | None,
    _slam_pose: int | None,
    _slam_stop: int | None,
    joy_linear_x: float | None,
    joy_linear_y: float | None,
    joy_angular_z: float | None,
    hand_side: str | None,
    say_text: str | None,
    slam_type: str | None,
    slam_save_path: str | None,
) -> tuple[str, str]:
    robot = get_robot()
    if robot is None:
        label, _ = robot_status()
        return f"Robot not connected ({label}).", "warning"

    trigger = dash.ctx.triggered_id
    try:
        if trigger == "btn-stand":
            robot.balanced_stand()
            return "Balanced stand command sent.", "primary"
        if trigger == "btn-stop":
            robot.stop()
            return "Stop command sent.", "secondary"
        if trigger == "btn-joy-send":
            vx = float(joy_linear_x or 0.0)
            vy = float(joy_linear_y or 0.0)
            vyaw = float(joy_angular_z or 0.0)
            robot.walk(vx=vx, vy=vy, vyaw=vyaw)
            return f"Joystick command sent: vx={vx:.2f}, vy={vy:.2f}, vyaw={vyaw:.2f}", "success"
        if trigger == "btn-joy-center":
            robot.stop()
            return "Joysticks centered. Stop command sent.", "secondary"
        if trigger == "btn-hand-open":
            selected_hand = hand_side or "right"
            robot.hand_open(hand=selected_hand, hold_s=0.5)
            return f"{selected_hand.title()} hand open sent.", "info"
        if trigger == "btn-hand-close":
            selected_hand = hand_side or "right"
            robot.hand_close(hand=selected_hand, hold_s=0.5)
            return f"{selected_hand.title()} hand close sent.", "info"
        if trigger == "btn-say":
            code = robot.say(say_text or "Hello from the G1 dashboard.")
            return f"say() returned {code}", "success"
        if trigger == "btn-slam-start":
            code = robot.start_slam(slam_type=slam_type or "indoor")
            return f"start_slam() returned {code}", "primary"
        if trigger == "btn-slam-pose":
            pose = robot.get_slam_pose(timeout_s=1.5)
            return f"SLAM pose: {pose}", "info"
        if trigger == "btn-slam-stop":
            code = robot.stop_slam(save_path=(slam_save_path or None))
            return f"stop_slam() returned {code}", "secondary"
    except Exception as exc:
        return f"Command failed: {exc}", "danger"

    return "Ready", "secondary"


@app.callback(
    Output("sensor-state", "children"),
    Output("imu-state", "children"),
    Input("sensor-interval", "n_intervals"),
)
def update_sensors(_tick: int) -> tuple[str, str]:
    robot = get_robot()
    if robot is None:
        return "Connect the robot to start sensor polling.", ""
    try:
        return pretty(robot.get_robot_state()), pretty(robot.get_imu())
    except Exception as exc:
        return f"Sensor read failed: {exc}", ""


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
