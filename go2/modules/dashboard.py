#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import threading
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
            ROBOT_INSTANCE = Robot(iface=iface, domain_id=domain_id, auto_start_sensors=True)
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
        html.H2("Go2 Robot Dashboard", className="mt-3"),
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
                                dbc.Col(dbc.Button("Release Mode", id="btn-release", color="warning", className="w-100"), md=3),
                                dbc.Col(dbc.Button("Stand Up", id="btn-stand", color="primary", className="w-100"), md=3),
                                dbc.Col(dbc.Button("Walk 0.3m", id="btn-walk", color="success", className="w-100"), md=3),
                                dbc.Col(dbc.Button("Stop", id="btn-stop", color="secondary", className="w-100"), md=3),
                            ],
                            className="g-2 mt-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dbc.Button("Turn 20deg", id="btn-turn", color="info", className="w-100"), md=4),
                                dbc.Col(dbc.Button("Balance Stand", id="btn-balance", color="primary", className="w-100"), md=4),
                                dbc.Col(dbc.Button("Stand Down", id="btn-stand-down", color="dark", className="w-100"), md=4),
                            ],
                            className="g-2 mt-2",
                        ),
                    ],
                ),
                dbc.Tab(
                    label="Body Height",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(dbc.Input(id="body-height", type="number", value=0.16, step=0.01), md=4),
                                dbc.Col(dbc.Button("Apply Height", id="btn-height", color="primary", className="w-100"), md=3),
                            ],
                            className="g-2 mt-3",
                        )
                    ],
                ),
                dbc.Tab(
                    label="Sensors",
                    children=[
                        dcc.Interval(id="sensor-interval", interval=1000, n_intervals=0),
                        html.H6("Robot State", className="mt-3"),
                        html.Pre(id="sensor-state", style={"maxHeight": "480px", "overflowY": "auto"}),
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
    Input("btn-release", "n_clicks"),
    Input("btn-stand", "n_clicks"),
    Input("btn-walk", "n_clicks"),
    Input("btn-stop", "n_clicks"),
    Input("btn-turn", "n_clicks"),
    Input("btn-balance", "n_clicks"),
    Input("btn-stand-down", "n_clicks"),
    Input("btn-height", "n_clicks"),
    State("body-height", "value"),
    prevent_initial_call=True,
)
def on_action(
    _release: int | None,
    _stand: int | None,
    _walk: int | None,
    _stop: int | None,
    _turn: int | None,
    _balance: int | None,
    _stand_down: int | None,
    _height: int | None,
    body_height: float | None,
) -> tuple[str, str]:
    robot = get_robot()
    if robot is None:
        label, _ = robot_status()
        return f"Robot not connected ({label}).", "warning"

    trigger = dash.ctx.triggered_id
    try:
        if trigger == "btn-release":
            ok = robot.release_active_mode()
            return f"release_active_mode(): {ok}", "warning"
        if trigger == "btn-stand":
            code = robot.stand_up()
            return f"stand_up() returned {code}", "primary"
        if trigger == "btn-walk":
            ok = robot.walk_for(distance=0.3, speed=0.25)
            return f"walk_for finished: {ok}", "success"
        if trigger == "btn-stop":
            code = robot.stop()
            return f"stop() returned {code}", "secondary"
        if trigger == "btn-turn":
            ok = robot.turn_for(angle_rad=math.radians(20.0), yaw_rate=0.4)
            return f"turn_for finished: {ok}", "info"
        if trigger == "btn-balance":
            code = robot.balance_stand()
            return f"balance_stand() returned {code}", "primary"
        if trigger == "btn-stand-down":
            code = robot.stand_down()
            return f"stand_down() returned {code}", "dark"
        if trigger == "btn-height":
            code = robot.set_body_height(float(body_height or 0.16))
            return f"set_body_height() returned {code}", "primary"
    except Exception as exc:
        return f"Command failed: {exc}", "danger"

    return "Ready", "secondary"


@app.callback(
    Output("sensor-state", "children"),
    Input("sensor-interval", "n_intervals"),
)
def update_sensors(_tick: int) -> str:
    robot = get_robot()
    if robot is None:
        return "Connect the robot to start sensor polling."
    try:
        return pretty(robot.get_robot_state())
    except Exception as exc:
        return f"Sensor read failed: {exc}"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8051)
