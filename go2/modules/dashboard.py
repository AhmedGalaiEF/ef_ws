#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import math
import threading
import time
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from sdk_client import Robot

try:
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc


ROBOT_LOCK = threading.Lock()
ROBOT_INSTANCE: Robot | None = None
ROBOT_ERROR: str | None = None
ROBOT_IFACE = "enp1s0"
ROBOT_DOMAIN_ID = 0


class LowLevelController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pub: ChannelPublisher | None = None
        self._crc = CRC()
        self._thread: threading.Thread | None = None
        self._enabled = False
        self._q = [0.0] * 20
        self._kp = [20.0] * 20
        self._kd = [2.0] * 20

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _ensure_publisher(self) -> None:
        if self._pub is not None:
            return
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()

    def _loop(self) -> None:
        while True:
            with self._lock:
                enabled = self._enabled
                q = list(self._q)
                kp = list(self._kp)
                kd = list(self._kd)
                pub = self._pub
            if enabled and pub is not None:
                cmd = unitree_go_msg_dds__LowCmd_()
                cmd.head[0] = 0xFE
                cmd.head[1] = 0xEF
                cmd.level_flag = 0xFF
                cmd.gpio = 0
                for i in range(20):
                    cmd.motor_cmd[i].mode = 0x01
                    cmd.motor_cmd[i].q = float(q[i])
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kp = float(kp[i])
                    cmd.motor_cmd[i].kd = float(kd[i])
                    cmd.motor_cmd[i].tau = 0.0
                cmd.crc = self._crc.Crc(cmd)
                pub.Write(cmd)
            time.sleep(0.02)

    def enable(self, robot: Robot) -> None:
        robot.release_active_mode()
        lowstate = robot.get_low_state()
        if lowstate is None:
            raise RuntimeError("lowstate is not available; cannot seed low-level targets")
        self._ensure_publisher()
        self._ensure_thread()
        with self._lock:
            for i in range(min(20, len(lowstate.motor_state))):
                try:
                    self._q[i] = float(lowstate.motor_state[i].q)
                except Exception:
                    self._q[i] = 0.0
                self._kp[i] = 20.0 if i < 12 else 0.0
                self._kd[i] = 2.0 if i < 12 else 0.0
            self._enabled = True

    def disable(self) -> None:
        with self._lock:
            self._enabled = False

    def set_joint(self, joint_index: int, q: float, kp: float, kd: float) -> None:
        idx = int(joint_index)
        if idx < 0 or idx >= 20:
            raise ValueError("joint_index must be between 0 and 19")
        with self._lock:
            self._q[idx] = float(q)
            self._kp[idx] = float(kp)
            self._kd[idx] = float(kd)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": bool(self._enabled),
                "q": list(self._q),
                "kp": list(self._kp),
                "kd": list(self._kd),
            }


LOW_LEVEL = LowLevelController()


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


def jpeg_to_data_uri(payload: bytes | None) -> str | None:
    if not payload:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


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
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Enable Low-Level Control")),
                dbc.ModalBody(
                    "This will start streaming rt/lowcmd targets and can destabilize the robot. "
                    "Make sure the robot is supported, clear of obstacles, and you understand the joint targets."
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button("Cancel", id="ll-modal-cancel", color="secondary"),
                        dbc.Button("Enable", id="ll-modal-confirm", color="danger"),
                    ]
                ),
            ],
            id="ll-confirm-modal",
            is_open=False,
        ),
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Locomotion",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(dbc.Button("Release Mode", id="btn-release", color="warning", className="w-100"), md=3),
                                dbc.Col(dbc.Button("Stand Up", id="btn-stand", color="primary", className="w-100"), md=3),
                                dbc.Col(dbc.Button("Stop", id="btn-stop", color="secondary", className="w-100"), md=3),
                            ],
                            className="g-2 mt-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dbc.Button("Balance Stand", id="btn-balance", color="primary", className="w-100"), md=4),
                                dbc.Col(dbc.Button("Stand Down", id="btn-stand-down", color="dark", className="w-100"), md=4),
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
                    label="Sensors",
                    children=[
                        dcc.Interval(id="sensor-interval", interval=1000, n_intervals=0),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H6("Camera", className="mt-3"),
                                        html.Img(
                                            id="camera-feed",
                                            style={"width": "100%", "borderRadius": "8px", "border": "1px solid #ccc"},
                                        ),
                                        html.Div(id="camera-status", className="small text-muted mt-2"),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        html.H6("IMU", className="mt-3"),
                                        html.Pre(id="imu-state", style={"maxHeight": "320px", "overflowY": "auto"}),
                                    ],
                                    md=6,
                                ),
                            ],
                            className="g-3",
                        ),
                        html.H6("Robot State", className="mt-4"),
                        html.Pre(id="sensor-state", style={"maxHeight": "360px", "overflowY": "auto"}),
                    ],
                ),
                dbc.Tab(
                    label="Low Level",
                    children=[
                        html.Div(
                            "This tab streams direct rt/lowcmd joint targets. Use only when the robot is safely supported.",
                            className="mt-3 mb-2 text-danger",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dbc.Button("Enable Low Level", id="btn-ll-enable", color="danger", className="w-100"), md=4),
                                dbc.Col(dbc.Button("Disable Low Level", id="btn-ll-disable", color="secondary", className="w-100"), md=4),
                            ],
                            className="g-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dbc.Input(id="ll-joint-index", type="number", value=0, min=0, max=19), md=2),
                                dbc.Col(dbc.Input(id="ll-q", type="number", value=0.0, step=0.01), md=3),
                                dbc.Col(dbc.Input(id="ll-kp", type="number", value=20.0, step=0.5), md=2),
                                dbc.Col(dbc.Input(id="ll-kd", type="number", value=2.0, step=0.1), md=2),
                                dbc.Col(dbc.Button("Apply Joint Target", id="btn-ll-apply", color="primary", className="w-100"), md=3),
                            ],
                            className="g-2 mt-2",
                        ),
                        html.Pre(id="ll-state", className="mt-3", style={"maxHeight": "260px", "overflowY": "auto"}),
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
    Input("btn-stop", "n_clicks"),
    Input("btn-balance", "n_clicks"),
    Input("btn-stand-down", "n_clicks"),
    Input("btn-joy-send", "n_clicks"),
    Input("btn-joy-center", "n_clicks"),
    State("joy-linear-x", "value"),
    State("joy-linear-y", "value"),
    State("joy-angular-z", "value"),
    prevent_initial_call=True,
)
def on_action(
    _release: int | None,
    _stand: int | None,
    _stop: int | None,
    _balance: int | None,
    _stand_down: int | None,
    _joy_send: int | None,
    _joy_center: int | None,
    joy_linear_x: float | None,
    joy_linear_y: float | None,
    joy_angular_z: float | None,
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
        if trigger == "btn-stop":
            code = robot.stop()
            return f"stop() returned {code}", "secondary"
        if trigger == "btn-balance":
            code = robot.balance_stand()
            return f"balance_stand() returned {code}", "primary"
        if trigger == "btn-stand-down":
            code = robot.stand_down()
            return f"stand_down() returned {code}", "dark"
        if trigger == "btn-joy-send":
            vx = float(joy_linear_x or 0.0)
            vy = float(joy_linear_y or 0.0)
            vyaw = float(joy_angular_z or 0.0)
            code = robot.move(vx=vx, vy=vy, vyaw=vyaw)
            return f"Joystick command sent: vx={vx:.2f}, vy={vy:.2f}, vyaw={vyaw:.2f}, rc={code}", "success"
        if trigger == "btn-joy-center":
            code = robot.stop()
            return f"Joysticks centered. stop() returned {code}", "secondary"
    except Exception as exc:
        return f"Command failed: {exc}", "danger"

    return "Ready", "secondary"


@app.callback(
    Output("ll-confirm-modal", "is_open"),
    Input("btn-ll-enable", "n_clicks"),
    Input("ll-modal-cancel", "n_clicks"),
    Input("ll-modal-confirm", "n_clicks"),
    State("ll-confirm-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_low_level_modal(
    _enable: int | None,
    _cancel: int | None,
    _confirm: int | None,
    is_open: bool,
) -> bool:
    trigger = dash.ctx.triggered_id
    if trigger == "btn-ll-enable":
        return True
    if trigger in ("ll-modal-cancel", "ll-modal-confirm"):
        return False
    return is_open


@app.callback(
    Output("ll-state", "children"),
    Output("action-status", "children", allow_duplicate=True),
    Output("action-status", "color", allow_duplicate=True),
    Input("ll-modal-confirm", "n_clicks"),
    Input("btn-ll-disable", "n_clicks"),
    Input("btn-ll-apply", "n_clicks"),
    State("ll-joint-index", "value"),
    State("ll-q", "value"),
    State("ll-kp", "value"),
    State("ll-kd", "value"),
    prevent_initial_call=True,
)
def on_low_level_action(
    _confirm: int | None,
    _disable: int | None,
    _apply: int | None,
    joint_index: float | None,
    q: float | None,
    kp: float | None,
    kd: float | None,
) -> tuple[str, str, str]:
    robot = get_robot()
    if robot is None:
        label, _ = robot_status()
        return pretty(LOW_LEVEL.status()), f"Robot not connected ({label}).", "warning"

    trigger = dash.ctx.triggered_id
    try:
        if trigger == "ll-modal-confirm":
            LOW_LEVEL.enable(robot)
            return pretty(LOW_LEVEL.status()), "Low-level control enabled.", "danger"
        if trigger == "btn-ll-disable":
            LOW_LEVEL.disable()
            return pretty(LOW_LEVEL.status()), "Low-level control disabled.", "secondary"
        if trigger == "btn-ll-apply":
            LOW_LEVEL.set_joint(
                joint_index=int(joint_index or 0),
                q=float(q or 0.0),
                kp=float(kp or 0.0),
                kd=float(kd or 0.0),
            )
            return pretty(LOW_LEVEL.status()), "Low-level joint target updated.", "primary"
    except Exception as exc:
        return pretty(LOW_LEVEL.status()), f"Low-level command failed: {exc}", "danger"

    return pretty(LOW_LEVEL.status()), "Ready", "secondary"


@app.callback(
    Output("sensor-state", "children"),
    Output("imu-state", "children"),
    Output("camera-feed", "src"),
    Output("camera-status", "children"),
    Input("sensor-interval", "n_intervals"),
)
def update_sensors(_tick: int) -> tuple[str, str, str | None, str]:
    robot = get_robot()
    if robot is None:
        return "Connect the robot to start sensor polling.", "", None, "camera unavailable"

    state_text = ""
    imu_text = ""
    camera_src = None
    camera_status = "camera unavailable"

    try:
        state_text = pretty(robot.get_robot_state())
    except Exception as exc:
        state_text = f"Sensor read failed: {exc}"

    try:
        imu_text = pretty(robot.get_imu())
    except Exception as exc:
        imu_text = f"IMU read failed: {exc}"

    try:
        camera_src = jpeg_to_data_uri(robot.get_camera_image_jpeg())
        camera_status = f"streaming at {time.strftime('%H:%M:%S')}"
    except Exception as exc:
        camera_status = f"camera read failed: {exc}"

    return state_text, imu_text, camera_src, camera_status


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8051)
