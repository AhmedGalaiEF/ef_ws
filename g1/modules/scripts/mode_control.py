#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

try:
    from sdk_client import Robot
except ImportError as exc:
    raise SystemExit(
        "Local sdk_client.Robot helper is required for this app."
    ) from exc

try:
    import dash
    import dash_bootstrap_components as dbc
    from dash import Input, Output, State, dcc, html
except ImportError as exc:
    raise SystemExit(
        "Dash and dash-bootstrap-components are required for this app.\n"
        "Install them with:\n"
        "  pip install dash dash-bootstrap-components"
    ) from exc

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.g1.loco.g1_loco_api import (
        ROBOT_API_ID_LOCO_GET_FSM_ID,
        ROBOT_API_ID_LOCO_GET_FSM_MODE,
    )
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc


FSM_ZERO_TORQUE = 0
FSM_DAMPING = 1
FSM_SIT = 3
FSM_PREPARE = 4
FSM_WALK = 501
FSM_RUN = 802
DEFAULT_CLIMB_FSM = 812
AI_MODE_NAME = "ai_sport"
MODE_ALIASES = {
    "ai": AI_MODE_NAME,
}
ERROR_HINTS = {
    0: "success",
    7001: "request parameter error",
    7002: "service busy; retry",
    7004: "unsupported mode name",
    7005: "internal command execute error",
    7006: "check command execute error",
    7007: "switch command execute error",
    7008: "release command execute error",
    7009: "custom config set error",
}


@dataclass(frozen=True)
class RobotState:
    mode: str
    fsm_id: int | None
    fsm_mode: int | None
    motion_mode: str | None
    motion_raw: Any
    motion_code: int | None = None
    loco_skipped: bool = False
    error: str | None = None


class ModeController:
    def __init__(self, iface: str, domain_id: int, timeout: float, climb_fsm_id: int) -> None:
        self.iface = str(iface)
        self.domain_id = int(domain_id)
        self.timeout = float(timeout)
        self.climb_fsm_id = int(climb_fsm_id)
        self._lock = threading.RLock()
        self._initialized = False
        self._loco: LocoClient | None = None
        self._motion: MotionSwitcherClient | None = None
        self._robot: Robot | None = None

    def _ensure_clients(self) -> None:
        if self._initialized:
            return
        ChannelFactoryInitialize(self.domain_id, self.iface)

        loco = LocoClient()
        loco.SetTimeout(self.timeout)
        loco.Init()

        motion = MotionSwitcherClient()
        motion.SetTimeout(self.timeout)
        motion.Init()

        self._loco = loco
        self._motion = motion
        self._initialized = True

    @staticmethod
    def _result_code(result: Any) -> int:
        if result is None:
            return 0
        if isinstance(result, tuple):
            return int(result[0])
        return int(result)

    @staticmethod
    def _rpc_get_int(client: LocoClient, api_id: int) -> int | None:
        try:
            code, data = client._Call(api_id, "{}")  # type: ignore[attr-defined]
            if code != 0 or not data:
                return None
            return int(json.loads(data).get("data"))
        except Exception:
            return None

    @staticmethod
    def _motion_mode_name(data: Any) -> str | None:
        if not isinstance(data, dict):
            return None
        for key in ("name", "mode", "alias"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _canonical_motion_name(name: str | None) -> str:
        value = "" if name is None else str(name).strip()
        return MODE_ALIASES.get(value, value)

    @classmethod
    def _motion_is_ai(cls, name: str | None) -> bool:
        return cls._canonical_motion_name(name) == AI_MODE_NAME

    def state(self) -> RobotState:
        with self._lock:
            try:
                self._ensure_clients()
                assert self._loco is not None
                assert self._motion is not None

                motion_raw = None
                motion_name = None
                motion_code = None
                try:
                    motion_code, motion_raw = self._motion.CheckMode()
                    motion_code = int(motion_code)
                    if motion_code == 0:
                        motion_name = self._motion_mode_name(motion_raw)
                except Exception as exc:
                    motion_raw = {"error": str(exc)}

                if motion_code == 0 and not self._motion_is_ai(motion_name):
                    return RobotState(
                        "dev",
                        None,
                        None,
                        motion_name or "<released>",
                        motion_raw,
                        motion_code=motion_code,
                        loco_skipped=True,
                    )

                fsm_id = self._rpc_get_int(self._loco, ROBOT_API_ID_LOCO_GET_FSM_ID)
                fsm_mode = self._rpc_get_int(self._loco, ROBOT_API_ID_LOCO_GET_FSM_MODE)
                mode = self._classify_mode(fsm_id)
                return RobotState(mode, fsm_id, fsm_mode, motion_name, motion_raw, motion_code=motion_code)
            except Exception as exc:
                return RobotState("unavailable", None, None, None, None, error=str(exc))

    def _classify_mode(self, fsm_id: int | None) -> str:
        if fsm_id == FSM_ZERO_TORQUE:
            return "zero_torque"
        if fsm_id == FSM_DAMPING:
            return "damping"
        if fsm_id == FSM_SIT:
            return "sit"
        if fsm_id == FSM_PREPARE:
            return "prepare"
        if fsm_id == FSM_WALK:
            return "walk"
        if fsm_id == FSM_RUN:
            return "run"
        if fsm_id == self.climb_fsm_id:
            return "climb"
        return "unknown"

    def command(self, name: str) -> str:
        with self._lock:
            self._ensure_clients()
            assert self._loco is not None
            assert self._motion is not None

            if name == "damping":
                self._result_code(self._loco.Damp())
                return "Damping command sent."
            if name == "zero_torque":
                state = self.state()
                if state.mode == "dev":
                    code = self._result_code(self._motion.SelectMode(AI_MODE_NAME))
                    time.sleep(0.2)
                    try:
                        self._loco.ZeroTorque()
                    except Exception:
                        pass
                    return f"AI mode selected for zero torque. code={code}"
                self._result_code(self._loco.ZeroTorque())
                return "Zero torque command sent."
            if name == "prepare":
                self._result_code(self._loco.SetFsmId(FSM_PREPARE))
                return f"Prepare command sent. fsm_id={FSM_PREPARE}"
            if name == "sit":
                self._result_code(self._loco.Sit())
                return "Sit command sent."
            if name == "walk":
                self._result_code(self._loco.SetFsmId(FSM_WALK))
                return "Walk command sent."
            if name == "run":
                self._result_code(self._loco.SetFsmId(FSM_RUN))
                return "Run command sent."
            if name == "climb":
                self._result_code(self._loco.SetFsmId(self.climb_fsm_id))
                return f"Climb command sent. fsm_id={self.climb_fsm_id}"
            if name == "dev":
                state = self.state()
                if state.mode == "dev":
                    code = self._result_code(self._motion.SelectMode(AI_MODE_NAME))
                    return f"AI mode selected. code={code}"
                code = self._result_code(self._motion.ReleaseMode())
                return f"AI mode released; dev mode active. code={code}"
            if name == "release_arms":
                robot = self.robot()
                robot.start_sensors()
                result = robot.release_arms()
                return f"Release arms command sent. result={json.dumps(result, default=str)}"
            if name == "stop":
                self.robot().stop()
                return "Stop command sent."
            raise ValueError(f"Unknown command: {name}")

    def robot(self) -> Robot:
        if self._robot is None:
            self._robot = Robot(
                iface=self.iface,
                domain_id=self.domain_id,
                auto_start_sensors=False,
            )
        return self._robot


def button_disabled(mode: str, button: str) -> bool:
    if button in {"release_arms", "stop"}:
        return False
    if mode == "unavailable":
        return True
    if mode == "zero_torque":
        return button not in {"damping", "dev"}
    if mode == "dev":
        return button != "dev"
    if button in {"walk", "run", "climb"}:
        return mode not in {"prepare", "walk", "run", "climb"}
    return False


def error_hint(code: int | None) -> str:
    if code is None:
        return ""
    hint = ERROR_HINTS.get(int(code))
    return "" if hint is None else f" ({hint})"


def state_text(state: RobotState) -> str:
    bits = [f"state={state.mode}"]
    if state.fsm_id is not None:
        bits.append(f"fsm_id={state.fsm_id}")
    if state.fsm_mode is not None:
        bits.append(f"fsm_mode={state.fsm_mode}")
    if state.motion_mode:
        bits.append(f"motion={state.motion_mode}")
    if state.motion_code is not None:
        bits.append(f"motion_code={state.motion_code}{error_hint(state.motion_code)}")
    if state.loco_skipped:
        bits.append("loco_rpc=skipped")
    if state.error:
        bits.append(f"error={state.error}")
    return "  ".join(bits)


def state_detail(state: RobotState) -> str:
    data = {
        "mode": state.mode,
        "fsm_id": state.fsm_id,
        "fsm_mode": state.fsm_mode,
        "motion_mode": state.motion_mode,
        "motion_code": state.motion_code,
        "motion_hint": ERROR_HINTS.get(state.motion_code) if state.motion_code is not None else None,
        "loco_rpc": "skipped while motion switcher is released/dev" if state.loco_skipped else "active",
        "motion_raw": state.motion_raw,
        "error": state.error,
    }
    return json.dumps(data, default=str, indent=2, sort_keys=True)


def make_app(controller: ModeController) -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="G1 Mode Control",
    )

    button_specs = [
        ("damping", "Damping", "warning"),
        ("zero_torque", "Zero Torque", "danger"),
        ("prepare", "Prepare", "primary"),
        ("sit", "Sit", "secondary"),
        ("walk", "Walk", "success"),
        ("run", "Run", "success"),
        ("climb", "Climb", "success"),
        ("dev", "Dev Off", "secondary"),
        ("release_arms", "Release Arms", "info"),
        ("stop", "Stop", "danger"),
    ]

    app.layout = html.Div(
        [
            dcc.Interval(id="state-interval", interval=1000, n_intervals=0),
            dcc.Store(id="event-log-store", data=[]),
            dbc.Container(
                dbc.Row(
                    dbc.Col(
                        [
                            html.H3("G1 Mode Control", className="text-center mb-3"),
                            html.Div(id="state-line", className="text-center text-muted mb-4"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Button(
                                            label,
                                            id=f"btn-{name}",
                                            color=color,
                                            className="w-100 mode-btn",
                                            size="lg",
                                            n_clicks=0,
                                        ),
                                        xs=12,
                                        className="mb-2",
                                    )
                                    for name, label, color in button_specs
                                ],
                                className="g-2",
                            ),
                            html.Div(id="command-status", className="text-center mt-4"),
                            html.Pre(id="state-detail", className="state-box mt-3 mb-2"),
                            html.Pre(id="event-log", className="state-box event-log"),
                        ],
                        xs=11,
                        sm=9,
                        md=7,
                        lg=5,
                    ),
                    className="min-vh-100 align-items-center justify-content-center",
                ),
                fluid=True,
            ),
        ]
    )

    app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { background: #f7f7f8; }
            .mode-btn { min-height: 3.25rem; font-weight: 600; }
            #btn-dev {
                opacity: 1;
            }
            .state-box {
                background: #111827;
                border-radius: 6px;
                color: #e5e7eb;
                font-size: 0.8rem;
                margin: 0;
                max-height: 12rem;
                overflow: auto;
                padding: 0.75rem;
                white-space: pre-wrap;
            }
            .event-log { max-height: 8rem; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

    outputs = [
        Output("state-line", "children"),
        Output("command-status", "children"),
        Output("state-detail", "children"),
        Output("event-log", "children"),
        Output("event-log-store", "data"),
        Output("btn-dev", "children"),
        Output("btn-dev", "active"),
        Output("btn-dev", "color"),
    ]
    outputs.extend(Output(f"btn-{name}", "disabled") for name, _, _ in button_specs)

    inputs = [Input("state-interval", "n_intervals")]
    inputs.extend(Input(f"btn-{name}", "n_clicks") for name, _, _ in button_specs)

    @app.callback(
        outputs,
        inputs,
        State("command-status", "children"),
        State("event-log-store", "data"),
        prevent_initial_call=False,
    )
    def update(_: int, *args: Any) -> tuple[Any, ...]:
        prior_status = args[-2] if len(args) >= 2 else ""
        event_log = args[-1] if args else []
        if not isinstance(event_log, list):
            event_log = []
        status = prior_status or ""
        trigger = dash.ctx.triggered_id

        button_names = [name for name, _, _ in button_specs]
        if isinstance(trigger, str) and trigger.startswith("btn-"):
            command_name = trigger[len("btn-"):]
            try:
                current = controller.state()
                if button_disabled(current.mode, command_name):
                    status = f"{command_name.replace('_', ' ').title()} is disabled from {current.mode}."
                else:
                    status = controller.command(command_name)
            except Exception as exc:
                status = f"Command failed: {exc}"
            event_log = [
                f"{time.strftime('%H:%M:%S')} {command_name}: {status}",
                *event_log,
            ][:20]

        current = controller.state()
        disabled = [button_disabled(current.mode, name) for name in button_names]
        dev_active = current.mode == "dev"
        dev_label = "Dev Off" if dev_active else "Dev On"
        dev_color = "dark" if dev_active else "secondary"
        return (
            state_text(current),
            status,
            state_detail(current),
            "\n".join(str(line) for line in event_log),
            event_log,
            dev_label,
            dev_active,
            dev_color,
            *disabled,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dash/DBC mode control for Unitree G1.")
    parser.add_argument("--iface", default=os.environ.get("G1_IFACE", "eth0"),
                        help="DDS network interface.")
    parser.add_argument("--domain-id", type=int,
                        default=int(os.environ.get("G1_DOMAIN_ID", "0")), help="DDS domain ID.")
    parser.add_argument("--timeout", type=float, default=10.0, help="SDK RPC timeout in seconds.")
    parser.add_argument("--host", default=os.environ.get("MODE_CONTROL_HOST",
                        "0.0.0.0"), help="Dash bind host.")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("MODE_CONTROL_PORT", "8051")), help="Dash bind port.")
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode.")
    parser.add_argument(
        "--climb-fsm-id",
        type=int,
        default=DEFAULT_CLIMB_FSM,
        help="FSM id to send for Climb mode. Defaults to G1_CLIMB_FSM_ID or 812.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    controller = ModeController(
        iface=args.iface,
        domain_id=args.domain_id,
        timeout=args.timeout,
        climb_fsm_id=args.climb_fsm_id,
    )
    app = make_app(controller)
    app.run(host=args.host, port=args.port, debug=bool(args.debug))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
