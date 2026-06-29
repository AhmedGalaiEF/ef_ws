#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import shlex
import sys
import time
from dataclasses import dataclass
from typing import Any

try:
    from rich import box
    from rich.align import Align
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.text import Text
except ImportError as exc:
    raise SystemExit(
        "rich is required for service_view.py.\n"
        "Install it with:\n"
        "  pip install rich"
    ) from exc

try:
    from dds_env import default_dds_iface, ensure_channel_factory_initialized
except ImportError as exc:
    raise SystemExit("Local dds_env.py helper is required for service_view.py.") from exc


SERVICE_CATALOG = {
    "ai_sport": "Main Motion Control Service",
    "basic_service": "Basic Service",
    "g1_arm_example": "Upper Limb Motion Service",
    "vui_service": "Audio and Lighting Control Service",
    "unitree_slam": "Navigation Service",
}

ROBOT_STATE_API_ID_REPORT_FREQ = 1002

ERROR_HINTS = {
    0: "success",
    3001: "RPC unknown error",
    3102: "RPC client send error",
    3103: "RPC API not registered",
    3104: "RPC timeout",
    3105: "RPC API mismatch",
    3106: "RPC client data error",
    3201: "RPC server send error",
    3202: "RPC server internal error",
    3203: "RPC API not implemented",
    3204: "RPC server parameter error",
    5201: "service switch execution error",
    5202: "service is protected",
}

STATUS_TEXT = {
    0: ("ON", "green"),
    1: ("OFF", "red"),
    5: ("PROTECTED", "yellow"),
}


@dataclass(frozen=True)
class ServiceRow:
    name: str
    description: str
    status: int | None = None
    protected: bool | None = None


class RobotStateController:
    def __init__(self, iface: str, domain_id: int, timeout: float) -> None:
        self.iface = iface
        self.domain_id = int(domain_id)
        self.timeout = float(timeout)
        self._client: Any | None = None
        self._client_source = ""

    @property
    def client_source(self) -> str:
        return self._client_source or "not initialized"

    def _load_client_type(self) -> tuple[type[Any], str]:
        errors: list[str] = []
        for module_name in (
            "unitree_sdk2py.b2.robot_state.robot_state_client",
            "unitree_sdk2py.go2.robot_state.robot_state_client",
        ):
            try:
                module = importlib.import_module(module_name)
                return module.RobotStateClient, module_name
            except ModuleNotFoundError as exc:
                errors.append(f"{module_name}: {exc}")
            except ImportError as exc:
                errors.append(f"{module_name}: {exc}")
        details = "\n  ".join(errors) if errors else "no candidate modules found"
        raise RuntimeError(f"RobotStateClient could not be imported:\n  {details}")

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        ensure_channel_factory_initialized(self.domain_id, self.iface)
        client_type, source = self._load_client_type()
        client = client_type()
        if hasattr(client, "SetTimeout"):
            client.SetTimeout(self.timeout)
        client.Init()
        self._client = client
        self._client_source = source
        return client

    def list_services(self) -> tuple[int, list[ServiceRow]]:
        client = self._ensure_client()
        if not hasattr(client, "ServiceList"):
            rows = [
                ServiceRow(name=name, description=description)
                for name, description in SERVICE_CATALOG.items()
            ]
            return 0, rows

        code, service_states = client.ServiceList()
        if int(code) != 0:
            return int(code), []

        rows: list[ServiceRow] = []
        for state in service_states or []:
            name = str(getattr(state, "name", "")).strip()
            if not name:
                continue
            rows.append(
                ServiceRow(
                    name=name,
                    description=SERVICE_CATALOG.get(name, ""),
                    status=_to_optional_int(getattr(state, "status", None)),
                    protected=_to_optional_bool(getattr(state, "protect", None)),
                )
            )

        known = {row.name for row in rows}
        for name, description in SERVICE_CATALOG.items():
            if name not in known:
                rows.append(ServiceRow(name=name, description=description))

        return 0, rows

    def switch(self, name: str, enabled: bool) -> int:
        client = self._ensure_client()
        return int(client.ServiceSwitch(name, bool(enabled)))

    def set_report_freq(self, interval: int, duration: int) -> int:
        client = self._ensure_client()
        if hasattr(client, "_Call"):
            parameter = json.dumps({"interval": int(interval), "duration": int(duration)})
            code, _data = client._Call(ROBOT_STATE_API_ID_REPORT_FREQ, parameter)
            return int(code)
        return int(client.SetReportFreq(int(interval), int(duration)))


def _to_optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _to_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def code_text(code: int | None) -> str:
    if code is None:
        return ""
    hint = ERROR_HINTS.get(int(code), "unknown")
    return f"{code}: {hint}"


def status_text(status: int | None) -> Text:
    if status is None:
        return Text("UNKNOWN", style="dim")
    label, style = STATUS_TEXT.get(status, (str(status), "yellow"))
    return Text(label, style=style)


def build_table(rows: list[ServiceRow]) -> Table:
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Service", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Protected", no_wrap=True)
    table.add_column("Description")

    for index, row in enumerate(rows, start=1):
        if row.protected is True:
            protected = Text("yes", style="yellow")
        elif row.protected is False:
            protected = Text("no", style="dim")
        else:
            protected = Text("unknown", style="dim")
        table.add_row(
            str(index),
            row.name,
            status_text(row.status),
            protected,
            row.description or Text("not in G1 docs", style="dim"),
        )
    return table


def build_screen(
    rows: list[ServiceRow],
    iface: str,
    domain_id: int,
    client_source: str,
    last_result: str,
) -> Group:
    title = Text("G1 Robot Service View", style="bold")
    subtitle = Text(
        f"iface={iface}  domain={domain_id}  client={client_source}",
        style="dim",
    )
    header = Panel(Align.left(Group(title, subtitle)), box=box.ROUNDED)

    commands = Table.grid(padding=(0, 2))
    commands.add_column(style="cyan", no_wrap=True)
    commands.add_column()
    commands.add_row("r", "refresh service list")
    commands.add_row("on <n|name>", "turn a service on")
    commands.add_row("off <n|name>", "turn a service off")
    commands.add_row("freq <interval> <duration>", "set report frequency in seconds")
    commands.add_row("q", "quit")

    result_text = Text(last_result or "Ready.", style="green" if "code=0" in last_result else "")
    footer = Panel(Group(commands, Text(""), result_text), title="Commands", box=box.ROUNDED)
    return Group(header, build_table(rows), footer)


def resolve_service(token: str, rows: list[ServiceRow]) -> str:
    value = token.strip()
    if not value:
        raise ValueError("missing service name or table number")
    if value.isdigit():
        index = int(value)
        if not 1 <= index <= len(rows):
            raise ValueError(f"service number {index} is outside the table")
        return rows[index - 1].name

    for row in rows:
        if row.name == value:
            return row.name
    for row in rows:
        if row.name.lower() == value.lower():
            return row.name
    raise ValueError(f"unknown service: {value}")


def run_once(controller: RobotStateController, console: Console) -> int:
    code, rows = controller.list_services()
    console.print(build_screen(rows, controller.iface, controller.domain_id, controller.client_source, code_text(code)))
    return 0 if code == 0 else 1


def run_tui(controller: RobotStateController, console: Console) -> int:
    rows: list[ServiceRow] = []
    last_result = ""

    while True:
        try:
            code, rows = controller.list_services()
            if code != 0:
                last_result = f"ServiceList failed: {code_text(code)}"
        except Exception as exc:
            code = -1
            last_result = f"{type(exc).__name__}: {exc}"

        console.clear()
        console.print(
            build_screen(
                rows,
                controller.iface,
                controller.domain_id,
                controller.client_source,
                last_result,
            )
        )

        try:
            raw = Prompt.ask("service_view").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0

        if not raw:
            continue
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            last_result = str(exc)
            continue

        command = parts[0].lower()
        try:
            if command in {"q", "quit", "exit"}:
                return 0
            if command in {"r", "refresh"}:
                last_result = f"Refreshed at {time.strftime('%H:%M:%S')}."
                continue
            if command in {"on", "off"}:
                if len(parts) != 2:
                    raise ValueError(f"usage: {command} <service-number|service-name>")
                name = resolve_service(parts[1], rows)
                switch_code = controller.switch(name, command == "on")
                last_result = f"{name} {command}: code={code_text(switch_code)}"
                continue
            if command == "freq":
                if len(parts) != 3:
                    raise ValueError("usage: freq <interval-sec> <duration-sec>")
                interval = int(parts[1])
                duration = int(parts[2])
                if interval <= 0 or duration <= 0:
                    raise ValueError("interval and duration must be positive seconds")
                freq_code = controller.set_report_freq(interval, duration)
                last_result = f"SetReportFreq interval={interval} duration={duration}: code={code_text(freq_code)}"
                continue
            last_result = f"unknown command: {command}"
        except Exception as exc:
            last_result = f"{type(exc).__name__}: {exc}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rich TUI for Unitree G1 robot_state service status and switches.",
    )
    parser.add_argument(
        "--iface",
        default=default_dds_iface("eth0"),
        help="DDS network interface. Defaults to eth0 when it is up, otherwise the first live NIC.",
    )
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument("--timeout", type=float, default=2.0, help="RPC timeout in seconds.")
    parser.add_argument("--once", action="store_true", help="Print the service table once and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    console = Console()
    controller = RobotStateController(args.iface, args.domain_id, args.timeout)
    if args.once:
        return run_once(controller, console)
    return run_tui(controller, console)


if __name__ == "__main__":
    raise SystemExit(main())
