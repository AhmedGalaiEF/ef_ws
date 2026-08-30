#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path
from typing import Any

# Make `from sdk_wrapper import G1` resolve regardless of cwd. This file
# lives in academy/visualizations/; sdk_wrapper.py sits one level up in
# academy/, and viz_util.py sits alongside this file.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ACADEMY_DIR = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _ACADEMY_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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

from viz_util import ServiceRow, SERVICE_STATUS_LABELS, service_error_text, service_rows, resolve_service_name

ROBOT_STATE_API_ID_REPORT_FREQ = 1002


class RobotStateController:
    """Thin adapter from service_view.py's TUI commands onto sdk_wrapper.G1 --
    G1.get_service()/set_service()/set_report_freq() do the actual RPC work
    (see academy/sdk_wrapper.py); this just lazily owns the G1 instance and
    converts its plain-dict rows into ServiceRow for the table."""

    def __init__(self, iface: str, domain_id: int, timeout: float) -> None:
        self.iface = iface
        self.domain_id = int(domain_id)
        self.timeout = float(timeout)
        self._g1: Any | None = None

    @property
    def client_source(self) -> str:
        return "sdk_wrapper.G1" if self._g1 is not None else "not initialized"

    def _ensure_g1(self) -> Any:
        if self._g1 is None:
            from sdk_wrapper import G1  # deferred: only needed once connecting

            self._g1 = G1(iface=self.iface, domain_id=self.domain_id)
            # G1._robot_state_client() hardcodes a 2s RPC timeout; override it
            # with what the user asked for on the command line.
            self._g1._robot_state_client().SetTimeout(self.timeout)
        return self._g1

    def list_services(self) -> tuple[int, list[ServiceRow]]:
        return 0, service_rows(self._ensure_g1())

    def switch(self, name: str, enabled: bool) -> int:
        return int(self._ensure_g1().set_service(name, enabled)["code"])

    def set_report_freq(self, interval: int, duration: int) -> int:
        return int(self._ensure_g1().set_report_freq(interval, duration))


def code_text(code: int | None) -> str:
    return "" if code is None else service_error_text(code)


def status_text(status: int | None) -> Text:
    if status is None:
        return Text("UNKNOWN", style="dim")
    label, style = SERVICE_STATUS_LABELS.get(status, (str(status), "yellow"))
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
                name = resolve_service_name(parts[1], rows)
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
    parser.add_argument("--iface", default="eth0", help="DDS network interface.")
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
