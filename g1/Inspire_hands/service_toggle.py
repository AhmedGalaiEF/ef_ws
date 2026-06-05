#!/usr/bin/env python3
"""
Toggle Unitree G1 robot services/modes by name.

This wraps Unitree's MotionSwitcherClient, which is exposed on ROS2 as:
  /api/motion_switcher/request
  /api/motion_switcher/response

Examples:
  ./service_toggle.py status
  ./service_toggle.py on ai_sport
  ./service_toggle.py off ai_sport
  ./service_toggle.py toggle ai_sport
  ./service_toggle.py on normal --iface eth0
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Sequence


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

MODE_ALIASES = {
    "ai": "ai_sport",
}

MODE_EQUIVALENTS = {
    canonical: {alias, canonical}
    for alias, canonical in MODE_ALIASES.items()
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Toggle Unitree G1 MotionSwitcher services/modes by name."
    )
    parser.add_argument(
        "action",
        choices=("on", "off", "toggle", "status", "select", "release"),
        help="Action to run. 'select' is an alias for 'on'; 'release' is an alias for 'off'.",
    )
    parser.add_argument(
        "name",
        nargs="?",
        help=(
            "Motion/service mode name, for example ai_sport, normal, advanced, or stand. "
            "Unknown names are passed through unchanged."
        ),
    )
    parser.add_argument(
        "--iface",
        help="Network interface for Unitree DDS, for example eth0 or enx....",
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=0,
        help="DDS domain id passed to ChannelFactoryInitialize. Default: 0.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Motion switcher RPC timeout in seconds. Default: 5.0.",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=0.5,
        help="Seconds to wait before reading status after a change. Default: 0.5.",
    )
    parser.add_argument(
        "--force-release",
        action="store_true",
        help="Release the active mode even when it does not match NAME.",
    )
    return parser.parse_args(argv)


def init_motion_switcher(args: argparse.Namespace) -> Any:
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )
    except ImportError as exc:
        raise SystemExit(
            "error: unitree_sdk2py is not importable. Activate the Unitree SDK environment first."
        ) from exc

    if args.iface:
        ChannelFactoryInitialize(args.domain_id, args.iface)
    else:
        ChannelFactoryInitialize(args.domain_id)

    client = MotionSwitcherClient()
    client.SetTimeout(args.timeout)
    client.Init()
    return client


def result_hint(code: int) -> str:
    hint = ERROR_HINTS.get(code)
    return f" ({hint})" if hint else ""


def check_mode(client: Any) -> tuple[int, dict[str, Any] | None]:
    code, data = client.CheckMode()
    if data is not None and not isinstance(data, dict):
        data = {"raw": data}
    return int(code), data


def current_name(data: dict[str, Any] | None) -> str:
    if not data:
        return ""
    name = data.get("name")
    return str(name) if name is not None else ""


def canonical_mode_name(name: str) -> str:
    return MODE_ALIASES.get(name, name)


def mode_names_match(left: str, right: str) -> bool:
    left = canonical_mode_name(left)
    right = canonical_mode_name(right)
    if left == right:
        return True

    left_equivalents = MODE_EQUIVALENTS.get(left, {left})
    right_equivalents = MODE_EQUIVALENTS.get(right, {right})
    return bool(left_equivalents & right_equivalents)


def print_status(prefix: str, code: int, data: dict[str, Any] | None) -> None:
    name = current_name(data)
    form = "" if not data else str(data.get("form", ""))
    detail = f"name={name or '<released>'}"
    if form:
        detail += f" form={form}"
    print(f"{prefix}: code={code}{result_hint(code)} {detail}")


def select_mode(client: Any, name: str) -> int:
    code, _data = client.SelectMode(name)
    code = int(code)
    print(f"SelectMode({name!r}): code={code}{result_hint(code)}")
    return code


def release_mode(client: Any) -> int:
    code, _data = client.ReleaseMode()
    code = int(code)
    print(f"ReleaseMode(): code={code}{result_hint(code)}")
    return code


def require_name(args: argparse.Namespace) -> str:
    if not args.name:
        raise SystemExit(f"error: action '{args.action}' requires NAME")
    name = canonical_mode_name(args.name)
    if name != args.name:
        print(f"using mode name {name!r} for alias {args.name!r}.")
    return name


def run(args: argparse.Namespace) -> int:
    action = {"select": "on", "release": "off"}.get(args.action, args.action)
    client = init_motion_switcher(args)

    before_code, before_data = check_mode(client)
    print_status("before", before_code, before_data)
    if before_code != 0:
        return before_code

    before_name = current_name(before_data)

    if action == "status":
        return 0

    name = require_name(args)
    result_code = 0

    if action == "on":
        if mode_names_match(before_name, name):
            print(f"{name!r} is already active.")
        else:
            result_code = select_mode(client, name)
    elif action == "off":
        if mode_names_match(before_name, name) or args.force_release:
            result_code = release_mode(client)
        elif before_name:
            print(
                f"active mode is {before_name!r}, not {name!r}; "
                "use --force-release to release it anyway."
            )
            result_code = 2
        else:
            print("no active mode to release.")
    elif action == "toggle":
        if before_name == name:
            result_code = release_mode(client)
        else:
            result_code = select_mode(client, name)
    else:
        raise SystemExit(f"error: unsupported action {action!r}")

    if args.wait > 0:
        time.sleep(args.wait)
    after_code, after_data = check_mode(client)
    print_status("after", after_code, after_data)
    return result_code or after_code


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
