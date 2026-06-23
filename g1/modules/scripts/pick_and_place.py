#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if TYPE_CHECKING:
    from sdk_client import Robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a teach/repeat helper flow for a pick-and-place style demo."
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("repeat", "teach"),
        default="repeat",
        help="Robot helper action to run. Default: repeat.",
    )
    parser.add_argument("--iface", default="eth0", help="DDS network interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from sdk_client import Robot

    robot = Robot(args.iface, domain_id=args.domain_id)

    if args.action == "teach":
        result = robot.teach()
        print(f"Robot.teach returned {result}")
    else:
        result = robot.repeat()
        print(f"Robot.repeat returned {result}")

    return int(result) if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
