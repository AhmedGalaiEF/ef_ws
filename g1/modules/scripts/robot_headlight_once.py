#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sdk_client import Robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set the robot headlight color once.")
    parser.add_argument("--color", default="#123456", help="Headlight color name, #RRGGBB, or R,G,B.")
    parser.add_argument("--intensity", type=int, default=100, help="Headlight intensity 0-100.")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration in seconds before turning off.")
    parser.add_argument("--iface", default="eth0", help="DDS network interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robot = Robot(
        iface=args.iface,
        domain_id=args.domain_id,
        safety_boot=False,
        recover_dev_mode_on_init=False,
        auto_start_sensors=False,
    )
    code = robot.headlight(
        color=str(args.color),
        intensity=max(0, min(100, int(args.intensity))),
        duration=args.duration,
    )
    print(f"Robot.headlight returned {code}")
    return int(code) if int(code) != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
