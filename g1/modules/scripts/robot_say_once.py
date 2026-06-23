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


def optional_volume_arg(value: str) -> int:
    volume = int(value)
    if not 0 <= volume <= 100:
        raise argparse.ArgumentTypeError("volume must be between 0 and 100")
    return volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speak one text string through the robot audio client.")
    parser.add_argument("text", help="Text to speak.")
    parser.add_argument("--iface", default="eth0", help="DDS network interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID.")
    parser.add_argument("--volume", type=optional_volume_arg, default=None, help="Optional playback volume 0-100.")
    parser.add_argument("--language", default=None,
                        help="Optional Piper language, for example en, de, fr, es, ar.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from sdk_client import Robot

    robot = Robot(
        iface=args.iface,
        domain_id=args.domain_id,
        safety_boot=False,
        recover_dev_mode_on_init=False,
        auto_start_sensors=False,
    )
    code = robot.say(args.text, volume=args.volume, language=args.language)
    print(f"Robot.say returned {code}")
    return int(code) if int(code) != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
