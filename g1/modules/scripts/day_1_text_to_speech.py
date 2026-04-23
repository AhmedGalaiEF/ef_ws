#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from sdk_client import Robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt for text and play it as speech on the G1 robot."
    )
    parser.add_argument("--iface", default="eth0", help="Network interface for the robot SDK.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument(
        "--safety-boot",
        action="store_true",
        help="Run the hanged safety boot sequence during initialization. Use only when the robot is properly supported.",
    )
    parser.add_argument(
        "--no-safety-boot",
        action="store_true",
        help="Deprecated no-op. Safety boot is skipped by default.",
    )
    parser.add_argument(
        "--volume",
        type=int,
        help="Optional robot playback volume to set before speech playback.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        robot = Robot(
            iface=args.iface,
            domain_id=args.domain_id,
            safety_boot=args.safety_boot,
            auto_start_sensors=True,
        )
    except Exception as exc:
        print(f"Failed to connect to robot: {exc}")
        return 1

    try:
        text = input("Enter text for the robot to speak: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled.")
        return 1

    if not text:
        print("No text provided. Nothing to play.")
        return 1

    try:
        code = robot.say(text, volume=args.volume)
    except Exception as exc:
        print(f"Text-to-speech failed: {exc}")
        return 1

    print(f"Speech command completed with code {code}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
