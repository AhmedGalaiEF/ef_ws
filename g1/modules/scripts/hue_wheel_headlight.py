#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import os
import sys
import time
from collections.abc import Iterator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sdk_client import Robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cycle the robot headlight around the HSV hue wheel.")
    parser.add_argument("--intensity", type=int, default=100, help="Headlight intensity 0-100.")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds between consecutive AudioClient.LedControl calls.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=360,
        help="Number of hue steps per full wheel rotation.",
    )
    parser.add_argument(
        "--cycles",
        type=float,
        default=0.0,
        help="Number of hue wheel rotations to run. Use 0 to run until Ctrl-C.",
    )
    parser.add_argument("--iface", default="eth0", help="DDS network interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID.")
    parser.add_argument("--turn-off", action="store_true",
                        help="Turn the headlight off before exiting.")
    return parser.parse_args()


def hue_wheel_rgb(steps: int, intensity: int) -> Iterator[tuple[int, int, int]]:
    steps = max(1, int(steps))
    scale = max(0, min(100, int(intensity))) / 100.0
    while True:
        for index in range(steps):
            red, green, blue = colorsys.hsv_to_rgb(index / steps, 1.0, scale)
            yield (round(red * 255), round(green * 255), round(blue * 255))


def sleep_until(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def main() -> int:
    args = parse_args()
    interval = max(0.0, float(args.interval))
    steps = max(1, int(args.steps))
    total_calls = None if float(args.cycles) <= 0.0 else max(1, round(float(args.cycles) * steps))

    robot = Robot(
        iface=args.iface,
        domain_id=args.domain_id,
        safety_boot=False,
        recover_dev_mode_on_init=False,
        auto_start_sensors=False,
    )
    client = robot._get_audio()._client

    next_call = time.monotonic()
    calls = 0
    last_code = 0
    try:
        for red, green, blue in hue_wheel_rgb(steps, args.intensity):
            if total_calls is not None and calls >= total_calls:
                break
            sleep_until(next_call)
            last_code = int(client.LedControl(red, green, blue))
            print(f"LedControl({red}, {green}, {blue}) returned {last_code}", flush=True)
            calls += 1
            if last_code != 0:
                return last_code
            next_call += interval
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        if args.turn_off:
            code = int(client.LedControl(0, 0, 0))
            print(f"LedControl(0, 0, 0) returned {code}", flush=True)
            if last_code == 0:
                last_code = code

    return int(last_code) if int(last_code) != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
