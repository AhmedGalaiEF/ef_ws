#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sdk_audio import parse_color, scale_color
from sdk_client import Robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistently hold the robot headlight at one RGB color.")
    parser.add_argument("color", help="RGB color as R,G,B, for example 255,0,0.")
    parser.add_argument("brightness", type=int, help="Brightness 0-100.")
    parser.add_argument("duration", type=float, help="Duration in seconds.")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds between repeated AudioClient.LedControl calls.",
    )
    parser.add_argument("--iface", default="eth0", help="DDS network interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID.")
    parser.add_argument("--turn-off", action="store_true", help="Turn the headlight off before exiting.")
    return parser.parse_args()


class HeadlightKeeper(threading.Thread):
    def __init__(
        self,
        client: object,
        rgb: tuple[int, int, int],
        duration: float,
        interval: float,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=False)
        self.client = client
        self.rgb = rgb
        self.duration = max(0.0, float(duration))
        self.interval = max(0.0, float(interval))
        self.stop_event = stop_event
        self.last_code = 0

    def run(self) -> None:
        end_time = time.monotonic() + self.duration
        next_call = time.monotonic()
        while not self.stop_event.is_set() and time.monotonic() < end_time:
            remaining = next_call - time.monotonic()
            if remaining > 0 and self.stop_event.wait(remaining):
                break

            self.last_code = int(self.client.LedControl(*self.rgb))
            print(f"LedControl({self.rgb[0]}, {self.rgb[1]}, {self.rgb[2]}) returned {self.last_code}", flush=True)
            if self.last_code != 0:
                self.stop_event.set()
                break
            next_call += self.interval


def main() -> int:
    args = parse_args()
    rgb = scale_color(parse_color(str(args.color)), int(args.brightness))

    robot = Robot(
        iface=args.iface,
        domain_id=args.domain_id,
        safety_boot=False,
        recover_dev_mode_on_init=False,
        auto_start_sensors=False,
    )
    client = robot._get_audio()._client

    stop_event = threading.Event()
    keeper = HeadlightKeeper(client, rgb, args.duration, args.interval, stop_event)
    keeper.start()
    try:
        keeper.join()
    except KeyboardInterrupt:
        print("Interrupted")
        stop_event.set()
        keeper.join()
    finally:
        if args.turn_off:
            code = int(client.LedControl(0, 0, 0))
            print(f"LedControl(0, 0, 0) returned {code}", flush=True)
            if keeper.last_code == 0:
                keeper.last_code = code

    return int(keeper.last_code) if int(keeper.last_code) != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
