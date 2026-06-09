#!/usr/bin/env python3
from __future__ import annotations
from sdk_client import Robot

import argparse
import os
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk forward for 5 seconds, then turn 90 degrees."
    )
    parser.add_argument("--iface", default="eth0", help="Network interface for the robot SDK.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument(
        "--forward-speed",
        type=float,
        default=0.3,
        help="Forward walking speed in m/s.",
    )
    parser.add_argument(
        "--forward-seconds",
        type=float,
        default=5.0,
        help="How long to walk forward.",
    )
    parser.add_argument(
        "--turn-angle-deg",
        type=float,
        default=90.0,
        help="Turn angle in degrees. Positive is counter-clockwise.",
    )
    parser.add_argument(
        "--turn-timeout",
        type=float,
        default=10.0,
        help="Timeout for the 90 degree turn.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        robot = Robot(
            iface=args.iface,
            domain_id=args.domain_id,
            safety_boot=False,
            auto_start_sensors=True,
        )
    except Exception as exc:
        print(f"Failed to connect to robot: {exc}")
        return 1

    try:
        print("Standing in balanced mode...")
        robot.balanced_stand()
        time.sleep(1.5)

        print(
            f"Walking forward at {args.forward_speed:.2f} m/s "
            f"for {args.forward_seconds:.2f} seconds..."
        )
        robot.walk(vx=args.forward_speed, vy=0.0, vyaw=0.0)
        time.sleep(max(0.0, float(args.forward_seconds)))
        robot.stop()
        time.sleep(0.75)

        print(f"Turning {args.turn_angle_deg:.1f} degrees...")
        turned = robot.turn_for(
            angle_deg=args.turn_angle_deg,
            timeout=args.turn_timeout,
        )
        print(f"Turn completed: {turned}")
    except KeyboardInterrupt:
        print("\nInterrupted. Sending stop command.")
        robot.stop()
        return 1
    except Exception as exc:
        print(f"Motion sequence failed: {exc}")
        robot.stop()
        return 1

    robot.stop()
    print("Sequence complete. Stop command sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
