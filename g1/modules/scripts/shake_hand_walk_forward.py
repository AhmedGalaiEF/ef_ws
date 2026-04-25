#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from sdk_client import Robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hanger boot, shake hands, then walk forward briefly."
    )
    parser.add_argument("--iface", default="eth0", help="Robot network interface.")
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
        default=2.0,
        help="How long to walk forward after shaking hands.",
    )
    parser.add_argument(
        "--extend-delay",
        type=float,
        default=0.75,
        help="Delay after the first ShakeHand call before walking forward.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the safety confirmation prompt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("WARNING: Please ensure there are no obstacles around the robot.")
    if not args.yes:
        input("Press Enter to run hanger boot, shake hands, then walk forward...")

    try:
        robot = Robot(
            iface=args.iface,
            domain_id=args.domain_id,
            safety_boot=True,
            auto_start_sensors=False,
        )
    except Exception as exc:
        print(f"Failed to boot/connect to robot: {exc}")
        return 1

    try:
        loco = robot._client
        if not hasattr(loco, "ShakeHand"):
            raise AttributeError("Current locomotion client does not support ShakeHand().")
        if not hasattr(loco, "Move"):
            raise AttributeError("Current locomotion client does not support Move().")

        print("Running loco client id 11: extend hand.")
        loco.ShakeHand()
        time.sleep(max(0.0, float(args.extend_delay)))

        print(
            f"Walking forward at {args.forward_speed:.2f} m/s "
            f"for {args.forward_seconds:.2f} seconds while hand is extended."
        )
        loco.Move(float(args.forward_speed), 0.0, 0.0, continous_move=True)
        time.sleep(max(0.0, float(args.forward_seconds)))
        robot.stop()
        time.sleep(0.25)

        print("Retracting hand with second loco client id 11 call.")
        loco.ShakeHand()
        time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nInterrupted. Sending stop command.")
        robot.stop()
        return 1
    except Exception as exc:
        print(f"Sequence failed: {exc}")
        robot.stop()
        return 1
    finally:
        try:
            robot.stop()
        except Exception:
            pass

    print("Sequence complete. Stop command sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
