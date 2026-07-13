from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import TYPE_CHECKING

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if TYPE_CHECKING:
    from sdk_client import Robot


def print_section(title: str, payload) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, (dict, list, tuple)):
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(payload)


def basic_locomotion(robot: Robot) -> None:
    print_section("Locomotion", "release mode -> stand -> walk -> turn -> stop")
    released = robot.release_active_mode()
    stand_up = robot.stand_up()
    time.sleep(1.0)
    balance = robot.balance_stand()
    walked = robot.walk_for(distance=0.3, speed=0.25)
    turned = robot.turn_for(angle_rad=math.radians(20.0), yaw_rate=0.4)
    stop_code = robot.stop()
    print_section(
        "Locomotion Result",
        {
            "released": released,
            "stand_up": stand_up,
            "balance_stand": balance,
            "walked": walked,
            "turned": turned,
            "stop": stop_code,
        },
    )


def basic_sensors(robot: Robot) -> None:
    time.sleep(0.5)
    print_section("Sensors", robot.get_robot_state())


def basic_posture(robot: Robot, height: float) -> None:
    code = robot.set_body_height(height)
    print_section("Body Height", {"requested_height_m": height, "code": code})


def confirm_motion(args: argparse.Namespace) -> None:
    if args.yes:
        return
    selected_motion = args.locomotion or args.posture or args.all
    if not selected_motion:
        return
    print(
        "This example can move the robot. Re-run with --yes to confirm, "
        "or select --sensors for a read-only check."
    )
    raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Basic usage example for the Go2 Robot wrapper.")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--body-height", type=float, default=0.16)
    parser.add_argument("--yes", action="store_true", help="Confirm that motion-capable examples may run.")
    parser.add_argument("--all", action="store_true", help="Run all examples.")
    parser.add_argument("--locomotion", action="store_true", help="Run the locomotion example.")
    parser.add_argument("--sensors", action="store_true", help="Print the current robot state.")
    parser.add_argument("--posture", action="store_true", help="Run the body-height example.")
    args = parser.parse_args()

    if not any((args.all, args.locomotion, args.sensors, args.posture)):
        args.sensors = True

    confirm_motion(args)

    from sdk_client import Robot

    robot = Robot(
        iface=args.iface,
        domain_id=args.domain_id,
        auto_start_sensors=True,
    )

    try:
        if args.all or args.locomotion:
            basic_locomotion(robot)
        if args.all or args.sensors:
            basic_sensors(robot)
        if args.all or args.posture:
            basic_posture(robot, height=args.body_height)
    finally:
        if args.all or args.locomotion:
            try:
                robot.stop()
            except Exception as exc:
                print_section("Stop Warning", str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
