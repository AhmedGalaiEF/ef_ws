from __future__ import annotations

import argparse
import json
import math
import time

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic usage example for the Go2 Robot wrapper.")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--body-height", type=float, default=0.16)
    args = parser.parse_args()

    robot = Robot(
        iface=args.iface,
        domain_id=args.domain_id,
        auto_start_sensors=True,
    )

    basic_locomotion(robot)
    basic_sensors(robot)
    basic_posture(robot, height=args.body_height)


if __name__ == "__main__":
    main()
