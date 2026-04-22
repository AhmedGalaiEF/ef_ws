#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from sdk_client import Robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Basic locomotion examples for the G1 robot."
    )
    parser.add_argument("--iface", default="eth0", help="Network interface for the robot SDK.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument(
        "--no-safety-boot",
        action="store_true",
        help="Skip the robot safety boot sequence during initialization.",
    )
    parser.add_argument(
        "--linear-vx",
        type=float,
        default=0.2,
        help="Forward velocity for the timed motion example in m/s.",
    )
    parser.add_argument(
        "--linear-vy",
        type=float,
        default=0.0,
        help="Lateral velocity for the timed motion example in m/s.",
    )
    parser.add_argument(
        "--move-seconds",
        type=float,
        default=2.0,
        help="How long to apply the timed linear velocity command.",
    )
    parser.add_argument(
        "--turn-vyaw",
        type=float,
        default=0.5,
        help="Angular velocity for the timed turn example in rad/s.",
    )
    parser.add_argument(
        "--turn-seconds",
        type=float,
        default=2.0,
        help="How long to apply the timed turn command.",
    )
    parser.add_argument(
        "--walk-distance",
        type=float,
        default=0.3,
        help="Distance in meters for the feedback-based walk_for example.",
    )
    parser.add_argument(
        "--turn-angle-deg",
        type=float,
        default=20.0,
        help="Angle in degrees for the feedback-based turn_for example.",
    )
    parser.add_argument(
        "--skip-strafe",
        action="store_true",
        help="Skip the lateral strafe example.",
    )
    return parser.parse_args()


def print_section(title: str, payload: object) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, (dict, list, tuple)):
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(payload)


def run_timed_velocity(
    robot: Robot,
    *,
    label: str,
    vx: float = 0.0,
    vy: float = 0.0,
    vyaw: float = 0.0,
    duration_s: float = 1.0,
) -> None:
    duration_s = max(0.0, float(duration_s))
    print_section(
        label,
        {
            "vx_mps": vx,
            "vy_mps": vy,
            "vyaw_radps": vyaw,
            "duration_s": duration_s,
        },
    )
    robot.walk(vx=vx, vy=vy, vyaw=vyaw)
    time.sleep(duration_s)
    robot.stop()
    time.sleep(0.75)


def basic_locomotion_examples(robot: Robot, args: argparse.Namespace) -> None:
    print_section("Posture", "balanced stand")
    robot.balanced_stand()
    time.sleep(1.5)

    run_timed_velocity(
        robot,
        label="Timed Linear Motion",
        vx=args.linear_vx,
        vy=args.linear_vy,
        duration_s=args.move_seconds,
    )

    run_timed_velocity(
        robot,
        label="Timed Turn",
        vyaw=args.turn_vyaw,
        duration_s=args.turn_seconds,
    )

    if not args.skip_strafe:
        run_timed_velocity(
            robot,
            label="Timed Strafe",
            vy=0.12,
            duration_s=1.5,
        )

    walked = robot.walk_for(distance=args.walk_distance, timeout=8.0)
    print_section(
        "walk_for Example",
        {"distance_m": args.walk_distance, "completed": walked},
    )
    time.sleep(0.75)

    turned = robot.turn_for(angle_deg=args.turn_angle_deg, timeout=6.0)
    print_section(
        "turn_for Example",
        {"angle_deg": args.turn_angle_deg, "completed": turned},
    )

    robot.stop()
    print_section("Stop", "robot stop command sent")


def main() -> int:
    args = parse_args()

    try:
        robot = Robot(
            iface=args.iface,
            domain_id=args.domain_id,
            safety_boot=not args.no_safety_boot,
            auto_start_sensors=True,
        )
    except Exception as exc:
        print(f"Failed to connect to robot: {exc}")
        return 1

    try:
        basic_locomotion_examples(robot, args)
    except KeyboardInterrupt:
        robot.stop()
        print("\nInterrupted. Stop command sent.")
        return 1
    except Exception as exc:
        robot.stop()
        print(f"Locomotion example failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
