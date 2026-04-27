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
        description="Show basic FSM switching examples for the G1 robot."
    )
    parser.add_argument("--iface", default="eth0", help="Network interface for the robot SDK.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.0,
        help="Pause after each FSM transition so the state can settle.",
    )
    parser.add_argument(
        "--run-hanged-boot",
        action="store_true",
        help="Run the full hanged boot sequence. Use only when the robot is properly supported on the hanger.",
    )
    return parser.parse_args()


def print_section(title: str, payload: object) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, (dict, list, tuple)):
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(payload)


def show_fsm(robot: Robot, label: str) -> None:
    print_section(
        label,
        {
            "fsm": robot.get_fsm(),
            "mode": robot.get_mode(),
            "gait": robot.get_gait(),
        },
    )


def switch_to_preparation(robot: Robot) -> None:
    if hasattr(robot._client, "SetFsmId"):
        robot._client.SetFsmId(4)
        return
    raise AttributeError("Current locomotion client does not support SetFsmId(4) for Preparation/HangedBoot.")


def main() -> int:
    args = parse_args()
    pause_s = max(0.0, float(args.sleep_seconds))

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
        show_fsm(robot, "Initial FSM")

        print_section("Switch", "ZeroTorque")
        robot.fsm_0_zt()
        time.sleep(pause_s)
        show_fsm(robot, "After ZeroTorque")

        print_section("Switch", "Damping")
        robot.fsm_1_damp()
        time.sleep(pause_s)
        show_fsm(robot, "After Damping")

        print_section("Switch", "Preparation / HangedBoot (SetFsmId(4))")
        switch_to_preparation(robot)
        time.sleep(pause_s)
        show_fsm(robot, "After Preparation")

        if args.run_hanged_boot:
            print_section(
                "Switch",
                "Running full hanged boot sequence. Ensure the robot is correctly supported on the hanger.",
            )
            robot.hanged_boot()
            time.sleep(pause_s)
            show_fsm(robot, "After HangedBoot Sequence")
        else:
            print_section(
                "HangedBoot",
                "Skipped full hanged boot sequence. Re-run with --run-hanged-boot when the robot is on the hanger.",
            )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    except Exception as exc:
        print(f"FSM example failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
