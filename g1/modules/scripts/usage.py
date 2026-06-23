#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import TYPE_CHECKING

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    print_section("Locomotion", "balanced stand -> walk -> turn -> stop")
    robot.balanced_stand()
    time.sleep(1.0)
    walked = robot.walk_for(distance=0.3, timeout=8.0)
    turned = robot.turn_for(angle_deg=20.0, timeout=6.0)
    robot.stop()
    print_section("Locomotion Result", {"walked": walked, "turned": turned})


def basic_arm_motion(robot: Robot, hand: str) -> None:
    print_section("Arm Motion", f"{hand} hand open -> close")
    robot.hand_open(hand=hand, hold_s=0.5)
    time.sleep(0.4)
    robot.hand_close(hand=hand, hold_s=0.5)


def basic_sensors(robot: Robot) -> None:
    time.sleep(0.5)
    state = robot.get_robot_state()
    imu = robot.get_imu()
    odom_pose = robot.get_odom_pose()
    lidar_points = robot.get_lidar_points(max_points=16)
    print_section(
        "Sensors",
        {
            "state": state,
            "imu": imu,
            "odom_pose": odom_pose,
            "lidar_sample_count": len(lidar_points),
            "lidar_sample_preview": lidar_points[:5],
        },
    )


def basic_text_to_speech(robot: Robot) -> None:
    code = robot.say("Hello from the G1 SDK Robot usage example.")
    print_section("Text To Speech", {"code": code})


def basic_slam(robot: Robot, slam_type: str, save_path: str | None) -> None:
    code = robot.start_slam(slam_type=slam_type)
    time.sleep(1.0)
    pose = robot.get_slam_pose(timeout_s=1.5)
    print_section("SLAM Start", {"code": code, "pose": pose})
    stop_code = robot.stop_slam(save_path=save_path)
    print_section("SLAM Stop", {"code": stop_code, "save_path": save_path})


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic usage example for the G1 Robot wrapper.")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--no-safety-boot", action="store_true")
    parser.add_argument("--skip-slam", action="store_true")
    parser.add_argument("--slam-type", default="indoor")
    parser.add_argument("--slam-save-path")
    args = parser.parse_args()
    from sdk_client import Robot

    robot = Robot(
        iface=args.iface,
        domain_id=args.domain_id,
        safety_boot=not args.no_safety_boot,
        auto_start_sensors=True,
    )

    basic_locomotion(robot)
    basic_arm_motion(robot, hand=args.hand)
    basic_sensors(robot)
    basic_text_to_speech(robot)

    if not args.skip_slam:
        basic_slam(robot, slam_type=args.slam_type, save_path=args.slam_save_path)


if __name__ == "__main__":
    main()
