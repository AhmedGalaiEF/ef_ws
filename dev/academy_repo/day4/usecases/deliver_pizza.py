#!/usr/bin/env python3
"""
deliver_pizza.py
================

Sequence:
1) Navigate to point A (pickup/start point).
2) Move both arms into a "hold pizza" front pose.
3) Wait while blinking headlight to signal "place pizza now".
4) Navigate to point B (delivery point).
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from typing import Iterable

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

ARM_MOTION_DIR = os.path.join(SCRIPTS_DIR, "arm_motion")
if ARM_MOTION_DIR not in sys.path:
    sys.path.insert(0, ARM_MOTION_DIR)

from arm_motion import ArmSdkController, JOINT_INDEX  # type: ignore


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pizza delivery sequence: navigate -> hold pose -> wait -> navigate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--map", required=True, help="Map path for dynamic path navigation (.npz/.pcd/.ply)")
    p.add_argument("--iface", default="eth0", help="Network interface for DDS")
    p.add_argument("--domain-id", type=int, default=0, help="DDS domain id for navigation process")
    p.add_argument("--start-x", type=float, required=True, help="Pickup point A x (m)")
    p.add_argument("--start-y", type=float, required=True, help="Pickup point A y (m)")
    p.add_argument("--goal-x", type=float, required=True, help="Delivery point B x (m)")
    p.add_argument("--goal-y", type=float, required=True, help="Delivery point B y (m)")
    p.add_argument("--wait-seconds", type=float, default=20.0, help="Wait time at pickup point")
    p.add_argument("--blink-period", type=float, default=0.6, help="Headlight blink period while waiting")
    p.add_argument("--waiting-color", default="yellow", help="Headlight color while waiting")
    p.add_argument("--idle-color", default="white", help="Headlight color between waiting blinks")
    p.add_argument("--end-color", default="green", help="Headlight color when sequence completes")
    p.add_argument("--arm-cmd-hz", type=float, default=50.0, help="Arm command frequency")
    p.add_argument("--arm-kp", type=float, default=45.0, help="Arm kp")
    p.add_argument("--arm-kd", type=float, default=1.2, help="Arm kd")
    p.add_argument("--arm-duration", type=float, default=1.8, help="Seconds to ramp into hold pose")
    p.add_argument("--arm-hold-seed", type=float, default=0.3, help="Initial hold after ramp (s)")
    p.add_argument(
        "--nav-extra-args",
        default="--smooth --allow-diagonal --use-live-map",
        help="Extra args forwarded to real_time_path_steps_dynamic.py",
    )
    return p.parse_args()


def _nav_script_path() -> str:
    return os.path.join(SCRIPTS_DIR, "navigation", "obstacle_avoidance", "real_time_path_steps_dynamic.py")


def _headlight_script_path() -> str:
    return os.path.join(SCRIPTS_DIR, "basic", "headlight_client", "headlight.py")


def _build_nav_cmd(args: argparse.Namespace, gx: float, gy: float) -> list[str]:
    cmd = [
        sys.executable,
        _nav_script_path(),
        "--map",
        args.map,
        "--iface",
        args.iface,
        "--domain-id",
        str(args.domain_id),
        "--goal-x",
        str(gx),
        "--goal-y",
        str(gy),
    ]
    if args.nav_extra_args:
        cmd.extend(shlex.split(args.nav_extra_args))
    return cmd


def _run_checked(cmd: list[str], label: str) -> None:
    print(f"[{label}] {' '.join(shlex.quote(c) for c in cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def _set_headlight(iface: str, color: str, intensity: int = 80) -> None:
    cmd = [
        sys.executable,
        _headlight_script_path(),
        "--iface",
        iface,
        "--color",
        color,
        "--intensity",
        str(intensity),
    ]
    subprocess.run(cmd, check=False)


def _indexed_pose(arm: str, pose_deg: dict[str, float]) -> list[tuple[int, float]]:
    import math

    idx = JOINT_INDEX[arm]
    return [(idx[name], math.radians(float(deg))) for name, deg in pose_deg.items()]


def _hold_pizza_pose(
    args: argparse.Namespace,
) -> list[tuple[ArmSdkController, list[tuple[int, float]]]]:
    # Symmetric "tray carry" pose with forearms in front.
    left_pose_deg = {
        "shoulder_pitch": -62.0,
        "shoulder_roll": 18.0,
        "shoulder_yaw": 6.0,
        "elbow": 82.0,
        "wrist_pitch": -12.0,
    }
    right_pose_deg = {
        "shoulder_pitch": -62.0,
        "shoulder_roll": -18.0,
        "shoulder_yaw": -6.0,
        "elbow": 82.0,
        "wrist_pitch": -12.0,
    }

    left = ArmSdkController(args.iface, "left", args.arm_cmd_hz, args.arm_kp, args.arm_kd)
    right = ArmSdkController(args.iface, "right", args.arm_cmd_hz, args.arm_kp, args.arm_kd)
    left.seed_from_lowstate()
    right.seed_from_lowstate()

    left_pose = _indexed_pose("left", left_pose_deg)
    right_pose = _indexed_pose("right", right_pose_deg)
    left.ramp_to_pose(left_pose, duration=args.arm_duration, easing="smooth")
    right.ramp_to_pose(right_pose, duration=args.arm_duration, easing="smooth")
    left.hold_pose(left_pose, args.arm_hold_seed)
    right.hold_pose(right_pose, args.arm_hold_seed)
    print("[arm] pizza-hold pose active")
    return [(left, left_pose), (right, right_pose)]


def _blink_wait(
    args: argparse.Namespace,
    arm_controllers: Iterable[tuple[ArmSdkController, list[tuple[int, float]]]],
) -> None:
    wait_s = max(0.0, args.wait_seconds)
    if wait_s <= 0.0:
        return

    period = max(0.1, args.blink_period)
    t_end = time.time() + wait_s
    toggle = False
    print(f"[wait] waiting {wait_s:.1f}s for pizza placement")
    while time.time() < t_end:
        color = args.waiting_color if toggle else args.idle_color
        toggle = not toggle
        _set_headlight(args.iface, color, intensity=90 if color == args.waiting_color else 35)
        for ctrl, pose in arm_controllers:
            ctrl.hold_pose(pose, min(0.2, period / 2.0))
        time.sleep(period / 2.0)


def main() -> None:
    args = _parse_args()

    if not os.path.exists(args.map):
        raise SystemExit(f"Map does not exist: {args.map}")
    if not os.path.exists(_nav_script_path()):
        raise SystemExit("Missing navigation script: navigation/obstacle_avoidance/real_time_path_steps_dynamic.py")
    if not os.path.exists(_headlight_script_path()):
        raise SystemExit("Missing headlight script: basic/headlight_client/headlight.py")

    _set_headlight(args.iface, "blue", intensity=60)
    _run_checked(_build_nav_cmd(args, args.start_x, args.start_y), "nav_to_start")

    controllers = _hold_pizza_pose(args)
    _blink_wait(args, controllers)

    _set_headlight(args.iface, "cyan", intensity=70)
    _run_checked(_build_nav_cmd(args, args.goal_x, args.goal_y), "nav_to_goal")
    _set_headlight(args.iface, args.end_color, intensity=90)
    print("[done] pizza delivered sequence complete")


if __name__ == "__main__":
    main()
