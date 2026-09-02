#!/usr/bin/env python3
"""Gently and repeatedly open/close only the left Dex3 middle finger.

All non-middle joints are sent with zero gains, so this program neither holds
nor targets the thumb or index finger.  Stop with Ctrl-C; it then sends a short
zero-gain release packet for every left-hand joint.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from sdk_hand import (
    Dex3HandController,
    FINGER_TO_IDXS,
    build_hand_msg,
    hand_grip_targets,
    pack_ris_mode,
)


MIDDLE_IDXS = FINGER_TO_IDXS["middle"]


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def percent(value: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 100.0:
        raise argparse.ArgumentTypeError("must be between 0 and 100")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously, gently open and close only the left Dex3 middle finger."
    )
    parser.add_argument("--iface", default="eth0", help="DDS network interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID.")
    parser.add_argument("--open-s", type=positive_float, default=4.0,
                        help="Seconds for the opening ramp (default: 4).")
    parser.add_argument("--close-s", type=positive_float, default=4.0,
                        help="Seconds for the closing ramp (default: 4).")
    parser.add_argument("--hold-s", type=float, default=0.4,
                        help="Seconds to pause at each end (default: 0.4).")
    parser.add_argument("--rate-hz", type=positive_float, default=25.0,
                        help="Command rate in Hz (default: 25).")
    parser.add_argument("--open-percent", type=percent, default=5.0,
                        help="Grip percentage at the open end; avoids the hard stop (default: 5).")
    parser.add_argument("--close-percent", type=percent, default=88.0,
                        help="Grip percentage at the closed end; avoids the hard stop (default: 88).")
    parser.add_argument("--kp", type=float, default=0.35,
                        help="Middle-finger proportional gain (default: 0.35).")
    parser.add_argument("--kd", type=float, default=0.08,
                        help="Middle-finger derivative gain (default: 0.08).")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Number of open/close cycles; 0 runs until Ctrl-C (default: 0).")
    args = parser.parse_args()
    if args.domain_id < 0:
        parser.error("--domain-id must be non-negative")
    if args.hold_s < 0.0:
        parser.error("--hold-s must be non-negative")
    if args.kp < 0.0 or args.kd < 0.0:
        parser.error("--kp and --kd must be non-negative")
    if args.close_percent <= args.open_percent:
        parser.error("--close-percent must be greater than --open-percent")
    if args.cycles < 0:
        parser.error("--cycles must be zero or positive")
    return args


def smoothstep(alpha: float) -> float:
    value = min(1.0, max(0.0, alpha))
    return value * value * (3.0 - 2.0 * value)


def write_middle_only(
    controller: Dex3HandController,
    middle_targets: list[float],
    *,
    kp: float,
    kd: float,
) -> None:
    """Release every other joint while position-controlling joints 3 and 4."""
    msg = build_hand_msg([0.0] * 7, kp=0.0, kd=0.0, tau=0.0, timeout=1)
    for local_idx, joint_idx in enumerate(MIDDLE_IDXS):
        command = msg.motor_cmd[joint_idx]
        command.mode = pack_ris_mode(joint_idx, timeout=0)
        command.q = float(middle_targets[local_idx])
        command.dq = 0.0
        command.kp = float(kp)
        command.kd = float(kd)
        command.tau = 0.0
    controller._pub.Write(msg)


def ramp(
    controller: Dex3HandController,
    start: list[float],
    stop: list[float],
    *,
    seconds: float,
    rate_hz: float,
    kp: float,
    kd: float,
) -> None:
    steps = max(2, round(seconds * rate_hz))
    interval_s = seconds / steps
    for step in range(1, steps + 1):
        blend = smoothstep(step / steps)
        targets = [current + (goal - current) * blend for current, goal in zip(start, stop)]
        write_middle_only(controller, targets, kp=kp, kd=kd)
        time.sleep(interval_s)


def hold(
    controller: Dex3HandController,
    targets: list[float],
    *,
    seconds: float,
    rate_hz: float,
    kp: float,
    kd: float,
) -> None:
    deadline = time.monotonic() + seconds
    interval_s = 1.0 / rate_hz
    while time.monotonic() < deadline:
        write_middle_only(controller, targets, kp=kp, kd=kd)
        time.sleep(interval_s)


def release_all(controller: Dex3HandController, rate_hz: float) -> None:
    message = build_hand_msg([0.0] * 7, kp=0.0, kd=0.0, tau=0.0, timeout=1)
    controller.publish_for(message, seconds=0.5, rate_hz=rate_hz)


def main() -> int:
    args = parse_args()
    controller = Dex3HandController("left", iface=args.iface, domain_id=args.domain_id)
    open_pose = hand_grip_targets("left", args.open_percent)
    close_pose = hand_grip_targets("left", args.close_percent)
    open_targets = [open_pose[index] for index in MIDDLE_IDXS]
    close_targets = [close_pose[index] for index in MIDDLE_IDXS]

    print("Left Dex3 middle-finger gentle cycle")
    print(f"  joints: {MIDDLE_IDXS}; open={open_targets}; close={close_targets}")
    print(f"  ramps: open={args.open_s:.1f}s close={args.close_s:.1f}s; Ctrl-C releases all joints")

    try:
        # Seed from live feedback so the first command cannot jump from an
        # assumed pose.  The middle-finger motors must be communicating.
        snapshot = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and snapshot is None:
            snapshot = controller.get_state_snapshot(max_age=1.0)
            if snapshot is None:
                time.sleep(0.02)
        if snapshot is None:
            raise RuntimeError("No fresh left Dex3 state; refusing to command the finger.")
        positions = list(snapshot["positions"])
        current_targets = [positions[index] for index in MIDDLE_IDXS]

        # Begin at the gentle open end, always from the measured pose.
        ramp(controller, current_targets, open_targets, seconds=args.open_s, rate_hz=args.rate_hz,
             kp=args.kp, kd=args.kd)
        cycle = 0
        while args.cycles == 0 or cycle < args.cycles:
            ramp(controller, open_targets, close_targets, seconds=args.close_s, rate_hz=args.rate_hz,
                 kp=args.kp, kd=args.kd)
            hold(controller, close_targets, seconds=args.hold_s, rate_hz=args.rate_hz,
                 kp=args.kp, kd=args.kd)
            ramp(controller, close_targets, open_targets, seconds=args.open_s, rate_hz=args.rate_hz,
                 kp=args.kp, kd=args.kd)
            hold(controller, open_targets, seconds=args.hold_s, rate_hz=args.rate_hz,
                 kp=args.kp, kd=args.kd)
            cycle += 1
    except KeyboardInterrupt:
        print("\nStopping: releasing all left-hand joints.")
    finally:
        release_all(controller, args.rate_hz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
