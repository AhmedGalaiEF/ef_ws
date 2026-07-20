#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from sdk_hand import (
        Dex3HandController,
        FINGER_TO_IDXS,
        build_hand_msg,
        hand_closed_targets,
        hand_open_targets,
    )
except ImportError as exc:
    raise SystemExit(
        "Could not import sdk_hand. Run this script from modules/scripts or keep sdk_hand.py in modules/."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slowly move Dex3-1 fingers on the right hand, then the left hand."
    )
    parser.add_argument("--iface", default="eth0", help="Network interface for DDS traffic.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument(
        "--close-s",
        type=float,
        default=3.0,
        help="Seconds to ramp from open to closed.",
    )
    parser.add_argument(
        "--open-s",
        type=float,
        default=3.0,
        help="Seconds to ramp from closed back to open.",
    )
    parser.add_argument("--hold-s", type=float, default=0.5, help="Hold time at each end.")
    parser.add_argument("--pause-s", type=float, default=0.8, help="Pause between hands.")
    parser.add_argument("--rate-hz", type=float, default=50.0, help="DDS publish rate.")
    parser.add_argument("--kp", type=float, default=0.8, help="Joint proportional gain.")
    parser.add_argument("--kd", type=float, default=0.05, help="Joint derivative gain.")
    parser.add_argument("--tau", type=float, default=0.03, help="Feed-forward torque.")
    parser.add_argument(
        "--per-finger",
        action="store_true",
        help="Move thumb, index, and middle individually instead of closing all fingers together.",
    )
    return parser.parse_args()


def blend_targets(start: list[float], stop: list[float], alpha: float) -> list[float]:
    blend = min(1.0, max(0.0, float(alpha)))
    return [src + (dst - src) * blend for src, dst in zip(start, stop)]


def publish_targets(
    controller: Dex3HandController,
    targets: list[float],
    *,
    seconds: float,
    rate_hz: float,
    kp: float,
    kd: float,
    tau: float,
) -> None:
    msg = build_hand_msg(targets, kp=kp, kd=kd, tau=tau)
    controller.publish_for(msg, seconds=seconds, rate_hz=rate_hz)


def ramp_targets(
    controller: Dex3HandController,
    start: list[float],
    stop: list[float],
    *,
    seconds: float,
    rate_hz: float,
    kp: float,
    kd: float,
    tau: float,
) -> None:
    rate = max(1.0, float(rate_hz))
    steps = max(2, int(round(max(0.1, float(seconds)) * rate)))
    dt = float(seconds) / float(steps)

    for step_idx in range(1, steps + 1):
        targets = blend_targets(start, stop, step_idx / steps)
        publish_targets(
            controller,
            targets,
            seconds=dt,
            rate_hz=rate,
            kp=kp,
            kd=kd,
            tau=tau,
        )


def finger_closed_target(hand: str, finger_name: str) -> list[float]:
    targets = hand_open_targets(hand)
    closed_targets = hand_closed_targets(hand)
    for idx in FINGER_TO_IDXS[finger_name]:
        targets[idx] = closed_targets[idx]
    return targets


def move_all_fingers(hand: str, controller: Dex3HandController, args: argparse.Namespace) -> None:
    open_targets = hand_open_targets(hand)
    closed_targets = hand_closed_targets(hand)
    publish_targets(
        controller,
        open_targets,
        seconds=args.hold_s,
        rate_hz=args.rate_hz,
        kp=args.kp,
        kd=args.kd,
        tau=args.tau,
    )
    ramp_targets(
        controller,
        open_targets,
        closed_targets,
        seconds=args.close_s,
        rate_hz=args.rate_hz,
        kp=args.kp,
        kd=args.kd,
        tau=args.tau,
    )
    publish_targets(
        controller,
        closed_targets,
        seconds=args.hold_s,
        rate_hz=args.rate_hz,
        kp=args.kp,
        kd=args.kd,
        tau=args.tau,
    )
    ramp_targets(
        controller,
        closed_targets,
        open_targets,
        seconds=args.open_s,
        rate_hz=args.rate_hz,
        kp=args.kp,
        kd=args.kd,
        tau=args.tau,
    )


def move_fingers_individually(hand: str, controller: Dex3HandController, args: argparse.Namespace) -> None:
    open_targets = hand_open_targets(hand)
    for finger_name in ("thumb", "index", "middle"):
        target = finger_closed_target(hand, finger_name)
        print(f"  moving {finger_name}")
        ramp_targets(
            controller,
            open_targets,
            target,
            seconds=args.close_s,
            rate_hz=args.rate_hz,
            kp=args.kp,
            kd=args.kd,
            tau=args.tau,
        )
        publish_targets(
            controller,
            target,
            seconds=args.hold_s,
            rate_hz=args.rate_hz,
            kp=args.kp,
            kd=args.kd,
            tau=args.tau,
        )
        ramp_targets(
            controller,
            target,
            open_targets,
            seconds=args.open_s,
            rate_hz=args.rate_hz,
            kp=args.kp,
            kd=args.kd,
            tau=args.tau,
        )


def move_hand(hand: str, args: argparse.Namespace) -> None:
    controller = Dex3HandController(hand=hand, iface=args.iface, domain_id=args.domain_id)
    print(f"Moving {hand} Dex3-1 hand")
    if args.per_finger:
        move_fingers_individually(hand, controller, args)
    else:
        move_all_fingers(hand, controller, args)

    publish_targets(
        controller,
        hand_open_targets(hand),
        seconds=args.hold_s,
        rate_hz=args.rate_hz,
        kp=args.kp,
        kd=args.kd,
        tau=0.0,
    )


def main() -> int:
    args = parse_args()
    print(
        "Dex3-1 slow finger motion: "
        f"iface={args.iface}, domain_id={args.domain_id}, rate_hz={args.rate_hz}"
    )

    move_hand("right", args)
    time.sleep(max(0.0, float(args.pause_s)))
    move_hand("left", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
