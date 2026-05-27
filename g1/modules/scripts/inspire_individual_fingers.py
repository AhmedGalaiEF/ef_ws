#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from inspire_sdk import HAND_CLOSE_TARGET, HAND_OPEN_TARGET, _move_hand
except ImportError as exc:
    raise SystemExit(
        "Could not import inspire_sdk. Run this script from modules/scripts or keep inspire_sdk.py in modules/."
    ) from exc


# Inspire RH56DFTP register order:
# little, ring, middle, index, thumb bending, thumb rotation.
FINGER_TO_IDXS: dict[str, tuple[int, ...]] = {
    "little": (0,),
    "pinky": (0,),
    "ring": (1,),
    "middle": (2,),
    "index": (3,),
    "thumb": (4, 5),
    "thumb_bend": (4,),
    "thumb_rotation": (5,),
    "thumb_rot": (5,),
}

DEFAULT_SEQUENCE = ("thumb", "index", "middle", "ring", "little")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move Inspire hand fingers one at a time."
    )
    parser.add_argument(
        "finger",
        nargs="?",
        default="all",
        help=(
            "Finger to move: all, thumb, index, middle, ring, little, "
            "thumb_bend, or thumb_rotation."
        ),
    )
    parser.add_argument(
        "--hand",
        choices=("left", "right", "both"),
        default="right",
        help="Which Inspire hand to command.",
    )
    parser.add_argument("--speed", type=int, default=200, help="Inspire speed register value.")
    parser.add_argument("--force", type=int, default=200, help="Inspire force register value.")
    parser.add_argument(
        "--hold-s",
        type=float,
        default=1.0,
        help="Seconds to hold the selected finger closed.",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.6,
        help="Seconds to hold the open hand before and after each finger move.",
    )
    parser.add_argument(
        "--pause-s",
        type=float,
        default=0.4,
        help="Pause between fingers when moving a sequence.",
    )
    parser.add_argument(
        "--close-value",
        type=int,
        default=None,
        help="Override the close target for non-thumb fingers.",
    )
    parser.add_argument(
        "--thumb-bend-close",
        type=int,
        default=None,
        help="Override the thumb bending close target.",
    )
    parser.add_argument(
        "--thumb-rotation-close",
        type=int,
        default=None,
        help="Override the thumb rotation close target.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target registers without sending commands to the hand.",
    )
    return parser.parse_args()


def hand_sides(value: str) -> tuple[str, ...]:
    return ("left", "right") if value == "both" else (value,)


def selected_fingers(value: str) -> tuple[str, ...]:
    finger = value.strip().lower().replace("-", "_")
    if finger == "all":
        return DEFAULT_SEQUENCE
    if finger not in FINGER_TO_IDXS:
        allowed = ", ".join(("all", *DEFAULT_SEQUENCE, "thumb_bend", "thumb_rotation"))
        raise SystemExit(f"Unknown finger '{value}'. Use one of: {allowed}.")
    return (finger,)


def finger_target(finger: str, args: argparse.Namespace) -> list[int]:
    target = list(HAND_OPEN_TARGET)
    close_target = list(HAND_CLOSE_TARGET)

    if args.close_value is not None:
        for idx in (0, 1, 2, 3):
            close_target[idx] = int(args.close_value)
    if args.thumb_bend_close is not None:
        close_target[4] = int(args.thumb_bend_close)
    if args.thumb_rotation_close is not None:
        close_target[5] = int(args.thumb_rotation_close)

    for idx in FINGER_TO_IDXS[finger]:
        target[idx] = close_target[idx]
    return target


def send_target(
    hand: str,
    target: Sequence[int],
    *,
    speed: int,
    force: int,
    hold_s: float,
    dry_run: bool,
) -> None:
    values = [int(value) for value in target]
    if dry_run:
        print(f"{hand}: target={values} speed={speed} force={force} hold_s={hold_s:g}")
        if hold_s > 0:
            time.sleep(float(hold_s))
        return

    _move_hand(hand, values, speed=int(speed), force=int(force), hold=float(hold_s))


def move_finger(hand: str, finger: str, args: argparse.Namespace) -> None:
    print(f"{hand}: opening hand")
    send_target(
        hand,
        HAND_OPEN_TARGET,
        speed=args.speed,
        force=args.force,
        hold_s=args.settle_s,
        dry_run=args.dry_run,
    )

    target = finger_target(finger, args)
    print(f"{hand}: moving {finger} target={target}")
    send_target(
        hand,
        target,
        speed=args.speed,
        force=args.force,
        hold_s=args.hold_s,
        dry_run=args.dry_run,
    )

    print(f"{hand}: reopening hand")
    send_target(
        hand,
        HAND_OPEN_TARGET,
        speed=args.speed,
        force=args.force,
        hold_s=args.settle_s,
        dry_run=args.dry_run,
    )


def main() -> None:
    args = parse_args()
    fingers = selected_fingers(args.finger)
    sides = hand_sides(args.hand)

    for hand in sides:
        for idx, finger in enumerate(fingers):
            if idx > 0 and args.pause_s > 0:
                time.sleep(float(args.pause_s))
            move_finger(hand, finger, args)


if __name__ == "__main__":
    main()
