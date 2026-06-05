#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from collections.abc import Iterable, Sequence
from typing import Literal

from inspire_sdk import (
    ANGLE_SET_REGISTER,
    CLEAR_ERROR_REGISTER,
    FORCE_SET_REGISTER,
    HAND_CLOSE_TARGET,
    HAND_CONFIGS,
    HAND_OPEN_TARGET,
    ModbusTcp,
    SPEED_SET_REGISTER,
)


Side = Literal["left", "right"]

# Inspire RH56DFTP angle register order:
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


def normalize_side(hand: str) -> Side:
    side = str(hand).strip().lower()
    if side in {"r", "right"}:
        return "right"
    if side in {"l", "left"}:
        return "left"
    raise ValueError("hand must be 'right' or 'left'")


def clamp_register(value: int) -> int:
    return max(0, min(1000, int(value)))


def send_angles(
    hand: str,
    angles: Sequence[int],
    *,
    speed: int = 200,
    force: int = 200,
    hold: float = 0.0,
) -> None:
    """Send one complete six-angle target to an Inspire hand."""
    if len(angles) != 6:
        raise ValueError("Inspire angle targets must contain exactly 6 values.")

    side = normalize_side(hand)
    config = HAND_CONFIGS[side]
    target = [clamp_register(value) for value in angles]

    with ModbusTcp(config.ip, config.port, config.unit_id) as client:
        client.write_single_register(CLEAR_ERROR_REGISTER, 1)
        client.write_registers(SPEED_SET_REGISTER, [clamp_register(speed)] * 6)
        client.write_registers(FORCE_SET_REGISTER, [clamp_register(force)] * 6)
        client.write_registers(ANGLE_SET_REGISTER, target)

    if hold > 0:
        time.sleep(float(hold))


def finger_target(
    fingers: Iterable[str],
    *,
    open_target: Sequence[int] = HAND_OPEN_TARGET,
    close_target: Sequence[int] = HAND_CLOSE_TARGET,
    percent: float = 100.0,
) -> list[int]:
    """Build a six-angle target with only the selected finger joints moved."""
    alpha = max(0.0, min(100.0, float(percent))) / 100.0
    target = [int(value) for value in open_target]
    closed = [int(value) for value in close_target]

    for finger_name in fingers:
        finger = str(finger_name).strip().lower().replace("-", "_")
        if finger not in FINGER_TO_IDXS:
            raise ValueError(f"Unknown finger '{finger_name}'.")
        for idx in FINGER_TO_IDXS[finger]:
            target[idx] = round(target[idx] + (closed[idx] - target[idx]) * alpha)

    return [clamp_register(value) for value in target]


def move_finger(
    hand: str,
    finger: str,
    *,
    percent: float = 100.0,
    speed: int = 200,
    force: int = 200,
    hold: float = 1.0,
    open_first: bool = True,
    reopen: bool = True,
    settle: float = 0.5,
) -> None:
    """Open the hand, move one finger, optionally reopen it."""
    if open_first:
        send_angles(hand, HAND_OPEN_TARGET, speed=speed, force=force, hold=settle)

    target = finger_target((finger,), percent=percent)
    send_angles(hand, target, speed=speed, force=force, hold=hold)

    if reopen:
        send_angles(hand, HAND_OPEN_TARGET, speed=speed, force=force, hold=settle)


def parse_fingers(value: str) -> tuple[str, ...]:
    normalized = value.strip().lower().replace("-", "_")
    if normalized == "all":
        return DEFAULT_SEQUENCE

    fingers = tuple(part.strip().lower().replace("-", "_") for part in normalized.split(",") if part.strip())
    if not fingers:
        raise argparse.ArgumentTypeError("at least one finger is required")

    unknown = [finger for finger in fingers if finger not in FINGER_TO_IDXS]
    if unknown:
        allowed = ", ".join(("all", *DEFAULT_SEQUENCE, "thumb_bend", "thumb_rotation"))
        raise argparse.ArgumentTypeError(f"unknown finger(s): {', '.join(unknown)}. Use one of: {allowed}")
    return fingers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move individual fingers on an Inspire RH56DFTP hand without editing inspire_sdk.py."
    )
    parser.add_argument(
        "finger",
        type=parse_fingers,
        help="Finger to move: all, thumb, index, middle, ring, little, thumb_bend, thumb_rotation, or comma-separated names.",
    )
    parser.add_argument("--hand", choices=("left", "right", "both"), default="right")
    parser.add_argument("--percent", type=float, default=100.0, help="How far to close selected finger(s), 0-100.")
    parser.add_argument("--speed", type=int, default=200)
    parser.add_argument("--force", type=int, default=200)
    parser.add_argument("--hold", type=float, default=1.0, help="Seconds to hold the selected finger target.")
    parser.add_argument("--settle", type=float, default=0.5, help="Seconds to hold open before/after movement.")
    parser.add_argument("--pause", type=float, default=0.35, help="Pause between fingers when finger=all.")
    parser.add_argument("--no-open-first", action="store_true", help="Do not open the hand before moving.")
    parser.add_argument("--no-reopen", action="store_true", help="Do not reopen the hand after moving.")
    parser.add_argument("--dry-run", action="store_true", help="Print targets instead of sending Modbus commands.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hands = ("left", "right") if args.hand == "both" else (args.hand,)

    for hand in hands:
        for idx, finger in enumerate(args.finger):
            if idx > 0 and args.pause > 0:
                time.sleep(float(args.pause))

            target = finger_target((finger,), percent=args.percent)
            if args.dry_run:
                print(
                    f"{hand}: finger={finger} target={target} "
                    f"speed={args.speed} force={args.force} percent={args.percent:g}"
                )
                continue

            move_finger(
                hand,
                finger,
                percent=args.percent,
                speed=args.speed,
                force=args.force,
                hold=args.hold,
                open_first=not args.no_open_first,
                reopen=not args.no_reopen,
                settle=args.settle,
            )


if __name__ == "__main__":
    main()
