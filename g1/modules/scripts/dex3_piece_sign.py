#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACADEMY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "academy"))
if ACADEMY_DIR not in sys.path:
    sys.path.insert(0, ACADEMY_DIR)

try:
    from sdk_wrapper import G1, HAND_CLOSED, HAND_OPEN
except ImportError as exc:
    raise SystemExit(
        "Could not import academy/sdk_wrapper.py. Keep this script in modules/scripts."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Close the Dex3 thumb, then open index and middle fingers into a piece sign."
    )
    parser.add_argument("--iface", default="eth0", help="Network interface for DDS traffic.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument(
        "--hand",
        choices=("left", "right", "both"),
        default="right",
        help="Dex3 hand to command.",
    )
    parser.add_argument(
        "--close-hold-s",
        type=float,
        default=0.8,
        help="Seconds to hold the closed hand before opening the sign.",
    )
    parser.add_argument(
        "--open-hold-s",
        type=float,
        default=1.5,
        help="Seconds to hold the piece-sign command.",
    )
    parser.add_argument("--rate-hz", type=float, default=50.0, help="DDS publish rate.")
    parser.add_argument(
        "--ramp-s",
        type=float,
        default=0.5,
        help="Seconds to ramp into each hand target.",
    )
    return parser.parse_args()


def _sides(hand: str) -> tuple[str, ...]:
    return ("left", "right") if hand == "both" else (hand,)


def piece_sign_targets(hand: str) -> list[float]:
    """Thumb joints stay closed; middle and index joints move to open targets."""
    closed = list(HAND_CLOSED[hand])
    open_pose = HAND_OPEN[hand]
    for joint_index in (3, 4, 5, 6):
        closed[joint_index] = open_pose[joint_index]
    return closed


def open_piece_sign(
    g1: G1,
    hand: str = "right",
    hold_s: float = 1.5,
    rate_hz: float = 50.0,
    ramp_s: float | None = 0.5,
) -> dict:
    out = {}
    for side in _sides(str(hand).strip().lower()):
        out[side] = g1.hand_pose(
            piece_sign_targets(side),
            hand=side,
            hold_s=hold_s,
            rate_hz=rate_hz,
            ramp_s=ramp_s,
        )
    return out if len(out) > 1 else next(iter(out.values()))


def main() -> int:
    args = parse_args()
    g1 = G1(iface=args.iface, domain_id=args.domain_id)

    print("Opening index and middle fingers into piece sign:")
    open_result = open_piece_sign(
        g1,
        hand=args.hand,
        hold_s=args.open_hold_s,
        rate_hz=args.rate_hz,
        ramp_s=args.ramp_s,
    )
    print(open_result)
    

    print("Closing Dex3 hand first:")
    close_result = g1.close_dex3_hand(
        hand=args.hand,
        hold_s=args.close_hold_s,
        rate_hz=args.rate_hz,
        ramp_s=args.ramp_s,
    )
    print(close_result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
