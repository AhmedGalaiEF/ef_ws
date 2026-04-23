#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from sdk_hand import Dex3HandController, HAND_CLOSED, HAND_OPEN, build_hand_msg
except ImportError as exc:
    raise SystemExit(
        "Could not import sdk_hand. Ensure this script is run from the modules/scripts directory."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move only the left dex3 hand finger joints."
    )
    parser.add_argument(
        "--iface",
        default="eth0",
        help="Network interface for DDS traffic.",
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=0,
        help="DDS domain id.",
    )
    parser.add_argument(
        "--preset",
        choices=("open", "closed"),
        help="Use a built-in joint target preset for the left hand.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Blend between HAND_OPEN (0.0) and HAND_CLOSED (1.0).",
    )
    parser.add_argument(
        "--targets",
        type=float,
        nargs=7,
        metavar=("J0", "J1", "J2", "J3", "J4", "J5", "J6"),
        help="Explicit target positions for the 7 left dex3 finger joints.",
    )
    parser.add_argument(
        "--hold-s",
        type=float,
        default=1.0,
        help="How long to publish the command for.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=50.0,
        help="Publish rate while holding the command.",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=1.2,
        help="Joint proportional gain.",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=0.05,
        help="Joint derivative gain.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
        help="Feed-forward torque.",
    )
    args = parser.parse_args()

    requested_modes = sum(
        value is not None for value in (args.preset, args.alpha, args.targets)
    )
    if requested_modes > 1:
        parser.error("Use only one of --preset, --alpha, or --targets.")
    return args


def blend_targets(alpha: float) -> list[float]:
    clamped = min(1.0, max(0.0, float(alpha)))
    return [
        start + (stop - start) * clamped
        for start, stop in zip(HAND_OPEN, HAND_CLOSED)
    ]


def resolve_targets(args: argparse.Namespace) -> list[float]:
    if args.targets is not None:
        return [float(value) for value in args.targets]
    if args.preset == "open":
        return list(HAND_OPEN)
    if args.preset == "closed":
        return list(HAND_CLOSED)
    if args.alpha is not None:
        return blend_targets(args.alpha)
    return list(HAND_OPEN)


def main() -> int:
    args = parse_args()
    targets = resolve_targets(args)

    controller = Dex3HandController(
        hand="left",
        iface=args.iface,
        domain_id=args.domain_id,
    )
    msg = build_hand_msg(
        targets,
        kp=args.kp,
        kd=args.kd,
        tau=args.tau,
    )

    print("Publishing left dex3 finger joint targets:")
    print("  hand: left")
    print(f"  iface: {args.iface}")
    print(f"  domain_id: {args.domain_id}")
    print(f"  hold_s: {args.hold_s}")
    print(f"  rate_hz: {args.rate_hz}")
    print(f"  targets: {targets}")

    controller.publish_for(
        msg,
        seconds=args.hold_s,
        rate_hz=args.rate_hz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
