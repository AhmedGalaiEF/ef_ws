from __future__ import annotations

import argparse
import os
import sys

# Add parent directory to path to import sdk_hand
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from sdk_hand import Dex3HandController, hand_grip_targets
except ImportError:
    print("Could not import sdk_hand. Ensure it is in the parent directory.")
    sys.exit(1)


def grip_targets(percent: float, hand: str = "right") -> list[float]:
    return hand_grip_targets(hand, percent)


def send_grip(
    controller: Dex3HandController,
    hand: str,
    percent: float,
    *,
    hold_s: float,
    rate_hz: float,
    ramp_s: float | None,
) -> None:
    clamped = min(100.0, max(0.0, float(percent)))
    controller.set_targets(
        hand_grip_targets(hand, clamped),
        hold_s=hold_s,
        rate_hz=rate_hz,
        ramp_s=ramp_s,
    )
    print(f"{hand.title()} hand grip set to {clamped:g}%")


def run_interactive(
    controller: Dex3HandController,
    hand: str,
    *,
    hold_s: float,
    rate_hz: float,
    ramp_s: float | None,
) -> None:
    print("Enter grip percentage 0-100, or 'q' to quit.")
    while True:
        try:
            raw_value = input("grip> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if raw_value.lower() in {"q", "quit", "exit"}:
            return

        try:
            percent = float(raw_value)
        except ValueError:
            print("Please enter a number from 0 to 100.")
            continue

        send_grip(
            controller,
            hand,
            percent,
            hold_s=hold_s,
            rate_hz=rate_hz,
            ramp_s=ramp_s,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dex3 hand grip control CLI.")
    parser.add_argument("hand", nargs="?", choices=("left", "right"), default="right")
    parser.add_argument(
        "percent",
        nargs="?",
        type=float,
        default=0.0,
        help="Grip percentage from 0 open to 100 fully closed.",
    )
    parser.add_argument("--iface", default="eth0", help="Network interface for DDS traffic.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument("--hold-s", type=float, default=0.6, help="Seconds to publish the target.")
    parser.add_argument("--rate-hz", type=float, default=50.0, help="Publish rate in Hz.")
    parser.add_argument(
        "--ramp-s",
        type=float,
        default=None,
        help="Seconds to ramp toward the target. Defaults to controller behavior.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Keep a prompt open for repeated grip percentage commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        controller = Dex3HandController(
            hand=args.hand,
            iface=args.iface,
            domain_id=args.domain_id,
        )
    except Exception as exc:
        print(f"Failed to initialize hand controller: {exc}")
        return 1

    if args.interactive:
        run_interactive(
            controller,
            args.hand,
            hold_s=args.hold_s,
            rate_hz=args.rate_hz,
            ramp_s=args.ramp_s,
        )
    else:
        send_grip(
            controller,
            args.hand,
            args.percent,
            hold_s=args.hold_s,
            rate_hz=args.rate_hz,
            ramp_s=args.ramp_s,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
