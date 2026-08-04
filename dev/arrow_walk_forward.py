#!/usr/bin/env python3
"""Drive the robot forward with the terminal Up arrow key."""

from __future__ import annotations

import argparse
import csv
import json
import math
import select
import statistics
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any

UP_ARROW = "\x1b[A"
DOWN_ARROW = "\x1b[B"
RIGHT_ARROW = "\x1b[C"
LEFT_ARROW = "\x1b[D"


def infer_forward_speed(csv_path: Path) -> float:
    """Estimate a positive forward speed from recorded controller input."""
    samples: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("topic") != "/wirelesscontroller":
                continue
            try:
                msg = json.loads(row["message_json"])
                ly = float(msg.get("ly", 0.0))
            except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if math.isfinite(ly) and abs(ly) > 0.05:
                samples.append(abs(ly))

    if not samples:
        raise RuntimeError(f"No active /wirelesscontroller ly samples found in {csv_path}")
    return float(statistics.median(samples))


def read_key(timeout_s: float) -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return None

    first = sys.stdin.read(1)
    if first != "\x1b":
        return first

    readable, _, _ = select.select([sys.stdin], [], [], 0.01)
    if not readable:
        return first
    second = sys.stdin.read(1)

    readable, _, _ = select.select([sys.stdin], [], [], 0.01)
    if not readable:
        return first + second
    third = sys.stdin.read(1)
    return first + second + third


def nonnegative_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite value >= 0")
    return parsed


def positive_finite_float(value: str) -> float:
    parsed = nonnegative_finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite value > 0")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Press and hold Up arrow to walk forward. Space stops. q exits."
    )
    parser.add_argument("--iface", default="eth0", help="Robot network interface. Default: eth0")
    parser.add_argument("--domain-id", type=nonnegative_int, default=0, help="DDS domain id. Default: 0")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("walk_forward.csv"),
        help="CSV recording used to infer default speed. Default: walk_forward.csv",
    )
    parser.add_argument(
        "--speed",
        type=nonnegative_finite_float,
        help="Forward velocity command. Defaults to median /wirelesscontroller ly from CSV.",
    )
    parser.add_argument(
        "--deadman-timeout",
        type=positive_finite_float,
        default=0.35,
        help="Stop if no Up-arrow repeat arrives for this many seconds. Default: 0.35",
    )
    parser.add_argument(
        "--no-fsm-walk",
        action="store_true",
        help="Do not switch the locomotion FSM to walk mode on startup.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        speed = float(args.speed) if args.speed is not None else infer_forward_speed(args.csv)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Cannot determine forward speed: {exc}", file=sys.stderr)
        return 2

    if not sys.stdin.isatty():
        print("This utility needs an interactive terminal for arrow-key input.", file=sys.stderr)
        return 2

    try:
        from .sdk_lib import G1
    except ImportError:
        try:
            from sdk_lib import G1
        except ImportError as exc:
            print(f"Cannot import the Unitree SDK wrapper: {exc}", file=sys.stderr)
            return 1

    print(f"Forward speed: {speed:.3f}")
    print("Hold Up arrow to walk forward. Space stops. q exits.")

    robot = None
    try:
        robot = G1(iface=args.iface, domain_id=args.domain_id)
    except Exception as exc:
        print(f"Cannot connect to robot: {exc}", file=sys.stderr)
        return 1

    original_termios: Any = None
    last_up = 0.0
    moving = False

    try:
        if not args.no_fsm_walk:
            robot.fsm_walk()
        original_termios = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        while True:
            key = read_key(0.05)
            now = time.monotonic()

            if key == UP_ARROW:
                robot.loco_move(speed, 0.0, 0.0)
                last_up = now
                moving = True
            elif key in (DOWN_ARROW, LEFT_ARROW, RIGHT_ARROW, " "):
                robot.stop()
                moving = False
                last_up = 0.0
            elif key in ("q", "Q", "\x03"):
                break

            if moving and now - last_up > max(0.05, float(args.deadman_timeout)):
                robot.stop()
                moving = False
    except Exception as exc:
        print(f"Walking utility failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            robot.stop()
        except Exception as exc:
            print(f"Warning: failed to stop robot cleanly: {exc}", file=sys.stderr)
        finally:
            if original_termios is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_termios)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
