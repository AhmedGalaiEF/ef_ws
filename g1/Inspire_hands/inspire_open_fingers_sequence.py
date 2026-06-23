#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from contextlib import ExitStack

from inspire_sdk import (
    CLEAR_ERROR_REGISTER,
    DEFAULT_OPEN_ORDER,
    HAND_CLOSE_TARGET,
    HAND_CONFIGS,
    ModbusTcp,
    NullClient,
    open_next_finger,
    ramp_to_target,
)


def parse_order(value: str) -> tuple[str, ...]:
    fingers = tuple(part.strip().lower().replace("-", "_") for part in value.split(",") if part.strip())
    if not fingers:
        raise argparse.ArgumentTypeError("at least one finger is required")

    allowed = set(DEFAULT_OPEN_ORDER) | {"thumb_bend", "thumb_rotation", "thumb_rot", "pinky"}
    unknown = [finger for finger in fingers if finger not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown finger(s): {', '.join(unknown)}")
    return fingers


def run_sequence(
    hand: str,
    *,
    order: tuple[str, ...],
    open_duration_s: float,
    reset_duration_s: float,
    closed_hold_s: float,
    opened_hold_s: float,
    between_fingers_s: float,
    loop_pause_s: float,
    rate_hz: float,
    speed: int,
    force: int,
    repeat: int,
    dry_run: bool,
) -> None:
    sides = ("left", "right") if hand == "both" else (hand,)
    cycle = 1

    with ExitStack() as stack:
        clients: dict[str, ModbusTcp | None] = {}
        for side in sides:
            config = HAND_CONFIGS[side]
            client = (
                stack.enter_context(ModbusTcp(config.ip, config.port, config.unit_id))
                if not dry_run
                else stack.enter_context(NullClient())
            )
            clients[side] = client
            if client is not None:
                client.write_single_register(CLEAR_ERROR_REGISTER, 1)

        current = {side: list(HAND_CLOSE_TARGET) for side in sides}
        while repeat <= 0 or cycle <= repeat:
            for side in sides:
                print(f"{side}: cycle {cycle} closing all fingers")
                current[side] = ramp_to_target(
                    clients[side],
                    current[side],
                    HAND_CLOSE_TARGET,
                    duration_s=reset_duration_s,
                    rate_hz=rate_hz,
                    speed=speed,
                    force=force,
                    dry_run=dry_run,
                )
            if closed_hold_s > 0:
                time.sleep(float(closed_hold_s))

            for finger in order:
                for side in sides:
                    print(f"{side}: opening {finger}")
                    current[side] = ramp_to_target(
                        clients[side],
                        current[side],
                        open_next_finger(current[side], finger),
                        duration_s=open_duration_s,
                        rate_hz=rate_hz,
                        speed=speed,
                        force=force,
                        dry_run=dry_run,
                    )
                if between_fingers_s > 0:
                    time.sleep(float(between_fingers_s))

            if opened_hold_s > 0:
                time.sleep(float(opened_hold_s))
            if loop_pause_s > 0 and (repeat <= 0 or cycle < repeat):
                time.sleep(float(loop_pause_s))
            cycle += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Close an Inspire hand over Modbus TCP, then open fingers one by one.")
    parser.add_argument("--hand", choices=("left", "right", "both"), default="right")
    parser.add_argument("--order", type=parse_order, default=DEFAULT_OPEN_ORDER)
    parser.add_argument("--open-duration-s", type=float, default=1.2)
    parser.add_argument("--reset-duration-s", type=float, default=1.0)
    parser.add_argument("--closed-hold-s", type=float, default=0.5)
    parser.add_argument("--opened-hold-s", type=float, default=1.0)
    parser.add_argument("--between-fingers-s", type=float, default=0.25)
    parser.add_argument("--loop-pause-s", type=float, default=0.3)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--speed", type=int, default=200)
    parser.add_argument("--force", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=0, help="Number of cycles to run. Use 0 for endless looping.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sequence(
        args.hand,
        order=args.order,
        open_duration_s=args.open_duration_s,
        reset_duration_s=args.reset_duration_s,
        closed_hold_s=args.closed_hold_s,
        opened_hold_s=args.opened_hold_s,
        between_fingers_s=args.between_fingers_s,
        loop_pause_s=args.loop_pause_s,
        rate_hz=args.rate_hz,
        speed=args.speed,
        force=args.force,
        repeat=args.repeat,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
