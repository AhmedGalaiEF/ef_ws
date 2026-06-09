#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from contextlib import ExitStack

from inspire_serial_common import (
    CLEAR_ERROR_REGISTER,
    DEFAULT_OPEN_ORDER,
    HAND_CLOSE_TARGET,
    NullClient,
    SerialHand,
    add_serial_connection_args,
    build_hand_configs,
    normalize_hands,
    open_next_finger,
    parse_order,
    ramp_to_target,
)
def run_sequence(
    hand: str,
    *,
    configs,
    order,
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
    verbose_serial: bool,
) -> None:
    sides = normalize_hands(hand)
    cycle = 1

    with ExitStack() as stack:
        clients: dict[str, SerialHand | None] = {}
        for side in sides:
            config = configs[side]
            client = (
                stack.enter_context(
                    SerialHand(
                        config.port,
                        baudrate=config.baudrate,
                        hand_id=config.hand_id,
                        timeout_s=config.timeout_s,
                        write_delay_s=config.write_delay_s,
                        verbose=verbose_serial,
                    )
                )
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
                    target = open_next_finger(current[side], finger)
                    current[side] = ramp_to_target(
                        clients[side],
                        current[side],
                        target,
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
    parser = argparse.ArgumentParser(
        description=(
            "Continuously close an Inspire hand over serial/TTY, then slowly open "
            "each finger one after another."
        )
    )
    parser.add_argument(
        "--order",
        type=parse_order,
        default=DEFAULT_OPEN_ORDER,
        help="Comma-separated opening order. Default: thumb,index,middle,ring,little.",
    )
    parser.add_argument("--open-duration-s", type=float, default=1.2, help="Seconds for each finger to open.")
    parser.add_argument("--reset-duration-s", type=float, default=1.0, help="Seconds to close all fingers before each loop.")
    parser.add_argument("--closed-hold-s", type=float, default=0.5, help="Seconds to hold the fully closed hand.")
    parser.add_argument("--opened-hold-s", type=float, default=1.0, help="Seconds to hold after all selected fingers are open.")
    parser.add_argument("--between-fingers-s", type=float, default=0.25, help="Pause between finger openings.")
    parser.add_argument("--loop-pause-s", type=float, default=0.3, help="Pause before closing again.")
    parser.add_argument("--rate-hz", type=float, default=20.0, help="Command update rate during slow motion.")
    parser.add_argument("--speed", type=int, default=200, help="Inspire speed register value.")
    parser.add_argument("--force", type=int, default=200, help="Inspire force register value.")
    parser.add_argument("--repeat", type=int, default=0, help="Number of cycles to run. Use 0 for endless looping.")
    add_serial_connection_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sequence(
        "both" if args.both_hands else args.hand,
        configs=build_hand_configs(args),
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
        verbose_serial=args.verbose_serial,
    )


if __name__ == "__main__":
    main()
