#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from contextlib import ExitStack

from inspire_serial_common import (
    CLEAR_ERROR_REGISTER,
    HAND_CLOSE_TARGET,
    HAND_OPEN_TARGET,
    NullClient,
    SerialHand,
    add_serial_connection_args,
    build_hand_configs,
    normalize_hands,
    ramp_to_target,
)


def run_sequence(
    hand: str,
    *,
    configs,
    open_duration_s: float,
    close_duration_s: float,
    open_hold_s: float,
    closed_hold_s: float,
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
                print(f"{side}: cycle {cycle} opening hand")
                current[side] = ramp_to_target(
                    clients[side],
                    current[side],
                    HAND_OPEN_TARGET,
                    duration_s=open_duration_s,
                    rate_hz=rate_hz,
                    speed=speed,
                    force=force,
                    dry_run=dry_run,
                )
            if open_hold_s > 0:
                time.sleep(float(open_hold_s))

            for side in sides:
                print(f"{side}: cycle {cycle} closing hand")
                current[side] = ramp_to_target(
                    clients[side],
                    current[side],
                    HAND_CLOSE_TARGET,
                    duration_s=close_duration_s,
                    rate_hz=rate_hz,
                    speed=speed,
                    force=force,
                    dry_run=dry_run,
                )
            if closed_hold_s > 0:
                time.sleep(float(closed_hold_s))
            if loop_pause_s > 0 and (repeat <= 0 or cycle < repeat):
                time.sleep(float(loop_pause_s))
            cycle += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously open and close an Inspire hand over serial/TTY."
    )
    parser.add_argument("--open-duration-s", type=float, default=1.0, help="Seconds to move from closed to open.")
    parser.add_argument("--close-duration-s", type=float, default=1.0, help="Seconds to move from open to closed.")
    parser.add_argument("--open-hold-s", type=float, default=0.8, help="Seconds to hold the open hand.")
    parser.add_argument("--closed-hold-s", type=float, default=0.8, help="Seconds to hold the closed hand.")
    parser.add_argument("--loop-pause-s", type=float, default=0.2, help="Pause before the next cycle.")
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
        open_duration_s=args.open_duration_s,
        close_duration_s=args.close_duration_s,
        open_hold_s=args.open_hold_s,
        closed_hold_s=args.closed_hold_s,
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
