#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from contextlib import ExitStack

from inspire_serial_common import (
    ACTION_RUN_REGISTER,
    ACTION_SEQUENCE_REGISTER,
    CLEAR_ERROR_REGISTER,
    NullClient,
    SerialHand,
    add_serial_connection_args,
    build_hand_configs,
    normalize_hands,
)


def run_sequence(
    hand: str,
    *,
    configs,
    sequence_id: int,
    repeat: int,
    select_delay_s: float,
    between_runs_s: float,
    startup_delay_s: float,
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

        if startup_delay_s > 0:
            time.sleep(float(startup_delay_s))

        while repeat <= 0 or cycle <= repeat:
            for side in sides:
                print(f"{side}: cycle {cycle} running action sequence {sequence_id}")
                if dry_run:
                    continue

                client = clients[side]
                if client is None:
                    raise RuntimeError("Serial client is required unless dry-run is enabled")
                client.write_single_register(ACTION_SEQUENCE_REGISTER, sequence_id)
                if select_delay_s > 0:
                    time.sleep(float(select_delay_s))
                client.write_single_register(ACTION_RUN_REGISTER, 1)

            if between_runs_s > 0 and (repeat <= 0 or cycle < repeat):
                time.sleep(float(between_runs_s))
            cycle += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trigger an Inspire hand built-in action sequence over serial/TTY."
    )
    parser.add_argument("--sequence-id", type=int, default=3, help="Built-in action sequence index to run.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of runs. Use 0 for endless looping.")
    parser.add_argument("--select-delay-s", type=float, default=0.05, help="Delay between selecting and running the sequence.")
    parser.add_argument("--between-runs-s", type=float, default=3.0, help="Delay before triggering the next run.")
    parser.add_argument("--startup-delay-s", type=float, default=0.0, help="Delay before the first action sequence run.")
    add_serial_connection_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sequence(
        "both" if args.both_hands else args.hand,
        configs=build_hand_configs(args),
        sequence_id=args.sequence_id,
        repeat=args.repeat,
        select_delay_s=args.select_delay_s,
        between_runs_s=args.between_runs_s,
        startup_delay_s=args.startup_delay_s,
        dry_run=args.dry_run,
        verbose_serial=args.verbose_serial,
    )


if __name__ == "__main__":
    main()
