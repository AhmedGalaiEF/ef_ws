#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys

from ollama_vla.sport_actor import SportCommandExecutor
from sdk_safety import init_channel_autodetect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn the Go2 in place at a fixed angular velocity for a fixed duration."
    )
    parser.add_argument("--iface", default="enp0s31f6")
    parser.add_argument("--deg-per-sec", type=float, default=10.0)
    parser.add_argument("--duration-sec", type=float, default=9.0)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--print-json", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    init_channel_autodetect(args.iface)

    executor = SportCommandExecutor(timeout_sec=5.0, dry_run=args.dry_run)
    executor.start()

    command = {
        "name": "move",
        "args": {
            "vx": 0.0,
            "vy": 0.0,
            "vyaw": args.deg_per_sec,
        },
        "duration_sec": args.duration_sec,
    }

    def _stop_handler(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    try:
        executed = executor.execute(command)
    except KeyboardInterrupt:
        executor.execute({"name": "stop_move", "args": {}, "duration_sec": 0.0})
        print("\nStopping turn test.")
        return 130

    if args.print_json:
        print(json.dumps(executed.__dict__, indent=2))
    else:
        print(
            f"Executed turn test: vyaw={command['args']['vyaw']:.6f} deg/s "
            f"({args.deg_per_sec:.2f} deg/s) for {args.duration_sec:.2f}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
