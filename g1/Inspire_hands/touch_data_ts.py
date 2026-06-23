#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from inspire_sdk import HAND_CONFIGS, ModbusTcp, decode_f32, normalize_side


def parse_channel(spec: str) -> tuple[str, int, float]:
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError("channel format must be name:start[:scale]")
    name = parts[0]
    start = int(parts[1], 0)
    scale = float(parts[2]) if len(parts) == 3 else 1.0
    return name, start, scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read higher-level tactile values over Modbus TCP. Provide one or more "
            "--channel entries in the form name:start[:scale]. Each channel reads two "
            "registers and decodes one float32 value."
        )
    )
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--channel", action="append", type=parse_channel, required=True)
    parser.add_argument("--interval-s", type=float, default=0.0, help="Repeat polling interval. Use 0 for a single read.")
    parser.add_argument(
        "--input-registers",
        action="store_true",
        help="Read from Modbus input registers (function 4) instead of holding registers (function 3).",
    )
    parser.add_argument(
        "--word-order",
        choices=("big", "little"),
        default="big",
        help="Word order used when reconstructing float32 values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    side = normalize_side(args.hand)
    config = HAND_CONFIGS[side]
    reader_name = "read_input_registers" if args.input_registers else "read_holding_registers"

    with ModbusTcp(config.ip, config.port, config.unit_id) as client:
        while True:
            for name, start, scale in args.channel:
                words = getattr(client, reader_name)(start, 2)
                value = decode_f32(words, word_order=args.word_order)[0] * scale
                print(f"{name}: {value:.6f}")
            if args.interval_s <= 0:
                break
            print("")
            time.sleep(float(args.interval_s))


if __name__ == "__main__":
    main()
