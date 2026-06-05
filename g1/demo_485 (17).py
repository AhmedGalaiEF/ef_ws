#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass

import serial


ANGLE_SET_REGISTER = 1486
FORCE_SET_REGISTER = 1498
SPEED_SET_REGISTER = 1522
CLEAR_ERROR_REGISTER = 1004


@dataclass(frozen=True)
class InspireSerialConfig:
    port: str
    baudrate: int = 115200
    hand_id: int = 1
    timeout_s: float = 0.05
    write_delay_s: float = 0.01


HAND_CONFIGS: dict[str, InspireSerialConfig] = {
    "right": InspireSerialConfig(port="/dev/ttyUSB0"),
    "left": InspireSerialConfig(port="/dev/ttyUSB1"),
}


# Match the existing Modbus sequence script so behavior stays familiar.
HAND_OPEN_TARGET = [700, 700, 700, 700, 800, 0]
HAND_CLOSE_TARGET = [0, 0, 0, 0, 1000, 600]

FINGER_TO_IDXS: dict[str, tuple[int, ...]] = {
    "little": (0,),
    "pinky": (0,),
    "ring": (1,),
    "middle": (2,),
    "index": (3,),
    "thumb": (4, 5),
    "thumb_bend": (4,),
    "thumb_rotation": (5,),
    "thumb_rot": (5,),
}

DEFAULT_OPEN_ORDER = ("thumb", "index", "middle", "ring", "little")


class SerialHand:
    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 115200,
        hand_id: int = 1,
        timeout_s: float = 0.05,
        write_delay_s: float = 0.01,
        verbose: bool = False,
    ) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self.hand_id = int(hand_id)
        self.timeout_s = float(timeout_s)
        self.write_delay_s = float(write_delay_s)
        self.verbose = bool(verbose)
        self.ser: serial.Serial | None = None

    def __enter__(self) -> "SerialHand":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout_s,
            write_timeout=self.timeout_s,
        )
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def write_single_register(self, address: int, value: int) -> None:
        value = int(value) & 0xFFFF
        payload = [value & 0xFF, (value >> 8) & 0xFF]
        self._write_register(address, payload)

    def write_registers(self, address: int, values: Iterable[int]) -> None:
        payload: list[int] = []
        for value in values:
            register = int(value) & 0xFFFF
            payload.append(register & 0xFF)
            payload.append((register >> 8) & 0xFF)
        self._write_register(address, payload)

    def _write_register(self, address: int, payload: Sequence[int]) -> None:
        if self.ser is None:
            raise RuntimeError("Serial port is not connected")

        frame = [0xEB, 0x90, self.hand_id, len(payload) + 3, 0x12, address & 0xFF, (address >> 8) & 0xFF]
        frame.extend(int(value) & 0xFF for value in payload)
        checksum = sum(frame[2:]) & 0xFF
        frame.append(checksum)

        if self.verbose:
            print(f"{self.port}: tx {[hex(value) for value in frame]}")

        self.ser.write(bytes(frame))
        self.ser.flush()
        time.sleep(self.write_delay_s)

        response = self.ser.read_all()
        if self.verbose and response:
            print(f"{self.port}: rx {response!r}")


def clamp_register(value: float | int) -> int:
    return max(0, min(1000, round(float(value))))


def normalize_hand(value: str) -> str:
    hand = str(value).strip().lower()
    if hand in {"r", "right"}:
        return "right"
    if hand in {"l", "left"}:
        return "left"
    raise ValueError("hand must be 'left' or 'right'")


def normalize_hands(value: str) -> tuple[str, ...]:
    hand = str(value).strip().lower()
    if hand in {"b", "both", "both_hands", "both-hands"}:
        return ("left", "right")
    return (normalize_hand(hand),)


def parse_order(value: str) -> tuple[str, ...]:
    fingers = tuple(part.strip().lower().replace("-", "_") for part in value.split(",") if part.strip())
    if not fingers:
        raise argparse.ArgumentTypeError("at least one finger is required")

    unknown = [finger for finger in fingers if finger not in FINGER_TO_IDXS]
    if unknown:
        allowed = ", ".join(DEFAULT_OPEN_ORDER)
        raise argparse.ArgumentTypeError(f"unknown finger(s): {', '.join(unknown)}. Default order is: {allowed}")
    return fingers


def interpolate(start: Sequence[int], stop: Sequence[int], alpha: float) -> list[int]:
    return [
        clamp_register(start_value + (stop_value - start_value) * alpha)
        for start_value, stop_value in zip(start, stop)
    ]


def send_target(
    client: SerialHand | None,
    target: Sequence[int],
    *,
    speed: int,
    force: int,
    dry_run: bool,
) -> None:
    values = [clamp_register(value) for value in target]
    if dry_run:
        print(f"target={values}")
        return

    if client is None:
        raise RuntimeError("Serial client is required unless dry-run is enabled")
    client.write_registers(SPEED_SET_REGISTER, [clamp_register(speed)] * 6)
    client.write_registers(FORCE_SET_REGISTER, [clamp_register(force)] * 6)
    client.write_registers(ANGLE_SET_REGISTER, values)


def ramp_to_target(
    client: SerialHand | None,
    current: Sequence[int],
    target: Sequence[int],
    *,
    duration_s: float,
    rate_hz: float,
    speed: int,
    force: int,
    dry_run: bool,
) -> list[int]:
    duration_s = max(0.0, float(duration_s))
    rate_hz = max(1.0, float(rate_hz))
    steps = max(1, round(duration_s * rate_hz))
    delay_s = duration_s / steps if duration_s > 0 else 0.0

    for step in range(1, steps + 1):
        alpha = step / steps
        next_target = interpolate(current, target, alpha)
        send_target(client, next_target, speed=speed, force=force, dry_run=dry_run)
        if delay_s > 0:
            time.sleep(delay_s)

    return [clamp_register(value) for value in target]


def open_next_finger(current: Sequence[int], finger: str) -> list[int]:
    target = [clamp_register(value) for value in current]
    for idx in FINGER_TO_IDXS[finger]:
        target[idx] = HAND_OPEN_TARGET[idx]
    return target


def build_hand_configs(args: argparse.Namespace) -> dict[str, InspireSerialConfig]:
    return {
        "right": InspireSerialConfig(
            port=args.right_port,
            baudrate=args.baudrate,
            hand_id=args.right_id,
            timeout_s=args.timeout_s,
            write_delay_s=args.write_delay_s,
        ),
        "left": InspireSerialConfig(
            port=args.left_port,
            baudrate=args.baudrate,
            hand_id=args.left_id,
            timeout_s=args.timeout_s,
            write_delay_s=args.write_delay_s,
        ),
    }


def run_sequence(
    hand: str,
    *,
    configs: dict[str, InspireSerialConfig],
    order: Sequence[str],
    open_duration_s: float,
    reset_duration_s: float,
    closed_hold_s: float,
    opened_hold_s: float,
    between_fingers_s: float,
    loop_pause_s: float,
    rate_hz: float,
    speed: int,
    force: int,
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
                else stack.enter_context(null_client())
            )
            clients[side] = client
            if client is not None:
                client.write_single_register(CLEAR_ERROR_REGISTER, 1)

        current = {side: list(HAND_CLOSE_TARGET) for side in sides}
        while True:
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
            if loop_pause_s > 0:
                time.sleep(float(loop_pause_s))
            cycle += 1


class null_client:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously close an Inspire hand over serial/TTY, then slowly open "
            "each finger one after another."
        )
    )
    parser.add_argument("--hand", choices=("left", "right", "both"), default="right")
    parser.add_argument("--both-hands", action="store_true", help="Run the sequence on both hands.")
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
    parser.add_argument("--right-port", default=HAND_CONFIGS["right"].port, help="TTY device for the right hand.")
    parser.add_argument("--left-port", default=HAND_CONFIGS["left"].port, help="TTY device for the left hand.")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate.")
    parser.add_argument("--right-id", type=int, default=1, help="Hand ID for the right hand.")
    parser.add_argument("--left-id", type=int, default=1, help="Hand ID for the left hand.")
    parser.add_argument("--timeout-s", type=float, default=0.05, help="Serial read/write timeout.")
    parser.add_argument("--write-delay-s", type=float, default=0.01, help="Delay after each serial frame.")
    parser.add_argument("--verbose-serial", action="store_true", help="Print raw serial frames.")
    parser.add_argument("--dry-run", action="store_true", help="Print targets without sending commands.")
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
        dry_run=args.dry_run,
        verbose_serial=args.verbose_serial,
    )


if __name__ == "__main__":
    main()
