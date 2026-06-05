#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import serial


ANGLE_SET_REGISTER = 1486
FORCE_SET_REGISTER = 1498
SPEED_SET_REGISTER = 1522
CLEAR_ERROR_REGISTER = 1004
ACTION_SEQUENCE_REGISTER = 2320
ACTION_RUN_REGISTER = 2322


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


class NullClient:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def add_serial_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hand", choices=("left", "right", "both"), default="right")
    parser.add_argument("--both-hands", action="store_true", help="Run the sequence on both hands.")
    parser.add_argument("--right-port", default=HAND_CONFIGS["right"].port, help="TTY device for the right hand.")
    parser.add_argument("--left-port", default=HAND_CONFIGS["left"].port, help="TTY device for the left hand.")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate.")
    parser.add_argument("--right-id", type=int, default=1, help="Hand ID for the right hand.")
    parser.add_argument("--left-id", type=int, default=1, help="Hand ID for the left hand.")
    parser.add_argument("--timeout-s", type=float, default=0.05, help="Serial read/write timeout.")
    parser.add_argument("--write-delay-s", type=float, default=0.01, help="Delay after each serial frame.")
    parser.add_argument("--verbose-serial", action="store_true", help="Print raw serial frames.")
    parser.add_argument("--dry-run", action="store_true", help="Print targets without sending commands.")


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


def open_next_finger(current: Sequence[int], finger: str) -> list[int]:
    target = [clamp_register(value) for value in current]
    for idx in FINGER_TO_IDXS[finger]:
        target[idx] = HAND_OPEN_TARGET[idx]
    return target


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
