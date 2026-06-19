#!/usr/bin/env python3

# importing time and socket modules for handling timing and network communication
from __future__ import annotations

import argparse
import socket
import struct
import time
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass

# Modbus register addresses for the Inspire hand
ANGLE_SET_REGISTER = 1486
FORCE_SET_REGISTER = 1498
SPEED_SET_REGISTER = 1522
CLEAR_ERROR_REGISTER = 1004


@dataclass(frozen=True)
class InspireHandConfig:
    ip: str
    port: int = 6000
    unit_id: int = 1

# Configuration for the left and right Inspire hands. Update the IP addresses as needed.


HAND_CONFIGS: dict[str, InspireHandConfig] = {
    "right": InspireHandConfig(ip="192.168.123.210"),
    "left": InspireHandConfig(ip="192.168.123.211"),
}

# Inspire RH56DFTP angle register order:
# little, ring, middle, index, thumb bending, thumb rotation.
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

# Default opening order for the fingers. Can be overridden with the --order argument.
DEFAULT_OPEN_ORDER = ("thumb", "index", "middle", "ring", "little")


# Modbus TCP client for communicating with the Inspire hand. Supports writing single and multiple registers.
class ModbusTcp:
    def __init__(self, host: str, port: int = 6000, unit_id: int = 1, timeout: float = 2.0):
        self.host = host
        self.port = int(port)
        self.unit_id = int(unit_id)
        self.timeout = float(timeout)
        self.transaction_id = 1
        self.sock: socket.socket | None = None

    def __enter__(self) -> "ModbusTcp":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), self.timeout)
        self.sock.settimeout(self.timeout)

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def write_single_register(self, address: int, value: int) -> None:
        pdu = struct.pack(">BHH", 6, int(address), int(value) & 0xFFFF)
        response = self._request(pdu)
        if response != pdu:
            raise RuntimeError("Unexpected Modbus write-single response")

    def write_registers(self, address: int, values: Iterable[int]) -> None:
        register_values = [int(value) & 0xFFFF for value in values]
        payload = struct.pack(">" + "H" * len(register_values), *register_values)
        pdu = struct.pack(">BHHB", 16, int(address), len(register_values), len(payload)) + payload
        response = self._request(pdu)
        expected = struct.pack(">BHH", 16, int(address), len(register_values))
        if response != expected:
            raise RuntimeError("Unexpected Modbus write-multiple response")

    def _request(self, pdu: bytes) -> bytes:
        if self.sock is None:
            raise RuntimeError("Modbus socket is not connected")

        tid = self.transaction_id & 0xFFFF
        self.transaction_id += 1
        header = struct.pack(">HHHB", tid, 0, len(pdu) + 1, self.unit_id)
        self.sock.sendall(header + pdu)

        response_header = self._recv_exact(7)
        response_tid, protocol, length, unit = struct.unpack(">HHHB", response_header)
        if response_tid != tid or protocol != 0 or unit != self.unit_id:
            raise RuntimeError("Unexpected Modbus response header")

        response_pdu = self._recv_exact(length - 1)
        function = response_pdu[0]
        if function & 0x80:
            code = response_pdu[1] if len(response_pdu) > 1 else None
            raise RuntimeError(f"Modbus exception function=0x{function:02x} code={code}")
        return response_pdu

    def _recv_exact(self, count: int) -> bytes:
        if self.sock is None:
            raise RuntimeError("Modbus socket is not connected")

        chunks: list[bytes] = []
        remaining = int(count)
        while remaining:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise RuntimeError("Socket closed while reading Modbus response")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


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
    fingers = tuple(part.strip().lower().replace("-", "_")
                    for part in value.split(",") if part.strip())
    if not fingers:
        raise argparse.ArgumentTypeError("at least one finger is required")

    unknown = [finger for finger in fingers if finger not in FINGER_TO_IDXS]
    if unknown:
        allowed = ", ".join(DEFAULT_OPEN_ORDER)
        raise argparse.ArgumentTypeError(
            f"unknown finger(s): {', '.join(unknown)}. Default order is: {allowed}")
    return fingers


def interpolate(start: Sequence[int], stop: Sequence[int], alpha: float) -> list[int]:
    return [
        clamp_register(start_value + (stop_value - start_value) * alpha)
        for start_value, stop_value in zip(start, stop)
    ]


def send_target(
    client: ModbusTcp | None,
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
        raise RuntimeError("Modbus client is required unless dry-run is enabled")
    client.write_registers(SPEED_SET_REGISTER, [clamp_register(speed)] * 6)
    client.write_registers(FORCE_SET_REGISTER, [clamp_register(force)] * 6)
    client.write_registers(ANGLE_SET_REGISTER, values)


def ramp_to_target(
    client: ModbusTcp | None,
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


def run_sequence(
    hand: str,
    *,
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
) -> None:
    sides = normalize_hands(hand)
    cycle = 1

    with ExitStack() as stack:
        clients: dict[str, ModbusTcp | None] = {}
        for side in sides:
            config = HAND_CONFIGS[side]
            client = (
                stack.enter_context(ModbusTcp(config.ip, config.port, config.unit_id))
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
            "Continuously close an Inspire hand, then slowly open each finger "
            "one after another."
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
    parser.add_argument("--open-duration-s", type=float, default=1.2,
                        help="Seconds for each finger to open.")
    parser.add_argument("--reset-duration-s", type=float, default=1.0,
                        help="Seconds to close all fingers before each loop.")
    parser.add_argument("--closed-hold-s", type=float, default=0.5,
                        help="Seconds to hold the fully closed hand.")
    parser.add_argument("--opened-hold-s", type=float, default=1.0,
                        help="Seconds to hold after all selected fingers are open.")
    parser.add_argument("--between-fingers-s", type=float, default=0.25,
                        help="Pause between finger openings.")
    parser.add_argument("--loop-pause-s", type=float, default=0.3,
                        help="Pause before closing again.")
    parser.add_argument("--rate-hz", type=float, default=20.0,
                        help="Command update rate during slow motion.")
    parser.add_argument("--speed", type=int, default=200, help="Inspire speed register value.")
    parser.add_argument("--force", type=int, default=200, help="Inspire force register value.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print targets without sending commands.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sequence(
        "both" if args.both_hands else args.hand,
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
    )


if __name__ == "__main__":
    main()
