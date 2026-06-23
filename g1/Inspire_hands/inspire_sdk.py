#!/usr/bin/env python3
from __future__ import annotations

import socket
import struct
import time
from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal


ANGLE_SET_REGISTER = 1486
FORCE_SET_REGISTER = 1498
SPEED_SET_REGISTER = 1522
CLEAR_ERROR_REGISTER = 1004


@dataclass(frozen=True)
class InspireHandConfig:
    ip: str
    port: int = 6000
    unit_id: int = 1


HAND_CONFIGS: dict[str, InspireHandConfig] = {
    "right": InspireHandConfig(ip="192.168.123.210"),
    "left": InspireHandConfig(ip="192.168.123.211"),
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
Side = Literal["left", "right"]


class ModbusTcp(AbstractContextManager["ModbusTcp"]):
    def __init__(self, host: str, port: int = 6000, unit_id: int = 1, timeout: float = 2.0):
        self.host = host
        self.port = int(port)
        self.unit_id = int(unit_id)
        self.timeout = float(timeout)
        self.transaction_id = 1
        self.sock: socket.socket | None = None

    def __enter__(self) -> ModbusTcp:
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
        registers = [int(value) & 0xFFFF for value in values]
        payload = struct.pack(">" + "H" * len(registers), *registers)
        pdu = struct.pack(">BHHB", 16, int(address), len(registers), len(payload)) + payload
        response = self._request(pdu)
        expected = struct.pack(">BHH", 16, int(address), len(registers))
        if response != expected:
            raise RuntimeError("Unexpected Modbus write-multiple response")

    def read_holding_registers(self, address: int, count: int) -> list[int]:
        return self._read_registers(3, address, count)

    def read_input_registers(self, address: int, count: int) -> list[int]:
        return self._read_registers(4, address, count)

    def _read_registers(self, function: int, address: int, count: int) -> list[int]:
        count = int(count)
        if count <= 0:
            raise ValueError("count must be positive")

        response = self._request(struct.pack(">BHH", function, int(address), count))
        if len(response) < 2:
            raise RuntimeError("Malformed Modbus read response")
        byte_count = response[1]
        payload = response[2:]
        if byte_count != len(payload) or byte_count != count * 2:
            raise RuntimeError("Unexpected Modbus read payload length")
        return list(struct.unpack(">" + "H" * count, payload))

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


class NullClient:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def clamp_register(value: float | int) -> int:
    return max(0, min(1000, round(float(value))))


def normalize_side(hand: str) -> Side:
    side = str(hand).strip().lower()
    if side in {"r", "right"}:
        return "right"
    if side in {"l", "left"}:
        return "left"
    raise ValueError("hand must be 'right' or 'left'")


def normalize_hands(value: str) -> tuple[str, ...]:
    hand = str(value).strip().lower()
    if hand in {"b", "both", "both_hands", "both-hands"}:
        return ("left", "right")
    return (normalize_side(hand),)


def interpolate(start: Sequence[int], stop: Sequence[int], alpha: float) -> list[int]:
    return [
        clamp_register(start_value + (stop_value - start_value) * alpha)
        for start_value, stop_value in zip(start, stop)
    ]


def send_angles(
    hand: str,
    angles: Sequence[int],
    *,
    speed: int = 200,
    force: int = 200,
    hold: float = 0.0,
) -> None:
    if len(angles) != 6:
        raise ValueError("Inspire angle targets must contain exactly 6 values.")

    side = normalize_side(hand)
    config = HAND_CONFIGS[side]
    target = [clamp_register(value) for value in angles]

    with ModbusTcp(config.ip, config.port, config.unit_id) as client:
        client.write_single_register(CLEAR_ERROR_REGISTER, 1)
        client.write_registers(SPEED_SET_REGISTER, [clamp_register(speed)] * 6)
        client.write_registers(FORCE_SET_REGISTER, [clamp_register(force)] * 6)
        client.write_registers(ANGLE_SET_REGISTER, target)

    if hold > 0:
        time.sleep(float(hold))


def _move_hand(hand: str, angles: Sequence[int], *, speed: int = 200, force: int = 200, hold: float = 0.0) -> None:
    send_angles(hand, angles, speed=speed, force=force, hold=hold)


def open_next_finger(current: Sequence[int], finger: str) -> list[int]:
    target = [clamp_register(value) for value in current]
    for idx in FINGER_TO_IDXS[finger]:
        target[idx] = HAND_OPEN_TARGET[idx]
    return target


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
        values = interpolate(current, target, alpha)
        if dry_run:
            print(f"target={values}")
        else:
            if client is None:
                raise RuntimeError("Modbus client is required unless dry-run is enabled")
            client.write_registers(SPEED_SET_REGISTER, [clamp_register(speed)] * 6)
            client.write_registers(FORCE_SET_REGISTER, [clamp_register(force)] * 6)
            client.write_registers(ANGLE_SET_REGISTER, values)
        if delay_s > 0:
            time.sleep(delay_s)

    return [clamp_register(value) for value in target]


def decode_u16(words: Sequence[int]) -> list[int]:
    return [int(value) & 0xFFFF for value in words]


def decode_i16(words: Sequence[int]) -> list[int]:
    values: list[int] = []
    for value in words:
        value &= 0xFFFF
        values.append(value - 0x10000 if value & 0x8000 else value)
    return values


def decode_f32(words: Sequence[int], *, word_order: str = "big") -> list[float]:
    if len(words) % 2:
        raise ValueError("word count must be even to decode float32 values")

    floats: list[float] = []
    for idx in range(0, len(words), 2):
        first, second = int(words[idx]) & 0xFFFF, int(words[idx + 1]) & 0xFFFF
        pair = (first, second) if word_order == "big" else (second, first)
        floats.append(struct.unpack(">f", struct.pack(">HH", *pair))[0])
    return floats
