from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from typing import Iterable, Literal


Side = Literal["left", "right"]


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
    "right": InspireHandConfig(ip="192.168.124.211"),
    "left": InspireHandConfig(ip="192.168.124.210"),
}

# Register values are six angle targets in the order used by the Inspire SDK.
# The open target follows the repo's dds_publish.py example. The close target is
# the conservative target already tested successfully on this right hand.
HAND_OPEN_TARGET = [0, 0, 0, 0, 1000, 1000]
HAND_CLOSE_TARGET = [700, 700, 700, 700, 300, 300]


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


def hand_open(hand: str = "right", *, speed: int = 200, force: int = 200, hold: float = 0.0) -> None:
    """Open an Inspire hand by side: 'right' or 'left'."""
    _move_hand(hand, HAND_OPEN_TARGET, speed=speed, force=force, hold=hold)


def hand_close(hand: str = "right", *, speed: int = 200, force: int = 200, hold: float = 0.0) -> None:
    """Close an Inspire hand by side: 'right' or 'left'."""
    _move_hand(hand, HAND_CLOSE_TARGET, speed=speed, force=force, hold=hold)


def _move_hand(hand: str, target: list[int], *, speed: int, force: int, hold: float) -> None:
    side = _normalize_side(hand)
    config = HAND_CONFIGS[side]
    with ModbusTcp(config.ip, config.port, config.unit_id) as client:
        client.write_single_register(CLEAR_ERROR_REGISTER, 1)
        client.write_registers(SPEED_SET_REGISTER, [speed] * 6)
        client.write_registers(FORCE_SET_REGISTER, [force] * 6)
        client.write_registers(ANGLE_SET_REGISTER, target)
        if hold > 0:
            time.sleep(float(hold))


def _normalize_side(hand: str) -> Side:
    side = str(hand).strip().lower()
    if side in {"r", "right"}:
        return "right"
    if side in {"l", "left"}:
        return "left"
    raise ValueError("hand must be 'right' or 'left'")


if __name__ == "__main__":
    hand_open("right", hold=1.0)
    hand_close("right", hold=1.0)
    hand_open("right", hold=1.0)
