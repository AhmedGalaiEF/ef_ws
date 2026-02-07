#!/usr/bin/env python3
"""
Simple high-level multi-step motion sequence for G1.

Sequence:
  1) Walk forward N meters
  2) Wave right hand
  3) Turn in place (degrees)
  4) Walk forward N meters

Connects over DDS using --iface (default: eth0).
"""
from __future__ import annotations

import argparse
import math
import time
from typing import Optional

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc


def _command_velocity(client: LocoClient, vx: float, vy: float, vyaw: float) -> None:
    if hasattr(client, "SetVelocity"):
        client.SetVelocity(float(vx), float(vy), float(vyaw))
    else:
        client.Move(float(vx), float(vy), float(vyaw))


def _stop(client: LocoClient) -> None:
    if hasattr(client, "StopMove"):
        client.StopMove()
    else:
        client.Move(0.0, 0.0, 0.0)


def _send_for_duration(client: LocoClient, vx: float, vy: float, vyaw: float, duration: float, cmd_hz: float) -> None:
    dt = 1.0 / max(1e-6, cmd_hz)
    end_time = time.monotonic() + max(0.0, duration)
    next_cmd = time.monotonic()
    while time.monotonic() < end_time:
        now = time.monotonic()
        if now >= next_cmd:
            _command_velocity(client, vx, vy, vyaw)
            next_cmd += dt
        else:
            time.sleep(min(0.005, next_cmd - now))


class _WaveHelper:
    def __init__(self) -> None:
        self._client = None
        self._mode_set = False
        self._init_attempted = False

    def _try_init(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        # Try a few likely client classes and paths.
        candidates = [
            ("unitree_sdk2py.g1.arm.g1_arm_client", "G1ArmClient"),
            ("unitree_sdk2py.g1.arm.g1_arm_client", "ArmClient"),
            ("unitree_sdk2py.g1.hand.g1_hand_client", "G1HandClient"),
            ("unitree_sdk2py.g1.hand.g1_hand_client", "HandClient"),
        ]
        for mod, cls in candidates:
            try:
                module = __import__(mod, fromlist=[cls])
                client_cls = getattr(module, cls)
                client = client_cls()
                if hasattr(client, "SetTimeout"):
                    client.SetTimeout(5.0)
                if hasattr(client, "Init"):
                    client.Init()
                self._client = client
                return
            except Exception:
                continue

    def wave_right(self) -> bool:
        self._try_init()
        if self._client is None:
            return False

        # Prefer a built-in wave if it exists.
        for name in ["WaveRight", "Wave", "WaveHand", "WaveRightHand"]:
            if hasattr(self._client, name):
                getattr(self._client, name)()
                return True

        # Try a minimal joint-motion style API if available.
        # These names are intentionally conservative to avoid unexpected behavior.
        if hasattr(self._client, "SetArmMode") and not self._mode_set:
            try:
                self._client.SetArmMode(1)
                self._mode_set = True
            except Exception:
                pass

        # If there is a generic "Pose" or "Move" call, we skip to avoid guessing.
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 simple multi-step motion sequence.")
    parser.add_argument("--iface", default="eth0", help="network interface for DDS")
    parser.add_argument("--walk-m", type=float, default=1.0, help="walk distance (meters)")
    parser.add_argument("--walk-v", type=float, default=0.3, help="walk speed (m/s)")
    parser.add_argument("--turn-deg", type=float, default=180.0, help="turn angle (degrees)")
    parser.add_argument("--turn-vyaw", type=float, default=0.8, help="turn rate (rad/s)")
    parser.add_argument("--cmd-hz", type=float, default=20.0, help="command rate (Hz)")
    parser.add_argument("--no-wave", action="store_true", help="skip right-hand wave")
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.iface)

    loco = LocoClient()
    if hasattr(loco, "SetTimeout"):
        loco.SetTimeout(5.0)
    loco.Init()

    wave = _WaveHelper()

    walk_time = abs(args.walk_m) / max(1e-6, abs(args.walk_v))
    turn_rad = math.radians(args.turn_deg)
    turn_time = abs(turn_rad) / max(1e-6, abs(args.turn_vyaw))
    turn_dir = 1.0 if turn_rad >= 0.0 else -1.0

    try:
        # 1) Walk forward
        _send_for_duration(loco, vx=abs(args.walk_v), vy=0.0, vyaw=0.0, duration=walk_time, cmd_hz=args.cmd_hz)
        _stop(loco)
        time.sleep(0.5)

        # 2) Wave right hand
        if not args.no_wave:
            ok = wave.wave_right()
            if not ok:
                print("Wave: right-hand wave not supported by available SDK; skipping.")
            time.sleep(1.0)

        # 3) Turn in place
        _send_for_duration(loco, vx=0.0, vy=0.0, vyaw=turn_dir * abs(args.turn_vyaw), duration=turn_time, cmd_hz=args.cmd_hz)
        _stop(loco)
        time.sleep(0.5)

        # 4) Walk forward
        _send_for_duration(loco, vx=abs(args.walk_v), vy=0.0, vyaw=0.0, duration=walk_time, cmd_hz=args.cmd_hz)
        _stop(loco)
    finally:
        _stop(loco)


if __name__ == "__main__":
    main()
