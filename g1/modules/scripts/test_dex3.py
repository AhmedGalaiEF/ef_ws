#!/usr/bin/env python3
"""Probe Dex3 hand command and state topics without moving the fingers."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from dds_env import default_dds_iface, ensure_channel_factory_initialized


HAND_STATE_ALIASES = {
    "left": (
        "rt/dex3/left/state",
        "rt/lf/dex3/left/state",
        "dex3/left/state",
        "lf/dex3/left/state",
    ),
    "right": (
        "rt/dex3/right/state",
        "rt/lf/dex3/right/state",
        "dex3/right/state",
        "lf/dex3/right/state",
    ),
}

HAND_COMMAND_TOPICS = {
    "left": "rt/dex3/left/cmd",
    "right": "rt/dex3/right/cmd",
}


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Dex3 state topics and command writer setup without changing hand targets."
    )
    parser.add_argument("--iface", default=default_dds_iface("eth0"), help="DDS network interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument("--seconds", type=positive_float, default=8.0, help="How long to listen.")
    parser.add_argument("--queue-size", type=int, default=10, help="DDS subscriber queue depth.")
    return parser.parse_args()


def resolve_hand_state_type() -> type[Any]:
    try:
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
    except ImportError as exc:
        raise RuntimeError("unitree_sdk2py HandState_ type is unavailable.") from exc
    return HandState_


def state_callback(topic: str, seen: dict[str, bool]) -> Callable[[Any], None]:
    def callback(message: Any) -> None:
        try:
            motors = len(getattr(message, "motor_state", []) or [])
            pressure_sensors = len(getattr(message, "press_sensor_state", []) or [])
        except Exception as exc:
            print(f"{topic}: received an unreadable state message: {exc}")
            return
        seen[topic] = True
        print(f"{topic}: state received, motors={motors}, pressure_sensors={pressure_sensors}")

    return callback


def probe_hand(hand: str, args: argparse.Namespace, hand_state_type: type[Any]) -> dict[str, bool]:
    from unitree_sdk2py.core.channel import ChannelSubscriber

    seen = {topic: False for topic in dict.fromkeys(HAND_STATE_ALIASES[hand])}
    subscribers = []
    for topic in seen:
        subscriber = ChannelSubscriber(topic, hand_state_type)
        subscriber.Init(state_callback(topic, seen), args.queue_size)
        subscribers.append(subscriber)

    print(f"{hand}: command topic={HAND_COMMAND_TOPICS[hand]}")
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        time.sleep(0.2)
    return seen


def main() -> int:
    args = parse_args()
    if args.domain_id < 0:
        raise SystemExit("--domain-id must be non-negative")
    if args.queue_size <= 0:
        raise SystemExit("--queue-size must be greater than zero")

    ensure_channel_factory_initialized(args.domain_id, args.iface)
    hand_state_type = resolve_hand_state_type()
    hands = ("left", "right") if args.hand == "both" else (args.hand,)
    all_seen: dict[str, bool] = {}
    for hand in hands:
        all_seen.update(probe_hand(hand, args, hand_state_type))

    print("\nState topics seen:")
    for topic, received in all_seen.items():
        print(f"  {topic}: {received}")
    return 0 if any(all_seen.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
