#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
    from sdk_hand import (
        Dex3HandController,
        HAND_MAX_LIMITS,
        HAND_MIN_LIMITS,
        HAND_THUMB_0_HOLD_TARGETS,
        hand_closed_targets,
        hand_open_targets,
    )
except ImportError as exc:
    raise SystemExit(
        "Could not import sdk_hand. Run this script from modules/scripts or keep sdk_hand.py in modules/."
    ) from exc


HANDS = ("left", "right")
HAND_STATE_TOPIC_BY_SIDE = {
    "left": "rt/dex3/left/state",
    "right": "rt/dex3/right/state",
}
RIGHT_HAND_STATE_TOPIC = HAND_STATE_TOPIC_BY_SIDE["right"]
CALIBRATION_PATH = os.path.join(SCRIPT_DIR, "dex3_closed_hand_reference.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Close the left and right Dex3 hands at the same time."
    )
    parser.add_argument("--iface", default="eth0", help="Network interface for DDS traffic.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument("--hold-s", type=float, default=2.0,
                        help="Seconds to keep publishing the closed command.")
    parser.add_argument("--ramp-s", type=float, default=1.0,
                        help="Seconds to ramp both hands closed.")
    parser.add_argument("--rate-hz", type=float, default=50.0, help="DDS publish rate.")
    parser.add_argument("--kp", type=float, default=1.2, help="Joint proportional gain.")
    parser.add_argument("--kd", type=float, default=0.05, help="Joint derivative gain.")
    parser.add_argument("--tau", type=float, default=0.05, help="Feed-forward torque.")
    parser.add_argument(
        "--calibration",
        default=CALIBRATION_PATH,
        help="JSON file used to save/load the measured right-hand closed pose.",
    )
    parser.add_argument(
        "--capture-right-closed",
        action="store_true",
        help="Read the current right-hand joint state and save it as the closed reference, then exit.",
    )
    parser.add_argument(
        "--capture-timeout-s",
        type=float,
        default=5.0,
        help="Seconds to wait for right-hand state while capturing calibration.",
    )
    parser.add_argument(
        "--state-timeout-s",
        type=float,
        default=1.0,
        help="Seconds to wait for current hand states before ramping.",
    )
    parser.add_argument(
        "--open-first-s",
        type=float,
        default=0.0,
        help="Optionally publish the open pose to both hands before closing.",
    )
    return parser.parse_args()


def clamp_targets(hand: str, targets: list[float]) -> list[float]:
    return [
        max(lo, min(hi, float(value)))
        for value, lo, hi in zip(targets, HAND_MIN_LIMITS[hand], HAND_MAX_LIMITS[hand])
    ]


def mirror_right_to_left(targets: list[float]) -> list[float]:
    mirrored = [-float(value) for value in targets]
    mirrored[0] = HAND_THUMB_0_HOLD_TARGETS["left"]
    return mirrored


class HandStateSubscriber:
    def __init__(self, hand: str) -> None:
        self.hand = str(hand)
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._positions: list[float] | None = None
        self._timestamp = 0.0
        self._sub = ChannelSubscriber(HAND_STATE_TOPIC_BY_SIDE[self.hand], HandState_)
        self._sub.Init(self._callback, 50)

    def _callback(self, msg: Any) -> None:
        try:
            positions = [float(msg.motor_state[idx].q) for idx in range(7)]
        except Exception:
            return
        with self._lock:
            self._positions = positions
            self._timestamp = time.time()
        self._event.set()

    def wait(self, timeout_s: float) -> tuple[list[float], float] | None:
        if not self._event.wait(max(0.1, float(timeout_s))):
            return None
        with self._lock:
            if self._positions is None:
                return None
            return list(self._positions), float(self._timestamp)


def load_calibrated_right_closed(path: str) -> list[float] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not load calibration from {path}: {exc}")
        return None

    try:
        targets = payload["right_closed"]
    except (TypeError, KeyError):
        print(f"Calibration file {path} does not contain right_closed.")
        return None
    if not isinstance(targets, list) or len(targets) != 7:
        print(f"Calibration file {path} has invalid right_closed targets.")
        return None
    return clamp_targets("right", [float(value) for value in targets])


def save_calibrated_right_closed(path: str, targets: list[float], args: argparse.Namespace) -> None:
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "source": RIGHT_HAND_STATE_TOPIC,
        "iface": str(args.iface),
        "domain_id": int(args.domain_id),
        "right_closed": clamp_targets("right", targets),
        "left_closed": clamp_targets("left", mirror_right_to_left(targets)),
        "note": "left_closed mirrors right_closed except thumb_0, then clips to left-hand limits",
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def capture_right_closed(args: argparse.Namespace) -> int:
    ChannelFactoryInitialize(int(args.domain_id), str(args.iface))
    subscriber = HandStateSubscriber("right")
    snapshot = subscriber.wait(float(args.capture_timeout_s))
    if snapshot is None:
        print(
            "No right-hand state received. Check iface/domain and that "
            f"{RIGHT_HAND_STATE_TOPIC} is publishing."
        )
        return 1

    positions, timestamp = snapshot
    right_closed = clamp_targets("right", positions)
    save_calibrated_right_closed(args.calibration, right_closed, args)
    print("Saved right-hand closed reference:")
    print(f"  file: {args.calibration}")
    print(f"  age_s: {max(0.0, time.time() - timestamp):.3f}")
    print(f"  right_closed: {right_closed}")
    print(f"  left_closed:  {clamp_targets('left', mirror_right_to_left(right_closed))}")
    return 0


def open_targets(hand: str) -> list[float]:
    return clamp_targets(hand, hand_open_targets(hand))


def fallback_closed_targets(hand: str) -> list[float]:
    return clamp_targets(hand, hand_closed_targets(hand))


def closed_targets(hand: str, calibrated_right_closed: list[float] | None) -> list[float]:
    if calibrated_right_closed is None:
        return fallback_closed_targets(hand)
    if hand == "right":
        return clamp_targets("right", calibrated_right_closed)
    return clamp_targets("left", mirror_right_to_left(calibrated_right_closed))


def read_current_hand_targets(args: argparse.Namespace) -> dict[str, list[float] | None]:
    subscribers = {hand: HandStateSubscriber(hand) for hand in HANDS}
    starts: dict[str, list[float] | None] = {}
    deadline = time.time() + max(0.1, float(args.state_timeout_s))
    for hand in HANDS:
        remaining = max(0.1, deadline - time.time())
        snapshot = subscribers[hand].wait(remaining)
        if snapshot is None:
            starts[hand] = None
            continue
        positions, _timestamp = snapshot
        starts[hand] = clamp_targets(hand, positions)
    return starts


def blend_targets(start: list[float], stop: list[float], alpha: float) -> list[float]:
    blend = min(1.0, max(0.0, float(alpha)))
    return [src + (dst - src) * blend for src, dst in zip(start, stop)]


def publish_both_once(
    controllers: dict[str, Dex3HandController],
    targets_by_hand: dict[str, list[float]],
    args: argparse.Namespace,
) -> None:
    for hand in HANDS:
        controllers[hand].write_targets_once(
            targets_by_hand[hand],
            kp=args.kp,
            kd=args.kd,
            tau=args.tau,
        )


def publish_both_for(
    controllers: dict[str, Dex3HandController],
    targets_by_hand: dict[str, list[float]],
    seconds: float,
    args: argparse.Namespace,
) -> None:
    rate = max(1.0, float(args.rate_hz))
    steps = max(1, int(round(max(0.0, float(seconds)) * rate)))
    dt = 1.0 / rate
    for _ in range(steps):
        publish_both_once(controllers, targets_by_hand, args)
        time.sleep(dt)


def ramp_both_closed(
    controllers: dict[str, Dex3HandController],
    calibrated_right_closed: list[float] | None,
    args: argparse.Namespace,
) -> dict[str, list[float]]:
    current_targets = read_current_hand_targets(args)
    starts = {
        hand: current_targets[hand] if current_targets[hand] is not None else open_targets(hand)
        for hand in HANDS
    }
    stops = {hand: closed_targets(hand, calibrated_right_closed) for hand in HANDS}

    missing_states = [hand for hand in HANDS if current_targets[hand] is None]
    if missing_states:
        print(
            "  state warning: no current state for "
            + ", ".join(missing_states)
            + "; ramping that hand from fallback open pose"
        )

    rate = max(1.0, float(args.rate_hz))
    steps = max(1, int(round(max(0.0, float(args.ramp_s)) * rate)))
    dt = 1.0 / rate

    for step_idx in range(1, steps + 1):
        alpha = step_idx / steps
        targets = {
            hand: blend_targets(starts[hand], stops[hand], alpha)
            for hand in HANDS
        }
        publish_both_once(controllers, targets, args)
        time.sleep(dt)

    return stops


def main() -> int:
    args = parse_args()
    if args.capture_right_closed:
        return capture_right_closed(args)

    calibrated_right_closed = load_calibrated_right_closed(args.calibration)
    controllers = {
        hand: Dex3HandController(hand=hand, iface=args.iface, domain_id=args.domain_id)
        for hand in HANDS
    }

    print(
        "Closing both Dex3 hands together: "
        f"iface={args.iface}, domain_id={args.domain_id}, rate_hz={args.rate_hz}"
    )
    if calibrated_right_closed is None:
        print(f"  calibration: not found at {args.calibration}; using fallback targets")
    else:
        print(f"  calibration: {args.calibration}")
    print(f"  left target:  {closed_targets('left', calibrated_right_closed)}")
    print(f"  right target: {closed_targets('right', calibrated_right_closed)}")

    if args.open_first_s > 0.0:
        publish_both_for(
            controllers,
            {hand: open_targets(hand) for hand in HANDS},
            args.open_first_s,
            args,
        )

    closed = ramp_both_closed(controllers, calibrated_right_closed, args)
    publish_both_for(controllers, closed, args.hold_s, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
