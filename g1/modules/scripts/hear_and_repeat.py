#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import TYPE_CHECKING, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from dds_env import ensure_cyclonedds_environment
from sdk_sensors import LatestSubscriber

ensure_cyclonedds_environment()

if TYPE_CHECKING:
    from sdk_client import Robot


AUDIO_TOPIC = "rt/audio_msg"


class AsrSubscriber(LatestSubscriber):
    def __init__(self, topic: str = AUDIO_TOPIC) -> None:
        super().__init__(topic, String_)

    def get_latest_text(self) -> tuple[str | None, float]:
        msg, ts = self.get_latest()
        if msg is None:
            return None, ts
        try:
            return str(msg.data), ts
        except Exception:
            try:
                return str(msg.data()), ts
            except Exception:
                return str(msg), ts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen to the robot microphone ASR stream and repeat recognized speech via Robot.say()."
    )
    parser.add_argument("--iface", default="eth0",
                        help="DDS network interface, for example eth0 or enp3s0.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID.")
    parser.add_argument("--topic", default=AUDIO_TOPIC, help="ASR topic to subscribe to.")
    parser.add_argument("--volume", type=int, default=None,
                        help="Optional playback volume 0-100 for Robot.say().")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Ignore ASR results below this confidence threshold.",
    )
    parser.add_argument(
        "--poll-s",
        type=float,
        default=0.1,
        help="Polling interval while waiting for ASR messages.",
    )
    return parser.parse_args()


def decode_asr_payload(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def is_final_result(payload: dict[str, Any]) -> bool:
    return bool(payload.get("is_final", True))


def main() -> int:
    args = parse_args()
    from sdk_client import Robot

    robot = Robot(
        iface=args.iface,
        domain_id=args.domain_id,
        safety_boot=False,
        recover_dev_mode_on_init=False,
        auto_start_sensors=False,
    )
    asr = AsrSubscriber(args.topic)
    asr.start()

    print(f"Subscribed to {args.topic} on iface={args.iface} domain_id={args.domain_id}")
    print("Speak near the robot microphone array. Press Ctrl+C to stop.")

    last_index: int | None = None
    last_text: str | None = None
    last_repeat_ts = 0.0

    try:
        while True:
            raw, _ts = asr.get_latest_text()
            if not raw:
                time.sleep(max(0.01, float(args.poll_s)))
                continue

            payload = decode_asr_payload(raw)
            if payload is None:
                time.sleep(max(0.01, float(args.poll_s)))
                continue

            text = str(payload.get("text", "")).strip()
            if not text:
                time.sleep(max(0.01, float(args.poll_s)))
                continue
            if not is_final_result(payload):
                time.sleep(max(0.01, float(args.poll_s)))
                continue

            confidence = float(payload.get("confidence", 0.0) or 0.0)
            if confidence < float(args.min_confidence):
                time.sleep(max(0.01, float(args.poll_s)))
                continue

            index_value = payload.get("index")
            try:
                index = int(index_value) if index_value is not None else None
            except Exception:
                index = None

            now = time.time()
            if index is not None and index == last_index:
                time.sleep(max(0.01, float(args.poll_s)))
                continue
            if index is None and text == last_text and (now - last_repeat_ts) < 2.0:
                time.sleep(max(0.01, float(args.poll_s)))
                continue

            angle = payload.get("angle")
            language = payload.get("language", "unknown")
            print(f'Heard: "{text}" confidence={confidence:.2f} angle={angle} language={language}')
            code = robot.say(text, volume=args.volume)
            print(f"Robot.say returned {code}")

            last_index = index
            last_text = text
            last_repeat_ts = now
            time.sleep(max(0.01, float(args.poll_s)))
    except KeyboardInterrupt:
        print("\nStopping hear_and_repeat.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
