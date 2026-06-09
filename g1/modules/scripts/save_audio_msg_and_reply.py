#!/usr/bin/env python3
from __future__ import annotations
from std_msgs.msg import String
from rclpy.node import Node
import rclpy

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save /audio_msg ASR JSON payloads and speak recognized text back through the robot."
    )
    parser.add_argument("--topic", default="/audio_msg", help="ROS 2 ASR topic to subscribe to.")
    parser.add_argument("--out", default="/tmp/audio_msg_reply.jsonl",
                        help="JSONL file for saved ASR payloads.")
    parser.add_argument("--text-out", default="/tmp/audio_msg_reply.txt",
                        help="Plain text transcript output file.")
    parser.add_argument("--iface", default="eth0", help="DDS interface for robot audio playback.")
    parser.add_argument("--domain-id", type=int, default=0,
                        help="DDS domain ID for robot audio playback.")
    parser.add_argument("--volume", type=int, default=None, help="Optional playback volume 0-100.")
    parser.add_argument("--tts-language", default=None,
                        help="Optional Piper language, for example en, de, fr, es, ar.")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Ignore ASR below this confidence.")
    parser.add_argument("--no-reply", action="store_true",
                        help="Only save messages; do not speak them back.")
    return parser.parse_args()


def decode_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"raw": raw}


class AudioMsgSaver(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("save_audio_msg_and_reply")
        self.args = args
        self.out_path = Path(args.out).expanduser()
        self.text_out_path = Path(args.text_out).expanduser()
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.text_out_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_index: int | None = None
        self.last_text: str | None = None
        self.last_reply_ts = 0.0
        self.create_subscription(String, args.topic, self.on_audio_msg, 10)
        self.get_logger().info(f"Saving {args.topic} to {self.out_path} and {self.text_out_path}")

    def on_audio_msg(self, msg: String) -> None:
        payload = decode_payload(str(msg.data))
        text = str(payload.get("text", "")).strip()
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        index = self._payload_index(payload)
        received_at = time.time()

        record = {
            "received_at": received_at,
            "topic": self.args.topic,
            "payload": payload,
            "text": text,
        }
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if text:
            with self.text_out_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

        if not text or confidence < float(self.args.min_confidence):
            return
        if not any(char.isalnum() for char in text):
            return
        if index is not None and index == self.last_index:
            return
        if index is None and text == self.last_text and (received_at - self.last_reply_ts) < 2.0:
            return

        self.get_logger().info(f'Heard: "{text}" confidence={confidence:.2f} index={index}')
        if not self.args.no_reply:
            code = self._speak_once(text)
            self.get_logger().info(f"robot_say_once exited {code}")

        self.last_index = index
        self.last_text = text
        self.last_reply_ts = received_at

    @staticmethod
    def _payload_index(payload: dict[str, Any]) -> int | None:
        try:
            value = payload.get("index")
            return int(value) if value is not None else None
        except Exception:
            return None

    def _speak_once(self, text: str) -> int:
        script = Path(__file__).with_name("robot_say_once.py")
        command = [
            sys.executable,
            str(script),
            text,
            "--iface",
            str(self.args.iface),
            "--domain-id",
            str(int(self.args.domain_id)),
        ]
        if self.args.volume is not None:
            command.extend(["--volume", str(int(self.args.volume))])
        if self.args.tts_language:
            command.extend(["--language", str(self.args.tts_language)])
        env = os.environ.copy()
        env.setdefault("CYCLONEDDS_HOME", "/home/unitree/cyclonedds_ws/install/cyclonedds")
        env.setdefault("CYCLONEDDS_URI", "/home/unitree/cyclonedds_ws/cyclonedds.xml")
        proc = subprocess.run(command, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.stdout.strip():
            self.get_logger().info(proc.stdout.strip())
        return int(proc.returncode)


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = AudioMsgSaver(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
