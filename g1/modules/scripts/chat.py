#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import string
import sys
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


DEFAULT_SYSTEM_PROMPT = (
    "You are the voice of a Unitree humanoid robot. Chat naturally with nearby people. "
    "Reply in no more than 25 words. "
    "Do not mention that you are a language model. Do not use markdown or hidden reasoning."
)
FILLER_TEXTS = {
    "ah",
    "eh",
    "er",
    "hmm",
    "hm",
    "mm",
    "uh",
    "um",
    "嗯",
    "呃",
    "啊",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen to robot ASR, answer with Ollama qwen3.5:9b, and speak replies through the robot."
    )
    parser.add_argument("--topic", default="/audio_msg", help="ROS 2 ASR topic to subscribe to.")
    parser.add_argument("--out", default="/tmp/robot_chat.jsonl", help="JSONL file for chat events.")
    parser.add_argument("--text-out", default="/tmp/robot_chat.txt", help="Plain text transcript output file.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL.")
    parser.add_argument("--model", default="qwen3.5:9b", help="Ollama model name.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt for the robot persona.")
    parser.add_argument("--temperature", type=float, default=0.4, help="Ollama sampling temperature.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Ollama request timeout in seconds.")
    parser.add_argument("--max-history", type=int, default=4, help="Maximum user/assistant messages to keep.")
    parser.add_argument("--num-predict", type=int, default=48, help="Maximum tokens to generate per reply.")
    parser.add_argument("--num-ctx", type=int, default=1024, help="Ollama context window size.")
    parser.add_argument("--keep-alive", default="15m", help="How long Ollama should keep the model loaded.")
    parser.add_argument("--num-thread", type=int, default=None, help="Optional Ollama CPU thread count.")
    parser.add_argument("--no-warmup", action="store_true", help="Do not preload the model at startup.")
    parser.add_argument("--iface", default="eth0", help="DDS interface for robot audio playback.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID for robot audio playback.")
    parser.add_argument("--volume", type=int, default=None, help="Optional playback volume 0-100.")
    parser.add_argument("--tts-language", default=None, help="Optional Piper language, for example en, de, fr, es, ar.")
    parser.add_argument("--startup-speech", default="chat mode activated", help="Text to speak when chat mode starts.")
    parser.add_argument("--no-startup-speech", action="store_true", help="Do not speak the startup phrase.")
    parser.add_argument("--headlight-color", default="#123456", help="Headlight color to set when chat mode starts.")
    parser.add_argument("--headlight-intensity", type=int, default=100, help="Startup headlight intensity 0-100.")
    parser.add_argument("--no-headlight", action="store_true", help="Do not change the headlight on startup.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Ignore ASR below this confidence.")
    parser.add_argument("--answer-fillers", action="store_true", help="Answer short filler utterances like um or hmm.")
    parser.add_argument("--no-reply", action="store_true", help="Generate and save replies; do not speak them.")
    parser.add_argument(
        "--post-speak-ignore-s",
        type=float,
        default=1.5,
        help="Ignore likely echo/self-ASR for this many seconds after speaking.",
    )
    return parser.parse_args()


def decode_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"raw": raw}


def clean_reply(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    # Qwen reasoning models can emit hidden-thought style sections. Never speak them.
    while "<think>" in text and "</think>" in text:
        before, rest = text.split("<think>", 1)
        _hidden, after = rest.split("</think>", 1)
        text = (before + after).strip()
    return " ".join(text.split())


class RobotChat(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("robot_chat")
        self.args = args
        self.out_path = Path(args.out).expanduser()
        self.text_out_path = Path(args.text_out).expanduser()
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.text_out_path.parent.mkdir(parents=True, exist_ok=True)

        self.messages: list[dict[str, str]] = [{"role": "system", "content": str(args.system_prompt)}]
        self.last_index: int | None = None
        self.last_text: str | None = None
        self.last_reply: str | None = None
        self.last_reply_ts = 0.0
        self.create_subscription(String, args.topic, self.on_audio_msg, 10)
        self.get_logger().info(
            f"Chatting from {args.topic} with Ollama model={args.model}; saving to {self.out_path}"
        )
        self._run_startup_actions()
        if not args.no_warmup:
            self._warm_up_ollama()

    def on_audio_msg(self, msg: String) -> None:
        payload = decode_payload(str(msg.data))
        text = str(payload.get("text", "")).strip()
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        index = self._payload_index(payload)
        received_at = time.time()

        if not self._should_answer(text, confidence, index, received_at):
            return

        self.get_logger().info(f'Heard: "{text}" confidence={confidence:.2f} index={index}')
        try:
            reply = self._ask_ollama(text)
        except Exception as exc:
            self.get_logger().error(f"Ollama request failed: {exc}")
            self._save_event(received_at, payload, text, None, error=str(exc))
            return

        self._save_event(received_at, payload, text, reply)
        self.get_logger().info(f'Reply: "{reply}"')

        if reply and not self.args.no_reply:
            code = self._speak_once(reply)
            self.get_logger().info(f"robot_say_once exited {code}")

        self.last_index = index
        self.last_text = text
        self.last_reply = reply
        self.last_reply_ts = time.time()

    def _should_answer(self, text: str, confidence: float, index: int | None, received_at: float) -> bool:
        if not text or confidence < float(self.args.min_confidence):
            return False
        if not self.args.answer_fillers and self._is_filler(text):
            return False
        if not any(char.isalnum() for char in text):
            return False
        if index is not None and index == self.last_index:
            return False
        if index is None and text == self.last_text and (received_at - self.last_reply_ts) < 2.0:
            return False
        if (received_at - self.last_reply_ts) < float(self.args.post_speak_ignore_s):
            return False
        if self.last_reply and self._similar(text, self.last_reply) >= 0.82:
            return False
        return True

    @staticmethod
    def _payload_index(payload: dict[str, Any]) -> int | None:
        try:
            value = payload.get("index")
            return int(value) if value is not None else None
        except Exception:
            return None

    @staticmethod
    def _similar(left: str, right: str) -> float:
        return SequenceMatcher(None, left.lower().strip(), right.lower().strip()).ratio()

    @staticmethod
    def _is_filler(text: str) -> bool:
        normalized = text.strip().lower().strip(string.punctuation + "，。！？、；：")
        return normalized in FILLER_TEXTS

    def _ask_ollama(self, user_text: str) -> str:
        max_history = max(1, int(self.args.max_history))
        messages = self.messages + [{"role": "user", "content": user_text}]
        del messages[1 : max(1, len(messages) - max_history)]

        body = {
            "model": str(self.args.model),
            "messages": messages,
            "stream": False,
            "keep_alive": str(self.args.keep_alive),
            "think": False,
            "options": {
                "temperature": float(self.args.temperature),
                "num_predict": int(self.args.num_predict),
                "num_ctx": int(self.args.num_ctx),
            },
        }
        if self.args.num_thread is not None:
            body["options"]["num_thread"] = int(self.args.num_thread)
        started = time.time()
        result = self._post_ollama_chat(body, timeout=float(self.args.timeout))
        self.get_logger().info(f"Ollama replied in {time.time() - started:.1f}s")

        reply = clean_reply(str(result.get("message", {}).get("content", "")))
        if not reply:
            reply = "I heard you, but I am not sure how to answer that yet."
        messages.append({"role": "assistant", "content": reply})
        self.messages = messages
        return reply

    def _warm_up_ollama(self) -> None:
        self.get_logger().info(f"Preloading Ollama model={self.args.model}")
        started = time.time()
        body = {
            "model": str(self.args.model),
            "messages": [
                {"role": "system", "content": "Answer with one short word."},
                {"role": "user", "content": "Ready?"},
            ],
            "stream": False,
            "keep_alive": str(self.args.keep_alive),
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 2,
                "num_ctx": int(self.args.num_ctx),
            },
        }
        if self.args.num_thread is not None:
            body["options"]["num_thread"] = int(self.args.num_thread)
        try:
            self._post_ollama_chat(body, timeout=float(self.args.timeout))
        except Exception as exc:
            self.get_logger().warning(f"Ollama warm-up failed: {exc}")
            return
        self.get_logger().info(f"Ollama warm-up finished in {time.time() - started:.1f}s")

    def _post_ollama_chat(self, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{str(self.args.ollama_url).rstrip('/')}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    def _run_startup_actions(self) -> None:
        if not self.args.no_headlight:
            try:
                code = self._set_headlight_once(str(self.args.headlight_color), int(self.args.headlight_intensity))
                self.get_logger().info(
                    f"Headlight set color={self.args.headlight_color} intensity={self.args.headlight_intensity} code={code}"
                )
            except Exception as exc:
                self.get_logger().warning(f"Headlight startup action failed: {exc}")

        startup_speech = str(self.args.startup_speech).strip()
        if startup_speech and not self.args.no_startup_speech:
            self.last_reply = startup_speech
            self.last_reply_ts = time.time()
            code = self._speak_once(startup_speech)
            self.last_reply_ts = time.time()
            self.get_logger().info(f'Startup speech "{startup_speech}" exited {code}')

    def _set_headlight_once(self, color: str, intensity: int) -> int:
        script = Path(__file__).with_name("robot_headlight_once.py")
        command = [
            sys.executable,
            str(script),
            "--color",
            str(color),
            "--intensity",
            str(max(0, min(100, int(intensity)))),
            "--iface",
            str(self.args.iface),
            "--domain-id",
            str(int(self.args.domain_id)),
        ]
        env = os.environ.copy()
        env.setdefault("CYCLONEDDS_HOME", "/home/unitree/cyclonedds_ws/install/cyclonedds")
        env.setdefault("CYCLONEDDS_URI", "/home/unitree/cyclonedds_ws/cyclonedds.xml")
        proc = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.stdout.strip():
            self.get_logger().info(proc.stdout.strip())
        return int(proc.returncode)

    def _save_event(
        self,
        received_at: float,
        payload: dict[str, Any],
        text: str,
        reply: str | None,
        *,
        error: str | None = None,
    ) -> None:
        record = {
            "received_at": received_at,
            "topic": self.args.topic,
            "model": self.args.model,
            "payload": payload,
            "text": text,
            "reply": reply,
        }
        if error:
            record["error"] = error
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with self.text_out_path.open("a", encoding="utf-8") as f:
            f.write(f"user: {text}\n")
            if reply:
                f.write(f"robot: {reply}\n")
            if error:
                f.write(f"error: {error}\n")

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
        proc = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.stdout.strip():
            self.get_logger().info(proc.stdout.strip())
        return int(proc.returncode)


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = RobotChat(args)
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
