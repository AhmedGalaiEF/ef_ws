#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import string
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FILLERS = {"ah", "eh", "er", "hmm", "hm", "mm", "uh", "um"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Dex3 tactile object-detection chatbot. No arm or motion control."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--token", default="",
                        help="Optional bearer/query/JSON token for HTTP /asr and /command.")
    parser.add_argument("--stdin", action="store_true",
                        help="Also read typed questions from stdin.")
    parser.add_argument("--audio-topic", default="/audio_msg")
    parser.add_argument("--filtered-audio-topic", default="/audio_msg/filter")
    parser.add_argument("--no-ros-audio-bridge", action="store_true",
                        help="Do not start the ROS /audio_msg to HTTP bridge subprocess.")
    parser.add_argument("--ros-audio-bridge-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--bridge-url", default="", help=argparse.SUPPRESS)
    parser.add_argument("--iface", default=os.environ.get("G1_IFACE", "eth0"))
    parser.add_argument("--domain-id", type=int, default=int(os.environ.get("G1_DOMAIN_ID", "0")))
    parser.add_argument("--queue-len", type=int, default=20)
    parser.add_argument("--tactile-threshold", type=float, default=100000.0,
                        help="Deprecated absolute threshold kept for compatibility; holding uses pressure delta.")
    parser.add_argument("--tactile-delta-threshold", type=float, default=5000.0,
                        help="Raw pressure increase from startup baseline required before a hand is considered holding.")
    parser.add_argument("--tactile-timeout-s", type=float, default=2.0,
                        help="Seconds to wait for initial Dex3 hand state at startup.")
    parser.add_argument("--tactile-max-age-s", type=float, default=2.0,
                        help="Maximum accepted age for tactile data used in answers.")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--post-speak-ignore-s", type=float, default=1.5)
    parser.add_argument("--answer-fillers", action="store_true")
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--volume", type=int, default=None)
    parser.add_argument("--tts-language", default=None)
    parser.add_argument("--startup-speech", default="Dex3 object detection ready")
    parser.add_argument("--no-startup-speech", action="store_true")
    return parser.parse_args()


def compact_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def decode_payload(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    return {"text": raw}


def is_holding_question(text: str) -> bool:
    normalized = " ".join(
        str(text).lower().strip().strip(string.punctuation + "，。！？、；：").split()
    )
    if not normalized:
        return False
    direct_phrases = (
        "are you holding something",
        "are you holding anything",
        "are you holding an object",
        "do you hold something",
        "do you hold anything",
        "do you have something in your hand",
        "do you have something in your hands",
        "is there something in your hand",
        "is there something in your hands",
        "what are you holding",
        "which hand is holding something",
        "which hand are you holding something in",
    )
    if any(phrase in normalized for phrase in direct_phrases):
        return True
    has_holding_word = any(word in normalized for word in ("holding", "hold", "gripping", "grip"))
    has_object_word = any(word in normalized for word in ("something", "anything", "object", "thing"))
    has_robot_word = any(word in normalized for word in ("you", "your"))
    has_hand_word = bool(re.search(r"\bhands?\b", normalized))
    return has_holding_word and has_robot_word and (has_object_word or has_hand_word)


def is_recalibrate_request(text: str) -> bool:
    normalized = " ".join(
        str(text).lower().strip().strip(string.punctuation + "，。！？、；：").split()
    )
    return any(
        phrase in normalized
        for phrase in (
            "recalibrate",
            "calibrate tactile",
            "calibrate your hands",
            "reset tactile baseline",
            "reset hand baseline",
            "empty hand baseline",
        )
    )


class Speaker:
    def __init__(self, args: argparse.Namespace, logger: Any) -> None:
        self.args = args
        self.logger = logger
        self.lock = threading.Lock()
        self.current_proc: subprocess.Popen[str] | None = None

    def say(self, text: str) -> int:
        speech = compact_text(text)
        if not speech:
            return 0
        self.logger.info(f"robot response text={speech!r}")
        if self.args.no_speech:
            return 0
        command = [
            sys.executable,
            str(SCRIPT_DIR / "robot_say_once.py"),
            speech,
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
        with self.lock:
            proc = subprocess.Popen(
                command,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.current_proc = proc
        try:
            output, _ = proc.communicate()
        finally:
            with self.lock:
                if self.current_proc is proc:
                    self.current_proc = None
        if output and output.strip():
            self.logger.info(output.strip())
        return int(proc.returncode or 0)

    def say_async(self, text: str) -> None:
        threading.Thread(target=self.say, args=(text,), daemon=True).start()


class PrintLogger:
    def info(self, message: str) -> None:
        print(message, flush=True)

    def warning(self, message: str) -> None:
        print(f"WARNING: {message}", file=sys.stderr, flush=True)

    def error(self, message: str) -> None:
        print(f"ERROR: {message}", file=sys.stderr, flush=True)


class Dex3HoldingState:
    def __init__(self, args: argparse.Namespace, logger: Any) -> None:
        from dds_env import default_dds_iface, ensure_channel_factory_initialized
        from test_dex3_tactile import (
            HAND_STATE_TOPIC_BY_SIDE,
            LatestTactileState,
            collect_latest_snapshots,
            is_invalid_tactile_value,
            wait_for_initial_snapshots,
        )

        self.args = args
        self.logger = logger
        self._collect_latest_snapshots = collect_latest_snapshots
        self._is_invalid_tactile_value = is_invalid_tactile_value
        iface = str(args.iface) if args.iface else default_dds_iface("eth0")
        self.iface = iface
        ensure_channel_factory_initialized(int(args.domain_id), iface)
        self.subscribers = [
            LatestTactileState(hand, HAND_STATE_TOPIC_BY_SIDE[hand], int(args.queue_len))
            for hand in ("left", "right")
        ]
        wait_for_initial_snapshots(self.subscribers, float(args.tactile_timeout_s))
        self.baselines = self._collect_latest_snapshots(self.subscribers)
        logger.info(
            "Dex3 tactile subscribers: "
            + ", ".join(f"{sub.hand}={sub.topic}" for sub in self.subscribers)
        )
        logger.info("Dex3 empty-hand tactile baseline captured.")

    def answer(self) -> dict[str, Any]:
        snapshots = self._collect_latest_snapshots(self.subscribers)
        threshold = float(self.args.tactile_delta_threshold)
        max_age_s = float(self.args.tactile_max_age_s)
        holding: list[str] = []
        hands: dict[str, Any] = {}
        fresh_hands = 0
        for hand in ("left", "right"):
            snapshot = snapshots.get(hand)
            if snapshot is None:
                hands[hand] = {"ok": False, "reason": "missing"}
                continue
            age_s = max(0.0, time.time() - float(snapshot.timestamp))
            fresh = age_s <= max_age_s
            if fresh:
                fresh_hands += 1
            max_delta = self._max_pressure_delta(hand, snapshot)
            max_value = snapshot.max_value
            is_holding = fresh and max_delta is not None and max_delta >= threshold
            if is_holding:
                holding.append(hand)
            hands[hand] = {
                "ok": fresh,
                "age_s": round(age_s, 3),
                "max": max_value,
                "max_delta": max_delta,
                "delta_threshold": threshold,
                "holding": is_holding,
                "valid_count": snapshot.valid_count,
                "active_count": snapshot.active_count,
            }
        if fresh_hands == 0:
            answer = "I cannot read my hand sensors right now."
            return {"ok": False, "answer": answer, "holding": holding, "hands": hands}
        if holding == ["left", "right"]:
            answer = "Yes, I am holding something in both hands."
        elif holding == ["left"]:
            answer = "Yes, I am holding something in my left hand."
        elif holding == ["right"]:
            answer = "Yes, I am holding something in my right hand."
        else:
            answer = "No."
        return {"ok": True, "answer": answer, "holding": holding, "hands": hands}

    def recalibrate(self) -> dict[str, Any]:
        self.baselines = self._collect_latest_snapshots(self.subscribers)
        answer = "Tactile baseline recalibrated."
        return {"ok": True, "answer": answer}

    def _max_pressure_delta(self, hand: str, snapshot: Any) -> float | None:
        baseline = self.baselines.get(hand)
        if baseline is None:
            return None
        max_delta: float | None = None
        for sensor_idx, sensor in enumerate(snapshot.sensors):
            if sensor_idx >= len(baseline.sensors):
                continue
            base_values = baseline.sensors[sensor_idx].values
            for value_idx, value in enumerate(sensor.values):
                if value_idx >= len(base_values):
                    continue
                base_value = base_values[value_idx]
                if self._is_invalid_tactile_value(value) or self._is_invalid_tactile_value(base_value):
                    continue
                delta = float(value) - float(base_value)
                if max_delta is None or delta > max_delta:
                    max_delta = delta
        return None if max_delta is None else round(max_delta, 1)


class Dex3ObjectDetectionBot:
    def __init__(self, args: argparse.Namespace, dex3: Dex3HoldingState, logger: Any) -> None:
        self.args = args
        self.logger = logger
        self.speaker = Speaker(args, logger)
        self.dex3 = dex3
        self.last_index: int | None = None
        self.last_text = ""
        self.last_reply = ""
        self.last_reply_ts = 0.0
        self.logger.info(
            f"dex3_obj_detection_bot ready http=http://{args.host}:{int(args.port)}/asr "
            f"threshold={float(args.tactile_threshold):.1f}"
        )
        if not args.no_startup_speech and compact_text(args.startup_speech):
            self.speaker.say_async(args.startup_speech)

    def handle_payload(self, payload: dict[str, Any], source: str) -> dict[str, Any]:
        text = compact_text(str(payload.get("text", payload.get("prompt", payload.get("raw", "")))))
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        index = self._payload_index(payload)
        if not self._should_answer(text, confidence, index):
            return {"ok": False, "accepted": False, "answer": "", "ignored": True, "source": source, "text": text}
        self.last_index = index
        self.last_text = text
        return self.handle_text(text, source=source)

    def handle_text(self, text: str, source: str) -> dict[str, Any]:
        self.logger.info(f"{source} accepted text={text!r}")
        if is_recalibrate_request(text):
            result = self.dex3.recalibrate()
        elif is_holding_question(text):
            result = self.dex3.answer()
        else:
            result = {"ok": False, "answer": "", "ignored": True, "reason": "unsupported_question"}
        result.update({"source": source, "text": text, "time": time.time()})
        self.publish_result(result)
        answer = compact_text(str(result.get("answer", "")))
        if answer and not result.get("ignored", False):
            self.last_reply = answer
            self.last_reply_ts = time.time()
            self.speaker.say_async(answer)
        return result

    def publish_result(self, result: dict[str, Any]) -> None:
        print(json.dumps(result, sort_keys=True, default=str), flush=True)

    def _should_answer(self, text: str, confidence: float, index: int | None) -> bool:
        if not text or confidence < float(self.args.min_confidence):
            return False
        normalized = text.strip().lower().strip(string.punctuation + "，。！？、；：")
        if not self.args.answer_fillers and normalized in FILLERS:
            return False
        if not any(char.isalnum() for char in text):
            return False
        if index is not None and index == self.last_index:
            return False
        if text == self.last_text and time.time() - self.last_reply_ts < 2.0:
            return False
        if time.time() - self.last_reply_ts < float(self.args.post_speak_ignore_s):
            return False
        return True

    @staticmethod
    def _payload_index(payload: dict[str, Any]) -> int | None:
        try:
            value = payload.get("index")
            return int(value) if value is not None else None
        except Exception:
            return None


def authorized(headers: Any, path: str, payload: dict[str, Any], token: str) -> bool:
    if not token:
        return True
    if str(headers.get("Authorization", "")) == f"Bearer {token}":
        return True
    if "?" in path and f"token={token}" in path.split("?", 1)[1]:
        return True
    return str(payload.get("token", "")) == token


def serve_http(bot: Dex3ObjectDetectionBot) -> http.server.ThreadingHTTPServer:
    token = str(bot.args.token or "")

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "Dex3ObjectDetectionBot/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            bot.logger.info("http " + (fmt % args))

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self.send_json(200, {"ok": True})

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self.send_json(200, {"ok": True, "service": "dex3_obj_detection_bot"})
                return
            self.send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path not in {"/asr", "/command"}:
                self.send_json(404, {"ok": False, "error": "not_found"})
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(min(length, 64_000)).decode("utf-8", errors="replace")
            payload = decode_payload(raw)
            if not authorized(self.headers, self.path, payload, token):
                self.send_json(401, {"ok": False, "error": "unauthorized"})
                return
            result = bot.handle_payload(payload, source="http")
            self.send_json(200, {"accepted": not result.get("ignored", False), **result})

    server = http.server.ThreadingHTTPServer((str(bot.args.host), int(bot.args.port)), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def stdin_loop(bot: Dex3ObjectDetectionBot) -> None:
    for line in sys.stdin:
        text = compact_text(line)
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            return
        bot.handle_text(text, source="stdin")


def run_ros_audio_bridge(args: argparse.Namespace) -> int:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class RosAudioBridge(Node):
        def __init__(self) -> None:
            super().__init__("dex3_obj_detection_audio_bridge")
            self.bridge_url = str(args.bridge_url or f"http://127.0.0.1:{int(args.port)}/asr")
            self.token = str(args.token or "")
            self.create_subscription(String, args.audio_topic, self.on_audio, 10)
            if str(args.filtered_audio_topic) and str(args.filtered_audio_topic) != str(args.audio_topic):
                self.create_subscription(String, args.filtered_audio_topic, self.on_audio, 10)
            self.get_logger().info(
                f"bridging ROS audio {args.audio_topic}"
                + (f" and {args.filtered_audio_topic}" if str(args.filtered_audio_topic) != str(args.audio_topic) else "")
                + f" -> {self.bridge_url}"
            )

        def on_audio(self, msg: String) -> None:
            payload = decode_payload(str(msg.data))
            if "text" not in payload and "raw" not in payload:
                return
            body = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            request = urllib.request.Request(self.bridge_url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=3.0) as response:
                    reply = response.read().decode("utf-8", errors="replace")
                self.get_logger().info(f"forwarded audio text={payload.get('text', payload.get('raw', ''))!r} response={reply}")
            except urllib.error.URLError as exc:
                self.get_logger().warning(f"failed to forward audio to {self.bridge_url}: {exc}")

    rclpy.init()
    node = RosAudioBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


def start_ros_audio_bridge(args: argparse.Namespace, logger: Any) -> subprocess.Popen[str] | None:
    if args.no_ros_audio_bridge:
        return None
    bridge_url = f"http://127.0.0.1:{int(args.port)}/asr"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--ros-audio-bridge-worker",
        "--bridge-url",
        bridge_url,
        "--audio-topic",
        str(args.audio_topic),
        "--filtered-audio-topic",
        str(args.filtered_audio_topic),
        "--port",
        str(int(args.port)),
    ]
    if args.token:
        command.extend(["--token", str(args.token)])
    logger.info("starting ROS audio bridge subprocess")
    return subprocess.Popen(command, text=True)


def stop_process(proc: subprocess.Popen[str] | None, logger: Any) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        logger.warning("ROS audio bridge did not stop; killing it.")
        proc.kill()


def main() -> int:
    args = parse_args()
    if args.ros_audio_bridge_worker:
        return run_ros_audio_bridge(args)
    logger = PrintLogger()
    dex3 = Dex3HoldingState(args, logger)
    bot = Dex3ObjectDetectionBot(args, dex3, logger)
    server = serve_http(bot)
    bridge_proc = start_ros_audio_bridge(args, logger)
    try:
        if args.stdin:
            stdin_loop(bot)
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop_process(bridge_proc, logger)
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
