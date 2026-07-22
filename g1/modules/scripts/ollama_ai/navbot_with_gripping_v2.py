#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import importlib.util
import json
import math
import os
import re
import string
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

WORKER_FLAGS = {"--slam-worker", "--robot-worker"}

if not any(flag in sys.argv for flag in WORKER_FLAGS):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
else:
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment,misc]
    String = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
G1_DIR = SCRIPT_DIR.parents[2] if SCRIPT_DIR.name == "ollama_ai" else SCRIPT_DIR.parent
MODULES_DIR = G1_DIR / "modules"
SCRIPTS_DIR = MODULES_DIR / "scripts"
WBC_DIR = G1_DIR / "WBC"
for path in (MODULES_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dds_env import ensure_channel_factory_initialized  # noqa: E402
from sdk_slam import SlamInfoSubscriber, SlamOperateClient  # noqa: E402


FILLERS = {
    "ah", "and", "eh", "er", "hmm", "hm", "i did not", "mm", "that's my",
    "thats my", "uh", "um",
}
SPOKEN_STATUS_ECHOES = (
    "navigation bot ready",
    "navigation and gripping bot ready",
    "navigation and gripping bot version two ready",
    "navigation not ready",
    "what should i call this point",
    "point saved",
    "starting mapping",
    "stopping mapping",
    "relocating",
    "i did not understand that navigation command",
    "the gripping system is not available right now",
    "i cannot read my hand sensors right now",
)
STOP_WORDS = {
    "a", "an", "and", "at", "called", "go", "i", "me", "my", "named", "navigate",
    "please", "point", "robot", "take", "the", "to",
}
NUMBER_WORDS = {
    "oh": "0",
    "zero": "0",
    "one": "1",
    "won": "1",
    "two": "2",
    "too": "2",
    "to": "2",
    "three": "3",
    "four": "4",
    "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "ate": "8",
    "nine": "9",
}
NAV_REACHED_DISTANCE_M = 0.35
NAV_TARGET_TIMEOUT_S = 120.0
NAV_POLL_INTERVAL_S = 0.5
LOCO_DURATION_S = 2.0
LOCO_LINEAR_SPEED_MPS = 1.2
LOCO_YAW_SPEED_RPS = 0.6
LOCO_RAMP_UP_S = 0.25
LOCO_RAMP_DOWN_S = 0.35
LOCO_RAMP_STEPS = 5
GRIPPING_SCRIPT = G1_DIR / "hand_pose_navigation" / "recognition_app.py"
GRIPPING_READY_TIMEOUT_S = 20.0
ROBOT_WORKER_READY_TIMEOUT_S = 8.0
WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
DEFAULT_SYSTEM_PROMPT = (
    "You are the voice of a Unitree G1 humanoid robot. Answer naturally and "
    "concisely. Do not mention hidden reasoning, tools, or model internals."
)
KNOWLEDGE_SYSTEM_PROMPT = (
    "Use the structured knowledge context when relevant. For questions about "
    "that knowledge, answer only from context. If context does not contain the "
    "answer, say you do not know yet. Keep it spoken and concise."
)
DEFAULT_KNOWLEDGE_FILE = SCRIPT_DIR / "robot_modules_knowledge.sample.json"
GESTURE_ACTIONS = {
    "wave": "face_wave",
    "face_wave": "face_wave",
    "hello_wave": "face_wave",
    "high_wave": "high_wave",
    "clap": "clap",
    "shake_hand": "shake_hand",
    "handshake": "shake_hand",
    "high_five": "high_five",
    "hug": "hug",
    "heart": "heart",
    "right_heart": "right_heart",
    "hands_up": "hands_up",
    "x_ray": "x_ray",
    "right_hand_up": "right_hand_up",
    "reject": "reject",
    "left_kiss": "left_kiss",
    "right_kiss": "right_kiss",
    "two_hand_kiss": "two_hand_kiss",
}
THINK_SEQUENCE = ["think"]
EXPLAIN_SEQUENCE = [
    "Explain_base",
    "Explain_left_0",
    "Explain_left_1",
    "Explain_base",
    "Explain_right_0",
    "Explain_right_1",
    "Explain_base",
    "Explain_both_0",
    "Explain_both_1",
]
DEFAULT_POSE_FILE = WBC_DIR / "saved_ik_pose_cli_v3_poses.json"


@dataclass
class PoseTarget:
    x: float
    y: float
    yaw: float = 0.0
    z: float = 0.0

    def quaternion(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, math.sin(self.yaw * 0.5), math.cos(self.yaw * 0.5))

    def xy_distance_to(self, other: "PoseTarget") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw}


@dataclass(frozen=True)
class KnowledgeEntry:
    title: str
    text: str
    source: str
    path: str
    tokens: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Voice-command SLAM navigation bot for Unitree G1 named points, "
        "plus smooth (ramped) locomotion steps/turns and voice control of the "
        "recognition_app.py grasp plugin, including its detection/gizmo/trajectory "
        "visualization (v2)."
    )
    parser.add_argument(
        "knowledge_file",
        nargs="*",
        default=[str(DEFAULT_KNOWLEDGE_FILE)],
        help="Optional structured JSON knowledge file(s). Defaults to robot_modules_knowledge.sample.json.",
    )
    parser.add_argument("--iface", default=os.environ.get("G1_IFACE", "eth0"))
    parser.add_argument("--domain-id", type=int, default=int(os.environ.get("G1_DOMAIN_ID", "0")))
    parser.add_argument("--map-path", default="/home/unitree/test.pcd")
    parser.add_argument("--points-file", default=str(SCRIPT_DIR / "nav_points.json"))
    parser.add_argument("--slam-type", default="indoor")
    parser.add_argument("--audio-topic", default="/audio_msg")
    parser.add_argument("--filtered-audio-topic", default="/audio_msg/filter")
    parser.add_argument("--command-topic", default="/model_api/navbot_gripping_v2_command")
    parser.add_argument("--response-topic", default="/model_api/navbot_gripping_v2_response")
    parser.add_argument("--external-asr-server", action="store_true")
    parser.add_argument("--external-asr-host", default="0.0.0.0")
    parser.add_argument("--external-asr-port", type=int, default=8098)
    parser.add_argument("--external-asr-public-host", default="192.168.2.41",
                        help="Robot hostname/IP that browser clients should use for the external ASR endpoint.")
    parser.add_argument("--external-asr-token", default="")
    parser.add_argument("--external-asr-only", "--no-ros-audio", dest="external_asr_only", action="store_true")
    parser.add_argument("--robot-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--enable-motion", action="store_true",
                        help="Actually execute step/turn/gesture/damp robot-body commands instead of only logging them.")
    parser.add_argument("--loco-duration-s", type=float, default=LOCO_DURATION_S)
    parser.add_argument("--loco-linear-speed", type=float, default=LOCO_LINEAR_SPEED_MPS)
    parser.add_argument("--loco-yaw-speed", type=float, default=LOCO_YAW_SPEED_RPS)
    parser.add_argument("--loco-ramp-up-s", type=float, default=LOCO_RAMP_UP_S,
                        help="Time to smoothly ramp locomotion velocity up from zero, to avoid a jerky start.")
    parser.add_argument("--loco-ramp-down-s", type=float, default=LOCO_RAMP_DOWN_S,
                        help="Time to smoothly ramp locomotion velocity down to zero, to avoid a jerky stop.")
    parser.add_argument("--no-gestures", dest="enable_gestures", action="store_false", default=True,
                        help="Disable high-level SDK arm gestures such as wave, clap, handshake, and high five.")
    parser.add_argument("--no-pose-gestures", dest="enable_pose_gestures", action="store_false", default=True,
                        help="Disable saved-pose gestures such as thinking, explain, and thanks.")
    parser.add_argument("--pose-file", default=str(DEFAULT_POSE_FILE))
    parser.add_argument("--motion-speed", type=float, default=0.32)
    parser.add_argument("--motion-kp", type=float, default=30.0)
    parser.add_argument("--motion-kd", type=float, default=1.5)
    parser.add_argument("--thinking-speed", type=float, default=0.23)
    parser.add_argument("--explain-speed", type=float, default=0.36)
    parser.add_argument("--sequence-gap", type=float, default=0.25)
    parser.add_argument("--pose-timeout-s", type=float, default=11.0)
    parser.add_argument("--post-sequence-hold-s", type=float, default=4.0)
    parser.add_argument("--thanks-hold-s", type=float, default=7.0)
    parser.add_argument("--release-after-sequence", action="store_true")
    parser.add_argument("--no-gripping", dest="enable_gripping", action="store_false", default=True,
                        help="Do not launch the recognition dashboard/grasp plugin.")
    parser.add_argument("--gripping-launch-mode", choices=("in-process", "subprocess"), default="subprocess",
                        help="Run the recognition Dash app inside this process, or as a legacy child process.")
    parser.add_argument("--gripping-mock", action="store_true",
                        help="Pass --mock to the recognition dashboard/grasp plugin (no camera/robot needed).")
    parser.add_argument("--gripping-zmq-rgbd", dest="gripping_sdk_rgbd", action="store_false", default=True,
                        help="Use the recognition app's standalone ZMQ RGB-D receiver instead of sdk_client.Robot.get_rgbd().")
    parser.add_argument("--gripping-host", default="192.168.2.41")
    parser.add_argument("--gripping-port", type=int, default=8060)
    parser.add_argument("--gripping-token", default="",
                        help="Bearer token required by the recognition_app.py /api/* routes.")
    parser.add_argument("--dashboard-host", default="0.0.0.0",
                        help="Host for navbot_with_gripping_v2's own control dashboard (separate port from "
                        "--gripping-host/--gripping-port, which belong to recognition_app.py's own Dash UI).")
    parser.add_argument("--dashboard-port", type=int, default=8099)
    parser.add_argument("--rgbd-host", default="192.168.2.41",
                        help="Forwarded to the recognition_app.py grasp plugin as --rgbd-host.")
    parser.add_argument("--rgbd-port", type=int, default=5555,
                        help="Forwarded to the recognition_app.py grasp plugin as --rgbd-port.")
    parser.add_argument("--rgbd-topic", default="",
                        help="Forwarded to the recognition_app.py grasp plugin as --rgbd-topic.")
    parser.add_argument("--vision-model", default="yolov8s-world.pt",
                        help="Forwarded to the recognition_app.py grasp plugin as --vision-model.")
    parser.add_argument("--no-tactile-status", action="store_true",
                        help="Disable Dex3 tactile status answers.")
    parser.add_argument("--tactile-threshold", type=float, default=100000.0,
                        help="Raw tactile pressure threshold for deciding that a hand is holding something.")
    parser.add_argument("--tactile-timeout-s", type=float, default=2.0,
                        help="Seconds to wait for initial Dex3 tactile state at startup.")
    parser.add_argument("--tactile-max-age-s", type=float, default=2.0,
                        help="Maximum age of tactile data used for holding status answers.")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--post-speak-ignore-s", type=float, default=1.5)
    parser.add_argument("--auto-relocate", dest="auto_relocate", action="store_true", default=True)
    parser.add_argument("--no-auto-relocate", dest="auto_relocate", action="store_false")
    parser.add_argument("--wait-for-arrival", dest="wait_for_arrival", action="store_true", default=False)
    parser.add_argument("--no-wait-for-arrival", dest="wait_for_arrival", action="store_false")
    parser.add_argument("--arrival-timeout-s", type=float, default=NAV_TARGET_TIMEOUT_S)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--num-predict", type=int, default=160)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--keep-alive", default="15m")
    parser.add_argument("--knowledge-top-k", type=int, default=4)
    parser.add_argument("--knowledge-min-score", type=float, default=0.06)
    parser.add_argument("--knowledge-max-chars", type=int, default=2600)
    parser.add_argument("--volume", type=int, default=None)
    parser.add_argument("--tts-language", default=None)
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--startup-speech", default="navigation and gripping bot version two ready")
    parser.add_argument("--no-startup-speech", action="store_true")
    parser.add_argument("--audit-log", default="/tmp/navbot_with_gripping_v2.jsonl")
    parser.add_argument("--slam-worker", default="", help=argparse.SUPPRESS)
    parser.add_argument("--point-json", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def compact_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def normalize_text(text: str) -> str:
    return compact_text(text).lower().strip(string.punctuation + "，。！？、；：")


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (match.group(0).lower() for match in WORD_RE.finditer(text))
        if len(token) > 1 and token not in STOP_WORDS
    )


def flatten_leaves(value: Any, path: str = "$") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        out: list[tuple[str, str]] = []
        for key, child in value.items():
            out.extend(flatten_leaves(child, f"{path}.{key}"))
        return out
    if isinstance(value, list):
        out = []
        for index, child in enumerate(value):
            out.extend(flatten_leaves(child, f"{path}[{index}]"))
        return out
    if value is None:
        return [(path, "null")]
    if isinstance(value, bool):
        return [(path, "true" if value else "false")]
    if isinstance(value, (int, float, str)):
        return [(path, str(value))]
    return [(path, json.dumps(value, ensure_ascii=False, sort_keys=True))]


def title_for(value: Any, fallback: str) -> str:
    if isinstance(value, dict):
        for key in ("title", "name", "question", "id", "label", "type", "category"):
            found = value.get(key)
            if isinstance(found, (str, int, float)) and str(found).strip():
                return str(found).strip()
    return fallback


def entry_from_value(value: Any, *, source: str, path: str, fallback_title: str) -> KnowledgeEntry | None:
    leaves = flatten_leaves(value, path)
    lines = [f"{leaf_path}: {leaf_value}" for leaf_path, leaf_value in leaves if str(leaf_value).strip()]
    title = title_for(value, fallback_title)
    text = f"{title}\n" + "\n".join(lines)
    tokens = tokenize(text)
    if not lines or not tokens:
        return None
    return KnowledgeEntry(title=title, text=text, source=source, path=path, tokens=tokens)


def entries_from_json(data: Any, *, source: str) -> list[KnowledgeEntry]:
    entries: list[KnowledgeEntry] = []
    if isinstance(data, list):
        for index, item in enumerate(data):
            entry = entry_from_value(item, source=source, path=f"$[{index}]", fallback_title=f"record {index + 1}")
            if entry:
                entries.append(entry)
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                for index, item in enumerate(value):
                    entry = entry_from_value(item, source=source, path=f"$.{key}[{index}]", fallback_title=f"{key} {index + 1}")
                    if entry:
                        entries.append(entry)
            elif isinstance(value, dict):
                entry = entry_from_value(value, source=source, path=f"$.{key}", fallback_title=str(key))
                if entry:
                    entries.append(entry)
    if not entries:
        root = entry_from_value(data, source=source, path="$", fallback_title=Path(source).stem)
        if root:
            entries.append(root)
    return entries


class KnowledgeRetriever:
    def __init__(self, paths: list[Path]) -> None:
        entries: list[KnowledgeEntry] = []
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries.extend(entries_from_json(data, source=str(path)))
        self.entries = entries
        doc_count = max(1, len(entries))
        df: dict[str, int] = {}
        for entry in entries:
            for token in set(entry.tokens):
                df[token] = df.get(token, 0) + 1
        self.idf = {token: math.log((doc_count + 1) / (freq + 0.5)) + 1.0 for token, freq in df.items()}

    def search(self, query: str, *, top_k: int, min_score: float) -> list[tuple[KnowledgeEntry, float]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_set = set(query_tokens)
        scored: list[tuple[KnowledgeEntry, float]] = []
        for entry in self.entries:
            counts: dict[str, int] = {}
            for token in entry.tokens:
                counts[token] = counts.get(token, 0) + 1
            score = 0.0
            for token in query_set:
                if token in counts:
                    score += self.idf.get(token, 1.0) * (1.0 + math.log(counts[token]))
            if query.strip().lower() in entry.text.lower():
                score += 2.0
            norm = math.sqrt(max(1, len(query_set)) * max(1, len(set(entry.tokens))))
            score /= norm
            if score >= min_score:
                scored.append((entry, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, top_k)]

    def format_context(self, query: str, *, top_k: int, min_score: float, max_chars: int) -> str:
        parts: list[str] = []
        total = 0
        for index, (entry, score) in enumerate(self.search(query, top_k=top_k, min_score=min_score), start=1):
            chunk = f"[{index}] source={entry.source} path={entry.path} score={score:.2f}\n{entry.text}"
            remaining = max(300, max_chars) - total
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk[:remaining].rsplit(" ", 1)[0].strip()
            parts.append(chunk)
            total += len(chunk) + 2
        return "\n\n".join(parts)


class OllamaClient:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.base_url = str(args.ollama_url).rstrip("/")

    def chat(self, messages: list[dict[str, str]], *, temperature: float | None = None, num_predict: int | None = None) -> str:
        body: dict[str, Any] = {
            "model": str(self.args.model),
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": str(self.args.keep_alive),
            "options": {
                "temperature": float(self.args.temperature if temperature is None else temperature),
                "num_predict": int(self.args.num_predict if num_predict is None else num_predict),
                "num_ctx": int(self.args.num_ctx),
            },
        }
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(self.args.timeout)) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        return compact_text(str(result.get("message", {}).get("content", "")))


def decode_payload(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    return {"text": raw}


def clean_point_name(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^(call it|name it|save it as|save as|called|named)\s+", "", text).strip()
    text = re.sub(r"[^a-z0-9 _-]+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -_")
    tokens = [NUMBER_WORDS.get(token, token) for token in text.split()]
    text = " ".join(tokens)
    text = re.sub(r"([a-z])\s+(\d)\b", r"\1 \2", text)
    return text


def response_dict(resp: Any) -> dict[str, Any]:
    raw = resp.raw
    try:
        raw = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        pass
    ok = int(resp.code) == 0
    if isinstance(raw, dict):
        ok = ok and int(raw.get("errorCode", 0)) == 0 and bool(raw.get("succeed", True))
    return {"code": int(resp.code), "ok": bool(ok), "raw": raw}


def short_error(result: dict[str, Any], fallback: str) -> str:
    raw = result.get("raw", "")
    if isinstance(raw, dict):
        for key in ("message", "error", "msg"):
            value = raw.get(key)
            if value:
                return compact_text(str(value))[:180]
        return fallback
    text = compact_text(str(raw))
    if not text:
        return fallback
    tail = text.splitlines()[-1] if "\n" in text else text
    return compact_text(tail)[:180]


def similar_to_any(text: str, phrases: tuple[str, ...], threshold: float = 0.82) -> bool:
    low = normalize_text(text)
    return any(SequenceMatcher(None, low, phrase).ratio() >= threshold for phrase in phrases)


def is_default_zero_pose(pose: PoseTarget) -> bool:
    return (
        abs(pose.x) < 1e-5
        and abs(pose.y) < 1e-5
        and abs(pose.z) < 1e-5
        and abs(pose.yaw) < 1e-5
    )


def parse_pose(payload_raw: str | None) -> PoseTarget | None:
    if not payload_raw:
        return None
    try:
        payload = json.loads(payload_raw)
        if int(payload.get("errorCode", 0)) != 0:
            return None
        cur = payload.get("data", {}).get("currentPose", {})
        x = float(cur.get("x", 0.0))
        y = float(cur.get("y", 0.0))
        z = float(cur.get("z", 0.0))
        if {"q_x", "q_y", "q_z", "q_w"} <= set(cur):
            qx = float(cur.get("q_x", 0.0))
            qy = float(cur.get("q_y", 0.0))
            qz = float(cur.get("q_z", 0.0))
            qw = float(cur.get("q_w", 1.0))
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)
        else:
            yaw = float(cur.get("yaw", 0.0))
        pose = PoseTarget(x=x, y=y, z=z, yaw=yaw)
        if is_default_zero_pose(pose):
            return None
        return pose
    except Exception:
        return None


class Speaker:
    def __init__(self, args: argparse.Namespace, logger: Any) -> None:
        self.args = args
        self.logger = logger
        self.lock = threading.Lock()
        self.current_proc: subprocess.Popen[str] | None = None

    def say(self, text: str) -> int:
        text = compact_text(text)
        if not text:
            return 0
        self.logger.info(f"robot response text={text!r}")
        if self.args.no_speech:
            self.logger.info(f"[speech disabled] {text}")
            return 0
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "robot_say_once.py"),
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
        with self.lock:
            proc = subprocess.Popen(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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

    def say_async(self, text: str) -> threading.Thread:
        thread = threading.Thread(target=self.say, args=(text,), daemon=True)
        thread.start()
        return thread

    def stop_current(self) -> None:
        with self.lock:
            proc = self.current_proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=0.8)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception as exc:
            self.logger.warning(f"Could not stop speech process: {exc}")


class NavState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.map_path = str(args.map_path)
        self.points_file = Path(args.points_file).expanduser()
        self.last_valid_pose: PoseTarget | None = None
        self.initial_pose: PoseTarget | None = None
        self.slam_running = False
        self.relocation_ready = False
        self.points = self._load_points()
        self.lock = threading.RLock()

    def _run_worker(self, operation: str, point: PoseTarget | None = None) -> dict[str, Any]:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--slam-worker",
            operation,
            "--iface",
            str(self.args.iface),
            "--domain-id",
            str(int(self.args.domain_id)),
            "--map-path",
            self.map_path,
            "--slam-type",
            str(self.args.slam_type),
        ]
        if point is not None:
            command.extend(["--point-json", json.dumps(point.as_dict(), sort_keys=True)])
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20.0)
        output = proc.stdout or ""
        result: dict[str, Any] | None = None
        for line in reversed(output.splitlines()):
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if isinstance(parsed, dict):
                result = parsed
                break
        if result is None:
            return {"code": proc.returncode or 1, "ok": False, "raw": output.strip() or "SLAM worker produced no JSON result."}
        if proc.returncode and result.get("ok", False):
            result["ok"] = False
            result["code"] = proc.returncode
        return result

    def _load_points(self) -> dict[str, PoseTarget]:
        if not self.points_file.exists():
            return {}
        try:
            data = json.loads(self.points_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw_points = data.get("points", data) if isinstance(data, dict) else {}
        points: dict[str, PoseTarget] = {}
        if isinstance(raw_points, dict):
            for name, raw in raw_points.items():
                if isinstance(raw, dict):
                    try:
                        clean = clean_point_name(str(name))
                        if clean:
                            points[clean] = PoseTarget(
                                x=float(raw["x"]),
                                y=float(raw["y"]),
                                z=float(raw.get("z", 0.0)),
                                yaw=float(raw.get("yaw", 0.0)),
                            )
                    except Exception:
                        continue
        return points

    def _save_points(self) -> None:
        payload = {
            "map_path": self.map_path,
            "updated": time.time(),
            "points": {name: pose.as_dict() for name, pose in sorted(self.points.items())},
        }
        self.points_file.parent.mkdir(parents=True, exist_ok=True)
        self.points_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def current_pose(self) -> PoseTarget | None:
        result = self._run_worker("pose")
        raw = result.get("pose") or result.get("raw")
        pose = None
        if isinstance(raw, dict):
            try:
                pose = PoseTarget(
                    x=float(raw["x"]),
                    y=float(raw["y"]),
                    z=float(raw.get("z", 0.0)),
                    yaw=float(raw.get("yaw", 0.0)),
                )
            except Exception:
                pose = None
        if pose is not None:
            self.last_valid_pose = pose
        return pose

    def start_mapping(self) -> dict[str, Any]:
        with self.lock:
            result = self._run_worker("start_mapping")
            self.slam_running = bool(result["ok"])
            self.relocation_ready = False
            return result

    def stop_mapping(self) -> dict[str, Any]:
        with self.lock:
            result = self._run_worker("stop_mapping")
            if result["ok"]:
                self.slam_running = False
            self._remember_pose_from_result(result)
            return result

    def close_slam(self) -> dict[str, Any]:
        with self.lock:
            result = self._run_worker("close_slam")
            self.slam_running = False
            self.relocation_ready = False
            return result

    def relocate(self) -> dict[str, Any]:
        with self.lock:
            pose = self.last_valid_pose or self.initial_pose
            result = self._run_worker("relocate", pose)
            self.relocation_ready = bool(result["ok"])
            self._remember_pose_from_result(result)
            return result

    def _remember_pose_from_result(self, result: dict[str, Any]) -> None:
        raw = result.get("pose")
        if not isinstance(raw, dict):
            return
        try:
            pose = PoseTarget(
                x=float(raw["x"]),
                y=float(raw["y"]),
                z=float(raw.get("z", 0.0)),
                yaw=float(raw.get("yaw", 0.0)),
            )
        except Exception:
            return
        self.last_valid_pose = pose

    def add_current_point(self, name: str) -> dict[str, Any]:
        clean = clean_point_name(name)
        if not clean:
            return {"code": 1, "ok": False, "raw": "Point name was empty."}
        with self.lock:
            pose = self.current_pose()
            if pose is None:
                return {"code": 1, "ok": False, "raw": "No current SLAM pose available."}
            self.points[clean] = pose
            self._save_points()
            return {"code": 0, "ok": True, "raw": {"name": clean, **pose.as_dict(), "point_count": len(self.points)}}

    def clear_points(self) -> dict[str, Any]:
        with self.lock:
            count = len(self.points)
            self.points = {}
            self._save_points()
            return {"code": 0, "ok": True, "raw": {"cleared": count}}

    def find_point(self, requested_name: str) -> tuple[str | None, PoseTarget | None, float]:
        wanted = clean_point_name(requested_name)
        if not wanted:
            return None, None, 0.0
        with self.lock:
            if wanted in self.points:
                return wanted, self.points[wanted], 1.0
            best_name = None
            best_score = 0.0
            wanted_compact = wanted.replace(" ", "")
            for name in self.points:
                name_compact = name.replace(" ", "")
                score = max(
                    SequenceMatcher(None, wanted, name).ratio(),
                    SequenceMatcher(None, wanted_compact, name_compact).ratio(),
                )
                if wanted_compact == name_compact:
                    score = 1.0
                wanted_tokens = set(wanted.split()) - STOP_WORDS
                name_tokens = set(name.split()) - STOP_WORDS
                if wanted_tokens and wanted_tokens <= name_tokens:
                    score = max(score, 0.92)
                if score > best_score:
                    best_name = name
                    best_score = score
            if best_name is None or best_score < 0.62:
                return None, None, best_score
            return best_name, self.points[best_name], best_score

    def go_to_point(self, name: str, *, auto_relocate: bool) -> dict[str, Any]:
        point_name, target, score = self.find_point(name)
        if target is None or point_name is None:
            return {"code": 1, "ok": False, "raw": f"I do not know a point named {name}.", "match_score": score}
        if not self.relocation_ready:
            if not auto_relocate:
                return {"code": 1, "ok": False, "raw": "Relocation is not active."}
            relocation = self.relocate()
            if not relocation["ok"]:
                return {"code": 1, "ok": False, "raw": {"relocate": relocation, "message": "Could not relocate before navigation."}}
        result = self._run_worker("pose_nav", target)
        result["point"] = point_name
        result["target"] = target.as_dict()
        return result

    def pause_nav(self) -> dict[str, Any]:
        return self._run_worker("pause_nav")

    def resume_nav(self) -> dict[str, Any]:
        return self._run_worker("resume_nav")

    def wait_for_target(self, target: PoseTarget, timeout_s: float) -> tuple[bool, PoseTarget | None, float]:
        start = time.time()
        last_pose: PoseTarget | None = None
        while time.time() - start < timeout_s:
            pose = self.current_pose()
            if pose is not None:
                last_pose = pose
                if pose.xy_distance_to(target) <= NAV_REACHED_DISTANCE_M:
                    return True, pose, time.time() - start
            time.sleep(NAV_POLL_INTERVAL_S)
        return False, last_pose, time.time() - start

    def status(self) -> dict[str, Any]:
        pose = self.current_pose()
        return {
            "map_path": self.map_path,
            "points_file": str(self.points_file),
            "slam_running": self.slam_running,
            "relocation_ready": self.relocation_ready,
            "pose": None if pose is None else pose.as_dict(),
            "points": sorted(self.points),
        }


class RobotControlWorker:
    """Persistent worker holding one warm sdk_client.Robot() connection.

    v1 spawned a brand-new subprocess (and a brand-new Robot()/DDS session)
    for every single step/turn/gesture/hand-action, which is what made
    locomotion and gestures feel snappy/laggy on real hardware: each command
    paid full reconnect latency before anything moved, and move_for()'s
    built-in stop() cuts velocity to zero instantly with no deceleration.
    This worker is started once and kept alive for the node's lifetime
    (same pattern as PoseGestureClient's IK worker below), and move_for()
    ramps velocity up/down instead of snapping, so consecutive voice
    commands feel continuous rather than start/stop/start/stop.
    """

    def __init__(self, args: argparse.Namespace, logger: Any) -> None:
        self.args = args
        self.logger = logger
        self.enabled = bool(args.enable_motion)
        self.proc: subprocess.Popen[str] | None = None
        self.stderr_thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.ready = False

    def _start(self) -> bool:
        if self.proc is not None and self.proc.poll() is None:
            return True
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--robot-worker",
            "--iface", str(self.args.iface),
            "--domain-id", str(int(self.args.domain_id)),
        ]
        env = os.environ.copy()
        env.setdefault("CYCLONEDDS_HOME", "/home/unitree/cyclonedds_ws/install/cyclonedds")
        env.setdefault("CYCLONEDDS_URI", "/home/unitree/cyclonedds_ws/cyclonedds.xml")
        try:
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            self.logger.warning(f"Could not start robot control worker: {exc}")
            self.proc = None
            return False
        self.stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self.stderr_thread.start()
        self.logger.info(f"Started robot control worker pid={self.proc.pid}")
        self.ready = True
        return True

    def _drain_stderr(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            text = line.strip()
            if text:
                self.logger.info(f"[robot worker] {text}")

    def _send(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        if not self.enabled:
            self.logger.info(f"[motion disabled] would send {payload}")
            return {"ok": False, "error": "motion_disabled"}
        with self.lock:
            if not self._start():
                return {"ok": False, "error": "robot_worker_unavailable"}
            proc = self.proc
            if proc is None or proc.stdin is None or proc.stdout is None or proc.poll() is not None:
                self.logger.warning("Robot control worker is not running.")
                self.ready = False
                return {"ok": False, "error": "robot_worker_not_running"}
            try:
                proc.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
                proc.stdin.flush()
            except BrokenPipeError:
                self.logger.warning("Robot control worker pipe is closed.")
                return {"ok": False, "error": "robot_worker_pipe_closed"}
            deadline = time.time() + timeout_s
            line = ""
            try:
                while time.time() < deadline:
                    line = proc.stdout.readline()
                    if line:
                        break
            except Exception as exc:
                return {"ok": False, "error": f"robot_worker_read_failed: {exc}"}
            if not line:
                return {"ok": False, "error": "robot_worker_timeout"}
            try:
                result = json.loads(line)
                return result if isinstance(result, dict) else {"ok": False, "error": "robot_worker_bad_response"}
            except Exception:
                return {"ok": False, "error": f"robot_worker_bad_json: {line.strip()!r}"}

    def move_for(self, vx: float, vy: float, vyaw: float) -> bool:
        duration_s = max(0.05, float(self.args.loco_duration_s))
        result = self._send(
            {
                "cmd": "move_for",
                "vx": float(vx),
                "vy": float(vy),
                "vyaw": float(vyaw),
                "duration_s": duration_s,
                "ramp_up_s": max(0.0, float(self.args.loco_ramp_up_s)),
                "ramp_down_s": max(0.0, float(self.args.loco_ramp_down_s)),
            },
            timeout_s=duration_s + float(self.args.loco_ramp_up_s) + float(self.args.loco_ramp_down_s) + 6.0,
        )
        if not result.get("ok"):
            self.logger.warning(f"Locomotion command failed: {result}")
        return bool(result.get("ok"))

    def stop(self) -> bool:
        result = self._send({"cmd": "stop"}, timeout_s=4.0)
        return bool(result.get("ok"))

    def damp(self) -> dict[str, Any]:
        return self._send({"cmd": "damp"}, timeout_s=6.0)

    def hl_action(self, action: str) -> dict[str, Any]:
        action_key = GESTURE_ACTIONS.get(str(action).strip().lower())
        if not action_key:
            return {"ok": False, "error": "unsupported_gesture", "action": action}
        if not bool(self.args.enable_gestures):
            self.logger.info(f"[gestures disabled] would run high-level action: {action_key}")
            return {"ok": False, "error": "gestures_disabled", "action": action_key}
        result = self._send({"cmd": "hl_action", "action": action_key}, timeout_s=20.0)
        result.setdefault("action", action_key)
        return result

    def close(self) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                proc.stdin.flush()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


class PoseGestureClient:
    """Uses the saved IK pose worker from chatbot_with_tactile_dex3 for pose gestures."""

    def __init__(self, args: argparse.Namespace, logger: Any) -> None:
        self.args = args
        self.logger = logger
        self.enabled = bool(args.enable_pose_gestures)
        self.proc: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()

    def _start_worker(self) -> bool:
        if not self.enabled:
            return False
        if self.proc is not None and self.proc.poll() is None:
            return True
        command = [
            sys.executable,
            str(SCRIPT_DIR / "chatbot_with_tactile_dex3.py"),
            "--motion-worker",
            "--enable-motion",
            "--iface", str(self.args.iface),
            "--domain-id", str(int(self.args.domain_id)),
            "--pose-file", str(self.args.pose_file),
            "--motion-speed", str(float(self.args.motion_speed)),
            "--motion-kp", str(float(self.args.motion_kp)),
            "--motion-kd", str(float(self.args.motion_kd)),
            "--thinking-speed", str(float(self.args.thinking_speed)),
            "--explain-speed", str(float(self.args.explain_speed)),
            "--sequence-gap", str(float(self.args.sequence_gap)),
            "--pose-timeout-s", str(float(self.args.pose_timeout_s)),
            "--post-sequence-hold-s", str(float(self.args.post_sequence_hold_s)),
            "--thanks-hold-s", str(float(self.args.thanks_hold_s)),
            "--no-speech",
            "--no-startup-speech",
        ]
        if bool(getattr(self.args, "release_after_sequence", False)):
            command.append("--release-after-sequence")
        env = os.environ.copy()
        env.setdefault("CYCLONEDDS_HOME", "/home/unitree/cyclonedds_ws/install/cyclonedds")
        env.setdefault("CYCLONEDDS_URI", "/home/unitree/cyclonedds_ws/cyclonedds.xml")
        try:
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            self.logger.warning(f"Could not start pose gesture worker: {exc}")
            self.proc = None
            return False
        threading.Thread(target=self._log_worker_output, daemon=True).start()
        self.logger.info(f"Started pose gesture worker pid={self.proc.pid}")
        return True

    def _log_worker_output(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            text = line.strip()
            if text:
                self.logger.info(f"[pose] {text}")

    def _send(self, payload: dict[str, Any]) -> bool:
        with self.lock:
            if not self._start_worker():
                return False
            proc = self.proc
            if proc is None or proc.stdin is None or proc.poll() is not None:
                self.logger.warning("Pose gesture worker is not running.")
                return False
            try:
                proc.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
                proc.stdin.flush()
                return True
            except BrokenPipeError:
                self.logger.warning("Pose gesture worker pipe is closed.")
                return False

    def play_thinking(self) -> bool:
        if not self.enabled:
            self.logger.info("[pose gestures disabled] would play thinking pose")
            return False
        return self._send({"cmd": "play", "names": THINK_SEQUENCE, "speed": float(self.args.thinking_speed), "loop": False})

    def play_explain(self) -> bool:
        if not self.enabled:
            self.logger.info("[pose gestures disabled] would play explain sequence")
            return False
        return self._send({"cmd": "play", "names": EXPLAIN_SEQUENCE, "speed": float(self.args.explain_speed), "loop": False})

    def play_thanks(self) -> bool:
        if not self.enabled:
            self.logger.info("[pose gestures disabled] would play thanks pose")
            return False
        return self._send({"cmd": "thanks", "speed": float(self.args.explain_speed)})

    def close(self) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
            proc.stdin.flush()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass


class GrippingClient:
    """Runs the recognition Dash grasp UI and talks to its /api/* control routes."""

    def __init__(self, args: argparse.Namespace, logger: Any) -> None:
        self.args = args
        self.logger = logger
        self.enabled = bool(args.enable_gripping)
        client_host = "127.0.0.1" if str(args.gripping_host) == "0.0.0.0" else str(args.gripping_host)
        self.base_url = f"http://{client_host}:{int(args.gripping_port)}"
        self.token = str(args.gripping_token or "")
        self.proc: subprocess.Popen[str] | None = None
        self.output_thread: threading.Thread | None = None
        self.app_thread: threading.Thread | None = None
        self.app_module: Any = None
        self.inprocess_server: Any = None
        self.ready = False
        if self.enabled:
            self._start()

    def _start(self) -> None:
        if str(self.args.gripping_launch_mode) == "subprocess":
            self._start_subprocess()
        else:
            self._start_in_process()
        threading.Thread(target=self._wait_ready, daemon=True).start()

    def _recognition_argv(self) -> list[str]:
        command = [
            str(GRIPPING_SCRIPT),
            "--iface", str(self.args.iface),
            "--domain-id", str(int(self.args.domain_id)),
            "--host", str(self.args.gripping_host),
            "--port", str(int(self.args.gripping_port)),
        ]
        if self.token:
            command.extend(["--api-token", self.token])
        if bool(self.args.gripping_mock):
            command.append("--mock")
        else:
            if bool(getattr(self.args, "gripping_sdk_rgbd", True)):
                command.append("--sdk-rgbd")
            if str(self.args.rgbd_host or ""):
                command.extend(["--rgbd-host", str(self.args.rgbd_host)])
            if int(self.args.rgbd_port or 0):
                command.extend(["--rgbd-port", str(int(self.args.rgbd_port))])
            if str(self.args.rgbd_topic or ""):
                command.extend(["--rgbd-topic", str(self.args.rgbd_topic)])
        if str(self.args.vision_model or ""):
            command.extend(["--vision-model", str(self.args.vision_model)])
        public_asr_host = str(self.args.external_asr_public_host or "")
        if not public_asr_host or public_asr_host in {"0.0.0.0", "::"}:
            public_asr_host = str(self.args.gripping_host)
        if public_asr_host in {"0.0.0.0", "::"}:
            public_asr_host = "127.0.0.1"
        command.extend([
            "--voice-asr-url",
            f"http://{public_asr_host}:{int(self.args.external_asr_port)}/asr",
        ])
        if str(self.args.external_asr_token or ""):
            command.extend(["--voice-asr-token", str(self.args.external_asr_token)])
        return command

    def _start_in_process(self) -> None:
        self.app_thread = threading.Thread(target=self._run_in_process_app, daemon=True)
        self.app_thread.start()
        self.logger.info(
            f"Gripping dashboard starting in-process on {self.base_url} "
            f"script={GRIPPING_SCRIPT}"
        )

    def _run_in_process_app(self) -> None:
        previous_argv = list(sys.argv)
        module_name = "_navbot_embedded_recognition_app"
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(GRIPPING_SCRIPT))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load {GRIPPING_SCRIPT}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            sys.argv = self._recognition_argv()
            try:
                spec.loader.exec_module(module)
            finally:
                sys.argv = previous_argv
            self.app_module = module
            from werkzeug.serving import make_server

            server = make_server(
                host=str(self.args.gripping_host),
                port=int(self.args.gripping_port),
                app=module.app.server,
                threaded=True,
            )
            self.inprocess_server = server
            server.serve_forever()
        except Exception as exc:
            self.logger.warning(f"In-process gripping dashboard failed: {exc}")

    def _start_subprocess(self) -> None:
        command = [sys.executable, *self._recognition_argv()]
        try:
            self.proc = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except Exception as exc:
            self.logger.warning(f"Could not start gripping plugin: {exc}")
            return
        self.logger.info(f"Gripping plugin starting: {' '.join(command)}")
        self.output_thread = threading.Thread(target=self._drain_output, daemon=True)
        self.output_thread.start()

    def _drain_output(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        noisy_dash_paths = (
            "/_dash-update-component",
            "/_dash-layout",
            "/_dash-dependencies",
        )
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    if any(noisy in line for noisy in noisy_dash_paths):
                        continue
                    self.logger.info(f"[gripping] {line}")
        except Exception as exc:
            self.logger.warning(f"Could not read gripping plugin output: {exc}")

    def _wait_ready(self) -> None:
        deadline = time.time() + GRIPPING_READY_TIMEOUT_S
        while time.time() < deadline:
            if self.status().get("ok"):
                self.ready = True
                self.logger.info("Gripping plugin ready.")
                return
            time.sleep(0.5)
        self.logger.warning("Gripping plugin did not become ready in time.")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "gripping_disabled"}
        data = json.dumps(body).encode("utf-8") if body is not None else None
        url = f"{self.base_url}{path}"
        self.logger.info(f"[gripping api] {method} {url} body={body if body is not None else {}}")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=5.0) as response:
                raw = response.read().decode("utf-8")
                result = json.loads(raw)
                self.logger.info(f"[gripping api] {method} {path} -> http={response.status} result={result}")
                return result
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read().decode("utf-8")
                result = json.loads(raw)
                self.logger.warning(f"[gripping api] {method} {path} -> http={exc.code} result={result}")
                return result
            except Exception:
                result = {"ok": False, "error": f"http_{exc.code}"}
                self.logger.warning(f"[gripping api] {method} {path} -> {result}")
                return result
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
            self.logger.warning(f"[gripping api] {method} {path} failed: {exc!r}")
            return result

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/api/status")

    def get_objects(self) -> dict[str, Any]:
        return self._request("GET", "/api/objects")

    def select_arm(self, side: str) -> dict[str, Any]:
        return self._request("POST", f"/api/select_arm/{side}")

    def extend_arm(self) -> dict[str, Any]:
        return self._request("POST", "/api/extend_arm")

    def release_arms(self) -> dict[str, Any]:
        return self._request("POST", "/api/release_arms")

    def unrelease_arms(self) -> dict[str, Any]:
        return self._request("POST", "/api/unrelease_arms")

    def set_prompt(self, text: str) -> dict[str, Any]:
        return self._request("POST", "/api/set_prompt", {"text": text})

    def grab(self, object_name: str) -> dict[str, Any]:
        return self._request("POST", "/api/grab", {"object": object_name})

    def hand_action(self, action: str, side: str) -> dict[str, Any]:
        return self._request("POST", f"/api/hand/{side}/{action}")

    def stable_hold(self) -> dict[str, Any]:
        return self._request("POST", "/api/stable_hold")

    def damp(self) -> dict[str, Any]:
        return self._request("POST", "/api/damp")

    def stop_grabbing(self) -> dict[str, Any]:
        return self._request("POST", "/api/stop_grabbing")

    def stop(self) -> None:
        if self.inprocess_server is not None:
            try:
                self.inprocess_server.shutdown()
            except Exception as exc:
                self.logger.warning(f"Could not stop in-process gripping dashboard: {exc}")
            finally:
                self.inprocess_server = None
        if self.app_thread is not None and self.app_thread.is_alive():
            self.app_thread.join(timeout=1.0)
        proc = self.proc
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception as exc:
            self.logger.warning(f"Could not stop gripping plugin: {exc}")



class TactileHoldingMonitor:
    def __init__(self, args: argparse.Namespace, logger: Any) -> None:
        self.args = args
        self.logger = logger
        self.enabled = not bool(args.no_tactile_status)
        self.ready = False
        self.error = ""
        self.subscribers: list[Any] = []
        if not self.enabled:
            return
        try:
            from test_dex3_tactile import (
                HAND_STATE_TOPIC_BY_SIDE,
                LatestTactileState,
                collect_latest_snapshots,
                wait_for_initial_snapshots,
            )

            self._collect_latest_snapshots = collect_latest_snapshots
            ensure_channel_factory_initialized(int(args.domain_id), str(args.iface))
            self.subscribers = [
                LatestTactileState(hand, HAND_STATE_TOPIC_BY_SIDE[hand], 20)
                for hand in ("left", "right")
            ]
            wait_for_initial_snapshots(self.subscribers, float(args.tactile_timeout_s))
            self.ready = True
            logger.info(
                "Dex3 tactile status ready: "
                + ", ".join(getattr(sub, "topic", "?") for sub in self.subscribers)
            )
        except Exception as exc:
            self.enabled = False
            self.error = str(exc)
            logger.warning(f"Dex3 tactile status unavailable: {exc}")

    def answer(self) -> tuple[str, dict[str, Any]]:
        if not self.enabled or not self.ready:
            answer = "I cannot read my hand sensors right now."
            return answer, {"ok": False, "answer": answer, "error": self.error or "tactile status disabled"}
        snapshots = self._collect_latest_snapshots(self.subscribers)
        threshold = float(self.args.tactile_threshold)
        max_age_s = float(self.args.tactile_max_age_s)
        holding: list[str] = []
        details: dict[str, Any] = {}
        fresh_hands = 0
        for hand in ("left", "right"):
            snapshot = snapshots.get(hand)
            if snapshot is None:
                details[hand] = {"ok": False, "reason": "missing"}
                continue
            age_s = max(0.0, time.time() - float(snapshot.timestamp))
            max_value = snapshot.max_value
            fresh = age_s <= max_age_s
            if fresh:
                fresh_hands += 1
            is_holding = fresh and max_value is not None and float(max_value) >= threshold
            if is_holding:
                holding.append(hand)
            details[hand] = {
                "ok": fresh,
                "age_s": round(age_s, 3),
                "max": max_value,
                "threshold": threshold,
                "holding": is_holding,
                "valid_count": snapshot.valid_count,
                "active_count": snapshot.active_count,
            }
        if fresh_hands == 0:
            answer = "I cannot read my hand sensors right now."
            return answer, {"ok": False, "answer": answer, "holding": holding, "hands": details}
        if holding == ["left", "right"]:
            answer = "Yes, I am holding something in both hands."
        elif holding == ["left"]:
            answer = "Yes, I am holding something in my left hand."
        elif holding == ["right"]:
            answer = "Yes, I am holding something in my right hand."
        else:
            answer = "No."
        return answer, {"ok": True, "answer": answer, "holding": holding, "hands": details}


class NavBotNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("g1_navbot_with_gripping_v2")
        self.args = args
        self.dds_status = self._init_dds_once(args)
        self.nav = NavState(args)
        self.robot_ctrl = RobotControlWorker(args, self.get_logger())
        self.pose_gestures = PoseGestureClient(args, self.get_logger())
        self.gripping = GrippingClient(args, self.get_logger())
        self.tactile = TactileHoldingMonitor(args, self.get_logger())
        knowledge_paths = [Path(item).expanduser() for item in args.knowledge_file]
        missing = [str(path) for path in knowledge_paths if not path.exists()]
        if missing:
            self.get_logger().warning("Knowledge file(s) not found: " + ", ".join(missing))
        existing_knowledge_paths = [path for path in knowledge_paths if path.exists()]
        self.retriever = KnowledgeRetriever(existing_knowledge_paths) if existing_knowledge_paths else None
        self.ollama = OllamaClient(args) if self.retriever is not None else None
        self.speaker = Speaker(args, self.get_logger())
        self.response_pub = self.create_publisher(String, args.response_topic, 10)
        self.audit_path = Path(args.audit_log).expanduser()
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_lock = threading.Lock()
        self.pending_add_point = False
        self._planned_announce_active = False
        self.last_index: int | None = None
        self.last_text = ""
        self.last_reply = ""
        self.last_reply_ts = 0.0
        self.external_asr_httpd: http.server.ThreadingHTTPServer | None = None
        self.external_asr_thread: threading.Thread | None = None
        self.dashboard_httpd: http.server.ThreadingHTTPServer | None = None
        self.dashboard_thread: threading.Thread | None = None
        if not bool(args.external_asr_only):
            self.create_subscription(String, args.audio_topic, self.on_audio, 10)
            if str(args.filtered_audio_topic) and str(args.filtered_audio_topic) != str(args.audio_topic):
                self.create_subscription(String, args.filtered_audio_topic, self.on_audio, 10)
        self.create_subscription(String, args.command_topic, self.on_command, 10)
        if bool(args.external_asr_server) or bool(args.external_asr_only):
            self._start_external_asr_server()
        self._start_dashboard_server()
        self.get_logger().info(
            f"navbot_with_gripping_v2 ready audio={'external-only' if args.external_asr_only else args.audio_topic} "
            f"map={args.map_path} points={args.points_file} "
            f"motion={'on' if args.enable_motion else 'off'} pose_gestures={'on' if args.enable_pose_gestures else 'off'} "
            f"gestures={'on' if args.enable_gestures else 'off'} gripping={'on' if args.enable_gripping else 'off'} "
            f"knowledge_entries={len(self.retriever.entries) if self.retriever is not None else 0} "
            f"dds={self.dds_status}"
        )
        if not args.no_startup_speech and compact_text(args.startup_speech):
            self._say(args.startup_speech)

    def _init_dds_once(self, args: argparse.Namespace) -> str:
        try:
            ensure_channel_factory_initialized(int(args.domain_id), str(args.iface))
            return f"ready iface={args.iface} domain_id={int(args.domain_id)}"
        except Exception as exc:
            message = f"init failed: {exc}"
            self.get_logger().warning(f"Unitree DDS {message}")
            return message

    def on_audio(self, msg: String) -> None:
        payload = decode_payload(str(msg.data))
        text = compact_text(str(payload.get("text", payload.get("raw", ""))))
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        index = self._payload_index(payload)
        now = time.time()
        self.get_logger().info(f"audio heard text={text!r} confidence={confidence:.3f} index={index}")
        reject_reason = self._filter_reason(
            text,
            confidence,
            index,
            now,
            allow_name_fragment=self.pending_add_point,
        )
        if reject_reason:
            self.get_logger().info(f"audio ignored: reason={reject_reason} text={text!r}")
            return
        self.last_index = index
        self.last_text = text
        threading.Thread(target=self._handle_text, args=(text, "audio"), daemon=True).start()

    def on_command(self, msg: String) -> None:
        payload = decode_payload(str(msg.data))
        text = compact_text(str(payload.get("text", payload.get("prompt", ""))))
        if text:
            self.get_logger().info(f"command accepted text={text!r}")
            threading.Thread(target=self._handle_text, args=(text, "command"), daemon=True).start()

    def submit_external_asr(self, text: str) -> bool:
        text = compact_text(text)
        if not text:
            return False
        reject_reason = self._filter_reason(text, 1.0, None, time.time(), allow_name_fragment=self.pending_add_point)
        if reject_reason in {"filler", "no_alphanumeric_text"}:
            self.get_logger().info(f"external ASR ignored: reason={reject_reason} text={text!r}")
            return False
        threading.Thread(target=self._handle_text, args=(text, "external_asr"), daemon=True).start()
        return True

    def _handle_text(self, text: str, source: str) -> None:
        started = time.time()
        with self.busy_lock:
            try:
                route = self._plan_intent(text)
                announce = compact_text(str(route.get("announce", "")))
                self.get_logger().info(
                    f"{source} planned: text={text!r} intent={route.get('intent')!r} "
                    f"route={json.dumps(route, sort_keys=True, default=str)}"
                )
                if announce:
                    self._say(announce)
                motion_name = str(route.get("motion", "")).strip().lower()
                planned_intent = str(route.get("intent", "")).strip().lower()
                if motion_name == "thinking" and planned_intent != "gesture":
                    self.pose_gestures.play_thinking()
                elif motion_name == "explain" and planned_intent != "gesture":
                    self.pose_gestures.play_explain()
                elif motion_name == "thanks" and planned_intent != "gesture":
                    self.pose_gestures.play_thanks()
                self._planned_announce_active = True
                try:
                    result = self._dispatch(text)
                finally:
                    self._planned_announce_active = False
                answer = str(result.pop("answer", "Done."))
                if not result.get("ok", True):
                    self.get_logger().warning(
                        f"nav command failed intent={result.get('intent')} code={result.get('code')} raw={result.get('raw')!r}"
                    )
                if answer and answer != announce:
                    self._say(answer)
                result.update({
                    "source": source,
                    "text": text,
                    "answer": answer or announce,
                    "planned_intent": route.get("intent"),
                    "announce": announce,
                    "elapsed_s": round(time.time() - started, 3),
                })
                self._publish(result)
            except Exception as exc:
                self._planned_announce_active = False
                self.get_logger().error(f"nav command failed: {exc}")
                answer = "Navigation command failed."
                self._say(answer)
                self._publish({"ok": False, "intent": "error", "source": source, "text": text, "answer": answer, "error": str(exc)})

    def _plan_intent(self, text: str) -> dict[str, Any]:
        low = normalize_text(text)
        if self.pending_add_point:
            return {"intent": "name_point", "announce": "Saving this point.", "needs_knowledge": False}
        inline_name = self._extract_add_point_name(low)
        if inline_name is not None:
            return {"intent": "add_current_point", "announce": "Saving this point.", "point": inline_name, "needs_knowledge": False}
        if self._wants_add_current_point(low):
            return {"intent": "ask_point_name", "announce": "What should I call this point?", "needs_knowledge": False}
        if self._wants_start_mapping(low):
            return {"intent": "start_mapping", "announce": "Starting mapping.", "needs_knowledge": False}
        if self._wants_stop_mapping(low):
            return {"intent": "stop_mapping", "announce": "Stopping mapping.", "needs_knowledge": False}
        if self._wants_relocate(low):
            return {"intent": "relocate", "announce": "Relocating.", "needs_knowledge": False}
        if self._wants_resume(low):
            return {"intent": "resume_navigation", "announce": "Resuming navigation.", "needs_knowledge": False}
        if self._wants_pause_or_stop(low):
            return {"intent": "pause_navigation", "announce": "Stopping navigation.", "needs_knowledge": False}
        if self._wants_close_slam(low):
            return {"intent": "close_slam", "announce": "Stopping SLAM.", "needs_knowledge": False}
        point_name = self._extract_go_to_name(low)
        if point_name:
            return {"intent": "go_to_point", "announce": f"Going to {point_name}.", "point": point_name, "needs_knowledge": False}
        loco = self._extract_locomotion(low, self.args)
        if loco is not None:
            vx, vy, vyaw, announce = loco
            return {"intent": "locomotion", "announce": announce, "vx": vx, "vy": vy, "vyaw": vyaw, "needs_knowledge": False}
        if self._is_holding_status_question(low):
            return {"intent": "holding_status", "announce": "Checking my hand sensors.", "needs_knowledge": False}
        if self._wants_objects_query(low):
            return {"intent": "list_objects", "announce": "Checking what I can see.", "needs_knowledge": False}
        if self._wants_stop_grabbing(low):
            return {"intent": "stop_grabbing", "announce": "Stopping grab.", "needs_knowledge": False}
        if self._wants_release_arms(low):
            return {"intent": "release_arms", "announce": "Releasing my arms.", "needs_knowledge": False}
        if self._wants_stable_hold(low):
            return {"intent": "stable_hold", "announce": "Moving to a stable hold.", "needs_knowledge": False}
        if self._wants_damp(low):
            return {"intent": "damp", "announce": "Damping.", "needs_knowledge": False}
        if self._wants_extend_arm(low):
            return {"intent": "extend_arm", "announce": "Extending my arm.", "needs_knowledge": False}
        arm_side = self._extract_select_arm_side(low)
        if arm_side is not None:
            return {"intent": "select_arm", "announce": f"Switching to the {arm_side} arm.", "arm": arm_side, "needs_knowledge": False}
        hand_action = self._extract_hand_action(low)
        if hand_action is not None:
            action, side = hand_action
            verb = "Opening" if action == "open" else "Closing"
            noun = "hands" if side == "both" else f"{side} hand"
            return {"intent": "hand_action", "announce": f"{verb} my {noun}.", "action": action, "side": side, "needs_knowledge": False}
        look_for = self._extract_look_for(low)
        if look_for is not None:
            return {"intent": "set_prompt", "announce": f"Looking for {look_for}.", "object": look_for, "needs_knowledge": False}
        grab_object = self._extract_grab_object(low)
        if grab_object is not None:
            return {"intent": "grab", "announce": f"Grabbing the {grab_object}.", "object": grab_object, "needs_knowledge": False}
        if "list" in low and "point" in low:
            return {"intent": "list_points", "announce": "Listing saved points.", "needs_knowledge": False}
        if self._wants_clear_points(low):
            return {"intent": "clear_points", "announce": "Clearing saved points.", "needs_knowledge": False}
        if "status" in low:
            return {"intent": "status", "announce": "Checking navigation status.", "needs_knowledge": False}
        if self._is_thanks_text(low):
            return {"intent": "thanks", "announce": self._thanks_answer(low), "needs_knowledge": False}
        if self._is_chat_text(low):
            return {"intent": "chat", "announce": "", "needs_knowledge": False}
        if self._is_knowledge_question(low):
            return {"intent": "rag_question", "announce": "Let me check my local knowledge.", "needs_knowledge": True, "motion": "thinking"}
        gesture = self._extract_gesture(low)
        if gesture is not None:
            action, announce = gesture
            motion = "thinking" if action == "thinking" else "explain" if action == "explain" else action
            return {"intent": "gesture", "announce": announce, "motion": motion, "needs_knowledge": False}
        return {
            "intent": "unknown",
            "announce": "I did not understand that navigation command.",
            "needs_knowledge": False,
        }

    def _say_action_announce(self, text: str) -> None:
        if not self._planned_announce_active:
            self._say(text)

    def _dispatch(self, text: str) -> dict[str, Any]:
        low = normalize_text(text)
        if self.pending_add_point:
            self.pending_add_point = False
            result = self.nav.add_current_point(text)
            name = result.get("raw", {}).get("name") if isinstance(result.get("raw"), dict) else clean_point_name(text)
            return {"intent": "name_point", **result, "answer": f"Point {name} saved." if result["ok"] else str(result["raw"])}

        inline_name = self._extract_add_point_name(low)
        if inline_name is not None:
            result = self.nav.add_current_point(inline_name)
            name = result.get("raw", {}).get("name") if isinstance(result.get("raw"), dict) else inline_name
            return {"intent": "add_current_point", **result, "answer": f"Point {name} saved." if result["ok"] else str(result["raw"])}

        if self._wants_add_current_point(low):
            self.pending_add_point = True
            return {"ok": True, "code": 0, "intent": "ask_point_name", "answer": "What should I call this point?"}

        if self._wants_start_mapping(low):
            result = self.nav.start_mapping()
            return {"intent": "start_mapping", **result, "answer": "Starting mapping." if result["ok"] else f"I could not start mapping: {short_error(result, 'SLAM returned an error')}."}

        if self._wants_stop_mapping(low):
            result = self.nav.stop_mapping()
            return {"intent": "stop_mapping", **result, "answer": f"Mapping stopped and saved to {self.nav.map_path}." if result["ok"] else f"I could not stop mapping: {short_error(result, 'SLAM returned an error')}."}

        if self._wants_relocate(low):
            result = self.nav.relocate()
            return {"intent": "relocate", **result, "answer": "Relocation started." if result["ok"] else f"I could not relocate: {short_error(result, 'SLAM returned an error')}."}

        if self._wants_resume(low):
            result = self.nav.resume_nav()
            return {"intent": "resume_navigation", **result, "answer": "Resuming navigation." if result["ok"] else "I could not resume navigation."}

        if self._wants_pause_or_stop(low):
            self.speaker.stop_current()
            result = self.nav.pause_nav()
            return {"intent": "pause_navigation", **result, "answer": "Stopping navigation." if result["ok"] else "I could not stop navigation."}

        if self._wants_close_slam(low):
            result = self.nav.close_slam()
            return {"intent": "close_slam", **result, "answer": "SLAM stopped." if result["ok"] else "I could not stop SLAM."}

        point_name = self._extract_go_to_name(low)
        if point_name:
            result = self.nav.go_to_point(point_name, auto_relocate=bool(self.args.auto_relocate))
            if result["ok"] and bool(self.args.wait_for_arrival):
                target = PoseTarget(**result["target"])
                reached, final_pose, elapsed = self.nav.wait_for_target(target, float(self.args.arrival_timeout_s))
                result["reached"] = reached
                result["elapsed_wait_s"] = round(elapsed, 2)
                if final_pose is not None:
                    result["final_pose"] = final_pose.as_dict()
                    result["final_distance_m"] = round(final_pose.xy_distance_to(target), 3)
                if not reached:
                    result["ok"] = False
                    result["code"] = 1
            point = str(result.get("point", point_name))
            answer = f"Going to {point}." if result["ok"] else str(result.get("raw", "I could not navigate to that point."))
            return {"intent": "go_to_point", **result, "answer": answer}

        # Locomotion/gripping checks run after every SLAM/point-nav intent above so that
        # e.g. "walk to office one" is always claimed by go_to_point first (both share the
        # word "walk") and only bare direction phrases ("walk forward") reach this point.
        loco = self._extract_locomotion(low, self.args)
        if loco is not None:
            vx, vy, vyaw, announce = loco
            self._say_action_announce(announce)
            self.robot_ctrl.move_for(vx, vy, vyaw)
            return {"ok": True, "code": 0, "intent": "locomotion", "vx": vx, "vy": vy, "vyaw": vyaw, "answer": ""}

        if self._is_holding_status_question(low):
            answer, tactile_result = self.tactile.answer()
            return {**tactile_result, "intent": "holding_status", "answer": answer}

        if self._wants_objects_query(low):
            if not self.gripping.enabled:
                return {"ok": False, "code": 1, "intent": "list_objects", "answer": "The gripping system is not available right now."}
            result = self.gripping.get_objects()
            if not result.get("ok"):
                return {"ok": False, "code": 1, "intent": "list_objects", "answer": "The gripping system is not available right now."}
            labels = [str(obj.get("label", "")) for obj in result.get("objects", []) if obj.get("label")]
            answer = "I do not see any objects right now." if not labels else "I can see " + ", ".join(labels) + "."
            return {"ok": True, "code": 0, "intent": "list_objects", "objects": labels, "answer": answer}

        if self._wants_stop_grabbing(low):
            if not self.gripping.enabled:
                return {"ok": False, "code": 1, "intent": "stop_grabbing", "answer": "The gripping system is not available right now."}
            self._say_action_announce("Stopping grab.")
            result = self.gripping.stop_grabbing()
            return {"ok": bool(result.get("ok")), "code": 0, "intent": "stop_grabbing", "answer": ""}

        if self._wants_release_arms(low):
            if not self.gripping.enabled:
                return {"ok": False, "code": 1, "intent": "release_arms", "answer": "The gripping system is not available right now."}
            self._say_action_announce("Releasing my arms.")
            result = self.gripping.release_arms()
            return {"ok": bool(result.get("ok")), "code": 0, "intent": "release_arms", "answer": ""}

        if self._wants_stable_hold(low):
            if not self.gripping.enabled:
                return {"ok": False, "code": 1, "intent": "stable_hold", "answer": "The gripping system is not available right now."}
            self._say_action_announce("Moving to a stable hold.")
            result = self.gripping.stable_hold()
            return {"ok": bool(result.get("ok")), "code": 0, "intent": "stable_hold", "answer": ""}

        if self._wants_damp(low):
            # Damp is an emergency/safety action: prefer the gripping backend
            # (recognition_app.py cancels any in-flight arm motion before
            # damping) when it is up, but always fall back to the persistent
            # robot control worker so damp still works if the grasp plugin
            # process is down or was disabled with --no-gripping.
            self._say_action_announce("Damping.")
            if self.gripping.enabled:
                result = self.gripping.damp()
                if not result.get("ok") and self.robot_ctrl.enabled:
                    result = self.robot_ctrl.damp()
            else:
                result = self.robot_ctrl.damp()
            ok = bool(result.get("ok"))
            return {
                "ok": ok,
                "code": 0 if ok else 1,
                "intent": "damp",
                "answer": "" if ok else f"I could not damp: {result.get('error', result.get('status', 'control failed'))}.",
                "raw": result,
            }

        if self._wants_extend_arm(low):
            if not self.gripping.enabled:
                return {"ok": False, "code": 1, "intent": "extend_arm", "answer": "The gripping system is not available right now."}
            self._say_action_announce("Extending my arm.")
            result = self.gripping.extend_arm()
            ok = bool(result.get("ok"))
            return {
                "ok": ok,
                "code": 0 if ok else 1,
                "intent": "extend_arm",
                "answer": "" if ok else f"I could not extend my arm: {result.get('error', result.get('status', 'control failed'))}.",
                "raw": result,
            }

        arm_side = self._extract_select_arm_side(low)
        if arm_side is not None:
            if not self.gripping.enabled:
                return {"ok": False, "code": 1, "intent": "select_arm", "answer": "The gripping system is not available right now."}
            self._say_action_announce(f"Switching to the {arm_side} arm.")
            result = self.gripping.select_arm(arm_side)
            return {"ok": bool(result.get("ok")), "code": 0, "intent": "select_arm", "arm": arm_side, "answer": ""}

        hand_action = self._extract_hand_action(low)
        if hand_action is not None:
            if not self.gripping.enabled:
                return {"ok": False, "code": 1, "intent": "hand_action", "answer": "The gripping system is not available right now."}
            action, side = hand_action
            verb = "Opening" if action == "open" else "Closing"
            noun = "hands" if side == "both" else f"{side} hand"
            self._say_action_announce(f"{verb} my {noun}.")
            if side == "both":
                result_left = self.gripping.hand_action(action, "left")
                result_right = self.gripping.hand_action(action, "right")
                ok = bool(result_left.get("ok")) and bool(result_right.get("ok"))
            else:
                result = self.gripping.hand_action(action, side)
                ok = bool(result.get("ok"))
            return {"ok": ok, "code": 0, "intent": "hand_action", "action": action, "side": side, "answer": ""}

        look_for = self._extract_look_for(low)
        if look_for is not None:
            if not self.gripping.enabled:
                return {"ok": False, "code": 1, "intent": "set_prompt", "answer": "The gripping system is not available right now."}
            self._say_action_announce(f"Looking for {look_for}.")
            result = self.gripping.set_prompt(look_for)
            return {"ok": bool(result.get("ok")), "code": 0, "intent": "set_prompt", "answer": ""}

        grab_object = self._extract_grab_object(low)
        if grab_object is not None:
            if not self.gripping.enabled:
                return {"ok": False, "code": 1, "intent": "grab", "answer": "The gripping system is not available right now."}
            self._say_action_announce(f"Grabbing the {grab_object}.")
            result = self.gripping.grab(grab_object)
            if result.get("ok"):
                return {"ok": True, "code": 0, "intent": "grab", "object": grab_object, "answer": ""}
            return {"ok": False, "code": 1, "intent": "grab", "object": grab_object, "answer": f"I do not see a {grab_object}."}

        if "list" in low and "point" in low:
            names = sorted(self.nav.points)
            answer = "I do not have any saved points." if not names else "Saved points are " + ", ".join(names) + "."
            return {"ok": True, "code": 0, "intent": "list_points", "points": names, "answer": answer}

        if self._wants_clear_points(low):
            result = self.nav.clear_points()
            count = result.get("raw", {}).get("cleared", 0) if isinstance(result.get("raw"), dict) else 0
            answer = f"Cleared {count} saved points." if result["ok"] else "I could not clear the saved points."
            return {"intent": "clear_points", **result, "answer": answer}

        if "status" in low:
            return {"ok": True, "code": 0, "intent": "status", "status": self.nav.status(), "answer": "Navigation status is available."}

        if self._is_chat_text(low):
            return {"ok": True, "code": 0, "intent": "chat", "answer": self._chat_answer(text)}

        if self._is_thanks_text(low):
            return {"ok": True, "code": 0, "intent": "thanks", "answer": self._thanks_answer(low)}

        if self._is_knowledge_question(low):
            answer, used_knowledge = self._rag_answer(text)
            if used_knowledge:
                return {"ok": True, "code": 0, "intent": "rag_question", "used_knowledge": True, "answer": answer}

        gesture = self._extract_gesture(low)
        if gesture is not None:
            action, announce = gesture
            if action == "thinking":
                ok = self.pose_gestures.play_thinking()
                return {"ok": ok, "code": 0 if ok else 1, "intent": "gesture", "motion": action, "answer": ""}
            if action == "explain":
                ok = self.pose_gestures.play_explain()
                return {"ok": ok, "code": 0 if ok else 1, "intent": "gesture", "motion": action, "answer": ""}
            result = self.robot_ctrl.hl_action(action)
            if result.get("ok"):
                return {"ok": True, "code": int(result.get("code", 0)), "intent": "gesture", "motion": action, "answer": ""}
            return {
                "ok": False,
                "code": int(result.get("code", 1) or 1),
                "intent": "gesture",
                "motion": action,
                "answer": f"I could not run that gesture: {result.get('error', 'gesture failed')}.",
                "raw": result,
            }

        return {
            "ok": False,
            "code": 1,
            "intent": "unknown",
            "answer": "I did not understand that navigation command.",
        }

    def _rag_answer(self, text: str) -> tuple[str, bool]:
        context = ""
        if self.retriever is not None:
            context = self.retriever.format_context(
                text,
                top_k=int(self.args.knowledge_top_k),
                min_score=float(self.args.knowledge_min_score),
                max_chars=int(self.args.knowledge_max_chars),
            )
        if not context:
            return "I do not know yet.", False
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "system", "content": f"{KNOWLEDGE_SYSTEM_PROMPT}\n\nStructured knowledge context:\n{context}"},
            {"role": "user", "content": text},
        ]
        try:
            if self.ollama is None:
                raise RuntimeError("Ollama client is not configured.")
            answer = self.ollama.chat(messages, temperature=float(self.args.temperature), num_predict=int(self.args.num_predict))
            return answer, True
        except Exception as exc:
            self.get_logger().warning(f"Ollama RAG answer failed; using local knowledge fallback: {exc}")
            fallback = self._local_knowledge_answer(text)
            return (fallback or "I do not know yet."), bool(fallback)

    def _local_knowledge_answer(self, text: str) -> str:
        if self.retriever is None:
            return ""
        matches = self.retriever.search(
            text,
            top_k=min(2, int(self.args.knowledge_top_k)),
            min_score=float(self.args.knowledge_min_score),
        )
        if not matches:
            return ""
        parts: list[str] = []
        for entry, _score in matches:
            cleaned_lines = []
            for line in entry.text.splitlines():
                line = compact_text(line)
                if not line or line.startswith("$.") or line.startswith("$["):
                    continue
                cleaned_lines.append(line)
                if len(" ".join(cleaned_lines)) >= 260:
                    break
            summary = " ".join(cleaned_lines) or compact_text(entry.text)
            if len(summary) > 280:
                summary = summary[:280].rsplit(" ", 1)[0].strip()
            parts.append(summary)
        answer = "From my local knowledge: " + " ".join(parts)
        if len(answer) > 650:
            answer = answer[:650].rsplit(" ", 1)[0].strip()
        return answer

    def _is_knowledge_question(self, low: str) -> bool:
        if self.retriever is None:
            return False
        if any(phrase in low for phrase in ("from your knowledge", "local knowledge", "knowledge file", "tell me about", "what is", "what are")):
            return True
        first = low.split(" ", 1)[0] if low else ""
        return first in {"what", "why", "how", "when", "where", "who", "which"} and bool(tokenize(low))

    @staticmethod
    def _extract_gesture(low: str) -> tuple[str, str] | None:
        if "thinking gesture" in low or "think gesture" in low or low in {"think", "thinking"}:
            return "thinking", "Let me think."
        if "explain gesture" in low or low in {"explain", "explain motion"}:
            return "explain", "Explaining."
        if "clap" in low or "applaud" in low or "lap your hands" in low or "sap your hands" in low:
            return "clap", "I will clap."
        if "high five" in low or "high-five" in low:
            return "high_five", "High five."
        if "shake hand" in low or "shake hands" in low or "handshake" in low:
            return "shake_hand", "Nice to meet you."
        if "hug" in low:
            return "hug", "Giving a hug gesture."
        if "right heart" in low:
            return "right_heart", "Making a right-heart gesture."
        if "heart" in low and any(word in low for word in ("make", "do", "show", "gesture", "pose")):
            return "heart", "Making a heart gesture."
        if "hands up" in low or "raise your hands" in low or "put your hands up" in low:
            return "hands_up", "Putting my hands up."
        if "x ray" in low or "x-ray" in low:
            return "x_ray", "Doing the x-ray gesture."
        if "right hand up" in low or "raise your right hand" in low:
            return "right_hand_up", "Raising my right hand."
        if "reject" in low or "refuse" in low:
            return "reject", "Rejecting."
        if "two hand kiss" in low or "two-hand kiss" in low:
            return "two_hand_kiss", "Blowing a kiss."
        if "right kiss" in low:
            return "right_kiss", "Blowing a right-hand kiss."
        if "left kiss" in low or "blow a kiss" in low:
            return "left_kiss", "Blowing a kiss."
        if "high wave" in low or "big wave" in low or "wave high" in low:
            return "high_wave", "Waving."
        if "wave" in low or "greet" in low or "say hello with your hand" in low:
            return "face_wave", "Hello."
        return None

    @staticmethod
    def _is_chat_text(low: str) -> bool:
        if low in {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}:
            return True
        return low in {"how are you", "how are you doing", "what can you do", "who are you"}

    @staticmethod
    def _is_thanks_text(low: str) -> bool:
        return any(word in low for word in ("thank", "thanks", "danke"))

    @staticmethod
    def _thanks_answer(low: str) -> str:
        if any(phrase in low for phrase in ("say thank you", "say thanks", "thank them", "thank everyone")):
            return "Thank you."
        return "You're welcome."

    @staticmethod
    def _chat_answer(text: str) -> str:
        low = normalize_text(text)
        if low in {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}:
            return "Hello. I am ready for navigation and gripping commands."
        if low == "who are you":
            return "I am the navigation and gripping voice controller for this G1 robot."
        if low == "what can you do":
            return "I can navigate to saved points, manage mapping, look for objects, and control gripping actions."
        return "I am ready."

    @staticmethod
    def _wants_clear_points(low: str) -> bool:
        if "point" not in low:
            return False
        return any(phrase in low for phrase in ("clear", "erase", "reset", "delete all", "forget all", "remove all"))

    @staticmethod
    def _wants_start_mapping(low: str) -> bool:
        phrases = ("start mapping", "begin mapping", "create map", "make a map")
        return any(phrase in low for phrase in phrases) or similar_to_any(low, phrases)

    @staticmethod
    def _wants_stop_mapping(low: str) -> bool:
        phrases = ("stop mapping", "finish mapping", "end mapping", "save map", "save the map")
        return any(phrase in low for phrase in phrases) or similar_to_any(low, phrases)

    @staticmethod
    def _wants_relocate(low: str) -> bool:
        phrases = ("relocate", "localize", "relocalize", "init pose")
        return any(word in low for word in phrases) or similar_to_any(low, phrases, threshold=0.68)

    @staticmethod
    def _wants_add_current_point(low: str) -> bool:
        if "current" not in low or "point" not in low:
            return False
        return any(word in low for word in ("add", "at", "save", "mark", "remember"))

    @staticmethod
    def _extract_add_point_name(low: str) -> str | None:
        match = re.search(r"(?:add|save|mark|remember)\s+(?:the\s+)?current\s+point\s+(?:as|called|named)\s+(.+)$", low)
        return clean_point_name(match.group(1)) if match else None

    @staticmethod
    def _wants_pause_or_stop(low: str) -> bool:
        return low in {"stop", "cancel", "halt"} or any(phrase in low for phrase in ("stop navigation", "pause navigation", "cancel navigation", "hold position"))

    @staticmethod
    def _wants_resume(low: str) -> bool:
        return any(phrase in low for phrase in ("resume navigation", "continue navigation", "keep going"))

    @staticmethod
    def _wants_close_slam(low: str) -> bool:
        return any(phrase in low for phrase in ("stop slam", "close slam", "shutdown slam", "shut down slam"))

    @staticmethod
    def _extract_go_to_name(low: str) -> str | None:
        patterns = (
            r"^(?:go|navigate|drive|walk)\s+to\s+(.+)$",
            r"^take\s+me\s+to\s+(.+)$",
            r"^go\s+to\s+point\s+(.+)$",
            r"^navigate\s+to\s+point\s+(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, low)
            if match:
                return clean_point_name(match.group(1))
        return None

    @staticmethod
    def _extract_locomotion(low: str, args: argparse.Namespace) -> tuple[float, float, float, str] | None:
        if not any(word in low for word in ("step", "walk", "move", "turn", "rotate")):
            return None
        linear = float(getattr(args, "loco_linear_speed", LOCO_LINEAR_SPEED_MPS))
        yaw = float(getattr(args, "loco_yaw_speed", LOCO_YAW_SPEED_RPS))
        if "forward" in low or "forwards" in low or "front" in low:
            return (linear, 0.0, 0.0, "Stepping forward.")
        if "backward" in low or "backwards" in low or "back" in low or "reverse" in low:
            return (-linear, 0.0, 0.0, "Stepping backward.")
        if "right" in low:
            if "turn" in low or "rotate" in low:
                return (0.0, 0.0, -yaw, "Turning right.")
            return (0.0, -linear, 0.0, "Stepping right.")
        if "left" in low:
            if "turn" in low or "rotate" in low:
                return (0.0, 0.0, yaw, "Turning left.")
            return (0.0, linear, 0.0, "Stepping left.")
        return None

    @staticmethod
    def _is_holding_status_question(low: str) -> bool:
        normalized = " ".join(str(low).strip().strip(string.punctuation + "，。！？、；：").split())
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
        has_hand_word = "hand" in normalized or "hands" in normalized
        return has_holding_word and (has_object_word or has_hand_word) and has_robot_word

    @staticmethod
    def _wants_objects_query(low: str) -> bool:
        phrases = ("what objects do you see", "what do you see", "what can you see", "what things do you see")
        return any(phrase in low for phrase in phrases)

    @staticmethod
    def _wants_stop_grabbing(low: str) -> bool:
        phrases = ("stop grabbing", "cancel grabbing", "cancel the grab", "abort grab", "abort the grab")
        return any(phrase in low for phrase in phrases)

    @staticmethod
    def _wants_release_arms(low: str) -> bool:
        phrases = ("release arm", "release arms", "release your arm", "release your arms")
        return any(phrase in low for phrase in phrases)

    @staticmethod
    def _wants_extend_arm(low: str) -> bool:
        phrases = ("extend arm", "extend your arm", "extend the arm")
        return any(phrase in low for phrase in phrases)

    @staticmethod
    def _wants_damp(low: str) -> bool:
        phrases = ("damp", "damping", "damp the robot", "damp arm", "damp arms", "damp the arms")
        return low in {"damp", "damping"} or any(phrase in low for phrase in phrases)

    @staticmethod
    def _wants_stable_hold(low: str) -> bool:
        phrases = ("stable hold", "hold stable", "stable pose", "safe hold")
        return any(phrase in low for phrase in phrases)

    @staticmethod
    def _extract_hand_action(low: str) -> tuple[str, str] | None:
        if "hand" not in low:
            return None
        if "open" in low:
            action = "open"
        elif "close" in low:
            action = "close"
        else:
            return None
        has_left = "left" in low
        has_right = "right" in low
        if has_left and has_right:
            side = "both"
        elif has_left:
            side = "left"
        elif has_right:
            side = "right"
        elif "hands" in low or "both" in low:
            side = "both"
        else:
            return None
        return action, side

    @staticmethod
    def _extract_select_arm_side(low: str) -> str | None:
        if not any(word in low for word in ("select", "switch", "use")):
            return None
        if "right arm" in low:
            return "right"
        if "left arm" in low:
            return "left"
        return None

    @staticmethod
    def _extract_look_for(low: str) -> str | None:
        match = re.search(r"^(?:look for|find|search for|detect)\s+(?:a\s+|an\s+|the\s+)?(.+)$", low)
        return compact_text(match.group(1)) if match else None

    @staticmethod
    def _extract_grab_object(low: str) -> str | None:
        match = re.search(r"^(?:grab|pick up)\s+(?:the\s+|a\s+|an\s+)?(.+)$", low)
        return compact_text(match.group(1)) if match else None

    def _say(self, text: str) -> None:
        self.last_reply = compact_text(text)
        self.last_reply_ts = time.time()
        self.speaker.say_async(self.last_reply)

    def _filter_reason(
        self,
        text: str,
        confidence: float,
        index: int | None,
        received_at: float,
        *,
        allow_name_fragment: bool = False,
    ) -> str | None:
        if not text or confidence < float(self.args.min_confidence):
            return f"empty_or_low_confidence confidence={confidence:.3f}"
        normalized = normalize_text(text)
        if normalized in FILLERS and not allow_name_fragment:
            return "filler"
        words = normalized.split()
        if not allow_name_fragment and len(words) <= 3 and not any(
            keyword in normalized
            for keyword in (
                "add", "arm", "close", "extend", "find", "go", "grab", "grip", "hand",
                "about", "hold", "how", "list", "look", "map", "mapping", "move", "navigate", "pick",
                "point", "relocate", "release", "resume", "rotate", "save", "search",
                "select", "slam", "start", "status", "step", "stop", "switch", "turn",
                "tell", "walk", "what", "when", "where", "which", "who", "why",
            )
        ):
            return "short_non_command_fragment"
        if not any(char.isalnum() for char in text):
            return "no_alphanumeric_text"
        if any(phrase in normalized for phrase in SPOKEN_STATUS_ECHOES):
            return "spoken_status_echo"
        if index is not None and index == self.last_index:
            return f"duplicate_index index={index}"
        if index is None and text == self.last_text and received_at - self.last_reply_ts < 2.0:
            return f"duplicate_text age_s={received_at - self.last_reply_ts:.2f}"
        if received_at - self.last_reply_ts < float(self.args.post_speak_ignore_s):
            return f"post_speak_ignore age_s={received_at - self.last_reply_ts:.2f}"
        if (
            self.last_reply
            and received_at - self.last_reply_ts < max(4.0, float(self.args.post_speak_ignore_s))
            and SequenceMatcher(None, text.lower(), self.last_reply.lower()).ratio() >= 0.82
        ):
            return "looks_like_tts_echo"
        return None

    def _publish(self, result: dict[str, Any]) -> None:
        result["time"] = time.time()
        self.response_pub.publish(String(data=json.dumps(result, sort_keys=True, default=str)))
        self._audit({"kind": "response", "result": result})

    def _audit(self, record: dict[str, Any]) -> None:
        record = {"time": time.time(), **record}
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def _gripping_browser_url(self) -> str:
        host = str(self.args.external_asr_public_host or "")
        if not host or host in {"0.0.0.0", "::"}:
            host = str(self.args.gripping_host)
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{int(self.args.gripping_port)}/"

    def _dashboard_html(self, title: str, note: str) -> str:
        gripping_url = self._gripping_browser_url()
        gripping_enabled = bool(self.args.enable_gripping)
        panel = (
            f'<iframe src="{gripping_url}" title="Grasp plugin (recognition_app.py)" '
            'referrerpolicy="no-referrer"></iframe>'
            if gripping_enabled
            else '<div class="status" style="padding:16px">Gripping is disabled (--no-gripping). '
            "No camera/detections/gizmo view is available.</div>"
        )
        return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light dark; }}
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #101114; color: #f4f4f5; }}
    header {{ padding: 14px 18px; border-bottom: 1px solid #30333a; display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
    main {{ display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 16px; padding: 16px; height: calc(100vh - 57px); box-sizing: border-box; }}
    h1 {{ font-size: 20px; margin: 0; }}
    .status {{ color: #b6bcc8; font-size: 13px; }}
    .grasp-panel {{ background: #050608; border: 1px solid #30333a; min-height: 360px; }}
    .grasp-panel iframe {{ display: block; width: 100%; height: 100%; border: 0; background: #fff; }}
    aside {{ display: flex; flex-direction: column; gap: 12px; overflow-y: auto; }}
    section {{ border: 1px solid #30333a; padding: 12px; background: #17191e; }}
    h2 {{ font-size: 15px; margin: 0 0 10px; }}
    button {{ font-size: 15px; padding: 9px 12px; margin: 3px; background: #2b65d9; color: white; border: 0; cursor: pointer; }}
    button.danger {{ background: #b3261e; }}
    button.secondary {{ background: #3b414d; }}
    input {{ box-sizing: border-box; width: 100%; font-size: 15px; padding: 10px; margin-bottom: 8px; background: #0f1115; color: #f4f4f5; border: 1px solid #3b414d; }}
    pre {{ background: #08090b; color: #d7dbe3; padding: 10px; min-height: 180px; max-height: 280px; overflow: auto; white-space: pre-wrap; font-size: 12px; }}
    @media (max-width: 860px) {{ main {{ grid-template-columns: 1fr; height: auto; }} .grasp-panel {{ min-height: 320px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="status" id="status">{note}</div>
  </header>
  <main>
    <div class="grasp-panel">{panel}</div>
    <aside>
      <section>
        <h2>Robot Control</h2>
        <button class="danger" onclick="post('/api/damp')">Damp</button>
        <button onclick="post('/api/extend_arm')">Extend Arm</button>
        <button class="secondary" onclick="post('/api/release_arms')">Release Arms</button>
        <button class="secondary" onclick="post('/api/unrelease_arms')">Unrelease Arms</button>
        <button class="secondary" onclick="post('/api/stable_hold')">Stable Hold</button>
      </section>
      <section>
        <h2>Voice/Text Command</h2>
        <p class="status">Locomotion, gestures, "grab the &lt;object&gt;", "look for &lt;object&gt;",
        "open/close left/right hand", holding-status questions, mapping/navigation, and
        knowledge questions all go through here — same routing as the ROS audio topic.</p>
        <input id="cmd" placeholder="Type a navigation or gripping command">
        <button onclick="command()">Send</button>
      </section>
      <section>
        <h2>Log</h2>
        <pre id="log"></pre>
      </section>
    </aside>
  </main>
  <script>
    const log = (msg) => {{
      const el = document.getElementById('log');
      el.textContent = new Date().toLocaleTimeString() + ' ' + msg + '\\n' + el.textContent;
    }};
    async function post(path, body) {{
      log('send ' + path);
      try {{
        const res = await fetch(path, {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify(body || {{}})
        }});
        log('response ' + res.status + ' ' + await res.text());
        refreshStatus();
      }} catch (err) {{
        log('error ' + err);
      }}
    }}
    function command() {{
      const input = document.getElementById('cmd');
      post('/command', {{text: input.value}});
    }}
    async function refreshStatus() {{
      try {{
        const res = await fetch('/api/status');
        document.getElementById('status').textContent = JSON.stringify(await res.json());
      }} catch (err) {{
        document.getElementById('status').textContent = String(err);
      }}
    }}
    setInterval(refreshStatus, 2000);
    refreshStatus();
  </script>
</body>
</html>
"""

    def _control_routes(self) -> dict[str, Any]:
        def damp() -> dict[str, Any]:
            if self.gripping.enabled:
                result = self.gripping.damp()
                if not result.get("ok") and self.robot_ctrl.enabled:
                    result = self.robot_ctrl.damp()
                return result
            return self.robot_ctrl.damp()

        return {
            "/api/damp": damp,
            "/api/extend_arm": self.gripping.extend_arm,
            "/api/release_arms": self.gripping.release_arms,
            "/api/unrelease_arms": self.gripping.unrelease_arms,
            "/api/stable_hold": self.gripping.stable_hold,
        }

    def _control_status_payload(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "navbot_with_gripping_v2",
            "control": self.gripping.status(),
            "robot_worker_ready": self.robot_ctrl.ready,
            "dds": self.dds_status,
            "gripping_url": self._gripping_browser_url() if self.args.enable_gripping else "",
            "last_text": self.last_text,
            "last_reply": self.last_reply,
        }

    def _start_dashboard_server(self) -> None:
        node = self
        token = str(self.args.external_asr_token or "")

        class DashboardHandler(http.server.BaseHTTPRequestHandler):
            server_version = "G1NavBotDashboard/1.0"

            def log_message(self, fmt: str, *args: Any) -> None:
                node.get_logger().info("dashboard_http " + (fmt % args))

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()
                self.wfile.write(body)

            def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _send_html(self, status: int, html_text: str) -> None:
                self._send_bytes(status, "text/html; charset=utf-8", html_text.encode("utf-8"))

            def _authorized(self, payload: dict[str, Any] | None = None) -> bool:
                if not token:
                    return True
                auth = str(self.headers.get("Authorization", ""))
                return auth == f"Bearer {token}" or bool(isinstance(payload, dict) and str(payload.get("token", "")) == token)

            def do_OPTIONS(self) -> None:
                self._send_json(200, {"ok": True})

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/":
                    self._send_html(
                        200,
                        node._dashboard_html(
                            "G1 NavBot + Gripping (v2)",
                            f"navbot dashboard on port {int(node.args.dashboard_port)}",
                        ),
                    )
                    return
                if path in {"/health", "/api/status"}:
                    self._send_json(200, node._control_status_payload())
                    return
                if path == "/api/objects":
                    self._send_json(200, node.gripping.get_objects())
                    return
                self._send_json(404, {"ok": False, "error": "not_found"})

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                routes = node._control_routes()
                if path not in routes and path not in {"/command", "/api/set_prompt", "/api/grab"}:
                    self._send_json(404, {"ok": False, "error": "not_found"})
                    return
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(min(length, 64_000)).decode("utf-8", errors="replace")
                payload: dict[str, Any] | None = None
                text = raw
                try:
                    parsed = json.loads(raw) if raw.strip() else {}
                    if isinstance(parsed, dict):
                        payload = parsed
                        text = str(parsed.get("text", parsed.get("prompt", parsed.get("raw", ""))))
                except Exception:
                    payload = None
                if not self._authorized(payload):
                    self._send_json(401, {"ok": False, "error": "unauthorized"})
                    return
                if path in routes:
                    node.get_logger().info(f"dashboard_http direct control route {path}")
                    result = routes[path]()
                    self._send_json(200 if result.get("ok") else 500, result)
                    return
                if path == "/api/set_prompt":
                    result = node.gripping.set_prompt(text)
                    self._send_json(200 if result.get("ok") else 500, result)
                    return
                if path == "/api/grab":
                    obj = str(payload.get("object", "") if isinstance(payload, dict) else "").strip()
                    result = node.gripping.grab(obj)
                    self._send_json(200 if result.get("ok") else 500, result)
                    return
                accepted = node.submit_external_asr(text)
                self._send_json(200, {"ok": True, "accepted": accepted, "text": compact_text(text)})

        host = str(self.args.dashboard_host)
        port = int(self.args.dashboard_port)
        try:
            self.dashboard_httpd = http.server.ThreadingHTTPServer((host, port), DashboardHandler)
        except Exception as exc:
            self.get_logger().warning(f"dashboard server failed to bind http://{host}:{port}: {exc}")
            return
        self.dashboard_thread = threading.Thread(target=self.dashboard_httpd.serve_forever, daemon=True)
        self.dashboard_thread.start()
        self.get_logger().info(f"navbot dashboard listening on http://{host}:{port}/ (grasp UI proxied from {self._gripping_browser_url()})")

    def _start_external_asr_server(self) -> None:
        node = self
        token = str(self.args.external_asr_token or "")

        class ExternalAsrHandler(http.server.BaseHTTPRequestHandler):
            server_version = "G1NavBotASR/1.0"

            def log_message(self, fmt: str, *args: Any) -> None:
                node.get_logger().info("external_asr_http " + (fmt % args))

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()
                self.wfile.write(body)

            def _send_html(self, status: int, html_text: str) -> None:
                body = html_text.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self, payload: dict[str, Any] | None = None) -> bool:
                if not token:
                    return True
                auth = str(self.headers.get("Authorization", ""))
                if auth == f"Bearer {token}":
                    return True
                query = self.path.split("?", 1)[1] if "?" in self.path else ""
                if f"token={token}" in query:
                    return True
                return bool(isinstance(payload, dict) and str(payload.get("token", "")) == token)

            def do_OPTIONS(self) -> None:
                self._send_json(200, {"ok": True})

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/health":
                    self._send_json(200, {"ok": True, "service": "navbot_with_gripping_v2"})
                    return
                if path == "/api/status":
                    self._send_json(200, node._control_status_payload())
                    return
                if path == "/":
                    self._send_html(
                        200,
                        f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>G1 NavBot Control (ASR endpoint)</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; max-width: 760px; }}
    button {{ font-size: 16px; padding: 10px 14px; margin: 4px; }}
    input {{ font-size: 16px; padding: 10px; width: min(520px, 100%); }}
    pre {{ background: #111; color: #eee; padding: 12px; min-height: 160px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>G1 NavBot Control (ASR endpoint)</h1>
  <p>This is the external-ASR endpoint on port {int(node.args.external_asr_port)}.
  The full control dashboard with the grasp/gizmo visualization is on port
  {int(node.args.dashboard_port)}.</p>
  <div>
    <button onclick="post('/api/damp')">Damp</button>
    <button onclick="post('/api/extend_arm')">Extend Arm</button>
    <button onclick="post('/api/release_arms')">Release Arms</button>
    <button onclick="post('/api/unrelease_arms')">Unrelease Arms</button>
    <button onclick="post('/api/stable_hold')">Stable Hold</button>
  </div>
  <p>
    <input id="cmd" placeholder="Type a navigation or gripping command">
    <button onclick="command()">Send</button>
  </p>
  <pre id="log"></pre>
  <script>
    const log = (msg) => {{
      const el = document.getElementById('log');
      el.textContent = new Date().toLocaleTimeString() + ' ' + msg + '\\n' + el.textContent;
    }};
    async function post(path, body) {{
      log('send ' + path);
      try {{
        const res = await fetch(path, {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify(body || {{}})
        }});
        log('response ' + res.status + ' ' + await res.text());
      }} catch (err) {{
        log('error ' + err);
      }}
    }}
    function command() {{
      const input = document.getElementById('cmd');
      post('/command', {{text: input.value}});
    }}
  </script>
</body>
</html>
""",
                    )
                    return
                self._send_json(404, {"ok": False, "error": "not_found"})

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                control_routes = node._control_routes()
                if path not in {"/asr", "/command"} and path not in control_routes:
                    self._send_json(404, {"ok": False, "error": "not_found"})
                    return
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(min(length, 64_000)).decode("utf-8", errors="replace")
                payload: dict[str, Any] | None = None
                text = raw
                try:
                    parsed = json.loads(raw) if raw.strip() else {}
                    if isinstance(parsed, dict):
                        payload = parsed
                        text = str(parsed.get("text", parsed.get("prompt", parsed.get("raw", ""))))
                except Exception:
                    payload = None
                if not self._authorized(payload):
                    self._send_json(401, {"ok": False, "error": "unauthorized"})
                    return
                if path in control_routes:
                    node.get_logger().info(f"external_asr_http direct control route {path}")
                    result = control_routes[path]()
                    self._send_json(200 if result.get("ok") else 500, result)
                    return
                accepted = node.submit_external_asr(text)
                self._send_json(200, {"ok": True, "accepted": accepted, "text": compact_text(text)})

        host = str(self.args.external_asr_host)
        port = int(self.args.external_asr_port)
        self.external_asr_httpd = http.server.ThreadingHTTPServer((host, port), ExternalAsrHandler)
        self.external_asr_thread = threading.Thread(target=self.external_asr_httpd.serve_forever, daemon=True)
        self.external_asr_thread.start()
        if not token and host not in {"127.0.0.1", "localhost"}:
            self.get_logger().warning("External ASR server has no token; use --external-asr-token on shared networks.")
        self.get_logger().info(f"external ASR endpoint listening on http://{host}:{port}/asr")

    @staticmethod
    def _payload_index(payload: dict[str, Any]) -> int | None:
        try:
            value = payload.get("index")
            return int(value) if value is not None else None
        except Exception:
            return None

    def destroy_node(self) -> bool:
        if self.external_asr_httpd is not None:
            self.external_asr_httpd.shutdown()
            self.external_asr_httpd.server_close()
            self.external_asr_httpd = None
        if self.external_asr_thread is not None and self.external_asr_thread.is_alive():
            self.external_asr_thread.join(timeout=0.5)
        if self.dashboard_httpd is not None:
            self.dashboard_httpd.shutdown()
            self.dashboard_httpd.server_close()
            self.dashboard_httpd = None
        if self.dashboard_thread is not None and self.dashboard_thread.is_alive():
            self.dashboard_thread.join(timeout=0.5)
        self.pose_gestures.close()
        self.robot_ctrl.close()
        self.gripping.stop()
        return super().destroy_node()


def worker_current_pose(info: SlamInfoSubscriber, timeout_s: float = 2.0) -> PoseTarget | None:
    deadline = time.time() + max(0.1, timeout_s)
    last_pose: PoseTarget | None = None
    while time.time() < deadline:
        pose = parse_pose(info.get_info()) or parse_pose(info.get_key())
        if pose is not None:
            return pose
        if last_pose is not None:
            return last_pose
        time.sleep(0.05)
    return None


def run_slam_worker(args: argparse.Namespace) -> int:
    ensure_channel_factory_initialized(int(args.domain_id), str(args.iface))
    client = SlamOperateClient()
    client.Init()
    client.SetTimeout(10.0)
    operation = str(args.slam_worker)
    result: dict[str, Any]

    try:
        if operation == "start_mapping":
            result = response_dict(client.start_mapping(slam_type=str(args.slam_type)))
        elif operation == "stop_mapping":
            info = SlamInfoSubscriber("rt/slam_info", "rt/slam_key_info")
            info.start()
            pose = worker_current_pose(info, timeout_s=1.5)
            result = response_dict(client.end_mapping(str(args.map_path)))
            if pose is not None:
                result["pose"] = pose.as_dict()
        elif operation == "close_slam":
            result = response_dict(client.close_slam())
        elif operation == "pause_nav":
            result = response_dict(client.pause_nav())
        elif operation == "resume_nav":
            result = response_dict(client.resume_nav())
        elif operation in {"pose", "relocate"}:
            pose = None
            if operation == "relocate" and args.point_json:
                try:
                    payload = json.loads(str(args.point_json))
                    pose = PoseTarget(
                        x=float(payload["x"]),
                        y=float(payload["y"]),
                        z=float(payload.get("z", 0.0)),
                        yaw=float(payload.get("yaw", 0.0)),
                    )
                except Exception:
                    pose = None
            if pose is None:
                info = SlamInfoSubscriber("rt/slam_info", "rt/slam_key_info")
                info.start()
                pose = worker_current_pose(info)
            if pose is None:
                result = {"code": 1, "ok": False, "raw": "No valid non-zero SLAM pose has been received yet."}
            elif operation == "pose":
                result = {"code": 0, "ok": True, "raw": pose.as_dict(), "pose": pose.as_dict()}
            else:
                qx, qy, qz, qw = pose.quaternion()
                result = response_dict(client.init_pose(pose.x, pose.y, pose.z, qx, qy, qz, qw, str(args.map_path)))
                result["pose"] = pose.as_dict()
        elif operation == "pose_nav":
            payload = json.loads(str(args.point_json or "{}"))
            target = PoseTarget(
                x=float(payload["x"]),
                y=float(payload["y"]),
                z=float(payload.get("z", 0.0)),
                yaw=float(payload.get("yaw", 0.0)),
            )
            qx, qy, qz, qw = target.quaternion()
            result = response_dict(client.pose_nav(target.x, target.y, target.z, qx, qy, qz, qw, mode=1))
            result["target"] = target.as_dict()
        else:
            result = {"code": 2, "ok": False, "raw": f"Unknown SLAM worker operation: {operation}"}
    except Exception as exc:
        result = {"code": 1, "ok": False, "raw": str(exc)}

    print(json.dumps(result, sort_keys=True, default=str), flush=True)
    return 0 if result.get("ok") else 1


def run_robot_worker(args: argparse.Namespace) -> int:
    """Persistent stdin/stdout worker behind RobotControlWorker.

    Keeps ONE sdk_client.Robot() connection warm for the whole process
    lifetime, instead of v1's per-command subprocess (which reconnected a
    fresh Robot()/DDS session for every single step/turn/gesture — that
    reconnect latency, plus move_for()'s built-in instant-stop, is what made
    locomotion feel snappy). move_for ramps velocity up and down instead of
    snapping to/from zero. Arm/hand detail actions (extend/release/hand
    open-close/stable-hold) are intentionally NOT handled here: they are
    owned by recognition_app.py's own control robot when gripping is
    enabled, so two processes are never fighting over arm_sdk authority at
    once. Only body-level actions that recognition_app.py doesn't own
    (locomotion, HL gestures, emergency damp) live in this worker.

    Protocol: one JSON object per line on stdin, one JSON object per line of
    response on stdout. Diagnostics go to stderr (drained by the parent's
    _drain_stderr) so they never get interleaved with stdout responses.
    """

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        ensure_channel_factory_initialized(int(args.domain_id), str(args.iface))
        from sdk_client import Robot

        robot = Robot(
            iface=str(args.iface),
            domain_id=int(args.domain_id),
            safety_boot=False,
            recover_dev_mode_on_init=False,
            auto_start_sensors=False,
        )
        log(f"robot control worker ready iface={args.iface} domain_id={int(args.domain_id)}")
    except Exception as exc:
        log(f"robot control worker failed to start: {exc}")
        robot = None

    def reply(result: dict[str, Any]) -> None:
        print(json.dumps(result, sort_keys=True, default=str), flush=True)

    def move_for(payload: dict[str, Any]) -> dict[str, Any]:
        if robot is None:
            return {"ok": False, "code": 1, "error": "robot_unavailable", "action": "move_for"}
        vx = float(payload.get("vx", 0.0))
        vy = float(payload.get("vy", 0.0))
        vyaw = float(payload.get("vyaw", 0.0))
        duration_s = max(0.05, float(payload.get("duration_s", LOCO_DURATION_S)))
        ramp_up_s = max(0.0, float(payload.get("ramp_up_s", LOCO_RAMP_UP_S)))
        ramp_down_s = max(0.0, float(payload.get("ramp_down_s", LOCO_RAMP_DOWN_S)))
        steps = max(1, LOCO_RAMP_STEPS)
        try:
            if ramp_up_s > 0.0:
                for step in range(1, steps + 1):
                    frac = step / steps
                    robot.loco_move(vx * frac, vy * frac, vyaw * frac)
                    time.sleep(ramp_up_s / steps)
            else:
                robot.loco_move(vx, vy, vyaw)
            hold_s = max(0.0, duration_s - ramp_up_s - ramp_down_s)
            if hold_s > 0.0:
                time.sleep(hold_s)
            if ramp_down_s > 0.0:
                for step in range(steps - 1, -1, -1):
                    frac = step / steps
                    robot.loco_move(vx * frac, vy * frac, vyaw * frac)
                    time.sleep(ramp_down_s / steps)
            robot.stop()
            return {
                "ok": True, "code": 0, "action": "move_for",
                "vx": vx, "vy": vy, "vyaw": vyaw, "duration_s": duration_s,
            }
        except Exception as exc:
            try:
                robot.stop()
            except Exception:
                pass
            return {"ok": False, "code": 1, "error": str(exc), "action": "move_for"}

    def stop() -> dict[str, Any]:
        if robot is None:
            return {"ok": False, "code": 1, "error": "robot_unavailable", "action": "stop"}
        try:
            robot.stop()
            return {"ok": True, "code": 0, "action": "stop"}
        except Exception as exc:
            return {"ok": False, "code": 1, "error": str(exc), "action": "stop"}

    def damp() -> dict[str, Any]:
        if robot is None:
            return {"ok": False, "code": 1, "error": "robot_unavailable", "action": "damp"}
        try:
            robot.damp()
            return {"ok": True, "code": 0, "action": "damp"}
        except Exception as exc:
            return {"ok": False, "code": 1, "error": str(exc), "action": "damp"}

    def hl_action(payload: dict[str, Any]) -> dict[str, Any]:
        action_key = str(payload.get("action", "")).strip().lower()
        if action_key not in set(GESTURE_ACTIONS.values()):
            return {"ok": False, "code": 2, "error": f"unsupported gesture: {action_key}", "action": action_key}
        if robot is None:
            return {"ok": False, "code": 1, "error": "robot_unavailable", "action": action_key}
        try:
            method = getattr(robot, action_key)
            code = int(method())
            return {"ok": code == 0, "code": code, "action": action_key}
        except Exception as exc:
            return {"ok": False, "code": 1, "error": str(exc), "action": action_key}

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            reply({"ok": False, "error": f"bad_json: {exc}"})
            continue
        if not isinstance(payload, dict):
            reply({"ok": False, "error": "bad_payload"})
            continue
        cmd = str(payload.get("cmd", "")).strip().lower()
        if cmd == "close":
            reply({"ok": True, "action": "close"})
            break
        if cmd == "move_for":
            reply(move_for(payload))
        elif cmd == "stop":
            reply(stop())
        elif cmd == "damp":
            reply(damp())
        elif cmd == "hl_action":
            reply(hl_action(payload))
        else:
            reply({"ok": False, "error": f"unsupported_cmd: {cmd}"})
    return 0


def main() -> int:
    args = parse_args()
    if args.slam_worker:
        return run_slam_worker(args)
    if args.robot_worker:
        return run_robot_worker(args)
    node: NavBotNode | None = None
    try:
        rclpy.init()
        node = NavBotNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
