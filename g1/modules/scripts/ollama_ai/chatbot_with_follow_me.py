#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import json
import math
import os
import re
import struct
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

if "--motion-worker" not in sys.argv:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
else:
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment,misc]
    String = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
G1_DIR = SCRIPT_DIR.parents[2] if SCRIPT_DIR.name == "ollama_ai" else SCRIPT_DIR.parent
if not (G1_DIR / "WBC").exists():
    G1_DIR = Path("/home/unitree/EF/ef_ws_clean/ef_ws/g1")
SCRIPTS_DIR = G1_DIR / "modules" / "scripts"
WBC_DIR = G1_DIR / "WBC"
for path in (SCRIPTS_DIR, WBC_DIR, G1_DIR / "modules"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ik_pose_cli_v3 import (  # noqa: E402
    IKPoseCLI,
    ControllerLockError,
    _parse_args as _parse_ik_args,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are the voice of a Unitree G1 humanoid robot. Answer naturally and "
    "concisely. Do not mention hidden reasoning, tools, or model internals."
)
ROUTER_PROMPT = (
    "Return only JSON with this schema: "
    "{\"intent\":\"one of: rag_question, chat, thanks, stop, gesture, unknown\","
    "\"announce\":\"short phrase the robot should say before acting\","
    "\"needs_knowledge\":true,"
    "\"motion\":\"one of: thinking, explain, thanks, face_wave, high_wave, clap, shake_hand, none\"}. "
    "Use rag_question for factual questions that need stored knowledge. "
    "Use chat for normal conversational questions. Use thanks for gratitude. "
    "Use gesture for requests to wave, greet, clap, or shake hands. "
    "Use stop for stop/cancel requests."
)
KNOWLEDGE_SYSTEM_PROMPT = (
    "Use the structured knowledge context when relevant. For questions about "
    "that knowledge, answer only from context. If context does not contain the "
    "answer, say you do not know yet. Keep it spoken and concise."
)
WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
CJK_OR_KANA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
SPOKEN_STATUS_ECHOES = (
    "chatbot ready",
    "let me think",
    "i had trouble answering that",
    "had trouble answering that",
    "trouble answering that",
    "trouble answer that",
    "trouble answering it",
    "answering that",
)
FILLERS = {"ah", "eh", "er", "hmm", "hm", "mm", "uh", "um"}
STOP_REQUESTS = {
    "stop",
    "stop talking",
    "stop speaking",
    "be quiet",
    "quiet",
    "cancel",
    "halt",
    "shut up",
}
UNINTELLIGIBLE_ASR = {
    "japanese letter",
    "japanese letters",
    "chinese letter",
    "chinese letters",
    "korean letter",
    "korean letters",
    "foreign letter",
    "foreign letters",
}
VALID_INTENTS = {"rag_question", "chat", "thanks", "stop", "gesture", "unknown", "release_arms", "diagnostic", "locomotion", "self_intro", "follow_me", "stop_follow"}
VALID_MOTIONS = {"thinking", "explain", "thanks", "face_wave", "high_wave", "clap", "shake_hand", "none"}
ROBOT_ACTION_INTENTS = {"release_arms", "diagnostic", "locomotion"}
LOCO_DURATION_S = 1.0
LOCO_LINEAR_SPEED_MPS = 0.18
LOCO_YAW_SPEED_RPS = 0.45
FOLLOW_TARGET_DISTANCE_M = 1.25
FOLLOW_MAX_VX_MPS = 0.16
FOLLOW_MAX_YAW_RPS = 0.35
FOLLOW_MAX_RANGE_M = 4.5
FOLLOW_MIN_PERSON_SCORE = 0.65
FOLLOW_MIN_BOX_DEPTH_FRACTION = 0.18
FOLLOW_LIDAR_CONFIRM_TOLERANCE_M = 0.65
STOP_WORDS = {
    "a", "about", "and", "are", "as", "at", "be", "can", "could", "do",
    "does", "for", "from", "how", "i", "in", "is", "it", "me", "of",
    "on", "or", "our", "please", "tell", "that", "the", "this", "to",
    "we", "what", "when", "where", "which", "who", "why", "with", "you",
    "your",
}

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
THINK_SEQUENCE = ["think"]
THANKS_SEQUENCE = ["thanks"]
REST_POSE = "unreleased"
THANKS_RETURN_POSE = REST_POSE
THINK_RETURN_POSE = REST_POSE
HL_ACTIONS = {
    "face_wave": "face_wave",
    "high_wave": "high_wave",
    "wave": "face_wave",
    "clap": "clap",
    "shake_hand": "shake_hand",
    "handshake": "shake_hand",
    "high_five": "high_five",
    "heart": "heart",
    "hands_up": "hands_up",
}


@dataclass(frozen=True)
class KnowledgeEntry:
    title: str
    text: str
    source: str
    path: str
    tokens: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ollama chatbot with lightweight intent routing, RAG answers, TTS, and saved IK pose gestures."
    )
    parser.add_argument("knowledge_file", nargs="*", help="Optional structured JSON knowledge file(s).")
    parser.add_argument("--audio-topic", default="/audio_msg")
    parser.add_argument("--filtered-audio-topic", default="/audio_msg/filter")
    parser.add_argument("--command-topic", default="/model_api/chatbot_command")
    parser.add_argument("--response-topic", default="/model_api/chatbot_response")
    parser.add_argument("--external-asr-server", action="store_true",
                        help="Start an HTTP endpoint for external ASR/headset text input.")
    parser.add_argument("--external-asr-host", default="0.0.0.0")
    parser.add_argument("--external-asr-port", type=int, default=8095)
    parser.add_argument("--external-asr-token", default="",
                        help="Optional bearer/query/JSON token required by the external ASR endpoint.")
    parser.add_argument("--external-asr-only", "--no-ros-audio", dest="external_asr_only",
                        action="store_true",
                        help="Do not subscribe to robot ROS ASR audio topics; use command/external ASR input only.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--router-model", default="qwen2.5:0.5b")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--num-predict", type=int, default=160)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--keep-alive", default="15m")
    parser.add_argument("--max-history", type=int, default=6)
    parser.add_argument("--knowledge-top-k", type=int, default=4)
    parser.add_argument("--knowledge-min-score", type=float, default=0.06)
    parser.add_argument("--knowledge-max-chars", type=int, default=2600)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--post-speak-ignore-s", type=float, default=1.5)
    parser.add_argument("--error-speech-cooldown-s", type=float, default=30.0,
                        help="Minimum seconds between spoken backend error messages.")
    parser.add_argument("--answer-fillers", action="store_true")
    parser.add_argument("--iface", default=os.environ.get("G1_IFACE", "eth0"))
    parser.add_argument("--domain-id", type=int, default=int(os.environ.get("G1_DOMAIN_ID", "0")))
    parser.add_argument("--volume", type=int, default=None)
    parser.add_argument("--tts-language", default=None)
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--pose-file", default=str(WBC_DIR / "saved_ik_pose_cli_v3_poses.json"))
    parser.add_argument("--motion-speed", type=float, default=0.32, help="IK joint ramp speed in rad/s.")
    parser.add_argument("--motion-kp", type=float, default=30.0, help="Arm hold kp for IK gesture motions.")
    parser.add_argument("--motion-kd", type=float, default=1.5, help="Arm hold kd for IK gesture motions.")
    parser.add_argument("--thinking-speed", type=float, default=0.23)
    parser.add_argument("--explain-speed", type=float, default=0.36)
    parser.add_argument("--sequence-gap", type=float, default=0.25)
    parser.add_argument("--pose-timeout-s", type=float, default=11.0)
    parser.add_argument("--post-sequence-hold-s", type=float, default=4.0,
                        help="Seconds to wait at the final pose before finishing the sequence.")
    parser.add_argument("--thanks-hold-s", type=float, default=7.0,
                        help=f"Seconds to hold the thanks pose before returning to {THANKS_RETURN_POSE}.")
    parser.add_argument("--release-after-sequence", action="store_true",
                        help="Release arm gains after a gesture sequence instead of holding the final pose.")
    parser.add_argument("--follow-web-host", default="127.0.0.1")
    parser.add_argument("--follow-web-port", type=int, default=8096)
    parser.add_argument("--follow-loop-s", type=float, default=0.7)
    parser.add_argument("--follow-rgbd-host", default=os.environ.get("G1_RGBD_HOST", "192.168.2.41"))
    parser.add_argument("--follow-rgbd-port", type=int, default=int(os.environ.get("G1_RGBD_PORT", "5555")))
    parser.add_argument("--follow-rgbd-topic", default=os.environ.get("G1_RGBD_TOPIC", ""))
    parser.add_argument("--follow-max-lidar-points", type=int, default=5000)
    parser.add_argument("--follow-disable-lidar-confirm", action="store_true",
                        help="Do not use lidar points to confirm/refine follow-me RGBD person targets.")
    parser.add_argument("--startup-speech", default="chatbot ready")
    parser.add_argument("--no-startup-speech", action="store_true")
    parser.add_argument("--audit-log", default="/tmp/ollama_chatbot.jsonl")
    parser.add_argument("--motion-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def clean_reply(text: str) -> str:
    text = str(text).strip()
    while "<think>" in text and "</think>" in text:
        before, rest = text.split("<think>", 1)
        _hidden, after = rest.split("</think>", 1)
        text = (before + after).strip()
    return " ".join(text.split())


def compact_text(text: str) -> str:
    return clean_reply(" ".join(str(text).split()))


def decode_payload(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    return {"text": raw}


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = compact_text(text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


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

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        timeout: float | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": model or str(self.args.model),
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
            with urllib.request.urlopen(request, timeout=float(timeout or self.args.timeout)) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        return compact_text(str(result.get("message", {}).get("content", "")))


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


class PrintLogger:
    def info(self, message: str) -> None:
        print(message, flush=True)

    def warning(self, message: str) -> None:
        print(f"WARNING: {message}", file=sys.stderr, flush=True)

    def error(self, message: str) -> None:
        print(f"ERROR: {message}", file=sys.stderr, flush=True)


class MotionPlayer:
    def __init__(self, args: argparse.Namespace, logger: Any) -> None:
        self.args = args
        self.logger = logger
        self.enabled = bool(args.enable_motion)
        self.ctrl: IKPoseCLI | None = None
        self.robot: Any | None = None
        self.poses_by_name: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.close_event = threading.Event()
        self.sequence_stop_event = threading.Event()
        self.sequence_active = threading.Event()
        self.tick_thread: threading.Thread | None = None
        self.sequence_thread: threading.Thread | None = None
        if self.enabled:
            self._start_controller()
        else:
            self._load_pose_names_only()

    def _start_controller(self) -> None:
        argv = [
            "ik_pose_cli_v3.py",
            "--iface", str(self.args.iface),
            "--domain-id", str(int(self.args.domain_id)),
            "--file", str(self.args.pose_file),
            "--speed-rad-s", str(float(self.args.motion_speed)),
            "--kp", str(float(self.args.motion_kp)),
            "--kd", str(float(self.args.motion_kd)),
            "--arm-control", "both",
            "--hand-control", "off",
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            ik_args = _parse_ik_args()
        finally:
            sys.argv = old_argv
        self.ctrl = IKPoseCLI(ik_args)
        self.ctrl._last_tick = time.monotonic()
        self.ctrl.arm_control_mode = "both"
        self.ctrl.max_speed = max(0.01, float(self.args.motion_speed))
        self.poses_by_name = {str(p.get("name", "")): p for p in self.ctrl.saved_poses}
        self.tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self.tick_thread.start()
        self._wait_seeded(timeout_s=5.0)
        self._release_for_high_level_action(reason="startup")
        self._check_required_poses()

    def _load_pose_names_only(self) -> None:
        path = Path(self.args.pose_file).expanduser()
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        poses = payload.get("poses", [])
        if isinstance(poses, list):
            self.poses_by_name = {str(p.get("name", "")): p for p in poses if isinstance(p, dict)}

    def _tick_loop(self) -> None:
        while not self.close_event.is_set():
            with self.lock:
                ctrl = self.ctrl
                if ctrl is not None and not ctrl._closed:
                    try:
                        ctrl.tick()
                    except Exception as exc:
                        ctrl.status = f"Tick error: {exc}"
                        self.logger.warning(str(exc))
            hz = self.ctrl.rate_hz if self.ctrl else 25.0
            time.sleep(1.0 / max(1.0, float(hz)))

    def _wait_seeded(self, timeout_s: float) -> None:
        deadline = time.time() + max(0.1, timeout_s)
        while time.time() < deadline:
            with self.lock:
                if self.ctrl is not None and self.ctrl.seeded:
                    return
            time.sleep(0.05)
        self.logger.warning("IK controller has not received lowstate yet; motions will start after seeding.")

    def _check_required_poses(self) -> None:
        required = set(EXPLAIN_SEQUENCE + THINK_SEQUENCE + THANKS_SEQUENCE + [THANKS_RETURN_POSE, THINK_RETURN_POSE])
        missing = sorted(name for name in required if name not in self.poses_by_name)
        if missing:
            self.logger.warning("Missing saved pose(s): " + ", ".join(missing))

    def play_async(self, names: list[str], *, speed: float | None = None, loop: bool = False) -> threading.Thread | None:
        if not names:
            return None
        if not self.enabled:
            self.logger.info("[motion disabled] would play: " + ", ".join(names))
            return None
        if self.sequence_active.is_set() or (self.sequence_thread and self.sequence_thread.is_alive()):
            self.logger.warning("Motion command ignored because a sequence is already active.")
            return None
        self.sequence_stop_event.clear()
        self.sequence_active.set()
        self.sequence_thread = threading.Thread(
            target=self._play_sequence,
            args=(list(names), speed, loop),
            daemon=True,
        )
        self.sequence_thread.start()
        return self.sequence_thread

    def play_thanks_async(self, *, speed: float | None = None) -> threading.Thread | None:
        if not self.enabled:
            self.logger.info(f"[motion disabled] would play thanks, hold, then {THANKS_RETURN_POSE}")
            return None
        if self.sequence_active.is_set() or (self.sequence_thread and self.sequence_thread.is_alive()):
            self.logger.warning("Thanks motion ignored because a sequence is already active.")
            return None
        self.sequence_stop_event.clear()
        self.sequence_active.set()
        self.sequence_thread = threading.Thread(
            target=self._play_thanks_sequence,
            args=(speed,),
            daemon=True,
        )
        self.sequence_thread.start()
        return self.sequence_thread

    def stop_sequence(self) -> None:
        self.sequence_stop_event.set()
        if self.sequence_thread and self.sequence_thread.is_alive():
            timeout_s = max(3.0, float(self.args.pose_timeout_s) + 3.0)
            self.sequence_thread.join(timeout=timeout_s)
            if self.sequence_thread.is_alive():
                self.logger.warning("Motion sequence did not stop before timeout.")

    def _play_sequence(self, names: list[str], speed: float | None, loop: bool) -> None:
        try:
            self._reengage_for_sequence()
            self._move_to_rest_pose_before_sequence(speed=speed)
            while not self.sequence_stop_event.is_set():
                for name in names:
                    if self.sequence_stop_event.is_set():
                        return
                    pose = self.poses_by_name.get(name)
                    if pose is None:
                        self.logger.warning(f"Skipping missing pose: {name}")
                        continue
                    self._apply_pose(pose, name=name, speed=speed)
                    self._wait_targets_reached(timeout_s=float(self.args.pose_timeout_s))
                    gap = max(0.0, float(self.args.sequence_gap))
                    if gap:
                        time.sleep(gap)
                if not loop:
                    return
        finally:
            if list(names) == THINK_SEQUENCE and not bool(getattr(self.args, "release_after_sequence", False)):
                self._return_to_pose(THINK_RETURN_POSE, speed=speed)
            self._release_after_sequence()
            self.sequence_active.clear()

    def _sleep_interruptible(self, duration_s: float) -> None:
        deadline = time.time() + max(0.0, float(duration_s))
        while time.time() < deadline and not self.sequence_stop_event.is_set():
            time.sleep(min(0.05, max(0.0, deadline - time.time())))

    def _play_thanks_sequence(self, speed: float | None) -> None:
        try:
            self._reengage_for_sequence()
            self._move_to_rest_pose_before_sequence(speed=speed)
            thanks_pose = self.poses_by_name.get("thanks")
            return_pose = self.poses_by_name.get(THANKS_RETURN_POSE)
            if thanks_pose is None:
                self.logger.warning("Skipping thanks motion: missing pose 'thanks'")
                return
            self._apply_pose(thanks_pose, name="thanks", speed=speed)
            self._wait_targets_reached(timeout_s=float(self.args.pose_timeout_s))
            self._sleep_interruptible(float(self.args.thanks_hold_s))
            if self.sequence_stop_event.is_set():
                return
            if return_pose is None:
                self.logger.warning(f"Skipping thanks return: missing pose '{THANKS_RETURN_POSE}'")
                return
            self._apply_pose(return_pose, name=THANKS_RETURN_POSE, speed=speed)
            self._wait_targets_reached(timeout_s=float(self.args.pose_timeout_s))
        finally:
            self._release_after_sequence()
            self.sequence_active.clear()

    def _return_to_pose(self, name: str, *, speed: float | None) -> None:
        pose = self.poses_by_name.get(name)
        if pose is None:
            self.logger.warning(f"Skipping return motion: missing pose '{name}'")
            return
        self.sequence_stop_event.clear()
        self._apply_pose(pose, name=name, speed=speed)
        self._wait_targets_reached(timeout_s=float(self.args.pose_timeout_s))

    def _move_to_rest_pose_before_sequence(self, *, speed: float | None) -> None:
        if self.sequence_stop_event.is_set():
            return
        pose = self.poses_by_name.get(REST_POSE)
        if pose is None:
            self.logger.warning(f"Skipping sequence start pose: missing pose '{REST_POSE}'")
            return
        self._apply_pose(pose, name=REST_POSE, speed=speed)
        self._wait_targets_reached(timeout_s=float(self.args.pose_timeout_s))

    def play_hl_action(self, action: str) -> bool:
        action_key = HL_ACTIONS.get(str(action).strip().lower())
        if not action_key:
            self.logger.warning(f"Unsupported high-level arm action: {action}")
            return False
        if not self.enabled:
            self.logger.info(f"[motion disabled] would run high-level action: {action_key}")
            return False
        if self.sequence_active.is_set() or (self.sequence_thread and self.sequence_thread.is_alive()):
            self.logger.warning("High-level action ignored because a sequence is already active.")
            return False
        self.sequence_active.set()
        try:
            self._release_for_high_level_action(reason=action_key)
            robot = self._get_robot()
            method = getattr(robot, action_key)
            code = int(method())
            self.logger.info(f"High-level arm action {action_key} returned {code}.")
            return code == 0
        except Exception as exc:
            self.logger.warning(f"High-level arm action {action_key} failed: {exc}")
            return False
        finally:
            self.sequence_active.clear()

    def _release_for_high_level_action(self, *, reason: str) -> None:
        with self.lock:
            ctrl = self.ctrl
            if ctrl is None or ctrl._closed:
                return
            if not ctrl.armed:
                self.logger.info(f"Arms already released for high-level action ({reason}).")
                return
            # High-level arm actions use the robot arm action service. Release
            # low-level arm_sdk authority first so the IK tick loop cannot hold
            # gains against the HL controller.
            ctrl._release_arms(duration_s=1.0)
            ctrl.armed = False
        self.logger.info(f"Arms released for high-level action ({reason}).")

    def release_arms_for_user(self) -> bool:
        if not self.enabled:
            self.logger.info("[motion disabled] would release arms")
            return False
        if self.sequence_active.is_set() or (self.sequence_thread and self.sequence_thread.is_alive()):
            self.stop_sequence()
        try:
            self._release_for_high_level_action(reason="user_release")
            robot = self._get_robot()
            code = int(robot.release_arm())
            self.logger.info(f"High-level release arm returned {code}.")
            return code == 0
        except Exception as exc:
            self.logger.warning(f"Release arms failed: {exc}")
            return False

    def move_for_user(self, vx: float, vy: float, vyaw: float, duration_s: float = LOCO_DURATION_S) -> bool:
        if not self.enabled:
            self.logger.info(f"[motion disabled] would move vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}")
            return False
        if self.sequence_active.is_set() or (self.sequence_thread and self.sequence_thread.is_alive()):
            self.logger.warning("Locomotion command ignored because a sequence is already active.")
            return False
        self.sequence_active.set()
        try:
            robot = self._get_robot()
            duration_s = max(0.05, min(LOCO_DURATION_S, float(duration_s)))
            code = int(robot.move_for(duration_s, vx=float(vx), vy=float(vy), vyaw=float(vyaw)))
            self.logger.info(
                f"Locomotion move_for returned {code}: duration={duration_s:.2f}s "
                f"vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}"
            )
            return code == 0
        except Exception as exc:
            self.logger.warning(f"Locomotion command failed: {exc}")
            return False
        finally:
            self.sequence_active.clear()

    def _get_robot(self) -> Any:
        if self.robot is None:
            from sdk_client import Robot
            self.robot = Robot(
                iface=str(self.args.iface),
                domain_id=int(self.args.domain_id),
                safety_boot=False,
                recover_dev_mode_on_init=False,
                auto_start_sensors=False,
                ollama_url=str(self.args.ollama_url),
                chat_model=str(self.args.model),
            )
        return self.robot

    def _reengage_for_sequence(self) -> None:
        try:
            with self.lock:
                ctrl = self.ctrl
                if ctrl is None:
                    return
                # Hold the worker lock across the whole ramp. The tick loop uses
                # the same lock before publishing rt/arm_sdk, so this prevents
                # release/reengage packets and normal target packets overlapping.
                ctrl._unrelease_arms(duration_s=0.8)
                ctrl.armed = True
            self.logger.info("Arms reengaged for motion sequence at current pose.")
        except Exception as exc:
            self.logger.warning(f"Could not reengage arms before sequence: {exc}")

    def _release_after_sequence(self) -> None:
        try:
            hold_s = max(0.0, float(self.args.post_sequence_hold_s))
            if hold_s:
                time.sleep(hold_s)
            with self.lock:
                ctrl = self.ctrl
                if ctrl is None:
                    return
                if not bool(getattr(self.args, "release_after_sequence", False)):
                    ctrl.current_targets = dict(ctrl.desired_targets)
                    ctrl.armed = True
                    ctrl.status = "Chatbot sequence complete; holding final pose"
                    ctrl.pub.publish(
                        ctrl.current_targets,
                        arm_kp=ctrl.arm_kp,
                        arm_kd=ctrl.arm_kd,
                        waist_pr_kp=ctrl.waist_pr_kp if ctrl.waist_enabled else 0.0,
                        waist_y_kp=ctrl.waist_y_kp if ctrl.waist_enabled else 0.0,
                        waist_kd=ctrl.waist_kd if ctrl.waist_enabled else 0.0,
                    )
                    self.logger.info("Holding final pose after motion sequence.")
                    return
                # See _reengage_for_sequence: this must be serialized with the
                # background tick publisher to avoid contradictory low-level arm
                # commands on rt/arm_sdk.
                ctrl._release_arms(duration_s=1.6)
                ctrl.armed = False
            self.logger.info("Arms released after motion sequence.")
        except Exception as exc:
            self.logger.warning(f"Could not release arms after sequence: {exc}")

    def _apply_pose(self, pose: dict[str, Any], *, name: str, speed: float | None) -> None:
        with self.lock:
            if self.ctrl is None:
                return
            self.ctrl.max_speed = max(0.01, float(speed if speed is not None else self.args.motion_speed))
            self.ctrl._apply_joint_pose(pose, include_waist=True)
            self.ctrl.status = f"Chatbot pose: {name}"

    def _wait_targets_reached(self, timeout_s: float) -> bool:
        deadline = time.time() + max(0.1, timeout_s)
        while time.time() < deadline and not self.sequence_stop_event.is_set():
            with self.lock:
                if self.ctrl is not None and self.ctrl._targets_reached():
                    return True
            time.sleep(0.05)
        return False

    def close(self) -> None:
        self.sequence_stop_event.set()
        self.close_event.set()
        if self.sequence_thread and self.sequence_thread.is_alive():
            self.sequence_thread.join(timeout=0.5)
        if self.tick_thread and self.tick_thread.is_alive():
            self.tick_thread.join(timeout=0.8)
        with self.lock:
            if self.ctrl is not None and not self.ctrl._closed:
                try:
                    self.ctrl.close()
                except KeyboardInterrupt:
                    self.logger.warning("Interrupted while closing IK controller.")


class MotionWorkerClient:
    def __init__(self, args: argparse.Namespace, logger: Any) -> None:
        self.args = args
        self.logger = logger
        self.enabled = bool(args.enable_motion)
        self.proc: subprocess.Popen[str] | None = None
        if self.enabled:
            self._start_worker()

    def _start_worker(self) -> None:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
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
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._log_worker_output, daemon=True).start()
        self.logger.info(f"Started motion worker pid={self.proc.pid}")

    def _log_worker_output(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            text = line.strip()
            if text:
                self.logger.info(f"[motion] {text}")

    def play_async(self, names: list[str], *, speed: float | None = None, loop: bool = False) -> None:
        if not names:
            return
        if not self.enabled:
            self.logger.info("[motion disabled] would play: " + ", ".join(names))
            return
        self._send({"cmd": "play", "names": names, "speed": speed, "loop": bool(loop)})

    def play_thanks_async(self, *, speed: float | None = None) -> None:
        if not self.enabled:
            self.logger.info(f"[motion disabled] would play thanks, hold, then {THANKS_RETURN_POSE}")
            return
        self._send({"cmd": "thanks", "speed": speed})

    def stop_sequence(self) -> None:
        if self.enabled:
            self._send({"cmd": "stop"})

    def hl_action(self, action: str) -> None:
        if not self.enabled:
            self.logger.info(f"[motion disabled] would run high-level action: {action}")
            return
        self._send({"cmd": "hl_action", "action": str(action)})

    def release_arms(self) -> None:
        if not self.enabled:
            self.logger.info("[motion disabled] would release arms")
            return
        self._send({"cmd": "release_arms"})

    def move_for(
        self,
        *,
        vx: float = 0.0,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration_s: float = LOCO_DURATION_S,
    ) -> None:
        if not self.enabled:
            self.logger.info(
                f"[motion disabled] would move duration={duration_s:.2f}s "
                f"vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}"
            )
            return
        self._send({
            "cmd": "move_for",
            "vx": float(vx),
            "vy": float(vy),
            "vyaw": float(vyaw),
            "duration_s": float(duration_s),
        })

    def close(self) -> None:
        if self.proc is None:
            return
        self._send({"cmd": "close"})
        try:
            self.proc.wait(timeout=2.0)
        except KeyboardInterrupt:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=0.5)
            except Exception:
                self.proc.kill()
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _send(self, payload: dict[str, Any]) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None:
            self.logger.warning("Motion worker is not running.")
            return
        if proc.poll() is not None:
            self.logger.warning(f"Motion worker exited with code {proc.returncode}.")
            return
        try:
            proc.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
            proc.stdin.flush()
        except BrokenPipeError:
            self.logger.warning("Motion worker pipe is closed.")


def run_motion_worker(args: argparse.Namespace) -> int:
    motion = MotionPlayer(args, PrintLogger())
    print("motion worker ready", flush=True)
    try:
        for line in sys.stdin:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"invalid command: {exc}", flush=True)
                continue
            cmd = str(payload.get("cmd", "")).strip().lower()
            if cmd == "play":
                names = payload.get("names", [])
                if isinstance(names, list):
                    motion.play_async(
                        [str(name) for name in names],
                        speed=payload.get("speed"),
                        loop=bool(payload.get("loop", False)),
                    )
            elif cmd == "thanks":
                motion.play_thanks_async(speed=payload.get("speed"))
            elif cmd == "stop":
                motion.stop_sequence()
            elif cmd == "hl_action":
                motion.play_hl_action(str(payload.get("action", "")))
            elif cmd == "release_arms":
                motion.release_arms_for_user()
            elif cmd == "move_for":
                motion.move_for_user(
                    vx=float(payload.get("vx", 0.0) or 0.0),
                    vy=float(payload.get("vy", 0.0) or 0.0),
                    vyaw=float(payload.get("vyaw", 0.0) or 0.0),
                    duration_s=float(payload.get("duration_s", LOCO_DURATION_S) or LOCO_DURATION_S),
                )
            elif cmd == "close":
                break
            else:
                print(f"unknown command: {cmd}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        motion.close()
    return 0


class FollowMeController:
    def __init__(self, args: argparse.Namespace, logger: Any, speaker: Speaker, motion: MotionWorkerClient) -> None:
        self.args = args
        self.logger = logger
        self.speaker = speaker
        self.motion = motion
        self.zmq_context: Any | None = None
        self.zmq_socket: Any | None = None
        self.lidar_robot: Any | None = None
        self.lock = threading.RLock()
        self.rgbd_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.httpd: http.server.ThreadingHTTPServer | None = None
        self.http_thread: threading.Thread | None = None
        self.latest_jpeg: bytes | None = None
        self.status: dict[str, Any] = {
            "active": False,
            "phase": "idle",
            "target": None,
            "rgbd_ok": False,
            "lidar_ok": False,
            "odom_pose": None,
            "rgbd_source": f"tcp://{self.args.follow_rgbd_host}:{self.args.follow_rgbd_port}",
            "last_frame_age_s": None,
            "last_error": "",
            "last_update": 0.0,
        }
        self._announced_recognition_complete = False
        self._announced_following = False
        self._start_web()

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            self.logger.info("Follow-me is already active.")
            return
        self.stop_event.clear()
        self._announced_recognition_complete = False
        self._announced_following = False
        with self.lock:
            self.status.update({"active": True, "phase": "recognizing", "last_error": "", "last_update": time.time()})
        self.speaker.say_async("recognizing human")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        try:
            self.motion.move_for(vx=0.0, vy=0.0, vyaw=0.0, duration_s=0.05)
        except Exception as exc:
            self.logger.warning(f"Could not stop locomotion after follow-me: {exc}")
        with self.lock:
            self.status.update({"active": False, "phase": "idle", "last_update": time.time()})

    def close(self) -> None:
        self.stop()
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        if self.http_thread and self.http_thread.is_alive():
            self.http_thread.join(timeout=0.5)
        self._close_rgbd_socket()
        if self.lidar_robot is not None:
            try:
                self.lidar_robot.close()
            except Exception:
                pass
            self.lidar_robot = None

    def _run(self) -> None:
        while not self.stop_event.is_set():
            started = time.time()
            try:
                target = self._update_rgb()
                with self.lock:
                    self.status.update({
                        "active": True,
                        "phase": "tracking" if target else "recognizing",
                        "target": target,
                        "odom_pose": None,
                        "last_error": "",
                        "last_frame_age_s": self._latest_frame_age_s(),
                        "last_update": time.time(),
                    })
                if target:
                    if not self._announced_recognition_complete:
                        self._announced_recognition_complete = True
                        self.speaker.say_async("recognition complete")
                    self._follow_target(target)
            except Exception as exc:
                with self.lock:
                    self.status.update({"last_error": str(exc), "last_update": time.time()})
                self.logger.warning(f"Follow-me loop error: {exc}")
            elapsed = time.time() - started
            time.sleep(max(0.05, float(self.args.follow_loop_s) - elapsed))

    def _ensure_rgbd_socket(self) -> Any:
        if self.zmq_socket is not None:
            return self.zmq_socket
        try:
            import zmq
        except Exception as exc:
            raise RuntimeError(f"RGBD stream requires pyzmq: {exc}") from exc
        self.zmq_context = zmq.Context.instance()
        socket = self.zmq_context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, str(self.args.follow_rgbd_topic).encode("utf-8"))
        socket.setsockopt(zmq.RCVTIMEO, 250)
        socket.setsockopt(zmq.RCVHWM, 1)
        endpoint = f"tcp://{self.args.follow_rgbd_host}:{int(self.args.follow_rgbd_port)}"
        socket.connect(endpoint)
        self.zmq_socket = socket
        self.logger.info(f"follow-me RGBD connected to {endpoint}")
        return socket

    def _close_rgbd_socket(self) -> None:
        socket = self.zmq_socket
        self.zmq_socket = None
        if socket is not None:
            try:
                socket.close(0)
            except Exception:
                pass

    def _latest_frame_age_s(self) -> float | None:
        with self.lock:
            updated = float(self.status.get("last_update", 0.0) or 0.0)
        if updated <= 0.0 or self.latest_jpeg is None:
            return None
        return max(0.0, time.time() - updated)

    def _recv_rgbd_frame(self, timeout: float = 0.35) -> dict[str, Any]:
        try:
            import zmq
        except Exception as exc:
            raise RuntimeError(f"RGBD stream requires pyzmq: {exc}") from exc
        with self.rgbd_lock:
            socket = self._ensure_rgbd_socket()
            deadline = time.time() + max(0.1, float(timeout))
            last_error = ""
            latest: list[bytes] | None = None
            while time.time() < deadline:
                try:
                    parts = socket.recv_multipart()
                except zmq.Again:
                    if latest is not None:
                        break
                    continue
                except Exception as exc:
                    self._close_rgbd_socket()
                    raise RuntimeError(f"RGBD receive failed: {exc}") from exc
                if len(parts) >= 4:
                    parts = parts[-3:]
                if len(parts) < 2:
                    last_error = f"expected RGBD multipart frame, got {len(parts)} part(s)"
                    continue
                latest = [bytes(part) for part in parts]
                while True:
                    try:
                        newer = socket.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    except Exception as exc:
                        self._close_rgbd_socket()
                        raise RuntimeError(f"RGBD receive failed while draining: {exc}") from exc
                    if len(newer) >= 4:
                        newer = newer[-3:]
                    if len(newer) >= 2:
                        latest = [bytes(part) for part in newer]
                if latest is None:
                    continue
                parts = latest
                depth_scale = 0.001
                if len(parts) >= 3 and len(parts[2]) >= 4:
                    try:
                        depth_scale = float(struct.unpack("f", parts[2][:4])[0])
                    except Exception:
                        depth_scale = 0.001
                return {
                    "timestamp": time.time(),
                    "rgb_jpeg": parts[0],
                    "depth_png": parts[1],
                    "depth_scale_m_per_unit": depth_scale,
                }
        endpoint = f"tcp://{self.args.follow_rgbd_host}:{int(self.args.follow_rgbd_port)}"
        detail = f": {last_error}" if last_error else ""
        raise RuntimeError(f"No RGBD frames received from {endpoint}{detail}")

    def _update_rgb(self) -> dict[str, Any] | None:
        try:
            frame = self._recv_rgbd_frame(timeout=0.35)
            frame = self._decode_rgbd_frame(frame)
            target = self._target_from_rgbd(frame)
            if target is not None and not bool(self.args.follow_disable_lidar_confirm):
                target = self._confirm_target_with_lidar(target)
            self.latest_jpeg = self._draw_human_box(frame, target)
            with self.lock:
                self.status["rgbd_ok"] = True
                self.status["last_rgbd_error"] = ""
                self.status["rgbd_source"] = f"tcp://{self.args.follow_rgbd_host}:{int(self.args.follow_rgbd_port)}"
                self.status["last_frame_age_s"] = 0.0
                self.status["last_update"] = time.time()
            return target
        except Exception as exc:
            with self.lock:
                self.status["rgbd_ok"] = False
                self.status["last_rgbd_error"] = str(exc)
                self.status["last_update"] = time.time()
            if self.latest_jpeg is None:
                self.latest_jpeg = None
            return None

    def _decode_rgbd_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        try:
            import cv2
            import numpy as np
        except Exception as exc:
            raise RuntimeError(f"RGBD decoding requires cv2 and numpy: {exc}") from exc
        rgb_jpeg = bytes(frame["rgb_jpeg"])
        depth_png = bytes(frame["depth_png"])
        depth_scale = float(frame.get("depth_scale_m_per_unit", 0.001) or 0.001)
        rgb = cv2.imdecode(np.frombuffer(rgb_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        depth_raw = cv2.imdecode(np.frombuffer(depth_png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if rgb is None:
            raise RuntimeError("Failed to decode RGB JPEG from RGBD stream.")
        if depth_raw is None:
            raise RuntimeError("Failed to decode depth PNG from RGBD stream.")
        if depth_raw.ndim == 3:
            depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
        depth_m = depth_raw.astype("float32") * depth_scale
        h, w = depth_raw.shape[:2]
        cx = w // 2
        cy = h // 2
        center_size = max(8, min(w, h) // 10)
        center = depth_m[
            max(0, cy - center_size): min(h, cy + center_size),
            max(0, cx - center_size): min(w, cx + center_size),
        ]
        valid_center = center[center > 0]
        center_depth_m = float(np.median(valid_center)) if valid_center.size else None
        frame.update({
            "rgb_bgr": rgb,
            "depth_raw": depth_raw,
            "depth_m": depth_m,
            "center_depth_m": center_depth_m,
            "valid_depth_fraction": float((depth_raw > 0).mean()) if depth_raw.size else 0.0,
        })
        return frame

    def _target_from_rgbd(self, frame: dict[str, Any]) -> dict[str, Any] | None:
        try:
            import numpy as np
        except Exception:
            return None

        rgb = frame.get("rgb_bgr")
        depth_m = frame.get("depth_m")
        if rgb is None or depth_m is None:
            return None
        h, w = rgb.shape[:2]
        candidates = self._detect_people(rgb)
        frame["person_candidates"] = candidates
        best: dict[str, Any] | None = None
        best_score = -1.0
        for cand in candidates:
            x, y, bw, bh = (int(cand["x"]), int(cand["y"]), int(cand["w"]), int(cand["h"]))
            pad_x = max(2, int(bw * 0.12))
            pad_top = max(2, int(bh * 0.12))
            pad_bottom = max(2, int(bh * 0.08))
            x0 = max(0, x + pad_x)
            x1 = min(w, x + bw - pad_x)
            y0 = max(0, y + pad_top)
            y1 = min(h, y + bh - pad_bottom)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = depth_m[y0:y1, x0:x1]
            valid = roi[(roi > 0.25) & (roi <= FOLLOW_MAX_RANGE_M)]
            valid_fraction = float(valid.size) / float(max(1, roi.size))
            if valid_fraction < FOLLOW_MIN_BOX_DEPTH_FRACTION:
                continue
            distance_m = float(np.median(valid))
            if not math.isfinite(distance_m) or distance_m <= 0.25 or distance_m > FOLLOW_MAX_RANGE_M:
                continue
            center_px = float(x + bw * 0.5)
            image_y_m = ((center_px / max(1.0, float(w))) - 0.5) * 2.0 * distance_m * 0.55
            score = float(cand.get("score", 0.0)) + min(0.35, valid_fraction * 0.35)
            if score > best_score:
                best_score = score
                best = {
                    "source": "rgbd_person",
                    "x_m": distance_m,
                    "y_m": image_y_m,
                    "confidence": min(0.95, score),
                    "box": {"x": x, "y": y, "w": bw, "h": bh},
                    "depth_valid_fraction": valid_fraction,
                }
        return best

    def _target_from_lidar(self, robot: Any) -> dict[str, Any] | None:
        points = robot.get_lidar_points(max_points=max(100, int(self.args.follow_max_lidar_points)))
        if not points:
            return None
        xs: list[float] = []
        ys: list[float] = []
        for point in points:
            try:
                x = float(point["x"])
                y = float(point["y"])
                z = float(point["z"])
            except Exception:
                continue
            if 0.45 <= x <= 3.5 and abs(y) <= 1.25 and -0.45 <= z <= 1.8:
                xs.append(x)
                ys.append(y)
        if len(xs) < 12:
            return None
        xs.sort()
        ys.sort()
        mid = len(xs) // 2
        return {"source": "lidar_front_cluster", "x_m": xs[mid], "y_m": ys[mid], "confidence": min(1.0, len(xs) / 200.0)}

    def _get_lidar_robot(self) -> Any:
        if self.lidar_robot is None:
            from sdk_client import Robot
            self.lidar_robot = Robot(
                iface=str(self.args.iface),
                domain_id=int(self.args.domain_id),
                safety_boot=False,
                recover_dev_mode_on_init=False,
                auto_start_sensors=True,
                ollama_url=str(self.args.ollama_url),
                chat_model=str(self.args.model),
            )
        return self.lidar_robot

    def _confirm_target_with_lidar(self, target: dict[str, Any]) -> dict[str, Any] | None:
        try:
            robot = self._get_lidar_robot()
            points = robot.get_lidar_points(max_points=max(100, int(self.args.follow_max_lidar_points)))
        except Exception as exc:
            with self.lock:
                self.status["lidar_ok"] = False
                self.status["last_lidar_error"] = str(exc)
            return target
        if not points:
            with self.lock:
                self.status["lidar_ok"] = False
                self.status["last_lidar_error"] = "no lidar points"
            return target

        x_hint = float(target.get("x_m", 0.0) or 0.0)
        y_hint = float(target.get("y_m", 0.0) or 0.0)
        xs: list[float] = []
        ys: list[float] = []
        for point in points:
            try:
                x = float(point["x"])
                y = float(point["y"])
                z = float(point["z"])
            except Exception:
                continue
            if not (0.35 <= x <= FOLLOW_MAX_RANGE_M and -0.55 <= z <= 1.9):
                continue
            if abs(x - x_hint) <= FOLLOW_LIDAR_CONFIRM_TOLERANCE_M and abs(y - y_hint) <= 0.65:
                xs.append(x)
                ys.append(y)
        if len(xs) < 8:
            with self.lock:
                self.status["lidar_ok"] = True
                self.status["last_lidar_error"] = "no confirming cluster near RGBD person"
            if float(target.get("confidence", 0.0) or 0.0) < 0.82:
                return None
            return target

        xs.sort()
        ys.sort()
        mid = len(xs) // 2
        refined = dict(target)
        refined.update({
            "source": "rgbd_person_lidar",
            "x_m": float(xs[mid]),
            "y_m": float(ys[mid]),
            "confidence": min(0.99, float(target.get("confidence", 0.0) or 0.0) + min(0.25, len(xs) / 120.0)),
            "lidar_points": len(xs),
        })
        with self.lock:
            self.status["lidar_ok"] = True
            self.status["last_lidar_error"] = ""
        return refined

    def _detect_people(self, image: Any) -> list[dict[str, Any]]:
        try:
            import cv2
        except Exception:
            return []
        h, w = image.shape[:2]
        scale = 1.0
        detect_image = image
        if w > 640:
            scale = 640.0 / float(w)
            detect_h = max(1, int(h * scale))
            detect_image = cv2.resize(image, (640, detect_h), interpolation=cv2.INTER_AREA)
        dh, dw = detect_image.shape[:2]
        min_height = max(64, int(h * 0.22))
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = hog.detectMultiScale(detect_image, winStride=(8, 8), padding=(8, 8), scale=1.05)
        candidates: list[dict[str, Any]] = []
        for (dx, dy, dbw, dbh), weight in zip(boxes, weights):
            x = int(dx / scale)
            y = int(dy / scale)
            bw = int(dbw / scale)
            bh = int(dbh / scale)
            score = float(weight)
            aspect = float(bw) / float(max(1, bh))
            if score < FOLLOW_MIN_PERSON_SCORE:
                continue
            if bh < min_height or not (0.22 <= aspect <= 0.85):
                continue
            if bw * bh < 0.015 * w * h or dbw * dbh < 0.015 * dw * dh:
                continue
            candidates.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "score": score})
        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        return candidates[:4]

    def _follow_target(self, target: dict[str, Any]) -> None:
        if not bool(self.args.enable_motion):
            return
        x = float(target.get("x_m", FOLLOW_TARGET_DISTANCE_M))
        y = float(target.get("y_m", 0.0))
        error_x = x - FOLLOW_TARGET_DISTANCE_M
        vx = max(-FOLLOW_MAX_VX_MPS, min(FOLLOW_MAX_VX_MPS, 0.45 * error_x))
        if abs(error_x) < 0.18:
            vx = 0.0
        vyaw = max(-FOLLOW_MAX_YAW_RPS, min(FOLLOW_MAX_YAW_RPS, 0.45 * y))
        if abs(y) < 0.18:
            vyaw = 0.0
        if abs(vx) < 0.01 and abs(vyaw) < 0.02:
            return
        if not self._announced_following:
            self._announced_following = True
            self.speaker.say_async("following")
        self.motion.move_for(vx=vx, vy=0.0, vyaw=vyaw, duration_s=0.35)
        with self.lock:
            self.status["last_command"] = {"vx": vx, "vy": 0.0, "vyaw": vyaw, "duration_s": 0.35}

    def _draw_human_box(self, frame: dict[str, Any], target: dict[str, Any] | None) -> bytes:
        rgb_jpeg = bytes(frame.get("rgb_jpeg", b""))
        try:
            import cv2
            import numpy as np
            image = frame.get("rgb_bgr")
            if image is not None:
                image = image.copy()
            else:
                image = cv2.imdecode(np.frombuffer(rgb_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return rgb_jpeg
            for cand in frame.get("person_candidates", []) or []:
                x, y, w, h = int(cand["x"]), int(cand["y"]), int(cand["w"]), int(cand["h"])
                cv2.rectangle(image, (x, y), (x + w, y + h), (80, 140, 255), 1)
            if target and target.get("box"):
                box = target["box"]
                x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"human {float(target.get('confidence', 0.0)):.2f} {float(target.get('x_m', 0.0)):.1f}m"
                cv2.putText(image, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            return encoded.tobytes() if ok else rgb_jpeg
        except Exception:
            return rgb_jpeg

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.status)

    def _start_web(self) -> None:
        controller = self

        class FollowHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/api/follow":
                    self._send_json(controller.snapshot())
                    return
                if path == "/rgb.jpg":
                    data = controller.latest_jpeg
                    if data:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                    else:
                        svg = controller._placeholder_svg().encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "image/svg+xml")
                        self.send_header("Content-Length", str(len(svg)))
                        self.end_headers()
                        self.wfile.write(svg)
                    return
                html = controller._html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)

            def _send_json(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        host = str(self.args.follow_web_host)
        port = int(self.args.follow_web_port)
        self.httpd = http.server.ThreadingHTTPServer((host, port), FollowHandler)
        self.http_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.http_thread.start()
        self.logger.info(f"follow-me web UI listening on http://{host}:{port}/")

    def _placeholder_svg(self) -> str:
        snap = self.snapshot()
        msg = str(snap.get("last_rgbd_error") or "No RGBD frame yet")
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540">
<rect width="100%" height="100%" fill="#101418"/>
<text x="40" y="80" fill="#dbe7ef" font-family="sans-serif" font-size="32">RGBD unavailable</text>
<text x="40" y="130" fill="#94a3ad" font-family="sans-serif" font-size="22">{msg[:120]}</text>
</svg>"""

    def _html(self) -> str:
        return """<!doctype html>
<html><head><meta charset="utf-8"><title>G1 Follow Me</title>
<style>
body { font-family: sans-serif; margin: 0; background: #111; color: #eee; }
main { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; padding: 16px; }
img { width: 100%; background: #000; border: 1px solid #333; }
pre { white-space: pre-wrap; background: #1b1f24; padding: 12px; border: 1px solid #333; }
</style></head><body>
<main>
  <section><h1>Follow Me RGB</h1><img id="rgb" src="/rgb.jpg"></section>
  <section><h1>Status</h1><pre id="status"></pre></section>
</main>
<script>
async function refresh() {
  document.getElementById('rgb').src = '/rgb.jpg?t=' + Date.now();
  const res = await fetch('/api/follow?t=' + Date.now());
  document.getElementById('status').textContent = JSON.stringify(await res.json(), null, 2);
}
setInterval(refresh, 500); refresh();
</script></body></html>"""


class ChatbotNode(Node):
    def __init__(self, args: argparse.Namespace, motion: MotionWorkerClient | None = None) -> None:
        super().__init__("ollama_ai_chatbot")
        self.args = args
        self.ollama = OllamaClient(args)
        self.speaker = Speaker(args, self.get_logger())
        self.motion = motion if motion is not None else MotionWorkerClient(args, self.get_logger())
        self.diagnostic_robot: Any | None = None
        self.follow_me = FollowMeController(args, self.get_logger(), self.speaker, self.motion)
        self.audit_path = Path(args.audit_log).expanduser()
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        knowledge_paths = [Path(item).expanduser() for item in args.knowledge_file]
        missing = [str(path) for path in knowledge_paths if not path.exists()]
        if missing:
            self.get_logger().warning("Knowledge file(s) not found: " + ", ".join(missing))
        self.retriever = KnowledgeRetriever([path for path in knowledge_paths if path.exists()]) if knowledge_paths else None
        self.history: list[dict[str, str]] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        self.last_index: int | None = None
        self.last_text = ""
        self.last_reply = ""
        self.last_reply_ts = 0.0
        self.last_error_speech_ts = 0.0
        self.busy_lock = threading.Lock()
        self.interrupt_event = threading.Event()
        self.last_unintelligible_ts = 0.0
        self.external_asr_httpd: http.server.ThreadingHTTPServer | None = None
        self.external_asr_thread: threading.Thread | None = None
        self.response_pub = self.create_publisher(String, args.response_topic, 10)
        if not bool(args.external_asr_only):
            self.create_subscription(String, args.audio_topic, self.on_audio, 10)
            if str(args.filtered_audio_topic) and str(args.filtered_audio_topic) != str(args.audio_topic):
                self.create_subscription(String, args.filtered_audio_topic, self.on_audio, 10)
        else:
            self.get_logger().info("robot ROS ASR audio subscriptions disabled; using command/external ASR input")
        self.create_subscription(String, args.command_topic, self.on_command, 10)
        if bool(args.external_asr_server):
            self._start_external_asr_server()
        self.get_logger().info(
            f"chatbot ready audio={'external-only' if args.external_asr_only else args.audio_topic} model={args.model} router={args.router_model} "
            f"motion={'on' if args.enable_motion else 'off'}"
        )
        if not args.no_startup_speech and compact_text(args.startup_speech):
            self.speaker.say_async(args.startup_speech)

    def on_audio(self, msg: String) -> None:
        payload = decode_payload(str(msg.data))
        text = compact_text(str(payload.get("text", payload.get("raw", ""))))
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        index = self._payload_index(payload)
        now = time.time()
        self._log_heard("audio", text, confidence=confidence, index=index)
        if self._is_stop_request(text):
            self.last_index = index
            self.last_text = text
            self.get_logger().info(f"ASR accepted stop request: {text!r}")
            self._interrupt_now("audio")
            return
        if self._is_unintelligible_asr(text):
            self.last_index = index
            self.last_text = text
            self.get_logger().info(f"ASR marked unintelligible: {text!r}")
            self._handle_unintelligible(now)
            return
        reject_reason = self._answer_filter_reason(text, confidence, index, now)
        if reject_reason:
            self.get_logger().info(f"ASR ignored: reason={reject_reason} text={text!r}")
            return
        self.last_index = index
        self.last_text = text
        self.get_logger().info(f"ASR accepted: text={text!r}")
        threading.Thread(target=self._handle_text, args=(text, "audio"), daemon=True).start()

    def on_command(self, msg: String) -> None:
        payload = decode_payload(str(msg.data))
        text = compact_text(str(payload.get("text", payload.get("prompt", ""))))
        self._log_heard("command", text, confidence=None, index=self._payload_index(payload))
        if text:
            if self._is_stop_request(text):
                self.get_logger().info(f"Command accepted stop request: {text!r}")
                self._interrupt_now("command")
                return
            if self._is_unintelligible_asr(text):
                self.get_logger().info(f"Command marked unintelligible: {text!r}")
                self._handle_unintelligible(time.time())
                return
            self.get_logger().info(f"Command accepted: text={text!r}")
            threading.Thread(target=self._handle_text, args=(text, "command"), daemon=True).start()

    def submit_external_asr(self, text: str, *, source: str = "external_asr") -> bool:
        text = compact_text(text)
        self._log_heard(source, text, confidence=None, index=None)
        if not text:
            self.get_logger().info(f"{source} ignored: empty text")
            return False
        if self._is_stop_request(text):
            self.get_logger().info(f"{source} accepted stop request: {text!r}")
            self._interrupt_now(source)
            return True
        if self._is_unintelligible_asr(text):
            self.get_logger().info(f"{source} marked unintelligible: {text!r}")
            self._handle_unintelligible(time.time())
            return False
        reject_reason = self._answer_filter_reason(text, 1.0, None, time.time())
        if reject_reason in {"filler", "no_alphanumeric_text", "short_numeric_fragment", "non_english_asr_fragment"}:
            self.get_logger().info(f"{source} ignored: reason={reject_reason} text={text!r}")
            return False
        self.get_logger().info(f"{source} accepted: text={text!r}")
        threading.Thread(target=self._handle_text, args=(text, source), daemon=True).start()
        return True

    def _start_external_asr_server(self) -> None:
        node = self
        token = str(self.args.external_asr_token or "")

        class ExternalAsrHandler(http.server.BaseHTTPRequestHandler):
            server_version = "G1ExternalASR/1.0"

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
                    self._send_json(200, {"ok": True, "service": "external_asr"})
                    return
                if path in {"/", "/headset"}:
                    body = node._headset_html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self._send_json(404, {"ok": False, "error": "not_found"})

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                if path not in {"/asr", "/command"}:
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
                accepted = node.submit_external_asr(text, source="external_asr")
                self._send_json(200, {"ok": True, "accepted": accepted, "text": compact_text(text)})

        host = str(self.args.external_asr_host)
        port = int(self.args.external_asr_port)
        self.external_asr_httpd = http.server.ThreadingHTTPServer((host, port), ExternalAsrHandler)
        self.external_asr_thread = threading.Thread(target=self.external_asr_httpd.serve_forever, daemon=True)
        self.external_asr_thread.start()
        if not token and host not in {"127.0.0.1", "localhost"}:
            self.get_logger().warning("External ASR server has no token; use --external-asr-token on shared networks.")
        self.get_logger().info(f"external ASR endpoint listening on http://{host}:{port}/asr; headset page /headset")

    def _headset_html(self) -> str:
        token = str(self.args.external_asr_token or "")
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>G1 Headset ASR</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; max-width: 46rem; }}
    button {{ font-size: 1rem; padding: .6rem 1rem; margin-right: .5rem; }}
    #status {{ margin: 1rem 0; font-weight: 600; }}
    #log {{ white-space: pre-wrap; border: 1px solid #ccc; padding: 1rem; min-height: 12rem; }}
  </style>
</head>
<body>
  <h1>G1 Headset ASR</h1>
  <button id="start">Start</button>
  <button id="stop">Stop</button>
  <form id="manual">
    <input id="manualText" autocomplete="off" placeholder="Type a command" style="font-size:1rem; padding:.55rem; width:70%; margin-top:1rem;">
    <button type="submit">Send</button>
  </form>
  <div id="status">Idle</div>
  <div id="log"></div>
  <script>
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const log = document.getElementById('log');
    const status = document.getElementById('status');
    let rec = null;
    function add(line) {{ log.textContent = new Date().toLocaleTimeString() + ' ' + line + '\\n' + log.textContent; }}
    async function send(text) {{
      add('send: ' + text);
      const headers = {{'Content-Type': 'application/json'}};
      const token = {json.dumps(token)};
      if (token) headers.Authorization = 'Bearer ' + token;
      const res = await fetch('/asr', {{method: 'POST', headers, body: JSON.stringify({{text}})}});
      add('response: ' + await res.text());
    }}
    document.getElementById('manual').onsubmit = e => {{
      e.preventDefault();
      const input = document.getElementById('manualText');
      const text = input.value.trim();
      if (text) send(text);
      input.value = '';
    }};
    document.getElementById('start').onclick = () => {{
      if (!SpeechRecognition) {{ status.textContent = 'SpeechRecognition is not supported in this browser.'; return; }}
      rec = new SpeechRecognition();
      rec.lang = 'en-US';
      rec.continuous = true;
      rec.interimResults = false;
      rec.onstart = () => status.textContent = 'Listening';
      rec.onerror = e => {{
        status.textContent = 'Speech recognition error: ' + e.error;
        add('error: ' + e.error + (e.error === 'network' ? ' (browser speech service unavailable; use the text box)' : ''));
      }};
      rec.onend = () => status.textContent = 'Stopped';
      rec.onresult = e => {{
        for (let i = e.resultIndex; i < e.results.length; i++) {{
          if (e.results[i].isFinal) send(e.results[i][0].transcript);
        }}
      }};
      rec.start();
    }};
    document.getElementById('stop').onclick = () => {{ if (rec) rec.stop(); }};
  </script>
</body>
</html>"""

    def _handle_text(self, text: str, source: str) -> None:
        if not self.busy_lock.acquire(blocking=False):
            route = self._route_fast(text)
            busy_intent = str(route.get("intent", "")).lower() if route else ""
            if busy_intent == "stop_follow":
                answer = compact_text(str(route.get("announce", ""))) or "Stopped following."
                self.get_logger().info(f"{source} accepted stop-follow while busy: text={text!r}")
                self.follow_me.stop()
                self.speaker.say_async(answer)
                self._publish({"ok": True, "intent": "stop_follow", "answer": answer, "busy_override": True})
                return
            if busy_intent == "release_arms":
                answer = compact_text(str(route.get("announce", ""))) or "Releasing my arms."
                self.get_logger().info(f"{source} accepted release while busy: text={text!r}")
                self.motion.release_arms()
                self.speaker.say_async(answer)
                self._publish({"ok": True, "intent": "release_arms", "answer": answer, "busy_override": True})
                return
            if not self._is_stop_request(text):
                self.get_logger().info(f"{source} ignored while busy: text={text!r}")
                self.speaker.say_async("I am still finishing the previous answer.")
            return
        self.interrupt_event.clear()
        started = time.time()
        try:
            route = self._route(text)
            intent = str(route.get("intent", "chat")).lower()
            announce = compact_text(str(route.get("announce", "")))
            self.get_logger().info(f"{source} routed: text={text!r} intent={intent} route={json.dumps(route, sort_keys=True, default=str)}")
            self._audit({"kind": "route", "source": source, "text": text, "route": route})
            if intent == "unknown":
                self._publish({"ok": False, "intent": intent, "answer": "", "ignored": True, "elapsed_s": time.time() - started})
                return
            if intent == "stop":
                self._interrupt_now(source)
                answer = announce or "Stopping."
                self.speaker.say_async(answer)
                self._publish({"ok": True, "intent": intent, "answer": answer, "elapsed_s": time.time() - started})
                return
            if intent == "thanks":
                answer = announce or "You're welcome."
                self.motion.play_thanks_async(speed=float(self.args.explain_speed))
                self.speaker.say_async(answer)
                self._publish({"ok": True, "intent": intent, "answer": answer, "elapsed_s": time.time() - started})
                return
            if intent == "gesture":
                action = self._hl_action_from_route(route)
                answer = announce or self._gesture_reply(action)
                if action:
                    self.motion.hl_action(action)
                self.speaker.say_async(answer)
                self._publish({"ok": True, "intent": intent, "answer": answer, "motion": action, "elapsed_s": time.time() - started})
                return
            if intent == "release_arms":
                answer = announce or "Releasing my arms."
                self.motion.release_arms()
                self.speaker.say_async(answer)
                self._publish({"ok": True, "intent": intent, "answer": answer, "elapsed_s": time.time() - started})
                return
            if intent == "locomotion":
                vx, vy, vyaw = self._locomotion_vector_from_route(route)
                answer = announce or self._locomotion_reply(vx, vy, vyaw)
                self.motion.move_for(vx=vx, vy=vy, vyaw=vyaw)
                self.speaker.say_async(answer)
                self._publish({
                    "ok": True,
                    "intent": intent,
                    "answer": answer,
                    "duration_s": LOCO_DURATION_S,
                    "vx": vx,
                    "vy": vy,
                    "vyaw": vyaw,
                    "elapsed_s": time.time() - started,
                })
                return
            if intent == "diagnostic":
                answer = self._diagnostic_answer(str(route.get("diagnostic", "") or route.get("kind", "")))
                self.speaker.say_async(answer)
                self._publish({"ok": True, "intent": intent, "answer": answer, "elapsed_s": time.time() - started})
                return
            if intent == "self_intro":
                answer = self._self_intro_answer()
                self.speaker.say_async(answer)
                self._publish({"ok": True, "intent": intent, "answer": answer, "elapsed_s": time.time() - started})
                return
            if intent == "follow_me":
                self.follow_me.start()
                answer = announce or "Starting follow me."
                self._publish({"ok": True, "intent": intent, "answer": answer, "elapsed_s": time.time() - started})
                return
            if intent == "stop_follow":
                self.follow_me.stop()
                answer = announce or "Stopped following."
                self.speaker.say_async(answer)
                self._publish({"ok": True, "intent": intent, "answer": answer, "elapsed_s": time.time() - started})
                return
            if announce:
                self.speaker.say(announce)
            if bool(route.get("needs_knowledge", intent == "rag_question")):
                self.motion.play_async(THINK_SEQUENCE, speed=float(self.args.thinking_speed), loop=True)
                answer, used_knowledge = self._rag_answer(text)
                self.motion.stop_sequence()
                if self.interrupt_event.is_set():
                    self._publish({"ok": True, "intent": intent, "answer": "", "interrupted": True, "elapsed_s": time.time() - started})
                    return
                if used_knowledge:
                    self.motion.play_async(EXPLAIN_SEQUENCE, speed=float(self.args.explain_speed), loop=False)
                self.speaker.say(answer)
            else:
                answer = self._chat_answer(text)
                if self.interrupt_event.is_set():
                    self._publish({"ok": True, "intent": intent, "answer": "", "interrupted": True, "elapsed_s": time.time() - started})
                    return
                self.speaker.say(answer)
            self.last_reply = answer
            self.last_reply_ts = time.time()
            self._publish({"ok": True, "intent": intent, "answer": answer, "elapsed_s": time.time() - started})
        except Exception as exc:
            self.get_logger().error(f"chatbot error: {exc}")
            self.motion.stop_sequence()
            answer = f"I hit an error: {exc}"
            spoken = "I had trouble answering that."
            self.last_reply = spoken
            self.last_reply_ts = time.time()
            if self._should_speak_backend_error(source):
                self.speaker.say_async(spoken)
            self._publish({"ok": False, "answer": answer, "error": str(exc), "elapsed_s": time.time() - started})
        finally:
            self.busy_lock.release()

    def _should_speak_backend_error(self, source: str) -> bool:
        if source == "audio":
            return False
        now = time.time()
        cooldown_s = max(0.0, float(self.args.error_speech_cooldown_s))
        if now - self.last_error_speech_ts < cooldown_s:
            self.get_logger().info("Backend error speech suppressed by cooldown.")
            return False
        self.last_error_speech_ts = now
        return True

    def _route(self, text: str) -> dict[str, Any]:
        fast = self._route_fast(text)
        if fast:
            return fast
        try:
            raw = self.ollama.chat(
                [
                    {"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": text},
                ],
                model=str(self.args.router_model or self.args.model),
                temperature=0.0,
                num_predict=96,
                timeout=min(12.0, float(self.args.timeout)),
            )
        except Exception as exc:
            self.get_logger().warning(f"Router unavailable; using fallback route: {exc}")
            if self.retriever is not None and tokenize(text):
                return {"intent": "rag_question", "announce": "Let me check my local knowledge.", "needs_knowledge": True, "motion": "thinking"}
            return {"intent": "unknown", "announce": "", "needs_knowledge": False, "motion": "none"}
        route = extract_json_object(raw)
        if isinstance(route, dict):
            return self._normalize_route(route)
        return {"intent": "chat", "announce": "", "needs_knowledge": False, "motion": "none"}

    def _normalize_route(self, route: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(route)
        intent = str(normalized.get("intent", "chat")).strip().lower()
        if intent not in VALID_INTENTS:
            self.get_logger().warning(f"Router returned invalid intent {intent!r}; treating as unknown.")
            intent = "unknown"
        motion = str(normalized.get("motion", "none")).strip().lower()
        if motion not in VALID_MOTIONS:
            self.get_logger().warning(f"Router returned invalid motion {motion!r}; using none.")
            motion = "none"
        normalized["intent"] = intent
        normalized["motion"] = motion
        normalized["announce"] = compact_text(str(normalized.get("announce", "")))
        normalized["needs_knowledge"] = bool(normalized.get("needs_knowledge", intent == "rag_question"))
        return normalized

    def _route_fast(self, text: str) -> dict[str, Any] | None:
        low = text.lower().strip()
        if self._is_stop_request(text):
            return {"intent": "stop", "announce": "Stopping.", "needs_knowledge": False, "motion": "none"}
        if any(phrase in low for phrase in ("stop following", "stop follow me", "stop follow", "cancel follow")):
            return {"intent": "stop_follow", "announce": "Stopped following.", "needs_knowledge": False, "motion": "none"}
        if any(phrase in low for phrase in ("follow me", "start following", "track me", "come with me")):
            return {"intent": "follow_me", "announce": "Starting follow me.", "needs_knowledge": False, "motion": "none"}
        if any(phrase in low for phrase in ("release arm", "release arms", "release your arm", "release your arms", "release hand", "release hands", "release your hands")):
            return {"intent": "release_arms", "announce": "Releasing my arms.", "needs_knowledge": False, "motion": "none"}
        loco = self._route_locomotion_fast(low)
        if loco is not None:
            return loco
        if any(phrase in low for phrase in ("fsm", "fsm mode", "which mode", "what mode", "in which mode", "self diagnosis", "self diagnostic", "diagnose yourself")):
            return {"intent": "diagnostic", "announce": "", "needs_knowledge": False, "motion": "none", "diagnostic": "fsm"}
        if any(phrase in low for phrase in ("wifi", "wi-fi", "wireless network", "which network", "connected network", "ssid")):
            return {"intent": "diagnostic", "announce": "", "needs_knowledge": False, "motion": "none", "diagnostic": "wifi"}
        if any(phrase in low for phrase in ("introduce yourself", "who are you", "what are you", "tell me who you are")):
            return {"intent": "self_intro", "announce": "", "needs_knowledge": False, "motion": "none"}
        if any(phrase in low for phrase in ("say thank you", "say thanks", "thank them", "thank everyone")):
            return {"intent": "thanks", "announce": "Thank you.", "needs_knowledge": False, "motion": "thanks"}
        if any(word in low for word in ("thank", "thanks", "danke")):
            return {"intent": "thanks", "announce": "You're welcome.", "needs_knowledge": False, "motion": "thanks"}
        if "clap" in low or "applaud" in low or "lap your hands" in low or "sap your hands" in low:
            return {"intent": "gesture", "announce": "I will clap.", "needs_knowledge": False, "motion": "clap"}
        if "shake hand" in low or "handshake" in low:
            return {"intent": "gesture", "announce": "Nice to meet you.", "needs_knowledge": False, "motion": "shake_hand"}
        if "high five" in low:
            return {"intent": "gesture", "announce": "High five.", "needs_knowledge": False, "motion": "high_five"}
        if "wave" in low or "hello" in low or "hi " in f"{low} " or "greet" in low:
            return {"intent": "gesture", "announce": "Hello.", "needs_knowledge": False, "motion": "face_wave"}
        knowledge_request = (
            low.startswith(("tell me about", "explain", "how to", "how do", "what is", "what are"))
            or "tell me about" in low
        )
        question_mark = knowledge_request or "?" in text or low.split(" ", 1)[0] in {"what", "why", "how", "when", "where", "who", "which"}
        if question_mark and self.retriever is not None:
            return {"intent": "rag_question", "announce": "Let me think.", "needs_knowledge": True, "motion": "thinking"}
        return None

    def _route_locomotion_fast(self, low: str) -> dict[str, Any] | None:
        if not any(word in low for word in ("walk", "move", "go", "step", "turn")):
            return None
        vx = vy = vyaw = 0.0
        direction = ""
        if "forward" in low or "forwards" in low or "front" in low:
            vx = LOCO_LINEAR_SPEED_MPS
            direction = "forward"
        elif "backward" in low or "backwards" in low or "back up" in low or "reverse" in low:
            vx = -LOCO_LINEAR_SPEED_MPS
            direction = "backward"
        elif "right" in low:
            if "turn" in low or "rotate" in low:
                vyaw = -LOCO_YAW_SPEED_RPS
                direction = "turn_right"
            else:
                vy = -LOCO_LINEAR_SPEED_MPS
                direction = "right"
        elif "left" in low:
            if "turn" in low or "rotate" in low:
                vyaw = LOCO_YAW_SPEED_RPS
                direction = "turn_left"
            else:
                vy = LOCO_LINEAR_SPEED_MPS
                direction = "left"
        elif "turn" in low or "rotate" in low:
            return None
        else:
            return None
        return {
            "intent": "locomotion",
            "announce": "",
            "needs_knowledge": False,
            "motion": "none",
            "direction": direction,
            "vx": vx,
            "vy": vy,
            "vyaw": vyaw,
        }

    def _interrupt_now(self, source: str) -> None:
        self.interrupt_event.set()
        self.speaker.stop_current()
        self.motion.stop_sequence()
        self.follow_me.stop()
        self.last_reply = "Stopping."
        self.last_reply_ts = time.time()
        self._publish({"ok": True, "intent": "stop", "source": source, "answer": "Stopping.", "interrupted": True})

    def _handle_unintelligible(self, now: float) -> None:
        if now - self.last_unintelligible_ts < 3.0:
            return
        self.last_unintelligible_ts = now
        self._publish({"ok": False, "intent": "unintelligible_asr", "answer": "", "ignored": True})

    @staticmethod
    def _is_stop_request(text: str) -> bool:
        low = " ".join(str(text).lower().strip().split())
        if low in STOP_REQUESTS:
            return True
        return any(
            phrase in low
            for phrase in (
                "stop talking",
                "stop speaking",
                "please stop",
                "stop the answer",
                "cancel the answer",
                "interrupt",
            )
        )

    @staticmethod
    def _is_unintelligible_asr(text: str) -> bool:
        low = " ".join(str(text).lower().strip().strip(string.punctuation + "，。！？、；：").split())
        if low in UNINTELLIGIBLE_ASR:
            return True
        return bool(re.fullmatch(r"(japanese|chinese|korean)\s+(letter|letters|character|characters)", low))

    @staticmethod
    def _hl_action_from_route(route: dict[str, Any]) -> str:
        motion = str(route.get("motion", "")).strip().lower()
        return HL_ACTIONS.get(motion, "")

    @staticmethod
    def _gesture_reply(action: str) -> str:
        if action == "clap":
            return "Clapping."
        if action in {"face_wave", "high_wave", "wave"}:
            return "Hello."
        if action in {"shake_hand", "handshake"}:
            return "Nice to meet you."
        return "Okay."

    @staticmethod
    def _self_intro_answer() -> str:
        return (
            "I am a Unitree G1 humanoid robot running a local chatbot interface. "
            "I can answer from my loaded knowledge file, report basic status, move for short one-second commands, "
            "and run arm gestures like wave or clap."
        )

    @staticmethod
    def _locomotion_vector_from_route(route: dict[str, Any]) -> tuple[float, float, float]:
        def bounded(value: Any, limit: float) -> float:
            try:
                numeric = float(value)
            except Exception:
                return 0.0
            return max(-limit, min(limit, numeric))

        return (
            bounded(route.get("vx", 0.0), LOCO_LINEAR_SPEED_MPS),
            bounded(route.get("vy", 0.0), LOCO_LINEAR_SPEED_MPS),
            bounded(route.get("vyaw", 0.0), LOCO_YAW_SPEED_RPS),
        )

    @staticmethod
    def _locomotion_reply(vx: float, vy: float, vyaw: float) -> str:
        if abs(vyaw) > 0.0:
            return "Turning right." if vyaw < 0.0 else "Turning left."
        if abs(vx) >= abs(vy):
            return "Walking forward." if vx >= 0.0 else "Walking backward."
        return "Stepping right." if vy < 0.0 else "Stepping left."

    def _diagnostic_answer(self, kind: str) -> str:
        kind = str(kind).strip().lower()
        if kind == "wifi":
            return self._wifi_answer()
        robot = self._get_diagnostic_robot()
        try:
            fsm = robot.get_fsm()
            mode = robot.get_mode()
            return f"My FSM id is {fsm.get('id')}, FSM mode is {fsm.get('mode')}, and sport mode is {mode}."
        except Exception as exc:
            self.get_logger().warning(f"Diagnostic query failed: {exc}")
            return "I could not read my FSM mode right now."

    def _wifi_answer(self) -> str:
        try:
            proc = subprocess.run(
                ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2.0,
            )
        except Exception as exc:
            self.get_logger().warning(f"Wi-Fi diagnostic failed: {exc}")
            return "I could not read the Wi-Fi network right now."
        if proc.returncode != 0:
            self.get_logger().warning(f"Wi-Fi diagnostic failed: {proc.stderr.strip()}")
            return "I could not read the Wi-Fi network right now."
        for line in proc.stdout.splitlines():
            active, _, ssid = line.partition(":")
            if active.lower() == "yes" and ssid.strip():
                return f"I am connected to Wi-Fi network {ssid.strip()}."
        return "I do not see an active Wi-Fi network."

    def _get_diagnostic_robot(self) -> Any:
        if self.diagnostic_robot is None:
            from sdk_client import Robot
            self.diagnostic_robot = Robot(
                iface=str(self.args.iface),
                domain_id=int(self.args.domain_id),
                safety_boot=False,
                recover_dev_mode_on_init=False,
                auto_start_sensors=True,
                ollama_url=str(self.args.ollama_url),
                chat_model=str(self.args.model),
            )
        return self.diagnostic_robot

    def _rag_answer(self, text: str) -> tuple[str, bool]:
        context = ""
        if self.retriever is not None:
            context = self.retriever.format_context(
                text,
                top_k=int(self.args.knowledge_top_k),
                min_score=float(self.args.knowledge_min_score),
                max_chars=int(self.args.knowledge_max_chars),
            )
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "system", "content": f"{KNOWLEDGE_SYSTEM_PROMPT}\n\nStructured knowledge context:\n{context}" if context else KNOWLEDGE_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        try:
            answer = self.ollama.chat(messages, temperature=float(self.args.temperature), num_predict=int(self.args.num_predict))
            return answer, bool(context)
        except Exception as exc:
            self.get_logger().warning(f"Ollama RAG answer failed; using local knowledge fallback: {exc}")
            if self.retriever is not None:
                fallback = self._local_knowledge_answer(text)
                if fallback:
                    return fallback, True
            raise

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

    def _chat_answer(self, text: str) -> str:
        history = self.history[-max(1, int(self.args.max_history)):]
        if not history or history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
        messages = [*history, {"role": "user", "content": text}]
        answer = self.ollama.chat(messages, temperature=float(self.args.temperature), num_predict=int(self.args.num_predict))
        self.history = [*messages, {"role": "assistant", "content": answer}][-(int(self.args.max_history) + 1):]
        return answer or "I heard you, but I am not sure how to answer yet."

    def _answer_filter_reason(self, text: str, confidence: float, index: int | None, received_at: float) -> str | None:
        if not text or confidence < float(self.args.min_confidence):
            return f"empty_or_low_confidence confidence={confidence:.3f} min={float(self.args.min_confidence):.3f}"
        normalized = text.strip().lower().strip(string.punctuation + "，。！？、；：")
        if not self.args.answer_fillers and normalized in FILLERS:
            return "filler"
        if not any(char.isalnum() for char in text):
            return "no_alphanumeric_text"
        if re.fullmatch(r"\d{1,2}", normalized):
            return "short_numeric_fragment"
        if CJK_OR_KANA_RE.search(normalized) and not ASCII_LETTER_RE.search(normalized):
            return "non_english_asr_fragment"
        if any(phrase in normalized for phrase in SPOKEN_STATUS_ECHOES):
            return "spoken_status_echo"
        if index is not None and index == self.last_index:
            return f"duplicate_index index={index}"
        if index is None and text == self.last_text and received_at - self.last_reply_ts < 2.0:
            return f"duplicate_text age_s={received_at - self.last_reply_ts:.2f}"
        if received_at - self.last_reply_ts < float(self.args.post_speak_ignore_s):
            return f"post_speak_ignore age_s={received_at - self.last_reply_ts:.2f} window_s={float(self.args.post_speak_ignore_s):.2f}"
        if self.last_reply and SequenceMatcher(None, text.lower(), self.last_reply.lower()).ratio() >= 0.82:
            return "looks_like_tts_echo"
        return None

    def _should_answer(self, text: str, confidence: float, index: int | None, received_at: float) -> bool:
        return self._answer_filter_reason(text, confidence, index, received_at) is None

    def _log_heard(self, source: str, text: str, *, confidence: float | None, index: int | None) -> None:
        parts = [f"{source} heard text={text!r}"]
        if confidence is not None:
            parts.append(f"confidence={confidence:.3f}")
        if index is not None:
            parts.append(f"index={index}")
        self.get_logger().info(" ".join(parts))

    def _publish(self, result: dict[str, Any]) -> None:
        result["time"] = time.time()
        self.response_pub.publish(String(data=json.dumps(result, sort_keys=True, default=str)))
        self._audit({"kind": "response", "result": result})

    def _audit(self, record: dict[str, Any]) -> None:
        record = {"time": time.time(), **record}
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    @staticmethod
    def _payload_index(payload: dict[str, Any]) -> int | None:
        try:
            value = payload.get("index")
            return int(value) if value is not None else None
        except Exception:
            return None

    def destroy_node(self) -> bool:
        self.follow_me.close()
        if self.external_asr_httpd is not None:
            self.external_asr_httpd.shutdown()
            self.external_asr_httpd.server_close()
            self.external_asr_httpd = None
        if self.external_asr_thread is not None and self.external_asr_thread.is_alive():
            self.external_asr_thread.join(timeout=0.5)
        self.motion.close()
        return super().destroy_node()


def main() -> int:
    args = parse_args()
    if args.motion_worker:
        return run_motion_worker(args)
    motion: MotionWorkerClient | None = None
    node: ChatbotNode | None = None
    try:
        rclpy.init()
        motion = MotionWorkerClient(args, PrintLogger()) if args.enable_motion else None
        node = ChatbotNode(args, motion=motion)
    except ControllerLockError as exc:
        print(f"chatbot: {exc}", file=sys.stderr)
        return 2
    except Exception:
        if motion is not None:
            motion.close()
        raise
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
            elif motion is not None:
                motion.close()
        except Exception:
            pass
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
