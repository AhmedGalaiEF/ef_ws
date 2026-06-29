#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


SCRIPT_DIR = Path(__file__).resolve().parent
G1_DIR = SCRIPT_DIR.parents[2] if SCRIPT_DIR.name == "ollama_ai" else SCRIPT_DIR.parent
if not (G1_DIR / "WBC").exists():
    G1_DIR = Path("/home/unitree/EF/ef_ws_clean/ef_ws/g1")
SCRIPTS_DIR = G1_DIR / "modules" / "scripts"
WBC_DIR = G1_DIR / "WBC"
for path in (SCRIPTS_DIR, WBC_DIR, G1_DIR / "modules"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chat import clean_reply  # noqa: E402
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
    "{\"intent\":\"rag_question|chat|thanks|stop|unknown\","
    "\"announce\":\"short phrase the robot should say before acting\","
    "\"needs_knowledge\":true,"
    "\"motion\":\"thinking|explain|thanks|none\"}. "
    "Use rag_question for factual questions that need stored knowledge. "
    "Use chat for normal conversational questions. Use thanks for gratitude. "
    "Use stop for stop/cancel requests."
)
KNOWLEDGE_SYSTEM_PROMPT = (
    "Use the structured knowledge context when relevant. For questions about "
    "that knowledge, answer only from context. If context does not contain the "
    "answer, say you do not know yet. Keep it spoken and concise."
)
WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
FILLERS = {"ah", "eh", "er", "hmm", "hm", "mm", "uh", "um"}
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
    parser.add_argument("--answer-fillers", action="store_true")
    parser.add_argument("--iface", default=os.environ.get("G1_IFACE", "eth0"))
    parser.add_argument("--domain-id", type=int, default=int(os.environ.get("G1_DOMAIN_ID", "0")))
    parser.add_argument("--volume", type=int, default=None)
    parser.add_argument("--tts-language", default=None)
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--pose-file", default=str(WBC_DIR / "saved_ik_pose_cli_v3_poses.json"))
    parser.add_argument("--motion-speed", type=float, default=0.22, help="IK joint ramp speed in rad/s.")
    parser.add_argument("--thinking-speed", type=float, default=0.16)
    parser.add_argument("--explain-speed", type=float, default=0.22)
    parser.add_argument("--sequence-gap", type=float, default=0.15)
    parser.add_argument("--pose-timeout-s", type=float, default=8.0)
    parser.add_argument("--startup-speech", default="chatbot ready")
    parser.add_argument("--no-startup-speech", action="store_true")
    parser.add_argument("--audit-log", default="/tmp/ollama_chatbot.jsonl")
    return parser.parse_args()


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

    def say(self, text: str) -> int:
        text = compact_text(text)
        if not text:
            return 0
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
            proc = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.stdout.strip():
            self.logger.info(proc.stdout.strip())
        return int(proc.returncode)

    def say_async(self, text: str) -> threading.Thread:
        thread = threading.Thread(target=self.say, args=(text,), daemon=True)
        thread.start()
        return thread


class MotionPlayer:
    def __init__(self, args: argparse.Namespace, logger: Any) -> None:
        self.args = args
        self.logger = logger
        self.enabled = bool(args.enable_motion)
        self.ctrl: IKPoseCLI | None = None
        self.poses_by_name: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
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
        while not self.stop_event.is_set():
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
        required = set(EXPLAIN_SEQUENCE + THINK_SEQUENCE + THANKS_SEQUENCE)
        missing = sorted(name for name in required if name not in self.poses_by_name)
        if missing:
            self.logger.warning("Missing saved pose(s): " + ", ".join(missing))

    def play_async(self, names: list[str], *, speed: float | None = None, loop: bool = False) -> threading.Thread | None:
        if not names:
            return None
        if not self.enabled:
            self.logger.info("[motion disabled] would play: " + ", ".join(names))
            return None
        self.stop_sequence()
        self.sequence_thread = threading.Thread(
            target=self._play_sequence,
            args=(list(names), speed, loop),
            daemon=True,
        )
        self.sequence_thread.start()
        return self.sequence_thread

    def stop_sequence(self) -> None:
        self.stop_event.set()
        if self.sequence_thread and self.sequence_thread.is_alive():
            self.sequence_thread.join(timeout=0.3)
        self.stop_event.clear()

    def _play_sequence(self, names: list[str], speed: float | None, loop: bool) -> None:
        while not self.stop_event.is_set():
            for name in names:
                if self.stop_event.is_set():
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

    def _apply_pose(self, pose: dict[str, Any], *, name: str, speed: float | None) -> None:
        with self.lock:
            if self.ctrl is None:
                return
            self.ctrl.max_speed = max(0.01, float(speed if speed is not None else self.args.motion_speed))
            self.ctrl._apply_joint_pose(pose, include_waist=True)
            self.ctrl.status = f"Chatbot pose: {name}"

    def _wait_targets_reached(self, timeout_s: float) -> bool:
        deadline = time.time() + max(0.1, timeout_s)
        while time.time() < deadline and not self.stop_event.is_set():
            with self.lock:
                if self.ctrl is not None and self.ctrl._targets_reached():
                    return True
            time.sleep(0.05)
        return False

    def close(self) -> None:
        self.stop_event.set()
        if self.sequence_thread and self.sequence_thread.is_alive():
            self.sequence_thread.join(timeout=0.5)
        if self.tick_thread and self.tick_thread.is_alive():
            self.tick_thread.join(timeout=0.8)
        with self.lock:
            if self.ctrl is not None and not self.ctrl._closed:
                self.ctrl.close()


class ChatbotNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("ollama_ai_chatbot")
        self.args = args
        self.ollama = OllamaClient(args)
        self.speaker = Speaker(args, self.get_logger())
        self.motion = MotionPlayer(args, self.get_logger())
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
        self.busy_lock = threading.Lock()
        self.response_pub = self.create_publisher(String, args.response_topic, 10)
        self.create_subscription(String, args.audio_topic, self.on_audio, 10)
        if str(args.filtered_audio_topic) and str(args.filtered_audio_topic) != str(args.audio_topic):
            self.create_subscription(String, args.filtered_audio_topic, self.on_audio, 10)
        self.create_subscription(String, args.command_topic, self.on_command, 10)
        self.get_logger().info(
            f"chatbot ready audio={args.audio_topic} model={args.model} router={args.router_model} "
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
        if not self._should_answer(text, confidence, index, now):
            return
        self.last_index = index
        self.last_text = text
        threading.Thread(target=self._handle_text, args=(text, "audio"), daemon=True).start()

    def on_command(self, msg: String) -> None:
        payload = decode_payload(str(msg.data))
        text = compact_text(str(payload.get("text", payload.get("prompt", ""))))
        if text:
            threading.Thread(target=self._handle_text, args=(text, "command"), daemon=True).start()

    def _handle_text(self, text: str, source: str) -> None:
        if not self.busy_lock.acquire(blocking=False):
            self.speaker.say_async("I am still finishing the previous answer.")
            return
        started = time.time()
        try:
            route = self._route(text)
            intent = str(route.get("intent", "chat")).lower()
            announce = compact_text(str(route.get("announce", "")))
            self._audit({"kind": "route", "source": source, "text": text, "route": route})
            if intent == "stop":
                self.motion.stop_sequence()
                answer = announce or "Stopping."
                self.speaker.say_async(answer)
                self._publish({"ok": True, "intent": intent, "answer": answer, "elapsed_s": time.time() - started})
                return
            if intent == "thanks":
                answer = announce or "You're welcome."
                self.motion.play_async(THANKS_SEQUENCE, speed=float(self.args.explain_speed))
                self.speaker.say_async(answer)
                self._publish({"ok": True, "intent": intent, "answer": answer, "elapsed_s": time.time() - started})
                return
            if announce:
                self.speaker.say(announce)
            if bool(route.get("needs_knowledge", intent == "rag_question")):
                self.motion.play_async(THINK_SEQUENCE, speed=float(self.args.thinking_speed), loop=True)
                answer = self._rag_answer(text)
                self.motion.stop_sequence()
                self.motion.play_async(EXPLAIN_SEQUENCE, speed=float(self.args.explain_speed), loop=True)
                self.speaker.say(answer)
                self.motion.stop_sequence()
                self.motion.play_async(["Explain_base"], speed=float(self.args.explain_speed))
            else:
                answer = self._chat_answer(text)
                self.motion.play_async(EXPLAIN_SEQUENCE, speed=float(self.args.explain_speed), loop=False)
                self.speaker.say(answer)
            self.last_reply = answer
            self.last_reply_ts = time.time()
            self._publish({"ok": True, "intent": intent, "answer": answer, "elapsed_s": time.time() - started})
        except Exception as exc:
            self.get_logger().error(f"chatbot error: {exc}")
            self.motion.stop_sequence()
            answer = f"I hit an error: {exc}"
            self.speaker.say_async("I had trouble answering that.")
            self._publish({"ok": False, "answer": answer, "error": str(exc), "elapsed_s": time.time() - started})
        finally:
            self.busy_lock.release()

    def _route(self, text: str) -> dict[str, Any]:
        fast = self._route_fast(text)
        if fast:
            return fast
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
        route = extract_json_object(raw)
        if isinstance(route, dict):
            return route
        return {"intent": "chat", "announce": "", "needs_knowledge": False, "motion": "explain"}

    def _route_fast(self, text: str) -> dict[str, Any] | None:
        low = text.lower().strip()
        if low in {"stop", "stop moving", "cancel", "halt"}:
            return {"intent": "stop", "announce": "Stopping.", "needs_knowledge": False, "motion": "none"}
        if any(word in low for word in ("thank", "thanks", "danke")):
            return {"intent": "thanks", "announce": "You're welcome.", "needs_knowledge": False, "motion": "thanks"}
        question_mark = "?" in text or low.split(" ", 1)[0] in {"what", "why", "how", "when", "where", "who", "which"}
        if question_mark and self.retriever is not None:
            return {"intent": "rag_question", "announce": "Let me think.", "needs_knowledge": True, "motion": "thinking"}
        return None

    def _rag_answer(self, text: str) -> str:
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
        return self.ollama.chat(messages, temperature=float(self.args.temperature), num_predict=int(self.args.num_predict))

    def _chat_answer(self, text: str) -> str:
        history = self.history[-max(1, int(self.args.max_history)):]
        if not history or history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
        messages = [*history, {"role": "user", "content": text}]
        answer = self.ollama.chat(messages, temperature=float(self.args.temperature), num_predict=int(self.args.num_predict))
        self.history = [*messages, {"role": "assistant", "content": answer}][-(int(self.args.max_history) + 1):]
        return answer or "I heard you, but I am not sure how to answer yet."

    def _should_answer(self, text: str, confidence: float, index: int | None, received_at: float) -> bool:
        if not text or confidence < float(self.args.min_confidence):
            return False
        normalized = text.strip().lower().strip(string.punctuation + "，。！？、；：")
        if not self.args.answer_fillers and normalized in FILLERS:
            return False
        if not any(char.isalnum() for char in text):
            return False
        if index is not None and index == self.last_index:
            return False
        if index is None and text == self.last_text and received_at - self.last_reply_ts < 2.0:
            return False
        if received_at - self.last_reply_ts < float(self.args.post_speak_ignore_s):
            return False
        if self.last_reply and SequenceMatcher(None, text.lower(), self.last_reply.lower()).ratio() >= 0.82:
            return False
        return True

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
        self.motion.close()
        return super().destroy_node()


def main() -> int:
    args = parse_args()
    try:
        rclpy.init()
        node = ChatbotNode(args)
    except ControllerLockError as exc:
        print(f"chatbot: {exc}", file=sys.stderr)
        return 2
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
