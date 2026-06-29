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
    "{\"intent\":\"rag_question|chat|thanks|stop|gesture|unknown\","
    "\"announce\":\"short phrase the robot should say before acting\","
    "\"needs_knowledge\":true,"
    "\"motion\":\"thinking|explain|thanks|face_wave|high_wave|clap|shake_hand|none\"}. "
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
    parser.add_argument("--motion-speed", type=float, default=0.32, help="IK joint ramp speed in rad/s.")
    parser.add_argument("--thinking-speed", type=float, default=0.23)
    parser.add_argument("--explain-speed", type=float, default=0.36)
    parser.add_argument("--sequence-gap", type=float, default=0.25)
    parser.add_argument("--pose-timeout-s", type=float, default=11.0)
    parser.add_argument("--post-sequence-hold-s", type=float, default=1.2,
                        help="Seconds to hold the final pose before releasing arm gains.")
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
            self._release_after_sequence()
            self.sequence_active.clear()

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
            with self.lock:
                ctrl = self.ctrl
                if ctrl is not None and not ctrl._closed:
                    # High-level arm actions use the robot arm action service.
                    # Release low-level arm_sdk authority first so the IK tick
                    # loop is not holding gains against the HL controller.
                    ctrl._release_arms(duration_s=1.0)
                    ctrl.armed = False
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
                ctrl._sync_targets_to_live()
            self.logger.info("Arms reengaged for motion sequence.")
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
                self.ctrl.close()


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
            "--thinking-speed", str(float(self.args.thinking_speed)),
            "--explain-speed", str(float(self.args.explain_speed)),
            "--sequence-gap", str(float(self.args.sequence_gap)),
            "--pose-timeout-s", str(float(self.args.pose_timeout_s)),
            "--post-sequence-hold-s", str(float(self.args.post_sequence_hold_s)),
            "--no-speech",
            "--no-startup-speech",
        ]
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

    def stop_sequence(self) -> None:
        if self.enabled:
            self._send({"cmd": "stop"})

    def hl_action(self, action: str) -> None:
        if not self.enabled:
            self.logger.info(f"[motion disabled] would run high-level action: {action}")
            return
        self._send({"cmd": "hl_action", "action": str(action)})

    def close(self) -> None:
        if self.proc is None:
            return
        self._send({"cmd": "close"})
        try:
            self.proc.wait(timeout=2.0)
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
            elif cmd == "stop":
                motion.stop_sequence()
            elif cmd == "hl_action":
                motion.play_hl_action(str(payload.get("action", "")))
            elif cmd == "close":
                break
            else:
                print(f"unknown command: {cmd}", flush=True)
    finally:
        motion.close()
    return 0


class ChatbotNode(Node):
    def __init__(self, args: argparse.Namespace, motion: MotionWorkerClient | None = None) -> None:
        super().__init__("ollama_ai_chatbot")
        self.args = args
        self.ollama = OllamaClient(args)
        self.speaker = Speaker(args, self.get_logger())
        self.motion = motion if motion is not None else MotionWorkerClient(args, self.get_logger())
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
        self.interrupt_event = threading.Event()
        self.last_unintelligible_ts = 0.0
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
        if self._is_stop_request(text):
            self.last_index = index
            self.last_text = text
            self._interrupt_now("audio")
            return
        if self._is_unintelligible_asr(text):
            self.last_index = index
            self.last_text = text
            self._handle_unintelligible(now)
            return
        if not self._should_answer(text, confidence, index, now):
            return
        self.last_index = index
        self.last_text = text
        threading.Thread(target=self._handle_text, args=(text, "audio"), daemon=True).start()

    def on_command(self, msg: String) -> None:
        payload = decode_payload(str(msg.data))
        text = compact_text(str(payload.get("text", payload.get("prompt", ""))))
        if text:
            if self._is_stop_request(text):
                self._interrupt_now("command")
                return
            if self._is_unintelligible_asr(text):
                self._handle_unintelligible(time.time())
                return
            threading.Thread(target=self._handle_text, args=(text, "command"), daemon=True).start()

    def _handle_text(self, text: str, source: str) -> None:
        if not self.busy_lock.acquire(blocking=False):
            if not self._is_stop_request(text):
                self.speaker.say_async("I am still finishing the previous answer.")
            return
        self.interrupt_event.clear()
        started = time.time()
        try:
            route = self._route(text)
            intent = str(route.get("intent", "chat")).lower()
            announce = compact_text(str(route.get("announce", "")))
            self._audit({"kind": "route", "source": source, "text": text, "route": route})
            if intent == "stop":
                self._interrupt_now(source)
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
            if intent == "gesture":
                action = self._hl_action_from_route(route)
                answer = announce or self._gesture_reply(action)
                if action:
                    self.motion.hl_action(action)
                self.speaker.say_async(answer)
                self._publish({"ok": True, "intent": intent, "answer": answer, "motion": action, "elapsed_s": time.time() - started})
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
        return {"intent": "chat", "announce": "", "needs_knowledge": False, "motion": "none"}

    def _route_fast(self, text: str) -> dict[str, Any] | None:
        low = text.lower().strip()
        if self._is_stop_request(text):
            return {"intent": "stop", "announce": "Stopping.", "needs_knowledge": False, "motion": "none"}
        if any(phrase in low for phrase in ("say thank you", "say thanks", "thank them", "thank everyone")):
            return {"intent": "thanks", "announce": "Thank you.", "needs_knowledge": False, "motion": "thanks"}
        if any(word in low for word in ("thank", "thanks", "danke")):
            return {"intent": "thanks", "announce": "You're welcome.", "needs_knowledge": False, "motion": "thanks"}
        if "clap" in low or "applaud" in low:
            return {"intent": "gesture", "announce": "I will clap.", "needs_knowledge": False, "motion": "clap"}
        if "shake hand" in low or "handshake" in low:
            return {"intent": "gesture", "announce": "Nice to meet you.", "needs_knowledge": False, "motion": "shake_hand"}
        if "high five" in low:
            return {"intent": "gesture", "announce": "High five.", "needs_knowledge": False, "motion": "high_five"}
        if "wave" in low or "hello" in low or "hi " in f"{low} " or "greet" in low:
            return {"intent": "gesture", "announce": "Hello.", "needs_knowledge": False, "motion": "face_wave"}
        question_mark = "?" in text or low.split(" ", 1)[0] in {"what", "why", "how", "when", "where", "who", "which"}
        if question_mark and self.retriever is not None:
            return {"intent": "rag_question", "announce": "Let me think.", "needs_knowledge": True, "motion": "thinking"}
        return None

    def _interrupt_now(self, source: str) -> None:
        self.interrupt_event.set()
        self.speaker.stop_current()
        self.motion.stop_sequence()
        self.last_reply = "Stopping."
        self.last_reply_ts = time.time()
        self._publish({"ok": True, "intent": "stop", "source": source, "answer": "Stopping.", "interrupted": True})

    def _handle_unintelligible(self, now: float) -> None:
        if now - self.last_unintelligible_ts < 3.0:
            return
        self.last_unintelligible_ts = now
        answer = "I wasn't able to understand that prompt."
        self.last_reply = answer
        self.last_reply_ts = time.time()
        self.speaker.say_async(answer)
        self._publish({"ok": False, "intent": "unintelligible_asr", "answer": answer})

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
        answer = self.ollama.chat(messages, temperature=float(self.args.temperature), num_predict=int(self.args.num_predict))
        return answer, bool(context)

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
