#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from sensor_msgs.msg import BatteryState
except Exception:  # pragma: no cover - lets this node run on minimal installs.
    BatteryState = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
MODULES_DIR = SCRIPT_DIR.parent.parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


DEFAULT_SYSTEM_PROMPT = (
    "You are the spoken AI interface for a Unitree G1 EDU robot. "
    "Be concise. Prefer safe diagnostic answers. "
    "For physical actions, choose only named skills and never raw joint commands."
)
ROUTER_PROMPT = (
    "Return only compact JSON. Schema: "
    "{\"route\":\"chat|diagnostic_query|skill_request\","
    "\"reply\":\"short text to say\","
    "\"tool_name\":\"optional diagnostic tool\","
    "\"skill_name\":\"optional skill\","
    "\"args\":{},"
    "\"gesture\":\"none|small_wave|open_hand|relax\"}. "
    "Allowed diagnostic tools: get_battery_status, get_cpu_memory, check_topic, "
    "check_service, get_available_skills, get_robot_state. "
    "Allowed skills: say, stop, hand_open, hand_close, reach_forward, move. "
    "Never invent a tool, skill, topic, service, or shell command."
)
FILLERS = {"ah", "eh", "er", "hmm", "hm", "mm", "uh", "um"}

ALLOWED_SERVICES = {
    "ollama": "ollama.service",
    "real_sense": "real-sense.service",
    "ros_sensors": "ros-sensors.service",
    "mode_control": "mode-control.service",
}
ALLOWED_TOPICS = {
    "battery": "/battery_state",
    "audio": "/audio_msg",
    "rgb": "/camera/color/image_raw",
    "depth": "/camera/depth/image_rect_raw",
    "segmentation": "/perception/overlay_image",
    "detections": "/perception/detections",
    "lidar": "/livox/points",
    "map": "/map",
    "tf": "/tf",
    "joint_states": "/joint_states",
    "brain_state": "/brain/state",
}
PHYSICAL_SKILLS = {"stop", "hand_open", "hand_close", "reach_forward", "move"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "General purpose Ollama AI node for chat, diagnostics, model-facing "
            "state topics, and allowlisted robot skill execution."
        )
    )
    parser.add_argument("--audio-topic", default="/audio_msg")
    parser.add_argument("--filtered-audio-topic", default="/audio_msg/filter")
    parser.add_argument("--command-topic", default="/model_api/command")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--router-model", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--num-predict", type=int, default=96)
    parser.add_argument("--num-ctx", type=int, default=1536)
    parser.add_argument("--keep-alive", default="15m")
    parser.add_argument("--num-thread", type=int, default=None)
    parser.add_argument("--max-history", type=int, default=6)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--post-speak-ignore-s", type=float, default=1.5)
    parser.add_argument("--answer-fillers", action="store_true")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--volume", type=int, default=None)
    parser.add_argument("--tts-language", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate commands without moving or speaking.")
    parser.add_argument("--enable-motion", action="store_true", help="Allow physical movement skills.")
    parser.add_argument("--allow-speech", action="store_true", help="Allow robot TTS responses.")
    parser.add_argument("--startup-speech", default="ollama ai ready")
    parser.add_argument("--no-startup-speech", action="store_true")
    parser.add_argument("--auth-token", default=os.environ.get("ROBOT_API_TOKEN", ""))
    parser.add_argument("--require-auth", action="store_true")
    parser.add_argument("--audit-log", default="/tmp/ollama_ai_audit.jsonl")
    parser.add_argument("--state-period", type=float, default=1.0)
    return parser.parse_args()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def compact_text(text: str) -> str:
    text = str(text).strip()
    while "<think>" in text and "</think>" in text:
        before, rest = text.split("<think>", 1)
        _hidden, after = rest.split("</think>", 1)
        text = before + after
    return " ".join(text.split())


def decode_jsonish(raw: str) -> dict[str, Any]:
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


def is_filler(text: str) -> bool:
    cleaned = text.lower().strip().strip(".,!?;: ")
    return cleaned in FILLERS


class OllamaClient:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.base_url = str(args.ollama_url).rstrip("/")

    def chat(
        self,
        messages: list[dict[str, Any]],
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
        if self.args.num_thread is not None:
            body["options"]["num_thread"] = int(self.args.num_thread)
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

    def warmup(self) -> None:
        self.chat(
            [
                {"role": "system", "content": "Answer with one word."},
                {"role": "user", "content": "Ready?"},
            ],
            temperature=0.0,
            num_predict=2,
            timeout=float(self.args.timeout),
        )


@dataclass
class Blackboard:
    battery_percent: float | None = None
    battery_voltage: float | None = None
    battery_current: float | None = None
    last_battery_ts: float = 0.0
    brain_state: str = "idle"
    last_command: dict[str, Any] = field(default_factory=dict)
    last_result: dict[str, Any] = field(default_factory=dict)


class DiagnosticRegistry:
    def __init__(self, node: Node, blackboard: Blackboard) -> None:
        self.node = node
        self.blackboard = blackboard
        self.tools: dict[str, Callable[..., dict[str, Any]]] = {
            "get_battery_status": self.get_battery_status,
            "get_cpu_memory": self.get_cpu_memory,
            "check_topic": self.check_topic,
            "check_service": self.check_service,
            "get_available_skills": self.get_available_skills,
            "get_robot_state": self.get_robot_state,
        }

    def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if name not in self.tools:
            return {"ok": False, "answer": f"Diagnostic tool {name} is not allowed."}
        try:
            return self.tools[name](**(args or {}))
        except Exception as exc:
            return {"ok": False, "answer": f"Diagnostic {name} failed: {exc}", "error": str(exc)}

    def get_battery_status(self) -> dict[str, Any]:
        pct = self.blackboard.battery_percent
        if pct is None:
            return {"ok": False, "answer": "I do not have battery data right now."}
        return {
            "ok": True,
            "battery_percent": round(float(pct), 1),
            "voltage": self.blackboard.battery_voltage,
            "current": self.blackboard.battery_current,
            "age_s": round(time.time() - self.blackboard.last_battery_ts, 2),
            "answer": f"My battery is at {pct:.0f} percent.",
        }

    def get_cpu_memory(self) -> dict[str, Any]:
        cpu = self._read_cpu_percent()
        mem = self._read_memory_percent()
        return {
            "ok": True,
            "cpu_percent": cpu,
            "memory_percent": mem,
            "answer": f"CPU usage is {cpu:.0f} percent and memory usage is {mem:.0f} percent.",
        }

    def check_topic(self, topic: str = "", name: str = "") -> dict[str, Any]:
        topic_name = self._allowed_topic(topic or name)
        if not topic_name:
            return {"ok": False, "answer": "That topic is not in my diagnostic allowlist."}
        result = self._run_fixed(["ros2", "topic", "list"], timeout_s=2.5)
        if not result["ok"]:
            return {**result, "answer": f"I could not list ROS topics: {result.get('error', 'unknown error')}"}
        topics = set(str(result["output"]).splitlines())
        exists = topic_name in topics
        return {
            "ok": exists,
            "topic": topic_name,
            "answer": f"The topic {topic_name} is active." if exists else f"I do not see {topic_name}.",
        }

    def check_service(self, service: str = "", name: str = "") -> dict[str, Any]:
        service_key = str(service or name).strip().lower()
        unit = ALLOWED_SERVICES.get(service_key)
        if not unit:
            return {"ok": False, "answer": "That service is not in my diagnostic allowlist."}
        result = self._run_fixed(["systemctl", "is-active", unit], timeout_s=2.0)
        status = str(result.get("output", "")).strip() or "unknown"
        return {
            "ok": status == "active",
            "service": unit,
            "status": status,
            "answer": f"The service {unit} is {status}.",
        }

    def get_available_skills(self) -> dict[str, Any]:
        skills = [
            {"name": "say", "risk": "none", "requires_motion": False},
            {"name": "stop", "risk": "low", "requires_motion": True},
            {"name": "hand_open", "risk": "low", "requires_motion": True},
            {"name": "hand_close", "risk": "medium", "requires_motion": True},
            {"name": "reach_forward", "risk": "medium", "requires_motion": True},
            {"name": "move", "risk": "medium", "requires_motion": True},
        ]
        return {"ok": True, "skills": skills, "answer": "I can speak, diagnose, stop, move, reach, and control the hands."}

    def get_robot_state(self) -> dict[str, Any]:
        state = {
            "brain_state": self.blackboard.brain_state,
            "battery_percent": self.blackboard.battery_percent,
            "last_command": self.blackboard.last_command,
            "last_result": self.blackboard.last_result,
        }
        return {"ok": True, "state": state, "answer": f"My current state is {self.blackboard.brain_state}."}

    @staticmethod
    def _allowed_topic(value: str) -> str | None:
        value = str(value).strip()
        if value in ALLOWED_TOPICS.values():
            return value
        return ALLOWED_TOPICS.get(value.lower())

    @staticmethod
    def _run_fixed(argv: list[str], timeout_s: float) -> dict[str, Any]:
        started = time.time()
        try:
            proc = subprocess.run(
                argv,
                shell=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=float(timeout_s),
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout if isinstance(exc.stdout, str) else ""
            return {"ok": False, "elapsed_s": round(time.time() - started, 3), "output": output, "error": "timeout"}
        return {
            "ok": proc.returncode == 0,
            "command": " ".join(shlex.quote(part) for part in argv),
            "returncode": int(proc.returncode),
            "elapsed_s": round(time.time() - started, 3),
            "output": proc.stdout or "",
        }

    @staticmethod
    def _read_cpu_percent() -> float:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                first = [float(v) for v in f.readline().split()[1:8]]
            time.sleep(0.1)
            with open("/proc/stat", "r", encoding="utf-8") as f:
                second = [float(v) for v in f.readline().split()[1:8]]
            idle = (second[3] + second[4]) - (first[3] + first[4])
            total = sum(second) - sum(first)
            return round(100.0 * (1.0 - idle / total), 1) if total > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _read_memory_percent() -> float:
        values: dict[str, float] = {}
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    key, raw = line.split(":", 1)
                    values[key] = float(raw.split()[0])
            total = values.get("MemTotal", 0.0)
            avail = values.get("MemAvailable", 0.0)
            return round(100.0 * (1.0 - avail / total), 1) if total > 0 else 0.0
        except Exception:
            return 0.0


class SkillExecutor:
    def __init__(self, args: argparse.Namespace, node: Node) -> None:
        self.args = args
        self.node = node
        self._robot: Any = None
        self._lock = threading.Lock()
        self._last_motion_ts = 0.0
        self.skills: dict[str, Callable[..., dict[str, Any]]] = {
            "say": self.say,
            "stop": self.stop,
            "hand_open": self.hand_open,
            "hand_close": self.hand_close,
            "reach_forward": self.reach_forward,
            "move": self.move,
        }

    def execute(self, name: str, args: dict[str, Any] | None, *, token: str = "", dry_run: bool = False) -> dict[str, Any]:
        skill = str(name).strip().lower()
        payload = args or {}
        auth_ok, auth_reason = self._authorize(skill, token)
        if not auth_ok:
            return {"ok": False, "skill": skill, "answer": auth_reason, "authorized": False}
        if skill not in self.skills:
            return {"ok": False, "skill": skill, "answer": f"Skill {skill} is not allowed."}
        if dry_run or self.args.dry_run:
            return {"ok": True, "skill": skill, "dry_run": True, "args": self._sanitize_args(payload)}
        with self._lock:
            return self.skills[skill](**payload)

    def say(self, text: str = "", **_: Any) -> dict[str, Any]:
        speech = compact_text(text)[:300]
        if not speech:
            return {"ok": False, "answer": "No speech text was provided."}
        if not self.args.allow_speech:
            return {"ok": True, "speech_allowed": False, "answer": speech}
        code = self._say_with_existing_script(speech)
        return {"ok": int(code) == 0, "returncode": int(code), "answer": speech}

    def stop(self, **_: Any) -> dict[str, Any]:
        if not self.args.enable_motion:
            return {"ok": True, "answer": "Motion is disabled, so there was nothing to stop."}
        self._get_robot().stop()
        return {"ok": True, "answer": "Stopped."}

    def hand_open(self, hand: str = "right", **_: Any) -> dict[str, Any]:
        self._motion_gate("hand_open")
        side = self._side(hand)
        self._get_robot().hand_open(side, hold_s=0.5)
        return {"ok": True, "answer": f"Opened {side} hand."}

    def hand_close(self, hand: str = "right", **_: Any) -> dict[str, Any]:
        self._motion_gate("hand_close")
        side = self._side(hand)
        self._get_robot().hand_close(side, hold_s=0.5)
        return {"ok": True, "answer": f"Closed {side} hand."}

    def reach_forward(self, arm: str = "right", duration_s: float = 3.0, **_: Any) -> dict[str, Any]:
        self._motion_gate("reach_forward")
        side = self._side(arm)
        result = self._get_robot().extend_arm_forward(arm=side, duration_s=clamp(duration_s, 0.8, 6.0))
        return {"ok": True, "result": result, "answer": f"Reached forward with my {side} arm."}

    def move(
        self,
        direction: str = "stop",
        distance_m: float = 0.25,
        yaw_rad: float = 0.0,
        speed_mps: float = 0.2,
        **_: Any,
    ) -> dict[str, Any]:
        self._motion_gate("move")
        direction = str(direction).strip().lower()
        if direction == "stop":
            self._get_robot().stop()
            return {"ok": True, "answer": "Stopped."}
        if direction not in {"forward", "backward", "left", "right"}:
            return {"ok": False, "answer": "Direction must be forward, backward, left, right, or stop."}
        speed = clamp(speed_mps, 0.05, 0.35)
        distance = clamp(distance_m, 0.0, 0.75)
        duration = distance / speed if speed > 0 else 0.0
        vx = speed if direction == "forward" else -speed if direction == "backward" else 0.0
        vy = speed if direction == "left" else -speed if direction == "right" else 0.0
        vyaw = clamp(yaw_rad, -0.8, 0.8) / duration if duration > 0.05 else 0.0
        self._get_robot().move_for(duration, vx=vx, vy=vy, vyaw=vyaw)
        return {"ok": True, "answer": f"Moved {direction} {distance:.2f} meters."}

    def gesture(self, name: str) -> dict[str, Any]:
        gesture = str(name).strip().lower()
        if gesture in {"", "none"}:
            return {"ok": True, "answer": "No gesture."}
        if gesture == "open_hand":
            return self.execute("hand_open", {"hand": "right"}, token=self.args.auth_token)
        if gesture == "relax":
            return self.execute("hand_open", {"hand": "right"}, token=self.args.auth_token)
        if gesture == "small_wave":
            if not self.args.enable_motion:
                return {"ok": False, "answer": "Motion is disabled."}
            self.hand_open("right")
            self.reach_forward("right", duration_s=1.0)
            self.hand_open("right")
            return {"ok": True, "answer": "Small wave gesture completed."}
        return {"ok": False, "answer": f"Gesture {gesture} is not allowed."}

    def _authorize(self, skill: str, token: str) -> tuple[bool, str]:
        if skill in PHYSICAL_SKILLS and skill != "stop" and not self.args.enable_motion:
            return False, "Motion skills are disabled."
        if self.args.require_auth and not self._token_matches(token):
            return False, "Unauthorized robot command."
        return True, "ok"

    def _token_matches(self, token: str) -> bool:
        expected = str(self.args.auth_token or "")
        if not expected:
            return False
        return hmac.compare_digest(str(token), expected)

    def _motion_gate(self, skill: str) -> None:
        if skill in PHYSICAL_SKILLS and skill != "stop" and not self.args.enable_motion:
            raise RuntimeError("Motion skills are disabled.")
        now = time.time()
        if now - self._last_motion_ts < 0.5:
            raise RuntimeError("Motion command rate limit is active.")
        self._last_motion_ts = now

    def _get_robot(self) -> Any:
        if self._robot is None:
            from sdk_client import Robot
            self._robot = Robot(
                iface=self.args.iface,
                domain_id=self.args.domain_id,
                safety_boot=False,
                recover_dev_mode_on_init=False,
                auto_start_sensors=False,
                ollama_url=self.args.ollama_url,
                chat_model=self.args.model,
            )
        return self._robot

    def _say_with_existing_script(self, speech: str) -> int:
        script = SCRIPTS_DIR / "robot_say_once.py"
        command = [
            sys.executable,
            str(script),
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
        try:
            proc = subprocess.run(
                command,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30.0,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout if isinstance(exc.stdout, str) else ""
            if output.strip():
                self.node.get_logger().warning(output.strip())
            return 124
        output = (proc.stdout or "").strip()
        if output:
            self.node.get_logger().info(output)
        return int(proc.returncode)

    @staticmethod
    def _side(value: str) -> str:
        side = str(value).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("hand/arm must be left or right")
        return side

    @staticmethod
    def _sanitize_args(args: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe[str(key)] = value
        return safe


class ModelApiNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("ollama_ai_model_api")
        self.args = args
        self.ollama = OllamaClient(args)
        self.blackboard = Blackboard()
        self.diagnostics = DiagnosticRegistry(self, self.blackboard)
        self.skills = SkillExecutor(args, self)
        self.audit_path = Path(args.audit_log).expanduser()
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, str]] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        self.last_audio_index: int | None = None
        self.last_audio_text = ""
        self.last_reply = ""
        self.last_reply_ts = 0.0

        self.response_pub = self.create_publisher(String, "/model_api/response", 10)
        self.brain_state_pub = self.create_publisher(String, "/brain/state", 10)
        self.robot_state_pub = self.create_publisher(String, "/model_api/robot_state", 10)
        self.skills_pub = self.create_publisher(String, "/model_api/available_skills", 10)
        self.audit_pub = self.create_publisher(String, "/model_api/audit", 10)
        self.create_subscription(String, args.audio_topic, self.on_audio, 10)
        if str(args.filtered_audio_topic) and str(args.filtered_audio_topic) != str(args.audio_topic):
            self.create_subscription(String, args.filtered_audio_topic, self.on_audio, 10)
        self.create_subscription(String, args.command_topic, self.on_command, 10)
        if BatteryState is not None:
            self.create_subscription(BatteryState, "/battery_state", self.on_battery, 10)
        self.create_timer(max(0.2, float(args.state_period)), self.publish_state)

        self.get_logger().info(
            f"ollama_ai ready: audio={args.audio_topic} command={args.command_topic} "
            f"model={args.model} motion={'on' if args.enable_motion else 'off'}"
        )
        if not args.no_warmup:
            try:
                started = time.time()
                self.ollama.warmup()
                self.get_logger().info(f"Ollama warmup finished in {time.time() - started:.1f}s")
            except Exception as exc:
                self.get_logger().warning(f"Ollama warmup failed: {exc}")
        if args.allow_speech and not args.no_startup_speech and compact_text(args.startup_speech):
            try:
                result = self.skills.execute("say", {"text": args.startup_speech}, token=args.auth_token)
            except Exception as exc:
                result = {"ok": False, "answer": f"Startup speech failed: {exc}", "error": str(exc)}
                self.get_logger().error(str(result["answer"]))
            self._audit({"kind": "startup_speech", "result": result})

    def on_battery(self, msg: Any) -> None:
        percentage = float(getattr(msg, "percentage", 0.0) or 0.0)
        self.blackboard.battery_percent = percentage * 100.0 if percentage <= 1.0 else percentage
        self.blackboard.battery_voltage = round(float(getattr(msg, "voltage", 0.0) or 0.0), 2)
        self.blackboard.battery_current = round(float(getattr(msg, "current", 0.0) or 0.0), 2)
        self.blackboard.last_battery_ts = time.time()

    def on_audio(self, msg: String) -> None:
        payload = decode_jsonish(str(msg.data))
        text = compact_text(str(payload.get("text", payload.get("raw", ""))))
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        index = self._payload_index(payload)
        now = time.time()
        if not self._should_answer(text, confidence, index, now):
            return
        self.get_logger().info(f'Heard: "{text}" confidence={confidence:.2f} index={index}')
        self.last_audio_index = index
        self.last_audio_text = text
        self._handle_user_text(text, source="audio", request_id=str(payload.get("index", "")))

    def on_command(self, msg: String) -> None:
        payload = decode_jsonish(str(msg.data))
        request_id = str(payload.get("request_id") or self._request_id(str(msg.data)))
        token = str(payload.get("auth_token", ""))
        dry_run = bool(payload.get("dry_run", False))
        if "skill_name" in payload:
            result = self._execute_skill_payload(payload, token=token, dry_run=dry_run, request_id=request_id)
            self._publish_response(result)
            return
        if "tool_name" in payload:
            result = self._diagnostic_payload(payload, request_id=request_id)
            self._publish_response(result)
            return
        text = compact_text(str(payload.get("text", "")))
        if text:
            self._handle_user_text(text, source="command", request_id=request_id, token=token, dry_run=dry_run)

    def _handle_user_text(
        self,
        text: str,
        *,
        source: str,
        request_id: str,
        token: str = "",
        dry_run: bool = False,
    ) -> None:
        self._set_brain_state("thinking")
        started = time.time()
        try:
            route = self._route_fast(text) or self._route_with_ollama(text)
            result = self._run_route(route, token=token, dry_run=dry_run, request_id=request_id)
        except Exception as exc:
            result = {"ok": False, "route": "error", "answer": f"I hit an error: {exc}", "error": str(exc)}
        result.update({"source": source, "request_id": request_id, "elapsed_s": round(time.time() - started, 3)})
        self.blackboard.last_result = result
        self._publish_response(result)
        answer = compact_text(str(result.get("answer", "")))
        if answer and self.args.allow_speech:
            speech_result = self._speak_with_optional_gesture(answer, str(result.get("gesture", "none")))
            self._audit({"kind": "speech", "text": answer, "result": speech_result})
        self.last_reply = answer
        self.last_reply_ts = time.time()
        self._set_brain_state("idle")

    def _route_fast(self, text: str) -> dict[str, Any] | None:
        low = text.lower()
        if any(word in low for word in ("battery", "charge")):
            return {"route": "diagnostic_query", "tool_name": "get_battery_status", "args": {}}
        if "cpu" in low or "memory" in low:
            return {"route": "diagnostic_query", "tool_name": "get_cpu_memory", "args": {}}
        if "ollama" in low and any(word in low for word in ("running", "active", "status")):
            return {"route": "diagnostic_query", "tool_name": "check_service", "args": {"name": "ollama"}}
        if "camera" in low and any(word in low for word in ("see", "active", "running")):
            return {"route": "diagnostic_query", "tool_name": "check_topic", "args": {"name": "rgb"}}
        if "lidar" in low:
            return {"route": "diagnostic_query", "tool_name": "check_topic", "args": {"name": "lidar"}}
        if low.strip() in {"stop", "stop moving", "halt"}:
            return {"route": "skill_request", "skill_name": "stop", "args": {}, "reply": "Stopping."}
        return None

    def _route_with_ollama(self, text: str) -> dict[str, Any]:
        model = str(self.args.router_model or self.args.model)
        messages = [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": text},
        ]
        raw = self.ollama.chat(messages, model=model, temperature=0.0, num_predict=120)
        route = extract_json_object(raw)
        if not route:
            return {"route": "chat", "reply": raw}
        return route

    def _run_route(self, route: dict[str, Any], *, token: str, dry_run: bool, request_id: str) -> dict[str, Any]:
        route_name = str(route.get("route", "chat")).strip().lower()
        self.blackboard.last_command = route
        if route_name == "diagnostic_query":
            result = self.diagnostics.call(str(route.get("tool_name", "")), self._dict(route.get("args")))
            return {**result, "route": route_name, "tool_name": route.get("tool_name"), "gesture": route.get("gesture", "none")}
        if route_name == "skill_request":
            result = self.skills.execute(
                str(route.get("skill_name", "")),
                self._dict(route.get("args")),
                token=token,
                dry_run=dry_run,
            )
            answer = compact_text(str(route.get("reply") or result.get("answer", "")))
            return {**result, "route": route_name, "request_id": request_id, "answer": answer, "gesture": route.get("gesture", "none")}
        return self._chat_answer(str(route.get("reply") or route.get("text") or ""))

    def _chat_answer(self, initial_reply: str) -> dict[str, Any]:
        if initial_reply and len(initial_reply.split()) <= 35:
            answer = initial_reply
        else:
            messages = self.history[-max(1, int(self.args.max_history)):]
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
            if initial_reply:
                messages.append({"role": "user", "content": initial_reply})
            answer = self.ollama.chat(messages, temperature=float(self.args.temperature), num_predict=int(self.args.num_predict))
        answer = compact_text(answer) or "I heard you, but I am not sure how to answer."
        self.history.append({"role": "assistant", "content": answer})
        self.history = self.history[-(int(self.args.max_history) + 1):]
        return {"ok": True, "route": "chat", "answer": answer, "gesture": "none"}

    def _execute_skill_payload(self, payload: dict[str, Any], *, token: str, dry_run: bool, request_id: str) -> dict[str, Any]:
        self._set_brain_state("acting")
        result = self.skills.execute(
            str(payload.get("skill_name", "")),
            self._dict(payload.get("args")),
            token=token,
            dry_run=dry_run,
        )
        result["request_id"] = request_id
        self._audit({"kind": "skill", "request": payload, "result": result})
        self._set_brain_state("idle")
        return result

    def _diagnostic_payload(self, payload: dict[str, Any], *, request_id: str) -> dict[str, Any]:
        self._set_brain_state("thinking")
        result = self.diagnostics.call(str(payload.get("tool_name", "")), self._dict(payload.get("args")))
        result["request_id"] = request_id
        self._audit({"kind": "diagnostic", "request": payload, "result": result})
        self._set_brain_state("idle")
        return result

    def _speak_with_optional_gesture(self, answer: str, gesture: str) -> dict[str, Any]:
        self._set_brain_state("acting")
        gesture_thread: threading.Thread | None = None
        if gesture and gesture != "none" and self.args.enable_motion:
            gesture_thread = threading.Thread(target=self.skills.gesture, args=(gesture,), daemon=True)
            gesture_thread.start()
        try:
            result = self.skills.execute("say", {"text": answer}, token=self.args.auth_token, dry_run=False)
        except Exception as exc:
            result = {"ok": False, "answer": f"Speech failed: {exc}", "error": str(exc)}
            self.get_logger().error(str(result["answer"]))
        finally:
            if gesture_thread is not None:
                gesture_thread.join(timeout=0.2)
        return result

    def publish_state(self) -> None:
        self.skills_pub.publish(String(data=json.dumps(self.diagnostics.get_available_skills(), sort_keys=True)))
        state = self.diagnostics.get_robot_state()
        self.robot_state_pub.publish(String(data=json.dumps(state, sort_keys=True, default=str)))
        self.brain_state_pub.publish(String(data=self.blackboard.brain_state))

    def _publish_response(self, result: dict[str, Any]) -> None:
        self.response_pub.publish(String(data=json.dumps(result, sort_keys=True, default=str)))
        self._audit({"kind": "response", "result": result})

    def _set_brain_state(self, state: str) -> None:
        self.blackboard.brain_state = state
        self.brain_state_pub.publish(String(data=state))

    def _audit(self, record: dict[str, Any]) -> None:
        record = {"time": time.time(), **record}
        line = json.dumps(record, sort_keys=True, default=str)
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.audit_pub.publish(String(data=line))

    def _should_answer(self, text: str, confidence: float, index: int | None, received_at: float) -> bool:
        if not text or confidence < float(self.args.min_confidence):
            return False
        if not self.args.answer_fillers and is_filler(text):
            return False
        if not any(char.isalnum() for char in text):
            return False
        if index is not None and index == self.last_audio_index:
            return False
        if index is None and text == self.last_audio_text and received_at - self.last_reply_ts < 2.0:
            return False
        if received_at - self.last_reply_ts < float(self.args.post_speak_ignore_s):
            return False
        if self.last_reply and SequenceMatcher(None, text.lower(), self.last_reply.lower()).ratio() >= 0.82:
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
    def _dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _request_id(text: str) -> str:
        return hashlib.sha256(f"{time.time()}:{text}".encode("utf-8")).hexdigest()[:16]


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = ModelApiNode(args)
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
