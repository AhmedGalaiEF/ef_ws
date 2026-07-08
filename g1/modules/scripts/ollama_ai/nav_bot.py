#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import json
import math
import os
import re
import string
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

if "--slam-worker" not in sys.argv:
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
    "navigation not ready",
    "what should i call this point",
    "point saved",
    "starting mapping",
    "stopping mapping",
    "relocating",
    "i did not understand that navigation command",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Voice-command SLAM navigation bot for Unitree G1 named points."
    )
    parser.add_argument("--iface", default=os.environ.get("G1_IFACE", "eth0"))
    parser.add_argument("--domain-id", type=int, default=int(os.environ.get("G1_DOMAIN_ID", "0")))
    parser.add_argument("--map-path", default="/home/unitree/test.pcd")
    parser.add_argument("--points-file", default=str(SCRIPT_DIR / "nav_points.json"))
    parser.add_argument("--slam-type", default="indoor")
    parser.add_argument("--audio-topic", default="/audio_msg")
    parser.add_argument("--filtered-audio-topic", default="/audio_msg/filter")
    parser.add_argument("--command-topic", default="/model_api/navbot_command")
    parser.add_argument("--response-topic", default="/model_api/navbot_response")
    parser.add_argument("--external-asr-server", action="store_true")
    parser.add_argument("--external-asr-host", default="0.0.0.0")
    parser.add_argument("--external-asr-port", type=int, default=8096)
    parser.add_argument("--external-asr-token", default="")
    parser.add_argument("--external-asr-only", "--no-ros-audio", dest="external_asr_only", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--post-speak-ignore-s", type=float, default=1.5)
    parser.add_argument("--auto-relocate", dest="auto_relocate", action="store_true", default=True)
    parser.add_argument("--no-auto-relocate", dest="auto_relocate", action="store_false")
    parser.add_argument("--wait-for-arrival", dest="wait_for_arrival", action="store_true", default=False)
    parser.add_argument("--no-wait-for-arrival", dest="wait_for_arrival", action="store_false")
    parser.add_argument("--arrival-timeout-s", type=float, default=NAV_TARGET_TIMEOUT_S)
    parser.add_argument("--volume", type=int, default=None)
    parser.add_argument("--tts-language", default=None)
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--startup-speech", default="navigation bot ready")
    parser.add_argument("--no-startup-speech", action="store_true")
    parser.add_argument("--audit-log", default="/tmp/nav_bot.jsonl")
    parser.add_argument("--slam-worker", default="", help=argparse.SUPPRESS)
    parser.add_argument("--point-json", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def compact_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def normalize_text(text: str) -> str:
    return compact_text(text).lower().strip(string.punctuation + "，。！？、；：")


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


class NavBotNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("g1_nav_bot")
        self.args = args
        self.nav = NavState(args)
        self.speaker = Speaker(args, self.get_logger())
        self.response_pub = self.create_publisher(String, args.response_topic, 10)
        self.audit_path = Path(args.audit_log).expanduser()
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_lock = threading.Lock()
        self.pending_add_point = False
        self.last_index: int | None = None
        self.last_text = ""
        self.last_reply = ""
        self.last_reply_ts = 0.0
        self.external_asr_httpd: http.server.ThreadingHTTPServer | None = None
        self.external_asr_thread: threading.Thread | None = None
        if not bool(args.external_asr_only):
            self.create_subscription(String, args.audio_topic, self.on_audio, 10)
            if str(args.filtered_audio_topic) and str(args.filtered_audio_topic) != str(args.audio_topic):
                self.create_subscription(String, args.filtered_audio_topic, self.on_audio, 10)
        self.create_subscription(String, args.command_topic, self.on_command, 10)
        if bool(args.external_asr_server):
            self._start_external_asr_server()
        self.get_logger().info(
            f"nav_bot ready audio={'external-only' if args.external_asr_only else args.audio_topic} "
            f"map={args.map_path} points={args.points_file}"
        )
        if not args.no_startup_speech and compact_text(args.startup_speech):
            self._say(args.startup_speech)

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
                result = self._dispatch(text)
                answer = str(result.pop("answer", "Done."))
                if not result.get("ok", True):
                    self.get_logger().warning(
                        f"nav command failed intent={result.get('intent')} code={result.get('code')} raw={result.get('raw')!r}"
                    )
                if answer:
                    self._say(answer)
                result.update({"source": source, "text": text, "answer": answer, "elapsed_s": round(time.time() - started, 3)})
                self._publish(result)
            except Exception as exc:
                self.get_logger().error(f"nav command failed: {exc}")
                answer = "Navigation command failed."
                self._say(answer)
                self._publish({"ok": False, "intent": "error", "source": source, "text": text, "answer": answer, "error": str(exc)})

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

        if "list" in low and "point" in low:
            names = sorted(self.nav.points)
            answer = "I do not have any saved points." if not names else "Saved points are " + ", ".join(names) + "."
            return {"ok": True, "code": 0, "intent": "list_points", "points": names, "answer": answer}

        if "status" in low:
            return {"ok": True, "code": 0, "intent": "status", "status": self.nav.status(), "answer": "Navigation status is available."}

        return {"ok": False, "code": 1, "intent": "unknown", "answer": ""}

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
            for keyword in ("add", "close", "go", "list", "map", "mapping", "navigate", "point", "relocate", "resume", "save", "slam", "start", "status", "stop")
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
                if self.path.split("?", 1)[0] == "/health":
                    self._send_json(200, {"ok": True, "service": "nav_bot"})
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


def main() -> int:
    args = parse_args()
    if args.slam_worker:
        return run_slam_worker(args)
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
