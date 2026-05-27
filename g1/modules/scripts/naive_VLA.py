#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import difflib
import html
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sdk_client import Robot

try:
    import cv2
    import numpy as np
    import zmq
except Exception as exc:  # pragma: no cover - reported at runtime.
    raise SystemExit(f"naive_VLA.py requires cv2, numpy, and pyzmq: {exc}") from exc


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise vision-language-action helper for a Unitree humanoid. "
    "Describe only visible context and practical robot-relevant affordances. "
    "Do not invent objects that are not visible."
)

DIAGNOSTIC_COMMANDS: dict[str, list[str]] = {
    "date": ["date"],
    "df": ["df", "-h"],
    "du": ["du", "-sh"],
    "free": ["free", "-h"],
    "hostname": ["hostname"],
    "ip_addr": ["ip", "-br", "addr"],
    "ip_route": ["ip", "route"],
    "journal_errors": ["journalctl", "-p", "3", "-n"],
    "lsusb": ["lsusb"],
    "ps": ["ps", "aux"],
    "ss_listen": ["ss", "-ltnp"],
    "systemctl_failed": ["systemctl", "--failed", "--no-pager"],
    "uptime": ["uptime"],
    "uname": ["uname", "-a"],
}

DIAGNOSTIC_PATH_COMMANDS = {"du"}
SYSTEM_DIAGNOSTIC_COMMANDS = (
    "date",
    "hostname",
    "uptime",
    "uname",
    "free",
    "df",
    "ip_addr",
    "ip_route",
    "ss_listen",
    "lsusb",
    "systemctl_failed",
    "journal_errors",
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
    "ok",
    "okay",
    "yes",
    "yeah",
    "what",
    "and",
    "i",
    "嗯",
    "嗯嗯",
    "呃",
    "啊",
    "う",
}
SPEECH_ECHO_PHRASES = (
    "i heard you",
    "i do not know how",
    "do not know how",
    "opened right hand",
    "opened left hand",
    "closed right hand",
    "closed left hand",
    "moved right end effector",
    "moved left end effector",
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def json_dumps(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode_audio_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"raw": raw, "text": raw}


def normalize_prompt(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def strip_prompt_punctuation(text: str) -> str:
    return str(text).strip(" \t\r\n.,!?;:，。！？；：、\"'")


def robot_say_once_script() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_say_once.py")


@dataclass
class RgbdSnapshot:
    rgb_jpeg: bytes | None
    depth_jpeg: bytes | None
    timestamp: float
    width: int
    height: int
    center_depth_m: float | None
    near_coverage_1m: float | None
    valid_depth_fraction: float
    depth_min_m: float | None
    depth_median_m: float | None
    depth_max_m: float | None
    error: str | None

    def to_context(self) -> dict[str, Any]:
        age_s = None if self.timestamp <= 0 else max(0.0, time.time() - self.timestamp)
        return {
            "timestamp": self.timestamp if self.timestamp > 0 else None,
            "age_s": age_s,
            "width": self.width,
            "height": self.height,
            "center_depth_m": self.center_depth_m,
            "near_coverage_1m": self.near_coverage_1m,
            "valid_depth_fraction": self.valid_depth_fraction,
            "depth_min_m": self.depth_min_m,
            "depth_median_m": self.depth_median_m,
            "depth_max_m": self.depth_max_m,
            "error": self.error,
        }


class RgbdReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        topic: str = "",
        *,
        max_depth_m: float = 4.0,
        fps: float = 12.0,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.topic = str(topic)
        self.max_depth_m = float(max_depth_m)
        self.fps = max(1.0, float(fps))
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._snapshot = RgbdSnapshot(
            rgb_jpeg=None,
            depth_jpeg=None,
            timestamp=0.0,
            width=0,
            height=0,
            center_depth_m=None,
            near_coverage_1m=None,
            valid_depth_fraction=0.0,
            depth_min_m=None,
            depth_median_m=None,
            depth_max_m=None,
            error="RGBD receiver not started.",
        )

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._set_error(f"Waiting for RGBD frames on {self.endpoint}")
        self._running = True
        self._thread = threading.Thread(target=self._run, name="rgbd-receiver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def snapshot(self) -> RgbdSnapshot:
        with self._lock:
            return RgbdSnapshot(**self._snapshot.__dict__)

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._snapshot.error = message

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, self.topic.encode("utf-8"))
        socket.setsockopt(zmq.RCVTIMEO, 500)
        socket.connect(self.endpoint)
        min_dt = 1.0 / self.fps
        last_update = 0.0
        try:
            while self._running:
                try:
                    parts = socket.recv_multipart()
                except zmq.Again:
                    if self.snapshot().timestamp <= 0:
                        self._set_error(f"Waiting for RGBD frames on {self.endpoint}")
                    continue
                except Exception as exc:
                    self._set_error(f"RGBD receive error: {exc}")
                    time.sleep(0.25)
                    continue

                if len(parts) >= 4:
                    parts = parts[-3:]
                if len(parts) < 2:
                    continue
                now = time.time()
                if now - last_update < min_dt:
                    continue
                last_update = now

                decoded = self._decode(parts)
                if decoded is None:
                    continue
                with self._lock:
                    self._snapshot = decoded
        finally:
            socket.close(0)
            context.term()

    def _decode(self, parts: list[bytes]) -> RgbdSnapshot | None:
        rgb_jpeg = bytes(parts[0])
        depth_png = bytes(parts[1])
        depth_scale = 0.001
        if len(parts) >= 3 and len(parts[2]) >= 4:
            try:
                depth_scale = float(struct.unpack("f", parts[2][:4])[0])
            except Exception:
                depth_scale = 0.001

        rgb = cv2.imdecode(np.frombuffer(rgb_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        depth_raw = cv2.imdecode(np.frombuffer(depth_png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth_raw is None:
            return None
        if depth_raw.ndim == 3:
            depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)

        depth_m = depth_raw.astype(np.float32) * float(depth_scale)
        valid = depth_raw > 0
        h, w = depth_raw.shape[:2]
        center_size = max(8, min(w, h) // 12)
        cx = w // 2
        cy = h // 2
        center = depth_m[
            max(0, cy - center_size) : min(h, cy + center_size),
            max(0, cx - center_size) : min(w, cx + center_size),
        ]
        roi = depth_m[int(h * 0.25) : int(h * 0.70), int(w * 0.30) : int(w * 0.70)]
        center_valid = center[center > 0]
        valid_values = depth_m[valid]

        depth_norm = np.zeros(depth_raw.shape, dtype=np.uint8)
        depth_norm[valid] = np.clip((depth_m[valid] / self.max_depth_m) * 255.0, 0, 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
        depth_vis[~valid] = (0, 0, 0)
        ok, depth_enc = cv2.imencode(".jpg", depth_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return None

        return RgbdSnapshot(
            rgb_jpeg=rgb_jpeg,
            depth_jpeg=depth_enc.tobytes(),
            timestamp=time.time(),
            width=int(w),
            height=int(h),
            center_depth_m=float(np.median(center_valid)) if center_valid.size else None,
            near_coverage_1m=float(np.mean((roi > 0) & (roi <= 1.0))) if roi.size else None,
            valid_depth_fraction=float(valid.mean()) if valid.size else 0.0,
            depth_min_m=float(np.min(valid_values)) if valid_values.size else None,
            depth_median_m=float(np.median(valid_values)) if valid_values.size else None,
            depth_max_m=float(np.max(valid_values)) if valid_values.size else None,
            error=None,
        )


class NaiveVLA:
    def __init__(
        self,
        receiver: RgbdReceiver,
        *,
        iface: str = "eth0",
        domain_id: int = 0,
        dry_run: bool = False,
        ollama_url: str = "http://127.0.0.1:11434",
        vision_model: str = "qwen2.5vl:7b",
        text_model: str = "qwen3.5:9b",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        allow_diagnostics: bool = True,
        allow_web_search: bool = True,
        speak_answers: bool = False,
        speech_volume: int | None = None,
        speech_language: str | None = None,
        hand_type: str = "dex3",
    ) -> None:
        self.receiver = receiver
        self.iface = "eth0"
        self.domain_id = 0
        self.dry_run = bool(dry_run)
        self.ollama_url = str(ollama_url).rstrip("/")
        self.vision_model = str(vision_model)
        self.text_model = str(text_model)
        self.system_prompt = str(system_prompt)
        self.allow_diagnostics = bool(allow_diagnostics)
        self.allow_web_search = bool(allow_web_search)
        self.speak_answers = bool(speak_answers)
        self.speech_volume = None if speech_volume is None else int(clamp(speech_volume, 0, 100))
        self.speech_language = speech_language
        self.hand_type = self._normalize_hand_type(hand_type)
        self._robot: Robot | None = None
        self._arm_sdk: Any = None
        self._lock = threading.Lock()

    def preinitialize_robot_sdk(self, *, arm: bool = True) -> dict[str, Any]:
        status: dict[str, Any] = {"ok": True, "iface": self.iface, "domain_id": self.domain_id}
        if self.dry_run:
            status["dry_run"] = True
            return status
        try:
            self._get_robot()
            status["robot"] = {"ok": True}
        except Exception as exc:
            status["ok"] = False
            status["robot"] = {"ok": False, "error": str(exc)}
        if arm:
            try:
                self._get_arm_sdk()
                status["arm_sdk"] = {"ok": True}
            except Exception as exc:
                status["ok"] = False
                status["arm_sdk"] = {"ok": False, "error": str(exc)}
        return status

    def tools(self) -> dict[str, Callable[..., Any]]:
        return {
            "get_visual_context": self.get_visual_context,
            "describe_visual_context": self.describe_visual_context,
            "speak": self.speak,
            "say": self.speak,
            "run_prompt": self.run_prompt,
            "set_hand_type": self.set_hand_type,
            "get_hand_type": self.get_hand_type,
            "hand_open": self.hand_open,
            "hand_close": self.hand_close,
            "loco_move": self.loco_move,
            "move_for": self.move_for,
            "stop": self.stop,
            "hand_shake": self.hand_shake,
            "shake_hand": self.hand_shake,
            "release_arms": self.release_arms,
            "ik_move_ee_pose": self.ik_move_ee_pose,
            "run_diagnostic_command": self.run_diagnostic_command,
            "diagnose_system": self.diagnose_system,
            "web_search": self.web_search,
        }

    def run_prompt(self, prompt: str, *, speak: bool = True) -> dict[str, Any]:
        text = " ".join(str(prompt).split())
        if not text:
            raise ValueError("prompt is required")
        normalized = normalize_prompt(text)
        result: dict[str, Any] = {"ok": True, "prompt": text, "matched": None}

        if self._prompt_requests_visual_description(normalized):
            desc = self.describe_visual_context(prompt=text, speak=bool(speak))
            result.update({"matched": "describe_visual_context", "result": desc})
            return result

        if self._prompt_requests_system_diagnosis(normalized):
            diag = self.diagnose_system(speak=bool(speak))
            result.update({"matched": "diagnose_system", "result": diag, "speech": diag.get("speech")})
            return result

        fuzzy_intent = self._fuzzy_builtin_intent(normalized)
        if fuzzy_intent is not None:
            return self._run_intent(text, fuzzy_intent, speak=bool(speak), source="fuzzy")

        hand_match = re.search(r"\b(open|upen|close|clothes|those)\s+(?:the\s+)?(left|right|both)\s+hands?\b", normalized)
        if hand_match is None:
            hand_match = re.search(r"\b(left|right|both)\s+hands?\s+(open|upen|close|clothes|those)\b", normalized)
            if hand_match is not None:
                hand = hand_match.group(1)
                action = hand_match.group(2)
            else:
                hand = action = ""
        else:
            action = hand_match.group(1)
            hand = hand_match.group(2)
        if action == "upen":
            action = "open"
        if action in {"clothes", "those"}:
            action = "close"
        if action in {"open", "close"} and hand:
            tool_result = self.hand_open(hand) if action == "open" else self.hand_close(hand)
            reply = f"{action.capitalize()}ed {hand} hand."
            speech = self._speak_feedback(reply, speak)
            result.update({"matched": f"hand_{action}", "result": tool_result, "speech": speech})
            return result

        if "shake" in normalized or "takeake" in normalized:
            tool_result = self.hand_shake()
            speech = self._speak_feedback("Shaking hand.", speak)
            result.update({"matched": "hand_shake", "result": tool_result, "speech": speech})
            return result

        ee = self._parse_ee_prompt(normalized)
        if ee is not None:
            tool_result = self.ik_move_ee_pose(**ee)
            axis = next((key[1:] for key in ("dx", "dy", "dz") if abs(float(ee[key])) > 0), "pose")
            reply = f"Moved {ee['arm']} end effector {axis} by {abs(next(float(ee[k]) for k in ('dx', 'dy', 'dz') if abs(float(ee[k])) > 0)):.2f} meters."
            speech = self._speak_feedback(reply, speak)
            result.update({"matched": "ik_move_ee_pose", "args": ee, "result": tool_result, "speech": speech})
            return result

        loco = self._parse_loco_prompt(normalized)
        if loco is not None:
            tool_result = self.move_for(**loco)
            speech = self._speak_feedback("Moving.", speak)
            result.update({"matched": "move_for", "args": loco, "result": tool_result, "speech": speech})
            return result

        if "release arm" in normalized or "release arms" in normalized:
            tool_result = self.release_arms()
            speech = self._speak_feedback("Released arms.", speak)
            result.update({"matched": "release_arms", "result": tool_result, "speech": speech})
            return result

        if "stop" in normalized:
            tool_result = self.stop()
            speech = self._speak_feedback("Stopped.", speak)
            result.update({"matched": "stop", "result": tool_result, "speech": speech})
            return result

        ollama_intent = self._interpret_prompt_with_ollama(text)
        if ollama_intent is not None:
            return self._run_intent(text, ollama_intent, speak=bool(speak), source="ollama")

        message = "I heard you, but I do not know how to run that command yet."
        result.update({"ok": False, "matched": None, "error": message, "speech": None})
        return result

    def _get_robot(self) -> Robot:
        if self.dry_run:
            raise RuntimeError("Robot SDK is disabled because --dry-run is active.")
        with self._lock:
            if self._robot is None:
                self._robot = Robot(iface=self.iface, domain_id=self.domain_id, auto_start_sensors=False)
            return self._robot

    def _get_arm_sdk(self) -> Any:
        if self.dry_run:
            raise RuntimeError("Arm SDK is disabled because --dry-run is active.")
        with self._lock:
            if self._arm_sdk is None:
                from arm_sdk import ArmSdk

                self._arm_sdk = ArmSdk(iface=self.iface, domain_id=self.domain_id)
                self._arm_sdk.resync()
            return self._arm_sdk

    def get_visual_context(self) -> dict[str, Any]:
        snap = self.receiver.snapshot()
        ctx = snap.to_context()
        ctx["source"] = self.receiver.endpoint
        ctx["hand_type"] = self.hand_type
        ctx["summary"] = self._heuristic_description(ctx)
        return ctx

    def describe_visual_context(
        self,
        prompt: str | None = None,
        *,
        use_ollama: bool = True,
        speak: bool | None = None,
    ) -> dict[str, Any]:
        ctx = self.get_visual_context()
        snap = self.receiver.snapshot()
        heuristic = str(ctx["summary"])
        should_speak = self.speak_answers if speak is None else bool(speak)
        if not use_ollama or snap.rgb_jpeg is None:
            return self._description_result(heuristic, ctx, None, should_speak)
        user_prompt = prompt or (
            "Describe the visible scene for a humanoid robot. Include nearby obstacles, "
            "reachable objects, people, floor space, and anything unsafe."
        )
        try:
            description = self._ask_ollama_vision(user_prompt, snap.rgb_jpeg, ctx)
            return self._description_result(description, ctx, self.vision_model, should_speak)
        except Exception as exc:
            result = self._description_result(heuristic, ctx, None, should_speak)
            result["error"] = str(exc)
            return result

    def speak(
        self,
        text: str,
        volume: int | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        message = " ".join(str(text).split())
        if not message:
            raise ValueError("text is required")
        if len(message) > 500:
            message = message[:500].rsplit(" ", 1)[0] + "..."
        use_volume = self.speech_volume if volume is None else int(clamp(volume, 0, 100))
        use_language = self.speech_language if language is None else language
        if self.dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "command": "speak",
                "text": message,
                "volume": use_volume,
                "language": use_language,
            }
        code = int(self._get_robot().say(message, volume=use_volume, language=use_language))
        return {
            "ok": code == 0,
            "code": code,
            "text": message,
            "volume": use_volume,
            "language": use_language,
            "output": f"Robot.say returned {code}",
        }

    def set_hand_type(self, hand_type: str) -> dict[str, Any]:
        selected = self._normalize_hand_type(hand_type)
        self.hand_type = selected
        return {"ok": True, "hand_type": self.hand_type}

    def get_hand_type(self) -> dict[str, Any]:
        return {"ok": True, "hand_type": self.hand_type, "available": ["dummy", "dex3", "inspire"]}

    def hand_open(
        self,
        hand: str = "right",
        hold_s: float = 0.6,
        rate_hz: float = 50.0,
        ramp_s: float | None = None,
        speed: int = 200,
        force: int = 200,
    ) -> dict[str, Any]:
        hands = self._normalize_hand_selection(hand)
        return self._dispatch_hand_motion(
            "open",
            hands,
            hold_s=hold_s,
            rate_hz=rate_hz,
            ramp_s=ramp_s,
            speed=speed,
            force=force,
        )

    def hand_close(
        self,
        hand: str = "right",
        hold_s: float = 0.6,
        rate_hz: float = 50.0,
        ramp_s: float | None = None,
        speed: int = 200,
        force: int = 200,
    ) -> dict[str, Any]:
        hands = self._normalize_hand_selection(hand)
        return self._dispatch_hand_motion(
            "close",
            hands,
            hold_s=hold_s,
            rate_hz=rate_hz,
            ramp_s=ramp_s,
            speed=speed,
            force=force,
        )

    def loco_move(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> dict[str, Any]:
        vx = clamp(vx, -0.6, 0.6)
        vy = clamp(vy, -0.4, 0.4)
        vyaw = clamp(vyaw, -1.2, 1.2)
        if self.dry_run:
            return {"ok": True, "dry_run": True, "command": "loco_move", "vx": vx, "vy": vy, "vyaw": vyaw}
        code = self._get_robot().loco_move(vx, vy, vyaw)
        return {"ok": int(code) == 0, "code": int(code), "vx": vx, "vy": vy, "vyaw": vyaw}

    def move_for(
        self,
        duration_s: float,
        vx: float = 0.0,
        vy: float = 0.0,
        vyaw: float = 0.0,
    ) -> dict[str, Any]:
        duration_s = clamp(duration_s, 0.0, 10.0)
        vx = clamp(vx, -0.6, 0.6)
        vy = clamp(vy, -0.4, 0.4)
        vyaw = clamp(vyaw, -1.2, 1.2)
        if self.dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "command": "move_for",
                "duration_s": duration_s,
                "vx": vx,
                "vy": vy,
                "vyaw": vyaw,
            }
        code = self._get_robot().move_for(duration_s, vx=vx, vy=vy, vyaw=vyaw)
        return {"ok": int(code) == 0, "code": int(code), "duration_s": duration_s, "vx": vx, "vy": vy, "vyaw": vyaw}

    def stop(self) -> dict[str, Any]:
        if self.dry_run:
            return {"ok": True, "dry_run": True, "command": "stop"}
        self._get_robot().stop()
        return {"ok": True}

    def hand_shake(self, release_after_s: float | None = 2.0) -> dict[str, Any]:
        if self.dry_run:
            return {"ok": True, "dry_run": True, "command": "hand_shake", "release_after_s": release_after_s}
        code = self._get_robot().shake_hand(release_after_s=release_after_s)
        return {"ok": int(code) == 0, "code": int(code), "release_after_s": release_after_s}

    def release_arms(
        self,
        duration_s: float = 3.0,
        command_rate_hz: float = 50.0,
    ) -> dict[str, Any]:
        duration_s = clamp(duration_s, 0.1, 10.0)
        command_rate_hz = clamp(command_rate_hz, 5.0, 100.0)
        if self.dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "command": "release_arms",
                "duration_s": duration_s,
                "command_rate_hz": command_rate_hz,
            }
        result = self._get_robot().release_arms(duration_s=duration_s, command_rate_hz=command_rate_hz)
        return {"ok": True, "result": result}

    def ik_move_ee_pose(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        droll: float = 0.0,
        dpitch: float = 0.0,
        dyaw: float = 0.0,
        arm: str = "right",
        mirror: bool = True,
        lock_orientation: bool = False,
    ) -> dict[str, Any]:
        side = str(arm).strip().lower()
        if side not in {"left", "right", "both"}:
            raise ValueError("arm must be left, right, or both")
        inc = [
            clamp(dx, -0.15, 0.15),
            clamp(dy, -0.15, 0.15),
            clamp(dz, -0.15, 0.15),
            clamp(droll, -0.5, 0.5),
            clamp(dpitch, -0.5, 0.5),
            clamp(dyaw, -0.5, 0.5),
        ]
        position_only = not bool(lock_orientation) and not any(abs(v) > 1e-9 for v in inc[3:])
        selected_axis = next((idx for idx, value in enumerate(inc[:3]) if abs(value) > 1e-9), None)
        if self.dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "command": "ik_move_ee_pose",
                "increment": inc,
                "arm": side,
                "position_only": position_only,
                "selected_axis": selected_axis,
            }
        info = self._get_arm_sdk().ik_move_EE(
            inc,
            arm=side,
            mirror=bool(mirror),
            position_only=position_only,
            selected_axis=selected_axis,
        )
        return {
            "ok": bool(info.get("success")),
            "arm": side,
            "increment": inc,
            "position_only": position_only,
            "selected_axis": selected_axis,
            "result": info,
        }

    def run_diagnostic_command(
        self,
        command: str,
        path: str | None = None,
        extra_args: list[str] | None = None,
        timeout_s: float = 5.0,
        max_output_chars: int = 12000,
    ) -> dict[str, Any]:
        if not self.allow_diagnostics:
            raise RuntimeError("Diagnostic shell tools are disabled.")
        name = str(command).strip()
        if name not in DIAGNOSTIC_COMMANDS:
            raise ValueError(
                "Unsupported diagnostic command. Use one of: "
                + ", ".join(sorted(DIAGNOSTIC_COMMANDS))
            )

        argv = list(DIAGNOSTIC_COMMANDS[name])
        if name in DIAGNOSTIC_PATH_COMMANDS:
            safe_path = self._safe_diagnostic_path(path or ".")
            argv.append(safe_path)
        if extra_args:
            argv.extend(self._safe_extra_args(extra_args))

        timeout = clamp(timeout_s, 1.0, 20.0)
        limit = int(clamp(max_output_chars, 1000, 50000))
        started = time.time()
        try:
            proc = subprocess.run(
                argv,
                shell=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                cwd=PROJECT_ROOT,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            return {
                "ok": False,
                "command": " ".join(shlex.quote(part) for part in argv),
                "returncode": None,
                "elapsed_s": round(time.time() - started, 3),
                "truncated": False,
                "output": output,
                "error": f"timed out after {timeout:.1f}s",
            }
        output = proc.stdout or ""
        truncated = len(output) > limit
        if truncated:
            output = output[:limit] + "\n...[truncated]..."
        return {
            "ok": proc.returncode == 0,
            "command": " ".join(shlex.quote(part) for part in argv),
            "returncode": int(proc.returncode),
            "elapsed_s": round(time.time() - started, 3),
            "truncated": truncated,
            "output": output,
        }

    def diagnose_system(
        self,
        *,
        speak: bool = False,
        timeout_s: float = 4.0,
        max_output_chars: int = 8000,
    ) -> dict[str, Any]:
        if not self.allow_diagnostics:
            raise RuntimeError("Diagnostic shell tools are disabled.")
        results: dict[str, Any] = {}
        for command in SYSTEM_DIAGNOSTIC_COMMANDS:
            results[command] = self.run_diagnostic_command(
                command,
                timeout_s=timeout_s,
                max_output_chars=max_output_chars,
            )
        summary = self._summarize_system_diagnostics(results)
        speech = self._speak_feedback(summary, speak)
        return {"ok": True, "summary": summary, "results": results, "speech": speech}

    def web_search(self, query: str, max_results: int = 5, timeout_s: float = 8.0) -> dict[str, Any]:
        if not self.allow_web_search:
            raise RuntimeError("Web search tool is disabled.")
        q = " ".join(str(query).split())
        if not q:
            raise ValueError("query is required")
        max_results = int(clamp(max_results, 1, 10))
        url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 naive_VLA robot diagnostic search",
                "Accept": "text/html,application/xhtml+xml",
            },
            method="GET",
        )
        started = time.time()
        with urllib.request.urlopen(request, timeout=clamp(timeout_s, 2.0, 20.0)) as response:
            raw = response.read(1_000_000).decode("utf-8", errors="replace")
        return {
            "ok": True,
            "query": q,
            "elapsed_s": round(time.time() - started, 3),
            "results": self._parse_duckduckgo_results(raw, max_results),
        }

    def _description_result(
        self,
        description: str,
        context: dict[str, Any],
        model: str | None,
        speak: bool,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"description": description, "context": context, "model": model}
        if speak:
            try:
                result["speech"] = self.speak(description)
            except Exception as exc:
                result["speech"] = {"ok": False, "error": str(exc)}
        return result

    @staticmethod
    def _prompt_requests_visual_description(normalized: str) -> bool:
        visual_words = ("camera", "visible", "see", "seeing", "objects", "object", "scene", "visual")
        describe_words = ("describe", "what", "tell me", "look")
        return any(word in normalized for word in visual_words) and any(word in normalized for word in describe_words)

    @staticmethod
    def _prompt_requests_system_diagnosis(normalized: str) -> bool:
        return (
            "diagnose" in normalized
            or "dgnognose" in normalized
            or "diagnognose" in normalized
            or "diagno" in normalized
            or "diagnostic" in normalized
            or "check system" in normalized
            or "check the system" in normalized
        ) and any(word in normalized for word in ("system", "jetson", "board", "computer", "robot"))

    def _fuzzy_builtin_intent(self, normalized: str) -> dict[str, Any] | None:
        words = re.findall(r"[a-z]+", normalized)
        best_diagnose = max((difflib.SequenceMatcher(None, word, "diagnose").ratio() for word in words), default=0.0)
        if best_diagnose >= 0.58 and any(word in normalized for word in ("system", "jetson", "board", "computer", "robot")):
            return {"intent": "diagnose_system", "confidence": best_diagnose}
        best_shake = max((difflib.SequenceMatcher(None, word, "shake").ratio() for word in words), default=0.0)
        if best_shake >= 0.72 and ("hand" in normalized or "hands" in normalized):
            return {"intent": "hand_shake", "confidence": best_shake}
        return None

    def _run_intent(
        self,
        prompt: str,
        intent: dict[str, Any],
        *,
        speak: bool,
        source: str,
    ) -> dict[str, Any]:
        name = str(intent.get("intent") or intent.get("tool") or "").strip()
        result: dict[str, Any] = {
            "ok": True,
            "prompt": prompt,
            "matched": name or None,
            "source": source,
            "intent": intent,
        }
        if name == "diagnose_system":
            diag = self.diagnose_system(speak=speak)
            result.update({"result": diag, "speech": diag.get("speech")})
            return result
        if name == "describe_visual_context":
            desc = self.describe_visual_context(prompt=prompt, speak=speak)
            result.update({"result": desc})
            return result
        if name == "hand_open":
            hand = str(intent.get("hand") or "right")
            tool_result = self.hand_open(hand)
            speech = self._speak_feedback(f"Opened {hand} hand.", speak)
            result.update({"result": tool_result, "speech": speech})
            return result
        if name == "hand_close":
            hand = str(intent.get("hand") or "right")
            tool_result = self.hand_close(hand)
            speech = self._speak_feedback(f"Closed {hand} hand.", speak)
            result.update({"result": tool_result, "speech": speech})
            return result
        if name == "hand_shake":
            tool_result = self.hand_shake()
            speech = self._speak_feedback("Shaking hand.", speak)
            result.update({"result": tool_result, "speech": speech})
            return result
        if name == "ik_move_ee_pose":
            args = {
                "arm": str(intent.get("arm") or "right"),
                "dx": float(intent.get("dx") or 0.0),
                "dy": float(intent.get("dy") or 0.0),
                "dz": float(intent.get("dz") or 0.0),
                "lock_orientation": False,
            }
            tool_result = self.ik_move_ee_pose(**args)
            speech = self._speak_feedback("Moved end effector.", speak)
            result.update({"args": args, "result": tool_result, "speech": speech})
            return result
        if name == "move_for":
            args = {
                "duration_s": float(intent.get("duration_s") or 1.0),
                "vx": float(intent.get("vx") or 0.0),
                "vy": float(intent.get("vy") or 0.0),
                "vyaw": float(intent.get("vyaw") or 0.0),
            }
            tool_result = self.move_for(**args)
            speech = self._speak_feedback("Moving.", speak)
            result.update({"args": args, "result": tool_result, "speech": speech})
            return result
        if name == "release_arms":
            tool_result = self.release_arms()
            speech = self._speak_feedback("Released arms.", speak)
            result.update({"result": tool_result, "speech": speech})
            return result
        if name == "stop":
            tool_result = self.stop()
            speech = self._speak_feedback("Stopped.", speak)
            result.update({"result": tool_result, "speech": speech})
            return result
        return {"ok": False, "prompt": prompt, "matched": None, "error": "No safe command intent matched.", "speech": None}

    def _interpret_prompt_with_ollama(self, prompt: str) -> dict[str, Any] | None:
        body = {
            "model": self.text_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Map noisy speech transcripts to one safe robot command. "
                        "Return only compact JSON. Allowed intents: diagnose_system, describe_visual_context, "
                        "hand_open, hand_close, hand_shake, ik_move_ee_pose, move_for, release_arms, stop, none. "
                        "Use iface eth0 and DDS domain 0 implicitly. For ik_move_ee_pose, use arm left/right/both, "
                        "dx/dy/dz meters, no orientation lock. forward/extend means dx positive; up means dz positive; "
                        "away from body means dy outward, use dy 0.03 for both/left and -0.03 for right. "
                        "For unclear unrelated speech, intent must be none."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Transcript: "
                        + json.dumps(prompt, ensure_ascii=False)
                        + "\nJSON schema: {\"intent\":\"...\",\"confidence\":0.0,\"hand\":\"right|left|both\","
                        + "\"arm\":\"right|left|both\",\"dx\":0,\"dy\":0,\"dz\":0,\"duration_s\":1,\"vx\":0,\"vy\":0,\"vyaw\":0}"
                    ),
                },
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 120},
        }
        request = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=json_dumps(body),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = str(payload.get("message", {}).get("content", "")).strip()
            intent = json.loads(content)
        except Exception:
            return None
        if not isinstance(intent, dict):
            return None
        name = str(intent.get("intent") or "").strip()
        confidence = float(intent.get("confidence") or 0.0)
        allowed = {
            "diagnose_system",
            "describe_visual_context",
            "hand_open",
            "hand_close",
            "hand_shake",
            "ik_move_ee_pose",
            "move_for",
            "release_arms",
            "stop",
        }
        if name not in allowed or confidence < 0.55:
            return None
        return intent

    @staticmethod
    def _parse_loco_prompt(normalized: str) -> dict[str, Any] | None:
        if not any(word in normalized for word in ("walk", "go", "move", "step", "turn")):
            return None
        args = {"duration_s": 1.0, "vx": 0.0, "vy": 0.0, "vyaw": 0.0}
        if "forward" in normalized or "ahead" in normalized:
            args["vx"] = 0.2
        elif "backward" in normalized or re.search(r"\bback\b", normalized):
            args["vx"] = -0.15
        elif "left" in normalized:
            args["vyaw"] = 0.4 if "turn" in normalized else 0.0
            args["vy"] = 0.15 if "turn" not in normalized else 0.0
        elif "right" in normalized:
            args["vyaw"] = -0.4 if "turn" in normalized else 0.0
            args["vy"] = -0.15 if "turn" not in normalized else 0.0
        else:
            return None
        return args

    def _parse_ee_prompt(self, normalized: str) -> dict[str, Any] | None:
        if not any(token in normalized for token in ("end effector", "ee", "arm", "hand", "hands", "wrist", "extend", "reach")):
            return None
        if "both" in normalized or ("hands" in normalized and "left" not in normalized and "right" not in normalized):
            arm = "both"
        elif "left" in normalized:
            arm = "left"
        else:
            arm = "right"
        axis = None
        sign = 1.0
        for candidate in ("x", "y", "z"):
            if re.search(rf"\b{candidate}\b", normalized):
                axis = candidate
                break
        if axis is None:
            if "up" in normalized or "higher" in normalized or "raise" in normalized or "lift" in normalized:
                axis = "z"
            elif "down" in normalized or "lower" in normalized:
                axis = "z"
                sign = -1.0
            elif "forward" in normalized or "extend" in normalized or "reach" in normalized or "ahead" in normalized:
                axis = "x"
            elif "backward" in normalized or re.search(r"\bback\b", normalized):
                axis = "x"
                sign = -1.0
            elif (
                "away from body" in normalized
                or "away from me" in normalized
                or "away from bud" in normalized
                or "away from bot" in normalized
                or "away from butt" in normalized
                or "away from bod" in normalized
                or "outward" in normalized
                or "to the side" in normalized
            ):
                axis = "y"
                sign = self._outward_y_sign(arm)
            elif "toward body" in normalized or "towards body" in normalized or "toward me" in normalized or "inward" in normalized:
                axis = "y"
                sign = -self._outward_y_sign(arm)
            elif "left" in normalized or "right" in normalized:
                axis = "y"
                sign = 1.0 if "left" in normalized else -1.0
        if axis is None:
            return None

        magnitude = 0.03
        number_match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(cm|centimeter|centimeters|m|meter|meters)?", normalized)
        if number_match:
            raw = abs(float(number_match.group(1)))
            unit = number_match.group(2) or "m"
            magnitude = raw / 100.0 if unit.startswith("cm") or unit.startswith("centimeter") else raw
        magnitude = clamp(magnitude, 0.005, 0.10)

        negative_words = ("decrease", "reduce", "minus", "negative")
        if any(word in normalized for word in negative_words):
            sign *= -1.0

        args = {
            "arm": arm,
            "dx": 0.0,
            "dy": 0.0,
            "dz": 0.0,
            "droll": 0.0,
            "dpitch": 0.0,
            "dyaw": 0.0,
            "lock_orientation": False,
        }
        args[f"d{axis}"] = sign * magnitude
        return args

    @staticmethod
    def _outward_y_sign(arm: str) -> float:
        if arm == "right":
            return -1.0
        return 1.0

    @staticmethod
    def _summarize_system_diagnostics(results: dict[str, Any]) -> str:
        failed = [name for name, result in results.items() if not result.get("ok")]
        parts: list[str] = []

        uptime = str(results.get("uptime", {}).get("output", "")).strip()
        if uptime:
            parts.append("Uptime: " + " ".join(uptime.split()))

        free = str(results.get("free", {}).get("output", "")).splitlines()
        mem_line = next((line for line in free if line.lower().startswith("mem:")), "")
        if mem_line:
            fields = mem_line.split()
            if len(fields) >= 7:
                parts.append(f"Memory: {fields[2]} used, {fields[6]} available of {fields[1]}.")

        df = str(results.get("df", {}).get("output", "")).splitlines()
        root_line = next((line for line in df if line.rstrip().endswith(" /")), "")
        if root_line:
            fields = root_line.split()
            if len(fields) >= 5:
                parts.append(f"Root disk: {fields[4]} used, {fields[3]} free.")

        services = str(results.get("systemctl_failed", {}).get("output", "")).strip()
        if services and "0 loaded units listed" not in services.lower():
            parts.append("There are failed systemd units.")
        else:
            parts.append("No failed systemd units reported.")

        journal = str(results.get("journal_errors", {}).get("output", "")).strip()
        if journal and "-- no entries --" not in journal.lower():
            parts.append("Recent journal errors are present.")
        else:
            parts.append("No recent journal priority errors reported.")

        if failed:
            parts.append("Some diagnostic commands failed: " + ", ".join(failed) + ".")

        return " ".join(parts) if parts else "System diagnostics completed, but no summary data was available."

    def _speak_feedback(self, text: str, speak: bool) -> dict[str, Any] | None:
        if not speak:
            return None
        try:
            return self.speak(text)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @staticmethod
    def _normalize_hand_type(hand_type: str) -> str:
        selected = str(hand_type).strip().lower()
        if selected not in {"dummy", "dex3", "inspire"}:
            raise ValueError("hand_type must be one of: dummy, dex3, inspire")
        return selected

    @staticmethod
    def _normalize_hand_selection(hand: str) -> tuple[str, ...]:
        side = str(hand).strip().lower()
        if side in {"both", "all"}:
            return ("left", "right")
        if side in {"l", "left"}:
            return ("left",)
        if side in {"r", "right"}:
            return ("right",)
        raise ValueError("hand must be left, right, or both")

    def _dispatch_hand_motion(
        self,
        action: str,
        hands: tuple[str, ...],
        *,
        hold_s: float,
        rate_hz: float,
        ramp_s: float | None,
        speed: int,
        force: int,
    ) -> dict[str, Any]:
        hold_s = clamp(hold_s, 0.0, 5.0)
        rate_hz = clamp(rate_hz, 5.0, 100.0)
        speed = int(clamp(speed, 0, 1000))
        force = int(clamp(force, 0, 1000))
        selected = self.hand_type

        if selected == "dummy" or self.dry_run:
            return {
                "ok": True,
                "dry_run": self.dry_run,
                "hand_type": selected,
                "action": action,
                "hands": list(hands),
                "note": "dummy/no-op hand motion" if selected == "dummy" else "dry-run hand motion",
            }

        if selected == "dex3":
            robot = self._get_robot()
            for side in hands:
                if action == "open":
                    robot.hand_open(side, hold_s=hold_s, rate_hz=rate_hz, ramp_s=ramp_s)
                else:
                    robot.hand_close(side, hold_s=hold_s, rate_hz=rate_hz, ramp_s=ramp_s)
            return {"ok": True, "hand_type": selected, "action": action, "hands": list(hands)}

        if selected == "inspire":
            from inspire_sdk import close_hand as inspire_close_hand
            from inspire_sdk import open_hand as inspire_open_hand

            for side in hands:
                if action == "open":
                    inspire_open_hand(side, speed=speed, force=force, hold=hold_s)
                else:
                    inspire_close_hand(side, speed=speed, force=force, hold=hold_s)
            return {
                "ok": True,
                "hand_type": selected,
                "action": action,
                "hands": list(hands),
                "speed": speed,
                "force": force,
            }

        raise RuntimeError(f"Unhandled hand_type: {selected}")

    @staticmethod
    def _safe_diagnostic_path(path: str) -> str:
        raw = str(path).strip() or "."
        resolved = os.path.abspath(os.path.join(PROJECT_ROOT, raw) if not os.path.isabs(raw) else raw)
        allowed_roots = [PROJECT_ROOT, "/home/unitree", "/tmp"]
        if not any(resolved == root or resolved.startswith(root + os.sep) for root in allowed_roots):
            raise ValueError("diagnostic path must be inside project root, /home/unitree, or /tmp")
        return resolved

    @staticmethod
    def _safe_extra_args(args: list[str]) -> list[str]:
        safe: list[str] = []
        for item in args[:8]:
            value = str(item)
            if not value or any(ch in value for ch in "\n\r\0"):
                raise ValueError("extra_args contains an unsafe value")
            if len(value) > 80:
                raise ValueError("extra_args entries must be <= 80 characters")
            safe.append(value)
        return safe

    @staticmethod
    def _parse_duckduckgo_results(raw_html: str, max_results: int) -> list[dict[str, str]]:
        import re

        results: list[dict[str, str]] = []
        pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(raw_html):
            href = html.unescape(match.group("href"))
            title = re.sub(r"<.*?>", "", match.group("title"), flags=re.DOTALL)
            title = html.unescape(" ".join(title.split()))
            url = href
            parsed = urllib.parse.urlparse(href)
            if parsed.path == "/l/":
                params = urllib.parse.parse_qs(parsed.query)
                if params.get("uddg"):
                    url = params["uddg"][0]
            if title and url:
                results.append({"title": title, "url": url})
            if len(results) >= max_results:
                break
        return results

    def _heuristic_description(self, ctx: dict[str, Any]) -> str:
        if ctx.get("error"):
            return str(ctx["error"])
        center = ctx.get("center_depth_m")
        near_cov = ctx.get("near_coverage_1m")
        valid = float(ctx.get("valid_depth_fraction") or 0.0)
        pieces = [f"RGBD frame {ctx.get('width')}x{ctx.get('height')} with {valid * 100:.0f}% valid depth."]
        if center is None:
            pieces.append("No reliable center depth is available.")
        elif center < 0.6:
            pieces.append(f"Something is very close in front of the robot at about {center:.2f} m.")
        elif center < 1.2:
            pieces.append(f"The central view has a nearby target or obstacle at about {center:.2f} m.")
        else:
            pieces.append(f"The central view is relatively open to about {center:.2f} m.")
        if near_cov is not None:
            pieces.append(f"{near_cov * 100:.0f}% of the central working area is within 1 m.")
        return " ".join(pieces)

    def _ask_ollama_vision(self, prompt: str, rgb_jpeg: bytes, context: dict[str, Any]) -> str:
        body = {
            "model": self.vision_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"{prompt}\n\nDepth/context JSON: {json.dumps(context, ensure_ascii=False)}",
                    "images": [base64.b64encode(rgb_jpeg).decode("ascii")],
                },
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2, "num_predict": 160},
        }
        request = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=json_dumps(body),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        text = str(result.get("message", {}).get("content", "")).strip()
        return " ".join(text.split()) or self._heuristic_description(context)


class MicrophonePromptListener:
    def __init__(
        self,
        vla: NaiveVLA,
        *,
        topic: str = "/audio_msg",
        min_confidence: float = 0.0,
        speak: bool = True,
    ) -> None:
        self.vla = vla
        self.topic = str(topic)
        self.min_confidence = float(min_confidence)
        self.speak = bool(speak)
        self._thread: threading.Thread | None = None
        self._node: Any = None
        self._rclpy: Any = None
        self._running = False
        self._last_index: int | None = None
        self._last_text: str | None = None
        self._last_ts = 0.0
        self.last_event: dict[str, Any] | None = None

    def start(self) -> bool:
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String
        except Exception as exc:
            print(f"Microphone prompts disabled: ROS 2 Python imports failed: {exc}", file=sys.stderr)
            return False

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)

        listener = self

        class PromptNode(Node):
            def __init__(self) -> None:
                super().__init__("naive_vla_microphone_prompts")
                self.create_subscription(String, listener.topic, self.on_audio_msg, 10)
                self.get_logger().info(f"Listening for microphone prompts on {listener.topic}")

            def on_audio_msg(self, msg: Any) -> None:
                listener._on_audio_msg(str(msg.data))

        try:
            self._node = PromptNode()
        except Exception as exc:
            print(f"Microphone prompts disabled: ROS 2 node creation failed: {exc}", file=sys.stderr)
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
            return False

        self._running = True
        self._thread = threading.Thread(target=self._spin, name="mic-prompt-listener", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        try:
            if self._node is not None:
                self._node.destroy_node()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            if self._rclpy is not None and self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:
            pass

    def _spin(self) -> None:
        assert self._rclpy is not None
        while self._running:
            try:
                self._rclpy.spin_once(self._node, timeout_sec=0.1)
            except Exception as exc:
                print(f"Microphone prompt listener error: {exc}", file=sys.stderr)
                time.sleep(0.2)

    def _on_audio_msg(self, raw: str) -> None:
        payload = decode_audio_payload(raw)
        text = str(payload.get("text") or payload.get("raw") or "").strip()
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        index = self._payload_index(payload)
        now = time.time()
        if not self._should_run(text, confidence, index, now):
            return
        print(f'[mic] prompt="{text}" confidence={confidence:.2f} index={index}', file=sys.stderr)
        try:
            result = self.vla.run_prompt(text, speak=self.speak)
        except Exception as exc:
            result = {"ok": False, "prompt": text, "error": str(exc)}
            self.vla._speak_feedback("I could not run that command.", self.speak)
        self.last_event = {"received_at": now, "payload": payload, "result": result}
        print(f"[mic] result={json.dumps(result, ensure_ascii=False)}", file=sys.stderr)
        self._last_index = index
        self._last_text = text
        self._last_ts = now

    def _should_run(self, text: str, confidence: float, index: int | None, now: float) -> bool:
        if not text or confidence < self.min_confidence:
            return False
        normalized = strip_prompt_punctuation(normalize_prompt(text))
        if normalized in FILLER_TEXTS:
            return False
        if any(phrase in normalized for phrase in SPEECH_ECHO_PHRASES):
            return False
        if len(normalized) <= 2 and normalized not in {"up"}:
            return False
        if not any(char.isalnum() for char in text):
            return False
        if index is not None and index == self._last_index:
            return False
        if index is None and text == self._last_text and now - self._last_ts < 2.0:
            return False
        return True

    @staticmethod
    def _payload_index(payload: dict[str, Any]) -> int | None:
        try:
            value = payload.get("index")
            return int(value) if value is not None else None
        except Exception:
            return None


def make_handler(vla: NaiveVLA) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "NaiveVLA/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            message = fmt % args
            if (
                ('"GET /rgb.jpg' in message or '"GET /depth.jpg' in message or '"GET /api/context' in message)
                and ('" 200 ' in message or '" 204 ' in message)
            ):
                return
            print(f"[web] {self.address_string()} {message}", file=sys.stderr)

        def do_GET(self) -> None:
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send_html()
            elif path == "/rgb.jpg":
                self._send_image("rgb")
            elif path == "/depth.jpg":
                self._send_image("depth")
            elif path == "/api/context":
                self._send_json(vla.get_visual_context())
            elif path == "/api/describe":
                self._send_json(vla.describe_visual_context())
            elif path == "/api/tools":
                self._send_json(
                    {
                        "tools": sorted(vla.tools().keys()),
                        "diagnostics_enabled": vla.allow_diagnostics,
                        "diagnostic_commands": sorted(DIAGNOSTIC_COMMANDS),
                        "web_search_enabled": vla.allow_web_search,
                        "speak_answers": vla.speak_answers,
                        "hand_type": vla.hand_type,
                        "hand_types": ["dummy", "dex3", "inspire"],
                    }
                )
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urllib.parse.urlparse(self.path).path
            if path != "/api/action":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                name = str(payload.get("tool") or payload.get("action") or "")
                args = payload.get("args") or {}
                if not isinstance(args, dict):
                    raise ValueError("args must be an object")
                tools = vla.tools()
                if name not in tools:
                    raise ValueError(f"Unknown tool: {name}")
                result = tools[name](**args)
                self._send_json({"ok": True, "tool": name, "result": result})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json_dumps(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._write_body(body)

        def _send_image(self, which: str) -> None:
            snap = vla.receiver.snapshot()
            data = snap.rgb_jpeg if which == "rgb" else snap.depth_jpeg
            if data is None:
                data = self._placeholder_jpeg(which, snap.error or "No RGBD frame yet")
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self._write_body(data)

        def _write_body(self, body: bytes) -> None:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _placeholder_jpeg(self, which: str, message: str) -> bytes:
            img = np.zeros((360, 640, 3), dtype=np.uint8)
            img[:] = (18, 22, 22)
            color = (60, 190, 220) if which == "rgb" else (210, 150, 70)
            cv2.putText(img, which.upper(), (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            words = str(message).split()
            line = ""
            y = 100
            for word in words:
                trial = f"{line} {word}".strip()
                if len(trial) > 58:
                    cv2.putText(img, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 230, 226), 1)
                    y += 30
                    line = word
                else:
                    line = trial
            if line:
                cv2.putText(img, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 230, 226), 1)
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            return enc.tobytes() if ok else b""

        def _send_html(self) -> None:
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._write_body(body)

    return Handler


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Naive VLA</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, -apple-system, sans-serif; }
    body { margin: 0; background: #101314; color: #eef1ed; }
    header { padding: 14px 18px; border-bottom: 1px solid #2b3030; display: flex; justify-content: space-between; gap: 12px; align-items: center; }
    h1 { font-size: 18px; margin: 0; font-weight: 650; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 18px; padding: 18px; }
    .feeds { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .panel { background: #171b1c; border: 1px solid #2b3030; border-radius: 8px; overflow: hidden; }
    .panel h2 { margin: 0; padding: 10px 12px; font-size: 14px; border-bottom: 1px solid #2b3030; color: #cbd7d0; }
    img { width: 100%; display: block; aspect-ratio: 4 / 3; object-fit: contain; background: #050606; }
    .side { display: grid; gap: 12px; align-content: start; }
    pre { margin: 0; padding: 12px; white-space: pre-wrap; overflow-wrap: anywhere; min-height: 120px; font-size: 12px; }
    .controls { padding: 12px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    button { border: 1px solid #3f4a48; background: #25302d; color: #f4f7f3; border-radius: 6px; padding: 9px 8px; font: inherit; cursor: pointer; }
    button:hover { background: #32403c; }
    input { grid-column: span 2; min-width: 0; border: 1px solid #3f4a48; background: #0f1313; color: #eef1ed; border-radius: 6px; padding: 9px 8px; font: inherit; }
    .radio-row { grid-column: span 3; display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .radio-row label { border: 1px solid #3f4a48; background: #0f1313; border-radius: 6px; padding: 8px; font-size: 13px; display: flex; gap: 6px; align-items: center; justify-content: center; }
    .radio-row input { grid-column: auto; min-width: auto; }
    .wide { grid-column: span 3; }
    .status { color: #adc0b8; font-size: 13px; padding-right: 4px; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } .feeds { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Naive VLA</h1>
    <div class="status" id="status">starting</div>
  </header>
  <main>
    <section class="feeds">
      <div class="panel"><h2>RGB</h2><img id="rgb" alt="RGB feed"></div>
      <div class="panel"><h2>Depth</h2><img id="depth" alt="Depth feed"></div>
    </section>
    <aside class="side">
      <div class="panel"><h2>Visual Context</h2><pre id="context">{}</pre></div>
      <div class="panel"><h2>Actions</h2>
        <div class="controls">
          <div class="radio-row" id="handTypeRadios">
            <label><input type="radio" name="handType" value="dummy"> Dummy</label>
            <label><input type="radio" name="handType" value="dex3" checked> Dex3</label>
            <label><input type="radio" name="handType" value="inspire"> Inspire</label>
          </div>
          <button onclick="act('hand_open',{hand:'left'})">Open L</button>
          <button onclick="act('hand_open',{hand:'both'})">Open Both</button>
          <button onclick="act('hand_open',{hand:'right'})">Open R</button>
          <button onclick="act('hand_close',{hand:'left'})">Close L</button>
          <button onclick="act('hand_close',{hand:'both'})">Close Both</button>
          <button onclick="act('hand_close',{hand:'right'})">Close R</button>
          <button onclick="act('move_for',{duration_s:1.0,vx:0.2})">Forward</button>
          <button onclick="act('stop',{})">Stop</button>
          <button onclick="act('move_for',{duration_s:1.0,vx:-0.15})">Back</button>
          <button onclick="act('move_for',{duration_s:0.8,vy:0.15})">Left</button>
          <button onclick="act('hand_shake',{release_after_s:2.0})">Handshake</button>
          <button onclick="act('move_for',{duration_s:0.8,vy:-0.15})">Right</button>
          <button class="wide" onclick="act('ik_move_ee_pose',{arm:'right',dx:0.04})">Right EE +X</button>
          <button class="wide" onclick="act('release_arms',{duration_s:3.0})">Release Arms</button>
          <button class="wide" onclick="describe()">Describe Visual Context</button>
          <button class="wide" onclick="describe(true)">Describe + Speak</button>
          <input id="prompttext" placeholder="test a spoken prompt">
          <button onclick="runPrompt()">Run Prompt</button>
          <input id="saytext" placeholder="text for robot to say">
          <button onclick="say()">Say</button>
          <button onclick="act('run_diagnostic_command',{command:'free'})">free</button>
          <button onclick="act('run_diagnostic_command',{command:'df'})">df</button>
          <button onclick="act('run_diagnostic_command',{command:'du',path:'.'})">du</button>
          <input id="searchq" placeholder="web search query">
          <button onclick="search()">Search</button>
        </div>
      </div>
      <div class="panel"><h2>Last Result</h2><pre id="result"></pre></div>
    </aside>
  </main>
  <script>
    const stamp = () => Date.now().toString();
    function refreshImages() {
      document.getElementById('rgb').src = '/rgb.jpg?t=' + stamp();
      document.getElementById('depth').src = '/depth.jpg?t=' + stamp();
    }
    async function refreshContext() {
      const r = await fetch('/api/context', {cache:'no-store'});
      const j = await r.json();
      document.getElementById('context').textContent = JSON.stringify(j, null, 2);
      document.getElementById('status').textContent = j.error || ('age ' + (j.age_s ?? 0).toFixed(2) + 's');
      if (j.hand_type) {
        const radio = document.querySelector(`input[name="handType"][value="${j.hand_type}"]`);
        if (radio) radio.checked = true;
      }
    }
    async function act(tool, args) {
      const r = await fetch('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({tool, args})});
      document.getElementById('result').textContent = JSON.stringify(await r.json(), null, 2);
      refreshContext();
    }
    async function describe(speak=false) {
      if (speak) {
        await act('describe_visual_context', {speak: true});
        return;
      }
      const r = await fetch('/api/describe', {cache:'no-store'});
      document.getElementById('result').textContent = JSON.stringify(await r.json(), null, 2);
    }
    async function say() {
      const text = document.getElementById('saytext').value;
      await act('speak', {text});
    }
    async function runPrompt() {
      const prompt = document.getElementById('prompttext').value;
      await act('run_prompt', {prompt, speak: true});
    }
    async function search() {
      const query = document.getElementById('searchq').value;
      await act('web_search', {query, max_results: 5});
    }
    document.querySelectorAll('input[name="handType"]').forEach((radio) => {
      radio.addEventListener('change', () => {
        if (radio.checked) act('set_hand_type', {hand_type: radio.value});
      });
    });
    setInterval(refreshImages, 500);
    setInterval(refreshContext, 1000);
    refreshImages();
    refreshContext();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Naive RGBD vision-language-action tools with a small web UI."
    )
    parser.add_argument("--iface", default="eth0", help="DDS interface for robot SDK commands; forced to eth0.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID; forced to 0.")
    parser.add_argument("--rgbd-host", "--robot-ip", dest="rgbd_host", default=os.environ.get("G1_RGBD_HOST", "10.34.0.83"))
    parser.add_argument("--rgbd-port", type=int, default=int(os.environ.get("G1_RGBD_PORT", "5555")))
    parser.add_argument("--rgbd-topic", default=os.environ.get("G1_RGBD_TOPIC", ""))
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=12.0, help="RGBD UI update/decode rate.")
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8088)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--vision-model", default="qwen2.5vl:7b")
    parser.add_argument("--text-model", default="qwen3.5:9b")
    parser.add_argument("--dry-run", action="store_true", help="Expose tools without sending robot SDK commands.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Disable allowlisted diagnostic shell tools.")
    parser.add_argument("--no-web-search", action="store_true", help="Disable web_search tool.")
    parser.add_argument("--speak-answers", action="store_true", help="Speak visual descriptions by default.")
    parser.add_argument("--speech-volume", type=int, default=None, help="Optional robot speech volume 0-100.")
    parser.add_argument("--speech-language", default=None, help="Optional TTS language, for example en or de.")
    parser.add_argument("--hand-type", choices=("dummy", "dex3", "inspire"), default="dex3", help="Default hand hardware mode.")
    parser.add_argument("--mic-topic", default="/audio_msg", help="ROS 2 ASR topic for spoken user prompts.")
    parser.add_argument("--mic-min-confidence", type=float, default=0.0, help="Ignore ASR prompts below this confidence.")
    parser.add_argument("--no-mic", action="store_true", help="Disable microphone prompt listener.")
    parser.add_argument("--no-mic-speech", action="store_true", help="Do not speak feedback for microphone commands.")
    parser.add_argument("--describe-once", action="store_true", help="Print one visual description, then exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.iface = "eth0"
    args.domain_id = 0
    receiver = RgbdReceiver(
        args.rgbd_host,
        args.rgbd_port,
        args.rgbd_topic,
        max_depth_m=args.max_depth_m,
        fps=args.fps,
    )
    receiver.start()
    vla = NaiveVLA(
        receiver,
        iface=args.iface,
        domain_id=args.domain_id,
        dry_run=args.dry_run,
        ollama_url=args.ollama_url,
        vision_model=args.vision_model,
        text_model=args.text_model,
        allow_diagnostics=not args.no_diagnostics,
        allow_web_search=not args.no_web_search,
        speak_answers=args.speak_answers,
        speech_volume=args.speech_volume,
        speech_language=args.speech_language,
        hand_type=args.hand_type,
    )
    sdk_startup = vla.preinitialize_robot_sdk(arm=not args.dry_run)

    if args.describe_once:
        time.sleep(0.5)
        print(json.dumps(vla.describe_visual_context(), ensure_ascii=False, indent=2))
        receiver.stop()
        return 0

    mic_listener: MicrophonePromptListener | None = None
    mic_started = False
    if not args.no_mic:
        mic_listener = MicrophonePromptListener(
            vla,
            topic=args.mic_topic,
            min_confidence=args.mic_min_confidence,
            speak=not args.no_mic_speech,
        )
        mic_started = mic_listener.start()

    server = ThreadingHTTPServer((str(args.web_host), int(args.web_port)), make_handler(vla))
    print(f"Naive VLA web UI: http://{args.web_host}:{args.web_port}")
    print(f"RGBD source: {receiver.endpoint}")
    print(f"Robot commands: {'disabled (--dry-run)' if args.dry_run else 'enabled'}")
    print(f"Robot SDK preinit: {json.dumps(sdk_startup, ensure_ascii=False)}")
    print(f"Diagnostics: {'disabled' if args.no_diagnostics else 'enabled (allowlisted)'}")
    print(f"Web search: {'disabled' if args.no_web_search else 'enabled'}")
    print(f"Speech: {'answers enabled' if args.speak_answers else 'available as speak tool'}")
    print(f"Hand type: {vla.hand_type}")
    print(f"Microphone prompts: {'enabled on ' + args.mic_topic if mic_started else 'disabled'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        if mic_listener is not None:
            mic_listener.stop()
        receiver.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
