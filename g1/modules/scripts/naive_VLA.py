#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import os
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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def json_dumps(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


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
    ) -> None:
        self.receiver = receiver
        self.iface = str(iface)
        self.domain_id = int(domain_id)
        self.dry_run = bool(dry_run)
        self.ollama_url = str(ollama_url).rstrip("/")
        self.vision_model = str(vision_model)
        self.text_model = str(text_model)
        self.system_prompt = str(system_prompt)
        self.allow_diagnostics = bool(allow_diagnostics)
        self.allow_web_search = bool(allow_web_search)
        self._robot: Robot | None = None
        self._arm_sdk: Any = None
        self._lock = threading.Lock()

    def tools(self) -> dict[str, Callable[..., Any]]:
        return {
            "get_visual_context": self.get_visual_context,
            "describe_visual_context": self.describe_visual_context,
            "loco_move": self.loco_move,
            "move_for": self.move_for,
            "stop": self.stop,
            "hand_shake": self.hand_shake,
            "shake_hand": self.hand_shake,
            "release_arms": self.release_arms,
            "ik_move_ee_pose": self.ik_move_ee_pose,
            "run_diagnostic_command": self.run_diagnostic_command,
            "web_search": self.web_search,
        }

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
        ctx["summary"] = self._heuristic_description(ctx)
        return ctx

    def describe_visual_context(self, prompt: str | None = None, *, use_ollama: bool = True) -> dict[str, Any]:
        ctx = self.get_visual_context()
        snap = self.receiver.snapshot()
        heuristic = str(ctx["summary"])
        if not use_ollama or snap.rgb_jpeg is None:
            return {"description": heuristic, "context": ctx, "model": None}
        user_prompt = prompt or (
            "Describe the visible scene for a humanoid robot. Include nearby obstacles, "
            "reachable objects, people, floor space, and anything unsafe."
        )
        try:
            description = self._ask_ollama_vision(user_prompt, snap.rgb_jpeg, ctx)
            return {"description": description, "context": ctx, "model": self.vision_model}
        except Exception as exc:
            return {"description": heuristic, "context": ctx, "model": None, "error": str(exc)}

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
        if self.dry_run:
            return {"ok": True, "dry_run": True, "command": "ik_move_ee_pose", "increment": inc, "arm": side}
        info = self._get_arm_sdk().ik_move_EE(inc, arm=side, mirror=bool(mirror))
        return {"ok": bool(info.get("success")), "arm": side, "increment": inc, "result": info}

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


def make_handler(vla: NaiveVLA) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "NaiveVLA/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[web] {self.address_string()} {fmt % args}", file=sys.stderr)

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
            self.wfile.write(body)

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
            self.wfile.write(data)

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
            self.wfile.write(body)

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
          <button onclick="act('move_for',{duration_s:1.0,vx:0.2})">Forward</button>
          <button onclick="act('stop',{})">Stop</button>
          <button onclick="act('move_for',{duration_s:1.0,vx:-0.15})">Back</button>
          <button onclick="act('move_for',{duration_s:0.8,vy:0.15})">Left</button>
          <button onclick="act('hand_shake',{release_after_s:2.0})">Handshake</button>
          <button onclick="act('move_for',{duration_s:0.8,vy:-0.15})">Right</button>
          <button class="wide" onclick="act('ik_move_ee_pose',{arm:'right',dx:0.04})">Right EE +X</button>
          <button class="wide" onclick="act('release_arms',{duration_s:3.0})">Release Arms</button>
          <button class="wide" onclick="describe()">Describe Visual Context</button>
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
    }
    async function act(tool, args) {
      const r = await fetch('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({tool, args})});
      document.getElementById('result').textContent = JSON.stringify(await r.json(), null, 2);
      refreshContext();
    }
    async function describe() {
      const r = await fetch('/api/describe', {cache:'no-store'});
      document.getElementById('result').textContent = JSON.stringify(await r.json(), null, 2);
    }
    async function search() {
      const query = document.getElementById('searchq').value;
      await act('web_search', {query, max_results: 5});
    }
    setInterval(refreshImages, 250);
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
    parser.add_argument("--iface", default="eth0", help="DDS interface for robot SDK commands.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID.")
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
    parser.add_argument("--describe-once", action="store_true", help="Print one visual description, then exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    )

    if args.describe_once:
        time.sleep(0.5)
        print(json.dumps(vla.describe_visual_context(), ensure_ascii=False, indent=2))
        receiver.stop()
        return 0

    server = ThreadingHTTPServer((str(args.web_host), int(args.web_port)), make_handler(vla))
    print(f"Naive VLA web UI: http://{args.web_host}:{args.web_port}")
    print(f"RGBD source: {receiver.endpoint}")
    print(f"Robot commands: {'disabled (--dry-run)' if args.dry_run else 'enabled'}")
    print(f"Diagnostics: {'disabled' if args.no_diagnostics else 'enabled (allowlisted)'}")
    print(f"Web search: {'disabled' if args.no_web_search else 'enabled'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        receiver.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
