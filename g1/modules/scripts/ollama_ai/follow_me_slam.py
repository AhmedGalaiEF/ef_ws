#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

if "--dash-worker" not in sys.argv:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
else:
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment,misc]
    String = None  # type: ignore[assignment]

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
G1_DIR = SCRIPT_DIR.parents[2] if SCRIPT_DIR.name == "ollama_ai" else SCRIPT_DIR.parent
MODULES_DIR = G1_DIR / "modules"
SCRIPTS_DIR = MODULES_DIR / "scripts"
if not (SCRIPTS_DIR / "slam_web_app.py").exists():
    G1_DIR = Path("/home/unitree/EF/ef_ws_clean/ef_ws/g1")
    MODULES_DIR = G1_DIR / "modules"
    SCRIPTS_DIR = MODULES_DIR / "scripts"
for path in (MODULES_DIR, SCRIPTS_DIR):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sdk_slam import SlamOdomSubscriber  # noqa: E402
from slam_web_app import (  # noqa: E402
    DEFAULT_TOPICS,
    LAYER_STYLE,
    PoseTarget,
    SlamWebState,
    decode_xyz,
    response_dict,
)


# ---------------------------------------------------------------------------
# Shared vocabulary / small helpers (used by both processes)
# ---------------------------------------------------------------------------

TRIGGER_PHRASES = ("follow me", "come follow me", "can you follow me", "please follow me")
CONFIRM_YES_PHRASES = ("yes", "yeah", "yep", "yup", "correct", "that's me", "that is me", "affirmative")
CONFIRM_NO_PHRASES = ("no", "nope", "not me", "negative", "wrong")
START_PHRASES = ("start", "begin", "start following", "start follow", "go ahead")
STOP_PHRASES = ("stop", "stop following", "stop follow", "halt", "cancel", "cancel follow")

PHASE_IDLE = "idle"
PHASE_RECOGNIZING = "recognizing"
PHASE_AWAITING_CONFIRMATION = "awaiting_confirmation"
PHASE_CONFIRMED = "confirmed"
PHASE_FOLLOWING = "following"
PHASE_SEARCHING = "searching"

FOLLOW_STABLE_HITS = 3
CLUSTER_CELL_M = 0.18
CLUSTER_MIN_POINTS = 14


def compact_text(text: str) -> str:
    return " ".join(str(text).split())


def decode_payload(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    return {"text": raw}


def matches_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    low = re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip()
    if not low:
        return False
    words = low.split()
    compact = " ".join(words)
    for phrase in phrases:
        if " " in phrase:
            if phrase in compact:
                return True
        elif phrase in words:
            return True
    return False


# ---------------------------------------------------------------------------
# Person detection: grid-bucket flood-fill clustering over a lidar point cloud
# (ported from the lidar clustering used elsewhere in this codebase for
# follow-me, applied directly to the same sensor-frame clouds the 3D view
# renders instead of a separate lidar RPC).
# ---------------------------------------------------------------------------

def cluster_person(
    points_xyz: np.ndarray,
    *,
    radius_m: float,
    front_max_y_m: float,
) -> Optional[tuple[float, float, dict[str, Any]]]:
    if points_xyz.size == 0:
        return None
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    mask = (
        np.isfinite(points_xyz).all(axis=1)
        & (x >= 0.35)
        & (x <= float(radius_m))
        & (np.abs(y) <= float(front_max_y_m))
        & (z >= -0.45)
        & (z <= 1.9)
    )
    pts = points_xyz[mask]
    if pts.shape[0] < CLUSTER_MIN_POINTS:
        return None

    cells = np.floor(pts[:, :2] / CLUSTER_CELL_M).astype(np.int32)
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, key in enumerate(cells):
        buckets.setdefault((int(key[0]), int(key[1])), []).append(idx)

    seen: set[tuple[int, int]] = set()
    clusters: list[list[int]] = []
    for key in buckets:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        indices: list[int] = []
        while stack:
            cur = stack.pop()
            indices.extend(buckets.get(cur, []))
            cx, cy = cur
            for nb in (
                (cx - 1, cy - 1), (cx - 1, cy), (cx - 1, cy + 1),
                (cx, cy - 1), (cx, cy + 1),
                (cx + 1, cy - 1), (cx + 1, cy), (cx + 1, cy + 1),
            ):
                if nb in buckets and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        clusters.append(indices)

    best: Optional[tuple[float, float, dict[str, Any]]] = None
    best_cost = float("inf")
    for indices in clusters:
        if len(indices) < CLUSTER_MIN_POINTS:
            continue
        cpts = pts[np.asarray(indices, dtype=np.int64)]
        xs, ys, zs = cpts[:, 0], cpts[:, 1], cpts[:, 2]
        width_x = float(np.percentile(xs, 90) - np.percentile(xs, 10))
        width_y = float(np.percentile(ys, 90) - np.percentile(ys, 10))
        height_z = float(np.percentile(zs, 90) - np.percentile(zs, 10))
        if width_x > 1.2 or width_y > 1.15:
            continue
        if height_z < 0.18 and len(indices) < 35:
            continue
        tx = float(np.median(xs))
        ty = float(np.median(ys))
        cost = tx + 0.75 * abs(ty) - min(0.35, len(indices) / 200.0)
        if cost < best_cost:
            best_cost = cost
            best = (
                tx,
                ty,
                {
                    "points": len(indices),
                    "width_x_m": round(width_x, 3),
                    "width_y_m": round(width_y, 3),
                    "height_z_m": round(height_z, 3),
                },
            )
    return best


def predicted_relative_target(rel_x: float, rel_y: float, target_distance_m: float) -> tuple[float, float, float]:
    dist = math.hypot(rel_x, rel_y)
    if dist <= float(target_distance_m) or dist < 1e-6:
        tx, ty = 0.0, 0.0
    else:
        scale = (dist - float(target_distance_m)) / dist
        tx, ty = rel_x * scale, rel_y * scale
    face_yaw = math.atan2(rel_y, rel_x)
    return tx, ty, face_yaw


def world_pose_from_relative(robot_pose: PoseTarget, rel_x: float, rel_y: float, rel_yaw: float) -> PoseTarget:
    c = math.cos(robot_pose.yaw)
    s = math.sin(robot_pose.yaw)
    wx = robot_pose.x + c * rel_x - s * rel_y
    wy = robot_pose.y + s * rel_x + c * rel_y
    wyaw = robot_pose.yaw + rel_yaw
    return PoseTarget(x=wx, y=wy, yaw=wyaw, z=robot_pose.z)


class TargetPredictor:
    """EMA position + finite-difference velocity lead: the 'dynamically predicted pose'."""

    def __init__(self, alpha: float = 0.4, lead_s: float = 0.35) -> None:
        self.alpha = float(alpha)
        self.lead_s = float(lead_s)
        self._ema: Optional[tuple[float, float]] = None
        self._last_raw: Optional[tuple[float, float]] = None
        self._last_ts: Optional[float] = None
        self._velocity: tuple[float, float] = (0.0, 0.0)

    def reset(self) -> None:
        self._ema = None
        self._last_raw = None
        self._last_ts = None
        self._velocity = (0.0, 0.0)

    def update(self, rel_x: float, rel_y: float) -> tuple[float, float]:
        now = time.time()
        if self._ema is None:
            self._ema = (rel_x, rel_y)
        else:
            a = self.alpha
            self._ema = (a * rel_x + (1.0 - a) * self._ema[0], a * rel_y + (1.0 - a) * self._ema[1])
        if self._last_raw is not None and self._last_ts is not None:
            dt = max(1e-3, now - self._last_ts)
            vx = (rel_x - self._last_raw[0]) / dt
            vy = (rel_y - self._last_raw[1]) / dt
            bv = 0.5
            self._velocity = (bv * vx + (1.0 - bv) * self._velocity[0], bv * vy + (1.0 - bv) * self._velocity[1])
        self._last_raw = (rel_x, rel_y)
        self._last_ts = now
        ex, ey = self._ema
        vx, vy = self._velocity
        return (ex + vx * self.lead_s, ey + vy * self.lead_s)


class Speaker:
    def __init__(self, args: argparse.Namespace, logger: Any) -> None:
        self.args = args
        self.logger = logger
        self.lock = threading.Lock()

    def say(self, text: str) -> int:
        text = compact_text(text)
        if not text:
            return 0
        self.logger.info(f"follow-me says: {text!r}")
        if getattr(self.args, "no_speech", False):
            return 0
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "robot_say_once.py"),
            text,
            "--iface", str(self.args.iface),
            "--domain-id", str(int(self.args.domain_id)),
        ]
        if getattr(self.args, "volume", None) is not None:
            command.extend(["--volume", str(int(self.args.volume))])
        if getattr(self.args, "tts_language", None):
            command.extend(["--language", str(self.args.tts_language)])
        env = os.environ.copy()
        env.setdefault("CYCLONEDDS_HOME", "/home/unitree/cyclonedds_ws/install/cyclonedds")
        env.setdefault("CYCLONEDDS_URI", "/home/unitree/cyclonedds_ws/cyclonedds.xml")
        with self.lock:
            proc = subprocess.Popen(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, _ = proc.communicate()
        if output and output.strip():
            self.logger.info(output.strip())
        return int(proc.returncode or 0)

    def say_async(self, text: str) -> threading.Thread:
        thread = threading.Thread(target=self.say, args=(text,), daemon=True)
        thread.start()
        return thread


class PrintLogger:
    def info(self, message: str) -> None:
        print(message, flush=True)

    def warning(self, message: str) -> None:
        print(f"WARNING: {message}", file=sys.stderr, flush=True)

    def error(self, message: str) -> None:
        print(f"ERROR: {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Dash-worker: owns the SDK/DDS connection, the 3D point cloud, detection,
# and the pose-nav follow loop. No rclpy in this process.
# ---------------------------------------------------------------------------

class FollowSlamState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.web = SlamWebState(args.iface, args.domain_id, DEFAULT_TOPICS, args.map_path)
        self.odom = SlamOdomSubscriber()
        self.odom.start()
        self.speaker = Speaker(args, PrintLogger())
        self.lock = threading.RLock()
        self.phase = PHASE_IDLE
        self.last_event = ""
        self.last_spoken = ""
        self.last_error = ""
        self.person_rel: Optional[tuple[float, float]] = None
        self.person_meta: dict[str, Any] = {}
        self.predicted_rel: Optional[tuple[float, float]] = None
        self.nav_target: Optional[PoseTarget] = None
        self.confirm_hits = 0
        self._predictor = TargetPredictor(lead_s=float(args.predict_lead_s))
        self._last_commanded: Optional[PoseTarget] = None
        self._last_command_ts = 0.0
        self._last_detect_ts = 0.0
        self._searching_since: Optional[float] = None
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._loop, daemon=True, name="follow-me-slam-loop")
        self._worker.start()

    def odom_pose(self) -> Optional[PoseTarget]:
        pose = self.odom.get_pose()
        if pose is None:
            return None
        x, y, yaw = pose
        return PoseTarget(x=x, y=y, yaw=yaw)

    def robot_pose(self) -> Optional[PoseTarget]:
        return self.odom_pose() or self.web.current_pose()

    def current_points(self) -> np.ndarray:
        frames: list[np.ndarray] = []
        for name in ("deskewed", "livox"):
            sub = self.web.subs.get(name)
            if sub is None:
                continue
            msg, _ts, _count = sub.latest()
            if msg is None:
                continue
            pts = decode_xyz(msg, stride=1, max_points=6000, z_min=-1.0, z_max=2.4)
            if pts.size:
                frames.append(pts)
        if not frames:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(frames, axis=0)

    def _speak_async(self, text: str) -> None:
        with self.lock:
            self.last_spoken = text
        self.speaker.say_async(text)

    def _ensure_relocated(self) -> bool:
        if self.web.relocation_ready:
            return True
        result = self.web.relocate(self.web.map_path)
        return bool(result.get("ok"))

    def handle_voice_event(self, event: str) -> dict[str, Any]:
        event = str(event or "").strip().lower()
        with self.lock:
            self.last_event = event
            phase = self.phase
            if event == "trigger_follow" and phase == PHASE_IDLE:
                self.phase = PHASE_RECOGNIZING
                self.confirm_hits = 0
                self._predictor.reset()
                self._speak_async("Looking for you.")
            elif event == "confirm_yes" and phase == PHASE_AWAITING_CONFIRMATION:
                self.phase = PHASE_CONFIRMED
                self._speak_async("Say start when you're ready.")
            elif event == "confirm_no" and phase == PHASE_AWAITING_CONFIRMATION:
                self.phase = PHASE_RECOGNIZING
                self.confirm_hits = 0
                self._speak_async("Let me look again.")
            elif event == "start" and phase == PHASE_CONFIRMED:
                if self._ensure_relocated():
                    self.phase = PHASE_FOLLOWING
                    self._last_commanded = None
                    self._last_command_ts = 0.0
                    self._searching_since = None
                    self._speak_async("Following.")
                else:
                    self._speak_async("I cannot navigate yet; relocation is not ready.")
            elif event == "stop" and phase != PHASE_IDLE:
                self.web.pause()
                self.phase = PHASE_IDLE
                self.person_rel = None
                self.predicted_rel = None
                self.nav_target = None
                self._speak_async("Stopped following.")
            return self._snapshot_locked()

    def _detect_tick(self, phase: str) -> None:
        points = self.current_points()
        result = cluster_person(
            points,
            radius_m=float(self.args.detect_radius_m),
            front_max_y_m=float(self.args.detect_front_max_y_m),
        )
        now = time.time()
        with self.lock:
            if result is None:
                self.person_rel = None
                self.person_meta = {}
                if phase == PHASE_RECOGNIZING:
                    self.confirm_hits = 0
                return
            rel_x, rel_y, meta = result
            self.person_rel = (rel_x, rel_y)
            self.person_meta = meta
            self._last_detect_ts = now
            if phase == PHASE_RECOGNIZING:
                self.confirm_hits += 1
                if self.confirm_hits >= FOLLOW_STABLE_HITS:
                    self.phase = PHASE_AWAITING_CONFIRMATION
                    self.confirm_hits = 0
                    self._speak_async("Are you in front of me?")

    def _maybe_send_nav(self, target: PoseTarget) -> None:
        now = time.time()
        with self.lock:
            last = self._last_commanded
            last_ts = self._last_command_ts
        moved = last is None or target.xy_distance_to(last) > float(self.args.resend_distance_m)
        stale = (now - last_ts) > float(self.args.resend_interval_s)
        if not (moved or stale):
            return
        qx, qy, qz, qw = target.quaternion()
        result = response_dict(self.web.client.pose_nav(target.x, target.y, target.z, qx, qy, qz, qw, mode=1))
        with self.lock:
            self._last_commanded = target
            self._last_command_ts = now
            self.web.last_action = {"label": "follow_pose_nav", **result, "stamp": now}

    def _follow_tick(self) -> None:
        with self.lock:
            person_rel = self.person_rel
            last_detect_ts = self._last_detect_ts
        now = time.time()
        if person_rel is None:
            age = (now - last_detect_ts) if last_detect_ts else float("inf")
            with self.lock:
                if self._searching_since is None:
                    self._searching_since = now
                    self.phase = PHASE_SEARCHING
                lost_for = now - self._searching_since
            if age > float(self.args.lost_timeout_s):
                self.web.pause()
            if lost_for > float(self.args.lost_abort_s):
                with self.lock:
                    self.phase = PHASE_IDLE
                    self.nav_target = None
                self._speak_async("I lost track of you.")
            return
        with self.lock:
            if self.phase == PHASE_SEARCHING:
                self.phase = PHASE_FOLLOWING
            self._searching_since = None
        rel_x, rel_y = person_rel
        pred_x, pred_y = self._predictor.update(rel_x, rel_y)
        tx, ty, tyaw = predicted_relative_target(pred_x, pred_y, float(self.args.target_distance_m))
        robot_pose = self.robot_pose()
        if robot_pose is None:
            return
        target = world_pose_from_relative(robot_pose, tx, ty, tyaw)
        with self.lock:
            self.predicted_rel = (pred_x, pred_y)
            self.nav_target = target
        self._maybe_send_nav(target)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.time()
            try:
                with self.lock:
                    phase = self.phase
                if phase in (PHASE_RECOGNIZING, PHASE_AWAITING_CONFIRMATION, PHASE_FOLLOWING, PHASE_SEARCHING):
                    self._detect_tick(phase)
                with self.lock:
                    phase = self.phase
                if phase in (PHASE_FOLLOWING, PHASE_SEARCHING):
                    self._follow_tick()
            except Exception as exc:
                with self.lock:
                    self.last_error = str(exc)
            elapsed = time.time() - started
            time.sleep(max(0.05, float(self.args.follow_loop_s) - elapsed))

    def _snapshot_locked(self) -> dict[str, Any]:
        pose = self.robot_pose()
        return {
            "phase": self.phase,
            "last_event": self.last_event,
            "last_spoken": self.last_spoken,
            "last_error": self.last_error,
            "person_rel": self.person_rel,
            "person_meta": self.person_meta,
            "predicted_rel": self.predicted_rel,
            "nav_target": None if self.nav_target is None else {
                "x": self.nav_target.x, "y": self.nav_target.y, "yaw": self.nav_target.yaw,
            },
            "robot_pose": None if pose is None else {"x": pose.x, "y": pose.y, "yaw": pose.yaw},
            "relocation_ready": self.web.relocation_ready,
            "slam_running": self.web.slam_running,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self._snapshot_locked()


def make_figure_3d(state: FollowSlamState, selected_layers: list[str], max_points: int) -> Any:
    import plotly.graph_objects as go

    fig = go.Figure()
    web = state.web
    for name, sub in web.subs.items():
        if name not in selected_layers:
            continue
        msg, ts, _count = sub.latest()
        if msg is None:
            continue
        stride = 1 if name.startswith("slam") else 4
        pts = decode_xyz(msg, stride=stride, max_points=max_points, z_min=-1.0, z_max=2.4)
        if pts.size == 0:
            continue
        label, color, size, opacity = LAYER_STYLE.get(name, (name, "#ffffff", 2, 0.6))
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                name=f"{label} ({time.time() - ts:.1f}s)",
                marker={"size": size, "color": color, "opacity": opacity},
            )
        )

    pose = state.robot_pose()
    if pose is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[pose.x], y=[pose.y], z=[pose.z],
                mode="markers",
                name="Robot pose (odometry)",
                marker={"size": 8, "color": "#ff3154", "symbol": "diamond"},
            )
        )
        radius = float(state.args.detect_radius_m)
        theta = np.linspace(0.0, 2.0 * np.pi, 48)
        ring_x = pose.x + radius * np.cos(theta)
        ring_y = pose.y + radius * np.sin(theta)
        fig.add_trace(
            go.Scatter3d(
                x=ring_x, y=ring_y, z=np.full_like(ring_x, pose.z),
                mode="lines",
                name="Detection radius",
                line={"color": "#3d5a80", "width": 3},
            )
        )

    with state.lock:
        person_rel = state.person_rel
        nav_target = state.nav_target

    if person_rel is not None and pose is not None:
        px, py = person_rel
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        wx = pose.x + c * px - s * py
        wy = pose.y + s * px + c * py
        fig.add_trace(
            go.Scatter3d(
                x=[wx], y=[wy], z=[pose.z],
                mode="markers",
                name="Detected person",
                marker={"size": 9, "color": "#37e05f", "symbol": "circle"},
            )
        )

    if nav_target is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[nav_target.x], y=[nav_target.y], z=[nav_target.z],
                mode="markers",
                name="Predicted nav target",
                marker={"size": 9, "color": "#00e5ff", "symbol": "x"},
            )
        )

    fig.update_layout(
        template="plotly_dark",
        uirevision="follow-me-slam-3d",
        margin={"l": 0, "r": 0, "t": 28, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        paper_bgcolor="#111318",
        scene={
            "xaxis": {"title": "x (m)", "backgroundcolor": "#171a21", "gridcolor": "#303642"},
            "yaxis": {"title": "y (m)", "backgroundcolor": "#171a21", "gridcolor": "#303642"},
            "zaxis": {"title": "z (m)", "backgroundcolor": "#171a21", "gridcolor": "#303642"},
            "aspectmode": "data",
        },
        height=760,
    )
    return fig


CSS = """
body { margin: 0; background: #111318; color: #e8ecf3; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.app { min-height: 100vh; }
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 12px 18px; background: #171a21; border-bottom: 1px solid #2b303b; }
h1 { font-size: 20px; line-height: 1.1; margin: 0 0 5px; font-weight: 650; }
.status-line { font-size: 12px; color: #aeb7c2; min-height: 16px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
button { background: #252b36; color: #f6f8fb; border: 1px solid #3b4352; border-radius: 6px; padding: 8px 10px; font-size: 13px; cursor: pointer; }
button:hover { background: #303849; }
button.stop { background: #8a2d2b; border-color: #aa3b38; }
.main { display: grid; grid-template-columns: 310px minmax(0, 1fr); min-height: calc(100vh - 72px); }
.side { padding: 14px; border-right: 1px solid #2b303b; background: #14171d; overflow: auto; }
.map-pane { min-width: 0; padding: 10px 12px 0; }
label { display: block; color: #e9eef7; font-size: 12px; margin: 13px 0 6px; }
input { width: 100%; box-sizing: border-box; background: #f7f9fc; color: #111827; border: 1px solid #596579; border-radius: 6px; padding: 8px; }
.side, .side * { color: #edf2fb; }
.side input, .side textarea { color: #111827; background: #f7f9fc; }
.layers label, .dash-checklist label { margin: 7px 0; color: #edf2fb !important; }
.rc-slider-mark-text, .rc-slider-tooltip-inner { color: #f8fafc !important; }
.rc-slider-track { background-color: #55c7ff; }
.rc-slider-handle { border-color: #55c7ff; background: #f8fafc; }
.follow-buttons { display: flex; flex-direction: column; gap: 8px; margin-top: 6px; }
.action-output { margin-top: 14px; padding: 10px; background: #0f1218; border: 1px solid #2b303b; border-radius: 6px; font-size: 12px; white-space: pre-wrap; }
"""


def app_layout(state: FollowSlamState):
    from dash import dcc, html

    layer_options = [
        {"label": LAYER_STYLE.get(name, (name,))[0], "value": name}
        for name in DEFAULT_TOPICS.keys()
    ]
    default_layers = ["slam_mapping", "slam_relocation", "deskewed"]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [html.H1("G1 Follow-Me (3D SLAM)"), html.Div(id="status-line", className="status-line")],
                        className="title-block",
                    ),
                    html.Div(
                        [
                            html.Button("Start Mapping", id="btn-start-mapping", n_clicks=0),
                            html.Button("Relocate", id="btn-relocate", n_clicks=0),
                            html.Button("Stop SLAM", id="btn-stop-slam", n_clicks=0),
                        ],
                        className="toolbar",
                    ),
                ],
                className="topbar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Map path"),
                            dcc.Input(id="map-path", value=state.web.map_path, type="text", debounce=True),
                            html.Label("Layers"),
                            dcc.Checklist(id="layers", options=layer_options, value=default_layers, className="layers"),
                            html.Label("Max points per layer"),
                            dcc.Slider(id="max-points", min=200, max=20000, step=200, value=4000,
                                       marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Label("Detection radius (m)"),
                            dcc.Slider(id="detect-radius", min=1.0, max=8.0, step=0.5,
                                       value=float(state.args.detect_radius_m),
                                       marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Label("Target distance (m)"),
                            dcc.Slider(id="target-distance", min=0.5, max=3.0, step=0.1,
                                       value=float(state.args.target_distance_m),
                                       marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Hr(),
                            html.Div(
                                [
                                    html.Button("Start Recognition", id="btn-follow-trigger", n_clicks=0),
                                    html.Button("Confirm Yes", id="btn-confirm-yes", n_clicks=0),
                                    html.Button("Confirm No", id="btn-confirm-no", n_clicks=0),
                                    html.Button("Begin Following", id="btn-start-following", n_clicks=0),
                                    html.Button("Stop Follow", id="btn-stop-follow", n_clicks=0, className="stop"),
                                ],
                                className="follow-buttons",
                            ),
                            html.Div(id="follow-output", className="action-output"),
                        ],
                        className="side",
                    ),
                    html.Div(
                        [dcc.Graph(id="map-graph-3d", config={"displayModeBar": True}, style={"height": "760px"})],
                        className="map-pane",
                    ),
                ],
                className="main",
            ),
            dcc.Interval(id="map-interval", interval=max(250, int(state.args.refresh_ms)), n_intervals=0),
        ],
        className="app",
    )


def create_dash_app(state: FollowSlamState):
    import dash
    from dash import Input, Output, State as DashState
    from flask import jsonify, request

    app = dash.Dash(__name__)
    app.index_string = f"""<!DOCTYPE html>
<html>
  <head>{{%metas%}}<title>G1 Follow-Me 3D</title>{{%favicon%}}{{%css%}}<style>{CSS}</style></head>
  <body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body>
</html>"""
    app.layout = app_layout(state)

    @app.callback(
        Output("map-graph-3d", "figure"),
        Output("status-line", "children"),
        Input("map-interval", "n_intervals"),
        Input("layers", "value"),
        Input("max-points", "value"),
        Input("detect-radius", "value"),
        Input("target-distance", "value"),
    )
    def update(_n: int, layers: list[str], max_points: int, detect_radius: float, target_distance: float):
        if detect_radius is not None:
            state.args.detect_radius_m = float(detect_radius)
        if target_distance is not None:
            state.args.target_distance_m = float(target_distance)
        fig = make_figure_3d(state, layers or [], int(max_points or 4000))
        snap = state.snapshot()
        line = (
            f"phase={snap['phase']} event={snap['last_event']} spoken={snap['last_spoken']!r} "
            f"relocation={'ready' if snap['relocation_ready'] else 'not ready'} "
            f"person={snap['person_rel']} target={snap['nav_target']}"
        )
        return fig, line

    @app.callback(
        Output("follow-output", "children"),
        Input("btn-follow-trigger", "n_clicks"),
        Input("btn-confirm-yes", "n_clicks"),
        Input("btn-confirm-no", "n_clicks"),
        Input("btn-start-following", "n_clicks"),
        Input("btn-stop-follow", "n_clicks"),
        prevent_initial_call=True,
    )
    def follow_action(_a, _b, _c, _d, _e):
        event_map = {
            "btn-follow-trigger": "trigger_follow",
            "btn-confirm-yes": "confirm_yes",
            "btn-confirm-no": "confirm_no",
            "btn-start-following": "start",
            "btn-stop-follow": "stop",
        }
        event = event_map.get(dash.ctx.triggered_id)
        if event is None:
            return dash.no_update
        result = state.handle_voice_event(event)
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @app.callback(
        Output("follow-output", "children", allow_duplicate=True),
        Input("btn-start-mapping", "n_clicks"),
        Input("btn-relocate", "n_clicks"),
        Input("btn-stop-slam", "n_clicks"),
        DashState("map-path", "value"),
        prevent_initial_call=True,
    )
    def slam_action(_start, _reloc, _stop, map_path: str):
        trigger = dash.ctx.triggered_id
        if trigger == "btn-start-mapping":
            result = state.web.start_mapping("indoor")
        elif trigger == "btn-relocate":
            result = state.web.relocate(str(map_path or state.web.map_path))
        elif trigger == "btn-stop-slam":
            result = state.web.stop_slam()
        else:
            result = state.web.last_action
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @app.server.route("/api/voice_event", methods=["POST"])
    def voice_event_route():
        payload = request.get_json(silent=True) or {}
        event = str(payload.get("event", "")).strip().lower()
        if not event:
            return jsonify({"ok": False, "error": "missing 'event'"}), 400
        result = state.handle_voice_event(event)
        return jsonify({"ok": True, **result})

    @app.server.route("/api/status", methods=["GET"])
    def status_route():
        return jsonify(state.snapshot())

    return app


def run_dash_worker(args: argparse.Namespace) -> None:
    state = FollowSlamState(args)
    app = create_dash_app(state)
    print(f"Follow-me SLAM web app: http://{args.dash_host}:{args.dash_port}", flush=True)
    app.run(host=args.dash_host, port=args.dash_port, debug=False)


# ---------------------------------------------------------------------------
# ASR bridge (default mode, rclpy): only understands the fixed follow-me
# vocabulary and forwards matched events to the dash-worker subprocess over
# HTTP. Never runs unitree_sdk2py's ChannelFactoryInitialize itself.
# ---------------------------------------------------------------------------

class VoiceBridgeNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("follow_me_slam_bridge")
        self.args = args
        self.speaker = Speaker(args, self.get_logger())
        self.dash_proc: subprocess.Popen[str] | None = None
        self.external_asr_httpd: http.server.ThreadingHTTPServer | None = None
        self.external_asr_thread: threading.Thread | None = None
        self.response_pub = self.create_publisher(String, args.response_topic, 10)
        self._start_dash_worker()
        self.create_subscription(String, args.audio_topic, self.on_audio, 10)
        if args.filtered_audio_topic and str(args.filtered_audio_topic) != str(args.audio_topic):
            self.create_subscription(String, args.filtered_audio_topic, self.on_audio, 10)
        self.create_subscription(String, args.command_topic, self.on_command, 10)
        if bool(args.external_asr_server):
            self._start_external_asr_server()
        self.get_logger().info(
            f"follow-me-slam bridge ready; dash worker at http://127.0.0.1:{int(args.dash_port)}"
        )

    def _start_dash_worker(self) -> None:
        command = [
            sys.executable, str(Path(__file__).resolve()), "--dash-worker",
            "--iface", str(self.args.iface),
            "--domain-id", str(int(self.args.domain_id)),
            "--map-path", str(self.args.map_path),
            "--dash-host", str(self.args.dash_host),
            "--dash-port", str(int(self.args.dash_port)),
            "--detect-radius-m", str(float(self.args.detect_radius_m)),
            "--detect-front-max-y-m", str(float(self.args.detect_front_max_y_m)),
            "--target-distance-m", str(float(self.args.target_distance_m)),
            "--predict-lead-s", str(float(self.args.predict_lead_s)),
            "--follow-loop-s", str(float(self.args.follow_loop_s)),
            "--resend-distance-m", str(float(self.args.resend_distance_m)),
            "--resend-interval-s", str(float(self.args.resend_interval_s)),
            "--lost-timeout-s", str(float(self.args.lost_timeout_s)),
            "--lost-abort-s", str(float(self.args.lost_abort_s)),
            "--refresh-ms", str(int(self.args.refresh_ms)),
        ]
        if self.args.no_speech:
            command.append("--no-speech")
        if self.args.volume is not None:
            command.extend(["--volume", str(int(self.args.volume))])
        if self.args.tts_language:
            command.extend(["--tts-language", str(self.args.tts_language)])
        env = os.environ.copy()
        env.setdefault("CYCLONEDDS_HOME", "/home/unitree/cyclonedds_ws/install/cyclonedds")
        env.setdefault("CYCLONEDDS_URI", "/home/unitree/cyclonedds_ws/cyclonedds.xml")
        self.dash_proc = subprocess.Popen(
            command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        threading.Thread(target=self._log_dash_output, daemon=True).start()
        self.get_logger().info(f"Started follow-me dash worker pid={self.dash_proc.pid}")

    def _log_dash_output(self) -> None:
        proc = self.dash_proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            text = line.strip()
            if text:
                self.get_logger().info(f"[dash] {text}")

    def _post_event(self, event: str) -> dict[str, Any]:
        url = f"http://127.0.0.1:{int(self.args.dash_port)}/api/voice_event"
        data = json.dumps({"event": event}).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            self.get_logger().warning(f"Could not reach dash worker: {exc}")
            return {"ok": False, "error": str(exc)}

    def on_audio(self, msg: String) -> None:
        payload = decode_payload(str(msg.data))
        self._handle_text(str(payload.get("text", payload.get("raw", ""))), "audio")

    def on_command(self, msg: String) -> None:
        payload = decode_payload(str(msg.data))
        self._handle_text(str(payload.get("text", payload.get("prompt", ""))), "command")

    def submit_external_asr(self, text: str) -> bool:
        return self._handle_text(text, "external_asr")

    def _handle_text(self, text: str, source: str) -> bool:
        text = compact_text(text)
        if not text:
            return False
        event: str | None = None
        if matches_phrase(text, STOP_PHRASES):
            event = "stop"
        elif matches_phrase(text, TRIGGER_PHRASES):
            event = "trigger_follow"
        elif matches_phrase(text, START_PHRASES):
            event = "start"
        elif matches_phrase(text, CONFIRM_YES_PHRASES):
            event = "confirm_yes"
        elif matches_phrase(text, CONFIRM_NO_PHRASES):
            event = "confirm_no"
        if event is None:
            self.get_logger().info(f"{source} ignored (not follow-me vocabulary): {text!r}")
            return False
        self.get_logger().info(f"{source} matched event={event}: {text!r}")
        result = self._post_event(event)
        self.response_pub.publish(String(data=json.dumps({"text": text, "event": event, "result": result}, default=str)))
        return True

    def _start_external_asr_server(self) -> None:
        node = self
        token = str(self.args.external_asr_token or "")

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *fmt_args: Any) -> None:
                return

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self, payload: dict[str, Any] | None) -> bool:
                if not token:
                    return True
                auth = str(self.headers.get("Authorization", ""))
                if auth == f"Bearer {token}":
                    return True
                return bool(isinstance(payload, dict) and str(payload.get("token", "")) == token)

            def do_GET(self) -> None:
                if self.path.split("?", 1)[0] == "/health":
                    self._send_json(200, {"ok": True, "service": "follow_me_slam_asr"})
                    return
                self._send_json(404, {"ok": False, "error": "not_found"})

            def do_POST(self) -> None:
                if self.path.split("?", 1)[0] not in {"/asr", "/command"}:
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
                        text = str(parsed.get("text", parsed.get("prompt", "")))
                except Exception:
                    payload = None
                if not self._authorized(payload):
                    self._send_json(401, {"ok": False, "error": "unauthorized"})
                    return
                accepted = node.submit_external_asr(text)
                self._send_json(200, {"ok": True, "accepted": accepted, "text": compact_text(text)})

        host = str(self.args.external_asr_host)
        port = int(self.args.external_asr_port)
        self.external_asr_httpd = http.server.ThreadingHTTPServer((host, port), Handler)
        self.external_asr_thread = threading.Thread(target=self.external_asr_httpd.serve_forever, daemon=True)
        self.external_asr_thread.start()
        self.get_logger().info(f"external ASR endpoint listening on http://{host}:{port}/asr")

    def destroy_node(self) -> bool:  # type: ignore[override]
        if self.external_asr_httpd is not None:
            self.external_asr_httpd.shutdown()
            self.external_asr_httpd.server_close()
        if self.dash_proc is not None:
            try:
                self.dash_proc.terminate()
                self.dash_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.dash_proc.kill()
            except Exception:
                pass
        return super().destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G1 follow-me: 3D SLAM point-cloud webapp with voice-gated pose-nav following."
    )
    parser.add_argument("--iface", default=os.environ.get("G1_IFACE", "eth0"))
    parser.add_argument("--domain-id", type=int, default=int(os.environ.get("G1_DOMAIN_ID", "0")))
    parser.add_argument("--map-path", default="/home/unitree/test.pcd")

    parser.add_argument("--dash-host", default="0.0.0.0")
    parser.add_argument("--dash-port", type=int, default=8098)
    parser.add_argument("--refresh-ms", type=int, default=700)

    parser.add_argument("--detect-radius-m", type=float, default=4.5)
    parser.add_argument("--detect-front-max-y-m", type=float, default=1.6)
    parser.add_argument("--target-distance-m", type=float, default=1.25)
    parser.add_argument("--predict-lead-s", type=float, default=0.35)
    parser.add_argument("--follow-loop-s", type=float, default=0.75)
    parser.add_argument("--resend-distance-m", type=float, default=0.35)
    parser.add_argument("--resend-interval-s", type=float, default=2.5)
    parser.add_argument("--lost-timeout-s", type=float, default=4.0)
    parser.add_argument("--lost-abort-s", type=float, default=20.0)

    parser.add_argument("--audio-topic", default="/audio_msg")
    parser.add_argument("--filtered-audio-topic", default="/audio_msg/filter")
    parser.add_argument("--command-topic", default="/model_api/chatbot_command")
    parser.add_argument("--response-topic", default="/model_api/follow_me_slam_response")
    parser.add_argument("--external-asr-server", action="store_true")
    parser.add_argument("--external-asr-host", default="0.0.0.0")
    parser.add_argument("--external-asr-port", type=int, default=8099)
    parser.add_argument("--external-asr-token", default="")
    parser.add_argument("--volume", type=int, default=None)
    parser.add_argument("--tts-language", default=None)
    parser.add_argument("--no-speech", action="store_true")

    parser.add_argument("--dash-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dash_worker:
        run_dash_worker(args)
        return 0

    node: VoiceBridgeNode | None = None
    try:
        rclpy.init()
        node = VoiceBridgeNode(args)
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
