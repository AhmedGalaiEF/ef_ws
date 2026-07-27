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
LEG_CELL_M = 0.12
LEG_MIN_POINTS = 8
TARGET_SIMILARITY_MIN = 0.52
TARGET_SIMILARITY_PROMPT_COOLDOWN_S = 6.0
YAW_DEADBAND_RAD = 0.22
RGB_MIN_PERSON_SCORE = 0.55
DEFAULT_CAMERA_X_M = 47.64571478 / 1000.0
DEFAULT_CAMERA_Z_M = 462.68178553 / 1000.0
DEFAULT_CAMERA_DOWN_PITCH_RAD = math.radians(90.0 - 42.0)


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


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def follow_cloud_signature(points: np.ndarray, meta: dict[str, Any]) -> dict[str, float]:
    signature: dict[str, float] = {
        "points": float(max(1, int(meta.get("points", points.shape[0] if points.size else 1)))),
        "width_x_m": _finite_float(meta.get("width_x_m"), 0.35),
        "width_y_m": _finite_float(meta.get("width_y_m"), 0.35),
        "height_z_m": _finite_float(meta.get("height_z_m"), 0.75),
        "source_code": 1.0 if str(meta.get("source", "")).startswith("leg") else 2.0,
    }
    if points.size:
        finite = points[np.isfinite(points).all(axis=1)]
        if finite.shape[0] >= 8:
            xs, ys, zs = finite[:, 0], finite[:, 1], finite[:, 2]
            signature.update({
                "points": float(max(1, finite.shape[0])),
                "width_x_m": max(0.03, float(np.percentile(xs, 90) - np.percentile(xs, 10))),
                "width_y_m": max(0.03, float(np.percentile(ys, 90) - np.percentile(ys, 10))),
                "height_z_m": max(0.05, float(np.percentile(zs, 90) - np.percentile(zs, 10))),
                "z_median": float(np.median(zs)),
            })
    return signature


def cluster_similarity(meta: dict[str, Any], reference: dict[str, float]) -> float:
    if not reference:
        return 1.0
    cand = follow_cloud_signature(np.empty((0, 3), dtype=np.float32), meta)
    cost = 0.0
    for key, weight in (("width_x_m", 1.0), ("width_y_m", 1.0), ("height_z_m", 1.15)):
        ref = max(0.03, _finite_float(reference.get(key), cand[key]))
        cur = max(0.03, _finite_float(cand.get(key), ref))
        cost += weight * abs(math.log(cur / ref))
    ref_points = max(1.0, _finite_float(reference.get("points"), cand["points"]))
    cur_points = max(1.0, _finite_float(cand.get("points"), ref_points))
    cost += 0.55 * abs(math.log(cur_points / ref_points))
    if int(reference.get("source_code", 0)) and int(cand.get("source_code", 0)) != int(reference.get("source_code", 0)):
        cost += 0.35
    return max(0.0, min(1.0, math.exp(-cost)))


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
    previous_rel: Optional[tuple[float, float]] = None,
    target_signature: Optional[dict[str, float]] = None,
) -> Optional[tuple[float, float, dict[str, Any]]]:
    candidates = cluster_person_candidates(
        points_xyz,
        radius_m=radius_m,
        front_max_y_m=front_max_y_m,
        previous_rel=previous_rel,
    )
    if not candidates:
        return None

    best: Optional[tuple[float, float, dict[str, Any]]] = None
    best_cost = float("inf")
    for rel_x, rel_y, raw_meta in candidates:
        meta = dict(raw_meta)
        cost = _finite_float(meta.get("_detect_cost"), rel_x + 0.75 * abs(rel_y))
        if target_signature:
            similarity = cluster_similarity(meta, target_signature)
            meta["target_similarity"] = round(similarity, 3)
            cost = 0.25 * cost + (1.0 - similarity) * (2.0 if previous_rel is not None else 1.35)
        if cost < best_cost:
            best_cost = cost
            meta.pop("_detect_cost", None)
            best = (rel_x, rel_y, meta)
    return best


def cluster_person_candidates(
    points_xyz: np.ndarray,
    *,
    radius_m: float,
    front_max_y_m: float,
    previous_rel: Optional[tuple[float, float]] = None,
) -> list[tuple[float, float, dict[str, Any]]]:
    if points_xyz.size == 0:
        return []
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
        return []

    candidates: list[tuple[float, float, dict[str, Any]]] = []
    leg_result = cluster_leg_person(
        pts,
        radius_m=radius_m,
        front_max_y_m=front_max_y_m,
        previous_rel=previous_rel,
    )
    if leg_result is not None:
        lx, ly, leg_meta = leg_result
        meta = dict(leg_meta)
        continuity = 0.0
        if previous_rel is not None:
            continuity = 0.8 * math.hypot(lx - previous_rel[0], ly - previous_rel[1])
        meta["_detect_cost"] = lx + 0.65 * abs(ly) + continuity - min(0.45, int(meta.get("points", 0)) / 120.0)
        candidates.append((lx, ly, meta))

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
        track_penalty = 0.0
        if previous_rel is not None:
            track_penalty = 0.55 * math.hypot(tx - previous_rel[0], ty - previous_rel[1])
        cost = tx + 0.75 * abs(ty) + track_penalty - min(0.35, len(indices) / 200.0)
        candidates.append((
            tx,
            ty,
            {
                "source": "body_cluster",
                "points": len(indices),
                "width_x_m": round(width_x, 3),
                "width_y_m": round(width_y, 3),
                "height_z_m": round(height_z, 3),
                "_detect_cost": cost,
            },
        ))
    candidates.sort(key=lambda item: _finite_float(item[2].get("_detect_cost"), item[0]))
    return candidates


def cluster_leg_person(
    pts: np.ndarray,
    *,
    radius_m: float,
    front_max_y_m: float,
    previous_rel: Optional[tuple[float, float]],
) -> Optional[tuple[float, float, dict[str, Any]]]:
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    leg_mask = (
        (x >= 0.45)
        & (x <= float(radius_m))
        & (np.abs(y) <= float(front_max_y_m))
        & (z >= -0.38)
        & (z <= 0.85)
    )
    leg_pts = pts[leg_mask]
    if leg_pts.shape[0] < LEG_MIN_POINTS:
        return None

    cells = np.floor(leg_pts[:, :2] / LEG_CELL_M).astype(np.int32)
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, key in enumerate(cells):
        buckets.setdefault((int(key[0]), int(key[1])), []).append(idx)

    seen: set[tuple[int, int]] = set()
    legs: list[dict[str, Any]] = []
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
        if len(indices) < LEG_MIN_POINTS:
            continue
        cpts = leg_pts[np.asarray(indices, dtype=np.int64)]
        xs, ys, zs = cpts[:, 0], cpts[:, 1], cpts[:, 2]
        width_x = float(np.percentile(xs, 90) - np.percentile(xs, 10))
        width_y = float(np.percentile(ys, 90) - np.percentile(ys, 10))
        height_z = float(np.percentile(zs, 95) - np.percentile(zs, 5))
        if not (0.03 <= width_x <= 0.55 and 0.03 <= width_y <= 0.42 and height_z >= 0.18):
            continue
        legs.append({
            "x": float(np.median(xs)),
            "y": float(np.median(ys)),
            "points": len(indices),
            "width_x_m": width_x,
            "width_y_m": width_y,
            "height_z_m": height_z,
        })

    if not legs:
        return None

    best: Optional[tuple[float, float, dict[str, Any]]] = None
    best_cost = float("inf")
    for i, left in enumerate(legs):
        for right in legs[i + 1:]:
            dx = abs(float(left["x"]) - float(right["x"]))
            dy = abs(float(left["y"]) - float(right["y"]))
            if dx > 0.45 or not (0.16 <= dy <= 0.75):
                continue
            cx = 0.5 * (float(left["x"]) + float(right["x"]))
            cy = 0.5 * (float(left["y"]) + float(right["y"]))
            if not (0.45 <= cx <= float(radius_m) and abs(cy) <= float(front_max_y_m)):
                continue
            points = int(left["points"]) + int(right["points"])
            continuity = 0.0
            if previous_rel is not None:
                continuity = 0.8 * math.hypot(cx - previous_rel[0], cy - previous_rel[1])
            cost = cx + 0.55 * abs(cy) + 0.45 * dx + continuity - min(0.45, points / 120.0)
            if cost < best_cost:
                best_cost = cost
                best = (
                    cx,
                    cy,
                    {
                        "source": "leg_pair",
                        "points": points,
                        "leg_count": 2,
                        "leg_separation_y_m": round(dy, 3),
                        "leg_x_delta_m": round(dx, 3),
                        "width_x_m": round(max(float(left["width_x_m"]), float(right["width_x_m"])) + dx, 3),
                        "width_y_m": round(dy + 0.5 * (float(left["width_y_m"]) + float(right["width_y_m"])), 3),
                        "height_z_m": round(max(float(left["height_z_m"]), float(right["height_z_m"])), 3),
                    },
                )
    if best is not None:
        return best

    leg = min(
        legs,
        key=lambda item: (
            float(item["x"])
            + 0.8 * abs(float(item["y"]))
            + (0.7 * math.hypot(float(item["x"]) - previous_rel[0], float(item["y"]) - previous_rel[1]) if previous_rel is not None else 0.0)
            - min(0.25, int(item["points"]) / 100.0)
        ),
    )
    return (
        float(leg["x"]),
        float(leg["y"]),
        {
            "source": "leg_single",
            "points": int(leg["points"]),
            "leg_count": 1,
            "width_x_m": round(float(leg["width_x_m"]), 3),
            "width_y_m": round(float(leg["width_y_m"]), 3),
            "height_z_m": round(float(leg["height_z_m"]), 3),
        },
    )


def predicted_relative_target(
    rel_x: float,
    rel_y: float,
    target_distance_m: float,
    lateral_deadband_m: float,
) -> tuple[float, float, float]:
    dist = math.hypot(rel_x, rel_y)
    if dist < 1e-6:
        return 0.0, 0.0, 0.0

    forward_error = rel_x - float(target_distance_m)
    if abs(forward_error) < 0.12:
        tx = 0.0
    else:
        tx = forward_error

    if abs(rel_y) < float(lateral_deadband_m):
        ty = 0.0
    else:
        ty = rel_y

    face_yaw = math.atan2(rel_y, max(0.25, rel_x))
    if abs(face_yaw) < YAW_DEADBAND_RAD or (abs(tx) < 0.05 and abs(ty) < 0.05):
        face_yaw = 0.0
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
        self.rgbd_lock = threading.Lock()
        self.rgbd_context: Any = None
        self.rgbd_socket: Any = None
        self.rgbd_proc: subprocess.Popen[str] | None = None
        self.rgbd_start_attempted = False
        self.video_client: Any = None
        self.latest_camera_jpeg_bytes: Optional[bytes] = None
        self.latest_depth_m: Optional[np.ndarray] = None
        self.latest_camera_source = ""
        self.latest_camera_ts = 0.0
        self.latest_rgb_person_rel: Optional[tuple[float, float]] = None
        self.latest_rgb_person_meta: dict[str, Any] = {}
        self.latest_rgb_person_points = np.empty((0, 3), dtype=np.float32)
        self.latest_rgb_person_ts = 0.0
        self.rgb_frame_seq = 0
        self.rgb_cached_detections: list[dict[str, Any]] = []
        self.phase = PHASE_IDLE
        self.last_event = ""
        self.last_spoken = ""
        self.last_error = ""
        self.last_camera_error = ""
        self.person_rel: Optional[tuple[float, float]] = None
        self.person_meta: dict[str, Any] = {}
        self.follow_points_rel = np.empty((0, 3), dtype=np.float32)
        self.target_signature: dict[str, float] = {}
        self._last_similarity_score = 1.0
        self._last_similarity_prompt_ts = 0.0
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
        self._camera_worker = threading.Thread(target=self._camera_loop, daemon=True, name="follow-me-camera-loop")
        self._camera_worker.start()

    def _start_rgbd_publisher(self) -> None:
        if not bool(self.args.auto_start_rgbd) or self.rgbd_start_attempted:
            return
        self.rgbd_start_attempted = True
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "real_sense.py"),
            "--reset",
            "--display", "off",
            "--width", str(int(self.args.rgbd_width)),
            "--height", str(int(self.args.rgbd_height)),
            "--fps", str(int(self.args.rgbd_fps)),
            "--timeout-ms", str(int(self.args.rgbd_timeout_ms)),
            "--publish",
            "--publish-host", "*",
            "--publish-port", str(int(self.args.rgbd_port)),
            "--publish-fps", str(int(self.args.rgbd_publish_fps)),
        ]
        log_path = "/tmp/follow_me_slam_realsense.log"
        try:
            log_file = open(log_path, "a", encoding="utf-8")
            self.rgbd_proc = subprocess.Popen(
                command,
                cwd=str(SCRIPTS_DIR),
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            with self.lock:
                self.last_camera_error = f"starting RGB-D publisher pid={self.rgbd_proc.pid}; log={log_path}"
        except Exception as exc:
            with self.lock:
                self.last_camera_error = f"failed to start RGB-D publisher: {exc}"

    def _ensure_rgbd_socket(self) -> Any:
        if self.rgbd_socket is not None:
            return self.rgbd_socket
        try:
            import zmq
        except Exception as exc:
            raise RuntimeError(f"RGB-D camera stream requires pyzmq: {exc}") from exc
        self.rgbd_context = zmq.Context.instance()
        socket = self.rgbd_context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, str(self.args.rgbd_topic).encode("utf-8"))
        socket.setsockopt(zmq.RCVTIMEO, max(20, min(250, int(float(self.args.camera_timeout_s) * 1000.0))))
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.connect(f"tcp://{self.args.rgbd_host}:{int(self.args.rgbd_port)}")
        self.rgbd_socket = socket
        return socket

    def _close_rgbd_socket(self) -> None:
        socket = self.rgbd_socket
        self.rgbd_socket = None
        if socket is not None:
            try:
                socket.close(0)
            except Exception:
                pass

    def latest_rgbd_jpeg(self, timeout: float = 0.35) -> bytes:
        try:
            import zmq
        except Exception as exc:
            raise RuntimeError(f"RGB-D camera stream requires pyzmq: {exc}") from exc
        with self.rgbd_lock:
            socket = self._ensure_rgbd_socket()
            socket.setsockopt(zmq.RCVTIMEO, max(20, min(250, int(float(timeout) * 1000.0))))
            deadline = time.time() + max(0.1, float(timeout))
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
                    raise RuntimeError(f"RGB-D receive failed: {exc}") from exc
                if len(parts) >= 4:
                    parts = parts[-3:]
                if len(parts) < 1:
                    continue
                latest = [bytes(part) for part in parts]
                while True:
                    try:
                        newer = socket.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    except Exception as exc:
                        self._close_rgbd_socket()
                        raise RuntimeError(f"RGB-D drain failed: {exc}") from exc
                    if len(newer) >= 4:
                        newer = newer[-3:]
                    if len(newer) >= 1:
                        latest = [bytes(part) for part in newer]
                break
            if not latest:
                raise RuntimeError(
                    f"No RGB-D camera frames from tcp://{self.args.rgbd_host}:{int(self.args.rgbd_port)}"
                )
            with self.lock:
                self.last_camera_error = ""
                if len(latest) >= 2:
                    try:
                        import cv2

                        depth_scale = 0.001
                        if len(latest) >= 3 and len(latest[2]) >= 4:
                            import struct

                            depth_scale = float(struct.unpack("f", latest[2][:4])[0])
                        depth_raw = cv2.imdecode(np.frombuffer(latest[1], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        if depth_raw is not None:
                            if depth_raw.ndim == 3:
                                depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
                            self.latest_depth_m = depth_raw.astype("float32") * float(depth_scale)
                    except Exception:
                        pass
            return latest[0]

    def _get_video_client(self) -> Any:
        if self.video_client is None:
            from sdk_sensors import load_video_client_type

            video_client_cls = load_video_client_type()
            client = video_client_cls()
            client.SetTimeout(2.0)
            client.Init()
            self.video_client = client
        return self.video_client

    def latest_camera_jpeg(self, timeout: float = 0.35) -> tuple[bytes, str]:
        source = str(self.args.camera_source).strip().lower()
        errors: list[str] = []
        if source in ("auto", "rgbd"):
            try:
                return self.latest_rgbd_jpeg(timeout=timeout), "rgbd"
            except Exception as exc:
                errors.append(f"RGB-D: {exc}")
                self._start_rgbd_publisher()
                if source == "auto" and bool(self.args.auto_start_rgbd):
                    time.sleep(max(0.0, float(self.args.rgbd_start_wait_s)))
                    try:
                        return self.latest_rgbd_jpeg(timeout=max(timeout, 1.0)), "rgbd"
                    except Exception as retry_exc:
                        errors.append(f"RGB-D retry: {retry_exc}")
                if source == "rgbd":
                    raise
        if source in ("auto", "videoclient", "video_client"):
            try:
                code, data = self._get_video_client().GetImageSample()
                if int(code) != 0:
                    raise RuntimeError(f"GetImageSample failed with code={code}")
                if data is None or len(bytes(data)) == 0:
                    raise RuntimeError("GetImageSample returned an empty image")
                with self.lock:
                    self.last_camera_error = ""
                    self.latest_depth_m = None
                return bytes(data), "video_client"
            except Exception as exc:
                errors.append(f"VideoClient: {exc}")
                if source in ("videoclient", "video_client"):
                    raise
        raise RuntimeError("; ".join(errors) if errors else f"unknown camera source: {self.args.camera_source}")

    def cached_camera_jpeg(self) -> Optional[tuple[bytes, str, float]]:
        with self.lock:
            if self.latest_camera_jpeg_bytes is None:
                return None
            return self.latest_camera_jpeg_bytes, self.latest_camera_source, self.latest_camera_ts

    def _camera_loop(self) -> None:
        period = 1.0 / max(1.0, float(self.args.camera_rate_hz))
        while not self._stop_event.is_set():
            started = time.time()
            try:
                jpeg, source = self.latest_camera_jpeg(timeout=float(self.args.camera_timeout_s))
                with self.lock:
                    self.latest_camera_jpeg_bytes = jpeg
                    self.latest_camera_source = source
                    self.latest_camera_ts = time.time()
                    self.last_camera_error = ""
            except Exception as exc:
                with self.lock:
                    self.last_camera_error = str(exc)
            elapsed = time.time() - started
            time.sleep(max(0.0, period - elapsed))

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
                self.target_signature = {}
                self._last_similarity_score = 1.0
                self._last_similarity_prompt_ts = 0.0
                self._predictor.reset()
                self._speak_async("Looking for you.")
            elif event == "confirm_yes" and phase == PHASE_AWAITING_CONFIRMATION:
                self.phase = PHASE_CONFIRMED
                self._speak_async("Say start when you're ready.")
            elif event == "confirm_no" and phase == PHASE_AWAITING_CONFIRMATION:
                self.phase = PHASE_RECOGNIZING
                self.confirm_hits = 0
                self.target_signature = {}
                self._last_similarity_score = 1.0
                self._last_similarity_prompt_ts = 0.0
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
                self.target_signature = {}
                self._last_similarity_score = 1.0
                self._last_similarity_prompt_ts = 0.0
                self.follow_points_rel = np.empty((0, 3), dtype=np.float32)
                self._speak_async("Stopped following.")
            return self._snapshot_locked()

    def _maybe_prompt_low_similarity(self, rel_y: Optional[float], similarity: float) -> None:
        now = time.time()
        if (now - self._last_similarity_prompt_ts) < float(self.args.target_prompt_cooldown_s):
            return
        self._last_similarity_prompt_ts = now
        if rel_y is not None and rel_y < -0.45:
            prompt = "Are you behind me or to my right side?"
        elif rel_y is not None and rel_y > 0.45:
            prompt = "Are you behind me or to my left side?"
        else:
            prompt = "Are you behind me or to my right side?"
        self._speak_async(prompt)

    def _extract_follow_points(
        self,
        points: np.ndarray,
        rel_x: float,
        rel_y: float,
        meta: dict[str, Any],
    ) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        source = str(meta.get("source", ""))
        x_window = 0.75 if source == "body_cluster" else 0.55
        y_window = 0.75 if source == "body_cluster" else 0.55
        z_max = 1.85 if source == "body_cluster" else 1.05
        mask = (
            np.isfinite(points).all(axis=1)
            & (np.abs(x - float(rel_x)) <= x_window)
            & (np.abs(y - float(rel_y)) <= y_window)
            & (z >= -0.45)
            & (z <= z_max)
        )
        filtered = points[mask]
        if filtered.shape[0] > int(self.args.follow_cloud_max_points):
            stride = max(1, filtered.shape[0] // int(self.args.follow_cloud_max_points))
            filtered = filtered[::stride][: int(self.args.follow_cloud_max_points)]
        return filtered.astype(np.float32, copy=False)

    def _detect_tick(self, phase: str) -> None:
        points = self.current_points()
        with self.lock:
            previous_rel = self.person_rel if phase in (PHASE_FOLLOWING, PHASE_SEARCHING) else None
            target_signature = dict(self.target_signature)
            rgb_age = time.time() - self.latest_rgb_person_ts if self.latest_rgb_person_ts else float("inf")
            rgb_result = None
            rgb_points = np.empty((0, 3), dtype=np.float32)
            use_rgb = not (phase in (PHASE_FOLLOWING, PHASE_SEARCHING) and target_signature)
            if use_rgb and bool(self.args.prefer_rgb_person) and self.latest_rgb_person_rel is not None and rgb_age <= float(self.args.rgb_person_hold_s):
                rx, ry = self.latest_rgb_person_rel
                rgb_result = (rx, ry, dict(self.latest_rgb_person_meta))
                rgb_points = self.latest_rgb_person_points.copy()
        if rgb_result is not None:
            result = rgb_result
        else:
            result = cluster_person(
                points,
                radius_m=float(self.args.detect_radius_m),
                front_max_y_m=float(self.args.detect_front_max_y_m),
                previous_rel=previous_rel,
                target_signature=target_signature if phase in (PHASE_FOLLOWING, PHASE_SEARCHING) else None,
            )
        now = time.time()
        with self.lock:
            if result is None:
                keep_visual = (now - self._last_detect_ts) <= float(self.args.follow_cloud_hold_s)
                if not keep_visual:
                    self.person_rel = None
                    self.person_meta = {}
                    self.follow_points_rel = np.empty((0, 3), dtype=np.float32)
                if phase == PHASE_RECOGNIZING:
                    self.confirm_hits = 0
                elif target_signature and phase in (PHASE_FOLLOWING, PHASE_SEARCHING) and not keep_visual:
                    self.web.pause()
                    self._maybe_prompt_low_similarity(None, 0.0)
                return
            rel_x, rel_y, meta = result
            if str(meta.get("source", "")).startswith("rgbd_") and rgb_points.size:
                follow_points = rgb_points
            else:
                follow_points = self._extract_follow_points(points, rel_x, rel_y, meta)
            similarity = _finite_float(meta.get("target_similarity"), 1.0)
            self._last_similarity_score = similarity
            min_similarity = float(self.args.target_similarity_min)
            if target_signature and phase in (PHASE_FOLLOWING, PHASE_SEARCHING) and similarity < min_similarity:
                self.web.pause()
                self.person_rel = None
                self.person_meta = {
                    "source": "target_rejected",
                    "target_similarity": round(similarity, 3),
                    "required_similarity": round(min_similarity, 3),
                    "candidate_y_m": round(rel_y, 3),
                }
                if (now - self._last_detect_ts) > float(self.args.follow_cloud_hold_s):
                    self.follow_points_rel = np.empty((0, 3), dtype=np.float32)
                self._maybe_prompt_low_similarity(rel_y, similarity)
                return
            self.person_rel = (rel_x, rel_y)
            self.person_meta = meta
            self.follow_points_rel = follow_points
            if not str(meta.get("source", "")).startswith("rgbd_") and follow_points.size and not self.target_signature:
                self.target_signature = follow_cloud_signature(follow_points, meta)
            self._last_detect_ts = now
            if phase == PHASE_RECOGNIZING:
                self.confirm_hits += 1
                if self.confirm_hits >= FOLLOW_STABLE_HITS:
                    if not str(meta.get("source", "")).startswith("rgbd_") and follow_points.size:
                        self.target_signature = follow_cloud_signature(follow_points, meta)
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
        tx, ty, tyaw = predicted_relative_target(
            pred_x,
            pred_y,
            float(self.args.target_distance_m),
            float(self.args.lateral_deadband_m),
        )
        step = math.hypot(tx, ty)
        max_step = max(0.1, float(self.args.max_nav_step_m))
        if step > max_step:
            scale = max_step / step
            tx *= scale
            ty *= scale
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
            "last_camera_error": self.last_camera_error,
            "person_rel": self.person_rel,
            "person_meta": self.person_meta,
            "follow_cloud_points": int(self.follow_points_rel.shape[0]),
            "target_similarity": self._last_similarity_score,
            "target_signature": self.target_signature,
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

    def close(self) -> None:
        self._stop_event.set()
        self._close_rgbd_socket()
        proc = self.rgbd_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception:
                pass


def make_figure_3d(state: FollowSlamState, selected_layers: list[str], max_points: int) -> Any:
    import plotly.graph_objects as go

    fig = go.Figure()
    web = state.web
    pose = state.robot_pose()
    with state.lock:
        follow_points = state.follow_points_rel.copy()
        person_rel = state.person_rel
        nav_target = state.nav_target
        person_meta = dict(state.person_meta)

    if "follow_target" in selected_layers and follow_points.size:
        pts = follow_points
        if pts.shape[0] > int(max_points):
            stride = max(1, pts.shape[0] // int(max_points))
            pts = pts[::stride][: int(max_points)]
        if bool(state.args.visualize_follow_relative):
            xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        elif pose is not None:
            c, s = math.cos(pose.yaw), math.sin(pose.yaw)
            xs = pose.x + c * pts[:, 0] - s * pts[:, 1]
            ys = pose.y + s * pts[:, 0] + c * pts[:, 1]
            zs = pose.z + pts[:, 2]
        else:
            xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                name=f"Follow target cloud ({person_meta.get('source', 'target')})",
                marker={"size": 5, "color": pts[:, 2], "colorscale": "Viridis", "opacity": 0.9},
            )
        )

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

    if bool(state.args.visualize_follow_relative):
        fig.add_trace(
            go.Scatter3d(
                x=[0.0], y=[0.0], z=[0.0],
                mode="markers",
                name="Robot",
                marker={"size": 8, "color": "#ff3154", "symbol": "diamond"},
            )
        )
        radius = float(state.args.detect_radius_m)
        theta = np.linspace(-math.pi / 2.0, math.pi / 2.0, 40)
        fig.add_trace(
            go.Scatter3d(
                x=radius * np.cos(theta),
                y=radius * np.sin(theta),
                z=np.zeros_like(theta),
                mode="lines",
                name="Detection front arc",
                line={"color": "#8da2bd", "width": 3},
            )
        )
    elif pose is not None:
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

    if person_rel is not None and bool(state.args.visualize_follow_relative):
        px, py = person_rel
        fig.add_trace(
            go.Scatter3d(
                x=[px], y=[py], z=[0.0],
                mode="markers",
                name="Detected person",
                marker={"size": 10, "color": "#37e05f", "symbol": "circle"},
            )
        )
    elif person_rel is not None and pose is not None:
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

    if nav_target is not None and bool(state.args.visualize_follow_relative):
        with state.lock:
            predicted_rel = state.predicted_rel
        if predicted_rel is not None:
            tx, ty, _yaw = predicted_relative_target(
                predicted_rel[0],
                predicted_rel[1],
                float(state.args.target_distance_m),
                float(state.args.lateral_deadband_m),
            )
        else:
            tx, ty = 0.0, 0.0
        fig.add_trace(
            go.Scatter3d(
                x=[tx], y=[ty], z=[0.0],
                mode="markers",
                name="Predicted nav target",
                marker={"size": 10, "color": "#00e5ff", "symbol": "x"},
            )
        )
    elif nav_target is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[nav_target.x], y=[nav_target.y], z=[nav_target.z],
                mode="markers",
                name="Predicted nav target",
                marker={"size": 9, "color": "#00e5ff", "symbol": "x"},
            )
        )

    scene = {
        "xaxis": {"title": "x (m)", "backgroundcolor": "#171a21", "gridcolor": "#303642"},
        "yaxis": {"title": "y (m)", "backgroundcolor": "#171a21", "gridcolor": "#303642"},
        "zaxis": {"title": "z (m)", "backgroundcolor": "#171a21", "gridcolor": "#303642"},
        "aspectmode": "data",
    }
    if bool(state.args.visualize_follow_relative):
        scene = {
            "xaxis": {"title": "forward x (m)", "range": [0.0, float(state.args.detect_radius_m)], "backgroundcolor": "#171a21", "gridcolor": "#303642"},
            "yaxis": {"title": "left y (m)", "range": [-float(state.args.detect_front_max_y_m), float(state.args.detect_front_max_y_m)], "backgroundcolor": "#171a21", "gridcolor": "#303642"},
            "zaxis": {"title": "height z (m)", "range": [-0.6, 1.8], "backgroundcolor": "#171a21", "gridcolor": "#303642"},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.4, "y": 1.0, "z": 0.65},
        }

    fig.update_layout(
        template="plotly_dark",
        uirevision="follow-me-slam-3d",
        margin={"l": 0, "r": 0, "t": 28, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        paper_bgcolor="#111318",
        scene=scene,
        height=560,
    )
    return fig


def render_leg_detection_jpeg(state: FollowSlamState) -> bytes:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"leg detection JPEG requires cv2: {exc}") from exc

    width, height = 720, 420
    image = np.full((height, width, 3), (18, 21, 27), dtype=np.uint8)
    meters_x = max(1.5, float(state.args.detect_radius_m))
    meters_y = max(1.0, float(state.args.detect_front_max_y_m))
    margin = 34
    x_scale = (height - 2 * margin) / meters_x
    y_scale = (width - 2 * margin) / (2.0 * meters_y)

    def project(px: float, py: float) -> tuple[int, int]:
        col = int(width * 0.5 + py * y_scale)
        row = int(height - margin - px * x_scale)
        return col, row

    for gx in np.arange(0.0, meters_x + 0.001, 0.5):
        _c0, r = project(float(gx), 0.0)
        cv2.line(image, (margin, r), (width - margin, r), (43, 49, 61), 1)
    for gy in np.arange(-meters_y, meters_y + 0.001, 0.5):
        c, _r0 = project(0.0, float(gy))
        cv2.line(image, (c, margin), (c, height - margin), (43, 49, 61), 1)

    with state.lock:
        pts = state.follow_points_rel.copy()
        person_rel = state.person_rel
        predicted_rel = state.predicted_rel
        meta = dict(state.person_meta)
        phase = state.phase

    if pts.size:
        z = pts[:, 2]
        z_min = float(np.min(z))
        z_range = max(0.05, float(np.max(z) - z_min))
        for px, py, pz in pts:
            c, r = project(float(px), float(py))
            if margin <= c < width - margin and margin <= r < height - margin:
                level = (float(pz) - z_min) / z_range
                color = (70, int(150 + 90 * level), int(90 + 120 * (1.0 - level)))
                cv2.circle(image, (c, r), 2, color, -1, lineType=cv2.LINE_AA)

    robot_c, robot_r = project(0.0, 0.0)
    cv2.circle(image, (robot_c, robot_r), 7, (49, 229, 255), -1, lineType=cv2.LINE_AA)
    cv2.line(image, (robot_c, robot_r), project(0.35, 0.0), (49, 229, 255), 2, lineType=cv2.LINE_AA)

    if person_rel is not None:
        pc, pr = project(float(person_rel[0]), float(person_rel[1]))
        cv2.drawMarker(image, (pc, pr), (74, 224, 95), markerType=cv2.MARKER_CROSS, markerSize=26, thickness=2)
        cv2.circle(image, (pc, pr), 16, (74, 224, 95), 2, lineType=cv2.LINE_AA)
    if predicted_rel is not None:
        pc, pr = project(float(predicted_rel[0]), float(predicted_rel[1]))
        cv2.drawMarker(image, (pc, pr), (255, 199, 70), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)

    text = f"{phase} | {meta.get('source', 'no_target')} | points={pts.shape[0]}"
    if person_rel is not None:
        text += f" | x={person_rel[0]:.2f}m y={person_rel[1]:.2f}m"
    cv2.putText(image, text, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (232, 238, 247), 1, cv2.LINE_AA)
    cv2.putText(image, "top-down RGB leg detector view", (14, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 180, 195), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(45, min(95, int(state.args.camera_jpeg_quality)))],
    )
    if not ok:
        raise RuntimeError("failed to encode leg detection JPEG")
    return encoded.tobytes()


def render_camera_detection_jpeg(state: FollowSlamState) -> bytes:
    try:
        import cv2
    except Exception as exc:
        return render_status_jpeg(f"camera view requires cv2: {exc}")
    try:
        cached = state.cached_camera_jpeg()
        if cached is None:
            jpeg, camera_source = state.latest_camera_jpeg(timeout=float(state.args.camera_timeout_s))
            camera_age_s = 0.0
        else:
            jpeg, camera_source, camera_ts = cached
            camera_age_s = max(0.0, time.time() - float(camera_ts))
            if camera_age_s > float(state.args.camera_stale_s):
                raise RuntimeError(f"stale RGB camera frame age={camera_age_s:.2f}s")
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("failed to decode RGB camera JPEG")
    except Exception as exc:
        with state.lock:
            state.last_camera_error = str(exc)
        return render_status_jpeg(str(exc))

    h, w = image.shape[:2]
    with state.lock:
        state.rgb_frame_seq += 1
        frame_seq = state.rgb_frame_seq
        cached_detections = [dict(item) for item in state.rgb_cached_detections]
    detect_every = max(1, int(state.args.rgb_detect_every_n))
    if frame_seq % detect_every == 1 or not cached_detections:
        detections = detect_people_in_rgb(image)
        with state.lock:
            state.rgb_cached_detections = [dict(item) for item in detections]
    else:
        detections = cached_detections
    with state.lock:
        depth_m = None if state.latest_depth_m is None else state.latest_depth_m.copy()
    rgb_target = rgbd_person_target(image, detections, depth_m, float(state.args.detect_radius_m))
    followed_box = None
    if rgb_target is not None:
        rel_x, rel_y, rgb_meta = rgb_target
        followed_box = dict(rgb_meta.get("box", {}))
        rgb_points = rgbd_person_cloud(
            image,
            depth_m,
            followed_box,
            max_points=int(state.args.follow_cloud_max_points),
        )
        with state.lock:
            state.latest_rgb_person_rel = (rel_x, rel_y)
            state.latest_rgb_person_meta = rgb_meta
            state.latest_rgb_person_points = rgb_points
            if rgb_points.size:
                state.follow_points_rel = rgb_points
            state.latest_rgb_person_ts = time.time()
    for det in detections:
        x, y, bw, bh = int(det["x"]), int(det["y"]), int(det["w"]), int(det["h"])
        score = float(det.get("score", 0.0))
        is_followed = bool(followed_box) and x == int(followed_box.get("x", -1)) and y == int(followed_box.get("y", -1))
        cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 70) if is_followed else (0, 220, 120), 3 if is_followed else 2)
        head_h = int(bh * 0.20)
        torso_y = y + head_h
        torso_h = int(bh * 0.35)
        leg_y = y + int(bh * 0.55)
        leg_h = max(1, y + bh - leg_y)
        mid_x = x + bw // 2
        cv2.rectangle(image, (x, y), (x + bw, y + head_h), (255, 190, 70), 1)
        cv2.rectangle(image, (x, torso_y), (x + bw, torso_y + torso_h), (70, 220, 255), 1)
        cv2.rectangle(image, (x, leg_y), (mid_x, leg_y + leg_h), (0, 190, 255), 2)
        cv2.rectangle(image, (mid_x, leg_y), (x + bw, leg_y + leg_h), (0, 190, 255), 2)
        label = f"{'FOLLOW ' if is_followed else ''}person {score:.2f}"
        cv2.putText(image, label, (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 120), 2, cv2.LINE_AA)
        cv2.putText(image, "head", (x + 3, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 190, 70), 1, cv2.LINE_AA)
        cv2.putText(image, "torso", (x + 3, torso_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "legs", (x, max(18, leg_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 190, 255), 1, cv2.LINE_AA)

    project_follow_cluster_to_rgb(image, state)

    with state.lock:
        person_rel = state.person_rel
        meta = dict(state.person_meta)
        phase = state.phase
        cloud_points = int(state.follow_points_rel.shape[0])

    source = str(meta.get("source", "no_target"))
    status = f"{phase} | camera={camera_source} age={camera_age_s:.2f}s | rgb people={len(detections)} | follow={source} cloud={cloud_points}"
    if person_rel is not None:
        status += f" | x={person_rel[0]:.2f}m y={person_rel[1]:.2f}m"
    cv2.rectangle(image, (0, 0), (w, 34), (15, 18, 24), -1)
    cv2.putText(image, status, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 241, 250), 1, cv2.LINE_AA)
    if person_rel is not None and not detections:
        hint = "lidar leg target active; no RGB person box from camera detector"
        cv2.putText(image, hint, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 210, 255), 1, cv2.LINE_AA)

    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
    if not ok:
        return render_status_jpeg("failed to encode RGB camera frame")
    return encoded.tobytes()


def detect_people_in_rgb(image: Any) -> list[dict[str, Any]]:
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
    min_height = max(64, int(h * 0.20))
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(detect_image, winStride=(8, 8), padding=(8, 8), scale=1.05)
    detections: list[dict[str, Any]] = []
    for (dx, dy, dbw, dbh), weight in zip(boxes, weights):
        x = int(dx / scale)
        y = int(dy / scale)
        bw = int(dbw / scale)
        bh = int(dbh / scale)
        aspect = float(bw) / float(max(1, bh))
        score = float(weight)
        if score < RGB_MIN_PERSON_SCORE:
            continue
        if bh < min_height or not (0.20 <= aspect <= 0.95):
            continue
        if bw * bh < 0.012 * w * h or dbw * dbh < 0.012 * dw * dh:
            continue
        detections.append({"x": x, "y": y, "w": bw, "h": bh, "score": score})
    detections.sort(key=lambda item: float(item["score"]), reverse=True)
    return detections[:4]


def rgbd_person_target(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    depth_m: Optional[np.ndarray],
    max_range_m: float,
) -> Optional[tuple[float, float, dict[str, Any]]]:
    if depth_m is None or not detections:
        return None
    h, w = image.shape[:2]
    if depth_m.shape[:2] != (h, w):
        return None
    best: Optional[tuple[float, float, dict[str, Any]]] = None
    best_score = -1.0
    for det in detections:
        x, y, bw, bh = int(det["x"]), int(det["y"]), int(det["w"]), int(det["h"])
        pad_x = max(2, int(bw * 0.14))
        pad_top = max(2, int(bh * 0.25))
        pad_bottom = max(2, int(bh * 0.08))
        x0 = max(0, x + pad_x)
        x1 = min(w, x + bw - pad_x)
        y0 = max(0, y + pad_top)
        y1 = min(h, y + bh - pad_bottom)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = depth_m[y0:y1, x0:x1]
        valid = roi[(roi > 0.25) & (roi <= float(max_range_m))]
        valid_fraction = float(valid.size) / float(max(1, roi.size))
        if valid_fraction < 0.12:
            continue
        distance_m = float(np.median(valid))
        if not math.isfinite(distance_m):
            continue
        center_px = float(x + bw * 0.5)
        lateral_m = ((center_px / max(1.0, float(w))) - 0.5) * 2.0 * distance_m * 0.55
        score = float(det.get("score", 0.0)) + min(0.35, valid_fraction * 0.35)
        if score > best_score:
            best_score = score
            best = (
                distance_m,
                lateral_m,
                {
                    "source": "rgbd_person",
                    "confidence": round(min(0.98, score), 3),
                    "box": {"x": x, "y": y, "w": bw, "h": bh},
                    "depth_valid_fraction": round(valid_fraction, 3),
                    "distance_m": round(distance_m, 3),
                    "lateral_m": round(lateral_m, 3),
                },
            )
    return best


def camera_intrinsics_for_shape(width: int, height: int) -> tuple[float, float, float, float]:
    intrinsics = {
        (424, 240): (302.1, 302.5, 214.3, 121.4),
        (640, 480): (604.2, 605.1, 324.7, 242.9),
    }
    if (width, height) in intrinsics:
        return intrinsics[(width, height)]
    return (
        604.2 * (float(width) / 640.0),
        605.1 * (float(height) / 480.0),
        324.7 * (float(width) / 640.0),
        242.9 * (float(height) / 480.0),
    )


def base_from_camera_transform() -> tuple[np.ndarray, np.ndarray]:
    roll = -math.pi / 2.0 - DEFAULT_CAMERA_DOWN_PITCH_RAD
    pitch = 0.0
    yaw = -math.pi / 2.0
    r_base_camera = rotation_matrix_from_rpy(roll, pitch, yaw)
    t_base_camera = np.asarray([DEFAULT_CAMERA_X_M, 0.0, DEFAULT_CAMERA_Z_M], dtype=np.float64)
    return r_base_camera, t_base_camera


def rgbd_person_cloud(
    image: np.ndarray,
    depth_m: Optional[np.ndarray],
    box: dict[str, Any],
    *,
    max_points: int,
) -> np.ndarray:
    if depth_m is None:
        return np.empty((0, 3), dtype=np.float32)
    h, w = image.shape[:2]
    if depth_m.shape[:2] != (h, w):
        return np.empty((0, 3), dtype=np.float32)
    x, y, bw, bh = int(box.get("x", 0)), int(box.get("y", 0)), int(box.get("w", 0)), int(box.get("h", 0))
    x0 = max(0, x + int(bw * 0.12))
    x1 = min(w, x + bw - int(bw * 0.12))
    y0 = max(0, y + int(bh * 0.18))
    y1 = min(h, y + bh - int(bh * 0.05))
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 3), dtype=np.float32)

    fx, fy, cx, cy = camera_intrinsics_for_shape(w, h)
    r_base_camera, t_base_camera = base_from_camera_transform()
    roi = depth_m[y0:y1, x0:x1]
    valid = np.argwhere((roi > 0.25) & (roi <= 5.0))
    if valid.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    stride = max(1, valid.shape[0] // max(1, int(max_points)))
    valid = valid[::stride][: max(1, int(max_points))]
    points: list[list[float]] = []
    for rv, cu in valid:
        v = int(y0 + rv)
        u = int(x0 + cu)
        z_cam = float(depth_m[v, u])
        x_cam = (float(u) - cx) * z_cam / fx
        y_cam = (float(v) - cy) * z_cam / fy
        p_base = r_base_camera @ np.asarray([x_cam, y_cam, z_cam], dtype=np.float64) + t_base_camera
        bx, by, bz = (float(value) for value in p_base)
        if 0.2 <= bx <= 5.0 and abs(by) <= 2.5 and -0.7 <= bz <= 2.2:
            points.append([bx, by, bz])
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def project_follow_cluster_to_rgb(image: np.ndarray, state: FollowSlamState) -> None:
    import cv2

    h, w = image.shape[:2]
    intrinsics = {
        (424, 240): (302.1, 302.5, 214.3, 121.4),
        (640, 480): (604.2, 605.1, 324.7, 242.9),
    }
    if (w, h) in intrinsics:
        fx, fy, cx, cy = intrinsics[(w, h)]
    else:
        fx = 604.2 * (float(w) / 640.0)
        fy = 605.1 * (float(h) / 480.0)
        cx = 324.7 * (float(w) / 640.0)
        cy = 242.9 * (float(h) / 480.0)

    roll = -math.pi / 2.0 - DEFAULT_CAMERA_DOWN_PITCH_RAD
    pitch = 0.0
    yaw = -math.pi / 2.0
    r_base_camera = rotation_matrix_from_rpy(roll, pitch, yaw)
    t_base_camera = np.asarray([DEFAULT_CAMERA_X_M, 0.0, DEFAULT_CAMERA_Z_M], dtype=np.float64)
    r_camera_base = r_base_camera.T

    with state.lock:
        pts = state.follow_points_rel.copy()
        person_rel = state.person_rel
        meta = dict(state.person_meta)
    if pts.size == 0:
        return

    step = max(1, pts.shape[0] // 280)
    uv_points: list[tuple[int, int]] = []
    for px, py, pz in pts[::step]:
        p_base = np.asarray([float(px), float(py), float(pz)], dtype=np.float64)
        p_cam = r_camera_base @ (p_base - t_base_camera)
        x_cam, y_cam, z_cam = (float(v) for v in p_cam)
        if z_cam <= 0.12:
            continue
        u = int(fx * x_cam / z_cam + cx)
        v = int(fy * y_cam / z_cam + cy)
        if 0 <= u < w and 0 <= v < h:
            uv_points.append((u, v))
            cv2.circle(image, (u, v), 3, (0, 210, 255), -1, lineType=cv2.LINE_AA)

    if uv_points:
        arr = np.asarray(uv_points, dtype=np.int32)
        x0, y0 = np.min(arr, axis=0)
        x1, y1 = np.max(arr, axis=0)
        cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 210, 255), 2)
        label = f"followed cluster: {meta.get('source', 'target')}"
        cv2.putText(image, label, (int(x0), max(18, int(y0) - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 210, 255), 1, cv2.LINE_AA)
    elif person_rel is not None:
        cv2.putText(
            image,
            "followed cluster active but outside projected RGB view",
            (12, max(46, h - 36)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 210, 255),
            1,
            cv2.LINE_AA,
        )


def render_status_jpeg(message: str) -> bytes:
    try:
        import cv2
    except Exception:
        return b""
    image = np.full((360, 640, 3), (18, 21, 27), dtype=np.uint8)
    lines = [str(message)[i:i + 72] for i in range(0, min(len(str(message)), 216), 72)] or ["camera unavailable"]
    cv2.putText(image, "Real RGB camera view", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 241, 250), 2, cv2.LINE_AA)
    for idx, line in enumerate(lines):
        cv2.putText(image, line, (18, 92 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (110, 190, 255), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
    return encoded.tobytes() if ok else b""


CSS = """
body { margin: 0; background: #07090d; color: #f8fafc; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.app { min-height: 100vh; }
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 12px 18px; background: #0c1118; border-bottom: 1px solid #536174; }
h1 { font-size: 20px; line-height: 1.1; margin: 0 0 5px; font-weight: 650; }
.status-line { font-size: 13px; color: #ffffff; min-height: 18px; font-weight: 600; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
button { background: #1f2937; color: #ffffff; border: 1px solid #8da2bd; border-radius: 6px; padding: 8px 10px; font-size: 13px; font-weight: 650; cursor: pointer; }
button:hover { background: #334155; }
button.stop { background: #8a2d2b; border-color: #aa3b38; }
.main { display: grid; grid-template-columns: 310px minmax(0, 1fr); min-height: calc(100vh - 72px); }
.side { padding: 14px; border-right: 1px solid #536174; background: #0b0f16; overflow: auto; }
.map-pane { min-width: 0; padding: 10px 12px 0; }
label { display: block; color: #ffffff; font-size: 13px; font-weight: 650; margin: 13px 0 6px; }
input { width: 100%; box-sizing: border-box; background: #ffffff; color: #020617; border: 2px solid #94a3b8; border-radius: 6px; padding: 8px; font-weight: 600; }
.side, .side * { color: #edf2fb; }
.side input, .side textarea { color: #111827; background: #f7f9fc; }
.layers label, .dash-checklist label { margin: 7px 0; color: #edf2fb !important; }
.side .rc-slider {
    margin: 12px 8px 36px;
    padding: 14px 8px 18px;
    background: #111827;
    border: 1px solid #475569;
    border-radius: 6px;
}
.side .rc-slider-rail {
    height: 8px !important;
    background-color: #94a3b8 !important;
    opacity: 1 !important;
}
.side .rc-slider-track {
    height: 8px !important;
    background-color: #22d3ee !important;
}
.side .rc-slider-step { height: 8px !important; }
.side .rc-slider-handle {
    width: 22px !important;
    height: 22px !important;
    margin-top: -7px !important;
    border: 3px solid #f8fafc !important;
    background: #0284c7 !important;
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.35), 0 4px 10px rgba(0, 0, 0, 0.55) !important;
    opacity: 1 !important;
}
.side .rc-slider-handle:hover,
.side .rc-slider-handle:focus,
.side .rc-slider-handle-dragging {
    border-color: #ffffff !important;
    box-shadow: 0 0 0 5px rgba(34, 211, 238, 0.45), 0 4px 12px rgba(0, 0, 0, 0.65) !important;
}
.rc-slider-tooltip,
.side .rc-slider-tooltip { opacity: 1 !important; z-index: 20; pointer-events: none; }
.rc-slider-tooltip-inner,
.side .rc-slider-tooltip-inner {
    min-width: 34px;
    height: auto;
    padding: 4px 8px;
    color: #020617 !important;
    background: #fde047 !important;
    border: 2px solid #111827 !important;
    border-radius: 5px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    font-size: 13px;
    font-weight: 800;
    line-height: 1.2;
    text-shadow: none !important;
}
.rc-slider-tooltip-inner *,
.side .rc-slider-tooltip-inner * { color: #020617 !important; }
.rc-slider-tooltip-placement-top .rc-slider-tooltip-arrow,
.rc-slider-tooltip-placement-topLeft .rc-slider-tooltip-arrow,
.rc-slider-tooltip-placement-topRight .rc-slider-tooltip-arrow { border-top-color: #fde047 !important; }
.rc-slider-tooltip-placement-bottom .rc-slider-tooltip-arrow,
.rc-slider-tooltip-placement-bottomLeft .rc-slider-tooltip-arrow,
.rc-slider-tooltip-placement-bottomRight .rc-slider-tooltip-arrow,
.side .rc-slider-tooltip-placement-bottom .rc-slider-tooltip-arrow,
.side .rc-slider-tooltip-placement-bottomLeft .rc-slider-tooltip-arrow,
.side .rc-slider-tooltip-placement-bottomRight .rc-slider-tooltip-arrow { border-bottom-color: #fde047 !important; }
.rc-slider-mark-text { color: #ffffff !important; font-weight: 750; text-shadow: 0 1px 2px #000000; }
.follow-buttons { display: flex; flex-direction: column; gap: 8px; margin-top: 6px; }
.action-output { margin-top: 14px; padding: 10px; background: #020617; color: #ffffff; border: 1px solid #8da2bd; border-radius: 6px; font-size: 12px; line-height: 1.45; white-space: pre-wrap; }
.video-pane { border-bottom: 1px solid #536174; padding-bottom: 10px; margin-bottom: 10px; }
.camera-video { display: block; width: 100%; max-height: 360px; object-fit: contain; background: #020617; border: 2px solid #8da2bd; border-radius: 6px; }
"""


def app_layout(state: FollowSlamState):
    from dash import dcc, html

    layer_options = [{"label": "Follow target cloud", "value": "follow_target"}] + [
        {"label": LAYER_STYLE.get(name, (name,))[0], "value": name}
        for name in DEFAULT_TOPICS.keys()
    ]
    default_layers = ["follow_target"]
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
                            html.Label("Max visual points"),
                            dcc.Slider(id="max-points", min=200, max=8000, step=200, value=1200,
                                       marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Label("Detection radius (m)"),
                            dcc.Slider(id="detect-radius", min=1.0, max=8.0, step=0.5,
                                       value=float(state.args.detect_radius_m),
                                       marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Label("Target distance (m)"),
                            dcc.Slider(id="target-distance", min=0.5, max=3.0, step=0.1,
                                       value=float(state.args.target_distance_m),
                                       marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Label("Lateral deadband (m)"),
                            dcc.Slider(id="lateral-deadband", min=0.05, max=0.6, step=0.05,
                                       value=float(state.args.lateral_deadband_m),
                                       marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Label("Max nav step (m)"),
                            dcc.Slider(id="max-nav-step", min=0.2, max=1.5, step=0.05,
                                       value=float(state.args.max_nav_step_m),
                                       marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Label("Cluster hold (s)"),
                            dcc.Slider(id="follow-cloud-hold", min=0.0, max=3.0, step=0.1,
                                       value=float(state.args.follow_cloud_hold_s),
                                       marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Label("RGB refresh (ms)"),
                            dcc.Slider(id="camera-refresh", min=50, max=500, step=25,
                                       value=float(state.args.camera_refresh_ms),
                                       marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Label("RGB detect every N frames"),
                            dcc.Slider(id="rgb-detect-every", min=1, max=12, step=1,
                                       value=int(state.args.rgb_detect_every_n),
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
                        [
                            html.Div([html.Img(id="camera-video", className="camera-video")], className="video-pane"),
                            dcc.Graph(id="map-graph-3d", config={"displayModeBar": True}, style={"height": "560px"}),
                        ],
                        className="map-pane",
                    ),
                ],
                className="main",
            ),
            dcc.Interval(id="map-interval", interval=max(250, int(state.args.refresh_ms)), n_intervals=0),
            dcc.Interval(id="camera-interval", interval=max(100, int(state.args.camera_refresh_ms)), n_intervals=0),
        ],
        className="app",
    )


def create_dash_app(state: FollowSlamState):
    import dash
    from dash import Input, Output, State as DashState
    from flask import Response, jsonify, request

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
        Output("camera-interval", "interval"),
        Input("map-interval", "n_intervals"),
        Input("layers", "value"),
        Input("max-points", "value"),
        Input("detect-radius", "value"),
        Input("target-distance", "value"),
        Input("lateral-deadband", "value"),
        Input("max-nav-step", "value"),
        Input("follow-cloud-hold", "value"),
        Input("camera-refresh", "value"),
        Input("rgb-detect-every", "value"),
    )
    def update(
        _n: int,
        layers: list[str],
        max_points: int,
        detect_radius: float,
        target_distance: float,
        lateral_deadband: float,
        max_nav_step: float,
        follow_cloud_hold: float,
        camera_refresh: float,
        rgb_detect_every: int,
    ):
        if detect_radius is not None:
            state.args.detect_radius_m = float(detect_radius)
        if target_distance is not None:
            state.args.target_distance_m = float(target_distance)
        if lateral_deadband is not None:
            state.args.lateral_deadband_m = float(lateral_deadband)
        if max_nav_step is not None:
            state.args.max_nav_step_m = float(max_nav_step)
        if follow_cloud_hold is not None:
            state.args.follow_cloud_hold_s = float(follow_cloud_hold)
        if camera_refresh is not None:
            state.args.camera_refresh_ms = int(camera_refresh)
        if rgb_detect_every is not None:
            state.args.rgb_detect_every_n = int(rgb_detect_every)
        fig = make_figure_3d(state, layers or [], int(max_points or 1200))
        snap = state.snapshot()
        line = (
            f"phase={snap['phase']} event={snap['last_event']} spoken={snap['last_spoken']!r} "
            f"relocation={'ready' if snap['relocation_ready'] else 'not ready'} "
            f"source={snap['person_meta'].get('source', '-')} cloud={snap['follow_cloud_points']} "
            f"similarity={float(snap.get('target_similarity', 1.0)):.2f} "
            f"camera={'ok' if not snap['last_camera_error'] else snap['last_camera_error']} "
            f"person={snap['person_rel']} target={snap['nav_target']}"
        )
        return fig, line, max(50, int(state.args.camera_refresh_ms))

    @app.callback(
        Output("camera-video", "src"),
        Input("camera-interval", "n_intervals"),
    )
    def update_camera(_n: int):
        return f"/camera_detection.jpg?t={int(_n or 0)}"

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

    @app.server.route("/leg_detection.jpg", methods=["GET"])
    def leg_detection_route():
        try:
            return Response(render_leg_detection_jpeg(state), mimetype="image/jpeg")
        except Exception as exc:
            return Response(str(exc), status=503, mimetype="text/plain")

    @app.server.route("/camera_detection.jpg", methods=["GET"])
    def camera_detection_route():
        return Response(render_camera_detection_jpeg(state), mimetype="image/jpeg")

    return app


def run_dash_worker(args: argparse.Namespace) -> None:
    state = FollowSlamState(args)
    app = create_dash_app(state)
    print(f"Follow-me SLAM web app: http://{args.dash_host}:{args.dash_port}", flush=True)
    try:
        app.run(host=args.dash_host, port=args.dash_port, debug=False)
    finally:
        state.close()


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
            "--target-similarity-min", str(float(self.args.target_similarity_min)),
            "--target-prompt-cooldown-s", str(float(self.args.target_prompt_cooldown_s)),
            "--follow-cloud-max-points", str(int(self.args.follow_cloud_max_points)),
            "--follow-cloud-hold-s", str(float(self.args.follow_cloud_hold_s)),
            "--visualize-follow-relative" if self.args.visualize_follow_relative else "--no-visualize-follow-relative",
            "--target-distance-m", str(float(self.args.target_distance_m)),
            "--lateral-deadband-m", str(float(self.args.lateral_deadband_m)),
            "--max-nav-step-m", str(float(self.args.max_nav_step_m)),
            "--predict-lead-s", str(float(self.args.predict_lead_s)),
            "--follow-loop-s", str(float(self.args.follow_loop_s)),
            "--resend-distance-m", str(float(self.args.resend_distance_m)),
            "--resend-interval-s", str(float(self.args.resend_interval_s)),
            "--lost-timeout-s", str(float(self.args.lost_timeout_s)),
            "--lost-abort-s", str(float(self.args.lost_abort_s)),
            "--refresh-ms", str(int(self.args.refresh_ms)),
            "--rgbd-host", str(self.args.rgbd_host),
            "--rgbd-port", str(int(self.args.rgbd_port)),
            "--rgbd-topic", str(self.args.rgbd_topic),
            "--camera-source", str(self.args.camera_source),
            "--camera-timeout-s", str(float(self.args.camera_timeout_s)),
            "--camera-refresh-ms", str(int(self.args.camera_refresh_ms)),
            "--camera-rate-hz", str(float(self.args.camera_rate_hz)),
            "--camera-stale-s", str(float(self.args.camera_stale_s)),
            "--camera-jpeg-quality", str(int(self.args.camera_jpeg_quality)),
            "--rgb-detect-every-n", str(int(self.args.rgb_detect_every_n)),
            "--auto-start-rgbd" if self.args.auto_start_rgbd else "--no-auto-start-rgbd",
            "--rgbd-width", str(int(self.args.rgbd_width)),
            "--rgbd-height", str(int(self.args.rgbd_height)),
            "--rgbd-fps", str(int(self.args.rgbd_fps)),
            "--rgbd-timeout-ms", str(int(self.args.rgbd_timeout_ms)),
            "--rgbd-publish-fps", str(int(self.args.rgbd_publish_fps)),
            "--rgbd-start-wait-s", str(float(self.args.rgbd_start_wait_s)),
            "--prefer-rgb-person" if self.args.prefer_rgb_person else "--no-prefer-rgb-person",
            "--rgb-person-hold-s", str(float(self.args.rgb_person_hold_s)),
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
    parser.add_argument("--refresh-ms", type=int, default=500)
    parser.add_argument("--rgbd-host", default=os.environ.get("G1_RGBD_HOST", "192.168.2.41"))
    parser.add_argument("--rgbd-port", type=int, default=int(os.environ.get("G1_RGBD_PORT", "5555")))
    parser.add_argument("--rgbd-topic", default=os.environ.get("G1_RGBD_TOPIC", ""))
    parser.add_argument("--camera-source", choices=("auto", "rgbd", "videoclient", "video_client"), default="rgbd")
    parser.add_argument("--camera-timeout-s", type=float, default=0.08)
    parser.add_argument("--camera-refresh-ms", type=int, default=100)
    parser.add_argument("--camera-rate-hz", type=float, default=15.0)
    parser.add_argument("--camera-stale-s", type=float, default=1.0)
    parser.add_argument("--camera-jpeg-quality", type=int, default=75)
    parser.add_argument("--rgb-detect-every-n", type=int, default=5)
    parser.add_argument("--auto-start-rgbd", dest="auto_start_rgbd", action="store_true", default=False)
    parser.add_argument("--no-auto-start-rgbd", dest="auto_start_rgbd", action="store_false")
    parser.add_argument("--rgbd-width", type=int, default=424)
    parser.add_argument("--rgbd-height", type=int, default=240)
    parser.add_argument("--rgbd-fps", type=int, default=15)
    parser.add_argument("--rgbd-timeout-ms", type=int, default=30000)
    parser.add_argument("--rgbd-publish-fps", type=int, default=15)
    parser.add_argument("--rgbd-start-wait-s", type=float, default=6.0)

    parser.add_argument("--detect-radius-m", type=float, default=4.5)
    parser.add_argument("--detect-front-max-y-m", type=float, default=1.6)
    parser.add_argument("--target-similarity-min", type=float, default=TARGET_SIMILARITY_MIN)
    parser.add_argument("--target-prompt-cooldown-s", type=float, default=TARGET_SIMILARITY_PROMPT_COOLDOWN_S)
    parser.add_argument("--follow-cloud-max-points", type=int, default=1600)
    parser.add_argument("--follow-cloud-hold-s", type=float, default=1.2)
    parser.add_argument("--visualize-follow-relative", dest="visualize_follow_relative", action="store_true", default=True)
    parser.add_argument("--no-visualize-follow-relative", dest="visualize_follow_relative", action="store_false")
    parser.add_argument("--prefer-rgb-person", dest="prefer_rgb_person", action="store_true", default=True)
    parser.add_argument("--no-prefer-rgb-person", dest="prefer_rgb_person", action="store_false")
    parser.add_argument("--rgb-person-hold-s", type=float, default=1.2)
    parser.add_argument("--target-distance-m", type=float, default=1.25)
    parser.add_argument("--lateral-deadband-m", type=float, default=0.18)
    parser.add_argument("--max-nav-step-m", type=float, default=0.65)
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
