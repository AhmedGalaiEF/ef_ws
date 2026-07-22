#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import http.server
import json
import math
import os
import struct
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
G1_DIR = SCRIPT_DIR.parents[2] if SCRIPT_DIR.name == "ollama_ai" else SCRIPT_DIR.parent
MODULES_DIR = G1_DIR / "modules"
WORKSPACE_DIR = G1_DIR.parent
for path in (G1_DIR, MODULES_DIR, G1_DIR / "modules" / "scripts", WORKSPACE_DIR):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


FOLLOW_MAX_RANGE_M = 4.5
FOLLOW_LIDAR_CONFIRM_TOLERANCE_M = 0.75


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone G1 follow-me web app with RGB-D AprilTag tracking and adjustable locomotion."
    )
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8096)
    parser.add_argument("--rgbd-host", default=os.environ.get("G1_RGBD_HOST", "192.168.2.41"))
    parser.add_argument("--rgbd-port", type=int, default=int(os.environ.get("G1_RGBD_PORT", "5555")))
    parser.add_argument("--rgbd-topic", default=os.environ.get("G1_RGBD_TOPIC", ""))
    parser.add_argument("--iface", default=os.environ.get("G1_IFACE", "eth0"))
    parser.add_argument("--domain-id", type=int, default=int(os.environ.get("G1_DOMAIN_ID", "0")))
    parser.add_argument("--loop-s", type=float, default=0.10)
    parser.add_argument("--capture-timeout-s", type=float, default=0.25)
    parser.add_argument("--target-distance", type=float, default=1.25)
    parser.add_argument("--distance-tolerance", type=float, default=0.18)
    parser.add_argument("--center-tolerance", type=float, default=0.18)
    parser.add_argument("--max-vx", type=float, default=0.16)
    parser.add_argument("--max-vyaw", type=float, default=0.35)
    parser.add_argument("--vx-gain", type=float, default=0.45)
    parser.add_argument("--yaw-gain", type=float, default=0.45)
    parser.add_argument("--command-duration", type=float, default=0.35)
    parser.add_argument("--tag-id", type=int, default=3)
    parser.add_argument("--tag-size-m", type=float, default=0.129)
    parser.add_argument("--tag-dict", default="apriltag_36h11")
    parser.add_argument("--tag-hold-s", type=float, default=0.75)
    parser.add_argument("--tag-roi-scale", type=float, default=3.0)
    parser.add_argument("--tag-full-search-interval-s", type=float, default=0.45)
    parser.add_argument("--tag-quad-decimate", type=float, default=1.0)
    parser.add_argument("--enable-lidar", action="store_true")
    parser.add_argument("--disable-lidar", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lidar-max-points", type=int, default=6000)
    parser.add_argument("--lidar-min-cluster-points", type=int, default=14)
    parser.add_argument("--lidar-front-max-y", type=float, default=1.6)
    parser.add_argument("--lidar-z-min", type=float, default=-0.45)
    parser.add_argument("--lidar-z-max", type=float, default=1.9)
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Begin following immediately. Otherwise use the web UI Start button.",
    )
    parser.add_argument(
        "--skip-stand",
        action="store_true",
        help="Do not run BalanceStand/Start/SetFsmId(501) before first motion command.",
    )
    return parser.parse_args()


class SharedState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.lock = threading.RLock()
        self.active = bool(args.auto_start)
        self.motion_enabled = bool(args.auto_start)
        self.target_distance_m = float(args.target_distance)
        self.distance_tolerance_m = float(args.distance_tolerance)
        self.center_tolerance_m = float(args.center_tolerance)
        self.max_vx_mps = float(args.max_vx)
        self.max_vyaw_rps = float(args.max_vyaw)
        self.vx_gain = float(args.vx_gain)
        self.yaw_gain = float(args.yaw_gain)
        self.command_duration_s = float(args.command_duration)
        self.tag_id = int(args.tag_id)
        self.tag_size_m = float(args.tag_size_m)
        self.tag_dict = str(args.tag_dict)
        self.tag_hold_s = float(args.tag_hold_s)
        self.tag_ok = False
        self.last_tag_error = ""
        self.tag_age_s: float | None = None
        self.phase = "starting"
        self.target: dict[str, Any] | None = None
        self.last_command = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "duration_s": 0.0}
        self.rgbd_ok = False
        self.lidar_ok = False
        self.robot_ok = False
        self.last_rgbd_error = ""
        self.last_lidar_error = ""
        self.last_robot_error = ""
        self.last_error = ""
        self.last_update = 0.0
        self.latest_jpeg: bytes | None = None
        self.rgbd_source = f"tcp://{args.rgbd_host}:{args.rgbd_port}"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "active": self.active,
                "motion_enabled": self.motion_enabled,
                "phase": self.phase,
                "target": self.target,
                "settings": {
                    "target_distance_m": self.target_distance_m,
                    "distance_tolerance_m": self.distance_tolerance_m,
                    "center_tolerance_m": self.center_tolerance_m,
                    "max_vx_mps": self.max_vx_mps,
                    "max_vyaw_rps": self.max_vyaw_rps,
                    "vx_gain": self.vx_gain,
                    "yaw_gain": self.yaw_gain,
                    "command_duration_s": self.command_duration_s,
                    "tag_id": self.tag_id,
                    "tag_size_m": self.tag_size_m,
                    "tag_dict": self.tag_dict,
                    "tag_hold_s": float(getattr(self, "tag_hold_s", 0.0) or 0.0),
                },
                "tag_ok": self.tag_ok,
                "last_tag_error": self.last_tag_error,
                "tag_age_s": self.tag_age_s,
                "last_command": dict(self.last_command),
                "rgbd_ok": self.rgbd_ok,
                "lidar_ok": self.lidar_ok,
                "robot_ok": self.robot_ok,
                "rgbd_source": self.rgbd_source,
                "last_rgbd_error": self.last_rgbd_error,
                "last_lidar_error": self.last_lidar_error,
                "last_robot_error": self.last_robot_error,
                "last_error": self.last_error,
                "last_update": self.last_update,
                "frame_age_s": None if self.last_update <= 0 else max(0.0, time.time() - self.last_update),
            }

    def update_settings(self, payload: dict[str, Any]) -> None:
        with self.lock:
            for key, attr, lo, hi in (
                ("target_distance_m", "target_distance_m", 0.55, 3.0),
                ("distance_tolerance_m", "distance_tolerance_m", 0.05, 0.75),
                ("center_tolerance_m", "center_tolerance_m", 0.03, 0.8),
                ("max_vx_mps", "max_vx_mps", 0.02, 0.45),
                ("max_vyaw_rps", "max_vyaw_rps", 0.05, 1.0),
                ("vx_gain", "vx_gain", 0.05, 1.2),
                ("yaw_gain", "yaw_gain", 0.05, 1.2),
                ("command_duration_s", "command_duration_s", 0.05, 1.0),
            ):
                if key in payload:
                    setattr(self, attr, clamp(float(payload[key]), lo, hi))


class RgbdReceiver:
    def __init__(self, host: str, port: int, topic: str) -> None:
        self.host = str(host)
        self.port = int(port)
        self.topic = str(topic)
        self._ctx: Any | None = None
        self._socket: Any | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            socket = self._socket
            self._socket = None
            if socket is not None:
                try:
                    socket.close(0)
                except Exception:
                    pass

    def _ensure_socket(self) -> Any:
        if self._socket is not None:
            return self._socket
        try:
            import zmq
        except Exception as exc:
            raise RuntimeError(f"pyzmq is required for RGB-D streaming: {exc}") from exc
        self._ctx = zmq.Context.instance()
        socket = self._ctx.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, self.topic.encode("utf-8"))
        socket.setsockopt(zmq.RCVTIMEO, 250)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.connect(f"tcp://{self.host}:{self.port}")
        self._socket = socket
        return socket

    def recv(self, timeout_s: float = 0.35) -> dict[str, Any]:
        try:
            import cv2
            import numpy as np
            import zmq
        except Exception as exc:
            raise RuntimeError(f"RGB-D decoding requires cv2, numpy, and pyzmq: {exc}") from exc

        with self._lock:
            socket = self._ensure_socket()
            deadline = time.time() + max(0.1, float(timeout_s))
            latest: list[bytes] | None = None
            last_error = ""
            while time.time() < deadline:
                try:
                    parts = socket.recv_multipart()
                except zmq.Again:
                    if latest is not None:
                        break
                    continue
                except Exception as exc:
                    self.close()
                    raise RuntimeError(f"RGB-D receive failed: {exc}") from exc
                if len(parts) >= 4:
                    parts = parts[-3:]
                if len(parts) < 2:
                    last_error = f"expected RGB-D multipart frame, got {len(parts)} part(s)"
                    continue
                latest = [bytes(part) for part in parts]
                while True:
                    try:
                        newer = socket.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if len(newer) >= 4:
                        newer = newer[-3:]
                    if len(newer) >= 2:
                        latest = [bytes(part) for part in newer]
                break
            if latest is None:
                detail = f": {last_error}" if last_error else ""
                raise RuntimeError(f"No RGB-D frame received from tcp://{self.host}:{self.port}{detail}")

        rgb_jpeg = latest[0]
        depth_png = latest[1]
        depth_scale = 0.001
        if len(latest) >= 3 and len(latest[2]) >= 4:
            try:
                depth_scale = float(struct.unpack("f", latest[2][:4])[0])
            except Exception:
                depth_scale = 0.001
        rgb = cv2.imdecode(np.frombuffer(rgb_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        depth_raw = cv2.imdecode(np.frombuffer(depth_png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if rgb is None:
            raise RuntimeError("failed to decode RGB JPEG")
        if depth_raw is None:
            raise RuntimeError("failed to decode depth PNG")
        if depth_raw.ndim == 3:
            depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
        return {
            "timestamp": time.time(),
            "rgb_jpeg": rgb_jpeg,
            "rgb_bgr": rgb,
            "depth_m": depth_raw.astype("float32") * float(depth_scale),
        }


class RobotMotion:
    def __init__(self, args: argparse.Namespace, state: SharedState) -> None:
        self.args = args
        self.state = state
        self.robot: Any | None = None
        self.prepared = False
        self.lock = threading.Lock()

    def close(self) -> None:
        self.stop()
        with self.lock:
            robot = self.robot
            self.robot = None
            if robot is not None and callable(getattr(robot, "close", None)):
                try:
                    robot.close()
                except Exception:
                    pass

    def _robot(self) -> Any:
        if self.robot is not None:
            return self.robot
        from sdk_client import Robot
        self.robot = Robot(
            iface=str(self.args.iface),
            domain_id=int(self.args.domain_id),
            safety_boot=False,
            recover_dev_mode_on_init=False,
            auto_start_sensors=True,
        )
        with self.state.lock:
            self.state.robot_ok = True
            self.state.last_robot_error = ""
        return self.robot

    def prepare(self) -> None:
        if self.prepared or bool(self.args.skip_stand):
            return
        robot = self._robot()
        client = getattr(robot, "_client", None)
        if client is not None:
            for method_name, method_args in (
                ("BalanceStand", (0,)),
                ("Start", ()),
                ("SetFsmId", (501,)),
            ):
                method = getattr(client, method_name, None)
                if callable(method):
                    method(*method_args)
                    time.sleep(0.3)
        self.prepared = True

    def send_velocity(self, vx: float, vy: float, vyaw: float, duration_s: float) -> int:
        with self.lock:
            self.prepare()
            robot = self._robot()
            client = getattr(robot, "_client", None)
            if client is not None and callable(getattr(client, "SetVelocity", None)):
                result = client.SetVelocity(float(vx), float(vy), float(vyaw), float(duration_s))
            elif callable(getattr(robot, "loco_move", None)):
                result = robot.loco_move(float(vx), float(vy), float(vyaw))
            else:
                result = robot.walk(float(vx), float(vy), float(vyaw))
            with self.state.lock:
                self.state.robot_ok = True
                self.state.last_robot_error = ""
            return 0 if result is None else int(result)

    def get_lidar_points(self, max_points: int) -> list[Any]:
        with self.lock:
            robot = self._robot()
            method = getattr(robot, "get_lidar_points", None)
            if not callable(method):
                raise RuntimeError("Robot wrapper does not provide get_lidar_points().")
            return list(method(max_points=max(100, int(max_points))))

    def stop(self) -> None:
        if self.robot is None:
            return
        try:
            robot = self.robot
            client = getattr(robot, "_client", None)
            if client is not None and callable(getattr(client, "SetVelocity", None)):
                client.SetVelocity(0.0, 0.0, 0.0, 0.05)
            elif callable(getattr(robot, "stop", None)):
                robot.stop()
            elif callable(getattr(robot, "loco_move", None)):
                robot.loco_move(0.0, 0.0, 0.0)
        except Exception as exc:
            with self.state.lock:
                self.state.robot_ok = False
                self.state.last_robot_error = str(exc)


class FollowMeApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.state = SharedState(args)
        self.rgbd = RgbdReceiver(args.rgbd_host, args.rgbd_port, args.rgbd_topic)
        self.motion = RobotMotion(args, self.state)
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame: dict[str, Any] | None = None
        self.latest_frame_seq = 0
        self.tag_detector: Any | None = None
        self.tag_dictionary: Any | None = None
        self.tag_lock = threading.RLock()
        self.last_tag_target: dict[str, Any] | None = None
        self.last_tag_seen_s = 0.0
        self.last_full_tag_search_s = 0.0
        self.capture_thread = threading.Thread(target=self._capture_loop, name="follow-me-capture", daemon=True)
        self.worker = threading.Thread(target=self._loop, name="follow-me-loop", daemon=True)
        self.httpd: http.server.ThreadingHTTPServer | None = None

    def start(self) -> None:
        self.capture_thread.start()
        self.worker.start()
        handler = self._make_handler()
        self.httpd = http.server.ThreadingHTTPServer((str(self.args.web_host), int(self.args.web_port)), handler)
        print(f"Follow-me web app: http://{self.args.web_host}:{self.args.web_port}/")
        self.httpd.serve_forever()

    def close(self) -> None:
        self.stop_event.set()
        with self.state.lock:
            self.state.active = False
            self.state.motion_enabled = False
        self.motion.close()
        self.rgbd.close()
        if self.httpd is not None:
            self.httpd.server_close()

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.rgbd.recv(timeout_s=float(self.args.capture_timeout_s))
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_frame_seq += 1
                    seq = self.latest_frame_seq
                with self.state.lock:
                    last_target = self.state.target
                preview_jpeg = self._draw_overlay(frame, last_target)
                with self.state.lock:
                    self.state.rgbd_ok = True
                    self.state.last_rgbd_error = ""
                    self.state.latest_jpeg = preview_jpeg
                    self.state.last_update = time.time()
                    if self.state.phase in ("starting", "error"):
                        self.state.phase = "camera"
                frame["seq"] = seq
            except Exception as exc:
                with self.state.lock:
                    self.state.rgbd_ok = False
                    self.state.last_rgbd_error = str(exc)
                    self.state.last_error = str(exc)
                    self.state.last_update = time.time()
                time.sleep(0.05)

    def _loop(self) -> None:
        processed_seq = 0
        while not self.stop_event.is_set():
            started = time.time()
            try:
                with self.frame_lock:
                    seq = self.latest_frame_seq
                    frame = self.latest_frame
                if frame is None or seq == processed_seq:
                    time.sleep(0.02)
                    continue
                processed_seq = seq
                tag_target = self._target_from_apriltag(frame)
                target = self._target_from_lidar(tag_target)
                jpeg = self._draw_overlay(frame, target)
                command = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "duration_s": 0.0}
                with self.state.lock:
                    active = bool(self.state.active)
                    motion_enabled = bool(self.state.motion_enabled)
                    self.state.target = target
                    self.state.latest_jpeg = jpeg
                    if target:
                        self.state.phase = "tracking"
                    else:
                        self.state.phase = "searching_tag"
                if active and motion_enabled:
                    command = self._command_for_target(target)
                    if target is None:
                        self.motion.send_velocity(0.0, 0.0, 0.0, 0.05)
                    elif abs(command["vx"]) > 0.005 or abs(command["vyaw"]) > 0.01:
                        self.motion.send_velocity(
                            command["vx"], command["vy"], command["vyaw"], command["duration_s"]
                        )
                    with self.state.lock:
                        self.state.last_command = dict(command)
            except Exception as exc:
                with self.state.lock:
                    self.state.phase = "error"
                    self.state.last_error = str(exc)
                    self.state.last_update = time.time()
                if self.state.active and self.state.motion_enabled:
                    self.motion.stop()
            elapsed = time.time() - started
            time.sleep(max(0.03, float(self.args.loop_s) - elapsed))

    def _target_from_apriltag(self, frame: dict[str, Any]) -> dict[str, Any] | None:
        import cv2
        import numpy as np

        rgb = frame["rgb_bgr"]
        depth_m = frame["depth_m"]
        h, w = rgb.shape[:2]
        try:
            detector = self._get_tag_detector()
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        except Exception as exc:
            with self.state.lock:
                self.state.tag_ok = False
                self.state.last_tag_error = str(exc)
                self.state.tag_age_s = None
            return None

        frame["tag_candidates"] = []
        now = time.time()
        search_regions: list[tuple[str, int, int, Any]] = []
        roi = self._last_tag_roi(w, h)
        if roi is not None:
            x0, y0, x1, y1 = roi
            search_regions.append(("roi", x0, y0, gray[y0:y1, x0:x1]))
        if not search_regions or now - self.last_full_tag_search_s >= float(self.args.tag_full_search_interval_s):
            search_regions.append(("full", 0, 0, gray))
            self.last_full_tag_search_s = now

        saw_wrong_tag = False
        for search_name, off_x, off_y, search_gray in search_regions:
            try:
                corners, ids, _rejected = detector.detectMarkers(search_gray)
            except Exception as exc:
                with self.state.lock:
                    self.state.tag_ok = False
                    self.state.last_tag_error = str(exc)
                    self.state.tag_age_s = None
                return None
            if ids is None:
                continue
            for idx, marker_id in enumerate(ids.flatten()):
                marker_id = int(marker_id)
                pts = corners[idx].reshape(4, 2).astype(np.float32)
                pts[:, 0] += float(off_x)
                pts[:, 1] += float(off_y)
                x0 = int(np.floor(float(pts[:, 0].min())))
                y0 = int(np.floor(float(pts[:, 1].min())))
                x1 = int(np.ceil(float(pts[:, 0].max())))
                y1 = int(np.ceil(float(pts[:, 1].max())))
                frame["tag_candidates"].append({
                    "id": marker_id,
                    "source": search_name,
                    "box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
                })
                if marker_id != int(self.args.tag_id):
                    saw_wrong_tag = True
                    continue
                target = self._target_from_tag_corners(pts, depth_m, w, h, marker_id, search_name)
                if target is None:
                    continue
                self.last_tag_target = dict(target)
                self.last_tag_seen_s = now
                with self.state.lock:
                    self.state.tag_ok = True
                    self.state.last_tag_error = ""
                    self.state.tag_age_s = 0.0
                return target

        held = self._held_tag_target(now)
        if held is not None:
            with self.state.lock:
                self.state.tag_ok = False
                self.state.last_tag_error = "tag temporarily lost; using last pose"
                self.state.tag_age_s = float(held.get("stale_age_s", 0.0))
            return held

        with self.state.lock:
            self.state.tag_ok = False
            self.state.last_tag_error = (
                f"visible tags do not include id {self.args.tag_id}"
                if saw_wrong_tag
                else f"tag id {self.args.tag_id} not visible"
            )
            self.state.tag_age_s = None
        return None

    def _target_from_tag_corners(
        self,
        pts: Any,
        depth_m: Any,
        image_w: int,
        image_h: int,
        marker_id: int,
        search_name: str,
    ) -> dict[str, Any] | None:
        import numpy as np

        x0 = int(np.floor(float(pts[:, 0].min())))
        y0 = int(np.floor(float(pts[:, 1].min())))
        x1 = int(np.ceil(float(pts[:, 0].max())))
        y1 = int(np.ceil(float(pts[:, 1].max())))
        center_px = float(pts[:, 0].mean())
        center_py = float(pts[:, 1].mean())
        half = max(4, int(max(x1 - x0, y1 - y0) * 0.18))
        rx0 = max(0, int(center_px) - half)
        rx1 = min(image_w, int(center_px) + half + 1)
        ry0 = max(0, int(center_py) - half)
        ry1 = min(image_h, int(center_py) + half + 1)
        roi = depth_m[ry0:ry1, rx0:rx1]
        valid = roi[(roi > 0.25) & (roi <= FOLLOW_MAX_RANGE_M)]
        distance_m = float(np.median(valid)) if valid.size else 0.0

        pnp_distance = self._tag_pnp_distance(pts)
        if not math.isfinite(distance_m) or distance_m <= 0.0:
            distance_m = pnp_distance
        elif math.isfinite(pnp_distance) and pnp_distance > 0.0:
            distance_m = 0.75 * distance_m + 0.25 * pnp_distance
        if not math.isfinite(distance_m) or distance_m <= 0.25 or distance_m > FOLLOW_MAX_RANGE_M:
            return None

        lateral_m = ((center_px / max(1.0, float(image_w))) - 0.5) * 2.0 * distance_m * 0.55
        return {
            "source": "apriltag",
            "label": f"AprilTag id {marker_id}",
            "tag_id": marker_id,
            "tag_size_m": float(self.args.tag_size_m),
            "x_m": distance_m,
            "y_m": lateral_m,
            "confidence": 0.98,
            "box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
            "pixel_uv": [center_px, center_py],
            "depth_valid_points": int(valid.size),
            "pnp_distance_m": pnp_distance if math.isfinite(pnp_distance) else None,
            "search": search_name,
        }

    def _last_tag_roi(self, image_w: int, image_h: int) -> tuple[int, int, int, int] | None:
        target = self.last_tag_target
        if not target or time.time() - self.last_tag_seen_s > max(0.1, float(self.args.tag_hold_s)):
            return None
        box = target.get("box") or {}
        try:
            x = float(box["x"])
            y = float(box["y"])
            w = float(box["w"])
            h = float(box["h"])
        except Exception:
            return None
        if w <= 2 or h <= 2:
            return None
        scale = max(1.2, float(self.args.tag_roi_scale))
        cx = x + w * 0.5
        cy = y + h * 0.5
        rw = max(80.0, w * scale)
        rh = max(80.0, h * scale)
        x0 = max(0, int(cx - rw * 0.5))
        y0 = max(0, int(cy - rh * 0.5))
        x1 = min(image_w, int(cx + rw * 0.5))
        y1 = min(image_h, int(cy + rh * 0.5))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        return x0, y0, x1, y1

    def _held_tag_target(self, now: float) -> dict[str, Any] | None:
        target = self.last_tag_target
        if not target:
            return None
        age = now - self.last_tag_seen_s
        hold_s = max(0.0, float(self.args.tag_hold_s))
        if age <= 0.0 or age > hold_s:
            return None
        held = dict(target)
        held["source"] = "apriltag_hold"
        held["confidence"] = max(0.2, float(target.get("confidence", 0.98)) * (1.0 - age / max(hold_s, 1e-3)))
        held["stale_age_s"] = age
        return held

    def _get_tag_detector(self) -> Any:
        import cv2

        with self.tag_lock:
            if self.tag_detector is not None:
                return self.tag_detector
            dict_name = str(self.args.tag_dict).strip().lower()
            if dict_name in {"apriltag_36h11", "36h11", "tag36h11"}:
                dict_id = cv2.aruco.DICT_APRILTAG_36h11
            elif dict_name in {"apriltag_25h9", "25h9", "tag25h9"}:
                dict_id = cv2.aruco.DICT_APRILTAG_25h9
            elif dict_name in {"apriltag_16h5", "16h5", "tag16h5"}:
                dict_id = cv2.aruco.DICT_APRILTAG_16h5
            else:
                raise RuntimeError(f"Unsupported tag dictionary: {self.args.tag_dict}")
            self.tag_dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            for name, value in (
                ("aprilTagQuadDecimate", max(0.0, float(self.args.tag_quad_decimate))),
                ("cornerRefinementMethod", getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1)),
                ("cornerRefinementMaxIterations", 12),
                ("adaptiveThreshWinSizeMin", 3),
                ("adaptiveThreshWinSizeMax", 15),
                ("adaptiveThreshWinSizeStep", 6),
            ):
                if hasattr(params, name):
                    try:
                        setattr(params, name, value)
                    except Exception:
                        pass
            if hasattr(cv2.aruco, "ArucoDetector"):
                self.tag_detector = cv2.aruco.ArucoDetector(self.tag_dictionary, params)
            else:
                class _CompatDetector:
                    def __init__(self, dictionary: Any, parameters: Any) -> None:
                        self.dictionary = dictionary
                        self.parameters = parameters

                    def detectMarkers(self, gray: Any) -> Any:
                        return cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

                self.tag_detector = _CompatDetector(self.tag_dictionary, params)
            return self.tag_detector

    def _tag_pnp_distance(self, corners_xy: Any) -> float:
        import cv2
        import numpy as np

        size_m = float(self.args.tag_size_m)
        s = size_m / 2.0
        obj_pts = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], dtype=np.float64)
        cam_mat = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.zeros((4,), dtype=np.float64)
        try:
            ok, _rvec, tvec = cv2.solvePnP(
                obj_pts,
                np.asarray(corners_xy, dtype=np.float64).reshape(4, 2),
                cam_mat,
                dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except Exception:
            return float("nan")
        if not ok:
            return float("nan")
        return float(tvec.reshape(3)[2])

    def _target_from_lidar(self, rgbd_target: dict[str, Any] | None) -> dict[str, Any] | None:
        if bool(self.args.disable_lidar) or not bool(self.args.enable_lidar):
            with self.state.lock:
                self.state.lidar_ok = False
                self.state.last_lidar_error = "disabled"
            return rgbd_target
        try:
            points = self.motion.get_lidar_points(max_points=int(self.args.lidar_max_points))
        except Exception as exc:
            with self.state.lock:
                self.state.lidar_ok = False
                self.state.last_lidar_error = str(exc)
            return rgbd_target
        lidar_target = self._cluster_lidar_points(points, rgbd_target)
        with self.state.lock:
            self.state.lidar_ok = bool(points)
            self.state.last_lidar_error = "" if points else "no lidar points"
        if rgbd_target is None:
            return None
        if lidar_target is None:
            return rgbd_target
        refined = dict(rgbd_target)
        refined.update(
            {
                "source": "apriltag_lidar",
                "x_m": lidar_target["x_m"],
                "y_m": lidar_target["y_m"],
                "confidence": min(0.99, float(rgbd_target.get("confidence", 0.0) or 0.0) + 0.15),
                "lidar_points": lidar_target.get("lidar_points", 0),
                "lidar_cluster": lidar_target.get("cluster"),
            }
        )
        return refined

    def _cluster_lidar_points(
        self,
        points: list[Any],
        rgbd_target: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        import numpy as np

        rows: list[tuple[float, float, float]] = []
        for point in points:
            try:
                if isinstance(point, dict):
                    x, y, z = float(point["x"]), float(point["y"]), float(point["z"])
                else:
                    x, y, z = float(point[0]), float(point[1]), float(point[2])
            except Exception:
                continue
            rows.append((x, y, z))
        if not rows:
            return None

        arr = np.asarray(rows, dtype=np.float32)
        x = arr[:, 0]
        y = arr[:, 1]
        z = arr[:, 2]
        x_hint = None if rgbd_target is None else float(rgbd_target.get("x_m", 0.0) or 0.0)
        y_hint = None if rgbd_target is None else float(rgbd_target.get("y_m", 0.0) or 0.0)
        mask = (
            np.isfinite(arr).all(axis=1)
            & (x >= 0.35)
            & (x <= FOLLOW_MAX_RANGE_M)
            & (np.abs(y) <= float(self.args.lidar_front_max_y))
            & (z >= float(self.args.lidar_z_min))
            & (z <= float(self.args.lidar_z_max))
        )
        if x_hint is not None and x_hint > 0.0:
            mask &= np.abs(x - x_hint) <= FOLLOW_LIDAR_CONFIRM_TOLERANCE_M
            mask &= np.abs(y - float(y_hint or 0.0)) <= 0.80
        pts = arr[mask]
        if pts.shape[0] < int(self.args.lidar_min_cluster_points):
            return None

        cell = 0.18
        cells = np.floor(pts[:, :2] / cell).astype(np.int32)
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
            cluster_indices: list[int] = []
            while stack:
                cur = stack.pop()
                cluster_indices.extend(buckets.get(cur, []))
                cx, cy = cur
                for nb in (
                    (cx - 1, cy - 1), (cx - 1, cy), (cx - 1, cy + 1),
                    (cx, cy - 1), (cx, cy + 1),
                    (cx + 1, cy - 1), (cx + 1, cy), (cx + 1, cy + 1),
                ):
                    if nb in buckets and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            clusters.append(cluster_indices)

        min_points = int(self.args.lidar_min_cluster_points)
        best: dict[str, Any] | None = None
        best_cost = float("inf")
        for indices in clusters:
            if len(indices) < min_points:
                continue
            cpts = pts[np.asarray(indices, dtype=np.int64)]
            xs = cpts[:, 0]
            ys = cpts[:, 1]
            zs = cpts[:, 2]
            width_x = float(np.percentile(xs, 90) - np.percentile(xs, 10))
            width_y = float(np.percentile(ys, 90) - np.percentile(ys, 10))
            height_z = float(np.percentile(zs, 90) - np.percentile(zs, 10))
            if width_x > 1.20 or width_y > 1.15:
                continue
            if height_z < 0.18 and len(indices) < max(min_points * 2, 35):
                continue
            tx = float(np.median(xs))
            ty = float(np.median(ys))
            if x_hint is not None and x_hint > 0.0:
                cost = abs(tx - x_hint) + 0.8 * abs(ty - float(y_hint or 0.0)) - min(0.25, len(indices) / 250.0)
            else:
                cost = tx + 0.75 * abs(ty) - min(0.35, len(indices) / 200.0)
            if cost < best_cost:
                best_cost = cost
                best = {
                    "source": "lidar_front_cluster",
                    "x_m": tx,
                    "y_m": ty,
                    "confidence": min(0.92, 0.45 + len(indices) / 180.0),
                    "lidar_points": len(indices),
                    "cluster": {
                        "width_x_m": round(width_x, 3),
                        "width_y_m": round(width_y, 3),
                        "height_z_m": round(height_z, 3),
                    },
                }
        return best

    def _command_for_target(self, target: dict[str, Any] | None) -> dict[str, float]:
        if target is None:
            return {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "duration_s": 0.05}
        with self.state.lock:
            target_distance = self.state.target_distance_m
            distance_tol = self.state.distance_tolerance_m
            center_tol = self.state.center_tolerance_m
            max_vx = self.state.max_vx_mps
            max_vyaw = self.state.max_vyaw_rps
            vx_gain = self.state.vx_gain
            yaw_gain = self.state.yaw_gain
            duration_s = self.state.command_duration_s
        x = float(target.get("x_m", target_distance))
        y = float(target.get("y_m", 0.0))
        error_x = x - target_distance
        vx = clamp(vx_gain * error_x, -max_vx, max_vx)
        if abs(error_x) < distance_tol:
            vx = 0.0
        vyaw = clamp(yaw_gain * y, -max_vyaw, max_vyaw)
        if abs(y) < center_tol:
            vyaw = 0.0
        if target.get("source") == "apriltag_hold":
            age = float(target.get("stale_age_s", 0.0) or 0.0)
            hold_s = max(0.001, float(self.args.tag_hold_s))
            scale = clamp(1.0 - age / hold_s, 0.20, 1.0)
            vx *= scale
            vyaw *= scale
        return {"vx": vx, "vy": 0.0, "vyaw": vyaw, "duration_s": duration_s}

    def _draw_overlay(self, frame: dict[str, Any], target: dict[str, Any] | None) -> bytes:
        import cv2

        image = frame["rgb_bgr"].copy()
        for cand in frame.get("tag_candidates", []) or []:
            box = cand.get("box", {})
            x, y, w, h = int(box.get("x", 0)), int(box.get("y", 0)), int(box.get("w", 0)), int(box.get("h", 0))
            color = (0, 255, 0) if int(cand.get("id", -1)) == int(self.args.tag_id) else (80, 140, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
            cv2.putText(image, f"id {cand.get('id')}", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if target and target.get("box"):
            box = target["box"]
            x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = (
                f"{target.get('label') or target.get('source', 'target')} "
                f"{float(target.get('confidence', 0.0)):.2f} "
                f"{float(target.get('x_m', 0.0)):.2f}m"
            )
            cv2.putText(image, label, (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        elif target:
            label = (
                f"{target.get('source', 'target')} "
                f"x={float(target.get('x_m', 0.0)):.2f}m "
                f"y={float(target.get('y_m', 0.0)):+.2f}m"
            )
            cv2.putText(image, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return encoded.tobytes() if ok else bytes(frame["rgb_jpeg"])

    def _make_handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        app = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/api/status":
                    self._send_json(app.state.snapshot())
                    return
                if path == "/rgb.jpg":
                    with app.state.lock:
                        data = app.state.latest_jpeg
                        error = app.state.last_rgbd_error
                    if data:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                    else:
                        body = app._placeholder_svg(error).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "image/svg+xml")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    return
                body = app._html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                try:
                    payload = json.loads(raw)
                except Exception:
                    payload = dict(urllib.parse.parse_qsl(raw))
                if path == "/api/settings":
                    app.state.update_settings(payload)
                    self._send_json(app.state.snapshot())
                    return
                if path == "/api/start":
                    app.state.update_settings(payload if isinstance(payload, dict) else {})
                    with app.state.lock:
                        app.state.active = True
                        app.state.motion_enabled = True
                        app.state.phase = "recognizing"
                    self._send_json(app.state.snapshot())
                    return
                if path == "/api/pause":
                    with app.state.lock:
                        app.state.active = False
                        app.state.motion_enabled = False
                        app.state.phase = "paused"
                        app.state.last_command = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "duration_s": 0.0}
                    app.motion.stop()
                    self._send_json(app.state.snapshot())
                    return
                if path == "/api/stop":
                    with app.state.lock:
                        app.state.active = False
                        app.state.motion_enabled = False
                        app.state.phase = "idle"
                        app.state.last_command = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "duration_s": 0.0}
                    app.motion.stop()
                    self._send_json(app.state.snapshot())
                    return
                self.send_error(404)

            def _send_json(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def _placeholder_svg(self, msg: str) -> str:
        safe = html.escape(str(msg or "Waiting for RGB-D frames")[:160])
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540">
<rect width="100%" height="100%" fill="#111418"/>
<text x="36" y="76" fill="#dbe7ef" font-family="sans-serif" font-size="30">RGB-D unavailable</text>
<text x="36" y="122" fill="#98a6b3" font-family="sans-serif" font-size="20">{safe}</text>
</svg>"""

    def _html(self) -> str:
        return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 Follow Me</title>
<style>
:root { color-scheme: dark; font-family: Arial, sans-serif; background: #121417; color: #e7edf2; }
body { margin: 0; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 16px; padding: 16px; }
.view { min-width: 0; }
img { display: block; width: 100%; max-height: calc(100vh - 32px); object-fit: contain; background: #050608; border: 1px solid #2e353b; }
aside { display: flex; flex-direction: column; gap: 12px; }
.panel { border: 1px solid #2e353b; border-radius: 6px; padding: 12px; background: #191d21; }
h1, h2 { margin: 0 0 10px; font-size: 18px; }
h2 { font-size: 14px; color: #bac7d2; text-transform: uppercase; }
.buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
button { height: 38px; border: 1px solid #44505a; border-radius: 6px; background: #26313a; color: #eef5fa; font-size: 14px; cursor: pointer; }
button.primary { background: #1f7a4d; border-color: #29925e; }
button.stop { background: #8a2d2b; border-color: #aa3b38; }
label { display: grid; gap: 4px; margin: 10px 0; font-size: 13px; color: #c9d4dc; }
input[type=range], input[type=text] { width: 100%; box-sizing: border-box; }
input[type=text] { height: 34px; border: 1px solid #44505a; border-radius: 5px; background: #11161b; color: #eef5fa; padding: 0 8px; font-size: 14px; }
.value { color: #ffffff; font-variant-numeric: tabular-nums; }
pre { margin: 0; white-space: pre-wrap; font-size: 12px; line-height: 1.35; color: #cfd9e2; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } img { max-height: 60vh; } }
</style></head><body>
<main>
  <section class="view"><img id="rgb" src="/rgb.jpg" alt="RGB camera with detected AprilTag overlay"></section>
  <aside>
    <section class="panel">
      <h1>Follow Me</h1>
      <div class="buttons">
        <button class="primary" onclick="post('/api/start')">Start</button>
        <button onclick="post('/api/pause')">Pause</button>
        <button class="stop" onclick="post('/api/stop')">Stop</button>
        <button onclick="saveSettings()">Apply</button>
      </div>
    </section>
    <section class="panel">
      <h2>Target</h2>
      <pre>AprilTag id 3
tag36h11
black edge 12.9 cm</pre>
    </section>
    <section class="panel">
      <h2>Distance</h2>
      <label>Target <span class="value" id="target_distance_m_v"></span><input id="target_distance_m" type="range" min="0.55" max="3.0" step="0.05"></label>
      <label>Tolerance <span class="value" id="distance_tolerance_m_v"></span><input id="distance_tolerance_m" type="range" min="0.05" max="0.75" step="0.01"></label>
      <label>Center tolerance <span class="value" id="center_tolerance_m_v"></span><input id="center_tolerance_m" type="range" min="0.03" max="0.8" step="0.01"></label>
    </section>
    <section class="panel">
      <h2>Locomotion</h2>
      <label>Max vx <span class="value" id="max_vx_mps_v"></span><input id="max_vx_mps" type="range" min="0.02" max="0.45" step="0.01"></label>
      <label>Max yaw <span class="value" id="max_vyaw_rps_v"></span><input id="max_vyaw_rps" type="range" min="0.05" max="1.0" step="0.01"></label>
      <label>Forward gain <span class="value" id="vx_gain_v"></span><input id="vx_gain" type="range" min="0.05" max="1.2" step="0.01"></label>
      <label>Yaw gain <span class="value" id="yaw_gain_v"></span><input id="yaw_gain" type="range" min="0.05" max="1.2" step="0.01"></label>
      <label>Command duration <span class="value" id="command_duration_s_v"></span><input id="command_duration_s" type="range" min="0.05" max="1.0" step="0.01"></label>
    </section>
    <section class="panel"><h2>Status</h2><pre id="status"></pre></section>
  </aside>
</main>
<script>
const keys = ['target_distance_m','distance_tolerance_m','center_tolerance_m','max_vx_mps','max_vyaw_rps','vx_gain','yaw_gain','command_duration_s'];
let dirty = false;
function payload() {
  const out = {};
  for (const k of keys) out[k] = Number(document.getElementById(k).value);
  return out;
}
async function post(path) {
  const res = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload())});
  render(await res.json());
}
async function saveSettings() { await post('/api/settings'); dirty = false; }
for (const k of keys) document.addEventListener('input', e => {
  if (e.target && e.target.id === k) { dirty = true; document.getElementById(k + '_v').textContent = Number(e.target.value).toFixed(2); }
});
function render(data) {
  if (data.settings && !dirty) {
    for (const k of keys) {
      const el = document.getElementById(k);
      el.value = data.settings[k];
      document.getElementById(k + '_v').textContent = Number(data.settings[k]).toFixed(2);
    }
  }
  document.getElementById('status').textContent = JSON.stringify(data, null, 2);
}
async function refresh() {
  document.getElementById('rgb').src = '/rgb.jpg?t=' + Date.now();
  const res = await fetch('/api/status?t=' + Date.now());
  render(await res.json());
}
setInterval(refresh, 200); refresh();
</script></body></html>"""


def main() -> int:
    args = parse_args()
    app = FollowMeApp(args)
    try:
        app.start()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
