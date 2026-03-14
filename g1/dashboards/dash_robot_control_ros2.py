#!/usr/bin/env python3
"""
Robot Control Dashboard — ROS2 native version.

Replaces the SDK-based dash_robot_control.py with direct rclpy publishers /
subscribers using unitree_api and unitree_go messages from this repository.

The app starts with no environment requirements.  Use the connection bar at
the top to auto-source the ROS2 workspace and connect the node.

Optional env vars (read at startup; also updated by Auto-setup):
    LIDAR_TOPIC       (default: rt/utlidar/cloud_livox_mid360)
    SPORT_STATE_TOPIC (default: lf/sportmodestate)
    LOW_STATE_TOPIC   (default: hf/lowstate)
    RGB_PORT          (default: 4000)
    DEPTH_PORT        (default: 4001)
    VIDEO_WIDTH       (default: 640)
    VIDEO_HEIGHT      (default: 480)
    VIDEO_FPS         (default: 30)
    LIVOX_MOUNT       (default: upside_down, or normal)
    HOST_IP           (default: 192.168.123.222, used by Livox SDK fallback)
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

# ---------------------------------------------------------------------------
# Sport API IDs  (source: example/src/include/common/ros2_sport_client.h)
# ---------------------------------------------------------------------------
SPORT_DAMP = 1001
SPORT_BALANCE_STAND = 1002
SPORT_STOP_MOVE = 1003
SPORT_STAND_UP = 1004
SPORT_STAND_DOWN = 1005
SPORT_RECOVERY_STAND = 1006
SPORT_MOVE = 1008
SPORT_STATIC_WALK = 1061
SPORT_TROT_RUN = 1062

# ---------------------------------------------------------------------------
# Configuration from environment (re-read inside helpers after auto-setup)
# ---------------------------------------------------------------------------
LIDAR_TOPIC = os.environ.get("LIDAR_TOPIC", "rt/utlidar/cloud_livox_mid360")
SPORT_STATE_TOPIC = os.environ.get("SPORT_STATE_TOPIC", "lf/sportmodestate")
LOW_STATE_TOPIC = os.environ.get("LOW_STATE_TOPIC", "hf/lowstate")

RGB_PORT = int(os.environ.get("RGB_PORT", "4000"))
DEPTH_PORT = int(os.environ.get("DEPTH_PORT", "4001"))
VIDEO_WIDTH = int(os.environ.get("VIDEO_WIDTH", "640"))
VIDEO_HEIGHT = int(os.environ.get("VIDEO_HEIGHT", "480"))
VIDEO_FPS = int(os.environ.get("VIDEO_FPS", "30"))

# ---------------------------------------------------------------------------
# IMU history buffer
# ---------------------------------------------------------------------------
IMU_HISTORY: deque[tuple[float, float, float, float]] = deque(maxlen=300)

# ---------------------------------------------------------------------------
# PointCloud2 parser
# ---------------------------------------------------------------------------

def _parse_pointcloud2_xyz(
    msg: Any, max_points: int = 4000
) -> list[tuple[float, float, float]]:
    """Extract up to *max_points* (x, y, z) tuples from a PointCloud2 message."""
    try:
        import numpy as np

        n = msg.width * msg.height
        if n == 0:
            return []
        step = msg.point_step
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, step)

        offsets: dict[str, int] = {
            f.name: f.offset for f in msg.fields if f.name in ("x", "y", "z")
        }
        if len(offsets) < 3:
            return []

        def _col(offset: int) -> np.ndarray:
            return np.frombuffer(
                raw[:, offset : offset + 4].tobytes(), dtype=np.float32
            )

        xs = _col(offsets["x"])
        ys = _col(offsets["y"])
        zs = _col(offsets["z"])

        valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
        xs, ys, zs = xs[valid], ys[valid], zs[valid]

        if len(xs) > max_points:
            stride = max(1, len(xs) // max_points)
            xs, ys, zs = xs[::stride], ys[::stride], zs[::stride]

        return list(zip(xs.tolist(), ys.tolist(), zs.tolist()))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# ROS2 environment setup
# ---------------------------------------------------------------------------

def _setup_ros2_environment() -> tuple[bool, str]:
    """
    Locate a ROS2 installation under /opt/ros, source it (plus the repo's
    cyclonedds_ws if present), and apply the resulting env vars to the current
    process so that rclpy / unitree_api become importable.

    Returns (success, message).
    """
    ros2_root = Path("/opt/ros")
    distros = ["jazzy", "humble", "iron", "rolling", "foxy", "galactic", "noetic"]
    found: str | None = None
    for d in distros:
        if (ros2_root / d).exists():
            found = d
            break

    if found is None:
        return False, "No ROS2 installation found under /opt/ros/."

    repo_root = Path(__file__).resolve().parent.parent
    cyclone_setup = repo_root / "cyclonedds_ws" / "install" / "setup.bash"

    cmds = [f"source /opt/ros/{found}/setup.bash"]
    if cyclone_setup.exists():
        cmds.append(f"source {cyclone_setup}")
    cmds += ["export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp", "env"]

    try:
        result = subprocess.run(
            ["bash", "-c", " && ".join(cmds)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return False, f"subprocess error: {exc}"

    if result.returncode != 0:
        return False, f"source failed: {result.stderr[:300]}"

    # Parse env output and update current process
    new_env: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            new_env[key] = val

    propagate_keys = (
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "PATH",
        "RMW_IMPLEMENTATION",
        "AMENT_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "CMAKE_PREFIX_PATH",
    )
    for key in propagate_keys:
        if key in new_env:
            os.environ[key] = new_env[key]

    if "PYTHONPATH" in new_env:
        for p in new_env["PYTHONPATH"].split(":"):
            if p and p not in sys.path:
                sys.path.insert(0, p)

    note = f"Sourced /opt/ros/{found}"
    if cyclone_setup.exists():
        note += f" + cyclonedds_ws"
    note += " | RMW=rmw_cyclonedds_cpp"
    return True, note


# ---------------------------------------------------------------------------
# ROS2 node (background thread, deferred start)
# ---------------------------------------------------------------------------

_rclpy_init_called = False  # rclpy.init() may only be called once per process


class _Ros2Node:
    """
    Wraps a rclpy node in a background daemon thread.

    Call connect() explicitly (via the Connect button) rather than connecting
    at construction time, so the dashboard can start without ROS2 installed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sport_state: Any = None
        self._low_state: Any = None
        self._lidar_pts: list[tuple[float, float, float]] | None = None
        self._lidar_ts: float = 0.0
        self._sport_state_ts: float = 0.0
        self._low_state_ts: float = 0.0
        self._status: str = "disconnected"   # disconnected | connecting | connected | error
        self._init_error: str | None = None
        self._node: Any = None
        self._pub: Any = None

    # --- public API -----------------------------------------------------

    def connect(self) -> None:
        """Start the ROS2 node (no-op if already connecting or connected)."""
        with self._lock:
            if self._status in ("connecting", "connected"):
                return
            self._status = "connecting"
            self._init_error = None

        ready = threading.Event()
        t = threading.Thread(target=self._init_and_spin, args=(ready,), daemon=True)
        t.start()
        ready.wait(timeout=10.0)

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def init_error(self) -> str | None:
        with self._lock:
            return self._init_error

    def status_badge(self) -> tuple[str, str]:
        """Return (label, bootstrap_color) for the connection status badge."""
        s = self.status
        if s == "connected":
            return "Connected", "success"
        if s == "connecting":
            return "Connecting…", "warning"
        if s == "error":
            return "Error", "danger"
        return "Disconnected", "secondary"

    # --- ROS2 init / spin -----------------------------------------------

    def _init_and_spin(self, ready: threading.Event) -> None:
        global _rclpy_init_called
        try:
            import rclpy
            from sensor_msgs.msg import PointCloud2
            from unitree_api.msg import Request
            from unitree_go.msg import LowState, SportModeState

            if not _rclpy_init_called:
                rclpy.init()
                _rclpy_init_called = True

            node = rclpy.create_node("dashboard_ros2_node")
            pub = node.create_publisher(Request, "/api/sport/request", 10)
            node.create_subscription(
                SportModeState, SPORT_STATE_TOPIC, self._sport_cb, 1
            )
            node.create_subscription(LowState, LOW_STATE_TOPIC, self._low_cb, 1)
            node.create_subscription(
                PointCloud2, LIDAR_TOPIC, self._lidar_cb, 1
            )

            with self._lock:
                self._node = node
                self._pub = pub
                self._status = "connected"

            ready.set()
            rclpy.spin(node)
        except Exception as exc:
            with self._lock:
                self._status = "error"
                self._init_error = str(exc)
            ready.set()

    # --- ROS2 topic callbacks -------------------------------------------

    def _sport_cb(self, msg: Any) -> None:
        with self._lock:
            self._sport_state = msg
            self._sport_state_ts = time.time()

    def _low_cb(self, msg: Any) -> None:
        with self._lock:
            self._low_state = msg
            self._low_state_ts = time.time()

    def _lidar_cb(self, msg: Any) -> None:
        pts = _parse_pointcloud2_xyz(msg, max_points=4000)
        with self._lock:
            self._lidar_pts = pts
            self._lidar_ts = time.time()

    # --- sport command publishers ---------------------------------------

    def _publish(self, api_id: int, parameter: str = "") -> None:
        with self._lock:
            pub = self._pub
            status = self._status
        if status != "connected" or pub is None:
            raise RuntimeError(f"Not connected (status={status})")
        from unitree_api.msg import Request

        req = Request()
        req.header.identity.api_id = api_id
        req.parameter = parameter
        pub.publish(req)

    def damp(self) -> None:
        self._publish(SPORT_DAMP)

    def zero_torque(self) -> None:
        """Sport API has no separate zero-torque; Damp (1001) is the closest."""
        self._publish(SPORT_DAMP)

    def stop(self) -> None:
        self._publish(SPORT_STOP_MOVE)

    def stand_up(self) -> None:
        self._publish(SPORT_STAND_UP)

    def balance_stand(self) -> None:
        self._publish(SPORT_BALANCE_STAND)

    def recovery_stand(self) -> None:
        self._publish(SPORT_RECOVERY_STAND)

    def move(self, vx: float, vy: float, vyaw: float) -> None:
        param = json.dumps({"x": float(vx), "y": float(vy), "z": float(vyaw)})
        self._publish(SPORT_MOVE, param)

    def set_gait_walk(self) -> None:
        self._publish(SPORT_STATIC_WALK)

    def set_gait_run(self) -> None:
        self._publish(SPORT_TROT_RUN)

    # --- state getters --------------------------------------------------

    def get_imu_rpy(self) -> tuple[float, float, float] | None:
        with self._lock:
            state = self._sport_state if self._sport_state is not None else self._low_state
            if state is None:
                return None
            rpy = state.imu_state.rpy
            return (float(rpy[0]), float(rpy[1]), float(rpy[2]))

    def get_lidar_points(self) -> tuple[list[tuple[float, float, float]], float]:
        with self._lock:
            return (self._lidar_pts or [], self._lidar_ts)

    def get_sensor_timestamps(self) -> dict[str, float]:
        with self._lock:
            return {
                "sport_state": self._sport_state_ts,
                "low_state": self._low_state_ts,
                "lidar_cloud": self._lidar_ts,
            }

    def sensors_stale(self, max_age: float = 1.5) -> bool:
        ts = self.get_sensor_timestamps()
        now = time.time()
        return all((now - v) > max_age for v in ts.values() if v > 0)


# ---------------------------------------------------------------------------
# Node singleton
# ---------------------------------------------------------------------------
_NODE_LOCK = threading.Lock()
_NODE_INSTANCE: _Ros2Node | None = None


def get_node() -> _Ros2Node:
    """Return the singleton node (creates it, does NOT auto-connect)."""
    global _NODE_INSTANCE
    with _NODE_LOCK:
        if _NODE_INSTANCE is None:
            _NODE_INSTANCE = _Ros2Node()
        return _NODE_INSTANCE


def _reset_node() -> None:
    """Discard the singleton so a fresh connect() attempt can be made.
    Only safe when rclpy.init() has not yet been called."""
    global _NODE_INSTANCE
    with _NODE_LOCK:
        if not _rclpy_init_called:
            _NODE_INSTANCE = None


# ---------------------------------------------------------------------------
# GStreamer video receivers (hardware-level UDP, independent of ROS2)
# ---------------------------------------------------------------------------

class _RgbPreviewReceiver:
    def __init__(self, rgb_port: int, width: int, height: int, fps: int) -> None:
        self.rgb_port = int(rgb_port)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_ts = 0.0
        self._error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def snapshot(self) -> tuple[bytes | None, float, str | None]:
        with self._lock:
            return (self._latest_jpeg, self._latest_ts, self._error)

    def _run(self) -> None:
        pipeline = None
        try:
            import cv2
            import gi
            import numpy as np

            gi.require_version("Gst", "1.0")
            gi.require_version("GstApp", "1.0")
            from gi.repository import Gst
        except Exception as exc:
            with self._lock:
                self._error = f"RGB receiver unavailable: {exc}"
            return
        try:
            Gst.init(None)
            pipeline = Gst.parse_launch(
                f"udpsrc port={self.rgb_port} caps=application/x-rtp,media=video,"
                "encoding-name=H264,payload=96 ! "
                "rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=true sync=false drop=true"
            )
            sink = pipeline.get_by_name("sink")
            if sink is None:
                raise RuntimeError("appsink not found")
            pipeline.set_state(Gst.State.PLAYING)
            wait_ns = int(Gst.SECOND // self.fps)
            while self._running:
                sample = sink.emit("try-pull-sample", wait_ns)
                if not sample:
                    time.sleep(0.01)
                    continue
                buf = sample.get_buffer()
                if buf is None:
                    continue
                raw = np.frombuffer(buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
                expected = self.width * self.height * 3
                if raw.size != expected:
                    continue
                bgr = raw.reshape((self.height, self.width, 3))
                ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    continue
                with self._lock:
                    self._latest_jpeg = enc.tobytes()
                    self._latest_ts = time.time()
                    self._error = None
        except Exception as exc:
            with self._lock:
                self._error = f"RGB stream error: {exc}"
        finally:
            try:
                if pipeline is not None:
                    pipeline.set_state(Gst.State.NULL)  # type: ignore[name-defined]
            except Exception:
                pass


class _DepthPreviewReceiver:
    def __init__(self, depth_port: int, width: int, height: int, fps: int) -> None:
        self.depth_port = int(depth_port)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_ts = 0.0
        self._latest_center_depth_m: float | None = None
        self._latest_near_coverage: float | None = None
        self._error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def snapshot(self) -> tuple[bytes | None, float, float | None, float | None, str | None]:
        with self._lock:
            return (
                self._latest_jpeg,
                self._latest_ts,
                self._latest_center_depth_m,
                self._latest_near_coverage,
                self._error,
            )

    def _run(self) -> None:
        pipeline = None
        try:
            import cv2
            import gi
            import numpy as np

            gi.require_version("Gst", "1.0")
            gi.require_version("GstApp", "1.0")
            from gi.repository import Gst
        except Exception as exc:
            with self._lock:
                self._error = f"Depth receiver unavailable: {exc}"
            return
        try:
            Gst.init(None)
            pipeline = Gst.parse_launch(
                f"udpsrc port={self.depth_port} caps=application/x-rtp,media=video,"
                "encoding-name=H264,payload=97 ! "
                "rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=true sync=false drop=true"
            )
            sink = pipeline.get_by_name("sink")
            if sink is None:
                raise RuntimeError("appsink not found")
            pipeline.set_state(Gst.State.PLAYING)
            cmap = cv2.applyColorMap(
                np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_PLASMA
            )
            cmap = cmap.reshape(256, 3).astype(np.int16)
            wait_ns = int(Gst.SECOND // self.fps)
            while self._running:
                sample = sink.emit("try-pull-sample", wait_ns)
                if not sample:
                    time.sleep(0.01)
                    continue
                buf = sample.get_buffer()
                if buf is None:
                    continue
                raw = np.frombuffer(buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
                expected = self.width * self.height * 3
                if raw.size != expected:
                    continue
                depth_bgr = raw.reshape((self.height, self.width, 3))
                ok, enc = cv2.imencode(".jpg", depth_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                center_size = max(8, min(self.width, self.height) // 12)
                cx, cy = self.width // 2, self.height // 2
                x0, x1 = max(0, cx - center_size), min(self.width, cx + center_size)
                y0, y1 = max(0, cy - center_size), min(self.height, cy + center_size)
                center = depth_bgr[y0:y1, x0:x1]
                roi = depth_bgr[
                    int(self.height * 0.25) : int(self.height * 0.70),
                    int(self.width * 0.30) : int(self.width * 0.70),
                ]
                center_depth_m: float | None = None
                near_cov: float | None = None
                if center.size > 0 and roi.size > 0:
                    cp = center.reshape(-1, 3).astype(np.int16)
                    diff = cp[:, None, :] - cmap[None, :, :]
                    center_idx = np.argmin((diff * diff).sum(axis=2), axis=1)
                    center_depth_m = float(np.median(center_idx) / 255.0 * 6.0)
                    rp = roi.reshape(-1, 3).astype(np.int16)
                    rdiff = rp[:, None, :] - cmap[None, :, :]
                    roi_idx = np.argmin((rdiff * rdiff).sum(axis=2), axis=1)
                    near_cov = float(np.mean(roi_idx <= int((1.0 / 6.0) * 255.0)))
                with self._lock:
                    self._latest_jpeg = enc.tobytes()
                    self._latest_ts = time.time()
                    self._latest_center_depth_m = center_depth_m
                    self._latest_near_coverage = near_cov
                    self._error = None
        except Exception as exc:
            with self._lock:
                self._error = f"Depth stream error: {exc}"
        finally:
            try:
                if pipeline is not None:
                    pipeline.set_state(Gst.State.NULL)  # type: ignore[name-defined]
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Livox SDK fallback
# ---------------------------------------------------------------------------

class _LivoxPointsReceiver:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._frames_xyz: deque[Any] = deque(maxlen=15)
        self._latest_ts = 0.0
        self._error: str | None = None
        self._mount = os.environ.get("LIVOX_MOUNT", "upside_down").lower()
        if self._mount not in {"normal", "upside_down"}:
            self._mount = "upside_down"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def snapshot(self) -> tuple[Any | None, float, str | None]:
        with self._lock:
            if not self._frames_xyz:
                return None, self._latest_ts, self._error
            try:
                import numpy as np

                merged = np.concatenate(list(self._frames_xyz), axis=0)
            except Exception as exc:
                self._error = f"Livox frame merge failed: {exc}"
                return None, self._latest_ts, self._error
            return merged, self._latest_ts, self._error

    def _run(self) -> None:
        try:
            import numpy as np
        except Exception as exc:
            with self._lock:
                self._error = f"numpy unavailable: {exc}"
            return

        sensors_dir = Path("/home/ag/ef_ws/g1/scripts/sensors")
        if str(sensors_dir) not in sys.path:
            sys.path.insert(0, str(sensors_dir))

        base_cls = None
        sdk2 = False
        try:
            from livox2_python import Livox2 as base_cls  # type: ignore[assignment]

            sdk2 = True
        except Exception:
            try:
                from livox_python import Livox as base_cls  # type: ignore[assignment]
            except Exception as exc:
                with self._lock:
                    self._error = f"Livox wrapper import failed: {exc}"
                return

        receiver = self

        class _DashLivox(base_cls):  # type: ignore[misc, valid-type]
            def __init__(self) -> None:
                if sdk2:
                    cfg = sensors_dir / "mid360_config.json"
                    if not cfg.exists():
                        raise RuntimeError(f"Missing Livox config: {cfg}")
                    host_ip = os.environ.get("HOST_IP", "192.168.123.222")
                    super().__init__(str(cfg), host_ip=host_ip)
                else:
                    super().__init__()

            def handle_points(self, xyz: Any) -> None:  # type: ignore[override]
                import numpy as np

                arr = np.asarray(xyz, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] < 3:
                    return
                arr = arr[:, :3]
                if receiver._mount == "upside_down":
                    arr = arr * np.array([1.0, -1.0, -1.0], dtype=np.float32)
                if arr.shape[0] > 100_000:
                    step = max(1, arr.shape[0] // 100_000)
                    arr = arr[::step]
                with receiver._lock:
                    receiver._frames_xyz.append(arr)
                    receiver._latest_ts = time.time()
                    receiver._error = None

        lidar = None
        try:
            lidar = _DashLivox()
            while self._running:
                time.sleep(0.02)
        except Exception as exc:
            with self._lock:
                self._error = f"Livox stream error: {exc}"
        finally:
            try:
                if lidar is not None:
                    lidar.shutdown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Video preview singletons
# ---------------------------------------------------------------------------
_DEPTH_LOCK = threading.Lock()
_DEPTH_PREVIEW: _DepthPreviewReceiver | None = None

_RGB_LOCK = threading.Lock()
_RGB_PREVIEW: _RgbPreviewReceiver | None = None

_LIVOX_LOCK = threading.Lock()
_LIVOX_PREVIEW: _LivoxPointsReceiver | None = None


def _get_rgb_preview() -> _RgbPreviewReceiver:
    global _RGB_PREVIEW
    with _RGB_LOCK:
        if _RGB_PREVIEW is None:
            _RGB_PREVIEW = _RgbPreviewReceiver(RGB_PORT, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS)
        return _RGB_PREVIEW


def _get_depth_preview() -> _DepthPreviewReceiver:
    global _DEPTH_PREVIEW
    with _DEPTH_LOCK:
        if _DEPTH_PREVIEW is None:
            _DEPTH_PREVIEW = _DepthPreviewReceiver(
                DEPTH_PORT, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS
            )
        return _DEPTH_PREVIEW


def _get_livox_preview() -> _LivoxPointsReceiver:
    global _LIVOX_PREVIEW
    with _LIVOX_LOCK:
        if _LIVOX_PREVIEW is None:
            _LIVOX_PREVIEW = _LivoxPointsReceiver()
        return _LIVOX_PREVIEW


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def _empty_lidar_figure(title: str = "LiDAR stream") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        margin={"l": 30, "r": 20, "t": 45, "b": 35},
        height=500,
    )
    return fig


def _empty_imu_figure(title: str = "IMU orientation (RPY)") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Time (s, recent window)",
        yaxis_title="Angle (rad)",
        margin={"l": 30, "r": 20, "t": 45, "b": 35},
        height=320,
    )
    return fig


# ---------------------------------------------------------------------------
# Dash application
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Robot Control (ROS2)"

app.layout = dbc.Container(
    [
        html.H3("Robot Control Dashboard (ROS2)", className="mt-3 mb-3"),

        # ----------------------------------------------------------------
        # Connection bar
        # ----------------------------------------------------------------
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Span("Status: ", className="fw-bold me-1"),
                                dbc.Badge("Disconnected", id="conn-badge", color="secondary"),
                            ],
                            md=4,
                            className="d-flex align-items-center",
                        ),
                        dbc.Col(
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        "Auto-setup ROS2 Environment",
                                        id="btn-autosetup",
                                        color="info",
                                        outline=True,
                                        size="sm",
                                    ),
                                    dbc.Button(
                                        "Connect",
                                        id="btn-connect",
                                        color="success",
                                        size="sm",
                                    ),
                                ]
                            ),
                            md=8,
                            className="d-flex justify-content-end align-items-center",
                        ),
                    ]
                )
            ),
            className="mb-2",
        ),
        html.Div(id="conn-result", className="mb-3 small text-muted"),
        dcc.Interval(id="conn-interval", interval=2000, n_intervals=0),

        dbc.Alert(id="status-alert", color="secondary", children="Ready", className="mb-3"),
        dbc.Tabs(
            [
                # ----------------------------------------------------------
                # Tab 1 — Control
                # ----------------------------------------------------------
                dbc.Tab(
                    label="Control",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button("Damp", id="btn-damp", color="warning", className="w-100"),
                                    md=4,
                                ),
                                dbc.Col(
                                    dbc.Button("Zero Torque", id="btn-zero", color="danger", className="w-100"),
                                    md=4,
                                ),
                                dbc.Col(
                                    dbc.Button("Stop", id="btn-stop", color="secondary", className="w-100"),
                                    md=4,
                                ),
                            ],
                            className="g-2 mt-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button("Stand Up", id="btn-stand-up", color="primary", className="w-100 mt-2"),
                                    md=4,
                                ),
                                dbc.Col(
                                    dbc.Button("Balance Stand", id="btn-balance-stand", color="primary", className="w-100 mt-2"),
                                    md=4,
                                ),
                                dbc.Col(
                                    dbc.Button("Recovery Stand", id="btn-recovery-stand", color="primary", className="w-100 mt-2"),
                                    md=4,
                                ),
                            ],
                            className="g-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div("Gait Type", className="mt-3 mb-1"),
                                        dbc.RadioItems(
                                            id="gait-toggle",
                                            options=[
                                                {"label": "Walk (StaticWalk 1061)", "value": "walk"},
                                                {"label": "Run (TrotRun 1062)", "value": "run"},
                                            ],
                                            value="walk",
                                            inline=True,
                                        ),
                                    ],
                                    md=12,
                                ),
                            ],
                            className="g-2",
                        ),
                        html.Div(id="control-result", className="mt-3"),
                    ],
                ),
                # ----------------------------------------------------------
                # Tab 2 — Navigation
                # ----------------------------------------------------------
                dbc.Tab(
                    label="Navigation",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div("vx (m/s)", className="mt-3 mb-1"),
                                        dbc.Input(id="nav-vx", type="number", value=0.0, step=0.05),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Div("vy (m/s)", className="mt-3 mb-1"),
                                        dbc.Input(id="nav-vy", type="number", value=0.0, step=0.05),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Div("vyaw (rad/s)", className="mt-3 mb-1"),
                                        dbc.Input(id="nav-vyaw", type="number", value=0.0, step=0.05),
                                    ],
                                    md=4,
                                ),
                            ],
                            className="g-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "move(vx, vy, vyaw)",
                                        id="btn-nav-move",
                                        color="primary",
                                        className="w-100 mt-3",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Stop Move",
                                        id="btn-nav-stop",
                                        color="secondary",
                                        className="w-100 mt-3",
                                    ),
                                    md=6,
                                ),
                            ],
                            className="g-2",
                        ),
                        html.Div(id="nav-result", className="mt-3"),
                    ],
                ),
                # ----------------------------------------------------------
                # Tab 3 — Sensors
                # ----------------------------------------------------------
                dbc.Tab(
                    label="Sensors",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            f"RGB camera feed (UDP port {RGB_PORT})",
                                            className="mt-3 mb-2",
                                        ),
                                        html.Img(
                                            id="rgb-feed",
                                            style={
                                                "width": "100%",
                                                "border": "1px solid #444",
                                                "borderRadius": "8px",
                                            },
                                        ),
                                        html.Div(id="rgb-status", className="mb-3"),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Div(
                                            f"Depth camera feed (PLASMA, UDP port {DEPTH_PORT})",
                                            className="mt-3 mb-2",
                                        ),
                                        html.Img(
                                            id="depth-feed",
                                            style={
                                                "width": "100%",
                                                "border": "1px solid #444",
                                                "borderRadius": "8px",
                                            },
                                        ),
                                        html.Div(id="depth-status", className="mb-3"),
                                    ],
                                    md=6,
                                ),
                            ],
                            className="g-2",
                        ),
                        html.Hr(),
                        html.Div(
                            f"LiDAR stream (topic: {LIDAR_TOPIC})",
                            className="mt-3 mb-2",
                        ),
                        dcc.Graph(id="lidar-graph", figure=_empty_lidar_figure()),
                        html.Div(id="lidar-status", className="mb-3"),
                        html.Hr(),
                        html.Div("IMU orientation (roll/pitch/yaw)", className="mt-3 mb-2"),
                        dcc.Graph(id="imu-graph", figure=_empty_imu_figure()),
                        html.Div(id="imu-status", className="mb-3"),
                    ],
                ),
                # ----------------------------------------------------------
                # Tab 4 — Speech
                # ----------------------------------------------------------
                dbc.Tab(
                    label="Speech",
                    children=[
                        dbc.InputGroup(
                            [
                                dbc.Input(id="say-text", placeholder="Type text to speak", type="text"),
                                dbc.Button("Say", id="btn-say", color="success"),
                            ],
                            className="mt-3",
                        ),
                        html.Div(
                            "Uses espeak/espeak-ng subprocess (sudo apt install espeak-ng).",
                            className="text-muted mt-1 small",
                        ),
                        html.Div(id="say-result", className="mt-3"),
                    ],
                ),
                # ----------------------------------------------------------
                # Tab 5 — Settings
                # ----------------------------------------------------------
                dbc.Tab(
                    label="Settings",
                    children=[
                        html.H6("ROS2 Topics", className="mt-3"),
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem(f"Sport commands → /api/sport/request"),
                                dbc.ListGroupItem(f"Sport state ← {SPORT_STATE_TOPIC}"),
                                dbc.ListGroupItem(f"Low state ← {LOW_STATE_TOPIC}"),
                                dbc.ListGroupItem(f"LiDAR ← {LIDAR_TOPIC}"),
                            ],
                            className="mb-3",
                        ),
                        html.H6("Video Stream Ports"),
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem(
                                    f"RGB  UDP port: {RGB_PORT}  ({VIDEO_WIDTH}×{VIDEO_HEIGHT} @ {VIDEO_FPS} fps)"
                                ),
                                dbc.ListGroupItem(f"Depth UDP port: {DEPTH_PORT}"),
                            ],
                            className="mb-3",
                        ),
                        dbc.Alert(
                            "Topics and ports are configured via environment variables. "
                            "Restart the dashboard to apply changes, or use Auto-setup to "
                            "source the ROS2 workspace before clicking Connect.",
                            color="info",
                        ),
                        html.H6("Node Status", className="mt-3"),
                        html.Div(id="ros2-node-status", className="mt-2"),
                        dbc.Button(
                            "Refresh",
                            id="btn-ros2-status",
                            color="secondary",
                            size="sm",
                            className="mt-2",
                        ),
                    ],
                ),
            ],
            className="mb-3",
        ),
        dcc.Interval(id="lidar-interval", interval=1000, n_intervals=0),
        dcc.Interval(id="rgb-interval", interval=500, n_intervals=0),
        dcc.Interval(id="depth-interval", interval=500, n_intervals=0),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# Callbacks — Connection bar
# ---------------------------------------------------------------------------

@app.callback(
    Output("conn-badge", "children"),
    Output("conn-badge", "color"),
    Output("conn-result", "children"),
    Input("btn-autosetup", "n_clicks"),
    Input("btn-connect", "n_clicks"),
    Input("conn-interval", "n_intervals"),
    prevent_initial_call=True,
)
def on_connection_action(
    _autosetup: int | None,
    _connect: int | None,
    _tick: int,
) -> tuple[str, str, str]:
    node = get_node()
    trigger = dash.ctx.triggered_id
    detail = ""

    if trigger == "btn-autosetup":
        ok, msg = _setup_ros2_environment()
        detail = msg
        if ok:
            # Reset node singleton if rclpy was never initialized so the
            # freshly updated sys.path is used on the next connect attempt.
            _reset_node()
        label, color = get_node().status_badge()
        return label, color, detail

    if trigger == "btn-connect":
        # Allow re-attempt if previous attempt never reached rclpy.init()
        if node.status == "error" and not _rclpy_init_called:
            _reset_node()
            node = get_node()
        node.connect()

    label, color = get_node().status_badge()
    err = get_node().init_error
    if err:
        detail = f"Error: {err}"
    return label, color, detail


# ---------------------------------------------------------------------------
# Callbacks — Control tab
# ---------------------------------------------------------------------------

@app.callback(
    Output("status-alert", "children"),
    Output("status-alert", "color"),
    Output("control-result", "children"),
    Input("btn-damp", "n_clicks"),
    Input("btn-zero", "n_clicks"),
    Input("btn-stop", "n_clicks"),
    Input("btn-stand-up", "n_clicks"),
    Input("btn-balance-stand", "n_clicks"),
    Input("btn-recovery-stand", "n_clicks"),
    Input("gait-toggle", "value"),
    prevent_initial_call=True,
)
def on_control(
    _damp: int | None,
    _zero: int | None,
    _stop: int | None,
    _stand_up: int | None,
    _balance_stand: int | None,
    _recovery_stand: int | None,
    gait_value: str,
) -> tuple[str, str, str]:
    node = get_node()
    if node.status != "connected":
        label, _ = node.status_badge()
        return f"Not connected ({label}). Use the Connect button above.", "warning", ""

    trigger = dash.ctx.triggered_id
    try:
        if trigger == "btn-damp":
            node.damp()
            return "Damp command sent.", "warning", "Sport API 1001 — Damp published."
        if trigger == "btn-zero":
            node.zero_torque()
            return "Zero torque command sent.", "danger", "Sport API 1001 — Damp (closest equivalent) published."
        if trigger == "btn-stop":
            node.stop()
            return "Stop command sent.", "secondary", "Sport API 1003 — StopMove published."
        if trigger == "btn-stand-up":
            node.stand_up()
            return "Stand Up sent.", "success", "Sport API 1004 — StandUp published."
        if trigger == "btn-balance-stand":
            node.balance_stand()
            return "Balance Stand sent.", "primary", "Sport API 1002 — BalanceStand published."
        if trigger == "btn-recovery-stand":
            node.recovery_stand()
            return "Recovery Stand sent.", "primary", "Sport API 1006 — RecoveryStand published."
        if trigger == "gait-toggle":
            if gait_value == "run":
                node.set_gait_run()
                return "Gait → run.", "primary", "Sport API 1062 — TrotRun published."
            node.set_gait_walk()
            return "Gait → walk.", "primary", "Sport API 1061 — StaticWalk published."
    except Exception as exc:
        return f"Command failed: {exc}", "danger", str(exc)

    return "Ready", "secondary", "No action taken."


# ---------------------------------------------------------------------------
# Callbacks — Navigation tab
# ---------------------------------------------------------------------------

@app.callback(
    Output("nav-result", "children"),
    Input("btn-nav-move", "n_clicks"),
    Input("btn-nav-stop", "n_clicks"),
    State("nav-vx", "value"),
    State("nav-vy", "value"),
    State("nav-vyaw", "value"),
    prevent_initial_call=True,
)
def on_navigation(
    _move: int | None,
    _stop: int | None,
    vx: float | None,
    vy: float | None,
    vyaw: float | None,
) -> str:
    node = get_node()
    if node.status != "connected":
        label, _ = node.status_badge()
        return f"Not connected ({label}). Use the Connect button above."

    trigger = dash.ctx.triggered_id
    try:
        if trigger == "btn-nav-stop":
            node.stop()
            return "StopMove (API 1003) sent."
        if trigger == "btn-nav-move":
            cmd_vx = float(vx or 0.0)
            cmd_vy = float(vy or 0.0)
            cmd_vyaw = float(vyaw or 0.0)
            node.move(cmd_vx, cmd_vy, cmd_vyaw)
            return (
                f"Move (API 1008) sent — "
                f"vx={cmd_vx:.3f} m/s, vy={cmd_vy:.3f} m/s, vyaw={cmd_vyaw:.3f} rad/s"
            )
    except Exception as exc:
        return f"Navigation command failed: {exc}"

    return "No navigation action taken."


# ---------------------------------------------------------------------------
# Callbacks — Speech tab
# ---------------------------------------------------------------------------

@app.callback(
    Output("say-result", "children"),
    Input("btn-say", "n_clicks"),
    State("say-text", "value"),
    prevent_initial_call=True,
)
def on_say(_n: int | None, text: str | None) -> str:
    phrase = (text or "").strip()
    if not phrase:
        return "Enter text before pressing Say."
    try:
        for cmd in (["espeak-ng", phrase], ["espeak", phrase]):
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode == 0:
                return f"Said: {phrase}"
        return "Say failed: neither espeak-ng nor espeak found."
    except FileNotFoundError:
        return "Say failed: install espeak-ng (sudo apt install espeak-ng)."
    except Exception as exc:
        return f"Say failed: {exc}"


# ---------------------------------------------------------------------------
# Callbacks — Settings tab
# ---------------------------------------------------------------------------

@app.callback(
    Output("ros2-node-status", "children"),
    Input("btn-ros2-status", "n_clicks"),
    prevent_initial_call=True,
)
def on_ros2_status(_n: int | None) -> Any:
    node = get_node()
    label, color = node.status_badge()
    err = node.init_error
    ts = node.get_sensor_timestamps()
    now = time.time()
    rows = [dbc.ListGroupItem(f"Node status: {label}", color=color)]
    if err:
        rows.append(dbc.ListGroupItem(f"Error: {err}", color="danger"))
    for name, t in ts.items():
        age = f"{now - t:.1f}s ago" if t > 0 else "no data"
        rows.append(dbc.ListGroupItem(f"{name}: {age}"))
    return dbc.ListGroup(rows)


# ---------------------------------------------------------------------------
# Callbacks — RGB feed
# ---------------------------------------------------------------------------

@app.callback(
    Output("rgb-feed", "src"),
    Output("rgb-status", "children"),
    Input("rgb-interval", "n_intervals"),
    State("rgb-feed", "src"),
)
def update_rgb_feed(_tick: int, prev_src: str | None) -> tuple[str | None, str]:
    try:
        preview = _get_rgb_preview()
        preview.start()
        jpeg, ts, err = preview.snapshot()
        if err is not None:
            return prev_src, err
        if jpeg is None:
            return prev_src, f"Waiting for RGB frames on UDP port {RGB_PORT}."
        payload = base64.b64encode(jpeg).decode("ascii")
        age_s = max(0.0, time.time() - ts) if ts > 0 else -1.0
        return (
            f"data:image/jpeg;base64,{payload}",
            f"RGB OK | bytes: {len(jpeg)} | age_s: {age_s:.2f}",
        )
    except Exception as exc:
        return prev_src, f"RGB read failed: {exc}"


# ---------------------------------------------------------------------------
# Callbacks — Depth feed
# ---------------------------------------------------------------------------

@app.callback(
    Output("depth-feed", "src"),
    Output("depth-status", "children"),
    Input("depth-interval", "n_intervals"),
    State("depth-feed", "src"),
)
def update_depth_feed(_tick: int, prev_src: str | None) -> tuple[str | None, str]:
    try:
        preview = _get_depth_preview()
        preview.start()
        jpeg, ts, center_depth_m, near_cov, err = preview.snapshot()
        if err is not None:
            return prev_src, err
        if jpeg is None:
            return prev_src, f"Waiting for depth frames on UDP port {DEPTH_PORT}."
        age_s = max(0.0, time.time() - ts) if ts > 0 else -1.0
        payload = base64.b64encode(jpeg).decode("ascii")
        center_text = f"{center_depth_m:.2f}m" if center_depth_m is not None else "n/a"
        near_text = f"{near_cov * 100.0:.1f}%" if near_cov is not None else "n/a"
        return (
            f"data:image/jpeg;base64,{payload}",
            f"Depth OK | bytes: {len(jpeg)} | center: {center_text} | near@1m: {near_text} | age_s: {age_s:.2f}",
        )
    except Exception as exc:
        return prev_src, f"Depth read failed: {exc}"


# ---------------------------------------------------------------------------
# Callbacks — LiDAR + IMU
# ---------------------------------------------------------------------------

@app.callback(
    Output("lidar-graph", "figure"),
    Output("lidar-status", "children"),
    Output("imu-graph", "figure"),
    Output("imu-status", "children"),
    Input("lidar-interval", "n_intervals"),
)
def update_sensors(_tick: int) -> tuple[go.Figure, str, go.Figure, str]:
    node = get_node()

    # --- LiDAR ----------------------------------------------------------
    if node.status != "connected":
        label, _ = node.status_badge()
        no_conn = f"Not connected ({label})."
        return (
            _empty_lidar_figure("LiDAR — not connected"),
            no_conn,
            _empty_imu_figure("IMU — not connected"),
            no_conn,
        )

    pts, lidar_ts = node.get_lidar_points()
    if not pts:
        live = _get_livox_preview()
        live.start()
        xyz, live_ts, live_err = live.snapshot()
        if xyz is not None:
            import numpy as np

            arr = np.asarray(xyz, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                mask = (
                    (arr[:, 2] >= -1.0)
                    & (arr[:, 2] <= 2.0)
                    & (np.abs(arr[:, 0]) <= 10.0)
                    & (np.abs(arr[:, 1]) <= 10.0)
                )
                arr = arr[mask]
                xs = arr[:, 0].tolist()
                ys = arr[:, 1].tolist()
                zs = arr[:, 2].tolist()
            else:
                xs, ys, zs = [], [], []
            lidar_fig = go.Figure(
                data=[
                    go.Scattergl(
                        x=xs,
                        y=ys,
                        mode="markers",
                        marker={"size": 3, "color": zs, "colorscale": "Viridis", "showscale": True},
                        name="LiDAR",
                    )
                ]
            )
            lidar_fig.update_layout(
                template="plotly_dark",
                title="LiDAR stream (Livox SDK fallback)",
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                margin={"l": 30, "r": 20, "t": 45, "b": 35},
                height=500,
            )
            age = max(0.0, time.time() - live_ts) if live_ts > 0 else -1.0
            lidar_status = f"Points: {len(xs)} | source: Livox SDK fallback | age_s: {age:.2f}"
        else:
            lidar_fig = _empty_lidar_figure("LiDAR — no points yet")
            extra = f" | {live_err}" if live_err else ""
            lidar_status = f"No LiDAR points on {LIDAR_TOPIC}{extra}"
    else:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        lidar_fig = go.Figure(
            data=[
                go.Scattergl(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker={"size": 3, "color": zs, "colorscale": "Viridis", "showscale": True},
                    name="LiDAR",
                )
            ]
        )
        lidar_fig.update_layout(
            template="plotly_dark",
            title=f"LiDAR stream ({LIDAR_TOPIC})",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            margin={"l": 30, "r": 20, "t": 45, "b": 35},
            height=500,
        )
        age = max(0.0, time.time() - lidar_ts) if lidar_ts > 0 else -1.0
        lidar_status = f"Points: {len(pts)} | topic: {LIDAR_TOPIC} | age_s: {age:.2f}"

    # --- IMU ------------------------------------------------------------
    rpy = node.get_imu_rpy()
    if rpy is None:
        imu_fig = _empty_imu_figure("IMU — no data yet")
        imu_status = f"No IMU data on {SPORT_STATE_TOPIC} or {LOW_STATE_TOPIC}"
    else:
        roll, pitch, yaw = rpy
        now = time.time()
        IMU_HISTORY.append((now, roll, pitch, yaw))
        t0 = IMU_HISTORY[0][0]
        rel_t = [row[0] - t0 for row in IMU_HISTORY]
        imu_fig = go.Figure(
            data=[
                go.Scatter(x=rel_t, y=[r[1] for r in IMU_HISTORY], mode="lines", name="roll"),
                go.Scatter(x=rel_t, y=[r[2] for r in IMU_HISTORY], mode="lines", name="pitch"),
                go.Scatter(x=rel_t, y=[r[3] for r in IMU_HISTORY], mode="lines", name="yaw"),
            ]
        )
        imu_fig.update_layout(
            template="plotly_dark",
            title="IMU orientation (RPY)",
            xaxis_title="Time (s, recent window)",
            yaxis_title="Angle (rad)",
            margin={"l": 30, "r": 20, "t": 45, "b": 35},
            height=320,
        )
        imu_status = (
            f"Latest RPY (rad): [{roll:.3f}, {pitch:.3f}, {yaw:.3f}] | "
            f"samples: {len(IMU_HISTORY)}"
        )

    return lidar_fig, lidar_status, imu_fig, imu_status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
