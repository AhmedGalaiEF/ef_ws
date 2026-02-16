"""
ef_client.py
============

Pragmatic high-level client for Unitree G1.

Highlights:
- Locomotion/FSM helpers
- Cached IMU + lidar subscriptions
- Local RGBD GST viewer launcher
- Local SLAM launcher (live_slam_save.py)
- Path point queue + navigation execution
- SLAM API debug routine
- Audio/headlight convenience wrappers
"""
from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
DEV_OTHER_DIR = THIS_FILE.parent
SCRIPTS_ROOT = DEV_OTHER_DIR.parent
DEV_OTHER_SAFETY_DIR = DEV_OTHER_DIR / "other" / "safety"

for p in (str(DEV_OTHER_DIR), str(SCRIPTS_ROOT), str(DEV_OTHER_SAFETY_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Unitree SDK imports
# ---------------------------------------------------------------------------

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, HeightMap_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc

try:
    from safety.hanger_boot_sequence import hanger_boot_sequence
except Exception:
    from hanger_boot_sequence import hanger_boot_sequence  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SPORT_TOPIC = "rt/odommodestate"
DEFAULT_LIDAR_MAP_TOPIC = "rt/utlidar/map_state"
DEFAULT_LIDAR_CLOUD_TOPIC = "rt/utlidar/cloud_deskewed"


@dataclass
class ImuData:
    rpy: tuple[float, float, float]
    gyro: tuple[float, float, float] | None
    acc: tuple[float, float, float] | None
    quat: tuple[float, float, float, float] | None
    temp: float | None


class _RgbdDepthGuard:
    """
    Lightweight RGBD depth guard using the GST receive pipeline.

    The incoming depth stream is expected to be PLASMA color-mapped depth
    (as produced by jetson_realsense_stream.py).
    """

    def __init__(
        self,
        depth_port: int,
        width: int,
        height: int,
        fps: int,
        near_distance_m: float = 0.75,
        min_coverage: float = 0.18,
        required_hits: int = 2,
    ) -> None:
        self.depth_port = int(depth_port)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self.near_distance_m = float(near_distance_m)
        self.min_coverage = float(min_coverage)
        self.required_hits = max(1, int(required_hits))

        self._available = False
        self._err: str | None = None
        self._blocked = False
        self._hits = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self._available)

    @property
    def error(self) -> str | None:
        return self._err

    def is_blocked(self) -> bool:
        with self._lock:
            return bool(self._blocked)

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

    def _run(self) -> None:
        try:
            import cv2
            import gi
            import numpy as np

            gi.require_version("Gst", "1.0")
            gi.require_version("GstApp", "1.0")
            from gi.repository import Gst
        except Exception as exc:
            self._err = f"RGBD depth guard unavailable: {exc}"
            self._available = False
            return

        try:
            Gst.init(None)
            pipeline = Gst.parse_launch(
                f"udpsrc port={self.depth_port} caps=application/x-rtp,media=video,encoding-name=H264,payload=97 ! "
                "rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=true sync=false drop=true"
            )
            sink = pipeline.get_by_name("sink")
            if sink is None:
                raise RuntimeError("appsink not found")
            pipeline.set_state(Gst.State.PLAYING)
            self._available = True

            # Build PLASMA lookup in BGR, index 0..255 -> distance 0..6m.
            cmap = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_PLASMA)
            cmap = cmap.reshape(256, 3).astype(np.int16)
            near_idx = int(max(0.0, min(255.0, (self.near_distance_m / 6.0) * 255.0)))

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

                # Forward ROI: center horizontally, upper-middle vertically to reduce floor hits.
                x0 = int(self.width * 0.30)
                x1 = int(self.width * 0.70)
                y0 = int(self.height * 0.25)
                y1 = int(self.height * 0.70)
                roi = depth_bgr[y0:y1, x0:x1]
                if roi.size == 0:
                    continue
                pix = roi.reshape(-1, 3).astype(np.int16)

                # Approximate inverse-colormap by nearest BGR entry.
                diff = pix[:, None, :] - cmap[None, :, :]
                dist2 = (diff * diff).sum(axis=2)
                idx = np.argmin(dist2, axis=1)
                near_cov = float(np.mean(idx <= near_idx))
                blocked_now = near_cov >= self.min_coverage

                with self._lock:
                    if blocked_now:
                        self._hits += 1
                    else:
                        self._hits = 0
                    self._blocked = self._hits >= self.required_hits
        except Exception as exc:
            self._err = str(exc)
            self._available = False
        finally:
            try:
                pipeline.set_state(Gst.State.NULL)  # type: ignore[name-defined]
            except Exception:
                pass


class Robot:
    """End-user wrapper around common G1 workflows."""

    def __init__(
        self,
        iface: str = "eth0",
        domain_id: int = 0,
        safety_boot: bool = True,
        auto_start_sensors: bool = True,
        sport_topic: str = DEFAULT_SPORT_TOPIC,
        lidar_map_topic: str = DEFAULT_LIDAR_MAP_TOPIC,
        lidar_cloud_topic: str = DEFAULT_LIDAR_CLOUD_TOPIC,
        slam_info_topic: str = "rt/slam_info",
        slam_key_topic: str = "rt/slam_key_info",
        rgb_port: int = 5600,
        depth_port: int = 5602,
        rgb_width: int = 640,
        rgb_height: int = 480,
        rgb_fps: int = 30,
        nav_map: str | None = None,
        nav_extra_args: str = "--smooth --use-live-map --no-viz",
        nav_use_external_astar: bool = False,
        rgbd_obs_near_m: float = 0.75,
        rgbd_obs_min_coverage: float = 0.18,
    ) -> None:
        self.iface = iface
        self.domain_id = int(domain_id)
        self.sport_topic = sport_topic
        self.lidar_map_topic = lidar_map_topic
        self.lidar_cloud_topic = lidar_cloud_topic
        self.slam_info_topic = slam_info_topic
        self.slam_key_topic = slam_key_topic

        self.rgb_port = int(rgb_port)
        self.depth_port = int(depth_port)
        self.rgb_width = int(rgb_width)
        self.rgb_height = int(rgb_height)
        self.rgb_fps = int(rgb_fps)

        self.nav_map = nav_map
        self.nav_extra_args = nav_extra_args
        self.nav_use_external_astar = bool(nav_use_external_astar)
        self.rgbd_obs_near_m = float(rgbd_obs_near_m)
        self.rgbd_obs_min_coverage = float(rgbd_obs_min_coverage)

        self._lock = threading.Lock()
        self._sport: SportModeState_ | None = None
        self._lidar_map: HeightMap_ | None = None
        self._lidar_cloud: PointCloud2_ | None = None
        self._last_sport_ts = 0.0
        self._last_lidar_map_ts = 0.0
        self._last_lidar_cloud_ts = 0.0

        self._sport_sub: ChannelSubscriber | None = None
        self._lidar_map_sub: ChannelSubscriber | None = None
        self._lidar_cloud_sub: ChannelSubscriber | None = None

        self._rgbd_proc: subprocess.Popen | None = None
        self._slam_proc: subprocess.Popen | None = None
        self._slam_log_fp: Any | None = None
        self.slam_is_running = False
        self._slam_save_dir: str | None = None

        self._path_points: list[tuple[float, float, float]] = []
        self._slam_service_windows: list[Any] = []

        if safety_boot:
            self._client = hanger_boot_sequence(iface=self.iface)
        else:
            ChannelFactoryInitialize(self.domain_id, self.iface)
            self._client = LocoClient()
            self._client.SetTimeout(10.0)
            self._client.Init()
        self._ensure_balanced_gait_mode()

        if auto_start_sensors:
            self.start_sensors()

    def _ensure_balanced_gait_mode(self) -> None:
        try:
            if hasattr(self._client, "BalanceStand"):
                self._client.BalanceStand(0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Sensor subscriptions
    # ------------------------------------------------------------------

    def start_sensors(self) -> None:
        if self._sport_sub is None:
            self._sport_sub = ChannelSubscriber(self.sport_topic, SportModeState_)
            self._sport_sub.Init(self._sport_cb, 10)

        if self._lidar_map_sub is None:
            self._lidar_map_sub = ChannelSubscriber(self.lidar_map_topic, HeightMap_)
            self._lidar_map_sub.Init(self._lidar_map_cb, 10)

        if self._lidar_cloud_sub is None:
            self._lidar_cloud_sub = ChannelSubscriber(self.lidar_cloud_topic, PointCloud2_)
            self._lidar_cloud_sub.Init(self._lidar_cloud_cb, 10)

    def _sport_cb(self, msg: SportModeState_) -> None:
        with self._lock:
            self._sport = msg
            self._last_sport_ts = time.time()

    def _lidar_map_cb(self, msg: HeightMap_) -> None:
        with self._lock:
            self._lidar_map = msg
            self._last_lidar_map_ts = time.time()

    def _lidar_cloud_cb(self, msg: PointCloud2_) -> None:
        with self._lock:
            self._lidar_cloud = msg
            self._last_lidar_cloud_ts = time.time()

    # ------------------------------------------------------------------
    # Generic state helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_attr(obj: Any, *path: str) -> Any:
        cur = obj
        for name in path:
            if cur is None or not hasattr(cur, name):
                return None
            cur = getattr(cur, name)
        return cur

    @staticmethod
    def _vector3_from(value: Any) -> tuple[float, float, float] | None:
        try:
            if value is None:
                return None
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return None
        return None

    def get_sport_state(self) -> SportModeState_ | None:
        with self._lock:
            return self._sport

    def get_lidar_map(self) -> HeightMap_ | None:
        with self._lock:
            return self._lidar_map

    def get_lidar_cloud(self) -> PointCloud2_ | None:
        with self._lock:
            return self._lidar_cloud

    def get_sensor_timestamps(self) -> dict[str, float]:
        with self._lock:
            return {
                "sport": float(self._last_sport_ts),
                "lidar_map": float(self._last_lidar_map_ts),
                "lidar_cloud": float(self._last_lidar_cloud_ts),
            }

    def sensors_stale(self, max_age: float = 1.0) -> dict[str, bool]:
        now = time.time()
        ts = self.get_sensor_timestamps()
        out: dict[str, bool] = {}
        for k, v in ts.items():
            out[k] = (v <= 0.0) or ((now - v) > max_age)
        return out

    def wait_for_sport_state(self, timeout: float = 2.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < max(0.0, timeout):
            if self.get_sport_state() is not None:
                return True
            time.sleep(0.05)
        return self.get_sport_state() is not None

    def get_mode(self) -> int | None:
        msg = self.get_sport_state()
        if msg is None:
            return None
        value = self._read_attr(msg, "mode")
        try:
            return int(value)
        except Exception:
            return None

    def get_gait(self) -> int | None:
        msg = self.get_sport_state()
        if msg is None:
            return None
        # field name differs across SDK revisions
        for key in ("gait_type", "gaitType", "gait"):
            value = self._read_attr(msg, key)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        return None

    def get_body_height(self) -> float | None:
        msg = self.get_sport_state()
        if msg is None:
            return None
        for key in ("body_height", "bodyHeight", "stand_height", "standHeight"):
            value = self._read_attr(msg, key)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue
        return None

    def get_position(self) -> tuple[float, float, float] | None:
        msg = self.get_sport_state()
        if msg is None:
            return None

        for key in ("position", "pos", "position_w"):
            vec = self._vector3_from(self._read_attr(msg, key))
            if vec is not None:
                return vec
        return None

    def get_velocity(self) -> tuple[float, float, float] | None:
        msg = self.get_sport_state()
        if msg is None:
            return None

        for key in ("velocity", "vel", "velocity_w"):
            vec = self._vector3_from(self._read_attr(msg, key))
            if vec is not None:
                return vec
        return None

    def get_yaw(self) -> float | None:
        imu = self.get_imu()
        if imu is None:
            return None
        return float(imu.rpy[2])

    def is_moving(self, linear_eps: float = 0.03, yaw_eps: float = 0.08) -> bool:
        v = self.get_velocity()
        if v is None:
            return False
        vx, vy, vz = v
        planar = math.hypot(vx, vy)
        return planar > linear_eps or abs(vz) > yaw_eps

    def get_robot_state(self) -> dict[str, Any]:
        """
        Consolidated state snapshot from cached DDS data + locomotion RPC.
        """
        return {
            "fsm": self.get_fsm(),
            "mode": self.get_mode(),
            "gait": self.get_gait(),
            "body_height": self.get_body_height(),
            "position": self.get_position(),
            "velocity": self.get_velocity(),
            "yaw": self.get_yaw(),
            "is_moving": self.is_moving(),
            "imu": self.get_imu(),
            "sensor_timestamps": self.get_sensor_timestamps(),
            "sensor_stale": self.sensors_stale(),
            "slam_is_running": bool(self.slam_is_running),
            "queued_path_points": len(self._path_points),
        }

    # ------------------------------------------------------------------
    # Locomotion + FSM
    # ------------------------------------------------------------------

    def loco_move(self, vx: float, vy: float, vyaw: float) -> int:
        return self._client.Move(float(vx), float(vy), float(vyaw), continous_move=True)

    def stop_moving(self) -> None:
        if hasattr(self._client, "StopMove"):
            self._client.StopMove()
        else:
            self._client.Move(0.0, 0.0, 0.0, continous_move=False)

    @staticmethod
    def _normalize_gait_type(gait_type: int | str) -> int:
        if isinstance(gait_type, str):
            key = gait_type.strip().lower().replace("-", "_").replace(" ", "_")
            alias = {
                "normal": 0,
                "balanced": 0,
                "balance": 0,
                "static": 0,
                "stand": 0,
                "continuous": 1,
                "walk": 1,
                "walking": 1,
                "dynamic": 1,
            }
            if key not in alias:
                raise ValueError(f"Unknown gait_type '{gait_type}'.")
            return int(alias[key])
        return int(gait_type)

    def set_gait_type(self, gait_type: int | str = 0) -> int:
        """
        Set locomotion gait/balance mode.

        Accepts integer mode or aliases:
          - 0: normal/balanced/static stand
          - 1: continuous walking mode
        """
        mode = self._normalize_gait_type(gait_type)
        if hasattr(self._client, "SetGaitType"):
            return int(self._client.SetGaitType(mode))
        if hasattr(self._client, "SetBalanceMode"):
            return int(self._client.SetBalanceMode(mode))
        raise AttributeError("Current locomotion client does not support gait mode setting API.")

    def _rpc_get_int(self, api_id: int) -> Optional[int]:
        try:
            code, data = self._client._Call(api_id, "{}")  # type: ignore[attr-defined]
            if code != 0 or not data:
                return None
            return int(json.loads(data).get("data"))
        except Exception:
            return None

    def get_fsm(self) -> dict[str, Optional[int]]:
        try:
            from unitree_sdk2py.g1.loco.g1_loco_api import (
                ROBOT_API_ID_LOCO_GET_FSM_ID,
                ROBOT_API_ID_LOCO_GET_FSM_MODE,
            )
        except Exception:
            return {"id": None, "mode": None}

        return {
            "id": self._rpc_get_int(ROBOT_API_ID_LOCO_GET_FSM_ID),
            "mode": self._rpc_get_int(ROBOT_API_ID_LOCO_GET_FSM_MODE),
        }

    def fsm_0_zt(self) -> None:
        if hasattr(self._client, "ZeroTorque"):
            self._client.ZeroTorque()
        elif hasattr(self._client, "SetFsmId"):
            self._client.SetFsmId(0)

    def fsm_1_damp(self) -> None:
        if hasattr(self._client, "Damp"):
            self._client.Damp()
        elif hasattr(self._client, "SetFsmId"):
            self._client.SetFsmId(1)

    def fsm_2_squat(self) -> None:
        if hasattr(self._client, "SetFsmId"):
            self._client.SetFsmId(2)

    # ------------------------------------------------------------------
    # IMU + lidar getters
    # ------------------------------------------------------------------

    def get_imu(self) -> ImuData | None:
        with self._lock:
            msg = self._sport
        if msg is None:
            return None

        rpy = (0.0, 0.0, 0.0)
        gyro = acc = quat = None
        temp = None

        try:
            rpy = (
                float(msg.imu_state.rpy[0]),
                float(msg.imu_state.rpy[1]),
                float(msg.imu_state.rpy[2]),
            )
        except Exception:
            pass
        try:
            gyro = (
                float(msg.imu_state.gyroscope[0]),
                float(msg.imu_state.gyroscope[1]),
                float(msg.imu_state.gyroscope[2]),
            )
        except Exception:
            pass
        try:
            acc = (
                float(msg.imu_state.accelerometer[0]),
                float(msg.imu_state.accelerometer[1]),
                float(msg.imu_state.accelerometer[2]),
            )
        except Exception:
            pass
        try:
            quat = (
                float(msg.imu_state.quaternion[0]),
                float(msg.imu_state.quaternion[1]),
                float(msg.imu_state.quaternion[2]),
                float(msg.imu_state.quaternion[3]),
            )
        except Exception:
            pass
        try:
            temp = float(msg.imu_state.temperature)
        except Exception:
            pass

        return ImuData(rpy=rpy, gyro=gyro, acc=acc, quat=quat, temp=temp)

    @staticmethod
    def _extract_xyz_from_cloud(msg: PointCloud2_, max_points: int | None = None) -> list[tuple[float, float, float]]:
        try:
            width = int(msg.width)
            height = int(msg.height)
            point_step = int(msg.point_step)
            raw = bytes(msg.data)
        except Exception:
            return []

        if point_step <= 0:
            return []

        x_off, y_off, z_off = 0, 4, 8
        try:
            fields = list(msg.fields)
            name_to_off = {str(f.name).lower(): int(f.offset) for f in fields}
            x_off = name_to_off.get("x", x_off)
            y_off = name_to_off.get("y", y_off)
            z_off = name_to_off.get("z", z_off)
        except Exception:
            pass

        total = max(0, width * height)
        if max_points is not None:
            total = min(total, max_points)

        import struct

        out: list[tuple[float, float, float]] = []
        for i in range(total):
            base = i * point_step
            try:
                x = struct.unpack_from("<f", raw, base + x_off)[0]
                y = struct.unpack_from("<f", raw, base + y_off)[0]
                z = struct.unpack_from("<f", raw, base + z_off)[0]
            except Exception:
                break
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                out.append((float(x), float(y), float(z)))
        return out

    def get_lidar_points(self, max_points: int | None = 20000) -> list[tuple[float, float, float]]:
        with self._lock:
            msg = self._lidar_cloud
        if msg is None:
            return []
        return self._extract_xyz_from_cloud(msg, max_points=max_points)

    # ------------------------------------------------------------------
    # RGBD GST helpers
    # ------------------------------------------------------------------

    def get_rgbd_gst(self, detect: str = "human") -> subprocess.Popen:
        """
        Start a local RGBD GST viewer.

        detect="" (or "none") => plain RGBD viewer.
        detect="human" (or any text) => CLIP detector viewer with custom prompt.
        """
        if self._rgbd_proc is not None and self._rgbd_proc.poll() is None:
            return self._rgbd_proc

        plain_script = SCRIPTS_ROOT / "sensors" / "manual_streaming" / "receive_realsense_gst.py"
        clip_script = SCRIPTS_ROOT / "sensors" / "manual_streaming" / "receive_realsense_gst_clip_can.py"

        det = (detect or "").strip().lower()
        if det in ("", "none", "off"):
            cmd = [sys.executable, str(plain_script)]
        else:
            prompt = f"a photo of a {det}"
            negative = f"a photo without a {det}"
            cmd = [
                sys.executable,
                str(clip_script),
                "--positive",
                prompt,
                "--negative",
                negative,
                "--threshold",
                "0.55",
            ]
            cmd.extend(
                [
                    "--rgb-port",
                    str(self.rgb_port),
                    "--depth-port",
                    str(self.depth_port),
                    "--width",
                    str(self.rgb_width),
                    "--height",
                    str(self.rgb_height),
                    "--fps",
                    str(self.rgb_fps),
                ]
            )

        self._rgbd_proc = subprocess.Popen(cmd, cwd=str(SCRIPTS_ROOT))
        return self._rgbd_proc

    # ------------------------------------------------------------------
    # Local SLAM helpers (live_slam_save.py)
    # ------------------------------------------------------------------

    def start_slam(
        self,
        save_folder: str = "./maps",
        save_every: int = 1,
        save_latest: bool = True,
        save_prefix: str = "live_slam_latest",
        viz: bool = False,
    ) -> subprocess.Popen:
        if self._slam_proc is not None and self._slam_proc.poll() is None:
            self.slam_is_running = True
            return self._slam_proc

        if viz:
            slam_script = DEV_OTHER_DIR / "other" / "slam_viz_nav_overlay_runner.py"
            slam_cwd = SCRIPTS_ROOT / "navigation" / "obstacle_avoidance"
        else:
            slam_script = DEV_OTHER_DIR / "other" / "slam_headless_save_runner.py"
            slam_cwd = slam_script.parent
        save_dir = Path(save_folder)
        if not save_dir.is_absolute():
            save_dir = (slam_cwd / save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(slam_script),
            "--save-dir",
            str(save_dir),
            "--save-every",
            str(max(1, int(save_every))),
            "--save-prefix",
            save_prefix,
        ]
        if save_latest:
            cmd.append("--save-latest")
        if viz:
            cmd.extend(["--iface", self.iface, "--domain-id", str(self.domain_id)])

        log_path = save_dir / "slam_runtime.log"
        self._slam_log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
        self._slam_log_fp.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] start viz={bool(viz)} cmd={cmd}\n")
        self._slam_proc = subprocess.Popen(
            cmd,
            cwd=str(slam_cwd),
            stdin=subprocess.DEVNULL,
            stdout=self._slam_log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        startup_deadline = time.time() + (5.0 if viz else 0.35)
        while time.time() < startup_deadline and self._slam_proc.poll() is None:
            time.sleep(0.1)

        if self._slam_proc.poll() is not None:
            tail = ""
            try:
                with open(log_path, "r", encoding="utf-8") as fp:
                    lines = fp.readlines()
                tail = "".join(lines[-40:]).strip()
            except Exception:
                pass
            tail_lower = tail.lower()
            viz_init_failed = (
                "failed to initialize glew" in tail_lower
                or "glfw error" in tail_lower
                or "failed to initialize gtk" in tail_lower
            )
            self.slam_is_running = False
            self._slam_proc = None
            if self._slam_log_fp is not None:
                try:
                    self._slam_log_fp.close()
                except Exception:
                    pass
                self._slam_log_fp = None
            if viz and viz_init_failed:
                # Typical on headless/Wayland sessions where Open3D GUI cannot be created.
                print("[start_slam] Visualization init failed; falling back to headless SLAM runner.")
                return self.start_slam(
                    save_folder=str(save_dir),
                    save_every=save_every,
                    save_latest=save_latest,
                    save_prefix=save_prefix,
                    viz=False,
                )
            detail = f"\nSLAM log tail:\n{tail}" if tail else ""
            raise RuntimeError(f"Failed to start SLAM process (viz={bool(viz)}).{detail}")
        self._slam_save_dir = str(save_dir)
        if save_latest:
            self.nav_map = str(save_dir / f"{save_prefix}_latest.pcd")
        self.slam_is_running = True
        return self._slam_proc

    def stop_slam(self, save_folder: str = "./maps") -> None:
        """
        Stop local SLAM subprocess started by start_slam().
        save_folder is kept for API compatibility with local live_slam_save usage.
        """
        _ = save_folder  # intentionally accepted even when process already has its own save-dir

        proc = self._slam_proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    proc.terminate()
                    proc.wait(timeout=3.0)
                except Exception:
                    proc.kill()
        self._slam_proc = None
        if self._slam_log_fp is not None:
            try:
                self._slam_log_fp.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] stop\n")
                self._slam_log_fp.close()
            except Exception:
                pass
            self._slam_log_fp = None
        self.slam_is_running = False

    # ------------------------------------------------------------------
    # Path points + navigation
    # ------------------------------------------------------------------

    def set_path_point(self, x: float, y: float, yaw: float = 0.0) -> None:
        if not self.slam_is_running:
            raise RuntimeError("set_path_point is only allowed while slam_is_running=True")
        self._path_points.append((float(x), float(y), float(yaw)))

    def get_path_points(self) -> list[tuple[float, float, float]]:
        return list(self._path_points)

    def clear_path_points(self) -> None:
        self._path_points.clear()

    def _snapshot_live_slam_map_npz(self, timeout_s: float = 2.0) -> str | None:
        """
        Snapshot current rt/utlidar map_state into a temporary .npz map file.
        This keeps planner map frame aligned with robot pose frame.
        """
        try:
            from navigation.obstacle_avoidance.slam_map import SlamMapSubscriber
        except Exception:
            return None

        sub = SlamMapSubscriber(self.lidar_map_topic)
        sub.start()
        t0 = time.time()
        while time.time() - t0 < max(0.2, float(timeout_s)):
            occ, _meta = sub.to_occupancy(height_threshold=0.15, max_height=None, origin_centered=True)
            if occ is not None:
                try:
                    with tempfile.NamedTemporaryFile(prefix="g1_live_nav_", suffix=".npz", delete=False) as tf:
                        out = tf.name
                    occ.save(out)
                    return out
                except Exception:
                    return None
            time.sleep(0.05)
        return None

    def _run_dynamic_nav(self, x: float, y: float, use_rgbd_depth_guard: bool = False) -> int:
        nav_script = SCRIPTS_ROOT / "navigation" / "obstacle_avoidance" / "real_time_path_steps_dynamic.py"

        map_for_run: str | None = self._snapshot_live_slam_map_npz(timeout_s=1.5)
        if map_for_run is None:
            map_for_run = self.nav_map

        if not map_for_run:
            candidates: list[Path] = []
            if self._slam_save_dir:
                save_dir = Path(self._slam_save_dir)
                candidates.extend(
                    sorted(save_dir.glob("*latest*.pcd"), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
                )
                candidates.extend(
                    sorted(save_dir.glob("*.pcd"), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
                )
            # common legacy fallback map path
            candidates.append(SCRIPTS_ROOT / "navigation" / "obstacle_avoidance" / "maps" / "live_slam_latest.pcd")
            for c in candidates:
                if c.exists():
                    map_for_run = str(c)
                    break
        if not map_for_run:
            print("[_run_dynamic_nav] no map path resolved for dynamic navigation.")
            return 97

        nav_map_path = Path(map_for_run)
        if not nav_map_path.exists():
            # map file can appear shortly after SLAM starts; wait briefly.
            t0 = time.time()
            while time.time() - t0 < 3.0 and not nav_map_path.exists():
                time.sleep(0.1)
            if not nav_map_path.exists():
                print(f"[_run_dynamic_nav] map file does not exist: {nav_map_path}")
                return 97

        cmd = [
            sys.executable,
            str(nav_script),
            "--map",
            str(map_for_run),
            "--iface",
            self.iface,
            "--domain-id",
            str(self.domain_id),
            "--goal-x",
            str(float(x)),
            "--goal-y",
            str(float(y)),
        ]
        start = self.get_position()
        if start is not None:
            cmd.extend(["--start-x", str(float(start[0])), "--start-y", str(float(start[1]))])
        if self.nav_extra_args:
            cmd.extend(shlex.split(self.nav_extra_args))
        if "--no-viz" not in cmd:
            cmd.append("--no-viz")

        guard: _RgbdDepthGuard | None = None
        if use_rgbd_depth_guard:
            guard = _RgbdDepthGuard(
                depth_port=self.depth_port,
                width=self.rgb_width,
                height=self.rgb_height,
                fps=self.rgb_fps,
                near_distance_m=self.rgbd_obs_near_m,
                min_coverage=self.rgbd_obs_min_coverage,
            )
            guard.start()
            # Give guard a brief moment to initialise pipeline.
            time.sleep(0.15)
            if not guard.available and guard.error:
                print(f"[_run_dynamic_nav] RGBD depth guard disabled: {guard.error}")
                guard = None

        print(f"[_run_dynamic_nav] start map={map_for_run} goal=({float(x):+.3f},{float(y):+.3f})")
        proc = subprocess.Popen(cmd, cwd=str(nav_script.parent), start_new_session=True)
        blocked_abort = False
        try:
            while proc.poll() is None:
                if guard is not None and guard.is_blocked():
                    blocked_abort = True
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    except Exception:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                    break
                time.sleep(0.05)
            rc = int(proc.wait())
            if blocked_abort:
                print("[_run_dynamic_nav] aborted due to RGBD depth obstacle in front ROI.")
                return 98
            return rc
        finally:
            if guard is not None:
                guard.stop()

    def _wait_pose_nav_arrival(self, timeout_s: float = 90.0) -> bool:
        from navigation.obstacle_avoidance.slam_map import SlamInfoSubscriber

        sub = SlamInfoSubscriber(self.slam_info_topic, self.slam_key_topic)
        sub.start()
        t0 = time.time()
        while time.time() - t0 < max(0.1, float(timeout_s)):
            for payload in (sub.get_key(), sub.get_info()):
                if not payload:
                    continue
                try:
                    msg = json.loads(payload)
                except Exception:
                    continue
                data = msg.get("data") if isinstance(msg, dict) else None
                if not isinstance(data, dict):
                    continue
                if data.get("is_arrived") is True:
                    return True
                if data.get("is_abort") is True or data.get("is_failed") is True:
                    return False
            time.sleep(0.05)
        return False

    def _run_pose_nav(self, x: float, y: float, yaw: float = 0.0, wait_timeout_s: float = 90.0) -> int:
        from navigation.obstacle_avoidance.slam_service import SlamOperateClient

        client = SlamOperateClient()
        client.Init()
        client.SetTimeout(10.0)

        qz = math.sin(float(yaw) * 0.5)
        qw = math.cos(float(yaw) * 0.5)
        resp = client.pose_nav(float(x), float(y), 0.0, 0.0, 0.0, qz, qw, mode=1)
        if int(resp.code) != 0:
            return int(resp.code)
        return 0 if self._wait_pose_nav_arrival(timeout_s=wait_timeout_s) else -1

    def navigate_path(self, obs_avoid: bool = True, clear_on_finish: bool = True) -> bool:
        if not self._path_points:
            raise RuntimeError("No path points queued. Call set_path_point(...) first.")
        try:
            self.set_gait_type(0)
        except Exception as exc:
            print(f"[navigate_path] warning: failed to set gait_type=0 ({exc})")

        ok = True
        try:
            for idx, (x, y, yaw) in enumerate(self._path_points, start=1):
                pos = self.get_position()
                if pos is not None:
                    dxy = math.hypot(float(x) - float(pos[0]), float(y) - float(pos[1]))
                    # pose_nav often rejects already-reached goals with rc=4.
                    if dxy <= 0.20:
                        continue
                if obs_avoid and self.nav_use_external_astar:
                    rc = self._run_dynamic_nav(x, y, use_rgbd_depth_guard=True)
                else:
                    # Default: robot-side planning/execution through SLAM pose_nav.
                    rc = self._run_pose_nav(x, y, yaw)
                    if rc == 4 and pos is not None:
                        dxy = math.hypot(float(x) - float(pos[0]), float(y) - float(pos[1]))
                        if dxy <= 0.30:
                            rc = 0
                    if rc != 0:
                        print(f"[navigate_path] pose_nav rejected (rc={rc}); trying local dynamic nav fallback.")
                        rc = self._run_dynamic_nav(x, y, use_rgbd_depth_guard=True)
                if rc != 0:
                    print(f"[navigate_path] failed at point {idx}: ({x:.3f},{y:.3f},{yaw:.3f}) rc={rc}")
                    ok = False
                    break
        finally:
            if clear_on_finish:
                self._path_points.clear()
        return ok

    # ------------------------------------------------------------------
    # Unified SLAM service GUI
    # ------------------------------------------------------------------

    def _keyboard_controller_path(self) -> Path:
        candidates = [
            SCRIPTS_ROOT / "basic" / "safety" / "keyboard_controller.py",
            DEV_OTHER_SAFETY_DIR / "keyboard_controller.py",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError("keyboard_controller.py not found in basic/safety or dev/other/safety")

    def _launch_teleop(self, input_mode: str = "curses") -> subprocess.Popen:
        script = self._keyboard_controller_path()
        base_cmd = [sys.executable, str(script), "--iface", self.iface, "--input", input_mode]

        if input_mode == "curses":
            term_launchers = [
                ["x-terminal-emulator", "-e"] if shutil.which("x-terminal-emulator") else None,
                ["gnome-terminal", "--"] if shutil.which("gnome-terminal") else None,
                ["konsole", "-e"] if shutil.which("konsole") else None,
                ["xterm", "-e"] if shutil.which("xterm") else None,
            ]
            for prefix in term_launchers:
                if not prefix:
                    continue
                try:
                    return subprocess.Popen(prefix + base_cmd, start_new_session=True)
                except Exception:
                    continue

        return subprocess.Popen(base_cmd, start_new_session=True)

    def slam_service(
        self,
        save_folder: str = "./maps",
        save_every: int = 1,
        save_latest: bool = True,
        save_prefix: str = "live_slam_latest",
        viz: bool = False,
    ) -> int:
        try:
            from PyQt5 import QtCore, QtWidgets
        except Exception:
            try:
                from PySide6 import QtCore, QtWidgets  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "PyQt is required for slam_service(). Install PyQt5 or PySide6."
                ) from exc

        robot = self

        class _SlamServiceWindow(QtWidgets.QWidget):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.setWindowTitle("SLAM Service")
                self.resize(620, 520)

                self._teleop_proc: subprocess.Popen | None = None
                self._follow_thread: threading.Thread | None = None
                self._follow_result: tuple[bool, str] | None = None

                self._status = QtWidgets.QLabel("idle")
                self._path_list = QtWidgets.QListWidget()
                self._obs_avoid = QtWidgets.QCheckBox("Obstacle Avoid Navigation")
                self._obs_avoid.setChecked(True)
                self._clear_on_finish = QtWidgets.QCheckBox("Clear Path On Finish")
                self._clear_on_finish.setChecked(True)

                self._start_btn = QtWidgets.QPushButton("Start SLAM")
                self._stop_btn = QtWidgets.QPushButton("Stop SLAM")
                self._teleop_start_btn = QtWidgets.QPushButton("Start Teleop (curses)")
                self._teleop_stop_btn = QtWidgets.QPushButton("Stop Teleop")
                self._add_pose_btn = QtWidgets.QPushButton("Add Current Pose")
                self._clear_btn = QtWidgets.QPushButton("Clear Path")
                self._follow_btn = QtWidgets.QPushButton("Follow Path")

                btn_row1 = QtWidgets.QHBoxLayout()
                btn_row1.addWidget(self._start_btn)
                btn_row1.addWidget(self._stop_btn)
                btn_row2 = QtWidgets.QHBoxLayout()
                btn_row2.addWidget(self._teleop_start_btn)
                btn_row2.addWidget(self._teleop_stop_btn)
                btn_row3 = QtWidgets.QHBoxLayout()
                btn_row3.addWidget(self._add_pose_btn)
                btn_row3.addWidget(self._clear_btn)
                btn_row3.addWidget(self._follow_btn)

                layout = QtWidgets.QVBoxLayout()
                layout.addWidget(self._status)
                layout.addLayout(btn_row1)
                layout.addLayout(btn_row2)
                layout.addWidget(self._obs_avoid)
                layout.addWidget(self._clear_on_finish)
                layout.addLayout(btn_row3)
                layout.addWidget(QtWidgets.QLabel("Queued Path Points (x, y, yaw):"))
                layout.addWidget(self._path_list)
                self.setLayout(layout)

                self._start_btn.clicked.connect(self._start_slam)
                self._stop_btn.clicked.connect(self._stop_slam)
                self._teleop_start_btn.clicked.connect(self._start_teleop)
                self._teleop_stop_btn.clicked.connect(self._stop_teleop)
                self._add_pose_btn.clicked.connect(self._add_pose)
                self._clear_btn.clicked.connect(self._clear_path)
                self._follow_btn.clicked.connect(self._follow_path)

                self._timer = QtCore.QTimer(self)
                self._timer.setInterval(400)
                self._timer.timeout.connect(self._refresh)
                self._timer.start()
                self._refresh()

            def _refresh(self) -> None:
                follow_alive = self._follow_thread is not None and self._follow_thread.is_alive()
                self._follow_btn.setEnabled(not follow_alive)

                pos = robot.get_position()
                yaw = robot.get_yaw()
                pose_txt = "pose unavailable"
                if pos is not None:
                    pose_txt = f"x={pos[0]:.3f} y={pos[1]:.3f} yaw={(yaw if yaw is not None else 0.0):.3f}"
                self._status.setText(
                    f"slam_running={robot.slam_is_running} | queued={len(robot.get_path_points())} | {pose_txt}"
                )

                self._path_list.clear()
                for x, y, yyaw in robot.get_path_points():
                    self._path_list.addItem(f"{x:.3f}, {y:.3f}, {yyaw:.3f}")

                if not follow_alive and self._follow_result is not None:
                    ok, err = self._follow_result
                    self._follow_result = None
                    if err:
                        QtWidgets.QMessageBox.critical(self, "Follow Path Failed", err)
                    elif not ok:
                        QtWidgets.QMessageBox.warning(self, "Follow Path", "Path execution did not complete.")

            def _start_slam(self) -> None:
                try:
                    robot.start_slam(
                        save_folder=save_folder,
                        save_every=save_every,
                        save_latest=save_latest,
                        save_prefix=save_prefix,
                        viz=viz,
                    )
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(self, "Start SLAM Failed", str(exc))
                self._refresh()

            def _stop_slam(self) -> None:
                try:
                    robot.stop_slam(save_folder=save_folder)
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(self, "Stop SLAM Failed", str(exc))
                self._refresh()

            def _start_teleop(self) -> None:
                if self._teleop_proc is not None and self._teleop_proc.poll() is None:
                    return
                try:
                    self._teleop_proc = robot._launch_teleop(input_mode="curses")
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(self, "Teleop Failed", str(exc))

            def _stop_teleop(self) -> None:
                if self._teleop_proc is None or self._teleop_proc.poll() is not None:
                    return
                try:
                    os.killpg(os.getpgid(self._teleop_proc.pid), signal.SIGINT)
                except Exception:
                    try:
                        self._teleop_proc.terminate()
                    except Exception:
                        pass

            def _add_pose(self) -> None:
                pos = robot.get_position()
                yaw = robot.get_yaw()
                if pos is None:
                    QtWidgets.QMessageBox.warning(self, "Pose Unavailable", "No current pose from sensors.")
                    return
                try:
                    robot.set_path_point(float(pos[0]), float(pos[1]), float(yaw if yaw is not None else 0.0))
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "Add Pose Failed", str(exc))
                self._refresh()

            def _clear_path(self) -> None:
                robot.clear_path_points()
                self._refresh()

            def _follow_path(self) -> None:
                if self._follow_thread is not None and self._follow_thread.is_alive():
                    return

                def _run() -> None:
                    ok = False
                    err = ""
                    try:
                        ok = robot.navigate_path(
                            obs_avoid=bool(self._obs_avoid.isChecked()),
                            clear_on_finish=bool(self._clear_on_finish.isChecked()),
                        )
                    except Exception as exc:
                        err = str(exc)
                    self._follow_result = (ok, err)

                self._follow_thread = threading.Thread(target=_run, daemon=True)
                self._follow_thread.start()

            def closeEvent(self, event: Any) -> None:  # noqa: N802
                try:
                    self._timer.stop()
                except Exception:
                    pass
                super().closeEvent(event)

        app = QtWidgets.QApplication.instance()
        own_app = app is None
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        window = _SlamServiceWindow()
        self._slam_service_windows.append(window)
        window.destroyed.connect(lambda *_: self._slam_service_windows.remove(window) if window in self._slam_service_windows else None)
        window.show()

        if own_app:
            return int(app.exec())
        return 0

    # ------------------------------------------------------------------
    # SLAM debug API (mirrors api_util.py sequence)
    # ------------------------------------------------------------------

    def debug_api(
        self,
        save_path: str = "/home/unitree/test1.pcd",
        load_path: str = "/home/unitree/test1.pcd",
        goal_x: float = 1.0,
        goal_y: float = 0.0,
        goal_yaw: float = 0.0,
        pause: bool = False,
        resume: bool = False,
        wait_task_result: bool = False,
    ) -> None:
        from navigation.obstacle_avoidance.slam_map import SlamInfoSubscriber
        from navigation.obstacle_avoidance.slam_service import SlamOperateClient, SlamResponse

        def _print_resp(label: str, req: dict[str, Any], resp: SlamResponse) -> None:
            print(f"\\n[{label}]")
            print("request:", json.dumps(req, indent=2))
            print(f"response: code={resp.code} raw={resp.raw}")

        def _wait_task(sub: SlamInfoSubscriber, timeout: float = 10.0) -> None:
            t0 = time.time()
            while time.time() - t0 < timeout:
                key = sub.get_key()
                if key:
                    try:
                        payload = json.loads(key)
                        if payload.get("type") == "task_result":
                            print("task_result:", json.dumps(payload, indent=2))
                            return
                    except Exception:
                        pass
                time.sleep(0.05)
            print("task_result: timeout")

        info_sub = SlamInfoSubscriber(self.slam_info_topic, self.slam_key_topic)
        info_sub.start()

        client = SlamOperateClient()
        client.Init()
        client.SetTimeout(10.0)

        req = {"data": {"slam_type": "indoor"}}
        _print_resp("start_mapping (1801)", req, client.start_mapping("indoor"))

        req = {"data": {"address": save_path}}
        _print_resp("end_mapping (1802)", req, client.end_mapping(save_path))

        req = {
            "data": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "q_x": 0.0,
                "q_y": 0.0,
                "q_z": 0.0,
                "q_w": 1.0,
                "address": load_path,
            }
        }
        _print_resp("init_pose (1804)", req, client.init_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, load_path))

        qz = math.sin(float(goal_yaw) * 0.5)
        qw = math.cos(float(goal_yaw) * 0.5)
        req = {
            "data": {
                "targetPose": {
                    "x": float(goal_x),
                    "y": float(goal_y),
                    "z": 0.0,
                    "q_x": 0.0,
                    "q_y": 0.0,
                    "q_z": qz,
                    "q_w": qw,
                },
                "mode": 1,
            }
        }
        _print_resp(
            "pose_nav (1102)",
            req,
            client.pose_nav(float(goal_x), float(goal_y), 0.0, 0.0, 0.0, qz, qw, mode=1),
        )

        if pause:
            _print_resp("pause_nav (1201)", {"data": {}}, client.pause_nav())
        if resume:
            _print_resp("resume_nav (1202)", {"data": {}}, client.resume_nav())
        if wait_task_result:
            _wait_task(info_sub)

        _print_resp("close_slam (1901)", {"data": {}}, client.close_slam())

    # ------------------------------------------------------------------
    # Safety / audio / lights
    # ------------------------------------------------------------------

    def hanged_boot(self) -> None:
        """Re-run hanger boot sequence and refresh locomotion client."""
        self._client = hanger_boot_sequence(iface=self.iface)
        self._ensure_balanced_gait_mode()

    def say(self, text: str = "what would you like me to say?") -> None:
        audio_dir = SCRIPTS_ROOT / "basic" / "audio"
        tts_script = audio_dir / "text_to_wav.py"
        greet_script = audio_dir / "greeting.py"

        with tempfile.TemporaryDirectory(prefix="g1_say_") as td:
            wav = Path(td) / "speech.wav"
            subprocess.run([sys.executable, str(tts_script), text, "-o", str(wav)], check=True)
            subprocess.run(
                [sys.executable, str(greet_script), "--robot", "--iface", self.iface, "--file", str(wav)],
                check=True,
            )

    def headlight(self, args: dict[str, Any] | list[str] | str | None = None) -> int:
        """
        Wrapper over basic/headlight_client/headlight.py.

        Examples:
            robot.headlight({"color": "yellow", "intensity": 70})
            robot.headlight("--color red --intensity 40")
            robot.headlight(["--color", "green", "--intensity", "90"])
        """
        script = SCRIPTS_ROOT / "basic" / "headlight_client" / "headlight.py"
        cmd = [sys.executable, str(script), "--iface", self.iface]

        if isinstance(args, dict):
            for k, v in args.items():
                key = f"--{str(k).replace('_', '-')}"
                if isinstance(v, bool):
                    if v:
                        cmd.append(key)
                elif v is not None:
                    cmd.extend([key, str(v)])
        elif isinstance(args, str) and args.strip():
            cmd.extend(shlex.split(args))
        elif isinstance(args, list):
            cmd.extend([str(x) for x in args])

        result = subprocess.run(cmd, check=False)
        return int(result.returncode)

    def huddle(self, args: dict[str, Any] | list[str] | str | None = None) -> int:
        """
        Wrapper over dev/other/huddle/huddle.py.

        Examples:
            robot.huddle()
            robot.huddle({"arm": "right", "volume": 80, "brightness": 40})
            robot.huddle("--arm left --file huddle.wav --volume 90")
        """
        script = SCRIPTS_ROOT / "dev" / "other" / "huddle" / "huddle.py"
        cmd = [sys.executable, str(script), "--iface", self.iface]

        if isinstance(args, dict):
            for k, v in args.items():
                key = f"--{str(k).replace('_', '-')}"
                if isinstance(v, bool):
                    if v:
                        cmd.append(key)
                elif v is not None:
                    cmd.extend([key, str(v)])
        elif isinstance(args, str) and args.strip():
            cmd.extend(shlex.split(args))
        elif isinstance(args, list):
            cmd.extend([str(x) for x in args])

        result = subprocess.run(cmd, cwd=str(script.parent), check=False)
        return int(result.returncode)


__all__ = ["Robot", "ImuData"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test for ef_client Robot wrapper")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--no-safety", action="store_true")
    args = parser.parse_args()

    bot = Robot(iface=args.iface, safety_boot=not args.no_safety)
    time.sleep(0.6)
    print("FSM:", bot.get_fsm())
    print("IMU:", bot.get_imu())
