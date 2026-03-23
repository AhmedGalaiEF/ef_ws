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
        nav_extra_args: str = "--smooth --allow-diagonal --use-live-map",
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
        self.slam_is_running = False
        self._slam_save_dir: str | None = None

        self._path_points: list[tuple[float, float, float]] = []

        if safety_boot:
            self._client = hanger_boot_sequence(iface=self.iface)
        else:
            ChannelFactoryInitialize(self.domain_id, self.iface)
            self._client = LocoClient()
            self._client.SetTimeout(10.0)
            self._client.Init()

        if auto_start_sensors:
            self.start_sensors()

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

    def fsm_2_airborne(self) -> None:
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
    ) -> subprocess.Popen:
        if self._slam_proc is not None and self._slam_proc.poll() is None:
            self.slam_is_running = True
            return self._slam_proc

        slam_script = SCRIPTS_ROOT / "navigation" / "obstacle_avoidance" / "live_slam_save.py"
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

        self._slam_proc = subprocess.Popen(cmd, cwd=str(slam_cwd))
        self._slam_save_dir = str(save_dir)
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
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    proc.terminate()
                    proc.wait(timeout=3.0)
                except Exception:
                    proc.kill()
        self._slam_proc = None
        self.slam_is_running = False

    # ------------------------------------------------------------------
    # Path points + navigation
    # ------------------------------------------------------------------

    def set_path_point(self, x: float, y: float, yaw: float = 0.0) -> None:
        if not self.slam_is_running:
            raise RuntimeError("set_path_point is only allowed while slam_is_running=True")
        self._path_points.append((float(x), float(y), float(yaw)))

    def _run_dynamic_nav(self, x: float, y: float) -> int:
        nav_script = SCRIPTS_ROOT / "navigation" / "obstacle_avoidance" / "real_time_path_steps_dynamic.py"

        if not self.nav_map:
            # common fallback map path
            fallback = SCRIPTS_ROOT / "navigation" / "obstacle_avoidance" / "maps" / "live_slam_latest.pcd"
            self.nav_map = str(fallback)

        cmd = [
            sys.executable,
            str(nav_script),
            "--map",
            str(self.nav_map),
            "--iface",
            self.iface,
            "--domain-id",
            str(self.domain_id),
            "--goal-x",
            str(float(x)),
            "--goal-y",
            str(float(y)),
        ]
        if self.nav_extra_args:
            cmd.extend(shlex.split(self.nav_extra_args))

        result = subprocess.run(cmd, cwd=str(nav_script.parent), check=False)
        return int(result.returncode)

    def _run_pose_nav(self, x: float, y: float, yaw: float = 0.0) -> int:
        from navigation.obstacle_avoidance.slam_service import SlamOperateClient

        client = SlamOperateClient()
        client.Init()
        client.SetTimeout(10.0)

        qz = math.sin(float(yaw) * 0.5)
        qw = math.cos(float(yaw) * 0.5)
        resp = client.pose_nav(float(x), float(y), 0.0, 0.0, 0.0, qz, qw, mode=1)
        return int(resp.code)

    def navigate_path(self, obs_avoid: bool = True, clear_on_finish: bool = True) -> bool:
        if not self._path_points:
            raise RuntimeError("No path points queued. Call set_path_point(...) first.")

        ok = True
        try:
            for idx, (x, y, yaw) in enumerate(self._path_points, start=1):
                if obs_avoid:
                    rc = self._run_dynamic_nav(x, y)
                else:
                    rc = self._run_pose_nav(x, y, yaw)
                if rc != 0:
                    print(f"[navigate_path] failed at point {idx}: ({x:.3f},{y:.3f},{yaw:.3f}) rc={rc}")
                    ok = False
                    break
        finally:
            if clear_on_finish:
                self._path_points.clear()
        return ok

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
