"""
sdk_client.py
=============

SDK-native Robot wrapper for Unitree G1.

The implementation is intentionally local to `modules/` and avoids imports
from `../scripts`. Script-backed workflows were replaced with direct SDK
helpers or removed from the core wrapper.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from sdk_audio import RobotAudio
from sdk_boot import create_loco_client, hanger_boot_sequence, rpc_get_int
from sdk_slam import SlamInfoSubscriber, SlamOperateClient, SlamResponse

try:
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_, SportModeState_
    from unitree_sdk2py.g1.loco.g1_loco_api import (
        ROBOT_API_ID_LOCO_GET_FSM_ID,
        ROBOT_API_ID_LOCO_GET_FSM_MODE,
    )
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc


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
    """End-user wrapper around common G1 SDK workflows."""

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
    ) -> None:
        self.iface = iface
        self.domain_id = int(domain_id)
        self.sport_topic = sport_topic
        self.lidar_map_topic = lidar_map_topic
        self.lidar_cloud_topic = lidar_cloud_topic
        self.slam_info_topic = slam_info_topic
        self.slam_key_topic = slam_key_topic

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

        self._path_points: list[tuple[float, float, float]] = []
        self._slam_client: SlamOperateClient | None = None
        self._audio: RobotAudio | None = None
        self.slam_is_running = False

        if safety_boot:
            self._client = hanger_boot_sequence(iface=self.iface, domain_id=self.domain_id)
        else:
            self._client = create_loco_client(domain_id=self.domain_id, iface=self.iface)

        if auto_start_sensors:
            self.start_sensors()

    def _get_slam_client(self) -> SlamOperateClient:
        if self._slam_client is None:
            self._slam_client = SlamOperateClient()
            self._slam_client.Init()
            self._slam_client.SetTimeout(10.0)
        return self._slam_client

    def _get_audio(self) -> RobotAudio:
        if self._audio is None:
            self._audio = RobotAudio()
        return self._audio

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
        return {
            name: (ts <= 0.0) or ((now - ts) > max_age)
            for name, ts in self.get_sensor_timestamps().items()
        }

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
        velocity = self.get_velocity()
        if velocity is None:
            return False
        vx, vy, vz = velocity
        return math.hypot(vx, vy) > linear_eps or abs(vz) > yaw_eps

    def get_robot_state(self) -> dict[str, Any]:
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
        return int(self._client.Move(float(vx), float(vy), float(vyaw), continous_move=True))

    def stop_moving(self) -> None:
        if hasattr(self._client, "StopMove"):
            self._client.StopMove()
            return
        self._client.Move(0.0, 0.0, 0.0, continous_move=False)

    def _rpc_get_int(self, api_id: int) -> Optional[int]:
        return rpc_get_int(self._client, api_id)

    def get_fsm(self) -> dict[str, Optional[int]]:
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
            rpy = tuple(float(msg.imu_state.rpy[i]) for i in range(3))  # type: ignore[assignment]
        except Exception:
            pass
        try:
            gyro = tuple(float(msg.imu_state.gyroscope[i]) for i in range(3))  # type: ignore[assignment]
        except Exception:
            pass
        try:
            acc = tuple(float(msg.imu_state.accelerometer[i]) for i in range(3))  # type: ignore[assignment]
        except Exception:
            pass
        try:
            quat = tuple(float(msg.imu_state.quaternion[i]) for i in range(4))  # type: ignore[assignment]
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
            name_to_off = {str(field.name).lower(): int(field.offset) for field in fields}
            x_off = name_to_off.get("x", x_off)
            y_off = name_to_off.get("y", y_off)
            z_off = name_to_off.get("z", z_off)
        except Exception:
            pass

        total = max(0, width * height)
        if max_points is not None:
            total = min(total, max_points)

        import struct

        points: list[tuple[float, float, float]] = []
        for idx in range(total):
            base = idx * point_step
            try:
                x = struct.unpack_from("<f", raw, base + x_off)[0]
                y = struct.unpack_from("<f", raw, base + y_off)[0]
                z = struct.unpack_from("<f", raw, base + z_off)[0]
            except Exception:
                break
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                points.append((float(x), float(y), float(z)))
        return points

    def get_lidar_points(self, max_points: int | None = 20000) -> list[tuple[float, float, float]]:
        with self._lock:
            msg = self._lidar_cloud
        if msg is None:
            return []
        return self._extract_xyz_from_cloud(msg, max_points=max_points)

    # ------------------------------------------------------------------
    # SLAM + navigation
    # ------------------------------------------------------------------

    def start_slam(self, slam_type: str = "indoor") -> int:
        response = self._get_slam_client().start_mapping(slam_type=slam_type)
        self.slam_is_running = response.code == 0
        return int(response.code)

    def stop_slam(self, save_path: str | None = None) -> int:
        client = self._get_slam_client()
        response = client.end_mapping(save_path) if save_path else client.close_slam()
        self.slam_is_running = False
        return int(response.code)

    def set_path_point(self, x: float, y: float, yaw: float = 0.0) -> None:
        self._path_points.append((float(x), float(y), float(yaw)))

    def _run_pose_nav(self, x: float, y: float, yaw: float = 0.0) -> int:
        client = self._get_slam_client()
        qz = math.sin(float(yaw) * 0.5)
        qw = math.cos(float(yaw) * 0.5)
        response = client.pose_nav(float(x), float(y), 0.0, 0.0, 0.0, qz, qw, mode=1)
        return int(response.code)

    def navigate_path(self, clear_on_finish: bool = True) -> bool:
        if not self._path_points:
            raise RuntimeError("No path points queued. Call set_path_point(...) first.")

        ok = True
        try:
            for idx, (x, y, yaw) in enumerate(self._path_points, start=1):
                rc = self._run_pose_nav(x, y, yaw)
                if rc != 0:
                    print(f"[navigate_path] failed at point {idx}: ({x:.3f},{y:.3f},{yaw:.3f}) rc={rc}")
                    ok = False
                    break
        finally:
            if clear_on_finish:
                self._path_points.clear()
        return ok

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
        def print_resp(label: str, req: dict[str, Any], resp: SlamResponse) -> None:
            print(f"\n[{label}]")
            print("request:", json.dumps(req, indent=2))
            print(f"response: code={resp.code} raw={resp.raw}")

        def wait_task(sub: SlamInfoSubscriber, timeout: float = 10.0) -> None:
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

        client = self._get_slam_client()

        req = {"data": {"slam_type": "indoor"}}
        print_resp("start_mapping (1801)", req, client.start_mapping("indoor"))

        req = {"data": {"address": save_path}}
        print_resp("end_mapping (1802)", req, client.end_mapping(save_path))

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
        print_resp("init_pose (1804)", req, client.init_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, load_path))

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
        print_resp("pose_nav (1102)", req, client.pose_nav(float(goal_x), float(goal_y), 0.0, 0.0, 0.0, qz, qw, mode=1))

        if pause:
            print_resp("pause_nav (1201)", {"data": {}}, client.pause_nav())
        if resume:
            print_resp("resume_nav (1202)", {"data": {}}, client.resume_nav())
        if wait_task_result:
            wait_task(info_sub)

        print_resp("close_slam (1901)", {"data": {}}, client.close_slam())

    # ------------------------------------------------------------------
    # Safety + audio
    # ------------------------------------------------------------------

    def hanged_boot(self) -> None:
        self._client = hanger_boot_sequence(iface=self.iface, domain_id=self.domain_id)

    def say(self, text: str = "what would you like me to say?", volume: int | None = None) -> int:
        return self._get_audio().speak(text, volume=volume)

    def play_wav(self, wav_path: str, volume: int | None = None) -> int:
        return self._get_audio().play_wav(wav_path, volume=volume)

    def headlight(
        self,
        color: str = "white",
        intensity: int = 100,
        duration: float | None = None,
    ) -> int:
        return self._get_audio().set_headlight(color=color, intensity=intensity, duration=duration)


__all__ = ["Robot", "ImuData"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test for sdk_client Robot wrapper")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--no-safety", action="store_true")
    args = parser.parse_args()

    bot = Robot(iface=args.iface, safety_boot=not args.no_safety)
    time.sleep(0.6)
    print("FSM:", bot.get_fsm())
    print("IMU:", bot.get_imu())
