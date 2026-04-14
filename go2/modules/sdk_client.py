"""
sdk_client.py
=============

SDK-native Robot wrapper for Unitree Go2.

This module consolidates the common Go2 control patterns currently spread
across `../scripts/` into a reusable `Robot` class backed by the official
`unitree_sdk2py` clients.
"""
from __future__ import annotations

import json
import math
import threading
import time
from typing import Any

try:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.go2.video.video_client import VideoClient
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc


DEFAULT_SPORT_TOPIC = "rt/sportmodestate"
DEFAULT_LOWSTATE_TOPIC = "rt/lowstate"
DEFAULT_ODOM_TOPIC = "rt/odom"
BODYHEIGHT_API_ID = 1013


class Robot:
    """Convenience wrapper around common Go2 sport-mode workflows."""

    def __init__(
        self,
        iface: str = "eth0",
        domain_id: int = 0,
        timeout: float = 10.0,
        auto_start_sensors: bool = True,
        sport_topic: str = DEFAULT_SPORT_TOPIC,
        lowstate_topic: str = DEFAULT_LOWSTATE_TOPIC,
        odom_topic: str = DEFAULT_ODOM_TOPIC,
    ) -> None:
        self.iface = iface
        self.domain_id = int(domain_id)
        self.timeout = float(timeout)
        self.sport_topic = sport_topic
        self.lowstate_topic = lowstate_topic
        self.odom_topic = odom_topic

        ChannelFactoryInitialize(self.domain_id, self.iface)

        self._lock = threading.Lock()
        self._sport: SportModeState_ | None = None
        self._lowstate: LowState_ | None = None
        self._odom: Odometry_ | None = None
        self._last_sport_ts = 0.0
        self._last_lowstate_ts = 0.0
        self._last_odom_ts = 0.0

        self._sport_sub: ChannelSubscriber | None = None
        self._lowstate_sub: ChannelSubscriber | None = None
        self._odom_sub: ChannelSubscriber | None = None
        self._video_client: VideoClient | None = None

        self._client = SportClient()
        self._client.SetTimeout(self.timeout)
        self._client.Init()

        self._motion_switcher = MotionSwitcherClient()
        self._motion_switcher.SetTimeout(self.timeout)
        self._motion_switcher.Init()

        if auto_start_sensors:
            self.start_sensors()

    # ------------------------------------------------------------------
    # Sensor subscriptions
    # ------------------------------------------------------------------

    def start_sensors(self) -> None:
        if self._sport_sub is None:
            self._sport_sub = ChannelSubscriber(self.sport_topic, SportModeState_)
            self._sport_sub.Init(self._sport_cb, 10)
        if self._lowstate_sub is None:
            self._lowstate_sub = ChannelSubscriber(self.lowstate_topic, LowState_)
            self._lowstate_sub.Init(self._lowstate_cb, 10)
        if self._odom_sub is None:
            self._odom_sub = ChannelSubscriber(self.odom_topic, Odometry_)
            self._odom_sub.Init(self._odom_cb, 10)

    def _sport_cb(self, msg: SportModeState_) -> None:
        with self._lock:
            self._sport = msg
            self._last_sport_ts = time.time()

    def _lowstate_cb(self, msg: LowState_) -> None:
        with self._lock:
            self._lowstate = msg
            self._last_lowstate_ts = time.time()

    def _odom_cb(self, msg: Odometry_) -> None:
        with self._lock:
            self._odom = msg
            self._last_odom_ts = time.time()

    def _get_video_client(self) -> VideoClient:
        if self._video_client is None:
            self._video_client = VideoClient()
            self._video_client.SetTimeout(3.0)
            self._video_client.Init()
        return self._video_client

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

    @staticmethod
    def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_angle(value: float) -> float:
        while value > math.pi:
            value -= 2.0 * math.pi
        while value < -math.pi:
            value += 2.0 * math.pi
        return value

    def get_sport_state(self) -> SportModeState_ | None:
        with self._lock:
            return self._sport

    def get_low_state(self) -> LowState_ | None:
        with self._lock:
            return self._lowstate

    def get_odom(self) -> Odometry_ | None:
        with self._lock:
            return self._odom

    def get_sensor_timestamps(self) -> dict[str, float]:
        with self._lock:
            return {
                "sport": float(self._last_sport_ts),
                "lowstate": float(self._last_lowstate_ts),
                "odom": float(self._last_odom_ts),
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

    def wait_for_low_state(self, timeout: float = 2.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < max(0.0, timeout):
            if self.get_low_state() is not None:
                return True
            time.sleep(0.05)
        return self.get_low_state() is not None

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
        value = self._read_attr(msg, "gait_type")
        try:
            return int(value)
        except Exception:
            return None

    def get_body_height(self) -> float | None:
        msg = self.get_sport_state()
        if msg is None:
            return None
        value = self._read_attr(msg, "body_height")
        try:
            return float(value)
        except Exception:
            return None

    def get_position(self) -> tuple[float, float, float] | None:
        msg = self.get_sport_state()
        if msg is None:
            return None
        return self._vector3_from(self._read_attr(msg, "position"))

    def get_velocity(self) -> tuple[float, float, float] | None:
        msg = self.get_sport_state()
        if msg is None:
            return None
        return self._vector3_from(self._read_attr(msg, "velocity"))

    def get_yaw_speed(self) -> float | None:
        msg = self.get_sport_state()
        if msg is None:
            return None
        value = self._read_attr(msg, "yaw_speed")
        try:
            return float(value)
        except Exception:
            return None

    def get_joint_positions(self) -> list[float]:
        msg = self.get_low_state()
        if msg is None:
            return []
        positions: list[float] = []
        try:
            for motor in msg.motor_state[:12]:
                positions.append(float(motor.q))
        except Exception:
            return []
        return positions

    def get_joint_velocities(self) -> list[float]:
        msg = self.get_low_state()
        if msg is None:
            return []
        values: list[float] = []
        try:
            for motor in msg.motor_state[:12]:
                values.append(float(motor.dq))
        except Exception:
            return []
        return values

    def get_joint_torques(self) -> list[float]:
        msg = self.get_low_state()
        if msg is None:
            return []
        values: list[float] = []
        try:
            for motor in msg.motor_state[:12]:
                values.append(float(motor.tau_est))
        except Exception:
            return []
        return values

    def get_joint_position(self, joint_index: int) -> float | None:
        positions = self.get_joint_positions()
        idx = int(joint_index)
        if idx < 0 or idx >= len(positions):
            return None
        return float(positions[idx])

    def get_odom_pose(self) -> tuple[float, float, float] | None:
        msg = self.get_odom()
        if msg is None:
            return None
        try:
            pos = msg.pose.pose.position
            quat = msg.pose.pose.orientation
            yaw = self._quat_to_yaw(float(quat.x), float(quat.y), float(quat.z), float(quat.w))
            return (float(pos.x), float(pos.y), float(yaw))
        except Exception:
            return None

    def get_imu(self) -> dict[str, Any] | None:
        msg = self.get_low_state()
        if msg is None:
            return None
        try:
            imu = msg.imu_state
            return {
                "rpy": [float(imu.rpy[i]) for i in range(3)],
                "gyro": [float(imu.gyroscope[i]) for i in range(3)],
                "acc": [float(imu.accelerometer[i]) for i in range(3)],
                "quat": [float(imu.quaternion[i]) for i in range(4)],
                "temp": float(getattr(imu, "temperature", 0.0)),
            }
        except Exception:
            return None

    def get_camera_image_jpeg(self) -> bytes:
        code, data = self._get_video_client().GetImageSample()
        if int(code) != 0:
            raise RuntimeError(f"GetImageSample failed with code={code}")
        return bytes(data)

    def is_moving(self, linear_eps: float = 0.03, yaw_eps: float = 0.08) -> bool:
        velocity = self.get_velocity()
        yaw_speed = self.get_yaw_speed()
        if velocity is None and yaw_speed is None:
            return False
        if velocity is not None and math.hypot(float(velocity[0]), float(velocity[1])) > linear_eps:
            return True
        return yaw_speed is not None and abs(float(yaw_speed)) > yaw_eps

    def get_robot_state(self) -> dict[str, Any]:
        return {
            "mode": self.get_mode(),
            "gait": self.get_gait(),
            "body_height": self.get_body_height(),
            "position": self.get_position(),
            "velocity": self.get_velocity(),
            "yaw_speed": self.get_yaw_speed(),
            "imu": self.get_imu(),
            "odom_pose": self.get_odom_pose(),
            "joint_count": len(self.get_joint_positions()),
            "is_moving": self.is_moving(),
            "sensor_timestamps": self.get_sensor_timestamps(),
            "sensor_stale": self.sensors_stale(),
        }

    # ------------------------------------------------------------------
    # Motion-switcher helpers
    # ------------------------------------------------------------------

    def check_mode(self) -> tuple[int, dict[str, Any] | None]:
        code, result = self._motion_switcher.CheckMode()
        return int(code), result

    def release_active_mode(self, retries: int = 10, delay: float = 1.0) -> bool:
        for _ in range(max(1, int(retries))):
            code, result = self.check_mode()
            if code == 0 and not (result and result.get("name")):
                return True
            try:
                self._client.StandDown()
            except Exception:
                pass
            self._motion_switcher.ReleaseMode()
            time.sleep(max(0.0, float(delay)))
        code, result = self.check_mode()
        return code == 0 and not (result and result.get("name"))

    # ------------------------------------------------------------------
    # Sport-mode commands
    # ------------------------------------------------------------------

    def damp(self) -> int:
        return int(self._client.Damp())

    def stand_up(self) -> int:
        return int(self._client.StandUp())

    def stand_down(self) -> int:
        return int(self._client.StandDown())

    def balance_stand(self) -> int:
        return int(self._client.BalanceStand())

    def recovery_stand(self) -> int:
        return int(self._client.RecoveryStand())

    def move(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> int:
        return int(self._client.Move(float(vx), float(vy), float(vyaw)))

    def walk(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> int:
        return self.move(vx=vx, vy=vy, vyaw=vyaw)

    def stop_moving(self) -> int:
        return int(self._client.StopMove())

    def stop(self) -> int:
        return self.stop_moving()

    def set_body_height(self, height: float) -> int:
        self._client._RegistApi(BODYHEIGHT_API_ID, 0)
        code, _ = self._client._Call(BODYHEIGHT_API_ID, json.dumps({"data": float(height)}))
        return int(code)

    def move_for(self, vx: float, vy: float, vyaw: float, duration: float) -> int:
        code = self.move(vx=vx, vy=vy, vyaw=vyaw)
        time.sleep(max(0.0, float(duration)))
        self.stop_moving()
        return int(code)

    def walk_for(self, distance: float, speed: float = 0.3, tick: float = 0.05) -> bool:
        speed = abs(float(speed))
        if speed <= 0.0:
            raise ValueError("speed must be positive")

        pose0 = self.get_odom_pose()
        if pose0 is None:
            raise RuntimeError("walk_for requires live odometry")

        start_x, start_y, start_yaw = pose0
        sign = 1.0 if float(distance) >= 0.0 else -1.0
        target_x = float(start_x) + float(distance) * math.cos(float(start_yaw))
        target_y = float(start_y) + float(distance) * math.sin(float(start_yaw))

        try:
            while True:
                pose = self.get_odom_pose()
                if pose is None:
                    time.sleep(tick)
                    continue
                dx = target_x - float(pose[0])
                dy = target_y - float(pose[1])
                remaining = math.hypot(dx, dy)
                if remaining <= 0.03:
                    return True
                vx = sign * min(speed, max(0.08, remaining))
                self.move(vx=vx, vy=0.0, vyaw=0.0)
                time.sleep(tick)
        finally:
            self.stop_moving()

    def turn_for(self, angle_rad: float, yaw_rate: float = 0.5, tick: float = 0.05) -> bool:
        yaw_rate = abs(float(yaw_rate))
        if yaw_rate <= 0.0:
            raise ValueError("yaw_rate must be positive")

        pose0 = self.get_odom_pose()
        if pose0 is None:
            raise RuntimeError("turn_for requires live odometry")

        start_yaw = float(pose0[2])
        target_yaw = self._wrap_angle(start_yaw + float(angle_rad))
        sign = 1.0 if float(angle_rad) >= 0.0 else -1.0

        try:
            while True:
                pose = self.get_odom_pose()
                if pose is None:
                    time.sleep(tick)
                    continue
                error = self._wrap_angle(target_yaw - float(pose[2]))
                if abs(error) <= 0.05:
                    return True
                cmd = sign * min(yaw_rate, max(0.15, abs(error)))
                self.move(vx=0.0, vy=0.0, vyaw=cmd)
                time.sleep(tick)
        finally:
            self.stop_moving()


__all__ = ["Robot", "BODYHEIGHT_API_ID"]
