"""
ef_client.py
============

Lightweight robot client wrapper for Unitree G1 troubleshooting.

Provides:
  - safe boot / FSM control via hanger_boot_sequence
  - cached sensor data (IMU, sport state, lidar map, lidar cloud)
  - simple motion helpers

This is intentionally conservative: it does not attempt autonomous motion
or planning; it only exposes basic SDK calls and cached telemetry.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

# Ensure scripts dir is on sys.path so we can import safety helpers.
_SCRIPTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LidarState_, HeightMap_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc

from safety.hanger_boot_sequence import hanger_boot_sequence


TOPIC_SPORT = "rt/sportmodestate"
TOPIC_LIDAR_STATE = "rt/utlidar/map_state"
TOPIC_LIDAR_CLOUD = "rt/utlidar/cloud_deskewed"
TOPIC_LIDAR_SWITCH = "rt/utlidar/switch"


@dataclass
class ImuData:
    rpy: tuple[float, float, float]
    gyro: tuple[float, float, float] | None
    acc: tuple[float, float, float] | None
    quat: tuple[float, float, float, float] | None
    temp: float | None


class Robot:
    """Minimal, safe wrapper around Unitree SDK2 for troubleshooting."""

    def __init__(self, iface: str, domain_id: int = 0, safety_boot: bool = True) -> None:
        self.iface = iface
        self.domain_id = domain_id
        self._lock = threading.Lock()

        self._sport: SportModeState_ | None = None
        self._lidar_state: Any | None = None
        self._lidar_cloud: PointCloud2_ | None = None
        self._last_sport_ts: float = 0.0
        self._last_lidar_ts: float = 0.0
        self._last_cloud_ts: float = 0.0

        self._sport_sub: ChannelSubscriber | None = None
        self._lidar_state_sub: ChannelSubscriber | None = None
        self._lidar_cloud_sub: ChannelSubscriber | None = None
        self._lidar_switch_pub: ChannelPublisher | None = None

        # SDK init + safe boot (preferred)
        if safety_boot:
            self._client: LocoClient = hanger_boot_sequence(iface=iface)
        else:
            ChannelFactoryInitialize(domain_id, iface)
            self._client = LocoClient()
            self._client.SetTimeout(10.0)
            self._client.Init()

        # Start subscribers lazily; caller can invoke start_sensors().

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def start_sensors(self) -> None:
        """Start DDS subscriptions for sport state and lidar topics."""
        if self._sport_sub is None:
            self._sport_sub = ChannelSubscriber(TOPIC_SPORT, SportModeState_)
            self._sport_sub.Init(self._sport_cb, 10)

        if self._lidar_state_sub is None:
            self._lidar_state_sub = ChannelSubscriber(TOPIC_LIDAR_STATE, HeightMap_)
            self._lidar_state_sub.Init(self._lidar_state_cb, 10)

        if self._lidar_cloud_sub is None:
            self._lidar_cloud_sub = ChannelSubscriber(TOPIC_LIDAR_CLOUD, PointCloud2_)
            self._lidar_cloud_sub.Init(self._lidar_cloud_cb, 10)

        if self._lidar_switch_pub is None:
            self._lidar_switch_pub = ChannelPublisher(TOPIC_LIDAR_SWITCH, String_)
            self._lidar_switch_pub.Init()

    def _sport_cb(self, msg: SportModeState_) -> None:
        with self._lock:
            self._sport = msg
            self._last_sport_ts = time.time()

    def _lidar_state_cb(self, msg: Any) -> None:
        with self._lock:
            self._lidar_state = msg
            self._last_lidar_ts = time.time()

    def _lidar_cloud_cb(self, msg: PointCloud2_) -> None:
        with self._lock:
            self._lidar_cloud = msg
            self._last_cloud_ts = time.time()

    # ------------------------------------------------------------------
    # Sensor getters (best-effort)
    # ------------------------------------------------------------------

    def get_sport_state(self) -> SportModeState_ | None:
        with self._lock:
            return self._sport

    def get_imu_data(self) -> ImuData | None:
        with self._lock:
            msg = self._sport
        if msg is None:
            return None

        rpy = (0.0, 0.0, 0.0)
        gyro = acc = quat = None
        temp = None

        try:
            rpy = (float(msg.imu_state.rpy[0]), float(msg.imu_state.rpy[1]), float(msg.imu_state.rpy[2]))
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

    def get_lidar_map(self) -> HeightMap_ | None:
        with self._lock:
            return self._lidar_state

    def get_lidar_cloud(self) -> PointCloud2_ | None:
        with self._lock:
            return self._lidar_cloud

    def lidar_switch(self, on: bool) -> int:
        if self._lidar_switch_pub is None:
            return -1
        self._lidar_switch_pub.Write(String_("ON" if on else "OFF"))
        return 0

    def sensors_stale(self, max_age: float = 1.0) -> bool:
        with self._lock:
            ts = self._last_sport_ts
        if ts == 0.0:
            return True
        return (time.time() - ts) > max_age

    # ------------------------------------------------------------------
    # Motion helpers
    # ------------------------------------------------------------------

    def move(self, vx: float, vy: float, vyaw: float, continuous: bool = True) -> int:
        return self._client.Move(float(vx), float(vy), float(vyaw), continous_move=continuous)

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        if hasattr(self._client, "SetVelocity"):
            self._client.SetVelocity(float(vx), float(vy), float(vyaw))
        else:
            self.move(vx, vy, vyaw, continuous=True)

    def stop(self) -> None:
        if hasattr(self._client, "StopMove"):
            self._client.StopMove()
        else:
            self.move(0.0, 0.0, 0.0, continuous=False)

    def emergency_stop(self) -> None:
        try:
            self.stop()
        finally:
            if hasattr(self._client, "ZeroTorque"):
                self._client.ZeroTorque()

    def damp(self) -> None:
        if hasattr(self._client, "Damp"):
            self._client.Damp()

    # ------------------------------------------------------------------
    # FSM helpers (best-effort)
    # ------------------------------------------------------------------

    def set_fsm_id(self, fsm_id: int) -> None:
        if hasattr(self._client, "SetFsmId"):
            self._client.SetFsmId(int(fsm_id))

    def start(self) -> None:
        if hasattr(self._client, "Start"):
            self._client.Start()

    def balance_stand(self, mode: int = 0) -> None:
        if hasattr(self._client, "BalanceStand"):
            self._client.BalanceStand(int(mode))

    # Low-level RPC access (same as hanger_boot_sequence)
    def _rpc_get_int(self, api_id: int) -> Optional[int]:
        try:
            code, data = self._client._Call(api_id, "{}")  # type: ignore[attr-defined]
            if code == 0 and data:
                import json

                return json.loads(data).get("data")
        except Exception:
            pass
        return None

    def fsm_id(self) -> Optional[int]:
        try:
            from unitree_sdk2py.g1.loco.g1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID
        except Exception:
            return None
        return self._rpc_get_int(ROBOT_API_ID_LOCO_GET_FSM_ID)

    def fsm_mode(self) -> Optional[int]:
        try:
            from unitree_sdk2py.g1.loco.g1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_MODE
        except Exception:
            return None
        return self._rpc_get_int(ROBOT_API_ID_LOCO_GET_FSM_MODE)


__all__ = ["Robot", "ImuData"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick smoke-test for ef_client Robot wrapper")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--no-safety", action="store_true", help="Do not run safety boot sequence")
    args = parser.parse_args()

    bot = Robot(args.iface, safety_boot=not args.no_safety)
    bot.start_sensors()

    print("Waiting for sport state...")
    time.sleep(1.0)

    imu = bot.get_imu_data()
    print("IMU:", imu)

    print("FSM id:", bot.fsm_id(), "mode:", bot.fsm_mode())
