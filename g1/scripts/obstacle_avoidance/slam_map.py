"""
slam_map.py
===========

Helpers for consuming the G1's built-in SLAM / mapping output over DDS.

This subscribes to the utlidar map_state topic and converts a HeightMap_ into
an OccupancyGrid for use by the obstacle_avoidance pipeline.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_, LidarState_
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed.  Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc

from create_map import OccupancyGrid


TOPIC_MAP_STATE = "rt/utlidar/map_state"
TOPIC_LIDAR_SWITCH = "rt/utlidar/switch"


@dataclass
class SlamMapMeta:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    timestamp: float


class SlamMapSubscriber:
    """Subscribe to utlidar map_state and expose the latest HeightMap_."""

    def __init__(self, topic: str = TOPIC_MAP_STATE) -> None:
        self.topic = topic
        self._lock = threading.Lock()
        self._map: HeightMap_ | None = None
        self._last_ts: float = 0.0
        self._sub: ChannelSubscriber | None = None

    def start(self) -> None:
        self._sub = ChannelSubscriber(self.topic, HeightMap_)
        self._sub.Init(self._callback, 10)

    def _callback(self, msg: HeightMap_) -> None:
        with self._lock:
            self._map = msg
            self._last_ts = time.time()

    def get_latest(self) -> tuple[HeightMap_ | None, float]:
        with self._lock:
            return self._map, self._last_ts

    def is_stale(self, max_age: float = 1.0) -> bool:
        with self._lock:
            ts = self._last_ts
        if ts == 0.0:
            return True
        return (time.time() - ts) > max_age

    def to_occupancy(
        self,
        height_threshold: float = 0.15,
        max_height: Optional[float] = None,
        origin_centered: bool = True,
    ) -> tuple[OccupancyGrid | None, SlamMapMeta | None]:
        """
        Convert the latest HeightMap_ into an OccupancyGrid.

        Args:
            height_threshold: Cells with height >= threshold become obstacles.
            max_height: Optional clamp; values above are treated as obstacles.
            origin_centered: If true, set origin to center the map at (0,0).
        """
        msg, ts = self.get_latest()
        if msg is None:
            return None, None

        try:
            width = int(msg.width)
            height = int(msg.height)
            resolution = float(msg.resolution)
            data = np.array(list(msg.data), dtype=float)
        except Exception:
            return None, None

        if width <= 0 or height <= 0 or resolution <= 0:
            return None, None
        if data.size != width * height:
            return None, None

        grid = data.reshape((height, width))
        if max_height is not None:
            grid = np.minimum(grid, max_height)

        occ = (grid >= height_threshold).astype(np.int8)

        if origin_centered:
            origin_x = -width * resolution / 2.0
            origin_y = -height * resolution / 2.0
        else:
            origin_x = 0.0
            origin_y = 0.0

        occ_grid = OccupancyGrid(
            width_m=width * resolution,
            height_m=height * resolution,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        occ_grid.grid = occ

        meta = SlamMapMeta(
            width=width,
            height=height,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            timestamp=ts,
        )
        return occ_grid, meta


class LidarSwitch:
    """Lightweight publisher to toggle the utlidar mapping stack."""

    def __init__(self, topic: str = TOPIC_LIDAR_SWITCH) -> None:
        self.topic = topic
        self._pub = ChannelPublisher(self.topic, String_)
        self._pub.Init()

    def set(self, status: str) -> int:
        msg = String_(status)
        self._pub.Write(msg)
        return 0


__all__ = [
    "SlamMapSubscriber",
    "SlamMapMeta",
    "LidarSwitch",
    "TOPIC_MAP_STATE",
    "TOPIC_LIDAR_SWITCH",
]
