from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.rpc.client import Client


SERVICE_NAME = "slam_operate"
SERVICE_VERSION = "1.0.0.1"

API_START_MAPPING = 1801
API_END_MAPPING = 1802
API_INIT_POSE = 1804
API_POSE_NAV = 1102
API_PAUSE_NAV = 1201
API_RESUME_NAV = 1202
API_CLOSE_SLAM = 1901


@dataclass
class SlamResponse:
    code: int
    raw: Any


class SlamOperateClient(Client):
    def __init__(self, enable_lease: bool = False) -> None:
        super().__init__(SERVICE_NAME, enable_lease)

    def Init(self) -> None:
        self._RegistApi(API_START_MAPPING, 0)
        self._RegistApi(API_END_MAPPING, 0)
        self._RegistApi(API_INIT_POSE, 0)
        self._RegistApi(API_POSE_NAV, 0)
        self._RegistApi(API_PAUSE_NAV, 0)
        self._RegistApi(API_RESUME_NAV, 0)
        self._RegistApi(API_CLOSE_SLAM, 0)
        self._SetApiVerson(SERVICE_VERSION)

    def _call(self, api_id: int, payload: dict[str, Any]) -> SlamResponse:
        code, data = self._Call(api_id, json.dumps(payload, ensure_ascii=True))
        return SlamResponse(code=int(code), raw=data)

    def start_mapping(self, slam_type: str = "indoor") -> SlamResponse:
        return self._call(API_START_MAPPING, {"data": {"slam_type": slam_type}})

    def end_mapping(self, address: str) -> SlamResponse:
        return self._call(API_END_MAPPING, {"data": {"address": address}})

    def init_pose(
        self,
        x: float,
        y: float,
        z: float,
        q_x: float,
        q_y: float,
        q_z: float,
        q_w: float,
        address: str,
    ) -> SlamResponse:
        return self._call(
            API_INIT_POSE,
            {
                "data": {
                    "x": x,
                    "y": y,
                    "z": z,
                    "q_x": q_x,
                    "q_y": q_y,
                    "q_z": q_z,
                    "q_w": q_w,
                    "address": address,
                }
            },
        )

    def pose_nav(
        self,
        x: float,
        y: float,
        z: float,
        q_x: float,
        q_y: float,
        q_z: float,
        q_w: float,
        mode: int = 1,
    ) -> SlamResponse:
        return self._call(
            API_POSE_NAV,
            {
                "data": {
                    "targetPose": {
                        "x": x,
                        "y": y,
                        "z": z,
                        "q_x": q_x,
                        "q_y": q_y,
                        "q_z": q_z,
                        "q_w": q_w,
                    },
                    "mode": mode,
                }
            },
        )

    def pause_nav(self) -> SlamResponse:
        return self._call(API_PAUSE_NAV, {"data": {}})

    def resume_nav(self) -> SlamResponse:
        return self._call(API_RESUME_NAV, {"data": {}})

    def close_slam(self) -> SlamResponse:
        return self._call(API_CLOSE_SLAM, {"data": {}})


class SlamInfoSubscriber:
    def __init__(self, info_topic: str = "rt/slam_info", key_topic: str = "rt/slam_key_info") -> None:
        self.info_topic = info_topic
        self.key_topic = key_topic
        self._lock = threading.Lock()
        self._info: str | None = None
        self._key: str | None = None
        self._last_info: float = 0.0
        self._last_key: float = 0.0
        self._info_sub: ChannelSubscriber | None = None
        self._key_sub: ChannelSubscriber | None = None

    def start(self) -> None:
        if self._info_sub is None:
            self._info_sub = ChannelSubscriber(self.info_topic, String_)
            self._info_sub.Init(self._info_cb, 10)
        if self._key_sub is None:
            self._key_sub = ChannelSubscriber(self.key_topic, String_)
            self._key_sub.Init(self._key_cb, 10)

    def _info_cb(self, msg: String_) -> None:
        with self._lock:
            self._info = str(msg.data)
            self._last_info = time.time()

    def _key_cb(self, msg: String_) -> None:
        with self._lock:
            self._key = str(msg.data)
            self._last_key = time.time()

    def get_info(self) -> str | None:
        with self._lock:
            return self._info

    def get_key(self) -> str | None:
        with self._lock:
            return self._key


__all__ = ["SlamInfoSubscriber", "SlamOperateClient", "SlamResponse"]
